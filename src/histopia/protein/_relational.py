"""Portable inductive cell-neighborhood protein regressors.

The prediction mean never consumes neighboring protein outcomes.  It queries
two training-only banks using target-free morphology and registered physical
coordinates, then applies GPU-fitted attention weights.  Training outcomes are
stored only to report local-neighborhood dispersion as uncertainty.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from histopia._atomic import write_binary_atomic

_RELATIONAL_ARCHITECTURES = frozenset({"dual_bank_attention", "graph_transformer"})
_RELATIONAL_LOSS_PROFILES = frozenset({"huber_v1", "tail_rank_v1"})
_LEGACY_OPERATOR = "dual-bank-attention-legacy-v1"
_DUAL_BANK_OPERATOR = "dual-bank-cross-attention-v2"
_GRAPH_OPERATOR = "residual-graph-message-v2"
_RELATIONAL_OPERATORS = frozenset(
    {_LEGACY_OPERATOR, _DUAL_BANK_OPERATOR, _GRAPH_OPERATOR}
)
_ARRAY_NAMES = (
    "feature_mean",
    "feature_scale",
    "bank_embedding",
    "bank_morphology_key",
    "bank_reference_um_xyz",
    "bank_od",
    "embed_weight",
    "embed_bias",
    "norm_weight",
    "norm_bias",
    "attention_weight",
    "attention_bias",
    "attention_output_weight",
    "attention_output_bias",
    "head0_weight",
    "head0_bias",
    "head1_weight",
    "head1_bias",
)


@dataclass(frozen=True, slots=True)
class PortableRelationalRegressor:
    """A two-bank cell regressor with NumPy and optional GPU inference."""

    architecture: str
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    bank_embedding: np.ndarray
    bank_morphology_key: np.ndarray
    bank_reference_um_xyz: np.ndarray
    bank_od: np.ndarray
    embed_weight: np.ndarray
    embed_bias: np.ndarray
    norm_weight: np.ndarray
    norm_bias: np.ndarray
    attention_weight: np.ndarray
    attention_bias: np.ndarray
    attention_output_weight: np.ndarray
    attention_output_bias: np.ndarray
    head0_weight: np.ndarray
    head0_bias: np.ndarray
    head1_weight: np.ndarray
    head1_bias: np.ndarray
    heads: int
    morphology_width: int
    target_od_upper: float
    provenance: dict[str, object]
    fingerprint: str | None = None

    def __post_init__(self) -> None:
        feature_count = len(np.asarray(self.feature_mean))
        width = len(np.asarray(self.embed_bias))
        bank_count = len(np.asarray(self.bank_embedding))
        if self.architecture not in _RELATIONAL_ARCHITECTURES:
            raise ValueError("unsupported portable relational architecture")
        operator = str(self.provenance.get("relational_operator", _LEGACY_OPERATOR))
        if operator not in _RELATIONAL_OPERATORS or (
            operator == _GRAPH_OPERATOR and self.architecture != "graph_transformer"
        ):
            raise ValueError("portable relational operator is incompatible")
        if (
            np.asarray(self.feature_scale).shape != (feature_count,)
            or np.asarray(self.embed_weight).shape != (width, feature_count)
            or np.asarray(self.norm_weight).shape != (width,)
            or np.asarray(self.norm_bias).shape != (width,)
        ):
            raise ValueError("relational feature projection is malformed")
        if (
            np.asarray(self.bank_embedding).shape != (bank_count, width)
            or np.asarray(self.bank_morphology_key).shape != (bank_count, 16)
            or np.asarray(self.bank_reference_um_xyz).shape != (bank_count, 3)
            or np.asarray(self.bank_od).shape != (bank_count,)
            or bank_count < 2
        ):
            raise ValueError("relational training banks are malformed")
        if (
            self.heads < 1
            or width % self.heads
            or np.asarray(self.attention_weight).shape != (3 * width, width)
            or np.asarray(self.attention_bias).shape != (3 * width,)
            or np.asarray(self.attention_output_weight).shape != (width, width)
            or np.asarray(self.attention_output_bias).shape != (width,)
            or np.asarray(self.head0_weight).shape != (width, 3 * width)
            or np.asarray(self.head0_bias).shape != (width,)
            or np.asarray(self.head1_weight).shape != (1, width)
            or np.asarray(self.head1_bias).shape != (1,)
        ):
            raise ValueError("relational attention weights are malformed")
        if not 1 <= self.morphology_width <= feature_count:
            raise ValueError("relational morphology width is invalid")
        if not math.isfinite(self.target_od_upper) or self.target_od_upper <= 0:
            raise ValueError("relational target OD bound is invalid")
        for value in _arrays(self).values():
            if not np.all(np.isfinite(value)):
                raise ValueError("relational model arrays must be finite")
        expected = _fingerprint(self)
        if self.fingerprint is not None and self.fingerprint != expected:
            raise ValueError("portable relational fingerprint does not match")
        object.__setattr__(self, "fingerprint", expected)

    def predict(
        self,
        features: np.ndarray,
        reference_um_xyz: np.ndarray,
        *,
        batch_size: int = 1024,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict bounded OD and local training-neighborhood dispersion."""

        values, xyz = self._validate_query(features, reference_um_xyz, batch_size)
        normalized = (values - self.feature_mean) / self.feature_scale
        morphology_index, spatial_index = self._query_indices(normalized, xyz)
        output = np.empty(len(values), dtype=np.float32)
        uncertainty = _neighbor_uncertainty(
            self.bank_od,
            morphology_index,
            spatial_index,
        )
        for start in range(0, len(values), batch_size):
            stop = min(start + batch_size, len(values))
            query = _embed(self, normalized[start:stop])
            morphology = _attention_context(
                self,
                query,
                morphology_index[start:stop],
            )
            spatial = _attention_context(
                self,
                query,
                spatial_index[start:stop],
            )
            morphology, spatial = _relational_messages(
                self,
                query,
                morphology,
                spatial,
            )
            hidden = _gelu(
                np.concatenate((query, morphology, spatial), axis=1)
                @ self.head0_weight.T
                + self.head0_bias
            )
            output[start:stop] = np.clip(
                (hidden @ self.head1_weight.T + self.head1_bias)[:, 0],
                0,
                self.target_od_upper,
            )
        return output, uncertainty

    def predict_accelerated(
        self,
        features: np.ndarray,
        reference_um_xyz: np.ndarray,
        *,
        batch_size: int = 4096,
        device: str = "auto",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Use optional PyTorch acceleration with the exact sealed weights."""

        try:
            from importlib import import_module

            import_module("torch")
        except ImportError:
            return self.predict(
                features,
                reference_um_xyz,
                batch_size=batch_size,
            )
        values, xyz = self._validate_query(features, reference_um_xyz, batch_size)
        normalized = (values - self.feature_mean) / self.feature_scale
        morphology_key = _morphology_neighbor_key(
            normalized[:, : self.morphology_width]
        )
        morphology_index = _accelerated_neighbor_index(
            morphology_key,
            self.bank_morphology_key,
            neighbors=16,
            device=device,
            batch_size=batch_size,
        )
        spatial_index = _neighbor_index(
            xyz,
            self.bank_reference_um_xyz,
            neighbors=16,
        )
        output = self._predict_torch(
            normalized,
            morphology_index,
            spatial_index,
            batch_size=batch_size,
            device=device,
        )
        return output, _neighbor_uncertainty(
            self.bank_od,
            morphology_index,
            spatial_index,
        )

    def predict_training_accelerated(
        self,
        *,
        batch_size: int = 4096,
        device: str = "auto",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict the fitted bank with self cells excluded from both contexts."""

        morphology = _neighbor_index(
            self.bank_morphology_key,
            self.bank_morphology_key,
            neighbors=16,
            aligned=True,
        )
        spatial = _neighbor_index(
            self.bank_reference_um_xyz,
            self.bank_reference_um_xyz,
            neighbors=16,
            aligned=True,
        )
        # Bank embeddings are already the output of normalization, projection,
        # layer normalization, and GELU.  The accelerated helper accepts them
        # directly to reproduce the training graph exactly.
        output = self._predict_embedded_torch(
            self.bank_embedding,
            morphology,
            spatial,
            batch_size=batch_size,
            device=device,
        )
        return output, _neighbor_uncertainty(self.bank_od, morphology, spatial)

    def predict_morphology_transfer(
        self,
        features: np.ndarray,
        *,
        neighbors: int = 16,
        batch_size: int = 16_384,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Transfer OD from training-only stain-neutral morphology neighbors.

        This is an inductive outcome-bank prediction: query cells contribute no
        outcomes, and exact feature-key matches are excluded to prevent a
        training cell from copying its own measured value.  The method is kept
        separate from the neural mean so a blend weight can be selected only on
        development mice and then frozen for untouched confirmation mice.
        """

        values = np.asarray(features, dtype=np.float32)
        if (
            values.ndim != 2
            or values.shape[1] != len(self.feature_mean)
            or not np.all(np.isfinite(values))
            or neighbors < 1
            or batch_size < 1
        ):
            raise ValueError("morphology-transfer query is malformed")
        normalized = (values - self.feature_mean) / self.feature_scale
        query_key = _morphology_neighbor_key(normalized[:, : self.morphology_width])
        return _inverse_distance_transfer(
            query_key,
            self.bank_morphology_key,
            self.bank_od,
            neighbors=neighbors,
            batch_size=batch_size,
        )

    def _predict_torch(
        self,
        normalized: np.ndarray,
        morphology_index: np.ndarray,
        spatial_index: np.ndarray,
        *,
        batch_size: int,
        device: str,
    ) -> np.ndarray:
        import torch

        selected = _torch_device(torch, device)
        tensor = lambda value: torch.from_numpy(  # noqa: E731
            np.asarray(value, dtype=np.float32)
        ).to(selected)
        raw = tensor(normalized)
        embedded = torch.nn.functional.gelu(
            torch.nn.functional.layer_norm(
                torch.nn.functional.linear(
                    raw,
                    tensor(self.embed_weight),
                    tensor(self.embed_bias),
                ),
                (len(self.embed_bias),),
                tensor(self.norm_weight),
                tensor(self.norm_bias),
            )
        )
        return self._predict_embedded_torch(
            embedded,
            morphology_index,
            spatial_index,
            batch_size=batch_size,
            device=str(selected),
        )

    def _predict_embedded_torch(
        self,
        embedded: object,
        morphology_index: np.ndarray,
        spatial_index: np.ndarray,
        *,
        batch_size: int,
        device: str,
    ) -> np.ndarray:
        import torch

        selected = _torch_device(torch, device)
        tensor = lambda value: torch.from_numpy(  # noqa: E731
            np.asarray(value, dtype=np.float32)
        ).to(selected)
        query_all = embedded if torch.is_tensor(embedded) else tensor(embedded)
        bank = tensor(self.bank_embedding)
        attention_weight = tensor(self.attention_weight)
        attention_bias = tensor(self.attention_bias)
        output_weight = tensor(self.attention_output_weight)
        output_bias = tensor(self.attention_output_bias)
        head0_weight = tensor(self.head0_weight)
        head0_bias = tensor(self.head0_bias)
        head1_weight = tensor(self.head1_weight)
        head1_bias = tensor(self.head1_bias)
        width = bank.shape[1]
        head_width = width // self.heads
        output = np.empty(len(morphology_index), dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, len(output), batch_size):
                stop = min(start + batch_size, len(output))
                query = query_all[start:stop]
                contexts = []
                for raw_index in (
                    morphology_index[start:stop],
                    spatial_index[start:stop],
                ):
                    index = torch.from_numpy(raw_index).to(selected)
                    neighbors = bank[index]
                    q = torch.nn.functional.linear(
                        query,
                        attention_weight[:width],
                        attention_bias[:width],
                    ).reshape(len(query), self.heads, head_width)
                    key = torch.nn.functional.linear(
                        neighbors,
                        attention_weight[width : 2 * width],
                        attention_bias[width : 2 * width],
                    ).reshape(
                        len(query),
                        neighbors.shape[1],
                        self.heads,
                        head_width,
                    )
                    value = torch.nn.functional.linear(
                        neighbors,
                        attention_weight[2 * width :],
                        attention_bias[2 * width :],
                    ).reshape(
                        len(query),
                        neighbors.shape[1],
                        self.heads,
                        head_width,
                    )
                    score = torch.einsum("bhd,bnhd->bhn", q, key) / math.sqrt(
                        head_width
                    )
                    weight = torch.softmax(score, dim=2)
                    context = torch.einsum(
                        "bhn,bnhd->bhd",
                        weight,
                        value,
                    ).reshape(len(query), width)
                    contexts.append(
                        torch.nn.functional.linear(
                            context,
                            output_weight,
                            output_bias,
                        )
                    )
                if self.provenance.get("relational_operator") == _GRAPH_OPERATOR:
                    contexts = [
                        torch.nn.functional.gelu(query + context)
                        for context in contexts
                    ]
                hidden = torch.nn.functional.gelu(
                    torch.nn.functional.linear(
                        torch.cat((query, *contexts), dim=1),
                        head0_weight,
                        head0_bias,
                    )
                )
                prediction = torch.nn.functional.linear(
                    hidden,
                    head1_weight,
                    head1_bias,
                )[:, 0]
                output[start:stop] = (
                    torch.clamp(
                        prediction,
                        0,
                        self.target_od_upper,
                    )
                    .cpu()
                    .numpy()
                )
        return output

    def _validate_query(
        self,
        features: np.ndarray,
        reference_um_xyz: np.ndarray,
        batch_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(features, dtype=np.float32)
        xyz = np.asarray(reference_um_xyz, dtype=np.float64)
        if (
            values.ndim != 2
            or values.shape[1] != len(self.feature_mean)
            or xyz.shape != (len(values), 3)
            or not np.all(np.isfinite(values))
            or not np.all(np.isfinite(xyz))
            or batch_size < 1
        ):
            raise ValueError("relational inference inputs are malformed")
        return values, xyz

    def _query_indices(
        self,
        normalized: np.ndarray,
        xyz: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        morphology_key = _morphology_neighbor_key(
            normalized[:, : self.morphology_width]
        )
        return (
            _neighbor_index(
                morphology_key,
                self.bank_morphology_key,
                neighbors=16,
            ),
            _neighbor_index(
                xyz,
                self.bank_reference_um_xyz,
                neighbors=16,
            ),
        )

    def save(self, path: Path | str) -> Path:
        """Write a fingerprinted, pickle-free model artifact."""

        target = Path(path)
        metadata = {
            "schema_version": 1,
            "architecture": self.architecture,
            "heads": self.heads,
            "morphology_width": self.morphology_width,
            "target_od_upper": self.target_od_upper,
            "provenance": self.provenance,
            "fingerprint": self.fingerprint,
        }

        def writer(stream) -> None:
            np.savez_compressed(
                stream,
                metadata_json=np.asarray(_canonical_json(metadata)),
                **_arrays(self),
            )

        return write_binary_atomic(target, writer)

    @classmethod
    def load(cls, path: Path | str) -> PortableRelationalRegressor:
        """Load and revalidate a portable relational artifact."""

        with np.load(Path(path), allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"]))
            if metadata.get("schema_version") != 1:
                raise ValueError("unsupported portable relational schema")
            return cls(
                architecture=str(metadata["architecture"]),
                heads=int(metadata["heads"]),
                morphology_width=int(metadata["morphology_width"]),
                target_od_upper=float(metadata["target_od_upper"]),
                provenance=dict(metadata["provenance"]),
                fingerprint=str(metadata["fingerprint"]),
                **{name: data[name] for name in _ARRAY_NAMES},
            )


def _relational_training_objective(
    prediction: object,
    target: object,
    *,
    profile: str,
    pair_index: object,
):
    """Return a training-only objective with optional expression-tail guards."""

    import torch

    if profile not in _RELATIONAL_LOSS_PROFILES:
        raise ValueError(f"unsupported relational loss profile: {profile}")
    if (
        not isinstance(prediction, torch.Tensor)
        or not isinstance(target, torch.Tensor)
        or not isinstance(pair_index, torch.Tensor)
        or prediction.ndim != 1
        or target.shape != prediction.shape
        or pair_index.shape != prediction.shape
        or pair_index.dtype != torch.long
    ):
        raise ValueError("relational objective tensors are malformed")
    per_cell = torch.nn.functional.smooth_l1_loss(
        prediction,
        target,
        reduction="none",
    )
    if profile == "huber_v1":
        return per_cell.mean()

    lower, upper = torch.quantile(
        target.detach(),
        torch.tensor((0.25, 0.90), device=target.device, dtype=target.dtype),
    )
    low = target <= lower
    high = target >= upper
    weight = torch.ones_like(target)
    weight = weight + 0.5 * low + 1.5 * high
    weight = weight + 1.5 * (low & (prediction > target))
    weight = weight + 2.5 * (high & (prediction < target))
    regression = torch.sum(per_cell * weight) / torch.sum(weight)

    low_flooding = torch.nn.functional.relu(prediction[low].mean() - target[low].mean())
    high_compression = torch.nn.functional.relu(
        target[high].mean() - prediction[high].mean()
    )
    target_delta = target - target[pair_index]
    comparable = torch.abs(target_delta) >= 0.1 * torch.clamp(upper - lower, min=1e-3)
    if torch.any(comparable):
        prediction_delta = prediction - prediction[pair_index]
        scale = torch.clamp(upper - lower, min=1e-3)
        ranking = torch.nn.functional.softplus(
            -prediction_delta[comparable] * torch.sign(target_delta[comparable]) / scale
        ).mean()
    else:
        ranking = torch.zeros((), device=target.device, dtype=target.dtype)
    return regression + 0.35 * (low_flooding + high_compression) + 0.025 * ranking


def fit_portable_relational(
    architecture: str,
    features: np.ndarray,
    measured_od: np.ndarray,
    train: np.ndarray,
    *,
    reference_um_xyz: np.ndarray,
    seed: int = 0,
    epochs: int = 40,
    loss_profile: str = "huber_v1",
) -> PortableRelationalRegressor:
    """Fit a leakage-safe relational regressor and export portable weights.

    Dual-bank attention uses morphology and spatial contexts directly.  The
    graph-transformer variant performs a residual nonlinear message update for
    each context before fusion.  The operator identity is sealed so older
    artifacts retain their original inference semantics.
    """

    try:
        import torch
        from torch import nn
    except ImportError as error:
        raise RuntimeError("relational fitting requires the 'cells' extra") from error
    if architecture not in _RELATIONAL_ARCHITECTURES:
        raise ValueError(f"unsupported relational architecture: {architecture}")
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(measured_od, dtype=np.float32)
    xyz = np.asarray(reference_um_xyz, dtype=np.float64)
    indices = np.asarray(train, dtype=np.int64)
    if (
        x.ndim != 2
        or y.shape != (len(x),)
        or xyz.shape != (len(x), 3)
        or epochs < 1
        or not np.all(np.isfinite(x))
        or not np.all(np.isfinite(xyz))
    ):
        raise ValueError("relational training arrays are malformed")
    if loss_profile not in _RELATIONAL_LOSS_PROFILES:
        raise ValueError(f"unsupported relational loss profile: {loss_profile}")
    valid = indices[np.isfinite(y[indices])]
    if len(valid) < 4:
        raise ValueError("relational fitting requires four measured cells")
    center = x[valid].mean(axis=0)
    scale = x[valid].std(axis=0)
    scale[scale < 1e-6] = 1
    train_values = np.ascontiguousarray((x[valid] - center) / scale)
    train_xyz = np.ascontiguousarray(xyz[valid])
    morphology_width = min(1536, x.shape[1])
    morphology_key = _morphology_neighbor_key(train_values[:, :morphology_width])
    morphology_index = _neighbor_index(
        morphology_key,
        morphology_key,
        neighbors=16,
        aligned=True,
    )
    spatial_index = _neighbor_index(
        train_xyz,
        train_xyz,
        neighbors=16,
        aligned=True,
    )
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_x = torch.from_numpy(train_values).to(device)
    train_y = torch.from_numpy(y[valid]).to(device)
    morphology_tensor = torch.from_numpy(morphology_index).to(device)
    spatial_tensor = torch.from_numpy(spatial_index).to(device)
    pair_index = torch.from_numpy(
        np.random.default_rng(seed ^ 0x5441494C).permutation(len(valid))
    ).to(device=device, dtype=torch.long)
    width = min(192, max(32, int(round(math.sqrt(x.shape[1]) * 3))))
    heads = 4 if width % 4 == 0 else 1
    operator = (
        _GRAPH_OPERATOR if architecture == "graph_transformer" else _DUAL_BANK_OPERATOR
    )

    class Network(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(x.shape[1], width)
            self.normalization = nn.LayerNorm(width)
            self.attention = nn.MultiheadAttention(
                width,
                num_heads=heads,
                batch_first=True,
            )
            self.head0 = nn.Linear(width * 3, width)
            self.head1 = nn.Linear(width, 1)

        def embed(self, value):  # type: ignore[no-untyped-def]
            return torch.nn.functional.gelu(self.normalization(self.linear(value)))

        def context(self, query, bank, index):  # type: ignore[no-untyped-def]
            neighbor = bank[index]
            context, _weight = self.attention(
                query[:, None, :],
                neighbor,
                neighbor,
                need_weights=False,
            )
            return context[:, 0, :]

        def forward(self, value, morph_index, space_index):  # type: ignore[no-untyped-def]
            embedded = self.embed(value)
            morphology = self.context(embedded, embedded, morph_index)
            spatial = self.context(embedded, embedded, space_index)
            if operator == _GRAPH_OPERATOR:
                morphology = torch.nn.functional.gelu(embedded + morphology)
                spatial = torch.nn.functional.gelu(embedded + spatial)
            hidden = torch.nn.functional.gelu(
                self.head0(torch.cat((embedded, morphology, spatial), dim=1))
            )
            return self.head1(hidden)[:, 0]

    model = Network().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    model.train()
    for _epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(train_x, morphology_tensor, spatial_tensor)
        loss = _relational_training_objective(
            prediction,
            train_y,
            profile=loss_profile,
            pair_index=pair_index,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    model.eval()
    with torch.inference_mode():
        bank_embedding = model.embed(train_x).cpu().numpy().astype(np.float32)

    def numpy(value: object) -> np.ndarray:
        return value.detach().cpu().numpy().astype(np.float32)

    target_upper = _robust_target_upper(y[valid])
    return PortableRelationalRegressor(
        architecture=architecture,
        feature_mean=center.astype(np.float32),
        feature_scale=scale.astype(np.float32),
        bank_embedding=bank_embedding,
        bank_morphology_key=morphology_key.astype(np.float32),
        bank_reference_um_xyz=train_xyz,
        bank_od=y[valid].astype(np.float32),
        embed_weight=numpy(model.linear.weight),
        embed_bias=numpy(model.linear.bias),
        norm_weight=numpy(model.normalization.weight),
        norm_bias=numpy(model.normalization.bias),
        attention_weight=numpy(model.attention.in_proj_weight),
        attention_bias=numpy(model.attention.in_proj_bias),
        attention_output_weight=numpy(model.attention.out_proj.weight),
        attention_output_bias=numpy(model.attention.out_proj.bias),
        head0_weight=numpy(model.head0.weight),
        head0_bias=numpy(model.head0.bias),
        head1_weight=numpy(model.head1.weight),
        head1_bias=numpy(model.head1.bias),
        heads=heads,
        morphology_width=morphology_width,
        target_od_upper=target_upper,
        provenance={
            "seed": seed,
            "epochs": epochs,
            "training_cells": len(valid),
            "neighbors_per_bank": min(16, len(valid) - 1),
            "prediction_banks": ("target-free-morphology+registered-reference-xyz-v1"),
            "neighbor_outcomes_used_for_mean": False,
            "neighbor_outcomes_used_for_uncertainty_only": True,
            "output_calibration": "training-od-q995-bound-v1",
            "relational_operator": operator,
            "training_loss_profile": loss_profile,
            "training_loss_tail_quantiles": (
                [0.25, 0.90] if loss_profile == "tail_rank_v1" else None
            ),
            "training_loss_tail_weights": (
                {
                    "low_tail": 0.5,
                    "high_tail": 1.5,
                    "low_overprediction": 1.5,
                    "high_underprediction": 2.5,
                    "directional_mean": 0.35,
                    "pairwise_rank": 0.025,
                }
                if loss_profile == "tail_rank_v1"
                else None
            ),
        },
    )


def _embed(
    model: PortableRelationalRegressor,
    normalized: np.ndarray,
) -> np.ndarray:
    projected = normalized @ model.embed_weight.T + model.embed_bias
    mean = projected.mean(axis=1, keepdims=True)
    variance = ((projected - mean) ** 2).mean(axis=1, keepdims=True)
    normalized_projection = (projected - mean) / np.sqrt(variance + 1e-5)
    return _gelu(normalized_projection * model.norm_weight + model.norm_bias)


def _attention_context(
    model: PortableRelationalRegressor,
    query: np.ndarray,
    index: np.ndarray,
) -> np.ndarray:
    bank = model.bank_embedding[index]
    width = query.shape[1]
    head_width = width // model.heads
    q = (query @ model.attention_weight[:width].T) + model.attention_bias[:width]
    key = (
        bank @ model.attention_weight[width : 2 * width].T
        + model.attention_bias[width : 2 * width]
    )
    value = (
        bank @ model.attention_weight[2 * width :].T + model.attention_bias[2 * width :]
    )
    q = q.reshape(len(query), model.heads, head_width)
    key = key.reshape(len(query), bank.shape[1], model.heads, head_width)
    value = value.reshape(len(query), bank.shape[1], model.heads, head_width)
    score = np.einsum("bhd,bnhd->bhn", q, key) / math.sqrt(head_width)
    score -= score.max(axis=2, keepdims=True)
    weight = np.exp(score)
    weight /= weight.sum(axis=2, keepdims=True)
    context = np.einsum("bhn,bnhd->bhd", weight, value).reshape(len(query), width)
    return context @ model.attention_output_weight.T + model.attention_output_bias


def _relational_messages(
    model: PortableRelationalRegressor,
    query: np.ndarray,
    morphology: np.ndarray,
    spatial: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the sealed relation operator while preserving legacy artifacts."""

    operator = str(model.provenance.get("relational_operator", _LEGACY_OPERATOR))
    if operator == _GRAPH_OPERATOR:
        return _gelu(query + morphology), _gelu(query + spatial)
    return morphology, spatial


def _morphology_neighbor_key(
    features: np.ndarray,
    components: int = 16,
) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("morphology neighbor features must be a finite matrix")
    width = min(components, values.shape[1])
    generator = np.random.default_rng(0x48495354)
    projection = generator.choice(
        np.asarray((-1.0, 1.0), dtype=np.float32),
        size=(values.shape[1], width),
    ) / math.sqrt(max(values.shape[1], 1))
    key = np.ascontiguousarray(values @ projection, dtype=np.float32)
    if width < components:
        key = np.pad(key, ((0, 0), (0, components - width)))
    return key


def _neighbor_index(
    query: np.ndarray,
    bank: np.ndarray,
    *,
    neighbors: int,
    aligned: bool = False,
) -> np.ndarray:
    from scipy.spatial import cKDTree

    query_values = np.asarray(query, dtype=np.float64)
    bank_values = np.asarray(bank, dtype=np.float64)
    if (
        query_values.ndim != 2
        or bank_values.ndim != 2
        or query_values.shape[1] != bank_values.shape[1]
        or not len(bank_values)
        or not np.all(np.isfinite(query_values))
        or not np.all(np.isfinite(bank_values))
        or neighbors < 1
        or (aligned and len(query_values) != len(bank_values))
    ):
        raise ValueError("neighbor query and bank are malformed")
    count = min(neighbors + int(aligned), len(bank_values))
    _distance, index = cKDTree(bank_values).query(query_values, k=count)
    index = np.asarray(index, dtype=np.int64)
    if index.ndim == 1:
        index = index[:, None]
    if aligned:
        keep = max(count - 1, 1)
        filtered = np.empty((len(query_values), keep), dtype=np.int64)
        for row in range(len(query_values)):
            candidate = index[row][index[row] != row]
            if not len(candidate):
                candidate = index[row, :1]
            if len(candidate) < keep:
                candidate = np.resize(candidate, keep)
            filtered[row] = candidate[:keep]
        index = filtered
    return np.ascontiguousarray(index)


def _inverse_distance_transfer(
    query: np.ndarray,
    bank: np.ndarray,
    bank_od: np.ndarray,
    *,
    neighbors: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return inverse-distance OD mean and dispersion from a frozen bank."""

    from scipy.spatial import cKDTree

    query_values = np.asarray(query, dtype=np.float64)
    bank_values = np.asarray(bank, dtype=np.float64)
    outcomes = np.asarray(bank_od, dtype=np.float64)
    if (
        query_values.ndim != 2
        or bank_values.ndim != 2
        or query_values.shape[1] != bank_values.shape[1]
        or outcomes.shape != (len(bank_values),)
        or len(bank_values) < 2
        or neighbors < 1
        or batch_size < 1
        or not np.all(np.isfinite(query_values))
        or not np.all(np.isfinite(bank_values))
        or not np.all(np.isfinite(outcomes))
    ):
        raise ValueError("morphology-transfer bank is malformed")
    tree = cKDTree(bank_values)
    count = min(neighbors + 1, len(bank_values))
    output = np.empty(len(query_values), dtype=np.float32)
    uncertainty = np.empty(len(query_values), dtype=np.float32)
    for start in range(0, len(query_values), batch_size):
        stop = min(start + batch_size, len(query_values))
        distance, index = tree.query(
            query_values[start:stop],
            k=count,
            workers=-1,
        )
        distance = np.asarray(distance, dtype=np.float64)
        index = np.asarray(index, dtype=np.int64)
        if distance.ndim == 1:
            distance = distance[:, None]
            index = index[:, None]
        valid = np.isfinite(distance)
        exact = distance[:, 0] <= 1e-7
        valid[:, 0] &= ~exact
        if count > neighbors:
            valid[~exact, neighbors:] = False
        weight = np.where(valid, 1.0 / np.maximum(distance, 1e-6), 0.0)
        denominator = weight.sum(axis=1)
        if np.any(denominator <= 0):
            raise ValueError("morphology-transfer query has no independent neighbor")
        selected = outcomes[index]
        mean = (selected * weight).sum(axis=1) / denominator
        variance = (np.square(selected - mean[:, None]) * weight).sum(
            axis=1
        ) / denominator
        output[start:stop] = mean.astype(np.float32)
        uncertainty[start:stop] = np.sqrt(np.maximum(variance, 0)).astype(np.float32)
    return output, uncertainty


def _accelerated_neighbor_index(
    query: np.ndarray,
    bank: np.ndarray,
    *,
    neighbors: int,
    device: str,
    batch_size: int,
) -> np.ndarray:
    """Find exact Euclidean neighbors with bounded optional GPU batches."""

    try:
        import torch
    except ImportError:
        return _neighbor_index(query, bank, neighbors=neighbors)
    selected = _torch_device(torch, device)
    if selected.type != "cuda":
        return _neighbor_index(query, bank, neighbors=neighbors)
    query_values = np.asarray(query, dtype=np.float32)
    bank_values = np.asarray(bank, dtype=np.float32)
    if (
        query_values.ndim != 2
        or bank_values.ndim != 2
        or query_values.shape[1] != bank_values.shape[1]
        or not len(bank_values)
        or neighbors < 1
        or batch_size < 1
        or not np.all(np.isfinite(query_values))
        or not np.all(np.isfinite(bank_values))
    ):
        raise ValueError("accelerated neighbor query and bank are malformed")
    count = min(neighbors, len(bank_values))
    bank_tensor = torch.from_numpy(bank_values).to(selected)
    bank_norm = torch.sum(bank_tensor.square(), dim=1)
    free_bytes: int | None = None
    try:
        free_bytes, _total_bytes = torch.cuda.mem_get_info(selected)
    except (AttributeError, RuntimeError, TypeError):
        pass
    effective_batch_size = _bounded_neighbor_batch_size(
        batch_size,
        len(bank_values),
        free_bytes=free_bytes,
    )
    output = np.empty((len(query_values), count), dtype=np.int64)
    with torch.inference_mode():
        for start in range(0, len(query_values), effective_batch_size):
            stop = min(start + effective_batch_size, len(query_values))
            value = torch.from_numpy(query_values[start:stop]).to(selected)
            # Maximizing the negative squared Euclidean distance is exactly
            # equivalent to the cKDTree query used by portable CPU inference.
            score = (
                2.0 * (value @ bank_tensor.T)
                - torch.sum(value.square(), dim=1)[:, None]
                - bank_norm[None, :]
            )
            output[start:stop] = (
                torch.topk(
                    score,
                    count,
                    dim=1,
                    largest=True,
                    sorted=True,
                )
                .indices.cpu()
                .numpy()
            )
    return output


def _bounded_neighbor_batch_size(
    requested: int,
    bank_rows: int,
    *,
    free_bytes: int | None = None,
) -> int:
    """Bound the exact-neighbor score matrix to a safe GPU working set."""

    if requested < 1 or bank_rows < 1:
        raise ValueError("neighbor batch dimensions must be positive")
    # The score is float32 ``batch x bank``. CUDA matrix multiplication and
    # subtraction can briefly retain several such arrays, so use no more than
    # one quarter of currently free memory and cap the score at 512 MiB.
    score_budget = 512 * 1024**2
    if free_bytes is not None:
        if free_bytes < 0:
            raise ValueError("free CUDA memory cannot be negative")
        score_budget = min(score_budget, max(16 * 1024**2, free_bytes // 4))
    bytes_per_query = bank_rows * np.dtype(np.float32).itemsize
    return max(1, min(requested, score_budget // bytes_per_query))


def _neighbor_uncertainty(
    bank_od: np.ndarray,
    morphology_index: np.ndarray,
    spatial_index: np.ndarray,
) -> np.ndarray:
    joined = np.concatenate((morphology_index, spatial_index), axis=1)
    return np.std(np.asarray(bank_od)[joined], axis=1).astype(np.float32)


def _robust_target_upper(values: np.ndarray) -> float:
    target = np.asarray(values, dtype=np.float64)
    target = target[np.isfinite(target) & (target >= 0)]
    if not len(target):
        raise ValueError("target calibration requires finite nonnegative OD")
    return max(
        float(np.quantile(target, 0.995)) * 1.25,
        float(np.quantile(target, 0.75)) * 2.0,
        1e-4,
    )


def _arrays(model: PortableRelationalRegressor) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(getattr(model, name), dtype=np.float32)
        for name in _ARRAY_NAMES
    }


def _fingerprint(model: PortableRelationalRegressor) -> str:
    digest = hashlib.sha256(b"histopia-portable-relational-v1\0")
    metadata = {
        "architecture": model.architecture,
        "heads": model.heads,
        "morphology_width": model.morphology_width,
        "target_od_upper": model.target_od_upper,
        "provenance": model.provenance,
    }
    digest.update(_canonical_json(metadata).encode())
    for name, value in sorted(_arrays(model).items()):
        array = np.ascontiguousarray(value)
        digest.update(name.encode())
        digest.update(array.dtype.str.encode())
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _gelu(values: np.ndarray) -> np.ndarray:
    from scipy.special import ndtr

    return np.asarray(values) * ndtr(values)


def _torch_device(torch: object, device: str):  # type: ignore[no-untyped-def]
    return torch.device(
        "cuda"
        if device == "auto" and torch.cuda.is_available()
        else ("cpu" if device == "auto" else device)
    )


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))
