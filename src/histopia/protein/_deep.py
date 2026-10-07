"""Optional deep protein-transfer candidates with lazy PyTorch imports.

Every relational candidate in this module is inductive: the training graph and
attention banks contain training cells only. Held-out cells are introduced
only after optimization, so their outcomes and morphology cannot leak through
message passing during validation.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from histopia._atomic import write_binary_atomic

_ARCHITECTURES = frozenset(
    {
        "graphsage",
        "gatv2",
        "cross_attention",
        "token_decoder",
        "multi_tower",
        "dual_bank_attention",
        "graph_transformer",
    }
)


@dataclass(frozen=True, slots=True)
class PortableSharedMultitaskRegressor:
    """Primary-antibody head exported from a shared multi-target trunk."""

    feature_mean: np.ndarray
    feature_scale: np.ndarray
    trunk0_weight: np.ndarray
    trunk0_bias: np.ndarray
    layernorm_weight: np.ndarray
    layernorm_bias: np.ndarray
    trunk1_weight: np.ndarray
    trunk1_bias: np.ndarray
    head_weight: np.ndarray
    head_bias: np.ndarray
    target_scale: float
    target_upper: float
    residual_scale: float
    provenance: dict[str, object]
    fingerprint: str | None = None

    def __post_init__(self) -> None:
        feature_count = len(np.asarray(self.feature_mean))
        width = len(np.asarray(self.trunk0_bias))
        expected_shapes = {
            "feature_scale": (feature_count,),
            "trunk0_weight": (feature_count, width),
            "layernorm_weight": (width,),
            "layernorm_bias": (width,),
            "trunk1_weight": (width, width),
            "trunk1_bias": (width,),
            "head_weight": (width, 1),
            "head_bias": (1,),
        }
        for name, shape in expected_shapes.items():
            if np.asarray(getattr(self, name)).shape != shape:
                raise ValueError(f"shared multitask {name} is malformed")
        if np.asarray(self.trunk0_weight).shape != (feature_count, width):
            raise ValueError("shared multitask trunk input is malformed")
        for name in ("target_scale", "target_upper", "residual_scale"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"shared multitask {name} must be positive")
        if self.target_upper < self.target_scale:
            raise ValueError("shared multitask target bound is inconsistent")
        for value in _shared_arrays(self).values():
            if not np.all(np.isfinite(value)):
                raise ValueError("shared multitask arrays must be finite")
        expected = _shared_fingerprint(self)
        if self.fingerprint is not None and self.fingerprint != expected:
            raise ValueError("shared multitask model fingerprint does not match")
        object.__setattr__(self, "fingerprint", expected)

    def predict(
        self,
        features: np.ndarray,
        *,
        batch_size: int = 4096,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict primary-target OD with sealed training-residual uncertainty."""

        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != len(self.feature_mean):
            raise ValueError("shared multitask inference features have the wrong shape")
        output = np.empty(len(values), dtype=np.float32)
        for start in range(0, len(values), batch_size):
            stop = min(start + batch_size, len(values))
            normalized = (values[start:stop] - self.feature_mean) / self.feature_scale
            hidden = normalized @ self.trunk0_weight + self.trunk0_bias
            mean = hidden.mean(axis=1, keepdims=True)
            variance = ((hidden - mean) ** 2).mean(axis=1, keepdims=True)
            hidden = (hidden - mean) / np.sqrt(
                variance + 1e-5
            ) * self.layernorm_weight + self.layernorm_bias
            hidden = _portable_gelu(hidden)
            hidden = _portable_gelu(hidden @ self.trunk1_weight + self.trunk1_bias)
            transformed = (hidden @ self.head_weight + self.head_bias)[:, 0]
            output[start:stop] = np.clip(
                np.sinh(transformed) * self.target_scale,
                0,
                self.target_upper,
            )
        uncertainty = np.full(len(values), self.residual_scale, dtype=np.float32)
        return output, uncertainty

    def predict_accelerated(
        self,
        features: np.ndarray,
        *,
        batch_size: int = 16_384,
        device: str = "auto",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Use optional PyTorch with the exact sealed primary-head weights."""

        try:
            import torch
        except ImportError:
            return self.predict(features, batch_size=batch_size)
        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != len(self.feature_mean):
            raise ValueError("shared multitask inference features have the wrong shape")
        selected = torch.device(
            "cuda"
            if device == "auto" and torch.cuda.is_available()
            else ("cpu" if device == "auto" else device)
        )
        tensor = lambda value: torch.from_numpy(  # noqa: E731
            np.asarray(value, dtype=np.float32)
        ).to(selected)
        center, scale = tensor(self.feature_mean), tensor(self.feature_scale)
        trunk0_weight, trunk0_bias = (
            tensor(self.trunk0_weight),
            tensor(self.trunk0_bias),
        )
        layernorm_weight, layernorm_bias = (
            tensor(self.layernorm_weight),
            tensor(self.layernorm_bias),
        )
        trunk1_weight, trunk1_bias = (
            tensor(self.trunk1_weight),
            tensor(self.trunk1_bias),
        )
        head_weight, head_bias = tensor(self.head_weight), tensor(self.head_bias)
        output = np.empty(len(values), dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, len(values), batch_size):
                stop = min(start + batch_size, len(values))
                normalized = (tensor(values[start:stop]) - center) / scale
                hidden = normalized @ trunk0_weight + trunk0_bias
                hidden = torch.nn.functional.layer_norm(
                    hidden,
                    (hidden.shape[1],),
                    layernorm_weight,
                    layernorm_bias,
                    1e-5,
                )
                hidden = torch.nn.functional.gelu(hidden, approximate="tanh")
                hidden = torch.nn.functional.gelu(
                    hidden @ trunk1_weight + trunk1_bias,
                    approximate="tanh",
                )
                transformed = (hidden @ head_weight + head_bias)[:, 0]
                output[start:stop] = (
                    torch.clamp(
                        torch.sinh(transformed) * self.target_scale,
                        0,
                        self.target_upper,
                    )
                    .cpu()
                    .numpy()
                )
        uncertainty = np.full(len(values), self.residual_scale, dtype=np.float32)
        return output, uncertainty

    def save(self, path: Path | str) -> Path:
        target = Path(path)
        metadata = {
            "schema_version": 1,
            "architecture": "shared_multitask",
            "target_scale": self.target_scale,
            "target_upper": self.target_upper,
            "residual_scale": self.residual_scale,
            "provenance": self.provenance,
            "fingerprint": self.fingerprint,
        }

        def writer(stream) -> None:
            np.savez_compressed(
                stream,
                metadata_json=np.asarray(
                    json.dumps(metadata, sort_keys=True, separators=(",", ":"))
                ),
                **_shared_arrays(self),
            )

        return write_binary_atomic(target, writer)

    @classmethod
    def load(cls, path: Path | str) -> PortableSharedMultitaskRegressor:
        with np.load(Path(path), allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"]))
            if (
                metadata.get("schema_version") != 1
                or metadata.get("architecture") != "shared_multitask"
            ):
                raise ValueError("unsupported portable shared multitask schema")
            return cls(
                **{name: data[name] for name in _SHARED_ARRAY_NAMES},
                target_scale=float(metadata["target_scale"]),
                target_upper=float(metadata["target_upper"]),
                residual_scale=float(metadata["residual_scale"]),
                provenance=dict(metadata["provenance"]),
                fingerprint=str(metadata["fingerprint"]),
            )


_SHARED_ARRAY_NAMES = (
    "feature_mean",
    "feature_scale",
    "trunk0_weight",
    "trunk0_bias",
    "layernorm_weight",
    "layernorm_bias",
    "trunk1_weight",
    "trunk1_bias",
    "head_weight",
    "head_bias",
)


def _shared_arrays(model: PortableSharedMultitaskRegressor) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(getattr(model, name), dtype=np.float32)
        for name in _SHARED_ARRAY_NAMES
    }


def _shared_fingerprint(model: PortableSharedMultitaskRegressor) -> str:
    digest = hashlib.sha256(b"portable-shared-multitask-v1\0")
    metadata = {
        "target_scale": model.target_scale,
        "target_upper": model.target_upper,
        "residual_scale": model.residual_scale,
        "provenance": model.provenance,
    }
    digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode())
    for name, value in _shared_arrays(model).items():
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def _portable_gelu(value: np.ndarray) -> np.ndarray:
    return (
        0.5
        * value
        * (1 + np.tanh(math.sqrt(2 / math.pi) * (value + 0.044715 * value**3)))
    )


def out_of_fold_deep_predictions(
    architecture: str,
    features: np.ndarray,
    measured_od: np.ndarray,
    groups: np.ndarray,
    *,
    reference_um_xyz: np.ndarray | None = None,
    section_groups: np.ndarray | None = None,
    seed: int = 0,
    epochs: int = 40,
) -> np.ndarray:
    """Return honest leave-one-group-out predictions in original row order."""

    from histopia.protein._splits import leave_one_group_out

    group_values = np.asarray(groups)
    if group_values.shape != (len(features),):
        raise ValueError("deep prediction groups must align with feature rows")
    output = np.full(len(group_values), np.nan, dtype=np.float32)
    for train, test in leave_one_group_out(group_values):
        output[test] = fit_deep_candidate(
            architecture,
            features,
            measured_od,
            train,
            test,
            reference_um_xyz=reference_um_xyz,
            section_groups=section_groups,
            seed=seed,
            epochs=epochs,
        )
    if np.any(~np.isfinite(output)):
        raise ValueError("out-of-fold deep prediction left uncovered rows")
    return output


def fit_deep_candidate(
    architecture: str,
    features: np.ndarray,
    measured_od: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    *,
    reference_um_xyz: np.ndarray | None = None,
    section_groups: np.ndarray | None = None,
    seed: int = 0,
    epochs: int = 40,
) -> np.ndarray:
    """Fit one leakage-safe neural candidate and predict held-out cells.

    ``graphsage``, ``gatv2``, and ``graph_transformer`` build their optimization
    graph from measured training cells only. ``dual_bank_attention`` has two
    outcome-free banks: registered-space neighbors and morphology neighbors.
    Test cells query those frozen training banks after fitting.
    """

    try:
        import torch
        from torch import nn
    except ImportError as error:
        raise RuntimeError(
            "deep protein candidates require the 'cells' extra"
        ) from error
    if architecture not in _ARCHITECTURES:
        raise ValueError(f"unsupported deep protein architecture: {architecture}")
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(measured_od, dtype=np.float32)
    train = np.asarray(train, dtype=np.int64)
    test = np.asarray(test, dtype=np.int64)
    if (
        x.ndim != 2
        or y.shape != (len(x),)
        or train.ndim != 1
        or test.ndim != 1
        or np.intersect1d(train, test).size
    ):
        raise ValueError("deep candidate train and test arrays do not align")
    valid = train[np.isfinite(y[train])]
    if len(valid) < 4:
        raise ValueError("deep candidate requires four measured training cells")
    if architecture == "cross_attention":
        from histopia.protein._attention import fit_portable_cross_attention

        model = fit_portable_cross_attention(x, y, valid, seed=seed, epochs=epochs)
        predicted, _uncertainty = model.predict_accelerated(x[test])
        return _bounded_od_prediction(predicted, y[valid])
    if architecture == "multi_tower":
        from histopia.protein._advanced import (
            feature_group_slices,
            fit_portable_multi_tower,
        )

        schema = (
            "native-hdab-neutral-cell-multiscale-v3"
            if x.shape[1] == 1844
            else "native-hdab-neutral-spatial-uni2h-v2"
        )
        model = fit_portable_multi_tower(
            x,
            y,
            valid,
            group_slices=feature_group_slices(schema, x.shape[1]),
            section_groups=section_groups,
            seed=seed,
            epochs=epochs,
        )
        predicted, _uncertainty = model.predict_accelerated(x[test])
        return _bounded_od_prediction(predicted, y[valid])
    if architecture in {"dual_bank_attention", "graph_transformer"}:
        from histopia.protein._relational import fit_portable_relational

        if reference_um_xyz is None:
            raise ValueError(f"{architecture} requires registered cell coordinates")
        model = fit_portable_relational(
            architecture,
            x,
            y,
            valid,
            reference_um_xyz=reference_um_xyz,
            seed=seed,
            epochs=epochs,
        )
        predicted, _uncertainty = model.predict_accelerated(
            x[test],
            np.asarray(reference_um_xyz)[test],
        )
        return _bounded_od_prediction(predicted, y[valid])

    spatial_architecture = architecture in {
        "graphsage",
        "gatv2",
        "dual_bank_attention",
        "graph_transformer",
    }
    if spatial_architecture and reference_um_xyz is None:
        raise ValueError(f"{architecture} requires registered cell coordinates")
    xyz = (
        None
        if reference_um_xyz is None
        else _validated_coordinates(reference_um_xyz, len(x))
    )
    center = x[valid].mean(0)
    scale = x[valid].std(0)
    scale[scale < 1e-6] = 1
    train_values = np.ascontiguousarray((x[valid] - center) / scale)
    test_values = np.ascontiguousarray((x[test] - center) / scale)
    morphology_width = min(1536, x.shape[1])
    train_key = _morphology_neighbor_key(train_values[:, :morphology_width])
    test_key = _morphology_neighbor_key(test_values[:, :morphology_width])
    morphology_train = _cross_neighbor_index(
        train_key,
        train_key,
        neighbors=16,
        aligned=True,
    )
    morphology_test = _cross_neighbor_index(
        test_key,
        train_key,
        neighbors=16,
    )
    if xyz is None:
        spatial_train = spatial_test = None
    else:
        spatial_train = _cross_neighbor_index(
            xyz[valid],
            xyz[valid],
            neighbors=16,
            aligned=True,
        )
        spatial_test = _cross_neighbor_index(
            xyz[test],
            xyz[valid],
            neighbors=16,
        )

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_x = torch.from_numpy(train_values).to(device)
    test_x = torch.from_numpy(test_values).to(device)
    train_y = torch.from_numpy(y[valid]).to(device)
    morph_train_index = torch.from_numpy(morphology_train).to(device)
    morph_test_index = torch.from_numpy(morphology_test).to(device)
    spatial_train_index = (
        None if spatial_train is None else torch.from_numpy(spatial_train).to(device)
    )
    spatial_test_index = (
        None if spatial_test is None else torch.from_numpy(spatial_test).to(device)
    )
    width = min(192, max(32, int(round(math.sqrt(x.shape[1]) * 3))))
    attention_contexts = (
        2 if architecture in {"dual_bank_attention", "graph_transformer"} else 1
    )

    class Candidate(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Sequential(
                nn.Linear(x.shape[1], width),
                nn.LayerNorm(width),
                nn.GELU(),
            )
            self.attention = nn.MultiheadAttention(
                width,
                num_heads=4 if width % 4 == 0 else 1,
                batch_first=True,
            )
            self.head = nn.Sequential(
                nn.Linear(width * (1 + attention_contexts), width),
                nn.GELU(),
                nn.Dropout(0.05),
                nn.Linear(width, 1),
            )

        def _context(self, query, bank, index, *, attentive):  # type: ignore[no-untyped-def]
            neighbor = bank[index]
            if attentive:
                context, _weights = self.attention(
                    query[:, None, :], neighbor, neighbor, need_weights=False
                )
                return context[:, 0, :]
            return neighbor.mean(dim=1)

        def forward(  # type: ignore[no-untyped-def]
            self,
            query,
            bank,
            morphology_index,
            spatial_index=None,
        ):
            query_embedding = self.embed(query)
            bank_embedding = self.embed(bank)
            if architecture == "token_decoder":
                contexts = [
                    self._context(
                        query_embedding,
                        bank_embedding,
                        morphology_index,
                        attentive=True,
                    )
                ]
            elif architecture == "graphsage":
                contexts = [
                    self._context(
                        query_embedding,
                        bank_embedding,
                        spatial_index,
                        attentive=False,
                    )
                ]
            elif architecture == "gatv2":
                contexts = [
                    self._context(
                        query_embedding,
                        bank_embedding,
                        spatial_index,
                        attentive=True,
                    )
                ]
            else:
                contexts = [
                    self._context(
                        query_embedding,
                        bank_embedding,
                        morphology_index,
                        attentive=True,
                    ),
                    self._context(
                        query_embedding,
                        bank_embedding,
                        spatial_index,
                        attentive=True,
                    ),
                ]
            return self.head(torch.cat((query_embedding, *contexts), dim=1))[:, 0]

    model = Candidate().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    model.train()
    for _epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(
            train_x,
            train_x,
            morph_train_index,
            spatial_train_index,
        )
        loss = torch.nn.functional.smooth_l1_loss(prediction, train_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    model.eval()
    with torch.inference_mode():
        prediction = model(
            test_x,
            train_x,
            morph_test_index,
            spatial_test_index,
        )
    return _bounded_od_prediction(prediction.cpu().numpy(), y[valid])


def fit_portable_shared_multitask(
    features: np.ndarray,
    measured_od: np.ndarray,
    train: np.ndarray,
    *,
    auxiliary: tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...],
    auxiliary_target_ids: tuple[str, ...] | None = None,
    seed: int = 0,
    epochs: int = 40,
    batch_size: int = 2048,
) -> PortableSharedMultitaskRegressor:
    """Fit a shared trunk and export its primary-antibody inference head.

    Auxiliary targets regularize only the target-free representation. Each
    antibody retains its own outcome head and scale, so incompatible OD scales
    are never pooled. Only the primary head is sealed for deployment. The
    caller must apply the same held-out-cohort exclusion to every auxiliary
    table.
    """

    try:
        import torch
        from torch import nn
    except ImportError as error:
        raise RuntimeError(
            "shared multi-target fitting requires the 'cells' extra"
        ) from error
    primary_x = np.asarray(features, dtype=np.float32)
    primary_y = np.asarray(measured_od, dtype=np.float32)
    train = np.asarray(train, dtype=np.int64)
    if (
        primary_x.ndim != 2
        or primary_y.shape != (len(primary_x),)
        or epochs < 1
        or batch_size < 1
    ):
        raise ValueError("shared multi-target primary arrays are malformed")
    auxiliary_ids = auxiliary_target_ids or tuple(
        f"auxiliary-{index + 1}" for index in range(len(auxiliary))
    )
    if len(auxiliary_ids) != len(auxiliary) or any(
        not isinstance(value, str) or not value for value in auxiliary_ids
    ):
        raise ValueError("shared multi-target auxiliary identities are malformed")
    datasets: list[tuple[np.ndarray, np.ndarray]] = []
    used_auxiliary_ids: list[str] = []
    primary_valid = train[np.isfinite(primary_y[train])]
    if len(primary_valid) < 4:
        raise ValueError("shared multi-target fitting needs four primary cells")
    datasets.append((primary_x[primary_valid], primary_y[primary_valid]))
    for target_id, (raw_x, raw_y, raw_train) in zip(
        auxiliary_ids, auxiliary, strict=True
    ):
        aux_x = np.asarray(raw_x, dtype=np.float32)
        aux_y = np.asarray(raw_y, dtype=np.float32)
        aux_train = np.asarray(raw_train, dtype=np.int64)
        if (
            aux_x.ndim != 2
            or aux_x.shape[1] != primary_x.shape[1]
            or aux_y.shape != (len(aux_x),)
        ):
            raise ValueError("shared multi-target auxiliary arrays are malformed")
        aux_valid = aux_train[np.isfinite(aux_y[aux_train])]
        if len(aux_valid) >= 4:
            datasets.append((aux_x[aux_valid], aux_y[aux_valid]))
            used_auxiliary_ids.append(target_id)
    if len(datasets) < 2:
        raise ValueError("shared multi-target fitting needs an auxiliary antibody")
    normalization_sample = np.concatenate(
        [
            values[
                np.linspace(
                    0,
                    len(values) - 1,
                    min(len(values), 4096),
                    dtype=np.int64,
                )
            ]
            for values, _target in datasets
        ],
        axis=0,
    )
    center = normalization_sample.mean(axis=0)
    scale = normalization_sample.std(axis=0)
    scale[scale < 1e-6] = 1
    transformed: list[tuple[np.ndarray, np.ndarray, float]] = []
    for values, target in datasets:
        positive = target[target > 0]
        target_scale = max(
            (
                float(np.quantile(positive, 0.75))
                if len(positive)
                else float(target.max())
            ),
            1e-4,
        )
        transformed.append(
            (
                np.ascontiguousarray((values - center) / scale),
                np.arcsinh(np.maximum(target, 0) / target_scale).astype(np.float32),
                target_scale,
            )
        )
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    width = min(256, max(64, int(round(math.sqrt(primary_x.shape[1]) * 4))))

    class SharedNetwork(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.trunk = nn.Sequential(
                nn.Linear(primary_x.shape[1], width),
                nn.LayerNorm(width),
                nn.GELU(approximate="tanh"),
                nn.Dropout(0.05),
                nn.Linear(width, width),
                nn.GELU(approximate="tanh"),
            )
            self.heads = nn.ModuleList(nn.Linear(width, 1) for _ in transformed)

        def forward(self, value, target_index):  # type: ignore[no-untyped-def]
            return self.heads[target_index](self.trunk(value))[:, 0]

    model = SharedNetwork().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-4)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    tensors = [
        (torch.from_numpy(value).to(device), torch.from_numpy(target).to(device))
        for value, target, _scale in transformed
    ]
    model.train()
    per_target_batch = max(batch_size // len(tensors), 32)
    for _epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for target_index, (value, target) in enumerate(tensors):
            selected = torch.randint(
                len(value),
                (min(per_target_batch, len(value)),),
                generator=generator,
                device=device,
            )
            predicted = model(value[selected], target_index)
            truth = target[selected]
            regression = torch.nn.functional.smooth_l1_loss(predicted, truth)
            shifted = torch.roll(torch.arange(len(selected), device=device), 1)
            direction = torch.sign(truth - truth[shifted])
            comparable = direction != 0
            rank = (
                torch.nn.functional.softplus(
                    -(predicted - predicted[shifted]) * direction
                )[comparable].mean()
                if torch.any(comparable)
                else torch.zeros((), device=device)
            )
            losses.append(regression + 0.05 * rank)
        loss = torch.stack(losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    model.eval()
    with torch.inference_mode():
        primary_prediction = torch.sinh(model(tensors[0][0], 0)) * transformed[0][2]
    bounded = _bounded_od_prediction(
        primary_prediction.cpu().numpy(), primary_y[primary_valid]
    )
    target_upper = max(
        float(np.quantile(primary_y[primary_valid], 0.995)) * 1.25,
        float(np.quantile(primary_y[primary_valid], 0.75)) * 2.0,
        1e-4,
    )
    residual = bounded - primary_y[primary_valid]
    residual_scale = max(
        float(np.sqrt(np.mean(np.square(residual), dtype=np.float64))),
        1e-4,
    )
    linear0 = model.trunk[0]
    layernorm = model.trunk[1]
    linear1 = model.trunk[4]
    head = model.heads[0]

    def numpy(value: object) -> np.ndarray:
        return value.detach().cpu().numpy().astype(np.float32)

    return PortableSharedMultitaskRegressor(
        feature_mean=center.astype(np.float32),
        feature_scale=scale.astype(np.float32),
        trunk0_weight=numpy(linear0.weight.T),
        trunk0_bias=numpy(linear0.bias),
        layernorm_weight=numpy(layernorm.weight),
        layernorm_bias=numpy(layernorm.bias),
        trunk1_weight=numpy(linear1.weight.T),
        trunk1_bias=numpy(linear1.bias),
        head_weight=numpy(head.weight.T),
        head_bias=numpy(head.bias),
        target_scale=transformed[0][2],
        target_upper=target_upper,
        residual_scale=residual_scale,
        provenance={
            "seed": seed,
            "epochs": epochs,
            "batch_size": batch_size,
            "training_cells": len(primary_valid),
            "auxiliary_targets": used_auxiliary_ids,
            "auxiliary_target_count": len(used_auxiliary_ids),
            "target_transform": "asinh-training-q75-v1",
            "output_calibration": "training-od-q995-bound-v1",
            "uncertainty": "primary-training-rmse-v1",
            "loss": "mean-per-target-huber+pairwise-rank-v1",
        },
    )


def fit_shared_multitask_candidate(
    features: np.ndarray,
    measured_od: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    *,
    auxiliary: tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...],
    seed: int = 0,
    epochs: int = 40,
    batch_size: int = 2048,
) -> np.ndarray:
    """Fit and evaluate the portable shared-trunk primary target head."""

    train_rows = np.asarray(train, dtype=np.int64)
    test_rows = np.asarray(test, dtype=np.int64)
    if np.intersect1d(train_rows, test_rows).size:
        raise ValueError("shared multi-target train and test rows overlap")
    model = fit_portable_shared_multitask(
        features,
        measured_od,
        train_rows,
        auxiliary=auxiliary,
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
    )
    return model.predict(np.asarray(features)[test_rows])[0]


def _bounded_od_prediction(
    prediction: np.ndarray,
    training_od: np.ndarray,
) -> np.ndarray:
    """Apply a training-only robust physical range to neural OD outputs."""

    values = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(training_od, dtype=np.float64)
    truth = truth[np.isfinite(truth) & (truth >= 0)]
    if not len(truth) or not np.all(np.isfinite(values)):
        raise ValueError("neural OD prediction or calibration is non-finite")
    upper = max(
        float(np.quantile(truth, 0.995)) * 1.25,
        float(np.quantile(truth, 0.75)) * 2.0,
        1e-4,
    )
    return np.clip(values, 0, upper).astype(np.float32)


def _validated_coordinates(values: np.ndarray, count: int) -> np.ndarray:
    xyz = np.asarray(values, dtype=np.float64)
    if xyz.shape != (count, 3) or not np.all(np.isfinite(xyz)):
        raise ValueError("registered cell coordinates must have shape (cells, 3)")
    return xyz


def _morphology_neighbor_key(features: np.ndarray, components: int = 16) -> np.ndarray:
    """Project morphology with a fixed target-free map for neighbor retrieval."""

    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("morphology neighbor features must be a finite matrix")
    width = min(components, values.shape[1])
    generator = np.random.default_rng(0x48495354)
    projection = generator.choice(
        np.asarray((-1.0, 1.0), dtype=np.float32),
        size=(values.shape[1], width),
    ) / math.sqrt(max(values.shape[1], 1))
    return np.ascontiguousarray(values @ projection, dtype=np.float32)


def _cross_neighbor_index(
    query: np.ndarray,
    bank: np.ndarray,
    *,
    neighbors: int,
    aligned: bool = False,
) -> np.ndarray:
    """Find deterministic bank indices, optionally excluding aligned selves."""

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
        row = np.arange(len(query_values), dtype=np.int64)[:, None]
        filtered = np.empty((len(query_values), max(count - 1, 1)), dtype=np.int64)
        for offset in range(len(query_values)):
            candidate = index[offset][index[offset] != row[offset, 0]]
            if not len(candidate):
                candidate = index[offset, :1]
            if len(candidate) < filtered.shape[1]:
                candidate = np.resize(candidate, filtered.shape[1])
            filtered[offset] = candidate[: filtered.shape[1]]
        index = filtered
    return np.ascontiguousarray(index)


def _spatial_neighbor_index(
    reference_um_xyz: np.ndarray, *, neighbors: int
) -> np.ndarray:
    """Build a deterministic registered-space kNN graph without outcomes."""

    xyz = np.asarray(reference_um_xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.all(np.isfinite(xyz)):
        raise ValueError("registered cell coordinates must have shape (cells, 3)")
    if len(xyz) < 2 or neighbors < 1:
        raise ValueError("a spatial graph requires at least two cells")
    return _cross_neighbor_index(xyz, xyz, neighbors=neighbors, aligned=True)
