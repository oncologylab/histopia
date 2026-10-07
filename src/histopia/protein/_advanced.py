"""Portable GPU-trained cell-aware protein regressors."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from histopia._atomic import write_binary_atomic


@dataclass(frozen=True, slots=True)
class PortableMultiTowerRegressor:
    """Feature-group towers, residual fusion, and heteroscedastic OD head."""

    feature_mean: np.ndarray
    feature_scale: np.ndarray
    group_slices: dict[str, tuple[int, int]]
    tower_weights: tuple[np.ndarray, ...]
    tower_biases: tuple[np.ndarray, ...]
    fusion_weight: np.ndarray
    fusion_bias: np.ndarray
    residual_weights: tuple[np.ndarray, ...]
    residual_biases: tuple[np.ndarray, ...]
    mean_weight: np.ndarray
    mean_bias: np.ndarray
    variance_weight: np.ndarray
    variance_bias: np.ndarray
    target_scale: float
    provenance: dict[str, object]
    morphology_bank_key: np.ndarray | None = None
    morphology_bank_od: np.ndarray | None = None
    morphology_transfer_slice: tuple[int, int] | None = None
    fingerprint: str | None = None

    def __post_init__(self) -> None:
        count = len(np.asarray(self.feature_mean))
        if np.asarray(self.feature_scale).shape != (count,):
            raise ValueError("multi-tower feature normalization is malformed")
        groups = tuple(sorted(self.group_slices))
        if len(groups) != len(self.tower_weights) or len(groups) != len(
            self.tower_biases
        ):
            raise ValueError("multi-tower feature groups and towers do not align")
        tower_width = 0
        covered = np.zeros(count, dtype=bool)
        for name, weight, bias in zip(
            groups, self.tower_weights, self.tower_biases, strict=True
        ):
            start, stop = self.group_slices[name]
            matrix, offset = np.asarray(weight), np.asarray(bias)
            if (
                start < 0
                or stop <= start
                or stop > count
                or np.any(covered[start:stop])
                or matrix.shape[0] != stop - start
                or offset.shape != (matrix.shape[1],)
            ):
                raise ValueError("multi-tower group geometry is malformed")
            covered[start:stop] = True
            tower_width += matrix.shape[1]
        if not np.all(covered):
            raise ValueError("multi-tower groups must cover every feature exactly once")
        fusion = np.asarray(self.fusion_weight)
        if fusion.ndim != 2 or fusion.shape[0] != tower_width:
            raise ValueError("multi-tower fusion weight is malformed")
        width = fusion.shape[1]
        if np.asarray(self.fusion_bias).shape != (width,):
            raise ValueError("multi-tower fusion bias is malformed")
        if len(self.residual_weights) != len(self.residual_biases):
            raise ValueError("multi-tower residual layers do not align")
        for weight, bias in zip(
            self.residual_weights, self.residual_biases, strict=True
        ):
            if np.asarray(weight).shape != (width, width) or np.asarray(bias).shape != (
                width,
            ):
                raise ValueError("multi-tower residual layer is malformed")
        for name, weight, bias in (
            ("mean", self.mean_weight, self.mean_bias),
            ("variance", self.variance_weight, self.variance_bias),
        ):
            if np.asarray(weight).shape != (width, 1) or np.asarray(bias).shape != (1,):
                raise ValueError(f"multi-tower {name} head is malformed")
        if not math.isfinite(self.target_scale) or self.target_scale <= 0:
            raise ValueError("multi-tower target scale must be positive and finite")
        transfer_values = (
            self.morphology_bank_key,
            self.morphology_bank_od,
            self.morphology_transfer_slice,
        )
        if any(value is not None for value in transfer_values):
            if any(value is None for value in transfer_values):
                raise ValueError("multi-tower morphology-transfer bank is incomplete")
            assert self.morphology_bank_key is not None
            assert self.morphology_bank_od is not None
            assert self.morphology_transfer_slice is not None
            bank_rows = len(np.asarray(self.morphology_bank_od))
            if (
                np.asarray(self.morphology_bank_key).shape != (bank_rows, 16)
                or bank_rows < 2
                or self.morphology_transfer_slice != self.group_slices.get("morphology")
            ):
                raise ValueError("multi-tower morphology-transfer bank is malformed")
        for value in _model_arrays(self).values():
            if not np.all(np.isfinite(value)):
                raise ValueError("multi-tower model arrays must be finite")
        expected = _fingerprint(self)
        if self.fingerprint is not None and self.fingerprint != expected:
            raise ValueError("multi-tower model fingerprint does not match")
        object.__setattr__(self, "fingerprint", expected)

    def with_morphology_transfer_bank(
        self,
        features: np.ndarray,
        measured_od: np.ndarray,
    ) -> PortableMultiTowerRegressor:
        """Seal a target-free morphology key and training-only OD outcome bank."""

        values = np.asarray(features, dtype=np.float32)
        outcomes = np.asarray(measured_od, dtype=np.float32)
        morphology_slice = self.group_slices.get("morphology")
        if (
            values.ndim != 2
            or values.shape[1] != len(self.feature_mean)
            or outcomes.shape != (len(values),)
            or len(values) < 2
            or morphology_slice is None
            or not np.all(np.isfinite(values))
            or not np.all(np.isfinite(outcomes))
            or np.any(outcomes < 0)
        ):
            raise ValueError("multi-tower morphology-transfer training bank is invalid")
        left, right = morphology_slice
        normalized = (
            values[:, left:right] - self.feature_mean[left:right]
        ) / self.feature_scale[left:right]
        from histopia.protein._relational import _morphology_neighbor_key

        return replace(
            self,
            morphology_bank_key=_morphology_neighbor_key(normalized),
            morphology_bank_od=outcomes,
            morphology_transfer_slice=(int(left), int(right)),
            fingerprint=None,
        )

    def predict_morphology_transfer(
        self,
        features: np.ndarray,
        *,
        neighbors: int = 16,
        batch_size: int = 16_384,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Transfer OD only from the model's sealed stain-neutral training bank."""

        if (
            self.morphology_bank_key is None
            or self.morphology_bank_od is None
            or self.morphology_transfer_slice is None
        ):
            raise ValueError("multi-tower model has no morphology-transfer bank")
        values = np.asarray(features, dtype=np.float32)
        if (
            values.ndim != 2
            or values.shape[1] != len(self.feature_mean)
            or not np.all(np.isfinite(values))
            or neighbors < 1
            or batch_size < 1
        ):
            raise ValueError("multi-tower morphology-transfer query is malformed")
        left, right = self.morphology_transfer_slice
        normalized = (
            values[:, left:right] - self.feature_mean[left:right]
        ) / self.feature_scale[left:right]
        from histopia.protein._relational import (
            _inverse_distance_transfer,
            _morphology_neighbor_key,
        )

        return _inverse_distance_transfer(
            _morphology_neighbor_key(normalized),
            self.morphology_bank_key,
            self.morphology_bank_od,
            neighbors=neighbors,
            batch_size=batch_size,
        )

    def predict(
        self,
        features: np.ndarray,
        *,
        batch_size: int = 4096,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict nonnegative OD and one-standard-deviation uncertainty."""

        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != len(self.feature_mean):
            raise ValueError("multi-tower inference features have the wrong shape")
        output = np.empty(len(values), dtype=np.float32)
        uncertainty = np.empty(len(values), dtype=np.float32)
        groups = tuple(sorted(self.group_slices))
        target_upper = _model_target_upper(self)
        transformed_upper = np.arcsinh(target_upper / self.target_scale)
        for start in range(0, len(values), batch_size):
            stop = min(start + batch_size, len(values))
            normalized = (values[start:stop] - self.feature_mean) / self.feature_scale
            towers = []
            for name, weight, bias in zip(
                groups, self.tower_weights, self.tower_biases, strict=True
            ):
                left, right = self.group_slices[name]
                towers.append(_gelu(normalized[:, left:right] @ weight + bias))
            hidden = _gelu(
                np.concatenate(towers, axis=1) @ self.fusion_weight + self.fusion_bias
            )
            for weight, bias in zip(
                self.residual_weights, self.residual_biases, strict=True
            ):
                hidden = hidden + 0.5 * _gelu(hidden @ weight + bias)
            mean = np.clip(
                (hidden @ self.mean_weight + self.mean_bias)[:, 0],
                0,
                transformed_upper,
            )
            log_variance = np.clip(
                (hidden @ self.variance_weight + self.variance_bias)[:, 0],
                -8,
                5,
            )
            sigma = np.sqrt(np.exp(log_variance))
            center = np.clip(np.sinh(mean) * self.target_scale, 0, target_upper)
            lower = np.clip(np.sinh(mean - sigma) * self.target_scale, 0, target_upper)
            upper = np.clip(np.sinh(mean + sigma) * self.target_scale, 0, target_upper)
            output[start:stop] = center
            uncertainty[start:stop] = (upper - lower) / 2
        return output, uncertainty

    def predict_accelerated(
        self,
        features: np.ndarray,
        *,
        batch_size: int = 16_384,
        device: str = "auto",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Use optional PyTorch for the same sealed weights."""

        try:
            import torch
        except ImportError:
            return self.predict(features, batch_size=batch_size)
        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != len(self.feature_mean):
            raise ValueError("multi-tower inference features have the wrong shape")
        selected = torch.device(
            "cuda"
            if device == "auto" and torch.cuda.is_available()
            else ("cpu" if device == "auto" else device)
        )
        tensor = lambda value: torch.from_numpy(  # noqa: E731
            np.asarray(value, dtype=np.float32)
        ).to(selected)
        mean_feature, scale_feature = (
            tensor(self.feature_mean),
            tensor(self.feature_scale),
        )
        tower_weights = tuple(tensor(value) for value in self.tower_weights)
        tower_biases = tuple(tensor(value) for value in self.tower_biases)
        fusion_weight, fusion_bias = (
            tensor(self.fusion_weight),
            tensor(self.fusion_bias),
        )
        residual_weights = tuple(tensor(value) for value in self.residual_weights)
        residual_biases = tuple(tensor(value) for value in self.residual_biases)
        mean_weight, mean_bias = tensor(self.mean_weight), tensor(self.mean_bias)
        variance_weight, variance_bias = (
            tensor(self.variance_weight),
            tensor(self.variance_bias),
        )
        output = np.empty(len(values), dtype=np.float32)
        uncertainty = np.empty(len(values), dtype=np.float32)
        groups = tuple(sorted(self.group_slices))
        target_upper = _model_target_upper(self)
        transformed_upper = math.asinh(target_upper / self.target_scale)
        with torch.inference_mode():
            for start in range(0, len(values), batch_size):
                stop = min(start + batch_size, len(values))
                normalized = (tensor(values[start:stop]) - mean_feature) / scale_feature
                towers = []
                for name, weight, bias in zip(
                    groups, tower_weights, tower_biases, strict=True
                ):
                    left, right = self.group_slices[name]
                    towers.append(
                        torch.nn.functional.gelu(
                            normalized[:, left:right] @ weight + bias,
                            approximate="tanh",
                        )
                    )
                hidden = torch.nn.functional.gelu(
                    torch.cat(towers, dim=1) @ fusion_weight + fusion_bias,
                    approximate="tanh",
                )
                for weight, bias in zip(residual_weights, residual_biases, strict=True):
                    hidden = hidden + 0.5 * torch.nn.functional.gelu(
                        hidden @ weight + bias, approximate="tanh"
                    )
                transformed = torch.clamp(
                    (hidden @ mean_weight + mean_bias)[:, 0],
                    0,
                    transformed_upper,
                )
                log_variance = torch.clamp(
                    (hidden @ variance_weight + variance_bias)[:, 0], -8, 5
                )
                sigma = torch.sqrt(torch.exp(log_variance))
                center = torch.clamp(
                    torch.sinh(transformed) * self.target_scale,
                    0,
                    target_upper,
                )
                lower = torch.clamp(
                    torch.sinh(transformed - sigma) * self.target_scale,
                    0,
                    target_upper,
                )
                upper = torch.clamp(
                    torch.sinh(transformed + sigma) * self.target_scale,
                    0,
                    target_upper,
                )
                output[start:stop] = center.cpu().numpy()
                uncertainty[start:stop] = ((upper - lower) / 2).cpu().numpy()
        return output, uncertainty

    def save(self, path: Path | str) -> Path:
        target = Path(path)
        metadata = {
            "schema_version": 1,
            "architecture": "multi_tower",
            "group_slices": {
                key: list(value) for key, value in sorted(self.group_slices.items())
            },
            "target_scale": self.target_scale,
            "residual_layers": len(self.residual_weights),
            "provenance": self.provenance,
            "fingerprint": self.fingerprint,
        }
        if self.morphology_transfer_slice is not None:
            metadata["morphology_transfer_slice"] = list(self.morphology_transfer_slice)

        def writer(stream) -> None:
            np.savez_compressed(
                stream,
                metadata_json=np.asarray(_canonical_json(metadata)),
                **_model_arrays(self),
            )

        return write_binary_atomic(target, writer)

    @classmethod
    def load(cls, path: Path | str) -> PortableMultiTowerRegressor:
        with np.load(Path(path), allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"]))
            if (
                metadata.get("schema_version") != 1
                or metadata.get("architecture") != "multi_tower"
            ):
                raise ValueError("unsupported portable multi-tower schema")
            group_slices = {
                str(key): (int(value[0]), int(value[1]))
                for key, value in metadata["group_slices"].items()
            }
            group_count = len(group_slices)
            residual_count = int(metadata["residual_layers"])
            raw_transfer_slice = metadata.get("morphology_transfer_slice")
            transfer_slice = (
                None
                if raw_transfer_slice is None
                else (int(raw_transfer_slice[0]), int(raw_transfer_slice[1]))
            )
            return cls(
                feature_mean=data["feature_mean"],
                feature_scale=data["feature_scale"],
                group_slices=group_slices,
                tower_weights=tuple(
                    data[f"tower_{index}_weight"] for index in range(group_count)
                ),
                tower_biases=tuple(
                    data[f"tower_{index}_bias"] for index in range(group_count)
                ),
                fusion_weight=data["fusion_weight"],
                fusion_bias=data["fusion_bias"],
                residual_weights=tuple(
                    data[f"residual_{index}_weight"] for index in range(residual_count)
                ),
                residual_biases=tuple(
                    data[f"residual_{index}_bias"] for index in range(residual_count)
                ),
                mean_weight=data["mean_weight"],
                mean_bias=data["mean_bias"],
                variance_weight=data["variance_weight"],
                variance_bias=data["variance_bias"],
                target_scale=float(metadata["target_scale"]),
                provenance=dict(metadata["provenance"]),
                morphology_bank_key=(
                    data["morphology_bank_key"] if transfer_slice is not None else None
                ),
                morphology_bank_od=(
                    data["morphology_bank_od"] if transfer_slice is not None else None
                ),
                morphology_transfer_slice=transfer_slice,
                fingerprint=str(metadata["fingerprint"]),
            )


def fit_portable_multi_tower(
    features: np.ndarray,
    measured_od: np.ndarray,
    train: np.ndarray,
    *,
    group_slices: dict[str, tuple[int, int]] | None = None,
    section_groups: np.ndarray | None = None,
    seed: int = 0,
    epochs: int = 80,
    batch_size: int = 1024,
    learning_rate: float = 5e-4,
    residual_layers: int = 2,
) -> PortableMultiTowerRegressor:
    """Fit a section-balanced cell-aware model and export NumPy weights."""

    try:
        import torch
        from torch import nn
    except ImportError as error:
        raise RuntimeError("multi-tower fitting requires the 'cells' extra") from error
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(measured_od, dtype=np.float32)
    indices = np.asarray(train, dtype=np.int64)
    valid = indices[np.isfinite(y[indices])]
    if x.ndim != 2 or y.shape != (len(x),) or len(valid) < 8:
        raise ValueError("multi-tower fitting requires aligned measured cells")
    groups = _validated_group_slices(group_slices, x.shape[1])
    if epochs < 1 or batch_size < 1 or learning_rate <= 0 or residual_layers < 1:
        raise ValueError("multi-tower training controls are invalid")
    center = x[valid].mean(axis=0)
    scale = x[valid].std(axis=0)
    scale[scale < 1e-6] = 1
    positive = y[valid][y[valid] > 0]
    target_scale = max(
        float(np.quantile(positive, 0.75)) if len(positive) else float(y[valid].max()),
        1e-4,
    )
    target_upper = _robust_target_upper(y[valid])
    normalized = (x[valid] - center) / scale
    target = np.arcsinh(np.maximum(y[valid], 0) / target_scale).astype(np.float32)
    sample_weight = _balanced_sample_weights(
        target,
        None if section_groups is None else np.asarray(section_groups)[valid],
    )
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_x = torch.from_numpy(normalized).to(device)
    train_y = torch.from_numpy(target).to(device)
    weights = torch.from_numpy(sample_weight).to(device)
    group_names = tuple(sorted(groups))
    tower_widths = tuple(
        min(128, max(16, int(round(math.sqrt(stop - start) * 4))))
        for start, stop in (groups[name] for name in group_names)
    )
    hidden_width = 256

    class Network(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.towers = nn.ModuleList(
                nn.Linear(groups[name][1] - groups[name][0], width)
                for name, width in zip(group_names, tower_widths, strict=True)
            )
            self.fusion = nn.Linear(sum(tower_widths), hidden_width)
            self.residual = nn.ModuleList(
                nn.Linear(hidden_width, hidden_width) for _ in range(residual_layers)
            )
            self.mean = nn.Linear(hidden_width, 1)
            self.log_variance = nn.Linear(hidden_width, 1)

        def forward(self, value):  # type: ignore[no-untyped-def]
            towers = []
            for name, tower in zip(group_names, self.towers, strict=True):
                start, stop = groups[name]
                towers.append(
                    torch.nn.functional.gelu(
                        tower(value[:, start:stop]), approximate="tanh"
                    )
                )
            hidden = torch.nn.functional.gelu(
                self.fusion(torch.cat(towers, dim=1)), approximate="tanh"
            )
            for layer in self.residual:
                hidden = hidden + 0.5 * torch.nn.functional.gelu(
                    layer(hidden), approximate="tanh"
                )
            return self.mean(hidden)[:, 0], torch.clamp(
                self.log_variance(hidden)[:, 0], -8, 5
            )

    model = Network().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    model.train()
    for _epoch in range(epochs):
        order = torch.multinomial(
            weights,
            num_samples=len(valid),
            replacement=True,
            generator=generator,
        )
        for start in range(0, len(valid), batch_size):
            selected = order[start : start + batch_size]
            predicted, log_variance = model(train_x[selected])
            truth = train_y[selected]
            residual = truth - predicted
            huber = torch.nn.functional.smooth_l1_loss(predicted, truth)
            heteroscedastic = torch.mean(
                residual.square() * torch.exp(-log_variance) + log_variance
            )
            if len(selected) > 1:
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
            else:
                rank = torch.zeros((), device=device)
            loss = huber + 0.05 * heteroscedastic + 0.10 * rank
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    model.eval()

    def numpy(value: object) -> np.ndarray:
        return value.detach().cpu().numpy().astype(np.float32)

    return PortableMultiTowerRegressor(
        feature_mean=center.astype(np.float32),
        feature_scale=scale.astype(np.float32),
        group_slices=groups,
        tower_weights=tuple(numpy(layer.weight.T) for layer in model.towers),
        tower_biases=tuple(numpy(layer.bias) for layer in model.towers),
        fusion_weight=numpy(model.fusion.weight.T),
        fusion_bias=numpy(model.fusion.bias),
        residual_weights=tuple(numpy(layer.weight.T) for layer in model.residual),
        residual_biases=tuple(numpy(layer.bias) for layer in model.residual),
        mean_weight=numpy(model.mean.weight.T),
        mean_bias=numpy(model.mean.bias),
        variance_weight=numpy(model.log_variance.weight.T),
        variance_bias=numpy(model.log_variance.bias),
        target_scale=target_scale,
        provenance={
            "seed": seed,
            "epochs": epochs,
            "training_cells": len(valid),
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "target_transform": "asinh_training_q75_v1",
            "output_calibration": "training-od-q995-bound-v1",
            "target_od_upper": target_upper,
            "loss": "huber+heteroscedastic+pairwise-rank-v1",
        },
    )


def feature_group_slices(
    feature_schema_id: str,
    feature_count: int,
    *,
    radii_count: int = 4,
) -> dict[str, tuple[int, int]]:
    """Return stable group slices for a known cell feature schema."""

    if feature_schema_id == "native-hdab-neutral-spatial-uni2h-v2":
        return {"all": (0, feature_count)}
    if feature_schema_id != "native-hdab-neutral-cell-multiscale-v3":
        raise ValueError(f"unsupported cell feature schema: {feature_schema_id}")
    morphology = 1536
    phenotype = 19
    neighborhood = radii_count * (2 * 32 + 3)
    position = 21
    expected = morphology + phenotype + neighborhood + position
    if feature_count != expected:
        raise ValueError(
            f"multiscale feature width is {feature_count}; expected {expected}"
        )
    return {
        "morphology": (0, morphology),
        "phenotype": (morphology, morphology + phenotype),
        "neighborhood": (
            morphology + phenotype,
            morphology + phenotype + neighborhood,
        ),
        "position": (expected - position, expected),
    }


def _validated_group_slices(
    raw: dict[str, tuple[int, int]] | None, feature_count: int
) -> dict[str, tuple[int, int]]:
    groups = dict(raw or {"all": (0, feature_count)})
    if not groups:
        raise ValueError("multi-tower fitting requires feature groups")
    normalized = {key: (int(value[0]), int(value[1])) for key, value in groups.items()}
    covered = np.zeros(feature_count, dtype=bool)
    for name, (start, stop) in normalized.items():
        if (
            not name
            or start < 0
            or stop <= start
            or stop > feature_count
            or np.any(covered[start:stop])
        ):
            raise ValueError("multi-tower feature groups are malformed")
        covered[start:stop] = True
    if not np.all(covered):
        raise ValueError("multi-tower feature groups must cover every feature")
    return normalized


def _balanced_sample_weights(
    transformed_od: np.ndarray, section_groups: np.ndarray | None
) -> np.ndarray:
    values = np.asarray(transformed_od, dtype=np.float64)
    quantiles = np.unique(np.quantile(values, np.linspace(0, 1, 9)[1:-1]))
    od_bin = np.searchsorted(quantiles, values, side="right")
    if section_groups is None:
        strata = od_bin[:, None]
    else:
        sections = np.asarray(section_groups).astype(str)
        if sections.shape != (len(values),):
            raise ValueError("multi-tower section groups do not align")
        _section, section_index = np.unique(sections, return_inverse=True)
        strata = np.column_stack((section_index, od_bin))
    _unique, inverse, counts = np.unique(
        strata, axis=0, return_inverse=True, return_counts=True
    )
    result = 1.0 / counts[inverse]
    return (result / result.sum()).astype(np.float32)


def _robust_target_upper(values: np.ndarray) -> float:
    """Estimate a training-only upper OD bound resistant to rare debris."""

    target = np.asarray(values, dtype=np.float64)
    target = target[np.isfinite(target) & (target >= 0)]
    if not len(target):
        raise ValueError("target calibration requires finite nonnegative OD")
    return max(
        float(np.quantile(target, 0.995)) * 1.25,
        float(np.quantile(target, 0.75)) * 2.0,
        1e-4,
    )


def _model_target_upper(model: PortableMultiTowerRegressor) -> float:
    value = model.provenance.get("target_od_upper")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        upper = float(value)
        if math.isfinite(upper) and upper > 0:
            return upper
    # Backward compatibility for early schema-v1 artifacts. Newly fitted
    # models always seal a finite training-derived bound in provenance.
    return float(np.finfo(np.float32).max)


def _model_arrays(model: PortableMultiTowerRegressor) -> dict[str, np.ndarray]:
    arrays = {
        "feature_mean": np.asarray(model.feature_mean, dtype=np.float32),
        "feature_scale": np.asarray(model.feature_scale, dtype=np.float32),
        "fusion_weight": np.asarray(model.fusion_weight, dtype=np.float32),
        "fusion_bias": np.asarray(model.fusion_bias, dtype=np.float32),
        "mean_weight": np.asarray(model.mean_weight, dtype=np.float32),
        "mean_bias": np.asarray(model.mean_bias, dtype=np.float32),
        "variance_weight": np.asarray(model.variance_weight, dtype=np.float32),
        "variance_bias": np.asarray(model.variance_bias, dtype=np.float32),
    }
    for index, (weight, bias) in enumerate(
        zip(model.tower_weights, model.tower_biases, strict=True)
    ):
        arrays[f"tower_{index}_weight"] = np.asarray(weight, dtype=np.float32)
        arrays[f"tower_{index}_bias"] = np.asarray(bias, dtype=np.float32)
    for index, (weight, bias) in enumerate(
        zip(model.residual_weights, model.residual_biases, strict=True)
    ):
        arrays[f"residual_{index}_weight"] = np.asarray(weight, dtype=np.float32)
        arrays[f"residual_{index}_bias"] = np.asarray(bias, dtype=np.float32)
    if model.morphology_bank_key is not None:
        assert model.morphology_bank_od is not None
        arrays["morphology_bank_key"] = np.asarray(
            model.morphology_bank_key, dtype=np.float32
        )
        arrays["morphology_bank_od"] = np.asarray(
            model.morphology_bank_od, dtype=np.float32
        )
    return arrays


def _fingerprint(model: PortableMultiTowerRegressor) -> str:
    digest = hashlib.sha256(b"histopia-portable-multi-tower-v1\0")
    metadata = {
        "group_slices": {
            key: list(value) for key, value in sorted(model.group_slices.items())
        },
        "target_scale": model.target_scale,
        "provenance": model.provenance,
    }
    if model.morphology_transfer_slice is not None:
        metadata["morphology_transfer_slice"] = list(model.morphology_transfer_slice)
    digest.update(_canonical_json(metadata).encode())
    for name, value in sorted(_model_arrays(model).items()):
        array = np.ascontiguousarray(value)
        digest.update(name.encode())
        digest.update(array.dtype.str.encode())
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _gelu(values: np.ndarray) -> np.ndarray:
    return (
        0.5
        * values
        * (
            1.0
            + np.tanh(
                math.sqrt(2.0 / math.pi) * (values + 0.044715 * np.power(values, 3))
            )
        )
    )


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))
