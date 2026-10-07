"""Portable hurdle models for semantic-guided protein prediction."""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from histopia._atomic import write_binary_atomic


@dataclass(frozen=True, slots=True)
class DenseNetwork:
    """Portable ReLU multilayer perceptron parameters."""

    weights: tuple[np.ndarray, ...]
    biases: tuple[np.ndarray, ...]

    def __post_init__(self) -> None:
        if not self.weights or len(self.weights) != len(self.biases):
            raise ValueError("network weights and biases must contain matching layers")
        previous = None
        for weight, bias in zip(self.weights, self.biases, strict=True):
            matrix = np.asarray(weight)
            offset = np.asarray(bias)
            if matrix.ndim != 2 or offset.shape != (matrix.shape[1],):
                raise ValueError("network layer dimensions are invalid")
            if previous is not None and matrix.shape[0] != previous:
                raise ValueError("network layers are not connected")
            if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(offset)):
                raise ValueError("network parameters must be finite")
            previous = matrix.shape[1]

    def forward(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        for index, (weight, bias) in enumerate(
            zip(self.weights, self.biases, strict=True)
        ):
            values = values @ weight + bias
            if index + 1 < len(self.weights):
                values = np.maximum(values, 0.0)
        return values.squeeze(axis=1)


@dataclass(frozen=True, slots=True)
class ProteinModel:
    """Stain-neutral transform, ensemble heads, and reference calibration."""

    target_id: str
    assay_domain: str
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    pca_mean: np.ndarray
    pca_basis: np.ndarray
    regressors: tuple[DenseNetwork, ...]
    classifiers: tuple[DenseNetwork, ...]
    reference_od: np.ndarray
    provenance: dict[str, object]
    fingerprint: str | None = None

    def __post_init__(self) -> None:
        feature_mean = np.asarray(self.feature_mean)
        feature_scale = np.asarray(self.feature_scale)
        pca_mean = np.asarray(self.pca_mean)
        pca_basis = np.asarray(self.pca_basis)
        if (
            feature_mean.ndim != 1
            or feature_scale.shape != feature_mean.shape
            or pca_mean.shape != feature_mean.shape
            or pca_basis.ndim != 2
            or pca_basis.shape[1] != len(feature_mean)
        ):
            raise ValueError("protein model feature transform is inconsistent")
        if np.any(feature_scale <= 0) or not np.all(np.isfinite(feature_scale)):
            raise ValueError("protein model feature scale must be positive")
        if not self.regressors:
            raise ValueError("protein model requires a regression ensemble")
        output_width = pca_basis.shape[0]
        networks = (*self.regressors, *self.classifiers)
        if any(network.weights[0].shape[0] != output_width for network in networks):
            raise ValueError("network input width differs from PCA output")
        reference = np.asarray(self.reference_od)
        if reference.ndim != 1 or not len(reference) or np.any(reference < 0):
            raise ValueError("reference OD must be a non-empty nonnegative vector")
        expected = _model_fingerprint(self)
        if self.fingerprint is not None and self.fingerprint != expected:
            raise ValueError("protein model fingerprint does not match")
        object.__setattr__(self, "fingerprint", expected)

    def transform(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.feature_mean):
            raise ValueError("prediction features differ from model inputs")
        if not np.all(np.isfinite(values)):
            raise ValueError("prediction features must be finite")
        standardized = (values - self.feature_mean) / self.feature_scale
        return (standardized - self.pca_mean) @ self.pca_basis.T

    def predict(
        self, features: np.ndarray
    ) -> tuple[np.ndarray | None, np.ndarray, np.ndarray, np.ndarray]:
        """Return probability, relative score, model-scale OD, and uncertainty."""

        transformed = self.transform(features)
        od_members = np.stack(
            [
                np.maximum(network.forward(transformed), 0.0)
                for network in self.regressors
            ]
        )
        od = od_members.mean(axis=0)
        ordered = np.sort(np.asarray(self.reference_od, dtype=np.float64))
        od_scale = max(float(np.quantile(ordered, 0.9)), 1e-8)
        uncertainty = od_members.std(axis=0) / od_scale
        relative = np.searchsorted(ordered, od, side="right") / len(ordered)
        probability = None
        if self.classifiers:
            logits = np.stack(
                [network.forward(transformed) for network in self.classifiers]
            )
            probability_members = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
            probability = probability_members.mean(axis=0).astype(np.float32)
            uncertainty = np.maximum(uncertainty, probability_members.std(axis=0))
        return (
            probability,
            relative.astype(np.float32),
            od.astype(np.float32),
            uncertainty.astype(np.float32),
        )

    def save(self, path: Path | str) -> Path:
        """Save portable arrays only; no estimator pickles."""

        target = Path(path)
        arrays = _model_arrays(self)
        metadata = {
            "schema_version": 1,
            "target_id": self.target_id,
            "assay_domain": self.assay_domain,
            "regressor_layers": [len(value.weights) for value in self.regressors],
            "classifier_layers": [len(value.weights) for value in self.classifiers],
            "provenance": self.provenance,
            "fingerprint": self.fingerprint,
        }

        def writer(stream) -> None:
            np.savez_compressed(
                stream,
                metadata_json=np.asarray(_canonical_json(metadata)),
                **arrays,
            )

        return write_binary_atomic(target, writer)

    @classmethod
    def load(cls, path: Path | str) -> ProteinModel:
        with np.load(Path(path), allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"]))
            regressors = _load_networks(data, "reg", metadata["regressor_layers"])
            classifiers = _load_networks(data, "cls", metadata["classifier_layers"])
            return cls(
                target_id=metadata["target_id"],
                assay_domain=metadata["assay_domain"],
                feature_mean=data["feature_mean"],
                feature_scale=data["feature_scale"],
                pca_mean=data["pca_mean"],
                pca_basis=data["pca_basis"],
                regressors=regressors,
                classifiers=classifiers,
                reference_od=data["reference_od"],
                provenance=metadata["provenance"],
                fingerprint=metadata["fingerprint"],
            )


def fit_protein_model(
    features: np.ndarray,
    measured_od: np.ndarray,
    binary_label: np.ndarray,
    train_indices: np.ndarray,
    *,
    target_id: str,
    assay_domain: str = "default",
    pca_components: int = 128,
    hidden_units: int = 256,
    hidden_layers: int = 4,
    ensemble_seeds: int = 3,
    max_iter: int = 200,
    seed: int = 0,
    provenance: dict[str, object] | None = None,
) -> ProteinModel:
    """Fit portable four-layer MLP heads on an explicit leakage-safe split."""

    try:
        from sklearn.decomposition import PCA
        from sklearn.neural_network import MLPClassifier, MLPRegressor
    except ImportError as error:
        raise RuntimeError(
            "protein fitting requires Histopia's 'protein' optional dependencies"
        ) from error
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(measured_od, dtype=np.float64)
    labels = np.asarray(binary_label, dtype=np.int8)
    train = np.asarray(train_indices, dtype=np.int64)
    if x.ndim != 2 or y.shape != (len(x),) or labels.shape != (len(x),):
        raise ValueError("training features and targets have inconsistent shapes")
    if train.ndim != 1 or not len(train) or np.any((train < 0) | (train >= len(x))):
        raise ValueError("train_indices must identify one or more feature rows")
    eligible = train[np.isfinite(y[train]) & (y[train] >= 0)]
    if len(eligible) < 4:
        raise ValueError("at least four measured training cells are required")
    feature_mean = x[eligible].mean(axis=0)
    feature_scale = x[eligible].std(axis=0)
    feature_scale[feature_scale < 1e-8] = 1.0
    standardized = (x[eligible] - feature_mean) / feature_scale
    components = min(pca_components, standardized.shape[0] - 1, standardized.shape[1])
    solver = "randomized" if min(standardized.shape) > 512 else "full"
    pca = PCA(n_components=max(1, components), svd_solver=solver, random_state=seed)
    transformed = pca.fit_transform(standardized)
    hidden = (hidden_units,) * hidden_layers
    regressors: list[DenseNetwork] = []
    classifiers: list[DenseNetwork] = []
    binary_rows = labels[eligible] >= 0
    has_binary = (
        binary_rows.sum() >= 4 and len(np.unique(labels[eligible][binary_rows])) == 2
    )
    for member in range(ensemble_seeds):
        member_seed = seed + member
        regressor = MLPRegressor(
            hidden_layer_sizes=hidden,
            activation="relu",
            solver="adam",
            early_stopping=len(eligible) >= 20,
            validation_fraction=0.1,
            n_iter_no_change=15,
            max_iter=max_iter,
            random_state=member_seed,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            regressor.fit(transformed, y[eligible])
        regressors.append(_portable_network(regressor))
        if has_binary:
            class_x = transformed[binary_rows]
            class_y = labels[eligible][binary_rows]
            balanced = _balanced_indices(class_y, member_seed)
            classifier = MLPClassifier(
                hidden_layer_sizes=hidden,
                activation="relu",
                solver="adam",
                early_stopping=len(balanced) >= 20,
                validation_fraction=0.1,
                n_iter_no_change=15,
                max_iter=max_iter,
                random_state=member_seed,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                classifier.fit(class_x[balanced], class_y[balanced])
            classifiers.append(_portable_network(classifier))
    return ProteinModel(
        target_id=target_id,
        assay_domain=assay_domain,
        feature_mean=feature_mean.astype(np.float32),
        feature_scale=feature_scale.astype(np.float32),
        pca_mean=pca.mean_.astype(np.float32),
        pca_basis=pca.components_.astype(np.float32),
        regressors=tuple(regressors),
        classifiers=tuple(classifiers),
        reference_od=np.sort(y[eligible]).astype(np.float32),
        provenance={
            **(provenance or {}),
            "algorithm": "pca_relu_mlp_hurdle_v1",
            "pca_components": int(components),
            "hidden_units": hidden_units,
            "hidden_layers": hidden_layers,
            "ensemble_seeds": ensemble_seeds,
            "seed": seed,
            "training_rows": int(len(eligible)),
            "binary_rows": int(binary_rows.sum()),
        },
    )


def evaluate_predictions(
    measured_od: np.ndarray,
    predicted_od: np.ndarray,
    *,
    binary_label: np.ndarray | None = None,
    probability: np.ndarray | None = None,
) -> dict[str, float | int | None]:
    """Return dependency-light continuous and binary validation metrics."""

    truth = np.asarray(measured_od, dtype=np.float64)
    predicted = np.asarray(predicted_od, dtype=np.float64)
    valid = np.isfinite(truth) & np.isfinite(predicted)
    metrics: dict[str, float | int | None] = {
        "rows": int(valid.sum()),
        "mae": float(np.mean(np.abs(truth[valid] - predicted[valid])))
        if np.any(valid)
        else None,
        "spearman": _spearman(truth[valid], predicted[valid])
        if valid.sum() >= 2
        else None,
        "auroc": None,
        "average_precision": None,
        "normalized_ap_gain": None,
        "ece": None,
    }
    if binary_label is None or probability is None:
        return metrics
    labels = np.asarray(binary_label, dtype=np.int8)
    scores = np.asarray(probability, dtype=np.float64)
    eligible = (labels >= 0) & np.isfinite(scores)
    if eligible.sum() < 2 or len(np.unique(labels[eligible])) != 2:
        return metrics
    y = labels[eligible]
    score = scores[eligible]
    prevalence = float(y.mean())
    ap = _average_precision(y, score)
    metrics.update(
        {
            "auroc": _auroc(y, score),
            "average_precision": ap,
            "normalized_ap_gain": (ap - prevalence) / max(1.0 - prevalence, 1e-8),
            "ece": _expected_calibration_error(y, score),
        }
    )
    return metrics


def evaluate_stain_invariance(
    neutral_features: np.ndarray,
    raw_features: np.ndarray,
    source_stains: np.ndarray,
    groups: np.ndarray,
    baseline_predictions: np.ndarray,
    perturbed_predictions: np.ndarray,
    *,
    seed: int = 0,
) -> dict[str, float]:
    """Audit source-stain predictability and synthetic chromogen sensitivity."""

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import GroupKFold, cross_val_predict
        from sklearn.preprocessing import LabelEncoder
    except ImportError as error:
        raise RuntimeError(
            "stain-invariance auditing requires the 'protein' dependencies"
        ) from error
    neutral = np.asarray(neutral_features, dtype=np.float64)
    raw = np.asarray(raw_features, dtype=np.float64)
    stains = np.asarray(source_stains)
    group_values = np.asarray(groups)
    baseline = np.asarray(baseline_predictions, dtype=np.float64)
    perturbed = np.asarray(perturbed_predictions, dtype=np.float64)
    if neutral.shape != raw.shape or neutral.ndim != 2:
        raise ValueError("neutral and raw features must share a two-dimensional shape")
    count = len(neutral)
    if any(value.shape != (count,) for value in (stains, group_values)):
        raise ValueError("stain labels and groups must align with features")
    if baseline.shape != perturbed.shape or baseline.ndim != 1:
        raise ValueError("baseline and perturbed predictions must be matching vectors")
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(stains)
    if len(encoder.classes_) < 2 or len(np.unique(group_values)) < 2:
        raise ValueError("invariance audit requires multiple stains and groups")
    folds = min(5, len(np.unique(group_values)))
    splitter = GroupKFold(n_splits=folds)

    def domain_auc(features: np.ndarray) -> float:
        estimator = LogisticRegression(
            max_iter=500,
            class_weight="balanced",
            random_state=seed,
        )
        probabilities = cross_val_predict(
            estimator,
            features,
            encoded,
            groups=group_values,
            cv=splitter,
            method="predict_proba",
        )
        if probabilities.shape[1] == 2:
            return float(roc_auc_score(encoded, probabilities[:, 1]))
        return float(
            roc_auc_score(encoded, probabilities, multi_class="ovr", average="macro")
        )

    return {
        "chromogen_prediction_delta": float(np.median(np.abs(baseline - perturbed))),
        "neutral_stain_macro_auroc": domain_auc(neutral),
        "raw_stain_macro_auroc": domain_auc(raw),
    }


def promotion_decision(
    metrics: dict[str, float | int | None], *, binary: bool
) -> tuple[bool, tuple[str, ...]]:
    """Apply the version-1 cross-mouse minimum quality gates."""

    reasons: list[str] = []
    if metrics.get("evaluation") != "leave_one_mouse_out":
        reasons.append("production promotion requires leave-one-mouse-out evaluation")
    spearman = metrics.get("spearman")
    if not isinstance(spearman, float) or spearman < 0.35:
        reasons.append("continuous Spearman is below 0.35")
    spearman_64um = metrics.get("spearman_64um")
    if not isinstance(spearman_64um, float) or spearman_64um < 0.50:
        reasons.append("64 µm aggregate Spearman is below 0.50")
    improvement = metrics.get("mae_improvement")
    if not isinstance(improvement, float) or improvement < 0.10:
        reasons.append("MAE improvement over semantic baseline is below 10%")
    fold_fraction = metrics.get("folds_beating_semantic_fraction")
    if not isinstance(fold_fraction, float) or fold_fraction < 0.75:
        reasons.append("fewer than 75% of mouse folds beat the semantic baseline")
    if binary:
        for key, threshold in (("auroc", 0.75), ("normalized_ap_gain", 0.25)):
            value = metrics.get(key)
            if not isinstance(value, float) or value < threshold:
                reasons.append(f"{key} is below {threshold:.2f}")
        ece = metrics.get("ece")
        if not isinstance(ece, float) or ece > 0.10:
            reasons.append("ECE is above 0.10")
    invariance = metrics.get("invariance")
    if not isinstance(invariance, dict):
        reasons.append("stain-invariance audit is missing")
    else:
        delta = invariance.get("chromogen_prediction_delta")
        neutral_auc = invariance.get("neutral_stain_macro_auroc")
        raw_auc = invariance.get("raw_stain_macro_auroc")
        if not isinstance(delta, (int, float)) or float(delta) > 0.02:
            reasons.append("synthetic chromogen prediction delta exceeds 0.02")
        if not isinstance(neutral_auc, (int, float)) or float(neutral_auc) > 0.60:
            reasons.append("neutral feature stain-domain AUROC exceeds 0.60")
        if (
            not isinstance(raw_auc, (int, float))
            or not isinstance(neutral_auc, (int, float))
            or float(raw_auc) - float(neutral_auc) < 0.10
        ):
            reasons.append("neutral stain-domain AUROC is not 0.10 below raw RGB")
    return not reasons, tuple(reasons)


def _portable_network(estimator) -> DenseNetwork:
    return DenseNetwork(
        weights=tuple(
            np.asarray(value, dtype=np.float32) for value in estimator.coefs_
        ),
        biases=tuple(
            np.asarray(value, dtype=np.float32) for value in estimator.intercepts_
        ),
    )


def _balanced_indices(labels: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    negative = np.flatnonzero(labels == 0)
    positive = np.flatnonzero(labels == 1)
    size = max(len(negative), len(positive))
    return rng.permutation(
        np.concatenate(
            [
                rng.choice(negative, size=size, replace=len(negative) < size),
                rng.choice(positive, size=size, replace=len(positive) < size),
            ]
        )
    )


def _model_arrays(model: ProteinModel) -> dict[str, np.ndarray]:
    arrays = {
        "feature_mean": np.asarray(model.feature_mean, dtype=np.float32),
        "feature_scale": np.asarray(model.feature_scale, dtype=np.float32),
        "pca_mean": np.asarray(model.pca_mean, dtype=np.float32),
        "pca_basis": np.asarray(model.pca_basis, dtype=np.float32),
        "reference_od": np.asarray(model.reference_od, dtype=np.float32),
    }
    for prefix, networks in (("reg", model.regressors), ("cls", model.classifiers)):
        for network_index, network in enumerate(networks):
            for layer, (weight, bias) in enumerate(
                zip(network.weights, network.biases, strict=True)
            ):
                arrays[f"{prefix}_{network_index}_w_{layer}"] = np.asarray(
                    weight, dtype=np.float32
                )
                arrays[f"{prefix}_{network_index}_b_{layer}"] = np.asarray(
                    bias, dtype=np.float32
                )
    return arrays


def _load_networks(
    data, prefix: str, layer_counts: list[int]
) -> tuple[DenseNetwork, ...]:
    return tuple(
        DenseNetwork(
            weights=tuple(
                data[f"{prefix}_{index}_w_{layer}"] for layer in range(count)
            ),
            biases=tuple(data[f"{prefix}_{index}_b_{layer}"] for layer in range(count)),
        )
        for index, count in enumerate(layer_counts)
    )


def _model_fingerprint(model: ProteinModel) -> str:
    digest = hashlib.sha256(b"histopia-protein-model-v1\0")
    metadata = {
        "target_id": model.target_id,
        "assay_domain": model.assay_domain,
        "provenance": model.provenance,
    }
    digest.update(_canonical_json(metadata).encode())
    for name, value in sorted(_model_arrays(model).items()):
        array = np.ascontiguousarray(value)
        digest.update(name.encode())
        digest.update(array.dtype.str.encode())
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0
        start = stop
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    a, b = _rank(left), _rank(right)
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    ranks = _rank(scores) + 1.0
    positives = labels == 1
    n_positive = int(positives.sum())
    n_negative = len(labels) - n_positive
    return float(
        (ranks[positives].sum() - n_positive * (n_positive + 1) / 2)
        / (n_positive * n_negative)
    )


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores, kind="stable")
    ordered = labels[order]
    cumulative = np.cumsum(ordered == 1)
    positive_positions = np.flatnonzero(ordered == 1)
    return float(np.mean(cumulative[positive_positions] / (positive_positions + 1)))


def _expected_calibration_error(
    labels: np.ndarray, scores: np.ndarray, bins: int = 10
) -> float:
    edges = np.linspace(0, 1, bins + 1)
    total = len(labels)
    result = 0.0
    for index in range(bins):
        selected = (scores >= edges[index]) & (
            scores <= edges[index + 1]
            if index + 1 == bins
            else scores < edges[index + 1]
        )
        if np.any(selected):
            result += selected.mean() * abs(
                scores[selected].mean() - labels[selected].mean()
            )
    return float(result) if total else float("nan")
