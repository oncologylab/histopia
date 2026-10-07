"""Supervised semantic linear probes over frozen patch embeddings."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from histopia._atomic import write_binary_atomic
from histopia._validation import positive_float, positive_int


@dataclass(frozen=True, slots=True)
class SemanticProbeConfig:
    """Deterministic configuration for a leakage-safe semantic linear probe."""

    pca_components: int = 32
    regularization_c: float = 1.0
    max_iter: int = 1_000
    seed: int = 0
    balanced_classes: bool = True

    def __post_init__(self) -> None:
        positive_int("pca_components", self.pca_components)
        positive_float("regularization_c", self.regularization_c)
        positive_int("max_iter", self.max_iter)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if not isinstance(self.balanced_classes, bool):
            raise TypeError("balanced_classes must be a boolean")


@dataclass(frozen=True, slots=True)
class SemanticProbeDiagnostics:
    """Training and held-out calibration diagnostics."""

    train_count: int
    validation_count: int
    class_counts: dict[str, int]
    pca_components: int
    train_accuracy: float
    validation_accuracy: float | None
    validation_nll_uncalibrated: float | None
    validation_nll_calibrated: float | None
    temperature: float


@dataclass(frozen=True, slots=True)
class SemanticLinearProbe:
    """Portable PCA and multinomial linear classifier without pickle state."""

    class_names: tuple[str, ...]
    feature_mean: np.ndarray
    pca_components: np.ndarray
    coefficients: np.ndarray
    intercept: np.ndarray
    temperature: float
    config: SemanticProbeConfig
    fingerprint: str | None = None

    def __post_init__(self) -> None:
        class_count = len(self.class_names)
        if class_count < 2 or len(set(self.class_names)) != class_count:
            raise ValueError("class_names must contain at least two unique values")
        if self.feature_mean.ndim != 1:
            raise ValueError("feature_mean must be one-dimensional")
        if self.pca_components.ndim != 2:
            raise ValueError("pca_components must be two-dimensional")
        if self.pca_components.shape[1] != len(self.feature_mean):
            raise ValueError("PCA components do not match the input feature dimension")
        expected_shape = (class_count, self.pca_components.shape[0])
        if self.coefficients.shape != expected_shape:
            raise ValueError("classifier coefficients do not match classes and PCA")
        if self.intercept.shape != (class_count,):
            raise ValueError("classifier intercept does not match class count")
        positive_float("temperature", self.temperature)
        for name, array in (
            ("feature_mean", self.feature_mean),
            ("pca_components", self.pca_components),
            ("coefficients", self.coefficients),
            ("intercept", self.intercept),
        ):
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must contain only finite values")
        expected = _probe_fingerprint(self)
        if self.fingerprint is not None and self.fingerprint != expected:
            raise ValueError("semantic probe fingerprint does not match model content")
        object.__setattr__(self, "fingerprint", expected)

    def decision_function(self, features: np.ndarray) -> np.ndarray:
        """Return calibrated class logits for feature rows."""

        matrix = _feature_matrix(features, expected_dim=len(self.feature_mean))
        projected = (_l2_normalize(matrix) - self.feature_mean) @ self.pca_components.T
        logits = projected @ self.coefficients.T + self.intercept
        return logits / self.temperature

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return one calibrated probability row per feature vector."""

        logits = self.decision_function(features)
        logits -= np.max(logits, axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= np.sum(probabilities, axis=1, keepdims=True)
        return probabilities.astype(np.float32)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return integer class indices aligned with ``class_names``."""

        return np.argmax(self.predict_proba(features), axis=1).astype(np.int32)

    def predict_names(self, features: np.ndarray) -> np.ndarray:
        """Return class names for feature rows."""

        names = np.asarray(self.class_names)
        return names[self.predict(features)]

    def save(self, path: Path | str) -> Path:
        """Write a portable, fingerprinted NumPy model artifact."""

        target = Path(path)

        def write(stream) -> None:
            np.savez_compressed(
                stream,
                schema_version=np.int16(1),
                class_names_json=np.asarray(json.dumps(self.class_names)),
                feature_mean=np.asarray(self.feature_mean, dtype=np.float32),
                pca_components=np.asarray(self.pca_components, dtype=np.float32),
                coefficients=np.asarray(self.coefficients, dtype=np.float32),
                intercept=np.asarray(self.intercept, dtype=np.float32),
                temperature=np.float64(self.temperature),
                config_json=np.asarray(_canonical_json(asdict(self.config))),
                fingerprint=np.asarray(self.fingerprint),
            )

        write_binary_atomic(target, write)
        self.load(target)
        return target

    @classmethod
    def load(cls, path: Path | str) -> SemanticLinearProbe:
        """Load and validate a portable semantic probe."""

        with np.load(Path(path), allow_pickle=False) as data:
            if int(data["schema_version"]) != 1:
                raise ValueError("unsupported semantic probe schema")
            config = SemanticProbeConfig(**json.loads(str(data["config_json"])))
            return cls(
                class_names=tuple(json.loads(str(data["class_names_json"]))),
                feature_mean=data["feature_mean"],
                pca_components=data["pca_components"],
                coefficients=data["coefficients"],
                intercept=data["intercept"],
                temperature=float(data["temperature"]),
                config=config,
                fingerprint=str(data["fingerprint"]),
            )


@dataclass(frozen=True, slots=True)
class SemanticProbeFit:
    """A fitted semantic probe and its explicit diagnostics."""

    model: SemanticLinearProbe
    diagnostics: SemanticProbeDiagnostics


def fit_semantic_probe(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    *,
    validation_features: np.ndarray | None = None,
    validation_labels: np.ndarray | None = None,
    config: SemanticProbeConfig | None = None,
) -> SemanticProbeFit:
    """Fit PCA and a linear classifier using training rows only.

    Validation rows are used solely to fit one scalar probability temperature.
    Callers remain responsible for case-level train/validation partitioning.
    """

    config = config or SemanticProbeConfig()
    matrix = _feature_matrix(train_features)
    labels, class_names = _encode_labels(train_labels, expected_rows=len(matrix))
    if len(class_names) < 2:
        raise ValueError("semantic probe training requires at least two classes")
    validation = _validation_inputs(
        validation_features,
        validation_labels,
        expected_dim=matrix.shape[1],
        class_names=class_names,
    )
    try:
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise RuntimeError(
            "semantic probe fitting requires the 'semantic' optional dependency"
        ) from exc

    normalized = _l2_normalize(matrix)
    component_count = min(
        config.pca_components,
        normalized.shape[0],
        normalized.shape[1],
    )
    pca = PCA(
        n_components=component_count,
        svd_solver="auto",
        random_state=config.seed,
    )
    projected = pca.fit_transform(normalized)
    classifier = LogisticRegression(
        C=config.regularization_c,
        class_weight="balanced" if config.balanced_classes else None,
        max_iter=config.max_iter,
        random_state=config.seed,
        solver="lbfgs",
    )
    classifier.fit(projected, labels)
    coefficients, intercept = _expanded_classifier_parameters(
        classifier.coef_,
        classifier.intercept_,
        class_count=len(class_names),
    )
    provisional = SemanticLinearProbe(
        class_names=class_names,
        feature_mean=np.asarray(pca.mean_, dtype=np.float32),
        pca_components=np.asarray(pca.components_, dtype=np.float32),
        coefficients=coefficients,
        intercept=intercept,
        temperature=1.0,
        config=config,
    )
    validation_accuracy = None
    nll_before = None
    nll_after = None
    temperature = 1.0
    if validation is not None:
        validation_matrix, validation_encoded = validation
        uncalibrated_logits = provisional.decision_function(validation_matrix)
        temperature = _fit_temperature(uncalibrated_logits, validation_encoded)
        nll_before = _negative_log_likelihood(
            uncalibrated_logits,
            validation_encoded,
        )
        nll_after = _negative_log_likelihood(
            uncalibrated_logits / temperature,
            validation_encoded,
        )
    model = SemanticLinearProbe(
        class_names=provisional.class_names,
        feature_mean=provisional.feature_mean,
        pca_components=provisional.pca_components,
        coefficients=provisional.coefficients,
        intercept=provisional.intercept,
        temperature=temperature,
        config=config,
    )
    if validation is not None:
        validation_accuracy = float(
            np.mean(model.predict(validation[0]) == validation[1])
        )
    diagnostics = SemanticProbeDiagnostics(
        train_count=len(matrix),
        validation_count=0 if validation is None else len(validation[0]),
        class_counts={
            name: int(np.sum(labels == index)) for index, name in enumerate(class_names)
        },
        pca_components=component_count,
        train_accuracy=float(np.mean(model.predict(matrix) == labels)),
        validation_accuracy=validation_accuracy,
        validation_nll_uncalibrated=nll_before,
        validation_nll_calibrated=nll_after,
        temperature=temperature,
    )
    return SemanticProbeFit(model=model, diagnostics=diagnostics)


def _feature_matrix(
    features: np.ndarray,
    *,
    expected_dim: int | None = None,
) -> np.ndarray:
    matrix = np.asarray(features, dtype=np.float32)
    if matrix.ndim != 2 or not len(matrix):
        raise ValueError("features must be a non-empty two-dimensional array")
    if expected_dim is not None and matrix.shape[1] != expected_dim:
        raise ValueError("feature dimension does not match the fitted probe")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("features must contain only finite values")
    return matrix


def _l2_normalize(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, np.finfo(np.float32).eps)


def _encode_labels(
    labels: np.ndarray,
    *,
    expected_rows: int,
) -> tuple[np.ndarray, tuple[str, ...]]:
    values = np.asarray(labels)
    if values.ndim != 1 or len(values) != expected_rows:
        raise ValueError("labels must contain one value per feature row")
    text = np.asarray([str(value) for value in values], dtype=np.str_)
    classes, encoded = np.unique(text, return_inverse=True)
    return encoded.astype(np.int32), tuple(str(value) for value in classes)


def _validation_inputs(
    features: np.ndarray | None,
    labels: np.ndarray | None,
    *,
    expected_dim: int,
    class_names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray] | None:
    if features is None and labels is None:
        return None
    if features is None or labels is None:
        raise ValueError("validation features and labels must be provided together")
    matrix = _feature_matrix(features, expected_dim=expected_dim)
    values = np.asarray(labels)
    if values.ndim != 1 or len(values) != len(matrix):
        raise ValueError("validation labels must contain one value per feature row")
    mapping = {name: index for index, name in enumerate(class_names)}
    try:
        encoded = np.asarray([mapping[str(value)] for value in values], dtype=np.int32)
    except KeyError as error:
        raise ValueError(
            f"validation contains unknown class: {error.args[0]}"
        ) from error
    return matrix, encoded


def _expanded_classifier_parameters(
    coefficients: np.ndarray,
    intercept: np.ndarray,
    *,
    class_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    coefficients = np.asarray(coefficients, dtype=np.float32)
    intercept = np.asarray(intercept, dtype=np.float32)
    if class_count == 2 and coefficients.shape[0] == 1:
        coefficients = np.vstack([np.zeros_like(coefficients), coefficients])
        intercept = np.concatenate([np.zeros(1, dtype=np.float32), intercept])
    return coefficients, intercept


def _fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    try:
        from scipy.optimize import minimize_scalar
    except ImportError as exc:  # pragma: no cover - semantic extra includes scipy
        raise RuntimeError("probability calibration requires scipy") from exc

    result = minimize_scalar(
        lambda log_temperature: _negative_log_likelihood(
            logits / float(np.exp(log_temperature)),
            labels,
        ),
        bounds=(-4.0, 4.0),
        method="bounded",
        options={"xatol": 1e-6},
    )
    if not result.success or not np.isfinite(result.x):
        return 1.0
    return float(np.clip(np.exp(result.x), 0.05, 20.0))


def _negative_log_likelihood(logits: np.ndarray, labels: np.ndarray) -> float:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    log_normalizer = np.log(np.sum(np.exp(shifted), axis=1))
    selected = shifted[np.arange(len(labels)), labels]
    return float(np.mean(log_normalizer - selected))


def _probe_fingerprint(model: SemanticLinearProbe) -> str:
    digest = hashlib.sha256(b"histopia-semantic-linear-probe-v1\0")
    metadata = {
        "class_names": model.class_names,
        "temperature": model.temperature,
        "config": asdict(model.config),
    }
    digest.update(_canonical_json(metadata).encode())
    for name, values in (
        ("feature_mean", model.feature_mean),
        ("pca_components", model.pca_components),
        ("coefficients", model.coefficients),
        ("intercept", model.intercept),
    ):
        array = np.ascontiguousarray(values, dtype="<f4")
        digest.update(name.encode())
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))
