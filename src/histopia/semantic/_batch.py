"""Conservative anchor-supported correction of section-level feature shifts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class BatchDiagnosticStage:
    """Quantitative diagnostics for one batch-correction feature space."""

    stage: str
    slide_variance_fraction: float
    slide_prediction_accuracy: float
    median_anchor_cosine_distance: float
    within_slide_knn_preservation: float
    correction_magnitude: float
    anchor_coverage: float


@dataclass(frozen=True, slots=True)
class BatchAcceptanceGuard:
    """Decision from the conservative scientific acceptance criteria."""

    accepted: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BatchCorrectionResult:
    """Accepted features, auditable proposal, offsets, and diagnostics."""

    corrected_features: np.ndarray
    proposed_features: np.ndarray
    section_corrections: np.ndarray
    unsupported_sections: tuple[int, ...]
    raw_diagnostics: BatchDiagnosticStage
    legacy_diagnostics: BatchDiagnosticStage
    corrected_diagnostics: BatchDiagnosticStage
    guard: BatchAcceptanceGuard


def correct_batch_offsets(
    projected_features: np.ndarray,
    section_offsets: np.ndarray,
    anchor_pairs: np.ndarray,
    anchor_weights: np.ndarray,
    *,
    seed: int = 0,
) -> BatchCorrectionResult:
    """Estimate robust additive section offsets from cross-section anchors.

    Each anchor-connected component uses its lowest section index as a fixed
    zero gauge. Sections without cross-section support are returned unchanged.
    """

    features, offsets, pairs, weights = _validated_inputs(
        projected_features, section_offsets, anchor_pairs, anchor_weights
    )
    section_for_patch = np.searchsorted(
        offsets[1:], np.arange(len(features)), side="right"
    )
    pair_sections = section_for_patch[pairs]
    corrections, unsupported = _solve_section_corrections(
        features, pairs, weights, pair_sections, len(offsets) - 1
    )
    proposed = features + corrections[section_for_patch]

    section_means = np.stack(
        [
            features[offsets[i] : offsets[i + 1]].mean(axis=0)
            for i in range(len(offsets) - 1)
        ]
    )
    global_mean = features.mean(axis=0)
    legacy_corrections = global_mean - section_means
    legacy = features + legacy_corrections[section_for_patch]
    coverage = float(len(np.unique(pairs)) / len(features))

    raw_diagnostics = _diagnose(
        "raw",
        features,
        features,
        offsets,
        pairs,
        np.zeros_like(corrections),
        coverage,
        seed,
    )
    legacy_diagnostics = _diagnose(
        "legacy",
        legacy,
        features,
        offsets,
        pairs,
        legacy_corrections,
        coverage,
        seed,
    )
    corrected_diagnostics = _diagnose(
        "anchor_corrected",
        proposed,
        features,
        offsets,
        pairs,
        corrections,
        coverage,
        seed,
    )
    reasons: list[str] = []
    if not (
        corrected_diagnostics.median_anchor_cosine_distance
        < raw_diagnostics.median_anchor_cosine_distance
    ):
        reasons.append("median_anchor_cosine_distance")
    if not (
        corrected_diagnostics.slide_variance_fraction
        < raw_diagnostics.slide_variance_fraction
    ):
        reasons.append("slide_variance_fraction")
    if corrected_diagnostics.within_slide_knn_preservation < 0.90:
        reasons.append("within_slide_knn_preservation")
    guard = BatchAcceptanceGuard(not reasons, tuple(reasons))
    return BatchCorrectionResult(
        corrected_features=proposed if guard.accepted else features.copy(),
        proposed_features=proposed,
        section_corrections=corrections,
        unsupported_sections=unsupported,
        raw_diagnostics=raw_diagnostics,
        legacy_diagnostics=legacy_diagnostics,
        corrected_diagnostics=corrected_diagnostics,
        guard=guard,
    )


def _validated_inputs(
    projected_features: np.ndarray,
    section_offsets: np.ndarray,
    anchor_pairs: np.ndarray,
    anchor_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features = np.asarray(projected_features, dtype=np.float64)
    offsets = _as_integer_array(section_offsets, "section_offsets")
    pairs = _as_integer_array(anchor_pairs, "anchor_pairs")
    weights = np.asarray(anchor_weights, dtype=np.float64)
    if features.ndim != 2 or not len(features) or not np.all(np.isfinite(features)):
        raise ValueError("projected_features must be a finite non-empty matrix")
    if (
        offsets.ndim != 1
        or len(offsets) < 2
        or offsets[0] != 0
        or offsets[-1] != len(features)
        or np.any(np.diff(offsets) <= 0)
    ):
        raise ValueError("section_offsets must strictly partition feature rows")
    if pairs.ndim != 2 or pairs.shape[1:] != (2,) or not len(pairs):
        raise ValueError("anchor_pairs must be a non-empty (anchors, 2) array")
    if np.any(pairs < 0) or np.any(pairs >= len(features)):
        raise ValueError("anchor pair indices are outside projected_features")
    if (
        weights.shape != (len(pairs),)
        or np.any(weights <= 0)
        or not np.all(np.isfinite(weights))
    ):
        raise ValueError(
            "anchor_weights must contain one positive finite value per pair"
        )
    patch_sections = np.searchsorted(offsets[1:], pairs, side="right")
    if np.any(patch_sections[:, 0] == patch_sections[:, 1]):
        raise ValueError("anchors must connect different sections")
    return features, offsets, pairs, weights


def _as_integer_array(values: np.ndarray, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.dtype.kind not in "iuf" or raw.dtype.kind == "b":
        raise ValueError(f"{name} must contain integer values")
    if raw.dtype.kind == "f":
        bounds = np.iinfo(np.int64)
        if (
            not np.all(np.isfinite(raw))
            or np.any(raw != np.floor(raw))
            or np.any(raw < bounds.min)
            or np.any(raw > bounds.max)
        ):
            raise ValueError(f"{name} must contain integer values")
    elif raw.dtype.kind == "u" and np.any(raw > np.iinfo(np.int64).max):
        raise ValueError(f"{name} must contain integer values")
    return raw.astype(np.int64)


def _solve_section_corrections(
    features: np.ndarray,
    pairs: np.ndarray,
    confidence: np.ndarray,
    pair_sections: np.ndarray,
    section_count: int,
) -> tuple[np.ndarray, tuple[int, ...]]:
    adjacency = [set() for _ in range(section_count)]
    for source, target in pair_sections:
        adjacency[int(source)].add(int(target))
        adjacency[int(target)].add(int(source))

    corrections = np.zeros((section_count, features.shape[1]), dtype=np.float64)
    supported = {index for index, neighbors in enumerate(adjacency) if neighbors}
    visited: set[int] = set()
    for start in sorted(supported):
        if start in visited:
            continue
        component: list[int] = []
        pending = [start]
        while pending:
            section = pending.pop()
            if section in visited:
                continue
            visited.add(section)
            component.append(section)
            pending.extend(sorted(adjacency[section] - visited, reverse=True))
        component.sort()
        component_set = set(component)
        selected = np.array(
            [
                left in component_set and right in component_set
                for left, right in pair_sections
            ],
            dtype=bool,
        )
        local_sections = component[1:]
        column = {section: index for index, section in enumerate(local_sections)}
        design = np.zeros((int(np.sum(selected)), len(local_sections)))
        selected_sections = pair_sections[selected]
        for row, (source, target) in enumerate(selected_sections):
            if int(source) in column:
                design[row, column[int(source)]] = 1.0
            if int(target) in column:
                design[row, column[int(target)]] = -1.0
        selected_pairs = pairs[selected]
        target = features[selected_pairs[:, 1]] - features[selected_pairs[:, 0]]
        solution = _robust_weighted_lstsq(design, target, confidence[selected])
        corrections[local_sections] = solution
    unsupported = tuple(
        index for index in range(section_count) if index not in supported
    )
    return corrections, unsupported


def _robust_weighted_lstsq(
    design: np.ndarray, target: np.ndarray, confidence: np.ndarray
) -> np.ndarray:
    robust = np.ones(len(design), dtype=np.float64)
    solution = np.zeros((design.shape[1], target.shape[1]), dtype=np.float64)
    for _ in range(20):
        combined = confidence * robust
        square_root = np.sqrt(combined)
        updated = np.linalg.lstsq(
            design * square_root[:, None],
            target * square_root[:, None],
            rcond=None,
        )[0]
        residual = np.linalg.norm(design @ updated - target, axis=1)
        scale = max(float(np.median(residual)), np.finfo(float).eps)
        threshold = 1.345 * scale
        robust = np.minimum(1.0, threshold / np.maximum(residual, threshold))
        if np.allclose(updated, solution, rtol=1e-10, atol=1e-12):
            solution = updated
            break
        solution = updated
    return solution


def _diagnose(
    stage: str,
    values: np.ndarray,
    raw: np.ndarray,
    offsets: np.ndarray,
    anchor_pairs: np.ndarray,
    corrections: np.ndarray,
    coverage: float,
    seed: int,
) -> BatchDiagnosticStage:
    return BatchDiagnosticStage(
        stage=stage,
        slide_variance_fraction=_slide_variance_fraction(values, offsets),
        slide_prediction_accuracy=_slide_prediction_accuracy(values, offsets, seed),
        median_anchor_cosine_distance=_median_anchor_cosine_distance(
            values, anchor_pairs
        ),
        within_slide_knn_preservation=_within_slide_knn_preservation(
            raw, values, offsets
        ),
        correction_magnitude=float(
            np.sqrt(np.mean(np.sum(np.square(corrections), axis=1)))
        ),
        anchor_coverage=coverage,
    )


def _slide_variance_fraction(values: np.ndarray, offsets: np.ndarray) -> float:
    overall = values.mean(axis=0)
    total = float(np.sum(np.square(values - overall)))
    if total <= np.finfo(float).eps:
        return 0.0
    between = 0.0
    for start, stop in zip(offsets[:-1], offsets[1:], strict=True):
        section_mean = values[start:stop].mean(axis=0)
        between += int(stop - start) * float(np.sum(np.square(section_mean - overall)))
    return between / total


def _slide_prediction_accuracy(
    values: np.ndarray, offsets: np.ndarray, seed: int
) -> float:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import balanced_accuracy_score
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
    except ImportError as exc:
        raise RuntimeError("batch diagnostics require the 'semantic' extra") from exc

    labels = np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))
    indices = _balanced_sample_indices(offsets, cap=5_000, seed=seed)
    sampled_labels = labels[indices]
    minimum_count = int(np.min(np.bincount(sampled_labels)))
    if len(offsets) == 2 or minimum_count < 2:
        return 1.0
    folds = StratifiedKFold(
        n_splits=min(5, minimum_count), shuffle=True, random_state=seed
    )
    estimator = LogisticRegression(max_iter=1_000, random_state=seed)
    predictions = cross_val_predict(
        estimator, values[indices], sampled_labels, cv=folds
    )
    return float(balanced_accuracy_score(sampled_labels, predictions))


def _balanced_sample_indices(offsets: np.ndarray, *, cap: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    per_section = max(1, cap // (len(offsets) - 1))
    selected: list[np.ndarray] = []
    for start, stop in zip(offsets[:-1], offsets[1:], strict=True):
        indices = np.arange(start, stop)
        if len(indices) > per_section:
            indices = np.sort(rng.choice(indices, per_section, replace=False))
        selected.append(indices)
    return np.concatenate(selected)


def _median_anchor_cosine_distance(
    values: np.ndarray, anchor_pairs: np.ndarray
) -> float:
    left = values[anchor_pairs[:, 0]]
    right = values[anchor_pairs[:, 1]]
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    similarity = np.sum(left * right, axis=1) / np.maximum(
        denominator, np.finfo(float).eps
    )
    return float(np.median(np.clip(1.0 - similarity, 0.0, 2.0)))


def _within_slide_knn_preservation(
    raw: np.ndarray, values: np.ndarray, offsets: np.ndarray
) -> float:
    preserved = 0.0
    comparisons = 0
    for start, stop in zip(offsets[:-1], offsets[1:], strict=True):
        count = int(stop - start)
        if count <= 1:
            continue
        raw_indices = _nonself_knn_indices(raw[start:stop])
        corrected_indices = _nonself_knn_indices(values[start:stop])
        neighbors = raw_indices.shape[1]
        for left, right in zip(raw_indices, corrected_indices, strict=True):
            preserved += len(set(left) & set(right)) / neighbors
            comparisons += 1
    return preserved / comparisons if comparisons else 1.0


def _nonself_knn_indices(values: np.ndarray) -> np.ndarray:
    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError as exc:
        raise RuntimeError("batch diagnostics require the 'semantic' extra") from exc

    neighbors = min(10, len(values) - 1)
    if neighbors <= 0:
        return np.empty((len(values), 0), dtype=np.int64)
    return (
        NearestNeighbors(n_neighbors=neighbors)
        .fit(values)
        .kneighbors(return_distance=False)
    )
