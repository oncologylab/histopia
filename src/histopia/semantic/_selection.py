"""Deterministic, topology-aware selection of semantic cluster count."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from itertools import combinations

import numpy as np


@dataclass(frozen=True, slots=True)
class ClusterSelectionMetrics:
    """Raw and equally ranked evidence for one independently fitted K."""

    k: int
    silhouette: float
    stability_ari: float
    within_section_edge_coherence: float
    cross_section_edge_continuity: float
    minimum_cluster_size: int
    required_cluster_size: int
    rejected: bool
    rejection_reasons: tuple[str, ...]
    normalized_metrics: tuple[float, float, float, float]
    composite_score: float


@dataclass(frozen=True, slots=True)
class ClusterSelectionResult:
    """Selected K with metrics, independent labels, and lineage display IDs."""

    selected_k: int
    evaluations: tuple[ClusterSelectionMetrics, ...]
    labels_by_k: dict[int, np.ndarray]
    display_ids_by_k: dict[int, np.ndarray]


def select_cluster_count(
    projected_features: np.ndarray,
    section_offsets: np.ndarray,
    within_section_edges: np.ndarray,
    accepted_cross_section_edges: np.ndarray,
    *,
    k_values: Iterable[int] = range(5, 16),
    seed: int = 0,
    max_evaluation_samples: int = 10_000,
    max_silhouette_samples: int = 2_000,
) -> ClusterSelectionResult:
    """Evaluate independent MiniBatchKMeans fits and select a conservative K."""

    features, offsets, within_edges, cross_edges, values = _validated_inputs(
        projected_features,
        section_offsets,
        within_section_edges,
        accepted_cross_section_edges,
        k_values,
        max_evaluation_samples,
        max_silhouette_samples,
    )
    try:
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.metrics import adjusted_rand_score, silhouette_score
    except ImportError as exc:
        raise RuntimeError("cluster selection requires the 'semantic' extra") from exc

    evaluation_indices = _sample_indices(len(features), max_evaluation_samples, seed)
    required_size = _required_cluster_size(len(evaluation_indices))
    labels_by_k: dict[int, np.ndarray] = {}
    raw_evaluations: list[ClusterSelectionMetrics] = []
    for k in values:
        seed_labels: list[np.ndarray] = []
        for seed_offset in range(5):
            model = MiniBatchKMeans(
                n_clusters=k,
                random_state=seed + seed_offset * 9_973,
                batch_size=min(4_096, max(256, len(features))),
                n_init=1,
                reassignment_ratio=0.0,
            )
            labels = model.fit_predict(features)
            labels, _ = _canonicalize(labels, model.cluster_centers_)
            seed_labels.append(labels)
        primary = seed_labels[0]
        labels_by_k[k] = primary

        evaluated_labels = primary[evaluation_indices]
        counts = np.bincount(evaluated_labels, minlength=k)
        minimum_size = int(np.min(counts))
        rejected = minimum_size < required_size
        silhouette_indices = evaluation_indices[
            _sample_indices(
                len(evaluation_indices),
                max_silhouette_samples,
                seed + k * 101,
            )
        ]
        sampled_labels = primary[silhouette_indices]
        if len(np.unique(sampled_labels)) < 2 or len(sampled_labels) <= k:
            silhouette = -1.0
        else:
            silhouette = float(
                silhouette_score(features[silhouette_indices], sampled_labels)
            )
        stability = float(
            np.mean(
                [
                    adjusted_rand_score(
                        left[evaluation_indices], right[evaluation_indices]
                    )
                    for left, right in combinations(seed_labels, 2)
                ]
            )
        )
        raw_evaluations.append(
            ClusterSelectionMetrics(
                k=k,
                silhouette=silhouette,
                stability_ari=stability,
                within_section_edge_coherence=_adjusted_edge_agreement(
                    primary, within_edges
                ),
                cross_section_edge_continuity=_adjusted_edge_agreement(
                    primary, cross_edges
                ),
                minimum_cluster_size=minimum_size,
                required_cluster_size=required_size,
                rejected=rejected,
                rejection_reasons=("tiny_cluster",) if rejected else (),
                normalized_metrics=(0.0, 0.0, 0.0, 0.0),
                composite_score=float("-inf"),
            )
        )

    evaluations = _rank_evaluations(raw_evaluations)
    selected_k = choose_best_k(
        {metric.k: metric.composite_score for metric in evaluations},
        rejected_k=tuple(metric.k for metric in evaluations if metric.rejected),
    )
    return ClusterSelectionResult(
        selected_k=selected_k,
        evaluations=tuple(evaluations),
        labels_by_k=labels_by_k,
        display_ids_by_k=assign_lineage_display_ids(labels_by_k),
    )


def choose_best_k(
    composite_scores: Mapping[int, float],
    *,
    rejected_k: Iterable[int] = (),
    tolerance: float = 0.02,
) -> int:
    """Choose the smallest non-rejected K within tolerance of the best score."""

    rejected = set(rejected_k)
    candidates = {
        int(k): float(score)
        for k, score in composite_scores.items()
        if k not in rejected and np.isfinite(score)
    }
    if not candidates:
        raise ValueError("at least one non-rejected K is required")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    best = max(candidates.values())
    epsilon = np.finfo(float).eps * max(1.0, abs(best)) * 8
    return min(
        k for k, score in candidates.items() if best - score <= tolerance + epsilon
    )


def assign_lineage_display_ids(
    labels_by_k: Mapping[int, np.ndarray],
) -> dict[int, np.ndarray]:
    """Assign stable display IDs between neighboring K values by max overlap."""

    if not labels_by_k:
        raise ValueError("labels_by_k must not be empty")
    ordered = sorted(labels_by_k)
    labels = {k: np.asarray(labels_by_k[k], dtype=np.int64) for k in ordered}
    lengths = {len(value) for value in labels.values()}
    if len(lengths) != 1 or next(iter(lengths)) == 0:
        raise ValueError("lineage label arrays must be non-empty and aligned")
    mappings: dict[int, np.ndarray] = {}
    first_k = ordered[0]
    _validate_labels(labels[first_k], first_k)
    mappings[first_k] = np.arange(first_k, dtype=np.int32)
    next_display_id = first_k

    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:
        raise RuntimeError("cluster lineage assignment requires scipy") from exc

    for previous_k, current_k in zip(ordered[:-1], ordered[1:], strict=True):
        previous_labels = labels[previous_k]
        current_labels = labels[current_k]
        _validate_labels(current_labels, current_k)
        current_count = current_k
        previous_display = mappings[previous_k][previous_labels]
        old_ids = np.unique(previous_display)
        overlap = np.zeros((len(old_ids), current_count), dtype=np.int64)
        for row, display_id in enumerate(old_ids):
            overlap[row] = np.bincount(
                current_labels[previous_display == display_id], minlength=current_count
            )
        row_indices, columns = linear_sum_assignment(-overlap)
        mapping = np.full(current_count, -1, dtype=np.int32)
        for row, column in zip(row_indices, columns, strict=True):
            if overlap[row, column] > 0:
                mapping[column] = int(old_ids[row])
        for cluster in np.flatnonzero(mapping < 0):
            mapping[cluster] = next_display_id
            next_display_id += 1
        mappings[current_k] = mapping
    return mappings


def _validated_inputs(
    projected_features: np.ndarray,
    section_offsets: np.ndarray,
    within_section_edges: np.ndarray,
    accepted_cross_section_edges: np.ndarray,
    k_values: Iterable[int],
    max_evaluation_samples: int,
    max_silhouette_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[int, ...]]:
    features = np.asarray(projected_features, dtype=np.float64)
    offsets = np.asarray(section_offsets, dtype=np.int64)
    within_edges = np.asarray(within_section_edges, dtype=np.int64)
    cross_edges = np.asarray(accepted_cross_section_edges, dtype=np.int64)
    values = tuple(sorted(set(int(value) for value in k_values)))
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
    if not values or any(k < 2 or k >= len(features) for k in values):
        raise ValueError("K values must be unique values between 2 and patch count")
    if max_evaluation_samples <= 0 or max_silhouette_samples <= 0:
        raise ValueError("sample caps must be positive")
    for name, edges in (
        ("within-section", within_edges),
        ("cross-section", cross_edges),
    ):
        if edges.ndim != 2 or edges.shape[1:] != (2,):
            raise ValueError(f"{name} edges must have shape (edges, 2)")
        if np.any(edges < 0) or np.any(edges >= len(features)):
            raise ValueError(f"{name} edge indices are outside projected_features")
    section_for_patch = np.searchsorted(
        offsets[1:], np.arange(len(features)), side="right"
    )
    if len(within_edges) and np.any(
        section_for_patch[within_edges[:, 0]] != section_for_patch[within_edges[:, 1]]
    ):
        raise ValueError("within-section edges must not cross sections")
    if len(cross_edges) and np.any(
        section_for_patch[cross_edges[:, 0]] == section_for_patch[cross_edges[:, 1]]
    ):
        raise ValueError("cross-section edges must connect different sections")
    return features, offsets, within_edges, cross_edges, values


def _sample_indices(count: int, cap: int, seed: int) -> np.ndarray:
    if count <= cap:
        return np.arange(count, dtype=np.int64)
    return np.sort(np.random.default_rng(seed).choice(count, cap, replace=False))


def _required_cluster_size(evaluated_count: int) -> int:
    base = max(50, int(np.ceil(0.001 * evaluated_count)))
    if evaluated_count >= 1_000:
        return base
    return min(base, max(2, int(np.ceil(0.02 * evaluated_count))))


def _adjusted_edge_agreement(labels: np.ndarray, edges: np.ndarray) -> float:
    if not len(edges):
        return 0.0
    source = labels[edges[:, 0]]
    target = labels[edges[:, 1]]
    observed = float(np.mean(source == target))
    count = int(max(np.max(source), np.max(target))) + 1
    source_frequency = np.bincount(source, minlength=count) / len(source)
    target_frequency = np.bincount(target, minlength=count) / len(target)
    chance = float(source_frequency @ target_frequency)
    if chance >= 1.0 - np.finfo(float).eps:
        return 1.0 if observed >= 1.0 - np.finfo(float).eps else 0.0
    return float(np.clip((observed - chance) / (1.0 - chance), -1.0, 1.0))


def _rank_evaluations(
    evaluations: list[ClusterSelectionMetrics],
) -> list[ClusterSelectionMetrics]:
    valid = [index for index, metric in enumerate(evaluations) if not metric.rejected]
    if not valid:
        raise ValueError("every supplied K was rejected by the minimum cluster size")
    raw = np.array(
        [
            [
                evaluations[index].silhouette,
                evaluations[index].stability_ari,
                evaluations[index].within_section_edge_coherence,
                evaluations[index].cross_section_edge_continuity,
            ]
            for index in valid
        ]
    )
    normalized = np.empty_like(raw)
    for column in range(raw.shape[1]):
        normalized[:, column] = _normalized_ranks(raw[:, column])
    result = list(evaluations)
    for row, index in enumerate(valid):
        metric_values = tuple(float(value) for value in normalized[row])
        result[index] = replace(
            evaluations[index],
            normalized_metrics=metric_values,
            composite_score=float(np.mean(normalized[row])),
        )
    return result


def _normalized_ranks(values: np.ndarray) -> np.ndarray:
    if len(values) == 1:
        return np.ones(1)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2
        start = stop
    return ranks / (len(values) - 1)


def _canonicalize(
    labels: np.ndarray, centroids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    order = np.lexsort(np.round(np.asarray(centroids), 8).T[::-1])
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return inverse[labels].astype(np.int32), np.asarray(centroids)[order]


def _validate_labels(labels: np.ndarray, k: int) -> None:
    if labels.ndim != 1 or np.any(labels < 0) or np.any(labels >= k):
        raise ValueError(f"labels for K={k} must be one-dimensional values below K")
