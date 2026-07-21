"""Sparse topology regularization across registered serial sections."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class GraphEdges:
    """Directed weighted edges over concatenated section patches."""

    source: np.ndarray
    target: np.ndarray
    weight: np.ndarray
    section_offsets: np.ndarray
    edge_kind: np.ndarray


@dataclass(frozen=True, slots=True)
class DiffusionResult:
    labels: np.ndarray
    probabilities: np.ndarray
    iterations: int
    converged: bool


@dataclass(frozen=True, slots=True)
class DiffusionGuard:
    accepted: bool
    changed_fraction: float
    reasons: tuple[str, ...]


def build_serial_graph(
    grid_rc_by_section: tuple[np.ndarray, ...],
    reference_um_xy_by_section: tuple[np.ndarray, ...],
    features_by_section: tuple[np.ndarray, ...],
    *,
    max_cross_section_distance_um: float,
) -> GraphEdges:
    """Build grid edges and reciprocal nearest edges between adjacent sections."""

    count = len(grid_rc_by_section)
    if count == 0 or not (
        len(reference_um_xy_by_section) == len(features_by_section) == count
    ):
        raise ValueError("section arrays must be non-empty and aligned")
    offsets = np.concatenate(
        [np.array([0], dtype=np.int64), np.cumsum([len(x) for x in grid_rc_by_section])]
    )
    sources: list[int] = []
    targets: list[int] = []
    weights: list[float] = []
    kinds: list[int] = []

    for section, grid in enumerate(grid_rc_by_section):
        lookup = {tuple(int(x) for x in rc): i for i, rc in enumerate(grid)}
        for local, (row, col) in enumerate(grid):
            for neighbor in ((int(row) + 1, int(col)), (int(row), int(col) + 1)):
                other = lookup.get(neighbor)
                if other is not None:
                    _append_undirected(
                        sources,
                        targets,
                        weights,
                        kinds,
                        int(offsets[section] + local),
                        int(offsets[section] + other),
                        _feature_weight(
                            features_by_section[section][local],
                            features_by_section[section][other],
                        ),
                        0,
                    )

    for section in range(count - 1):
        left = np.asarray(reference_um_xy_by_section[section], dtype=float)
        right = np.asarray(reference_um_xy_by_section[section + 1], dtype=float)
        left_to_right, left_distance = _nearest(left, right)
        right_to_left, _ = _nearest(right, left)
        for left_index, right_index in enumerate(left_to_right):
            if right_to_left[right_index] != left_index:
                continue
            distance = float(left_distance[left_index])
            if distance > max_cross_section_distance_um:
                continue
            spatial = np.exp(-((distance / max_cross_section_distance_um) ** 2))
            feature = _feature_weight(
                features_by_section[section][left_index],
                features_by_section[section + 1][right_index],
            )
            _append_undirected(
                sources,
                targets,
                weights,
                kinds,
                int(offsets[section] + left_index),
                int(offsets[section + 1] + right_index),
                float(spatial * feature),
                1,
            )

    return GraphEdges(
        source=np.asarray(sources, dtype=np.int64),
        target=np.asarray(targets, dtype=np.int64),
        weight=np.asarray(weights, dtype=np.float32),
        section_offsets=offsets,
        edge_kind=np.asarray(kinds, dtype=np.uint8),
    )


def diffuse_labels(
    labels: np.ndarray,
    graph: GraphEdges,
    *,
    n_clusters: int,
    alpha: float = 0.35,
    max_iterations: int = 20,
    tolerance: float = 1e-5,
) -> DiffusionResult:
    """Diffuse one-hot cluster evidence while retaining the joint-atlas prior."""

    labels = np.asarray(labels, dtype=np.int32)
    if not 0 <= alpha < 1:
        raise ValueError("alpha must be in [0, 1)")
    prior = np.eye(n_clusters, dtype=np.float32)[labels]
    probability = prior.copy()
    degree = np.zeros(len(labels), dtype=np.float32)
    np.add.at(degree, graph.source, graph.weight)
    degree = np.maximum(degree, np.finfo(np.float32).eps)
    converged = False
    iteration = 0
    for _iteration in range(1, max_iterations + 1):
        iteration = _iteration
        messages = np.zeros_like(probability)
        np.add.at(
            messages,
            graph.source,
            graph.weight[:, None] * probability[graph.target],
        )
        updated = (1 - alpha) * prior + alpha * messages / degree[:, None]
        if float(np.max(np.abs(updated - probability))) <= tolerance:
            converged = True
            probability = updated
            break
        probability = updated
    return DiffusionResult(
        labels=np.argmax(probability, axis=1).astype(np.int32),
        probabilities=probability,
        iterations=iteration,
        converged=converged,
    )


def evaluate_diffusion_guard(
    initial_labels: np.ndarray,
    proposed_labels: np.ndarray,
    *,
    adjacent_consistency_before: float,
    adjacent_consistency_after: float,
    centroid_distance_before: float,
    centroid_distance_after: float,
    max_changed_fraction: float,
    max_centroid_worsening_fraction: float,
) -> DiffusionGuard:
    """Accept graph regularization only when all conservative gates pass."""

    initial = np.asarray(initial_labels)
    proposed = np.asarray(proposed_labels)
    if initial.shape != proposed.shape or initial.size == 0:
        raise ValueError("label arrays must be non-empty and have equal shape")
    changed = float(np.mean(initial != proposed))
    reasons: list[str] = []
    if adjacent_consistency_after < adjacent_consistency_before:
        reasons.append("adjacent_consistency")
    if changed > max_changed_fraction:
        reasons.append("changed_fraction")
    allowed_distance = centroid_distance_before * (1 + max_centroid_worsening_fraction)
    if centroid_distance_after > allowed_distance:
        reasons.append("centroid_distance")
    return DiffusionGuard(not reasons, changed, tuple(reasons))


def _nearest(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not len(source) or not len(target):
        raise ValueError("nearest-neighbor sections must contain patches")
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise RuntimeError("semantic graph construction requires scipy") from exc
    distances, indices = cKDTree(target).query(source, k=1)
    return np.asarray(indices, dtype=np.int64), np.asarray(distances, dtype=float)


def _feature_weight(left: np.ndarray, right: np.ndarray) -> float:
    distance = float(np.linalg.norm(np.asarray(left) - np.asarray(right)))
    return float(np.exp(-(distance**2) / 2))


def _append_undirected(
    sources: list[int],
    targets: list[int],
    weights: list[float],
    kinds: list[int],
    left: int,
    right: int,
    weight: float,
    kind: int,
) -> None:
    sources.extend((left, right))
    targets.extend((right, left))
    weights.extend((weight, weight))
    kinds.extend((kind, kind))
