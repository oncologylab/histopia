"""Deterministic joint semantic atlas fitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from histopia.semantic._features import PatchFeatures
from histopia.semantic._graph import (
    DiffusionGuard,
    build_serial_graph,
    diffuse_labels,
    evaluate_diffusion_guard,
)


@dataclass(frozen=True, slots=True)
class AtlasClustering:
    cluster_count: int
    labels: np.ndarray
    joint_labels: np.ndarray
    centroids: np.ndarray
    diffusion_guard: DiffusionGuard | None


@dataclass(frozen=True, slots=True)
class JointAtlas:
    slide_ids: tuple[str, ...]
    section_offsets: np.ndarray
    pca_components: int
    pca_mean: np.ndarray
    pca_basis: np.ndarray
    clusterings: dict[int, AtlasClustering]


def balanced_sample_indices(
    section_sizes: tuple[int, ...],
    *,
    per_slide_cap: int,
    seed: int,
) -> np.ndarray:
    """Sample each section independently so large tissues do not dominate."""

    if per_slide_cap <= 0 or any(size < 0 for size in section_sizes):
        raise ValueError("sample sizes and cap must be valid")
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    offset = 0
    for size in section_sizes:
        count = min(size, per_slide_cap)
        local = (
            rng.choice(size, size=count, replace=False)
            if count < size
            else np.arange(size)
        )
        selected.append(np.sort(local) + offset)
        offset += size
    return np.concatenate(selected).astype(np.int64)


def fit_joint_atlas(
    sections: tuple[PatchFeatures, ...],
    *,
    cluster_counts: tuple[int, ...] = (7, 5, 10),
    pca_components: int = 64,
    balanced_patch_cap: int = 4096,
    seed: int = 0,
    regularize: bool = True,
    max_cross_section_distance_um: float = 112.0,
) -> JointAtlas:
    """Fit one normalized feature space and clustering across every section."""

    if not sections:
        raise ValueError("at least one section is required")
    feature_dims = {section.features.shape[1] for section in sections}
    if len(feature_dims) != 1:
        raise ValueError("all sections must use the same feature dimension")
    raw = np.concatenate([section.features.astype(np.float32) for section in sections])
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    normalized = raw / np.maximum(norms, np.finfo(np.float32).eps)
    sizes = tuple(len(section.features) for section in sections)
    sample = balanced_sample_indices(sizes, per_slide_cap=balanced_patch_cap, seed=seed)
    component_count = min(pca_components, normalized.shape[1], len(sample))
    if component_count <= 0:
        raise ValueError("PCA requires non-empty patch features")
    PCA, MiniBatchKMeans = _sklearn_estimators()
    pca = PCA(n_components=component_count, svd_solver="auto", random_state=seed)
    pca.fit(normalized[sample])
    projected = pca.transform(normalized).astype(np.float32)
    offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    graph = None
    if regularize:
        split_features = tuple(
            projected[offsets[i] : offsets[i + 1]] for i in range(len(sections))
        )
        graph = build_serial_graph(
            tuple(section.grid_rc for section in sections),
            tuple(section.reference_um_xy for section in sections),
            split_features,
            max_cross_section_distance_um=max_cross_section_distance_um,
        )

    clusterings: dict[int, AtlasClustering] = {}
    for cluster_count in cluster_counts:
        if cluster_count <= 1 or cluster_count > len(sample):
            raise ValueError("cluster count must be between 2 and sample size")
        model = MiniBatchKMeans(
            n_clusters=cluster_count,
            random_state=seed,
            batch_size=min(4096, max(256, len(sample))),
            n_init=10,
            reassignment_ratio=0.0,
        )
        model.fit(projected[sample])
        joint = model.predict(projected).astype(np.int32)
        joint, centroids = _canonicalize(joint, model.cluster_centers_)
        selected = joint
        guard = None
        if graph is not None:
            diffusion = diffuse_labels(joint, graph, n_clusters=cluster_count)
            before_consistency = _cross_edge_consistency(joint, graph)
            after_consistency = _cross_edge_consistency(diffusion.labels, graph)
            before_distance = _same_label_cross_distance(joint, graph, sections)
            after_distance = _same_label_cross_distance(
                diffusion.labels, graph, sections
            )
            guard = evaluate_diffusion_guard(
                joint,
                diffusion.labels,
                adjacent_consistency_before=before_consistency,
                adjacent_consistency_after=after_consistency,
                centroid_distance_before=before_distance,
                centroid_distance_after=after_distance,
                max_changed_fraction=0.25,
                max_centroid_worsening_fraction=0.10,
            )
            if guard.accepted:
                selected = diffusion.labels
        clusterings[cluster_count] = AtlasClustering(
            cluster_count,
            selected,
            joint,
            centroids.astype(np.float32),
            guard,
        )
    return JointAtlas(
        slide_ids=tuple(section.slide_id for section in sections),
        section_offsets=offsets,
        pca_components=component_count,
        pca_mean=np.asarray(pca.mean_, dtype=np.float32),
        pca_basis=np.asarray(pca.components_, dtype=np.float32),
        clusterings=clusterings,
    )


def _sklearn_estimators():
    try:
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.decomposition import PCA
    except ImportError as exc:
        raise RuntimeError(
            "joint semantic atlas fitting requires the 'semantic' extra"
        ) from exc
    return PCA, MiniBatchKMeans


def _canonicalize(
    labels: np.ndarray, centroids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    order = np.lexsort(np.round(np.asarray(centroids), 8).T[::-1])
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return inverse[labels].astype(np.int32), np.asarray(centroids)[order]


def _cross_edge_consistency(labels, graph) -> float:
    selected = graph.edge_kind == 1
    if not np.any(selected):
        return 1.0
    matches = labels[graph.source[selected]] == labels[graph.target[selected]]
    return float(np.mean(matches))


def _same_label_cross_distance(labels, graph, sections) -> float:
    selected = (graph.edge_kind == 1) & (graph.source < graph.target)
    same = selected & (labels[graph.source] == labels[graph.target])
    if not np.any(same):
        return float("inf")
    points = np.concatenate([section.reference_um_xy for section in sections])
    distances = np.linalg.norm(
        points[graph.source[same]] - points[graph.target[same]], axis=1
    )
    return float(np.mean(distances))
