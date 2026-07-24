"""Deterministic joint semantic atlas fitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from histopia.semantic._batch import BatchCorrectionResult, correct_batch_offsets
from histopia.semantic._correspondence import (
    AdjacentSectionCorrespondence,
    CorrespondenceConfig,
    match_adjacent_sections,
)
from histopia.semantic._features import PatchFeatures
from histopia.semantic._graph import (
    DiffusionGuard,
    EdgeKind,
    GraphEdges,
    build_serial_graph,
    diffuse_labels,
    evaluate_diffusion_guard,
)
from histopia.semantic._selection import ClusterSelectionResult, select_cluster_count


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
    selected_k: int | None = None
    graph: GraphEdges | None = None
    correspondences: tuple[AdjacentSectionCorrespondence, ...] = ()
    batch_correction: BatchCorrectionResult | None = None
    cluster_selection: ClusterSelectionResult | None = None


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
    sizes = tuple(len(section.features) for section in sections)
    normalized = _normalize_section_features(raw, sizes)
    sample = balanced_sample_indices(sizes, per_slide_cap=balanced_patch_cap, seed=seed)
    component_count = min(pca_components, normalized.shape[1], len(sample))
    if component_count <= 0:
        raise ValueError("PCA requires non-empty patch features")
    PCA, _ = _sklearn_estimators()
    pca = PCA(n_components=component_count, svd_solver="auto", random_state=seed)
    pca.fit(normalized[sample])
    projected = pca.transform(normalized).astype(np.float32)
    offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    graph = None
    correspondences: tuple[AdjacentSectionCorrespondence, ...] = ()
    batch_correction = None
    selection = None
    if regularize:
        split_features = tuple(
            projected[offsets[i] : offsets[i + 1]] for i in range(len(sections))
        )
        correspondences = _match_correspondences(
            sections,
            split_features,
        )
        graph = build_serial_graph(
            tuple(section.grid_rc for section in sections),
            tuple(section.reference_um_xy for section in sections),
            split_features,
            max_cross_section_distance_um=max_cross_section_distance_um,
            correspondences=correspondences,
        )

        consensus = (graph.edge_kind == EdgeKind.CROSS_SECTION_CONSENSUS) & (
            graph.source < graph.target
        )
        anchor_pairs = np.column_stack(
            [graph.source[consensus], graph.target[consensus]]
        ).astype(np.int64)
        if len(anchor_pairs):
            batch_correction = correct_batch_offsets(
                projected,
                offsets,
                anchor_pairs,
                graph.weight[consensus],
                seed=seed,
            )
            if batch_correction.guard.accepted:
                projected = batch_correction.corrected_features.astype(np.float32)
                split_features = tuple(
                    projected[offsets[i] : offsets[i + 1]] for i in range(len(sections))
                )
                correspondences = _match_correspondences(sections, split_features)
                graph = build_serial_graph(
                    tuple(section.grid_rc for section in sections),
                    tuple(section.reference_um_xy for section in sections),
                    split_features,
                    max_cross_section_distance_um=max_cross_section_distance_um,
                    correspondences=correspondences,
                )

    counts = tuple(dict.fromkeys(int(value) for value in cluster_counts))
    if any(count <= 1 or count > len(sample) for count in counts):
        raise ValueError("cluster count must be between 2 and sample size")
    within_edges = np.empty((0, 2), dtype=np.int64)
    cross_edges = np.empty((0, 2), dtype=np.int64)
    if graph is not None:
        within = (graph.edge_kind == EdgeKind.WITHIN_SECTION_SPATIAL) & (
            graph.source < graph.target
        )
        cross = (graph.edge_kind == EdgeKind.CROSS_SECTION_CONSENSUS) & (
            graph.source < graph.target
        )
        within_edges = np.column_stack([graph.source[within], graph.target[within]])
        cross_edges = np.column_stack([graph.source[cross], graph.target[cross]])
    selection = select_cluster_count(
        projected,
        offsets,
        within_edges,
        cross_edges,
        k_values=counts,
        seed=seed,
    )

    clusterings: dict[int, AtlasClustering] = {}
    for cluster_count in counts:
        joint = selection.labels_by_k[cluster_count]
        centroids = np.stack(
            [projected[joint == label].mean(axis=0) for label in range(cluster_count)]
        )
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
        selected_k=selection.selected_k,
        graph=graph,
        correspondences=correspondences,
        batch_correction=batch_correction,
        cluster_selection=selection,
    )


def _match_correspondences(
    sections: tuple[PatchFeatures, ...],
    features: tuple[np.ndarray, ...],
) -> tuple[AdjacentSectionCorrespondence, ...]:
    return tuple(
        match_adjacent_sections(
            sections[index].grid_rc,
            sections[index].reference_um_xy,
            features[index],
            sections[index + 1].grid_rc,
            sections[index + 1].reference_um_xy,
            features[index + 1],
            source_section=index,
            target_section=index + 1,
            config=CorrespondenceConfig(
                patch_width_um=0.5
                * (
                    sections[index].patch_size_px * sections[index].analysis_mpp
                    + sections[index + 1].patch_size_px
                    * sections[index + 1].analysis_mpp
                )
            ),
        )
        for index in range(len(sections) - 1)
    )


def _normalize_section_features(
    features: np.ndarray,
    section_sizes: tuple[int, ...],
) -> np.ndarray:
    """L2-normalize patches while preserving shifts for guarded correction."""

    features = np.asarray(features, dtype=np.float32)
    if sum(section_sizes) != len(features) or any(size <= 0 for size in section_sizes):
        raise ValueError("section sizes must partition all feature rows")
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, np.finfo(np.float32).eps)


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
    selected = graph.edge_kind == EdgeKind.CROSS_SECTION_CONSENSUS
    if not np.any(selected):
        return 1.0
    matches = labels[graph.source[selected]] == labels[graph.target[selected]]
    return float(np.mean(matches))


def _same_label_cross_distance(labels, graph, sections) -> float:
    selected = (graph.edge_kind == EdgeKind.CROSS_SECTION_CONSENSUS) & (
        graph.source < graph.target
    )
    same = selected & (labels[graph.source] == labels[graph.target])
    if not np.any(same):
        return float("inf")
    points = np.concatenate([section.reference_um_xy for section in sections])
    distances = np.linalg.norm(
        points[graph.source[same]] - points[graph.target[same]], axis=1
    )
    return float(np.mean(distances))
