from __future__ import annotations

import numpy as np
import pytest

from histopia.semantic._correspondence import AdjacentSectionCorrespondence
from histopia.semantic._graph import (
    EdgeKind,
    GraphEdges,
    build_serial_graph,
    diffuse_labels,
    evaluate_diffusion_guard,
)


def test_serial_graph_has_grid_and_reciprocal_adjacent_section_edges() -> None:
    grids = (
        np.array([[0, 0], [0, 1], [1, 0]], dtype=np.int32),
        np.array([[0, 0], [0, 1], [1, 0]], dtype=np.int32),
    )
    points = (
        np.array([[0, 0], [100, 0], [0, 100]], dtype=float),
        np.array([[2, 1], [102, 1], [300, 300]], dtype=float),
    )
    features = (
        np.array([[1, 0], [0, 1], [1, 1]], dtype=float),
        np.array([[1, 0], [0, 1], [-1, -1]], dtype=float),
    )

    graph = build_serial_graph(
        grids,
        points,
        features,
        max_cross_section_distance_um=20,
    )

    pairs = set(zip(graph.source.tolist(), graph.target.tolist(), strict=True))
    assert (0, 1) in pairs
    assert (0, 3) in pairs
    assert (1, 4) in pairs
    assert (2, 5) not in pairs
    np.testing.assert_array_equal(graph.section_offsets, [0, 3, 6])
    assert np.all((graph.weight > 0) & (graph.weight <= 1))
    assert set(graph.edge_kind.tolist()) == {
        EdgeKind.WITHIN_SECTION_SPATIAL,
        EdgeKind.CROSS_SECTION_SPATIAL,
    }


def test_edge_kinds_are_typed_and_preserve_legacy_integer_values() -> None:
    assert EdgeKind.WITHIN_SECTION_SPATIAL == 0
    assert EdgeKind.CROSS_SECTION_SPATIAL == 1
    assert EdgeKind.CROSS_SECTION_MORPHOLOGY == 2
    assert EdgeKind.CROSS_SECTION_CONSENSUS == 3


def test_cross_section_provenance_and_weights_represent_available_evidence() -> None:
    grids = tuple(np.array([[0, 0], [0, 2], [0, 4]], dtype=np.int32) for _ in range(2))
    points = (
        np.array([[0.0, 0.0], [100.0, 0.0], [200.0, 0.0]]),
        np.array([[1.0, 0.0], [101.0, 0.0], [1_000.0, 0.0]]),
    )
    features = (
        np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]]),
        np.array([[10.0, 0.0], [10.0, 0.0], [1.0, 0.0]]),
    )
    correspondence = AdjacentSectionCorrespondence(
        source_section=0,
        target_section=1,
        source_indices=np.array([0, 2], dtype=np.int64),
        target_indices=np.array([0, 2], dtype=np.int64),
        confidence=np.array([0.8, 0.7], dtype=np.float32),
        feature_similarity=np.array([0.9, 0.95], dtype=np.float32),
        reciprocal_margin=np.array([0.2, 0.2], dtype=np.float32),
        field_residual_um=np.array([5.0, 5.0], dtype=np.float32),
        neighborhood_consistency=np.array([0.9, 0.9], dtype=np.float32),
        estimated_displacement_um_xy=np.array(
            [[1.0, 0.0], [1.0, 0.0], [800.0, 0.0]], dtype=np.float32
        ),
        unmatched_source_indices=np.array([1], dtype=np.int64),
        unmatched_target_indices=np.array([1], dtype=np.int64),
    )

    graph = build_serial_graph(
        grids,
        points,
        features,
        max_cross_section_distance_um=10.0,
        correspondences=(correspondence,),
    )

    forward = graph.source < graph.target
    evidence = {
        (int(source), int(target)): (EdgeKind(kind), float(weight))
        for source, target, kind, weight in zip(
            graph.source[forward],
            graph.target[forward],
            graph.edge_kind[forward],
            graph.weight[forward],
            strict=True,
        )
    }
    assert set(evidence) == {(0, 3), (1, 4), (2, 5)}
    assert evidence[0, 3][0] is EdgeKind.CROSS_SECTION_CONSENSUS
    assert evidence[1, 4][0] is EdgeKind.CROSS_SECTION_SPATIAL
    assert evidence[2, 5][0] is EdgeKind.CROSS_SECTION_MORPHOLOGY
    spatial_weight = float(np.exp(-0.01))
    assert evidence[1, 4][1] == pytest.approx(spatial_weight)
    assert evidence[2, 5][1] == pytest.approx(0.7)
    assert evidence[0, 3][1] == pytest.approx(np.sqrt(spatial_weight * 0.8))


def test_diffusion_is_deterministic_and_preserves_confident_local_structure() -> None:
    graph = GraphEdges(
        source=np.array([0, 1, 1, 2, 2, 3], dtype=np.int64),
        target=np.array([1, 0, 2, 1, 3, 2], dtype=np.int64),
        weight=np.ones(6, dtype=np.float32),
        section_offsets=np.array([0, 2, 4], dtype=np.int64),
        edge_kind=np.zeros(6, dtype=np.uint8),
    )
    labels = np.array([0, 0, 1, 1], dtype=np.int32)

    first = diffuse_labels(labels, graph, n_clusters=2, alpha=0.35)
    second = diffuse_labels(labels, graph, n_clusters=2, alpha=0.35)

    np.testing.assert_array_equal(first.labels, labels)
    np.testing.assert_allclose(first.probabilities, second.probabilities)
    assert first.iterations <= 20


def test_diffusion_can_correct_a_patch_surrounded_by_another_region() -> None:
    graph = GraphEdges(
        source=np.array([0, 1, 1, 2, 1, 3, 0, 2, 2, 3, 3, 0], dtype=np.int64),
        target=np.array([1, 0, 2, 1, 3, 1, 2, 0, 3, 2, 0, 3], dtype=np.int64),
        weight=np.ones(12, dtype=np.float32),
        section_offsets=np.array([0, 4], dtype=np.int64),
        edge_kind=np.zeros(12, dtype=np.uint8),
    )
    labels = np.array([0, 1, 0, 0], dtype=np.int32)

    result = diffuse_labels(
        labels,
        graph,
        n_clusters=2,
        alpha=0.35,
        edge_kinds=(EdgeKind.WITHIN_SECTION_SPATIAL,),
    )

    assert result.labels.tolist() == [0, 0, 0, 0]


def test_diffusion_uses_only_consensus_edges_by_default() -> None:
    source = np.array([0, 0, 0], dtype=np.int64)
    target = np.array([1, 2, 3], dtype=np.int64)
    labels = np.array([1, 0, 0, 0], dtype=np.int32)

    morphology_graph = GraphEdges(
        source=source,
        target=target,
        weight=np.ones(3, dtype=np.float32),
        section_offsets=np.array([0, 1, 4], dtype=np.int64),
        edge_kind=np.full(3, EdgeKind.CROSS_SECTION_MORPHOLOGY, dtype=np.uint8),
    )
    consensus_graph = GraphEdges(
        source=source,
        target=target,
        weight=np.ones(3, dtype=np.float32),
        section_offsets=np.array([0, 1, 4], dtype=np.int64),
        edge_kind=np.full(3, EdgeKind.CROSS_SECTION_CONSENSUS, dtype=np.uint8),
    )
    within_graph = GraphEdges(
        source=source,
        target=target,
        weight=np.ones(3, dtype=np.float32),
        section_offsets=np.array([0, 4], dtype=np.int64),
        edge_kind=np.full(3, EdgeKind.WITHIN_SECTION_SPATIAL, dtype=np.uint8),
    )

    ignored = diffuse_labels(labels, morphology_graph, n_clusters=2)
    explicit = diffuse_labels(
        labels,
        morphology_graph,
        n_clusters=2,
        edge_kinds=(EdgeKind.CROSS_SECTION_MORPHOLOGY,),
    )
    consensus = diffuse_labels(labels, consensus_graph, n_clusters=2)
    within_ignored = diffuse_labels(labels, within_graph, n_clusters=2)
    within_explicit = diffuse_labels(
        labels,
        within_graph,
        n_clusters=2,
        edge_kinds=(EdgeKind.WITHIN_SECTION_SPATIAL,),
    )

    np.testing.assert_array_equal(ignored.labels, labels)
    assert explicit.labels.tolist() == [0, 0, 0, 0]
    assert consensus.labels.tolist() == [0, 0, 0, 0]
    np.testing.assert_array_equal(within_ignored.labels, labels)
    assert within_explicit.labels.tolist() == [0, 0, 0, 0]


def test_diffusion_guard_rejects_excessive_label_changes() -> None:
    initial = np.array([0, 0, 1, 1], dtype=np.int32)
    proposed = np.array([1, 1, 0, 1], dtype=np.int32)

    decision = evaluate_diffusion_guard(
        initial,
        proposed,
        adjacent_consistency_before=0.4,
        adjacent_consistency_after=0.6,
        centroid_distance_before=10.0,
        centroid_distance_after=10.5,
        max_changed_fraction=0.25,
        max_centroid_worsening_fraction=0.10,
    )

    assert not decision.accepted
    assert decision.changed_fraction == 0.75
    assert "changed_fraction" in decision.reasons
