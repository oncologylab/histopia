from __future__ import annotations

import numpy as np
import pytest

from histopia.semantic._selection import (
    ClusterSelectionResult,
    _adjusted_edge_agreement,
    _required_cluster_size,
    _sparse_overlap_matrix,
    assign_lineage_display_ids,
    choose_best_k,
    select_cluster_count,
)


def _cluster_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(73)
    centers = np.array(
        [
            [-5.0, 0.0, 0.0, 0.0],
            [0.0, 5.0, 0.0, 0.0],
            [5.0, 0.0, 0.0, 0.0],
        ]
    )
    sections: list[np.ndarray] = []
    for _section in range(3):
        sections.append(
            np.vstack([rng.normal(center, 0.25, size=(30, 4)) for center in centers])
        )
    features = np.vstack(sections)
    offsets = np.array([0, 90, 180, 270], dtype=np.int64)

    within_edges: list[tuple[int, int]] = []
    for section_start in offsets[:-1]:
        for cluster_start in (0, 30, 60):
            indices = section_start + cluster_start + np.arange(30)
            within_edges.extend(zip(indices[:-1], indices[1:], strict=True))
    cross_edges = np.vstack(
        [
            np.column_stack([np.arange(90), np.arange(90, 180)]),
            np.column_stack([np.arange(90, 180), np.arange(180, 270)]),
        ]
    )
    return (
        features,
        offsets,
        np.asarray(within_edges, dtype=np.int64),
        cross_edges,
    )


def test_selection_recovers_known_cluster_count_reproducibly() -> None:
    arguments = _cluster_fixture()

    first = select_cluster_count(
        *arguments,
        k_values=(2, 3, 4, 5),
        seed=11,
        max_evaluation_samples=180,
        max_silhouette_samples=120,
    )
    second = select_cluster_count(
        *arguments,
        k_values=(2, 3, 4, 5),
        seed=11,
        max_evaluation_samples=180,
        max_silhouette_samples=120,
    )

    assert isinstance(first, ClusterSelectionResult)
    assert first.selected_k == 3
    assert tuple(metric.k for metric in first.evaluations) == (2, 3, 4, 5)
    assert first.evaluations == second.evaluations
    for k in (2, 3, 4, 5):
        np.testing.assert_array_equal(first.labels_by_k[k], second.labels_by_k[k])
        np.testing.assert_array_equal(
            first.display_ids_by_k[k], second.display_ids_by_k[k]
        )
    selected = next(metric for metric in first.evaluations if metric.k == 3)
    assert not selected.rejected
    assert len(selected.normalized_metrics) == 4
    assert selected.composite_score == max(
        metric.composite_score for metric in first.evaluations
    )
    assert -1.0 <= selected.within_section_edge_coherence <= 1.0
    assert -1.0 <= selected.cross_section_edge_continuity <= 1.0


def test_selection_rejects_tiny_cluster_with_scaled_synthetic_threshold() -> None:
    rng = np.random.default_rng(8)
    features = np.vstack(
        [
            rng.normal([-4.0, 0.0], 0.2, size=(100, 2)),
            rng.normal([4.0, 0.0], 0.2, size=(100, 2)),
            rng.normal([0.0, 30.0], 0.02, size=(3, 2)),
        ]
    )
    offsets = np.array([0, 203], dtype=np.int64)
    edges = np.column_stack([np.arange(202), np.arange(1, 203)])

    result = select_cluster_count(
        features,
        offsets,
        edges,
        np.empty((0, 2), dtype=np.int64),
        k_values=(2, 3),
        seed=4,
    )

    by_k = {metric.k: metric for metric in result.evaluations}
    assert result.selected_k == 2
    assert by_k[3].rejected
    assert by_k[3].rejection_reasons == ("tiny_cluster",)
    assert by_k[3].minimum_cluster_size == 3
    assert by_k[3].required_cluster_size > 3
    assert by_k[3].composite_score == float("-inf")


def test_best_k_uses_smaller_value_within_score_tolerance() -> None:
    assert choose_best_k({5: 0.81, 6: 0.82, 7: 0.84}) == 6
    assert choose_best_k({5: 0.821, 6: 0.82, 7: 0.84}) == 5
    assert choose_best_k({5: 0.99, 6: 1.0}, rejected_k=(5,)) == 6


def test_lineage_ids_follow_maximum_overlap_between_neighboring_k() -> None:
    labels_by_k = {
        2: np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32),
        3: np.array([2, 2, 2, 2, 0, 0, 1, 1], dtype=np.int32),
        4: np.array([3, 3, 3, 2, 0, 0, 1, 1], dtype=np.int32),
    }

    display_ids = assign_lineage_display_ids(labels_by_k)

    np.testing.assert_array_equal(display_ids[2], [0, 1])
    np.testing.assert_array_equal(display_ids[3], [1, 2, 0])
    np.testing.assert_array_equal(display_ids[4], [1, 2, 3, 0])


def test_lineage_reports_display_id_for_an_empty_rejected_cluster() -> None:
    labels_by_k = {
        2: np.array([0, 0, 0, 1, 1, 1], dtype=np.int32),
        3: np.array([2, 2, 2, 0, 0, 0], dtype=np.int32),
    }

    display_ids = assign_lineage_display_ids(labels_by_k)

    np.testing.assert_array_equal(display_ids[3], [1, 2, 0])


def test_edge_agreement_is_neutral_for_degenerate_endpoint_distributions() -> None:
    labels = np.zeros(4, dtype=np.int32)
    edges = np.array([[0, 2], [1, 3]], dtype=np.int64)

    assert _adjusted_edge_agreement(labels, edges) == 0.0


@pytest.mark.parametrize(
    ("offsets", "within_edges", "cross_edges", "k_values"),
    [
        ([0.0, 3.5, 6.0], [[0, 1]], [[0, 3]], (2,)),
        ([0, 3, 6], [[0.0, 1.5]], [[0, 3]], (2,)),
        ([0, 3, 6], [[0, 1]], [[0.0, 3.5]], (2,)),
        ([0, 3, 6], [[0, 1]], [[0, 3]], (2.5,)),
    ],
)
def test_selection_rejects_fractional_indices_before_conversion(
    offsets: list[float],
    within_edges: list[list[float]],
    cross_edges: list[list[float]],
    k_values: tuple[float, ...],
) -> None:
    features = np.array(
        [[-2.0, 0.0], [-1.9, 0.0], [-2.1, 0.0], [2.0, 0.0], [1.9, 0.0], [2.1, 0.0]]
    )

    with pytest.raises(ValueError, match="integer"):
        select_cluster_count(
            features,
            np.asarray(offsets),
            np.asarray(within_edges),
            np.asarray(cross_edges),
            k_values=k_values,
        )


def test_lineage_rejects_fractional_labels_before_conversion() -> None:
    with pytest.raises(ValueError, match="integer"):
        assign_lineage_display_ids(
            {2: np.array([0.0, 0.5, 1.0]), 3: np.array([0, 1, 2])}
        )


def test_cluster_size_guard_uses_full_fitted_population_not_metric_sample() -> None:
    rng = np.random.default_rng(22)
    features = np.vstack(
        [
            rng.normal([-3.0, 0.0], 0.1, size=(600, 2)),
            rng.normal([3.0, 0.0], 0.1, size=(600, 2)),
        ]
    )

    result = select_cluster_count(
        features,
        np.array([0, 1_200]),
        np.empty((0, 2), dtype=np.int64),
        np.empty((0, 2), dtype=np.int64),
        k_values=(2,),
        seed=3,
        max_evaluation_samples=20,
        max_silhouette_samples=20,
    )

    metric = result.evaluations[0]
    assert metric.minimum_cluster_size == 600
    assert metric.required_cluster_size == 50
    assert not metric.rejected


def test_cluster_size_threshold_only_scales_for_genuinely_tiny_populations() -> None:
    assert _required_cluster_size(203) == 5
    assert _required_cluster_size(999) == 20
    assert _required_cluster_size(1_000) == 50
    assert _required_cluster_size(100_000) == 100


def test_high_k_lineage_overlap_storage_scales_with_observed_pairs() -> None:
    occupied = 2_000
    previous_display = np.arange(occupied, dtype=np.int64)
    current_labels = np.arange(occupied, dtype=np.int64)
    old_ids = previous_display.copy()

    overlap = _sparse_overlap_matrix(
        previous_display,
        current_labels,
        old_ids,
        current_count=1_000_000,
    )

    assert overlap.shape == (occupied, 1_000_000)
    assert overlap.nnz == occupied
    storage_bytes = overlap.data.nbytes + overlap.indices.nbytes + overlap.indptr.nbytes
    assert storage_bytes < 100_000
