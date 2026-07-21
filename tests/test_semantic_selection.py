from __future__ import annotations

import numpy as np

from histopia.semantic._selection import (
    ClusterSelectionResult,
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
