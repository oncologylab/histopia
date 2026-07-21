from __future__ import annotations

import numpy as np
import pytest

from histopia.semantic import PatchFeatures
from histopia.semantic import _atlas as atlas_module
from histopia.semantic._atlas import (
    _cross_edge_consistency,
    _normalize_section_features,
    _same_label_cross_distance,
    balanced_sample_indices,
    fit_joint_atlas,
)
from histopia.semantic._graph import EdgeKind, GraphEdges


def _section(slide_id: str, shift: float) -> PatchFeatures:
    rng = np.random.default_rng(4)
    first = rng.normal([1, 0, 0], 0.03, size=(6, 3))
    second = rng.normal([0, 1, 0], 0.03, size=(6, 3))
    features = np.vstack([first, second]).astype(np.float32)
    grid = np.array([(row, col) for row in range(3) for col in range(4)])
    xy = np.column_stack([grid[:, 1] * 112 + shift, grid[:, 0] * 112])
    return PatchFeatures(
        slide_id=slide_id,
        features=features,
        grid_rc=grid,
        native_xy=xy,
        reference_um_xy=xy,
        tissue_fraction=np.ones(12, dtype=np.float32),
        grid_shape=(3, 4),
        patch_size_px=224,
        analysis_mpp=0.5,
    )


def test_balanced_sample_caps_each_slide_deterministically() -> None:
    first = balanced_sample_indices((2, 8, 4), per_slide_cap=3, seed=17)
    second = balanced_sample_indices((2, 8, 4), per_slide_cap=3, seed=17)

    np.testing.assert_array_equal(first, second)
    assert len(first) == 8
    assert np.sum(first < 2) == 2
    assert np.sum((first >= 2) & (first < 10)) == 3
    assert np.sum(first >= 10) == 3


def test_section_normalization_removes_slide_level_feature_shift() -> None:
    first = np.array([[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    second = first + np.array([20.0, -7.0])

    normalized = _normalize_section_features(np.vstack([first, second]), (3, 3))

    np.testing.assert_allclose(normalized[:3], normalized[3:], atol=1e-6)
    np.testing.assert_allclose(normalized.mean(axis=0), 0.0, atol=1e-6)


def test_joint_atlas_is_deterministic_and_returns_each_sensitivity() -> None:
    sections = (_section("a", 0), _section("b", 2))

    first = fit_joint_atlas(
        sections,
        cluster_counts=(2, 3),
        pca_components=2,
        balanced_patch_cap=12,
        seed=8,
        regularize=False,
    )
    second = fit_joint_atlas(
        sections,
        cluster_counts=(2, 3),
        pca_components=2,
        balanced_patch_cap=12,
        seed=8,
        regularize=False,
    )

    assert set(first.clusterings) == {2, 3}
    np.testing.assert_array_equal(
        first.clusterings[2].labels,
        second.clusterings[2].labels,
    )
    assert first.section_offsets.tolist() == [0, 12, 24]
    assert first.pca_components == 2
    assert first.clusterings[2].labels.shape == (24,)


def test_regularized_atlas_builds_and_passes_adjacent_correspondences(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sections = (_section("a", 0), _section("b", 2), _section("c", 4))
    real_builder = atlas_module.build_serial_graph
    captured: list[object] = []

    def capture_builder(*args: object, **kwargs: object) -> GraphEdges:
        captured.append(kwargs.get("correspondences"))
        return real_builder(*args, **kwargs)

    monkeypatch.setattr(atlas_module, "build_serial_graph", capture_builder)

    result = fit_joint_atlas(
        sections,
        cluster_counts=(2,),
        pca_components=2,
        balanced_patch_cap=12,
        seed=8,
        regularize=True,
    )

    assert len(captured) == 1
    correspondences = captured[0]
    assert isinstance(correspondences, tuple)
    assert [(item.source_section, item.target_section) for item in correspondences] == [
        (0, 1),
        (1, 2),
    ]
    assert all(item.source_indices.size > 0 for item in correspondences)
    assert result.clusterings[2].diffusion_guard is not None


def test_atlas_guard_metrics_use_the_consensus_edges_used_by_diffusion() -> None:
    sections = (_section("a", 0), _section("b", 2))
    labels = np.zeros(24, dtype=np.int32)
    labels[12] = 1
    graph = GraphEdges(
        source=np.array([0, 0], dtype=np.int64),
        target=np.array([12, 13], dtype=np.int64),
        weight=np.ones(2, dtype=np.float32),
        section_offsets=np.array([0, 12, 24], dtype=np.int64),
        edge_kind=np.array(
            [EdgeKind.CROSS_SECTION_SPATIAL, EdgeKind.CROSS_SECTION_CONSENSUS],
            dtype=np.uint8,
        ),
    )

    assert _cross_edge_consistency(labels, graph) == 1.0
    assert _same_label_cross_distance(labels, graph, sections) == 114.0
