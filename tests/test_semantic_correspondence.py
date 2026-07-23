from __future__ import annotations

import tracemalloc

import numpy as np
import pytest

from histopia.semantic._correspondence import (
    AdjacentSectionCorrespondence,
    CorrespondenceConfig,
    match_adjacent_sections,
)


def test_correspondence_config_uses_coarse_to_fine_patch_width_defaults() -> None:
    config = CorrespondenceConfig(patch_width_um=100.0)

    assert config.search_radii_patch_widths == (8.0, 4.0, 2.0)
    assert config.context_radii_grid == (1, 2)
    assert config.patch_width_um == 100.0


def test_matching_reports_missing_and_distant_tiles_as_unmatched() -> None:
    grid = np.array([[0, 0], [0, 1], [0, 2]], dtype=np.int32)
    source_xy = np.array([[0.0, 0.0], [100.0, 0.0], [200.0, 0.0]])
    target_xy = np.array([[4.0, 2.0], [104.0, 2.0], [2_000.0, 0.0]])
    source_features = np.eye(3, dtype=np.float32)
    target_features = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )

    result = match_adjacent_sections(
        grid,
        source_xy,
        source_features,
        grid,
        target_xy,
        target_features,
        source_section=2,
        target_section=3,
        config=CorrespondenceConfig(patch_width_um=100.0),
    )

    assert isinstance(result, AdjacentSectionCorrespondence)
    np.testing.assert_array_equal(result.source_indices, [0, 1])
    np.testing.assert_array_equal(result.target_indices, [0, 1])
    np.testing.assert_array_equal(result.unmatched_source_indices, [2])
    np.testing.assert_array_equal(result.unmatched_target_indices, [2])
    assert result.source_section == 2
    assert result.target_section == 3
    assert result.confidence.shape == (2,)
    assert result.feature_similarity.shape == (2,)
    assert result.reciprocal_margin.shape == (2,)
    assert result.field_residual_um.shape == (2,)
    assert result.neighborhood_consistency.shape == (2,)
    assert result.estimated_displacement_um_xy.shape == (3, 2)

    with pytest.raises(ValueError, match="adjacent"):
        match_adjacent_sections(
            grid,
            source_xy,
            source_features,
            grid,
            target_xy,
            target_features,
            source_section=2,
            target_section=4,
            config=CorrespondenceConfig(patch_width_um=100.0),
        )


def test_matching_recovers_smooth_nonlinear_displacement_with_repeated_features() -> (
    None
):
    (
        source_grid,
        source_xy,
        source_features,
        target_grid,
        target_xy,
        target_features,
        target_truth,
        expected_target_xy,
    ) = _nonlinear_fixture()
    config = CorrespondenceConfig(patch_width_um=100.0)

    first = match_adjacent_sections(
        source_grid,
        source_xy,
        source_features,
        target_grid,
        target_xy,
        target_features,
        source_section=0,
        target_section=1,
        config=config,
    )
    second = match_adjacent_sections(
        source_grid,
        source_xy,
        source_features,
        target_grid,
        target_xy,
        target_features,
        source_section=0,
        target_section=1,
        config=config,
    )

    truth_for_matches = target_truth[first.target_indices]
    accepted_error = np.linalg.norm(
        target_xy[first.target_indices] - expected_target_xy[first.source_indices],
        axis=1,
    )
    assert len(first.source_indices) >= 0.65 * np.count_nonzero(target_truth >= 0)
    assert np.median(accepted_error) < config.patch_width_um
    assert np.mean(truth_for_matches == first.source_indices) > 0.9
    assert len(first.unmatched_source_indices) > 0
    assert len(first.unmatched_target_indices) > 0
    for name in AdjacentSectionCorrespondence.__dataclass_fields__:
        left = getattr(first, name)
        right = getattr(second, name)
        if isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        else:
            assert left == right


def test_matching_does_not_force_an_unsupported_distant_candidate() -> None:
    grid = np.array([[0, 0]], dtype=np.int32)
    features = np.array([[1.0, 0.0]], dtype=np.float32)

    result = match_adjacent_sections(
        grid,
        np.array([[0.0, 0.0]]),
        features,
        grid,
        np.array([[650.0, 0.0]]),
        features,
        source_section=0,
        target_section=1,
        config=CorrespondenceConfig(patch_width_um=100.0),
    )

    assert result.source_indices.size == 0
    assert result.target_indices.size == 0
    np.testing.assert_array_equal(result.unmatched_source_indices, [0])
    np.testing.assert_array_equal(result.unmatched_target_indices, [0])


def test_matching_uses_registered_geometry_when_cross_stain_features_shift() -> None:
    rows, columns = np.mgrid[:6, :7]
    grid = np.column_stack([rows.ravel(), columns.ravel()]).astype(np.int32)
    source_xy = grid[:, ::-1].astype(float) * 112.0
    target_xy = source_xy + np.array([24.0, -16.0])
    rng = np.random.default_rng(713)
    source_features = rng.normal(size=(len(grid), 16)).astype(np.float32)
    target_features = rng.normal(size=(len(grid), 16)).astype(np.float32)

    result = match_adjacent_sections(
        grid,
        source_xy,
        source_features,
        grid,
        target_xy,
        target_features,
        source_section=0,
        target_section=1,
        config=CorrespondenceConfig(patch_width_um=112.0),
    )

    assert len(result.source_indices) >= 0.75 * len(grid)
    assert np.mean(result.source_indices == result.target_indices) >= 0.95
    assert np.median(result.field_residual_um) < 0.25 * 112.0


def test_matching_rejects_an_isolated_near_decoy_without_runner_up() -> None:
    grid = np.array([[0, 0]], dtype=np.int32)
    features = np.array([[1.0, 0.0]], dtype=np.float32)

    result = match_adjacent_sections(
        grid,
        np.array([[0.0, 0.0]]),
        features,
        grid,
        np.array([[100.0, 0.0]]),
        features,
        source_section=0,
        target_section=1,
        config=CorrespondenceConfig(patch_width_um=100.0),
    )

    assert result.source_indices.size == 0
    np.testing.assert_array_equal(result.unmatched_source_indices, [0])
    np.testing.assert_array_equal(result.unmatched_target_indices, [0])


def test_returned_confidence_uses_the_final_refitted_field_residual() -> None:
    displacement = np.array(
        [
            112.347575,
            131.783295,
            -23.126909,
            23.122097,
            32.869060,
            0.947780,
            -134.870319,
            15.018651,
            2.112959,
            -7.248097,
            -30.651049,
            38.208470,
        ]
    )
    grid = np.column_stack(
        [
            np.zeros(len(displacement), dtype=np.int32),
            np.arange(len(displacement), dtype=np.int32),
        ]
    )
    source_xy = np.column_stack(
        [np.arange(len(displacement), dtype=float) * 100.0, np.zeros(len(grid))]
    )
    target_xy = source_xy + np.column_stack([displacement, np.zeros(len(displacement))])
    features = np.eye(len(displacement), dtype=np.float32)
    config = CorrespondenceConfig(
        patch_width_um=100.0,
        max_field_residual_patch_widths=0.8,
        min_neighborhood_consistency=0.30,
    )

    result = match_adjacent_sections(
        grid,
        source_xy,
        features,
        grid,
        target_xy,
        features,
        source_section=0,
        target_section=1,
        config=config,
    )

    feature_score = np.clip(result.feature_similarity, 0.0, 1.0)
    field_score = np.exp(-0.5 * (result.field_residual_um / config.patch_width_um) ** 2)
    evidence_score = np.maximum(
        feature_score, field_score * result.neighborhood_consistency
    )
    expected = np.power(
        evidence_score
        * np.clip(result.reciprocal_margin / 0.2, 0.0, 1.0)
        * field_score
        * result.neighborhood_consistency,
        0.25,
    )
    np.testing.assert_allclose(result.confidence, expected, rtol=1e-7, atol=1e-7)
    assert np.all(result.confidence >= config.min_confidence)
    assert np.all(
        result.field_residual_um
        <= config.max_field_residual_patch_widths * config.patch_width_um
    )


def test_sparse_matching_does_not_allocate_a_dense_pairwise_matrix() -> None:
    count = 3_000
    index = np.arange(count, dtype=np.float32)
    grid = np.column_stack(
        [np.zeros(count, dtype=np.int32), np.arange(count, dtype=np.int32)]
    )
    source_xy = np.column_stack([index * 500.0, np.zeros(count, dtype=np.float32)])
    target_xy = source_xy + np.array([50.0, 0.0], dtype=np.float32)
    features = np.column_stack(
        [
            np.sin(index * 0.013),
            np.cos(index * 0.013),
            np.sin(index * 0.031),
            np.cos(index * 0.031),
        ]
    )

    tracemalloc.start()
    result = match_adjacent_sections(
        grid,
        source_xy,
        features,
        grid,
        target_xy,
        features,
        source_section=0,
        target_section=1,
        config=CorrespondenceConfig(patch_width_um=100.0),
    )
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert peak_bytes < 32_000_000
    assert result.source_indices.size == 0
    assert result.unmatched_source_indices.size == count


def _nonlinear_fixture() -> tuple[np.ndarray, ...]:
    rows, columns = np.mgrid[:9, :11]
    source_grid = np.column_stack([rows.ravel(), columns.ravel()]).astype(np.int32)
    source_xy = source_grid[:, ::-1].astype(float) * 100.0
    rng = np.random.default_rng(41)
    morphology = rng.integers(0, 6, size=len(source_grid))
    source_features = np.eye(6, dtype=np.float32)[morphology]
    x, y = source_xy.T
    displacement = np.column_stack(
        [
            300.0 + 45.0 * np.sin(y / 220.0) + 0.025 * (y - 400.0),
            -180.0 + 35.0 * np.sin(x / 250.0) + 0.00008 * (x - 500.0) ** 2,
        ]
    )
    expected_target_xy = source_xy + displacement
    missing = (
        ((source_grid[:, 0] == 4) & np.isin(source_grid[:, 1], [4, 5, 6]))
        | ((source_grid[:, 0] == 1) & (source_grid[:, 1] == 8))
        | ((source_grid[:, 0] == 7) & (source_grid[:, 1] == 2))
    )
    retained = np.flatnonzero(~missing)

    false_source = np.arange(0, len(source_grid), 3, dtype=np.int64)
    false_grid = source_grid[false_source] + np.array([100, 100], dtype=np.int32)
    false_xy = source_xy[false_source] + np.array([650.0, -40.0])
    target_grid = np.concatenate([false_grid, source_grid[retained]])
    target_xy = np.concatenate([false_xy, expected_target_xy[retained]])
    target_features = np.concatenate(
        [source_features[false_source], source_features[retained]]
    )
    target_truth = np.concatenate(
        [np.full(len(false_source), -1, dtype=np.int64), retained]
    )
    return (
        source_grid,
        source_xy,
        source_features,
        target_grid,
        target_xy,
        target_features,
        target_truth,
        expected_target_xy,
    )
