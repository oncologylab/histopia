from __future__ import annotations

import tracemalloc

import numpy as np
import pytest

from histopia.semantic import _correspondence as correspondence_module
from histopia.semantic._correspondence import (
    AdjacentSectionCorrespondence,
    CorrespondenceConfig,
    _context_descriptors,
    _match_section_sequence,
    _neighborhood_consistency,
    _reciprocal_matches,
    _score_candidate_window,
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


def test_section_sequence_reuses_descriptors_and_matches_pairwise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows, columns = np.mgrid[:3, :4]
    grid = np.column_stack([rows.ravel(), columns.ravel()]).astype(np.int32)
    coordinates = tuple(
        grid[:, ::-1].astype(float) * 100.0 + np.array([shift, 0.0])
        for shift in (0.0, 2.0, 4.0)
    )
    rng = np.random.default_rng(912)
    features = tuple(
        rng.normal(size=(len(grid), 8)).astype(np.float32) for _ in coordinates
    )
    configs = (
        CorrespondenceConfig(patch_width_um=100.0),
        CorrespondenceConfig(patch_width_um=100.0),
    )
    expected = tuple(
        match_adjacent_sections(
            grid,
            coordinates[index],
            features[index],
            grid,
            coordinates[index + 1],
            features[index + 1],
            source_section=index,
            target_section=index + 1,
            config=configs[index],
        )
        for index in range(2)
    )
    real_descriptor = correspondence_module._context_descriptors
    descriptor_calls = 0

    def counted_descriptor(*args, **kwargs):
        nonlocal descriptor_calls
        descriptor_calls += 1
        return real_descriptor(*args, **kwargs)

    monkeypatch.setattr(
        correspondence_module,
        "_context_descriptors",
        counted_descriptor,
    )
    observed = _match_section_sequence(
        (grid, grid, grid),
        coordinates,
        features,
        configs=configs,
        workers=2,
    )

    assert descriptor_calls == 3
    for expected_pair, observed_pair in zip(expected, observed, strict=True):
        for name in AdjacentSectionCorrespondence.__dataclass_fields__:
            left = getattr(expected_pair, name)
            right = getattr(observed_pair, name)
            if isinstance(left, np.ndarray):
                np.testing.assert_array_equal(left, right)
            else:
                assert left == right


def test_section_sequence_rejects_invalid_worker_count() -> None:
    grid = np.array([[0, 0]], dtype=np.int32)
    xy = np.array([[0.0, 0.0]])
    features = np.array([[1.0, 0.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="workers"):
        _match_section_sequence(
            (grid, grid),
            (xy, xy),
            (features, features),
            configs=(CorrespondenceConfig(patch_width_um=100.0),),
            workers=0,
        )


def test_section_sequence_preserves_empty_section_results() -> None:
    empty_grid = np.empty((0, 2), dtype=np.int32)
    empty_xy = np.empty((0, 2), dtype=np.float64)
    empty_features = np.empty((0, 3), dtype=np.float32)
    grid = np.array([[0, 0]], dtype=np.int32)
    xy = np.array([[0.0, 0.0]])
    features = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    config = CorrespondenceConfig(patch_width_um=100.0)

    direct = match_adjacent_sections(
        empty_grid,
        empty_xy,
        empty_features,
        grid,
        xy,
        features,
        source_section=0,
        target_section=1,
        config=config,
    )
    sequence = _match_section_sequence(
        (empty_grid, grid),
        (empty_xy, xy),
        (empty_features, features),
        configs=(config,),
    )

    assert direct.source_indices.size == 0
    assert sequence[0].source_indices.size == 0
    np.testing.assert_array_equal(direct.unmatched_target_indices, [0])
    np.testing.assert_array_equal(sequence[0].unmatched_target_indices, [0])


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


def test_vectorized_neighborhood_consistency_matches_reference_definition() -> None:
    rng = np.random.default_rng(82)
    xy = rng.uniform(0, 500, size=(40, 2))
    displacement = rng.normal(0, 30, size=(40, 2))
    patch_width = 100.0
    expected = []
    for index in range(len(xy)):
        distance = np.linalg.norm(xy - xy[index], axis=1)
        neighbors = np.flatnonzero(
            (distance <= 3.0 * patch_width) & (np.arange(len(xy)) != index)
        )
        delta = np.linalg.norm(displacement[neighbors] - displacement[index], axis=1)
        expected.append(np.mean(np.exp(-0.5 * (delta / patch_width) ** 2)))

    observed = _neighborhood_consistency(xy, displacement, patch_width)

    np.testing.assert_allclose(observed, expected, rtol=1e-6, atol=1e-6)


def test_context_descriptors_retain_every_projected_component() -> None:
    grid = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int32)
    features = np.arange(1, 49, dtype=np.float32).reshape(4, 12)

    descriptors = _context_descriptors(grid, features, (1,))

    assert descriptors.shape == (4, 12 + 8 * 12 + 8)


def test_matching_rejects_duplicate_grid_coordinates() -> None:
    grid = np.array([[0, 0], [0, 0]], dtype=np.int32)
    xy = np.array([[0.0, 0.0], [10.0, 0.0]])
    features = np.eye(2, dtype=np.float32)

    with pytest.raises(ValueError, match="unique"):
        match_adjacent_sections(
            grid,
            xy,
            features,
            grid,
            xy,
            features,
            source_section=0,
            target_section=1,
            config=CorrespondenceConfig(patch_width_um=10.0),
        )


def test_reciprocal_search_considers_every_candidate_inside_radius() -> None:
    source_xy = np.array([[0.0, 0.0]])
    target_xy = np.column_stack([np.linspace(1.0, 80.0, 80), np.zeros(80)])
    source_descriptor = np.array([[1.0, 0.0]], dtype=np.float32)
    target_descriptor = np.tile(
        np.array([[-1.0, 0.0]], dtype=np.float32),
        (80, 1),
    )
    target_descriptor[-1] = source_descriptor[0]
    config = CorrespondenceConfig(patch_width_um=10.0)

    source, target, _, _ = _reciprocal_matches(
        source_xy,
        target_xy,
        source_descriptor,
        target_descriptor,
        np.zeros_like(source_xy),
        radius_um=100.0,
        config=config,
    )

    np.testing.assert_array_equal(source, [0])
    np.testing.assert_array_equal(target, [79])


def test_reciprocal_target_ranking_retains_runner_up_and_tie_breaking() -> None:
    source_xy = np.array([[0.0, 0.0], [1.0, 0.0]])
    target_xy = np.array([[0.0, 0.0], [1.0, 0.0]])
    source_descriptor = np.array([[1.0, 0.0], [0.8, 0.6]], dtype=np.float32)
    target_descriptor = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    config = CorrespondenceConfig(
        patch_width_um=10.0,
        geometry_score_weight=0.0,
    )

    source, target, similarity, margin = _reciprocal_matches(
        source_xy,
        target_xy,
        source_descriptor,
        target_descriptor,
        np.zeros_like(source_xy),
        radius_um=10.0,
        config=config,
    )

    np.testing.assert_array_equal(source, [0])
    np.testing.assert_array_equal(target, [0])
    np.testing.assert_array_equal(similarity, [1.0])
    np.testing.assert_allclose(margin, [0.1], rtol=0, atol=1e-7)

    tied_source = np.repeat(source_descriptor[:1], 2, axis=0)
    source, target, _, margin = _reciprocal_matches(
        source_xy,
        target_xy,
        tied_source,
        target_descriptor,
        np.zeros_like(source_xy),
        radius_um=10.0,
        config=config,
    )

    np.testing.assert_array_equal(source, [0])
    np.testing.assert_array_equal(target, [0])
    np.testing.assert_array_equal(margin, [0.0])


def test_batched_candidate_scores_are_bit_exact_with_scalar_ranking() -> None:
    rng = np.random.default_rng(401)
    source_count = 19
    target_count = 47
    dimensions = 64
    source_xy = rng.normal(size=(source_count, 2)) * 100
    target_xy = rng.normal(size=(target_count, 2)) * 100
    field = rng.normal(size=(source_count, 2)) * 5
    source_descriptor = rng.normal(size=(source_count, dimensions)).astype(np.float32)
    target_descriptor = rng.normal(size=(target_count, dimensions)).astype(np.float32)
    source_descriptor /= np.linalg.norm(
        source_descriptor,
        axis=1,
        keepdims=True,
    )
    target_descriptor /= np.linalg.norm(
        target_descriptor,
        axis=1,
        keepdims=True,
    )
    candidates = [
        sorted(
            rng.choice(
                target_count,
                size=index % 7,
                replace=False,
            ).tolist()
        )
        for index in range(source_count)
    ]
    config = CorrespondenceConfig(patch_width_um=112.0)

    observed = _score_candidate_window(
        candidates,
        source_start=0,
        source_xy=source_xy,
        target_xy=target_xy,
        source_descriptor=source_descriptor,
        target_descriptor=target_descriptor,
        field=field,
        config=config,
    )

    for source_index, candidate_indices in enumerate(candidates):
        if not candidate_indices:
            assert observed[source_index] is None
            continue
        indices = np.asarray(candidate_indices, dtype=np.int64)
        similarities = target_descriptor[indices] @ source_descriptor[source_index]
        distances = np.linalg.norm(
            target_xy[indices] - (source_xy[source_index] + field[source_index]),
            axis=1,
        )
        geometry = np.exp(-0.5 * (distances / config.patch_width_um) ** 2)
        feature_rank = np.clip((similarities + 1.0) / 2.0, 0.0, 1.0)
        geometry_weight = (
            min(0.20, config.geometry_score_weight)
            if float(np.max(similarities)) >= config.min_feature_similarity
            else config.geometry_score_weight
        )
        scores = (1.0 - geometry_weight) * feature_rank + geometry_weight * geometry
        ranked = observed[source_index]
        assert ranked is not None
        (
            observed_indices,
            observed_similarities,
            observed_scores,
            observed_best,
            observed_margin,
        ) = ranked
        np.testing.assert_array_equal(observed_indices, indices)
        np.testing.assert_array_equal(observed_similarities, similarities)
        np.testing.assert_array_equal(observed_scores, scores)
        order = np.lexsort((indices, -scores))
        assert observed_best == int(order[0])
        expected_margin = (
            float(scores[order[0]] - scores[order[1]]) if len(order) > 1 else 0.0
        )
        assert observed_margin == expected_margin


def test_batched_candidate_ranking_breaks_score_ties_by_target_index() -> None:
    candidates = [[2, 4, 7], [1]]
    source_xy = np.zeros((2, 2))
    target_xy = np.zeros((8, 2))
    source_descriptor = np.ones((2, 1), dtype=np.float32)
    target_descriptor = np.ones((8, 1), dtype=np.float32)

    observed = _score_candidate_window(
        candidates,
        source_start=0,
        source_xy=source_xy,
        target_xy=target_xy,
        source_descriptor=source_descriptor,
        target_descriptor=target_descriptor,
        field=np.zeros_like(source_xy),
        config=CorrespondenceConfig(
            patch_width_um=112.0,
            geometry_score_weight=0.0,
        ),
    )

    assert observed[0] is not None
    assert observed[0][3] == 0
    assert observed[0][4] == 0.0
    assert observed[1] is not None
    assert observed[1][3] == 0
    assert observed[1][4] == 0.0


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
