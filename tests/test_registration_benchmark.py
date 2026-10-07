import numpy as np
import pytest

from histopia.registration.benchmark import (
    acrobat_2022_score,
    affine_shape_metrics,
    block_bootstrap,
    change_frame,
    displacement_jacobian_metrics,
    landmark_metrics,
    mask_reference_metrics,
    match_landmarks,
    order_metrics,
    pixel_geometry,
    transform_points,
)


def test_forward_crop_anisotropic_units_and_inverse():
    src = pixel_geometry((2, 4), (10, 20), (0.5, 0.25))
    tgt = pixel_geometry((3, 2), (30, 40), (0.25, 0.5))
    m = np.array([[0, -1, 100], [1, 0, 50], [0, 0, 1.0]])
    p = np.array([[2.0, 8.0], [10.0, 20.0]])
    physical = change_frame(m, src, tgt)
    np.testing.assert_allclose(
        transform_points(transform_points(p, src), physical),
        transform_points(transform_points(p, m), tgt),
    )
    np.testing.assert_allclose(
        transform_points(transform_points(p, m), np.linalg.inv(m)), p
    )
    np.testing.assert_allclose(transform_points([[0, 0]], src), [[5.25, 5.375]])


def test_orientation_and_reflection():
    orient = np.array([[-1.0, 0, 99], [0, 1, 0], [0, 0, 1]])
    np.testing.assert_allclose(
        transform_points([[0, 0]], pixel_geometry(orientation=orient)), [[99, 0]]
    )
    assert affine_shape_metrics(orient)["reflection"]


def test_matching_is_by_id_and_missing_is_explicit():
    ids, a, b, absent = match_landmarks(
        {"b": [1, 2], "a": [3, 4], "c": [0, 0]}, {"a": [5, 6], "b": [7, 8], "d": [1, 1]}
    )
    assert ids == ["a", "b"]
    assert a.tolist() == [[3, 4], [1, 2]]
    assert absent == {"missing_source": ["d"], "missing_target": ["c"], "invalid": []}


def test_metrics_zero_error_and_failed_point_denominator():
    a, b = np.array([[3.0, 4.0], [6, 8]]), np.zeros((2, 2))
    result = landmark_metrics(a, b, [[0, 0], [np.nan, np.nan]], (60, 80), mpp_xy=(2, 1))
    assert result["n_missing_predictions"] == 1
    assert result["median_rtre"] == 0.05
    assert result["robustness"] == 0.5
    assert result["mean_distance_reduction_px"] == 2.5
    assert result["tre_um"][1] == pytest.approx(np.sqrt(208))


def test_challenge2022_does_not_pool_points_or_mean_pair_quantiles():
    assert acrobat_2022_score({"a": [0, 10], "b": [20, 20], "c": [100, 100]}) == 20


def test_challenge2022_averages_reviewer_distances_not_reviewer_positions():
    from histopia.registration.benchmark import mean_annotator_tre

    error = mean_annotator_tre([[0, 0]], [[[-1, 0], [1, 0]]])
    np.testing.assert_allclose(error, [1.0])
    with pytest.raises(ValueError):
        mean_annotator_tre([[0, 0]], [[[np.nan, 0], [1, 0]]])


def test_block_bootstrap_not_landmark_weighted():
    result = block_bootstrap([0, 0, 0, 10], ["a", "a", "a", "b"])
    assert result["mean"] == 5
    assert result["n_blocks"] == 2
    assert result == block_bootstrap([0, 0, 0, 10], ["a", "a", "a", "b"])


def test_order_reversal_and_incomplete_inputs():
    assert order_metrics(["a", "b", "c"], ["c", "b", "a"])["complete_recovery"]
    assert not order_metrics(["a", "b", "c"], ["c", "b", "a"], reversal_allowed=False)[
        "complete_recovery"
    ]
    with pytest.raises(ValueError):
        order_metrics(["a", "b", "c"], ["a", "b"])


def test_physical_affine_and_invalid_calibration():
    result = affine_shape_metrics(np.diag([2.0, 0.5, 1]))
    assert result["area_ratio"] == 1
    assert result["min_directional_scale"] == 0.5
    with pytest.raises(ValueError):
        pixel_geometry(mpp_xy=(0, 1))


def test_independent_mask_overlap_empty_and_anisotropic_boundary():
    ref = np.zeros((7, 7), bool)
    ref[2:5, 2:5] = True
    assert mask_reference_metrics(ref, ref)["dice"] == 1
    shifted = np.zeros_like(ref)
    shifted[2:5, 3:6] = True
    result = mask_reference_metrics(ref, shifted, spacing_xy=(2, 1))
    assert result["dice"] == pytest.approx(2 / 3)
    assert result["p95_boundary_distance"] == 2
    assert (
        mask_reference_metrics(ref, np.zeros_like(ref))["mean_boundary_distance"]
        is None
    )


def test_displacement_translation_and_folding():
    d = np.full((8, 9, 2), 3.0)
    assert displacement_jacobian_metrics(d)["jacobian_median"] == 1
    d[..., 0] = -2 * np.arange(9)[None, :]
    assert displacement_jacobian_metrics(d)["fold_fraction"] == 1
    with pytest.raises(ValueError):
        displacement_jacobian_metrics(d, np.zeros(d.shape[:2], bool))


def test_production_crop_composition_direction():
    from histopia.registration._pipeline import _compose_crop_transform, _Crop

    fixed = _Crop(np.zeros((1, 1, 3)), np.ones((1, 1)), np.array([17.0, 19.0]), 2.0)
    moving = _Crop(np.zeros((1, 1, 3)), np.ones((1, 1)), np.array([11.0, 13.0]), 4.0)
    m = np.array([[0.0, -1, 9], [1, 0, 8], [0, 0, 1]])
    points = np.array([[20.0, 25.0], [40.0, 50.0]])
    expected = (
        transform_points((points - moving.offset_xy) * moving.scale, m) / fixed.scale
        + fixed.offset_xy
    )
    np.testing.assert_allclose(
        transform_points(points, _compose_crop_transform(m, fixed, moving)), expected
    )


def test_invert_pull_points_handles_rotation_and_missing_support():
    from histopia.registration.benchmark import invert_pull_points

    yy, xx = np.mgrid[:50, :60]
    # A 180-degree pull rotation and translation; fixed-point iteration diverges.
    field = np.stack([59 - 2 * xx, 49 - 2 * yy], axis=-1).astype(float)
    source = np.array([[12.0, 17.0], [43.0, 34.0], [-100.0, 20.0]])
    mapped, valid, residual = invert_pull_points(
        source, field, np.array([[30.0, 20.0], [30.0, 20.0], [30.0, 20.0]])
    )
    np.testing.assert_allclose(mapped[:2], [[47.0, 32.0], [16.0, 15.0]], atol=1e-6)
    assert valid.tolist() == [True, True, False]
    assert np.isnan(mapped[2]).all()
    assert np.max(residual[:2]) < 1e-6


def test_invert_pull_points_rejects_singular_mapping():
    from histopia.registration.benchmark import invert_pull_points

    yy, xx = np.mgrid[:10, :10]
    field = np.stack([-xx, np.zeros_like(yy)], axis=-1).astype(float)
    mapped, valid, _ = invert_pull_points(
        np.array([[0.0, 3.0]]), field, np.array([[4.0, 3.0]])
    )
    assert not valid[0]
    assert np.isnan(mapped).all()
