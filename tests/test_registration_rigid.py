import numpy as np

from histopia.registration import (
    RigidTransformResult,
    estimate_rigid_transform,
    refine_rigid_transform,
)
from histopia.registration._rigid import _feature_or_mask_fallback


def test_phase_correlation_recovers_integer_translation() -> None:
    fixed = np.zeros((64, 64), dtype=np.float32)
    fixed[20:34, 25:40] = 1.0
    moving = np.roll(fixed, shift=(5, -7), axis=(0, 1))

    result = estimate_rigid_transform(
        fixed,
        moving,
        method="phase_correlation",
    )

    assert result.warnings == []
    assert result.matrix[0, 2] == 7
    assert result.matrix[1, 2] == -5


def test_mask_moments_recovers_rectangle_translation() -> None:
    fixed_mask = np.zeros((80, 90), dtype=bool)
    moving_mask = np.zeros((80, 90), dtype=bool)
    fixed_mask[20:36, 25:55] = True
    moving_mask[25:41, 18:48] = True

    result = estimate_rigid_transform(
        fixed_mask.astype(np.float32),
        moving_mask.astype(np.float32),
        fixed_mask=fixed_mask,
        moving_mask=moving_mask,
        method="mask_moments",
    )

    assert result.warnings == []
    assert result.method == "mask_moments"
    assert np.isclose(result.matrix[0, 2], 7, atol=1)
    assert np.isclose(result.matrix[1, 2], -5, atol=1)


def test_mask_refinement_improves_conservative_affine_initializer() -> None:
    fixed_mask = np.zeros((160, 180), dtype=bool)
    fixed_mask[30:120, 35:140] = True
    fixed_mask[45:75, 120:160] = True

    import cv2

    true_matrix = cv2.getRotationMatrix2D((90, 80), 6, 0.94)
    true_matrix[:, 2] += np.array([5, -4])
    moving_mask = cv2.warpAffine(
        fixed_mask.astype(np.uint8),
        cv2.invertAffineTransform(true_matrix),
        (180, 160),
        flags=cv2.INTER_NEAREST,
    ).astype(bool)
    initial_matrix = np.eye(3, dtype=float)
    initial_matrix[:2] = true_matrix
    initial_matrix[0, 2] += 8
    initial_matrix[1, 2] -= 7
    initial = RigidTransformResult(initial_matrix, "initial", 0, 0, [])

    result = refine_rigid_transform(fixed_mask, moving_mask, initial)

    assert result.method == "initial+mask_ecc_affine"
    assert result.inlier_count > 900


def test_feature_fallback_keeps_valid_low_overlap_mask_transform() -> None:
    fixed_mask = np.zeros((100, 100), dtype=bool)
    moving_mask = np.zeros((100, 100), dtype=bool)
    fixed_mask[10:90, 10:90] = True
    moving_mask[45:55, 45:55] = True

    result = _feature_or_mask_fallback(
        "not enough feature matches",
        fixed_mask,
        moving_mask,
        match_count=2,
    )

    assert result.method == "mask_moments"
    assert result.inlier_count > 0
    assert result.match_count == 2
    assert result.warnings[-1].endswith("used mask moments")
