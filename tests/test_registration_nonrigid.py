import cv2
import numpy as np

from histopia.registration._nonrigid import (
    displacement_jacobian,
    estimate_non_rigid_transform,
    warp_with_displacement,
)


def test_non_rigid_refinement_improves_smooth_synthetic_deformation() -> None:
    fixed = np.full((192, 192), 255, dtype=np.uint8)
    cv2.ellipse(fixed, (96, 96), (62, 52), 0, 0, 360, 170, -1)
    for x in range(48, 150, 20):
        cv2.circle(fixed, (x, 82 + int(10 * np.sin(x / 18))), 6, 60, -1)
    fixed_rgb = np.repeat(fixed[:, :, np.newaxis], 3, axis=2)
    rows, cols = np.indices(fixed.shape, dtype=np.float32)
    moving = cv2.remap(
        fixed_rgb,
        cols + 6 * np.sin(2 * np.pi * rows / fixed.shape[0]),
        rows + 4 * np.sin(2 * np.pi * cols / fixed.shape[1]),
        cv2.INTER_LINEAR,
        borderValue=(255, 255, 255),
    )
    fixed_mask = fixed < 245
    moving_mask = np.any(moving < 245, axis=2)

    result = estimate_non_rigid_transform(
        fixed_rgb,
        moving,
        fixed_mask=fixed_mask,
        rigid_moving_mask=moving_mask,
        smoothing_sigma_px=4,
        max_displacement_fraction=0.08,
    )

    assert result.accepted
    assert result.final_similarity > result.initial_similarity + 0.05
    assert result.final_mask_dice >= result.initial_mask_dice - 0.01
    assert result.jacobian_p01 > 0.25
    assert result.inverse_consistency_p95 < fixed.shape[0] * 0.02


def test_zero_displacement_is_identity() -> None:
    image = np.arange(64, dtype=np.uint8).reshape(8, 8)
    displacement = np.zeros((8, 8, 2), dtype=np.float32)

    warped = warp_with_displacement(image, displacement)

    assert np.array_equal(warped, image)
    assert np.allclose(displacement_jacobian(displacement), 1.0)
