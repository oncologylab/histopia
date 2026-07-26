import cv2
import numpy as np
import pytest

from histopia.registration import _pipeline
from histopia.registration._nonrigid import (
    NonRigidTransformResult,
    displacement_jacobian,
    estimate_non_rigid_transform,
    evaluate_non_rigid_feature_holdout,
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


def test_non_rigid_primitive_rejects_invalid_acceptance_gate() -> None:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    mask = np.ones((8, 8), dtype=bool)

    with pytest.raises(ValueError, match="min_jacobian_p01"):
        estimate_non_rigid_transform(
            image,
            image,
            fixed_mask=mask,
            rigid_moving_mask=mask,
            min_jacobian_p01=1.1,
        )


def test_sparse_feature_holdout_validates_known_translation() -> None:
    rng = np.random.default_rng(14)
    fixed = np.full((240, 240, 3), 245, dtype=np.uint8)
    mask = np.zeros(fixed.shape[:2], dtype=np.uint8)
    cv2.ellipse(mask, (120, 120), (92, 78), 0, 0, 360, 255, -1)
    for _ in range(160):
        x = int(rng.integers(35, 205))
        y = int(rng.integers(45, 195))
        if mask[y, x]:
            color = int(rng.integers(30, 210))
            cv2.circle(fixed, (x, y), int(rng.integers(1, 4)), (color,) * 3, -1)
    moving = cv2.warpAffine(
        fixed,
        np.float32([[1, 0, 6], [0, 1, -4]]),
        (fixed.shape[1], fixed.shape[0]),
        borderValue=(245, 245, 245),
    )
    moving_mask = cv2.warpAffine(
        mask,
        np.float32([[1, 0, 6], [0, 1, -4]]),
        (fixed.shape[1], fixed.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderValue=0,
    )
    displacement = np.empty((*fixed.shape[:2], 2), dtype=np.float32)
    displacement[:, :, 0] = 6
    displacement[:, :, 1] = -4

    validation = evaluate_non_rigid_feature_holdout(
        fixed,
        moving,
        displacement,
        fixed_mask=mask > 0,
        rigid_moving_mask=moving_mask > 0,
    )

    assert validation.status == "available"
    assert validation.coherent_matches >= 8
    assert validation.initial_median_residual_px is not None
    assert validation.final_median_residual_px is not None
    assert validation.final_median_residual_px < 0.75
    assert validation.initial_median_residual_px > 6
    assert validation.improved_fraction is not None
    assert validation.improved_fraction > 0.95


def test_rejected_non_rigid_candidate_remains_diagnostic_but_is_not_applied() -> None:
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
        min_similarity_improvement=2,
    )

    assert not result.accepted
    assert np.count_nonzero(result.displacement) == 0
    assert np.count_nonzero(result.diagnostic_displacement) > 0
    assert result.candidate_displacement is not None
    assert "candidate_displacement" not in result.to_json_dict()


def test_rejected_candidate_qc_renders_diagnostic_field(
    tmp_path,
    monkeypatch,
) -> None:
    image = np.full((32, 32, 3), 180, dtype=np.uint8)
    image[:, 8:12] = 20
    candidate = np.zeros((32, 32, 2), dtype=np.float32)
    candidate[:, :, 0] = 3
    result = NonRigidTransformResult(
        displacement=np.zeros_like(candidate),
        accepted=False,
        method="test",
        initial_similarity=0.1,
        final_similarity=0.2,
        initial_mask_dice=0.9,
        final_mask_dice=0.8,
        jacobian_p01=1.0,
        jacobian_p99=1.0,
        displacement_p95=3.0,
        inverse_consistency_p95=0.0,
        warnings=["test rejection"],
        candidate_displacement=candidate,
    )
    captured = {}

    def capture(path, *, panes, title, metadata):
        captured["panes"] = panes
        captured["title"] = title
        captured["metadata"] = metadata
        return path

    monkeypatch.setattr(_pipeline, "write_labeled_review_panel", capture)

    _pipeline._write_non_rigid_qc(
        tmp_path,
        tmp_path / "moving.ndpi",
        image,
        image,
        result,
    )

    assert "rejected; affine retained" in captured["title"]
    assert not np.array_equal(captured["panes"][1][1], captured["panes"][2][1])
    assert any("test rejection" in line for line in captured["metadata"])
