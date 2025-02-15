"""Conservative tissue-supported non-rigid thumbnail refinement."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import ndimage as ndi

from histopia.registration._errors import OptionalDependencyError


@dataclass(slots=True)
class NonRigidTransformResult:
    """Dense reference-to-rigid-moving displacement and acceptance metrics."""

    displacement: np.ndarray = field(repr=False)
    accepted: bool
    method: str
    initial_similarity: float
    final_similarity: float
    initial_mask_dice: float
    final_mask_dice: float
    jacobian_p01: float
    jacobian_p99: float
    displacement_p95: float
    inverse_consistency_p95: float
    warnings: list[str]
    displacement_path: str | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "method": self.method,
            "displacement_shape": list(self.displacement.shape),
            "displacement_path": self.displacement_path,
            "initial_similarity": self.initial_similarity,
            "final_similarity": self.final_similarity,
            "initial_mask_dice": self.initial_mask_dice,
            "final_mask_dice": self.final_mask_dice,
            "jacobian_p01": self.jacobian_p01,
            "jacobian_p99": self.jacobian_p99,
            "displacement_p95": self.displacement_p95,
            "inverse_consistency_p95": self.inverse_consistency_p95,
            "warnings": self.warnings,
        }


def estimate_non_rigid_transform(
    fixed: np.ndarray,
    rigid_moving: np.ndarray,
    *,
    fixed_mask: np.ndarray,
    rigid_moving_mask: np.ndarray,
    max_displacement_fraction: float = 0.03,
    smoothing_sigma_px: float = 12.0,
    support_dilation_fraction: float = 0.03,
    min_similarity_improvement: float = 0.01,
    max_mask_dice_loss: float = 0.01,
    min_jacobian_p01: float = 0.25,
    max_jacobian_p99: float = 4.0,
    max_inverse_consistency_fraction: float = 0.02,
) -> NonRigidTransformResult:
    """Estimate and acceptance-gate a dense flow after affine registration."""

    cv2 = _import_cv2()
    fixed_rgb = _as_rgb_u8(fixed)
    moving_rgb = _as_rgb_u8(rigid_moving)
    if fixed_rgb.shape != moving_rgb.shape:
        msg = "fixed and rigid_moving images must have matching shapes"
        raise ValueError(msg)
    fixed_mask_bool = np.asarray(fixed_mask, dtype=bool)
    moving_mask_bool = np.asarray(rigid_moving_mask, dtype=bool)
    if fixed_mask_bool.shape != fixed_rgb.shape[:2]:
        msg = "fixed_mask shape must match fixed image"
        raise ValueError(msg)
    if moving_mask_bool.shape != moving_rgb.shape[:2]:
        msg = "rigid_moving_mask shape must match rigid_moving image"
        raise ValueError(msg)

    support = fixed_mask_bool | moving_mask_bool
    dilation_px = max(
        1,
        int(round(max(support.shape) * support_dilation_fraction)),
    )
    support = ndi.binary_dilation(support, iterations=dilation_px)
    support_weight = cv2.GaussianBlur(
        support.astype(np.float32),
        (0, 0),
        max(1.0, smoothing_sigma_px),
    )

    fixed_structure = _structural_image(fixed_rgb, support, cv2)
    moving_structure = _structural_image(moving_rgb, support, cv2)
    maximum_displacement = max(support.shape) * max_displacement_fraction
    displacement = _estimate_displacement(
        fixed_structure,
        moving_structure,
        support_weight,
        maximum_displacement,
        smoothing_sigma_px,
        cv2,
    )
    reverse_displacement = _estimate_displacement(
        moving_structure,
        fixed_structure,
        support_weight,
        maximum_displacement,
        smoothing_sigma_px,
        cv2,
    )

    initial_similarity = _normalized_cross_correlation(
        fixed_structure,
        moving_structure,
        support,
    )
    warped_structure = warp_with_displacement(
        moving_structure,
        displacement,
        interpolation="linear",
        border_value=0,
    )
    final_similarity = _normalized_cross_correlation(
        fixed_structure,
        warped_structure,
        support,
    )
    warped_mask = warp_with_displacement(
        moving_mask_bool.astype(np.uint8),
        displacement,
        interpolation="nearest",
        border_value=0,
    ).astype(bool)
    initial_dice = _mask_dice(fixed_mask_bool, moving_mask_bool)
    final_dice = _mask_dice(fixed_mask_bool, warped_mask)
    jacobian = displacement_jacobian(displacement)
    support_values = jacobian[support]
    jacobian_p01 = float(np.percentile(support_values, 1))
    jacobian_p99 = float(np.percentile(support_values, 99))
    magnitudes = np.linalg.norm(displacement, axis=2)
    displacement_p95 = float(np.percentile(magnitudes[support], 95))
    reverse_at_forward_x = warp_with_displacement(
        reverse_displacement[:, :, 0],
        displacement,
        interpolation="linear",
        border_value=0,
    )
    reverse_at_forward_y = warp_with_displacement(
        reverse_displacement[:, :, 1],
        displacement,
        interpolation="linear",
        border_value=0,
    )
    inverse_residual = displacement + np.stack(
        [reverse_at_forward_x, reverse_at_forward_y],
        axis=2,
    )
    inverse_consistency = np.linalg.norm(inverse_residual, axis=2)
    inverse_consistency_p95 = float(np.percentile(inverse_consistency[support], 95))

    warnings: list[str] = []
    if final_similarity < initial_similarity + min_similarity_improvement:
        warnings.append("structural similarity did not improve enough")
    if final_dice < initial_dice - max_mask_dice_loss:
        warnings.append("tissue-mask Dice regressed")
    if jacobian_p01 < min_jacobian_p01:
        warnings.append("deformation compression exceeded Jacobian limit")
    if jacobian_p99 > max_jacobian_p99:
        warnings.append("deformation expansion exceeded Jacobian limit")
    if inverse_consistency_p95 > max(support.shape) * max_inverse_consistency_fraction:
        warnings.append("forward/backward flow consistency exceeded limit")
    accepted = not warnings
    if not accepted:
        displacement = np.zeros_like(displacement)
    return NonRigidTransformResult(
        displacement=displacement.astype(np.float32),
        accepted=accepted,
        method="dis_tissue_supported",
        initial_similarity=initial_similarity,
        final_similarity=final_similarity,
        initial_mask_dice=initial_dice,
        final_mask_dice=final_dice,
        jacobian_p01=jacobian_p01,
        jacobian_p99=jacobian_p99,
        displacement_p95=displacement_p95,
        inverse_consistency_p95=inverse_consistency_p95,
        warnings=warnings,
    )


def _estimate_displacement(
    fixed_structure: np.ndarray,
    moving_structure: np.ndarray,
    support_weight: np.ndarray,
    maximum_displacement: float,
    smoothing_sigma_px: float,
    cv2: Any,
) -> np.ndarray:
    estimator = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    estimator.setFinestScale(2)
    estimator.setGradientDescentIterations(15)
    estimator.setVariationalRefinementIterations(8)
    displacement = estimator.calc(fixed_structure, moving_structure, None)
    for channel in range(2):
        displacement[:, :, channel] = cv2.GaussianBlur(
            displacement[:, :, channel],
            (0, 0),
            smoothing_sigma_px,
        )
    displacement *= support_weight[:, :, np.newaxis]
    return _cap_displacement(displacement, maximum_displacement)


def warp_with_displacement(
    image: np.ndarray,
    displacement: np.ndarray,
    *,
    interpolation: str = "linear",
    border_value: int | tuple[int, int, int] = 255,
) -> np.ndarray:
    """Sample an image using a reference-to-moving displacement field."""

    cv2 = _import_cv2()
    array = np.asarray(image)
    flow = np.asarray(displacement, dtype=np.float32)
    if flow.shape != (*array.shape[:2], 2):
        msg = "displacement shape must be (height, width, 2)"
        raise ValueError(msg)
    rows, cols = np.indices(array.shape[:2], dtype=np.float32)
    interpolation_flag = (
        cv2.INTER_LINEAR if interpolation == "linear" else cv2.INTER_NEAREST
    )
    return cv2.remap(
        array,
        cols + flow[:, :, 0],
        rows + flow[:, :, 1],
        interpolation_flag,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def displacement_jacobian(displacement: np.ndarray) -> np.ndarray:
    """Return determinant of the reference-to-moving coordinate-map Jacobian."""

    flow = np.asarray(displacement, dtype=np.float32)
    du_dx = np.gradient(flow[:, :, 0], axis=1)
    du_dy = np.gradient(flow[:, :, 0], axis=0)
    dv_dx = np.gradient(flow[:, :, 1], axis=1)
    dv_dy = np.gradient(flow[:, :, 1], axis=0)
    return (1 + du_dx) * (1 + dv_dy) - du_dy * dv_dx


def _structural_image(image: np.ndarray, support: np.ndarray, cv2: Any) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    equalized = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    gradient_x = cv2.Scharr(equalized, cv2.CV_32F, 1, 0)
    gradient_y = cv2.Scharr(equalized, cv2.CV_32F, 0, 1)
    magnitude = cv2.magnitude(gradient_x, gradient_y)
    magnitude = cv2.GaussianBlur(magnitude, (0, 0), 1.5)
    normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    return (normalized * support).astype(np.uint8)


def _cap_displacement(displacement: np.ndarray, maximum: float) -> np.ndarray:
    magnitude = np.linalg.norm(displacement, axis=2)
    factor = np.minimum(1.0, maximum / np.maximum(magnitude, 1e-6))
    return displacement * factor[:, :, np.newaxis]


def _normalized_cross_correlation(
    fixed: np.ndarray,
    moving: np.ndarray,
    support: np.ndarray,
) -> float:
    fixed_values = np.asarray(fixed, dtype=float)[support]
    moving_values = np.asarray(moving, dtype=float)[support]
    fixed_values -= fixed_values.mean()
    moving_values -= moving_values.mean()
    denominator = np.linalg.norm(fixed_values) * np.linalg.norm(moving_values)
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(fixed_values, moving_values) / denominator)


def _mask_dice(fixed: np.ndarray, moving: np.ndarray) -> float:
    denominator = fixed.sum() + moving.sum()
    if denominator == 0:
        return 0.0
    return float(2 * np.logical_and(fixed, moving).sum() / denominator)


def _as_rgb_u8(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.repeat(array[:, :, np.newaxis], 3, axis=2)
    array = array[:, :, :3]
    if array.dtype != np.uint8:
        if array.max(initial=0) <= 1.5:
            array = array * 255
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def _import_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise OptionalDependencyError(
            "opencv-contrib-python-headless",
            "registration",
        ) from exc
    return cv2
