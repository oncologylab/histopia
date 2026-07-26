"""Conservative tissue-supported non-rigid thumbnail refinement."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import ndimage as ndi

from histopia.registration._config import NonRigidRefinementConfig
from histopia.registration._errors import OptionalDependencyError


@dataclass(slots=True)
class SparseFeatureValidation:
    """Held-out sparse correspondence evidence for a dense candidate field.

    The sparse features are not used by the DIS estimator. They provide an
    algorithmically separate diagnostic, but are not anatomical ground truth.
    """

    status: str
    detector: str
    fixed_keypoints: int
    moving_keypoints: int
    mutual_matches: int
    coherent_matches: int
    initial_median_residual_px: float | None
    final_median_residual_px: float | None
    initial_p95_residual_px: float | None
    final_p95_residual_px: float | None
    improved_fraction: float | None
    warnings: list[str]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "detector": self.detector,
            "fixed_keypoints": self.fixed_keypoints,
            "moving_keypoints": self.moving_keypoints,
            "mutual_matches": self.mutual_matches,
            "coherent_matches": self.coherent_matches,
            "initial_median_residual_px": self.initial_median_residual_px,
            "final_median_residual_px": self.final_median_residual_px,
            "initial_p95_residual_px": self.initial_p95_residual_px,
            "final_p95_residual_px": self.final_p95_residual_px,
            "improved_fraction": self.improved_fraction,
            "warnings": self.warnings,
        }


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
    sparse_feature_validation: SparseFeatureValidation | None = None
    candidate_displacement: np.ndarray | None = field(default=None, repr=False)

    @property
    def diagnostic_displacement(self) -> np.ndarray:
        """Return the estimated candidate, including when application was rejected."""

        if self.candidate_displacement is not None:
            return self.candidate_displacement
        return self.displacement

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
            "sparse_feature_validation": (
                self.sparse_feature_validation.to_json_dict()
                if self.sparse_feature_validation is not None
                else None
            ),
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

    settings = NonRigidRefinementConfig(
        enabled=True,
        max_displacement_fraction=max_displacement_fraction,
        smoothing_sigma_px=smoothing_sigma_px,
        support_dilation_fraction=support_dilation_fraction,
        min_similarity_improvement=min_similarity_improvement,
        max_mask_dice_loss=max_mask_dice_loss,
        min_jacobian_p01=min_jacobian_p01,
        max_jacobian_p99=max_jacobian_p99,
        max_inverse_consistency_fraction=max_inverse_consistency_fraction,
    )
    max_displacement_fraction = settings.max_displacement_fraction
    smoothing_sigma_px = settings.smoothing_sigma_px
    support_dilation_fraction = settings.support_dilation_fraction
    min_similarity_improvement = settings.min_similarity_improvement
    max_mask_dice_loss = settings.max_mask_dice_loss
    min_jacobian_p01 = settings.min_jacobian_p01
    max_jacobian_p99 = settings.max_jacobian_p99
    max_inverse_consistency_fraction = settings.max_inverse_consistency_fraction
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

    sparse_validation = evaluate_non_rigid_feature_holdout(
        fixed_rgb,
        moving_rgb,
        displacement,
        fixed_mask=fixed_mask_bool,
        rigid_moving_mask=moving_mask_bool,
    )

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
    candidate_displacement = None
    if not accepted:
        candidate_displacement = displacement.astype(np.float32)
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
        sparse_feature_validation=sparse_validation,
        candidate_displacement=candidate_displacement,
    )


def evaluate_non_rigid_feature_holdout(
    fixed: np.ndarray,
    rigid_moving: np.ndarray,
    displacement: np.ndarray,
    *,
    fixed_mask: np.ndarray,
    rigid_moving_mask: np.ndarray,
    ratio_threshold: float = 0.8,
    maximum_residual_fraction: float = 0.08,
    ransac_threshold_fraction: float = 0.012,
    minimum_coherent_matches: int = 8,
) -> SparseFeatureValidation:
    """Evaluate a dense field against mutual sparse features unused by DIS.

    Keypoints are detected after affine registration, matched in both
    directions, and filtered by a near-identity robust affine model. Residuals
    compare the matched moving point with the fixed point before and after
    sampling the candidate displacement at that point.
    """

    if not 0 < ratio_threshold < 1:
        raise ValueError("ratio_threshold must be between 0 and 1")
    if not 0 < maximum_residual_fraction < 0.5:
        raise ValueError("maximum_residual_fraction must be between 0 and 0.5")
    if not 0 < ransac_threshold_fraction < 0.5:
        raise ValueError("ransac_threshold_fraction must be between 0 and 0.5")
    if (
        isinstance(minimum_coherent_matches, bool)
        or not isinstance(minimum_coherent_matches, int)
        or minimum_coherent_matches < 4
    ):
        raise ValueError("minimum_coherent_matches must be an integer of at least 4")

    cv2 = _import_cv2()
    fixed_rgb = _as_rgb_u8(fixed)
    moving_rgb = _as_rgb_u8(rigid_moving)
    if fixed_rgb.shape != moving_rgb.shape:
        raise ValueError("fixed and rigid_moving images must have matching shapes")
    fixed_mask_bool = np.asarray(fixed_mask, dtype=bool)
    moving_mask_bool = np.asarray(rigid_moving_mask, dtype=bool)
    if fixed_mask_bool.shape != fixed_rgb.shape[:2]:
        raise ValueError("fixed_mask shape must match fixed image")
    if moving_mask_bool.shape != moving_rgb.shape[:2]:
        raise ValueError("rigid_moving_mask shape must match rigid_moving image")
    flow = np.asarray(displacement, dtype=np.float32)
    if flow.shape != (*fixed_rgb.shape[:2], 2):
        raise ValueError("displacement shape must be (height, width, 2)")

    overlap = fixed_mask_bool & moving_mask_bool
    erosion_px = max(1, int(round(max(overlap.shape) * 0.005)))
    overlap = ndi.binary_erosion(overlap, iterations=erosion_px)
    if not overlap.any():
        return _unavailable_sparse_validation(
            detector="orb",
            warning="fixed and moving tissue masks have no interior overlap",
        )

    detector, matcher_norm, detector_name = _create_holdout_detector(cv2)
    fixed_gray = _holdout_gray(fixed_rgb, cv2)
    moving_gray = _holdout_gray(moving_rgb, cv2)
    mask_u8 = (overlap * 255).astype(np.uint8)
    fixed_keypoints, fixed_descriptors = detector.detectAndCompute(
        fixed_gray,
        mask_u8,
    )
    moving_keypoints, moving_descriptors = detector.detectAndCompute(
        moving_gray,
        mask_u8,
    )
    fixed_count = len(fixed_keypoints)
    moving_count = len(moving_keypoints)
    if fixed_descriptors is None or moving_descriptors is None:
        return _unavailable_sparse_validation(
            detector=detector_name,
            warning="sparse holdout descriptors are unavailable",
            fixed_keypoints=fixed_count,
            moving_keypoints=moving_count,
        )

    matcher = cv2.BFMatcher(matcher_norm)
    fixed_to_moving = _ratio_matches(
        matcher.knnMatch(fixed_descriptors, moving_descriptors, k=2),
        ratio_threshold,
    )
    moving_to_fixed = _ratio_matches(
        matcher.knnMatch(moving_descriptors, fixed_descriptors, k=2),
        ratio_threshold,
    )
    reverse = {match.queryIdx: match for match in moving_to_fixed}
    mutual = [
        match
        for match in fixed_to_moving
        if (
            (reverse_match := reverse.get(match.trainIdx)) is not None
            and reverse_match.trainIdx == match.queryIdx
        )
    ]
    fixed_points = np.float32([fixed_keypoints[match.queryIdx].pt for match in mutual])
    moving_points = np.float32(
        [moving_keypoints[match.trainIdx].pt for match in mutual]
    )
    maximum_dimension = max(fixed_rgb.shape[:2])
    if len(mutual):
        initial_residual = np.linalg.norm(fixed_points - moving_points, axis=1)
        plausible = initial_residual <= maximum_dimension * maximum_residual_fraction
        fixed_points = fixed_points[plausible]
        moving_points = moving_points[plausible]

    coherent_count = 0
    if len(fixed_points) >= 4:
        if hasattr(cv2, "setRNGSeed"):
            cv2.setRNGSeed(0)
        _, inliers = cv2.estimateAffinePartial2D(
            fixed_points,
            moving_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=max(
                4.0,
                maximum_dimension * ransac_threshold_fraction,
            ),
            maxIters=5000,
            confidence=0.995,
        )
        if inliers is not None:
            coherent = inliers.ravel().astype(bool)
            fixed_points = fixed_points[coherent]
            moving_points = moving_points[coherent]
            coherent_count = len(fixed_points)

    if coherent_count < minimum_coherent_matches:
        return _unavailable_sparse_validation(
            detector=detector_name,
            warning=(
                "fewer than "
                f"{minimum_coherent_matches} geometrically coherent sparse matches"
            ),
            fixed_keypoints=fixed_count,
            moving_keypoints=moving_count,
            mutual_matches=len(mutual),
            coherent_matches=coherent_count,
        )

    initial_residual = np.linalg.norm(fixed_points - moving_points, axis=1)
    sampled_x = cv2.remap(
        flow[:, :, 0],
        fixed_points[:, 0].reshape(-1, 1),
        fixed_points[:, 1].reshape(-1, 1),
        cv2.INTER_LINEAR,
    ).ravel()
    sampled_y = cv2.remap(
        flow[:, :, 1],
        fixed_points[:, 0].reshape(-1, 1),
        fixed_points[:, 1].reshape(-1, 1),
        cv2.INTER_LINEAR,
    ).ravel()
    refined_points = fixed_points + np.column_stack((sampled_x, sampled_y))
    final_residual = np.linalg.norm(refined_points - moving_points, axis=1)
    return SparseFeatureValidation(
        status="available",
        detector=detector_name,
        fixed_keypoints=fixed_count,
        moving_keypoints=moving_count,
        mutual_matches=len(mutual),
        coherent_matches=coherent_count,
        initial_median_residual_px=float(np.median(initial_residual)),
        final_median_residual_px=float(np.median(final_residual)),
        initial_p95_residual_px=float(np.percentile(initial_residual, 95)),
        final_p95_residual_px=float(np.percentile(final_residual, 95)),
        improved_fraction=float(np.mean(final_residual < initial_residual)),
        warnings=[],
    )


def _create_holdout_detector(cv2: Any) -> tuple[Any, int, str]:
    if hasattr(cv2, "ORB_create"):
        return (
            cv2.ORB_create(nfeatures=8000, fastThreshold=5),
            cv2.NORM_HAMMING,
            "orb",
        )
    if hasattr(cv2, "SIFT_create"):
        return (
            cv2.SIFT_create(nfeatures=8000, contrastThreshold=0.02),
            cv2.NORM_L2,
            "sift",
        )
    raise RuntimeError("OpenCV has no supported sparse holdout feature detector")


def _holdout_gray(image: np.ndarray, cv2: Any) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def _ratio_matches(
    match_pairs: list[tuple[Any, ...]],
    ratio_threshold: float,
) -> list[Any]:
    return [
        pair[0]
        for pair in match_pairs
        if len(pair) == 2 and pair[0].distance < ratio_threshold * pair[1].distance
    ]


def _unavailable_sparse_validation(
    *,
    detector: str,
    warning: str,
    fixed_keypoints: int = 0,
    moving_keypoints: int = 0,
    mutual_matches: int = 0,
    coherent_matches: int = 0,
) -> SparseFeatureValidation:
    return SparseFeatureValidation(
        status="insufficient_matches",
        detector=detector,
        fixed_keypoints=fixed_keypoints,
        moving_keypoints=moving_keypoints,
        mutual_matches=mutual_matches,
        coherent_matches=coherent_matches,
        initial_median_residual_px=None,
        final_median_residual_px=None,
        initial_p95_residual_px=None,
        final_p95_residual_px=None,
        improved_fraction=None,
        warnings=[warning],
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
