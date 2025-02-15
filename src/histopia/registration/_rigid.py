"""Rigid registration primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from histopia.registration._errors import OptionalDependencyError


@dataclass(slots=True)
class RigidTransformResult:
    """Rigid transform estimate from moving image into fixed image coordinates."""

    matrix: np.ndarray
    method: str
    match_count: int
    inlier_count: int
    warnings: list[str]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "matrix": self.matrix.tolist(),
            "method": self.method,
            "match_count": self.match_count,
            "inlier_count": self.inlier_count,
            "warnings": self.warnings,
        }


def estimate_rigid_transform(
    fixed: np.ndarray,
    moving: np.ndarray,
    *,
    fixed_mask: np.ndarray | None = None,
    moving_mask: np.ndarray | None = None,
    method: str = "feature",
    refine: bool = False,
    refinement_max_dim_px: int = 500,
    min_dice_improvement: float = 0.01,
    max_relative_scale_change: float = 0.35,
    max_relative_anisotropy: float = 1.30,
) -> RigidTransformResult:
    """Estimate a rigid transform from ``moving`` into ``fixed`` coordinates."""

    if method == "phase_correlation":
        result = _estimate_translation_phase_correlation(fixed, moving)
    elif method == "mask_moments":
        result = _estimate_mask_moments_transform(fixed_mask, moving_mask)
    elif method == "feature":
        result = _estimate_feature_transform(fixed, moving, fixed_mask, moving_mask)
    else:
        msg = f"unsupported rigid registration method: {method!r}"
        raise ValueError(msg)

    if refine and fixed_mask is not None and moving_mask is not None:
        return refine_rigid_transform(
            fixed_mask,
            moving_mask,
            result,
            max_dim_px=refinement_max_dim_px,
            min_dice_improvement=min_dice_improvement,
            max_relative_scale_change=max_relative_scale_change,
            max_relative_anisotropy=max_relative_anisotropy,
        )
    return result


def refine_rigid_transform(
    fixed_mask: np.ndarray,
    moving_mask: np.ndarray,
    initial: RigidTransformResult,
    *,
    max_dim_px: int = 500,
    min_dice_improvement: float = 0.01,
    max_relative_scale_change: float = 0.35,
    max_relative_anisotropy: float = 1.30,
) -> RigidTransformResult:
    """Refine a moving-to-fixed transform using tissue-mask signed distances.

    The candidate is retained only when it improves mask Dice and remains close
    to the initializer in scale and anisotropy. This avoids using stain
    intensity as a correspondence signal across serial IHC sections.
    """

    try:
        import cv2
    except ImportError as exc:
        raise OptionalDependencyError(
            "opencv-contrib-python-headless",
            "registration",
        ) from exc

    fixed = np.asarray(fixed_mask, dtype=bool)
    moving = np.asarray(moving_mask, dtype=bool)
    if not fixed.any() or not moving.any():
        return initial

    initial_dice = _mask_alignment_dice(fixed, moving, initial.matrix)
    fixed_field, moving_field, scale = _mask_distance_fields(
        fixed,
        moving,
        max_dim_px,
        cv2,
    )
    scale_matrix = np.diag([scale, scale, 1.0])
    scaled_initial = scale_matrix @ initial.matrix @ np.linalg.inv(scale_matrix)
    inverse_initial = np.linalg.inv(scaled_initial)[:2, :].astype(np.float32)
    criteria = (
        cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,
        200,
        1e-6,
    )
    try:
        _, inverse_candidate = cv2.findTransformECC(
            fixed_field,
            moving_field,
            inverse_initial,
            cv2.MOTION_AFFINE,
            criteria,
            None,
            3,
        )
    except cv2.error:
        return initial

    inverse_homogeneous = np.vstack(
        [inverse_candidate.astype(float), np.array([0.0, 0.0, 1.0])]
    )
    scaled_candidate = np.linalg.inv(inverse_homogeneous)
    candidate = np.linalg.inv(scale_matrix) @ scaled_candidate @ scale_matrix
    if not _is_plausible_refinement(
        initial.matrix,
        candidate,
        max_relative_scale_change,
        max_relative_anisotropy,
    ):
        return initial

    candidate_dice = _mask_alignment_dice(fixed, moving, candidate)
    if candidate_dice < initial_dice + min_dice_improvement:
        return initial
    return RigidTransformResult(
        matrix=candidate,
        method=f"{initial.method}+mask_ecc_affine",
        match_count=initial.match_count,
        inlier_count=max(initial.inlier_count, int(round(candidate_dice * 1000))),
        warnings=list(initial.warnings),
    )


def _mask_distance_fields(
    fixed: np.ndarray,
    moving: np.ndarray,
    max_dim_px: int,
    cv2: Any,
) -> tuple[np.ndarray, np.ndarray, float]:
    height = max(fixed.shape[0], moving.shape[0])
    width = max(fixed.shape[1], moving.shape[1])
    scale = min(1.0, max_dim_px / max(height, width))
    output_size = (max(1, round(width * scale)), max(1, round(height * scale)))

    fixed_canvas = np.zeros((height, width), dtype=np.uint8)
    moving_canvas = np.zeros((height, width), dtype=np.uint8)
    fixed_canvas[: fixed.shape[0], : fixed.shape[1]] = fixed
    moving_canvas[: moving.shape[0], : moving.shape[1]] = moving
    fixed_resized = cv2.resize(
        fixed_canvas,
        output_size,
        interpolation=cv2.INTER_NEAREST,
    )
    moving_resized = cv2.resize(
        moving_canvas,
        output_size,
        interpolation=cv2.INTER_NEAREST,
    )
    return (
        _signed_distance_field(fixed_resized, cv2),
        _signed_distance_field(moving_resized, cv2),
        scale,
    )


def _signed_distance_field(mask: np.ndarray, cv2: Any) -> np.ndarray:
    foreground = np.asarray(mask, dtype=bool)
    inside = cv2.distanceTransform(
        foreground.astype(np.uint8),
        cv2.DIST_L2,
        5,
    )
    outside = cv2.distanceTransform(
        (~foreground).astype(np.uint8),
        cv2.DIST_L2,
        5,
    )
    field = np.clip(inside - outside, -20.0, 20.0)
    field -= field.min()
    field /= max(float(field.max()), 1e-6)
    return cv2.GaussianBlur(field.astype(np.float32), (0, 0), 2)


def _is_plausible_refinement(
    initial: np.ndarray,
    candidate: np.ndarray,
    max_relative_scale_change: float,
    max_relative_anisotropy: float,
) -> bool:
    if not np.isfinite(candidate).all() or np.linalg.det(candidate[:2, :2]) <= 0:
        return False
    relative = candidate @ np.linalg.inv(initial)
    singular_values = np.linalg.svd(relative[:2, :2], compute_uv=False)
    minimum = float(singular_values.min())
    maximum = float(singular_values.max())
    lower_bound = 1.0 - max_relative_scale_change
    upper_bound = 1.0 + max_relative_scale_change
    return (
        minimum >= lower_bound
        and maximum <= upper_bound
        and maximum / max(minimum, 1e-6) <= max_relative_anisotropy
    )


def _estimate_translation_phase_correlation(
    fixed: np.ndarray,
    moving: np.ndarray,
) -> RigidTransformResult:
    fixed_gray = _to_gray_float(fixed)
    moving_gray = _to_gray_float(moving)
    if fixed_gray.shape != moving_gray.shape:
        msg = "phase_correlation requires fixed and moving images with same shape"
        raise ValueError(msg)

    fixed_centered = fixed_gray - fixed_gray.mean()
    moving_centered = moving_gray - moving_gray.mean()
    cross_power = np.fft.fft2(fixed_centered) * np.fft.fft2(moving_centered).conj()
    cross_power /= np.maximum(np.abs(cross_power), 1e-12)
    correlation = np.fft.ifft2(cross_power)
    peak = np.unravel_index(np.argmax(np.abs(correlation)), correlation.shape)
    shift_rc = np.array(peak, dtype=float)
    shape = np.array(fixed_gray.shape, dtype=float)
    shift_rc[shift_rc > shape / 2] -= shape[shift_rc > shape / 2]

    matrix = np.eye(3, dtype=float)
    matrix[0, 2] = shift_rc[1]
    matrix[1, 2] = shift_rc[0]
    return RigidTransformResult(
        matrix=matrix,
        method="phase_correlation",
        match_count=1,
        inlier_count=1,
        warnings=[],
    )


def _estimate_feature_transform(
    fixed: np.ndarray,
    moving: np.ndarray,
    fixed_mask: np.ndarray | None,
    moving_mask: np.ndarray | None,
) -> RigidTransformResult:
    try:
        import cv2
    except ImportError as exc:
        raise OptionalDependencyError(
            "opencv-contrib-python-headless",
            "registration",
        ) from exc

    fixed_u8 = _to_gray_u8(fixed)
    moving_u8 = _to_gray_u8(moving)
    fixed_mask_u8 = _mask_to_u8(fixed_mask)
    moving_mask_u8 = _mask_to_u8(moving_mask)

    detector, matcher_norm, detector_name = _create_detector(cv2)
    fixed_keypoints, fixed_descriptors = detector.detectAndCompute(
        fixed_u8, fixed_mask_u8
    )
    moving_keypoints, moving_descriptors = detector.detectAndCompute(
        moving_u8, moving_mask_u8
    )
    if fixed_descriptors is None or moving_descriptors is None:
        return _feature_or_mask_fallback(
            "not enough descriptors",
            fixed_mask,
            moving_mask,
        )

    matcher = cv2.BFMatcher(matcher_norm)
    raw_matches = matcher.knnMatch(moving_descriptors, fixed_descriptors, k=2)
    good_matches = [
        pair[0]
        for pair in raw_matches
        if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance
    ]
    if len(good_matches) < 4:
        return _feature_or_mask_fallback(
            "not enough feature matches",
            fixed_mask,
            moving_mask,
            len(good_matches),
        )

    moving_points = np.float32(
        [moving_keypoints[match.queryIdx].pt for match in good_matches]
    )
    fixed_points = np.float32(
        [fixed_keypoints[match.trainIdx].pt for match in good_matches]
    )
    affine, inliers = cv2.estimateAffinePartial2D(
        moving_points,
        fixed_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=4.0,
        maxIters=5000,
        confidence=0.99,
    )
    if affine is None or inliers is None:
        return _feature_or_mask_fallback(
            "RANSAC failed",
            fixed_mask,
            moving_mask,
            len(good_matches),
        )

    matrix = np.eye(3, dtype=float)
    matrix[:2, :] = affine
    inlier_count = int(inliers.sum())
    if inlier_count < 10 or not _feature_transform_is_plausible(matrix):
        fallback = _estimate_mask_moments_transform(fixed_mask, moving_mask)
        if fallback.inlier_count > 0:
            reason = (
                f"feature inliers too low ({inlier_count})"
                if inlier_count < 10
                else "feature transform was degenerate"
            )
            fallback.warnings.append(f"{reason}; used mask moments")
            return fallback
        warning = (
            f"feature inliers too low ({inlier_count}); mask fallback failed"
            if inlier_count < 10
            else "feature transform was degenerate; mask fallback failed"
        )
        return _failed_feature_result(warning, len(good_matches))
    return RigidTransformResult(
        matrix=matrix,
        method=f"feature:{detector_name}",
        match_count=len(good_matches),
        inlier_count=inlier_count,
        warnings=[],
    )


def _failed_feature_result(
    warning: str,
    match_count: int = 0,
) -> RigidTransformResult:
    return RigidTransformResult(
        matrix=np.eye(3, dtype=float),
        method="feature",
        match_count=match_count,
        inlier_count=0,
        warnings=[warning],
    )


def _feature_or_mask_fallback(
    warning: str,
    fixed_mask: np.ndarray | None,
    moving_mask: np.ndarray | None,
    match_count: int = 0,
) -> RigidTransformResult:
    fallback = _estimate_mask_moments_transform(fixed_mask, moving_mask)
    if fallback.inlier_count <= 0:
        return _failed_feature_result(warning, match_count)
    fallback.match_count = match_count
    fallback.warnings.append(f"{warning}; used mask moments")
    return fallback


def _feature_transform_is_plausible(matrix: np.ndarray) -> bool:
    linear = np.asarray(matrix, dtype=float)[:2, :2]
    if not np.isfinite(linear).all() or np.linalg.det(linear) <= 0:
        return False
    singular_values = np.linalg.svd(linear, compute_uv=False)
    return bool(singular_values.min() >= 0.5 and singular_values.max() <= 2.0)


def _estimate_mask_moments_transform(
    fixed_mask: np.ndarray | None,
    moving_mask: np.ndarray | None,
) -> RigidTransformResult:
    if fixed_mask is None or moving_mask is None:
        return _failed_mask_result("mask moments require fixed and moving masks")

    fixed_props = _mask_moment_properties(fixed_mask)
    moving_props = _mask_moment_properties(moving_mask)
    if fixed_props is None or moving_props is None:
        return _failed_mask_result("mask moments require non-empty masks")

    best_matrix: np.ndarray | None = None
    best_dice = -1.0
    for angle_offset in (0.0, np.pi):
        angle = fixed_props["angle"] - moving_props["angle"] + angle_offset
        scale = fixed_props["scale"] / max(moving_props["scale"], 1e-6)
        matrix = _similarity_from_moments(
            moving_props["center"],
            fixed_props["center"],
            angle,
            scale,
        )
        dice = _mask_alignment_dice(fixed_mask, moving_mask, matrix)
        if dice > best_dice:
            best_matrix = matrix
            best_dice = dice

    if best_matrix is None:
        return _failed_mask_result("mask moments failed")
    warnings: list[str] = []
    if best_dice < 0.15:
        warnings.append(f"mask moment overlap is low ({best_dice:.3f})")
    return RigidTransformResult(
        matrix=best_matrix,
        method="mask_moments",
        match_count=0,
        inlier_count=int(round(best_dice * 1000)),
        warnings=warnings,
    )


def _failed_mask_result(warning: str) -> RigidTransformResult:
    return RigidTransformResult(
        matrix=np.eye(3, dtype=float),
        method="mask_moments",
        match_count=0,
        inlier_count=0,
        warnings=[warning],
    )


def _mask_moment_properties(mask: np.ndarray) -> dict[str, np.ndarray | float] | None:
    points_rc = np.column_stack(np.nonzero(np.asarray(mask, dtype=bool)))
    if points_rc.shape[0] < 10:
        return None
    points_xy = points_rc[:, ::-1].astype(float)
    center = points_xy.mean(axis=0)
    centered = points_xy - center
    covariance = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(covariance)
    major = eigvecs[:, int(np.argmax(eigvals))]
    angle = float(np.arctan2(major[1], major[0]))
    scale = float(np.sqrt(max(eigvals.max(), 1e-6)))
    return {"center": center, "angle": angle, "scale": scale}


def _similarity_from_moments(
    moving_center: np.ndarray,
    fixed_center: np.ndarray,
    angle: float,
    scale: float,
) -> np.ndarray:
    cos_angle = float(np.cos(angle) * scale)
    sin_angle = float(np.sin(angle) * scale)
    rotation = np.array(
        [[cos_angle, -sin_angle], [sin_angle, cos_angle]],
        dtype=float,
    )
    translation = fixed_center - rotation @ moving_center
    matrix = np.eye(3, dtype=float)
    matrix[:2, :2] = rotation
    matrix[:2, 2] = translation
    return matrix


def _mask_alignment_dice(
    fixed_mask: np.ndarray,
    moving_mask: np.ndarray,
    matrix: np.ndarray,
) -> float:
    try:
        import cv2
    except ImportError:
        return 0.0

    fixed = np.asarray(fixed_mask, dtype=bool)
    moving = (np.asarray(moving_mask, dtype=bool) * 255).astype(np.uint8)
    warped = cv2.warpAffine(
        moving,
        matrix[:2, :].astype(np.float32),
        dsize=(fixed.shape[1], fixed.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    warped_bool = warped > 0
    intersection = np.logical_and(fixed, warped_bool).sum()
    denominator = fixed.sum() + warped_bool.sum()
    if denominator == 0:
        return 0.0
    return float(2 * intersection / denominator)


def _to_gray_float(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 3:
        arr = arr[:, :, :3].astype(np.float32, copy=False)
        arr = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    else:
        arr = arr.astype(np.float32, copy=False)
    if arr.max(initial=0) > 1.5:
        arr = arr / 255.0
    return arr


def _to_gray_u8(image: np.ndarray) -> np.ndarray:
    gray = _to_gray_float(image)
    gray = np.clip(gray, 0.0, 1.0)
    return (gray * 255).astype(np.uint8)


def _mask_to_u8(mask: np.ndarray | None) -> np.ndarray | None:
    if mask is None:
        return None
    return (np.asarray(mask, dtype=bool) * 255).astype(np.uint8)


def _create_detector(cv2: object) -> tuple[object, int, str]:
    if hasattr(cv2, "AKAZE_create"):
        return cv2.AKAZE_create(), cv2.NORM_HAMMING, "akaze"
    if hasattr(cv2, "ORB_create"):
        return cv2.ORB_create(nfeatures=5000), cv2.NORM_HAMMING, "orb"
    if hasattr(cv2, "SIFT_create"):
        return cv2.SIFT_create(nfeatures=5000), cv2.NORM_L2, "sift"
    msg = "OpenCV has no supported local feature detector"
    raise RuntimeError(msg)
