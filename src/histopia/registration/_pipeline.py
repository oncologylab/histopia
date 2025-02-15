"""End-to-end registration pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from histopia.registration._config import RegistrationConfig
from histopia.registration._io import (
    blend_rgb,
    checkerboard_rgb,
    load_thumbnail,
    overlay_mask,
    resize_mask,
    resize_rgb,
    save_rgb,
    side_by_side,
    warp_mask_thumbnail,
    warp_rgb_thumbnail,
)
from histopia.registration._manifest import _natural_key
from histopia.registration._masking import (
    TissueMaskResult,
    _dominant_component_mask,
    create_tissue_mask,
)
from histopia.registration._nonrigid import (
    NonRigidTransformResult,
    estimate_non_rigid_transform,
    warp_with_displacement,
)
from histopia.registration._rigid import RigidTransformResult, estimate_rigid_transform
from histopia.registration._wsi import (
    WsiWarpResult,
    calculate_thumbnail_overlap_bbox,
    warp_slide_to_reference,
)

INPUT_SUFFIXES = {
    ".ndpi",
    ".scn",
    ".svs",
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
}


@dataclass(slots=True)
class AlignmentMetrics:
    """Mask-based coarse-registration quality measurements."""

    dice: float
    reference_coverage: float
    moving_coverage: float
    warped_area_ratio: float

    def to_json_dict(self) -> dict[str, float]:
        return {
            "dice": self.dice,
            "reference_coverage": self.reference_coverage,
            "moving_coverage": self.moving_coverage,
            "warped_area_ratio": self.warped_area_ratio,
        }


@dataclass(slots=True)
class SlideRegistration:
    """Registration metadata for one slide."""

    path: Path
    is_reference: bool
    mask: TissueMaskResult
    transform: RigidTransformResult
    alignment_metrics: AlignmentMetrics
    aligned_to: Path | None = None
    full_resolution_warp: WsiWarpResult | None = None
    non_rigid_transform: NonRigidTransformResult | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "is_reference": self.is_reference,
            "aligned_to": str(self.aligned_to) if self.aligned_to is not None else None,
            "mask": self.mask.to_json_dict(),
            "transform": self.transform.to_json_dict(),
            "alignment_metrics": self.alignment_metrics.to_json_dict(),
            "full_resolution_warp": (
                self.full_resolution_warp.to_json_dict()
                if self.full_resolution_warp is not None
                else None
            ),
            "non_rigid_transform": (
                self.non_rigid_transform.to_json_dict()
                if self.non_rigid_transform is not None
                else None
            ),
        }


@dataclass(slots=True)
class RegistrationResult:
    """Registration outputs and QC metadata."""

    output_dir: Path
    reference_slide: Path
    slides: tuple[SlideRegistration, ...]
    warnings: tuple[str, ...]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "output_dir": str(self.output_dir),
            "reference_slide": str(self.reference_slide),
            "slides": [slide.to_json_dict() for slide in self.slides],
            "warnings": list(self.warnings),
        }

    def write_json(self, path: Path | str | None = None) -> Path:
        path = (
            Path(path)
            if path is not None
            else self.output_dir / "registration_result.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json_dict(), indent=2) + "\n")
        return path


def register_sections(config: RegistrationConfig) -> RegistrationResult:
    """Run rigid thumbnail registration for a serial-section image folder."""

    slide_paths = _discover_input_slides(config.input_dir)
    if not slide_paths:
        msg = f"no registration input slides found in {config.input_dir}"
        raise FileNotFoundError(msg)

    reference_path = _select_reference(slide_paths, config.reference_slide)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = config.output_dir / "processed"
    qc_dir = config.output_dir / "qc"
    mask_candidate_dir = qc_dir / "mask_candidates"
    alignment_dir = qc_dir / "alignment"
    non_rigid_dir = qc_dir / "non_rigid"
    displacement_dir = config.output_dir / "transforms" / "non_rigid"

    thumbnails: dict[Path, np.ndarray] = {}
    masks: dict[Path, TissueMaskResult] = {}
    warnings: list[str] = []

    for path in slide_paths:
        image = load_thumbnail(path, config.max_processed_image_dim_px)
        mask = create_tissue_mask(image, config.mask)
        thumbnails[path] = image
        masks[path] = mask
        warnings.extend(f"{path.name}: {warning}" for warning in mask.warnings)
        if config.write_processed_images:
            save_rgb(processed_dir / f"{path.stem}.thumbnail.png", image)
            save_rgb(
                qc_dir / f"{path.stem}.mask_overlay.png",
                overlay_mask(image, mask.mask),
            )
            _write_candidate_mask_overlays(mask_candidate_dir, path, image, mask)

    reference_image = thumbnails[reference_path]
    crops = {
        path: _crop_to_mask(
            thumbnails[path],
            masks[path].mask,
            config.max_processed_image_dim_px,
        )
        for path in slide_paths
    }
    reference_crop = crops[reference_path]
    transforms_to_reference, aligned_to = _estimate_transforms_to_reference(
        slide_paths,
        reference_path,
        crops,
        config,
        alignment_dir,
    )

    slides: list[SlideRegistration] = []
    for path in slide_paths:
        if path == reference_path:
            transform = RigidTransformResult(
                matrix=np.eye(3, dtype=float),
                method="identity",
                match_count=0,
                inlier_count=0,
                warnings=[],
            )
        else:
            transform = transforms_to_reference[path]
            warnings.extend(f"{path.name}: {warning}" for warning in transform.warnings)
            if config.write_processed_images:
                _write_alignment_qc(
                    alignment_dir,
                    path,
                    reference_image,
                    thumbnails[path],
                    transform,
                )
                _write_alignment_qc(
                    alignment_dir / "crops",
                    path,
                    reference_crop.image,
                    crops[path].image,
                    _full_transform_to_crop_transform(
                        transform.matrix,
                        reference_crop,
                        crops[path],
                    ),
                )
        non_rigid_transform = None
        if path != reference_path and config.non_rigid_refinement.enabled:
            rigid_moving = warp_rgb_thumbnail(
                thumbnails[path],
                transform.matrix,
                reference_image.shape[:2],
            )
            rigid_moving_mask = warp_mask_thumbnail(
                masks[path].mask,
                transform.matrix,
                reference_image.shape[:2],
            )
            settings = config.non_rigid_refinement
            non_rigid_transform = estimate_non_rigid_transform(
                reference_image,
                rigid_moving,
                fixed_mask=masks[reference_path].mask,
                rigid_moving_mask=rigid_moving_mask,
                max_displacement_fraction=settings.max_displacement_fraction,
                smoothing_sigma_px=settings.smoothing_sigma_px,
                support_dilation_fraction=settings.support_dilation_fraction,
                min_similarity_improvement=settings.min_similarity_improvement,
                max_mask_dice_loss=settings.max_mask_dice_loss,
                min_jacobian_p01=settings.min_jacobian_p01,
                max_jacobian_p99=settings.max_jacobian_p99,
                max_inverse_consistency_fraction=(
                    settings.max_inverse_consistency_fraction
                ),
            )
            if non_rigid_transform.accepted:
                displacement_path = displacement_dir / f"{path.stem}.flow.npz"
                displacement_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    displacement_path,
                    displacement=non_rigid_transform.displacement,
                )
                non_rigid_transform.displacement_path = str(
                    displacement_path.relative_to(config.output_dir)
                )
            if config.write_processed_images:
                _write_non_rigid_qc(
                    non_rigid_dir,
                    path,
                    reference_image,
                    rigid_moving,
                    non_rigid_transform,
                )
        slides.append(
            SlideRegistration(
                path=path,
                is_reference=path == reference_path,
                mask=masks[path],
                transform=transform,
                alignment_metrics=(
                    AlignmentMetrics(1.0, 1.0, 1.0, 1.0)
                    if path == reference_path
                    else _alignment_metrics(
                        reference_crop,
                        crops[path],
                        transform.matrix,
                    )
                ),
                aligned_to=aligned_to.get(path),
                non_rigid_transform=non_rigid_transform,
            )
        )

    result = RegistrationResult(
        output_dir=config.output_dir,
        reference_slide=reference_path,
        slides=tuple(slides),
        warnings=tuple(warnings),
    )
    if config.write_warped_images:
        _write_full_resolution_warps(result, thumbnails, config)
    result.write_json()
    _write_validation_report(result)
    return result


def _write_full_resolution_warps(
    result: RegistrationResult,
    thumbnails: dict[Path, np.ndarray],
    config: RegistrationConfig,
) -> None:
    output_dir = config.registered_output_dir or config.output_dir / "registered"
    reference_shape = thumbnails[result.reference_slide].shape[:2]
    reference_thumbnail_bbox = None
    if config.crop_mode == "overlap":
        reference_thumbnail_bbox = calculate_thumbnail_overlap_bbox(
            [
                (thumbnails[slide.path].shape[:2], slide.transform.matrix)
                for slide in result.slides
            ],
            reference_shape,
        )
    for slide in result.slides:
        output_path = output_dir / f"{slide.path.stem}.registered.tiff"
        slide.full_resolution_warp = warp_slide_to_reference(
            slide.path,
            result.reference_slide,
            output_path,
            slide.transform.matrix,
            moving_thumbnail_shape=thumbnails[slide.path].shape[:2],
            reference_thumbnail_shape=reference_shape,
            compression=config.wsi_compression,
            jpeg_quality=config.wsi_jpeg_quality,
            tile_size=config.wsi_tile_size,
            reference_to_rigid_moving_displacement=(
                slide.non_rigid_transform.displacement
                if slide.non_rigid_transform is not None
                and slide.non_rigid_transform.accepted
                else None
            ),
            reference_thumbnail_bbox=reference_thumbnail_bbox,
        )


def _write_non_rigid_qc(
    output_dir: Path,
    path: Path,
    reference_image: np.ndarray,
    rigid_moving: np.ndarray,
    result: NonRigidTransformResult,
) -> None:
    refined = warp_with_displacement(
        rigid_moving,
        result.displacement,
        border_value=(255, 255, 255),
    )
    magnitude = np.linalg.norm(result.displacement, axis=2)
    maximum = max(float(magnitude.max()), 1e-6)
    magnitude_rgb = np.repeat(
        ((magnitude / maximum) * 255).astype(np.uint8)[:, :, np.newaxis],
        3,
        axis=2,
    )
    save_rgb(
        output_dir / f"{path.stem}.contact.png",
        side_by_side(
            [
                reference_image,
                rigid_moving,
                refined,
                blend_rgb(reference_image, refined),
                checkerboard_rgb(reference_image, refined),
                magnitude_rgb,
            ]
        ),
    )


def _estimate_transforms_to_reference(
    slide_paths: tuple[Path, ...],
    reference_path: Path,
    crops: dict[Path, _Crop],
    config: RegistrationConfig,
    alignment_dir: Path,
) -> tuple[dict[Path, RigidTransformResult], dict[Path, Path]]:
    if config.align_strategy == "reference":
        return _estimate_reference_transforms(
            slide_paths,
            reference_path,
            crops,
            config,
            alignment_dir,
        )
    if config.align_strategy == "serial":
        return _estimate_serial_transforms(
            slide_paths,
            reference_path,
            crops,
            config,
            alignment_dir,
        )
    if config.align_strategy == "hybrid":
        return _estimate_hybrid_transforms(
            slide_paths,
            reference_path,
            crops,
            config,
            alignment_dir,
        )
    msg = f"unsupported alignment strategy: {config.align_strategy!r}"
    raise ValueError(msg)


def _estimate_reference_transforms(
    slide_paths: tuple[Path, ...],
    reference_path: Path,
    crops: dict[Path, _Crop],
    config: RegistrationConfig,
    alignment_dir: Path,
) -> tuple[dict[Path, RigidTransformResult], dict[Path, Path]]:
    transforms: dict[Path, RigidTransformResult] = {}
    aligned_to: dict[Path, Path] = {}
    for path in slide_paths:
        if path == reference_path:
            continue
        transform, crop_transform = _estimate_pair_transform(
            reference_path,
            path,
            crops,
            config,
        )
        transforms[path] = transform
        aligned_to[path] = reference_path
        if config.write_processed_images:
            _write_alignment_qc(
                alignment_dir / "pair_crops",
                path,
                crops[reference_path].image,
                crops[path].image,
                crop_transform,
            )
    return transforms, aligned_to


def _estimate_serial_transforms(
    slide_paths: tuple[Path, ...],
    reference_path: Path,
    crops: dict[Path, _Crop],
    config: RegistrationConfig,
    alignment_dir: Path,
) -> tuple[dict[Path, RigidTransformResult], dict[Path, Path]]:
    reference_index = slide_paths.index(reference_path)
    transforms: dict[Path, RigidTransformResult] = {}
    aligned_to: dict[Path, Path] = {}
    cumulative: dict[Path, np.ndarray] = {reference_path: np.eye(3, dtype=float)}

    for index in range(reference_index + 1, len(slide_paths)):
        moving_path = slide_paths[index]
        fixed_path = slide_paths[index - 1]
        pair_transform, crop_transform = _estimate_pair_transform(
            fixed_path,
            moving_path,
            crops,
            config,
        )
        cumulative[moving_path] = cumulative[fixed_path] @ pair_transform.matrix
        transforms[moving_path] = _composed_transform_result(
            cumulative[moving_path],
            pair_transform,
        )
        aligned_to[moving_path] = fixed_path
        if config.write_processed_images:
            _write_alignment_qc(
                alignment_dir / "pair_crops",
                moving_path,
                crops[fixed_path].image,
                crops[moving_path].image,
                crop_transform,
            )

    for index in range(reference_index - 1, -1, -1):
        moving_path = slide_paths[index]
        fixed_path = slide_paths[index + 1]
        pair_transform, crop_transform = _estimate_pair_transform(
            fixed_path,
            moving_path,
            crops,
            config,
        )
        cumulative[moving_path] = cumulative[fixed_path] @ pair_transform.matrix
        transforms[moving_path] = _composed_transform_result(
            cumulative[moving_path],
            pair_transform,
        )
        aligned_to[moving_path] = fixed_path
        if config.write_processed_images:
            _write_alignment_qc(
                alignment_dir / "pair_crops",
                moving_path,
                crops[fixed_path].image,
                crops[moving_path].image,
                crop_transform,
            )

    return transforms, aligned_to


def _estimate_hybrid_transforms(
    slide_paths: tuple[Path, ...],
    reference_path: Path,
    crops: dict[Path, _Crop],
    config: RegistrationConfig,
    alignment_dir: Path,
) -> tuple[dict[Path, RigidTransformResult], dict[Path, Path]]:
    reference_transforms, reference_aligned_to = _estimate_reference_transforms(
        slide_paths,
        reference_path,
        crops,
        config,
        alignment_dir,
    )
    serial_transforms, serial_aligned_to = _estimate_serial_transforms(
        slide_paths,
        reference_path,
        crops,
        config,
        alignment_dir,
    )

    transforms: dict[Path, RigidTransformResult] = {}
    aligned_to: dict[Path, Path] = {}
    for path in slide_paths:
        if path == reference_path:
            continue
        reference_score = _final_mask_dice(
            crops[reference_path],
            crops[path],
            reference_transforms[path].matrix,
        )
        serial_score = _final_mask_dice(
            crops[reference_path],
            crops[path],
            serial_transforms[path].matrix,
        )
        if serial_score > reference_score + 0.03:
            selected = serial_transforms[path]
            selected_to = serial_aligned_to[path]
            selected_score = serial_score
            selected_label = "serial"
        else:
            selected = reference_transforms[path]
            selected_to = reference_aligned_to[path]
            selected_score = reference_score
            selected_label = "reference"
        selected_method = selected.method
        if selected_label == "serial" and selected_method.startswith("serial:"):
            selected_method = selected_method.removeprefix("serial:")
        transforms[path] = RigidTransformResult(
            matrix=selected.matrix,
            method=f"hybrid:{selected_label}:{selected_method}",
            match_count=selected.match_count,
            inlier_count=max(selected.inlier_count, int(round(selected_score * 1000))),
            warnings=list(selected.warnings),
        )
        aligned_to[path] = selected_to
    return transforms, aligned_to


def _estimate_pair_transform(
    fixed_path: Path,
    moving_path: Path,
    crops: dict[Path, _Crop],
    config: RegistrationConfig,
) -> tuple[RigidTransformResult, RigidTransformResult]:
    fixed_crop = crops[fixed_path]
    moving_crop = crops[moving_path]
    crop_transform = estimate_rigid_transform(
        fixed_crop.image,
        moving_crop.image,
        fixed_mask=fixed_crop.mask,
        moving_mask=moving_crop.mask,
        method=config.rigid_method,
        refine=config.refinement.enabled,
        refinement_max_dim_px=config.refinement.max_dim_px,
        min_dice_improvement=config.refinement.min_dice_improvement,
        max_relative_scale_change=config.refinement.max_relative_scale_change,
        max_relative_anisotropy=config.refinement.max_relative_anisotropy,
    )
    full_transform = RigidTransformResult(
        matrix=_compose_crop_transform(
            crop_transform.matrix,
            fixed_crop,
            moving_crop,
        ),
        method=crop_transform.method,
        match_count=crop_transform.match_count,
        inlier_count=crop_transform.inlier_count,
        warnings=list(crop_transform.warnings),
    )
    return full_transform, crop_transform


def _composed_transform_result(
    matrix: np.ndarray,
    pair_transform: RigidTransformResult,
) -> RigidTransformResult:
    warnings = list(pair_transform.warnings)
    return RigidTransformResult(
        matrix=matrix,
        method=f"serial:{pair_transform.method}",
        match_count=pair_transform.match_count,
        inlier_count=pair_transform.inlier_count,
        warnings=warnings,
    )


@dataclass(slots=True)
class _Crop:
    image: np.ndarray
    mask: np.ndarray
    offset_xy: np.ndarray
    scale: float


def _crop_to_mask(
    image: np.ndarray,
    mask: np.ndarray,
    target_dim_px: int,
    padding_fraction: float = 0.08,
) -> _Crop:
    mask_bool = np.asarray(mask, dtype=bool)
    if not mask_bool.any():
        scale = _crop_scale(image.shape[:2], target_dim_px)
        return _Crop(
            image=resize_rgb(image, scale),
            mask=resize_mask(mask_bool, scale),
            offset_xy=np.array([0.0, 0.0], dtype=float),
            scale=scale,
        )
    extent_mask = _dominant_component_mask(mask_bool)
    rows, cols = np.nonzero(extent_mask)
    height, width = mask_bool.shape
    pad = int(max(rows.max() - rows.min(), cols.max() - cols.min()) * padding_fraction)
    row0 = max(0, int(rows.min()) - pad)
    row1 = min(height, int(rows.max()) + pad + 1)
    col0 = max(0, int(cols.min()) - pad)
    col1 = min(width, int(cols.max()) + pad + 1)
    cropped_image = image[row0:row1, col0:col1]
    cropped_mask = mask_bool[row0:row1, col0:col1]
    scale = _crop_scale(cropped_image.shape[:2], target_dim_px)
    return _Crop(
        image=resize_rgb(cropped_image, scale),
        mask=resize_mask(cropped_mask, scale),
        offset_xy=np.array([col0, row0], dtype=float),
        scale=scale,
    )


def _crop_scale(shape_rc: tuple[int, int], target_dim_px: int) -> float:
    longest = max(shape_rc)
    if longest <= 0:
        return 1.0
    return target_dim_px / longest


def _compose_crop_transform(
    crop_matrix: np.ndarray,
    fixed_crop: _Crop,
    moving_crop: _Crop,
) -> np.ndarray:
    to_moving_crop = np.eye(3, dtype=float)
    to_moving_crop[:2, 2] = -moving_crop.offset_xy
    moving_scale = np.eye(3, dtype=float)
    moving_scale[0, 0] = moving_crop.scale
    moving_scale[1, 1] = moving_crop.scale
    fixed_unscale = np.eye(3, dtype=float)
    fixed_unscale[0, 0] = 1.0 / fixed_crop.scale
    fixed_unscale[1, 1] = 1.0 / fixed_crop.scale
    to_fixed_full = np.eye(3, dtype=float)
    to_fixed_full[:2, 2] = fixed_crop.offset_xy
    return to_fixed_full @ fixed_unscale @ crop_matrix @ moving_scale @ to_moving_crop


def _full_transform_to_crop_transform(
    full_matrix: np.ndarray,
    fixed_crop: _Crop,
    moving_crop: _Crop,
) -> RigidTransformResult:
    moving_unscale = np.eye(3, dtype=float)
    moving_unscale[0, 0] = 1.0 / moving_crop.scale
    moving_unscale[1, 1] = 1.0 / moving_crop.scale
    to_moving_full = np.eye(3, dtype=float)
    to_moving_full[:2, 2] = moving_crop.offset_xy

    to_fixed_crop = np.eye(3, dtype=float)
    to_fixed_crop[:2, 2] = -fixed_crop.offset_xy
    fixed_scale = np.eye(3, dtype=float)
    fixed_scale[0, 0] = fixed_crop.scale
    fixed_scale[1, 1] = fixed_crop.scale

    crop_matrix = (
        fixed_scale @ to_fixed_crop @ full_matrix @ to_moving_full @ moving_unscale
    )
    return RigidTransformResult(
        matrix=crop_matrix,
        method="crop_view",
        match_count=0,
        inlier_count=0,
        warnings=[],
    )


def _final_mask_dice(
    reference_crop: _Crop,
    moving_crop: _Crop,
    full_matrix: np.ndarray,
) -> float:
    return _alignment_metrics(reference_crop, moving_crop, full_matrix).dice


def _alignment_metrics(
    reference_crop: _Crop,
    moving_crop: _Crop,
    full_matrix: np.ndarray,
) -> AlignmentMetrics:
    try:
        import cv2
    except ImportError:
        return AlignmentMetrics(0.0, 0.0, 0.0, 0.0)

    crop_transform = _full_transform_to_crop_transform(
        full_matrix,
        reference_crop,
        moving_crop,
    )
    moving = moving_crop.mask.astype(np.uint8) * 255
    warped = cv2.warpAffine(
        moving,
        crop_transform.matrix[:2, :].astype(np.float32),
        dsize=(reference_crop.mask.shape[1], reference_crop.mask.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    warped_bool = warped > 0
    intersection = float(np.logical_and(reference_crop.mask, warped_bool).sum())
    reference_area = float(reference_crop.mask.sum())
    moving_area = float(warped_bool.sum())
    denominator = reference_area + moving_area
    if denominator == 0:
        return AlignmentMetrics(0.0, 0.0, 0.0, 0.0)
    return AlignmentMetrics(
        dice=2 * intersection / denominator,
        reference_coverage=intersection / reference_area if reference_area else 0.0,
        moving_coverage=intersection / moving_area if moving_area else 0.0,
        warped_area_ratio=moving_area / reference_area if reference_area else 0.0,
    )


def _write_candidate_mask_overlays(
    output_dir: Path,
    path: Path,
    image: np.ndarray,
    mask: TissueMaskResult,
) -> None:
    for method, candidate in mask.candidate_masks.items():
        save_rgb(
            output_dir / f"{path.stem}.{method}.png",
            overlay_mask(image, candidate),
        )


def _write_alignment_qc(
    output_dir: Path,
    path: Path,
    reference_image: np.ndarray,
    moving_image: np.ndarray,
    transform: RigidTransformResult,
) -> None:
    warped = warp_rgb_thumbnail(
        moving_image,
        transform.matrix,
        reference_image.shape[:2],
    )
    save_rgb(output_dir / f"{path.stem}.warped.png", warped)
    save_rgb(output_dir / f"{path.stem}.blend.png", blend_rgb(reference_image, warped))
    save_rgb(
        output_dir / f"{path.stem}.checkerboard.png",
        checkerboard_rgb(reference_image, warped),
    )
    save_rgb(
        output_dir / f"{path.stem}.contact.png",
        side_by_side(
            [
                reference_image,
                moving_image,
                warped,
                blend_rgb(reference_image, warped),
                checkerboard_rgb(reference_image, warped),
            ]
        ),
    )


def _write_validation_report(result: RegistrationResult) -> Path:
    lines = [
        "# Histopia Registration Validation Report",
        "",
        f"Output directory: `{result.output_dir}`",
        f"Reference slide: `{result.reference_slide.name}`",
        f"Slide count: {len(result.slides)}",
        f"Warning count: {len(result.warnings)}",
        "",
        "## Slide QC",
        "",
        (
            "| Slide | Status | Aligned to | Transform | Mask | Foreground | "
            "Border strip | Dice | Ref coverage | Moving coverage | Area ratio | "
            "Warnings |"
        ),
        "|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for slide in result.slides:
        mask_metrics = slide.mask.metrics
        warnings = [*slide.mask.warnings, *slide.transform.warnings]
        status = _slide_status(slide)
        lines.append(
            "| "
            f"{_escape_table(slide.path.name)} | "
            f"{status} | "
            f"{_escape_table(slide.aligned_to.name if slide.aligned_to else '')} | "
            f"{slide.transform.method} | "
            f"{slide.mask.method} | "
            f"{mask_metrics['foreground_fraction']:.3f} | "
            f"{mask_metrics['max_border_strip_foreground_fraction']:.3f} | "
            f"{slide.alignment_metrics.dice:.3f} | "
            f"{slide.alignment_metrics.reference_coverage:.3f} | "
            f"{slide.alignment_metrics.moving_coverage:.3f} | "
            f"{slide.alignment_metrics.warped_area_ratio:.3f} | "
            f"{_escape_table('; '.join(warnings))} |"
        )
    lines.append("")
    report_path = result.output_dir / "validation_report.md"
    report_path.write_text("\n".join(lines) + "\n")
    return report_path


def _slide_status(slide: SlideRegistration) -> str:
    if slide.is_reference:
        return "reference"
    metrics = slide.alignment_metrics
    if (
        metrics.dice < 0.25
        or metrics.reference_coverage < 0.15
        or metrics.moving_coverage < 0.25
        or slide.transform.inlier_count < 10
    ):
        return "fail"
    if (
        metrics.dice < 0.55
        or metrics.reference_coverage < 0.35
        or metrics.moving_coverage < 0.50
        or slide.mask.method == "full"
    ):
        return "review"
    return "pass"


def _escape_table(value: str) -> str:
    return value.replace("|", "\\|")


def _discover_input_slides(input_dir: Path) -> tuple[Path, ...]:
    return tuple(
        sorted(
            (
                path
                for path in input_dir.iterdir()
                if path.is_file() and path.suffix.lower() in INPUT_SUFFIXES
            ),
            key=lambda path: _natural_key(path.name),
        )
    )


def _select_reference(
    slide_paths: tuple[Path, ...],
    reference_slide: str | None,
) -> Path:
    if reference_slide is None:
        return slide_paths[0]
    for path in slide_paths:
        if path.name == reference_slide or path.stem == reference_slide:
            return path
    msg = f"reference slide {reference_slide!r} was not found"
    raise FileNotFoundError(msg)
