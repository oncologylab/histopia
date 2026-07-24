"""End-to-end registration pipeline."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from histopia.registration._config import RegistrationConfig
from histopia.registration._io import (
    blend_rgb,
    checkerboard_rgb,
    overlay_mask,
    resize_mask,
    resize_rgb,
    save_rgb,
    side_by_side,
    warp_mask_thumbnail,
    warp_rgb_thumbnail,
)
from histopia.registration._masking import (
    TissueMaskResult,
    _dominant_component_mask,
    create_tissue_mask,
    evaluate_tissue_mask,
    refine_group_tissue_masks,
)
from histopia.registration._nonrigid import (
    NonRigidTransformResult,
    estimate_non_rigid_transform,
    warp_with_displacement,
)
from histopia.registration._ordering import (
    order_is_approved,
    propose_anchored_order,
    write_order_proposal,
)
from histopia.registration._ordering_cache import (
    load_ordering_distance_cache,
    ordering_cache_fingerprint,
    write_ordering_distance_cache,
)
from histopia.registration._orientation import (
    apply_quarter_turn,
    load_orientation_overrides,
    quarter_turn_matrix,
)
from histopia.registration._qc import write_labeled_review_panel
from histopia.registration._review import (
    MaskReviewEntry,
    load_mask_review,
    resolve_reviewed_mask,
    write_mask_review,
)
from histopia.registration._rigid import (
    RigidTransformResult,
    estimate_rigid_transform,
)
from histopia.registration._slides import (
    SlideGeometry,
    discover_slides,
    load_slide_thumbnail,
)
from histopia.registration._wsi import (
    WsiWarpResult,
    calculate_thumbnail_overlap_bbox,
    warp_slide_to_reference,
)


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
    geometry: SlideGeometry | None = None
    mask_review: MaskReviewEntry | None = None
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
            "geometry": self.geometry.to_json_dict() if self.geometry else None,
            "mask_review": (
                self.mask_review.to_json_dict() if self.mask_review else None
            ),
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

    slide_paths = discover_slides(config.input_dir, wsi_only=config.wsi_only)
    if not slide_paths:
        msg = f"no registration input slides found in {config.input_dir}"
        raise FileNotFoundError(msg)

    if config.section_order_strategy != "anchored_similarity":
        slide_paths = _apply_section_order(slide_paths, config.section_order_path)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = config.output_dir / "processed"
    qc_dir = config.output_dir / "qc"
    mask_candidate_dir = qc_dir / "mask_candidates"
    alignment_dir = qc_dir / "alignment"
    non_rigid_dir = qc_dir / "non_rigid"
    displacement_dir = config.output_dir / "transforms" / "non_rigid"

    thumbnails, geometries = _load_registration_thumbnails(slide_paths, config)
    masks: dict[Path, TissueMaskResult] = {}
    review_entries = load_mask_review(config.mask_review_path)
    resolved_reviews: dict[Path, MaskReviewEntry] = {}
    warnings: list[str] = []

    if config.automatic_mask_snapshot_path is not None:
        masks = _load_automatic_mask_snapshot(
            config.automatic_mask_snapshot_path,
            slide_paths,
            thumbnails,
        )
    else:
        masks = _create_tissue_masks(thumbnails, config)
        physical_pixel_areas = {
            path: _thumbnail_physical_pixel_area(geometry)
            for path, geometry in geometries.items()
        }
        masks = refine_group_tissue_masks(
            masks,
            physical_pixel_areas=physical_pixel_areas,
            images=thumbnails,
        )
    for path in slide_paths:
        image = thumbnails[path]
        geometry = geometries[path]
        automatic_mask = masks[path]
        mask, review = resolve_reviewed_mask(
            slide_path=path,
            image=image,
            geometry=geometry,
            automatic=automatic_mask,
            review_entries=review_entries,
            override_dir=config.mask_override_dir,
            require_approved=False,
        )
        masks[path] = mask
        review_entries[path.name] = review
        resolved_reviews[path] = review
        warnings.extend(f"{path.name}: {warning}" for warning in mask.warnings)
        if config.write_processed_images:
            save_rgb(processed_dir / f"{path.stem}.thumbnail.png", image)
            save_rgb(
                processed_dir / f"{path.stem}.mask.png",
                (mask.mask * 255).astype(np.uint8),
            )
            save_rgb(
                qc_dir / f"{path.stem}.mask_overlay.png",
                overlay_mask(image, mask.mask),
            )
            _write_candidate_mask_overlays(mask_candidate_dir, path, image, mask)

    review_path = config.mask_review_path or config.output_dir / "mask_review.json"
    write_mask_review(review_path, review_entries)
    invalid_masks = [path.name for path in slide_paths if not masks[path].accepted]
    unapproved_masks = [
        path.name for path in slide_paths if not resolved_reviews[path].approved
    ]
    if invalid_masks:
        msg = "automatic tissue masks failed: " + ", ".join(invalid_masks)
        raise ValueError(msg)
    if config.require_approved_masks and unapproved_masks:
        msg = "registration requires approved masks: " + ", ".join(unapproved_masks)
        raise ValueError(msg)

    orientation_turns = load_orientation_overrides(
        config.section_orientation_path,
        tuple(path.name for path in slide_paths),
    )
    if any(orientation_turns.values()) and config.non_rigid_refinement.enabled:
        raise ValueError(
            "section orientation overrides cannot currently be combined with "
            "non-rigid refinement"
        )
    working_thumbnails = {
        path: apply_quarter_turn(thumbnails[path], orientation_turns[path.name])
        for path in slide_paths
    }
    working_masks = {
        path: apply_quarter_turn(masks[path].mask, orientation_turns[path.name])
        for path in slide_paths
    }

    crops = {
        path: _crop_to_mask(
            working_thumbnails[path],
            working_masks[path],
            config.max_processed_image_dim_px,
        )
        for path in slide_paths
    }
    if config.section_order_strategy == "similarity":
        slide_paths = _similarity_section_order(slide_paths, crops, config)
    elif config.section_order_strategy == "anchored_similarity":
        physical_areas = {
            path: _physical_mask_area(working_masks[path], geometries[path])
            for path in slide_paths
        }
        input_fingerprints = {
            path.name: _ordering_input_fingerprint(
                working_masks[path],
                geometries[path],
                orientation_turns[path.name],
            )
            for path in slide_paths
        }
        distance_fingerprint = ordering_cache_fingerprint(
            tuple(path.name for path in slide_paths),
            input_fingerprints,
            _ordering_distance_settings(config),
        )
        cache_path = config.output_dir / ".cache" / "section-order-distances.npz"
        distances = load_ordering_distance_cache(
            cache_path,
            expected_fingerprint=distance_fingerprint,
            expected_size=len(slide_paths),
        )
        if distances is None:
            distances = _section_distance_matrix(
                slide_paths, crops, config, physical_areas=physical_areas
            )
            write_ordering_distance_cache(
                cache_path,
                distances,
                fingerprint=distance_fingerprint,
            )
        fixed_positions = _read_fixed_positions(slide_paths, config.section_order_path)
        proposal = propose_anchored_order(
            tuple(path.name for path in slide_paths),
            distances,
            fixed_positions,
            physical_areas_um2={
                path.name: physical_areas[path] for path in slide_paths
            },
            input_fingerprints=input_fingerprints,
            orientation_quarter_turns=orientation_turns,
            cavity_fractions={
                path.name: _largest_internal_cavity_fraction(working_masks[path])
                for path in slide_paths
            },
        )
        order_review_path = (
            config.section_order_review_path
            or config.output_dir / "section_order_review.json"
        )
        write_order_proposal(order_review_path, proposal)
        if config.require_approved_order and not order_is_approved(
            order_review_path, proposal.fingerprint
        ):
            raise ValueError(
                "registration requires approval of the current section order: "
                f"{order_review_path}"
            )
        path_by_name = {path.name: path for path in slide_paths}
        slide_paths = tuple(path_by_name[name] for name in proposal.slides)
    if config.reference_slide is not None or config.reference_policy == "explicit":
        reference_path = _select_reference(slide_paths, config.reference_slide)
    else:
        reference_path = _select_best_connected_reference(slide_paths, crops, config)
    reference_image = working_thumbnails[reference_path]
    reference_crop = crops[reference_path]
    transforms_to_reference, aligned_to = _estimate_transforms_to_reference(
        slide_paths,
        reference_path,
        crops,
        config,
        alignment_dir,
    )
    _apply_affine_overrides(
        transforms_to_reference,
        aligned_to,
        slide_paths,
        reference_path,
        config.affine_override_path,
    )

    slides: list[SlideRegistration] = []
    for path in slide_paths:
        if path == reference_path:
            working_transform = RigidTransformResult(
                matrix=np.eye(3, dtype=float),
                method="identity",
                match_count=0,
                inlier_count=0,
                warnings=[],
            )
        else:
            working_transform = transforms_to_reference[path]
            warnings.extend(
                f"{path.name}: {warning}" for warning in working_transform.warnings
            )
            if config.write_processed_images:
                _write_alignment_qc(
                    alignment_dir,
                    path,
                    reference_image,
                    working_thumbnails[path],
                    working_transform,
                )
                _write_alignment_qc(
                    alignment_dir / "crops",
                    path,
                    reference_crop.image,
                    crops[path].image,
                    _full_transform_to_crop_transform(
                        working_transform.matrix,
                        reference_crop,
                        crops[path],
                    ),
                )
        non_rigid_transform = None
        if path != reference_path and config.non_rigid_refinement.enabled:
            rigid_moving = warp_rgb_thumbnail(
                working_thumbnails[path],
                working_transform.matrix,
                reference_image.shape[:2],
            )
            rigid_moving_mask = warp_mask_thumbnail(
                working_masks[path],
                working_transform.matrix,
                reference_image.shape[:2],
            )
            settings = config.non_rigid_refinement
            non_rigid_transform = estimate_non_rigid_transform(
                reference_image,
                rigid_moving,
                fixed_mask=working_masks[reference_path],
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
        transform = _transform_from_oriented_coordinates(
            working_transform,
            thumbnails[path].shape[:2],
            orientation_turns[path.name],
            thumbnails[reference_path].shape[:2],
            orientation_turns[reference_path.name],
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
                        working_transform.matrix,
                    )
                ),
                geometry=geometries[path],
                mask_review=resolved_reviews[path],
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
    if config.write_processed_images:
        _write_primary_review_panels(result, thumbnails, geometries, qc_dir / "review")
    if config.write_warped_images:
        _write_full_resolution_warps(result, thumbnails, geometries, config)
    result.write_json()
    _write_validation_report(result)
    return result


def _load_registration_thumbnails(
    slide_paths: tuple[Path, ...] | list[Path],
    config: RegistrationConfig,
) -> tuple[dict[Path, np.ndarray], dict[Path, SlideGeometry]]:
    """Decode slide thumbnails with bounded, order-preserving parallelism."""

    paths = tuple(slide_paths)

    def load(path: Path) -> tuple[np.ndarray, SlideGeometry]:
        return load_slide_thumbnail(path, config.max_processed_image_dim_px)

    if config.thumbnail_workers == 1:
        loaded = map(load, paths)
        return _unpack_loaded_thumbnails(paths, loaded)
    with ThreadPoolExecutor(max_workers=config.thumbnail_workers) as executor:
        return _unpack_loaded_thumbnails(paths, executor.map(load, paths))


def _unpack_loaded_thumbnails(
    paths: tuple[Path, ...],
    loaded: Iterable[tuple[np.ndarray, SlideGeometry]],
) -> tuple[dict[Path, np.ndarray], dict[Path, SlideGeometry]]:
    thumbnails: dict[Path, np.ndarray] = {}
    geometries: dict[Path, SlideGeometry] = {}
    for path, (image, geometry) in zip(paths, loaded, strict=True):
        thumbnails[path] = image
        geometries[path] = geometry
    return thumbnails, geometries


def _create_tissue_masks(
    thumbnails: dict[Path, np.ndarray],
    config: RegistrationConfig,
) -> dict[Path, TissueMaskResult]:
    """Create independent masks with bounded, deterministic CPU parallelism."""

    items = tuple(thumbnails.items())

    def create(item: tuple[Path, np.ndarray]) -> tuple[Path, TissueMaskResult]:
        path, image = item
        return path, create_tissue_mask(image, config.mask)

    if config.mask_workers == 1:
        return dict(map(create, items))
    with ThreadPoolExecutor(max_workers=config.mask_workers) as executor:
        return dict(executor.map(create, items))


def _write_primary_review_panels(
    result: RegistrationResult,
    thumbnails: dict[Path, np.ndarray],
    geometries: dict[Path, SlideGeometry],
    output_dir: Path,
) -> None:
    reference = thumbnails[result.reference_slide]
    for slide in result.slides:
        source = thumbnails[slide.path]
        registered = warp_rgb_thumbnail(
            source,
            slide.transform.matrix,
            reference.shape[:2],
        )
        mask_overlay = overlay_mask(source, slide.mask.mask)
        geometry = geometries[slide.path]
        review_status = slide.mask_review.status if slide.mask_review else "untracked"
        x, y, width, height = geometry.content_bbox_xywh
        write_labeled_review_panel(
            output_dir / f"{slide.path.stem}.review.png",
            panes=[
                ("Source WSI content", source),
                ("Approved tissue mask", mask_overlay),
                ("Affine registered", registered),
                ("Reference overlay", blend_rgb(reference, registered)),
            ],
            title=slide.path.name,
            metadata=[
                (
                    f"format={slide.path.suffix.lower()} "
                    f"native={geometry.native_shape[1]}x{geometry.native_shape[0]}"
                ),
                (
                    f"content_bounds=({x},{y},{width},{height}) "
                    f"source={geometry.bounds_source}"
                ),
                (
                    f"mask={review_status}:{slide.mask.method} "
                    f"affine={slide.transform.method}"
                ),
                f"reference={result.reference_slide.name}",
            ],
        )


def _write_full_resolution_warps(
    result: RegistrationResult,
    thumbnails: dict[Path, np.ndarray],
    geometries: dict[Path, SlideGeometry],
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
            moving_geometry=geometries[slide.path],
            reference_geometry=geometries[result.reference_slide],
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
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=True)
    for method, candidate in mask.candidate_masks.items():
        Image.fromarray(candidate.astype(np.uint8) * 255).save(
            output_dir / f"{path.stem}.{method}.mask.png"
        )
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


def _apply_section_order(
    slide_paths: tuple[Path, ...],
    order_path: Path | None,
) -> tuple[Path, ...]:
    if order_path is None:
        return slide_paths
    if not order_path.exists():
        raise FileNotFoundError(f"section order manifest not found: {order_path}")
    if order_path.suffix.lower() == ".json":
        payload = json.loads(order_path.read_text())
        rows = payload.get("slides", payload)
        if isinstance(rows, dict):
            order_by_name = {str(name): int(order) for name, order in rows.items()}
        else:
            order_by_name = {
                str(row.get("slide", row.get("filename"))): int(row["order"])
                for row in rows
            }
    else:
        with order_path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        order_by_name = {
            str(row.get("slide") or row.get("filename") or row.get("raw_name")): int(
                row["order"]
            )
            for row in rows
            if row.get("order") not in {None, "", "0"}
        }
    path_by_name = {path.name: path for path in slide_paths}
    path_by_stem = {path.stem: path for path in slide_paths}
    ordered: list[tuple[int, Path]] = []
    for name, order in order_by_name.items():
        path = path_by_name.get(name) or path_by_stem.get(Path(name).stem)
        if path is not None:
            ordered.append((order, path))
    ordered_paths = [path for _, path in sorted(ordered, key=lambda item: item[0])]
    remaining = [path for path in slide_paths if path not in set(ordered_paths)]
    return tuple([*ordered_paths, *remaining])


def _apply_affine_overrides(
    transforms: dict[Path, RigidTransformResult],
    aligned_to: dict[Path, Path],
    slide_paths: tuple[Path, ...],
    reference_path: Path,
    override_path: Path | None,
) -> None:
    if override_path is None:
        return
    payload = json.loads(override_path.read_text())
    rows = payload.get("slides", payload)
    if isinstance(rows, dict):
        matrix_by_name = rows
    else:
        matrix_by_name = {
            str(row.get("slide", row.get("filename"))): row["matrix"] for row in rows
        }
    for path in slide_paths:
        matrix_payload = matrix_by_name.get(path.name) or matrix_by_name.get(path.stem)
        if matrix_payload is None:
            continue
        matrix = np.asarray(matrix_payload, dtype=float)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            msg = f"invalid affine override for {path.name}"
            raise ValueError(msg)
        transforms[path] = RigidTransformResult(
            matrix=matrix,
            method="manual_affine_override",
            match_count=0,
            inlier_count=0,
            warnings=[],
        )
        aligned_to[path] = reference_path


def _select_best_connected_reference(
    slide_paths: tuple[Path, ...],
    crops: dict[Path, _Crop],
    config: RegistrationConfig,
) -> Path:
    if len(slide_paths) == 1:
        return slide_paths[0]
    midpoint = (len(slide_paths) - 1) / 2
    scored: list[tuple[float, Path]] = []
    for index, candidate in enumerate(slide_paths):
        neighbor_indices = {
            other
            for delta in (-2, -1, 1, 2)
            if 0 <= (other := index + delta) < len(slide_paths)
        }
        edge_score = 0.0
        successful_edges = 0
        for other_index in neighbor_indices:
            moving = slide_paths[other_index]
            try:
                transform, _ = _estimate_pair_transform(
                    candidate,
                    moving,
                    crops,
                    config,
                )
            except (ValueError, np.linalg.LinAlgError):
                continue
            dice = _final_mask_dice(crops[candidate], crops[moving], transform.matrix)
            edge_score += np.log1p(transform.inlier_count) + 3.0 * dice
            successful_edges += 1
        centrality = 1.0 - abs(index - midpoint) / max(midpoint, 1.0)
        scored.append((edge_score + successful_edges + centrality, candidate))
    return max(scored, key=lambda item: item[0])[1]


def _similarity_section_order(
    slide_paths: tuple[Path, ...],
    crops: dict[Path, _Crop],
    config: RegistrationConfig,
) -> tuple[Path, ...]:
    """Infer an undirected morphology order for registration, not physical z."""

    if len(slide_paths) < 3:
        return slide_paths
    from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering
    from scipy.spatial.distance import squareform

    distances = _section_distance_matrix(slide_paths, crops, config)
    condensed = squareform(distances, checks=False)
    tree = linkage(condensed, method="average")
    ordered_tree = optimal_leaf_ordering(tree, condensed)
    indices = leaves_list(ordered_tree)
    return tuple(slide_paths[int(index)] for index in indices)


def _section_distance_matrix(
    slide_paths: tuple[Path, ...],
    crops: dict[Path, _Crop],
    config: RegistrationConfig,
    *,
    physical_areas: dict[Path, float | None] | None = None,
) -> np.ndarray:
    """Return deterministic pairwise morphology distances for section ordering."""

    count = len(slide_paths)
    distances = np.ones((count, count), dtype=float)
    np.fill_diagonal(distances, 0.0)
    pairs = tuple(
        (first, second) for first in range(count) for second in range(first + 1, count)
    )

    def calculate(pair: tuple[int, int]) -> tuple[int, int, float]:
        first, second = pair
        try:
            transform, _ = _estimate_pair_transform(
                slide_paths[first],
                slide_paths[second],
                crops,
                config,
            )
            dice = _final_mask_dice(
                crops[slide_paths[first]],
                crops[slide_paths[second]],
                transform.matrix,
            )
            support = min(1.0, transform.inlier_count / 40.0)
            registration_distance = 1.0 - min(
                max(1e-3, 0.75 * dice + 0.25 * support), 1.0
            )
            shape_distance = _mask_shape_distance(
                crops[slide_paths[first]].mask,
                crops[slide_paths[second]].mask,
            )
            hole_distance = _mask_hole_topology_distance(
                crops[slide_paths[first]].mask,
                crops[slide_paths[second]].mask,
            )
            area_distance = _physical_area_distance(
                slide_paths[first],
                slide_paths[second],
                physical_areas,
            )
            distance = (
                0.60 * registration_distance
                + 0.15 * area_distance
                + 0.10 * shape_distance
                + 0.15 * hole_distance
            )
        except (ValueError, np.linalg.LinAlgError):
            distance = 1.0
        return first, second, float(distance)

    if config.ordering_workers == 1:
        results = map(calculate, pairs)
        executor = None
    else:
        executor = ThreadPoolExecutor(max_workers=config.ordering_workers)
        results = executor.map(calculate, pairs)
    try:
        for first, second, distance in results:
            distances[first, second] = distance
            distances[second, first] = distance
    finally:
        if executor is not None:
            executor.shutdown()
    return distances


def _physical_mask_area(mask: np.ndarray, geometry: SlideGeometry) -> float | None:
    """Return thumbnail-mask area in square micrometres when calibrated."""

    if geometry.mpp_xy is None:
        return None
    linear = geometry.thumbnail_to_physical[:2, :2]
    return float(np.count_nonzero(mask) * abs(np.linalg.det(linear)))


def _ordering_input_fingerprint(
    mask: np.ndarray,
    geometry: SlideGeometry,
    quarter_turns_ccw: int = 0,
) -> str:
    """Fingerprint the accepted mask and physical geometry used for ordering."""

    import hashlib

    digest = hashlib.sha256()
    digest.update(b"histopia-ordering-input-v2")
    digest.update(bytes([quarter_turns_ccw % 4]))
    digest.update(np.ascontiguousarray(mask, dtype=np.uint8).tobytes())
    digest.update(json.dumps(geometry.to_json_dict(), sort_keys=True).encode())
    return digest.hexdigest()


def _ordering_distance_settings(config: RegistrationConfig) -> dict[str, object]:
    """Return only settings that affect pairwise morphology distances."""

    return {
        "max_processed_image_dim_px": config.max_processed_image_dim_px,
        "rigid_method": config.rigid_method,
        "refinement": asdict(config.refinement),
        "weights": {
            "registration": 0.60,
            "physical_area": 0.15,
            "shape": 0.10,
            "hole_topology": 0.15,
        },
    }


def _transform_from_oriented_coordinates(
    transform: RigidTransformResult,
    moving_shape: tuple[int, int],
    moving_turns: int,
    reference_shape: tuple[int, int],
    reference_turns: int,
) -> RigidTransformResult:
    """Convert an oriented working transform back to source slide coordinates."""

    moving_orientation = quarter_turn_matrix(moving_shape, moving_turns)
    reference_orientation = quarter_turn_matrix(reference_shape, reference_turns)
    matrix = (
        np.linalg.inv(reference_orientation) @ transform.matrix @ moving_orientation
    )
    method = transform.method
    if moving_turns or reference_turns:
        method = f"oriented:{method}"
    return RigidTransformResult(
        matrix=matrix,
        method=method,
        match_count=transform.match_count,
        inlier_count=transform.inlier_count,
        warnings=list(transform.warnings),
    )


def _load_automatic_mask_snapshot(
    manifest_path: Path,
    slide_paths: tuple[Path, ...],
    thumbnails: dict[Path, np.ndarray],
) -> dict[Path, TissueMaskResult]:
    """Load hash-verified automatic masks previously accepted by a reviewer."""

    from PIL import Image

    payload = json.loads(manifest_path.read_text())
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported automatic mask snapshot schema")
    rows = payload.get("slides")
    if not isinstance(rows, list):
        raise ValueError("automatic mask snapshot must contain a slides list")
    by_name = {str(row["slide"]): row for row in rows}
    expected_names = {path.name for path in slide_paths}
    if set(by_name) != expected_names:
        missing = sorted(expected_names - set(by_name))
        extra = sorted(set(by_name) - expected_names)
        raise ValueError(
            "automatic mask snapshot must exactly match input slides "
            f"(missing={missing}, extra={extra})"
        )
    results: dict[Path, TissueMaskResult] = {}
    for path in slide_paths:
        row = by_name[path.name]
        mask_path = manifest_path.parent / str(row["mask"])
        encoded = mask_path.read_bytes()
        observed_hash = hashlib.sha256(encoded).hexdigest()
        if observed_hash != row.get("sha256"):
            raise ValueError(f"automatic mask snapshot hash mismatch: {path.name}")
        with Image.open(mask_path) as image:
            mask = np.asarray(image.convert("L")) > 127
        expected_shape = thumbnails[path].shape[:2]
        if mask.shape != expected_shape:
            raise ValueError(
                f"automatic mask snapshot shape mismatch for {path.name}: "
                f"{mask.shape} != {expected_shape}"
            )
        metrics, warnings = evaluate_tissue_mask(mask)
        results[path] = TissueMaskResult(
            mask=mask,
            method="approved_automatic_snapshot",
            metrics=metrics,
            accepted=not warnings,
            warnings=warnings,
        )
    return results


def _thumbnail_physical_pixel_area(geometry: SlideGeometry) -> float | None:
    if geometry.mpp_xy is None:
        return None
    return float(abs(np.linalg.det(geometry.thumbnail_to_physical[:2, :2])))


def _physical_area_distance(
    first: Path,
    second: Path,
    areas: dict[Path, float | None] | None,
) -> float:
    if areas is None or areas.get(first) is None or areas.get(second) is None:
        return 0.0
    first_area = float(areas[first])
    second_area = float(areas[second])
    if first_area <= 0 or second_area <= 0:
        return 1.0
    return min(1.0, abs(float(np.log(first_area / second_area))))


def _mask_shape_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Compare scale-independent extent and connected-component topology."""

    from scipy import ndimage as ndi

    def descriptor(mask: np.ndarray) -> tuple[float, float]:
        binary = np.asarray(mask, dtype=bool)
        rows, cols = np.nonzero(binary)
        if not rows.size:
            return 0.0, 0.0
        height = rows.max() - rows.min() + 1
        width = cols.max() - cols.min() + 1
        aspect = float(np.log(max(width, 1) / max(height, 1)))
        _, components = ndi.label(binary)
        return aspect, float(np.log1p(components))

    first_aspect, first_components = descriptor(first)
    second_aspect, second_components = descriptor(second)
    aspect_distance = min(1.0, abs(first_aspect - second_aspect))
    topology_distance = min(1.0, abs(first_components - second_components) / 2.0)
    return 0.7 * aspect_distance + 0.3 * topology_distance


def _mask_hole_topology_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Compare substantial internal cavities independently of outer shape."""

    first_fraction = _largest_internal_cavity_fraction(first)
    second_fraction = _largest_internal_cavity_fraction(second)
    first_has_hole = first_fraction >= 0.015
    second_has_hole = second_fraction >= 0.015
    if first_has_hole != second_has_hole:
        return 1.0
    if not first_has_hole:
        return 0.0
    return min(1.0, abs(first_fraction - second_fraction) / 0.10)


def _largest_internal_cavity_fraction(mask: np.ndarray) -> float:
    """Return the largest enclosed background component relative to filled tissue."""

    from scipy import ndimage as ndi

    binary = np.asarray(mask, dtype=bool)
    filled = ndi.binary_fill_holes(binary)
    filled_area = int(np.count_nonzero(filled))
    if filled_area == 0:
        return 0.0
    labels, count = ndi.label(filled & ~binary)
    if count == 0:
        return 0.0
    sizes = np.bincount(labels.ravel())
    return float(sizes[1:].max(initial=0) / filled_area)


def _read_fixed_positions(
    slide_paths: tuple[Path, ...], order_path: Path | None
) -> dict[str, int]:
    """Read positive manifest orders as hard one-based sequence positions."""

    if order_path is None:
        return {}
    if not order_path.exists():
        raise FileNotFoundError(f"section order manifest not found: {order_path}")
    if order_path.suffix.lower() == ".json":
        payload = json.loads(order_path.read_text())
        rows = payload.get("slides", payload)
        if isinstance(rows, dict):
            raw = rows.items()
        else:
            raw = (
                (row.get("slide", row.get("filename")), row.get("order"))
                for row in rows
            )
    else:
        with order_path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        raw = (
            (
                row.get("slide") or row.get("filename") or row.get("raw_name"),
                row.get("order"),
            )
            for row in rows
        )
    path_by_name = {path.name: path.name for path in slide_paths}
    path_by_stem = {path.stem: path.name for path in slide_paths}
    fixed: dict[str, int] = {}
    unresolved: list[str] = []
    for name, order in raw:
        if name in {None, ""} or order in {None, "", "0", 0}:
            continue
        requested = str(name)
        resolved = path_by_name.get(requested) or path_by_stem.get(Path(requested).stem)
        if resolved is None:
            unresolved.append(requested)
            continue
        fixed[resolved] = int(order)
    if unresolved:
        raise ValueError(
            "section order contains anchors that do not match discovered slides: "
            + ", ".join(sorted(unresolved))
        )
    return fixed


def _select_reference(
    slide_paths: tuple[Path, ...],
    reference_slide: str | None,
) -> Path:
    if reference_slide is None:
        msg = "reference_policy='explicit' requires reference_slide"
        raise ValueError(msg)
    for path in slide_paths:
        if path.name == reference_slide or path.stem == reference_slide:
            return path
    msg = f"reference slide {reference_slide!r} was not found"
    raise FileNotFoundError(msg)
