"""Full-resolution affine warping with lazy libvips evaluation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from histopia._atomic import write_json_atomic
from histopia._validation import positive_int, require_bool
from histopia._vips_image import normalize_vips_rgb_uchar
from histopia.registration._errors import OptionalDependencyError
from histopia.registration._slides import SlideGeometry


@dataclass(slots=True)
class WsiWarpResult:
    """Metadata for one full-resolution slide warp."""

    output_path: Path
    matrix: np.ndarray
    source_shape: tuple[int, int]
    reference_shape: tuple[int, int]
    non_rigid_applied: bool = False
    output_shape: tuple[int, int] | None = None
    reference_offset_xy: tuple[int, int] = (0, 0)
    provenance: dict[str, Any] | None = None

    def to_json_dict(self) -> dict[str, Any]:
        payload = {
            "output_path": str(self.output_path),
            "matrix": self.matrix.tolist(),
            "source_shape": list(self.source_shape),
            "reference_shape": list(self.reference_shape),
            "non_rigid_applied": self.non_rigid_applied,
            "output_shape": list(self.output_shape or self.reference_shape),
            "reference_offset_xy": list(self.reference_offset_xy),
        }
        if self.provenance is not None:
            payload["provenance"] = self.provenance
        return payload


@dataclass(frozen=True, slots=True)
class _WsiExportPlan:
    slide: dict[str, Any]
    moving_path: Path
    output_path: Path
    moving_thumbnail_shape: tuple[int, int]
    moving_shape: tuple[int, int]
    reference_shape: tuple[int, int]
    moving_geometry: SlideGeometry | None
    reference_geometry: SlideGeometry | None
    reference_bbox: tuple[int, int, int, int]
    full_matrix: np.ndarray
    non_rigid_accepted: bool
    displacement_path: Path | None
    request: dict[str, Any]
    fingerprint: str


def thumbnail_to_full_resolution_matrix(
    thumbnail_matrix: np.ndarray,
    *,
    moving_thumbnail_shape: tuple[int, int],
    moving_full_shape: tuple[int, int],
    reference_thumbnail_shape: tuple[int, int],
    reference_full_shape: tuple[int, int],
) -> np.ndarray:
    """Convert a moving-to-reference matrix from thumbnail to full coordinates."""

    moving_scale = _full_to_thumbnail_scale(
        moving_thumbnail_shape,
        moving_full_shape,
    )
    reference_scale = _full_to_thumbnail_scale(
        reference_thumbnail_shape,
        reference_full_shape,
    )
    matrix = np.asarray(thumbnail_matrix, dtype=float)
    if matrix.shape != (3, 3):
        msg = "thumbnail_matrix must have shape (3, 3)"
        raise ValueError(msg)
    return np.linalg.inv(reference_scale) @ matrix @ moving_scale


def geometry_thumbnail_to_native_matrix(
    thumbnail_matrix: np.ndarray,
    *,
    moving_geometry: SlideGeometry,
    reference_geometry: SlideGeometry,
) -> np.ndarray:
    """Convert a thumbnail transform using explicit scanner content bounds."""

    matrix = np.asarray(thumbnail_matrix, dtype=float)
    if matrix.shape != (3, 3):
        msg = "thumbnail_matrix must have shape (3, 3)"
        raise ValueError(msg)
    return (
        reference_geometry.thumbnail_to_native
        @ matrix
        @ moving_geometry.native_to_thumbnail
    )


def read_slide_shape(path: Path | str) -> tuple[int, int]:
    """Return auto-oriented full-resolution slide shape as ``(height, width)``."""

    image = _load_slide(Path(path))
    return image.height, image.width


def warp_slide_to_reference(
    moving_path: Path | str,
    reference_path: Path | str,
    output_path: Path | str,
    thumbnail_matrix: np.ndarray,
    *,
    moving_thumbnail_shape: tuple[int, int],
    reference_thumbnail_shape: tuple[int, int],
    moving_geometry: SlideGeometry | None = None,
    reference_geometry: SlideGeometry | None = None,
    compression: str = "jpeg",
    jpeg_quality: int = 95,
    tile_size: int = 512,
    pyramid: bool = True,
    reference_to_rigid_moving_displacement: np.ndarray | None = None,
    reference_thumbnail_bbox: tuple[int, int, int, int] | None = None,
) -> WsiWarpResult:
    """Warp one full-resolution slide into the reference slide canvas."""

    compression, jpeg_quality, tile_size, pyramid = _validate_writer_settings(
        compression,
        jpeg_quality,
        tile_size,
        pyramid,
    )
    moving_path = Path(moving_path)
    reference_path = Path(reference_path)
    output_path = Path(output_path)
    output_resolved = output_path.expanduser().resolve()
    if output_resolved in {
        moving_path.expanduser().resolve(),
        reference_path.expanduser().resolve(),
    }:
        raise ValueError("full-resolution output must not replace a source slide")
    moving_identity = _file_identity(moving_path)
    reference_identity = _file_identity(reference_path)
    pyvips = _import_pyvips()
    moving = _as_rgb_uchar(_load_slide(moving_path))
    reference = _load_slide(reference_path)
    moving_shape = (moving.height, moving.width)
    reference_shape = (reference.height, reference.width)
    if (moving_geometry is None) != (reference_geometry is None):
        raise ValueError("moving and reference geometry must be provided together")
    if moving_geometry is not None and reference_geometry is not None:
        if moving_geometry.native_shape != moving_shape:
            raise ValueError("moving slide shape differs from supplied geometry")
        if reference_geometry.native_shape != reference_shape:
            raise ValueError("reference slide shape differs from supplied geometry")
        if moving_geometry.thumbnail_shape != moving_thumbnail_shape:
            raise ValueError("moving thumbnail shape differs from supplied geometry")
        if reference_geometry.thumbnail_shape != reference_thumbnail_shape:
            raise ValueError("reference thumbnail shape differs from supplied geometry")
        full_matrix = geometry_thumbnail_to_native_matrix(
            thumbnail_matrix,
            moving_geometry=moving_geometry,
            reference_geometry=reference_geometry,
        )
        reference_bbox = _thumbnail_bbox_to_native(
            reference_thumbnail_bbox,
            reference_geometry,
        )
    else:
        full_matrix = thumbnail_to_full_resolution_matrix(
            thumbnail_matrix,
            moving_thumbnail_shape=moving_thumbnail_shape,
            moving_full_shape=moving_shape,
            reference_thumbnail_shape=reference_thumbnail_shape,
            reference_full_shape=reference_shape,
        )
        reference_bbox = _thumbnail_bbox_to_full_resolution(
            reference_thumbnail_bbox,
            reference_thumbnail_shape,
            reference_shape,
        )
    offset_x, offset_y, output_width, output_height = reference_bbox
    crop_translation = np.eye(3, dtype=float)
    crop_translation[:2, 2] = [-offset_x, -offset_y]
    output_matrix = crop_translation @ full_matrix
    inverse = np.linalg.inv(full_matrix)
    coordinates = pyvips.Image.xyz(output_width, output_height)
    x = coordinates[0] + offset_x
    y = coordinates[1] + offset_y
    target_x = x
    target_y = y
    non_rigid_applied = reference_to_rigid_moving_displacement is not None
    if reference_to_rigid_moving_displacement is not None:
        displacement = np.asarray(
            reference_to_rigid_moving_displacement,
            dtype=np.float32,
        )
        if displacement.shape != (*reference_thumbnail_shape, 2):
            msg = "non-rigid displacement must match reference thumbnail shape"
            raise ValueError(msg)
        flow_memory = displacement.tobytes()
        flow_image = pyvips.Image.new_from_memory(
            flow_memory,
            displacement.shape[1],
            displacement.shape[0],
            2,
            "float",
        ).copy_memory()
        if reference_geometry is None:
            reference_scale_x = reference_thumbnail_shape[1] / reference.width
            reference_scale_y = reference_thumbnail_shape[0] / reference.height
            flow_x = x * reference_scale_x
            flow_y = y * reference_scale_y
            flow_to_native_x = 1.0 / reference_scale_x
            flow_to_native_y = 1.0 / reference_scale_y
            flow_extend = "copy"
        else:
            thumbnail_to_native = reference_geometry.thumbnail_to_native
            native_to_thumbnail = reference_geometry.native_to_thumbnail
            flow_x = x * float(native_to_thumbnail[0, 0]) + float(
                native_to_thumbnail[0, 2]
            )
            flow_y = y * float(native_to_thumbnail[1, 1]) + float(
                native_to_thumbnail[1, 2]
            )
            flow_to_native_x = float(thumbnail_to_native[0, 0])
            flow_to_native_y = float(thumbnail_to_native[1, 1])
            flow_extend = "background"
        flow_index = flow_x.bandjoin(flow_y)
        full_flow = flow_image.mapim(
            flow_index,
            interpolate=pyvips.Interpolate.new("bilinear"),
            extend=flow_extend,
            background=[0.0, 0.0],
        )
        target_x = x + full_flow[0] * flow_to_native_x
        target_y = y + full_flow[1] * flow_to_native_y
    source_x = (
        target_x * float(inverse[0, 0])
        + target_y * float(inverse[0, 1])
        + float(inverse[0, 2])
    )
    source_y = (
        target_x * float(inverse[1, 0])
        + target_y * float(inverse[1, 1])
        + float(inverse[1, 2])
    )
    index = source_x.bandjoin(source_y)
    warped = moving.mapim(
        index,
        interpolate=pyvips.Interpolate.new("bilinear"),
        extend="background",
        background=[255.0, 255.0, 255.0],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tile_width = _valid_tiff_tile_size(tile_size, output_width)
    tile_height = _valid_tiff_tile_size(tile_size, output_height)
    save_options: dict[str, Any] = {
        "tile": True,
        "tile_width": tile_width,
        "tile_height": tile_height,
        "pyramid": pyramid,
        "bigtiff": True,
        "compression": compression,
        "xres": reference.xres,
        "yres": reference.yres,
    }
    if compression == "jpeg":
        save_options["Q"] = jpeg_quality
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    temporary_path.unlink(missing_ok=True)
    try:
        warped.tiffsave(str(temporary_path), **save_options)
        _require_unchanged_input(moving_path, moving_identity, "moving source slide")
        _require_unchanged_input(
            reference_path,
            reference_identity,
            "reference source slide",
        )
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return WsiWarpResult(
        output_path=output_path,
        matrix=output_matrix,
        source_shape=moving_shape,
        reference_shape=reference_shape,
        non_rigid_applied=non_rigid_applied,
        output_shape=(output_height, output_width),
        reference_offset_xy=(offset_x, offset_y),
    )


def warp_saved_registration(
    run_dir: Path | str,
    output_dir: Path | str | None = None,
    *,
    compression: str = "jpeg",
    jpeg_quality: int = 95,
    tile_size: int = 512,
    overwrite: bool = False,
    crop_mode: str = "reference",
    accepted_non_rigid_only: bool = False,
) -> tuple[WsiWarpResult, ...]:
    """Apply all transforms from an existing registration run to source slides."""

    from histopia.registration._io import load_thumbnail

    compression, jpeg_quality, tile_size, _ = _validate_writer_settings(
        compression,
        jpeg_quality,
        tile_size,
        True,
    )
    require_bool("overwrite", overwrite)
    require_bool("accepted_non_rigid_only", accepted_non_rigid_only)
    run_dir = Path(run_dir)
    result_path = run_dir / "registration_result.json"
    result_bytes = result_path.read_bytes()
    payload = json.loads(result_bytes)
    if not isinstance(payload, dict):
        raise ValueError("registration result JSON root must be an object")
    registration_result_sha256 = hashlib.sha256(result_bytes).hexdigest()
    reference_path = Path(payload["reference_slide"])
    output_dir = Path(output_dir) if output_dir else run_dir / "registered"
    processed_dir = run_dir / "processed"
    reference_thumbnail_path = processed_dir / f"{reference_path.stem}.thumbnail.png"
    if not reference_thumbnail_path.exists():
        msg = f"saved reference thumbnail not found: {reference_thumbnail_path}"
        raise FileNotFoundError(msg)
    reference_thumbnail = load_thumbnail(reference_thumbnail_path, 1_000_000)
    slides = payload.get("slides")
    if not isinstance(slides, list) or not slides:
        raise ValueError("registration result contains no slides")
    if any(not isinstance(slide, dict) for slide in slides):
        raise ValueError("registration result slides must contain objects")
    slide_thumbnails: dict[Path, np.ndarray] = {}
    slide_geometries: dict[Path, SlideGeometry | None] = {}
    for slide in slides:
        slide_path = Path(slide["path"])
        if slide_path in slide_thumbnails:
            raise ValueError(
                f"registration result contains duplicate slide: {slide_path}"
            )
        thumbnail_path = processed_dir / f"{slide_path.stem}.thumbnail.png"
        if not thumbnail_path.exists():
            msg = f"saved moving thumbnail not found: {thumbnail_path}"
            raise FileNotFoundError(msg)
        thumbnail = load_thumbnail(thumbnail_path, 1_000_000)
        slide_thumbnails[slide_path] = thumbnail
        slide_geometries[slide_path] = _geometry_from_result(
            slide.get("geometry"),
            thumbnail.shape[:2],
            slide_path,
        )
    if reference_path not in slide_thumbnails:
        raise ValueError("reference slide is not present in registration slides")
    geometry_presence = {geometry is not None for geometry in slide_geometries.values()}
    if len(geometry_presence) > 1:
        raise ValueError("registration result has incomplete slide geometry")
    reference_geometry = slide_geometries[reference_path]
    reference_thumbnail_bbox = None
    if crop_mode == "overlap":
        reference_thumbnail_bbox = calculate_thumbnail_overlap_bbox(
            [
                (
                    slide_thumbnails[Path(slide["path"])].shape[:2],
                    np.asarray(slide["transform"]["matrix"], dtype=float),
                )
                for slide in slides
            ],
            reference_thumbnail.shape[:2],
        )
    elif crop_mode != "reference":
        msg = "warp-run crop_mode must be 'reference' or 'overlap'"
        raise ValueError(msg)

    selected_slides = slides
    if accepted_non_rigid_only:
        selected_slides = [
            slide
            for slide in selected_slides
            if isinstance(slide.get("non_rigid_transform"), dict)
            and slide["non_rigid_transform"].get("accepted") is True
        ]
    summary_path = output_dir / "full_resolution_warps.json"
    try:
        prior_by_output = _load_warp_summary(summary_path)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        if not overwrite:
            raise
        prior_by_output = {}
    reference_shape = read_slide_shape(reference_path)
    if (
        reference_geometry is not None
        and reference_geometry.native_shape != reference_shape
    ):
        raise ValueError("reference slide shape differs from saved geometry")
    reference_bbox = (
        _thumbnail_bbox_to_native(reference_thumbnail_bbox, reference_geometry)
        if reference_geometry is not None
        else _thumbnail_bbox_to_full_resolution(
            reference_thumbnail_bbox,
            reference_thumbnail.shape[:2],
            reference_shape,
        )
    )
    reference_identity = _file_identity(reference_path)
    plans = [
        _create_export_plan(
            run_dir=run_dir,
            output_dir=output_dir,
            slide=slide,
            slide_thumbnail=slide_thumbnails[Path(slide["path"])],
            moving_geometry=slide_geometries[Path(slide["path"])],
            reference_thumbnail_shape=reference_thumbnail.shape[:2],
            reference_geometry=reference_geometry,
            reference_shape=reference_shape,
            reference_bbox=reference_bbox,
            reference_identity=reference_identity,
            registration_result_sha256=registration_result_sha256,
            crop_mode=crop_mode,
            compression=compression,
            jpeg_quality=jpeg_quality,
            tile_size=tile_size,
        )
        for slide in selected_slides
    ]
    plan_keys = [_path_key(plan.output_path) for plan in plans]
    if len(set(plan_keys)) != len(plan_keys):
        raise ValueError("registration slide stems produce duplicate output paths")
    if not overwrite:
        for plan in plans:
            if plan.output_path.exists():
                _validate_resumable_output(
                    plan,
                    prior_by_output.get(_path_key(plan.output_path)),
                )

    results: list[WsiWarpResult] = []
    summary_by_output = dict(prior_by_output)
    for plan in plans:
        offset_x, offset_y, expected_width, expected_height = plan.reference_bbox
        if plan.output_path.exists() and not overwrite:
            crop_translation = np.eye(3, dtype=float)
            crop_translation[:2, 2] = [-offset_x, -offset_y]
            result = WsiWarpResult(
                plan.output_path,
                crop_translation @ plan.full_matrix,
                plan.moving_shape,
                plan.reference_shape,
                plan.non_rigid_accepted,
                (expected_height, expected_width),
                (offset_x, offset_y),
            )
        else:
            displacement = None
            if plan.displacement_path is not None:
                with np.load(
                    plan.displacement_path,
                    allow_pickle=False,
                ) as data:
                    displacement = np.asarray(
                        data["displacement"],
                        dtype=np.float32,
                    )
            pending_path = plan.output_path.with_name(
                f".{plan.output_path.name}.pending"
            )
            pending_path.unlink(missing_ok=True)
            try:
                result = warp_slide_to_reference(
                    plan.moving_path,
                    reference_path,
                    pending_path,
                    np.asarray(plan.slide["transform"]["matrix"], dtype=float),
                    moving_thumbnail_shape=plan.moving_thumbnail_shape,
                    reference_thumbnail_shape=reference_thumbnail.shape[:2],
                    moving_geometry=plan.moving_geometry,
                    reference_geometry=plan.reference_geometry,
                    compression=compression,
                    jpeg_quality=jpeg_quality,
                    tile_size=tile_size,
                    reference_to_rigid_moving_displacement=displacement,
                    reference_thumbnail_bbox=reference_thumbnail_bbox,
                )
                _validate_export_inputs(
                    plan,
                    reference_path,
                    result_path,
                    registration_result_sha256,
                )
                pending_path.replace(plan.output_path)
                result.output_path = plan.output_path
            finally:
                pending_path.unlink(missing_ok=True)
        _validate_export_inputs(
            plan,
            reference_path,
            result_path,
            registration_result_sha256,
        )
        result.provenance = {
            "schema_version": 1,
            "export_fingerprint": plan.fingerprint,
            "registration_result_sha256": registration_result_sha256,
            "request": plan.request,
            "output": _file_identity(plan.output_path),
        }
        results.append(result)
        summary_by_output[_path_key(plan.output_path)] = result.to_json_dict()
        write_json_atomic(
            summary_path,
            _ordered_summary_rows(summary_by_output, plans),
        )
    return tuple(results)


def _validate_export_inputs(
    plan: _WsiExportPlan,
    reference_path: Path,
    result_path: Path,
    registration_result_sha256: str,
) -> None:
    _require_unchanged_input(
        plan.moving_path,
        plan.request["source"],
        "source slide",
    )
    _require_unchanged_input(
        reference_path,
        plan.request["reference"],
        "reference slide",
    )
    if plan.displacement_path is not None:
        current_sha256 = _sha256_file(plan.displacement_path)
        if current_sha256 != plan.request["non_rigid_displacement_sha256"]:
            raise ValueError(
                f"non-rigid displacement changed during export: "
                f"{plan.displacement_path}"
            )
    if hashlib.sha256(result_path.read_bytes()).hexdigest() != (
        registration_result_sha256
    ):
        raise ValueError("registration result changed during full-resolution export")


def _create_export_plan(
    *,
    run_dir: Path,
    output_dir: Path,
    slide: dict[str, Any],
    slide_thumbnail: np.ndarray,
    moving_geometry: SlideGeometry | None,
    reference_thumbnail_shape: tuple[int, int],
    reference_geometry: SlideGeometry | None,
    reference_shape: tuple[int, int],
    reference_bbox: tuple[int, int, int, int],
    reference_identity: dict[str, Any],
    registration_result_sha256: str,
    crop_mode: str,
    compression: str,
    jpeg_quality: int,
    tile_size: int,
) -> _WsiExportPlan:
    moving_path = Path(slide["path"])
    moving_shape = read_slide_shape(moving_path)
    if moving_geometry is not None and moving_geometry.native_shape != moving_shape:
        raise ValueError(f"slide shape differs from saved geometry: {moving_path}")
    matrix = np.asarray(slide.get("transform", {}).get("matrix"), dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"saved transform must be a finite 3x3 matrix: {moving_path}")
    if moving_geometry is not None and reference_geometry is not None:
        full_matrix = geometry_thumbnail_to_native_matrix(
            matrix,
            moving_geometry=moving_geometry,
            reference_geometry=reference_geometry,
        )
    else:
        full_matrix = thumbnail_to_full_resolution_matrix(
            matrix,
            moving_thumbnail_shape=slide_thumbnail.shape[:2],
            moving_full_shape=moving_shape,
            reference_thumbnail_shape=reference_thumbnail_shape,
            reference_full_shape=reference_shape,
        )
    if not np.all(np.isfinite(full_matrix)):
        raise ValueError(f"saved transform produces a non-finite matrix: {moving_path}")
    try:
        np.linalg.inv(full_matrix)
    except np.linalg.LinAlgError as error:
        raise ValueError(f"saved transform is singular: {moving_path}") from error

    non_rigid_payload = slide.get("non_rigid_transform")
    non_rigid_accepted = bool(
        isinstance(non_rigid_payload, dict)
        and non_rigid_payload.get("accepted") is True
    )
    displacement_path: Path | None = None
    displacement_sha256: str | None = None
    if non_rigid_accepted:
        displacement_value = non_rigid_payload.get("displacement_path")
        if not isinstance(displacement_value, str) or not displacement_value:
            raise ValueError(
                f"accepted non-rigid transform has no displacement path: {moving_path}"
            )
        displacement_path = Path(displacement_value)
        if not displacement_path.is_absolute():
            displacement_path = run_dir / displacement_path
        if not displacement_path.is_file():
            raise FileNotFoundError(
                f"accepted non-rigid displacement is missing: {displacement_path}"
            )
        displacement_sha256 = _sha256_file(displacement_path)

    request = {
        "schema_version": 1,
        "algorithm": "histopia-full-resolution-mapim-v2",
        "registration_result_sha256": registration_result_sha256,
        "source": _file_identity(moving_path),
        "reference": reference_identity,
        "thumbnail_transform": matrix.tolist(),
        "moving_thumbnail_shape": list(slide_thumbnail.shape[:2]),
        "reference_thumbnail_shape": list(reference_thumbnail_shape),
        "moving_geometry": (
            moving_geometry.to_json_dict() if moving_geometry is not None else None
        ),
        "reference_geometry": (
            reference_geometry.to_json_dict()
            if reference_geometry is not None
            else None
        ),
        "reference_bbox_xywh": list(reference_bbox),
        "crop_mode": crop_mode,
        "non_rigid_applied": non_rigid_accepted,
        "non_rigid_displacement_sha256": displacement_sha256,
        "writer": {
            "compression": compression,
            "jpeg_quality": jpeg_quality,
            "tile_size": tile_size,
            "pyramid": True,
        },
    }
    return _WsiExportPlan(
        slide=slide,
        moving_path=moving_path,
        output_path=output_dir / f"{moving_path.stem}.registered.tiff",
        moving_thumbnail_shape=slide_thumbnail.shape[:2],
        moving_shape=moving_shape,
        reference_shape=reference_shape,
        moving_geometry=moving_geometry,
        reference_geometry=reference_geometry,
        reference_bbox=reference_bbox,
        full_matrix=full_matrix,
        non_rigid_accepted=non_rigid_accepted,
        displacement_path=displacement_path,
        request=request,
        fingerprint=_sha256_json(request),
    )


def _geometry_from_result(
    payload: object,
    thumbnail_shape: tuple[int, int],
    slide_path: Path,
) -> SlideGeometry | None:
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ValueError(f"saved slide geometry must be an object: {slide_path}")
    try:
        native_shape = tuple(int(value) for value in payload["native_shape"])
        bbox = tuple(int(value) for value in payload["content_bbox_xywh"])
        saved_thumbnail_shape = tuple(
            int(value) for value in payload["thumbnail_shape"]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"saved slide geometry is invalid: {slide_path}") from error
    if len(native_shape) != 2 or min(native_shape) <= 0:
        raise ValueError(f"saved native slide shape is invalid: {slide_path}")
    if len(bbox) != 4:
        raise ValueError(f"saved slide content bounds are invalid: {slide_path}")
    x, y, width, height = bbox
    native_height, native_width = native_shape
    if (
        min(x, y) < 0
        or min(width, height) <= 0
        or x + width > native_width
        or y + height > native_height
    ):
        raise ValueError(f"saved slide content bounds are invalid: {slide_path}")
    if saved_thumbnail_shape != thumbnail_shape or min(thumbnail_shape) <= 0:
        raise ValueError(f"saved thumbnail shape does not match image: {slide_path}")
    bounds_source = payload.get("bounds_source")
    if not isinstance(bounds_source, str) or not bounds_source:
        raise ValueError(f"saved slide bounds source is invalid: {slide_path}")
    mpp_value = payload.get("mpp_xy")
    mpp: tuple[float, float] | None = None
    if mpp_value is not None:
        try:
            mpp = tuple(float(value) for value in mpp_value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"saved slide MPP is invalid: {slide_path}") from error
        if len(mpp) != 2 or not np.all(np.isfinite(mpp)) or min(mpp) <= 0:
            raise ValueError(f"saved slide MPP is invalid: {slide_path}")
    return SlideGeometry(
        native_shape=native_shape,
        content_bbox_xywh=bbox,
        thumbnail_shape=saved_thumbnail_shape,
        bounds_source=bounds_source,
        mpp_xy=mpp,
        mpp_source=str(payload.get("mpp_source", "unavailable")),
    )


def _load_warp_summary(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ValueError("full-resolution warp summary must contain a list")
    records: dict[str, dict[str, Any]] = {}
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("full-resolution warp summary rows must be objects")
        output_value = item.get("output_path")
        if not isinstance(output_value, str) or not output_value:
            raise ValueError("full-resolution warp summary output path is invalid")
        key = _path_key(Path(output_value))
        if key in records:
            raise ValueError("full-resolution warp summary has duplicate outputs")
        records[key] = item
    return records


def _validate_resumable_output(
    plan: _WsiExportPlan,
    prior: dict[str, Any] | None,
) -> None:
    if prior is None:
        raise ValueError(
            f"existing output has no provenance record: {plan.output_path}; "
            "rerun with --overwrite"
        )
    provenance = prior.get("provenance")
    if (
        not isinstance(provenance, dict)
        or provenance.get("schema_version") != 1
        or provenance.get("export_fingerprint") != plan.fingerprint
    ):
        raise ValueError(
            f"existing output provenance differs from this export request: "
            f"{plan.output_path}; rerun with --overwrite"
        )
    output_identity = provenance.get("output")
    if not isinstance(output_identity, dict):
        raise ValueError(
            f"existing output has incomplete file provenance: {plan.output_path}; "
            "rerun with --overwrite"
        )
    _require_unchanged_input(plan.output_path, output_identity, "registered output")
    existing = _load_slide(plan.output_path)
    _, _, expected_width, expected_height = plan.reference_bbox
    if (existing.width, existing.height) != (expected_width, expected_height):
        raise ValueError(
            f"existing output has a different canvas: {plan.output_path}; "
            "rerun with --overwrite"
        )


def _ordered_summary_rows(
    records: dict[str, dict[str, Any]],
    plans: list[_WsiExportPlan],
) -> list[dict[str, Any]]:
    preferred = [_path_key(plan.output_path) for plan in plans]
    remaining = sorted(set(records) - set(preferred))
    return [records[key] for key in [*preferred, *remaining] if key in records]


def _validate_writer_settings(
    compression: object,
    jpeg_quality: object,
    tile_size: object,
    pyramid: object,
) -> tuple[str, int, int, bool]:
    require_bool("pyramid", pyramid)
    if not isinstance(compression, str):
        raise TypeError("compression must be a string")
    if compression not in {"jpeg", "lzw", "deflate"}:
        raise ValueError("compression must be one of: jpeg, lzw, deflate")
    if pyramid and compression != "jpeg":
        raise ValueError(
            "pyramidal WSI output currently requires validated JPEG compression"
        )
    quality = positive_int("jpeg_quality", jpeg_quality)
    if quality > 100:
        raise ValueError("jpeg_quality must be at most 100")
    normalized_tile_size = positive_int("tile_size", tile_size)
    return compression, quality, normalized_tile_size, pyramid


def _file_identity(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _require_unchanged_input(
    path: Path,
    expected: dict[str, Any],
    label: str,
) -> None:
    if _file_identity(path) != expected:
        raise ValueError(f"{label} changed during or after export: {path}")


def _path_key(path: Path) -> str:
    return str(path.expanduser().resolve())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def calculate_thumbnail_overlap_bbox(
    slides: list[tuple[tuple[int, int], np.ndarray]],
    reference_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Return common valid canvas as ``(x, y, width, height)``."""

    try:
        import cv2
    except ImportError as exc:
        raise OptionalDependencyError(
            "opencv-contrib-python-headless",
            "registration",
        ) from exc
    reference_height, reference_width = reference_shape
    overlap = np.ones(reference_shape, dtype=bool)
    for moving_shape, matrix in slides:
        moving_valid = np.full(moving_shape, 255, dtype=np.uint8)
        warped = cv2.warpAffine(
            moving_valid,
            np.asarray(matrix, dtype=np.float32)[:2, :],
            (reference_width, reference_height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        overlap &= warped > 0
    if not overlap.any():
        msg = "registered slides have no common valid overlap"
        raise ValueError(msg)
    rows, cols = np.nonzero(overlap)
    x = int(cols.min())
    y = int(rows.min())
    return x, y, int(cols.max() - x + 1), int(rows.max() - y + 1)


def _full_to_thumbnail_scale(
    thumbnail_shape: tuple[int, int],
    full_shape: tuple[int, int],
) -> np.ndarray:
    thumbnail_height, thumbnail_width = thumbnail_shape
    full_height, full_width = full_shape
    if min(thumbnail_height, thumbnail_width, full_height, full_width) <= 0:
        msg = "image shapes must contain positive dimensions"
        raise ValueError(msg)
    return np.diag(
        [
            thumbnail_width / full_width,
            thumbnail_height / full_height,
            1.0,
        ]
    )


def _valid_tiff_tile_size(requested: int, dimension: int) -> int:
    maximum = max(16, (dimension // 16) * 16)
    return max(16, min(requested, maximum))


def _thumbnail_bbox_to_full_resolution(
    bbox: tuple[int, int, int, int] | None,
    thumbnail_shape: tuple[int, int],
    full_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    full_height, full_width = full_shape
    if bbox is None:
        return 0, 0, full_width, full_height
    x, y, width, height = bbox
    scale_x = full_width / thumbnail_shape[1]
    scale_y = full_height / thumbnail_shape[0]
    left = max(0, int(np.floor(x * scale_x)))
    top = max(0, int(np.floor(y * scale_y)))
    right = min(full_width, int(np.ceil((x + width) * scale_x)))
    bottom = min(full_height, int(np.ceil((y + height) * scale_y)))
    return left, top, right - left, bottom - top


def _thumbnail_bbox_to_native(
    bbox: tuple[int, int, int, int] | None,
    geometry: SlideGeometry,
) -> tuple[int, int, int, int]:
    native_height, native_width = geometry.native_shape
    if bbox is None:
        return 0, 0, native_width, native_height
    x, y, width, height = bbox
    matrix = geometry.thumbnail_to_native
    left = max(0, int(np.floor(matrix[0, 0] * x + matrix[0, 2])))
    top = max(0, int(np.floor(matrix[1, 1] * y + matrix[1, 2])))
    right = min(
        native_width,
        int(np.ceil(matrix[0, 0] * (x + width) + matrix[0, 2])),
    )
    bottom = min(
        native_height,
        int(np.ceil(matrix[1, 1] * (y + height) + matrix[1, 2])),
    )
    return left, top, right - left, bottom - top


def _import_pyvips() -> Any:
    try:
        import pyvips
    except ImportError as exc:
        raise OptionalDependencyError("pyvips", "wsi") from exc
    return pyvips


def _load_slide(path: Path) -> Any:
    pyvips = _import_pyvips()
    return pyvips.Image.new_from_file(str(path), access="sequential").autorot()


def _as_rgb_uchar(image: Any) -> Any:
    return normalize_vips_rgb_uchar(image)
