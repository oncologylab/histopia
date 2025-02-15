"""Full-resolution affine warping with lazy libvips evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from histopia.registration._errors import OptionalDependencyError


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

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "output_path": str(self.output_path),
            "matrix": self.matrix.tolist(),
            "source_shape": list(self.source_shape),
            "reference_shape": list(self.reference_shape),
            "non_rigid_applied": self.non_rigid_applied,
            "output_shape": list(self.output_shape or self.reference_shape),
            "reference_offset_xy": list(self.reference_offset_xy),
        }


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
    compression: str = "jpeg",
    jpeg_quality: int = 95,
    tile_size: int = 512,
    pyramid: bool = True,
    reference_to_rigid_moving_displacement: np.ndarray | None = None,
    reference_thumbnail_bbox: tuple[int, int, int, int] | None = None,
) -> WsiWarpResult:
    """Warp one full-resolution slide into the reference slide canvas."""

    if pyramid and compression != "jpeg":
        msg = "pyramidal WSI output currently requires validated JPEG compression"
        raise ValueError(msg)
    pyvips = _import_pyvips()
    moving = _as_rgb_uchar(_load_slide(Path(moving_path)))
    reference = _load_slide(Path(reference_path))
    moving_shape = (moving.height, moving.width)
    reference_shape = (reference.height, reference.width)
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
        reference_scale_x = reference_thumbnail_shape[1] / reference.width
        reference_scale_y = reference_thumbnail_shape[0] / reference.height
        flow_index = (x * reference_scale_x).bandjoin(y * reference_scale_y)
        full_flow = flow_image.mapim(
            flow_index,
            interpolate=pyvips.Interpolate.new("bilinear"),
            extend="copy",
        )
        target_x = x + full_flow[0] / reference_scale_x
        target_y = y + full_flow[1] / reference_scale_y
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

    output_path = Path(output_path)
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

    run_dir = Path(run_dir)
    payload = json.loads((run_dir / "registration_result.json").read_text())
    reference_path = Path(payload["reference_slide"])
    output_dir = Path(output_dir) if output_dir else run_dir / "registered"
    processed_dir = run_dir / "processed"
    reference_thumbnail_path = processed_dir / f"{reference_path.stem}.thumbnail.png"
    if not reference_thumbnail_path.exists():
        msg = f"saved reference thumbnail not found: {reference_thumbnail_path}"
        raise FileNotFoundError(msg)
    reference_thumbnail = load_thumbnail(reference_thumbnail_path, 1_000_000)
    slide_thumbnails: dict[Path, np.ndarray] = {}
    for slide in payload["slides"]:
        slide_path = Path(slide["path"])
        thumbnail_path = processed_dir / f"{slide_path.stem}.thumbnail.png"
        if not thumbnail_path.exists():
            msg = f"saved moving thumbnail not found: {thumbnail_path}"
            raise FileNotFoundError(msg)
        slide_thumbnails[slide_path] = load_thumbnail(thumbnail_path, 1_000_000)
    reference_thumbnail_bbox = None
    if crop_mode == "overlap":
        reference_thumbnail_bbox = calculate_thumbnail_overlap_bbox(
            [
                (
                    slide_thumbnails[Path(slide["path"])].shape[:2],
                    np.asarray(slide["transform"]["matrix"], dtype=float),
                )
                for slide in payload["slides"]
            ],
            reference_thumbnail.shape[:2],
        )
    elif crop_mode != "reference":
        msg = "warp-run crop_mode must be 'reference' or 'overlap'"
        raise ValueError(msg)

    selected_slides = payload["slides"]
    if accepted_non_rigid_only:
        selected_slides = [
            slide
            for slide in selected_slides
            if slide.get("non_rigid_transform")
            and slide["non_rigid_transform"].get("accepted")
        ]
    results: list[WsiWarpResult] = []
    summary_path = output_dir / "full_resolution_warps.json"
    prior_by_output: dict[str, dict[str, Any]] = {}
    if summary_path.exists():
        prior_by_output = {
            str(Path(item["output_path"])): item
            for item in json.loads(summary_path.read_text())
        }
    if not overwrite:
        reference_full_shape = read_slide_shape(reference_path)
        requested_bbox = _thumbnail_bbox_to_full_resolution(
            reference_thumbnail_bbox,
            reference_thumbnail.shape[:2],
            reference_full_shape,
        )
        _, _, requested_width, requested_height = requested_bbox
        for slide in selected_slides:
            slide_path = Path(slide["path"])
            output_path = output_dir / f"{slide_path.stem}.registered.tiff"
            if not output_path.exists():
                continue
            prior = prior_by_output.get(str(output_path))
            if prior is None:
                msg = f"existing output has no provenance record: {output_path}"
                raise ValueError(msg)
            non_rigid_payload = slide.get("non_rigid_transform")
            requested_non_rigid = bool(
                non_rigid_payload and non_rigid_payload.get("accepted")
            )
            if bool(prior.get("non_rigid_applied", False)) != requested_non_rigid:
                msg = (
                    f"existing output has different non-rigid provenance: "
                    f"{output_path}; rerun with --overwrite"
                )
                raise ValueError(msg)
            existing = _load_slide(output_path)
            if (existing.width, existing.height) != (
                requested_width,
                requested_height,
            ):
                msg = (
                    f"existing output has different crop: {output_path}; "
                    "rerun with --overwrite"
                )
                raise ValueError(msg)
    for slide in selected_slides:
        moving_path = Path(slide["path"])
        moving_thumbnail = slide_thumbnails[moving_path]
        output_path = output_dir / f"{moving_path.stem}.registered.tiff"
        non_rigid_payload = slide.get("non_rigid_transform")
        non_rigid_accepted = bool(
            non_rigid_payload and non_rigid_payload.get("accepted")
        )
        if output_path.exists() and not overwrite:
            moving_shape = read_slide_shape(moving_path)
            reference_shape = read_slide_shape(reference_path)
            full_matrix = thumbnail_to_full_resolution_matrix(
                np.asarray(slide["transform"]["matrix"], dtype=float),
                moving_thumbnail_shape=moving_thumbnail.shape[:2],
                moving_full_shape=moving_shape,
                reference_thumbnail_shape=reference_thumbnail.shape[:2],
                reference_full_shape=reference_shape,
            )
            reference_bbox = _thumbnail_bbox_to_full_resolution(
                reference_thumbnail_bbox,
                reference_thumbnail.shape[:2],
                reference_shape,
            )
            offset_x, offset_y, expected_width, expected_height = reference_bbox
            existing = _load_slide(output_path)
            if (existing.width, existing.height) != (
                expected_width,
                expected_height,
            ):
                msg = (
                    f"existing output has wrong crop for {output_path}; "
                    "rerun with --overwrite"
                )
                raise ValueError(msg)
            crop_translation = np.eye(3, dtype=float)
            crop_translation[:2, 2] = [-offset_x, -offset_y]
            result = WsiWarpResult(
                output_path,
                crop_translation @ full_matrix,
                moving_shape,
                reference_shape,
                non_rigid_accepted,
                (expected_height, expected_width),
                (offset_x, offset_y),
            )
        else:
            displacement = None
            if non_rigid_accepted:
                displacement_path = non_rigid_payload.get("displacement_path")
                if displacement_path:
                    displacement_path = Path(displacement_path)
                    if not displacement_path.is_absolute():
                        displacement_path = run_dir / displacement_path
                    displacement = np.load(
                        displacement_path,
                        allow_pickle=False,
                    )["displacement"]
            result = warp_slide_to_reference(
                moving_path,
                reference_path,
                output_path,
                np.asarray(slide["transform"]["matrix"], dtype=float),
                moving_thumbnail_shape=moving_thumbnail.shape[:2],
                reference_thumbnail_shape=reference_thumbnail.shape[:2],
                compression=compression,
                jpeg_quality=jpeg_quality,
                tile_size=tile_size,
                reference_to_rigid_moving_displacement=displacement,
                reference_thumbnail_bbox=reference_thumbnail_bbox,
            )
        results.append(result)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps([item.to_json_dict() for item in results], indent=2) + "\n"
        )
    return tuple(results)


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
    if image.bands > 3:
        image = image[:3]
    elif image.bands == 1:
        image = image.bandjoin([image, image])
    if image.format != "uchar":
        image = image.cast("uchar")
    return image
