"""Whole-slide discovery and coordinate geometry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from histopia.registration._errors import OptionalDependencyError

STANDARD_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png"})
WSI_EXTENSIONS = frozenset({".ndpi", ".scn", ".svs", ".tif", ".tiff"})


@dataclass(frozen=True, slots=True)
class SlideGeometry:
    """Mapping between a content thumbnail and native slide coordinates."""

    native_shape: tuple[int, int]
    content_bbox_xywh: tuple[int, int, int, int]
    thumbnail_shape: tuple[int, int]
    bounds_source: str
    mpp_xy: tuple[float, float] | None = None
    mpp_source: str = "unavailable"

    @property
    def thumbnail_to_native(self) -> np.ndarray:
        """Return a homogeneous transform from thumbnail to native pixels."""

        x, y, width, height = self.content_bbox_xywh
        thumb_height, thumb_width = self.thumbnail_shape
        return np.array(
            [
                [width / thumb_width, 0.0, float(x)],
                [0.0, height / thumb_height, float(y)],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

    @property
    def native_to_thumbnail(self) -> np.ndarray:
        """Return a homogeneous transform from native to thumbnail pixels."""

        return np.linalg.inv(self.thumbnail_to_native)

    @property
    def native_to_physical(self) -> np.ndarray:
        """Return a native-pixel to micrometre transform.

        Raises when the source does not provide calibrated pixel spacing. This
        prevents a silent fallback to pixel units for mixed-scanner studies.
        """

        if self.mpp_xy is None:
            raise ValueError("physical pixel spacing is unavailable")
        mpp_x, mpp_y = self.mpp_xy
        return np.array(
            [[mpp_x, 0.0, 0.0], [0.0, mpp_y, 0.0], [0.0, 0.0, 1.0]],
            dtype=float,
        )

    @property
    def physical_to_native(self) -> np.ndarray:
        """Return a micrometre to native-pixel transform."""

        return np.linalg.inv(self.native_to_physical)

    @property
    def thumbnail_to_physical(self) -> np.ndarray:
        """Return a thumbnail-pixel to micrometre transform."""

        return self.native_to_physical @ self.thumbnail_to_native

    def to_json_dict(self) -> dict[str, object]:
        return {
            "native_shape": list(self.native_shape),
            "content_bbox_xywh": list(self.content_bbox_xywh),
            "thumbnail_shape": list(self.thumbnail_shape),
            "bounds_source": self.bounds_source,
            "mpp_xy": list(self.mpp_xy) if self.mpp_xy is not None else None,
            "mpp_source": self.mpp_source,
            "thumbnail_to_native": self.thumbnail_to_native.tolist(),
            "thumbnail_to_physical": (
                self.thumbnail_to_physical.tolist() if self.mpp_xy is not None else None
            ),
        }


@dataclass(frozen=True, slots=True)
class SlideRecord:
    """A registration source with verified format and geometry."""

    path: Path
    format: str
    geometry: SlideGeometry

    def to_json_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "format": self.format,
            "geometry": self.geometry.to_json_dict(),
        }


def discover_slides(
    input_dir: Path | str,
    *,
    wsi_only: bool = False,
) -> tuple[Path, ...]:
    """Return natural-order registration inputs, excluding derived artifacts."""

    input_dir = Path(input_dir)
    extensions = (
        WSI_EXTENSIONS if wsi_only else WSI_EXTENSIONS | STANDARD_IMAGE_EXTENSIONS
    )
    paths = [
        path
        for path in input_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in extensions
        and not _is_derived_or_label_image(path)
    ]
    return tuple(sorted(paths, key=_natural_key))


def validate_slide_selection(
    paths: tuple[Path | str, ...],
    *,
    wsi_only: bool = False,
) -> tuple[Path, ...]:
    """Validate an exact, ordered slide selection supplied by an external UI."""

    extensions = (
        WSI_EXTENSIONS if wsi_only else WSI_EXTENSIONS | STANDARD_IMAGE_EXTENSIONS
    )
    selected: list[Path] = []
    names: set[str] = set()
    resolved_paths: set[Path] = set()
    for value in paths:
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"selected registration slide not found: {path}")
        if path.suffix.lower() not in extensions:
            raise ValueError(f"unsupported selected registration slide: {path}")
        if _is_derived_or_label_image(path):
            raise ValueError(f"derived image cannot be a registration input: {path}")
        if path in resolved_paths:
            raise ValueError(f"duplicate selected registration slide: {path}")
        if path.name in names:
            raise ValueError(
                f"selected registration slide filenames must be unique: {path.name}"
            )
        selected.append(path)
        resolved_paths.add(path)
        names.add(path.name)
    return tuple(selected)


def load_slide_thumbnail(
    path: Path | str,
    max_dim_px: int,
) -> tuple[np.ndarray, SlideGeometry]:
    """Load an RGB content thumbnail and its native-coordinate geometry."""

    path = Path(path)
    if path.suffix.lower() in WSI_EXTENSIONS:
        return _load_wsi_thumbnail(path, max_dim_px)
    return _load_standard_thumbnail(path, max_dim_px)


def inspect_slide(path: Path | str, max_dim_px: int = 1024) -> SlideRecord:
    """Read enough slide metadata to verify format and geometry."""

    path = Path(path)
    _, geometry = load_slide_thumbnail(path, max_dim_px)
    return SlideRecord(
        path=path,
        format=path.suffix.lower().lstrip("."),
        geometry=geometry,
    )


def _load_wsi_thumbnail(
    path: Path,
    max_dim_px: int,
) -> tuple[np.ndarray, SlideGeometry]:
    pyvips = _import_pyvips()
    native = pyvips.Image.new_from_file(str(path), access="sequential")
    native_shape = (native.height, native.width)
    mpp_xy, mpp_source = _openslide_mpp(native)
    bbox = _openslide_content_bbox(native)
    if bbox is None:
        bbox = (0, 0, native.width, native.height)
        bounds_source = "full_slide"
    else:
        bounds_source = "openslide.bounds"

    if bounds_source == "openslide.bounds":
        thumbnail_source = f"{path}[autocrop]"
    else:
        thumbnail_source = str(path)
    thumbnail = pyvips.Image.thumbnail(
        thumbnail_source,
        max_dim_px,
        height=max_dim_px,
        no_rotate=True,
    )
    try:
        thumbnail = _vips_to_rgb(thumbnail)
    except pyvips.Error as primary_error:
        thumbnail = _retry_wsi_thumbnail_levels(
            pyvips,
            path,
            native,
            max_dim_px,
            content_bbox_xywh=bbox if bounds_source == "openslide.bounds" else None,
            primary_error=primary_error,
        )
    geometry = SlideGeometry(
        native_shape=native_shape,
        content_bbox_xywh=bbox,
        thumbnail_shape=thumbnail.shape[:2],
        bounds_source=bounds_source,
        mpp_xy=mpp_xy,
        mpp_source=mpp_source,
    )
    return thumbnail, geometry


def _retry_wsi_thumbnail_levels(
    pyvips: Any,
    path: Path,
    native: Any,
    max_dim_px: int,
    *,
    content_bbox_xywh: tuple[int, int, int, int] | None,
    primary_error: Exception,
) -> np.ndarray:
    """Decode explicit pyramid levels when automatic thumbnail selection fails."""

    for level in _suitable_pyramid_levels(
        native,
        max_dim_px,
        content_bbox_xywh=content_bbox_xywh,
    ):
        options: dict[str, object] = {"access": "sequential", "level": level}
        if content_bbox_xywh is not None:
            options["autocrop"] = True
        try:
            image = pyvips.Image.new_from_file(str(path), **options)
            image = image.thumbnail_image(
                max_dim_px,
                height=max_dim_px,
                size="down",
                no_rotate=True,
            )
            return _vips_to_rgb(image)
        except pyvips.Error:
            continue
    raise primary_error


def _suitable_pyramid_levels(
    native: Any,
    max_dim_px: int,
    *,
    content_bbox_xywh: tuple[int, int, int, int] | None = None,
) -> tuple[int, ...]:
    fields = set(native.get_fields())
    if "openslide.level-count" not in fields:
        return (0,)
    level_count = int(native.get("openslide.level-count"))
    suitable = []
    for level in range(level_count - 1, -1, -1):
        width_key = f"openslide.level[{level}].width"
        height_key = f"openslide.level[{level}].height"
        if width_key not in fields or height_key not in fields:
            continue
        width = int(native.get(width_key))
        height = int(native.get(height_key))
        if content_bbox_xywh is None:
            effective_max_dim = max(width, height)
        else:
            _, _, content_width, content_height = content_bbox_xywh
            effective_max_dim = max(
                content_width * width / native.width,
                content_height * height / native.height,
            )
        if effective_max_dim >= max_dim_px:
            suitable.append(level)
    return tuple(suitable) or (0,)


def _load_standard_thumbnail(
    path: Path,
    max_dim_px: int,
) -> tuple[np.ndarray, SlideGeometry]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise OptionalDependencyError("pillow", "wsi") from exc

    with Image.open(path) as image:
        image = image.convert("RGB")
        native_width, native_height = image.size
        image.thumbnail((max_dim_px, max_dim_px))
        thumbnail = np.asarray(image).copy()
    geometry = SlideGeometry(
        native_shape=(native_height, native_width),
        content_bbox_xywh=(0, 0, native_width, native_height),
        thumbnail_shape=thumbnail.shape[:2],
        bounds_source="full_image",
    )
    return thumbnail, geometry


def _openslide_content_bbox(image: Any) -> tuple[int, int, int, int] | None:
    fields = set(image.get_fields())
    keys = (
        "openslide.bounds-x",
        "openslide.bounds-y",
        "openslide.bounds-width",
        "openslide.bounds-height",
    )
    if not all(key in fields for key in keys):
        return None
    values = tuple(int(image.get(key)) for key in keys)
    x, y, width, height = values
    if width <= 0 or height <= 0:
        return None
    if x < 0 or y < 0 or x + width > image.width or y + height > image.height:
        return None
    return values


def _openslide_mpp(image: Any) -> tuple[tuple[float, float] | None, str]:
    """Read calibrated micrometres-per-pixel from OpenSlide metadata."""

    fields = set(image.get_fields())
    keys = ("openslide.mpp-x", "openslide.mpp-y")
    if not all(key in fields for key in keys):
        return None, "unavailable"
    try:
        values = tuple(float(image.get(key)) for key in keys)
    except (TypeError, ValueError):
        return None, "invalid:openslide.mpp"
    if not all(np.isfinite(value) and value > 0 for value in values):
        return None, "invalid:openslide.mpp"
    return (values[0], values[1]), "openslide.mpp"


def _vips_to_rgb(image: Any) -> np.ndarray:
    if image.bands > 3:
        image = image[:3]
    if image.bands == 1:
        image = image.bandjoin([image, image])
    if image.format != "uchar":
        image = image.cast("uchar")
    memory = image.write_to_memory()
    return (
        np.frombuffer(memory, dtype=np.uint8)
        .reshape(
            image.height,
            image.width,
            image.bands,
        )[:, :, :3]
        .copy()
    )


def _is_derived_or_label_image(path: Path) -> bool:
    name = path.name.lower()
    return any(
        marker in name
        for marker in (
            "_final_label",
            ".registered.",
            "_registered.",
            ".thumbnail.",
            "mask_overlay",
        )
    )


def _natural_key(path: Path) -> tuple[object, ...]:
    import re

    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    )


def _import_pyvips() -> Any:
    try:
        import pyvips
    except ImportError as exc:
        raise OptionalDependencyError("pyvips", "wsi") from exc
    return pyvips
