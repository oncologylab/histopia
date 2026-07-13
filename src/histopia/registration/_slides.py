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

    def to_json_dict(self) -> dict[str, object]:
        return {
            "native_shape": list(self.native_shape),
            "content_bbox_xywh": list(self.content_bbox_xywh),
            "thumbnail_shape": list(self.thumbnail_shape),
            "bounds_source": self.bounds_source,
            "thumbnail_to_native": self.thumbnail_to_native.tolist(),
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
    thumbnail = _vips_to_rgb(thumbnail)
    geometry = SlideGeometry(
        native_shape=native_shape,
        content_bbox_xywh=bbox,
        thumbnail_shape=thumbnail.shape[:2],
        bounds_source=bounds_source,
    )
    return thumbnail, geometry


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
