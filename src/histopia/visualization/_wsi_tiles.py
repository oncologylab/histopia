"""Approval-bound, on-demand tiles for local whole-slide review."""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from histopia.registration._approval import validate_registration_approval
from histopia.registration._errors import OptionalDependencyError

_SECTION_RE = re.compile(r"[0-9]{3,6}")
_DIGEST_RE = re.compile(r"[0-9a-f]{64}")
_LAYERS = frozenset({"raw", "registered", "mask"})


@dataclass(frozen=True, slots=True)
class WsiLevel:
    """One native pyramid level, ordered from smallest to largest."""

    width: int
    height: int
    source_level: int


@dataclass(frozen=True, slots=True)
class WsiLayer:
    """One explicitly registered image source."""

    name: str
    path: Path
    digest: str
    levels: tuple[WsiLevel, ...]
    tile_size: int
    microns_per_pixel: float | None
    mask_path: Path | None = None
    crop_bbox_xywh: tuple[int, int, int, int] | None = None
    source_shape: tuple[int, int] | None = None


@dataclass(frozen=True, slots=True)
class WsiSection:
    """Path-free public identity plus private image sources for one section."""

    cohort: str
    section: str
    slide: str
    label: str
    reference: bool
    layers: dict[str, WsiLayer]


class WsiTileService:
    """Serve immutable tiles from an explicit, approval-bound cohort registry."""

    def __init__(
        self,
        sections: dict[tuple[str, str], WsiSection],
        *,
        max_concurrent_tiles: int = 8,
    ) -> None:
        if max_concurrent_tiles <= 0:
            raise ValueError("max_concurrent_tiles must be positive")
        self._sections = dict(sections)
        self._capacity = threading.BoundedSemaphore(max_concurrent_tiles)

    @classmethod
    def from_runs(
        cls,
        runs: dict[str, tuple[Path, Path]],
        *,
        max_concurrent_tiles: int = 8,
    ) -> WsiTileService:
        """Build a catalog from ``cohort -> (registration, registered_wsi)``."""

        sections: dict[tuple[str, str], WsiSection] = {}
        for cohort, (registration, registered_wsi) in sorted(runs.items()):
            for section in _load_cohort_sections(
                cohort,
                registration,
                registered_wsi,
            ):
                key = (cohort, section.section)
                if key in sections:
                    raise ValueError(
                        f"duplicate WSI section: {cohort}/{section.section}"
                    )
                sections[key] = section
        if not sections:
            raise ValueError("WSI registry contains no exported sections")
        return cls(sections, max_concurrent_tiles=max_concurrent_tiles)

    def metadata(self, cohort: str, section: str) -> dict[str, object]:
        """Return path-free metadata for one section."""

        item = self._section(cohort, section)
        return {
            "schema_version": 1,
            "cohort": item.cohort,
            "section": item.section,
            "slide": item.slide,
            "label": item.label,
            "reference": item.reference,
            "layers": {
                name: {
                    "digest": layer.digest,
                    "tile_size": layer.tile_size,
                    "width": layer.levels[-1].width,
                    "height": layer.levels[-1].height,
                    "levels": [
                        {"width": level.width, "height": level.height}
                        for level in layer.levels
                    ],
                    "microns_per_pixel": layer.microns_per_pixel,
                    "format": "png" if name == "mask" else "jpg",
                }
                for name, layer in sorted(item.layers.items())
            },
        }

    def section(self, cohort: str, section: str) -> WsiSection:
        """Return an explicitly configured section for trusted local exporters."""

        return self._section(cohort, section)

    def sections(self, cohort: str) -> tuple[str, ...]:
        """Return sorted configured section IDs for one cohort."""

        return tuple(
            section
            for item_cohort, section in sorted(self._sections)
            if item_cohort == cohort
        )

    def catalog(self, cohort: str) -> dict[str, object]:
        """Return path-free available-section metadata for one cohort."""

        rows = [
            {
                "section": section,
                "slide": item.slide,
                "label": item.label,
                "reference": item.reference,
                "layers": sorted(item.layers),
            }
            for (item_cohort, section), item in sorted(self._sections.items())
            if item_cohort == cohort
        ]
        if not rows:
            raise FileNotFoundError("unknown WSI cohort")
        return {
            "schema_version": 1,
            "cohort": cohort,
            "sections": rows,
        }

    def render_tile(
        self,
        cohort: str,
        section: str,
        layer_name: str,
        digest: str,
        level: int,
        x: int,
        y: int,
    ) -> tuple[bytes, str, str]:
        """Render one bounded tile and return bytes, media type, and ETag."""

        item = self._section(cohort, section)
        if layer_name not in _LAYERS or layer_name not in item.layers:
            raise FileNotFoundError("unknown WSI layer")
        layer = item.layers[layer_name]
        if not _DIGEST_RE.fullmatch(digest) or digest != layer.digest:
            raise FileNotFoundError("stale WSI layer")
        if level < 0 or level >= len(layer.levels) or x < 0 or y < 0:
            raise FileNotFoundError("invalid WSI tile coordinates")
        dimensions = layer.levels[level]
        columns = math.ceil(dimensions.width / layer.tile_size)
        rows = math.ceil(dimensions.height / layer.tile_size)
        if x >= columns or y >= rows:
            raise FileNotFoundError("invalid WSI tile coordinates")
        if not self._capacity.acquire(blocking=False):
            raise WsiTileCapacityError("WSI tile capacity is full")
        try:
            payload = _render_layer_tile(layer, level, x, y)
        finally:
            self._capacity.release()
        media_type = "image/png" if layer_name == "mask" else "image/jpeg"
        etag = f'"{digest}-{level}-{x}-{y}"'
        return payload, media_type, etag

    def _section(self, cohort: str, section: str) -> WsiSection:
        if not _SECTION_RE.fullmatch(section):
            raise FileNotFoundError("unknown WSI section")
        try:
            return self._sections[(cohort, section)]
        except KeyError as error:
            raise FileNotFoundError("unknown WSI section") from error


class WsiTileCapacityError(RuntimeError):
    """Raised when bounded native tile rendering has no free worker slot."""


def _load_cohort_sections(
    cohort: str,
    registration: Path,
    registered_wsi: Path,
) -> tuple[WsiSection, ...]:
    approval = validate_registration_approval(registration)
    result_path = registration / "registration_result.json"
    result = json.loads(result_path.read_text())
    slides = result.get("slides")
    if not isinstance(slides, list) or not slides:
        raise ValueError(f"registration contains no slides: {registration}")
    if any(not isinstance(row, dict) for row in slides):
        raise ValueError("registration slides must contain objects")

    summary_path = registered_wsi / "full_resolution_warps.json"
    summary = json.loads(summary_path.read_text())
    if not isinstance(summary, list):
        raise ValueError("full-resolution warp summary must contain a list")
    by_stem: dict[str, dict[str, Any]] = {}
    root = registered_wsi.expanduser().resolve()
    for row in summary:
        if not isinstance(row, dict):
            raise ValueError("full-resolution warp summary rows must be objects")
        output_value = row.get("output_path")
        provenance = row.get("provenance")
        if not isinstance(output_value, str) or not isinstance(provenance, dict):
            raise ValueError("full-resolution warp provenance is incomplete")
        output = Path(output_value).expanduser().resolve()
        if output.parent != root or not output.is_file():
            raise ValueError(
                "registered WSI output is outside its configured directory"
            )
        if (
            provenance.get("registration_result_sha256")
            != approval.registration_result_sha256
        ):
            raise ValueError("registered WSI is not bound to the approved registration")
        export_digest = provenance.get("export_fingerprint")
        if not isinstance(export_digest, str) or not _DIGEST_RE.fullmatch(
            export_digest
        ):
            raise ValueError("registered WSI export fingerprint is invalid")
        stem = output.name.removesuffix(".registered.tiff")
        if stem in by_stem:
            raise ValueError(f"duplicate registered WSI output stem: {stem}")
        by_stem[stem] = row

    sections: list[WsiSection] = []
    for order, slide in enumerate(slides, start=1):
        source = Path(str(slide.get("path", ""))).expanduser().resolve()
        row = by_stem.get(source.stem)
        if row is None:
            continue
        registered = Path(str(row["output_path"])).expanduser().resolve()
        export_digest = str(row["provenance"]["export_fingerprint"])
        raw_digest = _file_identity_digest(source)
        mask_path = registration / "processed" / f"{source.stem}.mask.png"
        geometry = slide.get("geometry")
        layers = {
            "raw": _image_layer(
                "raw",
                source,
                raw_digest,
                crop_bbox_xywh=_geometry_content_bbox(geometry),
            ),
            "registered": _image_layer(
                "registered",
                registered,
                export_digest,
            ),
        }
        if mask_path.is_file():
            layers["mask"] = _mask_layer(
                mask_path,
                layers["raw"],
            )
        sections.append(
            WsiSection(
                cohort=cohort,
                section=f"{order:03d}",
                slide=source.name,
                label=_marker_label(source.stem),
                reference=bool(slide.get("is_reference")),
                layers=layers,
            )
        )
    return tuple(sections)


def _image_layer(
    name: str,
    path: Path,
    digest: str,
    *,
    crop_bbox_xywh: tuple[int, int, int, int] | None = None,
) -> WsiLayer:
    levels, microns_per_pixel = _discover_levels(path)
    full = levels[-1]
    source_shape = (full.width, full.height)
    if crop_bbox_xywh is not None:
        _, _, crop_width, crop_height = crop_bbox_xywh
        levels = tuple(
            WsiLevel(
                width=max(1, int(round(crop_width * level.width / full.width))),
                height=max(1, int(round(crop_height * level.height / full.height))),
                source_level=level.source_level,
            )
            for level in levels
        )
    return WsiLayer(
        name=name,
        path=path,
        digest=digest,
        levels=levels,
        tile_size=512,
        microns_per_pixel=microns_per_pixel,
        crop_bbox_xywh=crop_bbox_xywh,
        source_shape=source_shape,
    )


def _mask_layer(
    mask_path: Path,
    raw: WsiLayer,
) -> WsiLayer:
    digest = hashlib.sha256(mask_path.read_bytes()).hexdigest()
    return WsiLayer(
        name="mask",
        path=raw.path,
        digest=digest,
        levels=raw.levels,
        tile_size=raw.tile_size,
        microns_per_pixel=raw.microns_per_pixel,
        mask_path=mask_path,
    )


def _discover_levels(path: Path) -> tuple[tuple[WsiLevel, ...], float | None]:
    pyvips = _import_pyvips()
    image = pyvips.Image.new_from_file(str(path), access="random")
    full_to_small: list[tuple[int, int, int]] = []
    level_count = _integer_field(image, "openslide.level-count")
    if level_count is not None and level_count > 0:
        for source_level in range(level_count):
            width = _integer_field(image, f"openslide.level[{source_level}].width")
            height = _integer_field(image, f"openslide.level[{source_level}].height")
            if width is None or height is None:
                level_image = pyvips.Image.new_from_file(
                    str(path),
                    level=source_level,
                    access="random",
                )
                width, height = level_image.width, level_image.height
            full_to_small.append((width, height, source_level))
    else:
        page_count = _integer_field(image, "n-pages") or 1
        for source_level in range(page_count):
            level_image = pyvips.Image.new_from_file(
                str(path),
                page=source_level,
                n=1,
                access="random",
            )
            full_to_small.append((level_image.width, level_image.height, source_level))
    unique: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int]] = set()
    for row in full_to_small:
        if row[:2] not in seen:
            unique.append(row)
            seen.add(row[:2])
    unique.sort(key=lambda row: row[0] * row[1])
    levels = tuple(
        WsiLevel(width=width, height=height, source_level=source_level)
        for width, height, source_level in unique
    )
    if not levels:
        raise ValueError(f"WSI contains no readable pyramid levels: {path}")
    xres = float(image.xres) if image.xres and image.xres > 0 else 0.0
    microns_per_pixel = 1000.0 / xres if xres > 0 else None
    return levels, microns_per_pixel


def _render_layer_tile(layer: WsiLayer, level: int, x: int, y: int) -> bytes:
    pyvips = _import_pyvips()
    target = layer.levels[level]
    if layer.mask_path is not None:
        image = pyvips.Image.new_from_file(str(layer.mask_path), access="random")
        scale_x = target.width / image.width
        scale_y = target.height / image.height
        image = image.resize(scale_x, vscale=scale_y, kernel="nearest")
    else:
        loader_options = {"access": "random"}
        probe = pyvips.Image.new_from_file(str(layer.path), access="random")
        if _integer_field(probe, "openslide.level-count") is not None:
            loader_options["level"] = target.source_level
        else:
            loader_options["page"] = target.source_level
            loader_options["n"] = 1
        image = pyvips.Image.new_from_file(str(layer.path), **loader_options)
    left = x * layer.tile_size
    top = y * layer.tile_size
    width = min(layer.tile_size, target.width - left)
    height = min(layer.tile_size, target.height - top)
    if layer.crop_bbox_xywh is not None:
        if layer.source_shape is None:
            raise ValueError("cropped WSI layer has no source shape")
        crop_x, crop_y, _, _ = layer.crop_bbox_xywh
        source_width, source_height = layer.source_shape
        left += int(round(crop_x * image.width / source_width))
        top += int(round(crop_y * image.height / source_height))
    tile = image.crop(left, top, width, height)
    if layer.mask_path is not None:
        if tile.bands > 1:
            tile = tile[0]
        if tile.format != "uchar":
            tile = tile.cast("uchar")
        return bytes(tile.pngsave_buffer(compression=3, strip=True))
    tile = _normalize_rgb_uchar(tile)
    return bytes(tile.jpegsave_buffer(Q=90, strip=True, optimize_coding=True))


def _normalize_rgb_uchar(image):  # type: ignore[no-untyped-def]
    if image.bands == 1:
        image = image.bandjoin([image, image])
    elif image.bands == 2:
        image = image[0].bandjoin([image[0], image[0]])
    elif image.bands > 3:
        image = image[:3]
    if image.format != "uchar":
        image = image.cast("uchar")
    return image


def _integer_field(image, name: str) -> int | None:  # type: ignore[no-untyped-def]
    try:
        value = image.get(name)
    except Exception:  # libvips exposes loader-specific metadata dynamically.
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _file_identity_digest(path: Path) -> str:
    stat = path.stat()
    payload = {
        "name": path.name,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _geometry_content_bbox(
    geometry: object,
) -> tuple[int, int, int, int] | None:
    if not isinstance(geometry, dict):
        return None
    value = geometry.get("content_bbox_xywh")
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(not isinstance(item, int) or item < 0 for item in value)
        or value[2] < 1
        or value[3] < 1
    ):
        return None
    return tuple(value)


def _marker_label(stem: str) -> str:
    match = re.search(r"panc[_-](.+?)(?:-\[|$)", stem, flags=re.IGNORECASE)
    return match.group(1) if match else stem


def _import_pyvips():
    try:
        import pyvips
    except (ImportError, OSError) as error:
        raise OptionalDependencyError("pyvips", "wsi") from error
    return pyvips
