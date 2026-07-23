"""Strict provenance checks before semantic feature extraction."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class SemanticPreflightSlide:
    """Validated inputs for one registered section."""

    slide_name: str
    source_path: str
    source_sha256: str
    thumbnail_sha256: str
    mask_sha256: str
    transform_sha256: str
    thumbnail_shape: tuple[int, int]
    mpp_xy: tuple[float, float]
    is_reference: bool


@dataclass(frozen=True, slots=True)
class SemanticPreflight:
    """Portable identity of a registration run accepted for extraction."""

    schema_version: int
    registration_run: str
    registration_result_sha256: str
    order_review_fingerprint: str | None
    reference_slide: str
    slides: tuple[SemanticPreflightSlide, ...]
    fingerprint: str

    @property
    def slide_count(self) -> int:
        return len(self.slides)


def preflight_registration(registration_run: Path | str) -> SemanticPreflight:
    """Validate a registration run and derive its extraction fingerprint."""

    run = Path(registration_run).expanduser().resolve()
    result_path = run / "registration_result.json"
    payload = json.loads(result_path.read_text())
    rows = payload.get("slides")
    if not isinstance(rows, list) or not rows:
        raise ValueError("registration result contains no slides")

    order_fingerprint = _approved_order_fingerprint(run)
    names = [Path(str(row.get("path", ""))).name for row in rows]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("registration slide names must be non-empty and unique")
    references = [
        name for name, row in zip(names, rows, strict=True) if row.get("is_reference")
    ]
    if len(references) != 1:
        raise ValueError("registration must contain exactly one reference slide")

    slides = tuple(
        _validate_slide(run, name, row) for name, row in zip(names, rows, strict=True)
    )
    core = {
        "schema_version": 1,
        "registration_result_sha256": _sha256_file(result_path),
        "order_review_fingerprint": order_fingerprint,
        "reference_slide": references[0],
        "slides": [_portable_slide(slide) for slide in slides],
    }
    fingerprint = _sha256_json(core)
    return SemanticPreflight(
        schema_version=1,
        registration_run=str(run),
        registration_result_sha256=core["registration_result_sha256"],
        order_review_fingerprint=order_fingerprint,
        reference_slide=references[0],
        slides=slides,
        fingerprint=fingerprint,
    )


def write_preflight(preflight: SemanticPreflight, output_path: Path | str) -> Path:
    """Write a validated preflight record outside the source registration."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(preflight)
    payload["slide_count"] = preflight.slide_count
    output.write_text(json.dumps(payload, indent=2) + "\n")
    return output


def _validate_slide(
    run: Path, slide_name: str, row: dict[str, Any]
) -> SemanticPreflightSlide:
    source = Path(str(row["path"])).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"{slide_name}: source slide is missing")
    geometry = row.get("geometry", {})
    shape = tuple(int(value) for value in geometry.get("thumbnail_shape", ()))
    if len(shape) != 2 or min(shape) <= 0:
        raise ValueError(f"{slide_name}: invalid thumbnail shape")
    mpp = tuple(float(value) for value in geometry.get("mpp_xy") or ())
    if len(mpp) != 2 or not np.all(np.isfinite(mpp)) or min(mpp) <= 0:
        raise ValueError(f"{slide_name}: positive finite MPP is required")
    matrix = np.asarray(row.get("transform", {}).get("matrix"), dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{slide_name}: transform must be a finite 3x3 matrix")

    stem = source.stem
    thumbnail = run / "processed" / f"{stem}.thumbnail.png"
    mask = run / "processed" / f"{stem}.mask.png"
    if not thumbnail.is_file():
        raise FileNotFoundError(f"{slide_name}: thumbnail is missing")
    if not mask.is_file():
        raise FileNotFoundError(f"{slide_name}: mask is missing")
    thumbnail_shape, mask_shape = _image_shapes(thumbnail, mask)
    if thumbnail_shape != shape:
        raise ValueError(f"{slide_name}: thumbnail shape does not match geometry")
    if mask_shape != shape:
        raise ValueError(f"{slide_name}: mask shape does not match geometry")
    return SemanticPreflightSlide(
        slide_name=slide_name,
        source_path=str(source.resolve()),
        source_sha256=_sha256_file(source),
        thumbnail_sha256=_sha256_file(thumbnail),
        mask_sha256=_sha256_file(mask),
        transform_sha256=_sha256_json(matrix.tolist()),
        thumbnail_shape=shape,
        mpp_xy=mpp,
        is_reference=bool(row.get("is_reference")),
    )


def _approved_order_fingerprint(run: Path) -> str | None:
    path = run / "section_order_review.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    if not payload.get("approved"):
        raise ValueError("section order is not approved")
    fingerprint = str(payload.get("fingerprint", ""))
    if not fingerprint:
        raise ValueError("approved section order has no fingerprint")
    return fingerprint


def _image_shapes(
    thumbnail: Path, mask: Path
) -> tuple[tuple[int, int], tuple[int, int]]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("semantic preflight requires the 'semantic' extra") from exc
    with Image.open(thumbnail) as image:
        thumbnail_shape = (image.height, image.width)
    with Image.open(mask) as image:
        mask_shape = (image.height, image.width)
    return thumbnail_shape, mask_shape


def _portable_slide(slide: SemanticPreflightSlide) -> dict[str, object]:
    payload = asdict(slide)
    payload.pop("source_path")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
