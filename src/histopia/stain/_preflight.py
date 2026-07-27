"""Registration and assay integrity checks before stain quantification."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from histopia._atomic import write_json_atomic
from histopia.registration._slides import SlideGeometry
from histopia.stain._assays import (
    SlideAssay,
    load_assay_manifest,
    resolve_slide_assays,
)
from histopia.stain._config import StainQuantificationConfig


@dataclass(frozen=True, slots=True)
class StainPreflightSlide:
    """Validated source, mask, transform, and assay identity."""

    slide_name: str
    source_path: str
    source_sha256: str
    mask_sha256: str
    transform_sha256: str
    thumbnail_shape: tuple[int, int]
    native_shape: tuple[int, int]
    content_bbox_xywh: tuple[int, int, int, int]
    mpp_xy: tuple[float, float]
    is_reference: bool
    assay: SlideAssay

    def portable_json_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload.pop("source_path")
        payload["assay"] = self.assay.to_json_dict()
        return payload


@dataclass(frozen=True, slots=True)
class StainPreflight:
    """Portable fingerprint for one stain campaign."""

    schema_version: int
    registration_run: str
    registration_result_sha256: str
    registration_approval_sha256: str | None
    assay_manifest_sha256: str | None
    reference_slide: str
    slides: tuple[StainPreflightSlide, ...]
    fingerprint: str

    @property
    def slide_count(self) -> int:
        return len(self.slides)


def preflight_stain_run(config: StainQuantificationConfig) -> StainPreflight:
    """Validate exact source slides, masks, transforms, and assay families."""

    run = config.registration_run.expanduser().resolve()
    result_path = run / "registration_result.json"
    payload = json.loads(result_path.read_text())
    rows = payload.get("slides")
    if not isinstance(rows, list) or not rows:
        raise ValueError("registration result contains no slides")
    slide_names = tuple(Path(str(row.get("path", ""))).name for row in rows)
    if any(not name for name in slide_names) or len(set(slide_names)) != len(
        slide_names
    ):
        raise ValueError("registration slide names must be non-empty and unique")
    manifest = (
        load_assay_manifest(config.assay_manifest)
        if config.assay_manifest is not None
        else None
    )
    assays = resolve_slide_assays(
        slide_names,
        manifest=manifest,
        default_family=config.resolved_default_family,
    )
    references = [
        name
        for name, row in zip(slide_names, rows, strict=True)
        if row.get("is_reference")
    ]
    if len(references) != 1:
        raise ValueError("registration must contain exactly one reference slide")
    approval_sha = _registration_approval_sha(
        run,
        required=config.require_registration_approval,
    )
    slides = tuple(
        _validate_slide(run, name, row, assay)
        for name, row, assay in zip(slide_names, rows, assays, strict=True)
    )
    result_sha = _sha256_file(result_path)
    manifest_sha = (
        _sha256_file(config.assay_manifest)
        if config.assay_manifest is not None
        else None
    )
    core = {
        "schema_version": 1,
        "registration_result_sha256": result_sha,
        "registration_approval_sha256": approval_sha,
        "assay_manifest_sha256": manifest_sha,
        "reference_slide": references[0],
        "slides": [slide.portable_json_dict() for slide in slides],
    }
    return StainPreflight(
        schema_version=1,
        registration_run=str(run),
        registration_result_sha256=result_sha,
        registration_approval_sha256=approval_sha,
        assay_manifest_sha256=manifest_sha,
        reference_slide=references[0],
        slides=slides,
        fingerprint=_sha256_json(core),
    )


def write_stain_preflight(
    preflight: StainPreflight,
    output_path: Path | str,
) -> Path:
    payload = asdict(preflight)
    payload["slides"] = [
        {
            **asdict(slide),
            "assay": slide.assay.to_json_dict(),
        }
        for slide in preflight.slides
    ]
    payload["slide_count"] = preflight.slide_count
    return write_json_atomic(output_path, payload)


def _validate_slide(
    run: Path,
    slide_name: str,
    row: dict[str, object],
    assay: SlideAssay,
) -> StainPreflightSlide:
    source = Path(str(row["path"])).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"{slide_name}: source WSI is missing")
    geometry = SlideGeometry.from_json_dict(row.get("geometry"))
    if geometry.mpp_xy is None:
        raise ValueError(f"{slide_name}: calibrated MPP is required")
    matrix = np.asarray(row.get("transform", {}).get("matrix"), dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{slide_name}: transform must be a finite 3x3 matrix")
    mask_payload = row.get("mask")
    if not isinstance(mask_payload, dict) or not mask_payload.get("accepted"):
        raise ValueError(f"{slide_name}: registration tissue mask is not accepted")
    mask_path = run / "processed" / f"{source.stem}.mask.png"
    if not mask_path.is_file():
        raise FileNotFoundError(f"{slide_name}: registration mask is missing")
    return StainPreflightSlide(
        slide_name=slide_name,
        source_path=str(source.resolve()),
        source_sha256=_sha256_file(source),
        mask_sha256=_sha256_file(mask_path),
        transform_sha256=_sha256_json(matrix.tolist()),
        thumbnail_shape=geometry.thumbnail_shape,
        native_shape=geometry.native_shape,
        content_bbox_xywh=geometry.content_bbox_xywh,
        mpp_xy=geometry.mpp_xy,
        is_reference=bool(row.get("is_reference")),
        assay=assay,
    )


def _registration_approval_sha(run: Path, *, required: bool) -> str | None:
    path = run / "registration_approval.json"
    try:
        from histopia.registration._approval import validate_registration_approval

        validate_registration_approval(run)
    except (FileNotFoundError, ValueError):
        if required:
            raise ValueError(
                "stain preflight requires a sealed registration approval"
            ) from None
        return None
    return _sha256_file(path)


def _sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
