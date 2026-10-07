"""Registration-bound validation for native-space cell segmentation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from histopia._atomic import write_json_atomic
from histopia._selection import load_analysis_selections, resolve_analysis_selections
from histopia.cells._config import CellSegmentationConfig
from histopia.registration._slides import SlideGeometry


@dataclass(frozen=True, slots=True)
class CellPreflightSlide:
    """Validated native slide, accepted mask, and registration transform."""

    section: str
    slide_name: str
    source_path: str
    source_identity: str
    mask_path: str
    mask_sha256: str
    transform_sha256: str
    native_shape: tuple[int, int]
    content_bbox_xywh: tuple[int, int, int, int]
    thumbnail_shape: tuple[int, int]
    mpp_xy: tuple[float, float]
    is_reference: bool

    def portable_json_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload.pop("source_path")
        payload.pop("mask_path")
        return payload


@dataclass(frozen=True, slots=True)
class CellPreflight:
    """Exact selected registration inputs for one cell run."""

    schema_version: int
    registration_run: str
    registration_result_sha256: str
    registration_approval_sha256: str | None
    analysis_manifest_sha256: str | None
    excluded_slides: tuple[dict[str, str | None], ...]
    slides: tuple[CellPreflightSlide, ...]
    fingerprint: str

    @property
    def slide_count(self) -> int:
        return len(self.slides)


def preflight_cell_run(config: CellSegmentationConfig) -> CellPreflight:
    """Resolve selected native slides and require accepted tissue masks."""

    run = config.registration_run.expanduser().resolve()
    result_path = run / "registration_result.json"
    payload = json.loads(result_path.read_text())
    rows = payload.get("slides")
    if (
        not isinstance(rows, list)
        or not rows
        or any(not isinstance(row, dict) for row in rows)
    ):
        raise ValueError("registration result contains no valid slides")
    slide_names = tuple(Path(str(row.get("path", ""))).name for row in rows)
    manifest = (
        load_analysis_selections(config.analysis_manifest)
        if config.analysis_manifest is not None
        else None
    )
    selections = resolve_analysis_selections(slide_names, manifest)
    excluded_names = {
        selection.slide_id for selection in selections if not selection.included
    }
    selected = _select_rows(rows, config.sections, excluded_names=excluded_names)
    approval_sha = _registration_approval_sha(
        run, required=config.require_registration_approval
    )
    slides = tuple(_validate_slide(run, order, row) for order, row in selected)
    core = {
        "schema_version": 2,
        "registration_result_sha256": _sha256_file(result_path),
        "registration_approval_sha256": approval_sha,
        "analysis_manifest_sha256": (
            _sha256_file(config.analysis_manifest)
            if config.analysis_manifest is not None
            else None
        ),
        "excluded_slides": [
            {"slide_id": selection.slide_id, "reason": selection.exclusion_reason}
            for selection in selections
            if not selection.included
        ],
        "slides": [slide.portable_json_dict() for slide in slides],
    }
    return CellPreflight(
        schema_version=2,
        registration_run=str(run),
        registration_result_sha256=str(core["registration_result_sha256"]),
        registration_approval_sha256=approval_sha,
        analysis_manifest_sha256=core["analysis_manifest_sha256"],
        excluded_slides=tuple(core["excluded_slides"]),
        slides=slides,
        fingerprint=_sha256_json(core),
    )


def write_cell_preflight(preflight: CellPreflight, output_path: Path | str) -> Path:
    """Write a local preflight including private paths needed by the runner."""

    payload = asdict(preflight)
    payload["slide_count"] = preflight.slide_count
    return write_json_atomic(output_path, payload)


def load_cell_preflight(path: Path | str) -> CellPreflight:
    """Load and fingerprint-validate a persisted cell preflight."""

    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict) or payload.get("schema_version") != 2:
        raise ValueError("cell preflight must use schema version 2")
    raw_slides = payload.get("slides")
    if not isinstance(raw_slides, list) or not raw_slides:
        raise ValueError("cell preflight contains no slides")
    slides: list[CellPreflightSlide] = []
    try:
        for raw in raw_slides:
            if not isinstance(raw, dict):
                raise TypeError
            values = dict(raw)
            for key in (
                "native_shape",
                "content_bbox_xywh",
                "thumbnail_shape",
                "mpp_xy",
            ):
                values[key] = tuple(values[key])
            slides.append(CellPreflightSlide(**values))
        excluded = tuple(dict(row) for row in payload.get("excluded_slides", []))
        preflight = CellPreflight(
            schema_version=2,
            registration_run=str(payload["registration_run"]),
            registration_result_sha256=str(payload["registration_result_sha256"]),
            registration_approval_sha256=(
                None
                if payload.get("registration_approval_sha256") is None
                else str(payload["registration_approval_sha256"])
            ),
            analysis_manifest_sha256=(
                None
                if payload.get("analysis_manifest_sha256") is None
                else str(payload["analysis_manifest_sha256"])
            ),
            excluded_slides=excluded,
            slides=tuple(slides),
            fingerprint=str(payload["fingerprint"]),
        )
    except (KeyError, TypeError, ValueError):
        raise ValueError("cell preflight is malformed") from None
    core = {
        "schema_version": 2,
        "registration_result_sha256": preflight.registration_result_sha256,
        "registration_approval_sha256": preflight.registration_approval_sha256,
        "analysis_manifest_sha256": preflight.analysis_manifest_sha256,
        "excluded_slides": list(preflight.excluded_slides),
        "slides": [slide.portable_json_dict() for slide in preflight.slides],
    }
    if (
        payload.get("slide_count") != preflight.slide_count
        or len({slide.section for slide in preflight.slides}) != preflight.slide_count
        or len({slide.slide_name for slide in preflight.slides})
        != preflight.slide_count
        or preflight.fingerprint != _sha256_json(core)
    ):
        raise ValueError("cell preflight fingerprint is stale")
    return preflight


def _select_rows(
    rows: list[dict[str, object]],
    sections: tuple[str, ...],
    *,
    excluded_names: set[str] | None = None,
) -> list[tuple[int, dict[str, object]]]:
    excluded = excluded_names or set()
    indexed = [
        (order, row)
        for order, row in enumerate(rows, start=1)
        if Path(str(row.get("path", ""))).name not in excluded
    ]
    if not sections:
        return indexed
    requested = set(sections)
    output = []
    matched: set[str] = set()
    for order, row in indexed:
        path = Path(str(row.get("path", "")))
        aliases = {f"{order:03d}", path.name, path.stem}
        hits = aliases & requested
        if hits:
            output.append((order, row))
            matched.update(hits)
    missing = requested - matched
    if missing:
        raise ValueError("unknown requested sections: " + ", ".join(sorted(missing)))
    return output


def _validate_slide(
    run: Path, order: int, row: dict[str, object]
) -> CellPreflightSlide:
    source = Path(str(row.get("path", ""))).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"section {order:03d}: source WSI is missing")
    geometry = SlideGeometry.from_json_dict(row.get("geometry"))
    if geometry.mpp_xy is None:
        raise ValueError(f"{source.name}: calibrated MPP is required")
    mask = row.get("mask")
    if not isinstance(mask, dict) or mask.get("accepted") is not True:
        raise ValueError(f"{source.name}: tissue mask has not been accepted")
    mask_path = run / "processed" / f"{source.stem}.mask.png"
    if not mask_path.is_file():
        raise FileNotFoundError(f"{source.name}: accepted tissue mask is missing")
    transform = row.get("transform")
    if not isinstance(transform, dict) or transform.get("matrix") is None:
        raise ValueError(f"{source.name}: registration transform is missing")
    return CellPreflightSlide(
        section=f"{order:03d}",
        slide_name=source.name,
        source_path=str(source),
        source_identity=_file_identity(source),
        mask_path=str(mask_path.resolve()),
        mask_sha256=_sha256_file(mask_path),
        transform_sha256=_sha256_json(transform["matrix"]),
        native_shape=geometry.native_shape,
        content_bbox_xywh=geometry.content_bbox_xywh,
        thumbnail_shape=geometry.thumbnail_shape,
        mpp_xy=geometry.mpp_xy,
        is_reference=bool(row.get("is_reference")),
    )


def _registration_approval_sha(run: Path, *, required: bool) -> str | None:
    path = run / "registration_approval.json"
    try:
        from histopia.registration._approval import validate_registration_approval

        validate_registration_approval(run)
    except (FileNotFoundError, ValueError):
        if required:
            raise ValueError(
                "cell preflight requires a sealed registration approval"
            ) from None
        return None
    return _sha256_file(path)


def _file_identity(path: Path) -> str:
    stat = path.stat()
    return _sha256_json(
        {"name": path.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
