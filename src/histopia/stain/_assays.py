"""Stain-family metadata and conservative filename inference."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class StainFamily(str, Enum):
    """Supported brightfield assay families."""

    H_DAB = "h-dab"
    SIRIUS_RED = "sirius-red"
    PAS = "pas"
    ALCIAN_BLUE = "alcian-blue"
    CONTEXT_HE = "context-he"


@dataclass(frozen=True, slots=True)
class SlideAssay:
    """Scientific assay identity for one registered section."""

    slide_id: str
    marker: str
    family: StainFamily
    batch_id: str | None = None

    def __post_init__(self) -> None:
        if not self.slide_id.strip():
            raise ValueError("slide_id must not be blank")
        if not self.marker.strip():
            raise ValueError("marker must not be blank")
        if self.batch_id is not None and not self.batch_id.strip():
            raise ValueError("batch_id must be non-empty when provided")

    def to_json_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["family"] = self.family.value
        return payload


def load_assay_manifest(path: Path | str) -> dict[str, SlideAssay]:
    """Load a JSON assay manifest keyed by exact source slide filename."""

    source = Path(path)
    payload = json.loads(source.read_text())
    rows: Any = payload.get("slides") if isinstance(payload, dict) else None
    if isinstance(rows, dict):
        iterable = ({"slide_id": slide_id, **value} for slide_id, value in rows.items())
    elif isinstance(rows, list):
        iterable = iter(rows)
    else:
        raise ValueError("assay manifest must contain a slides object or list")
    assays: dict[str, SlideAssay] = {}
    for raw in iterable:
        if not isinstance(raw, dict):
            raise ValueError("assay manifest slide rows must be objects")
        slide_id = str(raw.get("slide_id", "")).strip()
        try:
            family = StainFamily(str(raw.get("family", "")).strip().lower())
        except ValueError as error:
            raise ValueError(
                f"{slide_id or '<missing>'}: unsupported stain family"
            ) from error
        assay = SlideAssay(
            slide_id=slide_id,
            marker=str(raw.get("marker", "")).strip(),
            family=family,
            batch_id=(
                str(raw["batch_id"]).strip()
                if raw.get("batch_id") is not None
                else None
            ),
        )
        if assay.slide_id in assays:
            raise ValueError(f"duplicate assay entry: {assay.slide_id}")
        assays[assay.slide_id] = assay
    if not assays:
        raise ValueError("assay manifest contains no slides")
    return assays


def resolve_slide_assays(
    slide_ids: tuple[str, ...],
    *,
    manifest: dict[str, SlideAssay] | None,
    default_family: StainFamily | None,
) -> tuple[SlideAssay, ...]:
    """Resolve exact manifest rows and safe KPF-style filename inference."""

    extras = set(manifest or ()) - set(slide_ids)
    if extras:
        raise ValueError(
            "assay manifest references slides outside registration: "
            + ", ".join(sorted(extras))
        )
    resolved = []
    for slide_id in slide_ids:
        if manifest is not None and slide_id in manifest:
            resolved.append(manifest[slide_id])
            continue
        resolved.append(infer_slide_assay(slide_id, default_family=default_family))
    return tuple(resolved)


def infer_slide_assay(
    slide_id: str,
    *,
    default_family: StainFamily | None,
) -> SlideAssay:
    """Infer known special stains and require a default for other markers."""

    marker = marker_from_slide_id(slide_id)
    key = re.sub(r"[^a-z0-9]+", "", marker.lower())
    if "sirius" in key or "sirus" in key or key in {"sr", "siriusred"}:
        family = StainFamily.SIRIUS_RED
        marker = "Sirius Red"
    elif "alcian" in key:
        family = StainFamily.ALCIAN_BLUE
        marker = "Alcian Blue"
    elif key == "pas" or key.startswith("pascollection"):
        family = StainFamily.PAS
        marker = "PAS"
    elif key in {"he", "hande", "hematoxylineosin"}:
        family = StainFamily.CONTEXT_HE
        marker = "H&E"
    elif default_family is not None:
        family = default_family
    else:
        raise ValueError(
            f"{slide_id}: stain family is ambiguous; add it to the assay manifest"
        )
    return SlideAssay(slide_id=slide_id, marker=marker, family=family)


def marker_from_slide_id(slide_id: str) -> str:
    """Extract the marker portion from common KPF filenames."""

    stem = _without_slide_suffix(Path(slide_id).name)
    match = re.search(r"panc[_-](.+?)(?:-\[|$)", stem, flags=re.IGNORECASE)
    marker = match.group(1) if match else stem
    marker = re.sub(r"\((?:rab|rat|mouse)\)", "", marker, flags=re.IGNORECASE)
    marker = re.sub(r"^#+\d+\]?\s*", "", marker).strip(" _-")
    return marker or stem


def _without_slide_suffix(name: str) -> str:
    lowered = name.lower()
    for suffix in (".ome.tiff", ".ome.tif", ".ndpi", ".scn", ".tiff", ".tif"):
        if lowered.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem
