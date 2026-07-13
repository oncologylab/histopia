"""Persisted human review and dataset-specific registration overrides."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from histopia.registration._errors import OptionalDependencyError
from histopia.registration._masking import TissueMaskResult, evaluate_tissue_mask
from histopia.registration._slides import SlideGeometry

MaskReviewStatus = Literal["pending", "auto_pass", "override_pass", "rejected"]
APPROVED_MASK_STATUSES = frozenset({"auto_pass", "override_pass"})


@dataclass(slots=True)
class MaskReviewEntry:
    """Review state for one thumbnail mask."""

    slide: str
    thumbnail_sha256: str
    status: MaskReviewStatus = "pending"
    method: str = ""
    reviewer: str = ""
    notes: str = ""
    override_path: str | None = None

    @property
    def approved(self) -> bool:
        return self.status in APPROVED_MASK_STATUSES

    def to_json_dict(self) -> dict[str, object]:
        return {
            "slide": self.slide,
            "thumbnail_sha256": self.thumbnail_sha256,
            "status": self.status,
            "method": self.method,
            "reviewer": self.reviewer,
            "notes": self.notes,
            "override_path": self.override_path,
        }


def thumbnail_sha256(image: np.ndarray, geometry: SlideGeometry) -> str:
    """Fingerprint thumbnail pixels and their native-coordinate geometry."""

    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(image).tobytes())
    digest.update(json.dumps(geometry.to_json_dict(), sort_keys=True).encode())
    return digest.hexdigest()


def load_mask_review(path: Path | str | None) -> dict[str, MaskReviewEntry]:
    """Load a mask review manifest keyed by exact source filename."""

    if path is None or not Path(path).exists():
        return {}
    payload = json.loads(Path(path).read_text())
    entries = payload.get("slides", payload)
    return {item["slide"]: MaskReviewEntry(**item) for item in entries}


def write_mask_review(
    path: Path | str,
    entries: dict[str, MaskReviewEntry],
) -> Path:
    """Write a deterministic mask review manifest."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "slides": [entries[key].to_json_dict() for key in sorted(entries)],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def resolve_reviewed_mask(
    *,
    slide_path: Path,
    image: np.ndarray,
    geometry: SlideGeometry,
    automatic: TissueMaskResult,
    review_entries: dict[str, MaskReviewEntry],
    override_dir: Path | None,
    require_approved: bool,
) -> tuple[TissueMaskResult, MaskReviewEntry]:
    """Apply an override and enforce review approval for one slide."""

    fingerprint = thumbnail_sha256(image, geometry)
    entry = review_entries.get(slide_path.name)
    if entry is None or entry.thumbnail_sha256 != fingerprint:
        entry = MaskReviewEntry(
            slide=slide_path.name,
            thumbnail_sha256=fingerprint,
            method=automatic.method,
        )

    override_path = _find_override(slide_path, entry, override_dir)
    result = automatic
    if override_path is not None:
        override = _load_binary_mask(override_path, image.shape[:2])
        metrics, warnings = evaluate_tissue_mask(override)
        result = TissueMaskResult(
            mask=override,
            method="reviewed_override",
            metrics=metrics,
            accepted=not warnings,
            warnings=warnings,
            candidate_metrics=automatic.candidate_metrics,
            candidate_warnings=automatic.candidate_warnings,
            candidate_masks=automatic.candidate_masks,
        )
        entry.override_path = str(override_path)
        entry.method = result.method
    else:
        entry.method = automatic.method

    if require_approved and not entry.approved:
        msg = f"mask for {slide_path.name} is not approved (status={entry.status})"
        raise ValueError(msg)
    if entry.status == "override_pass" and override_path is None:
        msg = f"approved override is missing for {slide_path.name}"
        raise FileNotFoundError(msg)
    return result, entry


def _find_override(
    slide_path: Path,
    entry: MaskReviewEntry,
    override_dir: Path | None,
) -> Path | None:
    candidates: list[Path] = []
    if entry.override_path:
        candidates.append(Path(entry.override_path))
    if override_dir is not None:
        candidates.extend(
            [
                override_dir / f"{slide_path.name}.mask.png",
                override_dir / f"{slide_path.stem}.mask.png",
            ]
        )
    return next((candidate for candidate in candidates if candidate.exists()), None)


def _load_binary_mask(path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:
        raise OptionalDependencyError("pillow", "wsi") from exc

    with Image.open(path) as image:
        mask = np.asarray(image.convert("L")) > 127
    if mask.shape != expected_shape:
        msg = f"mask override {path} has shape {mask.shape}, expected {expected_shape}"
        raise ValueError(msg)
    if not mask.any():
        msg = f"mask override {path} is empty"
        raise ValueError(msg)
    return mask
