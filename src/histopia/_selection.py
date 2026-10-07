"""Exact, reusable analysis-inclusion manifests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class AnalysisSelection:
    """Whether one exact source slide belongs to the analysis scope."""

    slide_id: str
    included: bool = True
    exclusion_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.slide_id.strip():
            raise ValueError("slide_id must not be blank")
        if self.included and self.exclusion_reason is not None:
            raise ValueError("included slides cannot have an exclusion_reason")
        if not self.included and not (self.exclusion_reason or "").strip():
            raise ValueError("excluded slides require an exclusion_reason")


def load_analysis_selections(path: Path | str) -> dict[str, AnalysisSelection]:
    """Load additive inclusion fields from a slides-object/list manifest."""

    source = Path(path)
    payload = json.loads(source.read_text())
    rows = payload.get("slides") if isinstance(payload, dict) else None
    if isinstance(rows, dict):
        iterable = ({"slide_id": key, **value} for key, value in rows.items())
    elif isinstance(rows, list):
        iterable = iter(rows)
    else:
        raise ValueError("analysis manifest must contain a slides object or list")
    output: dict[str, AnalysisSelection] = {}
    for raw in iterable:
        if not isinstance(raw, dict):
            raise ValueError("analysis manifest slide rows must be objects")
        slide_id = str(raw.get("slide_id", "")).strip()
        included = raw.get("analysis_included", True)
        if not isinstance(included, bool):
            raise ValueError(
                f"{slide_id or '<missing>'}: analysis_included must be bool"
            )
        reason = raw.get("exclusion_reason")
        if reason is not None and not isinstance(reason, str):
            raise ValueError(
                f"{slide_id or '<missing>'}: exclusion_reason must be text"
            )
        selection = AnalysisSelection(slide_id, included, reason)
        if slide_id in output:
            raise ValueError(f"duplicate analysis entry: {slide_id}")
        output[slide_id] = selection
    if not output:
        raise ValueError("analysis manifest contains no slides")
    return output


def resolve_analysis_selections(
    slide_ids: tuple[str, ...],
    manifest: dict[str, AnalysisSelection] | None,
) -> tuple[AnalysisSelection, ...]:
    """Resolve all registered slides; omitted manifest rows remain included."""

    extras = set(manifest or ()) - set(slide_ids)
    if extras:
        raise ValueError(
            "analysis manifest references slides outside registration: "
            + ", ".join(sorted(extras))
        )
    return tuple(
        (manifest or {}).get(slide_id, AnalysisSelection(slide_id))
        for slide_id in slide_ids
    )
