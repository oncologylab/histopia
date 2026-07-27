"""Fingerprint-bound scientific approval for stain quantification."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from histopia._atomic import write_json_atomic
from histopia.stain._result_validation import validate_stain_result


@dataclass(frozen=True, slots=True)
class StainApproval:
    """Validated approval metadata for one exact stain result."""

    run_dir: Path
    fingerprint: str
    reviewer: str
    reviewed_at: str


def approve_stain_result(
    run_dir: Path | str,
    *,
    reviewer: str,
    notes: str,
    reviewed_at: str | None = None,
) -> StainApproval:
    """Approve the exact current result after QC and visual review."""

    root = Path(run_dir)
    result = validate_stain_result(root)
    reviewer = reviewer.strip()
    notes = notes.strip()
    if not reviewer or not notes:
        raise ValueError("reviewer and approval notes must not be blank")
    timestamp = reviewed_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    _validate_timestamp(timestamp)
    fingerprint = str(result["fingerprint"])
    write_json_atomic(
        root / "stain_review.json",
        {
            "schema_version": 1,
            "approved": True,
            "fingerprint": fingerprint,
            "reviewer": reviewer,
            "reviewed_at": timestamp,
            "notes": notes,
        },
    )
    return StainApproval(root, fingerprint, reviewer, timestamp)


def validate_stain_approval(run_dir: Path | str) -> StainApproval:
    """Verify approval still names the current sealed result."""

    root = Path(run_dir)
    result = validate_stain_result(root)
    payload = json.loads((root / "stain_review.json").read_text())
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or payload.get("approved") is not True
        or payload.get("fingerprint") != result.get("fingerprint")
    ):
        raise ValueError("stain result is not approved for the current fingerprint")
    reviewer = str(payload.get("reviewer", "")).strip()
    notes = str(payload.get("notes", "")).strip()
    timestamp = str(payload.get("reviewed_at", "")).strip()
    if not reviewer or not notes or not timestamp:
        raise ValueError("stain approval metadata is incomplete")
    _validate_timestamp(timestamp)
    return StainApproval(root, str(result["fingerprint"]), reviewer, timestamp)


def _validate_timestamp(value: str) -> None:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError("reviewed_at must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise ValueError("reviewed_at must include a timezone")
