"""Fingerprint-bound approval for reconstructed semantic topology."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from histopia._atomic import write_json_atomic
from histopia.topology._result import validate_topology_result


@dataclass(frozen=True, slots=True)
class TopologyApproval:
    """Review identity for one exact topology result."""

    run_dir: Path
    fingerprint: str
    reviewer: str
    reviewed_at: str
    notes: str


def approve_topology_result(
    run_dir: Path | str,
    *,
    reviewer: str,
    notes: str,
    reviewed_at: str | None = None,
) -> TopologyApproval:
    """Approve one exact result after all pair and surface evidence is reviewed."""

    root = Path(run_dir)
    result = validate_topology_result(root)
    _require_approved_inputs(root, result)
    reviewer = reviewer.strip()
    notes = notes.strip()
    if not reviewer or not notes:
        raise ValueError("reviewer and approval notes must not be blank")
    timestamp = reviewed_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    _parse_timestamp(timestamp)
    review = {
        "schema_version": 1,
        "approved": True,
        "fingerprint": result["fingerprint"],
        "reviewer": reviewer,
        "reviewed_at": timestamp,
        "notes": notes,
    }
    write_json_atomic(root / "topology_review.json", review)
    return TopologyApproval(
        root,
        str(result["fingerprint"]),
        reviewer,
        timestamp,
        notes,
    )


def validate_topology_approval(run_dir: Path | str) -> TopologyApproval:
    """Validate approval against the current sealed topology result."""

    root = Path(run_dir)
    result = validate_topology_result(root)
    _require_approved_inputs(root, result)
    payload = json.loads((root / "topology_review.json").read_text())
    if payload.get("schema_version") != 1 or payload.get("approved") is not True:
        raise ValueError("topology result is not approved")
    if payload.get("fingerprint") != result.get("fingerprint"):
        raise ValueError("topology review fingerprint is stale")
    reviewer = payload.get("reviewer")
    notes = payload.get("notes")
    timestamp = payload.get("reviewed_at")
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("topology approval reviewer is missing")
    if not isinstance(notes, str) or not notes.strip():
        raise ValueError("topology approval notes are missing")
    if not isinstance(timestamp, str):
        raise ValueError("topology approval timestamp is missing")
    _parse_timestamp(timestamp)
    return TopologyApproval(
        root,
        str(result["fingerprint"]),
        reviewer.strip(),
        timestamp,
        notes.strip(),
    )


def _parse_timestamp(value: str) -> None:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError("reviewed_at must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise ValueError("reviewed_at must include a timezone")


def _require_approved_inputs(
    root: Path,
    result: dict[str, object],
) -> None:
    preflight = json.loads((root / str(result["preflight"])).read_text())
    approval = preflight.get("semantic_approval")
    if not isinstance(approval, dict):
        raise ValueError(
            "topology approval requires approval-bound registration and semantic inputs"
        )
