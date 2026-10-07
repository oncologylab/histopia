"""Per-section review and fingerprint-bound cell-run approval."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from histopia._atomic import write_json_atomic
from histopia.cells._result import validate_cell_result


@dataclass(frozen=True, slots=True)
class CellApproval:
    """Validated approval for every section in one exact cell result."""

    run_dir: Path
    fingerprint: str
    sections: tuple[str, ...]


def review_cell_section(
    run_dir: Path | str,
    section: str,
    *,
    accepted: bool,
    reviewer: str,
    notes: str = "",
    issues: tuple[str, ...] = (),
    reviewed_at: str | None = None,
) -> dict[str, object]:
    """Record one section decision; acceptance itself marks it reviewed."""

    root = Path(run_dir)
    result = _load_result_for_review(root)
    review = _normalized_review(root, result)
    rows = review["sections"]
    assert isinstance(rows, dict)
    if section not in rows:
        raise ValueError(f"unknown cell-result section: {section}")
    reviewer = reviewer.strip()
    if not reviewer:
        raise ValueError("reviewer must not be blank")
    normalized_issues = tuple(
        dict.fromkeys(value.strip() for value in issues if value.strip())
    )
    timestamp = reviewed_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    _validate_timestamp(timestamp)
    rows[section] = {
        "accepted": bool(accepted),
        "reviewer": reviewer,
        "reviewed_at": timestamp,
        "notes": notes.strip(),
        "issues": list(normalized_issues),
    }
    review["approved"] = all(
        isinstance(row, dict) and row.get("accepted") is True for row in rows.values()
    )
    write_json_atomic(root / "cell_review.json", review)
    return cell_review_status(root, result)


def approve_cell_result(run_dir: Path | str) -> CellApproval:
    """Validate that every selected section has an accepted review."""

    root = Path(run_dir)
    result = validate_cell_result(root)
    review = _normalized_review(root, result)
    rows = review["sections"]
    assert isinstance(rows, dict)
    pending = [
        section
        for section, row in rows.items()
        if not isinstance(row, dict) or row.get("accepted") is not True
    ]
    if pending:
        raise ValueError(
            "cell approval requires accepted sections: " + ", ".join(pending)
        )
    review["approved"] = True
    write_json_atomic(root / "cell_review.json", review)
    return CellApproval(root, str(result["fingerprint"]), tuple(rows))


def validate_cell_approval(run_dir: Path | str) -> CellApproval:
    """Validate an existing all-section approval for the current result."""

    root = Path(run_dir)
    result = validate_cell_result(root)
    review = _normalized_review(root, result)
    rows = review["sections"]
    assert isinstance(rows, dict)
    pending = [
        section
        for section, row in rows.items()
        if not isinstance(row, dict) or row.get("accepted") is not True
    ]
    if pending or review.get("approved") is not True:
        raise ValueError("cell result does not have a complete current approval")
    return CellApproval(root, str(result["fingerprint"]), tuple(rows))


def cell_review_status(
    run_dir: Path | str,
    result: dict[str, object] | None = None,
) -> dict[str, object]:
    """Return path-free per-section review status."""

    root = Path(run_dir)
    result = _load_result_for_review(root) if result is None else result
    review = _normalized_review(root, result)
    rows = review["sections"]
    assert isinstance(rows, dict)
    accepted = [
        section
        for section, row in rows.items()
        if isinstance(row, dict) and row.get("accepted") is True
    ]
    return {
        "approved": len(accepted) == len(rows),
        "fingerprint_matches": True,
        "accepted_sections": accepted,
        "pending_sections": [section for section in rows if section not in accepted],
        "sections": rows,
    }


def _normalized_review(root: Path, result: dict[str, object]) -> dict[str, object]:
    from histopia.cells._result import _current_review

    return _current_review(root / "cell_review.json", result)


def _validate_timestamp(value: str) -> None:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError("reviewed_at must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise ValueError("reviewed_at must include a timezone")


def _load_result_for_review(root: Path) -> dict[str, object]:
    """Check the small manifest fingerprint; final approval hashes all artifacts."""

    payload = json.loads((root / "cell_result.json").read_text())
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("cell result must use schema version 1")
    fingerprint = payload.get("fingerprint")
    core = {key: value for key, value in payload.items() if key != "fingerprint"}
    expected = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if fingerprint != expected:
        raise ValueError("cell result fingerprint is stale")
    return payload
