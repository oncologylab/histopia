"""Fingerprint-bound approval of completed semantic atlases."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from histopia.semantic._registration_binding import (
    validate_semantic_registration_binding,
)
from histopia.semantic._result_validation import validate_semantic_result


@dataclass(frozen=True, slots=True)
class SemanticApproval:
    """Validated review metadata for one exact semantic result."""

    run_dir: Path
    fingerprint: str
    reviewer: str
    reviewed_at: str | None


def approve_semantic_result(
    run_dir: Path | str,
    *,
    registration_run: Path | str,
    reviewer: str,
    notes: str,
    reviewed_at: str | None = None,
) -> SemanticApproval:
    """Approve a result bound to the exact current registration run."""

    root = Path(run_dir)
    result = validate_semantic_result(root)
    binding = validate_semantic_registration_binding(
        registration_run,
        root,
        semantic_payload=result,
    )
    if not binding.approval_bound:
        raise ValueError(
            "semantic approval requires a preflight bound to the final "
            "registration approval"
        )
    review_path = root / "semantic_review.json"
    review = _load_review(review_path)
    if review.get("schema_version") != 3:
        raise ValueError("semantic review must use schema version 3")
    fingerprint = _required_fingerprint(result)
    if review.get("fingerprint") != fingerprint:
        raise ValueError("semantic review fingerprint is stale")
    reviewer, notes, timestamp = _validated_review_metadata(
        reviewer,
        notes,
        reviewed_at,
    )
    review.update(
        {
            "approved": True,
            "fingerprint": fingerprint,
            "reviewer": reviewer,
            "reviewed_at": timestamp,
            "notes": notes,
        }
    )
    _write_json_atomic(review_path, review)
    return SemanticApproval(root, fingerprint, reviewer, timestamp)


def validate_semantic_approval(run_dir: Path | str) -> SemanticApproval:
    """Verify that semantic approval still names the current sealed result."""

    root = Path(run_dir)
    result = validate_semantic_result(root)
    return _validate_semantic_approval_for_result(root, result)


def _validate_semantic_approval_for_result(
    root: Path,
    result: dict[str, object],
) -> SemanticApproval:
    """Validate review metadata for an already integrity-checked result."""

    review = _load_review(root / "semantic_review.json")
    if review.get("schema_version") != 3:
        raise ValueError("semantic review must use schema version 3")
    if review.get("approved") is not True:
        raise ValueError("semantic result is not approved")
    fingerprint = _required_fingerprint(result)
    if review.get("fingerprint") != fingerprint:
        raise ValueError("semantic review fingerprint is stale")
    reviewer = review.get("reviewer")
    notes = review.get("notes")
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("semantic approval reviewer is missing")
    if not isinstance(notes, str) or not notes.strip():
        raise ValueError("semantic approval notes are missing")
    reviewed_at = review.get("reviewed_at")
    if reviewed_at is not None:
        if not isinstance(reviewed_at, str) or not reviewed_at:
            raise ValueError("semantic approval timestamp is invalid")
        _parse_timestamp(reviewed_at)
    return SemanticApproval(root, fingerprint, reviewer.strip(), reviewed_at)


def _required_fingerprint(result: dict[str, object]) -> str:
    fingerprint = result.get("fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError("semantic result fingerprint is missing")
    return fingerprint


def _load_review(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("semantic review root must be an object")
    return payload


def _validated_review_metadata(
    reviewer: str,
    notes: str,
    reviewed_at: str | None,
) -> tuple[str, str, str]:
    reviewer = reviewer.strip()
    notes = notes.strip()
    if not reviewer:
        raise ValueError("reviewer must not be blank")
    if not notes:
        raise ValueError("approval notes must not be blank")
    timestamp = reviewed_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    _parse_timestamp(timestamp)
    return reviewer, notes, timestamp


def _parse_timestamp(value: str) -> None:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError("reviewed_at must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise ValueError("reviewed_at must include a timezone")


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
