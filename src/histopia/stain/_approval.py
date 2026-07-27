"""Fingerprint-bound scientific approval for stain quantification."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from histopia._atomic import write_json_atomic
from histopia.stain._assays import StainFamily
from histopia.stain._result_validation import validate_stain_result


@dataclass(frozen=True, slots=True)
class StainFamilyApproval:
    """Review metadata for one stain family in an exact result."""

    family: StainFamily
    reviewer: str
    reviewed_at: str
    notes: str


@dataclass(frozen=True, slots=True)
class StainApproval:
    """Validated family approvals for one exact stain result."""

    run_dir: Path
    fingerprint: str
    approvals: tuple[StainFamilyApproval, ...]

    @property
    def families(self) -> tuple[StainFamily, ...]:
        return tuple(approval.family for approval in self.approvals)


def approve_stain_result(
    run_dir: Path | str,
    *,
    reviewer: str,
    notes: str,
    families: Iterable[StainFamily | str] | None = None,
    reviewed_at: str | None = None,
) -> StainApproval:
    """Approve selected assay families in the exact current result."""

    root = Path(run_dir)
    result = validate_stain_result(root)
    reviewer = reviewer.strip()
    notes = notes.strip()
    if not reviewer or not notes:
        raise ValueError("reviewer and approval notes must not be blank")
    timestamp = reviewed_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    _validate_timestamp(timestamp)
    available = _result_families(result)
    selected = _selected_families(families, available)
    review = _normalized_review(root, result)
    rows = review["families"]
    assert isinstance(rows, dict)
    for family in selected:
        rows[family.value] = {
            "approved": True,
            "reviewer": reviewer,
            "reviewed_at": timestamp,
            "notes": notes,
        }
    write_json_atomic(root / "stain_review.json", review)
    return _validated_approval(root, result, selected)


def validate_stain_approval(
    run_dir: Path | str,
    *,
    family: StainFamily | str | None = None,
) -> StainApproval:
    """Validate one family, or every quantified family when omitted."""

    root = Path(run_dir)
    result = validate_stain_result(root)
    available = _result_families(result)
    selected = (_coerce_family(family),) if family is not None else available
    unknown = set(selected) - set(available)
    if unknown:
        names = ", ".join(sorted(item.value for item in unknown))
        raise ValueError(f"stain result does not contain quantified families: {names}")
    return _validated_approval(root, result, selected)


def stain_review_status(
    run_dir: Path | str,
    payload: dict[str, object] | None = None,
) -> dict[str, object]:
    """Return path-free family approval status for an exact result."""

    root = Path(run_dir)
    result = validate_stain_result(root, payload)
    review = _normalized_review(root, result)
    family_rows = review["families"]
    assert isinstance(family_rows, dict)
    available = _result_families(result)
    approved: list[str] = []
    for family in available:
        raw = family_rows.get(family.value)
        if isinstance(raw, dict) and raw.get("approved") is True:
            _approved_row(family, raw)
            approved.append(family.value)
    pending = [family.value for family in available if family.value not in approved]
    return {
        "approved": not pending,
        "fingerprint_matches": True,
        "approved_families": approved,
        "pending_families": pending,
    }


def _validated_approval(
    root: Path,
    result: dict[str, object],
    families: tuple[StainFamily, ...],
) -> StainApproval:
    review = _normalized_review(root, result)
    rows = review["families"]
    assert isinstance(rows, dict)
    approvals: list[StainFamilyApproval] = []
    for family in families:
        raw = rows.get(family.value)
        if not isinstance(raw, dict) or raw.get("approved") is not True:
            raise ValueError(f"stain family is not approved: {family.value}")
        reviewer, timestamp, notes = _approved_row(family, raw)
        approvals.append(
            StainFamilyApproval(
                family=family,
                reviewer=reviewer,
                reviewed_at=timestamp,
                notes=notes,
            )
        )
    return StainApproval(
        run_dir=root,
        fingerprint=str(result["fingerprint"]),
        approvals=tuple(approvals),
    )


def _normalized_review(
    root: Path,
    result: dict[str, object],
) -> dict[str, object]:
    fingerprint = str(result["fingerprint"])
    available = _result_families(result)
    default_rows = {
        family.value: {
            "approved": False,
            "reviewer": None,
            "reviewed_at": None,
            "notes": "",
        }
        for family in available
    }
    try:
        payload = json.loads((root / "stain_review.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        payload = None
    if not isinstance(payload, dict) or payload.get("fingerprint") != fingerprint:
        return {
            "schema_version": 2,
            "fingerprint": fingerprint,
            "families": default_rows,
        }
    if payload.get("schema_version") == 1:
        if payload.get("approved") is True:
            row = {
                "approved": True,
                "reviewer": payload.get("reviewer"),
                "reviewed_at": payload.get("reviewed_at"),
                "notes": payload.get("notes"),
            }
            default_rows = {family.value: dict(row) for family in available}
        return {
            "schema_version": 2,
            "fingerprint": fingerprint,
            "families": default_rows,
        }
    if payload.get("schema_version") != 2:
        return {
            "schema_version": 2,
            "fingerprint": fingerprint,
            "families": default_rows,
        }
    raw_rows = payload.get("families")
    if not isinstance(raw_rows, dict):
        return {
            "schema_version": 2,
            "fingerprint": fingerprint,
            "families": default_rows,
        }
    return {
        "schema_version": 2,
        "fingerprint": fingerprint,
        "families": {
            family.value: (
                dict(raw_rows[family.value])
                if isinstance(raw_rows.get(family.value), dict)
                else default_rows[family.value]
            )
            for family in available
        },
    }


def _result_families(result: dict[str, object]) -> tuple[StainFamily, ...]:
    names: set[StainFamily] = set()
    slides = result.get("slides")
    if not isinstance(slides, list):
        raise ValueError("stain result slides must be a list")
    for row in slides:
        if not isinstance(row, dict):
            raise ValueError("stain result slide rows must be objects")
        if row.get("quantified") is False:
            continue
        raw = row.get("family")
        if raw is None:
            continue
        family = _coerce_family(raw)
        if family is not StainFamily.CONTEXT_HE:
            names.add(family)
    if not names:
        raise ValueError("stain result has no quantified assay families")
    return tuple(sorted(names, key=lambda family: family.value))


def _selected_families(
    requested: Iterable[StainFamily | str] | None,
    available: tuple[StainFamily, ...],
) -> tuple[StainFamily, ...]:
    if requested is None:
        return available
    selected = tuple(dict.fromkeys(_coerce_family(item) for item in requested))
    if not selected:
        raise ValueError("at least one stain family must be selected")
    unknown = set(selected) - set(available)
    if unknown:
        names = ", ".join(sorted(item.value for item in unknown))
        raise ValueError(f"stain result does not contain quantified families: {names}")
    return selected


def _coerce_family(value: StainFamily | str | object) -> StainFamily:
    try:
        return value if isinstance(value, StainFamily) else StainFamily(str(value))
    except ValueError as error:
        raise ValueError(f"unsupported stain family: {value}") from error


def _approved_row(
    family: StainFamily,
    row: dict[str, object],
) -> tuple[str, str, str]:
    reviewer = str(row.get("reviewer", "")).strip()
    timestamp = str(row.get("reviewed_at", "")).strip()
    notes = str(row.get("notes", "")).strip()
    if not reviewer or not timestamp or not notes:
        raise ValueError(f"stain family approval is incomplete: {family.value}")
    _validate_timestamp(timestamp)
    return reviewer, timestamp, notes


def _validate_timestamp(value: str) -> None:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError("reviewed_at must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise ValueError("reviewed_at must include a timezone")
