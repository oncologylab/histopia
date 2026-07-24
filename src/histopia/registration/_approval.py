"""Fingerprint-bound approval of completed registration runs."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RegistrationApproval:
    """Validated approval metadata for one completed registration run."""

    run_dir: Path
    slide_count: int
    order_fingerprint: str
    reviewer: str
    reviewed_at: str
    registration_result_sha256: str


@dataclass(frozen=True, slots=True)
class MaskReviewApproval:
    """Approval metadata for one exact mask-review stage."""

    run_dir: Path
    slide_count: int
    mask_fingerprint: str
    reviewer: str
    reviewed_at: str


@dataclass(frozen=True, slots=True)
class SectionOrderApproval:
    """Approval metadata for one exact section-order proposal."""

    run_dir: Path
    slide_count: int
    order_fingerprint: str
    reviewer: str
    reviewed_at: str


def approve_mask_review(
    run_dir: Path | str,
    *,
    reviewer: str,
    notes: str,
    reviewed_at: str | None = None,
) -> MaskReviewApproval:
    """Approve every exact mask fingerprint in a prepared review stage."""

    root = Path(run_dir)
    reviewer, notes, timestamp = _validated_review_metadata(
        reviewer, notes, reviewed_at
    )
    review_path = root / "mask_review.json"
    payload = _load_object(review_path)
    if payload.get("schema_version") != 2:
        raise ValueError("mask review must use schema version 2")
    rows = _object_rows(payload, "slides", review_path)
    indexed = _unique_rows_by_name(rows, "slide", review_path)
    digest = hashlib.sha256(b"histopia-mask-review-approval-v1")
    for name in sorted(indexed):
        row = indexed[name]
        fingerprint = _required_string(row, "thumbnail_sha256", review_path)
        status = row.get("status")
        if status == "rejected":
            raise ValueError(f"mask review is rejected: {name}")
        if status not in {"pending", "auto_pass", "override_pass"}:
            raise ValueError(f"mask review has invalid status: {name}")
        _require_processed_review_artifacts(root, name)
        override_path = _resolved_override_path(root, row, review_path)
        row["status"] = "override_pass" if override_path is not None else "auto_pass"
        row["reviewer"] = reviewer
        row["notes"] = notes
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(fingerprint.encode())
        digest.update(b"\0")
    payload["reviewer"] = reviewer
    payload["reviewed_at"] = timestamp
    payload["notes"] = notes
    payload["fingerprint"] = digest.hexdigest()
    _write_json_atomic(review_path, payload)
    return MaskReviewApproval(
        run_dir=root,
        slide_count=len(rows),
        mask_fingerprint=digest.hexdigest(),
        reviewer=reviewer,
        reviewed_at=timestamp,
    )


def approve_section_order(
    run_dir: Path | str,
    *,
    reviewer: str,
    notes: str,
    reviewed_at: str | None = None,
) -> SectionOrderApproval:
    """Approve one exact prepared section-order fingerprint."""

    root = Path(run_dir)
    reviewer, notes, timestamp = _validated_review_metadata(
        reviewer, notes, reviewed_at
    )
    mask_path = root / "mask_review.json"
    mask_payload = _load_object(mask_path)
    mask_rows = _object_rows(mask_payload, "slides", mask_path)
    masks_by_name = _unique_rows_by_name(mask_rows, "slide", mask_path)
    unapproved = [
        name
        for name, row in masks_by_name.items()
        if row.get("status") not in {"auto_pass", "override_pass"}
    ]
    if unapproved:
        raise ValueError(
            "section order requires approved masks: " + ", ".join(sorted(unapproved))
        )

    order_path = root / "section_order_review.json"
    payload = _load_object(order_path)
    if payload.get("schema_version") != 3:
        raise ValueError("section order review must use schema version 3")
    fingerprint = _required_string(payload, "fingerprint", order_path)
    rows = _object_rows(payload, "slides", order_path)
    ordered_names: list[str] = []
    for expected_order, row in enumerate(rows, start=1):
        if row.get("order") != expected_order:
            raise ValueError("section order review positions must be consecutive")
        name = _required_string(row, "slide", order_path)
        if name in ordered_names:
            raise ValueError(f"section order review contains duplicate slide: {name}")
        ordered_names.append(name)
        _require_processed_review_artifacts(root, name)
    if set(ordered_names) != set(masks_by_name):
        raise ValueError("section order slides do not exactly match approved masks")
    inputs = payload.get("input_fingerprints")
    if (
        not isinstance(inputs, dict)
        or set(inputs) != set(ordered_names)
        or any(not isinstance(value, str) or not value for value in inputs.values())
    ):
        raise ValueError("section order input fingerprints are incomplete")

    payload["approved"] = True
    payload["reviewer"] = reviewer
    payload["reviewed_at"] = timestamp
    payload["notes"] = notes
    _write_json_atomic(order_path, payload)
    return SectionOrderApproval(
        run_dir=root,
        slide_count=len(rows),
        order_fingerprint=fingerprint,
        reviewer=reviewer,
        reviewed_at=timestamp,
    )


def approve_registration_run(
    run_dir: Path | str,
    *,
    reviewer: str,
    notes: str,
    reviewed_at: str | None = None,
) -> RegistrationApproval:
    """Approve exact masks and order, then seal a completed run atomically."""

    root = Path(run_dir)
    reviewer, notes, timestamp = _validated_review_metadata(
        reviewer, notes, reviewed_at
    )

    result_path = root / "registration_result.json"
    mask_path = root / "mask_review.json"
    order_path = root / "section_order_review.json"
    result = _load_object(result_path)
    mask_review = _load_object(mask_path)
    order_review = _load_object(order_path)

    result_slides = _object_rows(result, "slides", result_path)
    mask_slides = _object_rows(mask_review, "slides", mask_path)
    order_slides = _object_rows(order_review, "slides", order_path)
    result_by_name = _unique_rows_by_name(result_slides, "path", result_path)
    masks_by_name = _unique_rows_by_name(mask_slides, "slide", mask_path)
    ordered_names = [_required_string(row, "slide", order_path) for row in order_slides]
    result_names = [
        Path(_required_string(row, "path", result_path)).name for row in result_slides
    ]
    if set(result_by_name) != set(masks_by_name):
        raise ValueError("mask review slides do not exactly match registration result")
    if ordered_names != result_names:
        message = "approved order does not match registration result slide order"
        raise ValueError(message)

    fingerprint = order_review.get("fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError("section order review has no fingerprint")

    for name, row in result_by_name.items():
        mask = row.get("mask")
        if not isinstance(mask, dict) or mask.get("accepted") is not True:
            raise ValueError(f"registration mask is not accepted: {name}")
        embedded = row.get("mask_review")
        if not isinstance(embedded, dict):
            raise ValueError(f"registration result has no mask review: {name}")
        reviewed = masks_by_name[name]
        embedded_hash = embedded.get("thumbnail_sha256")
        reviewed_hash = reviewed.get("thumbnail_sha256")
        if (
            not isinstance(reviewed_hash, str)
            or not reviewed_hash
            or embedded_hash != reviewed_hash
        ):
            raise ValueError(f"mask review fingerprint mismatch: {name}")
        override_path = reviewed.get("override_path")
        if override_path is not None:
            override = Path(override_path) if isinstance(override_path, str) else None
            if override is not None and not override.is_absolute():
                override = root / override
            if override is None or not override.is_file():
                raise ValueError(f"approved mask override is missing: {name}")
            reviewed["status"] = "override_pass"
        else:
            reviewed["status"] = "auto_pass"
        reviewed["reviewer"] = reviewer
        reviewed["notes"] = notes
        row["mask_review"] = dict(reviewed)

    order_review["approved"] = True
    order_review["reviewer"] = reviewer
    order_review["reviewed_at"] = timestamp
    order_review["notes"] = notes
    mask_review["reviewed_at"] = timestamp

    _write_json_atomic(mask_path, mask_review)
    _write_json_atomic(order_path, order_review)
    _write_json_atomic(result_path, result)
    artifact_hashes = {
        path.name: _sha256_file(path) for path in (result_path, mask_path, order_path)
    }
    approval_payload = {
        "schema_version": 1,
        "reviewer": reviewer,
        "reviewed_at": timestamp,
        "notes": notes,
        "slide_count": len(result_slides),
        "order_fingerprint": fingerprint,
        "artifacts": artifact_hashes,
    }
    _write_json_atomic(root / "registration_approval.json", approval_payload)
    return RegistrationApproval(
        run_dir=root,
        slide_count=len(result_slides),
        order_fingerprint=fingerprint,
        reviewer=reviewer,
        reviewed_at=timestamp,
        registration_result_sha256=artifact_hashes[result_path.name],
    )


def validate_registration_approval(
    run_dir: Path | str,
) -> RegistrationApproval:
    """Verify that a registration approval still seals the current artifacts."""

    root = Path(run_dir)
    approval = _load_object(root / "registration_approval.json")
    if approval.get("schema_version") != 1:
        raise ValueError("registration approval must use schema version 1")
    artifacts = approval.get("artifacts")
    expected_names = {
        "registration_result.json",
        "mask_review.json",
        "section_order_review.json",
    }
    if not isinstance(artifacts, dict) or set(artifacts) != expected_names:
        raise ValueError("registration approval artifact manifest is incomplete")
    for name, expected in artifacts.items():
        artifact = root / name
        if not artifact.is_file() or _sha256_file(artifact) != expected:
            raise ValueError(f"registration approval artifact digest mismatch: {name}")

    order_review = _load_object(root / "section_order_review.json")
    fingerprint = approval.get("order_fingerprint")
    if (
        order_review.get("approved") is not True
        or not isinstance(fingerprint, str)
        or order_review.get("fingerprint") != fingerprint
    ):
        raise ValueError("registration approval order fingerprint is stale")
    result = _load_object(root / "registration_result.json")
    slides = _object_rows(result, "slides", root / "registration_result.json")
    slide_count = approval.get("slide_count")
    if not isinstance(slide_count, int) or slide_count != len(slides):
        raise ValueError("registration approval slide count is stale")
    if any(
        not isinstance(row.get("mask_review"), dict)
        or row["mask_review"].get("status") not in {"auto_pass", "override_pass"}
        for row in slides
    ):
        raise ValueError("registration approval contains an unapproved mask")
    reviewer = approval.get("reviewer")
    reviewed_at = approval.get("reviewed_at")
    if not isinstance(reviewer, str) or not reviewer:
        raise ValueError("registration approval reviewer is missing")
    if not isinstance(reviewed_at, str) or not reviewed_at:
        raise ValueError("registration approval timestamp is missing")
    return RegistrationApproval(
        run_dir=root,
        slide_count=slide_count,
        order_fingerprint=fingerprint,
        reviewer=reviewer,
        reviewed_at=reviewed_at,
        registration_result_sha256=str(artifacts["registration_result.json"]),
    )


def _load_object(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _object_rows(
    payload: dict[str, object],
    key: str,
    path: Path,
) -> list[dict[str, object]]:
    rows = payload.get(key)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path.name} contains no {key}")
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{path.name} {key} must contain objects")
    return rows


def _unique_rows_by_name(
    rows: list[dict[str, object]],
    key: str,
    path: Path,
) -> dict[str, dict[str, object]]:
    indexed: dict[str, dict[str, object]] = {}
    for row in rows:
        name = Path(_required_string(row, key, path)).name
        if name in indexed:
            raise ValueError(f"{path.name} contains duplicate slide: {name}")
        indexed[name] = row
    return indexed


def _required_string(row: dict[str, object], key: str, path: Path) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path.name} contains an invalid {key}")
    return value


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
    try:
        parsed_timestamp = datetime.fromisoformat(timestamp)
    except ValueError as error:
        raise ValueError("reviewed_at must be an ISO-8601 timestamp") from error
    if parsed_timestamp.tzinfo is None:
        raise ValueError("reviewed_at must include a timezone")
    return reviewer, notes, timestamp


def _require_processed_review_artifacts(root: Path, slide: str) -> None:
    stem = Path(slide).stem
    for suffix in ("thumbnail.png", "mask.png"):
        artifact = root / "processed" / f"{stem}.{suffix}"
        if not artifact.is_file():
            raise FileNotFoundError(f"mask review artifact is missing: {artifact}")


def _resolved_override_path(
    root: Path,
    row: dict[str, object],
    review_path: Path,
) -> Path | None:
    value = row.get("override_path")
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{review_path.name} contains an invalid override_path")
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        name = _required_string(row, "slide", review_path)
        raise FileNotFoundError(f"approved mask override is missing: {name}")
    return path


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
