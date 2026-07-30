"""Persistent, fingerprint-bound feedback for registration review evidence."""

from __future__ import annotations

import hashlib
import json
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from histopia._atomic import write_json_atomic

FEEDBACK_LABELS: dict[str, tuple[str, ...]] = {
    "mask": (
        "missing_tissue",
        "extra_debris",
        "glass_border",
        "stain_artifact",
        "internal_holes",
        "excess_whitespace",
        "fragmented_tissue",
        "other",
    ),
    "order": (
        "wrong_position",
        "abrupt_morphology_jump",
        "anchor_issue",
        "duplicate_section",
        "missing_section",
        "other",
    ),
    "alignment": (
        "wrong_orientation",
        "global_shift",
        "rotation_error",
        "scale_error",
        "local_misalignment",
        "poor_overlap",
        "crop_error",
        "wrong_reference",
        "other",
    ),
}
_DECISIONS = frozenset({"accept", "hold", "reject"})


class RegistrationFeedbackStore:
    """Read and append review records without changing scientific artifacts."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.Lock()

    def review(
        self,
        *,
        cohort: str,
        stage: str,
        registration_run: Path,
    ) -> dict[str, object]:
        """Return current evidence identity and latest feedback per slide."""

        evidence = registration_feedback_evidence(registration_run, stage)
        payload = self._load(cohort, stage, str(evidence["fingerprint"]))
        latest = _latest_records(payload["records"])
        return {
            **evidence,
            "cohort": cohort,
            "labels": list(FEEDBACK_LABELS[stage]),
            "feedback": latest,
            "summary": _record_summary(latest),
        }

    def save(
        self,
        request: dict[str, object],
        *,
        registration_run: Path,
    ) -> dict[str, object]:
        """Append one validated review decision and return refreshed feedback."""

        cohort = _required_text(request, "cohort")
        stage = _required_stage(request.get("stage"))
        evidence = registration_feedback_evidence(registration_run, stage)
        fingerprint = _required_text(request, "fingerprint")
        if fingerprint != evidence["fingerprint"]:
            raise ValueError("review evidence changed; reload before saving feedback")
        slide_id = _required_text(request, "slide_id")
        slides = {
            str(row["id"]): row
            for row in evidence["slides"]  # type: ignore[union-attr]
        }
        if slide_id not in slides:
            raise ValueError("review slide is not part of the current evidence")
        decision = _required_text(request, "decision")
        if decision not in _DECISIONS:
            raise ValueError("feedback decision must be accept, hold, or reject")
        labels = request.get("labels", [])
        if (
            not isinstance(labels, list)
            or any(not isinstance(label, str) for label in labels)
            or len(labels) != len(set(labels))
            or not set(labels).issubset(FEEDBACK_LABELS[stage])
        ):
            raise ValueError("feedback contains invalid or duplicate issue labels")
        if decision == "accept" and labels:
            raise ValueError("accepted feedback cannot contain issue labels")
        comment = _optional_text(request.get("comment"), "comment", maximum=4_000)
        reviewer = _required_text(request, "reviewer")
        if decision != "accept" and not labels and not comment:
            raise ValueError("hold or reject feedback requires an issue or comment")

        suggested_order = request.get("suggested_order")
        suggested_rotation = request.get("suggested_quarter_turns_ccw")
        if stage != "order" and (
            suggested_order is not None or suggested_rotation is not None
        ):
            raise ValueError("order corrections are only valid for section order")
        if suggested_order is not None and (
            isinstance(suggested_order, bool)
            or not isinstance(suggested_order, int)
            or not 1 <= suggested_order <= len(slides)
        ):
            raise ValueError("suggested order is outside the section range")
        if suggested_rotation is not None and suggested_rotation not in (0, 1, 2, 3):
            raise ValueError("suggested rotation must be 0, 1, 2, or 3")

        with self._write_lock:
            path = self._path(cohort, stage, fingerprint)
            payload = self._load(cohort, stage, fingerprint)
            timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
            record = {
                "record_id": uuid4().hex,
                "slide_id": slide_id,
                "slide_order": int(slides[slide_id]["order"]),
                "decision": decision,
                "labels": labels,
                "comment": comment,
                "reviewer": reviewer,
                "reviewed_at": timestamp,
            }
            if suggested_order is not None:
                record["suggested_order"] = suggested_order
            if suggested_rotation is not None:
                record["suggested_quarter_turns_ccw"] = suggested_rotation
            records = payload["records"]
            assert isinstance(records, list)
            records.append(record)
            payload["updated_at"] = timestamp
            write_json_atomic(path, payload)
        return self.review(
            cohort=cohort,
            stage=stage,
            registration_run=registration_run,
        )

    def summary(self) -> dict[str, object]:
        """Aggregate latest decisions into model-improvement signals."""

        stage_counts: Counter[str] = Counter()
        decision_counts: Counter[str] = Counter()
        label_counts: Counter[str] = Counter()
        cohort_counts: Counter[str] = Counter()
        reviewed_slides = 0
        for payload in load_registration_feedback(self.root):
            latest = _latest_records(payload["records"])
            for record in latest.values():
                reviewed_slides += 1
                stage = str(payload["stage"])
                cohort = str(payload["cohort"])
                stage_counts[stage] += 1
                cohort_counts[cohort] += 1
                decision_counts[str(record["decision"])] += 1
                label_counts.update(
                    f"{stage}:{label}" for label in record.get("labels", [])
                )
        return {
            "schema_version": 1,
            "reviewed_slides": reviewed_slides,
            "by_stage": dict(sorted(stage_counts.items())),
            "by_decision": dict(sorted(decision_counts.items())),
            "by_issue": dict(sorted(label_counts.items())),
            "by_cohort": dict(sorted(cohort_counts.items())),
        }

    def require_accepted(
        self,
        *,
        cohort: str,
        stage: str,
        registration_run: Path,
    ) -> None:
        """Require one current Accept decision for every displayed slide."""

        review = self.review(
            cohort=cohort,
            stage=stage,
            registration_run=registration_run,
        )
        records = review["feedback"]
        slides = review["slides"]
        assert isinstance(records, dict)
        assert isinstance(slides, list)
        expected = [str(slide["id"]) for slide in slides]
        missing = [slide for slide in expected if slide not in records]
        unresolved = [
            slide
            for slide in expected
            if slide in records and records[slide].get("decision") != "accept"
        ]
        if missing or unresolved:
            parts = []
            if missing:
                parts.append(f"{len(missing)} unreviewed")
            if unresolved:
                parts.append(f"{len(unresolved)} unresolved")
            raise ValueError(f"{stage} feedback is incomplete: " + ", ".join(parts))

    def _path(self, cohort: str, stage: str, fingerprint: str) -> Path:
        return self.root / cohort / stage / f"{fingerprint}.json"

    def _load(
        self,
        cohort: str,
        stage: str,
        fingerprint: str,
    ) -> dict[str, Any]:
        path = self._path(cohort, stage, fingerprint)
        if not path.is_file():
            return {
                "schema_version": 1,
                "cohort": cohort,
                "stage": stage,
                "artifact_fingerprint": fingerprint,
                "updated_at": None,
                "records": [],
            }
        payload = json.loads(path.read_text())
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != 1
            or payload.get("cohort") != cohort
            or payload.get("stage") != stage
            or payload.get("artifact_fingerprint") != fingerprint
            or not isinstance(payload.get("records"), list)
        ):
            raise ValueError("stored registration feedback is invalid")
        return payload


def registration_feedback_evidence(
    registration_run: Path | str,
    stage: str,
) -> dict[str, object]:
    """Describe exact slide evidence using the same viewer fingerprints."""

    run = Path(registration_run)
    stage = _required_stage(stage)
    if stage == "mask":
        from histopia.visualization._viewer import _mask_review_source_payload

        payload = _mask_review_source_payload(run)
        rows = payload.get("slides")
        if (
            not isinstance(rows, list)
            or not rows
            or any(not isinstance(row, dict) for row in rows)
        ):
            raise ValueError("mask review artifact contains invalid slides")
        digest = hashlib.sha256(b"histopia-mask-review-v2")
        slides = []
        for order, row in enumerate(rows, start=1):
            source = row.get("path") or row.get("slide")
            if not isinstance(source, str) or not source:
                raise ValueError("mask review artifact contains an invalid slide")
            name = Path(source).name
            stem = Path(name).stem
            for value in (
                name,
                _sha256(run / "processed" / f"{stem}.thumbnail.png"),
                _sha256(run / "processed" / f"{stem}.mask.png"),
            ):
                digest.update(value.encode())
                digest.update(b"\0")
            slides.append({"id": name, "order": order})
        fingerprint = digest.hexdigest()
    elif stage == "order":
        payload = _json_object(run / "section_order_review.json")
        rows = _slide_rows(payload, "slide")
        fingerprint = _required_text(payload, "fingerprint")
        slides = [
            {"id": Path(str(row["slide"])).name, "order": int(row["order"])}
            for row in rows
        ]
    else:
        path = run / "registration_result.json"
        payload = _json_object(path)
        rows = _slide_rows(payload, "path")
        fingerprint = _sha256(path)
        slides = [
            {"id": Path(str(row["path"])).name, "order": order}
            for order, row in enumerate(rows, start=1)
        ]
    return {
        "schema_version": 1,
        "stage": stage,
        "fingerprint": fingerprint,
        "slides": slides,
    }


def load_registration_feedback(root: Path | str) -> list[dict[str, Any]]:
    """Read all valid feedback files for analysis or model development."""

    feedback_root = Path(root)
    payloads: list[dict[str, Any]] = []
    if not feedback_root.is_dir():
        return payloads
    for path in sorted(feedback_root.glob("*/*/*.json")):
        payload = json.loads(path.read_text())
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != 1
            or not isinstance(payload.get("records"), list)
        ):
            raise ValueError(f"invalid registration feedback: {path}")
        payloads.append(payload)
    return payloads


def summarize_registration_feedback(root: Path | str) -> dict[str, object]:
    """Return path-free issue frequencies from latest per-slide decisions."""

    return RegistrationFeedbackStore(root).summary()


def registration_feedback_rows(root: Path | str) -> list[dict[str, object]]:
    """Flatten the latest per-slide decisions into supervised-learning rows."""

    rows: list[dict[str, object]] = []
    for payload in load_registration_feedback(root):
        for record in _latest_records(payload["records"]).values():
            rows.append(
                {
                    "cohort": str(payload["cohort"]),
                    "stage": str(payload["stage"]),
                    "artifact_fingerprint": str(payload["artifact_fingerprint"]),
                    **record,
                }
            )
    rows.sort(
        key=lambda row: (
            str(row["cohort"]),
            str(row["stage"]),
            int(row["slide_order"]),
        )
    )
    return rows


def _latest_records(records: object) -> dict[str, dict[str, Any]]:
    if not isinstance(records, list):
        raise ValueError("registration feedback records must be a list")
    latest: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("slide_id"), str):
            raise ValueError("registration feedback contains an invalid record")
        latest[record["slide_id"]] = record
    return latest


def _record_summary(records: dict[str, dict[str, Any]]) -> dict[str, object]:
    decisions = Counter(str(record["decision"]) for record in records.values())
    labels = Counter(
        label for record in records.values() for label in record.get("labels", [])
    )
    return {
        "reviewed": len(records),
        "decisions": dict(sorted(decisions.items())),
        "issues": dict(sorted(labels.items())),
    }


def _slide_rows(payload: dict[str, object], field: str) -> list[dict[str, Any]]:
    rows = payload.get("slides")
    if (
        not isinstance(rows, list)
        or not rows
        or any(
            not isinstance(row, dict) or not isinstance(row.get(field), str)
            for row in rows
        )
    ):
        raise ValueError("registration review artifact contains invalid slides")
    return rows


def _json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain an object")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _required_stage(value: object) -> str:
    if not isinstance(value, str) or value not in FEEDBACK_LABELS:
        raise ValueError("feedback stage must be mask, order, or alignment")
    return value


def _required_text(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must not be blank")
    return value.strip()


def _optional_text(value: object, name: str, *, maximum: int) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{name} must be text")
    value = value.strip()
    if len(value) > maximum:
        raise ValueError(f"{name} must contain at most {maximum} characters")
    return value
