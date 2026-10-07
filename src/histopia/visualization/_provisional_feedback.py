"""Fingerprint-bound provisional review for stain and cell artifacts."""

from __future__ import annotations

import json
import re
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from histopia._atomic import write_json_atomic

_DIGEST_RE = re.compile(r"[0-9a-f]{64}")
_DECISIONS = frozenset({"accept", "hold", "reject"})
_LABELS = {
    "stain": frozenset(
        {
            "non_target_nuclei",
            "missing_target_signal",
            "glass_or_debris",
            "invalid_tissue_support",
            "spatial_misalignment",
            "tile_seam",
            "other",
        }
    ),
    "cells": frozenset(
        {
            "missing_cells",
            "false_cells",
            "merged_cells",
            "split_cells",
            "boundary_too_tight",
            "boundary_too_broad",
            "tile_seam",
            "debris_or_necrosis",
            "other",
        }
    ),
}


class ProvisionalFeedbackStore:
    """Append review observations without changing scientific approvals."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.Lock()

    def review(
        self,
        *,
        cohort: str,
        stage: str,
        run: Path,
    ) -> dict[str, object]:
        """Return current artifact identity and latest provisional decisions."""

        evidence = provisional_feedback_evidence(run, stage)
        fingerprint = str(evidence["fingerprint"])
        payload = self._load(cohort, stage, fingerprint)
        latest = _latest(payload["records"])
        return {
            **evidence,
            "cohort": cohort,
            "labels": sorted(_LABELS[stage]),
            "feedback": latest,
            "summary": _summary(latest),
            "provisional": True,
        }

    def save(self, request: dict[str, object], *, run: Path) -> dict[str, object]:
        """Append one decision bound to the exact current result fingerprint."""

        cohort = _required_text(request, "cohort")
        stage = _stage(request.get("stage"))
        evidence = provisional_feedback_evidence(run, stage)
        fingerprint = _required_text(request, "fingerprint")
        if fingerprint != evidence["fingerprint"]:
            raise ValueError("review artifact changed; reload before saving")
        slide_id = _required_text(request, "slide_id")
        slides = {str(row["id"]): row for row in evidence["slides"]}  # type: ignore[index]
        if slide_id not in slides:
            raise ValueError("review slide is not part of the current artifact")
        decision = _required_text(request, "decision")
        if decision not in _DECISIONS:
            raise ValueError("decision must be accept, hold, or reject")
        labels = request.get("labels", [])
        if (
            not isinstance(labels, list)
            or any(not isinstance(label, str) for label in labels)
            or len(labels) != len(set(labels))
            or not set(labels).issubset(_LABELS[stage])
        ):
            raise ValueError("provisional feedback labels are invalid")
        checks = request.get("checks", {})
        if not isinstance(checks, dict) or any(
            not isinstance(key, str) or not isinstance(value, bool)
            for key, value in checks.items()
        ):
            raise ValueError("provisional feedback checks must map text to booleans")
        reviewer = _required_text(request, "reviewer")
        comment = _optional_text(request.get("comment"), maximum=4_000)
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
                "checks": dict(sorted(checks.items())),
                "comment": comment,
                "reviewer": reviewer,
                "reviewed_at": timestamp,
                "provisional": True,
            }
            records = payload["records"]
            assert isinstance(records, list)
            records.append(record)
            payload["updated_at"] = timestamp
            write_json_atomic(path, payload)
        return self.review(cohort=cohort, stage=stage, run=run)

    def summary(self) -> dict[str, object]:
        """Aggregate latest provisional findings without exposing paths."""

        decisions: Counter[str] = Counter()
        issues: Counter[str] = Counter()
        cohorts: Counter[str] = Counter()
        reviewed = 0
        for path in sorted(self.root.glob("*/*/*.json")):
            payload = json.loads(path.read_text())
            latest = _latest(payload.get("records", []))
            for record in latest.values():
                reviewed += 1
                decisions[str(record["decision"])] += 1
                cohorts[str(payload["cohort"])] += 1
                issues.update(
                    f"{payload['stage']}:{label}" for label in record.get("labels", [])
                )
        return {
            "schema_version": 1,
            "provisional": True,
            "reviewed_slides": reviewed,
            "by_decision": dict(sorted(decisions.items())),
            "by_issue": dict(sorted(issues.items())),
            "by_cohort": dict(sorted(cohorts.items())),
        }

    def _path(self, cohort: str, stage: str, fingerprint: str) -> Path:
        return self.root / cohort / stage / f"{fingerprint}.json"

    def _load(self, cohort: str, stage: str, fingerprint: str) -> dict[str, Any]:
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
            raise ValueError("stored provisional feedback is invalid")
        return payload


def provisional_feedback_evidence(run: Path | str, stage: str) -> dict[str, object]:
    """Return path-free identities for one sealed stain or cell result."""

    root = Path(run)
    stage = _stage(stage)
    if stage == "stain":
        from histopia.stain._result_validation import validate_stain_result

        result = validate_stain_result(root)
        rows = result.get("slides")
        artifacts = result.get("artifacts")
        if not isinstance(rows, list) or not isinstance(artifacts, dict):
            raise ValueError("stain review artifact is incomplete")
        slides = []
        for row in rows:
            if not isinstance(row, dict) or row.get("quantified") is not True:
                continue
            relative = row.get("map")
            digest = artifacts.get(relative) if isinstance(relative, str) else None
            if not isinstance(digest, str) or not _DIGEST_RE.fullmatch(digest):
                raise ValueError("stain review map digest is invalid")
            slides.append(
                {
                    "id": str(row["id"]),
                    "order": int(row["order"]),
                    "family": str(row["family"]),
                    "artifact_digest": digest,
                    "selected_source": (
                        "corrected"
                        if dict(row.get("qc", {})).get("correction_accepted")
                        else "raw"
                    ),
                }
            )
    else:
        from histopia.cells._result import validate_cell_result

        result = validate_cell_result(root)
        rows = result.get("slides")
        artifacts = result.get("artifacts")
        if not isinstance(rows, list) or not isinstance(artifacts, dict):
            raise ValueError("cell review artifact is incomplete")
        slides = []
        for order, row in enumerate(rows, start=1):
            if not isinstance(row, dict):
                raise ValueError("cell review slide row is invalid")
            relative = row.get("labels")
            digest = artifacts.get(relative) if isinstance(relative, str) else None
            if not isinstance(digest, str) or not _DIGEST_RE.fullmatch(digest):
                raise ValueError("cell review label digest is invalid")
            slides.append(
                {
                    "id": str(row["section"]),
                    "order": order,
                    "slide": str(row["slide"]),
                    "artifact_digest": digest,
                }
            )
    fingerprint = result.get("fingerprint")
    if not isinstance(fingerprint, str) or not _DIGEST_RE.fullmatch(fingerprint):
        raise ValueError(f"{stage} review fingerprint is invalid")
    if not slides:
        raise ValueError(f"{stage} review contains no reviewable slides")
    return {
        "schema_version": 1,
        "stage": stage,
        "fingerprint": fingerprint,
        "slides": slides,
    }


def _stage(value: object) -> str:
    if not isinstance(value, str) or value not in _LABELS:
        raise ValueError("provisional stage must be stain or cells")
    return value


def _required_text(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must not be blank")
    return value.strip()


def _optional_text(value: object, *, maximum: int) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError("comment must be text")
    stripped = value.strip()
    if len(stripped) > maximum:
        raise ValueError(f"comment must not exceed {maximum} characters")
    return stripped


def _latest(records: object) -> dict[str, dict[str, object]]:
    if not isinstance(records, list):
        raise ValueError("provisional feedback records must be a list")
    latest: dict[str, dict[str, object]] = {}
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("slide_id"), str):
            raise ValueError("provisional feedback record is invalid")
        latest[str(record["slide_id"])] = record
    return latest


def _summary(records: dict[str, dict[str, object]]) -> dict[str, int]:
    counts = Counter(str(record["decision"]) for record in records.values())
    return {
        "reviewed": len(records),
        "accept": counts["accept"],
        "hold": counts["hold"],
        "reject": counts["reject"],
    }
