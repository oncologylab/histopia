"""Resolve which registration-review artifacts belong to the latest run."""

from __future__ import annotations

import json
from pathlib import Path

_ALL_REVIEW_STAGES = frozenset({"mask", "order", "alignment"})
_FINISHED_STAGE_STATUSES = frozenset({"completed", "review_required"})
_PERFORMANCE_STATUSES = frozenset(
    {"running", "completed", "review_required", "failed", "interrupted"}
)


def current_registration_review_stages(
    registration_run: Path | str,
) -> frozenset[str]:
    """Return stages reached by the latest observable registration execution.

    Runs created before performance telemetry retain legacy artifact-based
    behavior. A valid current report prevents downstream files left by an older
    execution from appearing in a newly prepared review.
    """

    report_path = Path(registration_run) / "registration_performance.json"
    try:
        report = json.loads(report_path.read_text())
    except (OSError, json.JSONDecodeError):
        return _ALL_REVIEW_STAGES
    if (
        not isinstance(report, dict)
        or report.get("schema_version") != 1
        or report.get("workflow") != "registration"
        or report.get("observational_only") is not True
        or report.get("status") not in _PERFORMANCE_STATUSES
    ):
        return _ALL_REVIEW_STAGES

    status = report.get("status")
    if status == "completed":
        return _ALL_REVIEW_STAGES
    if status == "review_required":
        review_stage = report.get("review_stage")
        if review_stage == "masks":
            return frozenset({"mask"})
        if review_stage == "order":
            return frozenset({"mask", "order"})

    stages = report.get("stages")
    if not isinstance(stages, dict):
        return frozenset()
    reached: set[str] = set()
    mask_review = stages.get("mask_review")
    if (
        isinstance(mask_review, dict)
        and mask_review.get("status") in _FINISHED_STAGE_STATUSES
    ):
        reached.add("mask")
    ordering = stages.get("section_ordering")
    if (
        isinstance(ordering, dict)
        and ordering.get("status") in _FINISHED_STAGE_STATUSES
    ):
        reached.update({"mask", "order"})
    result = stages.get("result_write")
    if isinstance(result, dict) and result.get("status") == "completed":
        reached.update(_ALL_REVIEW_STAGES)
    return frozenset(reached)


def registration_artifact_slide_names(
    path: Path | str,
    *,
    field: str,
) -> frozenset[str] | None:
    """Return unique source filenames from one optional registration artifact."""

    artifact = Path(path)
    if not artifact.is_file():
        return None
    try:
        payload = json.loads(artifact.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{artifact.name} is not valid JSON") from exc
    rows = payload.get("slides") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{artifact.name} contains no slides")
    names: list[str] = []
    for row in rows:
        value = row.get(field) if isinstance(row, dict) else None
        if not isinstance(value, str) or not value:
            raise ValueError(f"{artifact.name} contains an invalid slide")
        names.append(Path(value).name)
    if len(names) != len(set(names)):
        raise ValueError(f"{artifact.name} contains duplicate slides")
    return frozenset(names)
