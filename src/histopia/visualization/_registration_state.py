"""Resolve which registration-review artifacts belong to the latest run."""

from __future__ import annotations

import json
from pathlib import Path

from histopia.registration._performance import load_performance_report

_ALL_REVIEW_STAGES = frozenset({"mask", "order", "alignment"})
_FINISHED_STAGE_STATUSES = frozenset({"completed", "review_required"})


def current_registration_review_stages(
    registration_run: Path | str,
) -> frozenset[str]:
    """Return stages reached by the latest observable registration execution.

    Runs created before performance telemetry retain legacy artifact-based
    behavior only when the telemetry file is absent. A present invalid report
    fails closed so downstream files left by an older execution cannot appear
    in a newly prepared review.
    """

    report_path = Path(registration_run) / "registration_performance.json"
    try:
        report = load_performance_report(report_path)
    except FileNotFoundError:
        return _ALL_REVIEW_STAGES
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(
            "registration_performance.json is invalid; refusing stale review stages"
        ) from exc

    stages = report.get("stages")
    if not isinstance(stages, dict):
        raise ValueError(
            "registration_performance.json is invalid; refusing stale review stages"
        )
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
