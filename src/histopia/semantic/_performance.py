"""Atomic observational performance records for semantic workflows."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from histopia._atomic import write_json_atomic

PERFORMANCE_FILENAME = "semantic_performance.json"
_STAGES = frozenset({"extraction", "fit"})


def utc_timestamp() -> str:
    """Return one second-resolution UTC timestamp."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def elapsed_seconds(started: float) -> float:
    """Return one stable, non-negative elapsed duration."""

    return round(max(0.0, time.perf_counter() - started), 6)


def write_performance_stage(
    output_dir: Path | str,
    stage: str,
    payload: dict[str, object],
    *,
    clear_stages: tuple[str, ...] = (),
) -> Path:
    """Atomically update one observational stage without sealing it as science."""

    if stage not in _STAGES or any(value not in _STAGES for value in clear_stages):
        raise ValueError("unsupported semantic performance stage")
    if not isinstance(payload, dict):
        raise TypeError("semantic performance payload must be a dictionary")
    output_dir = Path(output_dir)
    path = output_dir / PERFORMANCE_FILENAME
    report = _load_report(path)
    for name in clear_stages:
        report.pop(name, None)
    report[stage] = dict(payload)
    report["updated_at"] = utc_timestamp()
    return write_json_atomic(path, report)


def load_performance_report(path: Path | str) -> dict[str, object]:
    """Load and validate a semantic performance report."""

    report = json.loads(Path(path).read_text())
    if (
        not isinstance(report, dict)
        or report.get("schema_version") != 1
        or report.get("observational_only") is not True
    ):
        raise ValueError("invalid semantic performance report")
    if any(name not in _STAGES for name in report if name not in _ROOT_KEYS):
        raise ValueError("semantic performance report has an unsupported stage")
    for stage in _STAGES & report.keys():
        if not isinstance(report[stage], dict):
            raise ValueError("semantic performance stage must be an object")
    return report


_ROOT_KEYS = frozenset(
    {
        "schema_version",
        "observational_only",
        "fingerprint_scope",
        "updated_at",
    }
)


def _load_report(path: Path) -> dict[str, object]:
    try:
        return load_performance_report(path)
    except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError, ValueError):
        return {
            "schema_version": 1,
            "observational_only": True,
            "fingerprint_scope": "excluded-from-scientific-result-and-approval",
        }
