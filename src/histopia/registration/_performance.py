"""Atomic observational performance records for registration workflows."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from histopia._atomic import write_json_atomic

PERFORMANCE_FILENAME = "registration_performance.json"
_STAGES = frozenset(
    {
        "slide_discovery",
        "thumbnail_load",
        "mask_preparation",
        "mask_review",
        "orientation_and_crop",
        "section_ordering",
        "rigid_alignment",
        "refinement_and_metrics",
        "review_rendering",
        "full_resolution_warp",
        "result_write",
    }
)
_STATUSES = frozenset(
    {"running", "completed", "review_required", "failed", "interrupted"}
)


def utc_timestamp() -> str:
    """Return one second-resolution UTC timestamp."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def elapsed_seconds(started: float) -> float:
    """Return one stable, non-negative elapsed duration."""

    return round(max(0.0, time.perf_counter() - started), 6)


class RegistrationPerformance:
    """Checkpoint one registration execution without affecting its result."""

    def __init__(self, output_dir: Path | str, controls: dict[str, object]) -> None:
        self.path = Path(output_dir) / PERFORMANCE_FILENAME
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._started = time.perf_counter()
        self._stage_started: float | None = None
        self._current_stage: str | None = None
        self._report: dict[str, object] = {
            "schema_version": 1,
            "workflow": "registration",
            "observational_only": True,
            "fingerprint_scope": "excluded-from-scientific-result-and-approval",
            "status": "running",
            "started_at": utc_timestamp(),
            "updated_at": utc_timestamp(),
            "elapsed_seconds": 0.0,
            "controls": dict(controls),
            "stages": {},
        }
        self._checkpoint()

    def start_stage(
        self,
        stage: str,
        *,
        details: dict[str, object] | None = None,
    ) -> None:
        """Complete the prior stage and atomically start the next one."""

        if stage not in _STAGES:
            raise ValueError("unsupported registration performance stage")
        self._complete_current_stage()
        self._current_stage = stage
        self._stage_started = time.perf_counter()
        stages = self._stages()
        stages[stage] = {
            "status": "running",
            "started_at": utc_timestamp(),
            **(dict(details) if details is not None else {}),
        }
        self._report["current_stage"] = stage
        self._checkpoint()

    def update(self, **values: object) -> None:
        """Update root execution facts and checkpoint the active stage."""

        self._report.update(values)
        self._checkpoint()

    def complete(self, **values: object) -> None:
        """Mark the execution complete after finishing its active stage."""

        self._complete_current_stage()
        self._finish("completed", values)

    def review_required(
        self,
        stage: str,
        review_artifact: str,
        *,
        pending_slide_count: int = 0,
    ) -> None:
        """Record an intentional human-review pause rather than a failure."""

        self._mark_current_stage("review_required")
        self._finish(
            "review_required",
            {
                "review_stage": stage,
                "review_artifact": review_artifact,
                "pending_slide_count": pending_slide_count,
            },
        )

    def fail(self, exc: BaseException) -> None:
        """Record cancellation or failure while preserving the exception."""

        status = (
            "interrupted"
            if isinstance(exc, (KeyboardInterrupt, SystemExit))
            else "failed"
        )
        self._mark_current_stage(status)
        self._finish(status, {"failure_type": type(exc).__name__})

    def _complete_current_stage(self) -> None:
        self._mark_current_stage("completed")

    def _mark_current_stage(self, status: str) -> None:
        if self._current_stage is None or self._stage_started is None:
            return
        stage = self._stages()[self._current_stage]
        stage["status"] = status
        stage["elapsed_seconds"] = elapsed_seconds(self._stage_started)
        stage["completed_at"] = utc_timestamp()
        self._stage_started = None

    def _finish(self, status: str, values: dict[str, object]) -> None:
        if status not in _STATUSES - {"running"}:
            raise ValueError("invalid registration performance status")
        self._report.update(values)
        self._report["status"] = status
        self._report["completed_at"] = utc_timestamp()
        self._report.pop("current_stage", None)
        self._checkpoint()

    def _stages(self) -> dict[str, dict[str, object]]:
        stages = self._report["stages"]
        if not isinstance(stages, dict):
            raise TypeError("registration performance stages must be a dictionary")
        return stages

    def _checkpoint(self) -> None:
        self._report["elapsed_seconds"] = elapsed_seconds(self._started)
        self._report["updated_at"] = utc_timestamp()
        try:
            write_json_atomic(self.path, self._report)
        except OSError:
            # Observational telemetry must not determine scientific execution.
            pass


def load_performance_report(path: Path | str) -> dict[str, object]:
    """Load and validate a registration performance report."""

    report = json.loads(Path(path).read_text())
    if (
        not isinstance(report, dict)
        or report.get("schema_version") != 1
        or report.get("workflow") != "registration"
        or report.get("observational_only") is not True
        or report.get("status") not in _STATUSES
    ):
        raise ValueError("invalid registration performance report")
    stages = report.get("stages")
    if not isinstance(stages, dict) or any(name not in _STAGES for name in stages):
        raise ValueError("registration performance report has unsupported stages")
    if any(
        not isinstance(stage, dict) or stage.get("status") not in _STATUSES
        for stage in stages.values()
    ):
        raise ValueError("invalid registration performance stage")
    return report
