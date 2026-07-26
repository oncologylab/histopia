from __future__ import annotations

import json
from pathlib import Path

import pytest

from histopia.visualization._registration_state import (
    current_registration_review_stages,
)


def _write_performance(
    run: Path,
    *,
    status: str,
    stages: dict[str, dict[str, str]],
) -> None:
    run.mkdir(parents=True, exist_ok=True)
    (run / "registration_performance.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "workflow": "registration",
                "observational_only": True,
                "status": status,
                "stages": stages,
            }
        )
    )


def test_missing_performance_report_preserves_legacy_review_discovery(
    tmp_path: Path,
) -> None:
    assert current_registration_review_stages(tmp_path) == frozenset(
        {"mask", "order", "alignment"}
    )


@pytest.mark.parametrize(
    "payload",
    (
        "not JSON",
        json.dumps({"schema_version": 2}),
        json.dumps(
            {
                "schema_version": 1,
                "workflow": "registration",
                "observational_only": True,
                "status": "completed",
                "stages": {"unknown": {"status": "completed"}},
            }
        ),
    ),
)
def test_present_invalid_performance_report_fails_closed(
    tmp_path: Path,
    payload: str,
) -> None:
    (tmp_path / "registration_performance.json").write_text(payload)

    with pytest.raises(ValueError, match="refusing stale review stages"):
        current_registration_review_stages(tmp_path)


def test_current_review_stages_are_derived_from_completed_stage_records(
    tmp_path: Path,
) -> None:
    _write_performance(
        tmp_path,
        status="completed",
        stages={
            "mask_review": {"status": "completed"},
            "section_ordering": {"status": "completed"},
        },
    )

    assert current_registration_review_stages(tmp_path) == frozenset({"mask", "order"})

    _write_performance(
        tmp_path,
        status="completed",
        stages={
            "mask_review": {"status": "completed"},
            "section_ordering": {"status": "completed"},
            "result_write": {"status": "completed"},
        },
    )

    assert current_registration_review_stages(tmp_path) == frozenset(
        {"mask", "order", "alignment"}
    )
