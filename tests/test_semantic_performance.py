from __future__ import annotations

import json
from pathlib import Path

import pytest

from histopia.semantic._performance import (
    PERFORMANCE_FILENAME,
    load_performance_report,
    write_performance_stage,
)


def test_performance_stages_are_atomic_observational_records(tmp_path: Path) -> None:
    path = write_performance_stage(
        tmp_path,
        "extraction",
        {"status": "completed", "total_patches": 12},
    )
    write_performance_stage(
        tmp_path,
        "fit",
        {"status": "completed", "fit_threads": 4},
    )

    report = load_performance_report(path)

    assert path == tmp_path / PERFORMANCE_FILENAME
    assert report["observational_only"] is True
    assert report["fingerprint_scope"] == (
        "excluded-from-scientific-result-and-approval"
    )
    assert report["extraction"]["total_patches"] == 12
    assert report["fit"]["fit_threads"] == 4
    assert not tuple(tmp_path.glob(f".{PERFORMANCE_FILENAME}.*.tmp"))


def test_new_extraction_can_clear_stale_fit_and_replace_invalid_report(
    tmp_path: Path,
) -> None:
    path = tmp_path / PERFORMANCE_FILENAME
    path.write_text("{invalid")
    write_performance_stage(tmp_path, "fit", {"status": "completed"})
    write_performance_stage(
        tmp_path,
        "extraction",
        {"status": "running"},
        clear_stages=("fit",),
    )

    report = json.loads(path.read_text())

    assert report["extraction"]["status"] == "running"
    assert "fit" not in report


@pytest.mark.parametrize("stage", ("registration", "", "FIT"))
def test_performance_report_rejects_unknown_stage(tmp_path: Path, stage: str) -> None:
    with pytest.raises(ValueError, match="unsupported"):
        write_performance_stage(tmp_path, stage, {})
