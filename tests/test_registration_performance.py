from __future__ import annotations

import json
from pathlib import Path

import pytest

from histopia.registration._performance import (
    PERFORMANCE_FILENAME,
    RegistrationPerformance,
    load_performance_report,
)


def test_performance_records_completed_stages_atomically(tmp_path: Path) -> None:
    performance = RegistrationPerformance(tmp_path, {"thumbnail_workers": 2})
    performance.start_stage("slide_discovery")
    performance.update(slide_count=3)
    performance.start_stage("thumbnail_load")
    performance.complete(registered_slide_count=3)

    report = load_performance_report(tmp_path / PERFORMANCE_FILENAME)

    assert report["status"] == "completed"
    assert report["observational_only"] is True
    assert report["fingerprint_scope"] == (
        "excluded-from-scientific-result-and-approval"
    )
    assert report["slide_count"] == 3
    assert report["controls"] == {"thumbnail_workers": 2}
    assert report["stages"]["slide_discovery"]["status"] == "completed"
    assert report["stages"]["thumbnail_load"]["status"] == "completed"
    assert not tuple(tmp_path.glob(f".{PERFORMANCE_FILENAME}.*.tmp"))


def test_performance_distinguishes_review_pause_and_interruption(
    tmp_path: Path,
) -> None:
    review = RegistrationPerformance(tmp_path / "review", {})
    review.start_stage("mask_review")
    review.review_required(
        "masks",
        "mask_review.json",
        pending_slide_count=4,
    )
    review_report = load_performance_report(tmp_path / "review" / PERFORMANCE_FILENAME)

    interrupted = RegistrationPerformance(tmp_path / "interrupted", {})
    interrupted.start_stage("rigid_alignment")
    interrupted.fail(KeyboardInterrupt())
    interrupted_report = load_performance_report(
        tmp_path / "interrupted" / PERFORMANCE_FILENAME
    )

    assert review_report["status"] == "review_required"
    assert review_report["review_stage"] == "masks"
    assert review_report["review_artifact"] == "mask_review.json"
    assert review_report["pending_slide_count"] == 4
    assert review_report["stages"]["mask_review"]["status"] == "review_required"
    assert interrupted_report["status"] == "interrupted"
    assert interrupted_report["failure_type"] == "KeyboardInterrupt"
    assert interrupted_report["stages"]["rigid_alignment"]["status"] == "interrupted"


@pytest.mark.parametrize("stage", ("", "fit", "MASK"))
def test_performance_rejects_unknown_stage(tmp_path: Path, stage: str) -> None:
    performance = RegistrationPerformance(tmp_path, {})

    with pytest.raises(ValueError, match="unsupported"):
        performance.start_stage(stage)


def test_performance_loader_rejects_invalid_report(tmp_path: Path) -> None:
    path = tmp_path / PERFORMANCE_FILENAME
    path.write_text(json.dumps({"schema_version": 1, "status": "completed"}))

    with pytest.raises(ValueError, match="invalid"):
        load_performance_report(path)
