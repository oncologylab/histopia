from __future__ import annotations

import hashlib
import json
from pathlib import Path

from histopia.visualization import (
    audit_analysis_completeness,
    write_analysis_completeness,
)
from histopia.visualization._review_api import ReviewRuns


def _seal_registration(root: Path) -> None:
    result = json.loads((root / "registration_result.json").read_text())
    for row in result["slides"]:
        row["mask_review"] = {"status": "auto_pass"}
    (root / "registration_result.json").write_text(json.dumps(result))
    (root / "mask_review.json").write_text(json.dumps({"slides": []}))
    (root / "section_order_review.json").write_text(
        json.dumps({"approved": True, "fingerprint": "order"})
    )
    artifacts = {}
    for name in (
        "registration_result.json",
        "mask_review.json",
        "section_order_review.json",
    ):
        artifacts[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    (root / "registration_approval.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "reviewer": "synthetic",
                "reviewed_at": "2026-01-01T00:00:00+00:00",
                "slide_count": len(result["slides"]),
                "order_fingerprint": "order",
                "artifacts": artifacts,
            }
        )
    )


def test_completeness_reports_missing_stain_and_reference_cells_without_paths(
    tmp_path: Path,
) -> None:
    registration = tmp_path / "private-registration"
    registration.mkdir()
    (registration / "registration_result.json").write_text(
        json.dumps(
            {
                "slides": [
                    {"path": "/private/one.ndpi", "is_reference": False},
                    {"path": "/private/mouse_panc_HE.ndpi", "is_reference": True},
                    {"path": "/private/three.ndpi", "is_reference": False},
                ]
            }
        )
    )
    stain = tmp_path / "private-stain"
    stain.mkdir()
    (stain / "stain_result.json").write_text(
        json.dumps(
            {
                "fingerprint": "s" * 64,
                "slides": [
                    {"order": 1, "quantified": True},
                    {"order": 2, "quantified": False},
                    {"order": 3, "quantified": False},
                ],
            }
        )
    )

    report = audit_analysis_completeness(
        {
            "mouse": ReviewRuns(
                registration=registration,
                stain=stain,
            )
        }
    )
    payload = report.to_json_dict()

    assert report.complete is False
    assert payload["summary"] == {
        "cohort_count": 1,
        "complete_cohorts": 0,
        "expected_sections": 3,
        "registered_sections": 3,
        "approved_registration_cohorts": 0,
        "expected_stain_sections": 2,
        "quantified_stain_sections": 1,
        "expected_cell_sections": 3,
        "segmented_cell_sections": 0,
        "approved_cell_cohorts": 0,
    }
    assert payload["cohorts"][0]["stain"]["missing_orders"] == [3]
    assert payload["cohorts"][0]["cells"]["missing_sections"] == [
        "001",
        "002",
        "003",
    ]
    assert str(tmp_path) not in json.dumps(payload)


def test_completeness_accepts_exact_full_coverage_and_writes_atomically(
    tmp_path: Path,
) -> None:
    registration = tmp_path / "registration"
    registration.mkdir()
    (registration / "registration_result.json").write_text(
        json.dumps(
            {
                "slides": [
                    {"path": "mouse_panc_HE.ndpi", "is_reference": True},
                    {"path": "dab.ndpi", "is_reference": False},
                ]
            }
        )
    )
    _seal_registration(registration)
    stain = tmp_path / "stain"
    stain.mkdir()
    (stain / "stain_result.json").write_text(
        json.dumps(
            {
                "fingerprint": "a" * 64,
                "slides": [
                    {"order": 1, "quantified": False},
                    {"order": 2, "quantified": True},
                ],
            }
        )
    )
    cells = tmp_path / "cells"
    cells.mkdir()
    (cells / "cell_result.json").write_text(
        json.dumps(
            {
                "fingerprint": "b" * 64,
                "slides": [{"section": "001"}, {"section": "002"}],
            }
        )
    )
    (cells / "preflight.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "slides": [{"section": "001"}, {"section": "002"}],
            }
        )
    )
    (cells / "cell_review.json").write_text(
        json.dumps({"fingerprint": "b" * 64, "approved": True})
    )

    report = audit_analysis_completeness(
        {
            "mouse": ReviewRuns(
                registration=registration,
                stain=stain,
                cells=cells,
            )
        }
    )

    assert report.complete is True
    assert report.cohorts[0].cell_approved is True
    output = write_analysis_completeness(report, tmp_path / "report.json")
    assert json.loads(output.read_text())["complete"] is True


def test_completeness_requires_current_cell_approval(tmp_path: Path) -> None:
    registration = tmp_path / "registration"
    registration.mkdir()
    (registration / "registration_result.json").write_text(
        json.dumps(
            {
                "slides": [
                    {"path": "mouse_panc_HE.ndpi", "is_reference": True},
                    {"path": "dab.ndpi", "is_reference": False},
                ]
            }
        )
    )
    _seal_registration(registration)
    stain = tmp_path / "stain"
    stain.mkdir()
    (stain / "stain_result.json").write_text(
        json.dumps(
            {
                "slides": [
                    {"order": 1, "quantified": False},
                    {"order": 2, "quantified": True},
                ]
            }
        )
    )
    cells = tmp_path / "cells"
    cells.mkdir()
    (cells / "cell_result.json").write_text(
        json.dumps(
            {
                "fingerprint": "c" * 64,
                "slides": [{"section": "001"}, {"section": "002"}],
            }
        )
    )
    (cells / "preflight.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "slides": [{"section": "001"}, {"section": "002"}],
            }
        )
    )
    (cells / "cell_review.json").write_text(
        json.dumps({"fingerprint": "stale", "approved": True})
    )

    report = audit_analysis_completeness(
        {
            "mouse": ReviewRuns(
                registration=registration,
                stain=stain,
                cells=cells,
            )
        }
    )

    assert report.complete is False
    assert report.cohorts[0].cell_segmented == 2
    assert report.cohorts[0].cell_approved is False
    assert report.to_json_dict()["cohorts"][0]["cells"]["approved"] is False


def test_completeness_rejects_legacy_partial_cell_result_as_scope(
    tmp_path: Path,
) -> None:
    registration = tmp_path / "registration"
    registration.mkdir()
    (registration / "registration_result.json").write_text(
        json.dumps(
            {
                "slides": [
                    {"path": "mouse_panc_HE.ndpi", "is_reference": False},
                    {"path": "mouse_panc_Nr2f2.ndpi", "is_reference": True},
                    {"path": "mouse_panc_CK19.ndpi", "is_reference": False},
                ]
            }
        )
    )
    _seal_registration(registration)
    stain = tmp_path / "stain"
    stain.mkdir()
    (stain / "stain_result.json").write_text(
        json.dumps(
            {
                "fingerprint": "a" * 64,
                "slides": [
                    {"order": 1, "quantified": False},
                    {"order": 2, "quantified": True},
                    {"order": 3, "quantified": True},
                ],
            }
        )
    )
    cells = tmp_path / "cells"
    cells.mkdir()
    (cells / "cell_result.json").write_text(
        json.dumps(
            {
                "fingerprint": "b" * 64,
                "slides": [{"section": "003"}],
            }
        )
    )

    report = audit_analysis_completeness(
        {
            "mouse": ReviewRuns(
                registration=registration,
                stain=stain,
                cells=cells,
            )
        }
    )

    assert report.complete is False
    assert report.cohorts[0].cell_missing_sections == ("001", "002")


def test_completeness_honors_explicit_stain_exclusion(tmp_path: Path) -> None:
    registration = tmp_path / "registration"
    registration.mkdir()
    (registration / "registration_result.json").write_text(
        json.dumps(
            {
                "slides": [
                    {"path": "mouse_panc_HE.ndpi", "is_reference": True},
                    {"path": "B-catenin.ndpi", "is_reference": False},
                ]
            }
        )
    )
    stain = tmp_path / "stain"
    stain.mkdir()
    (stain / "stain_result.json").write_text(
        json.dumps(
            {
                "slides": [
                    {"order": 1, "quantified": False},
                    {
                        "order": 2,
                        "quantified": False,
                        "analysis_included": False,
                        "exclusion_reason": "Excluded by project owner.",
                    },
                ]
            }
        )
    )

    row = audit_analysis_completeness(
        {"mouse": ReviewRuns(registration=registration, stain=stain)}
    ).cohorts[0]

    assert row.stain_expected == 0
    assert row.stain_missing_orders == ()
    assert row.cell_expected == 1
    assert row.cell_missing_sections == ("001",)


def test_completeness_uses_prepared_mask_count_for_unfinished_registration(
    tmp_path: Path,
) -> None:
    registration = tmp_path / "registration"
    registration.mkdir()
    (registration / "mask_review.json").write_text(json.dumps({"slides": [{}, {}, {}]}))

    report = audit_analysis_completeness(
        {"mouse": ReviewRuns(registration=registration)}
    )
    row = report.to_json_dict()["cohorts"][0]

    assert row["registration"] == {
        "expected_sections": 3,
        "completed_sections": 0,
        "complete": False,
        "approved": False,
    }
    assert row["stain"]["expected_non_reference_sections"] == 2
