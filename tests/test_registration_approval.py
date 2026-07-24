from __future__ import annotations

import json
from pathlib import Path

import pytest

from histopia.registration._approval import (
    approve_registration_run,
    validate_registration_approval,
)


def test_approve_registration_run_seals_exact_reviewed_artifacts(
    tmp_path: Path,
) -> None:
    _write_run(tmp_path)

    approved = approve_registration_run(
        tmp_path,
        reviewer="Test Reviewer",
        notes="Masks and order visually reviewed.",
        reviewed_at="2026-07-24T10:00:00+00:00",
    )

    assert approved.slide_count == 2
    assert approved.order_fingerprint == "order-fingerprint"
    assert validate_registration_approval(tmp_path) == approved
    result = json.loads((tmp_path / "registration_result.json").read_text())
    assert {row["mask_review"]["status"] for row in result["slides"]} == {"auto_pass"}
    order = json.loads((tmp_path / "section_order_review.json").read_text())
    assert order["approved"] is True
    assert order["reviewer"] == "Test Reviewer"


def test_approve_registration_run_rejects_mask_fingerprint_mismatch(
    tmp_path: Path,
) -> None:
    _write_run(tmp_path)
    mask = json.loads((tmp_path / "mask_review.json").read_text())
    mask["slides"][0]["thumbnail_sha256"] = "changed"
    (tmp_path / "mask_review.json").write_text(json.dumps(mask))

    with pytest.raises(ValueError, match="mask review fingerprint mismatch"):
        approve_registration_run(
            tmp_path,
            reviewer="Test Reviewer",
            notes="Reviewed.",
        )

    assert not (tmp_path / "registration_approval.json").exists()


def test_validate_registration_approval_rejects_post_approval_changes(
    tmp_path: Path,
) -> None:
    _write_run(tmp_path)
    approve_registration_run(
        tmp_path,
        reviewer="Test Reviewer",
        notes="Reviewed.",
    )
    result_path = tmp_path / "registration_result.json"
    result_path.write_text(result_path.read_text() + "\n")

    with pytest.raises(ValueError, match="artifact digest mismatch"):
        validate_registration_approval(tmp_path)


def test_approve_registration_run_rejects_timestamp_without_timezone(
    tmp_path: Path,
) -> None:
    _write_run(tmp_path)

    with pytest.raises(ValueError, match="include a timezone"):
        approve_registration_run(
            tmp_path,
            reviewer="Test Reviewer",
            notes="Reviewed.",
            reviewed_at="2026-07-24T10:00:00",
        )


def _write_run(root: Path) -> None:
    names = ("HE.ndpi", "CK19.ndpi")
    reviews = [
        {
            "slide": name,
            "thumbnail_sha256": f"hash-{index}",
            "status": "pending",
            "method": "object_aware_fusion",
            "reviewer": "",
            "notes": "",
            "override_path": None,
        }
        for index, name in enumerate(names)
    ]
    result = {
        "output_dir": str(root),
        "reference_slide": str(root / names[0]),
        "slides": [
            {
                "path": str(root / name),
                "is_reference": index == 0,
                "mask": {"accepted": True},
                "mask_review": dict(reviews[index]),
            }
            for index, name in enumerate(names)
        ],
        "warnings": [],
    }
    order = {
        "schema_version": 3,
        "approved": False,
        "fingerprint": "order-fingerprint",
        "slides": [
            {"order": index + 1, "slide": name} for index, name in enumerate(names)
        ],
    }
    (root / "registration_result.json").write_text(json.dumps(result))
    (root / "mask_review.json").write_text(
        json.dumps({"schema_version": 2, "slides": reviews})
    )
    (root / "section_order_review.json").write_text(json.dumps(order))
