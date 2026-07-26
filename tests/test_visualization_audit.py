from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from histopia.semantic import approve_semantic_result
from histopia.semantic._preflight import preflight_registration, write_preflight
from histopia.semantic._result_validation import _seal_semantic_result
from histopia.visualization import audit_workflows, write_workflow_audit


def test_audit_accepts_exact_approved_workflow_and_current_viewer(
    tmp_path: Path,
) -> None:
    registration, semantic = _write_approved_workflow(tmp_path)
    registration_result = registration / "registration_result.json"
    registration_approval = json.loads(
        (registration / "registration_approval.json").read_text()
    )
    semantic_result = json.loads((semantic / "semantic_result.json").read_text())
    preflight = json.loads((semantic / "preflight.json").read_text())
    manifest = tmp_path / "viewer" / "manifest.json"
    manifest.parent.mkdir()
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "mice": [
                    {
                        "id": "mouse-a",
                        "slides": [{}, {}],
                        "registration_approval": {
                            "approved": True,
                            "order_fingerprint": "accepted-order",
                            "registration_result_sha256": _sha256(registration_result),
                        },
                        "semantic": {
                            "fingerprint": semantic_result["fingerprint"],
                            "registration_binding": {
                                "preflight_schema_version": 3,
                                "preflight_fingerprint": preflight["fingerprint"],
                                "registration_result_sha256": _sha256(
                                    registration_result
                                ),
                                "registration_approval_sha256": _sha256(
                                    registration / "registration_approval.json"
                                ),
                                "approval_bound": True,
                            },
                            "review": {
                                "approved": True,
                                "fingerprint_matches": True,
                            },
                        },
                    }
                ],
            }
        )
    )

    report = audit_workflows(
        {"mouse-a": registration},
        semantic_runs={"mouse-a": semantic},
        viewer_manifest=manifest,
    )

    payload = report.to_json_dict()
    assert report.status == "approved"
    assert report.exit_code == 0
    assert payload["summary"] == {
        "cohort_count": 1,
        "approved": 1,
        "review_required": 0,
        "incomplete": 0,
        "invalid": 0,
        "viewer_unmapped_count": 0,
    }
    assert payload["cohorts"][0]["registration"] == {
        "status": "approved",
        "slide_count": 2,
        "result_sha256": registration_approval["artifacts"]["registration_result.json"],
        "order_fingerprint": "accepted-order",
        "issue": None,
    }
    assert payload["cohorts"][0]["semantic"]["registration_binding"] == (
        "approval_bound"
    )
    assert payload["cohorts"][0]["viewer"]["status"] == "current"
    assert str(tmp_path) not in json.dumps(payload)

    output = write_workflow_audit(report, tmp_path / "audit.json")
    assert json.loads(output.read_text()) == payload


def test_audit_distinguishes_review_gate_from_invalid_approval(
    tmp_path: Path,
) -> None:
    registration, semantic = _write_approved_workflow(
        tmp_path,
        approve_semantic=False,
    )

    review_required = audit_workflows(
        {"mouse-a": registration},
        semantic_runs={"mouse-a": semantic},
    )

    assert review_required.status == "review_required"
    assert review_required.exit_code == 2
    assert review_required.cohorts[0].semantic.issue == "semantic_approval_required"

    mask_path = registration / "mask_review.json"
    mask = json.loads(mask_path.read_text())
    mask["slides"][0]["notes"] = "Changed after approval."
    mask_path.write_text(json.dumps(mask))

    invalid = audit_workflows(
        {"mouse-a": registration},
        semantic_runs={"mouse-a": semantic},
    )

    assert invalid.status == "invalid"
    assert invalid.exit_code == 1
    assert invalid.cohorts[0].registration.issue == ("registration_approval_invalid")
    assert invalid.cohorts[0].semantic.issue == ("semantic_result_or_binding_invalid")


def test_audit_reports_missing_results_and_viewer_cohorts(tmp_path: Path) -> None:
    missing = tmp_path / "missing-registration"
    viewer = tmp_path / "manifest.json"
    viewer.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "mice": [{"id": "extra", "slides": []}],
            }
        )
    )

    report = audit_workflows(
        {"mouse-a": missing},
        viewer_manifest=viewer,
    )

    assert report.status == "incomplete"
    assert report.exit_code == 1
    assert report.viewer_unmapped_ids == ("extra",)
    cohort = report.cohorts[0]
    assert cohort.registration.issue == "registration_result_missing"
    assert cohort.semantic.status == "not_requested"
    assert cohort.viewer.issue == "viewer_cohort_missing"


def test_audit_rejects_unknown_semantic_and_unsafe_cohort_ids(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="no matching registration"):
        audit_workflows(
            {"mouse-a": tmp_path / "registration"},
            semantic_runs={"mouse-b": tmp_path / "semantic"},
        )
    with pytest.raises(ValueError, match="invalid cohort IDs"):
        audit_workflows({"../mouse": tmp_path / "registration"})


def _write_approved_workflow(
    root: Path,
    *,
    approve_semantic: bool = True,
) -> tuple[Path, Path]:
    registration = root / "registration"
    processed = registration / "processed"
    raw = root / "raw"
    processed.mkdir(parents=True)
    raw.mkdir()
    slides = []
    reviews = []
    names = ("HE.ndpi", "CK19.ndpi")
    for index, name in enumerate(names):
        source = raw / name
        source.write_bytes(f"slide-{index}".encode())
        image = np.full((8, 10, 3), 220 - index * 20, dtype=np.uint8)
        mask = np.zeros((8, 10), dtype=np.uint8)
        mask[1:7, 2:9] = 255
        Image.fromarray(image).save(processed / f"{source.stem}.thumbnail.png")
        Image.fromarray(mask).save(processed / f"{source.stem}.mask.png")
        review = {
            "slide": name,
            "thumbnail_sha256": f"mask-fingerprint-{index}",
            "status": "auto_pass",
            "method": "group_consensus",
            "reviewer": "Test Reviewer",
            "notes": "Reviewed.",
            "override_path": None,
        }
        reviews.append(review)
        slides.append(
            {
                "path": str(source),
                "is_reference": index == 0,
                "geometry": {
                    "native_shape": [80, 100],
                    "content_bbox_xywh": [0, 0, 100, 80],
                    "thumbnail_shape": [8, 10],
                    "bounds_source": "test",
                    "mpp_xy": [0.5, 0.5],
                    "mpp_source": "test",
                },
                "transform": {"matrix": np.eye(3).tolist()},
                "mask": {"accepted": True, "method": "group_consensus"},
                "mask_review": dict(review),
            }
        )
    (registration / "registration_result.json").write_text(
        json.dumps({"reference_slide": slides[0]["path"], "slides": slides})
    )
    (registration / "mask_review.json").write_text(
        json.dumps({"schema_version": 2, "slides": reviews})
    )
    (registration / "section_order_review.json").write_text(
        json.dumps(
            {
                "schema_version": 3,
                "approved": True,
                "fingerprint": "accepted-order",
                "input_fingerprints": {
                    name: f"order-input-{index}" for index, name in enumerate(names)
                },
                "slides": [
                    {"order": index + 1, "slide": name}
                    for index, name in enumerate(names)
                ],
            }
        )
    )
    artifacts = {
        name: _sha256(registration / name)
        for name in (
            "registration_result.json",
            "mask_review.json",
            "section_order_review.json",
        )
    }
    (registration / "registration_approval.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "reviewer": "Test Reviewer",
                "reviewed_at": "2026-07-26T10:00:00+00:00",
                "notes": "Reviewed.",
                "slide_count": 2,
                "order_fingerprint": "accepted-order",
                "artifacts": artifacts,
            }
        )
    )

    semantic = root / "semantic"
    semantic.mkdir()
    preflight = preflight_registration(registration)
    write_preflight(preflight, semantic / "preflight.json")
    model_path = semantic / "atlas_model.npz"
    np.savez_compressed(model_path, pca_mean=np.zeros(2))
    label_dir = semantic / "labels"
    label_dir.mkdir()
    semantic_slides = []
    for index, name in enumerate(names, start=1):
        relative = f"labels/{index:03d}.npz"
        np.savez_compressed(
            semantic / relative,
            labels=np.array([0, 1], dtype=np.int16),
        )
        semantic_slides.append({"id": name, "labels": {"2": relative}})
    core = {
        "schema_version": 3,
        "model": model_path.name,
        "slides": semantic_slides,
        "topology_pairs": [],
        "feature_provenance": {
            "preflight_fingerprint": preflight.fingerprint,
        },
    }
    semantic_result = _seal_semantic_result(semantic, core)
    (semantic / "semantic_result.json").write_text(json.dumps(semantic_result))
    (semantic / "semantic_review.json").write_text(
        json.dumps(
            {
                "schema_version": 3,
                "approved": False,
                "fingerprint": semantic_result["fingerprint"],
                "reviewer": None,
                "notes": "",
            }
        )
    )
    if approve_semantic:
        approve_semantic_result(
            semantic,
            registration_run=registration,
            reviewer="Test Reviewer",
            notes="Semantic result reviewed.",
            reviewed_at="2026-07-26T11:00:00+00:00",
        )
    return registration, semantic


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
