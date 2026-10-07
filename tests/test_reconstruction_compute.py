from __future__ import annotations

import copy
import json
import subprocess
import sys

import pytest
from test_reconstruction_eligibility import example, rebind

from histopia.study import ReconstructionJobLedger, validate_reconstruction_job
from histopia.study._eligibility import POLICY_ID, assess_reconstruction_eligibility
from histopia.study._manifest import fingerprint


def inputs(*, pending=False):
    study, assessment = example()
    if pending:
        assessment["reconstruction_qc"]["alignment"] = "pending"
    decision = assess_reconstruction_eligibility(study, assessment)
    registry = dict(
        schema_version="reconstruction-registry-1",
        policy_id=POLICY_ID,
        entries=[dict(study=study, assessment=assessment)],
    )
    policy = dict(
        schema_version="reconstruction-compute-policy-1",
        policy_id=POLICY_ID,
        registry_path="registry.json",
        excluded_public_sources=["retired"],
        qualification_limits=dict(max_acquisitions=6, max_wall_seconds=600),
    )
    request = {
        k: decision[k]
        for k in (
            "reconstruction_id",
            "study_fingerprint",
            "source_id",
            "species",
            "organ",
            "subject_id",
            "specimen_id",
            "block_id",
            "external",
            "acquisition_ids",
        )
    }
    request.update(
        schema_version="reconstruction-job-1",
        stage="segmentation",
        eligibility_fingerprint=decision["fingerprint"],
        acquisition_modalities=decision["modalities"],
        method_fingerprint=fingerprint("method"),
        max_wall_seconds=60,
    )
    return request, registry, policy


def queue(tmp_path, *, pending=False):
    request, registry, policy = inputs(pending=pending)
    (tmp_path / "registry.json").write_text(json.dumps(registry))
    (tmp_path / "policy.json").write_text(json.dumps(policy))
    ledger = ReconstructionJobLedger(
        tmp_path / "queue", policy_path=tmp_path / "policy.json"
    )
    return ledger, request, registry, policy


def test_eligible_request_is_bound_to_physical_stack_and_method():
    request, registry, policy = inputs()
    receipt = validate_reconstruction_job(request, registry, policy)
    assert receipt["organ"] == "tongue" and not receipt["scientific_approval"]
    assert receipt["acquisition_ids"] == ["a0", "a1", "a2"]
    reordered = copy.deepcopy(request)
    reordered["acquisition_ids"].reverse()
    reordered["max_wall_seconds"] *= 2
    assert (
        validate_reconstruction_job(reordered, registry, policy)["artifact_key"]
        == receipt["artifact_key"]
    )
    reordered["method_fingerprint"] = fingerprint("new method")
    assert (
        validate_reconstruction_job(reordered, registry, policy)["artifact_key"]
        != receipt["artifact_key"]
    )


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": "unknown"},
        {"reconstruction_id": "missing"},
        {"eligibility_fingerprint": "0" * 64},
        {"study_fingerprint": "0" * 64},
        {"source_id": "retired"},
        {"block_id": "another"},
        {"external": 1},
        {"acquisition_ids": []},
        {"acquisition_ids": ["a1", "a1"]},
        {"acquisition_ids": ["foreign"]},
        {"acquisition_modalities": ["IF"]},
        {"acquisition_modalities": "IHC"},
        {"method_fingerprint": "unbound"},
        {"stage": "unrestricted"},
        {"max_wall_seconds": 0},
        {"max_wall_seconds": True},
    ],
)
def test_missing_or_conflicting_job_bindings_fail_closed(change):
    request, registry, policy = inputs()
    request.update(change)
    with pytest.raises(ValueError):
        validate_reconstruction_job(request, registry, policy)


@pytest.mark.parametrize("failure", ["if", "he_only", "mixed_blocks", "retired"])
def test_excluded_cohort_cannot_use_qualification_escape(failure):
    request, registry, policy = inputs()
    entry = registry["entries"][0]
    study, assessment = entry["study"], entry["assessment"]
    if failure == "if":
        assessment["acquisition_qc"]["a1"]["imaging_mode"] = "fluorescence"
    elif failure == "he_only":
        for acq in study["acquisitions"]:
            acq["modality"] = "H&E"
            assessment["acquisition_qc"][acq["acquisition_id"]]["verified_modality"] = (
                "H&E"
            )
    elif failure == "mixed_blocks":
        study["sections"][2]["block_id"] = "other"
    else:
        policy["excluded_public_sources"].append("serial")
    entry["study"], entry["assessment"] = rebind(study, assessment)
    decision = assess_reconstruction_eligibility(entry["study"], entry["assessment"])
    request.update(
        stage="provenance",
        qualification_purpose="Document acquisition",
        eligibility_fingerprint=decision["fingerprint"],
        study_fingerprint=decision["study_fingerprint"],
    )
    with pytest.raises(ValueError, match="excluded|retired"):
        validate_reconstruction_job(request, registry, policy)


def test_pending_candidate_requires_bounded_explicit_qualification():
    request, registry, policy = inputs(pending=True)
    with pytest.raises(ValueError, match="downstream"):
        validate_reconstruction_job(request, registry, policy)
    request["stage"] = "bounded_reconstruction"
    with pytest.raises(ValueError, match="purpose"):
        validate_reconstruction_job(request, registry, policy)
    request["qualification_purpose"] = "Image-only alignment with held-out landmarks"
    receipt = validate_reconstruction_job(request, registry, policy)
    assert receipt["qualification_only"] and not receipt["scientific_approval"]
    request["max_wall_seconds"] = 601
    with pytest.raises(ValueError, match="budget"):
        validate_reconstruction_job(request, registry, policy)
    request["max_wall_seconds"] = 60
    policy["qualification_limits"]["max_acquisitions"] = 2
    with pytest.raises(ValueError, match="budget"):
        validate_reconstruction_job(request, registry, policy)


@pytest.mark.parametrize("missing", ["image", "staining", "block"])
def test_geometry_cannot_run_before_native_input_qualification(missing):
    request, registry, policy = inputs(pending=True)
    entry = registry["entries"][0]
    assessment = entry["assessment"]
    if missing == "image":
        assessment["acquisition_qc"]["a1"]["image_available"] = False
    elif missing == "staining":
        assessment["acquisition_qc"]["a1"]["native_visual_qc"] = "pending"
    else:
        assessment["physical_provenance"]["block_verified"] = False
    decision = assess_reconstruction_eligibility(entry["study"], assessment)
    request.update(
        stage="bounded_reconstruction",
        qualification_purpose="Geometry pilot",
        eligibility_fingerprint=decision["fingerprint"],
    )
    with pytest.raises(ValueError, match="requires"):
        validate_reconstruction_job(request, registry, policy)
    request["stage"] = "provenance"
    assert validate_reconstruction_job(request, registry, policy)["qualification_only"]


def test_unique_claims_and_artifacts_do_not_automatically_retry(tmp_path):
    ledger, request, _, _ = queue(tmp_path)
    ledger.add("job1", "node2", request, {"input": "bound"})
    with pytest.raises(ValueError, match="elsewhere"):
        ledger.claim("job1", "node3")
    owner = ledger.claim("job1", "node2")
    with pytest.raises(ValueError, match="already claimed"):
        ledger.claim("job1", "node2")
    ledger.finish("job1", owner, status="failed", evidence={"checkpoint": "retained"})
    with pytest.raises(FileExistsError):
        ledger.add("renamed-retry", "node3", request, {"input": "bound"})
    assert ledger.rows()[0]["status"] == "failed"


@pytest.mark.parametrize("change", ["registry", "policy", "payload"])
def test_recheck_at_claim_prevents_stale_processing(tmp_path, change):
    ledger, request, registry, policy = queue(tmp_path)
    ledger.add("job1", "node2", request, {"input": "bound"})
    if change == "registry":
        registry["entries"][0]["assessment"]["acquisition_qc"]["a1"]["imaging_mode"] = (
            "fluorescence"
        )
        (tmp_path / "registry.json").write_text(json.dumps(registry))
    elif change == "policy":
        policy["excluded_public_sources"].append("serial")
        (tmp_path / "policy.json").write_text(json.dumps(policy))
    else:
        path = next((tmp_path / "queue/jobs").glob("*.json"))
        record = json.loads(path.read_text())
        record["payload"]["execution"]["input"] = "changed"
        path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        ledger.claim("job1", "node2")
    row = ledger.rows()[0]
    assert row["status"] == "blocked" and not row["evidence"]["processing_started"]
    with pytest.raises(ValueError, match="already claimed"):
        ledger.claim("job1", "node2")


def test_revocation_preserves_output_but_prevents_completion(tmp_path):
    ledger, request, registry, _ = queue(tmp_path)
    ledger.add("job1", "node2", request, {})
    owner = ledger.claim("job1", "node2")
    registry["entries"] = []
    (tmp_path / "registry.json").write_text(json.dumps(registry))
    with pytest.raises(ValueError):
        ledger.finish("job1", owner, status="complete", evidence={"checkpoint": "kept"})
    row = ledger.rows()[0]
    assert row["status"] == "blocked" and row["evidence"]["outputs_preserved"]
    assert row["evidence"]["checkpoint"] == "kept"


def test_normal_completion_retains_authorization_receipt(tmp_path):
    ledger, request, _, _ = queue(tmp_path)
    receipt = ledger.add("job1", "node2", request, {})
    owner = ledger.claim("job1", "node2")
    ledger.finish(
        "job1", owner, status="complete", evidence={"sha256": fingerprint("result")}
    )
    assert ledger.rows()[0]["evidence"]["authorization"] == receipt


def test_compute_entry_points_remain_lightweight():
    code = (
        "import sys; from histopia.study import ReconstructionJobLedger, "
        "validate_reconstruction_job; "
        "assert not {'numpy', 'torch', 'PIL', 'cv2'} & sys.modules.keys()"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
