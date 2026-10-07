"""Eligibility-bound compute assignments without image or GPU dependencies.

These checks authorize a reconstruction work scope, not a scientific result.
Model fitting must still use the study's separate fit/evaluation guardrails.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from histopia.study._eligibility import (
    POLICY_ID,
    _digest,
    validate_reconstruction_registry,
)
from histopia.study._ledger import AssignmentLedger
from histopia.study._manifest import fingerprint

_QUALIFICATION = {"provenance", "native_qc", "bounded_reconstruction"}
_DOWNSTREAM = {
    "registration",
    "segmentation",
    "features",
    "prediction",
    "annotation",
    "region_summary",
    "neighborhoods",
    "audit",
    "export",
}
_IDENTITY = ("source_id", "species", "organ", "subject_id", "specimen_id", "block_id")


def validate_reconstruction_job(
    request: Mapping, registry: Mapping, policy: Mapping
) -> dict:
    """Return a receipt or reject an unknown, stale or out-of-scope assignment.

    Pending candidates permit only bounded provenance, native-image QC and
    reconstruction qualification. Geometry additionally requires verified serial
    identity, accessible images and verified brightfield staining. A receipt
    never changes the registry or authorizes publication. Workers must enforce
    the declared acquisition scope and wall-time limit and recheck at launch,
    checkpoints and completion; this pure function does not execute work.
    """
    if (
        policy.get("schema_version") != "reconstruction-compute-policy-1"
        or policy.get("policy_id") != POLICY_ID
    ):
        raise ValueError("a versioned reconstruction compute policy is required")
    excluded = policy.get("excluded_public_sources")
    limits = policy.get("qualification_limits", {})
    if not isinstance(excluded, list) or not all(
        isinstance(s, str) and s for s in excluded
    ):
        raise ValueError("compute policy requires explicit excluded public sources")
    for key in ("max_acquisitions", "max_wall_seconds"):
        if type(limits.get(key)) is not int or limits[key] <= 0:
            raise ValueError("compute policy requires positive qualification limits")
    rules = dict(
        schema_version=policy["schema_version"],
        policy_id=POLICY_ID,
        excluded_public_sources=sorted(set(excluded)),
        qualification_limits=limits,
    )
    if request.get("schema_version") != "reconstruction-job-1":
        raise ValueError("a versioned reconstruction job is required")
    stage = request.get("stage")
    if stage not in _QUALIFICATION | _DOWNSTREAM:
        raise ValueError("unknown reconstruction processing stage")
    decisions = validate_reconstruction_registry(registry)["decisions"]
    decision = next(
        (
            d
            for d in decisions
            if d["reconstruction_id"] == request.get("reconstruction_id")
        ),
        None,
    )
    if decision is None or decision["status"] == "excluded":
        raise ValueError("reconstruction is unknown or excluded")
    if decision["external"] and decision["source_id"] in excluded:
        raise ValueError("public source has been retired from compute")
    if (
        request.get("eligibility_fingerprint") != decision["fingerprint"]
        or request.get("study_fingerprint") != decision["study_fingerprint"]
    ):
        raise ValueError("job eligibility or study binding is missing or stale")
    if any(request.get(k) != decision[k] for k in _IDENTITY) or (
        request.get("external") is not decision["external"]
    ):
        raise ValueError("job reconstruction identity conflicts with registry")
    acquisitions = request.get("acquisition_ids")
    if (
        not isinstance(acquisitions, list)
        or not acquisitions
        or not all(isinstance(a, str) for a in acquisitions)
        or len(set(acquisitions)) != len(acquisitions)
        or not set(acquisitions) <= set(decision["acquisition_ids"])
    ):
        raise ValueError("job requires distinct acquisitions within its reconstruction")
    modalities = request.get("acquisition_modalities")
    if (
        not isinstance(modalities, list)
        or not all(isinstance(m, str) for m in modalities)
        or set(modalities)
        != {decision["acquisition_modalities"][a] for a in acquisitions}
    ):
        raise ValueError("job modality binding conflicts with its acquisitions")
    if not _digest(request.get("method_fingerprint")):
        raise ValueError("job requires a bound method fingerprint")
    seconds = request.get("max_wall_seconds")
    if type(seconds) is not int or seconds <= 0:
        raise ValueError("job requires a positive wall-time limit")
    if stage in _QUALIFICATION:
        purpose = request.get("qualification_purpose")
        if not isinstance(purpose, str) or not purpose.strip():
            raise ValueError("qualification requires an explicit purpose")
        if (
            len(acquisitions) > limits["max_acquisitions"]
            or seconds > limits["max_wall_seconds"]
        ):
            raise ValueError("qualification exceeds its bounded compute budget")
        reasons = set(decision["reasons"])
        if stage in {"native_qc", "bounded_reconstruction"} and reasons & {
            "image_unavailable_or_unbound",
            "unverified_acquisition_binding",
        }:
            raise ValueError("image qualification requires accessible bound inputs")
        if stage == "bounded_reconstruction" and reasons & {
            "staining_not_verified",
            "physical_serial_identity_unverified",
        }:
            raise ValueError("geometry requires verified serial identity and staining")
    elif decision["status"] != "eligible":
        raise ValueError("downstream processing requires an eligible reconstruction")
    binding = {k: decision[k] for k in _IDENTITY}
    binding.update(
        reconstruction_id=decision["reconstruction_id"],
        external=decision["external"],
        eligibility_fingerprint=decision["fingerprint"],
        study_fingerprint=decision["study_fingerprint"],
        stage=stage,
        acquisition_ids=sorted(acquisitions),
        acquisition_modalities=sorted(set(modalities)),
        method_fingerprint=request["method_fingerprint"],
    )
    result = dict(
        **binding,
        policy_fingerprint=fingerprint(rules),
        request_fingerprint=fingerprint(request),
        # Renaming a job or increasing its budget cannot repeat the same work.
        artifact_key=fingerprint(binding),
        max_wall_seconds=seconds,
        qualification_only=stage in _QUALIFICATION,
        scientific_approval=False,
    )
    result["fingerprint"] = fingerprint(result)
    return result


class ReconstructionJobLedger:
    """Durable single-owner jobs gated by the live policy and registry files.

    Rejected claims terminate as blocked, so polling cannot retry them. Finished
    and failed artifacts retain their reservations; a new method or new input
    evidence requires an explicit new assignment. No age-based claim release or
    automatic retry is provided. Payloads identify the actual executable and
    inputs; executors must verify their bound bytes before executing them.
    """

    def __init__(self, path: Path | str, *, policy_path: Path | str):
        self.ledger = AssignmentLedger(path)
        self.policy_path = Path(policy_path)

    def _validate(self, request: Mapping) -> dict:
        policy = json.loads(self.policy_path.read_text())
        registry_path = Path(policy["registry_path"])
        if not registry_path.is_absolute():
            registry_path = self.policy_path.parent / registry_path
        registry = json.loads(registry_path.read_text())
        return validate_reconstruction_job(request, registry, policy)

    def add(self, job_id: str, node: str, request: dict, payload: dict) -> dict:
        """Validate before reserving the immutable job and scientific artifact."""
        receipt = self._validate(request)
        self.ledger.add(
            job_id,
            receipt["artifact_key"],
            node,
            dict(
                request=request,
                execution=payload,
                execution_fingerprint=fingerprint(payload),
                authorization=receipt,
            ),
        )
        return receipt

    def authorize(self, job_id: str) -> dict:
        """Re-read current evidence, including revocations, at each checkpoint."""
        payload = self.ledger.read(job_id)["payload"]
        receipt = self._validate(payload["request"])
        if receipt != payload["authorization"] or (
            fingerprint(payload["execution"]) != payload["execution_fingerprint"]
        ):
            raise ValueError("assignment or compute policy changed after submission")
        return receipt

    def claim(self, job_id: str, node: str) -> str:
        """Reserve once, then gate launch; a rejected claim is durably blocked."""
        owner = self.ledger.claim(job_id, node)
        try:
            self.authorize(job_id)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            self.ledger.finish(
                job_id,
                owner,
                status="blocked",
                evidence={
                    "eligibility_error": str(exc),
                    "processing_started": False,
                },
            )
            raise
        return owner

    def finish(self, job_id: str, owner: str, *, status: str, evidence: dict) -> None:
        """Preserve outputs, but block completion if inputs were revoked."""
        if status == "complete":
            try:
                receipt = self.authorize(job_id)
            except (ValueError, KeyError, TypeError, OSError) as exc:
                self.ledger.finish(
                    job_id,
                    owner,
                    status="blocked",
                    evidence={
                        **evidence,
                        "eligibility_error": str(exc),
                        "outputs_preserved": True,
                    },
                )
                raise
            evidence = {**evidence, "authorization": receipt}
        self.ledger.finish(job_id, owner, status=status, evidence=evidence)

    def read(self, job_id: str) -> dict:
        """Read the original assignment, independent of claim/outcome overlays."""
        return self.ledger.read(job_id)

    def rows(self) -> list[dict]:
        return self.ledger.rows()
