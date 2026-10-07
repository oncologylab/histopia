"""Donor-bound public brightfield benchmarks, independent of 3D eligibility.

This scope authorizes research processing only. It cannot authorize a serial
reconstruction or imply validation of a mouse-trained predictor.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import urlsplit

from histopia.study._ledger import AssignmentLedger
from histopia.study._manifest import fingerprint

_SHA = re.compile(r"[a-f0-9]{64}\Z")
_ROLES = {"train", "validation", "test", "qualification"}


def validate_external_benchmark(manifest: Mapping) -> dict:
    """Validate modality, donor independence and immutable study identity."""
    if manifest.get("schema_version") != "external-brightfield-benchmark-1":
        raise ValueError("a versioned external benchmark is required")
    if manifest.get("purpose") != "2d-protein-benchmark":
        raise ValueError("external benchmarks cannot authorize reconstruction")
    expected = fingerprint({k: v for k, v in manifest.items() if k != "fingerprint"})
    if manifest.get("fingerprint") != expected:
        raise ValueError("benchmark manifest fingerprint changed")
    records = manifest.get("images", [])
    if not records:
        raise ValueError("benchmark requires explicit image records")
    identities, donors, duplicates = set(), {}, {}
    exposed = set(manifest.get("development_exposed_donors", []))
    for row in records:
        key = row.get("image_id")
        if not isinstance(key, str) or not key or key in identities:
            raise ValueError("benchmark image identifiers must be unique")
        identities.add(key)
        if (
            row.get("modality") not in {"IHC", "H&E"}
            or row.get("detection") != "brightfield"
        ):
            raise ValueError("benchmark requires verified brightfield H&E or IHC")
        if row["modality"] == "IHC" and row.get("chromogen") != "DAB":
            raise ValueError("this benchmark requires documented DAB IHC")
        for name in (
            "source_id",
            "organ",
            "donor_id",
            "acquisition_id",
            "source_methods_url",
        ):
            if not isinstance(row.get(name), str) or not row[name]:
                raise ValueError("benchmark acquisition provenance is incomplete")
        for name in ("image_url", "source_methods_url"):
            url = urlsplit(row.get(name, ""))
            if url.scheme != "https" or not url.netloc:
                raise ValueError("benchmark inputs require HTTPS provenance")
        role = row.get("role")
        if role not in _ROLES:
            raise ValueError("unknown benchmark role")
        donor = (row["source_id"], row["donor_id"])
        if donors.setdefault(donor, role) != role:
            raise ValueError(
                "all organs and acquisitions from a donor must share a split"
            )
        if row["source_id"] + ":" + row["donor_id"] in exposed and role in {
            "validation",
            "test",
        }:
            raise ValueError("previously exposed donors must remain in development")
        for identity in (row["image_url"], row.get("image_sha256")):
            if identity and duplicates.setdefault(identity, role) != role:
                raise ValueError("duplicate acquisitions cannot cross donor splits")
        if row.get("image_sha256") and not _SHA.fullmatch(row["image_sha256"]):
            raise ValueError("invalid image byte binding")
    if not any(r["modality"] == "IHC" for r in records):
        raise ValueError("a protein benchmark requires chromogenic IHC targets")
    return dict(
        fingerprint=expected,
        images=len(records),
        donors=len(donors),
        purpose=manifest["purpose"],
    )


def validate_external_job(request: Mapping, manifest: Mapping) -> dict:
    """Bind one processing job without granting any 3D publication scope."""
    validate_external_benchmark(manifest)
    if request.get("manifest_fingerprint") != manifest["fingerprint"]:
        raise ValueError("stale external benchmark assignment")
    if request.get("stage") not in {
        "stain_features",
        "audit",
        "fit",
        "evaluate",
        "export",
    }:
        raise ValueError("unknown external processing stage")
    if not _SHA.fullmatch(str(request.get("method_fingerprint", ""))):
        raise ValueError("external job requires an exact method binding")
    images = {r["image_id"]: r for r in manifest["images"]}
    ids = request.get("image_ids", [])
    if not ids or len(set(ids)) != len(ids) or not set(ids) <= images.keys():
        raise ValueError("external job acquisitions are missing or unresolved")
    for field, role in (
        ("fit_ids", "train"),
        ("validation_ids", "validation"),
        ("test_ids", "test"),
    ):
        for key in request.get(field, []):
            if key not in ids or images[key]["role"] != role:
                raise ValueError("model role conflicts with frozen donor split")
    if request["stage"] == "fit":
        if request.get("test_ids") or any(images[k]["role"] == "test" for k in ids):
            raise ValueError("test outcomes cannot enter model fitting")
        if not request.get("fit_ids"):
            raise ValueError("model fitting requires explicit training acquisitions")
    if request["stage"] == "evaluate" and not _SHA.fullmatch(
        str(request.get("model_selection_sha256", ""))
    ):
        raise ValueError("evaluation requires a frozen model-selection receipt")
    seconds = request.get("max_wall_seconds")
    if type(seconds) is not int or not 0 < seconds <= 14400:
        raise ValueError("external assignments require a bounded wall time")
    binding = dict(
        purpose=manifest["purpose"],
        manifest_fingerprint=manifest["fingerprint"],
        stage=request["stage"],
        image_ids=sorted(ids),
        method_fingerprint=request["method_fingerprint"],
        roles={
            k: sorted(request.get(k, []))
            for k in ("fit_ids", "validation_ids", "test_ids")
        },
    )
    if request["stage"] == "evaluate":
        binding["model_selection_sha256"] = request["model_selection_sha256"]
    return dict(
        **binding,
        artifact_key=fingerprint(binding),
        request_fingerprint=fingerprint(request),
        max_wall_seconds=seconds,
        reconstruction_allowed=False,
        scientific_approval=False,
    )


class ExternalBenchmarkLedger:
    """Single-owner assignments that recheck live manifest and method bindings."""

    def __init__(self, path: Path | str, *, manifest_path: Path | str):
        self.ledger = AssignmentLedger(path)
        self.manifest_path = Path(manifest_path)

    def add(self, job_id: str, request: dict, execution: dict) -> None:
        manifest = json.loads(self.manifest_path.read_text())
        authorization = validate_external_job(request, manifest)
        self.ledger.add(
            job_id,
            authorization["artifact_key"],
            "pool",
            dict(
                request=request,
                execution=execution,
                execution_fingerprint=fingerprint(execution),
                authorization=authorization,
            ),
        )

    def authorize(self, job_id: str) -> dict:
        payload = self.ledger.read(job_id)["payload"]
        result = validate_external_job(
            payload["request"], json.loads(self.manifest_path.read_text())
        )
        if (
            result != payload["authorization"]
            or fingerprint(payload["execution"]) != payload["execution_fingerprint"]
        ):
            raise ValueError("external assignment changed after submission")
        return result

    def claim(self, job_id: str) -> str:
        owner = self.ledger.claim(job_id, "pool")
        try:
            self.authorize(job_id)
        except (ValueError, OSError, KeyError) as error:
            self.ledger.finish(
                job_id, owner, status="blocked", evidence={"error": str(error)}
            )
            raise
        return owner

    def finish(self, job_id: str, owner: str, *, status: str, evidence: dict) -> None:
        if status == "complete":
            self.authorize(job_id)
        self.ledger.finish(job_id, owner, status=status, evidence=evidence)

    def rows(self) -> list[dict]:
        return self.ledger.rows()
