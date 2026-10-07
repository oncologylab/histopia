"""Export separate 2D public benchmarks without a reconstruction bypass."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from histopia.study._external_benchmark import validate_external_benchmark
from histopia.study._manifest import file_sha256
from histopia.visualization._results_catalog import _write_catalog


def external_validation_tab(catalog: Mapping) -> dict:
    """Bind portal recovery links to exact images in a published 2D catalog.

    A source ID alone cannot authorize recovery from a serial-workflow URL:
    old reconstruction links and unknown datasets must remain unavailable.
    """
    policy = catalog.get("benchmark_policy", {})
    if (
        policy.get("purpose") != "2d-protein-benchmark"
        or policy.get("reconstruction_allowed") is not False
    ):
        raise ValueError("external navigation requires a published 2D benchmark")
    source_ids = [source["id"] for source in catalog["sources"]]
    contexts = {}
    for row in catalog["datasets"]:
        if (
            row["id"] in contexts
            or row["source_id"] not in source_ids
            or row.get("reconstruction_id")
            or row.get("reconstruction_view")
            or row.get("evidence_kind") != "image"
            or not row.get("media")
        ):
            raise ValueError("external navigation requires unique published images")
        contexts[row["id"]] = {
            key: row[key] for key in ("source_id", "organ", "subject_id")
        }
    return dict(
        id="external-validation",
        label="External validation",
        href="external-validation/index.html",
        catalog_sources=source_ids,
        default_source=source_ids[0] if source_ids else "",
        export_fingerprint=catalog["fingerprint"],
        image_contexts=contexts,
    )


def build_external_validation(
    manifest: Mapping,
    datasets: Sequence[Mapping],
    output_dir: Path | str,
    *,
    updated_at: str,
    compute: Sequence[Mapping] = (),
) -> Path:
    """Bind every visible image/result to a versioned 2D benchmark acquisition.

    Test images require completed evaluation. The resulting catalog carries a
    benchmark policy only and is never accepted by the serial-stack publisher.
    """
    decision = validate_external_benchmark(manifest)
    images = {r["image_id"]: r for r in manifest["images"]}
    for row in datasets:
        source = images.get(row.get("image_id"))
        if (
            source is None
            or row.get("benchmark_fingerprint") != decision["fingerprint"]
        ):
            raise ValueError("external review record lacks an exact benchmark binding")
        if (
            any(row.get(k) != source[k] for k in ("source_id", "organ"))
            or row.get("subject_id") != source["donor_id"]
            or row.get("role") != source["role"]
        ):
            raise ValueError("external review identity conflicts with its acquisition")
        if source["role"] == "test":
            evidence = row.get("evaluation_evidence", {})
            if not row.get("evaluation_complete") or not evidence:
                raise ValueError("test images stay sealed until evaluation completes")
            evaluation_path = Path(evidence["path"])
            selection_path = Path(evidence["selection_path"])
            if (
                file_sha256(evaluation_path) != evidence["sha256"]
                or file_sha256(selection_path) != evidence["selection_sha256"]
            ):
                raise ValueError("evaluation evidence changed after completion")
            evaluation = json.loads(evaluation_path.read_text())
            selection = json.loads(selection_path.read_text())
            auth = evaluation.get("authorization", {})
            fit_auth = selection.get("authorization", {})
            if (
                evaluation.get("status") != "evaluated"
                or selection.get("stage") != "frozen_before_test"
                or auth.get("stage") != "evaluate"
                or auth.get("manifest_fingerprint") != decision["fingerprint"]
                or fit_auth.get("manifest_fingerprint") != decision["fingerprint"]
                or fit_auth.get("stage") != "fit"
                or not fit_auth.get("roles", {}).get("fit_ids")
                or source["image_id"] not in auth.get("image_ids", [])
                or auth.get("model_selection_sha256") != evidence["selection_sha256"]
                or evaluation.get("model_selection_sha256")
                != evidence["selection_sha256"]
                or any(
                    images[key]["role"] == "test"
                    for key in selection.get("authorization", {}).get("image_ids", [])
                )
            ):
                raise ValueError("test evidence lacks a frozen independent evaluation")
            audit_path = Path(evidence["audit_path"])
            if file_sha256(audit_path) != evidence["audit_sha256"]:
                raise ValueError("independent numerical audit changed")
            audit = json.loads(audit_path.read_text())
            if (
                audit.get("status") != "pass"
                or audit.get("evaluation_sha256") != evidence["sha256"]
                or audit.get("selection_sha256") != evidence["selection_sha256"]
            ):
                raise ValueError("completed evaluation requires its numerical audit")
        if row.get("evidence_kind") != "image" or not row.get("media"):
            raise ValueError("external validation requires actual image evidence")
        if row.get("reconstruction_id") or row.get("reconstruction_view"):
            raise ValueError("2D results cannot advertise reconstruction eligibility")
    source_ids = {r["source_id"] for r in datasets}
    sources = [s for s in manifest["sources"] if s["id"] in source_ids]
    return _write_catalog(
        sources,
        datasets,
        Path(output_dir),
        updated_at=updated_at,
        compute=compute,
        benchmark_policy=dict(
            id="external-brightfield-benchmark-1",
            manifest_fingerprint=decision["fingerprint"],
            purpose="2d-protein-benchmark",
            reconstruction_allowed=False,
        ),
    )
