"""Publish explicitly bound internal input QC separately from serial stacks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from histopia._atomic import write_text_atomic
from histopia.study._manifest import fingerprint
from histopia.visualization._results_catalog import (
    _DIGEST,
    _identifier,
    _write_catalog,
    catalog_navigation_context,
)


def build_internal_review(
    manifest: Mapping,
    datasets: Sequence[Mapping],
    output_dir: Path | str,
    *,
    updated_at: str,
    inventory_scan_count: int | None = None,
) -> Path:
    """Export completed internal image QC without implying reconstruction.

    Acquisitions are scans, not additional physical planes. The versioned
    manifest binds each result, its exact presentation and its scan identities.
    Public benchmarks and serial reconstruction use their own eligibility gates.
    Asset bytes are verified by the shared exporter before publication.
    """
    if manifest.get("schema_version") != "internal-input-review-1" or manifest.get(
        "fingerprint"
    ) != fingerprint({k: v for k, v in manifest.items() if k != "fingerprint"}):
        raise ValueError("internal review requires a fingerprinted input manifest")
    sources = manifest.get("sources", [])
    source_ids = {_identifier(s["id"]) for s in sources}
    if len(source_ids) != len(sources) or any(
        s.get("external") is not False for s in sources
    ):
        raise ValueError("internal input review accepts only internal sources")
    acquisitions = {}
    for scan in manifest.get("acquisitions", []):
        key = _identifier(scan["id"])
        if key in acquisitions or scan["source_id"] not in source_ids:
            raise ValueError("input acquisitions require unique known identities")
        if not _DIGEST.fullmatch(str(scan.get("source_sha256", ""))):
            raise ValueError("input acquisitions require a source image binding")
        acquisitions[key] = scan
    records = manifest.get("records", {})
    seen = set()
    for row in datasets:
        key = _identifier(row["id"])
        if key in seen or records.get(key) != fingerprint(row):
            raise ValueError("internal result differs from its manifest binding")
        seen.add(key)
        if (
            row.get("status") != "ready"
            or row.get("evidence_kind") != "image"
            or not row.get("media")
            or row.get("scientific_approval") is not False
            or any(
                row.get(k)
                for k in (
                    "reconstruction_id",
                    "reconstruction_view",
                    "eligibility_fingerprint",
                )
            )
        ):
            raise ValueError("internal inputs require completed provisional image QC")
        scan_ids = row.get("acquisition_ids", [])
        if not scan_ids or len(scan_ids) != len(set(scan_ids)):
            raise ValueError("internal results must bind their original acquisitions")
        for scan_id in scan_ids:
            scan = acquisitions.get(scan_id)
            if scan is None or any(
                row.get(k) != scan.get(k) for k in ("source_id", "organ", "subject_id")
            ):
                raise ValueError("internal result conflicts with its scan identity")
        if row.get("input_scan_id") not in scan_ids:
            raise ValueError("internal result needs an explicit displayed scan")
    if seen != set(records):
        raise ValueError("internal review is missing a manifest-bound result")
    path = _write_catalog(
        sources,
        datasets,
        Path(output_dir),
        updated_at=updated_at,
        input_review_policy={
            "id": "internal-input-review-1",
            "manifest_fingerprint": manifest["fingerprint"],
            "reconstruction_allowed": False,
        },
    )
    if inventory_scan_count is not None:
        if inventory_scan_count < len(acquisitions):
            raise ValueError("inventory count cannot be smaller than reviewed inputs")
        page = Path(output_dir) / "index.html"
        link = (
            '<a id="inventory-link" href="../index.html?view=organ-metadata" '
            'target="_top" style="font-size:12px;margin-left:auto">'
            f"{len(acquisitions):,} reviewed scans · "
            f"All {inventory_scan_count:,} scans ↗</a>"
        )
        write_text_atomic(
            page, page.read_text().replace("</header>", link + "</header>", 1)
        )
    return path


def internal_review_tab(catalog: Mapping) -> dict:
    """Describe the separate, image-first internal input review surface."""
    policy = catalog.get("input_review_policy", {})
    if (
        policy.get("id") != "internal-input-review-1"
        or policy.get("reconstruction_allowed") is not False
        or any(s.get("external") is not False for s in catalog["sources"])
    ):
        raise ValueError("internal navigation requires a published internal catalog")
    return dict(
        id="internal-inputs",
        label="Internal tissues",
        href="internal-inputs/index.html",
        export_fingerprint=catalog["fingerprint"],
        **catalog_navigation_context(catalog),
    )
