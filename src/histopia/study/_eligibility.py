"""Evidence-bound eligibility for serial brightfield IHC review.

Eligibility is a review scope, not biological validation. A saved assessment
binds source documentation and visual QC to one immutable serial manifest.
Restains are acquisitions of one physical section, never extra depth planes.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from histopia.study._manifest import fingerprint
from histopia.study._serial import validate_serial_study

POLICY_ID = "serial-brightfield-ihc-1"
_DIGEST = re.compile(r"[a-f0-9]{64}\Z")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


def _digest(value: object) -> bool:
    return bool(_DIGEST.fullmatch(str(value)))


def assess_reconstruction_eligibility(study: Mapping, assessment: Mapping) -> dict:
    """Assess one block using explicit acquisition and reconstruction evidence.

    Public studies require documented physical Z. Previously reviewed internal
    studies may retain explicitly assumed Z and its original approval binding.
    Unknown/missing evidence fails closed. The assessment supplies acquisition QC
    bound to source identities, and whole-stack QC bound to the same study hash.
    No image dependencies are imported and no upstream approval is modified.
    """
    manifest = validate_serial_study(study)
    if study.get("fingerprint") != manifest["fingerprint"]:
        raise ValueError("serial manifest fingerprint is missing or stale")
    key = assessment.get("reconstruction_id")
    if not isinstance(key, str) or not _ID.fullmatch(key):
        raise ValueError("a URL-safe reconstruction identity is required")
    if assessment.get("study_fingerprint") != manifest["fingerprint"]:
        raise ValueError("eligibility assessment belongs to a different study")
    if not isinstance(assessment.get("external"), bool):
        raise ValueError("eligibility requires an explicit external flag")
    blocked: list[str] = []
    excluded: list[str] = []
    sections = {s["physical_section_id"]: s for s in manifest["sections"]}
    acquisitions = {a["acquisition_id"]: a for a in manifest["acquisitions"]}
    chosen = assessment.get("acquisition_ids", [])
    if (
        not chosen
        or len(set(chosen)) != len(chosen)
        or set(chosen) - acquisitions.keys()
    ):
        raise ValueError("select distinct known acquisitions for the reconstruction")
    selected = [acquisitions[a] for a in chosen]
    physical = sorted({a["physical_section_id"] for a in selected})
    planes = [sections[p] for p in physical]
    identities = {
        tuple(
            s[k]
            for k in (
                "source_id",
                "species",
                "subject_id",
                "organ",
                "specimen_id",
                "block_id",
            )
        )
        for s in planes
    }
    if len(identities) != 1:
        excluded.append("mixed_tissue_blocks")
    if len(physical) < 3:
        excluded.append("fewer_than_three_physical_sections")
    modalities = sorted({a["modality"] for a in selected})
    if set(modalities) - {"H&E", "IHC"}:
        excluded.append("non_brightfield_he_ihc_acquisition")
    if "IHC" not in modalities:
        excluded.append("no_chromogenic_ihc")
    qc = assessment.get("acquisition_qc", {})
    image_planes: dict[str, str] = {}
    for acquisition in selected:
        check = qc.get(acquisition["acquisition_id"], {})
        digest = check.get("image_sha256")
        if _digest(digest):
            previous = image_planes.setdefault(
                digest, acquisition["physical_section_id"]
            )
            if previous != acquisition["physical_section_id"]:
                excluded.append("repeated_image_counted_as_physical_sections")
        if check.get("source_binding_fingerprint") != fingerprint(
            acquisition["source_binding"]
        ):
            blocked.append("unverified_acquisition_binding")
        if check.get("imaging_mode") not in {None, "brightfield"}:
            excluded.append("fluorescence_or_other_imaging")
        if check.get("verified_modality") not in {None, acquisition["modality"]}:
            excluded.append("stain_identity_conflict")
        if (
            check.get("imaging_mode") != "brightfield"
            or check.get("verified_modality") != acquisition["modality"]
            or not _digest(check.get("documentation_sha256"))
            or check.get("native_visual_qc") != "pass"
            or not _digest(check.get("native_visual_qc_sha256"))
        ):
            blocked.append("staining_not_verified")
        if check.get("image_available") is not True or not _digest(
            check.get("image_sha256")
        ):
            blocked.append("image_unavailable_or_unbound")
    provenance = assessment.get("physical_provenance", {})
    if not all(
        provenance.get(k) is True for k in ("block_verified", "order_verified")
    ) or not _digest(provenance.get("evidence_sha256")):
        blocked.append("physical_serial_identity_unverified")
    z_kind = provenance.get("spacing_status")
    if (
        any(s.get("z_um") is None for s in planes)
        or len({s.get("z_reference") for s in planes}) != 1
    ):
        blocked.append("physical_depth_unknown_or_inconsistent")
    if z_kind != "documented":
        if not (
            z_kind == "assumed"
            and assessment["external"] is False
            and _digest(provenance.get("prior_approval_sha256"))
        ):
            blocked.append("documented_spacing_required")
    reconstruction = assessment.get("reconstruction_qc", {})
    if (
        reconstruction.get("study_fingerprint") != manifest["fingerprint"]
        or set(reconstruction.get("reviewed_section_ids", [])) != set(physical)
        or not all(
            reconstruction.get(k) == "pass"
            for k in ("continuity", "alignment", "coverage", "stack_visual")
        )
        or not all(
            _digest(reconstruction.get(k))
            for k in ("registration_fingerprint", "stack_sha256", "visual_qc_sha256")
        )
    ):
        blocked.append("bounded_reconstruction_qc_pending")
    identity = (
        dict(
            zip(
                (
                    "source_id",
                    "species",
                    "subject_id",
                    "organ",
                    "specimen_id",
                    "block_id",
                ),
                next(iter(identities)),
                strict=True,
            )
        )
        if len(identities) == 1
        else {}
    )
    result = dict(
        policy_id=POLICY_ID,
        reconstruction_id=key,
        **identity,
        external=assessment["external"],
        status="excluded" if excluded else "pending" if blocked else "eligible",
        reasons=sorted(set(excluded + blocked)),
        modalities=modalities,
        acquisition_ids=sorted(chosen),
        acquisition_modalities={a["acquisition_id"]: a["modality"] for a in selected},
        physical_section_ids=physical,
        spacing_status=z_kind,
        study_fingerprint=manifest["fingerprint"],
        assessment_fingerprint=fingerprint(assessment),
        scientific_approval=False,
    )
    result["fingerprint"] = fingerprint(result)
    return result


def validate_reconstruction_registry(registry: Mapping) -> dict:
    """Recompute every decision; never trust a caller-supplied eligible flag."""
    if (
        registry.get("schema_version") != "reconstruction-registry-1"
        or registry.get("policy_id") != POLICY_ID
    ):
        raise ValueError("a versioned serial IHC eligibility registry is required")
    decisions = [
        assess_reconstruction_eligibility(e["study"], e["assessment"])
        for e in registry.get("entries", [])
    ]
    if len({d["reconstruction_id"] for d in decisions}) != len(decisions):
        raise ValueError("duplicate reconstruction identity")
    result = dict(
        schema_version="reconstruction-registry-1",
        policy_id=POLICY_ID,
        decisions=decisions,
    )
    result["fingerprint"] = fingerprint(result)
    return result


def filter_reconstruction_catalog(
    sources: Sequence[Mapping], datasets: Sequence[Mapping], registry: Mapping
) -> tuple[list, list, list, dict]:
    """Keep only results explicitly bound to an eligible physical stack.

    Filtering precedes asset copying and all public serialization. Exclusion
    reasons stay in the caller's operational audit, outside the review catalog.
    """
    validated = validate_reconstruction_registry(registry)
    decisions = {d["reconstruction_id"]: d for d in validated["decisions"]}
    source_map = {s["id"]: s for s in sources}
    kept, removed = [], []
    for row in datasets:
        decision = decisions.get(row.get("reconstruction_id"))
        reason = None
        if not decision or decision["status"] != "eligible":
            reason = "reconstruction_not_eligible"
        elif row.get("eligibility_fingerprint") != decision["fingerprint"]:
            reason = "eligibility_binding_missing_or_stale"
        elif (
            any(
                row.get(k) != decision[k]
                for k in ("source_id", "organ", "subject_id", "specimen_id", "block_id")
            )
            or source_map.get(row.get("source_id"), {}).get("external")
            != decision["external"]
        ):
            reason = "reconstruction_identity_conflict"
        elif not row.get("acquisition_ids") or not set(row["acquisition_ids"]) <= set(
            decision["acquisition_ids"]
        ):
            reason = "acquisition_binding_missing_or_conflicting"
        elif set(row.get("acquisition_modalities", [])) != {
            decision["acquisition_modalities"][a] for a in row["acquisition_ids"]
        }:
            reason = "modality_binding_missing_or_conflicting"
        elif (
            row.get("physical_section_id")
            and row["physical_section_id"] not in decision["physical_section_ids"]
        ):
            reason = "section_outside_reconstruction"
        elif row.get("evidence_kind") == "inventory":
            reason = "unrelated_inventory"
        if reason:
            removed.append(dict(id=row["id"], reason=reason))
        else:
            kept.append(dict(row))
    present = {r["source_id"] for r in kept}
    return [dict(s) for s in sources if s["id"] in present], kept, removed, validated
