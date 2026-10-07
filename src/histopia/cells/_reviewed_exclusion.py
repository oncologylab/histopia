"""Immutable provenance for bounded, visually reviewed artifact exclusions.

Some slide artifacts are real but cannot be distinguished safely by one
cohort-wide color or focus threshold.  This module defines a deliberately
narrow escape hatch: a reviewer may approve exact whole-label removals in
native source coordinates, but the decision must be path-free and bound to
the source result, preflight, slide identity, label digest, reviewed evidence,
and sorted label IDs.  The refilter never infers or expands this manifest.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Any

REVIEWED_ARTIFACT_EXCLUSION_ALGORITHM_VERSION = 91
REVIEWED_ARTIFACT_EXCLUSION_METHOD_PROFILE = "combined-containment-cpsam-wsi-v86"
REVIEWED_ADDITION_EXCLUSION_ALGORITHM_VERSION = 98
REVIEWED_ADDITION_EXCLUSION_METHOD_PROFILE = (
    "combined-containment-cpsam-reviewed-cache-addition-and-exclusion-v1"
)
REVIEWED_ARTIFACT_EXCLUSION_SCOPE = "post-inference-reviewed-artifact-exclusion-v1"
REVIEWED_ARTIFACT_EXCLUSION_INSTANCE_ACTION = (
    "remove_whole_labels_selected_by_reviewed_native_regions"
)
REVIEWED_ARTIFACT_EXCLUSION_SOURCE_ALGORITHM_VERSION = 75
REVIEWED_ADDITION_EXCLUSION_SOURCE_ALGORITHM_VERSION = 97
REVIEWED_ARTIFACT_EXCLUSION_SOURCE_ALGORITHM_VERSIONS = frozenset(
    {
        REVIEWED_ARTIFACT_EXCLUSION_SOURCE_ALGORITHM_VERSION,
        REVIEWED_ADDITION_EXCLUSION_SOURCE_ALGORITHM_VERSION,
    }
)

_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "scope",
        "source_algorithm_version",
        "source_result_fingerprint",
        "source_preflight_fingerprint",
        "sections",
    }
)
_SECTION_KEYS = frozenset(
    {
        "section",
        "source_identity",
        "source_labels_sha256",
        "label_ids",
        "label_ids_sha256",
        "selection",
        "review",
    }
)
_SELECTION_KEYS = frozenset({"instance_action", "regions"})
_REGION_KEYS = frozenset(
    {
        "region_id",
        "kind",
        "native_coordinate_space",
        "geometry",
        "selection_rule",
        "parameters",
    }
)
_REVIEW_KEYS = frozenset(
    {"decision", "reviewer", "reviewed_at", "evidence_sha256s", "notes"}
)
_SECTION_PAYLOAD_KEYS = frozenset(
    {
        "schema_version",
        "scope",
        "source_algorithm_version",
        "source_result_fingerprint",
        "source_preflight_fingerprint",
        "section",
    }
)


def validate_reviewed_artifact_exclusion_manifest(
    value: object,
    *,
    expected_sections: Iterable[str] | None = None,
) -> dict[str, dict[str, object]]:
    """Validate an exact reviewed-exclusion manifest and return its sections."""

    manifest = _object(value, "reviewed artifact exclusion manifest")
    _require_exact_keys(manifest, _MANIFEST_KEYS, "reviewed exclusion manifest")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("scope") != REVIEWED_ARTIFACT_EXCLUSION_SCOPE
        or manifest.get("source_algorithm_version")
        not in REVIEWED_ARTIFACT_EXCLUSION_SOURCE_ALGORITHM_VERSIONS
        or not _is_sha256(manifest.get("source_result_fingerprint"))
        or not _is_sha256(manifest.get("source_preflight_fingerprint"))
    ):
        raise ValueError("reviewed exclusion source seal is invalid")
    rows = manifest.get("sections")
    if (
        not isinstance(rows, list)
        or not rows
        or len(rows) > 10_000
        or any(not isinstance(row, dict) for row in rows)
    ):
        raise ValueError("reviewed exclusion sections are invalid")

    output: dict[str, dict[str, object]] = {}
    for raw in rows:
        assert isinstance(raw, dict)
        _validate_section(raw)
        section = str(raw["section"])
        if section in output:
            raise ValueError("reviewed exclusion sections must be unique")
        output[section] = raw
    expected = tuple(str(section) for section in expected_sections or output)
    if tuple(output) != expected:
        raise ValueError("reviewed exclusion section order is stale")
    _require_path_free_json(manifest)
    return output


def reviewed_artifact_exclusion_manifest_sha256(value: object) -> str:
    """Return the canonical digest after validating a complete manifest."""

    validate_reviewed_artifact_exclusion_manifest(value)
    return _json_sha256(value)


def reviewed_artifact_exclusion_section_sha256(value: object) -> str:
    """Return the canonical digest after validating one section entry."""

    section = _object(value, "reviewed exclusion section")
    _validate_section(section)
    _require_path_free_json(section)
    return _json_sha256(section)


def reviewed_artifact_exclusion_section_payload(
    manifest: object,
    section: str,
) -> dict[str, object]:
    """Extract a self-contained section seal for a composed result."""

    value = _object(manifest, "reviewed artifact exclusion manifest")
    sections = validate_reviewed_artifact_exclusion_manifest(value)
    if section not in sections:
        raise ValueError(f"reviewed exclusion manifest omits section {section}")
    return {
        "schema_version": 1,
        "scope": value["scope"],
        "source_algorithm_version": value["source_algorithm_version"],
        "source_result_fingerprint": value["source_result_fingerprint"],
        "source_preflight_fingerprint": value["source_preflight_fingerprint"],
        "section": sections[section],
    }


def validate_reviewed_artifact_exclusion_section_payload(
    value: object,
    *,
    expected_section: str,
) -> dict[str, object]:
    """Validate the self-contained reviewed seal retained by a composite."""

    payload = _object(value, "composite reviewed exclusion payload")
    _require_exact_keys(
        payload,
        _SECTION_PAYLOAD_KEYS,
        "composite reviewed exclusion payload",
    )
    section = _object(payload.get("section"), "composite reviewed section")
    manifest = {
        "schema_version": payload.get("schema_version"),
        "scope": payload.get("scope"),
        "source_algorithm_version": payload.get("source_algorithm_version"),
        "source_result_fingerprint": payload.get("source_result_fingerprint"),
        "source_preflight_fingerprint": payload.get("source_preflight_fingerprint"),
        "sections": [section],
    }
    validate_reviewed_artifact_exclusion_manifest(
        manifest,
        expected_sections=(expected_section,),
    )
    _require_path_free_json(payload)
    return payload


def reviewed_artifact_exclusion_qc_evidence(
    manifest: object,
    section: str,
) -> dict[str, object]:
    """Return the compact evidence seal stored in section QC."""

    value = _object(manifest, "reviewed artifact exclusion manifest")
    rows = validate_reviewed_artifact_exclusion_manifest(value)
    if section not in rows:
        raise ValueError(f"reviewed exclusion manifest omits section {section}")
    row = rows[section]
    selection = _object(row["selection"], "reviewed exclusion selection")
    review = _object(row["review"], "reviewed exclusion review")
    return {
        "schema_version": 1,
        "scope": REVIEWED_ARTIFACT_EXCLUSION_SCOPE,
        "manifest_sha256": _json_sha256(value),
        "section_sha256": _json_sha256(row),
        "label_ids_sha256": row["label_ids_sha256"],
        "instance_action": selection["instance_action"],
        "review_decision": review["decision"],
        "evidence_sha256s": review["evidence_sha256s"],
    }


def _validate_section(section: dict[str, object]) -> None:
    _require_exact_keys(section, _SECTION_KEYS, "reviewed exclusion section")
    identifier = section.get("section")
    if (
        not isinstance(identifier, str)
        or len(identifier) != 3
        or not identifier.isdigit()
    ):
        raise ValueError("reviewed exclusion section ID must use three digits")
    if not _is_sha256(section.get("source_identity")) or not _is_sha256(
        section.get("source_labels_sha256")
    ):
        raise ValueError(f"section {identifier}: reviewed source binding is invalid")
    label_ids = section.get("label_ids")
    if (
        not isinstance(label_ids, list)
        or not label_ids
        or len(label_ids) > 2_000_000
        or any(
            not isinstance(label_id, int) or isinstance(label_id, bool) or label_id <= 0
            for label_id in label_ids
        )
        or label_ids != sorted(set(label_ids))
        or section.get("label_ids_sha256") != _json_sha256(label_ids)
    ):
        raise ValueError(f"section {identifier}: reviewed label IDs are invalid")

    selection = _object(section.get("selection"), "reviewed exclusion selection")
    _require_exact_keys(selection, _SELECTION_KEYS, "reviewed exclusion selection")
    regions = selection.get("regions")
    if (
        selection.get("instance_action") != REVIEWED_ARTIFACT_EXCLUSION_INSTANCE_ACTION
        or not isinstance(regions, list)
        or not regions
        or len(regions) > 100
        or any(not isinstance(region, dict) for region in regions)
    ):
        raise ValueError(f"section {identifier}: reviewed selection is invalid")
    region_ids: set[str] = set()
    for region in regions:
        assert isinstance(region, dict)
        _require_exact_keys(region, _REGION_KEYS, "reviewed exclusion region")
        region_id = region.get("region_id")
        if (
            not isinstance(region_id, str)
            or not region_id
            or len(region_id) > 128
            or region_id in region_ids
            or not isinstance(region.get("kind"), str)
            or not str(region["kind"])
            or len(str(region["kind"])) > 128
            or region.get("native_coordinate_space") != "source_wsi_pixels"
            or not isinstance(region.get("geometry"), dict)
            or not isinstance(region.get("selection_rule"), str)
            or not str(region["selection_rule"])
            or len(str(region["selection_rule"])) > 512
            or not isinstance(region.get("parameters"), dict)
        ):
            raise ValueError(f"section {identifier}: reviewed region is invalid")
        region_ids.add(region_id)

    review = _object(section.get("review"), "reviewed exclusion review")
    _require_exact_keys(review, _REVIEW_KEYS, "reviewed exclusion review")
    reviewer = review.get("reviewer")
    reviewed_at = review.get("reviewed_at")
    evidence = review.get("evidence_sha256s")
    notes = review.get("notes")
    if (
        review.get("decision") != "approved"
        or not isinstance(reviewer, str)
        or not reviewer.strip()
        or len(reviewer) > 128
        or not _valid_timestamp(reviewed_at)
        or not isinstance(evidence, list)
        or not evidence
        or evidence != list(dict.fromkeys(evidence))
        or any(not _is_sha256(digest) for digest in evidence)
        or not isinstance(notes, str)
        or len(notes) > 4096
    ):
        raise ValueError(f"section {identifier}: reviewed decision is invalid")


def _valid_timestamp(value: object) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None


def _require_exact_keys(
    value: Mapping[str, object], expected: frozenset[str], label: str
) -> None:
    if frozenset(value) != expected:
        raise ValueError(f"{label} fields are invalid")


def _require_path_free_json(value: object, *, key: str | None = None) -> None:
    if key is not None and "path" in key.casefold():
        raise ValueError("reviewed exclusion metadata must be path-free")
    if isinstance(value, dict):
        for child_key, child in value.items():
            if not isinstance(child_key, str):
                raise ValueError("reviewed exclusion metadata keys must be strings")
            _require_path_free_json(child, key=child_key)
        return
    if isinstance(value, list):
        for child in value:
            _require_path_free_json(child)
        return
    if isinstance(value, str):
        lowered = value.casefold()
        if value.startswith(("/", "\\\\")) or lowered.startswith("file:"):
            raise ValueError("reviewed exclusion metadata must be path-free")
        return
    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, float) and math.isfinite(value):
        return
    raise ValueError("reviewed exclusion metadata is not canonical JSON")


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
