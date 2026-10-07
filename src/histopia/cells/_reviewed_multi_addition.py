"""Sealed whole-instance recovery in several disjoint reviewed native regions.

Each patch retains the single-tile recovery contract. Requiring disjoint
support rectangles makes cross-patch duplication impossible without clipping,
merging, or changing any accepted source instance. Overlapping proposals need
an explicitly reviewed partition and are refused by this narrow profile.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from histopia.cells._reviewed_addition import (
    _is_sha256,
    _json_sha256,
    _require_path_free_json,
    _validate_section,
)

REVIEWED_MULTI_ADDITION_ALGORITHM_VERSION = 101
REVIEWED_MULTI_ADDITION_METHOD_PROFILE = (
    "combined-containment-cpsam-reviewed-disjoint-cache-addition-v1"
)
REVIEWED_MULTI_ADDITION_SCOPE = "post-inference-reviewed-disjoint-cache-addition-v1"


@dataclass(frozen=True)
class ReviewedCachePatch:
    """Exact raw labels and evidence reviewed inside one native rectangle."""

    raw_tile: str
    support_bbox_xywh: tuple[int, int, int, int]
    selected_tile_label_ids: tuple[int, ...]
    evidence_paths: tuple[Path | str, ...]
    reviewer: str
    notes: str


def require_disjoint_supports(boxes: tuple[tuple[int, int, int, int], ...]) -> None:
    """Reject invalid or intersecting supports; touching edges are disjoint."""
    if not 2 <= len(boxes) <= 256:
        raise ValueError("reviewed multi-addition requires 2 to 256 patches")
    for index, box in enumerate(boxes):
        if (
            len(box) != 4
            or any(type(v) is not int for v in box)
            or min(box[:2]) < 0
            or min(box[2:]) <= 0
        ):
            raise ValueError("invalid reviewed support rectangle")
        x, y, w, h = box
        for xx, yy, ww, hh in boxes[:index]:
            if min(x + w, xx + ww) > max(x, xx) and min(y + h, yy + hh) > max(y, yy):
                raise ValueError("reviewed support rectangles overlap")


def validate_reviewed_multi_addition_manifest(value: object) -> dict[str, Any]:
    """Validate every tile seal, review, disjoint support, and output ID range."""
    keys = {
        "schema_version",
        "scope",
        "source_algorithm_version",
        "source_result_fingerprint",
        "source_preflight_fingerprint",
        "sections",
    }
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError("invalid reviewed multi-addition manifest fields")
    if (
        value["schema_version"] != 1
        or value["scope"] != REVIEWED_MULTI_ADDITION_SCOPE
        or value["source_algorithm_version"] != 75
        or not all(
            _is_sha256(value[k])
            for k in ("source_result_fingerprint", "source_preflight_fingerprint")
        )
    ):
        raise ValueError("invalid reviewed multi-addition source seal")
    rows = value["sections"]
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        raise ValueError("reviewed multi-addition requires exactly one section")
    row = rows[0]
    source_keys = {
        "section",
        "source_identity",
        "source_labels_sha256",
        "source_qc_sha256",
    }
    if set(row) != source_keys | {"patches"}:
        raise ValueError("invalid reviewed multi-addition section fields")
    patches = row["patches"]
    if not isinstance(patches, list) or not 2 <= len(patches) <= 256:
        raise ValueError("reviewed multi-addition requires 2 to 256 patches")
    boxes = []
    next_id = None
    for patch in patches:
        if not isinstance(patch, dict):
            raise ValueError("invalid reviewed patch")
        _validate_section(patch)
        if any(patch[key] != row[key] for key in source_keys):
            raise ValueError("reviewed patches have different source identities")
        selection = patch["selection"]
        boxes.append(tuple(selection["support_geometry"]["bbox_xywh"]))
        if next_id is not None and selection["first_output_id"] != next_id:
            raise ValueError("reviewed patch output IDs are not contiguous")
        next_id = selection["last_output_id"] + 1
    require_disjoint_supports(tuple(boxes))
    _require_path_free_json(value)
    return row


def reviewed_multi_addition_qc_evidence(manifest: object) -> dict[str, Any]:
    """Bind aggregate counts to every independently reviewed patch."""
    row = validate_reviewed_multi_addition_manifest(manifest)
    return _reviewed_addition_qc_evidence(manifest, row, REVIEWED_MULTI_ADDITION_SCOPE)


def _reviewed_addition_qc_evidence(
    manifest: object, row: Mapping[str, Any], scope: str
) -> dict[str, Any]:
    selections = [patch["selection"] for patch in row["patches"]]
    return {
        "schema_version": 1,
        "scope": scope,
        "manifest_sha256": _json_sha256(manifest),
        "section_sha256": _json_sha256(row),
        "patch_count": len(selections),
        "selected_instance_count": sum(
            s["selected_instance_count"] for s in selections
        ),
        "selected_pixels": sum(s["selected_pixels"] for s in selections),
        "first_output_id": selections[0]["first_output_id"],
        "last_output_id": selections[-1]["last_output_id"],
        "changed_source_pixels": 0,
        "changed_outside_support_pixels": 0,
    }


def validate_reviewed_multi_addition_qc(
    qc: Mapping[str, Any], manifest: object
) -> None:
    """Reject stale source, aggregate, inference, or geometry counters."""
    row = validate_reviewed_multi_addition_manifest(manifest)
    evidence = reviewed_multi_addition_qc_evidence(manifest)
    _validate_reviewed_addition_qc(qc, row, evidence, "reviewed_multi_cache_addition")


def _validate_reviewed_addition_qc(
    qc: Mapping[str, Any],
    row: Mapping[str, Any],
    evidence: Mapping[str, Any],
    key: str,
) -> None:
    if (
        qc.get(key) != evidence
        or qc.get("tiles_inferred") != 0
        or not isinstance(qc.get("tiles_reused_from_prior_run"), int)
        or qc["tiles_reused_from_prior_run"] <= 0
        or qc.get("filter_upgrade_from_algorithm_version") != 75
        or qc.get("filter_upgrade_source_labels_sha256") != row["source_labels_sha256"]
        or qc.get("outside_tissue_pixels_final") != 0
        or qc.get(f"{key}_instances_added") != evidence["selected_instance_count"]
        or qc.get(f"{key}_pixels_added") != evidence["selected_pixels"]
        or qc.get(f"{key}_pixels_removed") != 0
        or qc.get(f"{key}_changed_source_pixels") != 0
        or qc.get(f"{key}_changed_outside_support_pixels") != 0
        or not isinstance(qc.get("cell_count"), int)
        or qc["cell_count"] <= evidence["selected_instance_count"]
        or not isinstance(qc.get("foreground_pixels"), int)
        or qc["foreground_pixels"] <= evidence["selected_pixels"]
    ):
        raise ValueError("reviewed multi-addition QC is stale")


def validate_reviewed_multi_addition_provenance(
    result: Mapping[str, Any],
    *,
    composite_entry: bool = False,
    expected_source_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Validate exact source binding for a section or its composite entry."""
    container = result if composite_entry else result.get("request", {})
    manifest = container.get("reviewed_multi_cache_addition")
    row = validate_reviewed_multi_addition_manifest(manifest)
    _validate_reviewed_addition_provenance(
        result,
        manifest,
        row,
        scope=REVIEWED_MULTI_ADDITION_SCOPE,
        version=REVIEWED_MULTI_ADDITION_ALGORITHM_VERSION,
        composite_entry=composite_entry,
        expected_source_fingerprint=expected_source_fingerprint,
    )
    return manifest


def _validate_reviewed_addition_provenance(
    result: Mapping[str, Any],
    manifest: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    scope: str,
    version: int,
    composite_entry: bool,
    expected_source_fingerprint: str | None,
) -> None:
    subset = result.get("subset_source")
    if (
        not isinstance(subset, dict)
        or set(subset)
        != {
            "scope",
            "source_result_fingerprint",
            "source_preflight_fingerprint",
            "source_profile_fingerprint",
            "source_section_fingerprint",
            "source_labels_sha256",
        }
        or subset["scope"] != scope
        or subset["source_result_fingerprint"] != manifest["source_result_fingerprint"]
        or subset["source_preflight_fingerprint"]
        != manifest["source_preflight_fingerprint"]
        or subset["source_labels_sha256"] != row["source_labels_sha256"]
        or not all(
            _is_sha256(subset[k])
            for k in ("source_profile_fingerprint", "source_section_fingerprint")
        )
        or (
            expected_source_fingerprint is not None
            and manifest["source_result_fingerprint"] != expected_source_fingerprint
        )
    ):
        raise ValueError("reviewed multi-addition source provenance is stale")
    if composite_entry:
        if (
            result.get("source_preflight_fingerprint")
            != manifest["source_preflight_fingerprint"]
        ):
            raise ValueError("composite reviewed multi-addition preflight is stale")
    else:
        rows = result.get("slides")
        if (
            not isinstance(rows, list)
            or len(rows) != 1
            or rows[0].get("section") != row["section"]
            or rows[0].get("source_identity") != row["source_identity"]
            or result.get("preflight_fingerprint")
            != manifest["source_preflight_fingerprint"]
            or result.get("profile_fingerprint")
            != _json_sha256(
                {
                    "algorithm_version": version,
                    "preflight_fingerprint": result.get("preflight_fingerprint"),
                    "model": result.get("model"),
                    "request": result.get("request"),
                }
            )
        ):
            raise ValueError("reviewed multi-addition result profile is stale")
