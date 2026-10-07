"""Immutable provenance for native-ROI recovery from sealed raw cell tiles.

Whole-slide debris filters are intentionally conservative and can occasionally
remove a small, coherent tissue island.  This module defines a narrow recovery
path for that case.  A recovery must use one fingerprinted raw CPSAM tile from
the exact accepted source run, add only complete tile instances contained in a
reviewed native-coordinate rectangle, preserve every accepted source pixel,
and carry immutable visual-review evidence.

The schema is path-free so it remains valid after a section is incorporated
into a complete per-mouse composite.  File access and full-raster equivalence
checks are performed by the operational composer; this module validates the
scientific and cryptographic record retained in the result and section QC.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Any

REVIEWED_CACHE_ADDITION_ALGORITHM_VERSION = 97
REVIEWED_CACHE_ADDITION_METHOD_PROFILE = (
    "combined-containment-cpsam-reviewed-cache-addition-v1"
)
REVIEWED_CACHE_ADDITION_SCOPE = "post-inference-reviewed-cache-addition-v1"
REVIEWED_CACHE_ADDITION_SOURCE_ALGORITHM_VERSION = 75
REVIEWED_CACHE_ADDITION_INSTANCE_ACTION = (
    "add_whole_raw_cache_labels_inside_reviewed_native_region"
)

_TILE_KEY = re.compile(r"^r[0-9]{4}_c[0-9]{4}_y(?P<y>[0-9]+)_x(?P<x>[0-9]+)\.npz$")
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
        "source_qc_sha256",
        "raw_tile",
        "selected_tile_label_ids",
        "selected_tile_label_ids_sha256",
        "selection",
        "review",
    }
)
_TILE_KEYS = frozenset(
    {
        "tile",
        "x",
        "y",
        "width",
        "height",
        "source_cache_sha256",
        "source_tile_fingerprint",
    }
)
_SELECTION_KEYS = frozenset(
    {
        "instance_action",
        "native_coordinate_space",
        "support_geometry",
        "containment_rule",
        "source_overlap_rule",
        "minimum_area_px",
        "selected_instance_count",
        "selected_pixels",
        "first_output_id",
        "last_output_id",
        "changed_source_pixels",
        "changed_outside_support_pixels",
    }
)
_REVIEW_KEYS = frozenset(
    {"decision", "reviewer", "reviewed_at", "evidence_sha256s", "notes"}
)


def validate_reviewed_cache_addition_manifest(
    value: object,
    *,
    expected_sections: Iterable[str] | None = None,
) -> dict[str, dict[str, object]]:
    """Validate an exact reviewed raw-cache addition manifest."""

    manifest = _object(value, "reviewed cache addition manifest")
    _exact(manifest, _MANIFEST_KEYS, "reviewed cache addition manifest")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("scope") != REVIEWED_CACHE_ADDITION_SCOPE
        or manifest.get("source_algorithm_version")
        != REVIEWED_CACHE_ADDITION_SOURCE_ALGORITHM_VERSION
        or not _is_sha256(manifest.get("source_result_fingerprint"))
        or not _is_sha256(manifest.get("source_preflight_fingerprint"))
    ):
        raise ValueError("reviewed cache addition source seal is invalid")
    rows = manifest.get("sections")
    if (
        not isinstance(rows, list)
        or not rows
        or len(rows) > 10_000
        or any(not isinstance(row, dict) for row in rows)
    ):
        raise ValueError("reviewed cache addition sections are invalid")
    output: dict[str, dict[str, object]] = {}
    for raw in rows:
        assert isinstance(raw, dict)
        _validate_section(raw)
        section = str(raw["section"])
        if section in output:
            raise ValueError("reviewed cache addition sections must be unique")
        output[section] = raw
    expected = tuple(str(section) for section in expected_sections or output)
    if tuple(output) != expected:
        raise ValueError("reviewed cache addition section order is stale")
    _require_path_free_json(manifest)
    return output


def reviewed_cache_addition_manifest_sha256(value: object) -> str:
    """Return the canonical digest of a validated manifest."""

    validate_reviewed_cache_addition_manifest(value)
    return _json_sha256(value)


def reviewed_cache_addition_section_sha256(value: object) -> str:
    """Return the canonical digest of one validated section record."""

    section = _object(value, "reviewed cache addition section")
    _validate_section(section)
    _require_path_free_json(section)
    return _json_sha256(section)


def reviewed_cache_addition_qc_evidence(
    manifest: object,
    section: str,
) -> dict[str, object]:
    """Return the compact immutable seal stored in section QC."""

    value = _object(manifest, "reviewed cache addition manifest")
    rows = validate_reviewed_cache_addition_manifest(value)
    if section not in rows:
        raise ValueError(f"reviewed cache addition omits section {section}")
    row = rows[section]
    review = _object(row["review"], "reviewed cache addition review")
    selection = _object(row["selection"], "reviewed cache addition selection")
    return {
        "schema_version": 1,
        "scope": REVIEWED_CACHE_ADDITION_SCOPE,
        "manifest_sha256": _json_sha256(value),
        "section_sha256": _json_sha256(row),
        "selected_tile_label_ids_sha256": row["selected_tile_label_ids_sha256"],
        "instance_action": selection["instance_action"],
        "selected_instance_count": selection["selected_instance_count"],
        "selected_pixels": selection["selected_pixels"],
        "review_decision": review["decision"],
        "evidence_sha256s": review["evidence_sha256s"],
    }


def validate_reviewed_cache_addition_qc(
    qc: Mapping[str, object],
    manifest: Mapping[str, object],
    *,
    section: str,
    latest_filter_source_algorithm_version: int | None = None,
    latest_filter_source_labels_sha256: str | None = None,
) -> None:
    """Bind output QC counters to one reviewed addition manifest row.

    A direct reviewed addition is the latest filtering stage and therefore
    binds the generic filter-upgrade fields to its algorithm-75 source.  A
    later reviewed exclusion may preserve the addition while rebinding those
    generic fields to the intermediate algorithm-97 labels; callers must pass
    that exact source version and digest for the chained case.
    """

    rows = validate_reviewed_cache_addition_manifest(
        manifest,
        expected_sections=(section,),
    )
    row = rows[section]
    selection = _object(row["selection"], "reviewed cache addition selection")
    selected = int(selection["selected_instance_count"])
    pixels = int(selection["selected_pixels"])
    expected_source_algorithm = (
        REVIEWED_CACHE_ADDITION_SOURCE_ALGORITHM_VERSION
        if latest_filter_source_algorithm_version is None
        else latest_filter_source_algorithm_version
    )
    expected_source_labels = (
        str(row["source_labels_sha256"])
        if latest_filter_source_labels_sha256 is None
        else latest_filter_source_labels_sha256
    )
    if (
        qc.get("reviewed_cache_addition")
        != reviewed_cache_addition_qc_evidence(manifest, section)
        or qc.get("tiles_inferred") != 0
        or not _positive_int(qc.get("tiles_reused_from_prior_run"))
        or qc.get("filter_upgrade_from_algorithm_version") != expected_source_algorithm
        or qc.get("filter_upgrade_source_labels_sha256") != expected_source_labels
        or qc.get("reviewed_cache_addition_instances_added") != selected
        or qc.get("reviewed_cache_addition_pixels_added") != pixels
        or qc.get("reviewed_cache_addition_pixels_removed") != 0
        or qc.get("reviewed_cache_addition_changed_source_pixels") != 0
        or qc.get("reviewed_cache_addition_changed_outside_support_pixels") != 0
        or not _positive_int(qc.get("cell_count"))
        or int(qc["cell_count"]) <= selected
        or not _positive_int(qc.get("foreground_pixels"))
        or int(qc["foreground_pixels"]) <= pixels
    ):
        raise ValueError("reviewed cache addition QC provenance is stale")


def _validate_section(section: dict[str, object]) -> None:
    _exact(section, _SECTION_KEYS, "reviewed cache addition section")
    identifier = section.get("section")
    if (
        not isinstance(identifier, str)
        or len(identifier) != 3
        or not identifier.isdigit()
        or not _is_sha256(section.get("source_identity"))
        or not _is_sha256(section.get("source_labels_sha256"))
        or not _is_sha256(section.get("source_qc_sha256"))
    ):
        raise ValueError("reviewed cache addition section source is invalid")

    tile = _object(section.get("raw_tile"), "reviewed cache addition raw tile")
    _exact(tile, _TILE_KEYS, "reviewed cache addition raw tile")
    key = tile.get("tile")
    match = _TILE_KEY.fullmatch(str(key)) if isinstance(key, str) else None
    if (
        match is None
        or not _nonnegative_int(tile.get("x"))
        or not _nonnegative_int(tile.get("y"))
        or int(tile["x"]) != int(match.group("x"))
        or int(tile["y"]) != int(match.group("y"))
        or not _positive_int(tile.get("width"))
        or not _positive_int(tile.get("height"))
        or not _is_sha256(tile.get("source_cache_sha256"))
        or not _is_sha256(tile.get("source_tile_fingerprint"))
    ):
        raise ValueError(f"section {identifier}: reviewed raw tile is invalid")

    _validate_selected_geometry(section, tile)


def _validate_selected_geometry(
    section: dict[str, object],
    tile: Mapping[str, Any],
    *,
    instance_action: str = REVIEWED_CACHE_ADDITION_INSTANCE_ACTION,
) -> None:
    """Validate whole-instance geometry/review shared by explicit input profiles."""
    identifier = section.get("section")
    label_ids = section.get("selected_tile_label_ids")
    if (
        not isinstance(label_ids, list)
        or not label_ids
        or len(label_ids) > 1_000_000
        or any(not _positive_int(value) for value in label_ids)
        or label_ids != sorted(set(label_ids))
        or section.get("selected_tile_label_ids_sha256") != _json_sha256(label_ids)
    ):
        raise ValueError(f"section {identifier}: reviewed tile labels are invalid")

    selection = _object(section.get("selection"), "reviewed cache addition selection")
    _exact(selection, _SELECTION_KEYS, "reviewed cache addition selection")
    geometry = _object(
        selection.get("support_geometry"),
        "reviewed cache addition support geometry",
    )
    if set(geometry) != {"bbox_xywh"}:
        raise ValueError("reviewed cache addition support geometry is invalid")
    bbox = geometry.get("bbox_xywh")
    if (
        selection.get("instance_action") != instance_action
        or selection.get("native_coordinate_space") != "source_wsi_pixels"
        or selection.get("containment_rule")
        != "all-selected-instance-pixels-inside-support-v1"
        or selection.get("source_overlap_rule") != "zero-source-overlap-pixels-v1"
        or selection.get("minimum_area_px") != 15
        or selection.get("selected_instance_count") != len(label_ids)
        or not _positive_int(selection.get("selected_pixels"))
        or int(selection["selected_pixels"]) < 15 * len(label_ids)
        or not _positive_int(selection.get("first_output_id"))
        or selection.get("last_output_id")
        != int(selection["first_output_id"]) + len(label_ids) - 1
        or selection.get("changed_source_pixels") != 0
        or selection.get("changed_outside_support_pixels") != 0
        or not isinstance(bbox, list)
        or len(bbox) != 4
        or not all(_nonnegative_int(value) for value in bbox[:2])
        or not all(_positive_int(value) for value in bbox[2:])
    ):
        raise ValueError(f"section {identifier}: reviewed selection is invalid")
    assert isinstance(bbox, list)
    support_x, support_y, support_width, support_height = map(int, bbox)
    tile_x, tile_y = int(tile["x"]), int(tile["y"])
    if (
        support_x < tile_x
        or support_y < tile_y
        or support_x + support_width > tile_x + int(tile["width"])
        or support_y + support_height > tile_y + int(tile["height"])
    ):
        raise ValueError("reviewed addition support must be inside its raw tile")

    review = _object(section.get("review"), "reviewed cache addition review")
    _exact(review, _REVIEW_KEYS, "reviewed cache addition review")
    evidence = review.get("evidence_sha256s")
    notes = review.get("notes")
    if (
        review.get("decision") != "approved"
        or not isinstance(review.get("reviewer"), str)
        or not str(review["reviewer"]).strip()
        or len(str(review["reviewer"])) > 128
        or not _valid_timestamp(review.get("reviewed_at"))
        or not isinstance(evidence, list)
        or not evidence
        or evidence != list(dict.fromkeys(evidence))
        or any(not _is_sha256(value) for value in evidence)
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


def _exact(value: Mapping[str, object], expected: frozenset[str], label: str) -> None:
    if frozenset(value) != expected:
        raise ValueError(f"{label} fields are invalid")


def _require_path_free_json(value: object, *, key: str | None = None) -> None:
    if key is not None and "path" in key.casefold():
        raise ValueError("reviewed cache addition metadata must be path-free")
    if isinstance(value, dict):
        for child_key, child in value.items():
            if not isinstance(child_key, str):
                raise ValueError("reviewed cache addition keys must be strings")
            _require_path_free_json(child, key=child_key)
        return
    if isinstance(value, list):
        for child in value:
            _require_path_free_json(child)
        return
    if isinstance(value, str):
        lowered = value.casefold()
        if value.startswith(("/", "\\\\")) or lowered.startswith("file:"):
            raise ValueError("reviewed cache addition metadata must be path-free")
        return
    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, float) and math.isfinite(value):
        return
    raise ValueError("reviewed cache addition metadata is not canonical JSON")


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


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
