"""Provenance for reviewed containment masks generated from a new H-DAB input.

This profile preserves the original labels and adds only reviewed whole instances.
Unlike original-cache recovery, each candidate binds the native RGB pixels, the
fixed hematoxylin projection, and its own inference identity and mask archive.
Coverage improvement alone is never a review or a whole-section approval.
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
    _validate_selected_geometry,
)
from histopia.cells._reviewed_multi_addition import (
    _reviewed_addition_qc_evidence,
    _validate_reviewed_addition_provenance,
    _validate_reviewed_addition_qc,
    require_disjoint_supports,
)

REVIEWED_INPUT_ADDITION_ALGORITHM_VERSION = 102
REVIEWED_INPUT_ADDITION_METHOD_PROFILE = (
    "combined-containment-cpsam-reviewed-fixed-hdab-input-addition-v1"
)
REVIEWED_INPUT_ADDITION_SCOPE = "reviewed-fixed-hdab-input-whole-instance-addition-v1"
REVIEWED_INPUT_ADDITION_KEY = "reviewed_input_addition"
REVIEWED_INPUT_ADDITION_ACTION = "add_whole_reviewed_new_input_containment_labels"

FIXED_HDAB_PROJECTION = {
    "name": "fixed-hdab-hematoxylin-p99-purple-v1",
    "basis": [[0.650, 0.268], [0.704, 0.570], [0.286, 0.776]],
    "positive_hematoxylin_quantile": 0.99,
    "white_rgb": [255, 255, 255],
    "purple_rgb": [72, 54, 122],
}
_SOURCE_KEYS = {
    "section",
    "source_identity",
    "source_labels_sha256",
    "source_qc_sha256",
}
_PARAMETERS = {
    "model": "cpsam_v2",
    "method": "containment",
    "first_cellprob": -2.5,
    "first_diameter": 26,
    "second_cellprob": -1.75,
    "second_diameter": 15,
    "flow_threshold": 0.0,
    "merge_threshold": 0.1,
    "min_size": 15,
    "inference_batch_size": 32,
    "multiscale_nuclear_support": True,
    "nuclear_minimum_optical_density": 0.12,
    "nuclear_minimum_pixels": 3,
    "nuclear_minimum_fraction": 0.005,
}


@dataclass(frozen=True)
class ReviewedInputPatch:
    """A reviewed subset of one independently sealed new-input mask archive."""

    mask_archive: Path | str
    input_receipt: Path | str
    input_receipt_sha256: str
    support_bbox_xywh: tuple[int, int, int, int]
    selected_tile_label_ids: tuple[int, ...]
    evidence_paths: tuple[Path | str, ...]
    reviewer: str
    notes: str


def validate_input_tile(tile: object) -> dict[str, Any]:
    """Validate an explicit native-input and inference seal, without file access."""
    keys = {
        "x",
        "y",
        "width",
        "height",
        "mask_archive_sha256",
        "mask_array_sha256",
        "mask_fingerprint",
        "source_rgb_sha256",
        "projection",
        "input_identity",
        "generation_plan_sha256",
        "generation_worker_sha256",
    }
    if not isinstance(tile, dict) or set(tile) != keys:
        raise ValueError("invalid new-input tile fields")
    if any(type(tile[k]) is not int or tile[k] < 0 for k in ("x", "y")) or any(
        type(tile[k]) is not int or not 0 < tile[k] <= 8192 for k in ("width", "height")
    ):
        raise ValueError("invalid new-input tile geometry")
    if any(not _is_sha256(tile[k]) for k in keys if k.endswith("sha256")):
        raise ValueError("invalid new-input digest")
    projection = tile["projection"]
    if (
        not isinstance(projection, dict)
        or set(projection) != set(FIXED_HDAB_PROJECTION) | {"implementation_sha256"}
        or any(projection[k] != v for k, v in FIXED_HDAB_PROJECTION.items())
        or not _is_sha256(projection["implementation_sha256"])
    ):
        raise ValueError("new-input projection differs from the frozen H-DAB recipe")
    identity = tile["input_identity"]
    if (
        not isinstance(identity, dict)
        or set(identity)
        != {"schema_version", "rgb_sha256", "rgb_shape", "inference_binding"}
        or identity["schema_version"] != 1
        or identity["rgb_shape"] != [tile["height"], tile["width"], 3]
        or not _is_sha256(identity["rgb_sha256"])
        or tile["mask_fingerprint"] != _json_sha256(identity)
    ):
        raise ValueError("new-input inference identity is stale")
    binding = identity["inference_binding"]
    if (
        not isinstance(binding, dict)
        or set(binding)
        != {
            "model_weight_sha256",
            "algorithm_sha256",
            "adapter_sha256",
            "cellpose_version",
            "torch_version",
            "parameters",
        }
        or any(
            not _is_sha256(binding[k])
            for k in ("model_weight_sha256", "algorithm_sha256", "adapter_sha256")
        )
        or any(
            not isinstance(binding[k], str) or not binding[k].strip()
            for k in ("cellpose_version", "torch_version")
        )
        or binding["parameters"] != _PARAMETERS
    ):
        raise ValueError("new-input containment inference binding differs")
    _require_path_free_json(tile)
    return tile


def validate_input_receipt(value: object) -> dict[str, Any]:
    """Validate a source-bound tile receipt before composing any pixels."""
    if not isinstance(value, dict) or set(value) != _SOURCE_KEYS | {
        "schema_version",
        "source_result_fingerprint",
        "source_preflight_fingerprint",
        "input_tile",
    }:
        raise ValueError("invalid reviewed input receipt")
    if (
        value["schema_version"] != 1
        or not isinstance(value["section"], str)
        or len(value["section"]) != 3
        or not value["section"].isdigit()
        or any(
            not _is_sha256(value[k])
            for k in (
                "source_identity",
                "source_labels_sha256",
                "source_qc_sha256",
                "source_result_fingerprint",
                "source_preflight_fingerprint",
            )
        )
    ):
        raise ValueError("reviewed input source seal is invalid")
    validate_input_tile(value["input_tile"])
    _require_path_free_json(value)
    return value


def validate_reviewed_input_addition_manifest(value: object) -> dict[str, Any]:
    """Require exact new-input provenance and disjoint reviewed whole instances."""
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "scope",
        "source_algorithm_version",
        "source_result_fingerprint",
        "source_preflight_fingerprint",
        "sections",
    }:
        raise ValueError("invalid reviewed input addition manifest fields")
    if (
        value["schema_version"] != 1
        or value["scope"] != REVIEWED_INPUT_ADDITION_SCOPE
        or value["source_algorithm_version"] != 75
        or not all(
            _is_sha256(value[k])
            for k in ("source_result_fingerprint", "source_preflight_fingerprint")
        )
    ):
        raise ValueError("invalid reviewed input addition source")
    rows = value["sections"]
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        raise ValueError("reviewed input addition requires one section")
    row = rows[0]
    if set(row) != _SOURCE_KEYS | {"patches"}:
        raise ValueError("invalid reviewed input addition section")
    patches = row["patches"]
    if not isinstance(patches, list) or not 2 <= len(patches) <= 256:
        raise ValueError("reviewed input addition requires 2 to 256 patches")
    boxes, next_id, previous_binding = [], None, None
    for patch in patches:
        if not isinstance(patch, dict) or set(patch) != _SOURCE_KEYS | {
            "input_tile",
            "input_receipt_sha256",
            "selected_tile_label_ids",
            "selected_tile_label_ids_sha256",
            "selection",
            "review",
        }:
            raise ValueError("invalid reviewed new-input patch fields")
        if any(patch[k] != row[k] for k in _SOURCE_KEYS):
            raise ValueError("reviewed input patches have different sources")
        validate_input_receipt(
            {
                "schema_version": 1,
                **{k: row[k] for k in _SOURCE_KEYS},
                "source_result_fingerprint": value["source_result_fingerprint"],
                "source_preflight_fingerprint": value["source_preflight_fingerprint"],
                "input_tile": patch["input_tile"],
            }
        )
        if not _is_sha256(patch["input_receipt_sha256"]):
            raise ValueError("reviewed input receipt digest is missing")
        tile = patch["input_tile"]
        _validate_selected_geometry(
            patch, tile, instance_action=REVIEWED_INPUT_ADDITION_ACTION
        )
        binding = tile["input_identity"]["inference_binding"]
        if previous_binding is not None and binding != previous_binding:
            raise ValueError("mixed containment inference bindings")
        previous_binding = binding
        selection = patch["selection"]
        boxes.append(tuple(selection["support_geometry"]["bbox_xywh"]))
        if next_id is not None and selection["first_output_id"] != next_id:
            raise ValueError("reviewed input output IDs are not contiguous")
        next_id = selection["last_output_id"] + 1
    require_disjoint_supports(tuple(boxes))
    _require_path_free_json(value)
    return row


def validate_input_binding_to_source(
    manifest: object, model: Mapping[str, Any], request: Mapping[str, Any]
) -> None:
    """Keep source weights and all actual per-tile containment parameters fixed."""
    row = validate_reviewed_input_addition_manifest(manifest)
    binding = row["patches"][0]["input_tile"]["input_identity"]["inference_binding"]
    if binding["model_weight_sha256"] != model.get("weight_sha256") or binding[
        "cellpose_version"
    ] != model.get("cellpose_version"):
        raise ValueError("new-input model differs from original source")
    nuclear = request.get("multiscale_nuclear_support", {})
    parameters = {
        k: request.get(k)
        for k in _PARAMETERS
        if k
        not in {
            "multiscale_nuclear_support",
            "nuclear_minimum_optical_density",
            "nuclear_minimum_pixels",
            "nuclear_minimum_fraction",
        }
    }
    if not isinstance(nuclear, dict):
        raise ValueError("source nuclear support parameters are missing")
    parameters.update(
        multiscale_nuclear_support=nuclear.get("enabled"),
        nuclear_minimum_optical_density=nuclear.get("minimum_optical_density"),
        nuclear_minimum_pixels=nuclear.get("minimum_pixels"),
        nuclear_minimum_fraction=nuclear.get("minimum_fraction"),
    )
    if parameters != binding["parameters"]:
        raise ValueError("new-input containment parameters differ from source")


def reviewed_input_addition_qc_evidence(manifest: object) -> dict[str, Any]:
    row = validate_reviewed_input_addition_manifest(manifest)
    return _reviewed_addition_qc_evidence(manifest, row, REVIEWED_INPUT_ADDITION_SCOPE)


def validate_reviewed_input_addition_qc(
    qc: Mapping[str, Any], manifest: object
) -> None:
    row = validate_reviewed_input_addition_manifest(manifest)
    _validate_reviewed_addition_qc(
        qc,
        row,
        reviewed_input_addition_qc_evidence(manifest),
        REVIEWED_INPUT_ADDITION_KEY,
    )
    if qc.get("new_input_mask_archives_reused") != len(
        {patch["input_tile"]["mask_fingerprint"] for patch in row["patches"]}
    ):
        raise ValueError("new-input archive reuse count is stale")


def validate_reviewed_input_addition_provenance(
    result: Mapping[str, Any],
    *,
    composite_entry: bool = False,
    expected_source_fingerprint: str | None = None,
) -> dict[str, Any]:
    container = result if composite_entry else result.get("request", {})
    manifest = container.get(REVIEWED_INPUT_ADDITION_KEY)
    row = validate_reviewed_input_addition_manifest(manifest)
    _validate_reviewed_addition_provenance(
        result,
        manifest,
        row,
        scope=REVIEWED_INPUT_ADDITION_SCOPE,
        version=REVIEWED_INPUT_ADDITION_ALGORITHM_VERSION,
        composite_entry=composite_entry,
        expected_source_fingerprint=expected_source_fingerprint,
    )
    if composite_entry:
        binding = row["patches"][0]["input_tile"]["input_identity"]["inference_binding"]
        if binding["model_weight_sha256"] != result.get("model_weight_sha256"):
            raise ValueError("composite new-input model differs from source")
    else:
        validate_input_binding_to_source(manifest, result.get("model", {}), container)
    return manifest
