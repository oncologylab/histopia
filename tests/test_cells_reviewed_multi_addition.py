from __future__ import annotations

import copy

import pytest

from histopia.cells._reviewed_addition import (
    REVIEWED_CACHE_ADDITION_INSTANCE_ACTION,
    _json_sha256,
)
from histopia.cells._reviewed_multi_addition import (
    REVIEWED_MULTI_ADDITION_SCOPE,
    require_disjoint_supports,
    reviewed_multi_addition_qc_evidence,
    validate_reviewed_multi_addition_manifest,
    validate_reviewed_multi_addition_qc,
)


def _manifest():
    source = {
        "section": "005",
        "source_identity": "c" * 64,
        "source_labels_sha256": "d" * 64,
        "source_qc_sha256": "e" * 64,
    }
    patches = []
    for index, x in enumerate((0, 100)):
        ids = [4, 8]
        patches.append(
            {
                **source,
                "raw_tile": {
                    "tile": f"r0000_c{index:04d}_y0_x{x}.npz",
                    "x": x,
                    "y": 0,
                    "width": 100,
                    "height": 100,
                    "source_cache_sha256": "f" * 64,
                    "source_tile_fingerprint": "1" * 64,
                },
                "selected_tile_label_ids": ids,
                "selected_tile_label_ids_sha256": _json_sha256(ids),
                "selection": {
                    "instance_action": REVIEWED_CACHE_ADDITION_INSTANCE_ACTION,
                    "native_coordinate_space": "source_wsi_pixels",
                    "support_geometry": {"bbox_xywh": [x, 0, 100, 100]},
                    "containment_rule": (
                        "all-selected-instance-pixels-inside-support-v1"
                    ),
                    "source_overlap_rule": "zero-source-overlap-pixels-v1",
                    "minimum_area_px": 15,
                    "selected_instance_count": 2,
                    "selected_pixels": 40,
                    "first_output_id": 101 + 2 * index,
                    "last_output_id": 102 + 2 * index,
                    "changed_source_pixels": 0,
                    "changed_outside_support_pixels": 0,
                },
                "review": {
                    "decision": "approved",
                    "reviewer": "test-reviewer",
                    "reviewed_at": "2026-09-08T23:00:00+00:00",
                    "evidence_sha256s": [str(index + 2) * 64],
                    "notes": "Distinct tissue island.",
                },
            }
        )
    return {
        "schema_version": 1,
        "scope": REVIEWED_MULTI_ADDITION_SCOPE,
        "source_algorithm_version": 75,
        "source_result_fingerprint": "a" * 64,
        "source_preflight_fingerprint": "b" * 64,
        "sections": [{**source, "patches": patches}],
    }


def test_distinct_tiles_may_reuse_raw_ids_but_receive_distinct_output_ids():
    manifest = _manifest()
    row = validate_reviewed_multi_addition_manifest(manifest)
    assert len(row["patches"]) == 2
    evidence = reviewed_multi_addition_qc_evidence(manifest)
    assert evidence["selected_instance_count"] == 4
    assert evidence["selected_pixels"] == 80
    assert (evidence["first_output_id"], evidence["last_output_id"]) == (101, 104)


def test_touching_supports_pass_but_one_pixel_overlap_fails():
    require_disjoint_supports(((0, 0, 100, 100), (100, 0, 100, 100)))
    with pytest.raises(ValueError, match="overlap"):
        require_disjoint_supports(((0, 0, 100, 100), (99, 0, 100, 100)))


@pytest.mark.parametrize(
    "mutation",
    ["source", "duplicate_output", "raw_ids", "outside_tile", "no_review", "path"],
)
def test_each_patch_requires_exact_source_geometry_ids_and_review(mutation):
    manifest = _manifest()
    patch = manifest["sections"][0]["patches"][1]
    if mutation == "source":
        patch["source_identity"] = "9" * 64
    elif mutation == "duplicate_output":
        patch["selection"].update(first_output_id=101, last_output_id=102)
    elif mutation == "raw_ids":
        patch["selected_tile_label_ids"][0] = 3
    elif mutation == "outside_tile":
        patch["selection"]["support_geometry"]["bbox_xywh"][2] = 101
    elif mutation == "no_review":
        patch["review"]["decision"] = "pending"
    else:
        patch["review"]["notes"] = "/private/image.tiff"
    with pytest.raises(ValueError):
        validate_reviewed_multi_addition_manifest(manifest)


def test_qc_binds_every_patch_and_refuses_source_changes():
    manifest = _manifest()
    qc = {
        "reviewed_multi_cache_addition": reviewed_multi_addition_qc_evidence(manifest),
        "tiles_inferred": 0,
        "tiles_reused_from_prior_run": 3,
        "filter_upgrade_from_algorithm_version": 75,
        "filter_upgrade_source_labels_sha256": "d" * 64,
        "outside_tissue_pixels_final": 0,
        "cell_count": 104,
        "foreground_pixels": 2000,
        "reviewed_multi_cache_addition_instances_added": 4,
        "reviewed_multi_cache_addition_pixels_added": 80,
        "reviewed_multi_cache_addition_pixels_removed": 0,
        "reviewed_multi_cache_addition_changed_source_pixels": 0,
        "reviewed_multi_cache_addition_changed_outside_support_pixels": 0,
    }
    validate_reviewed_multi_addition_qc(qc, manifest)
    changed = copy.deepcopy(manifest)
    changed["sections"][0]["patches"][1]["review"]["evidence_sha256s"] = ["9" * 64]
    with pytest.raises(ValueError, match="stale"):
        validate_reviewed_multi_addition_qc(qc, changed)
    qc["reviewed_multi_cache_addition_changed_source_pixels"] = 1
    with pytest.raises(ValueError, match="stale"):
        validate_reviewed_multi_addition_qc(qc, manifest)
