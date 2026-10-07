from __future__ import annotations

import copy
import hashlib
import json

import pytest

from histopia.cells._reviewed_addition import (
    REVIEWED_CACHE_ADDITION_INSTANCE_ACTION,
    REVIEWED_CACHE_ADDITION_SCOPE,
    reviewed_cache_addition_manifest_sha256,
    reviewed_cache_addition_qc_evidence,
    validate_reviewed_cache_addition_manifest,
    validate_reviewed_cache_addition_qc,
)


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _manifest() -> dict[str, object]:
    ids = [4, 8, 12]
    return {
        "schema_version": 1,
        "scope": REVIEWED_CACHE_ADDITION_SCOPE,
        "source_algorithm_version": 75,
        "source_result_fingerprint": "a" * 64,
        "source_preflight_fingerprint": "b" * 64,
        "sections": [
            {
                "section": "005",
                "source_identity": "c" * 64,
                "source_labels_sha256": "d" * 64,
                "source_qc_sha256": "e" * 64,
                "raw_tile": {
                    "tile": "r0023_c0018_y20608_x16128.npz",
                    "x": 16128,
                    "y": 20608,
                    "width": 1024,
                    "height": 1024,
                    "source_cache_sha256": "f" * 64,
                    "source_tile_fingerprint": "1" * 64,
                },
                "selected_tile_label_ids": ids,
                "selected_tile_label_ids_sha256": _sha(ids),
                "selection": {
                    "instance_action": REVIEWED_CACHE_ADDITION_INSTANCE_ACTION,
                    "native_coordinate_space": "source_wsi_pixels",
                    "support_geometry": {"bbox_xywh": [16280, 20900, 800, 600]},
                    "containment_rule": (
                        "all-selected-instance-pixels-inside-support-v1"
                    ),
                    "source_overlap_rule": "zero-source-overlap-pixels-v1",
                    "minimum_area_px": 15,
                    "selected_instance_count": 3,
                    "selected_pixels": 300,
                    "first_output_id": 101,
                    "last_output_id": 103,
                    "changed_source_pixels": 0,
                    "changed_outside_support_pixels": 0,
                },
                "review": {
                    "decision": "approved",
                    "reviewer": "visual-audit",
                    "reviewed_at": "2026-09-02T11:30:00-04:00",
                    "evidence_sha256s": ["2" * 64],
                    "notes": "Coherent detached epithelium; raw contours match cells.",
                },
            }
        ],
    }


def test_reviewed_cache_addition_manifest_is_path_free_and_fingerprinted() -> None:
    manifest = _manifest()
    rows = validate_reviewed_cache_addition_manifest(
        manifest,
        expected_sections=("005",),
    )
    assert rows["005"]["selected_tile_label_ids"] == [4, 8, 12]
    assert len(reviewed_cache_addition_manifest_sha256(manifest)) == 64

    escaped = copy.deepcopy(manifest)
    escaped["sections"][0]["review"]["notes"] = "/private/source.ndpi"  # type: ignore[index]
    with pytest.raises(ValueError, match="path-free"):
        validate_reviewed_cache_addition_manifest(escaped)


def test_reviewed_cache_addition_requires_support_inside_raw_tile() -> None:
    manifest = _manifest()
    manifest["sections"][0]["selection"]["support_geometry"] = {  # type: ignore[index]
        "bbox_xywh": [16000, 20900, 800, 600]
    }
    with pytest.raises(ValueError, match="inside its raw tile"):
        validate_reviewed_cache_addition_manifest(manifest)


def test_reviewed_label_subset_only_restricts_existing_geometry_gates() -> None:
    from histopia.cells._reviewed_addition_compose import _reviewed_label_subset

    assert _reviewed_label_subset((12, 4), (4, 8, 12)) == (4, 12)
    with pytest.raises(ValueError, match="geometry gates"):
        _reviewed_label_subset((4, 99), (4, 8, 12))


@pytest.mark.parametrize("requested", [(), (4, 4), (True,), (0,), (-1,), (4.0,)])
def test_reviewed_label_subset_rejects_ambiguous_instance_ids(requested) -> None:
    from histopia.cells._reviewed_addition_compose import _reviewed_label_subset

    with pytest.raises(ValueError, match="unique positive integers"):
        _reviewed_label_subset(requested, (4, 8, 12))


def test_reviewed_cache_addition_rejects_stale_selected_ids() -> None:
    manifest = _manifest()
    manifest["sections"][0]["selected_tile_label_ids"] = [4, 12, 8]  # type: ignore[index]
    with pytest.raises(ValueError, match="tile labels"):
        validate_reviewed_cache_addition_manifest(manifest)


def test_reviewed_cache_addition_qc_binds_zero_mutation_counts() -> None:
    manifest = _manifest()
    qc = {
        "reviewed_cache_addition": reviewed_cache_addition_qc_evidence(
            manifest,
            "005",
        ),
        "tiles_inferred": 0,
        "tiles_reused_from_prior_run": 587,
        "filter_upgrade_from_algorithm_version": 75,
        "filter_upgrade_source_labels_sha256": "d" * 64,
        "reviewed_cache_addition_instances_added": 3,
        "reviewed_cache_addition_pixels_added": 300,
        "reviewed_cache_addition_pixels_removed": 0,
        "reviewed_cache_addition_changed_source_pixels": 0,
        "reviewed_cache_addition_changed_outside_support_pixels": 0,
        "cell_count": 1003,
        "foreground_pixels": 100_300,
    }
    validate_reviewed_cache_addition_qc(qc, manifest, section="005")
    qc["reviewed_cache_addition_changed_source_pixels"] = 1
    with pytest.raises(ValueError, match="QC provenance"):
        validate_reviewed_cache_addition_qc(qc, manifest, section="005")


def test_reviewed_cache_addition_qc_allows_exact_later_exclusion_source() -> None:
    manifest = _manifest()
    qc = {
        "reviewed_cache_addition": reviewed_cache_addition_qc_evidence(
            manifest,
            "005",
        ),
        "tiles_inferred": 0,
        "tiles_reused_from_prior_run": 587,
        "filter_upgrade_from_algorithm_version": 97,
        "filter_upgrade_source_labels_sha256": "3" * 64,
        "reviewed_cache_addition_instances_added": 3,
        "reviewed_cache_addition_pixels_added": 300,
        "reviewed_cache_addition_pixels_removed": 0,
        "reviewed_cache_addition_changed_source_pixels": 0,
        "reviewed_cache_addition_changed_outside_support_pixels": 0,
        "cell_count": 1001,
        "foreground_pixels": 100_100,
    }

    validate_reviewed_cache_addition_qc(
        qc,
        manifest,
        section="005",
        latest_filter_source_algorithm_version=97,
        latest_filter_source_labels_sha256="3" * 64,
    )
    qc["filter_upgrade_source_labels_sha256"] = "4" * 64
    with pytest.raises(ValueError, match="QC provenance"):
        validate_reviewed_cache_addition_qc(
            qc,
            manifest,
            section="005",
            latest_filter_source_algorithm_version=97,
            latest_filter_source_labels_sha256="3" * 64,
        )
