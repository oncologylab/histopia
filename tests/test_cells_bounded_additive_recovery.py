import hashlib
import json

import numpy as np
import pytest

from histopia.cells._bounded_additive_recovery import (
    BOUNDED_ADDITIVE_RECOVERY_SCOPE,
    apply_bounded_additions,
    bounded_additive_label_lookup,
    bounded_additive_recovery_parameters,
    bounded_support_geometry,
    select_bounded_additive_instances,
    validate_bounded_additive_recovery_provenance,
    validate_bounded_additive_recovery_qc,
)


def _selection():
    # IDs: 1 accepted, 2 too small, 3 crosses support, 4 overlaps source.
    return select_bounded_additive_instances(
        np.asarray([0, 6, 2, 7, 5], dtype=np.uint64),
        np.asarray([0, 6, 2, 6, 5], dtype=np.uint64),
        np.asarray([0, 0, 0, 0, 1], dtype=np.uint64),
        np.asarray([0, 6, 2, 6, 4], dtype=np.uint64),
        minimum_area_px=3,
    )


def test_bounded_additive_selection_is_exact_and_deterministic() -> None:
    selection = _selection()

    assert selection.selected_candidate_ids == (1,)
    assert selection.selected_areas_px == (6,)
    assert selection.candidate_instance_count == 4
    assert selection.rejected_below_minimum_count == 1
    assert selection.rejected_outside_support_count == 1
    assert selection.rejected_source_overlap_count == 1
    assert selection.selected_instance_count == 1
    assert selection.added_pixels == 6
    assert len(selection.selected_ids_sha256) == 64


def test_bounded_additive_lookup_uses_new_contiguous_ids() -> None:
    lookup = bounded_additive_label_lookup(
        _selection(), maximum_candidate_id=4, first_output_id=21
    )

    assert lookup.dtype == np.uint32
    assert lookup.tolist() == [0, 21, 0, 0, 0]


def test_bounded_additions_leave_source_and_outside_support_unchanged() -> None:
    source = np.asarray([[9, 0, 0], [0, 0, 7]], dtype=np.uint32)
    candidate = np.asarray([[0, 1, 1], [0, 1, 0]], dtype=np.uint32)
    support = np.asarray([[False, True, True], [False, True, False]])
    lookup = np.asarray([0, 10], dtype=np.uint32)

    output, added = apply_bounded_additions(source, candidate, support, lookup)

    assert added == 3
    assert output.tolist() == [[9, 10, 10], [0, 10, 7]]
    assert np.array_equal(output[~support], source[~support])


def test_bounded_additions_reject_mapped_pixels_outside_support() -> None:
    with pytest.raises(ValueError, match="outside bounded support"):
        apply_bounded_additions(
            np.zeros((1, 2), dtype=np.uint32),
            np.asarray([[1, 1]], dtype=np.uint32),
            np.asarray([[True, False]]),
            np.asarray([0, 2], dtype=np.uint32),
        )


def test_bounded_additions_reject_source_overlap() -> None:
    with pytest.raises(ValueError, match="overlaps a source"):
        apply_bounded_additions(
            np.asarray([[4]], dtype=np.uint32),
            np.asarray([[1]], dtype=np.uint32),
            np.asarray([[True]]),
            np.asarray([0, 5], dtype=np.uint32),
        )


def test_bounded_additive_parameters_are_explicit() -> None:
    parameters = bounded_additive_recovery_parameters(minimum_area_px=15)

    assert parameters["scope"] == BOUNDED_ADDITIVE_RECOVERY_SCOPE
    assert parameters["minimum_area_px"] == 15
    assert parameters["source_mutation"] == "none"


def _provenance() -> tuple[dict[str, object], dict[str, object]]:
    tiles = [
        {
            "tile": "r0000_c0000_y0_x0.npz",
            "x": 0,
            "y": 0,
            "width": 8,
            "height": 8,
            "recovery_fingerprint": "1" * 64,
            "file_sha256": "2" * 64,
        },
        {
            "tile": "r0000_c0001_y0_x6.npz",
            "x": 6,
            "y": 0,
            "width": 8,
            "height": 8,
            "recovery_fingerprint": "3" * 64,
            "file_sha256": "4" * 64,
        },
    ]
    support_core = {"kind": "exact-recovery-tile-union-v1", "tiles": tiles}
    fingerprint = hashlib.sha256(
        json.dumps(support_core, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    bbox, area = bounded_support_geometry(tiles)
    source = {
        "result_fingerprint": "a" * 64,
        "algorithm_version": 75,
        "profile_fingerprint": "b" * 64,
        "preflight_fingerprint": "c" * 64,
        "section_fingerprint": "d" * 64,
        "labels_sha256": "e" * 64,
        "qc_sha256": "f" * 64,
    }
    recovery = {
        "schema_version": 1,
        "scope": BOUNDED_ADDITIVE_RECOVERY_SCOPE,
        "parameters": bounded_additive_recovery_parameters(minimum_area_px=15),
        "source": source,
        "candidate": {
            "result_fingerprint": "0" * 64,
            "algorithm_version": 95,
            "profile_fingerprint": "1" * 64,
            "preflight_fingerprint": "2" * 64,
            "section_fingerprint": "3" * 64,
            "labels_sha256": "4" * 64,
            "qc_sha256": "5" * 64,
            "recovery_manifest_fingerprint": "6" * 64,
        },
        "section": "016",
        "source_identity": "7" * 64,
        "support": {
            **support_core,
            "fingerprint": fingerprint,
            "union_bbox_xywh": bbox,
            "union_pixels": area,
        },
        "selection": {
            "candidate_instance_count": 5,
            "selected_instance_count": 2,
            "rejected_below_minimum_count": 1,
            "rejected_outside_support_count": 1,
            "rejected_source_overlap_count": 1,
            "selected_ids_sha256": "8" * 64,
            "added_pixels": 48,
            "removed_pixels": 0,
            "changed_outside_support_pixels": 0,
            "first_output_id": 101,
            "last_output_id": 102,
        },
    }
    subset = {
        "scope": BOUNDED_ADDITIVE_RECOVERY_SCOPE,
        "source_result_fingerprint": source["result_fingerprint"],
        "source_preflight_fingerprint": source["preflight_fingerprint"],
        "source_profile_fingerprint": source["profile_fingerprint"],
        "source_section_fingerprint": source["section_fingerprint"],
        "source_labels_sha256": source["labels_sha256"],
    }
    return recovery, subset


def test_bounded_additive_provenance_and_qc_validate() -> None:
    recovery, subset = _provenance()
    validated = validate_bounded_additive_recovery_provenance(
        recovery,
        subset,
        section="016",
        source_identity="7" * 64,
    )
    validate_bounded_additive_recovery_qc(
        {
            "bounded_additive_recovery": validated,
            "tiles_inferred": 0,
            "tiles_reused_from_prior_run": 4,
            "filter_upgrade_from_algorithm_version": 75,
            "filter_upgrade_source_labels_sha256": "e" * 64,
            "bounded_additive_candidate_instances": 5,
            "bounded_additive_instances_added": 2,
            "bounded_additive_pixels_added": 48,
            "bounded_additive_pixels_removed": 0,
            "bounded_additive_changed_outside_support_pixels": 0,
            "cell_count": 12,
            "foreground_pixels": 1000,
        },
        validated,
    )


def test_bounded_additive_provenance_rejects_stale_union() -> None:
    recovery, subset = _provenance()
    recovery["support"]["union_pixels"] += 1

    with pytest.raises(ValueError, match="support geometry"):
        validate_bounded_additive_recovery_provenance(
            recovery,
            subset,
            section="016",
            source_identity="7" * 64,
        )


def test_bounded_additive_qc_rejects_any_removal() -> None:
    recovery, _subset = _provenance()
    with pytest.raises(ValueError, match="QC provenance"):
        validate_bounded_additive_recovery_qc(
            {
                "bounded_additive_recovery": recovery,
                "tiles_inferred": 0,
                "tiles_reused_from_prior_run": 4,
                "filter_upgrade_from_algorithm_version": 75,
                "filter_upgrade_source_labels_sha256": "e" * 64,
                "bounded_additive_candidate_instances": 5,
                "bounded_additive_instances_added": 2,
                "bounded_additive_pixels_added": 48,
                "bounded_additive_pixels_removed": 1,
                "bounded_additive_changed_outside_support_pixels": 0,
                "cell_count": 12,
                "foreground_pixels": 1000,
            },
            recovery,
        )
