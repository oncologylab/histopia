import hashlib
import json
from pathlib import Path

import pytest

from histopia.cells import (
    approve_cell_result,
    cell_review_status,
    chromatic_microcluster_gate,
    compose_cell_result,
    review_cell_section,
    saturated_cyan_foreign_material_gate,
    validate_cell_approval,
    validate_cell_promotion_candidate,
    validate_cell_result,
)
from histopia.cells._bounded_additive_recovery import (
    BOUNDED_ADDITIVE_RECOVERY_METHOD_PROFILE,
    BOUNDED_ADDITIVE_RECOVERY_SCOPE,
    bounded_additive_recovery_parameters,
    bounded_support_geometry,
)
from histopia.cells._dense_small_cell_recovery import (
    DENSE_SMALL_CELL_ADJACENCY_RECOVERY_METHOD,
    DENSE_SMALL_CELL_RECOVERY_METHOD,
    dense_small_cell_adjacency_recovery_parameters,
    dense_small_cell_recovery_parameters,
)
from histopia.cells._result import write_cell_result
from histopia.cells._reviewed_exclusion import (
    REVIEWED_ARTIFACT_EXCLUSION_INSTANCE_ACTION,
    REVIEWED_ARTIFACT_EXCLUSION_SCOPE,
    reviewed_artifact_exclusion_manifest_sha256,
    reviewed_artifact_exclusion_qc_evidence,
    reviewed_artifact_exclusion_section_sha256,
)


@pytest.mark.parametrize("new_input", [False, True])
def test_reviewed_multi_addition_composes_real_small_rasters_once(
    tmp_path: Path,
    new_input: bool,
) -> None:
    """Two raw tiles reuse local IDs but preserve all source pixels and reviews."""
    import numpy as np
    from PIL import Image

    pyvips = pytest.importorskip("pyvips")
    from test_cells_reviewed_input_addition import _make_input_archive

    from histopia.cells import (
        ReviewedCachePatch,
        ReviewedInputPatch,
        compose_reviewed_input_addition,
        compose_reviewed_multi_cache_addition,
    )

    composer = (
        compose_reviewed_input_addition
        if new_input
        else compose_reviewed_multi_cache_addition
    )
    expected_model = "cpsam_v2" if new_input else "cpsam"

    source_root = tmp_path / "source"
    source_root.mkdir()
    _promotion_run(source_root)
    before = np.zeros((128, 256), dtype=np.uint32)
    before[10:15, 10:15] = 1
    labels_path = source_root / "labels/001.cells.tiff"
    pyvips.Image.new_from_memory(before.tobytes(), 256, 128, 1, "uint").tiffsave(
        str(labels_path)
    )
    mask_path = tmp_path / "tissue.png"
    Image.fromarray(np.full(before.shape, 255, dtype=np.uint8)).save(mask_path)
    preflight_path = source_root / "preflight.json"
    preflight = json.loads(preflight_path.read_text())
    preflight["fingerprint"] = "b" * 64
    geometry = {
        "source_identity": "c" * 64,
        "native_shape": [128, 256],
        "content_bbox_xywh": [0, 0, 256, 128],
        "mpp_xy": [1.0, 1.0],
        "is_reference": False,
        "transform_sha256": "e" * 64,
    }
    preflight["slides"][0].update(
        **geometry,
        thumbnail_shape=[128, 256],
        mask_path=str(mask_path),
        mask_sha256=hashlib.sha256(mask_path.read_bytes()).hexdigest(),
    )
    rgb = np.full((128, 256, 3), [130, 110, 160], dtype=np.uint8)
    native_path = tmp_path / "native.tiff"
    Image.fromarray(rgb).save(native_path)
    preflight["slides"][0]["source_path"] = str(native_path)
    preflight_path.write_text(json.dumps(preflight))
    core = json.loads((source_root / "cell_result.json").read_text())
    core.pop("fingerprint")
    core.pop("artifacts")
    core["profile_fingerprint"] = "a" * 64
    core["preflight_fingerprint"] = "b" * 64
    core["coordinate_space"] = "native_source"
    core["model"] = {"name": "cpsam", "weight_name": "cpsam", "weight_sha256": "f" * 64}
    if new_input:
        core["model"].update(
            name="cpsam_v2", weight_name="cpsam_v2", cellpose_version="4.2.1"
        )
        core["request"].update(
            model="cpsam_v2",
            inference_batch_size=32,
            first_diameter=26,
            second_diameter=15,
            min_size=15,
            first_cellprob=-2.5,
            second_cellprob=-1.75,
            flow_threshold=0.0,
            merge_threshold=0.1,
        )
    core["slides"][0].update(**geometry, cell_count=1)
    qc_path = source_root / "qc/001.json"
    qc = json.loads(qc_path.read_text())
    qc.update(
        cell_count=1,
        foreground_pixels=25,
        labels_sha256=hashlib.sha256(labels_path.read_bytes()).hexdigest(),
        profile_fingerprint="a" * 64,
        section_fingerprint="d" * 64,
    )
    qc_path.write_text(json.dumps(qc))
    write_cell_result(source_root, core)
    source_result = validate_cell_promotion_candidate(
        source_root, expected_model=expected_model
    )
    patches = []
    for index, x in enumerate((0, 128)):
        raw = np.zeros((128, 128), dtype=np.int32)
        raw[30:34, 30:34] = 4
        raw[50:55, 50:55] = 8
        # An extra raw instance overlaps the preserved source and is not selected.
        raw[10:15, 10:15] = 9
        tile = f"r0000_c{index:04d}_y0_x{x}.npz"
        cache = source_root / ".cell-cache/001" / tile
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache, mask=raw, fingerprint=np.asarray(str(index + 1) * 64)
        )
        evidence = tmp_path / f"review-{index}.txt"
        evidence.write_text(f"Reviewed synthetic tissue island {index}.")
        if new_input:
            archive, input_tile = _make_input_archive(
                tmp_path, raw, rgb[:, x : x + 128], x=x, index=index
            )
            receipt_path = tmp_path / f"receipt-{index}.json"
            receipt_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "section": "001",
                        "source_result_fingerprint": source_result["fingerprint"],
                        "source_preflight_fingerprint": source_result[
                            "preflight_fingerprint"
                        ],
                        "source_identity": geometry["source_identity"],
                        "source_labels_sha256": source_result["artifacts"][
                            "labels/001.cells.tiff"
                        ],
                        "source_qc_sha256": source_result["artifacts"]["qc/001.json"],
                        "input_tile": input_tile,
                    }
                )
            )
            patches.append(
                ReviewedInputPatch(
                    archive,
                    receipt_path,
                    hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
                    (x, 0, 128, 128),
                    (4, 8),
                    (evidence,),
                    "test-reviewer",
                    "Two intact synthetic cells.",
                )
            )
            continue
        patches.append(
            ReviewedCachePatch(
                tile,
                (x, 0, 128, 128),
                (4, 8),
                (evidence,),
                "test-reviewer",
                "Two intact synthetic cells.",
            )
        )
    output = tmp_path / "composed"
    phases = []
    composer(source_root, "001", tuple(patches), output, progress=phases.append)
    result = validate_cell_promotion_candidate(output, expected_model=expected_model)
    image = pyvips.Image.new_from_file(str(output / "labels/001.cells.tiff"))
    after = np.frombuffer(image.write_to_memory(), dtype=np.uint32).reshape(
        before.shape
    )
    expected = before.copy()
    for index, x in enumerate((0, 128)):
        expected[30:34, x + 30 : x + 34] = 2 + 2 * index
        expected[50:55, x + 50 : x + 55] = 3 + 2 * index
    np.testing.assert_array_equal(after, expected)
    assert result["slides"][0]["cell_count"] == 5
    assert [p["phase"] for p in phases].count("copying_exact_source_labels") == 1
    assert [p["phase"] for p in phases].count("writing_native_pyramidal_labels") == 1
    cohort = tmp_path / "cohort"
    compose_cell_result(
        source_root, cohort, {"001": output}, expected_model=expected_model
    )
    composite = validate_cell_promotion_candidate(cohort, expected_model=expected_model)
    assert composite["composition"]["section_sources"][0][
        "source_algorithm_version"
    ] == (102 if new_input else 101)
    # A second request must not replace or recompute the sealed candidate.
    with pytest.raises(FileExistsError):
        composer(source_root, "001", tuple(patches), output)

    if new_input:
        from dataclasses import replace

        # The source-overlapping whole instance must fail before allocating a canvas.
        invalid = (replace(patches[0], selected_tile_label_ids=(9,)), patches[1])
        bad_output = tmp_path / "invalid-overlap"
        with pytest.raises(ValueError):
            composer(source_root, "001", invalid, bad_output)
        assert not bad_output.exists()
        assert not list(tmp_path.glob(".invalid-overlap.*"))
        np.testing.assert_array_equal(
            np.frombuffer(
                pyvips.Image.new_from_file(str(labels_path)).write_to_memory(),
                np.uint32,
            ).reshape(before.shape),
            before,
        )


def _cell_run(tmp_path: Path) -> Path:
    (tmp_path / "preflight.json").write_text("{}\n")
    (tmp_path / "labels").mkdir()
    (tmp_path / "qc").mkdir()
    (tmp_path / "labels/001.cells.tiff").write_bytes(b"labels")
    (tmp_path / "qc/001.json").write_text("{}\n")
    write_cell_result(
        tmp_path,
        {
            "schema_version": 1,
            "registration_result_sha256": "a" * 64,
            "preflight": "preflight.json",
            "slides": [
                {
                    "section": "001",
                    "slide": "slide.ndpi",
                    "labels": "labels/001.cells.tiff",
                    "qc": "qc/001.json",
                }
            ],
        },
    )
    return tmp_path


def test_section_acceptance_is_one_click_and_final_approval_is_bound(
    tmp_path: Path,
) -> None:
    run = _cell_run(tmp_path)
    status = review_cell_section(
        run,
        "001",
        accepted=True,
        reviewer="browser",
    )
    assert status["approved"] is True
    approval = approve_cell_result(run)
    assert approval.sections == ("001",)
    assert validate_cell_approval(run).fingerprint == approval.fingerprint


def test_changed_artifact_invalidates_final_approval(tmp_path: Path) -> None:
    run = _cell_run(tmp_path)
    review_cell_section(run, "001", accepted=True, reviewer="browser")
    (run / "labels/001.cells.tiff").write_bytes(b"changed")

    with pytest.raises(ValueError, match="digest mismatch"):
        approve_cell_result(run)


def test_changed_result_resets_section_reviews(tmp_path: Path) -> None:
    run = _cell_run(tmp_path)
    review_cell_section(run, "001", accepted=True, reviewer="browser")
    result = validate_cell_result(run)
    core = {
        key: value
        for key, value in result.items()
        if key not in {"artifacts", "fingerprint"}
    }
    core["request"] = {"model": "cpsam"}
    write_cell_result(run, core)

    assert cell_review_status(run)["pending_sections"] == ["001"]
    assert json.loads((run / "cell_review.json").read_text())["approved"] is False


def test_cell_promotion_gate_requires_complete_guarded_reference_profile(
    tmp_path: Path,
) -> None:
    run = _promotion_run(tmp_path)

    result = validate_cell_promotion_candidate(run)

    assert result["algorithm_version"] == 75
    assert result["slides"][0]["section"] == "001"


def test_cell_promotion_gate_accepts_validated_dark_fold_profile(
    tmp_path: Path,
) -> None:
    run = _promotion_run(
        tmp_path,
        mutation={"fold_red_blue": 85.0, "fold_intensity": 130.0},
    )

    result = validate_cell_promotion_candidate(run)

    assert (
        result["request"]["isolated_debris_gate"][
            "fold_maximum_mean_red_blue_difference"
        ]
        == 85.0
    )


def test_cell_promotion_gate_accepts_validated_bright_microfoam_profile(
    tmp_path: Path,
) -> None:
    run = _promotion_run(
        tmp_path,
        mutation={
            "foam_core_pixels": 2,
            "foam_core_bright_fraction": 0.65,
            "foam_core_intensity": 120.0,
            "foam_core_instance_area_um2": 180.0,
            "foam_core_component_area_um2": 3000.0,
            "foam_core_aspect": 2.50,
        },
    )

    result = validate_cell_promotion_candidate(run)

    assert result["request"]["isolated_debris_gate"]["foam_core_minimum_pixels"] == 2


def test_cell_promotion_gate_accepts_validated_residual_foam_profile(
    tmp_path: Path,
) -> None:
    run = _promotion_run(
        tmp_path,
        mutation={
            "foam_core_pixels": 1,
            "foam_core_bright_fraction": 0.45,
            "foam_core_intensity": 100.0,
            "foam_core_instance_area_um2": 180.0,
            "foam_core_component_area_um2": 1000.0,
            "foam_core_aspect": 2.50,
        },
    )

    result = validate_cell_promotion_candidate(run)

    assert result["request"]["isolated_debris_gate"]["foam_core_minimum_pixels"] == 1


def test_cell_promotion_gate_accepts_validated_v82_cache_only_refilter(
    tmp_path: Path,
) -> None:
    run = _promotion_run(tmp_path, mutation={"algorithm_version": 82})

    result = validate_cell_promotion_candidate(run)

    assert result["algorithm_version"] == 82
    assert result["request"]["method_reference"]["profile"].endswith("v78")


def test_cell_promotion_gate_rejects_unsealed_v82_shape_policy(
    tmp_path: Path,
) -> None:
    run = _promotion_run(
        tmp_path,
        mutation={
            "algorithm_version": 82,
            "v82_sparse_glass_tissue_fraction": 0.05,
        },
    )

    with pytest.raises(ValueError, match="validated v82 refilter gate"):
        validate_cell_promotion_candidate(run)


def test_cell_promotion_gate_accepts_validated_v83_focus_refilter(
    tmp_path: Path,
) -> None:
    run = _promotion_run(tmp_path, mutation={"algorithm_version": 83})

    result = validate_cell_promotion_candidate(run)

    assert result["algorithm_version"] == 83
    assert result["request"]["method_reference"]["profile"].endswith("v79")
    assert result["filter_upgrade"]["source_algorithm_version"] == 82


def test_cell_promotion_gate_rejects_unsealed_v83_focus_policy(
    tmp_path: Path,
) -> None:
    run = _promotion_run(
        tmp_path,
        mutation={
            "algorithm_version": 83,
            "v83_focus_median_laplacian": 701.0,
        },
    )

    with pytest.raises(ValueError, match="validated v83 focus gate"):
        validate_cell_promotion_candidate(run)


def test_cell_promotion_gate_accepts_v83_from_validated_v75_source(
    tmp_path: Path,
) -> None:
    run = _promotion_run(
        tmp_path,
        mutation={
            "algorithm_version": 83,
            "v83_source_algorithm_version": 75,
        },
    )

    result = validate_cell_promotion_candidate(run)

    assert result["filter_upgrade"]["source_algorithm_version"] == 75


def test_cell_promotion_gate_rejects_v83_from_unvalidated_source(
    tmp_path: Path,
) -> None:
    run = _promotion_run(
        tmp_path,
        mutation={
            "algorithm_version": 83,
            "v83_source_algorithm_version": 74,
        },
    )

    with pytest.raises(ValueError, match="validated v83 source seal"):
        validate_cell_promotion_candidate(run)


def test_cell_promotion_gate_rejects_stale_v83_section_evidence(
    tmp_path: Path,
) -> None:
    run = _promotion_run(
        tmp_path,
        mutation={
            "algorithm_version": 83,
            "v83_qc_component_count": 0,
        },
    )

    with pytest.raises(ValueError, match="v83 focus provenance is stale"):
        validate_cell_promotion_candidate(run)


def test_cell_promotion_gate_accepts_validated_v84_clustered_brown_refilter(
    tmp_path: Path,
) -> None:
    run = _promotion_run(tmp_path, mutation={"algorithm_version": 84})

    result = validate_cell_promotion_candidate(run)

    assert result["algorithm_version"] == 84
    assert result["request"]["method_reference"]["profile"].endswith("v80")


def test_cell_promotion_gate_accepts_validated_v86_chromatic_refilter(
    tmp_path: Path,
) -> None:
    run = _promotion_run(tmp_path, mutation={"algorithm_version": 86})

    result = validate_cell_promotion_candidate(run)

    assert result["algorithm_version"] == 86
    assert result["request"]["method_reference"]["profile"].endswith("v81")


def test_cell_promotion_gate_rejects_stale_v86_chromatic_policy(
    tmp_path: Path,
) -> None:
    run = _promotion_run(
        tmp_path,
        mutation={
            "algorithm_version": 86,
            "v86_minimum_component_instances": 11,
        },
    )

    with pytest.raises(ValueError, match="validated v86 chromatic gate"):
        validate_cell_promotion_candidate(run)


def test_cell_promotion_gate_accepts_validated_v87_organized_escape_refilter(
    tmp_path: Path,
) -> None:
    run = _promotion_run(tmp_path, mutation={"algorithm_version": 87})

    result = validate_cell_promotion_candidate(run)

    assert result["algorithm_version"] == 87
    assert result["request"]["method_reference"]["profile"].endswith("v82")


def test_cell_promotion_gate_accepts_validated_v88_pas_refilter(
    tmp_path: Path,
) -> None:
    run = _promotion_run(tmp_path, mutation={"algorithm_version": 88})

    result = validate_cell_promotion_candidate(run)

    assert result["algorithm_version"] == 88
    assert result["request"]["method_reference"]["profile"].endswith("v83")
    assert result["filter_upgrade"]["source_algorithm_version"] == 87


def test_cell_promotion_gate_accepts_bounded_v89_satellite_interior_protection(
    tmp_path: Path,
) -> None:
    run = _promotion_run(tmp_path, mutation={"algorithm_version": 89})

    result = validate_cell_promotion_candidate(run)

    assert result["algorithm_version"] == 89
    assert result["request"]["method_reference"]["profile"].endswith("v84")
    assert result["filter_upgrade"]["source_algorithm_version"] == 75


def test_cell_promotion_gate_accepts_validated_v90_foreign_material_refilter(
    tmp_path: Path,
) -> None:
    run = _promotion_run(tmp_path, mutation={"algorithm_version": 90})

    result = validate_cell_promotion_candidate(run)

    assert result["algorithm_version"] == 90
    assert result["request"]["method_reference"]["profile"].endswith("v85")
    assert result["filter_upgrade"]["source_algorithm_version"] == 75


def test_cell_promotion_gate_rejects_stale_v90_foreign_material_policy(
    tmp_path: Path,
) -> None:
    run = _promotion_run(
        tmp_path,
        mutation={
            "algorithm_version": 90,
            "v90_minimum_instance_fraction": 0.31,
        },
    )

    with pytest.raises(ValueError, match="validated v90 foreign-material gate"):
        validate_cell_promotion_candidate(run)


def test_cell_promotion_gate_accepts_validated_v91_reviewed_exclusion(
    tmp_path: Path,
) -> None:
    run = _promotion_run(tmp_path, mutation={"algorithm_version": 91})

    result = validate_cell_promotion_candidate(run)

    assert result["algorithm_version"] == 91
    assert result["request"]["method_reference"]["profile"].endswith("v86")
    assert result["filter_upgrade"]["source_algorithm_version"] == 75


def test_cell_promotion_gate_accepts_bounded_v93_neutral_precipitate_refilter(
    tmp_path: Path,
) -> None:
    run = _promotion_run(tmp_path, mutation={"algorithm_version": 93})

    result = validate_cell_promotion_candidate(run)

    assert result["algorithm_version"] == 93
    assert result["request"]["method_reference"]["profile"].endswith("v88")
    assert result["request"]["isolated_debris_gate"]["neutral_precipitate_gate"]
    assert result["subset_source"]["scope"] == "cache-only-refilter-v1"


def test_cell_promotion_gate_rejects_stale_v93_neutral_policy(
    tmp_path: Path,
) -> None:
    run = _promotion_run(
        tmp_path,
        mutation={
            "algorithm_version": 93,
            "v93_maximum_mean_intensity": 160.0,
        },
    )

    with pytest.raises(ValueError, match="neutral-precipitate QC provenance"):
        validate_cell_promotion_candidate(run)


@pytest.mark.parametrize(
    ("algorithm_version", "expected_method"),
    (
        (94, DENSE_SMALL_CELL_RECOVERY_METHOD),
        (95, DENSE_SMALL_CELL_ADJACENCY_RECOVERY_METHOD),
    ),
)
def test_cell_promotion_gate_accepts_dense_small_cell_recovery(
    tmp_path: Path,
    algorithm_version: int,
    expected_method: str,
) -> None:
    run = _promotion_run(
        tmp_path,
        mutation={"algorithm_version": algorithm_version},
    )

    result = validate_cell_promotion_candidate(run)

    recovery = result["request"]["dense_small_cell_recovery"]
    assert result["algorithm_version"] == algorithm_version
    assert recovery["method"] == expected_method
    assert recovery["manifest_fingerprint"] == "f" * 64
    assert "/" not in json.dumps(recovery, sort_keys=True)


def test_cell_promotion_gate_accepts_bounded_additive_recovery(
    tmp_path: Path,
) -> None:
    run = _promotion_run(tmp_path, mutation={"algorithm_version": 96})

    result = validate_cell_promotion_candidate(run)

    recovery = result["request"]["bounded_additive_recovery"]
    assert result["algorithm_version"] == 96
    assert recovery["selection"]["selected_instance_count"] == 2
    assert recovery["selection"]["removed_pixels"] == 0
    assert "/" not in json.dumps(recovery, sort_keys=True)


def test_cell_promotion_gate_rejects_bounded_additive_removal(
    tmp_path: Path,
) -> None:
    run = _promotion_run(
        tmp_path,
        mutation={"algorithm_version": 96, "v96_removed_pixels": 1},
    )

    with pytest.raises(ValueError, match="bounded-additive QC provenance"):
        validate_cell_promotion_candidate(run)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            {"v94_manifest_fingerprint": "e" * 64},
            "dense-recovery QC provenance",
        ),
        ({"v94_qc_recovery_tiles": 2}, "dense-recovery QC provenance"),
        (
            {"v94_source_labels_sha256": "e" * 64},
            "dense-recovery source binding",
        ),
        ({"v94_tile_key": "../unsafe.npz"}, "dense-recovery tile seal"),
    ),
)
def test_cell_promotion_gate_rejects_tampered_dense_small_cell_recovery(
    tmp_path: Path,
    mutation: dict[str, object],
    message: str,
) -> None:
    run = _promotion_run(
        tmp_path,
        mutation={"algorithm_version": 94, **mutation},
    )

    with pytest.raises(ValueError, match=message):
        validate_cell_promotion_candidate(run)


def test_cell_promotion_gate_rejects_stale_v91_label_digest(
    tmp_path: Path,
) -> None:
    run = _promotion_run(
        tmp_path,
        mutation={"algorithm_version": 91, "v91_label_ids_sha256": "f" * 64},
    )

    with pytest.raises(ValueError, match="reviewed label IDs"):
        validate_cell_promotion_candidate(run)


@pytest.mark.parametrize(
    "mutation,message",
    (
        ({"v89_scope": "stale"}, "bounded v89"),
        ({"v89_source_algorithm": 74}, "bounded v89"),
        ({"satellite_debris_instances_removed": 0}, "v89 satellite-gate"),
        ({"satellite_interior_instances_protected": 0}, "v89 satellite-gate"),
        ({"satellite_mask_distance_um": 199.0}, "bounded v89"),
    ),
)
def test_cell_promotion_gate_rejects_stale_v89_satellite_provenance(
    tmp_path: Path,
    mutation: dict[str, object],
    message: str,
) -> None:
    run = _promotion_run(
        tmp_path,
        mutation={"algorithm_version": 89, **mutation},
    )

    with pytest.raises(ValueError, match=message):
        validate_cell_promotion_candidate(run)


def test_compose_cell_result_snapshots_a_validated_section_replacement(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base"
    replacement = tmp_path / "replacement"
    base.mkdir()
    replacement.mkdir()
    _composition_ready(
        _promotion_run(base),
        label_bytes=b"base labels",
        section_fingerprint="1" * 64,
    )
    _composition_ready(
        _promotion_run(replacement, mutation={"algorithm_version": 84}),
        label_bytes=b"replacement labels",
        section_fingerprint="2" * 64,
    )
    replacement_result = validate_cell_promotion_candidate(replacement)

    result_path = compose_cell_result(
        base,
        tmp_path / "composite",
        {"001": replacement},
    )
    result = validate_cell_promotion_candidate(result_path.parent)

    assert result["algorithm_version"] == 85
    assert result["slides"][0]["section"] == "001"
    assert (
        result["composition"]["base_result_fingerprint"]
        == (validate_cell_promotion_candidate(base)["fingerprint"])
    )
    source = result["composition"]["section_sources"][0]
    assert source["replacement"] is True
    assert source["source_result_fingerprint"] == replacement_result["fingerprint"]
    assert (result_path.parent / "labels/001.cells.tiff").read_bytes() == (
        b"replacement labels"
    )
    assert (result_path.parent / "labels/001.cells.tiff").stat().st_ino == (
        replacement / "labels/001.cells.tiff"
    ).stat().st_ino


def test_compose_cell_result_preserves_v89_satellite_source_seal(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base"
    replacement = tmp_path / "replacement"
    base.mkdir()
    replacement.mkdir()
    _composition_ready(
        _promotion_run(base),
        label_bytes=b"base labels",
        section_fingerprint="1" * 64,
    )
    _composition_ready(
        _promotion_run(replacement, mutation={"algorithm_version": 89}),
        label_bytes=b"replacement labels",
        section_fingerprint="2" * 64,
    )

    result_path = compose_cell_result(
        base,
        tmp_path / "composite",
        {"001": replacement},
    )
    result = validate_cell_promotion_candidate(result_path.parent)
    source = result["composition"]["section_sources"][0]

    assert source["source_algorithm_version"] == 89
    assert source["satellite_gate"]["enabled"] is True
    assert source["satellite_gate"]["interior_protection"] == {
        "minimum_mask_distance_um": 200.0,
        "minimum_sample_fraction": 0.50,
    }
    assert source["filter_upgrade"]["source_algorithm_version"] == 75


def test_compose_cell_result_preserves_v91_reviewed_exclusion_seal(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base"
    replacement = tmp_path / "replacement"
    base.mkdir()
    replacement.mkdir()
    _composition_ready(
        _promotion_run(base),
        label_bytes=b"base labels",
        section_fingerprint="1" * 64,
    )
    _composition_ready(
        _promotion_run(replacement, mutation={"algorithm_version": 91}),
        label_bytes=b"replacement labels",
        section_fingerprint="2" * 64,
    )

    result_path = compose_cell_result(
        base,
        tmp_path / "composite",
        {"001": replacement},
    )
    result = validate_cell_promotion_candidate(result_path.parent)
    source = result["composition"]["section_sources"][0]

    assert source["source_algorithm_version"] == 91
    assert source["reviewed_artifact_exclusion"]["sections"][0]["label_ids"] == [
        3,
        7,
    ]
    assert source["filter_upgrade"]["source_result_fingerprint"] == "b" * 64


def test_compose_cell_result_preserves_dense_small_cell_recovery_seal(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base"
    replacement = tmp_path / "replacement"
    base.mkdir()
    replacement.mkdir()
    base_labels = b"base labels"
    base_labels_sha256 = hashlib.sha256(base_labels).hexdigest()
    _composition_ready(
        _promotion_run(base),
        label_bytes=base_labels,
        section_fingerprint="1" * 64,
    )
    _composition_ready(
        _promotion_run(replacement, mutation={"algorithm_version": 94}),
        label_bytes=b"dense-recovered labels",
        section_fingerprint="2" * 64,
        dense_source_labels_sha256=base_labels_sha256,
        dense_source_section_fingerprint="1" * 64,
    )

    result_path = compose_cell_result(
        base,
        tmp_path / "composite",
        {"001": replacement},
    )
    result = validate_cell_promotion_candidate(result_path.parent)
    source = result["composition"]["section_sources"][0]

    assert source["source_algorithm_version"] == 94
    assert source["dense_small_cell_recovery"]["manifest_fingerprint"] == "f" * 64
    assert source["subset_source"]["source_labels_sha256"] == base_labels_sha256


def test_compose_cell_result_rejects_dense_recovery_from_stale_base_labels(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base"
    replacement = tmp_path / "replacement"
    base.mkdir()
    replacement.mkdir()
    _composition_ready(
        _promotion_run(base),
        label_bytes=b"current base labels",
        section_fingerprint="1" * 64,
    )
    _composition_ready(
        _promotion_run(replacement, mutation={"algorithm_version": 94}),
        label_bytes=b"dense-recovered labels",
        section_fingerprint="2" * 64,
        dense_source_labels_sha256="e" * 64,
        dense_source_section_fingerprint="1" * 64,
    )

    with pytest.raises(ValueError, match="not derived from the selected base labels"):
        compose_cell_result(
            base,
            tmp_path / "composite",
            {"001": replacement},
        )


def test_compose_cell_result_rejects_a_different_source_identity(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base"
    replacement = tmp_path / "replacement"
    base.mkdir()
    replacement.mkdir()
    _composition_ready(
        _promotion_run(base),
        label_bytes=b"base labels",
        section_fingerprint="1" * 64,
    )
    _composition_ready(
        _promotion_run(replacement, mutation={"algorithm_version": 84}),
        label_bytes=b"replacement labels",
        section_fingerprint="2" * 64,
        source_identity="different-source",
    )

    with pytest.raises(ValueError, match="geometry or source identity differs"):
        compose_cell_result(
            base,
            tmp_path / "composite",
            {"001": replacement},
        )


def test_cell_promotion_gate_rejects_unsealed_v84_clustered_brown_policy(
    tmp_path: Path,
) -> None:
    run = _promotion_run(
        tmp_path,
        mutation={"algorithm_version": 84, "v84_clustered_brown_gate": False},
    )

    with pytest.raises(ValueError, match="validated v84 clustered-brown gate"):
        validate_cell_promotion_candidate(run)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ({"method": "combined"}, "method profile"),
        ({"method_reference_sha": "0" * 64}, "method reference is stale"),
        ({"method_reference_profile": "legacy"}, "method reference is stale"),
        ({"global_minimum_pixels": 2}, "global_nuclear_support threshold"),
        ({"qc_algorithm_version": 56}, "QC method provenance"),
        ({"qc_method_profile": "legacy"}, "QC method provenance"),
        ({"qc_profile_fingerprint": "z" * 64}, "QC method provenance"),
        ({"source_context": False}, "source tissue-context evidence"),
        ({"adaptive_nuclear_core": False}, "adaptive nuclear-core gate"),
        ({"adaptive_minimum_area_um2": 500.0}, "adaptive nuclear-core gate"),
        ({"adaptive_minimum_fraction": 0.02}, "adaptive nuclear-core gate"),
        ({"adaptive_minimum_blue_ratio": 1.0}, "adaptive nuclear-core gate"),
        ({"isolated_debris_gate": False}, "isolated-debris gate"),
        ({"allow_nuclear_escape": False}, "isolated-debris gate"),
        ({"organized_component_pixels": 1_000}, "isolated-debris gate"),
        ({"organized_mean_pixels": 100.0}, "isolated-debris gate"),
        ({"organized_window_size": 2048}, "isolated-debris gate"),
        ({"organized_strong_fraction": 0.5}, "isolated-debris gate"),
        ({"compact_unsupported_area_um2": 20_000.0}, "isolated-debris gate"),
        ({"compact_unsupported_mean_pixels": 400.0}, "isolated-debris gate"),
        ({"compact_unsupported_red_blue": 16.0}, "isolated-debris gate"),
        ({"compact_unsupported_strong_red_blue": 50.0}, "isolated-debris gate"),
        ({"compact_unsupported_strong_fraction": 0.2}, "isolated-debris gate"),
        ({"compact_unsupported_elongation_ratio": 3.0}, "isolated-debris gate"),
        ({"compact_unsupported_elongated_fraction": 0.7}, "isolated-debris gate"),
        ({"compact_unsupported_context_nuclear": 0.5}, "isolated-debris gate"),
        ({"compact_unsupported_fill": 0.1}, "isolated-debris gate"),
        ({"compact_unsupported_instances_removed": -1}, "required QC counters"),
        ({"foam_minimum_intensity": 140.0}, "isolated-debris gate"),
        ({"foam_mean_pixels": 300.0}, "isolated-debris gate"),
        ({"foam_strong_fraction": 0.2}, "isolated-debris gate"),
        ({"foam_context_nuclear": 0.7}, "isolated-debris gate"),
        ({"foam_elongated_fraction": 0.7}, "isolated-debris gate"),
        ({"foam_core_pixels": 12}, "isolated-debris gate"),
        ({"foam_core_bright_fraction": 0.7}, "isolated-debris gate"),
        ({"foam_core_intensity": 170.0}, "isolated-debris gate"),
        ({"foam_core_instance_area_um2": 150.0}, "isolated-debris gate"),
        ({"foam_core_component_area_um2": 5000.0}, "isolated-debris gate"),
        ({"foam_core_aspect": 3.0}, "isolated-debris gate"),
        ({"foam_instances_removed": -1}, "required QC counters"),
        ({"oversized_area_um2": 2_000.0}, "isolated-debris gate"),
        ({"oversized_fraction": 0.75}, "isolated-debris gate"),
        ({"fold_instance_area_um2": 600.0}, "isolated-debris gate"),
        ({"fold_component_area_um2": 1500.0}, "isolated-debris gate"),
        ({"fold_red_blue": 20.0}, "isolated-debris gate"),
        ({"fold_intensity": 220.0}, "isolated-debris gate"),
        ({"fold_dense_instances": 4}, "isolated-debris gate"),
        ({"fold_dense_aspect": 4.0}, "isolated-debris gate"),
        ({"fold_dense_connectivity": 9}, "isolated-debris gate"),
        ({"fold_dense_red_blue": 50.0}, "isolated-debris gate"),
        ({"fold_dense_intensity": 125.0}, "isolated-debris gate"),
        ({"detached_compact_instance_area_um2": 600.0}, "isolated-debris gate"),
        ({"detached_compact_component_area_um2": 1500.0}, "isolated-debris gate"),
        ({"detached_compact_instances": 3}, "isolated-debris gate"),
        ({"detached_compact_red_blue": 10.0}, "isolated-debris gate"),
        ({"detached_compact_intensity": 200.0}, "isolated-debris gate"),
        ({"glass_prediction_fraction": 0.2}, "isolated-debris gate"),
        ({"glass_context_stain_fraction": 0.4}, "isolated-debris gate"),
        ({"glass_low_stain_red_blue": 50.0}, "isolated-debris gate"),
        ({"glass_low_stain_intensity": 180.0}, "isolated-debris gate"),
        ({"necrotic_component_area_um2": 10_000.0}, "isolated-debris gate"),
        ({"necrotic_fill": 0.60}, "isolated-debris gate"),
        ({"necrotic_instances_removed": -1}, "required QC counters"),
        ({"brown_component_area_um2": 10_000.0}, "isolated-debris gate"),
        ({"brown_red_blue": 40.0}, "isolated-debris gate"),
        ({"brown_source_area_um2": 100_000.0}, "isolated-debris gate"),
        ({"brown_source_intensity": 200.0}, "isolated-debris gate"),
        ({"brown_source_dilation": 1}, "isolated-debris gate"),
        ({"brown_instances_removed": -1}, "required QC counters"),
        ({"isolated_context_fraction": 0.03}, "isolated-debris gate"),
        ({"source_context_pixels_removed": -1}, "required QC counters"),
        ({"adaptive_core_pixels_removed": -1}, "required QC counters"),
        ({"isolated_debris_pixels_removed": -1}, "required QC counters"),
        ({"oversized_chromatic_instances_removed": -1}, "required QC counters"),
        ({"oversized_chromatic_pixels_removed": -1}, "required QC counters"),
        ({"fold_artifact_instances_removed": -1}, "required QC counters"),
        ({"fold_artifact_pixels_removed": -1}, "required QC counters"),
        ({"detached_fragment_instances_removed": -1}, "required QC counters"),
        ({"detached_fragment_pixels_removed": -1}, "required QC counters"),
        ({"satellite_debris_instances_removed": -1}, "required QC counters"),
        ({"satellite_debris_pixels_removed": -1}, "required QC counters"),
        ({"satellite_maximum_instance_area_um2": 301.0}, "satellite-debris safeguards"),
        (
            {"satellite_minimum_component_area_um2": 501.0},
            "satellite-debris safeguards",
        ),
        ({"satellite_minimum_instances": 4}, "satellite-debris safeguards"),
        ({"satellite_connectivity_dilation_bins": 9}, "satellite-debris safeguards"),
        ({"satellite_maximum_nuclear_fraction": 0.06}, "satellite-debris safeguards"),
        ({"satellite_minimum_red_blue": 6.0}, "satellite-debris safeguards"),
        ({"satellite_maximum_red_blue": 71.0}, "satellite-debris safeguards"),
        ({"satellite_minimum_intensity": 91.0}, "satellite-debris safeguards"),
        ({"satellite_maximum_intensity": 221.0}, "satellite-debris safeguards"),
        ({"satellite_context_window_size_px": 257}, "satellite-debris safeguards"),
        (
            {"satellite_maximum_prediction_fraction": 0.36},
            "satellite-debris safeguards",
        ),
        ({"glass_artifact_instances_removed": -1}, "required QC counters"),
        ({"glass_artifact_pixels_removed": -1}, "required QC counters"),
        ({"outside_tissue_pixels_final": 1}, "leave the tissue mask"),
        ({"tiles_total": 9}, "tile accounting"),
        (
            {"post_constraint_cells_below_min_size": 1},
            "sub-minimum fragments",
        ),
        ({"preflight_slide_count": 2}, "every selected section"),
    ),
)
def test_cell_promotion_gate_rejects_unsafe_candidates(
    tmp_path: Path,
    mutation: dict[str, object],
    message: str,
) -> None:
    run = _promotion_run(tmp_path, mutation=mutation)

    with pytest.raises(ValueError, match=message):
        validate_cell_promotion_candidate(run)


def _promotion_run(
    root: Path,
    *,
    mutation: dict[str, object] | None = None,
) -> Path:
    mutation = mutation or {}
    algorithm_version = int(mutation.get("algorithm_version", 75))
    requires_v82 = algorithm_version in {82, 84} or (
        algorithm_version == 83
        and mutation.get("v83_source_algorithm_version", 82) == 82
    )
    method_profile = {
        82: "combined-containment-cpsam-wsi-v78",
        83: "combined-containment-cpsam-wsi-v79",
        84: "combined-containment-cpsam-wsi-v80",
        86: "combined-containment-cpsam-wsi-v81",
        87: "combined-containment-cpsam-wsi-v82",
        88: "combined-containment-cpsam-wsi-v83",
        89: "combined-containment-cpsam-wsi-v84",
        90: "combined-containment-cpsam-wsi-v85",
        91: "combined-containment-cpsam-wsi-v86",
        93: "combined-containment-cpsam-wsi-v88",
        94: "combined-containment-cpsam-stardist-dense-recovery-v1",
        95: "combined-containment-cpsam-stardist-dense-recovery-v2",
        96: BOUNDED_ADDITIVE_RECOVERY_METHOD_PROFILE,
    }.get(algorithm_version, "combined-containment-cpsam-wsi-v71")
    bounded_tiles = [
        {
            "tile": "r0000_c0000_y0_x0.npz",
            "x": 0,
            "y": 0,
            "width": 1024,
            "height": 1024,
            "recovery_fingerprint": "6" * 64,
            "file_sha256": "7" * 64,
        }
    ]
    bounded_support_core = {
        "kind": "exact-recovery-tile-union-v1",
        "tiles": bounded_tiles,
    }
    bounded_bbox, bounded_area = bounded_support_geometry(bounded_tiles)
    bounded_source = {
        "result_fingerprint": "b" * 64,
        "algorithm_version": 75,
        "profile_fingerprint": "9" * 64,
        "preflight_fingerprint": "5" * 64,
        "section_fingerprint": "6" * 64,
        "labels_sha256": "a" * 64,
        "qc_sha256": "d" * 64,
    }
    bounded_recovery = {
        "schema_version": 1,
        "scope": BOUNDED_ADDITIVE_RECOVERY_SCOPE,
        "parameters": bounded_additive_recovery_parameters(minimum_area_px=15),
        "source": bounded_source,
        "candidate": {
            "result_fingerprint": "e" * 64,
            "algorithm_version": 95,
            "profile_fingerprint": "1" * 64,
            "preflight_fingerprint": "2" * 64,
            "section_fingerprint": "3" * 64,
            "labels_sha256": "4" * 64,
            "qc_sha256": "5" * 64,
            "recovery_manifest_fingerprint": "f" * 64,
        },
        "section": "001",
        "source_identity": "c" * 64,
        "support": {
            **bounded_support_core,
            "fingerprint": hashlib.sha256(
                json.dumps(
                    bounded_support_core,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
            "union_bbox_xywh": bounded_bbox,
            "union_pixels": bounded_area,
        },
        "selection": {
            "candidate_instance_count": 5,
            "selected_instance_count": 2,
            "rejected_below_minimum_count": 1,
            "rejected_outside_support_count": 1,
            "rejected_source_overlap_count": 1,
            "selected_ids_sha256": "8" * 64,
            "added_pixels": 30,
            "removed_pixels": 0,
            "changed_outside_support_pixels": 0,
            "first_output_id": 11,
            "last_output_id": 12,
        },
    }
    chromatic_profile = (
        "pas-anucleate-microobjects-v2"
        if algorithm_version == 88
        else "residual-foam-microobjects-v3"
    )
    chromatic_gate = chromatic_microcluster_gate(chromatic_profile)
    chromatic_gate["minimum_component_instances"] = mutation.get(
        "v86_minimum_component_instances",
        chromatic_gate["minimum_component_instances"],
    )
    foreign_material_gate = saturated_cyan_foreign_material_gate()
    foreign_material_gate["minimum_instance_fraction"] = mutation.get(
        "v90_minimum_instance_fraction",
        foreign_material_gate["minimum_instance_fraction"],
    )
    preflight_fingerprint = "1" * 64 if algorithm_version == 91 else "p" * 64
    reviewed_label_ids = [3, 7]
    reviewed_manifest = {
        "schema_version": 1,
        "scope": REVIEWED_ARTIFACT_EXCLUSION_SCOPE,
        "source_algorithm_version": 75,
        "source_result_fingerprint": "b" * 64,
        "source_preflight_fingerprint": preflight_fingerprint,
        "sections": [
            {
                "section": "001",
                "source_identity": "c" * 64,
                "source_labels_sha256": "a" * 64,
                "label_ids": reviewed_label_ids,
                "label_ids_sha256": mutation.get(
                    "v91_label_ids_sha256",
                    hashlib.sha256(
                        json.dumps(
                            reviewed_label_ids,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode()
                    ).hexdigest(),
                ),
                "selection": {
                    "instance_action": REVIEWED_ARTIFACT_EXCLUSION_INSTANCE_ACTION,
                    "regions": [
                        {
                            "region_id": "reviewed-debris-1",
                            "kind": "detached-slide-processing-debris",
                            "native_coordinate_space": "source_wsi_pixels",
                            "geometry": {"polygon_xy": [[0, 0], [8, 0], [4, 8]]},
                            "selection_rule": "whole labels with centroids in polygon",
                            "parameters": {"minimum_region_fraction": 0.2},
                        }
                    ],
                },
                "review": {
                    "decision": "approved",
                    "reviewer": "visual-audit",
                    "reviewed_at": "2026-08-31T20:00:00+00:00",
                    "evidence_sha256s": ["e" * 64],
                    "notes": "Synthetic reviewed exclusion fixture.",
                },
            }
        ],
    }
    focus_gate = {
        "enabled": True,
        "analysis_downsample": 4.0,
        "native_block_size": 128,
        "maximum_mean_intensity": 120.0,
        "minimum_dark_fraction": 0.30,
        "dark_intensity_threshold": 100.0,
        "maximum_block_laplacian_variance": 1500.0,
        "minimum_tissue_fraction": 0.25,
        "minimum_component_blocks": 12,
        "minimum_component_fill_fraction": 0.70,
        "maximum_component_aspect_ratio": 2.0,
        "maximum_median_laplacian_variance": mutation.get(
            "v83_focus_median_laplacian", 700.0
        ),
        "growth_maximum_smoothed_intensity": 145.0,
        "growth_gaussian_sigma_px": 8.0,
        "growth_minimum_seed_pixels": 256,
        "instance_action": "remove_whole_labels_touching_native_growth_region",
    }
    focus_component = {
        "blocks": 12,
        "native_content_bbox_xyxy": [0, 0, 512, 384],
        "block_indices_rc": [[row, column] for row in range(3) for column in range(4)],
        "fill_fraction": 1.0,
        "aspect_ratio": 4 / 3,
        "median_laplacian_variance": 500.0,
        "native_growth_bbox_xyxy": [0, 0, 1024, 896],
        "seed_pixels": 196_608,
        "grown_pixels": 210_000,
        "touching_labels": 3,
    }
    preflight = {
        "schema_version": 2,
        "fingerprint": preflight_fingerprint,
        "slide_count": mutation.get("preflight_slide_count", 1),
        "slides": [
            {
                "section": "001",
                "slide_name": "slide.ndpi",
                **(
                    {"source_identity": "c" * 64}
                    if algorithm_version in {91, 94, 95, 96}
                    else {}
                ),
            }
        ],
    }
    (root / "preflight.json").write_text(json.dumps(preflight))
    (root / "labels").mkdir()
    (root / "qc").mkdir()
    (root / "labels/001.cells.tiff").write_bytes(b"labels")
    qc = {
        "algorithm_version": mutation.get("qc_algorithm_version", algorithm_version),
        "method_profile": mutation.get("qc_method_profile", method_profile),
        "profile_fingerprint": mutation.get("qc_profile_fingerprint", "q" * 64),
        "cell_count": 12,
        "outside_tissue_pixels_final": mutation.get("outside_tissue_pixels_final", 0),
        "tiles_total": mutation.get("tiles_total", 10),
        "tiles_inferred": (
            0
            if algorithm_version in {83, 86, 87, 88, 89, 90, 91, 93, 94, 95, 96}
            else 4
        ),
        "tiles_reused": (
            7
            if algorithm_version in {83, 86, 87, 88, 89, 90, 91, 93, 94, 95, 96}
            else 3
        ),
        "tiles_skipped_outside_tissue": 3,
        "post_constraint_small_instances_removed": 4,
        "post_constraint_cells_below_min_size": mutation.get(
            "post_constraint_cells_below_min_size", 0
        ),
        "flat_background_instances_removed": 2,
        "nuclear_unsupported_instances_removed": 5,
        "source_context_unsupported_instances_removed": 3,
        "source_context_unsupported_pixels_removed": mutation.get(
            "source_context_pixels_removed", 240
        ),
        "adaptive_nuclear_core_unsupported_instances_removed": 2,
        "adaptive_nuclear_core_unsupported_pixels_removed": mutation.get(
            "adaptive_core_pixels_removed", 1200
        ),
        "isolated_debris_unsupported_instances_removed": 7,
        "isolated_debris_unsupported_pixels_removed": mutation.get(
            "isolated_debris_pixels_removed", 640
        ),
        "micro_island_debris_instances_removed": 3,
        "micro_island_debris_pixels_removed": 320,
        "organized_tissue_escape_instances": (
            0 if algorithm_version in {87, 88} else 4
        ),
        "organized_tissue_escape_pixels": (
            0 if algorithm_version in {87, 88} else 2_400
        ),
        "compact_unsupported_mosaic_instances_removed": mutation.get(
            "compact_unsupported_instances_removed", 12
        ),
        "compact_unsupported_mosaic_pixels_removed": 8_000,
        "foam_mosaic_instances_removed": mutation.get("foam_instances_removed", 20),
        "foam_mosaic_pixels_removed": 12_000,
        "oversized_chromatic_artifact_instances_removed": mutation.get(
            "oversized_chromatic_instances_removed", 1
        ),
        "oversized_chromatic_artifact_pixels_removed": mutation.get(
            "oversized_chromatic_pixels_removed", 4_000
        ),
        "fold_artifact_instances_removed": mutation.get(
            "fold_artifact_instances_removed", 4
        ),
        "fold_artifact_pixels_removed": mutation.get(
            "fold_artifact_pixels_removed", 12_000
        ),
        "detached_fragment_instances_removed": mutation.get(
            "detached_fragment_instances_removed", 6
        ),
        "detached_fragment_pixels_removed": mutation.get(
            "detached_fragment_pixels_removed", 18_000
        ),
        **(
            {
                "clustered_brown_fragment_instances_removed": 4,
                "clustered_brown_fragment_pixels_removed": 12_000,
            }
            if algorithm_version == 84
            else {}
        ),
        "satellite_debris_instances_removed": mutation.get(
            "satellite_debris_instances_removed",
            31 if algorithm_version == 89 else 83,
        ),
        "satellite_debris_pixels_removed": mutation.get(
            "satellite_debris_pixels_removed",
            14_520 if algorithm_version == 89 else 40_311,
        ),
        **(
            {
                "satellite_interior_instances_protected": mutation.get(
                    "satellite_interior_instances_protected", 52
                ),
                "satellite_interior_pixels_protected": mutation.get(
                    "satellite_interior_pixels_protected", 25_791
                ),
            }
            if algorithm_version == 89
            else {}
        ),
        "glass_artifact_instances_removed": mutation.get(
            "glass_artifact_instances_removed", 2
        ),
        "glass_artifact_pixels_removed": mutation.get(
            "glass_artifact_pixels_removed", 800
        ),
        "necrotic_artifact_instances_removed": mutation.get(
            "necrotic_instances_removed", 900
        ),
        "necrotic_artifact_pixels_removed": 300_000,
        "brown_debris_artifact_instances_removed": mutation.get(
            "brown_instances_removed", 400
        ),
        "brown_debris_artifact_pixels_removed": 180_000,
        "instance_evidence": {
            "isolated_debris_satellite_maximum_instance_area_um2": mutation.get(
                "satellite_maximum_instance_area_um2", 300.0
            ),
            "isolated_debris_satellite_minimum_component_area_um2": mutation.get(
                "satellite_minimum_component_area_um2", 500.0
            ),
            "isolated_debris_satellite_minimum_instances": mutation.get(
                "satellite_minimum_instances", 3
            ),
            "isolated_debris_satellite_connectivity_dilation_bins": mutation.get(
                "satellite_connectivity_dilation_bins", 8
            ),
            "isolated_debris_satellite_maximum_nuclear_fraction": mutation.get(
                "satellite_maximum_nuclear_fraction", 0.05
            ),
            "isolated_debris_satellite_minimum_mean_red_blue_difference": mutation.get(
                "satellite_minimum_red_blue", 5.0
            ),
            "isolated_debris_satellite_maximum_mean_red_blue_difference": mutation.get(
                "satellite_maximum_red_blue", 70.0
            ),
            "isolated_debris_satellite_minimum_mean_intensity": mutation.get(
                "satellite_minimum_intensity", 90.0
            ),
            "isolated_debris_satellite_maximum_mean_intensity": mutation.get(
                "satellite_maximum_intensity", 220.0
            ),
            "isolated_debris_satellite_context_window_size_px": mutation.get(
                "satellite_context_window_size_px", 256
            ),
            "isolated_debris_satellite_maximum_prediction_fraction": mutation.get(
                "satellite_maximum_prediction_fraction", 0.35
            ),
            **(
                {
                    "isolated_debris_satellite_gate_enabled": True,
                    "isolated_debris_satellite_minimum_mask_distance_um": (
                        mutation.get("satellite_mask_distance_um", 200.0)
                    ),
                    "isolated_debris_satellite_minimum_sample_fraction": 0.50,
                    "isolated_debris_satellite_mask_shape": [1200, 731],
                    "isolated_debris_satellite_mpp_xy": [0.5, 0.5],
                }
                if algorithm_version == 89
                else {}
            ),
            **(
                {
                    "isolated_debris_clustered_brown_gate": True,
                    "isolated_debris_clustered_brown_minimum_instance_area_um2": 225.0,
                    (
                        "isolated_debris_clustered_brown_maximum_instance_area_um2"
                    ): 1_200.0,
                    (
                        "isolated_debris_clustered_brown_minimum_component_area_um2"
                    ): 1_500.0,
                    "isolated_debris_clustered_brown_minimum_instances": 4,
                    "isolated_debris_clustered_brown_connectivity_dilation_bins": 12,
                    "isolated_debris_clustered_brown_maximum_nuclear_fraction": 0.02,
                    "isolated_debris_clustered_brown_minimum_red_green_difference": 5.0,
                    "isolated_debris_clustered_brown_minimum_red_blue_difference": 30.0,
                    "isolated_debris_clustered_brown_maximum_mean_intensity": 175.0,
                    "isolated_debris_clustered_brown_context_window_size_px": 256,
                    "isolated_debris_clustered_brown_maximum_prediction_fraction": 0.50,
                    (
                        "isolated_debris_clustered_brown_minimum_sparse_sample_fraction"
                    ): 0.75,
                }
                if algorithm_version == 84
                else {}
            ),
            **(
                {
                    "isolated_debris_neutral_precipitate_gate": True,
                    (
                        "isolated_debris_neutral_precipitate_"
                        "minimum_instance_area_pixels"
                    ): 100,
                    (
                        "isolated_debris_neutral_precipitate_"
                        "maximum_instance_area_pixels"
                    ): 2400,
                    "isolated_debris_neutral_precipitate_minimum_core_fraction": 0.28,
                    (
                        "isolated_debris_neutral_precipitate_minimum_very_dark_fraction"
                    ): 0.20,
                    (
                        "isolated_debris_neutral_precipitate_maximum_nuclear_fraction"
                    ): 0.35,
                    "isolated_debris_neutral_precipitate_mean_red_blue_range": [
                        -10.0,
                        15.0,
                    ],
                    (
                        "isolated_debris_neutral_precipitate_maximum_mean_intensity"
                    ): mutation.get("v93_maximum_mean_intensity", 135.0),
                    "isolated_debris_neutral_precipitate_context_window_size_px": 256,
                    (
                        "isolated_debris_neutral_precipitate_"
                        "maximum_prediction_fraction"
                    ): 0.25,
                    (
                        "isolated_debris_neutral_precipitate_"
                        "minimum_sparse_sample_fraction"
                    ): 0.50,
                    "isolated_debris_glass_minimum_area_pixels": 100,
                    "isolated_debris_satellite_maximum_instance_area_pixels": 1200,
                }
                if algorithm_version == 93
                else {}
            ),
            **({"focus_artifact_gate": focus_gate} if algorithm_version == 83 else {}),
            **(
                {"chromatic_microcluster_gate": chromatic_gate}
                if algorithm_version in {86, 88}
                else {}
            ),
            **(
                {"saturated_cyan_foreign_material_gate": (foreign_material_gate)}
                if algorithm_version == 90
                else {}
            ),
            **(
                {
                    "reviewed_artifact_exclusion": (
                        reviewed_artifact_exclusion_qc_evidence(
                            reviewed_manifest,
                            "001",
                        )
                    )
                }
                if algorithm_version == 91
                and mutation.get("v91_label_ids_sha256") is None
                else {}
            ),
            **(
                {
                    "isolated_debris_allow_nuclear_escape": False,
                    "isolated_debris_allow_organized_escape": False,
                }
                if algorithm_version in {87, 88}
                else {}
            ),
        },
        **(
            {
                "tiles_reused_from_prior_run": 0,
                "filter_upgrade_from_algorithm_version": mutation.get(
                    "v83_source_algorithm_version", 82
                ),
                "filter_upgrade_source_labels_sha256": "a" * 64,
                "focus_artifact_components_excluded": mutation.get(
                    "v83_qc_component_count", 1
                ),
                "focus_artifact_instances_removed": 3,
                "focus_artifact_pixels_removed": 12_000,
                "focus_artifact_components": [focus_component],
            }
            if algorithm_version == 83
            else {}
        ),
        **(
            {
                "tiles_reused_from_prior_run": 0,
                "filter_upgrade_from_algorithm_version": (
                    87 if algorithm_version == 88 else 75
                ),
                "filter_upgrade_source_labels_sha256": "a" * 64,
                "chromatic_microcluster_candidate_instances": 14,
                "chromatic_microcluster_instances_removed": 12,
                "chromatic_microcluster_pixels_removed": 5_000,
            }
            if algorithm_version in {86, 88}
            else {}
        ),
        **(
            {
                "tiles_reused_from_prior_run": 0,
                "filter_upgrade_from_algorithm_version": 75,
                "filter_upgrade_source_labels_sha256": "a" * 64,
                "saturated_cyan_foreign_material_candidate_instances": 3,
                "saturated_cyan_foreign_material_instances_removed": 3,
                "saturated_cyan_foreign_material_pixels_removed": 900,
                "saturated_cyan_foreign_material_support_pixels": 700,
            }
            if algorithm_version == 90
            else {}
        ),
        **(
            {
                "tiles_reused_from_prior_run": 0,
                "filter_upgrade_from_algorithm_version": 75,
                "filter_upgrade_source_labels_sha256": "a" * 64,
                "reviewed_artifact_candidate_instances": 2,
                "reviewed_artifact_instances_removed": 2,
                "reviewed_artifact_pixels_removed": 900,
                "reviewed_artifact_manifest_sha256": (
                    reviewed_artifact_exclusion_manifest_sha256(reviewed_manifest)
                ),
                "reviewed_artifact_section_sha256": (
                    reviewed_artifact_exclusion_section_sha256(
                        reviewed_manifest["sections"][0]
                    )
                ),
                "reviewed_artifact_label_ids_sha256": (
                    reviewed_manifest["sections"][0]["label_ids_sha256"]
                ),
            }
            if algorithm_version == 91 and mutation.get("v91_label_ids_sha256") is None
            else {}
        ),
        **(
            {
                "tiles_reused_from_prior_run": 7,
                **(
                    {
                        "filter_upgrade_from_algorithm_version": None,
                        "filter_upgrade_source_labels_sha256": None,
                        "neutral_precipitate_instances_removed": 2,
                        "neutral_precipitate_pixels_removed": 500,
                        "labels_sha256": "e" * 64,
                    }
                    if algorithm_version == 93
                    else {}
                ),
            }
            if algorithm_version in {87, 89, 93, 94, 95}
            else {}
        ),
        **(
            {
                "dense_small_cell_recovery_tiles": mutation.get(
                    "v94_qc_recovery_tiles", 1
                ),
                "dense_small_cell_recovery_manifest_fingerprint": mutation.get(
                    "v94_qc_manifest_fingerprint", "f" * 64
                ),
            }
            if algorithm_version in {94, 95}
            else {}
        ),
        **(
            {
                "foreground_pixels": 100,
                "tiles_reused_from_prior_run": 7,
                "filter_upgrade_from_algorithm_version": 75,
                "filter_upgrade_source_labels_sha256": "a" * 64,
                "bounded_additive_recovery": bounded_recovery,
                "bounded_additive_candidate_instances": 5,
                "bounded_additive_instances_added": 2,
                "bounded_additive_pixels_added": 30,
                "bounded_additive_pixels_removed": mutation.get(
                    "v96_removed_pixels", 0
                ),
                "bounded_additive_changed_outside_support_pixels": 0,
            }
            if algorithm_version == 96
            else {}
        ),
    }
    (root / "qc/001.json").write_text(json.dumps(qc))
    request = {
        "analysis_selection": "registered_minus_manifest_exclusions",
        "model": "cpsam",
        "method": mutation.get("method", "containment"),
        "method_reference": {
            "name": "run_best_method_whole_image.py",
            "sha256": mutation.get(
                "method_reference_sha",
                "d827cea0f33d9cdc08b95a7b26fa715ea805fff1a464c105088277a01bfcd548",
            ),
            "profile": mutation.get(
                "method_reference_profile",
                method_profile,
            ),
        },
        "tissue_constraint": "accepted_mask_nearest",
        "multiscale_nuclear_support": {
            "enabled": True,
            "minimum_optical_density": 0.12,
            "minimum_pixels": 3,
            "minimum_fraction": 0.005,
        },
        "global_nuclear_support": {
            "enabled": True,
            "minimum_optical_density": 0.15,
            "minimum_pixels": mutation.get("global_minimum_pixels", 8),
            "minimum_fraction": 0.02,
        },
        "instance_evidence": {"method": "local_optical_density"},
        **({"focus_artifact_gate": focus_gate} if algorithm_version == 83 else {}),
        **(
            {
                "chromatic_microcluster_gate": {
                    "schema_version": 1,
                    "sections": [{"section": "001", "gate": chromatic_gate}],
                }
            }
            if algorithm_version in {86, 88}
            else {}
        ),
        **(
            {
                "saturated_cyan_foreign_material_gate": {
                    "schema_version": 1,
                    "sections": [{"section": "001", "gate": foreign_material_gate}],
                }
            }
            if algorithm_version == 90
            else {}
        ),
        **(
            {"reviewed_artifact_exclusion": reviewed_manifest}
            if algorithm_version == 91
            else {}
        ),
        **(
            {
                "dense_small_cell_recovery": {
                    "schema_version": 1,
                    "method": (
                        DENSE_SMALL_CELL_ADJACENCY_RECOVERY_METHOD
                        if algorithm_version == 95
                        else DENSE_SMALL_CELL_RECOVERY_METHOD
                    ),
                    "manifest_fingerprint": mutation.get(
                        "v94_manifest_fingerprint", "f" * 64
                    ),
                    "source_preflight_fingerprint": "5" * 64,
                    "source_section": "001",
                    "source_identity": "c" * 64,
                    "source_section_qc_sha256": "d" * 64,
                    "source_labels_sha256": mutation.get(
                        "v94_source_labels_sha256", "a" * 64
                    ),
                    "model": {
                        "name": "2D_versatile_he",
                        "stardist_version": "0.9.1",
                        "tensorflow_version": "2.19.0",
                        "files": {
                            "config.json": "1" * 64,
                            "thresholds.json": "2" * 64,
                            "weights_best.h5": "3" * 64,
                        },
                    },
                    "parameters": (
                        dense_small_cell_adjacency_recovery_parameters()
                        if algorithm_version == 95
                        else dense_small_cell_recovery_parameters()
                    ),
                    "tiles": [
                        {
                            "tile": mutation.get(
                                "v94_tile_key", "r0000_c0000_y0_x0.npz"
                            ),
                            "source_cache_sha256": "4" * 64,
                            "source_tile_fingerprint": "5" * 64,
                            "recovery_fingerprint": "6" * 64,
                            "file_sha256": "7" * 64,
                            "shape": [1024, 1024],
                        }
                    ],
                }
            }
            if algorithm_version in {94, 95}
            else {}
        ),
        **(
            {"bounded_additive_recovery": bounded_recovery}
            if algorithm_version == 96
            else {}
        ),
        "source_tissue_context": {
            "enabled": mutation.get("source_context", True),
            "bin_size_px": 256,
            "minimum_stain_fraction": 0.03,
            "minimum_instance_fraction": 0.01,
        },
        "adaptive_nuclear_core": {
            "enabled": mutation.get("adaptive_nuclear_core", True),
            "percentile": 70.0,
            "minimum_area_um2": mutation.get("adaptive_minimum_area_um2", 225.0),
            "minimum_pixels": 8,
            "minimum_fraction": mutation.get("adaptive_minimum_fraction", 0.10),
            "minimum_blue_ratio": mutation.get("adaptive_minimum_blue_ratio", 1.08),
        },
        "isolated_debris_gate": {
            "enabled": mutation.get("isolated_debris_gate", True),
            "context_bin_size_px": 64,
            "context_minimum_stain_fraction": mutation.get(
                "isolated_context_fraction", 0.80
            ),
            "context_minimum_instance_fraction": 0.75,
            "component_minimum_nuclear_pixels": 8,
            "allow_nuclear_escape": mutation.get(
                "allow_nuclear_escape", algorithm_version not in {87, 88}
            ),
            **(
                {"allow_organized_escape": False}
                if algorithm_version in {87, 88}
                else {}
            ),
            **(
                {
                    "satellite_gate": {
                        "enabled": True,
                        "section": "001",
                        "scope": mutation.get(
                            "v89_scope",
                            "post-inference-section-satellite-interior-protection-v1",
                        ),
                        "interior_protection": {
                            "minimum_mask_distance_um": mutation.get(
                                "satellite_mask_distance_um", 200.0
                            ),
                            "minimum_sample_fraction": 0.50,
                        },
                    }
                }
                if algorithm_version == 89
                else {}
            ),
            "nuclear_minimum_pixels": 2,
            "nuclear_minimum_fraction": 0.005,
            "nuclear_minimum_blue_ratio": 1.08,
            "neutral_dark_maximum_value": 90,
            "neutral_dark_maximum_chroma": 30,
            "neutral_dark_maximum_fraction": 0.50,
            "very_dark_maximum_value": 60,
            "very_dark_maximum_chroma": 20,
            "very_dark_maximum_fraction": 0.45,
            "micro_bin_size_px": 16,
            "micro_minimum_stain_fraction": 0.80,
            "micro_maximum_component_bins": 512,
            "micro_minimum_nuclear_fraction": 0.01,
            "micro_minimum_instance_fraction": 0.50,
            "organized_bin_size_px": 4,
            "organized_window_size_px": mutation.get("organized_window_size", 1024),
            "organized_window_overlap_px": 128,
            "organized_minimum_bin_occupancy_fraction": 0.10,
            "organized_minimum_component_pixels": mutation.get(
                "organized_component_pixels", 10_000
            ),
            "organized_compact_minimum_component_pixels": 20_000,
            "organized_minimum_aspect_ratio": 4.0,
            "organized_minimum_mean_instance_pixels": mutation.get(
                "organized_mean_pixels", 350.0
            ),
            "organized_minimum_instance_fraction": 0.50,
            "organized_strong_red_blue_difference": 40.0,
            "organized_minimum_strong_chromatic_fraction": mutation.get(
                "organized_strong_fraction", 0.25
            ),
            "compact_unsupported_minimum_area_um2": mutation.get(
                "compact_unsupported_area_um2", 10_000.0
            ),
            "compact_unsupported_maximum_aspect_ratio": 1.75,
            "compact_unsupported_maximum_mean_instance_pixels": mutation.get(
                "compact_unsupported_mean_pixels", 250.0
            ),
            "compact_unsupported_maximum_mean_red_blue_difference": mutation.get(
                "compact_unsupported_red_blue", 21.0
            ),
            "compact_unsupported_strong_red_blue_difference": mutation.get(
                "compact_unsupported_strong_red_blue", 40.0
            ),
            "compact_unsupported_maximum_strong_chromatic_fraction": mutation.get(
                "compact_unsupported_strong_fraction", 0.10
            ),
            "compact_unsupported_elongation_ratio": mutation.get(
                "compact_unsupported_elongation_ratio", 2.0
            ),
            "compact_unsupported_maximum_elongated_instance_fraction": mutation.get(
                "compact_unsupported_elongated_fraction", 0.50
            ),
            "compact_unsupported_maximum_context_nuclear_fraction": mutation.get(
                "compact_unsupported_context_nuclear", 0.20
            ),
            "compact_unsupported_minimum_component_fill_fraction": mutation.get(
                "compact_unsupported_fill", 0.20
            ),
            "compact_unsupported_minimum_instance_fraction": 0.50,
            "foam_minimum_mean_intensity": mutation.get(
                "foam_minimum_intensity", 160.0
            ),
            "foam_maximum_mean_instance_pixels": mutation.get(
                "foam_mean_pixels", 250.0
            ),
            "foam_maximum_strong_chromatic_fraction": mutation.get(
                "foam_strong_fraction", 0.10
            ),
            "foam_maximum_context_nuclear_fraction": mutation.get(
                "foam_context_nuclear", 0.50
            ),
            "foam_maximum_elongated_instance_fraction": mutation.get(
                "foam_elongated_fraction", 0.50
            ),
            "foam_core_minimum_pixels": mutation.get("foam_core_pixels", 8),
            "foam_core_minimum_bright_fraction": mutation.get(
                "foam_core_bright_fraction", 0.80
            ),
            "foam_core_minimum_intensity": mutation.get("foam_core_intensity", 180.0),
            "foam_core_maximum_instance_area_um2": mutation.get(
                "foam_core_instance_area_um2", 110.0
            ),
            "foam_core_minimum_component_area_um2": mutation.get(
                "foam_core_component_area_um2", 6500.0
            ),
            "foam_core_maximum_aspect_ratio": mutation.get("foam_core_aspect", 2.50),
            "oversized_chromatic_minimum_area_um2": mutation.get(
                "oversized_area_um2", 1_000.0
            ),
            "oversized_chromatic_minimum_fraction": mutation.get(
                "oversized_fraction", 0.50
            ),
            "fold_minimum_instance_area_um2": mutation.get(
                "fold_instance_area_um2", 300.0
            ),
            "fold_minimum_component_area_um2": mutation.get(
                "fold_component_area_um2", 750.0
            ),
            "fold_maximum_mean_red_blue_difference": mutation.get(
                "fold_red_blue", 40.0
            ),
            "fold_maximum_mean_intensity": mutation.get("fold_intensity", 180.0),
            "fold_dense_minimum_instances": mutation.get("fold_dense_instances", 3),
            "fold_dense_minimum_aspect_ratio": mutation.get("fold_dense_aspect", 3.0),
            "fold_dense_connectivity_dilation_bins": mutation.get(
                "fold_dense_connectivity", 8
            ),
            "fold_dense_maximum_mean_red_blue_difference": mutation.get(
                "fold_dense_red_blue", 60.0
            ),
            "fold_dense_maximum_mean_intensity": mutation.get(
                "fold_dense_intensity", 115.0
            ),
            "detached_minimum_instance_area_um2": 200.0,
            "detached_minimum_component_area_um2": 1000.0,
            "detached_minimum_instances": 3,
            "detached_minimum_aspect_ratio": 3.0,
            "detached_maximum_mean_red_blue_difference": 35.0,
            "detached_maximum_mean_intensity": 220.0,
            "detached_context_window_size_px": 512,
            "detached_maximum_prediction_fraction": 0.20,
            "detached_compact_minimum_instance_area_um2": mutation.get(
                "detached_compact_instance_area_um2", 300.0
            ),
            "detached_compact_minimum_component_area_um2": mutation.get(
                "detached_compact_component_area_um2", 750.0
            ),
            "detached_compact_minimum_instances": mutation.get(
                "detached_compact_instances", 2
            ),
            "detached_compact_maximum_mean_red_blue_difference": mutation.get(
                "detached_compact_red_blue", 0.0
            ),
            "detached_compact_maximum_mean_intensity": mutation.get(
                "detached_compact_intensity", 180.0
            ),
            "glass_minimum_area_um2": 25.0,
            "glass_context_window_size_px": 256,
            "glass_maximum_prediction_fraction": mutation.get(
                "glass_prediction_fraction", 0.40
            ),
            "glass_maximum_context_stain_fraction": mutation.get(
                "glass_context_stain_fraction", 0.20
            ),
            "glass_low_stain_maximum_mean_red_blue_difference": mutation.get(
                "glass_low_stain_red_blue", 35.0
            ),
            "glass_low_stain_minimum_mean_intensity": mutation.get(
                "glass_low_stain_intensity", 120.0
            ),
            "glass_minimum_context_fraction": 0.50,
            "glass_maximum_mean_red_blue_difference": 25.0,
            "glass_minimum_mean_intensity": 190.0,
            **({"neutral_precipitate_gate": True} if algorithm_version == 93 else {}),
            "necrotic_minimum_component_area_um2": mutation.get(
                "necrotic_component_area_um2", 5_000.0
            ),
            "necrotic_maximum_aspect_ratio": 2.5,
            "necrotic_minimum_fill_fraction": 0.05,
            "necrotic_maximum_fill_fraction": mutation.get("necrotic_fill", 0.45),
            "necrotic_minimum_mean_red_blue_difference": -15.0,
            "necrotic_maximum_mean_red_blue_difference": 15.0,
            "necrotic_minimum_mean_intensity": 160.0,
            "necrotic_minimum_instance_fraction": 0.50,
            "necrotic_window_size_px": 1024,
            "necrotic_window_overlap_px": 128,
            "necrotic_context_window_size_px": 256,
            "necrotic_maximum_local_prediction_fraction": 0.40,
            "brown_minimum_component_area_um2": mutation.get(
                "brown_component_area_um2", 4.0
            ),
            "brown_maximum_aspect_ratio": 3.0,
            "brown_minimum_fill_fraction": 0.12,
            "brown_maximum_fill_fraction": 0.45,
            "brown_minimum_mean_red_blue_difference": mutation.get(
                "brown_red_blue", 50.0
            ),
            "brown_maximum_mean_intensity": 120.0,
            "brown_minimum_instance_fraction": 0.50,
            "brown_requires_nuclear_unsupported": False,
            "brown_expands_to_source_component": True,
            "brown_window_size_px": 1024,
            "brown_window_overlap_px": 128,
            "brown_context_window_size_px": 256,
            "brown_maximum_local_prediction_fraction": 0.50,
            "brown_maximum_source_component_area_um2": mutation.get(
                "brown_source_area_um2", 50_000.0
            ),
            "brown_source_maximum_mean_intensity": mutation.get(
                "brown_source_intensity", 218.0
            ),
            "brown_source_dilation_bins": mutation.get("brown_source_dilation", 0),
            **(
                {
                    "self_dense_glass_gate": True,
                    "oversized_brown_gate": True,
                    "oversized_brown_maximum_mean_intensity": 180.0,
                    **(
                        {
                            "clustered_brown_gate": mutation.get(
                                "v84_clustered_brown_gate", True
                            )
                        }
                        if algorithm_version == 84
                        else {}
                    ),
                    "reconcile_enclosed_cytoplasmic_children": True,
                    "protect_organized_from_necrotic": True,
                    "organized_necrotic_protection_maximum_instance_area_um2": 500.0,
                    (
                        "organized_necrotic_protection_"
                        "minimum_local_prediction_fraction"
                    ): 0.025,
                    (
                        "organized_necrotic_protection_local_prediction_source"
                    ): "preartifact_independent_nuclear_external",
                    "organized_necrotic_sparse_glass_shape_gate": {
                        "context_window_size_px": 128,
                        "maximum_mean_intensity": 220.0,
                        "minimum_tissue_fraction": mutation.get(
                            "v82_sparse_glass_tissue_fraction", 0.14
                        ),
                        "maximum_sampled_fill_fraction": 0.40,
                        "minimum_sampled_elongation": 2.50,
                    },
                }
                if requires_v82
                else {}
            ),
        },
    }
    filter_upgrade: dict[str, object] | None = None
    if algorithm_version == 83:
        filter_upgrade = {
            "source_algorithm_version": mutation.get(
                "v83_source_algorithm_version", 82
            ),
            "source_result_fingerprint": "b" * 64,
            "scope": "post-inference-focus-artifact-exclusion-v1",
        }
    if algorithm_version == 86:
        filter_upgrade = {
            "source_algorithm_version": 75,
            "source_result_fingerprint": "b" * 64,
            "scope": "post-inference-chromatic-microcluster-exclusion-v1",
        }
    if algorithm_version == 87:
        filter_upgrade = {
            "source_algorithm_version": 75,
            "source_result_fingerprint": "b" * 64,
            "scope": "post-inference-organized-escape-exclusion-v1",
        }
    if algorithm_version == 88:
        filter_upgrade = {
            "source_algorithm_version": 87,
            "source_result_fingerprint": "b" * 64,
            "scope": "post-inference-pas-anucleate-microcluster-exclusion-v1",
        }
    if algorithm_version == 89:
        filter_upgrade = {
            "source_algorithm_version": mutation.get("v89_source_algorithm", 75),
            "source_result_fingerprint": "b" * 64,
            "scope": mutation.get(
                "v89_scope",
                "post-inference-section-satellite-interior-protection-v1",
            ),
        }
    if algorithm_version == 90:
        filter_upgrade = {
            "source_algorithm_version": 75,
            "source_result_fingerprint": "b" * 64,
            "scope": ("post-inference-saturated-cyan-foreign-material-exclusion-v1"),
        }
    if algorithm_version == 91:
        filter_upgrade = {
            "source_algorithm_version": 75,
            "source_result_fingerprint": "b" * 64,
            "scope": REVIEWED_ARTIFACT_EXCLUSION_SCOPE,
        }
    profile_fingerprint = "q" * 64
    if algorithm_version in {86, 88, 89, 90, 91, 93, 94, 95, 96}:
        profile_payload = {
            "algorithm_version": algorithm_version,
            "preflight_fingerprint": preflight["fingerprint"],
            "model": None,
            "request": request,
        }
        if filter_upgrade is not None:
            profile_payload["filter_upgrade"] = filter_upgrade
        profile_fingerprint = hashlib.sha256(
            json.dumps(
                profile_payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        qc["profile_fingerprint"] = profile_fingerprint
        (root / "qc/001.json").write_text(json.dumps(qc))
    write_cell_result(
        root,
        {
            "schema_version": 1,
            "algorithm_version": algorithm_version,
            "registration_result_sha256": "a" * 64,
            "preflight": "preflight.json",
            "preflight_fingerprint": preflight["fingerprint"],
            "profile_fingerprint": profile_fingerprint,
            "request": request,
            "slides": [
                {
                    "section": "001",
                    "slide": "slide.ndpi",
                    "labels": "labels/001.cells.tiff",
                    "qc": "qc/001.json",
                    "cell_count": 12,
                    "mpp_xy": [0.5, 0.5],
                    **(
                        {"source_identity": "c" * 64}
                        if algorithm_version in {91, 94, 95, 96}
                        else {}
                    ),
                }
            ],
            **(
                {"filter_upgrade": filter_upgrade} if filter_upgrade is not None else {}
            ),
            **(
                {
                    "subset_source": {
                        "scope": "cache-only-refilter-v1",
                        "source_preflight_fingerprint": "5" * 64,
                        "source_profile_fingerprint": "b" * 64,
                        "source_section_fingerprint": "6" * 64,
                        "source_labels_sha256": mutation.get(
                            "v94_subset_source_labels_sha256", "a" * 64
                        ),
                    }
                }
                if algorithm_version in {94, 95}
                else {}
            ),
            **(
                {
                    "subset_source": {
                        "scope": "cache-only-refilter-v1",
                        "source_result_fingerprint": "b" * 64,
                        "source_section_fingerprint": "6" * 64,
                        "source_labels_sha256": "a" * 64,
                    }
                }
                if algorithm_version == 93
                else {}
            ),
            **(
                {
                    "subset_source": {
                        "scope": BOUNDED_ADDITIVE_RECOVERY_SCOPE,
                        "source_result_fingerprint": bounded_source[
                            "result_fingerprint"
                        ],
                        "source_preflight_fingerprint": bounded_source[
                            "preflight_fingerprint"
                        ],
                        "source_profile_fingerprint": bounded_source[
                            "profile_fingerprint"
                        ],
                        "source_section_fingerprint": bounded_source[
                            "section_fingerprint"
                        ],
                        "source_labels_sha256": bounded_source["labels_sha256"],
                    }
                }
                if algorithm_version == 96
                else {}
            ),
        },
    )
    return root


def _composition_ready(
    run: Path,
    *,
    label_bytes: bytes,
    section_fingerprint: str,
    source_identity: str = "c" * 64,
    dense_source_labels_sha256: str | None = None,
    dense_source_section_fingerprint: str | None = None,
) -> Path:
    """Add exact slide/model identity required for section composition tests."""

    result = validate_cell_result(run)
    preflight = json.loads((run / "preflight.json").read_text())
    preflight["fingerprint"] = "8" * 64
    preflight["slides"][0].update(
        {
            "source_identity": source_identity,
            "mask_sha256": "e" * 64,
            "transform_sha256": "d" * 64,
            "native_shape": [384, 512],
            "content_bbox_xywh": [0, 0, 512, 384],
            "thumbnail_shape": [96, 128],
            "mpp_xy": [0.5, 0.5],
            "is_reference": True,
        }
    )
    (run / "preflight.json").write_text(json.dumps(preflight))
    (run / "labels/001.cells.tiff").write_bytes(label_bytes)
    qc = json.loads((run / "qc/001.json").read_text())
    qc.update(
        {
            "profile_fingerprint": "7" * 64,
            "section_fingerprint": section_fingerprint,
            "labels_sha256": hashlib.sha256(label_bytes).hexdigest(),
        }
    )
    (run / "qc/001.json").write_text(json.dumps(qc))
    core = {
        key: value
        for key, value in result.items()
        if key not in {"artifacts", "fingerprint"}
    }
    core.update(
        {
            "coordinate_space": "native_content_bbox",
            "preflight_fingerprint": "8" * 64,
            "profile_fingerprint": "7" * 64,
            "registration_approval_sha256": "f" * 64,
            "model": {
                "cellpose_version": "4.2.1",
                "weight_name": "cpsam",
                "weight_sha256": "9" * 64,
                "device": {
                    "requested": "cpu",
                    "resolved": "cpu",
                    "backend": "cpu",
                },
            },
        }
    )
    core["slides"][0].update(
        {
            "source_identity": source_identity,
            "is_reference": True,
            "mpp_xy": [0.5, 0.5],
            "native_shape": [384, 512],
            "content_bbox_xywh": [0, 0, 512, 384],
            "transform_sha256": "d" * 64,
        }
    )
    if core.get("algorithm_version") == 91:
        reviewed_manifest = core["request"]["reviewed_artifact_exclusion"]
        reviewed_manifest["source_preflight_fingerprint"] = "8" * 64
        reviewed_manifest["sections"][0]["source_identity"] = source_identity
        qc["reviewed_artifact_manifest_sha256"] = (
            reviewed_artifact_exclusion_manifest_sha256(reviewed_manifest)
        )
        qc["reviewed_artifact_section_sha256"] = (
            reviewed_artifact_exclusion_section_sha256(reviewed_manifest["sections"][0])
        )
        qc["instance_evidence"]["reviewed_artifact_exclusion"] = (
            reviewed_artifact_exclusion_qc_evidence(reviewed_manifest, "001")
        )
    if core.get("algorithm_version") in {94, 95}:
        recovery = core["request"]["dense_small_cell_recovery"]
        subset_source = core["subset_source"]
        recovery["source_identity"] = source_identity
        recovery["source_preflight_fingerprint"] = "8" * 64
        subset_source["source_preflight_fingerprint"] = "8" * 64
        subset_source["source_profile_fingerprint"] = "7" * 64
        if dense_source_labels_sha256 is not None:
            recovery["source_labels_sha256"] = dense_source_labels_sha256
            subset_source["source_labels_sha256"] = dense_source_labels_sha256
        if dense_source_section_fingerprint is not None:
            subset_source["source_section_fingerprint"] = (
                dense_source_section_fingerprint
            )
    if core.get("algorithm_version") in {89, 90, 91, 93, 94, 95}:
        profile_payload = {
            "algorithm_version": core["algorithm_version"],
            "preflight_fingerprint": core["preflight_fingerprint"],
            "model": core["model"],
            "request": core["request"],
        }
        if "filter_upgrade" in core:
            profile_payload["filter_upgrade"] = core["filter_upgrade"]
        profile_fingerprint = hashlib.sha256(
            json.dumps(
                profile_payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        core["profile_fingerprint"] = profile_fingerprint
        qc["profile_fingerprint"] = profile_fingerprint
        (run / "qc/001.json").write_text(json.dumps(qc))
    write_cell_result(run, core)
    return run
