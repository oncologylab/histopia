from __future__ import annotations

import copy
import json
import subprocess
import sys

import numpy as np
import pytest

from histopia.protein._vector_transfer import (
    match_registered_cells,
    transfer_protein_vectors,
)
from histopia.study._labels import (
    CellLabelProbabilities,
    annotation_review_sample,
    evaluate_cell_labels,
    interpolate_label_probabilities,
)
from histopia.study._ledger import AssignmentLedger
from histopia.study._manifest import (
    feature_identity,
    load_study_manifest,
    validate_fit_scope,
    validate_study_manifest,
    write_study_manifest,
)
from histopia.study._neighborhoods import (
    physical_neighborhoods,
    within_tissue_permutation,
)
from histopia.study._preprocessing import prepare_he_rgb
from histopia.study._regions import (
    assign_cells_by_overlap,
    assign_cells_from_overlap,
    connected_tissue_regions,
    region_expression_summary,
    sample_region_labels,
)


def study():
    return {
        "schema_version": 1,
        "study_id": "synthetic-v1",
        "mice": [
            {"mouse_id": "m1", "role": "development", "previously_exposed": True},
            {
                "mouse_id": "m2",
                "role": "test",
                "previously_exposed": False,
                "exposure_evidence": "prospective enrollment",
            },
        ],
        "slides": [
            {
                "slide_id": "s1",
                "mouse_id": "m1",
                "organ": "pancreas",
                "original_scan_id": "o1",
            },
            {
                "slide_id": "s2",
                "mouse_id": "m1",
                "organ": "lung",
                "original_scan_id": "o2",
            },
        ],
    }


def regions(labels=None):
    return connected_tissue_regions(
        np.array([[0, 0, -1, 0], [0, 0, -1, 1]]) if labels is None else labels,
        semantic_fingerprint="semantic",
        approval_fingerprint="approval",
        section_id="s1",
        pixel_size_um=10,
    )


def correspondence(**kwargs):
    return match_registered_cells(
        ["s:1", "s:2", "s:3"],
        ["t:1", "t:2", "t:3"],
        [[0, 0, 0], [10, 0, 0], [20, 0, 0]],
        [[0, 0, 4], [20, 0, 4], [500, 0, 4]],
        [[-1, 0], [0, 0], [1, 0]],
        [[-1, 0], [1, 0], [1, 0]],
        upstream_fingerprints={"geometry": "g", "morphology": "m"},
        neighbors=2,
        candidates=3,
        maximum_distance_um=32,
        **kwargs,
    )


def test_study_prevents_mouse_leakage_and_exposure_reclassification(tmp_path):
    base = study()
    validate_study_manifest(base)
    broken = copy.deepcopy(base)
    broken["slides"][1]["role"] = "test"
    with pytest.raises(ValueError, match="one split"):
        validate_study_manifest(broken)
    for exposure in (True, None):
        broken = copy.deepcopy(base)
        broken["mice"][1]["previously_exposed"] = exposure
        with pytest.raises(ValueError, match="exposure"):
            validate_study_manifest(broken)
    with pytest.raises(ValueError, match="development"):
        validate_fit_scope(base, ["m2"])
    with pytest.raises(ValueError, match="overlap"):
        validate_fit_scope(base, ["m1"], evaluation_mice=["m1"])
    path = write_study_manifest(tmp_path / "study.json", base)
    assert load_study_manifest(path)["study_id"] == "synthetic-v1"
    base["study_id"] = "changed"
    with pytest.raises(ValueError, match="frozen"):
        write_study_manifest(path, base)
    changed = json.loads(path.read_text())
    changed["slides"][0]["organ"] = "kidney"
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="fingerprint"):
        load_study_manifest(path)


def test_derivative_and_spacing_evidence_are_required():
    base = study()
    row = base["slides"][0]
    row["is_derivative"] = True
    with pytest.raises(ValueError, match="derivative_of"):
        validate_study_manifest(base)
    row["derivative_of"] = "o1"
    row.update(z_spacing_um=4, z_spacing_kind="physical")
    with pytest.raises(ValueError, match="source evidence"):
        validate_study_manifest(base)
    row["z_spacing_kind"] = "assumed"
    validate_study_manifest(base)


def test_feature_identity_changes_for_scientific_binding_changes():
    from histopia.study._manifest import _FEATURE_BINDINGS

    bindings = {key: "bound" for key in _FEATURE_BINDINGS}
    before = feature_identity(bindings)
    for key in bindings:
        assert feature_identity({**bindings, key: "changed"}) != before
    with pytest.raises(ValueError, match="incomplete"):
        feature_identity({"source_sha256": "x"})
    image = np.array([[[100, 50, 150], [255, 255, 255]]], np.uint8)
    assert np.array_equal(image, prepare_he_rgb(image))


def test_vector_transfer_missing_proteins_order_support_and_uncertainty():
    graph = correspondence()
    values = np.array([[1, np.nan], [2, np.nan], [4, 8]])
    result = transfer_protein_vectors(
        graph,
        ["s:1", "s:2", "s:3"],
        ["p1", "p2"],
        values,
        source_fingerprint="source",
        source_uncertainty=np.full(values.shape, 0.2),
    )
    assert result.values.shape == (3, 2)
    assert np.isnan(result.values[0, 1])
    assert not result.support[2].any()
    assert np.isnan(result.values[2]).all()
    assert np.isnan(result.uncertainty[2]).all()
    assert np.isfinite(result.uncertainty[1, 0])
    with pytest.raises(ValueError, match="ordering"):
        transfer_protein_vectors(
            graph,
            ["s:3", "s:2", "s:1"],
            ["p1", "p2"],
            values,
            source_fingerprint="source",
        )
    changed = transfer_protein_vectors(
        graph,
        graph.source_cell_ids,
        ["p1", "p2"],
        values * 2,
        source_fingerprint="source2",
    )
    assert np.array_equal(changed.correspondence.source_indices, graph.source_indices)
    assert changed.fingerprint != result.fingerprint
    assert np.isnan(changed.uncertainty).all()
    blocked = correspondence(target_support=np.array([False, True, False]))
    assert (blocked.source_indices[0] == -1).all()
    assert blocked.fingerprint != graph.fingerprint


@pytest.mark.parametrize("mode", ["combined", "spatial", "morphology"])
def test_correspondence_controls_are_deterministic_and_distance_gated(mode):
    graph = correspondence(mode=mode)
    repeat = correspondence(mode=mode, batch_size=1)
    np.testing.assert_array_equal(graph.source_indices, repeat.source_indices)
    np.testing.assert_allclose(graph.weights, repeat.weights)
    assert (graph.source_indices[2] == -1).all()
    assert np.all(graph.distances_um[np.isfinite(graph.distances_um)] <= 32)


def test_connected_regions_keep_disconnected_class_and_version_identity():
    atlas = regions()
    assert len(atlas.regions) == 3
    assert atlas.regions[0]["semantic_class"] == atlas.regions[1]["semantic_class"]
    assert atlas.regions[0]["region_id"] != atlas.regions[1]["region_id"]
    assert atlas.fingerprint == regions().fingerprint
    changed = connected_tissue_regions(
        np.array([[0, -1], [-1, 0]]),
        semantic_fingerprint="semantic",
        approval_fingerprint="approval",
        section_id="s1",
        pixel_size_um=10,
    )
    assert len(changed.regions) == 2  # corner touch is not connected
    assert changed.fingerprint != atlas.fingerprint


def test_coordinate_transform_roundtrip_and_outside_support():
    atlas = regions()
    matrix = np.array([[0.5, 0, -1], [0, 0.5, -2], [0, 0, 1]])
    grid = np.array([[0.5, 0.5], [3.5, 0.5], [-1, 0], [2.5, 1.5]])
    native = (np.column_stack((grid, np.ones(4))) @ np.linalg.inv(matrix).T)[:, :2]
    result = sample_region_labels(atlas, native, matrix)
    assert result.tolist() == [1, 2, 0, 0]
    with pytest.raises(ValueError, match="invertible"):
        sample_region_labels(atlas, native, np.zeros((3, 3)))


def test_boundary_membership_ambiguity_and_missing_summary():
    atlas = regions(np.array([[0, 0, 1, 1], [0, 0, 1, 1]]))
    cells = np.array([[1, 1, 2, 2], [3, 4, 4, 5]])
    assignment = assign_cells_by_overlap(
        cells, atlas.labels, atlas, cell_fingerprint="cells"
    )
    assert assignment.status.tolist() == [
        "assigned",
        "assigned",
        "assigned",
        "ambiguous",
        "assigned",
    ]
    assert assignment.region_ids[3] == ""
    values = np.array([[1.0], [4.0], [3.0], [999.0], [np.nan]])
    rows = region_expression_summary(
        atlas,
        assignment,
        [1, 2, 3, 4, 5],
        ["yap"],
        values,
        np.ones_like(values, bool),
        evidence_kind="measured",
        expression_fingerprint="expression",
    )
    assert rows[0]["cell_count"] == 2
    assert rows[0]["mean"] == rows[0]["median"] == 2
    assert rows[0]["dispersion_sd"] == 1
    assert rows[1]["supported_coverage"] == 0.5
    assert all(row["positive_fraction"] is None for row in rows)
    with pytest.raises(ValueError, match="validated"):
        region_expression_summary(
            atlas,
            assignment,
            [1, 2, 3, 4, 5],
            ["yap"],
            values,
            np.ones_like(values, bool),
            evidence_kind="measured",
            expression_fingerprint="e",
            positivity_thresholds={"yap": {"value": 1}},
        )
    outside = assign_cells_from_overlap(
        [1], np.array([[9, 1, 0]]), atlas, cell_fingerprint="c"
    )
    assert outside.status[0] == "ambiguous"


def test_labels_interpolation_abstention_and_no_fake_reference():
    graph = correspondence()
    labels = CellLabelProbabilities(
        graph.source_cell_ids,
        np.array([[0.8, 0.2], [0.5, 0.5], [0.1, 0.9]]),
        np.ones(3, bool),
        "upstream",
        "morphology",
        classes=("epithelial", "uncertain"),
    )
    result = interpolate_label_probabilities(labels, graph)
    assert result.evidence_kind == "interpolated"
    assert np.isnan(result.probabilities[2]).all()
    np.testing.assert_allclose(result.probabilities[result.support].sum(axis=1), 1)
    with pytest.raises(ValueError, match="lab-reviewed"):
        evaluate_cell_labels(labels, {})
    reference = {
        "independent_lab_review": True,
        "blinded": True,
        "review_fingerprint": "review",
        "cell_ids": graph.source_cell_ids,
        "class_ids": ["epithelial", "epithelial", "uncertain"],
    }
    metrics = evaluate_cell_labels(labels, reference)
    assert metrics["called_cells"] == 2
    assert metrics["accuracy"] == 1
    assert metrics["abstention_fraction"] == pytest.approx(1 / 3)
    sample = annotation_review_sample(
        labels,
        ["m1", "m2", "m2"],
        {"m1": "development", "m2": "test"},
        blinded=True,
        per_class=1,
    )
    assert all(
        "suggested_class" not in row and row["reference_label"] == ""
        for row in sample["rows"]
    )


def test_neighborhoods_never_cross_mouse_tissue_or_unsupported_geometry():
    xyz = np.array([[0, 0, 0], [10, 0, 0], [0, 0, 4], [0, 0, 0], [0, 0, 0]])
    values = np.array([[1, np.nan], [3, 5], [9, 9], [99, 99], [999, 999.0]])
    args = (
        ["a", "b", "c", "d", "e"],
        xyz,
        values,
        np.isfinite(values),
        ["class", "protein"],
        ["m1", "m1", "m1", "m2", "m1"],
        [0, 0, 1, 0, 0],
        ["t", "t", "t", "t", "other"],
    )
    result = physical_neighborhoods(*args, upstream_fingerprint="u")
    assert result.neighbor_count.tolist() == [1, 1, 0, 0, 0]
    np.testing.assert_allclose(result.means[0], [3, 5])
    assert np.isnan(result.means[1, 1])
    with pytest.raises(ValueError, match="Z spacing"):
        physical_neighborhoods(*args, upstream_fingerprint="u", adjacent_sections=True)
    adjacent = physical_neighborhoods(
        *args,
        upstream_fingerprint="u",
        adjacent_sections=True,
        z_spacing_kind="assumed",
    )
    assert adjacent.neighbor_count[:3].tolist() == [2, 2, 2]
    permuted, valid, order = within_tissue_permutation(
        values, np.isfinite(values), args[5], args[6], args[7], seed=1
    )
    assert set(order[:2]) == {0, 1}
    assert order[2:].tolist() == [2, 3, 4]
    np.testing.assert_array_equal(valid, np.isfinite(permuted))


def test_assignment_claim_is_unique_durable_and_not_age_released(tmp_path):
    ledger = AssignmentLedger(tmp_path / "assignments")
    ledger.add("job", "artifact", "node2", {"dependency": "approved"})
    with pytest.raises(FileExistsError):
        ledger.add("job2", "artifact", "node3", {})
    with pytest.raises(ValueError):
        ledger.claim("job", "node3")
    owner = ledger.claim("job", "node2")
    with pytest.raises(ValueError):
        AssignmentLedger(ledger.path).claim("job", "node2")
    ledger.finish("job", owner, status="complete", evidence={"sha256": "evidence"})
    assert ledger.rows()[0]["status"] == "complete"


def test_study_import_and_manifest_cli_stay_lightweight():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import histopia.study; "
            "from histopia.study import load_study_manifest; "
            "assert 'numpy' not in sys.modules; assert 'torch' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_vector_evaluation_preserves_mouse_and_section_bins_and_missing_support():
    from histopia.study._evaluation import evaluate_protein_vectors

    kwargs = dict(
        study=study(),
        cell_ids=["a", "b", "c", "d", "e"],
        mouse_ids=["m2"] * 5,
        section_ids=["s1", "s1", "s1", "s2", "s2"],
        xy_um=[[0, 0], [2, 0], [65, 0], [0, 0], [0, 0]],
        protein_ids=["YAP"],
        measured=[[0], [2], [4], [10], [99]],
        predicted=[[2], [4], [4], [8], [np.nan]],
        support=np.array([[True], [True], [True], [True], [False]]),
        fitting_mice_by_holdout={"m2": ["m1"]},
        protocol="direct_he_prediction",
        measurement_mpp=4,
    )
    result = evaluate_protein_vectors(**kwargs)
    row = result["mouse_metrics"][0]
    assert row["supported_coverage"] == 0.8
    assert row["supported_cell_metrics"]["mae"] == 1.5
    assert row["aggregate_metrics"]["n"] == 3
    assert row["aggregate_metrics"]["mae"] == pytest.approx(4 / 3)
    assert result["measurement_mpp"] == 4
    assert len(result["spatial_errors"]) == 3
    kwargs["fitting_mice_by_holdout"] = {"m2": ["m2"]}
    with pytest.raises(ValueError, match="development"):
        evaluate_protein_vectors(**kwargs)
    kwargs["fitting_mice_by_holdout"] = {"m2": ["m1"]}
    kwargs["study"]["mice"][1]["role"] = "protected"
    with pytest.raises(ValueError, match="protected"):
        evaluate_protein_vectors(**kwargs)


def test_analysis_arrays_bind_cache_and_refuse_pickle(tmp_path):
    from histopia.study._artifacts import load_analysis_arrays, write_analysis_arrays

    output = tmp_path / "vectors.npz"
    arrays = {
        "cell_ids": np.array(["s:a", "s:b"]),
        "values": np.array([[1, np.nan], [2, 3]]),
        "support": np.array([[True, False], [True, True]]),
    }
    metadata = {"coordinate_units": "um", "upstream_fingerprints": ["bound-input"]}
    write_analysis_arrays(output, arrays, metadata=metadata)
    actual, bound = load_analysis_arrays(output)
    np.testing.assert_array_equal(actual["values"], arrays["values"])
    assert bound == metadata
    np.savez(output, values=np.zeros((2, 2)))
    with pytest.raises(ValueError, match="binding changed"):
        load_analysis_arrays(output)
    with pytest.raises(ValueError, match="object arrays"):
        write_analysis_arrays(output, {"bad": np.array([{}])}, metadata=metadata)


def test_sparse_region_counts_preserve_background_ties_and_unsupported_cells():
    from scipy.sparse import csr_matrix

    tissue = regions()
    counts = np.array([[5, 5, 0, 0], [1, 6, 3, 0], [0, 0, 0, 0], [0, 0, 4, 4]])
    dense = assign_cells_from_overlap(
        [1, 2, 3, 4], counts, tissue, cell_fingerprint="cells"
    )
    sparse = assign_cells_from_overlap(
        [1, 2, 3, 4], csr_matrix(counts), tissue, cell_fingerprint="cells"
    )
    np.testing.assert_array_equal(sparse.status, dense.status)
    assert sparse.status.tolist() == [
        "ambiguous",
        "assigned",
        "unsupported",
        "ambiguous",
    ]
    assert sparse.dominant_fraction[1] == 0.6


def test_hpa_adapter_preserves_antibody_species_and_specimen_separation(tmp_path):
    from histopia.study._external import hpa_ihc_inventory

    path = tmp_path / "entry.xml"
    path.write_text("""<proteinAtlas><entry><name>KRT19</name>
<identifier id="ENSG-example"/><antibody id="HPA-one">
<tissueExpression technology="IHC" assayType="tissue"><data><tissue>liver</tissue>
<tissueCell><cellType>cholangiocytes</cellType><level type="staining">high</level>
</tissueCell><patient><patientId>p1</patientId><image><imageUrl>https://example/a.jpg
</imageUrl></image></patient></data></tissueExpression></antibody>
<antibody id="HPA-two"><tissueExpression technology="IHC" assayType="tissue">
<data><tissue>liver</tissue><patient><patientId>p2</patientId><image>
<imageUrl>https://example/b.jpg</imageUrl></image></patient></data>
</tissueExpression></antibody></entry></proteinAtlas>""")
    rows = hpa_ihc_inventory(path)
    assert [r["antibody_id"] for r in rows] == ["HPA-one", "HPA-two"]
    assert [r["specimen_id"] for r in rows] == ["p1", "p2"]
    assert all(r["species"] == "Homo sapiens" for r in rows)
    assert all(not r["matched_tcga_target"] for r in rows)
    assert all(r["phospho_total_equivalence"] == "unverified" for r in rows)
    assert hpa_ihc_inventory(path, organs=("pancreas",)) == []


def test_missing_marker_priors_cannot_create_supported_broad_classes():
    from histopia.study._astir import fit_astir_comparator, marker_prior_coverage

    priors = {"epithelial": ["CK19"], "immune": ["CD45"], "vascular": ["CD31"]}
    coverage = marker_prior_coverage(["CK19"], priors)
    assert coverage["epithelial"]["supported"]
    assert coverage["immune"]["missing"] == ["CD45"]
    assert not coverage["stromal"]["supported"]
    with pytest.raises(ValueError, match="at least two"):
        fit_astir_comparator(
            study(),
            ["a", "b"],
            ["m1", "m1"],
            ["CK19"],
            [[1], [2]],
            np.ones((2, 1), bool),
            priors=priors,
            prior_fingerprint="prior",
            input_fingerprint="input",
        )
