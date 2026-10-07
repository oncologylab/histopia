from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from histopia.protein import (
    CellExpressionTable,
    CellFeatureSet,
    CellTokenFeatures,
    PortableCrossAttentionRegressor,
    PortableMultiTowerRegressor,
    PortableRelationalRegressor,
    PortableSharedMultitaskRegressor,
    ProteinModel,
    ProteinPredictionConfig,
    ProteinPredictions,
    ProteinTarget,
    RegisteredAnchorTransfer,
    TargetFreeTailSwitch,
    VipsPatchReader,
    aggregate_cell_measurements,
    build_cell_feature_set,
    build_spatial_split,
    cell_neighborhood_features,
    cell_phenotype_features,
    cell_shape_features,
    compartment_sampled_labels,
    confidence_weighted_registered_residual_correction,
    evaluate_predictions,
    evaluate_protein_promotion,
    evaluate_stain_invariance,
    extend_cell_expression_table,
    extract_cell_token_features,
    fit_equal_section_od_harmonization,
    fit_portable_shared_multitask,
    fit_protein_model,
    fit_registered_od_calibration,
    fixed_stain_neutral_projection,
    label_centroids_from_tiff,
    leave_one_section_out_consensus,
    morphology_compatible_smoothing,
    morphospatial_features,
    multiscale_neighborhood_features,
    neutralize_hdab_morphology,
    normalize_target_id,
    out_of_fold_deep_predictions,
    pool_patch_features_to_cells,
    registered_morphospatial_anchor_transfer,
    registered_position_features,
    render_neutral_morphology,
    sampled_label_expected_analysis_pixels,
    select_bracketing_sections,
    select_protein_architecture_candidate,
    tail_guarded_registered_anchor_blend,
    validate_protein_approval,
    validate_protein_result,
    validated_adaptive_target_measurement,
    write_protein_result,
)
from histopia.protein._cell_features import (
    cell_token_feature_index,
    validate_cell_token_feature_binding,
)
from histopia.protein._cli import _load_artifact, _region_centered_blend
from histopia.protein._deep import _spatial_neighbor_index
from histopia.protein._manifest import approve_protein_result
from histopia.protein._measurements import inner_boundary_mask
from histopia.protein._multiscale import _content_aligned_source
from histopia.protein._real_data import _validate_analysis_slide_order
from histopia.protein._relational import (
    _bounded_neighbor_batch_size,
    _inverse_distance_transfer,
    _relational_messages,
    _relational_training_objective,
)
from histopia.protein._splits import leave_one_group_out


def test_real_run_slide_binding_honors_only_sealed_exclusions() -> None:
    complete = ["he", "yap", "b-catenin"]
    _validate_analysis_slide_order(
        ["he", "yap"],
        complete,
        complete,
        [{"slide_id": "b-catenin", "reason": "Owner excluded from analysis."}],
    )

    with pytest.raises(ValueError, match="sealed exclusions"):
        _validate_analysis_slide_order(["he"], complete, complete, [])
    with pytest.raises(ValueError, match="stain and semantic"):
        _validate_analysis_slide_order(
            ["he", "yap"],
            complete,
            ["he", "b-catenin", "yap"],
            [{"slide_id": "b-catenin", "reason": "Owner excluded."}],
        )
    with pytest.raises(ValueError, match="unknown registration"):
        _validate_analysis_slide_order(
            complete,
            complete,
            complete,
            [{"slide_id": "missing", "reason": "Owner excluded."}],
        )


def test_multiscale_cell_features_are_fingerprinted_path_free_and_round_trip(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(71)
    labels = np.arange(1, 7, dtype=np.uint32)
    native = np.column_stack((np.arange(6) * 4.0, np.arange(6) * 2.0))
    xyz = np.column_stack((native, np.arange(6) * 5.0))
    artifact = build_cell_feature_set(
        slide_id="mouse-001",
        label_ids=labels,
        native_xy=native,
        reference_um_xyz=xyz,
        morphology=rng.normal(size=(6, 8)).astype(np.float32),
        phenotype=rng.normal(size=(6, 19)).astype(np.float32),
        supported=np.ones(6, dtype=bool),
        radii_um=(16.0, 32.0),
        provenance={"source_identity": "a" * 64, "schema_version": 1},
    )

    loaded = CellFeatureSet.load(artifact.save(tmp_path / "features.npz"))

    assert loaded.fingerprint == artifact.fingerprint
    assert loaded.features.shape == (6, 8 + 19 + 2 * 19 + 21)
    assert str(tmp_path) not in json.dumps(loaded.provenance)
    np.testing.assert_allclose(loaded.features, artifact.features)
    with pytest.raises(ValueError, match="path"):
        build_cell_feature_set(
            slide_id="mouse-001",
            label_ids=labels,
            native_xy=native,
            reference_um_xyz=xyz,
            morphology=rng.normal(size=(6, 8)).astype(np.float32),
            phenotype=rng.normal(size=(6, 19)).astype(np.float32),
            supported=np.ones(6, dtype=bool),
            provenance={"source_path": "/private/slide.ndpi"},
        )


def test_hematoxylin_cell_phenotype_is_invariant_to_synthetic_dab() -> None:
    labels = np.zeros((12, 12), dtype=np.uint32)
    labels[2:10, 2:10] = 1
    basis = np.asarray(
        [[0.650, 0.268], [0.704, 0.570], [0.286, 0.776]],
        dtype=np.float64,
    )

    def render(dab: float) -> np.ndarray:
        concentration = np.zeros((12, 12, 2), dtype=np.float64)
        concentration[..., 0] = 0.55
        concentration[..., 1] = dab
        od = concentration @ basis.T
        return np.clip(np.rint(256 * np.exp(-od) - 1), 0, 255).astype(np.uint8)

    _ids, no_dab = cell_phenotype_features(
        labels,
        render(0.0),
        pixel_size_um_xy=(0.5, 0.5),
    )
    _ids, strong_dab = cell_phenotype_features(
        labels,
        render(1.2),
        pixel_size_um_xy=(0.5, 0.5),
    )

    np.testing.assert_allclose(no_dab[:, :9], strong_dab[:, :9], atol=0)
    np.testing.assert_allclose(no_dab[:, 9:], strong_dab[:, 9:], atol=0.03)


def test_native_source_is_cropped_to_the_cell_content_box() -> None:
    labels = SimpleNamespace(width=20, height=10)
    crops: list[tuple[int, int, int, int]] = []

    class Source:
        width = 100
        height = 80

        def crop(self, x: int, y: int, width: int, height: int) -> object:
            crops.append((x, y, width, height))
            return SimpleNamespace(width=width, height=height)

    aligned = _content_aligned_source(
        labels,
        Source(),
        content_bbox_native_xywh=(11, 17, 20, 10),
    )
    assert (aligned.width, aligned.height) == (20, 10)
    assert crops == [(11, 17, 20, 10)]

    with pytest.raises(ValueError, match="does not match"):
        _content_aligned_source(
            labels,
            Source(),
            content_bbox_native_xywh=(11, 17, 19, 10),
        )


def test_multiscale_neighborhood_and_registered_position_are_local() -> None:
    coordinates = np.asarray([[0.0, 0.0], [3.0, 0.0], [100.0, 0.0]])
    morphology = np.asarray([[0.0, 1.0], [1.0, 0.0], [4.0, 4.0]])
    neighborhood, names = multiscale_neighborhood_features(
        coordinates,
        morphology,
        radii_um=(8.0,),
        neighbors=2,
        projection_components=2,
    )
    position, position_names = registered_position_features(
        np.column_stack((coordinates, [0.0, 5.0, 10.0]))
    )

    count_column = names.index("r8_log_count")
    np.testing.assert_allclose(neighborhood[:, count_column], np.log1p([1, 1, 0]))
    assert position.shape == (3, 21)
    assert len(position_names) == 21


def test_portable_multi_tower_round_trip_and_accelerated_prediction(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(72)
    model = PortableMultiTowerRegressor(
        feature_mean=np.zeros(4, dtype=np.float32),
        feature_scale=np.ones(4, dtype=np.float32),
        group_slices={"morphology": (0, 2), "phenotype": (2, 4)},
        tower_weights=(
            rng.normal(size=(2, 2)).astype(np.float32),
            rng.normal(size=(2, 2)).astype(np.float32),
        ),
        tower_biases=(np.zeros(2, np.float32), np.zeros(2, np.float32)),
        fusion_weight=rng.normal(size=(4, 3)).astype(np.float32),
        fusion_bias=np.zeros(3, np.float32),
        residual_weights=(rng.normal(size=(3, 3)).astype(np.float32),),
        residual_biases=(np.zeros(3, np.float32),),
        mean_weight=rng.normal(size=(3, 1)).astype(np.float32),
        mean_bias=np.zeros(1, np.float32),
        variance_weight=np.zeros((3, 1), np.float32),
        variance_bias=np.asarray([-3.0], np.float32),
        target_scale=0.8,
        provenance={"fixture": True, "target_od_upper": 1.5},
    )
    query = rng.normal(size=(5, 4)).astype(np.float32)
    expected = model.predict(query)
    loaded = PortableMultiTowerRegressor.load(model.save(tmp_path / "multi-tower.npz"))
    accelerated = loaded.predict_accelerated(query, device="cpu")

    assert loaded.fingerprint == model.fingerprint
    np.testing.assert_allclose(loaded.predict(query)[0], expected[0])
    np.testing.assert_allclose(accelerated[0], expected[0], atol=1e-5)
    np.testing.assert_allclose(accelerated[1], expected[1], atol=1e-5)
    extreme = loaded.predict_accelerated(
        np.full((2, 4), 100.0, dtype=np.float32), device="cpu"
    )
    assert np.all(np.isfinite(extreme[0]))
    assert np.all(extreme[0] <= 1.5)
    assert np.all(extreme[1] <= 0.75)


def test_portable_multi_tower_seals_training_only_morphology_bank(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(720)
    model = PortableMultiTowerRegressor(
        feature_mean=np.zeros(4, dtype=np.float32),
        feature_scale=np.ones(4, dtype=np.float32),
        group_slices={"morphology": (0, 2), "phenotype": (2, 4)},
        tower_weights=(
            rng.normal(size=(2, 2)).astype(np.float32),
            rng.normal(size=(2, 2)).astype(np.float32),
        ),
        tower_biases=(np.zeros(2, np.float32), np.zeros(2, np.float32)),
        fusion_weight=rng.normal(size=(4, 3)).astype(np.float32),
        fusion_bias=np.zeros(3, np.float32),
        residual_weights=(rng.normal(size=(3, 3)).astype(np.float32),),
        residual_biases=(np.zeros(3, np.float32),),
        mean_weight=rng.normal(size=(3, 1)).astype(np.float32),
        mean_bias=np.zeros(1, np.float32),
        variance_weight=np.zeros((3, 1), np.float32),
        variance_bias=np.asarray([-3.0], np.float32),
        target_scale=0.8,
        provenance={"fixture": True, "target_od_upper": 1.5},
    )
    bank_features = np.asarray(
        [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [3.0, 3.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    banked = model.with_morphology_transfer_bank(
        bank_features,
        np.asarray([0.1, 0.5, 1.0], dtype=np.float32),
    )
    loaded = PortableMultiTowerRegressor.load(
        banked.save(tmp_path / "multi-tower-with-bank.npz")
    )

    prediction, uncertainty = loaded.predict_morphology_transfer(
        bank_features[:1], neighbors=1
    )

    assert loaded.fingerprint == banked.fingerprint
    assert loaded.fingerprint != model.fingerprint
    assert loaded.morphology_bank_key is not None
    np.testing.assert_allclose(prediction, [0.5])
    np.testing.assert_allclose(uncertainty, 0.0)


def test_portable_shared_multitask_round_trip_and_primary_head_fit(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(721)
    features = rng.normal(size=(24, 6)).astype(np.float32)
    target = np.maximum(0.4 + features[:, 0] * 0.2, 0).astype(np.float32)
    auxiliary_features = rng.normal(size=(20, 6)).astype(np.float32)
    auxiliary_target = np.maximum(
        0.6 - auxiliary_features[:, 1] * 0.15,
        0,
    ).astype(np.float32)
    model = fit_portable_shared_multitask(
        features,
        target,
        np.arange(20),
        auxiliary=((auxiliary_features, auxiliary_target, np.arange(18)),),
        auxiliary_target_ids=("different-antibody",),
        seed=12,
        epochs=2,
        batch_size=16,
    )
    query = features[20:]
    expected = model.predict(query)
    loaded = PortableSharedMultitaskRegressor.load(
        model.save(tmp_path / "shared-multitask.npz")
    )
    accelerated = loaded.predict_accelerated(query, device="cpu")

    assert loaded.fingerprint == model.fingerprint
    assert loaded.provenance["auxiliary_targets"] == ["different-antibody"]
    np.testing.assert_allclose(accelerated[0], expected[0], atol=2e-4)
    np.testing.assert_allclose(accelerated[1], expected[1], atol=0)
    assert np.all(np.isfinite(accelerated[0]))
    assert np.all(accelerated[0] >= 0)


def test_portable_relational_round_trip_matches_accelerated_inference(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(73)
    feature_count = 4
    width = 8
    bank_count = 6
    model = PortableRelationalRegressor(
        architecture="graph_transformer",
        feature_mean=np.zeros(feature_count, np.float32),
        feature_scale=np.ones(feature_count, np.float32),
        bank_embedding=rng.normal(size=(bank_count, width)).astype(np.float32),
        bank_morphology_key=rng.normal(size=(bank_count, 16)).astype(np.float32),
        bank_reference_um_xyz=rng.normal(size=(bank_count, 3)).astype(np.float32),
        bank_od=np.linspace(0.1, 0.9, bank_count, dtype=np.float32),
        embed_weight=rng.normal(size=(width, feature_count)).astype(np.float32),
        embed_bias=rng.normal(size=width).astype(np.float32),
        norm_weight=np.ones(width, np.float32),
        norm_bias=np.zeros(width, np.float32),
        attention_weight=rng.normal(size=(3 * width, width)).astype(np.float32),
        attention_bias=rng.normal(size=3 * width).astype(np.float32),
        attention_output_weight=rng.normal(size=(width, width)).astype(np.float32),
        attention_output_bias=rng.normal(size=width).astype(np.float32),
        head0_weight=rng.normal(size=(width, 3 * width)).astype(np.float32),
        head0_bias=rng.normal(size=width).astype(np.float32),
        head1_weight=rng.normal(size=(1, width)).astype(np.float32),
        head1_bias=np.zeros(1, np.float32),
        heads=2,
        morphology_width=feature_count,
        target_od_upper=1.4,
        provenance={
            "fixture": True,
            "neighbor_outcomes_used_for_mean": False,
        },
    )
    query = rng.normal(size=(3, feature_count)).astype(np.float32)
    xyz = rng.normal(size=(3, 3)).astype(np.float32)
    expected = model.predict(query, xyz)
    path = model.save(tmp_path / "relational.npz")
    loaded = PortableRelationalRegressor.load(path)
    validated = _load_artifact(path, "model")
    accelerated = loaded.predict_accelerated(query, xyz, device="cpu")

    assert loaded.fingerprint == model.fingerprint
    assert isinstance(validated, PortableRelationalRegressor)
    assert validated.fingerprint == model.fingerprint
    np.testing.assert_allclose(accelerated[0], expected[0], atol=2e-5)
    np.testing.assert_allclose(accelerated[1], expected[1], atol=1e-6)
    assert np.all((accelerated[0] >= 0) & (accelerated[0] <= 1.4))

    graph = replace(
        model,
        provenance={
            **model.provenance,
            "relational_operator": "residual-graph-message-v2",
        },
        fingerprint=None,
    )
    graph_expected = graph.predict(query, xyz)
    graph_accelerated = graph.predict_accelerated(query, xyz, device="cpu")
    np.testing.assert_allclose(graph_accelerated[0], graph_expected[0], atol=2e-5)
    message_query = np.full((2, width), 0.2, dtype=np.float32)
    morphology = np.full((2, width), -0.4, dtype=np.float32)
    spatial = np.full((2, width), 0.8, dtype=np.float32)
    legacy_messages = _relational_messages(model, message_query, morphology, spatial)
    graph_messages = _relational_messages(graph, message_query, morphology, spatial)
    assert not np.allclose(graph_messages[0], legacy_messages[0])
    assert not np.allclose(graph_messages[1], legacy_messages[1])

    with pytest.raises(ValueError, match="operator is incompatible"):
        replace(graph, architecture="dual_bank_attention", fingerprint=None)


def test_morphology_transfer_excludes_exact_self_and_weights_neighbors() -> None:
    prediction, uncertainty = _inverse_distance_transfer(
        np.asarray([[0.0], [10.0]]),
        np.asarray([[0.0], [1.0], [9.0], [12.0]]),
        np.asarray([100.0, 1.0, 9.0, 12.0]),
        neighbors=1,
        batch_size=2,
    )

    np.testing.assert_allclose(prediction, [1.0, 9.0])
    np.testing.assert_allclose(uncertainty, 0.0)


def test_confidence_adaptive_transfer_preserves_confident_expression_tails() -> None:
    from histopia.protein._calibration import (
        confidence_adaptive_morphology_transfer,
    )

    baseline = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0])
    transferred = np.full(5, 0.5)
    relative = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0])

    refined, effective_weight = confidence_adaptive_morphology_transfer(
        baseline,
        transferred,
        relative,
        base_weight=0.25,
        gate_power=0.5,
    )

    np.testing.assert_allclose(
        effective_weight,
        [0.0, np.sqrt(0.75) * 0.25, 0.25, np.sqrt(0.75) * 0.25, 0.0],
    )
    np.testing.assert_allclose(refined[[0, -1]], baseline[[0, -1]])
    assert refined[1] > baseline[1]
    assert refined[3] < baseline[3]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"base_weight": 1.1}, "base_weight"),
        ({"gate_power": 0.0}, "gate_power"),
        ({"gate_floor": -0.1}, "gate_floor"),
    ],
)
def test_confidence_adaptive_transfer_rejects_invalid_controls(
    overrides: dict[str, float],
    message: str,
) -> None:
    from histopia.protein._calibration import (
        confidence_adaptive_morphology_transfer,
    )

    controls = {"base_weight": 0.25, "gate_power": 0.5, "gate_floor": 0.0}
    controls.update(overrides)
    with pytest.raises(ValueError, match=message):
        confidence_adaptive_morphology_transfer(
            np.asarray([0.2, 0.8]),
            np.asarray([0.4, 0.6]),
            np.asarray([0.1, 0.9]),
            **controls,
        )


def test_candidate_blends_calibrated_relational_mean_with_frozen_od_bank() -> None:
    from histopia.protein._study_workflow import _Candidate

    class Estimator:
        def predict_accelerated(
            self,
            features: np.ndarray,
            reference_um_xyz: np.ndarray,
            *,
            device: str,
        ) -> tuple[np.ndarray, np.ndarray]:
            del features, reference_um_xyz, device
            return np.asarray([0.2, 0.6]), np.asarray([0.1, 0.2])

        def predict_morphology_transfer(
            self,
            features: np.ndarray,
            *,
            neighbors: int,
        ) -> tuple[np.ndarray, np.ndarray]:
            del features
            assert neighbors == 8
            return np.asarray([1.0, 0.2]), np.asarray([0.05, 0.1])

    candidate = _Candidate(
        architecture="dual_bank_attention",
        estimator=Estimator(),
        source_knots=np.asarray([0.0, 1.0]),
        target_knots=np.asarray([0.0, 1.0]),
        reference_od=np.asarray([0.0, 0.25, 0.5, 0.75, 1.0]),
        fingerprint="a" * 64,
        morphology_transfer_weight=0.25,
        morphology_transfer_neighbors=8,
    )

    _probability, relative, od, uncertainty = candidate.predict(
        np.zeros((2, 3), dtype=np.float32),
        reference_um_xyz=np.zeros((2, 3)),
        device="cpu",
    )

    np.testing.assert_allclose(od, [0.4, 0.5])
    np.testing.assert_allclose(relative, [0.4, 0.6])
    expected_variance = (
        0.75**2 * np.square([0.1, 0.2])
        + 0.25**2 * np.square([0.05, 0.1])
        + 0.25 * 0.75 * np.square(np.asarray([0.2, 0.6]) - [1.0, 0.2])
    )
    np.testing.assert_allclose(uncertainty, np.sqrt(expected_variance))

    with pytest.raises(ValueError, match="compatible neural architecture"):
        _Candidate(
            architecture="extra_trees",
            estimator=Estimator(),
            source_knots=np.asarray([0.0, 1.0]),
            target_knots=np.asarray([0.0, 1.0]),
            reference_od=np.asarray([0.0, 1.0]),
            fingerprint="b" * 64,
            morphology_transfer_weight=0.2,
        )


def test_reused_candidate_refinement_is_distinct_and_evidence_bound(
    tmp_path: Path,
) -> None:
    from histopia.protein._study_workflow import (
        _Candidate,
        _refine_reused_candidate,
        _validate_existing_postfit_refinement,
        _write_candidate_calibration,
    )

    class Estimator:
        def predict_morphology_transfer(
            self,
            features: np.ndarray,
            *,
            neighbors: int,
        ) -> tuple[np.ndarray, np.ndarray]:
            del neighbors
            return np.zeros(len(features)), np.zeros(len(features))

    candidate = _Candidate(
        architecture="dual_bank_attention",
        estimator=Estimator(),
        source_knots=np.asarray([0.0, 1.0]),
        target_knots=np.asarray([0.0, 1.0]),
        reference_od=np.asarray([0.1, 0.5, 0.9], np.float32),
        fingerprint="a" * 64,
    )
    refinement = {
        "method": ("calibrated-neural-plus-training-only-morphology-od-transfer-v1"),
        "weight": 0.2,
        "neighbors": 16,
        "evidence_sha256": "b" * 64,
    }
    refined = _refine_reused_candidate(candidate, refinement)
    sidecar = _write_candidate_calibration(refined, tmp_path / "final")
    payload = json.loads(sidecar.read_text())

    assert refined.estimator is candidate.estimator
    assert refined.fingerprint != candidate.fingerprint
    assert refined.morphology_transfer_weight == pytest.approx(0.2)
    assert refined.morphology_transfer_neighbors == 16
    assert payload["fingerprint"] == refined.fingerprint
    assert payload["morphology_transfer_weight"] == pytest.approx(0.2)
    assert payload["morphology_transfer_neighbors"] == 16
    assert _refine_reused_candidate(refined, refinement) is refined
    _validate_existing_postfit_refinement(refined, refinement)

    with pytest.raises(ValueError, match="refinement controls differ"):
        _validate_existing_postfit_refinement(candidate, refinement)

    conflicting = {**refinement, "weight": 0.3}
    with pytest.raises(ValueError, match="different post-fit refinement"):
        _refine_reused_candidate(refined, conflicting)


def test_expanded_relational_refinement_binds_exact_training_bank(
    tmp_path: Path,
) -> None:
    from histopia.protein._study_workflow import (
        _validate_expanded_morphology_transfer_evidence,
    )

    class Model:
        fingerprint = "a" * 64
        bank_od = np.asarray([0.1, 0.2], dtype=np.float32)

    evidence = {
        "training_cohorts": ["train"],
        "development_policy": {
            "selection_scope": ("expanded-development-before-4714-8567-outcomes-v1"),
            "selection_fingerprint": "b" * 64,
            "selection_sha256": "c" * 64,
        },
        "transfer_bank": {
            "scope": "sealed-relational-training-bank-v1",
            "rows": 2,
            "transfer_model_fingerprint": "a" * 64,
        },
    }
    _validate_expanded_morphology_transfer_evidence(
        evidence,
        source_result={"training_cohorts": ["train"]},
        source_root=tmp_path,
        architecture="dual_bank_attention",
        model=Model(),
    )

    stale = {
        **evidence,
        "transfer_bank": {**evidence["transfer_bank"], "rows": 3},
    }
    with pytest.raises(ValueError, match="relational transfer bank differs"):
        _validate_expanded_morphology_transfer_evidence(
            stale,
            source_result={"training_cohorts": ["train"]},
            source_root=tmp_path,
            architecture="dual_bank_attention",
            model=Model(),
        )


def test_embedded_morphology_transfer_requires_every_sealed_candidate() -> None:
    from histopia.protein._study_workflow import (
        _embedded_transfer_controls_match,
    )

    expected = {
        "models/final.calibration.json": "a" * 64,
        "models/leave-one-mouse-out-4312.calibration.json": "b" * 64,
    }
    calibrations = {
        relative: {
            "schema_version": 1,
            "fingerprint": fingerprint,
            "morphology_transfer_weight": 0.2,
            "morphology_transfer_neighbors": 16,
        }
        for relative, fingerprint in expected.items()
    }

    assert _embedded_transfer_controls_match(
        calibrations,
        expected,
        weight=0.2,
        neighbors=16,
    )
    calibrations["models/leave-one-mouse-out-4312.calibration.json"][
        "morphology_transfer_weight"
    ] = 0.0
    assert not _embedded_transfer_controls_match(
        calibrations,
        expected,
        weight=0.2,
        neighbors=16,
    )


def test_relational_neighbor_batch_bounds_gpu_score_working_set() -> None:
    bank_rows = 571_000
    batch = _bounded_neighbor_batch_size(
        4096,
        bank_rows,
        free_bytes=4 * 1024**3,
    )

    assert batch == (512 * 1024**2) // (bank_rows * 4)
    assert batch < 4096
    assert _bounded_neighbor_batch_size(64, 100) == 64
    assert _bounded_neighbor_batch_size(4096, bank_rows, free_bytes=0) >= 1
    with pytest.raises(ValueError, match="positive"):
        _bounded_neighbor_batch_size(0, bank_rows)
    with pytest.raises(ValueError, match="cannot be negative"):
        _bounded_neighbor_batch_size(1, bank_rows, free_bytes=-1)


def test_training_visible_variant_can_reuse_sealed_models_without_refitting(
    tmp_path: Path,
) -> None:
    from sklearn.ensemble import ExtraTreesRegressor

    from histopia.protein._study_workflow import (
        _Candidate,
        _fingerprint_json,
        _resume_partial_study_candidates,
        _reuse_study_candidates,
        _save_candidate,
    )

    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source_models = source / "models"
    destination_models = destination / "models"
    source_models.mkdir(parents=True)
    destination_models.mkdir(parents=True)
    estimator = ExtraTreesRegressor(n_estimators=2, random_state=4).fit(
        np.asarray([[0.0], [1.0], [2.0], [3.0]]),
        np.asarray([0.1, 0.4, 0.7, 1.0]),
    )
    candidate = _Candidate(
        architecture="extra_trees",
        estimator=estimator,
        source_knots=np.asarray([0.0, 1.0]),
        target_knots=np.asarray([0.0, 1.0]),
        reference_od=np.asarray([0.1, 0.4, 0.7, 1.0], np.float32),
        fingerprint="b" * 64,
    )
    artifacts = []
    for stem in (
        "leave-one-mouse-out-1",
        "leave-one-mouse-out-2",
        "final",
    ):
        artifacts.extend(
            path.relative_to(source).as_posix()
            for path in _save_candidate(candidate, source_models / stem)
        )
    bundle = {
        "schema_version": 1,
        "fold_fingerprints": {
            "1": candidate.fingerprint,
            "2": candidate.fingerprint,
            "3": candidate.fingerprint,
        },
        "final_fingerprint": candidate.fingerprint,
        "training_epochs": None,
    }
    bundle["fingerprint"] = _fingerprint_json(bundle)
    source.joinpath("model_bundle.json").write_text(json.dumps(bundle))
    bindings = {
        mouse: {
            "registration_result_sha256": "a" * 64,
            "cell_result_fingerprint": "b" * 64,
            "stain_result_fingerprint": "c" * 64,
            "semantic_result_fingerprint": "d" * 64,
        }
        for mouse in ("1", "2", "3")
    }

    def expression_table(
        mouse_ids: tuple[str, ...],
        cohort_bindings: dict[str, object],
        *,
        alter_training_feature: bool = False,
    ) -> CellExpressionTable:
        rows = len(mouse_ids) * 2
        features = np.arange(rows * 3, dtype=np.float32).reshape(rows, 3)
        if alter_training_feature:
            features[0, 0] += 0.25
        return CellExpressionTable(
            target_id="yap",
            label_ids=np.tile(np.asarray([1, 2], np.uint32), len(mouse_ids)),
            section_ids=np.asarray(["001", "001"] * len(mouse_ids)),
            mouse_ids=np.repeat(np.asarray(mouse_ids), 2),
            native_xy=np.arange(rows * 2, dtype=np.float64).reshape(rows, 2),
            reference_um_xy=np.arange(rows * 2, dtype=np.float64).reshape(rows, 2),
            features=features,
            measured_od=np.tile(
                np.asarray([0.1, 0.9], np.float32),
                len(mouse_ids),
            ),
            binary_label=np.tile(np.asarray([0, 1], np.int8), len(mouse_ids)),
            measurement_coverage=np.ones(rows, np.float32),
            semantic_region=np.zeros(rows, np.int16),
            semantic_support=np.ones(rows, bool),
            provenance={
                "feature_view": "native-hdab-neutral-cell-multiscale-v3",
                "measurement_view": (
                    "tissue-masked-adaptive-corrected-target-od-4um-v1"
                ),
                "cohort_bindings": cohort_bindings,
            },
        )

    source_table = expression_table(("1", "2", "3"), bindings)
    source_table.save(source / "table.npz")
    source.joinpath("audit.json").write_text("{}")
    slides = []
    for mouse in bindings:
        relative = f"predictions/{mouse}.npz"
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(mouse.encode())
        slides.append(
            {
                "cohort": mouse,
                "section": "001",
                "predictions": relative,
            }
        )
    write_protein_result(
        source,
        {
            "schema_version": 4,
            "model_id": "yap-extra-trees-1-2",
            "model_version": "fixture-v1",
            "target_id": "yap",
            "architecture": "extra_trees",
            "prediction_protocol": "leave-one-mouse-out",
            "training_cohorts": ["1", "2"],
            "cohort_bindings": bindings,
            "measurement_view": ("tissue-masked-adaptive-corrected-target-od-4um-v1"),
            "measurement_statistic": "mean",
            "feature_schema_id": "native-hdab-neutral-cell-multiscale-v3",
            "table_fingerprint": source_table.fingerprint,
            "model": "model_bundle.json",
            "model_artifacts": artifacts,
            "training_table": "table.npz",
            "measurement_audit": "audit.json",
            "slides": slides,
            "model_fingerprint": bundle["fingerprint"],
            "metrics": {"folds": [{"held_out": "1"}, {"held_out": "2"}]},
        },
    )
    table = source_table

    fitted, final, copied, folds, source_fingerprint, epochs = _reuse_study_candidates(
        source,
        architecture="extra_trees",
        table=table,
        available_mice=("1", "2", "3"),
        training_mice=("1", "2"),
        root=destination,
        model_root=destination_models,
        measurement_view=("tissue-masked-adaptive-corrected-target-od-4um-v1"),
        feature_view="native-hdab-neutral-cell-multiscale-v3",
        cohort_bindings=bindings,
    )

    assert set(fitted) == {"1", "2", "3"}
    assert final.fingerprint == candidate.fingerprint
    assert set(copied) == set(artifacts)
    assert len(folds) == 2
    assert len(source_fingerprint) == 64
    assert epochs is None

    expanded_bindings = {
        **bindings,
        "4": {
            "registration_result_sha256": "1" * 64,
            "cell_result_fingerprint": "2" * 64,
            "stain_result_fingerprint": "3" * 64,
            "semantic_result_fingerprint": "4" * 64,
        },
    }
    expanded = expression_table(("1", "2", "3", "4"), expanded_bindings)
    expanded_root = tmp_path / "expanded"
    expanded_models = expanded_root / "models"
    expanded_models.mkdir(parents=True)
    expanded_fit = _reuse_study_candidates(
        source,
        architecture="extra_trees",
        table=expanded,
        available_mice=("1", "2", "3", "4"),
        training_mice=("1", "2"),
        root=expanded_root,
        model_root=expanded_models,
        measurement_view=("tissue-masked-adaptive-corrected-target-od-4um-v1"),
        feature_view="native-hdab-neutral-cell-multiscale-v3",
        cohort_bindings=expanded_bindings,
    )
    assert set(expanded_fit[0]) == {"1", "2", "3", "4"}

    altered = expression_table(
        ("1", "2", "3", "4"),
        expanded_bindings,
        alter_training_feature=True,
    )
    with pytest.raises(ValueError, match="training rows differ"):
        _reuse_study_candidates(
            source,
            architecture="extra_trees",
            table=altered,
            available_mice=("1", "2", "3", "4"),
            training_mice=("1", "2"),
            root=tmp_path / "altered",
            model_root=tmp_path / "altered" / "models",
            measurement_view=("tissue-masked-adaptive-corrected-target-od-4um-v1"),
            feature_view="native-hdab-neutral-cell-multiscale-v3",
            cohort_bindings=expanded_bindings,
        )

    changed_training_binding = {
        **expanded_bindings,
        "1": {**expanded_bindings["1"], "cell_result_fingerprint": "9" * 64},
    }
    with pytest.raises(ValueError, match="training-cohort bindings differ"):
        _reuse_study_candidates(
            source,
            architecture="extra_trees",
            table=expanded,
            available_mice=("1", "2", "3", "4"),
            training_mice=("1", "2"),
            root=tmp_path / "changed-binding",
            model_root=tmp_path / "changed-binding" / "models",
            measurement_view=("tissue-masked-adaptive-corrected-target-od-4um-v1"),
            feature_view="native-hdab-neutral-cell-multiscale-v3",
            cohort_bindings=changed_training_binding,
        )
    resume_root = tmp_path / "resume"
    resume_models = resume_root / "models"
    resume_models.mkdir(parents=True)
    _save_candidate(candidate, resume_models / "final")
    resume_bundle = {
        "schema_version": 1,
        "target_id": "yap",
        "architecture": "extra_trees",
        "prediction_protocol": "training-visible",
        "training_cohorts": ["1"],
        "table_fingerprint": source_table.fingerprint,
        "feature_schema_id": "native-hdab-neutral-cell-multiscale-v3",
        "fold_fingerprints": {"1": candidate.fingerprint},
        "final_fingerprint": candidate.fingerprint,
        "measurement_view": ("tissue-masked-adaptive-corrected-target-od-4um-v1"),
        "measurement_compartment": "whole_cell",
        "measurement_statistic": "mean",
        "training_epochs": None,
    }
    resume_bundle["fingerprint"] = _fingerprint_json(resume_bundle)
    resume_root.joinpath("model_bundle.json").write_text(json.dumps(resume_bundle))
    resumed = _resume_partial_study_candidates(
        "extra_trees",
        SimpleNamespace(
            target=SimpleNamespace(
                compartment="whole_cell",
                measurement_statistic="mean",
            )
        ),
        SimpleNamespace(target_id="yap", fingerprint=source_table.fingerprint),
        all_training=np.asarray([0]),
        available_mice=("1",),
        training_mice=("1",),
        mouse_values=np.asarray(["1"]),
        root=resume_root,
        model_root=resume_models,
        device="cpu",
        epochs=None,
        prediction_protocol="training-visible",
        measurement_view=("tissue-masked-adaptive-corrected-target-od-4um-v1"),
        feature_view="native-hdab-neutral-cell-multiscale-v3",
    )
    assert resumed is not None
    assert resumed[1].fingerprint == candidate.fingerprint
    assert resumed[3] == []


def test_shared_multitask_reuse_preserves_training_only_auxiliary_lineage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from histopia.protein import _manifest
    from histopia.protein._study_workflow import _resolve_shared_auxiliary_scope

    primary = SimpleNamespace(target_id="ck18")
    expected = {"erp44": "a" * 64, "gfp": "b" * 64}
    monkeypatch.setattr(
        _manifest,
        "validate_protein_result_index",
        lambda _root: {"auxiliary_table_fingerprints": expected},
    )

    tables, bindings = _resolve_shared_auxiliary_scope(
        (),
        primary=primary,
        architecture="shared_multitask",
        reuse_models_from="frozen-source",
    )

    assert tables == ()
    assert bindings == expected


def test_shared_multitask_reuse_rejects_malformed_auxiliary_lineage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from histopia.protein import _manifest
    from histopia.protein._study_workflow import _resolve_shared_auxiliary_scope

    monkeypatch.setattr(
        _manifest,
        "validate_protein_result_index",
        lambda _root: {"auxiliary_table_fingerprints": {"ck18": "short"}},
    )
    with pytest.raises(ValueError, match="auxiliary lineage"):
        _resolve_shared_auxiliary_scope(
            (),
            primary=SimpleNamespace(target_id="ck18"),
            architecture="shared_multitask",
            reuse_models_from="frozen-source",
        )


def test_sklearn_model_reuse_fails_closed_on_runtime_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import joblib
    from sklearn.exceptions import InconsistentVersionWarning

    from histopia.protein._study_workflow import _load_sklearn_estimator

    def incompatible_load(_path: Path) -> object:
        import warnings

        warnings.warn(
            InconsistentVersionWarning(
                estimator_name="ExtraTreesRegressor",
                current_sklearn_version="1.7.2",
                original_sklearn_version="1.9.0",
            ),
            stacklevel=2,
        )
        return object()

    monkeypatch.setattr(joblib, "load", incompatible_load)
    with pytest.raises(ValueError, match="1.9.0 required, 1.7.2 active"):
        _load_sklearn_estimator(tmp_path / "model.joblib")


def test_study_section_sampling_seed_is_cohort_order_independent() -> None:
    from histopia.protein._study_workflow import _study_section_sampling_seed

    first = _study_section_sampling_seed(17, "yap", "4630", "012")
    repeated = _study_section_sampling_seed(17, "yap", "4630", "012")

    assert first == repeated
    assert first != _study_section_sampling_seed(17, "yap", "4314", "012")
    assert first != _study_section_sampling_seed(17, "ecad", "4630", "012")
    assert first != _study_section_sampling_seed(18, "yap", "4630", "012")


def test_sweep_manifest_claims_unique_fingerprinted_tasks(
    tmp_path: Path,
) -> None:
    from histopia.protein import create_protein_sweep, protein_sweep_status
    from histopia.protein._sweep import _claim_task

    entries = []
    for target in ("yap", "ki67"):
        target_root = tmp_path / target
        target_root.mkdir()
        config = target_root / "config.json"
        config.write_text(
            json.dumps(
                {
                    "output_dir": str(target_root / "output"),
                    "target": {"target_id": target},
                }
            )
        )
        study = target_root / "study.json"
        study.write_text('{"schema_version":1}\n')
        table = CellExpressionTable(
            target_id=target,
            label_ids=np.arange(1, 13, dtype=np.uint32),
            section_ids=np.asarray(["001"] * 6 + ["003"] * 6),
            mouse_ids=np.asarray(["a"] * 6 + ["b"] * 6),
            native_xy=np.column_stack((np.arange(12), np.arange(12))),
            reference_um_xy=np.column_stack((np.arange(12), np.arange(12))),
            features=np.arange(36, dtype=np.float32).reshape(12, 3),
            measured_od=np.linspace(0, 1, 12, dtype=np.float32),
            binary_label=np.asarray([0] * 6 + [1] * 6, dtype=np.int8),
            measurement_coverage=np.ones(12, dtype=np.float32),
            semantic_region=np.zeros(12, dtype=np.int16),
            semantic_support=np.ones(12, dtype=bool),
            provenance={"feature_view": "native-hdab-neutral-spatial-uni2h-v2"},
        ).save(target_root / "table.npz")
        geometry = target_root / "geometry"
        geometry.mkdir()
        entries.append(
            {
                "target_id": target,
                "config": str(config),
                "study": str(study),
                "table": str(table),
                "geometry_cache": str(geometry),
            }
        )
    specification = tmp_path / "spec.json"
    specification.write_text(json.dumps({"schema_version": 1, "entries": entries}))
    manifest = create_protein_sweep(
        specification,
        tmp_path / "sweep",
        architectures=("extra_trees", "multi_tower"),
        maximum_training_cells=64,
        maximum_evaluation_cells=64,
    )

    first = _claim_task(manifest, "worker-a", {"extra_trees", "multi_tower"})
    second = _claim_task(manifest, "worker-b", {"extra_trees", "multi_tower"})
    payload = json.loads(manifest.read_text())

    assert first is not None and second is not None
    assert first["id"] != second["id"]
    assert len({task["fingerprint"] for task in payload["tasks"]}) == 4
    assert protein_sweep_status(manifest)["running"] == 2
    assert "config" not in protein_sweep_status(manifest)


def test_disjoint_sweep_partitions_merge_without_duplicate_tasks(
    tmp_path: Path,
) -> None:
    from histopia.protein import (
        create_protein_sweep,
        select_protein_sweep_winners,
    )

    entries = []
    for target in ("yap", "ki67"):
        target_root = tmp_path / target
        target_root.mkdir()
        config = target_root / "config.json"
        config.write_text(
            json.dumps(
                {
                    "output_dir": str(target_root / "output"),
                    "target": {"target_id": target},
                }
            )
        )
        study = target_root / "study.json"
        study.write_text('{"schema_version":1}\n')
        table = CellExpressionTable(
            target_id=target,
            label_ids=np.arange(1, 13, dtype=np.uint32),
            section_ids=np.asarray(["001"] * 12),
            mouse_ids=np.asarray(["a"] * 6 + ["b"] * 6),
            native_xy=np.zeros((12, 2), dtype=np.float64),
            reference_um_xy=np.zeros((12, 2), dtype=np.float64),
            features=np.arange(36, dtype=np.float32).reshape(12, 3),
            measured_od=np.linspace(0, 1, 12, dtype=np.float32),
            binary_label=np.full(12, -1, dtype=np.int8),
            measurement_coverage=np.ones(12, dtype=np.float32),
            semantic_region=np.zeros(12, dtype=np.int16),
            semantic_support=np.ones(12, dtype=bool),
            provenance={"feature_view": "native-hdab-neutral-spatial-uni2h-v2"},
        ).save(target_root / "table.npz")
        geometry = target_root / "geometry"
        geometry.mkdir()
        entries.append(
            {
                "target_id": target,
                "config": str(config),
                "study": str(study),
                "table": str(table),
                "geometry_cache": str(geometry),
            }
        )
    specification = tmp_path / "spec.json"
    specification.write_text(json.dumps({"schema_version": 1, "entries": entries}))
    manifests = tuple(
        create_protein_sweep(
            specification,
            tmp_path / architecture,
            architectures=(architecture,),
            maximum_training_cells=64,
            maximum_evaluation_cells=64,
        )
        for architecture in ("extra_trees", "multi_tower")
    )
    for rank, manifest in enumerate(manifests, start=1):
        payload = json.loads(manifest.read_text())
        for task in payload["tasks"]:
            result = {
                "schema_version": 1,
                "target_id": task["target_id"],
                "architecture": task["architecture"],
                "feature_schema_id": task["feature_schema_id"],
                "table_fingerprint": task["table_fingerprint"],
                "task_fingerprint": task["fingerprint"],
                "transfer": {
                    "available": True,
                    "spearman_64um": rank / 10,
                    "spearman": rank / 10,
                    "mean_bias_fraction": 0.0,
                    "mae": 1 / rank,
                    "mae_improvement": rank / 10,
                    "folds_beating_semantic_fraction": 1.0,
                    "folds": [
                        {
                            "held_out": "a",
                            "spearman_64um": rank / 10,
                            "mae_improvement": rank / 10,
                            "mean_bias_fraction": 0.0,
                            "beats_semantic_baseline": True,
                        }
                    ],
                },
                "measured_fit": {
                    "available": True,
                    "spearman_64um": rank / 10,
                    "spearman": rank / 10,
                    "mean_bias_fraction": 0.0,
                    "mae": 1 / rank,
                },
            }
            relative = f"results/{task['id']}.json"
            (manifest.parent / relative).write_text(json.dumps(result))
            task.update(state="completed", result=relative)
        manifest.write_text(json.dumps(payload))

    winners = select_protein_sweep_winners(
        manifests,
        tmp_path / "winners.json",
    )
    payload = json.loads(winners.read_text())
    selected = payload["targets"]

    assert payload["selection"] == "guardrailed-transfer-and-measured-fit-v2"
    assert {row["target_id"] for row in selected} == {"yap", "ki67"}
    assert {row["transfer_architecture"] for row in selected} == {"multi_tower"}
    assert all(
        row["transfer_eligible_for_protected_evaluation"] is True for row in selected
    )


def test_study_feature_preparation_can_partition_cohorts_and_sections(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from histopia.protein import prepare_real_protein_study_features
    from histopia.protein._study_workflow import RealProteinStudyCohort

    contexts = {
        mouse: SimpleNamespace(
            cell_by_section={section: {} for section in ("001", "002")}
        )
        for mouse in ("a", "b")
    }
    seen = []

    def section_design(context, section, _config, **kwargs):
        seen.append((context, section, kwargs["include_semantic"]))
        return SimpleNamespace(
            label_ids=np.arange(3, dtype=np.uint32),
            features=np.ones((3, 1844), dtype=np.float32),
        )

    verification = []

    def load_contexts(_config, selected, *, verify_artifacts=True):
        verification.append((verify_artifacts, tuple(sorted(selected))))
        return {mouse: contexts[mouse] for mouse in selected}

    monkeypatch.setattr(
        "histopia.protein._study_workflow._contexts",
        load_contexts,
    )
    monkeypatch.setattr(
        "histopia.protein._study_workflow._section_design", section_design
    )
    source = RealProteinStudyCohort(*(tmp_path for _ in range(5)))

    result = prepare_real_protein_study_features(
        SimpleNamespace(feature_schema_id="native-hdab-neutral-cell-multiscale-v3"),
        {"a": source, "b": source},
        geometry_cache=tmp_path,
        cohort_ids=("b",),
        sections=("002",),
    )

    assert result["section_count"] == 1
    assert result["cell_count"] == 3
    assert seen == [(contexts["b"], "002", False)]
    assert verification == [(False, ("b",))]


def test_registered_od_calibration_recovers_repeated_stain_shift() -> None:
    rng = np.random.default_rng(17)
    grid = np.asarray([(x, y) for y in range(16) for x in range(16)], dtype=float)
    xy = np.repeat(grid * 64.0 + 1.0, 8, axis=0)
    base = np.repeat(np.linspace(0.1, 1.8, len(grid)), 8)
    sections = np.asarray(["003"] * len(xy) + ["013"] * len(xy))
    coordinates = np.vstack((xy, xy + rng.normal(0, 0.1, xy.shape)))
    measured = np.r_[base, base / 2.0]
    calibration = fit_registered_od_calibration(
        sections,
        coordinates,
        measured,
        np.zeros(len(sections), dtype=np.int16),
        np.ones(len(sections), dtype=bool),
        minimum_anchors=128,
    )
    light = calibration.for_section("013")
    assert light.status == "accepted"
    assert 1.8 < light.scale < 2.2
    transformed = light.apply(np.asarray([0.0, 0.25, 0.5]))
    assert transformed[0] == 0
    assert np.all(np.diff(transformed) >= 0)
    np.testing.assert_allclose(transformed[1:], [0.5, 1.0], atol=0.08)


def test_registered_od_calibration_falls_back_to_identity() -> None:
    calibration = fit_registered_od_calibration(
        np.asarray(["003"] * 4 + ["013"] * 4),
        np.asarray([[0, 0], [1, 1], [2, 2], [3, 3]] * 2, dtype=float),
        np.asarray([0.1, 0.2, 0.3, 0.4] * 2),
        np.zeros(8, dtype=np.int16),
        np.ones(8, dtype=bool),
        minimum_cells_per_bin=1,
        minimum_anchors=8,
    )
    row = calibration.for_section("003")
    assert row.status == "identity"
    np.testing.assert_allclose(row.apply(np.asarray([0.0, 0.5])), [0.0, 0.5])


def test_equal_section_harmonization_is_cohort_aware_and_monotonic() -> None:
    base = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    cohorts = np.asarray(["4312"] * 512 + ["4630"] * 256)
    sections = np.asarray(["003"] * 256 + ["013"] * 256 + ["003"] * 256)
    measured = np.concatenate((base * 2.0, base * 0.5, base))
    harmonization = fit_equal_section_od_harmonization(
        cohorts,
        sections,
        measured,
        np.ones(len(measured), dtype=bool),
        minimum_cells=128,
    )

    dark = harmonization.for_section("4312", "003")
    light = harmonization.for_section("4312", "013")
    other_mouse = harmonization.for_section("4630", "003")
    assert dark.scale < other_mouse.scale < light.scale
    assert dark.apply(np.asarray([0.0]))[0] == 0
    transformed = [
        row.apply(values)
        for row, values in (
            (dark, base * 2.0),
            (light, base * 0.5),
            (other_mouse, base),
        )
    ]
    assert all(np.all(np.diff(values) >= 0) for values in transformed)
    np.testing.assert_allclose(
        [np.quantile(values, 0.5) for values in transformed],
        [0.5, 0.5, 0.5],
        atol=0.02,
    )
    payload = harmonization.as_dict()
    assert payload["method"] == "equal-section-adaptive-quantile-v2"
    assert len(payload["fingerprint"]) == 64
    assert len({(row["cohort"], row["section"]) for row in payload["sections"]}) == 3


def test_equal_section_harmonization_preserves_support_and_nan() -> None:
    values = np.r_[np.linspace(0.0, 1.0, 128), np.linspace(0.0, 2.0, 128)]
    harmonization = fit_equal_section_od_harmonization(
        np.asarray(["a"] * 128 + ["b"] * 128),
        np.asarray(["001"] * 256),
        values,
        np.ones(256, dtype=bool),
        minimum_cells=128,
    )
    transformed = harmonization.for_section("a", "001").apply(
        np.asarray([np.nan, -1.0, 0.0, 0.5])
    )
    assert np.isnan(transformed[:2]).all()
    assert transformed[2] == 0
    assert transformed[3] > 0


def test_equal_section_harmonization_freezes_reference_to_training_mice() -> None:
    base = np.linspace(0.0, 1.0, 128, dtype=np.float32)
    cohorts = np.asarray(["train"] * 256 + ["holdout"] * 128)
    sections = np.asarray(["001"] * 128 + ["002"] * 128 + ["001"] * 128)
    measured = np.concatenate((base, base * 2.0, base * 20.0))

    frozen = fit_equal_section_od_harmonization(
        cohorts,
        sections,
        measured,
        np.ones(len(measured), dtype=bool),
        minimum_cells=128,
        reference_cohorts=("train",),
    )
    training_only = fit_equal_section_od_harmonization(
        cohorts[:256],
        sections[:256],
        measured[:256],
        np.ones(256, dtype=bool),
        minimum_cells=128,
    )

    np.testing.assert_allclose(
        frozen.reference_quantiles,
        training_only.reference_quantiles,
    )
    assert frozen.reference_cohorts == ("train",)
    assert frozen.as_dict()["reference_cohorts"] == ["train"]
    assert frozen.for_section("holdout", "001").status == "accepted"


def test_external_holdout_harmonization_accepts_training_subset_only() -> None:
    from histopia.protein import _study_workflow

    assert _study_workflow._training_only_harmonization_references(
        ["mouse-b", "mouse-a"],
        ("mouse-a", "mouse-b", "mouse-c"),
    ) == ("mouse-a", "mouse-b")
    for invalid in ([], ["mouse-a", "mouse-a"], ["mouse-a", "holdout"]):
        with pytest.raises(ValueError, match="selected training cohorts"):
            _study_workflow._training_only_harmonization_references(
                invalid,
                ("mouse-a", "mouse-b", "mouse-c"),
            )


def test_external_holdout_metrics_are_separate_from_model_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from histopia.protein import _study_workflow

    table = _table()
    training = np.flatnonzero(table.mouse_ids == "mouse-a")

    def predict(
        candidate: object,
        selected_table: object,
        train: np.ndarray,
        test: np.ndarray,
        config: object,
        *,
        device: str,
    ) -> np.ndarray:
        del candidate, train, config, device
        return np.asarray(selected_table.measured_od)[test]

    monkeypatch.setattr(_study_workflow, "_predict_table_candidate", predict)
    metrics = _study_workflow._evaluate_external_holdouts(
        SimpleNamespace(),
        architecture="extra_trees",
        config=SimpleNamespace(),
        table=table,
        training=training,
        holdout_mice=("mouse-b",),
        device="cpu",
    )

    assert metrics["evaluation"] == "frozen_external_mouse_holdout"
    assert metrics["training_cohorts"] == ["mouse-a"]
    assert metrics["holdout_cohorts"] == ["mouse-b"]
    assert metrics["model_selection_eligible"] is False
    assert metrics["spearman"] == pytest.approx(1.0)


def test_promotion_gate_requires_accuracy_baseline_parity_and_bias() -> None:
    accepted = evaluate_protein_promotion(
        {
            "spearman": 0.44,
            "spearman_64um": 0.57,
            "mae": 0.532,
            "mean_bias_fraction": 0.04,
            "median_bias_fraction": -0.08,
            "folds": [
                {
                    "held_out": "003",
                    "mae_improvement": -0.002,
                    "mean_bias_fraction": 0.04,
                    "median_bias_fraction": -0.08,
                },
                {
                    "held_out": "013",
                    "mae_improvement": 0.03,
                    "mean_bias_fraction": -0.02,
                    "median_bias_fraction": 0.06,
                },
            ],
        }
    )
    assert accepted.accepted
    rejected = evaluate_protein_promotion(
        {
            "spearman": 0.44,
            "spearman_64um": 0.57,
            "mae": 0.532,
            "mean_bias_fraction": 0.18,
            "median_bias_fraction": 0.02,
            "folds": [{"held_out": "003", "mae_improvement": -0.02}],
        }
    )
    assert not rejected.accepted
    assert any("mean_bias" in reason for reason in rejected.reasons)
    assert any("003" in reason for reason in rejected.reasons)

    masked_fold_bias = evaluate_protein_promotion(
        {
            "spearman": 0.6,
            "spearman_64um": 0.7,
            "mae": 0.2,
            "mean_bias_fraction": 0.01,
            "median_bias_fraction": 0.02,
            "folds": [
                {
                    "held_out": "6180",
                    "mae_improvement": 0.2,
                    "mean_bias_fraction": 0.24,
                    "median_bias_fraction": 0.29,
                }
            ],
        }
    )
    assert not masked_fold_bias.accepted
    assert any("fold 6180" in reason for reason in masked_fold_bias.reasons)


def test_architecture_selection_uses_identical_mouse_folds_and_spatial_rank() -> None:
    def metrics(rows):
        return {
            "evaluation": "leave_one_mouse_out",
            "folds": [
                {
                    "held_out": held_out,
                    "spearman_64um": spearman,
                    "mae_improvement": improvement,
                    "mean_bias_fraction": bias,
                    "beats_semantic_baseline": beats_baseline,
                }
                for held_out, spearman, improvement, bias, beats_baseline in rows
            ],
        }

    selection = select_protein_architecture_candidate(
        {
            "graph_transformer": metrics(
                [
                    ("mouse-a", 0.70, 0.10, 0.10, True),
                    ("mouse-b", 0.65, 0.05, -0.20, True),
                    ("mouse-c", 0.60, 0.02, 0.15, True),
                ]
            ),
            "multi_tower": metrics(
                [
                    ("mouse-a", 0.80, 0.30, 0.05, True),
                    ("mouse-b", 0.55, 0.30, 0.05, True),
                    ("mouse-c", 0.54, 0.30, 0.05, True),
                ]
            ),
            "dual_bank_attention": metrics(
                [
                    ("mouse-a", 0.90, 0.20, 0.05, True),
                    ("mouse-b", 0.85, -0.01, 0.05, False),
                    ("mouse-c", 0.80, 0.20, 0.05, True),
                ]
            ),
        },
        expected_holdouts=("mouse-a", "mouse-b", "mouse-c"),
    )

    assert selection.selected == "graph_transformer"
    scores = {score.candidate: score for score in selection.candidates}
    assert scores["graph_transformer"].eligible
    assert scores["graph_transformer"].median_spearman_64um == pytest.approx(0.65)
    assert not scores["dual_bank_attention"].eligible
    assert any(
        "mouse-b" in reason and "semantic baseline" in reason
        for reason in scores["dual_bank_attention"].reasons
    )


def test_architecture_selection_fails_closed_on_incomplete_or_biased_folds() -> None:
    selection = select_protein_architecture_candidate(
        {
            "incomplete": {
                "evaluation": "leave_one_mouse_out",
                "folds": [
                    {
                        "held_out": "mouse-a",
                        "spearman_64um": 0.9,
                        "mae_improvement": 0.2,
                        "mean_bias_fraction": 0.0,
                        "beats_semantic_baseline": True,
                    }
                ],
            },
            "biased": {
                "evaluation": "leave_one_mouse_out",
                "folds": [
                    {
                        "held_out": mouse,
                        "spearman_64um": 0.8,
                        "mae_improvement": 0.2,
                        "mean_bias_fraction": 0.26 if mouse == "mouse-b" else 0.0,
                        "beats_semantic_baseline": True,
                    }
                    for mouse in ("mouse-a", "mouse-b")
                ],
            },
        },
        expected_holdouts=("mouse-a", "mouse-b"),
    )

    assert selection.selected is None
    scores = {score.candidate: score for score in selection.candidates}
    assert any("coverage differs" in reason for reason in scores["incomplete"].reasons)
    assert any("mean bias" in reason for reason in scores["biased"].reasons)
    with pytest.raises(ValueError, match="unique"):
        select_protein_architecture_candidate(
            {"candidate": {}},
            expected_holdouts=("mouse-a", "mouse-a"),
        )
    with pytest.raises(ValueError, match="sequence"):
        select_protein_architecture_candidate(
            {"candidate": {}},
            expected_holdouts="mouse-a",
        )
    with pytest.raises(ValueError, match="candidate identifiers"):
        select_protein_architecture_candidate(
            {1: {}},  # type: ignore[dict-item]
            expected_holdouts=("mouse-a",),
        )


def test_region_centered_blend_preserves_within_region_order() -> None:
    predicted = np.asarray([0.1, 0.4, 1.0, 1.4])
    semantic = np.asarray([0.8, 0.8, 1.8, 1.8])
    regions = np.asarray([0, 0, 1, 1])
    result = _region_centered_blend(predicted, semantic, regions, weight=0.7)
    assert result[1] > result[0]
    assert result[3] > result[2]
    np.testing.assert_allclose(
        [np.median(result[:2]), np.median(result[2:])], [0.8, 1.8]
    )


def test_deep_oof_predictions_hold_out_each_section(monkeypatch) -> None:
    seen: list[tuple[set[int], set[int]]] = []

    def fake_fit(_architecture, _features, _od, train, test, **_kwargs):
        seen.append((set(train.tolist()), set(test.tolist())))
        return np.asarray(test, dtype=np.float32) + 1

    monkeypatch.setattr("histopia.protein._deep.fit_deep_candidate", fake_fit)
    result = out_of_fold_deep_predictions(
        "cross_attention",
        np.ones((6, 2), dtype=np.float32),
        np.arange(6, dtype=np.float32),
        np.asarray(["003"] * 3 + ["013"] * 3),
    )
    np.testing.assert_array_equal(result, np.arange(6, dtype=np.float32) + 1)
    assert all(train.isdisjoint(test) for train, test in seen)


def test_portable_cross_attention_round_trip_requires_no_torch(tmp_path: Path) -> None:
    rng = np.random.default_rng(8)
    model = PortableCrossAttentionRegressor(
        feature_mean=np.zeros(2, np.float32),
        feature_scale=np.ones(2, np.float32),
        bank_embedding=rng.normal(size=(5, 4)).astype(np.float32),
        bank_od=np.linspace(0.1, 0.9, 5, dtype=np.float32),
        embed_weight=rng.normal(size=(4, 2)).astype(np.float32),
        embed_bias=np.zeros(4, np.float32),
        attention_weight=rng.normal(size=(12, 4)).astype(np.float32),
        attention_bias=np.zeros(12, np.float32),
        attention_output_weight=rng.normal(size=(4, 4)).astype(np.float32),
        attention_output_bias=np.zeros(4, np.float32),
        head0_weight=rng.normal(size=(4, 8)).astype(np.float32),
        head0_bias=np.zeros(4, np.float32),
        head1_weight=rng.normal(size=(1, 4)).astype(np.float32),
        head1_bias=np.zeros(1, np.float32),
        heads=2,
        provenance={"fixture": True},
    )
    expected = model.predict(np.asarray([[0.2, -0.1]], dtype=np.float32))
    path = model.save(tmp_path / "attention.npz")
    loaded = PortableCrossAttentionRegressor.load(path)
    actual = loaded.predict(np.asarray([[0.2, -0.1]], dtype=np.float32))
    accelerated = loaded.predict_accelerated(
        np.asarray([[0.2, -0.1]], dtype=np.float32), device="cpu"
    )
    assert loaded.fingerprint == model.fingerprint
    np.testing.assert_allclose(actual[0], expected[0])
    np.testing.assert_allclose(actual[1], expected[1])
    np.testing.assert_allclose(accelerated[0], expected[0], atol=1e-5)
    np.testing.assert_allclose(accelerated[1], expected[1], atol=1e-5)


def test_dense_tokens_give_distinct_features_to_cells_in_one_crop(
    tmp_path: Path,
) -> None:
    class Encoder:
        model_fingerprint = "uni2h-test"

        def encode_tokens(self, images: np.ndarray) -> np.ndarray:
            assert images.shape == (1, 224, 224, 3)
            yy, xx = np.mgrid[:16, :16]
            grid = np.stack((xx, yy), axis=-1).astype(np.float32)
            return grid[None]

    artifact = extract_cell_token_features(
        slide_id="mouse-section",
        content_bbox_native_xywh=(0, 0, 224, 224),
        source_mpp_xy=(0.5, 0.5),
        tissue_mask=np.ones((224, 224), dtype=bool),
        native_to_mask=np.eye(3),
        label_ids=np.asarray([1, 2], dtype=np.uint32),
        cell_native_xy=np.asarray([[56.0, 112.0], [168.0, 112.0]]),
        reader=lambda *_request: np.zeros((224, 224, 3), dtype=np.uint8),
        neutralizer=lambda image, _crop: image,
        encoder=Encoder(),
        batch_size=1,
    )

    assert artifact.features.shape == (2, 2)
    assert np.all(artifact.supported)
    assert artifact.features[0, 0] < artifact.features[1, 0]
    path = artifact.save(tmp_path / "tokens.npz")
    loaded = CellTokenFeatures.load(path)
    assert loaded.fingerprint == artifact.fingerprint
    np.testing.assert_allclose(loaded.features, artifact.features, rtol=1e-3)


def test_cell_token_feature_index_validates_exact_cache_binding(
    tmp_path: Path,
) -> None:
    artifact = CellTokenFeatures(
        slide_id="mouse-section",
        label_ids=np.asarray([1], dtype=np.uint32),
        native_xy=np.asarray([[10.0, 20.0]], dtype=np.float64),
        features=np.asarray([[0.25, -0.5]], dtype=np.float32),
        support_weight=np.asarray([1.0], dtype=np.float32),
        analysis_mpp=0.5,
        token_spacing_um=7.0,
        provenance={
            "source_identity": "source-v1",
            "mask_sha256": "mask-v1",
            "cell_result_fingerprint": "cells-v1",
            "feature_view": "native-hdab-neutral-spatial-uni2h-v2",
            "model_fingerprint": "uni2h-v1",
            "view": "native-stain-neutral-uni2h-spatial-tokens-v1",
            "crop_size_px": 224,
            "crop_stride_px": 112,
            "tissue_masked": True,
        },
    )
    path = artifact.save(tmp_path / "tokens.npz")

    index = cell_token_feature_index(path)
    assert index.fingerprint == artifact.fingerprint
    assert index.slide_id == "mouse-section"
    validate_cell_token_feature_binding(
        path,
        slide_id="mouse-section",
        cell_result_fingerprint="cells-v1",
        source_identity="source-v1",
        mask_sha256="mask-v1",
    )
    with pytest.raises(ValueError, match="cell_result_fingerprint"):
        validate_cell_token_feature_binding(
            path,
            slide_id="mouse-section",
            cell_result_fingerprint="cells-v2",
            source_identity="source-v1",
            mask_sha256="mask-v1",
        )


def test_hdab_neutral_view_removes_target_component() -> None:
    basis = np.asarray(
        [[0.650, 0.268], [0.704, 0.570], [0.286, 0.776]], dtype=np.float32
    )
    hematoxylin = np.linspace(0, 0.8, 16, dtype=np.float32).reshape(4, 4)

    def render(dab: float) -> np.ndarray:
        concentrations = np.column_stack(
            (hematoxylin.ravel(), np.full(hematoxylin.size, dab))
        )
        od = concentrations @ basis.T
        return (
            np.clip(np.rint(256 * np.exp(-od) - 1), 0, 255)
            .astype(np.uint8)
            .reshape(4, 4, 3)
        )

    low = neutralize_hdab_morphology(render(0.1), (0, 0, 4, 4))
    high = neutralize_hdab_morphology(render(0.9), (0, 0, 4, 4))
    np.testing.assert_allclose(low, high, atol=3)


def test_cell_shape_and_neighborhood_features_are_cell_resolved() -> None:
    labels = np.zeros((12, 12), dtype=np.uint32)
    labels[1:3, 1:3] = 1
    labels[5:10, 6:10] = 2
    ids, shape = cell_shape_features(labels, pixel_size_um_xy=(0.5, 0.5))
    neighborhood = cell_neighborhood_features(
        np.asarray([[1.5, 1.5], [7.5, 7.0]]),
        shape,
        neighbors=1,
    )

    assert ids.tolist() == [1, 2]
    assert shape.shape == (2, 5)
    assert shape[1, 0] > shape[0, 0]
    assert neighborhood.shape == (2, 8)
    assert np.all(np.isfinite(neighborhood))


def test_spatial_graph_uses_registered_xyz_neighbors() -> None:
    graph = _spatial_neighbor_index(
        np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [100.0, 0.0, 5.0]]),
        neighbors=1,
    )

    assert graph.shape == (3, 1)
    assert graph[:, 0].tolist() == [1, 0, 1]


def test_native_label_centroids_stream_exact_ids(tmp_path: Path) -> None:
    pyvips = pytest.importorskip("pyvips")
    labels = np.zeros((6, 8), dtype=np.uint32)
    labels[1:3, 2:4] = 3
    labels[4:6, 6:8] = 9
    image = pyvips.Image.new_from_memory(
        labels.tobytes(), labels.shape[1], labels.shape[0], 1, "uint"
    )
    path = tmp_path / "labels.tiff"
    image.tiffsave(str(path), tile=True, compression="lzw")

    ids, xy, areas = label_centroids_from_tiff(path, stripe_height=2)

    assert ids.tolist() == [3, 9]
    np.testing.assert_allclose(xy, [[2.5, 1.5], [6.5, 4.5]])
    assert areas.tolist() == [4, 4]


def test_vips_cell_token_reader_batches_neighboring_crops(tmp_path: Path) -> None:
    pyvips = pytest.importorskip("pyvips")
    rgb = np.zeros((8, 16, 3), dtype=np.uint8)
    rgb[:, :8, 0] = 40
    rgb[:, 8:, 0] = 180
    image = pyvips.Image.new_from_memory(rgb.tobytes(), 16, 8, 3, "uchar")
    path = tmp_path / "source.tiff"
    image.tiffsave(str(path))

    patches = VipsPatchReader(path).read_many(((0, 0, 8, 8, 8), (8, 0, 8, 8, 8)))

    assert len(patches) == 2
    assert patches[0].shape == (8, 8, 3)
    assert int(patches[0][0, 0, 0]) == 40
    assert int(patches[1][0, 0, 0]) == 180


def test_target_aliases_and_config_are_explicit(tmp_path: Path) -> None:
    assert normalize_target_id("E-Cad") == "ecad"
    assert normalize_target_id("Ecad") == "ecad"
    target = ProteinTarget(
        "E-Cad",
        aliases=("Ecad",),
        assay_domain="pooled-v1",
        compartment="inner_boundary",
        measurement_statistic="q90",
        binary_enabled=False,
    )
    config = ProteinPredictionConfig(
        output_dir=tmp_path,
        target=target,
        od_harmonization="none",
        relational_morphology_transfer_weight=0.2,
        relational_morphology_transfer_neighbors=16,
        relational_loss_profile="tail_rank_v1",
    )
    assert config.target.target_id == "ecad"
    assert config.target.display_name == "E-Cad"
    assert config.target.measurement_statistic == "q90"
    assert config.od_harmonization == "none"
    assert config.relational_morphology_transfer_weight == pytest.approx(0.2)
    assert config.relational_morphology_transfer_neighbors == 16
    assert config.relational_loss_profile == "tail_rank_v1"
    clone_specific = ProteinTarget("MYC(CST)")
    assert clone_specific.target_id == "myccst"
    assert clone_specific.display_name == "MYC(CST)"
    with pytest.raises(ValueError, match="normalize"):
        ProteinTarget("ecad", aliases=("pERK",))
    with pytest.raises(ValueError, match="morphology_transfer_weight"):
        ProteinPredictionConfig(
            output_dir=tmp_path,
            target=target,
            relational_morphology_transfer_weight=1.1,
        )
    with pytest.raises(ValueError, match="relational_loss_profile"):
        ProteinPredictionConfig(
            output_dir=tmp_path,
            target=target,
            relational_loss_profile="unknown",
        )


def test_tail_rank_relational_objective_penalizes_tail_compression() -> None:
    torch = pytest.importorskip("torch")
    target = torch.tensor([0.02, 0.08, 0.15, 0.35, 0.60, 0.95, 1.40, 2.00])
    compressed = torch.full_like(target, 0.70)
    repaired = target + torch.tensor([0.01, -0.01, 0.01, 0.0, -0.01, 0.01, -0.02, 0.02])
    pair_index = torch.tensor([7, 6, 5, 4, 3, 2, 1, 0], dtype=torch.long)

    huber = _relational_training_objective(
        compressed,
        target,
        profile="huber_v1",
        pair_index=pair_index,
    )
    expected_huber = torch.nn.functional.smooth_l1_loss(compressed, target)
    compressed_tail = _relational_training_objective(
        compressed,
        target,
        profile="tail_rank_v1",
        pair_index=pair_index,
    )
    repaired_tail = _relational_training_objective(
        repaired,
        target,
        profile="tail_rank_v1",
        pair_index=pair_index,
    )

    torch.testing.assert_close(huber, expected_huber)
    assert compressed_tail > repaired_tail


def test_default_relational_loss_profile_is_backward_compatible(tmp_path: Path) -> None:
    config = ProteinPredictionConfig(
        output_dir=tmp_path,
        target=ProteinTarget("YAP"),
    )

    assert config.relational_loss_profile == "huber_v1"


def test_fractional_cell_measurements_and_validated_binary_policy() -> None:
    result = aggregate_cell_measurements(
        np.asarray([1, 1, 1, 2, 2, 2, 0]),
        np.asarray([0.1, 0.4, 0.8, 0.1, 0.2, 0.3, 9.0]),
        overlap_weights=np.asarray([1.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0]),
        positive=np.asarray([False, True, True, False, False, False, True]),
        expected_area=np.asarray([2.0, 3.0]),
    )
    np.testing.assert_array_equal(result.label_ids, [1, 2])
    np.testing.assert_allclose(result.mean_od, [0.35, 0.2])
    np.testing.assert_allclose(result.positive_area_fraction, [0.5, 0.0])
    np.testing.assert_array_equal(result.binary_label, [1, 0])
    np.testing.assert_array_equal(result.measured, [True, True])


def test_measurements_without_accepted_threshold_are_continuous_only() -> None:
    result = aggregate_cell_measurements(
        np.asarray([1, 1]),
        np.asarray([0.2, 0.3]),
        expected_area=np.asarray([2.0]),
    )
    assert result.binary_label.tolist() == [-1]
    assert np.isnan(result.positive_area_fraction[0])


def test_inner_boundary_band_preserves_label_ids() -> None:
    labels = np.zeros((7, 7), dtype=np.uint32)
    labels[1:6, 1:6] = 4
    boundary = inner_boundary_mask(labels)
    assert np.count_nonzero(boundary == 4) == 16
    assert boundary[3, 3] == 0


def test_neutral_render_is_white_outside_tissue_and_target_free() -> None:
    od = np.asarray([[0.0, 1.0], [2.0, 4.0]])
    tissue = np.asarray([[False, True], [True, True]])
    rendered = render_neutral_morphology(od, tissue, display_max=2.0)
    np.testing.assert_array_equal(rendered[0, 0], [255, 255, 255])
    np.testing.assert_array_equal(rendered[1, 0], [74, 54, 122])
    np.testing.assert_array_equal(rendered[1, 1], rendered[1, 0])


def test_patch_pooling_has_bounded_support() -> None:
    pooled, supported = pool_patch_features_to_cells(
        np.asarray([[0.0, 0.0], [10.0, 0.0]]),
        np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        np.asarray([[5.0, 0.0], [1000.0, 1000.0]]),
        maximum_distance_um=20.0,
    )
    np.testing.assert_allclose(pooled[0], [0.5, 0.5])
    assert supported.tolist() == [True, False]


def test_leave_one_section_out_never_uses_target_section() -> None:
    probabilities = np.asarray(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0]],
            [[0.0, 1.0], [1.0, 0.0]],
        ]
    )
    consensus, support = leave_one_section_out_consensus(probabilities, 0)
    np.testing.assert_allclose(consensus, [[0.0, 1.0], [1.0, 0.0]])
    assert support.all()


def test_spatial_split_keeps_blocks_whole_and_buffers_boundaries() -> None:
    coordinates = np.asarray(
        [
            [x * 1000.0 + offset, y * 1000.0 + offset]
            for x in range(8)
            for y in range(8)
            for offset in (50.0, 500.0)
        ]
    )
    split = build_spatial_split(coordinates, block_um=1000.0, buffer_um=100.0, seed=7)
    fold_by_index = np.full(len(coordinates), -1)
    fold_by_index[split.train] = 0
    fold_by_index[split.validation] = 1
    fold_by_index[split.test] = 2
    for block in np.unique(split.block_ids):
        assignments = np.unique(fold_by_index[split.block_ids == block])
        assert len(assignments[assignments >= 0]) <= 1
    assert len(split.excluded) == 64
    assert len(split.train) and len(split.validation) and len(split.test)


def test_leave_one_group_out_holds_complete_mice() -> None:
    groups = np.asarray(["a", "a", "b", "c"])
    folds = leave_one_group_out(groups)
    assert len(folds) == 3
    assert all(len(np.intersect1d(train, test)) == 0 for train, test in folds)


def _table() -> CellExpressionTable:
    count = 24
    rng = np.random.default_rng(4)
    features = rng.normal(size=(count, 5)).astype(np.float32)
    measured = np.maximum(features[:, 0] + 0.5 * features[:, 1] + 2.0, 0.0)
    return CellExpressionTable(
        target_id="perk",
        label_ids=np.arange(1, count + 1, dtype=np.uint32),
        section_ids=np.asarray(["001"] * 12 + ["002"] * 12),
        mouse_ids=np.asarray(["mouse-a"] * 12 + ["mouse-b"] * 12),
        native_xy=np.column_stack([np.arange(count), np.arange(count)]),
        reference_um_xy=np.column_stack(
            [np.arange(count) * 1200.0, np.arange(count) * 1200.0]
        ),
        features=features,
        measured_od=measured.astype(np.float32),
        binary_label=(measured > np.median(measured)).astype(np.int8),
        measurement_coverage=np.ones(count, dtype=np.float32),
        semantic_region=np.arange(count, dtype=np.int16) % 3,
        semantic_support=np.ones(count, dtype=bool),
        provenance={"feature_view": "stain_neutral_v1", "source": "synthetic"},
    )


def test_cell_table_is_portable_and_detects_stale_content(tmp_path: Path) -> None:
    table = _table()
    path = table.save(tmp_path / "cells.npz")
    loaded = CellExpressionTable.load(path)
    assert loaded.fingerprint == table.fingerprint
    np.testing.assert_array_equal(loaded.section_ids, table.section_ids)
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"]))
        arrays = {
            name: archive[name] for name in archive.files if name != "metadata_json"
        }
    arrays["features"] = arrays["features"].copy()
    arrays["features"][0, 0] += 1
    np.savez_compressed(path, metadata_json=json.dumps(metadata), **arrays)
    with pytest.raises(ValueError, match="fingerprint"):
        CellExpressionTable.load(path)


def test_cell_table_extension_preserves_exact_base_rows_and_scope(
    tmp_path: Path,
) -> None:
    def make_table(
        mice: tuple[str, ...],
        *,
        feature_offset: float,
        binding_suffix: str = "",
    ) -> CellExpressionTable:
        count = len(mice) * 2
        mouse_ids = np.repeat(np.asarray(mice), 2)
        section_ids = np.asarray(
            [f"{index + 1:03d}" for index in range(len(mice)) for _ in range(2)]
        )
        labels = np.tile(np.asarray([1, 2], dtype=np.uint32), len(mice))
        features = (
            np.arange(count * 3, dtype=np.float32).reshape(count, 3) + feature_offset
        )
        bindings = {
            mouse: {"cell_result_fingerprint": f"cells-{mouse}{binding_suffix}"}
            for mouse in mice
        }
        return CellExpressionTable(
            target_id="yap",
            label_ids=labels,
            section_ids=section_ids,
            mouse_ids=mouse_ids,
            native_xy=np.arange(count * 2, dtype=float).reshape(count, 2),
            reference_um_xy=np.arange(count * 2, dtype=float).reshape(count, 2),
            features=features,
            measured_od=np.linspace(0.1, 0.8, count, dtype=np.float32),
            binary_label=np.tile(np.asarray([0, 1], dtype=np.int8), len(mice)),
            measurement_coverage=np.ones(count, dtype=np.float32),
            semantic_region=np.zeros(count, dtype=np.int16),
            semantic_support=np.ones(count, dtype=bool),
            provenance={
                "schema_version": 3,
                "measurement_view": "counterstain-v3",
                "measurement_source_view": "counterstain-v3",
                "measurement_compartment": "whole_cell",
                "measurement_statistic": "mean",
                "coverage_definition": "tissue_supported_compartment_area_fraction",
                "feature_view": "stain-neutral-cell-v3",
                "cohort_bindings": bindings,
                "target_sections": [
                    {"cohort": mouse, "section": f"{index + 1:03d}"}
                    for index, mouse in enumerate(mice)
                ],
                "od_calibration": None,
                "harmonization_scope": "none",
                "semantic_role": "soft_region_prior_not_cell_type_bound",
            },
        )

    base = make_table(("a", "b"), feature_offset=0)
    # Rows for a/b deliberately differ. Only the truly new cohort may be used.
    candidate = make_table(("a", "b", "c"), feature_offset=100)
    merged = extend_cell_expression_table(base, candidate)
    loaded = CellExpressionTable.load(merged.save(tmp_path / "extended.npz"))

    assert loaded.fingerprint == merged.fingerprint
    assert tuple(loaded.mouse_ids) == ("a", "a", "b", "b", "c", "c")
    for name in (
        "label_ids",
        "section_ids",
        "mouse_ids",
        "native_xy",
        "reference_um_xy",
        "features",
        "measured_od",
        "binary_label",
        "measurement_coverage",
        "semantic_region",
        "semantic_support",
    ):
        np.testing.assert_array_equal(
            np.asarray(getattr(loaded, name))[: len(base.label_ids)],
            np.asarray(getattr(base, name)),
        )
    assert loaded.provenance["table_extension"] == {
        "protocol": "exact-base-plus-disjoint-outcome-cohorts-v1",
        "base_table_fingerprint": base.fingerprint,
        "addition_table_fingerprint": candidate.fingerprint,
        "addition_cohorts": ["c"],
    }

    with pytest.raises(ValueError, match="must be new"):
        extend_cell_expression_table(base, candidate, addition_cohorts=("a",))
    conflicting = make_table(
        ("a", "b", "c"), feature_offset=100, binding_suffix="-changed"
    )
    with pytest.raises(ValueError, match="binding differs"):
        extend_cell_expression_table(base, conflicting)
    different_scope = replace(
        candidate,
        provenance={**candidate.provenance, "measurement_view": "wrong"},
        fingerprint=None,
    )
    with pytest.raises(ValueError, match="scientific scopes differ"):
        extend_cell_expression_table(base, different_scope)


@pytest.mark.protein
def test_model_and_predictions_are_portable_without_pickle(tmp_path: Path) -> None:
    table = _table()
    model = fit_protein_model(
        table.features,
        table.measured_od,
        table.binary_label,
        np.arange(18),
        target_id="perk",
        pca_components=3,
        hidden_units=4,
        hidden_layers=2,
        ensemble_seeds=2,
        max_iter=20,
        seed=3,
        provenance={"feature_view": "stain_neutral_v1"},
    )
    model_path = model.save(tmp_path / "model.npz")
    loaded = ProteinModel.load(model_path)
    expected = model.predict(table.features)
    actual = loaded.predict(table.features)
    for left, right in zip(expected, actual, strict=True):
        np.testing.assert_allclose(left, right)
    probability, relative, od, uncertainty = actual
    assert probability is not None
    predictions = ProteinPredictions(
        target_id="perk",
        model_fingerprint=str(loaded.fingerprint),
        label_ids=table.label_ids,
        section_ids=table.section_ids,
        expression_probability=probability,
        relative_expression=relative,
        predicted_od_reference=od,
        measured_od=table.measured_od,
        uncertainty=uncertainty,
        supported=table.semantic_support,
        provenance={"table_fingerprint": table.fingerprint},
    )
    path = predictions.save(tmp_path / "predictions.npz")
    assert ProteinPredictions.load(path).fingerprint == predictions.fingerprint


def test_metrics_report_roc_pr_and_continuous_accuracy() -> None:
    truth = np.asarray([0.0, 0.1, 0.8, 1.0])
    scores = np.asarray([0.05, 0.2, 0.7, 0.9])
    labels = np.asarray([0, 0, 1, 1])
    metrics = evaluate_predictions(
        truth, scores, binary_label=labels, probability=scores
    )
    assert metrics["auroc"] == 1.0
    assert metrics["average_precision"] == 1.0
    assert metrics["spearman"] == 1.0


@pytest.mark.protein
def test_candidate_benchmark_selects_by_held_out_spatial_metric(tmp_path: Path) -> None:
    from histopia.protein._cli import _benchmark

    config = ProteinPredictionConfig(
        output_dir=tmp_path,
        target=ProteinTarget("pERK"),
        model_candidates=("morphospatial_knn", "linear", "tree"),
    )
    metrics = _benchmark(config, _table(), group="section")
    assert metrics["schema_version"] == 2
    assert metrics["selected_candidate"] in config.model_candidates
    assert metrics["selection_metric"] == "spearman_64um_then_cell_spearman"
    assert set(metrics["candidates"]) == set(config.model_candidates)


@pytest.mark.protein
def test_stain_invariance_audit_compares_neutral_with_raw_features() -> None:
    rng = np.random.default_rng(8)
    stains = np.tile(np.asarray(["dab", "pas"]), 20)
    groups = np.repeat(np.asarray(["m1", "m2", "m3", "m4"]), 10)
    neutral = rng.normal(size=(40, 3))
    raw = neutral.copy()
    raw[:, 0] += np.where(stains == "dab", -4.0, 4.0)
    audit = evaluate_stain_invariance(
        neutral,
        raw,
        stains,
        groups,
        np.linspace(0, 1, 40),
        np.linspace(0, 1, 40) + 0.005,
    )
    assert audit["chromogen_prediction_delta"] == pytest.approx(0.005)
    assert audit["raw_stain_macro_auroc"] > audit["neutral_stain_macro_auroc"]


def test_result_manifest_seals_artifacts_and_requires_approval(tmp_path: Path) -> None:
    (tmp_path / "model.npz").write_bytes(b"model")
    (tmp_path / "predictions.npz").write_bytes(b"predictions")
    (tmp_path / "measurement-audit.npz").write_bytes(b"raw-and-calibrated")
    write_protein_result(
        tmp_path,
        {
            "schema_version": 2,
            "target_id": "ecad",
            "assay_domain": "pooled",
            "model": "model.npz",
            "measurement_audit": "measurement-audit.npz",
            "slides": [
                {
                    "section": "001",
                    "predictions": "predictions.npz",
                    "prediction_fingerprint": "0" * 64,
                }
            ],
            "metrics": {},
        },
    )
    result = validate_protein_result(tmp_path)
    assert "model.npz" in result["artifacts"]
    assert "measurement-audit.npz" in result["artifacts"]
    with pytest.raises(ValueError, match="not approved"):
        validate_protein_approval(tmp_path)
    approve_protein_result(tmp_path, reviewer="test", accepted=True)
    assert validate_protein_approval(tmp_path)["accepted"] is True
    (tmp_path / "predictions.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="stale"):
        validate_protein_result(tmp_path)


def test_real_map_aggregation_uses_only_tissue_pixels() -> None:
    from histopia.protein._real_data import aggregate_map_to_sampled_labels

    labels = np.array([[1, 1], [2, 2]], dtype=np.uint32)
    od = np.array([[1.0, 3.0], [8.0, 4.0]], dtype=np.float32)
    tissue = np.array([[1, 1], [0, 1]], dtype=np.uint8)
    mean, effective, fraction = aggregate_map_to_sampled_labels(
        labels,
        od,
        tissue,
        od >= 2.0,
        source_mpp_xy=np.array([4.0, 4.0]),
        analysis_mpp=4.0,
        content_origin_native_xy=np.array([0, 0]),
        pyramid_scale=1.0,
        cell_count=2,
    )
    assert mean[1] == pytest.approx(2.0)
    assert mean[2] == pytest.approx(4.0)
    assert effective.tolist() == [0.0, 2.0, 1.0]
    assert fraction[1] == pytest.approx(0.5)


def test_real_map_aggregation_supports_per_cell_q90() -> None:
    from histopia.protein._real_data import aggregate_map_to_sampled_labels

    labels = np.array([[1, 1], [2, 2]], dtype=np.uint32)
    od = np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32)
    q90, effective, _fraction = aggregate_map_to_sampled_labels(
        labels,
        od,
        np.ones_like(labels, dtype=bool),
        np.zeros_like(labels, dtype=bool),
        source_mpp_xy=np.array([4.0, 4.0]),
        analysis_mpp=4.0,
        content_origin_native_xy=np.array([0, 0]),
        pyramid_scale=1.0,
        cell_count=2,
        statistic="q90",
    )

    assert q90[1] == pytest.approx(3.0)
    assert q90[2] == pytest.approx(4.0)
    assert effective.tolist() == [0.0, 2.0, 2.0]


def test_real_map_aggregation_uses_content_local_label_coordinates() -> None:
    from histopia.protein._real_data import aggregate_map_to_sampled_labels

    labels = np.array([[1, 1], [2, 2]], dtype=np.uint32)
    od = np.array([[1.0, 3.0], [8.0, 4.0]], dtype=np.float32)
    tissue = np.array([[1, 1], [0, 1]], dtype=np.uint8)

    mean, effective, fraction = aggregate_map_to_sampled_labels(
        labels,
        od,
        tissue,
        od >= 2.0,
        source_mpp_xy=np.array([4.0, 4.0]),
        analysis_mpp=4.0,
        content_origin_native_xy=np.array([21_446, 59_540]),
        pyramid_scale=1.0,
        cell_count=2,
    )

    assert mean[1] == pytest.approx(2.0)
    assert mean[2] == pytest.approx(4.0)
    assert effective.tolist() == [0.0, 2.0, 1.0]
    assert fraction[1] == pytest.approx(0.5)


def test_adaptive_target_measurement_is_strict_tissue_only_and_fingerprinted() -> None:
    class Map:
        analysis_mpp = 4.0
        corrected_target_od = np.asarray([[0.2, 0.8], [2.0, 0.5]], dtype=np.float32)
        tissue_mask = np.asarray([[1, 1], [0, 1]], dtype=bool)
        content_fingerprint = "a" * 64

    row = {
        "map_artifact_digest": "b" * 64,
        "qc": {
            "correction_accepted": True,
            "threshold_accepted": True,
            "positive_threshold_od": 0.7,
            "adaptive_background": {
                "method": "inferred_floor-v2",
                "accepted": True,
                "floor_od": 0.3,
            },
        },
    }
    result = validated_adaptive_target_measurement(Map(), row)
    np.testing.assert_allclose(result.target_od, [[0.0, 0.5], [0.0, 0.2]])
    assert result.tissue_mask.tolist() == [[True, True], [False, True]]
    assert result.positive_mask.tolist() == [[False, True], [False, False]]
    assert result.positive_threshold_od == pytest.approx(0.4)
    assert result.floor_od == pytest.approx(0.3)
    assert len(result.fingerprint) == 64


def test_counterstain_target_uses_sealed_sidecar_without_reflooring() -> None:
    diagnostics = {
        "method": "counterstain-conditioned-v3",
        "accepted": True,
    }

    class Map:
        slide_id = "slide.ndpi"
        analysis_mpp = 4.0
        corrected_target_od = np.ones((2, 2), dtype=np.float32)
        tissue_mask = np.asarray([[1, 1], [0, 1]], dtype=bool)
        content_origin_native_xy = (10, 20)
        source_mpp_xy = (0.5, 0.5)
        content_fingerprint = "a" * 64

    class Adaptive:
        slide_id = Map.slide_id
        analysis_mpp = Map.analysis_mpp
        target_od = np.asarray([[0.1, 0.8], [0.0, 0.3]], dtype=np.float32)
        tissue_mask = Map.tissue_mask
        content_origin_native_xy = Map.content_origin_native_xy
        source_mpp_xy = Map.source_mpp_xy
        source_content_fingerprint = Map.content_fingerprint
        method = "counterstain-conditioned-v3"
        diagnostics = {
            "method": "counterstain-conditioned-v3",
            "accepted": True,
        }
        content_fingerprint = "b" * 64

    row = {
        "adaptive_map_fingerprint": Adaptive.content_fingerprint,
        "qc": {
            "correction_accepted": True,
            "adaptive_background": diagnostics,
        },
    }
    result = validated_adaptive_target_measurement(
        Map(),
        row,
        adaptive_stain_map=Adaptive(),
    )

    np.testing.assert_array_equal(result.target_od, Adaptive.target_od)
    assert result.floor_od is None
    assert result.positive_threshold_od is None
    assert result.measurement_view.endswith("counterstain-conditioned-target-od-4um-v3")
    with pytest.raises(ValueError, match="digest differs"):
        validated_adaptive_target_measurement(
            Map(),
            {**row, "adaptive_map_fingerprint": "c" * 64},
            adaptive_stain_map=Adaptive(),
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("correction_accepted", False, "nuisance correction"),
        ("adaptive_accepted", False, "adaptive background"),
    ],
)
def test_adaptive_target_measurement_refuses_fallbacks(
    field: str, value: bool, message: str
) -> None:
    class Map:
        analysis_mpp = 4.0
        corrected_target_od = np.ones((2, 2), dtype=np.float32)
        tissue_mask = np.ones((2, 2), dtype=bool)
        content_fingerprint = "a" * 64

    qc = {
        "correction_accepted": True,
        "adaptive_background": {
            "method": "inferred_floor-v2",
            "accepted": True,
            "floor_od": 0.2,
        },
    }
    if field == "adaptive_accepted":
        qc["adaptive_background"]["accepted"] = value
    else:
        qc[field] = value
    with pytest.raises(ValueError, match=message):
        validated_adaptive_target_measurement(Map(), {"qc": qc})


def test_continuous_target_sections_do_not_require_binary_threshold() -> None:
    from histopia.protein._real_data import target_sections

    stain = {
        "slides": [
            {
                "order": 2,
                "marker": "Yap",
                "quantified": True,
                "map": "maps/002.npz",
                "qc": {
                    "correction_accepted": True,
                    "threshold_accepted": True,
                    "adaptive_background": {"accepted": True},
                },
            },
            {
                "order": 7,
                "marker": "YAP",
                "quantified": True,
                "map": "maps/007.npz",
                "qc": {
                    "correction_accepted": True,
                    "threshold_accepted": False,
                    "adaptive_background": {"accepted": True},
                },
            },
        ]
    }

    with pytest.raises(ValueError, match="two accepted"):
        target_sections(stain, "YAP")
    assert target_sections(stain, "YAP", require_threshold=False) == ("002", "007")
    assert (
        target_sections(
            stain,
            "N-Cad",
            require_threshold=False,
            minimum_sections=0,
        )
        == ()
    )


def test_real_feature_pool_keeps_semantics_out_of_primary_features() -> None:
    from histopia.protein._real_data import pool_neutral_features

    features, support, regions = pool_neutral_features(
        np.array([[0.0, 0.0], [100.0, 0.0]]),
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        np.array([0, 2], dtype=np.int16),
        np.array([[2.0, 0.0], [102.0, 0.0], [500.0, 0.0]]),
        region_count=3,
        maximum_distance=20.0,
    )
    assert features.shape == (3, 2)
    assert support.tolist() == [True, True, False]
    assert regions.tolist() == [0, 2, 2]
    assert features[0].tolist() == [1.0, 2.0]
    assert features[1].tolist() == [3.0, 4.0]
    assert np.all(features[2] == 0)


def test_semantic_label_subset_joins_source_patches_by_grid_coordinate() -> None:
    from histopia.protein._real_workflow import _align_semantic_patch_rows

    native_xy = np.asarray([[10.0, 20.0], [30.0, 20.0], [50.0, 20.0], [10.0, 40.0]])
    aligned = _align_semantic_patch_rows(
        np.asarray([[0, 0], [0, 1], [0, 2], [1, 0]]),
        native_xy,
        (2, 3),
        np.asarray([[1, 0], [0, 2], [0, 0]]),
        np.asarray([2, 1, 0]),
        region_count=3,
    )

    np.testing.assert_array_equal(aligned, native_xy[[3, 2, 0]])


def test_semantic_label_subset_refuses_unknown_patch_coordinates() -> None:
    from histopia.protein._real_workflow import _align_semantic_patch_rows

    with pytest.raises(ValueError, match="not a subset"):
        _align_semantic_patch_rows(
            np.asarray([[0, 0], [0, 1]]),
            np.asarray([[10.0, 20.0], [30.0, 20.0]]),
            (2, 2),
            np.asarray([[1, 1]]),
            np.asarray([0]),
            region_count=2,
        )


def test_morphospatial_features_include_shape_neighborhood_and_xyz() -> None:
    result = morphospatial_features(
        np.asarray([[1.0, 2.0], [3.0, 4.0]]),
        np.asarray([[10.0, 20.0, 5.0], [30.0, 40.0, 10.0]]),
        shape_features=np.asarray([[0.2], [0.4]]),
        neighborhood_features=np.asarray([[0.1], [0.3]]),
        coordinate_scale_um=10.0,
    )
    assert result.shape == (2, 7)
    np.testing.assert_allclose(result[0], [1, 2, 0.2, 0.1, 1, 2, 0.5])


def test_fixed_stain_neutral_projection_is_deterministic_and_bounded() -> None:
    morphology = np.arange(60, dtype=np.float32).reshape(10, 6)
    first = fixed_stain_neutral_projection(
        morphology,
        components=4,
        batch_size=3,
    )
    second = fixed_stain_neutral_projection(
        morphology,
        components=4,
        batch_size=7,
    )
    assert first.shape == (10, 4)
    np.testing.assert_array_equal(first, second)


def test_bracketing_sections_do_not_use_distant_fallback_inside_range() -> None:
    measured = {"003": 15.0, "009": 45.0, "015": 75.0}
    assert select_bracketing_sections(50.0, measured) == ("009", "015")
    assert select_bracketing_sections(5.0, measured) == ("003",)
    assert select_bracketing_sections(90.0, measured) == ("015",)


def test_registered_anchor_transfer_uses_cell_morphology_and_registered_space() -> None:
    result = registered_morphospatial_anchor_transfer(
        np.asarray([[1.0, 0.05], [-1.0, 0.05]], dtype=np.float32),
        np.asarray([[5.0, 0.0, 10.0], [105.0, 0.0, 10.0]]),
        np.asarray(
            [[1.0, 0.0], [1.0, 0.1], [-1.0, 0.0], [-1.0, 0.1]],
            dtype=np.float32,
        ),
        np.asarray(
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [100.0, 0.0, 0.0], [110.0, 0.0, 0.0]]
        ),
        np.asarray([0.8, 0.9, 0.05, 0.1]),
        spatial_candidates=4,
        neighbors=2,
        minimum_cosine_similarity=0.5,
        xy_bandwidth_um=40.0,
        z_bandwidth_um=40.0,
    )
    assert isinstance(result, RegisteredAnchorTransfer)
    assert result.prediction.tolist() == pytest.approx([0.8, 0.05])
    assert np.all(result.confidence > 0)
    assert result.support_count.tolist() == [2, 2]


def test_registered_anchor_transfer_excludes_exact_measured_cell() -> None:
    result = registered_morphospatial_anchor_transfer(
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        np.asarray([[0.0, 0.0, 0.0]]),
        np.asarray([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]], dtype=np.float32),
        np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]]),
        np.asarray([0.99, 0.11, 0.12]),
        query_anchor_indices=np.asarray([0]),
        spatial_candidates=3,
        neighbors=1,
        minimum_cosine_similarity=-1.0,
    )
    assert result.prediction[0] == pytest.approx(0.11)
    assert result.nearest_distance_um[0] == pytest.approx(2.0)


def test_registered_anchor_transfer_automatically_excludes_exact_constant_key() -> None:
    result = registered_morphospatial_anchor_transfer(
        np.ones((1, 2), dtype=np.float32),
        np.asarray([[0.0, 0.0, 0.0]]),
        np.ones((3, 2), dtype=np.float32),
        np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]]),
        np.asarray([0.99, 0.11, 0.12]),
        spatial_candidates=3,
        neighbors=1,
        minimum_cosine_similarity=-1.0,
    )
    assert result.prediction[0] == pytest.approx(0.11)


def test_registered_anchor_transfer_retains_local_modes_instead_of_flooding() -> None:
    anchor_xy = np.asarray(
        [[0, 0, 0], [2, 0, 0], [4, 0, 0], [100, 0, 0], [102, 0, 0], [104, 0, 0]],
        dtype=float,
    )
    result = registered_morphospatial_anchor_transfer(
        np.asarray([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32),
        np.asarray([[2.0, 1.0, 8.0], [102.0, 1.0, 8.0]]),
        np.ones((6, 2), dtype=np.float32),
        anchor_xy,
        np.asarray([0.04, 0.05, 0.8, 0.85, 0.9, 0.06]),
        spatial_candidates=3,
        neighbors=3,
        minimum_cosine_similarity=-1.0,
        xy_bandwidth_um=50.0,
        z_bandwidth_um=50.0,
        aggregation="weighted_median",
    )
    assert result.prediction.tolist() == pytest.approx([0.05, 0.85])


def test_registered_anchor_transfer_semantics_are_only_a_soft_gate() -> None:
    result = registered_morphospatial_anchor_transfer(
        np.asarray([[1.0, 1.0]], dtype=np.float32),
        np.asarray([[0.0, 0.0, 2.0]]),
        np.ones((3, 2), dtype=np.float32),
        np.asarray([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        np.asarray([0.1, 0.9, 0.8]),
        query_regions=np.asarray([2]),
        anchor_regions=np.asarray([2, 7, 7]),
        spatial_candidates=3,
        neighbors=3,
        minimum_cosine_similarity=-1.0,
        semantic_mismatch_weight=0.01,
    )
    assert result.prediction[0] == pytest.approx(0.1)
    assert result.support_count[0] == 3


def test_registered_anchor_transfer_refuses_unsupported_extrapolation() -> None:
    result = registered_morphospatial_anchor_transfer(
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        np.asarray([[10_000.0, 0.0, 0.0]]),
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        np.asarray([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
        np.asarray([0.1, 0.9]),
        spatial_candidates=2,
        neighbors=2,
        maximum_xy_distance_um=100.0,
    )
    assert np.isnan(result.prediction[0])
    assert result.confidence[0] == 0
    assert result.support_count[0] == 0
    assert np.isinf(result.nearest_distance_um[0])


def test_registered_anchor_transfer_supports_explicit_signed_residuals() -> None:
    result = registered_morphospatial_anchor_transfer(
        np.ones((1, 2), dtype=np.float32),
        np.asarray([[1.0, 0.0, 2.0]]),
        np.ones((3, 2), dtype=np.float32),
        np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]]),
        np.asarray([-0.4, -0.3, 0.2]),
        spatial_candidates=3,
        neighbors=3,
        minimum_cosine_similarity=-1.0,
        allow_signed_outcomes=True,
    )
    assert result.prediction[0] == pytest.approx(-0.3)


def test_registered_anchor_transfer_rejects_signed_absolute_od() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        registered_morphospatial_anchor_transfer(
            np.ones((1, 2)),
            np.zeros((1, 3)),
            np.ones((2, 2)),
            np.asarray([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            np.asarray([-0.1, 0.2]),
        )


def test_tail_guarded_anchor_blend_preserves_background_and_bright_tails() -> None:
    refined, weight = tail_guarded_registered_anchor_blend(
        np.asarray([0.1, 0.5, 0.9]),
        np.asarray([0.5, 0.8, 0.5]),
        np.asarray([0.2, 0.2, 0.2]),
        maximum_weight=1.0,
        directional_tail_guard=1.0,
    )
    np.testing.assert_allclose(weight, [0.04, 0.2, 0.04], atol=1e-7)
    np.testing.assert_allclose(refined, [0.116, 0.56, 0.884], atol=1e-7)


def test_tail_guarded_anchor_blend_preserves_high_confidence_and_fallback() -> None:
    refined, weight = tail_guarded_registered_anchor_blend(
        np.asarray([0.1, 0.9]),
        np.asarray([0.7, np.nan]),
        np.asarray([1.0, 0.8]),
        maximum_weight=0.5,
    )
    np.testing.assert_allclose(refined, [0.4, 0.9])
    np.testing.assert_allclose(weight, [0.5, 0.0])


def test_registered_residual_correction_repairs_background_and_bright_tail() -> None:
    refined, weight = confidence_weighted_registered_residual_correction(
        np.asarray([0.4, 0.5, 0.6, 0.9]),
        np.asarray([-0.3, np.nan, 0.2, -0.4]),
        np.asarray([1.0, 0.8, 0.5, 0.2]),
        maximum_weight=1.0,
        confidence_power=1.0,
        directional_tail_guard=1.0,
        maximum_absolute_correction_od=0.25,
    )
    np.testing.assert_allclose(weight, [1.0, 0.0, 0.5, 0.04], atol=1e-7)
    np.testing.assert_allclose(refined, [0.15, 0.5, 0.7, 0.884], atol=1e-7)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"maximum_weight": 1.1}, "maximum_weight"),
        ({"maximum_weight": 0.5, "confidence_power": 0.0}, "confidence_power"),
        (
            {"maximum_weight": 0.5, "directional_tail_guard": 1.1},
            "directional_tail_guard",
        ),
        (
            {"maximum_weight": 0.5, "maximum_absolute_correction_od": 0.0},
            "maximum_absolute_correction_od",
        ),
    ],
)
def test_registered_residual_correction_rejects_invalid_controls(
    kwargs: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        confidence_weighted_registered_residual_correction(
            np.asarray([0.1, 0.9]),
            np.asarray([-0.1, 0.1]),
            np.asarray([0.5, 0.5]),
            **kwargs,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"maximum_weight": 1.1}, "maximum_weight"),
        ({"maximum_weight": 0.5, "confidence_power": 0.0}, "confidence_power"),
        (
            {"maximum_weight": 0.5, "directional_tail_guard": -0.1},
            "directional_tail_guard",
        ),
        (
            {"maximum_weight": 0.5, "lower_quantile": 0.9, "upper_quantile": 0.2},
            "tail quantiles",
        ),
    ],
)
def test_tail_guarded_anchor_blend_rejects_invalid_controls(
    kwargs: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        tail_guarded_registered_anchor_blend(
            np.asarray([0.1, 0.9]),
            np.asarray([0.2, 0.8]),
            np.asarray([0.5, 0.5]),
            **kwargs,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"neighbors": 9, "spatial_candidates": 8}, "controls"),
        ({"workers": 0}, "controls"),
        ({"semantic_mismatch_weight": 0.0}, "semantic_mismatch_weight"),
        ({"aggregation": "maximum"}, "aggregation"),
    ],
)
def test_registered_anchor_transfer_rejects_invalid_controls(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        registered_morphospatial_anchor_transfer(
            np.ones((1, 2)),
            np.zeros((1, 3)),
            np.ones((2, 2)),
            np.zeros((2, 3)),
            np.asarray([0.1, 0.2]),
            **kwargs,
        )


def test_smoothing_requires_spatial_and_morphology_compatibility() -> None:
    values = np.asarray([0.0, 1.0, 0.8, 0.2])
    morphology = np.asarray([[1, 0], [0, 1], [1, 0], [1, 0]], dtype=float)
    xyz = np.asarray([[0, 0, 0], [2, 0, 0], [4, 0, 0], [500, 0, 0]], dtype=float)
    smoothed = morphology_compatible_smoothing(
        values,
        morphology,
        xyz,
        maximum_distance_um=10,
        minimum_cosine_similarity=0.9,
        blend=0.5,
    )
    assert smoothed[0] == pytest.approx(0.4)
    assert smoothed[1] == pytest.approx(1.0)
    assert smoothed[3] == pytest.approx(0.2)


def test_real_workflow_reports_metrics_for_portable_deployed_candidate() -> None:
    from histopia.protein._real_workflow import _deployed_candidate_metrics

    metrics = _deployed_candidate_metrics(
        {
            "selected_candidate": "cross_attention",
            "selection_metric": "spearman_64um_then_cell_spearman",
            "spearman": 0.61,
            "candidates": {
                "cross_attention": {"available": True, "spearman": 0.61},
                "hurdle_mlp": {"available": True, "spearman": 0.31},
            },
        },
        deployed_candidate="hurdle_mlp",
    )

    assert metrics["spearman"] == 0.31
    assert metrics["selected_candidate"] == "hurdle_mlp"
    assert metrics["deployed_candidate"] == "hurdle_mlp"
    assert metrics["benchmark_selected_candidate"] == "cross_attention"
    assert "not the portable deployed artifact" in metrics["deployment_note"]


def test_schema_v4_study_result_seals_prediction_and_training_cohorts(
    tmp_path: Path,
) -> None:
    for name in (
        "bundle.json",
        "table.npz",
        "audit.json",
        "fold.npz",
        "fold.calibration.json",
        "a.npz",
        "b.npz",
    ):
        (tmp_path / name).write_bytes(name.encode())
    binding = {
        "registration_result_sha256": "1" * 64,
        "cell_result_fingerprint": "2" * 64,
        "stain_result_fingerprint": "3" * 64,
        "semantic_result_fingerprint": "4" * 64,
    }
    calibration = fit_equal_section_od_harmonization(
        np.asarray(["a"] * 128 + ["b"] * 128),
        np.asarray(["001"] * 256),
        np.r_[np.linspace(0, 1, 128), np.linspace(0, 2, 128)],
        np.ones(256, dtype=bool),
        minimum_cells=128,
    ).as_dict()
    core = {
        "schema_version": 4,
        "model_id": "yap-extra-trees-a-adaptive-harmonized-v2",
        "target_id": "yap",
        "architecture": "extra_trees",
        "implementation": {
            "serialization": "joblib-pickle-exact-runtime-v1",
            "scikit_learn": "1.9.0",
            "joblib": "1.5.3",
            "numpy": "2.2.6",
        },
        "model_version": "adaptive-harmonized-v2",
        "training_cohorts": ["a"],
        "cohort_bindings": {"a": binding, "b": binding},
        "measurement_view": (
            "tissue-masked-harmonized-adaptive-corrected-target-od-4um-v2"
        ),
        "measurement_source_view": (
            "tissue-masked-adaptive-corrected-target-od-4um-v1"
        ),
        "od_calibration": calibration,
        "model": "bundle.json",
        "training_table": "table.npz",
        "measurement_audit": "audit.json",
        "model_artifacts": ["fold.npz", "fold.calibration.json"],
        "model_reuse": {
            "scope": "identical-training-scope-v1",
            "source_result_fingerprint": "f" * 64,
        },
        "metrics": {
            "spearman": 0.6,
            "spearman_64um": 0.7,
            "mae": 0.2,
            "mean_bias_fraction": 0.01,
            "median_bias_fraction": 0.02,
            "folds": [
                {
                    "held_out": "b",
                    "mae_improvement": 0.2,
                    "mean_bias_fraction": 0.24,
                    "median_bias_fraction": 0.29,
                }
            ],
        },
        "slides": [
            {"cohort": "a", "section": "001", "predictions": "a.npz"},
            {"cohort": "b", "section": "001", "predictions": "b.npz"},
        ],
    }

    result_path = write_protein_result(tmp_path, core)
    result = validate_protein_result(tmp_path)

    assert result_path == tmp_path / "protein_result.json"
    assert set(result["artifacts"]) == {
        "bundle.json",
        "table.npz",
        "audit.json",
        "fold.npz",
        "fold.calibration.json",
        "a.npz",
        "b.npz",
    }
    assert result["training_cohorts"] == ["a"]
    assert set(result["cohort_bindings"]) == {"a", "b"}
    assert result["model_reuse"]["source_result_fingerprint"] == "f" * 64
    assert result["implementation"]["scikit_learn"] == "1.9.0"

    from histopia.protein import reassess_protein_result

    reassess_protein_result(tmp_path)
    reassessed = validate_protein_result(tmp_path)
    assert reassessed["status"] == "candidate"
    assert any(
        "fold b" in reason for reason in reassessed["candidate_promotion"]["reasons"]
    )

    invalid = dict(core)
    invalid["model_reuse"] = {
        "scope": "unverified",
        "source_result_fingerprint": "f" * 64,
    }
    with pytest.raises(ValueError, match="reuse provenance"):
        write_protein_result(tmp_path, invalid)

    invalid_runtime = dict(core)
    invalid_runtime["implementation"] = {
        **core["implementation"],
        "serialization": "unversioned-pickle",
    }
    with pytest.raises(ValueError, match="runtime implementation"):
        write_protein_result(tmp_path, invalid_runtime)


def test_schema_v4_relational_refinement_requires_disjoint_frozen_evidence(
    tmp_path: Path,
) -> None:
    for name in ("bundle.json", "table.npz", "audit.json", "final.npz", "a.npz"):
        (tmp_path / name).write_bytes(name.encode())
    binding = {
        "registration_result_sha256": "1" * 64,
        "cell_result_fingerprint": "2" * 64,
        "stain_result_fingerprint": "3" * 64,
        "semantic_result_fingerprint": "4" * 64,
    }
    refinement = {
        "schema_version": 1,
        "method": ("calibrated-neural-plus-training-only-morphology-od-transfer-v1"),
        "weight": 0.2,
        "neighbors": 16,
        "evidence_sha256": "5" * 64,
        "base_model_fingerprint": "6" * 64,
        "base_model_sha256": "7" * 64,
        "development_cohorts": ["dev"],
        "confirmation_cohorts": ["confirm"],
        "confirmation_gate": "accepted-once-after-development-freeze",
        "evidence_source_result_fingerprints": ["8" * 64],
    }
    core = {
        "schema_version": 4,
        "model_id": "yap-dual-bank-attention-a-adaptive-v1-transfer",
        "target_id": "yap",
        "architecture": "dual_bank_attention",
        "model_version": "adaptive-v1-morphology-transfer-v1",
        "prediction_protocol": "leave-one-mouse-out",
        "training_cohorts": ["a"],
        "cohort_bindings": {"a": binding},
        "measurement_view": ("tissue-masked-adaptive-corrected-target-od-4um-v1"),
        "measurement_statistic": "mean",
        "feature_schema_id": "native-hdab-neutral-cell-multiscale-v3",
        "model": "bundle.json",
        "training_table": "table.npz",
        "measurement_audit": "audit.json",
        "model_artifacts": ["final.npz"],
        "postfit_refinement": refinement,
        "slides": [{"cohort": "a", "section": "001", "predictions": "a.npz"}],
    }

    write_protein_result(tmp_path, core)
    result = validate_protein_result(tmp_path)
    assert result["postfit_refinement"] == refinement

    expanded = dict(core)
    expanded["postfit_refinement"] = {
        **refinement,
        "method": "neural-plus-training-only-morphology-od-transfer-v2",
    }
    write_protein_result(tmp_path, expanded)
    assert (
        validate_protein_result(tmp_path)["postfit_refinement"]
        == expanded["postfit_refinement"]
    )

    overlapping = dict(core)
    overlapping["postfit_refinement"] = {
        **refinement,
        "confirmation_cohorts": ["dev"],
    }
    with pytest.raises(ValueError, match="cohorts overlap"):
        write_protein_result(tmp_path, overlapping)

    wrong_architecture = dict(core)
    wrong_architecture["architecture"] = "extra_trees"
    with pytest.raises(ValueError, match="post-fit refinement"):
        write_protein_result(tmp_path, wrong_architecture)


def test_schema_v4_registered_residual_requires_sealed_confirmed_anchors(
    tmp_path: Path,
) -> None:
    artifacts = (
        "bundle.json",
        "table.npz",
        "audit.json",
        "final.npz",
        "anchor.npz",
        "frozen-design.json",
        "quantitative-confirmation.json",
        "visual-acceptance.json",
        "a.npz",
    )
    for name in artifacts:
        (tmp_path / name).write_bytes(name.encode())
    binding = {
        "registration_result_sha256": "1" * 64,
        "cell_result_fingerprint": "2" * 64,
        "stain_result_fingerprint": "3" * 64,
        "semantic_result_fingerprint": "4" * 64,
    }
    refinement = {
        "schema_version": 1,
        "method": "confidence-weighted-registered-residual-transfer-v5",
        "base_model_fingerprint": "5" * 64,
        "base_result_fingerprint": "6" * 64,
        "frozen_design_fingerprint": "7" * 64,
        "development_cohorts": ["dev"],
        "confirmation_cohorts": ["a"],
        "anchor_cohorts": ["a"],
        "anchor_sections": {"a": ["001"]},
        "anchor_bank_artifact": "anchor.npz",
        "evidence_artifacts": {
            "frozen_design": "frozen-design.json",
            "quantitative_confirmation": "quantitative-confirmation.json",
            "visual_acceptance": "visual-acceptance.json",
        },
        "quantitative_gate": "passed-once-after-design-freeze",
        "visual_gate": "accepted-cell-resolved-native-audit",
        "unsupported_behavior": "exact-frozen-inductive-baseline-fallback",
    }
    core = {
        "schema_version": 4,
        "model_id": "yap-dual-bank-attention-registered-residual-v5",
        "target_id": "yap",
        "architecture": "dual_bank_attention",
        "model_version": "registered-residual-v5",
        "prediction_protocol": "leave-one-mouse-out",
        "training_cohorts": ["dev"],
        "prediction_cohorts": ["a"],
        "cohort_bindings": {"dev": binding, "a": binding},
        "measurement_view": ("tissue-masked-adaptive-corrected-target-od-4um-v1"),
        "measurement_statistic": "mean",
        "feature_schema_id": "native-hdab-neutral-cell-multiscale-v3",
        "model": "bundle.json",
        "training_table": "table.npz",
        "measurement_audit": "audit.json",
        "model_artifacts": [
            "final.npz",
            "anchor.npz",
            "frozen-design.json",
            "quantitative-confirmation.json",
            "visual-acceptance.json",
        ],
        "registered_residual_refinement": refinement,
        "slides": [{"cohort": "a", "section": "001", "predictions": "a.npz"}],
    }

    write_protein_result(tmp_path, core)
    result = validate_protein_result(tmp_path)
    assert result["registered_residual_refinement"] == refinement

    unconfirmed = dict(core)
    unconfirmed["registered_residual_refinement"] = {
        **refinement,
        "anchor_cohorts": ["unseen"],
        "anchor_sections": {"unseen": ["001"]},
    }
    with pytest.raises(ValueError, match="anchors are not confirmed"):
        write_protein_result(tmp_path, unconfirmed)

    missing_evidence = dict(core)
    missing_evidence["model_artifacts"] = [
        value for value in core["model_artifacts"] if value != "visual-acceptance.json"
    ]
    with pytest.raises(ValueError, match="artifacts are invalid"):
        write_protein_result(tmp_path, missing_evidence)

    overlapping = dict(core)
    overlapping["registered_residual_refinement"] = {
        **refinement,
        "development_cohorts": ["a"],
    }
    with pytest.raises(ValueError, match="cohorts overlap"):
        write_protein_result(tmp_path, overlapping)


def test_schema_v4_study_result_requires_predictions_for_every_binding(
    tmp_path: Path,
) -> None:
    for name in ("bundle.json", "table.npz", "a.npz"):
        (tmp_path / name).write_bytes(name.encode())
    binding = {
        "registration_result_sha256": "1" * 64,
        "cell_result_fingerprint": "2" * 64,
        "stain_result_fingerprint": "3" * 64,
        "semantic_result_fingerprint": "4" * 64,
    }
    with pytest.raises(ValueError, match="slide cohorts differ"):
        write_protein_result(
            tmp_path,
            {
                "schema_version": 4,
                "model_id": "yap-extra-trees-all2-adaptive-v1",
                "target_id": "yap",
                "architecture": "extra_trees",
                "model_version": "adaptive-v1",
                "training_cohorts": ["a", "b"],
                "cohort_bindings": {"a": binding, "b": binding},
                "measurement_view": (
                    "tissue-masked-adaptive-corrected-target-od-4um-v1"
                ),
                "model": "bundle.json",
                "training_table": "table.npz",
                "model_artifacts": [],
                "slides": [{"cohort": "a", "section": "001", "predictions": "a.npz"}],
            },
        )


def test_schema_v4_study_result_allows_explicit_prediction_cohort_subset(
    tmp_path: Path,
) -> None:
    for name in ("bundle.json", "table.npz", "b.npz"):
        (tmp_path / name).write_bytes(name.encode())
    binding = {
        "registration_result_sha256": "1" * 64,
        "cell_result_fingerprint": "2" * 64,
        "stain_result_fingerprint": "3" * 64,
        "semantic_result_fingerprint": "4" * 64,
    }
    core = {
        "schema_version": 4,
        "model_id": "yap-extra-trees-a-holdout-b-adaptive-v1",
        "target_id": "yap",
        "architecture": "extra_trees",
        "model_version": "adaptive-v1",
        "training_cohorts": ["a"],
        "prediction_cohorts": ["b"],
        "cohort_bindings": {"a": binding, "b": binding},
        "measurement_view": ("tissue-masked-adaptive-corrected-target-od-4um-v1"),
        "model": "bundle.json",
        "training_table": "table.npz",
        "model_artifacts": [],
        "slides": [{"cohort": "b", "section": "001", "predictions": "b.npz"}],
    }

    write_protein_result(tmp_path, core)
    result = validate_protein_result(tmp_path)
    assert result["prediction_cohorts"] == ["b"]
    assert {row["cohort"] for row in result["slides"]} == {"b"}
    assert set(result["cohort_bindings"]) == {"a", "b"}

    invalid = dict(core)
    invalid["prediction_cohorts"] = ["missing"]
    with pytest.raises(ValueError, match="prediction cohorts"):
        write_protein_result(tmp_path, invalid)


def test_study_quantile_calibration_reduces_global_overprediction() -> None:
    from histopia.protein._study_workflow import (
        _apply_quantile_calibration,
        _fit_quantile_calibration,
    )

    predicted = np.linspace(0.0, 3.0, 200, dtype=np.float32)
    measured = np.linspace(0.0, 0.9, 200, dtype=np.float32) ** 1.3
    source, target = _fit_quantile_calibration(predicted, measured)
    calibrated = _apply_quantile_calibration(predicted, source, target)

    assert np.all(np.diff(calibrated) >= 0)
    assert np.quantile(calibrated, 0.9) == pytest.approx(
        np.quantile(measured, 0.9), rel=0.02
    )
    assert calibrated.mean() < predicted.mean() * 0.5


def test_training_visible_metrics_retain_held_out_reference() -> None:
    from histopia.protein._study_workflow import _training_visible_metrics

    table = _table()
    selected = np.arange(len(table.measured_od), dtype=np.int64)
    held_out = {
        "schema_version": 3,
        "evaluation": "leave_one_mouse_out",
        "available": True,
        "spearman": 0.4,
        "spearman_64um": 0.5,
        "folds": [],
    }

    metrics = _training_visible_metrics(
        table,
        selected,
        table.measured_od.copy(),
        held_out_reference=held_out,
    )

    assert metrics["evaluation"] == "training_visible_fit"
    assert metrics["training_cells"] == len(selected)
    assert metrics["mae"] == pytest.approx(0)
    assert metrics["spearman"] == pytest.approx(1)
    assert metrics["spearman_64um"] == pytest.approx(1)
    assert metrics["held_out_reference"] is held_out


def test_study_prediction_roles_make_training_visibility_explicit() -> None:
    from histopia.protein._study_workflow import _study_prediction_role

    common = {
        "training_mice": ("4312", "6180"),
        "target_sections": ("004",),
        "prediction_protocol": "training-visible",
    }
    assert (
        _study_prediction_role(mouse_id="4312", section="004", **common)
        == "training-visible-fit"
    )
    assert (
        _study_prediction_role(mouse_id="4312", section="006", **common)
        == "training-visible-transfer"
    )
    assert (
        _study_prediction_role(mouse_id="4630", section="004", **common)
        == "external-transfer"
    )
    assert (
        _study_prediction_role(
            mouse_id="4312",
            section="004",
            **{**common, "prediction_protocol": "leave-one-mouse-out"},
        )
        == "leave-one-mouse-out"
    )


def test_membrane_measurement_uses_four_micron_inward_band() -> None:
    labels = np.zeros((9, 9), dtype=np.uint32)
    labels[1:8, 1:8] = 1

    boundary = compartment_sampled_labels(
        labels,
        "inner_boundary",
        source_mpp_xy=np.asarray([0.5, 0.5]),
        pyramid_scale=4.0,
    )

    assert boundary[1, 4] == 1
    assert boundary[2, 4] == 1
    assert boundary[3, 4] == 0
    np.testing.assert_array_equal(
        compartment_sampled_labels(
            labels,
            "whole_cell",
            source_mpp_xy=np.asarray([0.5, 0.5]),
            pyramid_scale=4.0,
        ),
        labels,
    )
    expected = sampled_label_expected_analysis_pixels(
        boundary,
        cell_count=1,
        source_mpp_xy=np.asarray([0.5, 0.5]),
        pyramid_scale=4.0,
        analysis_mpp=4.0,
    )
    assert expected[1] == pytest.approx(np.count_nonzero(boundary) / 4.0)


def test_extra_trees_batched_prediction_preserves_member_statistics() -> None:
    from sklearn.ensemble import ExtraTreesRegressor

    from histopia.protein._study_workflow import _predict_extra_trees

    features = np.arange(144, dtype=np.float32).reshape(24, 6) / 13.0
    measured = np.sin(features[:, 0]) + 1.0
    estimator = ExtraTreesRegressor(
        n_estimators=7,
        min_samples_leaf=2,
        random_state=3,
        n_jobs=-1,
    ).fit(features, measured)
    expected = np.stack([tree.predict(features) for tree in estimator.estimators_])

    predicted, uncertainty = _predict_extra_trees(
        estimator,
        features,
        batch_size=5,
    )

    np.testing.assert_allclose(predicted, expected.mean(axis=0), rtol=1e-6)
    np.testing.assert_allclose(uncertainty, expected.std(axis=0), rtol=1e-6)


def test_study_resume_accepts_only_exact_fingerprinted_prediction() -> None:
    from histopia.protein._study_workflow import _validate_resumable_prediction

    prediction = ProteinPredictions(
        target_id="ck19",
        model_fingerprint="1" * 64,
        label_ids=np.asarray([1, 2], dtype=np.uint32),
        section_ids=np.asarray(["001", "001"]),
        expression_probability=np.asarray([np.nan, np.nan], dtype=np.float32),
        relative_expression=np.asarray([0.2, 0.7], dtype=np.float32),
        predicted_od_reference=np.asarray([0.1, 0.6], dtype=np.float32),
        measured_od=np.asarray([np.nan, np.nan], dtype=np.float32),
        uncertainty=np.asarray([0.1, 0.2], dtype=np.float32),
        supported=np.asarray([True, True]),
        provenance={
            "feature_view": "native-hdab-neutral-spatial-uni2h-v2",
            "measurement_view": ("tissue-masked-adaptive-corrected-target-od-4um-v1"),
            "cohort": "4312",
            "architecture": "extra_trees",
            "fold_model_fingerprint": "2" * 64,
            "evaluation_role": "leave-one-mouse-out",
        },
    )
    options = {
        "target_id": "ck19",
        "model_fingerprint": "1" * 64,
        "fold_model_fingerprint": "2" * 64,
        "architecture": "extra_trees",
        "cohort": "4312",
        "section": "001",
        "cell_count": 2,
        "evaluation_role": "leave-one-mouse-out",
    }

    _validate_resumable_prediction(prediction, **options)
    with pytest.raises(ValueError, match="stale"):
        _validate_resumable_prediction(
            prediction,
            **{**options, "cohort": "4630"},
        )


def test_equal_study_ensemble_seals_parent_models_and_cell_predictions(
    tmp_path: Path,
) -> None:
    from histopia.protein import ensemble_protein_study_results

    table = _table()
    binding = {
        "registration_result_sha256": "1" * 64,
        "cell_result_fingerprint": "2" * 64,
        "stain_result_fingerprint": "3" * 64,
        "semantic_result_fingerprint": "4" * 64,
    }

    def parent(name: str, scale: float, architecture: str) -> Path:
        root = tmp_path / name
        root.mkdir()
        table.save(root / "table.npz")
        (root / "model.json").write_text("{}")
        (root / "audit.json").write_text("{}")
        slides = []
        for cohort, section in (("mouse-a", "001"), ("mouse-b", "002")):
            selected = np.flatnonzero(table.mouse_ids == cohort)
            values = table.measured_od[selected] * scale
            prediction = ProteinPredictions(
                target_id="perk",
                model_fingerprint=("a" if name == "left" else "b") * 64,
                label_ids=table.label_ids[selected],
                section_ids=table.section_ids[selected],
                expression_probability=np.full(len(selected), np.nan, np.float32),
                relative_expression=np.linspace(0, 1, len(selected), dtype=np.float32),
                predicted_od_reference=values,
                measured_od=table.measured_od[selected],
                uncertainty=np.full(len(selected), 0.1, np.float32),
                supported=np.ones(len(selected), dtype=bool),
                provenance={
                    "feature_view": "native-hdab-neutral-spatial-uni2h-v2",
                    "measurement_view": (
                        "tissue-masked-adaptive-corrected-target-od-4um-v1"
                    ),
                    "cohort": cohort,
                    "architecture": architecture,
                    "evaluation_role": "leave-one-mouse-out",
                },
            )
            path = prediction.save(root / "predictions" / cohort / f"{section}.npz")
            slides.append(
                {
                    "cohort": cohort,
                    "section": section,
                    "predictions": path.relative_to(root).as_posix(),
                    "prediction_fingerprint": prediction.fingerprint,
                    "cells": len(selected),
                    "measured_cells": len(selected),
                    "evaluation_role": "leave-one-mouse-out",
                }
            )
        write_protein_result(
            root,
            {
                "schema_version": 4,
                "model_id": f"perk-{name}-adaptive-v1",
                "target_id": "perk",
                "architecture": architecture,
                "model_version": "adaptive-v1",
                "training_cohorts": ["mouse-a", "mouse-b"],
                "cohort_bindings": {"mouse-a": binding, "mouse-b": binding},
                "measurement_view": (
                    "tissue-masked-adaptive-corrected-target-od-4um-v1"
                ),
                "assay_domain": "synthetic",
                "model": "model.json",
                "training_table": "table.npz",
                "measurement_audit": "audit.json",
                "model_artifacts": [],
                "model_fingerprint": ("a" if name == "left" else "b") * 64,
                "table_fingerprint": table.fingerprint,
                "binary_enabled": False,
                "slides": slides,
            },
        )
        return root

    left = parent("left", 0.8, "extra_trees")
    right = parent("right", 1.2, "cross_attention")
    output = tmp_path / "ensemble"

    ensemble_protein_study_results((left, right), output)
    result = validate_protein_result(output)
    prediction = ProteinPredictions.load(output / result["slides"][0]["predictions"])
    selected = np.flatnonzero(table.mouse_ids == result["slides"][0]["cohort"])

    assert result["architecture"] == "equal_ensemble"
    assert result["status"] == "promoted"
    assert len(result["model_artifacts"]) == 2
    np.testing.assert_allclose(
        prediction.predicted_od_reference,
        table.measured_od[selected],
        rtol=1e-6,
    )
    assert prediction.provenance["parent_model_fingerprints"] == [
        "a" * 64,
        "b" * 64,
    ]


def test_target_free_tail_switch_matches_frozen_query_rule(tmp_path: Path) -> None:
    switch = TargetFreeTailSwitch(
        component_bundle_fingerprints=("1" * 64, "2" * 64, "3" * 64),
        component_bundle_sha256=("4" * 64, "5" * 64, "6" * 64),
    )
    base = np.asarray([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1])
    tail = np.asarray([0.2, 0.3, 0.5, 0.6, 0.8, 1.0, 1.3, 1.6])
    graph = np.asarray([0.3, 0.4, 0.6, 0.7, 0.9, 1.1, 1.5, 2.0])
    result = switch.combine(
        base,
        tail,
        graph,
        base_uncertainty=np.full(8, 0.1),
        tail_uncertainty=np.full(8, 0.2),
        graph_uncertainty=np.full(8, 0.3),
    )

    signal = 0.5 * (base + graph)
    lower, upper = np.quantile(signal, (0.25, 0.90))
    expected = base.copy()
    low = signal <= lower
    high = signal >= upper
    expected[low] = np.minimum(base[low], tail[low])
    high_target = np.maximum.reduce((base[high], tail[high], graph[high]))
    expected[high] += 0.4 * (high_target - expected[high])

    np.testing.assert_allclose(result.predicted_od, expected)
    np.testing.assert_array_equal(result.low_switched, low)
    np.testing.assert_array_equal(result.high_switched, high)
    assert np.all(result.uncertainty >= 0)
    path = switch.save(tmp_path / "tail-switch.json")
    loaded = TargetFreeTailSwitch.load(path)
    assert loaded.fingerprint == switch.fingerprint
    np.testing.assert_allclose(
        loaded.combine(base, tail, graph).predicted_od,
        expected,
    )


def test_target_free_tail_switch_uses_query_group_quantiles() -> None:
    switch = TargetFreeTailSwitch(
        component_bundle_fingerprints=("1" * 64, "2" * 64, "3" * 64),
        component_bundle_sha256=("4" * 64, "5" * 64, "6" * 64),
    )
    base = np.asarray([0.1, 0.2, 0.3, 0.4, 10.0, 20.0, 30.0, 40.0])
    tail = base + np.asarray([-0.05, 0.0, 0.0, 1.0] * 2)
    graph = base + np.asarray([0.0, 0.0, 0.0, 2.0] * 2)
    groups = np.asarray(["a"] * 4 + ["b"] * 4)

    result = switch.combine(base, tail, graph, group_ids=groups)

    np.testing.assert_array_equal(
        result.low_switched,
        [True, False, False, False, True, False, False, False],
    )
    np.testing.assert_array_equal(
        result.high_switched,
        [False, False, False, True, False, False, False, True],
    )
    assert result.predicted_od[0] == pytest.approx(0.05)
    assert result.predicted_od[4] == pytest.approx(9.95)
    assert result.predicted_od[3] == pytest.approx(1.2)
    assert result.predicted_od[7] == pytest.approx(40.8)
