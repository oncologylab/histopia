"""Stain-invariant, semantic-guided per-cell protein prediction."""

from importlib import import_module

_PUBLIC_IMPORTS = {
    "CellCorrespondence": (
        "histopia.protein._vector_transfer",
        "CellCorrespondence",
    ),
    "ProteinVectorTransfer": (
        "histopia.protein._vector_transfer",
        "ProteinVectorTransfer",
    ),
    "match_registered_cells": (
        "histopia.protein._vector_transfer",
        "match_registered_cells",
    ),
    "transfer_protein_vectors": (
        "histopia.protein._vector_transfer",
        "transfer_protein_vectors",
    ),
    "CellTokenFeatures": (
        "histopia.protein._cell_features",
        "CellTokenFeatures",
    ),
    "VipsPatchReader": (
        "histopia.protein._cell_features",
        "VipsPatchReader",
    ),
    "CellExpressionTable": ("histopia.protein._result", "CellExpressionTable"),
    "CellFeatureSet": (
        "histopia.protein._multiscale",
        "CellFeatureSet",
    ),
    "AdaptiveTargetMeasurement": (
        "histopia.protein._real_data",
        "AdaptiveTargetMeasurement",
    ),
    "RegisteredOdCalibration": (
        "histopia.protein._calibration",
        "RegisteredOdCalibration",
    ),
    "RegisteredAnchorTransfer": (
        "histopia.protein._transfer",
        "RegisteredAnchorTransfer",
    ),
    "EqualSectionOdHarmonization": (
        "histopia.protein._calibration",
        "EqualSectionOdHarmonization",
    ),
    "RealProteinStudyCohort": (
        "histopia.protein._study_workflow",
        "RealProteinStudyCohort",
    ),
    "SectionOdCalibration": (
        "histopia.protein._calibration",
        "SectionOdCalibration",
    ),
    "ProteinModel": ("histopia.protein._model", "ProteinModel"),
    "PortableCrossAttentionRegressor": (
        "histopia.protein._attention",
        "PortableCrossAttentionRegressor",
    ),
    "PortableMultiTowerRegressor": (
        "histopia.protein._advanced",
        "PortableMultiTowerRegressor",
    ),
    "PortableRelationalRegressor": (
        "histopia.protein._relational",
        "PortableRelationalRegressor",
    ),
    "PortableSharedMultitaskRegressor": (
        "histopia.protein._deep",
        "PortableSharedMultitaskRegressor",
    ),
    "ProteinPredictionConfig": (
        "histopia.protein._config",
        "ProteinPredictionConfig",
    ),
    "ProteinPredictions": ("histopia.protein._result", "ProteinPredictions"),
    "TailSwitchResult": ("histopia.protein._ensemble", "TailSwitchResult"),
    "TargetFreeTailSwitch": (
        "histopia.protein._ensemble",
        "TargetFreeTailSwitch",
    ),
    "ProteinTarget": ("histopia.protein._config", "ProteinTarget"),
    "SpatialSplit": ("histopia.protein._splits", "SpatialSplit"),
    "aggregate_cell_measurements": (
        "histopia.protein._measurements",
        "aggregate_cell_measurements",
    ),
    "approve_protein_result": (
        "histopia.protein._manifest",
        "approve_protein_result",
    ),
    "build_spatial_split": ("histopia.protein._splits", "build_spatial_split"),
    "evaluate_predictions": ("histopia.protein._model", "evaluate_predictions"),
    "evaluate_stain_invariance": (
        "histopia.protein._model",
        "evaluate_stain_invariance",
    ),
    "ensemble_protein_study_results": (
        "histopia.protein._study_workflow",
        "ensemble_protein_study_results",
    ),
    "extend_cell_expression_table": (
        "histopia.protein._result",
        "extend_cell_expression_table",
    ),
    "extract_cell_token_features": (
        "histopia.protein._cell_features",
        "extract_cell_token_features",
    ),
    "cell_neighborhood_features": (
        "histopia.protein._cell_features",
        "cell_neighborhood_features",
    ),
    "cell_shape_features": (
        "histopia.protein._cell_features",
        "cell_shape_features",
    ),
    "confidence_weighted_registered_residual_correction": (
        "histopia.protein._calibration",
        "confidence_weighted_registered_residual_correction",
    ),
    "cell_phenotype_features": (
        "histopia.protein._multiscale",
        "cell_phenotype_features",
    ),
    "build_cell_feature_set": (
        "histopia.protein._multiscale",
        "build_cell_feature_set",
    ),
    "create_protein_sweep": (
        "histopia.protein._sweep",
        "create_protein_sweep",
    ),
    "compartment_sampled_labels": (
        "histopia.protein._real_data",
        "compartment_sampled_labels",
    ),
    "fit_protein_model": ("histopia.protein._model", "fit_protein_model"),
    "fit_portable_cross_attention": (
        "histopia.protein._attention",
        "fit_portable_cross_attention",
    ),
    "fit_portable_multi_tower": (
        "histopia.protein._advanced",
        "fit_portable_multi_tower",
    ),
    "fit_portable_relational": (
        "histopia.protein._relational",
        "fit_portable_relational",
    ),
    "fit_portable_shared_multitask": (
        "histopia.protein._deep",
        "fit_portable_shared_multitask",
    ),
    "fit_real_protein_study_variant": (
        "histopia.protein._study_workflow",
        "fit_real_protein_study_variant",
    ),
    "fit_registered_od_calibration": (
        "histopia.protein._calibration",
        "fit_registered_od_calibration",
    ),
    "fixed_stain_neutral_projection": (
        "histopia.protein._transfer",
        "fixed_stain_neutral_projection",
    ),
    "fit_equal_section_od_harmonization": (
        "histopia.protein._calibration",
        "fit_equal_section_od_harmonization",
    ),
    "ProteinPromotionDecision": (
        "histopia.protein._selection",
        "ProteinPromotionDecision",
    ),
    "ProteinCandidateScore": (
        "histopia.protein._selection",
        "ProteinCandidateScore",
    ),
    "ProteinCandidateSelection": (
        "histopia.protein._selection",
        "ProteinCandidateSelection",
    ),
    "evaluate_protein_promotion": (
        "histopia.protein._selection",
        "evaluate_protein_promotion",
    ),
    "select_protein_architecture_candidate": (
        "histopia.protein._selection",
        "select_protein_architecture_candidate",
    ),
    "out_of_fold_deep_predictions": (
        "histopia.protein._deep",
        "out_of_fold_deep_predictions",
    ),
    "leave_one_section_out_consensus": (
        "histopia.protein._features",
        "leave_one_section_out_consensus",
    ),
    "load_real_protein_study_manifest": (
        "histopia.protein._study_workflow",
        "load_real_protein_study_manifest",
    ),
    "label_centroids_from_tiff": (
        "histopia.protein._cell_features",
        "label_centroids_from_tiff",
    ),
    "load_protein_config": ("histopia.protein._config", "load_protein_config"),
    "normalize_target_id": ("histopia.protein._config", "normalize_target_id"),
    "morphology_compatible_smoothing": (
        "histopia.protein._transfer",
        "morphology_compatible_smoothing",
    ),
    "morphospatial_features": (
        "histopia.protein._transfer",
        "morphospatial_features",
    ),
    "multiscale_neighborhood_features": (
        "histopia.protein._multiscale",
        "multiscale_neighborhood_features",
    ),
    "neutralize_hdab_morphology": (
        "histopia.protein._cell_features",
        "neutralize_hdab_morphology",
    ),
    "pool_patch_features_to_cells": (
        "histopia.protein._features",
        "pool_patch_features_to_cells",
    ),
    "protein_sweep_status": (
        "histopia.protein._sweep",
        "protein_sweep_status",
    ),
    "prepare_real_protein_study_table": (
        "histopia.protein._study_workflow",
        "prepare_real_protein_study_table",
    ),
    "prepare_real_protein_study_features": (
        "histopia.protein._study_workflow",
        "prepare_real_protein_study_features",
    ),
    "render_neutral_morphology": (
        "histopia.protein._features",
        "render_neutral_morphology",
    ),
    "registered_position_features": (
        "histopia.protein._multiscale",
        "registered_position_features",
    ),
    "registered_morphospatial_anchor_transfer": (
        "histopia.protein._transfer",
        "registered_morphospatial_anchor_transfer",
    ),
    "run_protein_sweep_worker": (
        "histopia.protein._sweep",
        "run_protein_sweep_worker",
    ),
    "reassess_protein_result": (
        "histopia.protein._manifest",
        "reassess_protein_result",
    ),
    "sampled_label_expected_analysis_pixels": (
        "histopia.protein._real_data",
        "sampled_label_expected_analysis_pixels",
    ),
    "select_bracketing_sections": (
        "histopia.protein._transfer",
        "select_bracketing_sections",
    ),
    "stream_cell_phenotype_features": (
        "histopia.protein._multiscale",
        "stream_cell_phenotype_features",
    ),
    "tail_guarded_registered_anchor_blend": (
        "histopia.protein._calibration",
        "tail_guarded_registered_anchor_blend",
    ),
    "select_protein_sweep_winners": (
        "histopia.protein._sweep",
        "select_protein_sweep_winners",
    ),
    "validate_protein_approval": (
        "histopia.protein._manifest",
        "validate_protein_approval",
    ),
    "validate_protein_result": (
        "histopia.protein._manifest",
        "validate_protein_result",
    ),
    "validated_adaptive_target_measurement": (
        "histopia.protein._real_data",
        "validated_adaptive_target_measurement",
    ),
    "write_protein_result": (
        "histopia.protein._manifest",
        "write_protein_result",
    ),
}


def __getattr__(name: str):
    try:
        module_name, attribute = _PUBLIC_IMPORTS[name]
    except KeyError as error:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from error
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


__all__ = sorted(_PUBLIC_IMPORTS)
