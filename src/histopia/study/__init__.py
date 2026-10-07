"""Versioned, evidence-bound study analyses; optional dependencies load on demand."""

from importlib import import_module

_EXPORTS = {
    "validate_external_benchmark": (
        "_external_benchmark",
        "validate_external_benchmark",
    ),
    "validate_external_job": ("_external_benchmark", "validate_external_job"),
    "ExternalBenchmarkLedger": ("_external_benchmark", "ExternalBenchmarkLedger"),
    "boundary_overlap_counts": ("_boundary_counts", "boundary_overlap_counts"),
    "validate_reconstruction_job": ("_compute", "validate_reconstruction_job"),
    "ReconstructionJobLedger": ("_compute", "ReconstructionJobLedger"),
    "assess_reconstruction_eligibility": (
        "_eligibility",
        "assess_reconstruction_eligibility",
    ),
    "validate_reconstruction_registry": (
        "_eligibility",
        "validate_reconstruction_registry",
    ),
    "filter_reconstruction_catalog": ("_eligibility", "filter_reconstruction_catalog"),
    "validate_serial_study": ("_serial", "validate_serial_study"),
    "RangeReader": ("_remote_tiff", "RangeReader"),
    "read_tiled_region": ("_remote_tiff", "read_tiled_region"),
    "TissueRegions": ("_regions", "TissueRegions"),
    "RegionAssignments": ("_regions", "RegionAssignments"),
    "NeighborhoodResult": ("_neighborhoods", "NeighborhoodResult"),
    "AssignmentLedger": ("_ledger", "AssignmentLedger"),
    "prepare_he_rgb": ("_preprocessing", "prepare_he_rgb"),
    "sample_region_labels": ("_regions", "sample_region_labels"),
    "assign_cells_from_overlap": ("_regions", "assign_cells_from_overlap"),
    "write_summary_tables": ("_regions", "write_summary_tables"),
    "write_analysis_arrays": ("_artifacts", "write_analysis_arrays"),
    "load_analysis_arrays": ("_artifacts", "load_analysis_arrays"),
    "evaluate_protein_vectors": ("_evaluation", "evaluate_protein_vectors"),
    "annotation_review_sample": ("_labels", "annotation_review_sample"),
    "evaluate_cell_labels": ("_labels", "evaluate_cell_labels"),
    "marker_prior_coverage": ("_astir", "marker_prior_coverage"),
    "fit_astir_comparator": ("_astir", "fit_astir_comparator"),
    "within_tissue_permutation": ("_neighborhoods", "within_tissue_permutation"),
    "niche_stability": ("_neighborhoods", "niche_stability"),
    "hpa_ihc_inventory": ("_external", "hpa_ihc_inventory"),
    "load_study_manifest": ("_manifest", "load_study_manifest"),
    "write_study_manifest": ("_manifest", "write_study_manifest"),
    "validate_study_manifest": ("_manifest", "validate_study_manifest"),
    "validate_fit_scope": ("_manifest", "validate_fit_scope"),
    "feature_identity": ("_manifest", "feature_identity"),
    "connected_tissue_regions": ("_regions", "connected_tissue_regions"),
    "assign_cells_by_overlap": ("_regions", "assign_cells_by_overlap"),
    "region_expression_summary": ("_regions", "region_expression_summary"),
    "CellLabelProbabilities": ("_labels", "CellLabelProbabilities"),
    "interpolate_label_probabilities": ("_labels", "interpolate_label_probabilities"),
    "physical_neighborhoods": ("_neighborhoods", "physical_neighborhoods"),
    "build_study_figure": ("_figure", "build_study_figure"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(f"histopia.study.{module}"), attribute)
    globals()[name] = value
    return value
