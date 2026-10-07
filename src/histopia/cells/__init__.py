"""Native-space cell-boundary detection for registered serial sections."""

from importlib import import_module

_PUBLIC_IMPORTS = {
    "ReviewedInputPatch": (
        "histopia.cells._reviewed_input_addition",
        "ReviewedInputPatch",
    ),
    "compose_reviewed_input_addition": (
        "histopia.cells._reviewed_input_addition_compose",
        "compose_reviewed_input_addition",
    ),
    "CellApproval": ("histopia.cells._approval", "CellApproval"),
    "ChromaticMicroclusterPolicy": (
        "histopia.cells._chromatic",
        "ChromaticMicroclusterPolicy",
    ),
    "CellPreflight": ("histopia.cells._preflight", "CellPreflight"),
    "CellPreflightSlide": ("histopia.cells._preflight", "CellPreflightSlide"),
    "CellSegmentationConfig": ("histopia.cells._config", "CellSegmentationConfig"),
    "compose_cell_result": ("histopia.cells._composition", "compose_cell_result"),
    "compose_reviewed_cache_addition": (
        "histopia.cells._reviewed_addition_compose",
        "compose_reviewed_cache_addition",
    ),
    "ReviewedCachePatch": (
        "histopia.cells._reviewed_multi_addition",
        "ReviewedCachePatch",
    ),
    "compose_reviewed_multi_cache_addition": (
        "histopia.cells._reviewed_multi_addition_compose",
        "compose_reviewed_multi_cache_addition",
    ),
    "FocusArtifactComponent": (
        "histopia.cells._focus",
        "FocusArtifactComponent",
    ),
    "FocusArtifactPolicy": ("histopia.cells._focus", "FocusArtifactPolicy"),
    "SaturatedCyanForeignMaterialPolicy": (
        "histopia.cells._foreign_material",
        "SaturatedCyanForeignMaterialPolicy",
    ),
    "approve_cell_result": ("histopia.cells._approval", "approve_cell_result"),
    "benchmark_cell_methods": (
        "histopia.cells._benchmark",
        "benchmark_cell_methods",
    ),
    "blue_counterstain_core": (
        "histopia.cells._chromatic",
        "blue_counterstain_core",
    ),
    "cell_review_status": ("histopia.cells._approval", "cell_review_status"),
    "chromatic_microcluster_gate": (
        "histopia.cells._chromatic",
        "chromatic_microcluster_gate",
    ),
    "clustered_chromatic_microobject_instances": (
        "histopia.cells._chromatic",
        "clustered_chromatic_microobject_instances",
    ),
    "centroid_instance_metrics": (
        "histopia.cells._benchmark",
        "centroid_instance_metrics",
    ),
    "containment_merge": ("histopia.cells._algorithms", "containment_merge"),
    "constrain_labels_to_tissue": (
        "histopia.cells._algorithms",
        "constrain_labels_to_tissue",
    ),
    "export_qupath_cell_roi": (
        "histopia.cells._qupath",
        "export_qupath_cell_roi",
    ),
    "detect_focus_artifact_blocks": (
        "histopia.cells._focus",
        "detect_focus_artifact_blocks",
    ),
    "grow_focus_artifact_regions": (
        "histopia.cells._focus",
        "grow_focus_artifact_regions",
    ),
    "labels_touching_mask": (
        "histopia.cells._focus",
        "labels_touching_mask",
    ),
    "load_cell_config": ("histopia.cells._config", "load_cell_config"),
    "merge_small_into_large": (
        "histopia.cells._algorithms",
        "merge_small_into_large",
    ),
    "preflight_cell_run": ("histopia.cells._preflight", "preflight_cell_run"),
    "refilter_chromatic_microclusters": (
        "histopia.cells._chromatic_refilter",
        "refilter_chromatic_microclusters",
    ),
    "refilter_saturated_cyan_foreign_material": (
        "histopia.cells._foreign_material_refilter",
        "refilter_saturated_cyan_foreign_material",
    ),
    "refilter_reviewed_artifact_exclusions": (
        "histopia.cells._reviewed_exclusion_refilter",
        "refilter_reviewed_artifact_exclusions",
    ),
    "remove_labels_touching_mask": (
        "histopia.cells._focus",
        "remove_labels_touching_mask",
    ),
    "review_cell_section": ("histopia.cells._approval", "review_cell_section"),
    "run_cell_segmentation": (
        "histopia.cells._pipeline",
        "run_cell_segmentation",
    ),
    "saturated_cyan_foreign_material_core": (
        "histopia.cells._foreign_material",
        "saturated_cyan_foreign_material_core",
    ),
    "saturated_cyan_foreign_material_gate": (
        "histopia.cells._foreign_material",
        "saturated_cyan_foreign_material_gate",
    ),
    "saturated_cyan_foreign_material_instances": (
        "histopia.cells._foreign_material",
        "saturated_cyan_foreign_material_instances",
    ),
    "reviewed_artifact_exclusion_manifest_sha256": (
        "histopia.cells._reviewed_exclusion",
        "reviewed_artifact_exclusion_manifest_sha256",
    ),
    "reviewed_cache_addition_manifest_sha256": (
        "histopia.cells._reviewed_addition",
        "reviewed_cache_addition_manifest_sha256",
    ),
    "validate_reviewed_artifact_exclusion_manifest": (
        "histopia.cells._reviewed_exclusion",
        "validate_reviewed_artifact_exclusion_manifest",
    ),
    "validate_reviewed_cache_addition_manifest": (
        "histopia.cells._reviewed_addition",
        "validate_reviewed_cache_addition_manifest",
    ),
    "summarize_benchmark": ("histopia.cells._benchmark", "summarize_benchmark"),
    "validate_cell_approval": (
        "histopia.cells._approval",
        "validate_cell_approval",
    ),
    "validate_cell_result": ("histopia.cells._result", "validate_cell_result"),
    "validate_cell_promotion_candidate": (
        "histopia.cells._promotion",
        "validate_cell_promotion_candidate",
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
