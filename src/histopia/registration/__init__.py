"""Registration utilities for serial-section histology images."""

from importlib import import_module

_PUBLIC_IMPORTS = {
    "AlignmentMetrics": ("histopia.registration._pipeline", "AlignmentMetrics"),
    "BrightfieldMaskConfig": (
        "histopia.registration._config",
        "BrightfieldMaskConfig",
    ),
    "CavityContinuitySummary": (
        "histopia.registration._ordering",
        "CavityContinuitySummary",
    ),
    "DuplicateSlideContent": (
        "histopia.registration._slides",
        "DuplicateSlideContent",
    ),
    "PhysicalAreaContinuitySummary": (
        "histopia.registration._ordering",
        "PhysicalAreaContinuitySummary",
    ),
    "GroupOrientation": (
        "histopia.registration._orientation",
        "GroupOrientation",
    ),
    "KpfManifest": ("histopia.registration._manifest", "KpfManifest"),
    "MaskRefinementConfig": (
        "histopia.registration._config",
        "MaskRefinementConfig",
    ),
    "load_registration_config": (
        "histopia.registration._config",
        "load_registration_config",
    ),
    "MaskReviewApproval": (
        "histopia.registration._approval",
        "MaskReviewApproval",
    ),
    "MaskReviewEntry": ("histopia.registration._review", "MaskReviewEntry"),
    "NonRigidRefinementConfig": (
        "histopia.registration._config",
        "NonRigidRefinementConfig",
    ),
    "NonRigidTransformResult": (
        "histopia.registration._nonrigid",
        "NonRigidTransformResult",
    ),
    "SparseFeatureValidation": (
        "histopia.registration._nonrigid",
        "SparseFeatureValidation",
    ),
    "OrientationDecision": (
        "histopia.registration._orientation",
        "OrientationDecision",
    ),
    "RegistrationApproval": (
        "histopia.registration._approval",
        "RegistrationApproval",
    ),
    "RegistrationConfig": ("histopia.registration._config", "RegistrationConfig"),
    "RegistrationResult": (
        "histopia.registration._pipeline",
        "RegistrationResult",
    ),
    "RigidTransformResult": (
        "histopia.registration._rigid",
        "RigidTransformResult",
    ),
    "SectionOrderApproval": (
        "histopia.registration._approval",
        "SectionOrderApproval",
    ),
    "SectionOrderProposal": (
        "histopia.registration._ordering",
        "SectionOrderProposal",
    ),
    "SlideGeometry": ("histopia.registration._slides", "SlideGeometry"),
    "SlidePair": ("histopia.registration._manifest", "SlidePair"),
    "SlideRecord": ("histopia.registration._slides", "SlideRecord"),
    "WsiWarpResult": ("histopia.registration._wsi", "WsiWarpResult"),
    "apply_quarter_turn": (
        "histopia.registration._orientation",
        "apply_quarter_turn",
    ),
    "approve_mask_review": (
        "histopia.registration._approval",
        "approve_mask_review",
    ),
    "approve_registration_run": (
        "histopia.registration._approval",
        "approve_registration_run",
    ),
    "approve_section_order": (
        "histopia.registration._approval",
        "approve_section_order",
    ),
    "build_kpf_manifest": (
        "histopia.registration._manifest",
        "build_kpf_manifest",
    ),
    "calculate_thumbnail_overlap_bbox": (
        "histopia.registration._wsi",
        "calculate_thumbnail_overlap_bbox",
    ),
    "create_tissue_mask": (
        "histopia.registration._masking",
        "create_tissue_mask",
    ),
    "discover_slides": ("histopia.registration._slides", "discover_slides"),
    "estimate_non_rigid_transform": (
        "histopia.registration._nonrigid",
        "estimate_non_rigid_transform",
    ),
    "evaluate_non_rigid_feature_holdout": (
        "histopia.registration._nonrigid",
        "evaluate_non_rigid_feature_holdout",
    ),
    "estimate_rigid_transform": (
        "histopia.registration._rigid",
        "estimate_rigid_transform",
    ),
    "find_duplicate_slide_content": (
        "histopia.registration._slides",
        "find_duplicate_slide_content",
    ),
    "geometry_thumbnail_to_native_matrix": (
        "histopia.registration._wsi",
        "geometry_thumbnail_to_native_matrix",
    ),
    "normalize_slide_stem": (
        "histopia.registration._manifest",
        "normalize_slide_stem",
    ),
    "orient_section_group": (
        "histopia.registration._orientation",
        "orient_section_group",
    ),
    "refine_group_tissue_masks": (
        "histopia.registration._masking",
        "refine_group_tissue_masks",
    ),
    "registration_config_from_mapping": (
        "histopia.registration._config",
        "registration_config_from_mapping",
    ),
    "refine_rigid_transform": (
        "histopia.registration._rigid",
        "refine_rigid_transform",
    ),
    "register_sections": (
        "histopia.registration._pipeline",
        "register_sections",
    ),
    "summarize_cavity_continuity": (
        "histopia.registration._ordering",
        "summarize_cavity_continuity",
    ),
    "summarize_physical_area_continuity": (
        "histopia.registration._ordering",
        "summarize_physical_area_continuity",
    ),
    "thumbnail_to_full_resolution_matrix": (
        "histopia.registration._wsi",
        "thumbnail_to_full_resolution_matrix",
    ),
    "validate_registration_approval": (
        "histopia.registration._approval",
        "validate_registration_approval",
    ),
    "validate_unique_slide_content": (
        "histopia.registration._slides",
        "validate_unique_slide_content",
    ),
    "warp_saved_registration": (
        "histopia.registration._wsi",
        "warp_saved_registration",
    ),
    "warp_slide_to_reference": (
        "histopia.registration._wsi",
        "warp_slide_to_reference",
    ),
    "warp_with_displacement": (
        "histopia.registration._nonrigid",
        "warp_with_displacement",
    ),
}


def __getattr__(name: str):
    try:
        module_name, attribute = _PUBLIC_IMPORTS[name]
    except KeyError as error:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message) from error
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


__all__ = sorted(_PUBLIC_IMPORTS)
