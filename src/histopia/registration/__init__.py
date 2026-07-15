"""Registration utilities for serial-section histology images."""

from histopia.registration._config import (
    BrightfieldMaskConfig,
    MaskRefinementConfig,
    NonRigidRefinementConfig,
    RegistrationConfig,
)
from histopia.registration._manifest import (
    KpfManifest,
    SlidePair,
    build_kpf_manifest,
    normalize_slide_stem,
)
from histopia.registration._masking import create_tissue_mask, refine_group_tissue_masks
from histopia.registration._nonrigid import (
    NonRigidTransformResult,
    estimate_non_rigid_transform,
    warp_with_displacement,
)
from histopia.registration._ordering import SectionOrderProposal
from histopia.registration._orientation import (
    GroupOrientation,
    OrientationDecision,
    apply_quarter_turn,
    orient_section_group,
)
from histopia.registration._pipeline import (
    AlignmentMetrics,
    RegistrationResult,
    register_sections,
)
from histopia.registration._review import MaskReviewEntry
from histopia.registration._rigid import (
    RigidTransformResult,
    estimate_rigid_transform,
    refine_rigid_transform,
)
from histopia.registration._slides import SlideGeometry, SlideRecord, discover_slides
from histopia.registration._viewer import build_section_viewer
from histopia.registration._wsi import (
    WsiWarpResult,
    calculate_thumbnail_overlap_bbox,
    thumbnail_to_full_resolution_matrix,
    warp_saved_registration,
    warp_slide_to_reference,
)

__all__ = [
    "BrightfieldMaskConfig",
    "AlignmentMetrics",
    "MaskRefinementConfig",
    "MaskReviewEntry",
    "NonRigidRefinementConfig",
    "NonRigidTransformResult",
    "GroupOrientation",
    "OrientationDecision",
    "KpfManifest",
    "RegistrationConfig",
    "RegistrationResult",
    "RigidTransformResult",
    "SlidePair",
    "SlideGeometry",
    "SlideRecord",
    "SectionOrderProposal",
    "WsiWarpResult",
    "build_kpf_manifest",
    "build_section_viewer",
    "apply_quarter_turn",
    "calculate_thumbnail_overlap_bbox",
    "create_tissue_mask",
    "refine_group_tissue_masks",
    "discover_slides",
    "estimate_rigid_transform",
    "estimate_non_rigid_transform",
    "refine_rigid_transform",
    "normalize_slide_stem",
    "orient_section_group",
    "register_sections",
    "thumbnail_to_full_resolution_matrix",
    "warp_saved_registration",
    "warp_slide_to_reference",
    "warp_with_displacement",
]
