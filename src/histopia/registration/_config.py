"""Typed configuration objects for registration workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

MaskMode = Literal["auto_tissue", "full"]
CropMode = Literal["overlap", "reference", "none"]
RigidMethod = Literal["feature", "mask_moments", "phase_correlation"]
AlignStrategy = Literal["hybrid", "serial", "reference"]
SectionOrderStrategy = Literal[
    "natural", "manifest", "similarity", "anchored_similarity"
]
WsiCompression = Literal["jpeg", "lzw", "deflate"]


@dataclass(slots=True)
class BrightfieldMaskConfig:
    """Configuration for brightfield/IHC tissue mask generation.

    ``auto_tissue`` is the intended production mode. ``full`` exists to
    reproduce legacy full-mask runs or as a transparent fallback when all
    tissue-mask candidates fail QC.
    """

    mode: MaskMode = "auto_tissue"
    allow_full_fallback: bool = False
    min_foreground_fraction: float = 0.002
    max_foreground_fraction: float = 0.85
    min_largest_component_fraction: float = 0.05
    min_bbox_fraction: float = 0.01
    max_border_strip_fraction: float = 0.50
    max_component_border_fraction: float = 0.35
    max_frame_component_border_fraction: float = 0.10
    min_object_area_px: int = 64
    close_radius_px: int = 4
    open_radius_px: int = 2


@dataclass(slots=True)
class MaskRefinementConfig:
    """Conservative affine refinement using stain-independent tissue geometry."""

    enabled: bool = True
    max_dim_px: int = 500
    min_dice_improvement: float = 0.01
    max_relative_scale_change: float = 0.35
    max_relative_anisotropy: float = 1.30

    def __post_init__(self) -> None:
        if self.max_dim_px <= 0:
            msg = "refinement max_dim_px must be positive"
            raise ValueError(msg)
        if self.min_dice_improvement < 0:
            msg = "refinement min_dice_improvement must be non-negative"
            raise ValueError(msg)
        if not 0 < self.max_relative_scale_change < 1:
            msg = "refinement max_relative_scale_change must be between 0 and 1"
            raise ValueError(msg)
        if self.max_relative_anisotropy < 1:
            msg = "refinement max_relative_anisotropy must be at least 1"
            raise ValueError(msg)


@dataclass(slots=True)
class NonRigidRefinementConfig:
    """Acceptance-gated dense refinement after affine registration."""

    enabled: bool = False
    max_displacement_fraction: float = 0.03
    smoothing_sigma_px: float = 12.0
    support_dilation_fraction: float = 0.03
    min_similarity_improvement: float = 0.01
    max_mask_dice_loss: float = 0.01
    min_jacobian_p01: float = 0.25
    max_jacobian_p99: float = 4.0
    max_inverse_consistency_fraction: float = 0.02

    def __post_init__(self) -> None:
        if not 0 < self.max_displacement_fraction < 0.5:
            msg = "max_displacement_fraction must be between 0 and 0.5"
            raise ValueError(msg)
        if self.smoothing_sigma_px <= 0:
            msg = "smoothing_sigma_px must be positive"
            raise ValueError(msg)
        if not 0 <= self.support_dilation_fraction < 0.5:
            msg = "support_dilation_fraction must be between 0 and 0.5"
            raise ValueError(msg)
        if not 0 < self.max_inverse_consistency_fraction < 0.5:
            msg = "max_inverse_consistency_fraction must be between 0 and 0.5"
            raise ValueError(msg)


@dataclass(slots=True)
class RegistrationConfig:
    """Configuration for one rigid serial-section registration run."""

    input_dir: Path
    output_dir: Path
    reference_slide: str | None = None
    reference_policy: Literal["explicit", "best_connected"] = "best_connected"
    section_order_path: Path | None = None
    section_order_strategy: SectionOrderStrategy = "natural"
    section_order_review_path: Path | None = None
    require_approved_order: bool = False
    mask_review_path: Path | None = None
    mask_override_dir: Path | None = None
    affine_override_path: Path | None = None
    require_approved_masks: bool = False
    wsi_only: bool = False
    registered_reference_dir: Path | None = None
    max_processed_image_dim_px: int = 1200
    crop_mode: CropMode = "reference"
    rigid_method: RigidMethod = "feature"
    align_strategy: AlignStrategy = "hybrid"
    non_rigid: bool = False
    mask: BrightfieldMaskConfig = field(default_factory=BrightfieldMaskConfig)
    refinement: MaskRefinementConfig = field(default_factory=MaskRefinementConfig)
    non_rigid_refinement: NonRigidRefinementConfig = field(
        default_factory=NonRigidRefinementConfig
    )
    write_processed_images: bool = True
    write_warped_images: bool = False
    registered_output_dir: Path | None = None
    wsi_compression: WsiCompression = "jpeg"
    wsi_jpeg_quality: int = 95
    wsi_tile_size: int = 512

    def __post_init__(self) -> None:
        self.input_dir = Path(self.input_dir)
        self.output_dir = Path(self.output_dir)
        if self.registered_reference_dir is not None:
            self.registered_reference_dir = Path(self.registered_reference_dir)
        if self.section_order_path is not None:
            self.section_order_path = Path(self.section_order_path)
        if self.section_order_review_path is not None:
            self.section_order_review_path = Path(self.section_order_review_path)
        if self.mask_review_path is not None:
            self.mask_review_path = Path(self.mask_review_path)
        if self.mask_override_dir is not None:
            self.mask_override_dir = Path(self.mask_override_dir)
        if self.affine_override_path is not None:
            self.affine_override_path = Path(self.affine_override_path)
        if self.registered_output_dir is not None:
            self.registered_output_dir = Path(self.registered_output_dir)
        if self.max_processed_image_dim_px <= 0:
            msg = "max_processed_image_dim_px must be positive"
            raise ValueError(msg)
        if self.non_rigid:
            self.non_rigid_refinement.enabled = True
        if not 1 <= self.wsi_jpeg_quality <= 100:
            msg = "wsi_jpeg_quality must be between 1 and 100"
            raise ValueError(msg)
        if self.wsi_tile_size <= 0:
            msg = "wsi_tile_size must be positive"
            raise ValueError(msg)
