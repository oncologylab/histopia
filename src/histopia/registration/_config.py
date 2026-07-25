"""Typed configuration objects for registration workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from histopia._validation import (
    finite_float,
    nonnegative_int,
    positive_int,
    require_bool,
    require_choice,
)

MaskMode = Literal["auto_tissue", "full"]
CropMode = Literal["overlap", "reference"]
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

    def __post_init__(self) -> None:
        require_choice("mask mode", self.mode, ("auto_tissue", "full"))
        require_bool("allow_full_fallback", self.allow_full_fallback)
        fraction_names = (
            "min_foreground_fraction",
            "max_foreground_fraction",
            "min_largest_component_fraction",
            "min_bbox_fraction",
            "max_border_strip_fraction",
            "max_component_border_fraction",
            "max_frame_component_border_fraction",
        )
        for name in fraction_names:
            value = finite_float(name, getattr(self, name))
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
            setattr(self, name, value)
        if self.min_foreground_fraction > self.max_foreground_fraction:
            raise ValueError(
                "min_foreground_fraction must not exceed max_foreground_fraction"
            )
        self.min_object_area_px = positive_int(
            "min_object_area_px",
            self.min_object_area_px,
        )
        self.close_radius_px = nonnegative_int(
            "close_radius_px",
            self.close_radius_px,
        )
        self.open_radius_px = nonnegative_int(
            "open_radius_px",
            self.open_radius_px,
        )


@dataclass(slots=True)
class MaskRefinementConfig:
    """Conservative affine refinement using stain-independent tissue geometry."""

    enabled: bool = True
    max_dim_px: int = 500
    min_dice_improvement: float = 0.01
    max_relative_scale_change: float = 0.35
    max_relative_anisotropy: float = 1.30

    def __post_init__(self) -> None:
        require_bool("refinement enabled", self.enabled)
        self.max_dim_px = positive_int("refinement max_dim_px", self.max_dim_px)
        self.min_dice_improvement = finite_float(
            "refinement min_dice_improvement",
            self.min_dice_improvement,
        )
        if not 0 <= self.min_dice_improvement <= 1:
            msg = "refinement min_dice_improvement must be between 0 and 1"
            raise ValueError(msg)
        self.max_relative_scale_change = finite_float(
            "refinement max_relative_scale_change",
            self.max_relative_scale_change,
        )
        if not 0 < self.max_relative_scale_change < 1:
            msg = "refinement max_relative_scale_change must be between 0 and 1"
            raise ValueError(msg)
        self.max_relative_anisotropy = finite_float(
            "refinement max_relative_anisotropy",
            self.max_relative_anisotropy,
        )
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
        require_bool("non-rigid enabled", self.enabled)
        self.max_displacement_fraction = finite_float(
            "max_displacement_fraction",
            self.max_displacement_fraction,
        )
        if not 0 < self.max_displacement_fraction < 0.5:
            msg = "max_displacement_fraction must be between 0 and 0.5"
            raise ValueError(msg)
        self.smoothing_sigma_px = finite_float(
            "smoothing_sigma_px",
            self.smoothing_sigma_px,
        )
        if self.smoothing_sigma_px <= 0:
            msg = "smoothing_sigma_px must be positive"
            raise ValueError(msg)
        self.support_dilation_fraction = finite_float(
            "support_dilation_fraction",
            self.support_dilation_fraction,
        )
        if not 0 <= self.support_dilation_fraction < 0.5:
            msg = "support_dilation_fraction must be between 0 and 0.5"
            raise ValueError(msg)
        self.min_similarity_improvement = finite_float(
            "min_similarity_improvement",
            self.min_similarity_improvement,
        )
        if not 0 <= self.min_similarity_improvement <= 2:
            raise ValueError("min_similarity_improvement must be between 0 and 2")
        self.max_mask_dice_loss = finite_float(
            "max_mask_dice_loss",
            self.max_mask_dice_loss,
        )
        if not 0 <= self.max_mask_dice_loss <= 1:
            raise ValueError("max_mask_dice_loss must be between 0 and 1")
        self.min_jacobian_p01 = finite_float(
            "min_jacobian_p01",
            self.min_jacobian_p01,
        )
        if not 0 < self.min_jacobian_p01 <= 1:
            raise ValueError("min_jacobian_p01 must be between 0 and 1")
        self.max_jacobian_p99 = finite_float(
            "max_jacobian_p99",
            self.max_jacobian_p99,
        )
        if self.max_jacobian_p99 < 1:
            raise ValueError("max_jacobian_p99 must be at least 1")
        self.max_inverse_consistency_fraction = finite_float(
            "max_inverse_consistency_fraction",
            self.max_inverse_consistency_fraction,
        )
        if not 0 < self.max_inverse_consistency_fraction < 0.5:
            msg = "max_inverse_consistency_fraction must be between 0 and 0.5"
            raise ValueError(msg)


@dataclass(slots=True)
class RegistrationConfig:
    """Configuration for one rigid serial-section registration run."""

    input_dir: Path
    output_dir: Path
    input_slides: tuple[Path, ...] = ()
    reference_slide: str | None = None
    reference_policy: Literal["explicit", "best_connected"] = "best_connected"
    section_order_path: Path | None = None
    section_order_strategy: SectionOrderStrategy = "natural"
    section_order_review_path: Path | None = None
    section_orientation_path: Path | None = None
    thumbnail_workers: int = 1
    mask_workers: int = 1
    ordering_workers: int = 1
    preprocessing_cache: bool = True
    alignment_cache: bool = True
    require_approved_order: bool = False
    mask_review_path: Path | None = None
    mask_override_dir: Path | None = None
    automatic_mask_snapshot_path: Path | None = None
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
        if isinstance(self.input_slides, (str, bytes, Path)):
            raise TypeError("input_slides must be an iterable of paths")
        self.input_slides = tuple(Path(path) for path in self.input_slides)
        if self.registered_reference_dir is not None:
            self.registered_reference_dir = Path(self.registered_reference_dir)
        if self.section_order_path is not None:
            self.section_order_path = Path(self.section_order_path)
        if self.section_order_review_path is not None:
            self.section_order_review_path = Path(self.section_order_review_path)
        if self.section_orientation_path is not None:
            self.section_orientation_path = Path(self.section_orientation_path)
        if self.mask_review_path is not None:
            self.mask_review_path = Path(self.mask_review_path)
        if self.mask_override_dir is not None:
            self.mask_override_dir = Path(self.mask_override_dir)
        if self.automatic_mask_snapshot_path is not None:
            self.automatic_mask_snapshot_path = Path(self.automatic_mask_snapshot_path)
        if self.affine_override_path is not None:
            self.affine_override_path = Path(self.affine_override_path)
        if self.registered_output_dir is not None:
            self.registered_output_dir = Path(self.registered_output_dir)
        if not isinstance(self.mask, BrightfieldMaskConfig):
            raise TypeError("mask must be a BrightfieldMaskConfig")
        if not isinstance(self.refinement, MaskRefinementConfig):
            raise TypeError("refinement must be a MaskRefinementConfig")
        if not isinstance(self.non_rigid_refinement, NonRigidRefinementConfig):
            raise TypeError("non_rigid_refinement must be a NonRigidRefinementConfig")
        require_choice(
            "reference_policy",
            self.reference_policy,
            ("explicit", "best_connected"),
        )
        require_choice(
            "section_order_strategy",
            self.section_order_strategy,
            ("natural", "manifest", "similarity", "anchored_similarity"),
        )
        require_choice("crop_mode", self.crop_mode, ("reference", "overlap"))
        require_choice(
            "rigid_method",
            self.rigid_method,
            ("feature", "mask_moments", "phase_correlation"),
        )
        require_choice(
            "align_strategy",
            self.align_strategy,
            ("hybrid", "serial", "reference"),
        )
        require_choice(
            "wsi_compression",
            self.wsi_compression,
            ("jpeg", "lzw", "deflate"),
        )
        if self.reference_slide is not None:
            if not isinstance(self.reference_slide, str):
                raise TypeError("reference_slide must be a string")
            if not self.reference_slide.strip():
                raise ValueError("reference_slide must not be blank")
        if self.reference_policy == "explicit" and self.reference_slide is None:
            raise ValueError(
                "reference_slide is required when reference_policy is explicit"
            )
        if (
            self.section_order_strategy == "manifest"
            and self.section_order_path is None
        ):
            raise ValueError(
                "section_order_path is required when section_order_strategy is manifest"
            )
        self.max_processed_image_dim_px = positive_int(
            "max_processed_image_dim_px",
            self.max_processed_image_dim_px,
        )
        self.ordering_workers = positive_int(
            "ordering_workers",
            self.ordering_workers,
        )
        self.thumbnail_workers = positive_int(
            "thumbnail_workers",
            self.thumbnail_workers,
        )
        self.mask_workers = positive_int("mask_workers", self.mask_workers)
        for name in (
            "preprocessing_cache",
            "alignment_cache",
            "require_approved_order",
            "require_approved_masks",
            "wsi_only",
            "non_rigid",
            "write_processed_images",
            "write_warped_images",
        ):
            require_bool(name, getattr(self, name))
        if self.non_rigid:
            self.non_rigid_refinement.enabled = True
        self.wsi_jpeg_quality = positive_int(
            "wsi_jpeg_quality",
            self.wsi_jpeg_quality,
        )
        if self.wsi_jpeg_quality > 100:
            msg = "wsi_jpeg_quality must be between 1 and 100"
            raise ValueError(msg)
        self.wsi_tile_size = positive_int("wsi_tile_size", self.wsi_tile_size)
