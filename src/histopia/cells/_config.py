"""Configuration for native-space cell-boundary detection."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from histopia._validation import finite_float, positive_int, require_bool

CELLPOSE_MODELS = ("cpsam", "cpsam_v2")
CELL_METHODS = ("direct", "multiscale", "combined", "containment")


@dataclass(slots=True)
class CellSegmentationConfig:
    """Scientific controls and bounded runtime settings for one cell run."""

    registration_run: Path
    output_dir: Path
    tile_cache_reuse_run: Path | None = None
    analysis_manifest: Path | None = None
    sections: tuple[str, ...] = ()
    model: str = "cpsam"
    method: str = "containment"
    device: str = "auto"
    inference_batch_size: int = 8
    tile_size: int = 1024
    overlap: int = 128
    tissue_fraction_threshold: float = 0.01
    first_cellprob: float = -2.5
    first_diameter: int = 26
    second_cellprob: float = -1.75
    second_diameter: int = 15
    flow_threshold: float = 0.0
    merge_threshold: float = 0.1
    match_ios: float = 0.5
    min_size: int = 15
    multiscale_nuclear_support: bool = True
    nuclear_minimum_optical_density: float = 0.12
    nuclear_minimum_pixels: int = 3
    nuclear_minimum_fraction: float = 0.005
    global_nuclear_support: bool = True
    global_nuclear_minimum_optical_density: float = 0.15
    global_nuclear_minimum_pixels: int = 8
    global_nuclear_minimum_fraction: float = 0.02
    source_tissue_context: bool = True
    source_tissue_context_bin_size: int = 256
    source_tissue_context_minimum_fraction: float = 0.03
    source_tissue_context_minimum_instance_fraction: float = 0.01
    adaptive_nuclear_core: bool = True
    adaptive_nuclear_core_percentile: float = 70.0
    adaptive_nuclear_core_minimum_area_um2: float = 225.0
    adaptive_nuclear_core_minimum_pixels: int = 8
    adaptive_nuclear_core_minimum_fraction: float = 0.10
    adaptive_nuclear_core_minimum_blue_ratio: float = 1.08
    isolated_debris_gate: bool = True
    isolated_debris_context_bin_size: int = 64
    isolated_debris_context_minimum_stain_fraction: float = 0.80
    isolated_debris_context_minimum_instance_fraction: float = 0.75
    isolated_debris_component_minimum_nuclear_pixels: int = 8
    isolated_debris_allow_nuclear_escape: bool = True
    isolated_debris_allow_organized_escape: bool = True
    isolated_debris_nuclear_minimum_pixels: int = 2
    isolated_debris_nuclear_minimum_fraction: float = 0.005
    isolated_debris_nuclear_minimum_blue_ratio: float = 1.08
    isolated_debris_neutral_dark_maximum_value: int = 90
    isolated_debris_neutral_dark_maximum_chroma: int = 30
    isolated_debris_neutral_dark_maximum_fraction: float = 0.50
    isolated_debris_very_dark_maximum_value: int = 60
    isolated_debris_very_dark_maximum_chroma: int = 20
    isolated_debris_very_dark_maximum_fraction: float = 0.45
    isolated_debris_micro_bin_size: int = 16
    isolated_debris_micro_minimum_stain_fraction: float = 0.80
    isolated_debris_micro_maximum_component_bins: int = 512
    isolated_debris_micro_minimum_nuclear_fraction: float = 0.01
    isolated_debris_micro_minimum_instance_fraction: float = 0.50
    isolated_debris_organized_bin_size: int = 4
    isolated_debris_organized_window_size: int = 1024
    isolated_debris_organized_window_overlap: int = 128
    isolated_debris_organized_minimum_bin_occupancy_fraction: float = 0.10
    isolated_debris_organized_minimum_component_pixels: int = 10_000
    isolated_debris_organized_compact_minimum_component_pixels: int = 20_000
    isolated_debris_organized_minimum_aspect_ratio: float = 4.0
    isolated_debris_organized_minimum_mean_instance_pixels: float = 350.0
    isolated_debris_organized_minimum_instance_fraction: float = 0.50
    isolated_debris_organized_strong_red_blue_difference: float = 40.0
    isolated_debris_organized_minimum_strong_chromatic_fraction: float = 0.25
    isolated_debris_compact_unsupported_minimum_area_um2: float = 10_000.0
    isolated_debris_compact_unsupported_maximum_aspect_ratio: float = 1.75
    isolated_debris_compact_unsupported_maximum_mean_instance_pixels: float = 250.0
    isolated_debris_compact_unsupported_maximum_mean_red_blue_difference: float = 21.0
    isolated_debris_compact_unsupported_strong_red_blue_difference: float = 40.0
    isolated_debris_compact_unsupported_maximum_strong_chromatic_fraction: float = 0.10
    isolated_debris_compact_unsupported_elongation_ratio: float = 2.0
    isolated_debris_compact_unsupported_maximum_elongated_instance_fraction: float = (
        0.50
    )
    isolated_debris_compact_unsupported_maximum_context_nuclear_fraction: float = 0.20
    isolated_debris_compact_unsupported_minimum_component_fill_fraction: float = 0.20
    isolated_debris_compact_unsupported_minimum_instance_fraction: float = 0.50
    # A second, deliberately narrower compact-mosaic gate targets pale luminal
    # foam.  It may tolerate slightly more nearby nuclear support than the
    # general pseudo-cell gate only when the source component is bright and
    # predominantly round; this avoids reopening the compact viable-tissue
    # failure that motivated the stricter general threshold.
    isolated_debris_foam_minimum_mean_intensity: float = 160.0
    isolated_debris_foam_maximum_mean_instance_pixels: float = 250.0
    isolated_debris_foam_maximum_strong_chromatic_fraction: float = 0.10
    isolated_debris_foam_maximum_context_nuclear_fraction: float = 0.50
    isolated_debris_foam_maximum_elongated_instance_fraction: float = 0.50
    isolated_debris_foam_core_minimum_pixels: int = 8
    isolated_debris_foam_core_minimum_bright_fraction: float = 0.80
    isolated_debris_foam_core_minimum_intensity: float = 180.0
    isolated_debris_foam_core_maximum_instance_area_um2: float = 110.0
    isolated_debris_foam_core_minimum_component_area_um2: float = 6500.0
    isolated_debris_foam_core_maximum_aspect_ratio: float = 2.50
    isolated_debris_oversized_chromatic_minimum_area_um2: float = 1000.0
    isolated_debris_oversized_chromatic_minimum_fraction: float = 0.50
    isolated_debris_fold_minimum_instance_area_um2: float = 300.0
    isolated_debris_fold_minimum_component_area_um2: float = 750.0
    isolated_debris_fold_maximum_mean_red_blue_difference: float = 40.0
    isolated_debris_fold_maximum_mean_intensity: float = 180.0
    isolated_debris_fold_dense_minimum_instances: int = 3
    isolated_debris_fold_dense_minimum_aspect_ratio: float = 3.0
    isolated_debris_fold_dense_connectivity_dilation_bins: int = 8
    isolated_debris_fold_dense_maximum_mean_red_blue_difference: float = 60.0
    isolated_debris_fold_dense_maximum_mean_intensity: float = 115.0
    # Detached scanning/debris fragments can be split into several very large
    # pseudo-cells despite containing stain-like pixels.  This gate is narrow:
    # it requires a sparse, elongated cluster of multiple muted oversized
    # instances, so isolated large cells and populated tissue are preserved.
    isolated_debris_detached_minimum_instance_area_um2: float = 200.0
    isolated_debris_detached_minimum_component_area_um2: float = 1000.0
    isolated_debris_detached_minimum_instances: int = 3
    isolated_debris_detached_minimum_aspect_ratio: float = 3.0
    isolated_debris_detached_maximum_mean_red_blue_difference: float = 35.0
    isolated_debris_detached_maximum_mean_intensity: float = 220.0
    isolated_debris_detached_context_window_size: int = 512
    isolated_debris_detached_maximum_prediction_fraction: float = 0.20
    # A compact crushed fragment can evade the elongated-fragment topology
    # while still being split into two giant, muted pseudo-cells.  The compact
    # branch is intentionally stricter in color and density and retains the
    # same sparse-context requirement.
    isolated_debris_detached_compact_minimum_instance_area_um2: float = 300.0
    isolated_debris_detached_compact_minimum_component_area_um2: float = 750.0
    isolated_debris_detached_compact_minimum_instances: int = 2
    isolated_debris_detached_compact_maximum_mean_red_blue_difference: float = 0.0
    isolated_debris_detached_compact_maximum_mean_intensity: float = 180.0
    isolated_debris_glass_minimum_area_um2: float = 25.0
    isolated_debris_glass_context_window_size: int = 256
    isolated_debris_glass_maximum_prediction_fraction: float = 0.40
    isolated_debris_glass_maximum_context_stain_fraction: float = 0.20
    isolated_debris_glass_low_stain_maximum_mean_red_blue_difference: float = 35.0
    isolated_debris_glass_low_stain_minimum_mean_intensity: float = 120.0
    isolated_debris_glass_minimum_context_fraction: float = 0.50
    isolated_debris_glass_maximum_mean_red_blue_difference: float = 25.0
    isolated_debris_glass_minimum_mean_intensity: float = 190.0
    isolated_debris_necrotic_minimum_component_area_um2: float = 5000.0
    isolated_debris_necrotic_maximum_aspect_ratio: float = 2.5
    isolated_debris_necrotic_minimum_fill_fraction: float = 0.05
    isolated_debris_necrotic_maximum_fill_fraction: float = 0.45
    isolated_debris_necrotic_minimum_mean_red_blue_difference: float = -15.0
    isolated_debris_necrotic_maximum_mean_red_blue_difference: float = 15.0
    isolated_debris_necrotic_minimum_mean_intensity: float = 160.0
    isolated_debris_necrotic_minimum_instance_fraction: float = 0.50
    isolated_debris_brown_minimum_component_area_um2: float = 4.0
    isolated_debris_brown_maximum_aspect_ratio: float = 3.0
    isolated_debris_brown_minimum_fill_fraction: float = 0.12
    isolated_debris_brown_maximum_fill_fraction: float = 0.45
    isolated_debris_brown_minimum_mean_red_blue_difference: float = 50.0
    isolated_debris_brown_maximum_mean_intensity: float = 120.0
    isolated_debris_brown_minimum_instance_fraction: float = 0.50
    isolated_debris_brown_maximum_source_component_area_um2: float = 50_000.0
    isolated_debris_brown_source_maximum_mean_intensity: float = 218.0
    isolated_debris_brown_source_dilation_bins: int = 0
    model_cache: Path | None = None
    allow_model_download: bool = False
    require_registration_approval: bool = True
    keep_tile_masks: bool = False
    # Runtime-only cache policy and cohort-wide refinements are appended to
    # preserve the positional constructor order of the established API.
    require_complete_tile_cache_reuse: bool = False
    # The satellite gate is part of the validated v75 debris policy and may
    # not be disabled. A later, one-section cache-only profile can protect
    # candidates far inside the accepted mask while retaining the gate at
    # debris-prone tissue boundaries.
    isolated_debris_satellite_gate: bool = True
    isolated_debris_satellite_minimum_mask_distance_um: float | None = None
    # Refinements remain opt-in until their positive and negative controls are
    # sealed. Disabled controls are omitted from the request fingerprint so an
    # older validated run retains its exact v75 raw-tile profile.
    isolated_debris_self_dense_glass_gate: bool = False
    isolated_debris_oversized_brown_gate: bool = False
    isolated_debris_oversized_brown_maximum_mean_intensity: float = 180.0
    isolated_debris_clustered_brown_gate: bool = False
    isolated_debris_diffuse_degenerated_gate: bool = False
    isolated_debris_neutral_precipitate_gate: bool = False
    reconcile_enclosed_cytoplasmic_children: bool = False
    isolated_debris_protect_organized_from_necrotic: bool = False
    # An organized-looking object can still be one oversized, detached
    # pseudo-cell (for example a pale processing loop over glass).  A physical
    # area ceiling lets a later profile bound the organized-necrotic rescue
    # without changing the established unbounded v76 canary semantics.
    isolated_debris_organized_necrotic_protection_maximum_instance_area_um2: (
        float | None
    ) = None
    # A cell-sized fragment can still look organized in isolation over glass.
    # Newer profiles may require a modest surrounding prediction density while
    # older profiles retain their exact unbounded behavior.
    isolated_debris_organized_necrotic_protection_minimum_local_prediction_fraction: (
        float | None
    ) = None
    isolated_debris_organized_necrotic_protection_use_eligible_context: bool = False
    # Shape-only rescues must not mutually validate a detached processing
    # fragment.  When enabled, local support is built only from instances with
    # independent nuclear/core evidence, while retaining the full-cell area of
    # those independently supported neighbours.
    isolated_debris_organized_necrotic_protection_require_independent_nuclear_context: bool = False  # noqa: E501
    isolated_debris_organized_necrotic_sparse_glass_shape_gate: bool = False
    # Optional fingerprint-bound StarDist replacements for exact raw CPSAM
    # cache tiles with independently detected dense-small-cell dropout.
    dense_small_cell_recovery_manifest: Path | None = None

    def __post_init__(self) -> None:
        self.registration_run = Path(self.registration_run)
        self.output_dir = Path(self.output_dir)
        if self.tile_cache_reuse_run is not None:
            self.tile_cache_reuse_run = Path(self.tile_cache_reuse_run)
        if self.dense_small_cell_recovery_manifest is not None:
            self.dense_small_cell_recovery_manifest = Path(
                self.dense_small_cell_recovery_manifest
            )
        require_bool(
            "require_complete_tile_cache_reuse",
            self.require_complete_tile_cache_reuse,
        )
        require_bool(
            "isolated_debris_satellite_gate",
            self.isolated_debris_satellite_gate,
        )
        if self.require_complete_tile_cache_reuse and self.tile_cache_reuse_run is None:
            raise ValueError(
                "require_complete_tile_cache_reuse requires tile_cache_reuse_run"
            )
        if self.dense_small_cell_recovery_manifest is not None and (
            self.tile_cache_reuse_run is None
            or not self.require_complete_tile_cache_reuse
            or len(self.sections) != 1
        ):
            raise ValueError(
                "dense_small_cell_recovery_manifest requires one explicit section "
                "and complete sealed tile-cache reuse"
            )
        if self.analysis_manifest is not None:
            self.analysis_manifest = Path(self.analysis_manifest)
        if self.model_cache is not None:
            self.model_cache = Path(self.model_cache)
        self.sections = tuple(
            dict.fromkeys(str(value).strip() for value in self.sections)
        )
        if any(not value for value in self.sections):
            raise ValueError(
                "sections must contain non-empty slide names or section IDs"
            )
        self.model = self.model.strip().lower()
        if self.model not in CELLPOSE_MODELS:
            raise ValueError("model must be one of: " + ", ".join(CELLPOSE_MODELS))
        self.method = self.method.strip().lower()
        if self.method not in CELL_METHODS:
            raise ValueError("method must be one of: " + ", ".join(CELL_METHODS))
        if not self.device.strip():
            raise ValueError("device must not be blank")
        self.inference_batch_size = positive_int(
            "inference_batch_size", self.inference_batch_size
        )
        self.tile_size = positive_int("tile_size", self.tile_size)
        self.overlap = positive_int("overlap", self.overlap)
        if self.overlap >= self.tile_size:
            raise ValueError("overlap must be smaller than tile_size")
        self.tissue_fraction_threshold = finite_float(
            "tissue_fraction_threshold", self.tissue_fraction_threshold
        )
        if not 0 <= self.tissue_fraction_threshold <= 1:
            raise ValueError("tissue_fraction_threshold must be between zero and one")
        self.first_cellprob = finite_float("first_cellprob", self.first_cellprob)
        self.second_cellprob = finite_float("second_cellprob", self.second_cellprob)
        self.first_diameter = positive_int("first_diameter", self.first_diameter)
        self.second_diameter = positive_int("second_diameter", self.second_diameter)
        self.flow_threshold = finite_float("flow_threshold", self.flow_threshold)
        if self.flow_threshold < 0:
            raise ValueError("flow_threshold must be nonnegative")
        self.merge_threshold = finite_float("merge_threshold", self.merge_threshold)
        self.match_ios = finite_float("match_ios", self.match_ios)
        if not 0 <= self.merge_threshold <= 1:
            raise ValueError("merge_threshold must be between zero and one")
        if not 0 <= self.match_ios <= 1:
            raise ValueError("match_ios must be between zero and one")
        self.min_size = positive_int("min_size", self.min_size)
        require_bool("multiscale_nuclear_support", self.multiscale_nuclear_support)
        self.nuclear_minimum_optical_density = finite_float(
            "nuclear_minimum_optical_density",
            self.nuclear_minimum_optical_density,
        )
        if self.nuclear_minimum_optical_density <= 0:
            raise ValueError("nuclear_minimum_optical_density must be positive")
        self.nuclear_minimum_pixels = positive_int(
            "nuclear_minimum_pixels", self.nuclear_minimum_pixels
        )
        self.nuclear_minimum_fraction = finite_float(
            "nuclear_minimum_fraction", self.nuclear_minimum_fraction
        )
        if not 0 <= self.nuclear_minimum_fraction <= 1:
            raise ValueError("nuclear_minimum_fraction must be between zero and one")
        require_bool("global_nuclear_support", self.global_nuclear_support)
        self.global_nuclear_minimum_optical_density = finite_float(
            "global_nuclear_minimum_optical_density",
            self.global_nuclear_minimum_optical_density,
        )
        if self.global_nuclear_minimum_optical_density <= 0:
            raise ValueError("global_nuclear_minimum_optical_density must be positive")
        self.global_nuclear_minimum_pixels = positive_int(
            "global_nuclear_minimum_pixels", self.global_nuclear_minimum_pixels
        )
        self.global_nuclear_minimum_fraction = finite_float(
            "global_nuclear_minimum_fraction",
            self.global_nuclear_minimum_fraction,
        )
        if not 0 <= self.global_nuclear_minimum_fraction <= 1:
            raise ValueError(
                "global_nuclear_minimum_fraction must be between zero and one"
            )
        require_bool("source_tissue_context", self.source_tissue_context)
        self.source_tissue_context_bin_size = positive_int(
            "source_tissue_context_bin_size", self.source_tissue_context_bin_size
        )
        self.source_tissue_context_minimum_fraction = finite_float(
            "source_tissue_context_minimum_fraction",
            self.source_tissue_context_minimum_fraction,
        )
        if not 0 < self.source_tissue_context_minimum_fraction <= 1:
            raise ValueError(
                "source_tissue_context_minimum_fraction must be between zero and one"
            )
        self.source_tissue_context_minimum_instance_fraction = finite_float(
            "source_tissue_context_minimum_instance_fraction",
            self.source_tissue_context_minimum_instance_fraction,
        )
        if not 0 < self.source_tissue_context_minimum_instance_fraction <= 1:
            raise ValueError(
                "source_tissue_context_minimum_instance_fraction must be between "
                "zero and one"
            )
        require_bool("adaptive_nuclear_core", self.adaptive_nuclear_core)
        self.adaptive_nuclear_core_percentile = finite_float(
            "adaptive_nuclear_core_percentile",
            self.adaptive_nuclear_core_percentile,
        )
        if not 0 < self.adaptive_nuclear_core_percentile < 100:
            raise ValueError(
                "adaptive_nuclear_core_percentile must be between zero and 100"
            )
        self.adaptive_nuclear_core_minimum_area_um2 = finite_float(
            "adaptive_nuclear_core_minimum_area_um2",
            self.adaptive_nuclear_core_minimum_area_um2,
        )
        if self.adaptive_nuclear_core_minimum_area_um2 <= 0:
            raise ValueError("adaptive_nuclear_core_minimum_area_um2 must be positive")
        self.adaptive_nuclear_core_minimum_pixels = positive_int(
            "adaptive_nuclear_core_minimum_pixels",
            self.adaptive_nuclear_core_minimum_pixels,
        )
        self.adaptive_nuclear_core_minimum_fraction = finite_float(
            "adaptive_nuclear_core_minimum_fraction",
            self.adaptive_nuclear_core_minimum_fraction,
        )
        if not 0 < self.adaptive_nuclear_core_minimum_fraction <= 1:
            raise ValueError(
                "adaptive_nuclear_core_minimum_fraction must be between zero and one"
            )
        self.adaptive_nuclear_core_minimum_blue_ratio = finite_float(
            "adaptive_nuclear_core_minimum_blue_ratio",
            self.adaptive_nuclear_core_minimum_blue_ratio,
        )
        if self.adaptive_nuclear_core_minimum_blue_ratio <= 1:
            raise ValueError(
                "adaptive_nuclear_core_minimum_blue_ratio must be greater than one"
            )
        require_bool("isolated_debris_gate", self.isolated_debris_gate)
        if not self.isolated_debris_satellite_gate:
            raise ValueError(
                "isolated_debris_satellite_gate cannot be disabled; use the "
                "bounded interior-protection policy"
            )
        if self.isolated_debris_satellite_minimum_mask_distance_um is not None:
            if not self.isolated_debris_gate:
                raise ValueError(
                    "satellite interior protection requires isolated-debris evidence"
                )
            distance = finite_float(
                "isolated_debris_satellite_minimum_mask_distance_um",
                self.isolated_debris_satellite_minimum_mask_distance_um,
            )
            if distance <= 0:
                raise ValueError(
                    "satellite interior-protection distance must be positive"
                )
            self.isolated_debris_satellite_minimum_mask_distance_um = distance
        self.isolated_debris_context_bin_size = positive_int(
            "isolated_debris_context_bin_size",
            self.isolated_debris_context_bin_size,
        )
        self.isolated_debris_context_minimum_stain_fraction = finite_float(
            "isolated_debris_context_minimum_stain_fraction",
            self.isolated_debris_context_minimum_stain_fraction,
        )
        if not 0 < self.isolated_debris_context_minimum_stain_fraction <= 1:
            raise ValueError(
                "isolated_debris_context_minimum_stain_fraction must be between "
                "zero and one"
            )
        self.isolated_debris_context_minimum_instance_fraction = finite_float(
            "isolated_debris_context_minimum_instance_fraction",
            self.isolated_debris_context_minimum_instance_fraction,
        )
        if not 0 < self.isolated_debris_context_minimum_instance_fraction <= 1:
            raise ValueError(
                "isolated_debris_context_minimum_instance_fraction must be between "
                "zero and one"
            )
        self.isolated_debris_component_minimum_nuclear_pixels = positive_int(
            "isolated_debris_component_minimum_nuclear_pixels",
            self.isolated_debris_component_minimum_nuclear_pixels,
        )
        require_bool(
            "isolated_debris_allow_nuclear_escape",
            self.isolated_debris_allow_nuclear_escape,
        )
        require_bool(
            "isolated_debris_allow_organized_escape",
            self.isolated_debris_allow_organized_escape,
        )
        if (
            not self.isolated_debris_allow_organized_escape
            and self.isolated_debris_allow_nuclear_escape
        ):
            raise ValueError(
                "disabling organized escape also requires disabling nuclear escape"
            )
        self.isolated_debris_nuclear_minimum_pixels = positive_int(
            "isolated_debris_nuclear_minimum_pixels",
            self.isolated_debris_nuclear_minimum_pixels,
        )
        self.isolated_debris_nuclear_minimum_fraction = finite_float(
            "isolated_debris_nuclear_minimum_fraction",
            self.isolated_debris_nuclear_minimum_fraction,
        )
        if not 0 < self.isolated_debris_nuclear_minimum_fraction <= 1:
            raise ValueError(
                "isolated_debris_nuclear_minimum_fraction must be between zero and one"
            )
        self.isolated_debris_nuclear_minimum_blue_ratio = finite_float(
            "isolated_debris_nuclear_minimum_blue_ratio",
            self.isolated_debris_nuclear_minimum_blue_ratio,
        )
        if self.isolated_debris_nuclear_minimum_blue_ratio <= 1:
            raise ValueError(
                "isolated_debris_nuclear_minimum_blue_ratio must be greater than one"
            )
        self.isolated_debris_neutral_dark_maximum_value = positive_int(
            "isolated_debris_neutral_dark_maximum_value",
            self.isolated_debris_neutral_dark_maximum_value,
        )
        self.isolated_debris_neutral_dark_maximum_chroma = positive_int(
            "isolated_debris_neutral_dark_maximum_chroma",
            self.isolated_debris_neutral_dark_maximum_chroma,
        )
        if self.isolated_debris_neutral_dark_maximum_value > 255:
            raise ValueError(
                "isolated_debris_neutral_dark_maximum_value must not exceed 255"
            )
        if self.isolated_debris_neutral_dark_maximum_chroma > 255:
            raise ValueError(
                "isolated_debris_neutral_dark_maximum_chroma must not exceed 255"
            )
        self.isolated_debris_neutral_dark_maximum_fraction = finite_float(
            "isolated_debris_neutral_dark_maximum_fraction",
            self.isolated_debris_neutral_dark_maximum_fraction,
        )
        if not 0 < self.isolated_debris_neutral_dark_maximum_fraction <= 1:
            raise ValueError(
                "isolated_debris_neutral_dark_maximum_fraction must be between "
                "zero and one"
            )
        self.isolated_debris_very_dark_maximum_value = positive_int(
            "isolated_debris_very_dark_maximum_value",
            self.isolated_debris_very_dark_maximum_value,
        )
        if self.isolated_debris_very_dark_maximum_value > 255:
            raise ValueError(
                "isolated_debris_very_dark_maximum_value must not exceed 255"
            )
        self.isolated_debris_very_dark_maximum_chroma = positive_int(
            "isolated_debris_very_dark_maximum_chroma",
            self.isolated_debris_very_dark_maximum_chroma,
        )
        if self.isolated_debris_very_dark_maximum_chroma > 255:
            raise ValueError(
                "isolated_debris_very_dark_maximum_chroma must not exceed 255"
            )
        self.isolated_debris_very_dark_maximum_fraction = finite_float(
            "isolated_debris_very_dark_maximum_fraction",
            self.isolated_debris_very_dark_maximum_fraction,
        )
        if not 0 < self.isolated_debris_very_dark_maximum_fraction <= 1:
            raise ValueError(
                "isolated_debris_very_dark_maximum_fraction must be between zero "
                "and one"
            )
        self.isolated_debris_micro_bin_size = positive_int(
            "isolated_debris_micro_bin_size",
            self.isolated_debris_micro_bin_size,
        )
        self.isolated_debris_micro_minimum_stain_fraction = finite_float(
            "isolated_debris_micro_minimum_stain_fraction",
            self.isolated_debris_micro_minimum_stain_fraction,
        )
        if not 0 < self.isolated_debris_micro_minimum_stain_fraction <= 1:
            raise ValueError(
                "isolated_debris_micro_minimum_stain_fraction must be between "
                "zero and one"
            )
        self.isolated_debris_micro_maximum_component_bins = positive_int(
            "isolated_debris_micro_maximum_component_bins",
            self.isolated_debris_micro_maximum_component_bins,
        )
        self.isolated_debris_micro_minimum_nuclear_fraction = finite_float(
            "isolated_debris_micro_minimum_nuclear_fraction",
            self.isolated_debris_micro_minimum_nuclear_fraction,
        )
        if not 0 < self.isolated_debris_micro_minimum_nuclear_fraction <= 1:
            raise ValueError(
                "isolated_debris_micro_minimum_nuclear_fraction must be between "
                "zero and one"
            )
        self.isolated_debris_micro_minimum_instance_fraction = finite_float(
            "isolated_debris_micro_minimum_instance_fraction",
            self.isolated_debris_micro_minimum_instance_fraction,
        )
        if not 0 < self.isolated_debris_micro_minimum_instance_fraction <= 1:
            raise ValueError(
                "isolated_debris_micro_minimum_instance_fraction must be between "
                "zero and one"
            )
        self.isolated_debris_organized_bin_size = positive_int(
            "isolated_debris_organized_bin_size",
            self.isolated_debris_organized_bin_size,
        )
        self.isolated_debris_organized_window_size = positive_int(
            "isolated_debris_organized_window_size",
            self.isolated_debris_organized_window_size,
        )
        self.isolated_debris_organized_window_overlap = positive_int(
            "isolated_debris_organized_window_overlap",
            self.isolated_debris_organized_window_overlap,
        )
        if (
            self.isolated_debris_organized_window_overlap
            >= self.isolated_debris_organized_window_size
        ):
            raise ValueError(
                "isolated_debris_organized_window_overlap must be smaller than "
                "isolated_debris_organized_window_size"
            )
        if (
            self.isolated_debris_organized_window_size
            % self.isolated_debris_organized_bin_size
            or self.isolated_debris_organized_window_overlap
            % self.isolated_debris_organized_bin_size
        ):
            raise ValueError(
                "isolated debris organized window size and overlap must be "
                "multiples of the organized bin size"
            )
        self.isolated_debris_organized_minimum_bin_occupancy_fraction = finite_float(
            "isolated_debris_organized_minimum_bin_occupancy_fraction",
            self.isolated_debris_organized_minimum_bin_occupancy_fraction,
        )
        if not (0 < self.isolated_debris_organized_minimum_bin_occupancy_fraction <= 1):
            raise ValueError(
                "isolated_debris_organized_minimum_bin_occupancy_fraction must be "
                "between zero and one"
            )
        self.isolated_debris_organized_minimum_component_pixels = positive_int(
            "isolated_debris_organized_minimum_component_pixels",
            self.isolated_debris_organized_minimum_component_pixels,
        )
        self.isolated_debris_organized_compact_minimum_component_pixels = positive_int(
            "isolated_debris_organized_compact_minimum_component_pixels",
            self.isolated_debris_organized_compact_minimum_component_pixels,
        )
        if (
            self.isolated_debris_organized_compact_minimum_component_pixels
            < self.isolated_debris_organized_minimum_component_pixels
        ):
            raise ValueError(
                "isolated_debris_organized_compact_minimum_component_pixels must "
                "not be smaller than the organized minimum component size"
            )
        self.isolated_debris_organized_minimum_aspect_ratio = finite_float(
            "isolated_debris_organized_minimum_aspect_ratio",
            self.isolated_debris_organized_minimum_aspect_ratio,
        )
        if self.isolated_debris_organized_minimum_aspect_ratio < 1:
            raise ValueError(
                "isolated_debris_organized_minimum_aspect_ratio must be at least one"
            )
        self.isolated_debris_organized_minimum_mean_instance_pixels = finite_float(
            "isolated_debris_organized_minimum_mean_instance_pixels",
            self.isolated_debris_organized_minimum_mean_instance_pixels,
        )
        if self.isolated_debris_organized_minimum_mean_instance_pixels <= 0:
            raise ValueError(
                "isolated_debris_organized_minimum_mean_instance_pixels must be "
                "positive"
            )
        self.isolated_debris_organized_minimum_instance_fraction = finite_float(
            "isolated_debris_organized_minimum_instance_fraction",
            self.isolated_debris_organized_minimum_instance_fraction,
        )
        if not 0 < self.isolated_debris_organized_minimum_instance_fraction <= 1:
            raise ValueError(
                "isolated_debris_organized_minimum_instance_fraction must be "
                "between zero and one"
            )
        self.isolated_debris_organized_strong_red_blue_difference = finite_float(
            "isolated_debris_organized_strong_red_blue_difference",
            self.isolated_debris_organized_strong_red_blue_difference,
        )
        self.isolated_debris_organized_minimum_strong_chromatic_fraction = finite_float(
            "isolated_debris_organized_minimum_strong_chromatic_fraction",
            self.isolated_debris_organized_minimum_strong_chromatic_fraction,
        )
        if not (
            0 <= self.isolated_debris_organized_minimum_strong_chromatic_fraction <= 1
        ):
            raise ValueError(
                "isolated_debris_organized_minimum_strong_chromatic_fraction "
                "must be between zero and one"
            )
        self.isolated_debris_compact_unsupported_minimum_area_um2 = finite_float(
            "isolated_debris_compact_unsupported_minimum_area_um2",
            self.isolated_debris_compact_unsupported_minimum_area_um2,
        )
        if self.isolated_debris_compact_unsupported_minimum_area_um2 <= 0:
            raise ValueError(
                "isolated_debris_compact_unsupported_minimum_area_um2 must be positive"
            )
        self.isolated_debris_compact_unsupported_maximum_aspect_ratio = finite_float(
            "isolated_debris_compact_unsupported_maximum_aspect_ratio",
            self.isolated_debris_compact_unsupported_maximum_aspect_ratio,
        )
        if self.isolated_debris_compact_unsupported_maximum_aspect_ratio < 1:
            raise ValueError(
                "isolated_debris_compact_unsupported_maximum_aspect_ratio must be "
                "at least one"
            )
        self.isolated_debris_compact_unsupported_maximum_mean_instance_pixels = (
            finite_float(
                "isolated_debris_compact_unsupported_maximum_mean_instance_pixels",
                self.isolated_debris_compact_unsupported_maximum_mean_instance_pixels,
            )
        )
        if self.isolated_debris_compact_unsupported_maximum_mean_instance_pixels <= 0:
            raise ValueError(
                "isolated_debris_compact_unsupported_maximum_mean_instance_pixels "
                "must be positive"
            )
        compact_red_blue = finite_float(
            "isolated_debris_compact_unsupported_maximum_mean_red_blue_difference",
            self.isolated_debris_compact_unsupported_maximum_mean_red_blue_difference,
        )
        self.isolated_debris_compact_unsupported_maximum_mean_red_blue_difference = (
            compact_red_blue
        )
        self.isolated_debris_compact_unsupported_strong_red_blue_difference = (
            finite_float(
                "isolated_debris_compact_unsupported_strong_red_blue_difference",
                self.isolated_debris_compact_unsupported_strong_red_blue_difference,
            )
        )
        compact_strong_fraction = finite_float(
            "isolated_debris_compact_unsupported_maximum_strong_chromatic_fraction",
            self.isolated_debris_compact_unsupported_maximum_strong_chromatic_fraction,
        )
        self.isolated_debris_compact_unsupported_maximum_strong_chromatic_fraction = (
            compact_strong_fraction
        )
        strong_chromatic_fraction = (
            self.isolated_debris_compact_unsupported_maximum_strong_chromatic_fraction
        )
        if not 0 <= strong_chromatic_fraction <= 1:
            raise ValueError(
                "isolated_debris_compact_unsupported_maximum_strong_chromatic_fraction "
                "must be between zero and one"
            )
        self.isolated_debris_compact_unsupported_elongation_ratio = finite_float(
            "isolated_debris_compact_unsupported_elongation_ratio",
            self.isolated_debris_compact_unsupported_elongation_ratio,
        )
        if self.isolated_debris_compact_unsupported_elongation_ratio <= 1:
            raise ValueError(
                "isolated_debris_compact_unsupported_elongation_ratio must be "
                "greater than one"
            )
        compact_elongated_fraction = finite_float(
            "isolated_debris_compact_unsupported_maximum_elongated_instance_fraction",
            self.isolated_debris_compact_unsupported_maximum_elongated_instance_fraction,
        )
        self.isolated_debris_compact_unsupported_maximum_elongated_instance_fraction = (
            compact_elongated_fraction
        )
        elongated_fraction = (
            self.isolated_debris_compact_unsupported_maximum_elongated_instance_fraction
        )
        if not 0 <= elongated_fraction <= 1:
            raise ValueError(
                "isolated_debris_compact_unsupported_maximum_elongated_"
                "instance_fraction "
                "must be between zero and one"
            )
        compact_context_nuclear_fraction = finite_float(
            "isolated_debris_compact_unsupported_maximum_context_nuclear_fraction",
            self.isolated_debris_compact_unsupported_maximum_context_nuclear_fraction,
        )
        self.isolated_debris_compact_unsupported_maximum_context_nuclear_fraction = (
            compact_context_nuclear_fraction
        )
        if not (
            0
            <= self.isolated_debris_compact_unsupported_maximum_context_nuclear_fraction
            <= 1
        ):
            raise ValueError(
                "isolated_debris_compact_unsupported_maximum_context_nuclear_fraction "
                "must be between zero and one"
            )
        compact_component_fill_fraction = finite_float(
            "isolated_debris_compact_unsupported_minimum_component_fill_fraction",
            self.isolated_debris_compact_unsupported_minimum_component_fill_fraction,
        )
        self.isolated_debris_compact_unsupported_minimum_component_fill_fraction = (
            compact_component_fill_fraction
        )
        if not (
            0
            < self.isolated_debris_compact_unsupported_minimum_component_fill_fraction
            <= 1
        ):
            raise ValueError(
                "isolated_debris_compact_unsupported_minimum_component_fill_fraction "
                "must be between zero and one"
            )
        self.isolated_debris_compact_unsupported_minimum_instance_fraction = (
            finite_float(
                "isolated_debris_compact_unsupported_minimum_instance_fraction",
                self.isolated_debris_compact_unsupported_minimum_instance_fraction,
            )
        )
        if not (
            0 < self.isolated_debris_compact_unsupported_minimum_instance_fraction <= 1
        ):
            raise ValueError(
                "isolated_debris_compact_unsupported_minimum_instance_fraction "
                "must be between zero and one"
            )
        self.isolated_debris_foam_minimum_mean_intensity = finite_float(
            "isolated_debris_foam_minimum_mean_intensity",
            self.isolated_debris_foam_minimum_mean_intensity,
        )
        if not 0 < self.isolated_debris_foam_minimum_mean_intensity <= 255:
            raise ValueError(
                "isolated_debris_foam_minimum_mean_intensity must be between "
                "zero and 255"
            )
        self.isolated_debris_foam_maximum_mean_instance_pixels = finite_float(
            "isolated_debris_foam_maximum_mean_instance_pixels",
            self.isolated_debris_foam_maximum_mean_instance_pixels,
        )
        if self.isolated_debris_foam_maximum_mean_instance_pixels <= 0:
            raise ValueError(
                "isolated_debris_foam_maximum_mean_instance_pixels must be positive"
            )
        self.isolated_debris_foam_maximum_strong_chromatic_fraction = finite_float(
            "isolated_debris_foam_maximum_strong_chromatic_fraction",
            self.isolated_debris_foam_maximum_strong_chromatic_fraction,
        )
        if not (0 <= self.isolated_debris_foam_maximum_strong_chromatic_fraction <= 1):
            raise ValueError(
                "isolated_debris_foam_maximum_strong_chromatic_fraction must be "
                "between zero and one"
            )
        self.isolated_debris_foam_maximum_context_nuclear_fraction = finite_float(
            "isolated_debris_foam_maximum_context_nuclear_fraction",
            self.isolated_debris_foam_maximum_context_nuclear_fraction,
        )
        if not (0 <= self.isolated_debris_foam_maximum_context_nuclear_fraction <= 1):
            raise ValueError(
                "isolated_debris_foam_maximum_context_nuclear_fraction must be "
                "between zero and one"
            )
        self.isolated_debris_foam_maximum_elongated_instance_fraction = finite_float(
            "isolated_debris_foam_maximum_elongated_instance_fraction",
            self.isolated_debris_foam_maximum_elongated_instance_fraction,
        )
        if not (
            0 <= self.isolated_debris_foam_maximum_elongated_instance_fraction <= 1
        ):
            raise ValueError(
                "isolated_debris_foam_maximum_elongated_instance_fraction must be "
                "between zero and one"
            )
        self.isolated_debris_foam_core_minimum_pixels = positive_int(
            "isolated_debris_foam_core_minimum_pixels",
            self.isolated_debris_foam_core_minimum_pixels,
        )
        self.isolated_debris_foam_core_minimum_bright_fraction = finite_float(
            "isolated_debris_foam_core_minimum_bright_fraction",
            self.isolated_debris_foam_core_minimum_bright_fraction,
        )
        if not 0 < self.isolated_debris_foam_core_minimum_bright_fraction <= 1:
            raise ValueError(
                "isolated_debris_foam_core_minimum_bright_fraction must be "
                "between zero and one"
            )
        self.isolated_debris_foam_core_minimum_intensity = finite_float(
            "isolated_debris_foam_core_minimum_intensity",
            self.isolated_debris_foam_core_minimum_intensity,
        )
        if not 0 < self.isolated_debris_foam_core_minimum_intensity <= 255:
            raise ValueError(
                "isolated_debris_foam_core_minimum_intensity must be between "
                "zero and 255"
            )
        independent_nuclear_context_name = (
            "isolated_debris_organized_necrotic_protection_"
            "require_independent_nuclear_context"
        )
        for name in (
            "isolated_debris_foam_core_maximum_instance_area_um2",
            "isolated_debris_foam_core_minimum_component_area_um2",
            "isolated_debris_foam_core_maximum_aspect_ratio",
        ):
            value = finite_float(name, getattr(self, name))
            setattr(self, name, value)
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.isolated_debris_foam_core_maximum_aspect_ratio < 1:
            raise ValueError(
                "isolated_debris_foam_core_maximum_aspect_ratio must be at least one"
            )
        self.isolated_debris_oversized_chromatic_minimum_area_um2 = finite_float(
            "isolated_debris_oversized_chromatic_minimum_area_um2",
            self.isolated_debris_oversized_chromatic_minimum_area_um2,
        )
        if self.isolated_debris_oversized_chromatic_minimum_area_um2 <= 0:
            raise ValueError(
                "isolated_debris_oversized_chromatic_minimum_area_um2 must be positive"
            )
        self.isolated_debris_oversized_chromatic_minimum_fraction = finite_float(
            "isolated_debris_oversized_chromatic_minimum_fraction",
            self.isolated_debris_oversized_chromatic_minimum_fraction,
        )
        if not (0 < self.isolated_debris_oversized_chromatic_minimum_fraction <= 1):
            raise ValueError(
                "isolated_debris_oversized_chromatic_minimum_fraction must be "
                "between zero and one"
            )
        for name in (
            "isolated_debris_fold_minimum_instance_area_um2",
            "isolated_debris_fold_minimum_component_area_um2",
        ):
            value = finite_float(name, getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            setattr(self, name, value)
        if (
            self.isolated_debris_fold_minimum_component_area_um2
            < self.isolated_debris_fold_minimum_instance_area_um2
        ):
            raise ValueError(
                "isolated_debris_fold_minimum_component_area_um2 must not be "
                "smaller than the instance area"
            )
        self.isolated_debris_fold_maximum_mean_red_blue_difference = finite_float(
            "isolated_debris_fold_maximum_mean_red_blue_difference",
            self.isolated_debris_fold_maximum_mean_red_blue_difference,
        )
        self.isolated_debris_fold_maximum_mean_intensity = finite_float(
            "isolated_debris_fold_maximum_mean_intensity",
            self.isolated_debris_fold_maximum_mean_intensity,
        )
        if not 0 < self.isolated_debris_fold_maximum_mean_intensity <= 255:
            raise ValueError(
                "isolated_debris_fold_maximum_mean_intensity must be between "
                "zero and 255"
            )
        self.isolated_debris_fold_dense_minimum_instances = positive_int(
            "isolated_debris_fold_dense_minimum_instances",
            self.isolated_debris_fold_dense_minimum_instances,
        )
        self.isolated_debris_fold_dense_minimum_aspect_ratio = finite_float(
            "isolated_debris_fold_dense_minimum_aspect_ratio",
            self.isolated_debris_fold_dense_minimum_aspect_ratio,
        )
        if self.isolated_debris_fold_dense_minimum_aspect_ratio < 1:
            raise ValueError(
                "isolated_debris_fold_dense_minimum_aspect_ratio must be at least one"
            )
        self.isolated_debris_fold_dense_connectivity_dilation_bins = positive_int(
            "isolated_debris_fold_dense_connectivity_dilation_bins",
            self.isolated_debris_fold_dense_connectivity_dilation_bins,
        )
        self.isolated_debris_fold_dense_maximum_mean_red_blue_difference = finite_float(
            "isolated_debris_fold_dense_maximum_mean_red_blue_difference",
            self.isolated_debris_fold_dense_maximum_mean_red_blue_difference,
        )
        self.isolated_debris_fold_dense_maximum_mean_intensity = finite_float(
            "isolated_debris_fold_dense_maximum_mean_intensity",
            self.isolated_debris_fold_dense_maximum_mean_intensity,
        )
        if not 0 < self.isolated_debris_fold_dense_maximum_mean_intensity <= 255:
            raise ValueError(
                "isolated_debris_fold_dense_maximum_mean_intensity must be between "
                "zero and 255"
            )
        for name in (
            "isolated_debris_detached_minimum_instance_area_um2",
            "isolated_debris_detached_minimum_component_area_um2",
        ):
            value = finite_float(name, getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            setattr(self, name, value)
        if (
            self.isolated_debris_detached_minimum_component_area_um2
            < self.isolated_debris_detached_minimum_instance_area_um2
        ):
            raise ValueError(
                "isolated_debris_detached_minimum_component_area_um2 must not "
                "be smaller than the instance area"
            )
        self.isolated_debris_detached_minimum_instances = positive_int(
            "isolated_debris_detached_minimum_instances",
            self.isolated_debris_detached_minimum_instances,
        )
        self.isolated_debris_detached_minimum_aspect_ratio = finite_float(
            "isolated_debris_detached_minimum_aspect_ratio",
            self.isolated_debris_detached_minimum_aspect_ratio,
        )
        if self.isolated_debris_detached_minimum_aspect_ratio < 1:
            raise ValueError(
                "isolated_debris_detached_minimum_aspect_ratio must be at least one"
            )
        self.isolated_debris_detached_maximum_mean_red_blue_difference = finite_float(
            "isolated_debris_detached_maximum_mean_red_blue_difference",
            self.isolated_debris_detached_maximum_mean_red_blue_difference,
        )
        self.isolated_debris_detached_maximum_mean_intensity = finite_float(
            "isolated_debris_detached_maximum_mean_intensity",
            self.isolated_debris_detached_maximum_mean_intensity,
        )
        if not 0 < self.isolated_debris_detached_maximum_mean_intensity <= 255:
            raise ValueError(
                "isolated_debris_detached_maximum_mean_intensity must be between "
                "zero and 255"
            )
        self.isolated_debris_detached_context_window_size = positive_int(
            "isolated_debris_detached_context_window_size",
            self.isolated_debris_detached_context_window_size,
        )
        if (
            self.isolated_debris_detached_context_window_size
            % self.isolated_debris_organized_bin_size
        ):
            raise ValueError(
                "isolated_debris_detached_context_window_size must be a multiple "
                "of the organized bin size"
            )
        self.isolated_debris_detached_maximum_prediction_fraction = finite_float(
            "isolated_debris_detached_maximum_prediction_fraction",
            self.isolated_debris_detached_maximum_prediction_fraction,
        )
        if not 0 < self.isolated_debris_detached_maximum_prediction_fraction <= 1:
            raise ValueError(
                "isolated_debris_detached_maximum_prediction_fraction must be "
                "between zero and one"
            )
        for name in (
            "isolated_debris_detached_compact_minimum_instance_area_um2",
            "isolated_debris_detached_compact_minimum_component_area_um2",
        ):
            value = finite_float(name, getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            setattr(self, name, value)
        if (
            self.isolated_debris_detached_compact_minimum_component_area_um2
            < self.isolated_debris_detached_compact_minimum_instance_area_um2
        ):
            raise ValueError(
                "isolated_debris_detached_compact_minimum_component_area_um2 "
                "must not be smaller than the instance area"
            )
        self.isolated_debris_detached_compact_minimum_instances = positive_int(
            "isolated_debris_detached_compact_minimum_instances",
            self.isolated_debris_detached_compact_minimum_instances,
        )
        self.isolated_debris_detached_compact_maximum_mean_red_blue_difference = (
            finite_float(
                "isolated_debris_detached_compact_maximum_mean_red_blue_difference",
                self.isolated_debris_detached_compact_maximum_mean_red_blue_difference,
            )
        )
        self.isolated_debris_detached_compact_maximum_mean_intensity = finite_float(
            "isolated_debris_detached_compact_maximum_mean_intensity",
            self.isolated_debris_detached_compact_maximum_mean_intensity,
        )
        if not (
            0 < self.isolated_debris_detached_compact_maximum_mean_intensity <= 255
        ):
            raise ValueError(
                "isolated_debris_detached_compact_maximum_mean_intensity must be "
                "between zero and 255"
            )
        self.isolated_debris_glass_minimum_area_um2 = finite_float(
            "isolated_debris_glass_minimum_area_um2",
            self.isolated_debris_glass_minimum_area_um2,
        )
        if self.isolated_debris_glass_minimum_area_um2 <= 0:
            raise ValueError("isolated_debris_glass_minimum_area_um2 must be positive")
        if (
            self.isolated_debris_glass_context_window_size <= 0
            or self.isolated_debris_glass_context_window_size
            % self.isolated_debris_organized_bin_size
        ):
            raise ValueError(
                "isolated_debris_glass_context_window_size must be a positive "
                "multiple of the organized bin size"
            )
        for name in (
            "isolated_debris_glass_maximum_prediction_fraction",
            "isolated_debris_glass_maximum_context_stain_fraction",
            "isolated_debris_glass_minimum_context_fraction",
        ):
            value = finite_float(name, getattr(self, name))
            if not 0 < value <= 1:
                raise ValueError(f"{name} must be between zero and one")
            setattr(self, name, value)
        self.isolated_debris_glass_maximum_mean_red_blue_difference = finite_float(
            "isolated_debris_glass_maximum_mean_red_blue_difference",
            self.isolated_debris_glass_maximum_mean_red_blue_difference,
        )
        self.isolated_debris_glass_low_stain_maximum_mean_red_blue_difference = (
            finite_float(
                "isolated_debris_glass_low_stain_maximum_mean_red_blue_difference",
                self.isolated_debris_glass_low_stain_maximum_mean_red_blue_difference,
            )
        )
        self.isolated_debris_glass_low_stain_minimum_mean_intensity = finite_float(
            "isolated_debris_glass_low_stain_minimum_mean_intensity",
            self.isolated_debris_glass_low_stain_minimum_mean_intensity,
        )
        self.isolated_debris_glass_minimum_mean_intensity = finite_float(
            "isolated_debris_glass_minimum_mean_intensity",
            self.isolated_debris_glass_minimum_mean_intensity,
        )
        if not 0 < self.isolated_debris_glass_minimum_mean_intensity <= 255:
            raise ValueError(
                "isolated_debris_glass_minimum_mean_intensity must be between "
                "zero and 255"
            )
        if not 0 < self.isolated_debris_glass_low_stain_minimum_mean_intensity <= 255:
            raise ValueError(
                "isolated_debris_glass_low_stain_minimum_mean_intensity must be "
                "between zero and 255"
            )
        for name in (
            "isolated_debris_self_dense_glass_gate",
            "isolated_debris_oversized_brown_gate",
            "isolated_debris_clustered_brown_gate",
            "isolated_debris_diffuse_degenerated_gate",
            "isolated_debris_neutral_precipitate_gate",
            "reconcile_enclosed_cytoplasmic_children",
            "isolated_debris_protect_organized_from_necrotic",
            "isolated_debris_organized_necrotic_protection_use_eligible_context",
            independent_nuclear_context_name,
            "isolated_debris_organized_necrotic_sparse_glass_shape_gate",
        ):
            require_bool(name, getattr(self, name))
        enabled_refinements = (
            self.isolated_debris_self_dense_glass_gate
            or self.isolated_debris_oversized_brown_gate
            or self.isolated_debris_clustered_brown_gate
            or self.isolated_debris_diffuse_degenerated_gate
            or self.isolated_debris_neutral_precipitate_gate
            or self.reconcile_enclosed_cytoplasmic_children
            or self.isolated_debris_protect_organized_from_necrotic
            or self.isolated_debris_organized_necrotic_protection_use_eligible_context
            or getattr(self, independent_nuclear_context_name)
            or self.isolated_debris_organized_necrotic_sparse_glass_shape_gate
        )
        if enabled_refinements and not self.isolated_debris_gate:
            raise ValueError(
                "validated isolated-debris refinements require isolated_debris_gate"
            )
        organized_necrotic_area_name = (
            "isolated_debris_organized_necrotic_protection_maximum_instance_area_um2"
        )
        if getattr(self, organized_necrotic_area_name) is not None:
            maximum_area = finite_float(
                organized_necrotic_area_name,
                getattr(self, organized_necrotic_area_name),
            )
            if maximum_area <= 0:
                raise ValueError(
                    "organized-necrotic protection maximum instance area must be "
                    "positive"
                )
            setattr(self, organized_necrotic_area_name, maximum_area)
        organized_necrotic_density_name = (
            "isolated_debris_organized_necrotic_protection_"
            "minimum_local_prediction_fraction"
        )
        if getattr(self, organized_necrotic_density_name) is not None:
            minimum_density = finite_float(
                organized_necrotic_density_name,
                getattr(self, organized_necrotic_density_name),
            )
            if not 0 < minimum_density <= 1:
                raise ValueError(
                    "organized-necrotic protection minimum local prediction "
                    "fraction must be between zero and one"
                )
            setattr(self, organized_necrotic_density_name, minimum_density)
        if (
            self.isolated_debris_organized_necrotic_protection_use_eligible_context
            and getattr(self, organized_necrotic_density_name) is None
        ):
            raise ValueError(
                "eligible organized-necrotic context requires a minimum local "
                "prediction fraction"
            )
        if (
            self.isolated_debris_organized_necrotic_protection_use_eligible_context
            and not self.isolated_debris_protect_organized_from_necrotic
        ):
            raise ValueError(
                "eligible organized-necrotic context requires organized-necrotic "
                "protection"
            )
        if getattr(self, independent_nuclear_context_name) and not (
            self.isolated_debris_organized_necrotic_protection_use_eligible_context
        ):
            raise ValueError(
                "independent nuclear organized-necrotic context requires eligible "
                "context"
            )
        if (
            self.isolated_debris_organized_necrotic_sparse_glass_shape_gate
            and not self.isolated_debris_protect_organized_from_necrotic
        ):
            raise ValueError(
                "organized-necrotic sparse-glass shape gate requires "
                "organized-necrotic protection"
            )
        self.isolated_debris_oversized_brown_maximum_mean_intensity = finite_float(
            "isolated_debris_oversized_brown_maximum_mean_intensity",
            self.isolated_debris_oversized_brown_maximum_mean_intensity,
        )
        if not (0 < self.isolated_debris_oversized_brown_maximum_mean_intensity <= 255):
            raise ValueError(
                "isolated_debris_oversized_brown_maximum_mean_intensity must be "
                "between zero and 255"
            )
        self.isolated_debris_necrotic_minimum_component_area_um2 = finite_float(
            "isolated_debris_necrotic_minimum_component_area_um2",
            self.isolated_debris_necrotic_minimum_component_area_um2,
        )
        if self.isolated_debris_necrotic_minimum_component_area_um2 <= 0:
            raise ValueError(
                "isolated_debris_necrotic_minimum_component_area_um2 must be positive"
            )
        self.isolated_debris_necrotic_maximum_aspect_ratio = finite_float(
            "isolated_debris_necrotic_maximum_aspect_ratio",
            self.isolated_debris_necrotic_maximum_aspect_ratio,
        )
        if self.isolated_debris_necrotic_maximum_aspect_ratio < 1:
            raise ValueError(
                "isolated_debris_necrotic_maximum_aspect_ratio must be at least one"
            )
        for name in (
            "isolated_debris_necrotic_minimum_fill_fraction",
            "isolated_debris_necrotic_maximum_fill_fraction",
            "isolated_debris_necrotic_minimum_instance_fraction",
        ):
            value = finite_float(name, getattr(self, name))
            if not 0 < value <= 1:
                raise ValueError(f"{name} must be between zero and one")
            setattr(self, name, value)
        if (
            self.isolated_debris_necrotic_minimum_fill_fraction
            >= self.isolated_debris_necrotic_maximum_fill_fraction
        ):
            raise ValueError(
                "isolated_debris_necrotic_minimum_fill_fraction must be smaller "
                "than the maximum"
            )
        for name in (
            "isolated_debris_necrotic_minimum_mean_red_blue_difference",
            "isolated_debris_necrotic_maximum_mean_red_blue_difference",
        ):
            setattr(self, name, finite_float(name, getattr(self, name)))
        if (
            self.isolated_debris_necrotic_minimum_mean_red_blue_difference
            >= self.isolated_debris_necrotic_maximum_mean_red_blue_difference
        ):
            raise ValueError(
                "isolated debris necrotic minimum red-blue difference must be "
                "smaller than the maximum"
            )
        self.isolated_debris_necrotic_minimum_mean_intensity = finite_float(
            "isolated_debris_necrotic_minimum_mean_intensity",
            self.isolated_debris_necrotic_minimum_mean_intensity,
        )
        if not 0 < self.isolated_debris_necrotic_minimum_mean_intensity <= 255:
            raise ValueError(
                "isolated_debris_necrotic_minimum_mean_intensity must be between "
                "zero and 255"
            )
        self.isolated_debris_brown_minimum_component_area_um2 = finite_float(
            "isolated_debris_brown_minimum_component_area_um2",
            self.isolated_debris_brown_minimum_component_area_um2,
        )
        if self.isolated_debris_brown_minimum_component_area_um2 <= 0:
            raise ValueError(
                "isolated_debris_brown_minimum_component_area_um2 must be positive"
            )
        self.isolated_debris_brown_maximum_aspect_ratio = finite_float(
            "isolated_debris_brown_maximum_aspect_ratio",
            self.isolated_debris_brown_maximum_aspect_ratio,
        )
        if self.isolated_debris_brown_maximum_aspect_ratio < 1:
            raise ValueError(
                "isolated_debris_brown_maximum_aspect_ratio must be at least one"
            )
        for name in (
            "isolated_debris_brown_minimum_fill_fraction",
            "isolated_debris_brown_maximum_fill_fraction",
            "isolated_debris_brown_minimum_instance_fraction",
        ):
            value = finite_float(name, getattr(self, name))
            if not 0 < value <= 1:
                raise ValueError(f"{name} must be between zero and one")
            setattr(self, name, value)
        if (
            self.isolated_debris_brown_minimum_fill_fraction
            >= self.isolated_debris_brown_maximum_fill_fraction
        ):
            raise ValueError(
                "isolated_debris_brown_minimum_fill_fraction must be smaller "
                "than the maximum"
            )
        self.isolated_debris_brown_minimum_mean_red_blue_difference = finite_float(
            "isolated_debris_brown_minimum_mean_red_blue_difference",
            self.isolated_debris_brown_minimum_mean_red_blue_difference,
        )
        self.isolated_debris_brown_maximum_mean_intensity = finite_float(
            "isolated_debris_brown_maximum_mean_intensity",
            self.isolated_debris_brown_maximum_mean_intensity,
        )
        if not 0 < self.isolated_debris_brown_maximum_mean_intensity <= 255:
            raise ValueError(
                "isolated_debris_brown_maximum_mean_intensity must be between "
                "zero and 255"
            )
        self.isolated_debris_brown_maximum_source_component_area_um2 = finite_float(
            "isolated_debris_brown_maximum_source_component_area_um2",
            self.isolated_debris_brown_maximum_source_component_area_um2,
        )
        if self.isolated_debris_brown_maximum_source_component_area_um2 <= 0:
            raise ValueError(
                "isolated_debris_brown_maximum_source_component_area_um2 must "
                "be positive"
            )
        self.isolated_debris_brown_source_maximum_mean_intensity = finite_float(
            "isolated_debris_brown_source_maximum_mean_intensity",
            self.isolated_debris_brown_source_maximum_mean_intensity,
        )
        if not 0 < self.isolated_debris_brown_source_maximum_mean_intensity <= 255:
            raise ValueError(
                "isolated_debris_brown_source_maximum_mean_intensity must be "
                "between zero and 255"
            )
        if (
            not isinstance(self.isolated_debris_brown_source_dilation_bins, int)
            or isinstance(self.isolated_debris_brown_source_dilation_bins, bool)
            or self.isolated_debris_brown_source_dilation_bins < 0
        ):
            raise ValueError(
                "isolated_debris_brown_source_dilation_bins must be a "
                "nonnegative integer"
            )
        require_bool("allow_model_download", self.allow_model_download)
        require_bool(
            "require_registration_approval", self.require_registration_approval
        )
        require_bool("keep_tile_masks", self.keep_tile_masks)


def load_cell_config(path: Path | str) -> CellSegmentationConfig:
    """Load JSON or TOML without importing numerical or WSI dependencies."""

    source = Path(path)
    if source.suffix.lower() == ".json":
        payload: dict[str, Any] = json.loads(source.read_text())
    elif source.suffix.lower() in {".toml", ".tml"}:
        if sys.version_info >= (3, 11):
            import tomllib
        else:
            import tomli as tomllib
        payload = tomllib.loads(source.read_text())
    else:
        raise ValueError("cell config must be JSON or TOML")
    values = dict(payload)
    values["sections"] = tuple(values.get("sections", ()))
    return CellSegmentationConfig(**values)
