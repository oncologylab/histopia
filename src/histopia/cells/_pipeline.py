"""Resumable native-WSI cell-boundary inference and stitching."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from histopia._atomic import write_json_atomic
from histopia._vips_image import normalize_vips_rgb_uchar
from histopia.cells._algorithms import (
    constrain_labels_to_tissue,
    hematoxylin_concentration,
    local_stain_evidence,
)
from histopia.cells._cellpose import (
    CellposeRuntime,
    load_cellpose_runtime,
    runtime_provenance,
    segment_tiles,
)
from histopia.cells._config import CellSegmentationConfig
from histopia.cells._dense_small_cell_recovery import (
    DENSE_SMALL_CELL_RECOVERY_ALGORITHM_VERSIONS,
    DenseSmallCellRecoveryManifest,
    dense_small_cell_recovery_manifest_profile,
    dense_small_cell_recovery_profile,
    load_dense_small_cell_recovery_manifest,
)
from histopia.cells._preflight import (
    CellPreflight,
    CellPreflightSlide,
    load_cell_preflight,
    preflight_cell_run,
    write_cell_preflight,
)
from histopia.cells._promotion import (
    CELL_METHOD_REFERENCE_NAME,
    CELL_METHOD_REFERENCE_PROFILE,
    CELL_METHOD_REFERENCE_SHA256,
    CELL_SATELLITE_PROTECTION_ALGORITHM_VERSION,
    CELL_SATELLITE_PROTECTION_METHOD_PROFILE,
    CELL_SATELLITE_PROTECTION_MINIMUM_MASK_DISTANCE_UM,
    CELL_SATELLITE_PROTECTION_MINIMUM_SAMPLE_FRACTION,
    CELL_SATELLITE_PROTECTION_SCOPE,
)
from histopia.cells._result import validate_cell_result_index, write_cell_result
from histopia.cells._tiles import CellTile, DiskLabelCanvas, make_tiles, tile_starts
from histopia.compute import configure_vips_threads

_ALGORITHM_VERSION = 75
_ORGANIZED_ESCAPE_ALGORITHM_VERSION = 87
_ORGANIZED_ESCAPE_METHOD_PROFILE = "combined-containment-cpsam-wsi-v82"
_ORGANIZED_ESCAPE_SCOPE = "post-inference-organized-escape-exclusion-v1"
_EVIDENCE_BACKGROUND_PERCENTILE = 98.0
_EVIDENCE_MINIMUM_OPTICAL_DENSITY = 0.08
_EVIDENCE_MINIMUM_INSTANCE_FRACTION = 0.1
_EVIDENCE_BLOCK_SIZE = 4096
_ORGANIZED_NECROTIC_MINIMUM_DENSITY_FIELD = (
    "isolated_debris_organized_necrotic_protection_minimum_local_prediction_fraction"
)
_ORGANIZED_NECROTIC_MAXIMUM_AREA_FIELD = (
    "isolated_debris_organized_necrotic_protection_maximum_instance_area_um2"
)
_ORGANIZED_NECROTIC_INDEPENDENT_CONTEXT_FIELD = (
    "isolated_debris_organized_necrotic_protection_require_independent_nuclear_context"
)
_ORGANIZED_NECROTIC_SPARSE_GLASS_CONTEXT_WINDOW = 128
_ORGANIZED_NECROTIC_SPARSE_GLASS_MAXIMUM_MEAN_INTENSITY = 220.0
_ORGANIZED_NECROTIC_SPARSE_GLASS_MINIMUM_TISSUE_FRACTION = 0.14
_ORGANIZED_NECROTIC_SPARSE_GLASS_MAXIMUM_SAMPLED_FILL_FRACTION = 0.40
_ORGANIZED_NECROTIC_SPARSE_GLASS_MINIMUM_SAMPLED_ELONGATION = 2.50


@dataclass(frozen=True, slots=True)
class _TileCacheReuse:
    """One provenance-validated source of native Cellpose tile predictions."""

    cache_root: Path
    preflight: CellPreflight
    slide: CellPreflightSlide
    profile_fingerprints: tuple[str, ...]


def _active_method_profile(config: CellSegmentationConfig) -> tuple[int, str]:
    """Select the immutable result profile for the requested evidence policy."""

    if config.dense_small_cell_recovery_manifest is not None:
        organized_necrotic_area = getattr(
            config, _ORGANIZED_NECROTIC_MAXIMUM_AREA_FIELD
        )
        organized_necrotic_density = getattr(
            config, _ORGANIZED_NECROTIC_MINIMUM_DENSITY_FIELD
        )
        independent_context = getattr(
            config, _ORGANIZED_NECROTIC_INDEPENDENT_CONTEXT_FIELD
        )
        incompatible_refinements = (
            config.isolated_debris_satellite_minimum_mask_distance_um is not None
            or config.isolated_debris_self_dense_glass_gate
            or config.isolated_debris_oversized_brown_gate
            or config.isolated_debris_clustered_brown_gate
            or config.isolated_debris_diffuse_degenerated_gate
            or config.isolated_debris_neutral_precipitate_gate
            or config.reconcile_enclosed_cytoplasmic_children
            or config.isolated_debris_protect_organized_from_necrotic
            or config.isolated_debris_organized_necrotic_protection_use_eligible_context
            or independent_context
            or config.isolated_debris_organized_necrotic_sparse_glass_shape_gate
            or organized_necrotic_area is not None
            or organized_necrotic_density is not None
        )
        if (
            not config.isolated_debris_gate
            or not config.isolated_debris_allow_nuclear_escape
            or not config.isolated_debris_allow_organized_escape
            or not config.require_complete_tile_cache_reuse
            or config.tile_cache_reuse_run is None
            or len(config.sections) != 1
            or incompatible_refinements
        ):
            raise ValueError(
                "dense-small-cell recovery requires one explicit section, the "
                "otherwise exact v75 evidence policy, and complete sealed raw-tile "
                "reuse"
            )
        return dense_small_cell_recovery_manifest_profile(
            config.dense_small_cell_recovery_manifest
        )
    satellite_distance = config.isolated_debris_satellite_minimum_mask_distance_um
    if satellite_distance is not None:
        organized_necrotic_area = getattr(
            config, _ORGANIZED_NECROTIC_MAXIMUM_AREA_FIELD
        )
        independent_context = getattr(
            config, _ORGANIZED_NECROTIC_INDEPENDENT_CONTEXT_FIELD
        )
        incompatible_refinements = (
            config.isolated_debris_self_dense_glass_gate
            or config.isolated_debris_oversized_brown_gate
            or config.isolated_debris_clustered_brown_gate
            or config.isolated_debris_diffuse_degenerated_gate
            or config.isolated_debris_neutral_precipitate_gate
            or config.reconcile_enclosed_cytoplasmic_children
            or config.isolated_debris_protect_organized_from_necrotic
            or config.isolated_debris_organized_necrotic_protection_use_eligible_context
            or independent_context
            or config.isolated_debris_organized_necrotic_sparse_glass_shape_gate
            or organized_necrotic_area is not None
            or getattr(config, _ORGANIZED_NECROTIC_MINIMUM_DENSITY_FIELD) is not None
        )
        if (
            not config.isolated_debris_gate
            or not config.isolated_debris_allow_nuclear_escape
            or not config.isolated_debris_allow_organized_escape
            or satellite_distance != CELL_SATELLITE_PROTECTION_MINIMUM_MASK_DISTANCE_UM
            or not config.require_complete_tile_cache_reuse
            or config.tile_cache_reuse_run is None
            or len(config.sections) != 1
            or incompatible_refinements
        ):
            raise ValueError(
                "satellite interior protection requires the validated 200-um "
                "distance, one explicit section, the otherwise exact v75 evidence "
                "policy, and complete sealed raw-tile reuse"
            )
        return (
            CELL_SATELLITE_PROTECTION_ALGORITHM_VERSION,
            CELL_SATELLITE_PROTECTION_METHOD_PROFILE,
        )
    if config.isolated_debris_allow_organized_escape:
        return _ALGORITHM_VERSION, CELL_METHOD_REFERENCE_PROFILE
    if (
        config.isolated_debris_allow_nuclear_escape
        or not config.require_complete_tile_cache_reuse
        or config.tile_cache_reuse_run is None
    ):
        raise ValueError(
            "organized-escape exclusion requires nuclear escape disabled and "
            "complete sealed raw-tile reuse"
        )
    return _ORGANIZED_ESCAPE_ALGORITHM_VERSION, _ORGANIZED_ESCAPE_METHOD_PROFILE


def run_cell_segmentation(
    config: CellSegmentationConfig,
    *,
    processing_sections: tuple[str, ...] | None = None,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Segment native slides, optionally as a non-final distributed worker.

    ``processing_sections`` limits computation without changing the complete
    preflight or profile fingerprints. This lets disjoint workers populate
    independently validated section checkpoints in one run directory. A worker
    never writes ``cell_result.json``; a later unrestricted invocation reopens
    and validates every checkpoint before sealing the authoritative result.
    """

    started = time.perf_counter()
    algorithm_version, method_profile = _active_method_profile(config)
    configure_vips_threads(None)
    preflight = preflight_cell_run(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    write_cell_preflight(preflight, config.output_dir / "preflight.json")
    runtime: CellposeRuntime | None
    source_result: dict[str, object] | None = None
    filter_upgrade: dict[str, object] | None = None
    if config.require_complete_tile_cache_reuse:
        assert config.tile_cache_reuse_run is not None
        source_result = validate_cell_result_index(config.tile_cache_reuse_run)
        if (
            algorithm_version
            in {
                _ORGANIZED_ESCAPE_ALGORITHM_VERSION,
                CELL_SATELLITE_PROTECTION_ALGORITHM_VERSION,
                *DENSE_SMALL_CELL_RECOVERY_ALGORITHM_VERSIONS,
            }
            and source_result.get("algorithm_version") != 75
        ):
            raise ValueError(
                "cache-only evidence refinement requires an exact algorithm-75 source"
            )
        if algorithm_version == CELL_SATELLITE_PROTECTION_ALGORITHM_VERSION:
            filter_upgrade = {
                "source_algorithm_version": source_result.get("algorithm_version"),
                "source_result_fingerprint": source_result.get("fingerprint"),
                "scope": CELL_SATELLITE_PROTECTION_SCOPE,
            }
        source_model = source_result.get("model")
        if not isinstance(source_model, dict):
            raise ValueError("tile cache reuse source model provenance is invalid")
        runtime = None
        model = dict(source_model)
    else:
        runtime = load_cellpose_runtime(config)
        model = runtime_provenance(runtime)
    dense_recovery: DenseSmallCellRecoveryManifest | None = None
    if config.dense_small_cell_recovery_manifest is not None:
        if source_result is None or config.tile_cache_reuse_run is None:
            raise RuntimeError("dense-small-cell recovery source seal is missing")
        if len(preflight.slides) != 1:
            raise ValueError("dense-small-cell recovery requires exactly one slide")
        source_preflight_fingerprint = source_result.get("preflight_fingerprint")
        if not isinstance(source_preflight_fingerprint, str):
            raise ValueError("dense-small-cell recovery source preflight is invalid")
        recovery_slide = preflight.slides[0]
        dense_recovery = load_dense_small_cell_recovery_manifest(
            config.dense_small_cell_recovery_manifest,
            source_run=config.tile_cache_reuse_run,
            expected_preflight_fingerprint=source_preflight_fingerprint,
            expected_section=recovery_slide.section,
            expected_source_identity=recovery_slide.source_identity,
        )
        if dense_small_cell_recovery_profile(dense_recovery.method) != (
            algorithm_version,
            method_profile,
        ):
            raise ValueError("dense recovery method and result profile differ")
    request = _request_payload(
        config,
        method_profile=method_profile,
        dense_recovery_payload=(
            dense_recovery.request_payload() if dense_recovery is not None else None
        ),
    )
    profile_payload: dict[str, object] = {
        "algorithm_version": algorithm_version,
        "preflight_fingerprint": preflight.fingerprint,
        "model": model,
        "request": request,
    }
    if filter_upgrade is not None:
        profile_payload["filter_upgrade"] = filter_upgrade
    profile_fingerprint = _json_sha256(profile_payload)
    # Postfilter tuning does not alter CPSAM inference. Accept the exact
    # baseline-v75 tile fingerprint when only the validated dark-fold color
    # profile changed, so a cohort-specific refinement rebuilds from raw tile
    # predictions instead of either rerunning the GPU model or filtering an
    # already-filtered label pyramid a second time.
    baseline_fold_tile_profiles = _baseline_fold_tile_profile_fingerprints(
        preflight_fingerprint=preflight.fingerprint,
        model=model,
        request=request,
        algorithm_version=algorithm_version,
    )
    baseline_foam_tile_profiles = _baseline_foam_tile_profile_fingerprints(
        preflight_fingerprint=preflight.fingerprint,
        model=model,
        request=request,
        algorithm_version=algorithm_version,
    )
    # Algorithm 76 enables only the independently validated post-inference
    # refinements.  Strip those additive request keys to reconstruct the exact
    # algorithm-75 raw-tile profile before walking the older compatibility
    # chain.  Starting the chain from the v76 request would silently poison all
    # historical fingerprints even though CPSAM inference itself is unchanged.
    version_75_request = _version_75_postfilter_request(request)
    version_75_profile_fingerprint = _json_sha256(
        {
            "algorithm_version": 75,
            "preflight_fingerprint": preflight.fingerprint,
            "model": model,
            "request": version_75_request,
        }
    )
    # Algorithm 75 removes clustered mid-intensity anuclear satellites without
    # weakening the protection for isolated cells or organized tissue. Rebuild
    # algorithm 74 exactly so its native CPSAM tile cache remains reusable.
    version_74_request = json.loads(json.dumps(version_75_request))
    version_74_reference = dict(version_74_request["method_reference"])
    version_74_reference["profile"] = "combined-containment-cpsam-wsi-v70"
    version_74_request["method_reference"] = version_74_reference
    version_74_profile_fingerprint = _json_sha256(
        {
            "algorithm_version": 74,
            "preflight_fingerprint": preflight.fingerprint,
            "model": model,
            "request": version_74_request,
        }
    )
    # Algorithm 74 permits the independently gated dense-fold branch to reject
    # attached crushed tissue. Reconstruct algorithm 73 exactly for raw CPSAM
    # cache reuse.
    version_73_request = json.loads(json.dumps(version_74_request))
    version_73_reference = dict(version_73_request["method_reference"])
    version_73_reference["profile"] = "combined-containment-cpsam-wsi-v69"
    version_73_request["method_reference"] = version_73_reference
    version_73_profile_fingerprint = _json_sha256(
        {
            "algorithm_version": 73,
            "preflight_fingerprint": preflight.fingerprint,
            "model": model,
            "request": version_73_request,
        }
    )
    # Algorithm 73 adds a dark elongated dense-fold branch without weakening
    # the general nuclear safeguard. Reconstruct algorithm 72 exactly for raw
    # CPSAM cache reuse.
    version_72_request = json.loads(json.dumps(version_73_request))
    version_72_isolated = version_72_request["isolated_debris_gate"]
    for key in (
        "fold_dense_minimum_instances",
        "fold_dense_minimum_aspect_ratio",
        "fold_dense_connectivity_dilation_bins",
        "fold_dense_maximum_mean_red_blue_difference",
        "fold_dense_maximum_mean_intensity",
    ):
        version_72_isolated.pop(key, None)
    version_72_reference = dict(version_72_request["method_reference"])
    version_72_reference["profile"] = "combined-containment-cpsam-wsi-v68"
    version_72_request["method_reference"] = version_72_reference
    version_72_profile_fingerprint = _json_sha256(
        {
            "algorithm_version": 72,
            "preflight_fingerprint": preflight.fingerprint,
            "model": model,
            "request": version_72_request,
        }
    )
    # Algorithm 72 adds a compact, sparse branch for crushed fragments split
    # into two giant muted pseudo-cells. Reconstruct algorithm 71 exactly for
    # raw CPSAM cache reuse.
    version_71_request = json.loads(json.dumps(version_72_request))
    version_71_isolated = version_71_request["isolated_debris_gate"]
    for key in (
        "detached_compact_minimum_instance_area_um2",
        "detached_compact_minimum_component_area_um2",
        "detached_compact_minimum_instances",
        "detached_compact_maximum_mean_red_blue_difference",
        "detached_compact_maximum_mean_intensity",
    ):
        version_71_isolated.pop(key, None)
    version_71_reference = dict(version_71_request["method_reference"])
    version_71_reference["profile"] = "combined-containment-cpsam-wsi-v67"
    version_71_request["method_reference"] = version_71_reference
    version_71_profile_fingerprint = _json_sha256(
        {
            "algorithm_version": 71,
            "preflight_fingerprint": preflight.fingerprint,
            "model": model,
            "request": version_71_request,
        }
    )
    # Algorithm 71 expands a qualified detached cluster to its immediately
    # touching muted fragments, including model splits below the oversized
    # threshold. Reconstruct algorithm 70 for raw CPSAM cache reuse.
    version_70_request = json.loads(json.dumps(version_71_request))
    version_70_reference = dict(version_70_request["method_reference"])
    version_70_reference["profile"] = "combined-containment-cpsam-wsi-v66"
    version_70_request["method_reference"] = version_70_reference
    version_70_profile_fingerprint = _json_sha256(
        {
            "algorithm_version": 70,
            "preflight_fingerprint": preflight.fingerprint,
            "model": model,
            "request": version_70_request,
        }
    )
    # Algorithm 70 measures detached-fragment aspect ratio on candidate pixels
    # rather than the one-bin connectivity dilation. Reconstruct algorithm 69
    # exactly for raw CPSAM cache reuse.
    version_69_request = json.loads(json.dumps(version_70_request))
    version_69_reference = dict(version_69_request["method_reference"])
    version_69_reference["profile"] = "combined-containment-cpsam-wsi-v65"
    version_69_request["method_reference"] = version_69_reference
    version_69_profile_fingerprint = _json_sha256(
        {
            "algorithm_version": 69,
            "preflight_fingerprint": preflight.fingerprint,
            "model": model,
            "request": version_69_request,
        }
    )
    # Algorithm 69 adds a bounded topology gate for sparse, elongated clusters
    # of oversized muted pseudo-cells split from detached scanning/debris
    # fragments. Reconstruct algorithm 68 exactly for raw CPSAM cache reuse.
    version_68_request = json.loads(json.dumps(version_69_request))
    version_68_isolated = version_68_request["isolated_debris_gate"]
    for key in (
        "detached_minimum_instance_area_um2",
        "detached_minimum_component_area_um2",
        "detached_minimum_instances",
        "detached_minimum_aspect_ratio",
        "detached_maximum_mean_red_blue_difference",
        "detached_maximum_mean_intensity",
        "detached_context_window_size_px",
        "detached_maximum_prediction_fraction",
    ):
        version_68_isolated.pop(key, None)
    version_68_reference = dict(version_68_request["method_reference"])
    version_68_reference["profile"] = "combined-containment-cpsam-wsi-v64"
    version_68_request["method_reference"] = version_68_reference
    version_68_profile_fingerprint = _json_sha256(
        {
            "algorithm_version": 68,
            "preflight_fingerprint": preflight.fingerprint,
            "model": model,
            "request": version_68_request,
        }
    )
    # Algorithm 68 lets an instance that passes the global hematoxylin gate
    # escape only the blue-core-dependent isolated-context, micro-island, and
    # pale-glass tests.  The remaining explicit debris/artifact gates still apply. This
    # preserves real nuclei on weak-counterstain and brown nuclear IHC without
    # treating DAB chromaticity itself as nuclear evidence.  Reconstruct
    # algorithm 67 exactly for raw CPSAM tile-cache reuse.
    version_67_request = json.loads(json.dumps(version_68_request))
    version_67_reference = dict(version_67_request["method_reference"])
    version_67_reference["profile"] = "combined-containment-cpsam-wsi-v63"
    version_67_request["method_reference"] = version_67_reference
    version_67_profile_fingerprint = _json_sha256(
        {
            "algorithm_version": 67,
            "preflight_fingerprint": preflight.fingerprint,
            "model": model,
            "request": version_67_request,
        }
    )
    # Algorithm 67 replaces the broad sampled foam thresholds with a clustered
    # exact bright-interior test. Reconstruct algorithm 66 exactly for raw
    # CPSAM tile-cache reuse.
    version_66_request = json.loads(json.dumps(version_67_request))
    version_66_isolated = version_66_request["isolated_debris_gate"]
    for key in (
        "foam_core_minimum_pixels",
        "foam_core_minimum_bright_fraction",
        "foam_core_minimum_intensity",
        "foam_core_maximum_instance_area_um2",
        "foam_core_minimum_component_area_um2",
        "foam_core_maximum_aspect_ratio",
    ):
        version_66_isolated.pop(key, None)
    version_66_isolated["foam_maximum_mean_instance_pixels"] = 350.0
    version_66_isolated["foam_maximum_strong_chromatic_fraction"] = 0.15
    version_66_isolated["foam_maximum_elongated_instance_fraction"] = 0.60
    version_66_reference = dict(version_66_request["method_reference"])
    version_66_reference["profile"] = "combined-containment-cpsam-wsi-v62"
    version_66_request["method_reference"] = version_66_reference
    version_66_profile_fingerprint = _json_sha256(
        {
            "algorithm_version": 66,
            "preflight_fingerprint": preflight.fingerprint,
            "model": model,
            "request": version_66_request,
        }
    )
    # Algorithm 66 uses foam-specific sampled-size, chromatic-fraction, and
    # elongation limits measured from the rejected native field. Reconstruct
    # algorithm 65 exactly for raw CPSAM tile-cache reuse.
    version_65_request = json.loads(json.dumps(version_66_request))
    version_65_isolated = version_65_request["isolated_debris_gate"]
    version_65_isolated.pop("foam_maximum_mean_instance_pixels", None)
    version_65_isolated.pop("foam_maximum_strong_chromatic_fraction", None)
    version_65_isolated["foam_maximum_elongated_instance_fraction"] = 0.50
    version_65_reference = dict(version_65_request["method_reference"])
    version_65_reference["profile"] = "combined-containment-cpsam-wsi-v61"
    version_65_request["method_reference"] = version_65_reference
    version_65_profile_fingerprint = _json_sha256(
        {
            "algorithm_version": 65,
            "preflight_fingerprint": preflight.fingerprint,
            "model": model,
            "request": version_65_request,
        }
    )
    # Algorithm 65 permits only nuclear-supported bins immediately enclosed by
    # unsupported foam to bridge its topology. Algorithm 64 admitted every
    # predicted supported bin, which could connect the foam to surrounding
    # viable tissue and make the combined component fail the bounded shape
    # gate. Reconstruct its fingerprint for raw CPSAM tile-cache reuse.
    version_64_request = json.loads(json.dumps(version_65_request))
    version_64_reference = dict(version_64_request["method_reference"])
    version_64_reference["profile"] = "combined-containment-cpsam-wsi-v60"
    version_64_request["method_reference"] = version_64_reference
    version_64_profile_fingerprint = _json_sha256(
        {
            "algorithm_version": 64,
            "preflight_fingerprint": preflight.fingerprint,
            "model": model,
            "request": version_64_request,
        }
    )
    # Algorithm 64 lets nuclear-supported cells bridge the foam topology while
    # retaining them outside removal evidence, and accepts at most 50% nuclear
    # context in an otherwise bright/round foam field.
    version_63_request = json.loads(json.dumps(version_64_request))
    version_63_request["isolated_debris_gate"][
        "foam_maximum_context_nuclear_fraction"
    ] = 0.30
    version_63_reference = dict(version_63_request["method_reference"])
    version_63_reference["profile"] = "combined-containment-cpsam-wsi-v59"
    version_63_request["method_reference"] = version_63_reference
    version_63_profile_fingerprint = _json_sha256(
        {
            "algorithm_version": 63,
            "preflight_fingerprint": preflight.fingerprint,
            "model": model,
            "request": version_63_request,
        }
    )
    # Algorithm 63 raises only the foam-specific elongated-instance ceiling
    # from 0.45 to 0.50 after native measurement placed the luminal field
    # between those bounds while viable controls remained above 0.60.
    version_62_request = json.loads(json.dumps(version_63_request))
    version_62_request["isolated_debris_gate"][
        "foam_maximum_elongated_instance_fraction"
    ] = 0.45
    version_62_reference = dict(version_62_request["method_reference"])
    version_62_reference["profile"] = "combined-containment-cpsam-wsi-v58"
    version_62_request["method_reference"] = version_62_reference
    version_62_profile_fingerprint = _json_sha256(
        {
            "algorithm_version": 62,
            "preflight_fingerprint": preflight.fingerprint,
            "model": model,
            "request": version_62_request,
        }
    )
    # Algorithm 62 lets the sampled foam topology nominate a component without
    # using center-sampled bins as an instance-area estimate; the later exact
    # native-pixel 50% gate still decides every removed label. Reconstruct
    # algorithm 61 exactly for raw-tile reuse.
    version_61_request = json.loads(json.dumps(version_62_request))
    version_61_reference = dict(version_61_request["method_reference"])
    version_61_reference["profile"] = "combined-containment-cpsam-wsi-v57"
    version_61_request["method_reference"] = version_61_reference
    version_61_profile_fingerprint = _json_sha256(
        {
            "algorithm_version": 61,
            "preflight_fingerprint": preflight.fingerprint,
            "model": model,
            "request": version_61_request,
        }
    )
    # Algorithm 61 makes brightness a per-bin prerequisite for the foam gate,
    # preventing darker adjacent tissue from bridging into and diluting a pale
    # luminal component. Reconstruct algorithm 60 exactly for raw-tile reuse.
    version_60_request = json.loads(json.dumps(version_62_request))
    version_60_reference = dict(version_60_request["method_reference"])
    version_60_reference["profile"] = "combined-containment-cpsam-wsi-v56"
    version_60_request["method_reference"] = version_60_reference
    version_60_profile_fingerprint = _json_sha256(
        {
            "algorithm_version": 60,
            "preflight_fingerprint": preflight.fingerprint,
            "model": model,
            "request": version_60_request,
        }
    )
    # Algorithm 60 added a bright, round, low-nuclear-context compact-mosaic
    # gate for luminal foam.  Reconstruct algorithm 59 exactly so expensive raw
    # CPSAM tiles remain reusable while every derived label/QC artifact is
    # rebuilt under the new sealed filter profile.
    version_59_request = json.loads(json.dumps(request))
    version_59_isolated = version_59_request["isolated_debris_gate"]
    for key in (
        "foam_minimum_mean_intensity",
        "foam_maximum_mean_instance_pixels",
        "foam_maximum_strong_chromatic_fraction",
        "foam_maximum_context_nuclear_fraction",
        "foam_maximum_elongated_instance_fraction",
        "foam_core_minimum_pixels",
        "foam_core_minimum_bright_fraction",
        "foam_core_minimum_intensity",
        "foam_core_maximum_instance_area_um2",
        "foam_core_minimum_component_area_um2",
        "foam_core_maximum_aspect_ratio",
    ):
        version_59_isolated.pop(key, None)
    version_59_reference = dict(version_59_request["method_reference"])
    version_59_reference["profile"] = "combined-containment-cpsam-wsi-v55"
    version_59_request["method_reference"] = version_59_reference
    version_59_profile_fingerprint = _json_sha256(
        {
            "algorithm_version": 59,
            "preflight_fingerprint": preflight.fingerprint,
            "model": model,
            "request": version_59_request,
        }
    )
    # Algorithm 58 used a source-component ceiling of 220, which let very pale
    # glass-adjacent pixels bridge compact brown sloughed-cell clusters back to
    # viable tissue. Algorithm 59 lowers only that connectivity ceiling to 218;
    # the bounded component size, local sparsity, shape, color, and tissue
    # escape safeguards remain unchanged.
    version_58_request = json.loads(json.dumps(version_59_request))
    version_58_request["isolated_debris_gate"][
        "brown_source_maximum_mean_intensity"
    ] = 220.0
    version_58_reference = dict(version_58_request["method_reference"])
    version_58_reference["profile"] = "combined-containment-cpsam-wsi-v54"
    version_58_request["method_reference"] = version_58_reference
    version_58_profile_fingerprint = _json_sha256(
        {
            "algorithm_version": 58,
            "preflight_fingerprint": preflight.fingerprint,
            "model": model,
            "request": version_58_request,
        }
    )
    # Algorithm 57 added the adaptive nuclear rescue but did not seal the
    # algorithm/profile provenance into every section QC artifact. Algorithm
    # 58 makes that provenance explicit so interrupted mixed-version rebuilds
    # cannot be accepted or visually audited as current.
    version_57_request = json.loads(json.dumps(version_59_request))
    version_57_reference = dict(version_57_request["method_reference"])
    version_57_reference["profile"] = "combined-containment-cpsam-wsi-v53"
    version_57_request["method_reference"] = version_57_reference
    version_57_profile_fingerprint = _json_sha256(
        {
            "algorithm_version": 57,
            "preflight_fingerprint": preflight.fingerprint,
            "model": model,
            "request": version_57_request,
        }
    )
    # Algorithm 56 required the fixed global hematoxylin threshold even when
    # the adaptive chromatic-nucleus evidence used by the isolated-debris gate
    # had already validated an instance.  That removed intact pale stromal
    # cells on weak-counterstain IHC sections.  Algorithm 57 lets the existing
    # bounded nuclear escape rescue those instances; every downstream debris
    # and artifact gate remains in force.
    version_56_request = json.loads(json.dumps(version_59_request))
    version_56_reference = dict(version_56_request["method_reference"])
    version_56_reference["profile"] = "combined-containment-cpsam-wsi-v52"
    version_56_request["method_reference"] = version_56_reference
    version_56_profile_fingerprint = _json_sha256(
        {
            "algorithm_version": 56,
            "preflight_fingerprint": preflight.fingerprint,
            "model": model,
            "request": version_56_request,
        }
    )
    # Algorithm 55 allowed the compact-mosaic gate to remove a field with up
    # to 42.5% nuclear-supported local context.  That rejected viable compact
    # tissue on weak/neutral IHC sections despite a substantial intermixed
    # population of supported nuclei.  Algorithm 56 limits the gate to fields
    # with at most 20% supported context; the independent isolated, glass,
    # brown, and necrotic guards continue to reject particulate debris.
    version_55_request = json.loads(json.dumps(version_59_request))
    version_55_request["isolated_debris_gate"][
        "compact_unsupported_maximum_context_nuclear_fraction"
    ] = 0.425
    version_55_reference = dict(version_55_request["method_reference"])
    version_55_reference["profile"] = "combined-containment-cpsam-wsi-v51"
    version_55_request["method_reference"] = version_55_reference
    version_55_profile_fingerprint = _json_sha256(
        {
            "algorithm_version": 55,
            "preflight_fingerprint": preflight.fingerprint,
            "model": model,
            "request": version_55_request,
        }
    )
    # Algorithm 54 used low-stain context only for pale instances. Reconstruct
    # its immutable fingerprint; algorithm 55 also catches muted dark fragments
    # in low-stain fields without broadening the sparse-field color criteria.
    version_54_request = json.loads(json.dumps(version_55_request))
    version_54_isolated = version_54_request["isolated_debris_gate"]
    version_54_isolated.pop("glass_low_stain_maximum_mean_red_blue_difference")
    version_54_isolated.pop("glass_low_stain_minimum_mean_intensity")
    version_54_reference = dict(version_54_request["method_reference"])
    version_54_reference["profile"] = "combined-containment-cpsam-wsi-v50"
    version_54_request["method_reference"] = version_54_reference
    version_54_profile_fingerprint = _json_sha256(
        {
            "algorithm_version": 54,
            "preflight_fingerprint": preflight.fingerprint,
            "model": model,
            "request": version_54_request,
        }
    )
    # Algorithm 53 did not use the already-computed source-stain context in its
    # pale-artifact decision. Reconstruct its immutable fingerprint; algorithm
    # 54 also rejects pale instances whose 256 px context is below 20% stain.
    version_53_request = json.loads(json.dumps(version_54_request))
    version_53_request["isolated_debris_gate"].pop(
        "glass_maximum_context_stain_fraction"
    )
    version_53_reference = dict(version_53_request["method_reference"])
    version_53_reference["profile"] = "combined-containment-cpsam-wsi-v49"
    version_53_request["method_reference"] = version_53_reference
    version_53_profile_fingerprint = _json_sha256(
        {
            "algorithm_version": 53,
            "preflight_fingerprint": preflight.fingerprint,
            "model": model,
            "request": version_53_request,
        }
    )
    # Algorithm 52 used a 30% sparse-field cutoff. Reconstruct its immutable
    # fingerprint; algorithm 53 extends the evidence-backed cutoff to 40% to
    # remove residual particulate clusters while dense tissue remains guarded.
    version_52_request = json.loads(json.dumps(version_53_request))
    version_52_request["isolated_debris_gate"]["glass_maximum_prediction_fraction"] = (
        0.30
    )
    version_52_reference = dict(version_52_request["method_reference"])
    version_52_reference["profile"] = "combined-containment-cpsam-wsi-v48"
    version_52_request["method_reference"] = version_52_reference
    version_52_profile_fingerprint = _json_sha256(
        {
            "algorithm_version": 52,
            "preflight_fingerprint": preflight.fingerprint,
            "model": model,
            "request": version_52_request,
        }
    )
    # Algorithm 51 used exact instance color but admitted only extremely sparse
    # fields below 10% local coverage. Reconstruct its immutable fingerprint;
    # algorithm 52 retains exact color evidence across fragmented debris fields
    # below 30% coverage while dense viable tissue remains protected.
    version_51_request = json.loads(json.dumps(version_52_request))
    version_51_request["isolated_debris_gate"]["glass_maximum_prediction_fraction"] = (
        0.10
    )
    version_51_reference = dict(version_51_request["method_reference"])
    version_51_reference["profile"] = "combined-containment-cpsam-wsi-v47"
    version_51_request["method_reference"] = version_51_reference
    version_51_profile_fingerprint = _json_sha256(
        {
            "algorithm_version": 51,
            "preflight_fingerprint": preflight.fingerprint,
            "model": model,
            "request": version_51_request,
        }
    )
    # Algorithm 50 sampled sparse pale-artifact color only at the center of
    # each 4 px context bin. Reconstruct its immutable fingerprint for raw
    # tile-cache reuse; algorithm 51 uses exact per-instance color statistics.
    version_50_request = json.loads(json.dumps(version_51_request))
    version_50_request["isolated_debris_gate"][
        "glass_maximum_mean_red_blue_difference"
    ] = 0.0
    version_50_reference = dict(version_50_request["method_reference"])
    version_50_reference["profile"] = "combined-containment-cpsam-wsi-v46"
    version_50_request["method_reference"] = version_50_reference
    version_50_profile_fingerprint = _json_sha256(
        {
            "algorithm_version": 50,
            "preflight_fingerprint": preflight.fingerprint,
            "model": model,
            "request": version_50_request,
        }
    )
    # Algorithm 49 used the same inference settings but ignored brown debris
    # components below 25 square micrometres. Retain that fingerprint strictly
    # for raw tile-cache reuse; algorithm 50 reapplies the evidence gates down
    # to the minimum plausible CPSAM instance scale.
    version_49_request = json.loads(json.dumps(version_50_request))
    version_49_request["isolated_debris_gate"]["brown_minimum_component_area_um2"] = (
        25.0
    )
    version_49_reference = dict(version_49_request["method_reference"])
    version_49_reference["profile"] = "combined-containment-cpsam-wsi-v45"
    version_49_request["method_reference"] = version_49_reference
    version_49_profile_fingerprint = _json_sha256(
        {
            "algorithm_version": 49,
            "preflight_fingerprint": preflight.fingerprint,
            "model": model,
            "request": version_49_request,
        }
    )
    version_48_request = json.loads(json.dumps(version_49_request))
    version_48_request["isolated_debris_gate"]["brown_minimum_component_area_um2"] = (
        200.0
    )
    version_48_reference = dict(version_48_request["method_reference"])
    version_48_reference["profile"] = "combined-containment-cpsam-wsi-v44"
    version_48_request["method_reference"] = version_48_reference
    version_48_profile_fingerprint = _json_sha256(
        {
            "algorithm_version": 48,
            "preflight_fingerprint": preflight.fingerprint,
            "model": model,
            "request": version_48_request,
        }
    )
    version_47_request = json.loads(json.dumps(version_48_request))
    version_47_request["isolated_debris_gate"].pop(
        "brown_expands_to_source_component", None
    )
    version_47_reference = dict(version_47_request["method_reference"])
    version_47_reference["profile"] = "combined-containment-cpsam-wsi-v43"
    version_47_request["method_reference"] = version_47_reference
    version_47_profile_fingerprint = _json_sha256(
        {
            "algorithm_version": 47,
            "preflight_fingerprint": preflight.fingerprint,
            "model": model,
            "request": version_47_request,
        }
    )
    version_46_request = json.loads(json.dumps(version_47_request))
    version_46_request["isolated_debris_gate"]["brown_requires_nuclear_unsupported"] = (
        True
    )
    version_46_reference = dict(version_46_request["method_reference"])
    version_46_reference["profile"] = "combined-containment-cpsam-wsi-v42"
    version_46_request["method_reference"] = version_46_reference
    version_46_profile_fingerprint = _json_sha256(
        {
            "algorithm_version": 46,
            "preflight_fingerprint": preflight.fingerprint,
            "model": model,
            "request": version_46_request,
        }
    )
    version_45_request = json.loads(json.dumps(version_46_request))
    version_45_request["isolated_debris_gate"]["brown_source_dilation_bins"] = 2
    version_45_reference = dict(version_45_request["method_reference"])
    version_45_reference["profile"] = "combined-containment-cpsam-wsi-v41"
    version_45_request["method_reference"] = version_45_reference
    version_45_profile_fingerprint = _json_sha256(
        {
            "algorithm_version": 45,
            "preflight_fingerprint": preflight.fingerprint,
            "model": model,
            "request": version_45_request,
        }
    )
    version_44_request = json.loads(json.dumps(version_45_request))
    version_44_isolated = version_44_request["isolated_debris_gate"]
    version_44_isolated["brown_minimum_component_area_um2"] = 750.0
    version_44_isolated["brown_minimum_fill_fraction"] = 0.15
    version_44_isolated.pop("brown_maximum_source_component_area_um2", None)
    version_44_isolated.pop("brown_source_maximum_mean_intensity", None)
    version_44_isolated.pop("brown_source_dilation_bins", None)
    version_44_reference = dict(version_44_request["method_reference"])
    version_44_reference["profile"] = "combined-containment-cpsam-wsi-v40"
    version_44_request["method_reference"] = version_44_reference
    version_44_profile_fingerprint = _json_sha256(
        {
            "algorithm_version": 44,
            "preflight_fingerprint": preflight.fingerprint,
            "model": model,
            "request": version_44_request,
        }
    )
    version_43_request = json.loads(json.dumps(version_44_request))
    version_43_isolated = version_43_request["isolated_debris_gate"]
    version_43_isolated.pop("necrotic_window_size_px", None)
    version_43_isolated.pop("necrotic_window_overlap_px", None)
    version_43_isolated.pop("necrotic_context_window_size_px", None)
    version_43_isolated.pop("necrotic_maximum_local_prediction_fraction", None)
    version_43_isolated["necrotic_requires_nuclear_unsupported"] = True
    version_43_reference = dict(version_43_request["method_reference"])
    version_43_reference["profile"] = "combined-containment-cpsam-wsi-v39"
    version_43_request["method_reference"] = version_43_reference
    version_43_profile_fingerprint = _json_sha256(
        {
            "algorithm_version": 43,
            "preflight_fingerprint": preflight.fingerprint,
            "model": model,
            "request": version_43_request,
        }
    )
    version_42_request = json.loads(json.dumps(version_43_request))
    version_42_request["isolated_debris_gate"].pop(
        "necrotic_requires_nuclear_unsupported", None
    )
    version_42_reference = dict(version_42_request["method_reference"])
    version_42_reference["profile"] = "combined-containment-cpsam-wsi-v38"
    version_42_request["method_reference"] = version_42_reference
    version_42_profile_fingerprint = _json_sha256(
        {
            "algorithm_version": 42,
            "preflight_fingerprint": preflight.fingerprint,
            "model": model,
            "request": version_42_request,
        }
    )
    version_41_request = json.loads(json.dumps(version_42_request))
    version_41_isolated = version_41_request["isolated_debris_gate"]
    version_41_isolated["brown_minimum_component_area_um2"] = 5000.0
    version_41_isolated["brown_minimum_fill_fraction"] = 0.22
    version_41_reference = dict(version_41_request["method_reference"])
    version_41_reference["profile"] = "combined-containment-cpsam-wsi-v37"
    version_41_request["method_reference"] = version_41_reference
    version_41_profile_fingerprint = _json_sha256(
        {
            "algorithm_version": 41,
            "preflight_fingerprint": preflight.fingerprint,
            "model": model,
            "request": version_41_request,
        }
    )
    version_40_request = json.loads(json.dumps(version_41_request))
    version_40_isolated = version_40_request["isolated_debris_gate"]
    version_40_isolated.pop("brown_context_window_size_px", None)
    version_40_isolated.pop("brown_maximum_local_prediction_fraction", None)
    version_40_isolated["brown_preserves_organized_tissue"] = True
    version_40_reference = dict(version_40_request["method_reference"])
    version_40_reference["profile"] = "combined-containment-cpsam-wsi-v36"
    version_40_request["method_reference"] = version_40_reference
    version_40_profile_fingerprint = _json_sha256(
        {
            "algorithm_version": 40,
            "preflight_fingerprint": preflight.fingerprint,
            "model": model,
            "request": version_40_request,
        }
    )
    version_39_request = json.loads(json.dumps(version_40_request))
    version_39_request["isolated_debris_gate"].pop(
        "brown_preserves_organized_tissue", None
    )
    version_39_reference = dict(version_39_request["method_reference"])
    version_39_reference["profile"] = "combined-containment-cpsam-wsi-v35"
    version_39_request["method_reference"] = version_39_reference
    version_39_profile_fingerprint = _json_sha256(
        {
            "algorithm_version": 39,
            "preflight_fingerprint": preflight.fingerprint,
            "model": model,
            "request": version_39_request,
        }
    )
    version_38_request = json.loads(json.dumps(version_39_request))
    version_38_isolated = version_38_request["isolated_debris_gate"]
    version_38_isolated.pop("brown_window_size_px", None)
    version_38_isolated.pop("brown_window_overlap_px", None)
    version_38_reference = dict(version_38_request["method_reference"])
    version_38_reference["profile"] = "combined-containment-cpsam-wsi-v34"
    version_38_request["method_reference"] = version_38_reference
    version_38_profile_fingerprint = _json_sha256(
        {
            "algorithm_version": 38,
            "preflight_fingerprint": preflight.fingerprint,
            "model": model,
            "request": version_38_request,
        }
    )
    version_37_request = json.loads(json.dumps(version_38_request))
    version_37_request["isolated_debris_gate"].pop(
        "brown_requires_nuclear_unsupported", None
    )
    version_37_reference = dict(version_37_request["method_reference"])
    version_37_reference["profile"] = "combined-containment-cpsam-wsi-v33"
    version_37_request["method_reference"] = version_37_reference
    version_37_profile_fingerprint = _json_sha256(
        {
            "algorithm_version": 37,
            "preflight_fingerprint": preflight.fingerprint,
            "model": model,
            "request": version_37_request,
        }
    )
    version_36_request = json.loads(json.dumps(version_37_request))
    version_36_request["isolated_debris_gate"]["brown_minimum_fill_fraction"] = 0.20
    version_36_reference = dict(version_36_request["method_reference"])
    version_36_reference["profile"] = "combined-containment-cpsam-wsi-v32"
    version_36_request["method_reference"] = version_36_reference
    version_36_profile_fingerprint = _json_sha256(
        {
            "algorithm_version": 36,
            "preflight_fingerprint": preflight.fingerprint,
            "model": model,
            "request": version_36_request,
        }
    )
    version_35_request = json.loads(json.dumps(version_36_request))
    version_35_isolated = version_35_request["isolated_debris_gate"]
    for key in (
        "brown_minimum_component_area_um2",
        "brown_maximum_aspect_ratio",
        "brown_minimum_fill_fraction",
        "brown_maximum_fill_fraction",
        "brown_minimum_mean_red_blue_difference",
        "brown_maximum_mean_intensity",
        "brown_minimum_instance_fraction",
    ):
        version_35_isolated.pop(key, None)
    version_35_reference = dict(version_35_request["method_reference"])
    version_35_reference["profile"] = "combined-containment-cpsam-wsi-v31"
    version_35_request["method_reference"] = version_35_reference
    version_35_profile_fingerprint = _json_sha256(
        {
            "algorithm_version": 35,
            "preflight_fingerprint": preflight.fingerprint,
            "model": model,
            "request": version_35_request,
        }
    )
    version_34_request = json.loads(json.dumps(version_35_request))
    version_34_reference = dict(version_34_request["method_reference"])
    version_34_reference["profile"] = "combined-containment-cpsam-wsi-v30"
    version_34_request["method_reference"] = version_34_reference
    version_34_profile_fingerprint = _json_sha256(
        {
            "algorithm_version": 34,
            "preflight_fingerprint": preflight.fingerprint,
            "model": model,
            "request": version_34_request,
        }
    )
    version_33_request = json.loads(json.dumps(version_34_request))
    version_33_isolated = version_33_request["isolated_debris_gate"]
    for key in (
        "necrotic_minimum_component_area_um2",
        "necrotic_maximum_aspect_ratio",
        "necrotic_minimum_fill_fraction",
        "necrotic_maximum_fill_fraction",
        "necrotic_minimum_mean_red_blue_difference",
        "necrotic_maximum_mean_red_blue_difference",
        "necrotic_minimum_mean_intensity",
        "necrotic_minimum_instance_fraction",
    ):
        version_33_isolated.pop(key, None)
    version_33_reference = dict(version_33_request["method_reference"])
    version_33_reference["profile"] = "combined-containment-cpsam-wsi-v29"
    version_33_request["method_reference"] = version_33_reference
    version_33_profile_fingerprint = _json_sha256(
        {
            "algorithm_version": 33,
            "preflight_fingerprint": preflight.fingerprint,
            "model": model,
            "request": version_33_request,
        }
    )
    version_32_request = json.loads(json.dumps(version_33_request))
    version_32_isolated = version_32_request["isolated_debris_gate"]
    for key in (
        "fold_minimum_instance_area_um2",
        "fold_minimum_component_area_um2",
        "fold_maximum_mean_red_blue_difference",
        "fold_maximum_mean_intensity",
        "organized_strong_red_blue_difference",
        "organized_minimum_strong_chromatic_fraction",
        "glass_minimum_area_um2",
        "glass_context_window_size_px",
        "glass_maximum_prediction_fraction",
        "glass_minimum_context_fraction",
        "glass_maximum_mean_red_blue_difference",
        "glass_minimum_mean_intensity",
    ):
        version_32_isolated.pop(key, None)
    version_32_reference = dict(version_32_request["method_reference"])
    version_32_reference["profile"] = "combined-containment-cpsam-wsi-v28"
    version_32_request["method_reference"] = version_32_reference
    version_32_profile_fingerprint = _json_sha256(
        {
            "algorithm_version": 32,
            "preflight_fingerprint": preflight.fingerprint,
            "model": model,
            "request": version_32_request,
        }
    )
    version_31_request = json.loads(json.dumps(version_32_request))
    version_31_isolated = version_31_request["isolated_debris_gate"]
    version_31_isolated["organized_minimum_component_pixels"] = 25_000
    version_31_isolated["organized_compact_minimum_component_pixels"] = 50_000
    version_31_isolated["organized_minimum_mean_instance_pixels"] = 150.0
    version_31_reference = dict(version_31_request["method_reference"])
    version_31_reference["profile"] = "combined-containment-cpsam-wsi-v27"
    version_31_request["method_reference"] = version_31_reference
    version_31_profile_fingerprint = _json_sha256(
        {
            "algorithm_version": 31,
            "preflight_fingerprint": preflight.fingerprint,
            "model": model,
            "request": version_31_request,
        }
    )
    version_30_request = json.loads(json.dumps(version_31_request))
    version_30_isolated = version_30_request["isolated_debris_gate"]
    version_30_isolated["organized_minimum_mean_instance_pixels"] = 350.0
    version_30_reference = dict(version_30_request["method_reference"])
    version_30_reference["profile"] = "combined-containment-cpsam-wsi-v26"
    version_30_request["method_reference"] = version_30_reference
    version_30_profile_fingerprint = _json_sha256(
        {
            "algorithm_version": 30,
            "preflight_fingerprint": preflight.fingerprint,
            "model": model,
            "request": version_30_request,
        }
    )
    version_29_request = json.loads(json.dumps(version_30_request))
    version_29_reference = dict(version_29_request["method_reference"])
    version_29_reference["profile"] = "combined-containment-cpsam-wsi-v25"
    version_29_request["method_reference"] = version_29_reference
    version_29_profile_fingerprint = _json_sha256(
        {
            "algorithm_version": 29,
            "preflight_fingerprint": preflight.fingerprint,
            "model": model,
            "request": version_29_request,
        }
    )
    version_28_request = json.loads(json.dumps(version_29_request))
    version_28_reference = dict(version_28_request["method_reference"])
    version_28_reference["profile"] = "combined-containment-cpsam-wsi-v24"
    version_28_request["method_reference"] = version_28_reference
    version_28_profile_fingerprint = _json_sha256(
        {
            "algorithm_version": 28,
            "preflight_fingerprint": preflight.fingerprint,
            "model": model,
            "request": version_28_request,
        }
    )
    version_27_request = json.loads(json.dumps(version_28_request))
    version_27_isolated = version_27_request["isolated_debris_gate"]
    version_27_isolated["organized_minimum_bin_occupancy_fraction"] = 0.25
    version_27_reference = dict(version_27_request["method_reference"])
    version_27_reference["profile"] = "combined-containment-cpsam-wsi-v23"
    version_27_request["method_reference"] = version_27_reference
    version_27_profile_fingerprint = _json_sha256(
        {
            "algorithm_version": 27,
            "preflight_fingerprint": preflight.fingerprint,
            "model": model,
            "request": version_27_request,
        }
    )
    version_26_request = json.loads(json.dumps(version_27_request))
    version_26_isolated = version_26_request["isolated_debris_gate"]
    version_26_isolated.pop("compact_unsupported_elongation_ratio", None)
    version_26_isolated.pop(
        "compact_unsupported_maximum_elongated_instance_fraction", None
    )
    version_26_reference = dict(version_26_request["method_reference"])
    version_26_reference["profile"] = "combined-containment-cpsam-wsi-v22"
    version_26_request["method_reference"] = version_26_reference
    version_26_profile_fingerprint = _json_sha256(
        {
            "algorithm_version": 26,
            "preflight_fingerprint": preflight.fingerprint,
            "model": model,
            "request": version_26_request,
        }
    )
    version_25_request = json.loads(json.dumps(version_26_request))
    version_25_isolated = version_25_request["isolated_debris_gate"]
    version_25_isolated.pop("compact_unsupported_strong_red_blue_difference", None)
    version_25_isolated.pop(
        "compact_unsupported_maximum_strong_chromatic_fraction", None
    )
    version_25_reference = dict(version_25_request["method_reference"])
    version_25_reference["profile"] = "combined-containment-cpsam-wsi-v21"
    version_25_request["method_reference"] = version_25_reference
    version_25_profile_fingerprint = _json_sha256(
        {
            "algorithm_version": 25,
            "preflight_fingerprint": preflight.fingerprint,
            "model": model,
            "request": version_25_request,
        }
    )
    version_24_request = json.loads(json.dumps(version_25_request))
    version_24_reference = dict(version_24_request["method_reference"])
    version_24_reference["profile"] = "combined-containment-cpsam-wsi-v20"
    version_24_request["method_reference"] = version_24_reference
    version_24_profile_fingerprint = _json_sha256(
        {
            "algorithm_version": 24,
            "preflight_fingerprint": preflight.fingerprint,
            "model": model,
            "request": version_24_request,
        }
    )
    version_23_request = json.loads(json.dumps(version_24_request))
    version_23_isolated = version_23_request["isolated_debris_gate"]
    version_23_isolated.pop("compact_unsupported_minimum_component_fill_fraction", None)
    version_23_reference = dict(version_23_request["method_reference"])
    version_23_reference["profile"] = "combined-containment-cpsam-wsi-v19"
    version_23_request["method_reference"] = version_23_reference
    version_23_profile_fingerprint = _json_sha256(
        {
            "algorithm_version": 23,
            "preflight_fingerprint": preflight.fingerprint,
            "model": model,
            "request": version_23_request,
        }
    )
    version_22_request = json.loads(json.dumps(version_23_request))
    version_22_isolated = version_22_request["isolated_debris_gate"]
    version_22_isolated.pop(
        "compact_unsupported_maximum_context_nuclear_fraction", None
    )
    version_22_isolated["compact_unsupported_maximum_aspect_ratio"] = 4.0
    version_22_isolated["compact_unsupported_maximum_mean_red_blue_difference"] = 15.0
    version_22_reference = dict(version_22_request["method_reference"])
    version_22_reference["profile"] = "combined-containment-cpsam-wsi-v18"
    version_22_request["method_reference"] = version_22_reference
    version_22_profile_fingerprint = _json_sha256(
        {
            "algorithm_version": 22,
            "preflight_fingerprint": preflight.fingerprint,
            "model": model,
            "request": version_22_request,
        }
    )
    version_21_request = json.loads(json.dumps(version_22_request))
    version_21_isolated = version_21_request["isolated_debris_gate"]
    version_21_isolated.pop(
        "compact_unsupported_maximum_mean_red_blue_difference", None
    )
    version_21_reference = dict(version_21_request["method_reference"])
    version_21_reference["profile"] = "combined-containment-cpsam-wsi-v17"
    version_21_request["method_reference"] = version_21_reference
    version_21_profile_fingerprint = _json_sha256(
        {
            "algorithm_version": 21,
            "preflight_fingerprint": preflight.fingerprint,
            "model": model,
            "request": version_21_request,
        }
    )
    version_20_request = json.loads(json.dumps(version_21_request))
    version_20_isolated = version_20_request["isolated_debris_gate"]
    for key in (
        "compact_unsupported_minimum_area_um2",
        "compact_unsupported_maximum_aspect_ratio",
        "compact_unsupported_maximum_mean_instance_pixels",
        "compact_unsupported_minimum_instance_fraction",
    ):
        version_20_isolated.pop(key, None)
    version_20_reference = dict(version_20_request["method_reference"])
    version_20_reference["profile"] = "combined-containment-cpsam-wsi-v16"
    version_20_request["method_reference"] = version_20_reference
    version_20_profile_fingerprint = _json_sha256(
        {
            "algorithm_version": 20,
            "preflight_fingerprint": preflight.fingerprint,
            "model": model,
            "request": version_20_request,
        }
    )
    version_19_request = json.loads(json.dumps(version_20_request))
    version_19_isolated = version_19_request["isolated_debris_gate"]
    version_19_isolated.pop("oversized_chromatic_minimum_area_um2", None)
    version_19_isolated.pop("oversized_chromatic_minimum_fraction", None)
    version_19_reference = dict(version_19_request["method_reference"])
    version_19_reference["profile"] = "combined-containment-cpsam-wsi-v15"
    version_19_request["method_reference"] = version_19_reference
    version_19_profile_fingerprint = _json_sha256(
        {
            "algorithm_version": 19,
            "preflight_fingerprint": preflight.fingerprint,
            "model": model,
            "request": version_19_request,
        }
    )
    version_18_request = json.loads(json.dumps(version_19_request))
    version_18_isolated = version_18_request["isolated_debris_gate"]
    version_18_isolated.pop("organized_window_size_px", None)
    version_18_isolated.pop("organized_window_overlap_px", None)
    version_18_isolated["organized_compact_minimum_component_pixels"] = 100_000
    version_18_isolated["organized_minimum_aspect_ratio"] = 1.25
    version_18_reference = dict(version_18_request["method_reference"])
    version_18_reference["profile"] = "combined-containment-cpsam-wsi-v14"
    version_18_request["method_reference"] = version_18_reference
    version_18_profile_fingerprint = _json_sha256(
        {
            "algorithm_version": 18,
            "preflight_fingerprint": preflight.fingerprint,
            "model": model,
            "request": version_18_request,
        }
    )
    version_17_request = json.loads(json.dumps(version_18_request))
    version_17_isolated = version_17_request["isolated_debris_gate"]
    version_17_isolated.pop("organized_minimum_mean_instance_pixels", None)
    version_17_reference = dict(version_17_request["method_reference"])
    version_17_reference["profile"] = "combined-containment-cpsam-wsi-v13"
    version_17_request["method_reference"] = version_17_reference
    version_17_profile_fingerprint = _json_sha256(
        {
            "algorithm_version": 17,
            "preflight_fingerprint": preflight.fingerprint,
            "model": model,
            "request": version_17_request,
        }
    )
    version_16_request = json.loads(json.dumps(version_17_request))
    version_16_isolated = version_16_request["isolated_debris_gate"]
    for key in (
        "organized_bin_size_px",
        "organized_minimum_bin_occupancy_fraction",
        "organized_minimum_component_pixels",
        "organized_compact_minimum_component_pixels",
        "organized_minimum_aspect_ratio",
        "organized_minimum_instance_fraction",
    ):
        version_16_isolated.pop(key, None)
    version_16_reference = dict(version_16_request["method_reference"])
    version_16_reference["profile"] = "combined-containment-cpsam-wsi-v12"
    version_16_request["method_reference"] = version_16_reference
    version_16_profile_fingerprint = _json_sha256(
        {
            "algorithm_version": 16,
            "preflight_fingerprint": preflight.fingerprint,
            "model": model,
            "request": version_16_request,
        }
    )
    version_15_request = json.loads(json.dumps(version_16_request))
    version_15_isolated = version_15_request["isolated_debris_gate"]
    version_15_isolated["allow_nuclear_escape"] = False
    version_15_reference = dict(version_15_request["method_reference"])
    version_15_reference["profile"] = "combined-containment-cpsam-wsi-v11"
    version_15_request["method_reference"] = version_15_reference
    version_15_profile_fingerprint = _json_sha256(
        {
            "algorithm_version": 15,
            "preflight_fingerprint": preflight.fingerprint,
            "model": model,
            "request": version_15_request,
        }
    )
    version_14_request = json.loads(json.dumps(version_15_request))
    version_14_isolated = version_14_request["isolated_debris_gate"]
    for key in (
        "micro_bin_size_px",
        "micro_minimum_stain_fraction",
        "micro_maximum_component_bins",
        "micro_minimum_nuclear_fraction",
        "micro_minimum_instance_fraction",
    ):
        version_14_isolated.pop(key, None)
    version_14_reference = dict(version_14_request["method_reference"])
    version_14_reference["profile"] = "combined-containment-cpsam-wsi-v10"
    version_14_request["method_reference"] = version_14_reference
    version_14_profile_fingerprint = _json_sha256(
        {
            "algorithm_version": 14,
            "preflight_fingerprint": preflight.fingerprint,
            "model": model,
            "request": version_14_request,
        }
    )
    version_13_request = json.loads(json.dumps(version_14_request))
    version_13_isolated = version_13_request["isolated_debris_gate"]
    version_13_isolated["neutral_dark_maximum_chroma"] = 20
    version_13_reference = dict(version_13_request["method_reference"])
    version_13_reference["profile"] = "combined-containment-cpsam-wsi-v9"
    version_13_request["method_reference"] = version_13_reference
    version_13_profile_fingerprint = _json_sha256(
        {
            "algorithm_version": 13,
            "preflight_fingerprint": preflight.fingerprint,
            "model": model,
            "request": version_13_request,
        }
    )
    version_12_request = json.loads(json.dumps(version_13_request))
    version_12_isolated = version_12_request["isolated_debris_gate"]
    version_12_isolated.pop("very_dark_maximum_chroma", None)
    version_12_reference = dict(version_12_request["method_reference"])
    version_12_reference["profile"] = "combined-containment-cpsam-wsi-v8"
    version_12_request["method_reference"] = version_12_reference
    version_12_profile_fingerprint = _json_sha256(
        {
            "algorithm_version": 12,
            "preflight_fingerprint": preflight.fingerprint,
            "model": model,
            "request": version_12_request,
        }
    )
    version_11_request = json.loads(json.dumps(version_12_request))
    version_11_isolated = version_11_request["isolated_debris_gate"]
    for key in (
        "allow_nuclear_escape",
        "neutral_dark_maximum_value",
        "neutral_dark_maximum_chroma",
        "neutral_dark_maximum_fraction",
        "very_dark_maximum_value",
        "very_dark_maximum_fraction",
    ):
        version_11_isolated.pop(key, None)
    version_11_reference = dict(version_11_request["method_reference"])
    version_11_reference["profile"] = "combined-containment-cpsam-wsi-v7"
    version_11_request["method_reference"] = version_11_reference
    version_11_profile_fingerprint = _json_sha256(
        {
            "algorithm_version": 11,
            "preflight_fingerprint": preflight.fingerprint,
            "model": model,
            "request": version_11_request,
        }
    )
    nuclear_draft_version_11_request = json.loads(json.dumps(version_11_request))
    nuclear_draft_isolated = nuclear_draft_version_11_request["isolated_debris_gate"]
    nuclear_draft_isolated["context_bin_size_px"] = 32
    nuclear_draft_isolated.pop("context_minimum_stain_fraction", None)
    nuclear_draft_isolated["context_minimum_nuclear_fraction"] = 0.01
    nuclear_draft_isolated["context_minimum_instance_fraction"] = 0.50
    nuclear_draft_isolated.pop("component_minimum_nuclear_pixels", None)
    nuclear_draft_version_11_profile_fingerprint = _json_sha256(
        {
            "algorithm_version": 11,
            "preflight_fingerprint": preflight.fingerprint,
            "model": model,
            "request": nuclear_draft_version_11_request,
        }
    )
    stain_draft_version_11_request = json.loads(json.dumps(version_11_request))
    stain_draft_isolated = stain_draft_version_11_request["isolated_debris_gate"]
    stain_draft_isolated["context_bin_size_px"] = 64
    stain_draft_isolated["context_minimum_stain_fraction"] = 0.30
    stain_draft_isolated["context_minimum_instance_fraction"] = 0.50
    stain_draft_isolated.pop("component_minimum_nuclear_pixels", None)
    stain_draft_version_11_profile_fingerprint = _json_sha256(
        {
            "algorithm_version": 11,
            "preflight_fingerprint": preflight.fingerprint,
            "model": model,
            "request": stain_draft_version_11_request,
        }
    )
    version_10_request = json.loads(json.dumps(version_59_request))
    version_10_request.pop("isolated_debris_gate", None)
    version_10_reference = dict(version_10_request["method_reference"])
    version_10_reference["profile"] = "combined-containment-cpsam-wsi-v6"
    version_10_request["method_reference"] = version_10_reference
    version_10_profile_fingerprint = _json_sha256(
        {
            "algorithm_version": 10,
            "preflight_fingerprint": preflight.fingerprint,
            "model": model,
            "request": version_10_request,
        }
    )
    version_9_request = json.loads(json.dumps(version_10_request))
    version_9_core = version_9_request["adaptive_nuclear_core"]
    version_9_core.pop("minimum_blue_ratio", None)
    version_9_core["minimum_fraction"] = 0.02
    version_9_reference = dict(version_9_request["method_reference"])
    version_9_reference["profile"] = "combined-containment-cpsam-wsi-v5"
    version_9_request["method_reference"] = version_9_reference
    version_9_profile_fingerprint = _json_sha256(
        {
            "algorithm_version": 9,
            "preflight_fingerprint": preflight.fingerprint,
            "model": model,
            "request": version_9_request,
        }
    )
    version_8_request = dict(version_9_request)
    version_8_request.pop("adaptive_nuclear_core", None)
    version_8_reference = dict(version_8_request["method_reference"])
    version_8_reference["profile"] = "combined-containment-cpsam-wsi-v4"
    version_8_request["method_reference"] = version_8_reference
    version_8_profile_fingerprint = _json_sha256(
        {
            "algorithm_version": 8,
            "preflight_fingerprint": preflight.fingerprint,
            "model": model,
            "request": version_8_request,
        }
    )
    version_7_request = dict(version_8_request)
    version_7_request.pop("source_tissue_context", None)
    version_7_reference = dict(version_7_request["method_reference"])
    version_7_reference["profile"] = "combined-containment-cpsam-wsi-v3"
    version_7_request["method_reference"] = version_7_reference
    version_7_profile_fingerprint = _json_sha256(
        {
            "algorithm_version": 7,
            "preflight_fingerprint": preflight.fingerprint,
            "model": model,
            "request": version_7_request,
        }
    )
    version_5_request = dict(version_7_request)
    version_5_request.pop("global_nuclear_support", None)
    version_5_profile_fingerprint = _json_sha256(
        {
            "algorithm_version": 5,
            "preflight_fingerprint": preflight.fingerprint,
            "model": model,
            "request": version_5_request,
        }
    )
    historical_tile_profiles = (
        version_75_profile_fingerprint,
        version_74_profile_fingerprint,
        version_73_profile_fingerprint,
        version_72_profile_fingerprint,
        version_71_profile_fingerprint,
        version_70_profile_fingerprint,
        version_69_profile_fingerprint,
        version_68_profile_fingerprint,
        version_67_profile_fingerprint,
        version_66_profile_fingerprint,
        version_65_profile_fingerprint,
        version_64_profile_fingerprint,
        version_63_profile_fingerprint,
        version_62_profile_fingerprint,
        version_61_profile_fingerprint,
        version_60_profile_fingerprint,
        version_59_profile_fingerprint,
        version_58_profile_fingerprint,
        version_57_profile_fingerprint,
        version_56_profile_fingerprint,
        version_55_profile_fingerprint,
        version_54_profile_fingerprint,
        version_53_profile_fingerprint,
        version_52_profile_fingerprint,
        version_51_profile_fingerprint,
        version_50_profile_fingerprint,
        version_49_profile_fingerprint,
        version_48_profile_fingerprint,
        version_47_profile_fingerprint,
        version_46_profile_fingerprint,
        version_45_profile_fingerprint,
        version_44_profile_fingerprint,
        version_43_profile_fingerprint,
        version_42_profile_fingerprint,
        version_41_profile_fingerprint,
        version_40_profile_fingerprint,
        version_39_profile_fingerprint,
        version_38_profile_fingerprint,
        version_37_profile_fingerprint,
        version_36_profile_fingerprint,
        version_35_profile_fingerprint,
        version_34_profile_fingerprint,
        version_33_profile_fingerprint,
        version_32_profile_fingerprint,
        version_31_profile_fingerprint,
        version_30_profile_fingerprint,
        version_29_profile_fingerprint,
        version_28_profile_fingerprint,
        version_27_profile_fingerprint,
        version_26_profile_fingerprint,
        version_25_profile_fingerprint,
        version_24_profile_fingerprint,
        version_23_profile_fingerprint,
        version_22_profile_fingerprint,
        version_21_profile_fingerprint,
        version_20_profile_fingerprint,
        version_19_profile_fingerprint,
        version_18_profile_fingerprint,
        version_17_profile_fingerprint,
        version_16_profile_fingerprint,
        version_15_profile_fingerprint,
        version_14_profile_fingerprint,
        version_13_profile_fingerprint,
        version_12_profile_fingerprint,
        version_11_profile_fingerprint,
        nuclear_draft_version_11_profile_fingerprint,
        stain_draft_version_11_profile_fingerprint,
        version_10_profile_fingerprint,
        version_9_profile_fingerprint,
        version_8_profile_fingerprint,
        version_7_profile_fingerprint,
        version_5_profile_fingerprint,
    )
    tile_cache_reuse = _prepare_tile_cache_reuse(
        config,
        preflight,
        model=model,
        request=request,
        algorithm_version=algorithm_version,
        compatible_profile_fingerprints=historical_tile_profiles,
    )
    slides = _select_processing_slides(preflight, processing_sections)
    rows: list[dict[str, object]] = []
    for index, slide in enumerate(slides, start=1):
        _progress(
            progress,
            f"[{index}/{len(slides)}] {slide.section} {slide.slide_name}",
        )
        rows.append(
            _run_slide(
                config,
                preflight,
                slide,
                runtime,
                (
                    profile_fingerprint,
                    *baseline_fold_tile_profiles,
                    *baseline_foam_tile_profiles,
                    *historical_tile_profiles,
                ),
                filter_upgrade_profile_fingerprints=(
                    # v19 already applied the same non-idempotent evidence gates.
                    # Rebuild from its cached raw tile predictions so v20 applies
                    # the full filter stack exactly once, with only the new
                    # oversized-chromatic safeguard changing the result.
                    (10, version_10_profile_fingerprint),
                ),
                tile_cache_reuse=tile_cache_reuse.get(slide.slide_name),
                dense_recovery=dense_recovery,
                algorithm_version=algorithm_version,
                method_profile=method_profile,
                progress=progress,
            )
        )
    core = {
        "schema_version": 1,
        "algorithm_version": algorithm_version,
        "coordinate_space": "native_content_bbox",
        "preflight": "preflight.json",
        "preflight_fingerprint": preflight.fingerprint,
        "registration_result_sha256": preflight.registration_result_sha256,
        "registration_approval_sha256": preflight.registration_approval_sha256,
        "model": model,
        "request": request,
        "profile_fingerprint": profile_fingerprint,
        "slides": rows,
    }
    if algorithm_version == _ORGANIZED_ESCAPE_ALGORITHM_VERSION:
        if source_result is None:
            raise RuntimeError("organized-escape refilter source seal is missing")
        core["filter_upgrade"] = {
            "source_algorithm_version": source_result.get("algorithm_version"),
            "source_result_fingerprint": source_result.get("fingerprint"),
            "scope": _ORGANIZED_ESCAPE_SCOPE,
        }
    if algorithm_version == CELL_SATELLITE_PROTECTION_ALGORITHM_VERSION:
        if filter_upgrade is None:
            raise RuntimeError("satellite interior-protection source seal is missing")
        core["filter_upgrade"] = filter_upgrade
    if dense_recovery is not None:
        core["tile_recovery"] = dense_recovery.request_payload()
    if processing_sections is not None:
        worker_core = {
            "schema_version": 1,
            "complete_result": False,
            "algorithm_version": algorithm_version,
            "preflight_fingerprint": preflight.fingerprint,
            "profile_fingerprint": profile_fingerprint,
            "requested_sections": list(processing_sections),
            "slides": rows,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
        worker_payload = {
            **worker_core,
            "fingerprint": _json_sha256(worker_core),
        }
        worker_key = _json_sha256(
            {
                "preflight": preflight.fingerprint,
                "sections": [slide.section for slide in slides],
            }
        )[:16]
        return write_json_atomic(
            config.output_dir / "worker-checkpoints" / f"sections-{worker_key}.json",
            worker_payload,
        )
    result_path = write_cell_result(config.output_dir, core)
    result = json.loads(result_path.read_text())
    write_json_atomic(
        config.output_dir / "cell_performance.json",
        {
            "schema_version": 1,
            "result_fingerprint": result["fingerprint"],
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "section_count": len(rows),
            "device": model["device"],
        },
    )
    return result_path


def _select_processing_slides(
    preflight: CellPreflight,
    requested: tuple[str, ...] | None,
) -> tuple[CellPreflightSlide, ...]:
    """Resolve worker selectors while retaining canonical preflight order."""

    if requested is None:
        return preflight.slides
    if not requested:
        raise ValueError("processing_sections must not be empty")
    selected: set[str] = set()
    for raw_selector in requested:
        selector = raw_selector.strip()
        if not selector:
            raise ValueError("processing section selectors must not be blank")
        matches = [
            slide
            for slide in preflight.slides
            if selector
            in {slide.section, slide.slide_name, Path(slide.slide_name).stem}
        ]
        if len(matches) != 1:
            qualifier = "no" if not matches else "multiple"
            raise ValueError(
                f"processing section selector {selector!r} matched {qualifier} slides"
            )
        selected.add(matches[0].section)
    return tuple(slide for slide in preflight.slides if slide.section in selected)


def _run_slide(
    config: CellSegmentationConfig,
    preflight: CellPreflight,
    slide: CellPreflightSlide,
    runtime: CellposeRuntime | None,
    profile_fingerprints: tuple[str, ...],
    filter_upgrade_profile_fingerprints: tuple[tuple[int, str], ...],
    *,
    progress: Callable[[str], None] | None,
    tile_cache_reuse: _TileCacheReuse | None = None,
    dense_recovery: DenseSmallCellRecoveryManifest | None = None,
    algorithm_version: int = _ALGORITHM_VERSION,
    method_profile: str = CELL_METHOD_REFERENCE_PROFILE,
) -> dict[str, object]:
    completed = _load_completed_slide(
        config,
        preflight,
        slide,
        profile_fingerprints[0],
    )
    if completed is not None:
        _progress(progress, "  reusing sealed completed section")
        return completed

    pyvips = _import_pyvips()
    from PIL import Image

    source = normalize_vips_rgb_uchar(
        pyvips.Image.new_from_file(slide.source_path, access="random")
    )
    crop_x, crop_y, width, height = slide.content_bbox_xywh
    tissue_mask = np.asarray(Image.open(slide.mask_path).convert("L")) > 0
    tiles = make_tiles(
        height, width, tile_size=config.tile_size, overlap=config.overlap
    )
    cache_root = config.output_dir / ".cell-cache" / slide.section
    work_root = config.output_dir / ".cell-work" / slide.section
    canvas = DiskLabelCanvas(work_root, (height, width), resume=False)
    executed = reused = skipped = 0
    reused_from_prior_run = 0
    dense_recovery_tiles: set[str] = set()
    outside_prediction_pixels_removed = 0
    tiles_with_outside_predictions = 0
    legacy_qc: dict[str, object] | None = None
    pending: list[tuple[int, CellTile, Path, str, np.ndarray]] = []

    def merge_local(tile_index: int, tile: CellTile, local: np.ndarray) -> None:
        nonlocal outside_prediction_pixels_removed, tiles_with_outside_predictions
        tile_tissue = _tile_tissue_mask(
            tissue_mask,
            tile,
            content_shape=(height, width),
        )
        outside_count = int(np.count_nonzero((local > 0) & (~tile_tissue)))
        outside_prediction_pixels_removed += outside_count
        tiles_with_outside_predictions += int(outside_count > 0)
        constrained = constrain_labels_to_tissue(local, tile_tissue)
        canvas.merge(
            tile,
            constrained,
            match_ios=config.match_ios,
            sanitize=not bool(np.all(tile_tissue)),
        )
        if tile_index % 16 == 0:
            _progress(progress, f"  tile {tile_index}/{len(tiles)}")

    def flush_pending() -> None:
        nonlocal executed
        if not pending:
            return
        if runtime is None:
            raise RuntimeError(
                "cache-only cell refilter attempted to invoke Cellpose inference"
            )
        inferred = segment_tiles([item[4] for item in pending], runtime, config)
        if len(inferred) != len(pending):
            raise RuntimeError("Cellpose returned an incomplete tile batch")
        for (tile_index, tile, tile_path, fingerprint, _image), local in zip(
            pending, inferred, strict=True
        ):
            _write_cached_tile(tile_path, local, fingerprint)
            executed += 1
            merge_local(tile_index, tile, local)
        pending.clear()

    legacy_completed = None
    legacy_algorithm_version: int | None = None
    for candidate_version, candidate_fingerprint in filter_upgrade_profile_fingerprints:
        legacy_completed = _load_completed_slide(
            config,
            preflight,
            slide,
            candidate_fingerprint,
        )
        if legacy_completed is not None:
            legacy_algorithm_version = candidate_version
            break
    if legacy_completed is not None:
        _progress(
            progress,
            "  reusing sealed labels for filter-only upgrade from algorithm "
            f"v{legacy_algorithm_version}",
        )
        label_path = config.output_dir / str(legacy_completed["labels"])
        legacy_qc = json.loads(
            (config.output_dir / str(legacy_completed["qc"])).read_text()
        )
        canvas.next_label = _copy_labels_to_canvas(
            label_path, canvas.labels, block_size=4096
        )
        skipped = int(legacy_qc["tiles_skipped_outside_tissue"])
        reused = len(tiles) - skipped
        outside_prediction_pixels_removed = int(
            legacy_qc.get("outside_prediction_pixels_removed", 0)
        )
        tiles_with_outside_predictions = int(
            legacy_qc.get("tiles_with_outside_predictions", 0)
        )
    else:
        for tile_index, tile in enumerate(tiles, start=1):
            if not _tile_has_tissue(
                tissue_mask,
                tile,
                content_shape=(height, width),
                minimum=config.tissue_fraction_threshold,
            ):
                skipped += 1
                continue
            tile_fingerprints = tuple(
                _tile_fingerprint(preflight, slide, profile_fingerprint, tile)
                for profile_fingerprint in profile_fingerprints
            )
            tile_path = cache_root / f"{tile.key}.npz"
            local = _load_cached_tile(tile_path, tile_fingerprints)
            recovery_tile = (
                dense_recovery.tiles.get(f"{tile.key}.npz")
                if dense_recovery is not None
                else None
            )
            loaded_from_prior = False
            if local is None and tile_cache_reuse is not None:
                local = _load_reusable_tile(tile_cache_reuse, tile)
                if local is not None:
                    reused_from_prior_run += 1
                    loaded_from_prior = True
            if local is not None and recovery_tile is not None:
                # Reapply and revalidate the immutable recovery even when a
                # candidate-local tile cache already exists.  This makes a
                # safely interrupted recovery resumable without trusting that
                # the local cache was written after the replacement step.
                local, _replaced = _replace_reusable_tile_with_dense_recovery(
                    dense_recovery,
                    tile,
                    local,
                )
                _write_cached_tile(tile_path, local, tile_fingerprints[0])
                dense_recovery_tiles.add(recovery_tile.key)
            elif local is not None and loaded_from_prior:
                _write_cached_tile(tile_path, local, tile_fingerprints[0])
            if local is None:
                if config.require_complete_tile_cache_reuse:
                    raise RuntimeError(
                        "required raw tile cache is missing or stale for "
                        f"section {slide.section}, tile {tile.key}"
                    )
                crop = source.crop(
                    crop_x + tile.x0,
                    crop_y + tile.y0,
                    tile.x1 - tile.x0,
                    tile.y1 - tile.y0,
                )
                image = np.frombuffer(crop.write_to_memory(), dtype=np.uint8).reshape(
                    crop.height,
                    crop.width,
                    crop.bands,
                )
                pending.append(
                    (tile_index, tile, tile_path, tile_fingerprints[0], image)
                )
                if len(pending) >= config.inference_batch_size:
                    flush_pending()
            else:
                flush_pending()
                reused += 1
                merge_local(tile_index, tile, local)
    flush_pending()
    if dense_recovery is not None and dense_recovery_tiles != set(dense_recovery.tiles):
        missing = sorted(set(dense_recovery.tiles) - dense_recovery_tiles)
        raise RuntimeError(
            "dense recovery tiles did not match the selected native tiling: "
            + ", ".join(missing)
        )
    canvas.flush()
    labels_dir = config.output_dir / "labels"
    qc_dir = config.output_dir / "qc"
    labels_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)
    _progress(progress, "  filtering flat-background instances")
    organized_necrotic_maximum_area_um2 = (
        config.isolated_debris_organized_necrotic_protection_maximum_instance_area_um2
    )
    organized_necrotic_minimum_density = getattr(
        config, _ORGANIZED_NECROTIC_MINIMUM_DENSITY_FIELD
    )
    organized_necrotic_use_eligible_context = (
        config.isolated_debris_organized_necrotic_protection_use_eligible_context
    )
    organized_necrotic_require_independent_nuclear_context = getattr(
        config, _ORGANIZED_NECROTIC_INDEPENDENT_CONTEXT_FIELD
    )
    organized_necrotic_sparse_glass_shape_gate = (
        config.isolated_debris_organized_necrotic_sparse_glass_shape_gate
    )
    evidence_qc = _filter_flat_background_instances(
        canvas.labels,
        canvas.next_label,
        read_rgb=lambda x, y, block_width, block_height: _read_source_rgb(
            source,
            crop_x + x,
            crop_y + y,
            block_width,
            block_height,
        ),
        require_nuclear_support=config.global_nuclear_support,
        nuclear_minimum_optical_density=(config.global_nuclear_minimum_optical_density),
        nuclear_minimum_pixels=config.global_nuclear_minimum_pixels,
        nuclear_minimum_fraction=config.global_nuclear_minimum_fraction,
        require_source_tissue_context=config.source_tissue_context,
        source_tissue_context_bin_size=config.source_tissue_context_bin_size,
        source_tissue_context_minimum_fraction=(
            config.source_tissue_context_minimum_fraction
        ),
        source_tissue_context_minimum_instance_fraction=(
            config.source_tissue_context_minimum_instance_fraction
        ),
        require_adaptive_nuclear_core=config.adaptive_nuclear_core,
        adaptive_nuclear_core_percentile=(config.adaptive_nuclear_core_percentile),
        adaptive_nuclear_core_minimum_area=math.ceil(
            config.adaptive_nuclear_core_minimum_area_um2
            / (slide.mpp_xy[0] * slide.mpp_xy[1])
        ),
        adaptive_nuclear_core_minimum_area_um2=(
            config.adaptive_nuclear_core_minimum_area_um2
        ),
        adaptive_nuclear_core_minimum_pixels=(
            config.adaptive_nuclear_core_minimum_pixels
        ),
        adaptive_nuclear_core_minimum_fraction=(
            config.adaptive_nuclear_core_minimum_fraction
        ),
        adaptive_nuclear_core_minimum_blue_ratio=(
            config.adaptive_nuclear_core_minimum_blue_ratio
        ),
        require_isolated_debris_gate=config.isolated_debris_gate,
        isolated_debris_context_bin_size=(config.isolated_debris_context_bin_size),
        isolated_debris_context_minimum_stain_fraction=(
            config.isolated_debris_context_minimum_stain_fraction
        ),
        isolated_debris_context_minimum_instance_fraction=(
            config.isolated_debris_context_minimum_instance_fraction
        ),
        isolated_debris_component_minimum_nuclear_pixels=(
            config.isolated_debris_component_minimum_nuclear_pixels
        ),
        isolated_debris_allow_nuclear_escape=(
            config.isolated_debris_allow_nuclear_escape
        ),
        isolated_debris_allow_organized_escape=(
            config.isolated_debris_allow_organized_escape
        ),
        isolated_debris_nuclear_minimum_pixels=(
            config.isolated_debris_nuclear_minimum_pixels
        ),
        isolated_debris_nuclear_minimum_fraction=(
            config.isolated_debris_nuclear_minimum_fraction
        ),
        isolated_debris_nuclear_minimum_blue_ratio=(
            config.isolated_debris_nuclear_minimum_blue_ratio
        ),
        isolated_debris_neutral_dark_maximum_value=(
            config.isolated_debris_neutral_dark_maximum_value
        ),
        isolated_debris_neutral_dark_maximum_chroma=(
            config.isolated_debris_neutral_dark_maximum_chroma
        ),
        isolated_debris_neutral_dark_maximum_fraction=(
            config.isolated_debris_neutral_dark_maximum_fraction
        ),
        isolated_debris_very_dark_maximum_value=(
            config.isolated_debris_very_dark_maximum_value
        ),
        isolated_debris_very_dark_maximum_chroma=(
            config.isolated_debris_very_dark_maximum_chroma
        ),
        isolated_debris_very_dark_maximum_fraction=(
            config.isolated_debris_very_dark_maximum_fraction
        ),
        isolated_debris_micro_bin_size=(config.isolated_debris_micro_bin_size),
        isolated_debris_micro_minimum_stain_fraction=(
            config.isolated_debris_micro_minimum_stain_fraction
        ),
        isolated_debris_micro_maximum_component_bins=(
            config.isolated_debris_micro_maximum_component_bins
        ),
        isolated_debris_micro_minimum_nuclear_fraction=(
            config.isolated_debris_micro_minimum_nuclear_fraction
        ),
        isolated_debris_micro_minimum_instance_fraction=(
            config.isolated_debris_micro_minimum_instance_fraction
        ),
        isolated_debris_organized_bin_size=(config.isolated_debris_organized_bin_size),
        isolated_debris_organized_window_size=(
            config.isolated_debris_organized_window_size
        ),
        isolated_debris_organized_window_overlap=(
            config.isolated_debris_organized_window_overlap
        ),
        isolated_debris_organized_minimum_bin_occupancy_fraction=(
            config.isolated_debris_organized_minimum_bin_occupancy_fraction
        ),
        isolated_debris_organized_minimum_component_pixels=(
            config.isolated_debris_organized_minimum_component_pixels
        ),
        isolated_debris_organized_compact_minimum_component_pixels=(
            config.isolated_debris_organized_compact_minimum_component_pixels
        ),
        isolated_debris_organized_minimum_aspect_ratio=(
            config.isolated_debris_organized_minimum_aspect_ratio
        ),
        isolated_debris_organized_minimum_mean_instance_pixels=(
            config.isolated_debris_organized_minimum_mean_instance_pixels
        ),
        isolated_debris_organized_minimum_instance_fraction=(
            config.isolated_debris_organized_minimum_instance_fraction
        ),
        isolated_debris_organized_strong_red_blue_difference=(
            config.isolated_debris_organized_strong_red_blue_difference
        ),
        isolated_debris_organized_minimum_strong_chromatic_fraction=(
            config.isolated_debris_organized_minimum_strong_chromatic_fraction
        ),
        isolated_debris_compact_unsupported_minimum_area=math.ceil(
            config.isolated_debris_compact_unsupported_minimum_area_um2
            / (slide.mpp_xy[0] * slide.mpp_xy[1])
        ),
        isolated_debris_compact_unsupported_minimum_area_um2=(
            config.isolated_debris_compact_unsupported_minimum_area_um2
        ),
        isolated_debris_compact_unsupported_maximum_aspect_ratio=(
            config.isolated_debris_compact_unsupported_maximum_aspect_ratio
        ),
        isolated_debris_compact_unsupported_maximum_mean_instance_pixels=(
            config.isolated_debris_compact_unsupported_maximum_mean_instance_pixels
        ),
        isolated_debris_compact_unsupported_maximum_mean_red_blue_difference=(
            config.isolated_debris_compact_unsupported_maximum_mean_red_blue_difference
        ),
        isolated_debris_compact_unsupported_strong_red_blue_difference=(
            config.isolated_debris_compact_unsupported_strong_red_blue_difference
        ),
        isolated_debris_compact_unsupported_maximum_strong_chromatic_fraction=(
            config.isolated_debris_compact_unsupported_maximum_strong_chromatic_fraction
        ),
        isolated_debris_compact_unsupported_elongation_ratio=(
            config.isolated_debris_compact_unsupported_elongation_ratio
        ),
        isolated_debris_compact_unsupported_maximum_elongated_instance_fraction=(
            config.isolated_debris_compact_unsupported_maximum_elongated_instance_fraction
        ),
        isolated_debris_compact_unsupported_maximum_context_nuclear_fraction=(
            config.isolated_debris_compact_unsupported_maximum_context_nuclear_fraction
        ),
        isolated_debris_compact_unsupported_minimum_component_fill_fraction=(
            config.isolated_debris_compact_unsupported_minimum_component_fill_fraction
        ),
        isolated_debris_compact_unsupported_minimum_instance_fraction=(
            config.isolated_debris_compact_unsupported_minimum_instance_fraction
        ),
        isolated_debris_foam_minimum_mean_intensity=(
            config.isolated_debris_foam_minimum_mean_intensity
        ),
        isolated_debris_foam_maximum_mean_instance_pixels=(
            config.isolated_debris_foam_maximum_mean_instance_pixels
        ),
        isolated_debris_foam_maximum_strong_chromatic_fraction=(
            config.isolated_debris_foam_maximum_strong_chromatic_fraction
        ),
        isolated_debris_foam_maximum_context_nuclear_fraction=(
            config.isolated_debris_foam_maximum_context_nuclear_fraction
        ),
        isolated_debris_foam_maximum_elongated_instance_fraction=(
            config.isolated_debris_foam_maximum_elongated_instance_fraction
        ),
        isolated_debris_foam_core_minimum_pixels=(
            config.isolated_debris_foam_core_minimum_pixels
        ),
        isolated_debris_foam_core_minimum_bright_fraction=(
            config.isolated_debris_foam_core_minimum_bright_fraction
        ),
        isolated_debris_foam_core_minimum_intensity=(
            config.isolated_debris_foam_core_minimum_intensity
        ),
        isolated_debris_foam_core_maximum_instance_area=math.ceil(
            config.isolated_debris_foam_core_maximum_instance_area_um2
            / (slide.mpp_xy[0] * slide.mpp_xy[1])
        ),
        isolated_debris_foam_core_maximum_instance_area_um2=(
            config.isolated_debris_foam_core_maximum_instance_area_um2
        ),
        isolated_debris_foam_core_minimum_component_area=math.ceil(
            config.isolated_debris_foam_core_minimum_component_area_um2
            / (slide.mpp_xy[0] * slide.mpp_xy[1])
        ),
        isolated_debris_foam_core_minimum_component_area_um2=(
            config.isolated_debris_foam_core_minimum_component_area_um2
        ),
        isolated_debris_foam_core_maximum_aspect_ratio=(
            config.isolated_debris_foam_core_maximum_aspect_ratio
        ),
        isolated_debris_oversized_chromatic_minimum_area=math.ceil(
            config.isolated_debris_oversized_chromatic_minimum_area_um2
            / (slide.mpp_xy[0] * slide.mpp_xy[1])
        ),
        isolated_debris_oversized_chromatic_minimum_area_um2=(
            config.isolated_debris_oversized_chromatic_minimum_area_um2
        ),
        isolated_debris_oversized_chromatic_minimum_fraction=(
            config.isolated_debris_oversized_chromatic_minimum_fraction
        ),
        isolated_debris_fold_minimum_instance_area=math.ceil(
            config.isolated_debris_fold_minimum_instance_area_um2
            / (slide.mpp_xy[0] * slide.mpp_xy[1])
        ),
        isolated_debris_fold_minimum_instance_area_um2=(
            config.isolated_debris_fold_minimum_instance_area_um2
        ),
        isolated_debris_fold_minimum_component_area=math.ceil(
            config.isolated_debris_fold_minimum_component_area_um2
            / (slide.mpp_xy[0] * slide.mpp_xy[1])
        ),
        isolated_debris_fold_minimum_component_area_um2=(
            config.isolated_debris_fold_minimum_component_area_um2
        ),
        isolated_debris_fold_maximum_mean_red_blue_difference=(
            config.isolated_debris_fold_maximum_mean_red_blue_difference
        ),
        isolated_debris_fold_maximum_mean_intensity=(
            config.isolated_debris_fold_maximum_mean_intensity
        ),
        isolated_debris_fold_dense_minimum_instances=(
            config.isolated_debris_fold_dense_minimum_instances
        ),
        isolated_debris_fold_dense_minimum_aspect_ratio=(
            config.isolated_debris_fold_dense_minimum_aspect_ratio
        ),
        isolated_debris_fold_dense_connectivity_dilation_bins=(
            config.isolated_debris_fold_dense_connectivity_dilation_bins
        ),
        isolated_debris_fold_dense_maximum_mean_red_blue_difference=(
            config.isolated_debris_fold_dense_maximum_mean_red_blue_difference
        ),
        isolated_debris_fold_dense_maximum_mean_intensity=(
            config.isolated_debris_fold_dense_maximum_mean_intensity
        ),
        isolated_debris_detached_minimum_instance_area=math.ceil(
            config.isolated_debris_detached_minimum_instance_area_um2
            / (slide.mpp_xy[0] * slide.mpp_xy[1])
        ),
        isolated_debris_detached_minimum_instance_area_um2=(
            config.isolated_debris_detached_minimum_instance_area_um2
        ),
        isolated_debris_detached_minimum_component_area=math.ceil(
            config.isolated_debris_detached_minimum_component_area_um2
            / (slide.mpp_xy[0] * slide.mpp_xy[1])
        ),
        isolated_debris_detached_minimum_component_area_um2=(
            config.isolated_debris_detached_minimum_component_area_um2
        ),
        isolated_debris_detached_minimum_instances=(
            config.isolated_debris_detached_minimum_instances
        ),
        isolated_debris_detached_minimum_aspect_ratio=(
            config.isolated_debris_detached_minimum_aspect_ratio
        ),
        isolated_debris_detached_maximum_mean_red_blue_difference=(
            config.isolated_debris_detached_maximum_mean_red_blue_difference
        ),
        isolated_debris_detached_maximum_mean_intensity=(
            config.isolated_debris_detached_maximum_mean_intensity
        ),
        isolated_debris_detached_context_window_size=(
            config.isolated_debris_detached_context_window_size
        ),
        isolated_debris_detached_maximum_prediction_fraction=(
            config.isolated_debris_detached_maximum_prediction_fraction
        ),
        isolated_debris_detached_compact_minimum_instance_area=math.ceil(
            config.isolated_debris_detached_compact_minimum_instance_area_um2
            / (slide.mpp_xy[0] * slide.mpp_xy[1])
        ),
        isolated_debris_detached_compact_minimum_instance_area_um2=(
            config.isolated_debris_detached_compact_minimum_instance_area_um2
        ),
        isolated_debris_detached_compact_minimum_component_area=math.ceil(
            config.isolated_debris_detached_compact_minimum_component_area_um2
            / (slide.mpp_xy[0] * slide.mpp_xy[1])
        ),
        isolated_debris_detached_compact_minimum_component_area_um2=(
            config.isolated_debris_detached_compact_minimum_component_area_um2
        ),
        isolated_debris_detached_compact_minimum_instances=(
            config.isolated_debris_detached_compact_minimum_instances
        ),
        isolated_debris_detached_compact_maximum_mean_red_blue_difference=(
            config.isolated_debris_detached_compact_maximum_mean_red_blue_difference
        ),
        isolated_debris_detached_compact_maximum_mean_intensity=(
            config.isolated_debris_detached_compact_maximum_mean_intensity
        ),
        isolated_debris_satellite_maximum_instance_area=math.ceil(
            300.0 / (slide.mpp_xy[0] * slide.mpp_xy[1])
        ),
        isolated_debris_satellite_maximum_instance_area_um2=300.0,
        isolated_debris_satellite_minimum_component_area=math.ceil(
            500.0 / (slide.mpp_xy[0] * slide.mpp_xy[1])
        ),
        isolated_debris_satellite_minimum_component_area_um2=500.0,
        isolated_debris_satellite_gate=config.isolated_debris_satellite_gate,
        isolated_debris_satellite_minimum_mask_distance_um=(
            config.isolated_debris_satellite_minimum_mask_distance_um
        ),
        isolated_debris_satellite_tissue_mask=tissue_mask,
        isolated_debris_satellite_mpp_xy=slide.mpp_xy,
        isolated_debris_glass_minimum_area=math.ceil(
            config.isolated_debris_glass_minimum_area_um2
            / (slide.mpp_xy[0] * slide.mpp_xy[1])
        ),
        isolated_debris_glass_minimum_area_um2=(
            config.isolated_debris_glass_minimum_area_um2
        ),
        isolated_debris_glass_context_window_size=(
            config.isolated_debris_glass_context_window_size
        ),
        isolated_debris_glass_maximum_prediction_fraction=(
            config.isolated_debris_glass_maximum_prediction_fraction
        ),
        isolated_debris_glass_maximum_context_stain_fraction=(
            config.isolated_debris_glass_maximum_context_stain_fraction
        ),
        isolated_debris_glass_low_stain_maximum_mean_red_blue_difference=(
            config.isolated_debris_glass_low_stain_maximum_mean_red_blue_difference
        ),
        isolated_debris_glass_low_stain_minimum_mean_intensity=(
            config.isolated_debris_glass_low_stain_minimum_mean_intensity
        ),
        isolated_debris_glass_minimum_context_fraction=(
            config.isolated_debris_glass_minimum_context_fraction
        ),
        isolated_debris_glass_maximum_mean_red_blue_difference=(
            config.isolated_debris_glass_maximum_mean_red_blue_difference
        ),
        isolated_debris_glass_minimum_mean_intensity=(
            config.isolated_debris_glass_minimum_mean_intensity
        ),
        isolated_debris_self_dense_glass_gate=(
            config.isolated_debris_self_dense_glass_gate
        ),
        isolated_debris_oversized_brown_gate=(
            config.isolated_debris_oversized_brown_gate
        ),
        isolated_debris_oversized_brown_maximum_mean_intensity=(
            config.isolated_debris_oversized_brown_maximum_mean_intensity
        ),
        isolated_debris_clustered_brown_gate=(
            config.isolated_debris_clustered_brown_gate
        ),
        isolated_debris_clustered_brown_minimum_instance_area=math.ceil(
            225.0 / (slide.mpp_xy[0] * slide.mpp_xy[1])
        ),
        isolated_debris_clustered_brown_minimum_instance_area_um2=225.0,
        isolated_debris_clustered_brown_maximum_instance_area=math.ceil(
            1_200.0 / (slide.mpp_xy[0] * slide.mpp_xy[1])
        ),
        isolated_debris_clustered_brown_maximum_instance_area_um2=1_200.0,
        isolated_debris_clustered_brown_minimum_component_area=math.ceil(
            1_500.0 / (slide.mpp_xy[0] * slide.mpp_xy[1])
        ),
        isolated_debris_clustered_brown_minimum_component_area_um2=1_500.0,
        isolated_debris_diffuse_degenerated_gate=(
            config.isolated_debris_diffuse_degenerated_gate
        ),
        isolated_debris_neutral_precipitate_gate=(
            config.isolated_debris_neutral_precipitate_gate
        ),
        reconcile_enclosed_cytoplasmic_children=(
            config.reconcile_enclosed_cytoplasmic_children
        ),
        isolated_debris_necrotic_minimum_component_area=math.ceil(
            config.isolated_debris_necrotic_minimum_component_area_um2
            / (slide.mpp_xy[0] * slide.mpp_xy[1])
        ),
        isolated_debris_necrotic_minimum_component_area_um2=(
            config.isolated_debris_necrotic_minimum_component_area_um2
        ),
        isolated_debris_necrotic_maximum_aspect_ratio=(
            config.isolated_debris_necrotic_maximum_aspect_ratio
        ),
        isolated_debris_necrotic_minimum_fill_fraction=(
            config.isolated_debris_necrotic_minimum_fill_fraction
        ),
        isolated_debris_necrotic_maximum_fill_fraction=(
            config.isolated_debris_necrotic_maximum_fill_fraction
        ),
        isolated_debris_necrotic_minimum_mean_red_blue_difference=(
            config.isolated_debris_necrotic_minimum_mean_red_blue_difference
        ),
        isolated_debris_necrotic_maximum_mean_red_blue_difference=(
            config.isolated_debris_necrotic_maximum_mean_red_blue_difference
        ),
        isolated_debris_necrotic_minimum_mean_intensity=(
            config.isolated_debris_necrotic_minimum_mean_intensity
        ),
        isolated_debris_necrotic_minimum_instance_fraction=(
            config.isolated_debris_necrotic_minimum_instance_fraction
        ),
        isolated_debris_protect_organized_from_necrotic=(
            config.isolated_debris_protect_organized_from_necrotic
        ),
        isolated_debris_organized_necrotic_protection_maximum_instance_area=(
            math.ceil(
                organized_necrotic_maximum_area_um2
                / (slide.mpp_xy[0] * slide.mpp_xy[1])
            )
            if organized_necrotic_maximum_area_um2 is not None
            else None
        ),
        isolated_debris_organized_necrotic_protection_maximum_instance_area_um2=(
            organized_necrotic_maximum_area_um2
        ),
        organized_necrotic_protection_minimum_local_prediction_fraction=(
            organized_necrotic_minimum_density
        ),
        organized_necrotic_protection_use_eligible_context=(
            organized_necrotic_use_eligible_context
        ),
        organized_necrotic_protection_require_independent_nuclear_context=(
            organized_necrotic_require_independent_nuclear_context
        ),
        organized_necrotic_sparse_glass_shape_gate=(
            organized_necrotic_sparse_glass_shape_gate
        ),
        isolated_debris_brown_minimum_component_area=math.ceil(
            config.isolated_debris_brown_minimum_component_area_um2
            / (slide.mpp_xy[0] * slide.mpp_xy[1])
        ),
        isolated_debris_brown_minimum_component_area_um2=(
            config.isolated_debris_brown_minimum_component_area_um2
        ),
        isolated_debris_brown_maximum_aspect_ratio=(
            config.isolated_debris_brown_maximum_aspect_ratio
        ),
        isolated_debris_brown_minimum_fill_fraction=(
            config.isolated_debris_brown_minimum_fill_fraction
        ),
        isolated_debris_brown_maximum_fill_fraction=(
            config.isolated_debris_brown_maximum_fill_fraction
        ),
        isolated_debris_brown_minimum_mean_red_blue_difference=(
            config.isolated_debris_brown_minimum_mean_red_blue_difference
        ),
        isolated_debris_brown_maximum_mean_intensity=(
            config.isolated_debris_brown_maximum_mean_intensity
        ),
        isolated_debris_brown_minimum_instance_fraction=(
            config.isolated_debris_brown_minimum_instance_fraction
        ),
        isolated_debris_brown_maximum_source_component_area=math.ceil(
            config.isolated_debris_brown_maximum_source_component_area_um2
            / (slide.mpp_xy[0] * slide.mpp_xy[1])
        ),
        isolated_debris_brown_maximum_source_component_area_um2=(
            config.isolated_debris_brown_maximum_source_component_area_um2
        ),
        isolated_debris_brown_source_maximum_mean_intensity=(
            config.isolated_debris_brown_source_maximum_mean_intensity
        ),
        isolated_debris_brown_source_dilation_bins=(
            config.isolated_debris_brown_source_dilation_bins
        ),
        minimum_area=config.min_size,
    )
    if legacy_qc is not None:
        for key in (
            "post_constraint_small_instances_removed",
            "post_constraint_small_pixels_removed",
            "flat_background_instances_removed",
            "flat_background_pixels_removed",
            "nuclear_unsupported_instances_removed",
            "nuclear_unsupported_pixels_removed",
            "source_context_unsupported_instances_removed",
            "source_context_unsupported_pixels_removed",
            "adaptive_nuclear_core_unsupported_instances_removed",
            "adaptive_nuclear_core_unsupported_pixels_removed",
        ):
            evidence_qc[key] = int(evidence_qc[key]) + int(legacy_qc.get(key, 0))
    canvas.flush()
    _progress(
        progress,
        "  removed "
        f"{evidence_qc['post_constraint_small_instances_removed']} "
        "post-constraint small instances",
    )
    _progress(
        progress,
        "  removed "
        f"{evidence_qc['flat_background_instances_removed']} flat-background "
        "instances",
    )
    if config.global_nuclear_support:
        _progress(
            progress,
            "  removed "
            f"{evidence_qc['nuclear_unsupported_instances_removed']} "
            "nuclear-unsupported instances",
        )
    if config.source_tissue_context:
        _progress(
            progress,
            "  removed "
            f"{evidence_qc['source_context_unsupported_instances_removed']} "
            "source-context-unsupported instances",
        )
    if config.adaptive_nuclear_core:
        _progress(
            progress,
            "  removed "
            f"{evidence_qc['adaptive_nuclear_core_unsupported_instances_removed']} "
            "large adaptive-core-unsupported instances",
        )
    if config.isolated_debris_gate:
        _progress(
            progress,
            "  removed "
            f"{evidence_qc['isolated_debris_unsupported_instances_removed']} "
            "isolated/luminal debris instances",
        )
        _progress(
            progress,
            "  removed "
            f"{evidence_qc['micro_island_debris_instances_removed']} "
            "micro-island debris instances",
        )
        _progress(
            progress,
            "  preserved "
            f"{evidence_qc['organized_tissue_escape_instances']} "
            "organized-tissue instances",
        )
        _progress(
            progress,
            "  removed "
            f"{evidence_qc['compact_unsupported_mosaic_instances_removed']} "
            "compact unsupported mosaic instances",
        )
        _progress(
            progress,
            "  removed "
            f"{evidence_qc['foam_mosaic_instances_removed']} "
            "bright luminal foam mosaic instances",
        )
        _progress(
            progress,
            "  removed "
            f"{evidence_qc['oversized_chromatic_artifact_instances_removed']} "
            "oversized chromatic artifact instances",
        )
        _progress(
            progress,
            "  removed "
            f"{evidence_qc['detached_fragment_instances_removed']} "
            "detached-fragment artifact instances",
        )
        _progress(
            progress,
            "  removed "
            f"{evidence_qc['satellite_debris_instances_removed']} "
            "sparse anuclear satellite-debris instances",
        )
    label_relative = Path("labels") / f"{slide.section}.cells.tiff"
    label_path = config.output_dir / label_relative
    _progress(progress, "  writing pyramidal labels")
    _write_pyramidal_labels(canvas.label_path, label_path, width=width, height=height)
    _progress(progress, "  auditing final label geometry")
    qc = _disk_label_qc(
        canvas.labels,
        canvas.next_label,
        tissue_mask=tissue_mask,
        content_shape=(height, width),
        min_size=config.min_size,
    )
    if qc["outside_tissue_pixels_final"] != 0:
        raise RuntimeError("final cell labels extend outside the accepted tissue mask")
    qc.update(
        {
            "checkpoint_schema_version": 1,
            "algorithm_version": algorithm_version,
            "method_profile": method_profile,
            "profile_fingerprint": profile_fingerprints[0],
            "section_fingerprint": _section_fingerprint(
                preflight, slide, profile_fingerprints[0]
            ),
            "labels_sha256": _file_sha256(label_path),
            "tiles_total": len(tiles),
            "tiles_inferred": executed,
            "tiles_reused": reused,
            "tiles_reused_from_prior_run": reused_from_prior_run,
            "dense_small_cell_recovery_tiles": len(dense_recovery_tiles),
            "dense_small_cell_recovery_manifest_fingerprint": (
                dense_recovery.fingerprint if dense_recovery is not None else None
            ),
            "tiles_skipped_outside_tissue": skipped,
            "tiles_with_outside_predictions": tiles_with_outside_predictions,
            "outside_prediction_pixels_removed": outside_prediction_pixels_removed,
            "filter_upgrade_from_algorithm_version": (
                legacy_algorithm_version if legacy_qc is not None else None
            ),
            "filter_upgrade_source_labels_sha256": (
                legacy_qc.get("labels_sha256") if legacy_qc is not None else None
            ),
            **evidence_qc,
        }
    )
    qc_relative = Path("qc") / f"{slide.section}.json"
    write_json_atomic(config.output_dir / qc_relative, qc)
    del canvas.labels
    del canvas.weights
    if not config.keep_tile_masks:
        shutil.rmtree(cache_root, ignore_errors=True)
    shutil.rmtree(work_root, ignore_errors=True)
    return {
        "section": slide.section,
        "slide": slide.slide_name,
        "source_identity": slide.source_identity,
        "is_reference": slide.is_reference,
        "mpp_xy": list(slide.mpp_xy),
        "native_shape": list(slide.native_shape),
        "content_bbox_xywh": list(slide.content_bbox_xywh),
        "transform_sha256": slide.transform_sha256,
        "labels": label_relative.as_posix(),
        "qc": qc_relative.as_posix(),
        "cell_count": qc["cell_count"],
    }


def _load_completed_slide(
    config: CellSegmentationConfig,
    preflight: CellPreflight,
    slide: CellPreflightSlide,
    profile_fingerprint: str,
) -> dict[str, object] | None:
    """Reuse only a complete section bound to current inputs and output bytes."""

    label_relative = Path("labels") / f"{slide.section}.cells.tiff"
    qc_relative = Path("qc") / f"{slide.section}.json"
    label_path = config.output_dir / label_relative
    qc_path = config.output_dir / qc_relative
    try:
        qc = json.loads(qc_path.read_text())
        if not isinstance(qc, dict) or qc.get("checkpoint_schema_version") != 1:
            return None
        if qc.get("section_fingerprint") != _section_fingerprint(
            preflight, slide, profile_fingerprint
        ):
            return None
        if qc.get("labels_sha256") != _file_sha256(label_path):
            return None
        cell_count = qc.get("cell_count")
        if (
            not isinstance(cell_count, int)
            or isinstance(cell_count, bool)
            or cell_count <= 0
            or qc.get("outside_tissue_pixels_final") != 0
        ):
            return None
        tile_counts = tuple(
            qc.get(key)
            for key in (
                "tiles_total",
                "tiles_inferred",
                "tiles_reused",
                "tiles_skipped_outside_tissue",
            )
        )
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in tile_counts
        ) or tile_counts[0] != sum(tile_counts[1:]):
            return None
        label = _import_pyvips().Image.new_from_file(
            str(label_path), access="sequential"
        )
        _x, _y, width, height = slide.content_bbox_xywh
        if (
            label.width != width
            or label.height != height
            or label.bands != 1
            or label.format != "uint"
        ):
            return None
    except (FileNotFoundError, OSError, TypeError, ValueError):
        return None
    return {
        "section": slide.section,
        "slide": slide.slide_name,
        "source_identity": slide.source_identity,
        "is_reference": slide.is_reference,
        "mpp_xy": list(slide.mpp_xy),
        "native_shape": list(slide.native_shape),
        "content_bbox_xywh": list(slide.content_bbox_xywh),
        "transform_sha256": slide.transform_sha256,
        "labels": label_relative.as_posix(),
        "qc": qc_relative.as_posix(),
        "cell_count": cell_count,
    }


def _section_fingerprint(
    preflight: CellPreflight,
    slide: CellPreflightSlide,
    profile_fingerprint: str,
) -> str:
    return _json_sha256(
        {
            "checkpoint_schema_version": 1,
            "preflight": preflight.fingerprint,
            "section": slide.section,
            "source": slide.source_identity,
            "mask": slide.mask_sha256,
            "profile": profile_fingerprint,
        }
    )


def _tile_has_tissue(
    mask: np.ndarray,
    tile: CellTile,
    *,
    content_shape: tuple[int, int],
    minimum: float,
) -> bool:
    height, width = content_shape
    mask_height, mask_width = mask.shape
    y0 = max(0, int(np.floor(tile.y0 * mask_height / height)))
    y1 = min(mask_height, max(y0 + 1, int(np.ceil(tile.y1 * mask_height / height))))
    x0 = max(0, int(np.floor(tile.x0 * mask_width / width)))
    x1 = min(mask_width, max(x0 + 1, int(np.ceil(tile.x1 * mask_width / width))))
    return float(np.mean(mask[y0:y1, x0:x1])) >= minimum


def _tile_tissue_mask(
    mask: np.ndarray,
    tile: CellTile,
    *,
    content_shape: tuple[int, int],
) -> np.ndarray:
    """Sample a thumbnail mask over one native tile with nearest neighbours."""

    height, width = content_shape
    mask_height, mask_width = mask.shape
    rows = np.floor(
        (np.arange(tile.y0, tile.y1, dtype=np.float64) + 0.5) * mask_height / height
    ).astype(np.int64)
    columns = np.floor(
        (np.arange(tile.x0, tile.x1, dtype=np.float64) + 0.5) * mask_width / width
    ).astype(np.int64)
    np.clip(rows, 0, mask_height - 1, out=rows)
    np.clip(columns, 0, mask_width - 1, out=columns)
    return np.asarray(mask[np.ix_(rows, columns)], dtype=bool)


def _tile_fingerprint(
    preflight: CellPreflight,
    slide: CellPreflightSlide,
    profile_fingerprint: str,
    tile: CellTile,
) -> str:
    return _json_sha256(
        {
            "preflight": preflight.fingerprint,
            "source": slide.source_identity,
            "mask": slide.mask_sha256,
            "profile": profile_fingerprint,
            "tile": [tile.row, tile.column, tile.y0, tile.y1, tile.x0, tile.x1],
        }
    )


def _prepare_tile_cache_reuse(
    config: CellSegmentationConfig,
    preflight: CellPreflight,
    *,
    model: dict[str, object],
    request: dict[str, object],
    algorithm_version: int = _ALGORITHM_VERSION,
    compatible_profile_fingerprints: tuple[str, ...] = (),
) -> dict[str, _TileCacheReuse]:
    """Validate an optional prior run for raw native-tile prediction reuse."""

    if config.tile_cache_reuse_run is None:
        return {}
    reuse_root = config.tile_cache_reuse_run.expanduser().resolve()
    if reuse_root == config.output_dir.expanduser().resolve():
        raise ValueError("tile_cache_reuse_run must differ from output_dir")
    old = load_cell_preflight(reuse_root / "preflight.json")
    old_by_name = {slide.slide_name: slide for slide in old.slides}
    old_profile = _json_sha256(
        {
            "algorithm_version": algorithm_version,
            "preflight_fingerprint": old.fingerprint,
            "model": model,
            "request": request,
        }
    )
    sealed_source_profiles = _sealed_source_tile_profile_fingerprints(
        reuse_root,
        old,
        model=model,
        request=request,
        algorithm_version=algorithm_version,
    )
    # A cross-run postfilter probe has a different output preflight and request,
    # but its source tiles can still be the exact algorithm-v75 baseline.  Admit
    # only the narrowly reconstructed baseline fingerprints already used for
    # same-run cache lookup.  Without these candidates, a scientifically
    # post-inference-only threshold change silently reruns Cellpose even though
    # source identity, geometry, model, and inference controls are unchanged.
    old_profiles = tuple(
        dict.fromkeys(
            (
                old_profile,
                *sealed_source_profiles,
                *_baseline_fold_tile_profile_fingerprints(
                    preflight_fingerprint=old.fingerprint,
                    model=model,
                    request=request,
                    algorithm_version=algorithm_version,
                ),
                *_baseline_foam_tile_profile_fingerprints(
                    preflight_fingerprint=old.fingerprint,
                    model=model,
                    request=request,
                    algorithm_version=algorithm_version,
                ),
                *compatible_profile_fingerprints,
            )
        )
    )
    output: dict[str, _TileCacheReuse] = {}
    for slide in preflight.slides:
        candidate = old_by_name.get(slide.slide_name)
        if candidate is None:
            raise ValueError(
                f"tile cache reuse run is missing slide: {slide.slide_name}"
            )
        invariants = (
            "source_identity",
            "native_shape",
            "content_bbox_xywh",
            "mpp_xy",
        )
        if any(getattr(candidate, name) != getattr(slide, name) for name in invariants):
            raise ValueError(
                "tile cache reuse source geometry or identity changed for "
                f"{slide.slide_name}"
            )
        output[slide.slide_name] = _TileCacheReuse(
            cache_root=reuse_root / ".cell-cache" / candidate.section,
            preflight=old,
            slide=candidate,
            profile_fingerprints=old_profiles,
        )
    return output


def _sealed_source_tile_profile_fingerprints(
    reuse_root: Path,
    preflight: CellPreflight,
    *,
    model: dict[str, object],
    request: dict[str, object],
    algorithm_version: int = _ALGORITHM_VERSION,
) -> tuple[str, ...]:
    """Return the exact profile sealed by a compatible completed source run.

    Slide selection and downstream evidence filters do not alter Cellpose's raw
    native-tile masks.  A subset refinement may therefore reuse the source
    run's exact cache profile, but only after its sealed result, model,
    inference controls, and preflight binding have all been validated.
    """

    result_path = reuse_root / "cell_result.json"
    if not result_path.is_file():
        return ()
    result = validate_cell_result_index(reuse_root)
    source_algorithm_version = result.get("algorithm_version")
    # The v76 cohort refilter changes only post-inference evidence logic.  Its
    # raw CPSAM masks are therefore byte-for-byte reusable from a sealed v75
    # run when the projected inference request, model, source, and geometry
    # all still match.  Keep this bridge deliberately one-way and explicit;
    # arbitrary historical or future algorithm versions are not admitted.
    if source_algorithm_version not in {algorithm_version, 75}:
        return ()
    if result.get("preflight_fingerprint") != preflight.fingerprint:
        raise ValueError("tile cache reuse result is bound to another preflight")
    source_model = result.get("model")
    if source_model != model:
        raise ValueError("tile cache reuse source model provenance changed")
    source_request = result.get("request")
    if not isinstance(source_request, dict):
        raise ValueError("tile cache reuse source request is invalid")
    if _raw_tile_inference_request(source_request) != _raw_tile_inference_request(
        request
    ):
        raise ValueError("tile cache reuse inference controls changed")
    expected = _json_sha256(
        {
            "algorithm_version": source_algorithm_version,
            "preflight_fingerprint": preflight.fingerprint,
            "model": source_model,
            "request": source_request,
        }
    )
    profile = result.get("profile_fingerprint")
    if profile != expected:
        raise ValueError("tile cache reuse source profile is stale")
    return (expected,)


def _raw_tile_inference_request(request: dict[str, object]) -> dict[str, object]:
    """Project a full request onto fields that determine a cached tile mask."""

    keys = (
        "model",
        "method",
        "inference_batch_size",
        "tile_size",
        "overlap",
        "tissue_fraction_threshold",
        "first_cellprob",
        "first_diameter",
        "second_cellprob",
        "second_diameter",
        "flow_threshold",
        "merge_threshold",
        "min_size",
        "multiscale_nuclear_support",
    )
    projected = {key: request.get(key) for key in keys}
    reference = request.get("method_reference")
    if isinstance(reference, dict):
        # The profile label advances when post-inference evidence changes; the
        # source script identity remains the raw CPSAM provenance boundary.
        projected["method_reference"] = {
            "name": reference.get("name"),
            "sha256": reference.get("sha256"),
        }
    else:
        projected["method_reference"] = reference
    return projected


def _version_75_postfilter_request(request: dict[str, object]) -> dict[str, object]:
    """Project later post-inference refinements onto the exact v75 request."""

    baseline = json.loads(json.dumps(request))
    # A recovery manifest replaces a bounded set of already inferred raw masks;
    # it is deliberately part of the derived result fingerprint, but it did not
    # participate in the source CPSAM inference.  Remove it before rebuilding
    # the exact algorithm-v75 tile fingerprint or every otherwise valid source
    # cache entry appears stale and a cache-only recovery cannot run.
    baseline.pop("dense_small_cell_recovery", None)
    reference = baseline.get("method_reference")
    if isinstance(reference, dict):
        reference["profile"] = "combined-containment-cpsam-wsi-v71"
    isolated = baseline.get("isolated_debris_gate")
    if isinstance(isolated, dict):
        isolated["allow_nuclear_escape"] = True
        isolated.pop("allow_organized_escape", None)
        for key in (
            "satellite_gate",
            "self_dense_glass_gate",
            "oversized_brown_gate",
            "oversized_brown_maximum_mean_intensity",
            "clustered_brown_gate",
            "diffuse_degenerated_gate",
            "neutral_precipitate_gate",
            "reconcile_enclosed_cytoplasmic_children",
            "protect_organized_from_necrotic",
            "organized_necrotic_protection_maximum_instance_area_um2",
            "organized_necrotic_protection_minimum_local_prediction_fraction",
            "organized_necrotic_protection_local_prediction_source",
            "organized_necrotic_sparse_glass_shape_gate",
        ):
            isolated.pop(key, None)
    return baseline


def _load_reusable_tile(
    reuse: _TileCacheReuse,
    tile: CellTile,
) -> np.ndarray | None:
    """Load a prior raw tile only when its old exact cache seal still matches."""

    fingerprints = tuple(
        _tile_fingerprint(reuse.preflight, reuse.slide, profile, tile)
        for profile in reuse.profile_fingerprints
    )
    return _load_cached_tile(reuse.cache_root / f"{tile.key}.npz", fingerprints)


def _replace_reusable_tile_with_dense_recovery(
    recovery: DenseSmallCellRecoveryManifest | None,
    tile: CellTile,
    source_mask: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Replace only an exact targeted source tile with its validated recovery."""

    if recovery is None:
        return source_mask, False
    replacement = recovery.tiles.get(f"{tile.key}.npz")
    if replacement is None:
        return source_mask, False
    mask = replacement.load_mask()
    expected_shape = (tile.y1 - tile.y0, tile.x1 - tile.x0)
    if mask.shape != expected_shape:
        raise ValueError(f"dense recovery tile geometry changed: {tile.key}")
    return mask, True


def _load_cached_tile(path: Path, fingerprints: tuple[str, ...]) -> np.ndarray | None:
    try:
        with np.load(path, allow_pickle=False) as payload:
            if str(payload["fingerprint"].item()) not in fingerprints:
                return None
            mask = np.asarray(payload["mask"], dtype=np.int32)
    except (FileNotFoundError, KeyError, OSError, ValueError):
        return None
    return mask if mask.ndim == 2 else None


def _filter_flat_background_instances(
    labels: np.ndarray,
    maximum: int,
    *,
    read_rgb: Callable[[int, int, int, int], np.ndarray],
    block_size: int = _EVIDENCE_BLOCK_SIZE,
    require_nuclear_support: bool = False,
    nuclear_minimum_optical_density: float = 0.12,
    nuclear_minimum_pixels: int = 3,
    nuclear_minimum_fraction: float = 0.005,
    require_source_tissue_context: bool = False,
    source_tissue_context_bin_size: int = 256,
    source_tissue_context_minimum_fraction: float = 0.03,
    source_tissue_context_minimum_instance_fraction: float = 0.01,
    require_adaptive_nuclear_core: bool = False,
    adaptive_nuclear_core_percentile: float = 70.0,
    adaptive_nuclear_core_minimum_area: int = 1000,
    adaptive_nuclear_core_minimum_area_um2: float | None = None,
    adaptive_nuclear_core_minimum_pixels: int = 8,
    adaptive_nuclear_core_minimum_fraction: float = 0.10,
    adaptive_nuclear_core_minimum_blue_ratio: float = 1.08,
    require_isolated_debris_gate: bool = False,
    isolated_debris_context_bin_size: int = 64,
    isolated_debris_context_minimum_stain_fraction: float = 0.80,
    isolated_debris_context_minimum_instance_fraction: float = 0.75,
    isolated_debris_component_minimum_nuclear_pixels: int = 8,
    isolated_debris_allow_nuclear_escape: bool = True,
    isolated_debris_allow_organized_escape: bool = True,
    isolated_debris_nuclear_minimum_pixels: int = 2,
    isolated_debris_nuclear_minimum_fraction: float = 0.005,
    isolated_debris_nuclear_minimum_blue_ratio: float = 1.08,
    isolated_debris_global_nuclear_escape_maximum_fraction: float = 0.75,
    isolated_debris_neutral_dark_maximum_value: int = 90,
    isolated_debris_neutral_dark_maximum_chroma: int = 30,
    isolated_debris_neutral_dark_maximum_fraction: float = 0.50,
    isolated_debris_very_dark_maximum_value: int = 60,
    isolated_debris_very_dark_maximum_chroma: int = 20,
    isolated_debris_very_dark_maximum_fraction: float = 0.45,
    isolated_debris_micro_bin_size: int = 16,
    isolated_debris_micro_minimum_stain_fraction: float = 0.80,
    isolated_debris_micro_maximum_component_bins: int = 512,
    isolated_debris_micro_minimum_nuclear_fraction: float = 0.01,
    isolated_debris_micro_minimum_instance_fraction: float = 0.50,
    isolated_debris_organized_bin_size: int = 4,
    isolated_debris_organized_window_size: int = 1024,
    isolated_debris_organized_window_overlap: int = 128,
    isolated_debris_organized_minimum_bin_occupancy_fraction: float = 0.10,
    isolated_debris_organized_minimum_component_pixels: int = 10_000,
    isolated_debris_organized_compact_minimum_component_pixels: int = 20_000,
    isolated_debris_organized_minimum_aspect_ratio: float = 4.0,
    isolated_debris_organized_minimum_mean_instance_pixels: float = 350.0,
    isolated_debris_organized_minimum_instance_fraction: float = 0.50,
    isolated_debris_organized_strong_red_blue_difference: float = 40.0,
    isolated_debris_organized_minimum_strong_chromatic_fraction: float = 0.25,
    isolated_debris_compact_unsupported_minimum_area: int = 50_000,
    isolated_debris_compact_unsupported_minimum_area_um2: float | None = None,
    isolated_debris_compact_unsupported_maximum_aspect_ratio: float = 1.75,
    isolated_debris_compact_unsupported_maximum_mean_instance_pixels: float = 250.0,
    isolated_debris_compact_unsupported_maximum_mean_red_blue_difference: float = 21.0,
    isolated_debris_compact_unsupported_strong_red_blue_difference: float = 40.0,
    isolated_debris_compact_unsupported_maximum_strong_chromatic_fraction: float = 0.10,
    isolated_debris_compact_unsupported_elongation_ratio: float = 2.0,
    isolated_debris_compact_unsupported_maximum_elongated_instance_fraction: float = (
        0.50
    ),
    isolated_debris_compact_unsupported_maximum_context_nuclear_fraction: float = 0.20,
    isolated_debris_compact_unsupported_minimum_component_fill_fraction: float = 0.20,
    isolated_debris_compact_unsupported_minimum_instance_fraction: float = 0.50,
    isolated_debris_foam_minimum_mean_intensity: float = 160.0,
    isolated_debris_foam_maximum_mean_instance_pixels: float = 250.0,
    isolated_debris_foam_maximum_strong_chromatic_fraction: float = 0.10,
    isolated_debris_foam_maximum_context_nuclear_fraction: float = 0.50,
    isolated_debris_foam_maximum_elongated_instance_fraction: float = 0.50,
    isolated_debris_foam_core_minimum_pixels: int = 8,
    isolated_debris_foam_core_minimum_bright_fraction: float = 0.80,
    isolated_debris_foam_core_minimum_intensity: float = 180.0,
    isolated_debris_foam_core_maximum_instance_area: int = 520,
    isolated_debris_foam_core_maximum_instance_area_um2: float | None = None,
    isolated_debris_foam_core_minimum_component_area: int = 30_000,
    isolated_debris_foam_core_minimum_component_area_um2: float | None = None,
    isolated_debris_foam_core_maximum_aspect_ratio: float = 2.50,
    isolated_debris_oversized_chromatic_minimum_area: int = 4000,
    isolated_debris_oversized_chromatic_minimum_area_um2: float | None = None,
    isolated_debris_oversized_chromatic_minimum_fraction: float = 0.50,
    isolated_debris_fold_minimum_instance_area: int = 1200,
    isolated_debris_fold_minimum_instance_area_um2: float | None = None,
    isolated_debris_fold_minimum_component_area: int = 3000,
    isolated_debris_fold_minimum_component_area_um2: float | None = None,
    isolated_debris_fold_maximum_mean_red_blue_difference: float = 40.0,
    isolated_debris_fold_maximum_mean_intensity: float = 180.0,
    isolated_debris_fold_dense_minimum_instances: int = 3,
    isolated_debris_fold_dense_minimum_aspect_ratio: float = 3.0,
    isolated_debris_fold_dense_connectivity_dilation_bins: int = 8,
    isolated_debris_fold_dense_maximum_mean_red_blue_difference: float = 60.0,
    isolated_debris_fold_dense_maximum_mean_intensity: float = 115.0,
    isolated_debris_detached_minimum_instance_area: int = 800,
    isolated_debris_detached_minimum_instance_area_um2: float | None = None,
    isolated_debris_detached_minimum_component_area: int = 4000,
    isolated_debris_detached_minimum_component_area_um2: float | None = None,
    isolated_debris_detached_minimum_instances: int = 3,
    isolated_debris_detached_minimum_aspect_ratio: float = 3.0,
    isolated_debris_detached_maximum_mean_red_blue_difference: float = 35.0,
    isolated_debris_detached_maximum_mean_intensity: float = 220.0,
    isolated_debris_detached_context_window_size: int = 512,
    isolated_debris_detached_maximum_prediction_fraction: float = 0.20,
    isolated_debris_detached_compact_minimum_instance_area: int = 1200,
    isolated_debris_detached_compact_minimum_instance_area_um2: float | None = None,
    isolated_debris_detached_compact_minimum_component_area: int = 3000,
    isolated_debris_detached_compact_minimum_component_area_um2: float | None = None,
    isolated_debris_detached_compact_minimum_instances: int = 2,
    isolated_debris_detached_compact_maximum_mean_red_blue_difference: float = 0.0,
    isolated_debris_detached_compact_maximum_mean_intensity: float = 180.0,
    isolated_debris_satellite_maximum_instance_area: int = 1500,
    isolated_debris_satellite_maximum_instance_area_um2: float | None = None,
    isolated_debris_satellite_minimum_component_area: int = 2500,
    isolated_debris_satellite_minimum_component_area_um2: float | None = None,
    isolated_debris_satellite_gate: bool = True,
    isolated_debris_satellite_minimum_mask_distance_um: float | None = None,
    isolated_debris_satellite_tissue_mask: np.ndarray | None = None,
    isolated_debris_satellite_mpp_xy: tuple[float, float] | None = None,
    isolated_debris_glass_minimum_area: int = 100,
    isolated_debris_glass_minimum_area_um2: float | None = None,
    isolated_debris_glass_context_window_size: int = 256,
    isolated_debris_glass_maximum_prediction_fraction: float = 0.40,
    isolated_debris_glass_maximum_context_stain_fraction: float = 0.20,
    isolated_debris_glass_low_stain_maximum_mean_red_blue_difference: float = 35.0,
    isolated_debris_glass_low_stain_minimum_mean_intensity: float = 120.0,
    isolated_debris_glass_minimum_context_fraction: float = 0.50,
    isolated_debris_glass_maximum_mean_red_blue_difference: float = 25.0,
    isolated_debris_glass_minimum_mean_intensity: float = 190.0,
    isolated_debris_self_dense_glass_gate: bool = False,
    isolated_debris_oversized_brown_gate: bool = False,
    isolated_debris_oversized_brown_maximum_mean_intensity: float = 180.0,
    isolated_debris_clustered_brown_gate: bool = False,
    isolated_debris_clustered_brown_minimum_instance_area: int = 900,
    isolated_debris_clustered_brown_minimum_instance_area_um2: float | None = None,
    isolated_debris_clustered_brown_maximum_instance_area: int = 4_800,
    isolated_debris_clustered_brown_maximum_instance_area_um2: float | None = None,
    isolated_debris_clustered_brown_minimum_component_area: int = 6_000,
    isolated_debris_clustered_brown_minimum_component_area_um2: float | None = None,
    isolated_debris_diffuse_degenerated_gate: bool = False,
    isolated_debris_neutral_precipitate_gate: bool = False,
    reconcile_enclosed_cytoplasmic_children: bool = False,
    isolated_debris_necrotic_minimum_component_area: int = 20_000,
    isolated_debris_necrotic_minimum_component_area_um2: float | None = None,
    isolated_debris_necrotic_maximum_aspect_ratio: float = 2.5,
    isolated_debris_necrotic_minimum_fill_fraction: float = 0.05,
    isolated_debris_necrotic_maximum_fill_fraction: float = 0.45,
    isolated_debris_necrotic_minimum_mean_red_blue_difference: float = -15.0,
    isolated_debris_necrotic_maximum_mean_red_blue_difference: float = 15.0,
    isolated_debris_necrotic_minimum_mean_intensity: float = 160.0,
    isolated_debris_necrotic_minimum_instance_fraction: float = 0.50,
    isolated_debris_protect_organized_from_necrotic: bool = False,
    isolated_debris_organized_necrotic_protection_maximum_instance_area: (
        int | None
    ) = None,
    isolated_debris_organized_necrotic_protection_maximum_instance_area_um2: (
        float | None
    ) = None,
    organized_necrotic_protection_minimum_local_prediction_fraction: (
        float | None
    ) = None,
    organized_necrotic_protection_use_eligible_context: bool = False,
    organized_necrotic_protection_require_independent_nuclear_context: bool = False,
    organized_necrotic_sparse_glass_shape_gate: bool = False,
    isolated_debris_brown_minimum_component_area: int = 1_000,
    isolated_debris_brown_minimum_component_area_um2: float | None = None,
    isolated_debris_brown_maximum_aspect_ratio: float = 3.0,
    isolated_debris_brown_minimum_fill_fraction: float = 0.12,
    isolated_debris_brown_maximum_fill_fraction: float = 0.45,
    isolated_debris_brown_minimum_mean_red_blue_difference: float = 50.0,
    isolated_debris_brown_maximum_mean_intensity: float = 120.0,
    isolated_debris_brown_minimum_instance_fraction: float = 0.50,
    isolated_debris_brown_maximum_source_component_area: int = 250_000,
    isolated_debris_brown_maximum_source_component_area_um2: float | None = None,
    isolated_debris_brown_source_maximum_mean_intensity: float = 218.0,
    isolated_debris_brown_source_dilation_bins: int = 0,
    minimum_area: int = 1,
) -> dict[str, object]:
    """Remove stitched instances failing size, stain, or nuclear evidence.

    The final size gate is deliberately applied after accepted-mask clipping
    and overlap stitching because both operations can create fragments smaller
    than Cellpose's tile-local ``min_size`` threshold.
    """

    if labels.ndim != 2:
        raise ValueError("labels must be a 2D array")
    if not isinstance(minimum_area, int) or isinstance(minimum_area, bool):
        raise TypeError("minimum_area must be an integer")
    if minimum_area <= 0:
        raise ValueError("minimum_area must be positive")
    if source_tissue_context_bin_size <= 0:
        raise ValueError("source tissue-context bin size must be positive")
    if not 0 < source_tissue_context_minimum_fraction <= 1:
        raise ValueError("source tissue-context fraction must be between zero and one")
    if not 0 < source_tissue_context_minimum_instance_fraction <= 1:
        raise ValueError(
            "source tissue-context instance fraction must be between zero and one"
        )
    if not 0 < adaptive_nuclear_core_percentile < 100:
        raise ValueError("adaptive nuclear-core percentile must be between 0 and 100")
    if adaptive_nuclear_core_minimum_area <= 0:
        raise ValueError("adaptive nuclear-core minimum area must be positive")
    if adaptive_nuclear_core_minimum_pixels <= 0:
        raise ValueError("adaptive nuclear-core minimum pixels must be positive")
    if not 0 < adaptive_nuclear_core_minimum_fraction <= 1:
        raise ValueError(
            "adaptive nuclear-core minimum fraction must be between zero and one"
        )
    if adaptive_nuclear_core_minimum_blue_ratio <= 1 or not np.isfinite(
        adaptive_nuclear_core_minimum_blue_ratio
    ):
        raise ValueError(
            "adaptive nuclear-core minimum blue ratio must be finite and greater "
            "than one"
        )
    if isolated_debris_context_bin_size <= 0:
        raise ValueError("isolated debris context bin size must be positive")
    if not 0 < isolated_debris_context_minimum_stain_fraction <= 1:
        raise ValueError(
            "isolated debris context stain fraction must be between zero and one"
        )
    if not 0 < isolated_debris_context_minimum_instance_fraction <= 1:
        raise ValueError(
            "isolated debris context instance fraction must be between zero and one"
        )
    if isolated_debris_component_minimum_nuclear_pixels <= 0:
        raise ValueError(
            "isolated debris component minimum nuclear pixels must be positive"
        )
    if not isinstance(isolated_debris_allow_nuclear_escape, bool):
        raise TypeError("isolated debris nuclear escape must be a boolean")
    if not isinstance(isolated_debris_allow_organized_escape, bool):
        raise TypeError("isolated debris organized escape must be a boolean")
    if isolated_debris_nuclear_minimum_pixels <= 0:
        raise ValueError("isolated debris nuclear minimum pixels must be positive")
    if not 0 < isolated_debris_nuclear_minimum_fraction <= 1:
        raise ValueError(
            "isolated debris nuclear fraction must be between zero and one"
        )
    if isolated_debris_nuclear_minimum_blue_ratio <= 1 or not np.isfinite(
        isolated_debris_nuclear_minimum_blue_ratio
    ):
        raise ValueError(
            "isolated debris nuclear blue ratio must be finite and greater than one"
        )
    if not 0 < isolated_debris_global_nuclear_escape_maximum_fraction < 1:
        raise ValueError(
            "isolated debris global-nuclear escape maximum fraction must be "
            "between zero and one"
        )
    if not 0 < isolated_debris_neutral_dark_maximum_value <= 255:
        raise ValueError(
            "isolated debris neutral-dark maximum value must be between 1 and 255"
        )
    if not 0 < isolated_debris_neutral_dark_maximum_chroma <= 255:
        raise ValueError(
            "isolated debris neutral-dark maximum chroma must be between 1 and 255"
        )
    if not 0 < isolated_debris_neutral_dark_maximum_fraction <= 1:
        raise ValueError(
            "isolated debris neutral-dark fraction must be between zero and one"
        )
    if not 0 < isolated_debris_very_dark_maximum_value <= 255:
        raise ValueError(
            "isolated debris very-dark maximum value must be between 1 and 255"
        )
    if not 0 < isolated_debris_very_dark_maximum_chroma <= 255:
        raise ValueError(
            "isolated debris very-dark maximum chroma must be between 1 and 255"
        )
    if not 0 < isolated_debris_very_dark_maximum_fraction <= 1:
        raise ValueError(
            "isolated debris very-dark fraction must be between zero and one"
        )
    if isolated_debris_micro_bin_size <= 0:
        raise ValueError("isolated debris micro bin size must be positive")
    if not 0 < isolated_debris_micro_minimum_stain_fraction <= 1:
        raise ValueError(
            "isolated debris micro stain fraction must be between zero and one"
        )
    if isolated_debris_micro_maximum_component_bins <= 0:
        raise ValueError(
            "isolated debris micro maximum component bins must be positive"
        )
    if not 0 < isolated_debris_micro_minimum_nuclear_fraction <= 1:
        raise ValueError(
            "isolated debris micro nuclear fraction must be between zero and one"
        )
    if not 0 < isolated_debris_micro_minimum_instance_fraction <= 1:
        raise ValueError(
            "isolated debris micro instance fraction must be between zero and one"
        )
    if isolated_debris_organized_bin_size <= 0:
        raise ValueError("isolated debris organized bin size must be positive")
    if isolated_debris_organized_window_size <= 0:
        raise ValueError("isolated debris organized window size must be positive")
    if (
        not 0
        <= isolated_debris_organized_window_overlap
        < (isolated_debris_organized_window_size)
    ):
        raise ValueError(
            "isolated debris organized window overlap must be nonnegative and "
            "smaller than its window size"
        )
    if (
        isolated_debris_organized_window_size % isolated_debris_organized_bin_size
        or isolated_debris_organized_window_overlap % isolated_debris_organized_bin_size
    ):
        raise ValueError(
            "isolated debris organized window size and overlap must be multiples "
            "of its bin size"
        )
    if not 0 < isolated_debris_organized_minimum_bin_occupancy_fraction <= 1:
        raise ValueError(
            "isolated debris organized bin occupancy fraction must be between "
            "zero and one"
        )
    if isolated_debris_organized_minimum_component_pixels <= 0:
        raise ValueError(
            "isolated debris organized minimum component pixels must be positive"
        )
    if (
        isolated_debris_organized_compact_minimum_component_pixels
        < isolated_debris_organized_minimum_component_pixels
    ):
        raise ValueError(
            "isolated debris organized compact component pixels must not be "
            "smaller than the minimum component size"
        )
    if isolated_debris_organized_minimum_aspect_ratio < 1 or not np.isfinite(
        isolated_debris_organized_minimum_aspect_ratio
    ):
        raise ValueError(
            "isolated debris organized aspect ratio must be finite and at least one"
        )
    if isolated_debris_organized_minimum_mean_instance_pixels <= 0 or not np.isfinite(
        isolated_debris_organized_minimum_mean_instance_pixels
    ):
        raise ValueError(
            "isolated debris organized mean instance pixels must be finite and positive"
        )
    if not 0 < isolated_debris_organized_minimum_instance_fraction <= 1:
        raise ValueError(
            "isolated debris organized instance fraction must be between zero and one"
        )
    if not np.isfinite(isolated_debris_organized_strong_red_blue_difference):
        raise ValueError("isolated debris organized red-blue difference must be finite")
    if not (0 <= isolated_debris_organized_minimum_strong_chromatic_fraction <= 1):
        raise ValueError(
            "isolated debris organized strong chromatic fraction must be between "
            "zero and one"
        )
    if isolated_debris_compact_unsupported_minimum_area <= 0:
        raise ValueError(
            "isolated debris compact unsupported minimum area must be positive"
        )
    if isolated_debris_compact_unsupported_maximum_aspect_ratio < 1 or not np.isfinite(
        isolated_debris_compact_unsupported_maximum_aspect_ratio
    ):
        raise ValueError(
            "isolated debris compact unsupported maximum aspect ratio must be "
            "finite and at least one"
        )
    if (
        isolated_debris_compact_unsupported_maximum_mean_instance_pixels <= 0
        or not np.isfinite(
            isolated_debris_compact_unsupported_maximum_mean_instance_pixels
        )
    ):
        raise ValueError(
            "isolated debris compact unsupported maximum mean instance pixels "
            "must be finite and positive"
        )
    if not np.isfinite(
        isolated_debris_compact_unsupported_maximum_mean_red_blue_difference
    ):
        raise ValueError(
            "isolated debris compact unsupported red-blue difference must be finite"
        )
    if not np.isfinite(isolated_debris_compact_unsupported_strong_red_blue_difference):
        raise ValueError(
            "isolated debris compact unsupported strong red-blue difference must "
            "be finite"
        )
    if not (
        0 <= isolated_debris_compact_unsupported_maximum_strong_chromatic_fraction <= 1
    ):
        raise ValueError(
            "isolated debris compact unsupported strong chromatic fraction must "
            "be between zero and one"
        )
    if isolated_debris_compact_unsupported_elongation_ratio <= 1 or not np.isfinite(
        isolated_debris_compact_unsupported_elongation_ratio
    ):
        raise ValueError(
            "isolated debris compact unsupported elongation ratio must be finite "
            "and greater than one"
        )
    if not (
        0
        <= isolated_debris_compact_unsupported_maximum_elongated_instance_fraction
        <= 1
    ):
        raise ValueError(
            "isolated debris compact unsupported elongated-instance fraction must "
            "be between zero and one"
        )
    if not (
        0 <= isolated_debris_compact_unsupported_maximum_context_nuclear_fraction <= 1
    ):
        raise ValueError(
            "isolated debris compact unsupported context nuclear-supported "
            "fraction must be between zero and one"
        )
    if not (
        0 < isolated_debris_compact_unsupported_minimum_component_fill_fraction <= 1
    ):
        raise ValueError(
            "isolated debris compact unsupported component fill fraction must be "
            "between zero and one"
        )
    if not (0 < isolated_debris_compact_unsupported_minimum_instance_fraction <= 1):
        raise ValueError(
            "isolated debris compact unsupported instance fraction must be "
            "between zero and one"
        )
    if not (
        0 < isolated_debris_foam_minimum_mean_intensity <= 255
        and np.isfinite(isolated_debris_foam_minimum_mean_intensity)
    ):
        raise ValueError(
            "isolated debris foam minimum mean intensity must be finite and "
            "between zero and 255"
        )
    if not 0 <= isolated_debris_foam_maximum_context_nuclear_fraction <= 1:
        raise ValueError(
            "isolated debris foam context nuclear-supported fraction must be "
            "between zero and one"
        )
    if not (
        isolated_debris_foam_maximum_mean_instance_pixels > 0
        and np.isfinite(isolated_debris_foam_maximum_mean_instance_pixels)
    ):
        raise ValueError("isolated debris foam mean instance size is invalid")
    if not 0 <= isolated_debris_foam_maximum_strong_chromatic_fraction <= 1:
        raise ValueError("isolated debris foam chromatic fraction is invalid")
    if not 0 <= isolated_debris_foam_maximum_elongated_instance_fraction <= 1:
        raise ValueError(
            "isolated debris foam elongated-instance fraction must be between "
            "zero and one"
        )
    if isolated_debris_foam_core_minimum_pixels <= 0:
        raise ValueError("isolated debris foam core minimum pixels is invalid")
    if not 0 < isolated_debris_foam_core_minimum_bright_fraction <= 1:
        raise ValueError("isolated debris foam core bright fraction is invalid")
    if not 0 < isolated_debris_foam_core_minimum_intensity <= 255:
        raise ValueError("isolated debris foam core intensity is invalid")
    if isolated_debris_foam_core_maximum_instance_area <= 0:
        raise ValueError("isolated debris foam core instance area is invalid")
    if isolated_debris_foam_core_minimum_component_area <= 0:
        raise ValueError("isolated debris foam core component area is invalid")
    if not (
        isolated_debris_foam_core_maximum_aspect_ratio >= 1
        and np.isfinite(isolated_debris_foam_core_maximum_aspect_ratio)
    ):
        raise ValueError("isolated debris foam core aspect ratio is invalid")
    if isolated_debris_oversized_chromatic_minimum_area <= 0:
        raise ValueError(
            "isolated debris oversized chromatic minimum area must be positive"
        )
    if not 0 < isolated_debris_oversized_chromatic_minimum_fraction <= 1:
        raise ValueError(
            "isolated debris oversized chromatic fraction must be between zero and one"
        )
    if isolated_debris_fold_minimum_instance_area <= 0:
        raise ValueError("isolated debris fold instance area must be positive")
    if (
        isolated_debris_fold_minimum_component_area
        < isolated_debris_fold_minimum_instance_area
    ):
        raise ValueError(
            "isolated debris fold component area must not be smaller than the "
            "instance area"
        )
    if not np.isfinite(isolated_debris_fold_maximum_mean_red_blue_difference):
        raise ValueError("isolated debris fold red-blue difference must be finite")
    if not (
        0 < isolated_debris_fold_maximum_mean_intensity <= 255
        and np.isfinite(isolated_debris_fold_maximum_mean_intensity)
    ):
        raise ValueError(
            "isolated debris fold mean intensity must be finite and between zero "
            "and 255"
        )
    if isolated_debris_fold_dense_minimum_instances < 2:
        raise ValueError(
            "isolated debris dense-fold instance count must be at least two"
        )
    if not (
        isolated_debris_fold_dense_minimum_aspect_ratio >= 1
        and np.isfinite(isolated_debris_fold_dense_minimum_aspect_ratio)
    ):
        raise ValueError("isolated debris dense-fold aspect ratio is invalid")
    if isolated_debris_fold_dense_connectivity_dilation_bins <= 0:
        raise ValueError("isolated debris dense-fold connectivity is invalid")
    if not np.isfinite(isolated_debris_fold_dense_maximum_mean_red_blue_difference):
        raise ValueError("isolated debris dense-fold red-blue difference is invalid")
    if not (
        0 < isolated_debris_fold_dense_maximum_mean_intensity <= 255
        and np.isfinite(isolated_debris_fold_dense_maximum_mean_intensity)
    ):
        raise ValueError("isolated debris dense-fold mean intensity is invalid")
    if isolated_debris_detached_minimum_instance_area <= 0:
        raise ValueError("isolated debris detached instance area must be positive")
    if (
        isolated_debris_detached_minimum_component_area
        < isolated_debris_detached_minimum_instance_area
    ):
        raise ValueError(
            "isolated debris detached component area must not be smaller than "
            "the instance area"
        )
    if isolated_debris_detached_minimum_instances < 2:
        raise ValueError("isolated debris detached instance count must be at least two")
    if not (
        isolated_debris_detached_minimum_aspect_ratio >= 1
        and np.isfinite(isolated_debris_detached_minimum_aspect_ratio)
    ):
        raise ValueError("isolated debris detached aspect ratio is invalid")
    if not np.isfinite(isolated_debris_detached_maximum_mean_red_blue_difference):
        raise ValueError("isolated debris detached red-blue difference is invalid")
    if not (
        0 < isolated_debris_detached_maximum_mean_intensity <= 255
        and np.isfinite(isolated_debris_detached_maximum_mean_intensity)
    ):
        raise ValueError("isolated debris detached mean intensity is invalid")
    if (
        isolated_debris_detached_context_window_size <= 0
        or isolated_debris_detached_context_window_size
        % isolated_debris_organized_bin_size
    ):
        raise ValueError(
            "isolated debris detached context window must be a positive multiple "
            "of the organized bin size"
        )
    if not 0 < isolated_debris_detached_maximum_prediction_fraction <= 1:
        raise ValueError("isolated debris detached prediction fraction is invalid")
    if isolated_debris_satellite_maximum_instance_area <= 0:
        raise ValueError("isolated debris satellite instance area is invalid")
    if isolated_debris_satellite_minimum_component_area <= 0:
        raise ValueError("isolated debris satellite component area is invalid")
    if not isinstance(isolated_debris_satellite_gate, bool):
        raise TypeError("isolated debris satellite gate flag must be boolean")
    if not isolated_debris_satellite_gate and not require_isolated_debris_gate:
        raise ValueError("satellite-gate policy requires isolated debris evidence")
    if isolated_debris_satellite_minimum_mask_distance_um is not None:
        if (
            not isolated_debris_satellite_gate
            or not require_isolated_debris_gate
            or not np.isfinite(isolated_debris_satellite_minimum_mask_distance_um)
            or isolated_debris_satellite_minimum_mask_distance_um <= 0
        ):
            raise ValueError("satellite interior-protection policy is invalid")
        if (
            isolated_debris_satellite_tissue_mask is None
            or isolated_debris_satellite_tissue_mask.ndim != 2
            or isolated_debris_satellite_mpp_xy is None
            or len(isolated_debris_satellite_mpp_xy) != 2
            or any(
                not np.isfinite(value) or value <= 0
                for value in isolated_debris_satellite_mpp_xy
            )
        ):
            raise ValueError(
                "satellite interior protection requires tissue-mask geometry"
            )
    if isolated_debris_glass_minimum_area <= 0:
        raise ValueError("isolated debris glass minimum area must be positive")
    if (
        isolated_debris_glass_context_window_size <= 0
        or isolated_debris_glass_context_window_size
        % isolated_debris_organized_bin_size
    ):
        raise ValueError(
            "isolated debris glass context window must be a positive multiple "
            "of the organized bin size"
        )
    if not 0 < isolated_debris_glass_maximum_prediction_fraction <= 1:
        raise ValueError(
            "isolated debris glass prediction fraction must be between zero and one"
        )
    if not isinstance(isolated_debris_self_dense_glass_gate, bool):
        raise TypeError("self-dense glass gate flag must be boolean")
    if not isinstance(isolated_debris_oversized_brown_gate, bool):
        raise TypeError("oversized-brown gate flag must be boolean")
    if isolated_debris_oversized_brown_gate and not require_isolated_debris_gate:
        raise ValueError("oversized-brown gate requires isolated debris evidence")
    if not isinstance(isolated_debris_clustered_brown_gate, bool):
        raise TypeError("clustered-brown gate flag must be boolean")
    if isolated_debris_clustered_brown_gate and not require_isolated_debris_gate:
        raise ValueError("clustered-brown gate requires isolated debris evidence")
    if (
        isolated_debris_clustered_brown_minimum_instance_area <= 0
        or isolated_debris_clustered_brown_maximum_instance_area
        < isolated_debris_clustered_brown_minimum_instance_area
        or isolated_debris_clustered_brown_minimum_component_area
        < isolated_debris_clustered_brown_minimum_instance_area
    ):
        raise ValueError("clustered-brown areas must be positive and ordered")
    if not (
        0 < isolated_debris_oversized_brown_maximum_mean_intensity <= 255
        and np.isfinite(isolated_debris_oversized_brown_maximum_mean_intensity)
    ):
        raise ValueError("oversized-brown mean intensity is invalid")
    if not isinstance(isolated_debris_diffuse_degenerated_gate, bool):
        raise TypeError("diffuse-degenerated debris gate flag must be boolean")
    if not isinstance(isolated_debris_neutral_precipitate_gate, bool):
        raise TypeError("neutral-precipitate debris gate flag must be boolean")
    if isolated_debris_neutral_precipitate_gate and not require_isolated_debris_gate:
        raise ValueError("neutral-precipitate gate requires isolated debris evidence")
    if not isinstance(reconcile_enclosed_cytoplasmic_children, bool):
        raise TypeError("enclosed cytoplasmic reconciliation flag must be boolean")
    if reconcile_enclosed_cytoplasmic_children and not require_isolated_debris_gate:
        raise ValueError(
            "enclosed cytoplasmic reconciliation requires isolated debris evidence"
        )
    if not 0 < isolated_debris_glass_maximum_context_stain_fraction <= 1:
        raise ValueError(
            "isolated debris glass context stain fraction must be between zero and one"
        )
    if not np.isfinite(
        isolated_debris_glass_low_stain_maximum_mean_red_blue_difference
    ):
        raise ValueError("isolated debris low-stain red-blue difference is invalid")
    if not 0 < isolated_debris_glass_low_stain_minimum_mean_intensity <= 255:
        raise ValueError("isolated debris low-stain intensity is invalid")
    if not 0 < isolated_debris_glass_minimum_context_fraction <= 1:
        raise ValueError(
            "isolated debris glass context fraction must be between zero and one"
        )
    if not np.isfinite(isolated_debris_glass_maximum_mean_red_blue_difference):
        raise ValueError("isolated debris glass red-blue difference must be finite")
    if not (
        0 < isolated_debris_glass_minimum_mean_intensity <= 255
        and np.isfinite(isolated_debris_glass_minimum_mean_intensity)
    ):
        raise ValueError(
            "isolated debris glass mean intensity must be finite and between zero "
            "and 255"
        )
    if isolated_debris_necrotic_minimum_component_area <= 0:
        raise ValueError("isolated debris necrotic component area must be positive")
    if isolated_debris_necrotic_maximum_aspect_ratio < 1 or not np.isfinite(
        isolated_debris_necrotic_maximum_aspect_ratio
    ):
        raise ValueError(
            "isolated debris necrotic aspect ratio must be finite and at least one"
        )
    if not (
        0
        < isolated_debris_necrotic_minimum_fill_fraction
        < isolated_debris_necrotic_maximum_fill_fraction
        <= 1
    ):
        raise ValueError("isolated debris necrotic fill fractions are invalid")
    if not (
        np.isfinite(isolated_debris_necrotic_minimum_mean_red_blue_difference)
        and np.isfinite(isolated_debris_necrotic_maximum_mean_red_blue_difference)
        and isolated_debris_necrotic_minimum_mean_red_blue_difference
        < isolated_debris_necrotic_maximum_mean_red_blue_difference
    ):
        raise ValueError("isolated debris necrotic red-blue range is invalid")
    if not (
        0 < isolated_debris_necrotic_minimum_mean_intensity <= 255
        and np.isfinite(isolated_debris_necrotic_minimum_mean_intensity)
    ):
        raise ValueError("isolated debris necrotic mean intensity is invalid")
    if not 0 < isolated_debris_necrotic_minimum_instance_fraction <= 1:
        raise ValueError("isolated debris necrotic instance fraction is invalid")
    if not isinstance(isolated_debris_protect_organized_from_necrotic, bool):
        raise TypeError("organized-necrotic protection flag must be boolean")
    if (
        isolated_debris_protect_organized_from_necrotic
        and not require_isolated_debris_gate
    ):
        raise ValueError(
            "organized-necrotic protection requires isolated debris evidence"
        )
    if (
        isolated_debris_organized_necrotic_protection_maximum_instance_area is not None
        and isolated_debris_organized_necrotic_protection_maximum_instance_area <= 0
    ):
        raise ValueError(
            "organized-necrotic protection maximum instance area must be positive"
        )
    if (
        isolated_debris_organized_necrotic_protection_maximum_instance_area_um2
        is not None
        and (
            not np.isfinite(
                isolated_debris_organized_necrotic_protection_maximum_instance_area_um2
            )
            or isolated_debris_organized_necrotic_protection_maximum_instance_area_um2
            <= 0
        )
    ):
        raise ValueError(
            "organized-necrotic protection maximum physical instance area must "
            "be positive"
        )
    if (
        organized_necrotic_protection_minimum_local_prediction_fraction is not None
        and not (
            0 < organized_necrotic_protection_minimum_local_prediction_fraction <= 1
        )
    ):
        raise ValueError(
            "organized-necrotic protection minimum local prediction fraction "
            "must be between zero and one"
        )
    if not isinstance(organized_necrotic_protection_use_eligible_context, bool):
        raise TypeError("organized-necrotic eligible-context flag must be boolean")
    if not isinstance(
        organized_necrotic_protection_require_independent_nuclear_context, bool
    ):
        raise TypeError(
            "organized-necrotic independent-nuclear-context flag must be boolean"
        )
    if not isinstance(organized_necrotic_sparse_glass_shape_gate, bool):
        raise TypeError("organized-necrotic sparse-glass shape flag must be boolean")
    if (
        organized_necrotic_protection_use_eligible_context
        and organized_necrotic_protection_minimum_local_prediction_fraction is None
    ):
        raise ValueError(
            "organized-necrotic eligible context requires a local prediction fraction"
        )
    if (
        organized_necrotic_protection_require_independent_nuclear_context
        and not organized_necrotic_protection_use_eligible_context
    ):
        raise ValueError(
            "organized-necrotic independent nuclear context requires eligible context"
        )
    if (
        organized_necrotic_sparse_glass_shape_gate
        and not isolated_debris_protect_organized_from_necrotic
    ):
        raise ValueError(
            "organized-necrotic sparse-glass shape gate requires protection"
        )
    if isolated_debris_brown_minimum_component_area <= 0:
        raise ValueError("isolated debris brown component area must be positive")
    if isolated_debris_brown_maximum_aspect_ratio < 1 or not np.isfinite(
        isolated_debris_brown_maximum_aspect_ratio
    ):
        raise ValueError(
            "isolated debris brown aspect ratio must be finite and at least one"
        )
    if not (
        0
        < isolated_debris_brown_minimum_fill_fraction
        < isolated_debris_brown_maximum_fill_fraction
        <= 1
    ):
        raise ValueError("isolated debris brown fill fractions are invalid")
    if not np.isfinite(isolated_debris_brown_minimum_mean_red_blue_difference):
        raise ValueError("isolated debris brown red-blue threshold is invalid")
    if not (
        0 < isolated_debris_brown_maximum_mean_intensity <= 255
        and np.isfinite(isolated_debris_brown_maximum_mean_intensity)
    ):
        raise ValueError("isolated debris brown mean intensity is invalid")
    if not 0 < isolated_debris_brown_minimum_instance_fraction <= 1:
        raise ValueError("isolated debris brown instance fraction is invalid")
    if isolated_debris_brown_maximum_source_component_area <= 0:
        raise ValueError("isolated debris brown source component area is invalid")
    if not (
        0 < isolated_debris_brown_source_maximum_mean_intensity <= 255
        and np.isfinite(isolated_debris_brown_source_maximum_mean_intensity)
    ):
        raise ValueError("isolated debris brown source intensity is invalid")
    if isolated_debris_brown_source_dilation_bins < 0:
        raise ValueError("isolated debris brown source dilation is invalid")
    height, width = labels.shape
    areas = np.zeros(maximum + 1, dtype=np.uint64)
    supported = np.zeros(maximum + 1, dtype=np.uint64)
    nuclear_supported = np.zeros(maximum + 1, dtype=np.uint64)
    adaptive_core_supported = np.zeros(maximum + 1, dtype=np.uint64)
    isolated_nuclear_supported = np.zeros(maximum + 1, dtype=np.uint64)
    isolated_neutral_dark = np.zeros(maximum + 1, dtype=np.uint64)
    isolated_very_dark = np.zeros(maximum + 1, dtype=np.uint64)
    isolated_micro_debris_supported = np.zeros(maximum + 1, dtype=np.uint64)
    isolated_organized_supported = np.zeros(maximum + 1, dtype=np.uint64)
    isolated_compact_unsupported = np.zeros(maximum + 1, dtype=np.uint64)
    isolated_foam_unsupported = np.zeros(maximum + 1, dtype=np.uint64)
    isolated_necrotic_supported = np.zeros(maximum + 1, dtype=np.uint64)
    isolated_brown_debris_supported = np.zeros(maximum + 1, dtype=np.uint64)
    isolated_red_blue_sum = np.zeros(maximum + 1, dtype=np.float64)
    isolated_intensity_sum = np.zeros(maximum + 1, dtype=np.float64)
    isolated_red_sum = (
        np.zeros(maximum + 1, dtype=np.float64)
        if (
            isolated_debris_oversized_brown_gate or isolated_debris_clustered_brown_gate
        )
        else None
    )
    isolated_green_sum = (
        np.zeros(maximum + 1, dtype=np.float64)
        if (
            isolated_debris_oversized_brown_gate or isolated_debris_clustered_brown_gate
        )
        else None
    )
    isolated_blue_sum = (
        np.zeros(maximum + 1, dtype=np.float64)
        if (
            isolated_debris_oversized_brown_gate or isolated_debris_clustered_brown_gate
        )
        else None
    )
    adaptive_core_thresholds: list[float] = []
    if require_source_tissue_context:
        grid_height = (height + source_tissue_context_bin_size - 1) // (
            source_tissue_context_bin_size
        )
        grid_width = (width + source_tissue_context_bin_size - 1) // (
            source_tissue_context_bin_size
        )
        context_evidence = np.zeros((grid_height, grid_width), dtype=np.uint64)
        context_pixels = np.zeros((grid_height, grid_width), dtype=np.uint64)
    if require_isolated_debris_gate:
        isolated_grid_height = (
            height + isolated_debris_context_bin_size - 1
        ) // isolated_debris_context_bin_size
        isolated_grid_width = (
            width + isolated_debris_context_bin_size - 1
        ) // isolated_debris_context_bin_size
        isolated_context_evidence = np.zeros(
            (isolated_grid_height, isolated_grid_width), dtype=np.uint64
        )
        isolated_context_pixels = np.zeros(
            (isolated_grid_height, isolated_grid_width), dtype=np.uint64
        )
        isolated_context_nuclear = np.zeros(
            (isolated_grid_height, isolated_grid_width), dtype=np.uint64
        )
        isolated_context_nuclear_pixels = np.zeros(
            (isolated_grid_height, isolated_grid_width), dtype=np.uint64
        )
        micro_grid_height = (
            height + isolated_debris_micro_bin_size - 1
        ) // isolated_debris_micro_bin_size
        micro_grid_width = (
            width + isolated_debris_micro_bin_size - 1
        ) // isolated_debris_micro_bin_size
        micro_context_evidence = np.zeros(
            (micro_grid_height, micro_grid_width), dtype=np.uint64
        )
        micro_context_pixels = np.zeros(
            (micro_grid_height, micro_grid_width), dtype=np.uint64
        )
        micro_context_nuclear = np.zeros(
            (micro_grid_height, micro_grid_width), dtype=np.uint64
        )
        micro_context_nuclear_pixels = np.zeros(
            (micro_grid_height, micro_grid_width), dtype=np.uint64
        )
        organized_grid_height = (
            height + isolated_debris_organized_bin_size - 1
        ) // isolated_debris_organized_bin_size
        organized_grid_width = (
            width + isolated_debris_organized_bin_size - 1
        ) // isolated_debris_organized_bin_size
        organized_context_prediction = np.zeros(
            (organized_grid_height, organized_grid_width), dtype=np.uint32
        )
        organized_context_pixels = np.zeros(
            (organized_grid_height, organized_grid_width), dtype=np.uint32
        )
        organized_context_labels = np.zeros(
            (organized_grid_height, organized_grid_width), dtype=np.uint32
        )
        organized_context_red_blue = np.zeros(
            (organized_grid_height, organized_grid_width), dtype=np.int16
        )
        organized_context_intensity = np.zeros(
            (organized_grid_height, organized_grid_width), dtype=np.uint8
        )
    for y0 in range(0, height, block_size):
        y1 = min(height, y0 + block_size)
        for x0 in range(0, width, block_size):
            x1 = min(width, x0 + block_size)
            block = np.asarray(labels[y0:y1, x0:x1])
            rgb = np.asarray(read_rgb(x0, y0, x1 - x0, y1 - y0))
            if rgb.shape != (*block.shape, 3):
                raise ValueError("RGB evidence block does not match label geometry")
            evidence = local_stain_evidence(
                rgb,
                background_percentile=_EVIDENCE_BACKGROUND_PERCENTILE,
                minimum_optical_density=_EVIDENCE_MINIMUM_OPTICAL_DENSITY,
            )
            if require_source_tissue_context:
                _accumulate_context_grid(
                    context_evidence,
                    context_pixels,
                    evidence,
                    x0=x0,
                    y0=y0,
                    bin_size=source_tissue_context_bin_size,
                )
            if require_isolated_debris_gate:
                red_blue = rgb[..., 0].astype(np.int16) - rgb[..., 2].astype(np.int16)
                intensity = np.mean(rgb, axis=2)
                isolated_red_blue_sum += np.bincount(
                    block.ravel(),
                    weights=red_blue.ravel(),
                    minlength=maximum + 1,
                )[: maximum + 1]
                isolated_intensity_sum += np.bincount(
                    block.ravel(),
                    weights=intensity.ravel(),
                    minlength=maximum + 1,
                )[: maximum + 1]
                if (
                    isolated_debris_oversized_brown_gate
                    or isolated_debris_clustered_brown_gate
                ):
                    if (
                        isolated_red_sum is None
                        or isolated_green_sum is None
                        or isolated_blue_sum is None
                    ):
                        raise RuntimeError("oversized-brown channel sums are absent")
                    isolated_red_sum += np.bincount(
                        block.ravel(),
                        weights=rgb[..., 0].ravel(),
                        minlength=maximum + 1,
                    )[: maximum + 1]
                    isolated_green_sum += np.bincount(
                        block.ravel(),
                        weights=rgb[..., 1].ravel(),
                        minlength=maximum + 1,
                    )[: maximum + 1]
                    isolated_blue_sum += np.bincount(
                        block.ravel(),
                        weights=rgb[..., 2].ravel(),
                        minlength=maximum + 1,
                    )[: maximum + 1]
                _accumulate_context_grid(
                    isolated_context_evidence,
                    isolated_context_pixels,
                    evidence,
                    x0=x0,
                    y0=y0,
                    bin_size=isolated_debris_context_bin_size,
                )
                _accumulate_context_grid(
                    micro_context_evidence,
                    micro_context_pixels,
                    evidence,
                    x0=x0,
                    y0=y0,
                    bin_size=isolated_debris_micro_bin_size,
                )
                _accumulate_context_grid(
                    organized_context_prediction,
                    organized_context_pixels,
                    block > 0,
                    x0=x0,
                    y0=y0,
                    bin_size=isolated_debris_organized_bin_size,
                )
                organized_rows = np.minimum(
                    np.arange(
                        y0 // isolated_debris_organized_bin_size,
                        (y1 + isolated_debris_organized_bin_size - 1)
                        // isolated_debris_organized_bin_size,
                    )
                    * isolated_debris_organized_bin_size
                    + isolated_debris_organized_bin_size // 2,
                    height - 1,
                )
                organized_columns = np.minimum(
                    np.arange(
                        x0 // isolated_debris_organized_bin_size,
                        (x1 + isolated_debris_organized_bin_size - 1)
                        // isolated_debris_organized_bin_size,
                    )
                    * isolated_debris_organized_bin_size
                    + isolated_debris_organized_bin_size // 2,
                    width - 1,
                )
                organized_context_labels[
                    y0 // isolated_debris_organized_bin_size : (
                        y1 + isolated_debris_organized_bin_size - 1
                    )
                    // isolated_debris_organized_bin_size,
                    x0 // isolated_debris_organized_bin_size : (
                        x1 + isolated_debris_organized_bin_size - 1
                    )
                    // isolated_debris_organized_bin_size,
                ] = block[np.ix_(organized_rows - y0, organized_columns - x0)]
                organized_context_red_blue[
                    y0 // isolated_debris_organized_bin_size : (
                        y1 + isolated_debris_organized_bin_size - 1
                    )
                    // isolated_debris_organized_bin_size,
                    x0 // isolated_debris_organized_bin_size : (
                        x1 + isolated_debris_organized_bin_size - 1
                    )
                    // isolated_debris_organized_bin_size,
                ] = rgb[np.ix_(organized_rows - y0, organized_columns - x0)][
                    ..., 0
                ].astype(np.int16) - rgb[
                    np.ix_(organized_rows - y0, organized_columns - x0)
                ][..., 2].astype(np.int16)
                organized_context_intensity[
                    y0 // isolated_debris_organized_bin_size : (
                        y1 + isolated_debris_organized_bin_size - 1
                    )
                    // isolated_debris_organized_bin_size,
                    x0 // isolated_debris_organized_bin_size : (
                        x1 + isolated_debris_organized_bin_size - 1
                    )
                    // isolated_debris_organized_bin_size,
                ] = np.mean(
                    rgb[np.ix_(organized_rows - y0, organized_columns - x0)],
                    axis=2,
                ).astype(np.uint8)
            counts = np.bincount(block.ravel(), minlength=maximum + 1)
            evidence_counts = np.bincount(
                block[evidence].ravel(), minlength=maximum + 1
            )
            areas += counts[: maximum + 1].astype(np.uint64)
            supported += evidence_counts[: maximum + 1].astype(np.uint64)
            if (
                require_nuclear_support
                or require_adaptive_nuclear_core
                or require_isolated_debris_gate
            ):
                concentration = hematoxylin_concentration(
                    rgb, background_percentile=_EVIDENCE_BACKGROUND_PERCENTILE
                )
            if require_nuclear_support:
                nuclear = concentration >= nuclear_minimum_optical_density
                nuclear_counts = np.bincount(
                    block[nuclear].ravel(), minlength=maximum + 1
                )
                nuclear_supported += nuclear_counts[: maximum + 1].astype(np.uint64)
            if (
                require_adaptive_nuclear_core or require_isolated_debris_gate
            ) and np.any(block > 0):
                adaptive_threshold = float(
                    np.percentile(
                        concentration[block > 0], adaptive_nuclear_core_percentile
                    )
                )
                adaptive_core_thresholds.append(adaptive_threshold)
                channel_maximum = np.maximum(rgb[..., 0], rgb[..., 1]).astype(
                    np.float32
                )
                if require_adaptive_nuclear_core:
                    chromatic_core = rgb[..., 2].astype(np.float32) >= (
                        channel_maximum * adaptive_nuclear_core_minimum_blue_ratio
                    )
                    adaptive_counts = np.bincount(
                        block[
                            (concentration >= adaptive_threshold) & chromatic_core
                        ].ravel(),
                        minlength=maximum + 1,
                    )
                    adaptive_core_supported += adaptive_counts[: maximum + 1].astype(
                        np.uint64
                    )
                if require_isolated_debris_gate:
                    channel_minimum = np.min(rgb, axis=2)
                    channel_maximum_uint8 = np.max(rgb, axis=2)
                    neutral_dark = (
                        channel_maximum_uint8
                        <= isolated_debris_neutral_dark_maximum_value
                    ) & (
                        channel_maximum_uint8 - channel_minimum
                        <= isolated_debris_neutral_dark_maximum_chroma
                    )
                    very_dark = (
                        channel_maximum_uint8 <= isolated_debris_very_dark_maximum_value
                    ) & (
                        channel_maximum_uint8 - channel_minimum
                        <= isolated_debris_very_dark_maximum_chroma
                    )
                    neutral_counts = np.bincount(
                        block[neutral_dark].ravel(), minlength=maximum + 1
                    )
                    very_dark_counts = np.bincount(
                        block[very_dark].ravel(), minlength=maximum + 1
                    )
                    isolated_neutral_dark += neutral_counts[: maximum + 1].astype(
                        np.uint64
                    )
                    isolated_very_dark += very_dark_counts[: maximum + 1].astype(
                        np.uint64
                    )
                    isolated_chromatic_core = rgb[..., 2].astype(np.float32) >= (
                        channel_maximum * isolated_debris_nuclear_minimum_blue_ratio
                    )
                    isolated_nuclear_evidence = (
                        concentration >= adaptive_threshold
                    ) & isolated_chromatic_core
                    _accumulate_context_grid(
                        isolated_context_nuclear,
                        isolated_context_nuclear_pixels,
                        isolated_nuclear_evidence,
                        x0=x0,
                        y0=y0,
                        bin_size=isolated_debris_context_bin_size,
                    )
                    _accumulate_context_grid(
                        micro_context_nuclear,
                        micro_context_nuclear_pixels,
                        isolated_nuclear_evidence,
                        x0=x0,
                        y0=y0,
                        bin_size=isolated_debris_micro_bin_size,
                    )
                    isolated_counts = np.bincount(
                        block[isolated_nuclear_evidence].ravel(),
                        minlength=maximum + 1,
                    )
                    isolated_nuclear_supported += isolated_counts[: maximum + 1].astype(
                        np.uint64
                    )
    context_supported = np.zeros(maximum + 1, dtype=np.uint64)
    if require_source_tissue_context:
        context_fraction = context_evidence / np.maximum(context_pixels, 1)
        context_grid = context_fraction >= source_tissue_context_minimum_fraction
        for y0 in range(0, height, block_size):
            y1 = min(height, y0 + block_size)
            rows = np.arange(y0, y1) // source_tissue_context_bin_size
            for x0 in range(0, width, block_size):
                x1 = min(width, x0 + block_size)
                columns = np.arange(x0, x1) // source_tissue_context_bin_size
                block = np.asarray(labels[y0:y1, x0:x1])
                supported_pixels = context_grid[np.ix_(rows, columns)]
                counts = np.bincount(
                    block[supported_pixels].ravel(), minlength=maximum + 1
                )
                context_supported += counts[: maximum + 1].astype(np.uint64)
    isolated_dense_context_supported = np.zeros(maximum + 1, dtype=np.uint64)
    if require_isolated_debris_gate:
        isolated_context_fraction = isolated_context_evidence / np.maximum(
            isolated_context_pixels, 1
        )
        isolated_stain_grid = (
            isolated_context_fraction >= isolated_debris_context_minimum_stain_fraction
        )
        from scipy import ndimage as ndi

        isolated_components, component_count = ndi.label(
            isolated_stain_grid,
            structure=np.ones((3, 3), dtype=np.uint8),
        )
        component_nuclear_pixels = np.bincount(
            isolated_components.ravel(),
            weights=isolated_context_nuclear.ravel(),
            minlength=component_count + 1,
        )
        isolated_dense_grid = isolated_stain_grid & (
            component_nuclear_pixels[isolated_components]
            >= isolated_debris_component_minimum_nuclear_pixels
        )
        for y0 in range(0, height, block_size):
            y1 = min(height, y0 + block_size)
            rows = np.arange(y0, y1) // isolated_debris_context_bin_size
            for x0 in range(0, width, block_size):
                x1 = min(width, x0 + block_size)
                columns = np.arange(x0, x1) // isolated_debris_context_bin_size
                block = np.asarray(labels[y0:y1, x0:x1])
                dense_pixels = isolated_dense_grid[np.ix_(rows, columns)]
                counts = np.bincount(block[dense_pixels].ravel(), minlength=maximum + 1)
                isolated_dense_context_supported += counts[: maximum + 1].astype(
                    np.uint64
                )
        micro_context_fraction = micro_context_evidence / np.maximum(
            micro_context_pixels, 1
        )
        micro_stain_grid = (
            micro_context_fraction >= isolated_debris_micro_minimum_stain_fraction
        )
        micro_components, micro_component_count = ndi.label(
            micro_stain_grid,
            structure=np.ones((3, 3), dtype=np.uint8),
        )
        micro_component_bins = np.bincount(
            micro_components.ravel(), minlength=micro_component_count + 1
        )
        micro_component_nuclear = np.bincount(
            micro_components.ravel(),
            weights=micro_context_nuclear.ravel(),
            minlength=micro_component_count + 1,
        )
        micro_component_pixels = np.bincount(
            micro_components.ravel(),
            weights=micro_context_pixels.ravel(),
            minlength=micro_component_count + 1,
        )
        micro_component_nuclear_fraction = np.divide(
            micro_component_nuclear,
            np.maximum(micro_component_pixels, 1),
        )
        micro_debris_components = (
            micro_component_bins <= isolated_debris_micro_maximum_component_bins
        ) & (
            micro_component_nuclear_fraction
            < isolated_debris_micro_minimum_nuclear_fraction
        )
        micro_debris_components[0] = False
        micro_debris_grid = micro_stain_grid & micro_debris_components[micro_components]
        isolated_nuclear_required_for_grid = np.maximum(
            isolated_debris_nuclear_minimum_pixels,
            np.ceil(areas * isolated_debris_nuclear_minimum_fraction).astype(np.uint64),
        )
        isolated_nuclear_keep_for_grid = (
            isolated_nuclear_supported >= isolated_nuclear_required_for_grid
        )
        isolated_nuclear_keep_for_grid[0] = False
        organized_grid = _locally_organized_prediction_grid(
            organized_context_prediction,
            organized_context_pixels,
            organized_context_labels,
            organized_context_red_blue,
            bin_size=isolated_debris_organized_bin_size,
            window_size=isolated_debris_organized_window_size,
            window_overlap=isolated_debris_organized_window_overlap,
            minimum_bin_occupancy_fraction=(
                isolated_debris_organized_minimum_bin_occupancy_fraction
            ),
            minimum_component_pixels=(
                isolated_debris_organized_minimum_component_pixels
            ),
            compact_minimum_component_pixels=(
                isolated_debris_organized_compact_minimum_component_pixels
            ),
            minimum_aspect_ratio=(isolated_debris_organized_minimum_aspect_ratio),
            minimum_mean_instance_pixels=(
                isolated_debris_organized_minimum_mean_instance_pixels
            ),
            strong_red_blue_difference=(
                isolated_debris_organized_strong_red_blue_difference
            ),
            minimum_strong_chromatic_fraction=(
                isolated_debris_organized_minimum_strong_chromatic_fraction
            ),
        )
        for y0 in range(0, height, block_size):
            y1 = min(height, y0 + block_size)
            rows = np.arange(y0, y1) // isolated_debris_micro_bin_size
            for x0 in range(0, width, block_size):
                x1 = min(width, x0 + block_size)
                columns = np.arange(x0, x1) // isolated_debris_micro_bin_size
                block = np.asarray(labels[y0:y1, x0:x1])
                debris_pixels = micro_debris_grid[np.ix_(rows, columns)]
                counts = np.bincount(
                    block[debris_pixels].ravel(), minlength=maximum + 1
                )
                isolated_micro_debris_supported += counts[: maximum + 1].astype(
                    np.uint64
                )
        for y0 in range(0, height, block_size):
            y1 = min(height, y0 + block_size)
            rows = np.arange(y0, y1) // isolated_debris_organized_bin_size
            for x0 in range(0, width, block_size):
                x1 = min(width, x0 + block_size)
                columns = np.arange(x0, x1) // isolated_debris_organized_bin_size
                block = np.asarray(labels[y0:y1, x0:x1])
                organized_pixels = organized_grid[np.ix_(rows, columns)]
                counts = np.bincount(
                    block[organized_pixels].ravel(), minlength=maximum + 1
                )
                isolated_organized_supported += counts[: maximum + 1].astype(np.uint64)
    size_keep = (areas == 0) | (areas >= minimum_area)
    size_keep[0] = True
    required = np.ceil(areas * _EVIDENCE_MINIMUM_INSTANCE_FRACTION).astype(np.uint64)
    stain_keep = (areas > 0) & (supported >= required)
    stain_keep[0] = True
    nuclear_required = np.maximum(
        nuclear_minimum_pixels,
        np.ceil(areas * nuclear_minimum_fraction).astype(np.uint64),
    )
    nuclear_keep = nuclear_supported >= nuclear_required
    nuclear_keep[0] = True
    context_required = np.ceil(
        areas * source_tissue_context_minimum_instance_fraction
    ).astype(np.uint64)
    context_keep = context_supported >= context_required
    context_keep[0] = True
    adaptive_core_required = np.maximum(
        adaptive_nuclear_core_minimum_pixels,
        np.ceil(areas * adaptive_nuclear_core_minimum_fraction).astype(np.uint64),
    )
    adaptive_core_keep = (areas < adaptive_nuclear_core_minimum_area) | (
        adaptive_core_supported >= adaptive_core_required
    )
    adaptive_core_keep[0] = True
    isolated_context_required = np.ceil(
        areas * isolated_debris_context_minimum_instance_fraction
    ).astype(np.uint64)
    isolated_dense_keep = isolated_dense_context_supported >= isolated_context_required
    isolated_nuclear_required = np.maximum(
        isolated_debris_nuclear_minimum_pixels,
        np.ceil(areas * isolated_debris_nuclear_minimum_fraction).astype(np.uint64),
    )
    isolated_nuclear_keep = isolated_nuclear_supported >= isolated_nuclear_required
    isolated_organized_required = np.ceil(
        areas * isolated_debris_organized_minimum_instance_fraction
    ).astype(np.uint64)
    isolated_organized_keep = (
        isolated_organized_supported >= isolated_organized_required
    )
    isolated_organized_keep &= isolated_debris_allow_organized_escape
    organized_necrotic_sparse_glass_excluded = np.zeros(maximum + 1, dtype=bool)
    nuclear_escape_keep = (
        isolated_nuclear_keep & isolated_debris_allow_nuclear_escape
        if require_nuclear_support and require_isolated_debris_gate
        else False
    )
    global_nuclear_escape_keep = (
        nuclear_keep
        & (
            nuclear_supported
            < np.ceil(
                areas * isolated_debris_global_nuclear_escape_maximum_fraction
            ).astype(np.uint64)
        )
        & isolated_debris_allow_nuclear_escape
        if require_nuclear_support and require_isolated_debris_gate
        else np.zeros(maximum + 1, dtype=bool)
    )
    global_nuclear_escape_keep[0] = False
    nuclear_effective_keep = (
        nuclear_keep
        | (
            isolated_organized_keep
            if require_nuclear_support and require_isolated_debris_gate
            else False
        )
        | nuclear_escape_keep
    )
    nuclear_effective_keep[0] = True
    adaptive_core_effective_keep = adaptive_core_keep | (
        isolated_organized_keep
        if require_adaptive_nuclear_core and require_isolated_debris_gate
        else False
    )
    adaptive_core_effective_keep[0] = True
    isolated_context_keep = (
        isolated_dense_keep
        | (isolated_nuclear_keep & isolated_debris_allow_nuclear_escape)
        | global_nuclear_escape_keep
        | isolated_organized_keep
    )
    neutral_dark_artifact = isolated_neutral_dark >= np.ceil(
        areas * isolated_debris_neutral_dark_maximum_fraction
    ).astype(np.uint64)
    very_dark_artifact = isolated_very_dark >= np.ceil(
        areas * isolated_debris_very_dark_maximum_fraction
    ).astype(np.uint64)
    micro_island_artifact = isolated_micro_debris_supported >= np.ceil(
        areas * isolated_debris_micro_minimum_instance_fraction
    ).astype(np.uint64)
    micro_island_artifact &= ~isolated_organized_keep & ~global_nuclear_escape_keep
    if require_isolated_debris_gate:
        compact_removal_eligible = (
            size_keep
            & stain_keep
            & (nuclear_effective_keep if require_nuclear_support else True)
            & (context_keep if require_source_tissue_context else True)
            & (adaptive_core_effective_keep if require_adaptive_nuclear_core else True)
            & isolated_context_keep
            & (~neutral_dark_artifact)
            & (~very_dark_artifact)
            & (~micro_island_artifact)
        )
        compact_unsupported_grid = _compact_unsupported_prediction_grid(
            organized_context_prediction,
            organized_context_pixels,
            organized_context_labels,
            organized_context_red_blue,
            isolated_nuclear_keep_for_grid,
            areas,
            compact_removal_eligible,
            bin_size=isolated_debris_organized_bin_size,
            minimum_bin_occupancy_fraction=(
                isolated_debris_organized_minimum_bin_occupancy_fraction
            ),
            minimum_component_pixels=(isolated_debris_compact_unsupported_minimum_area),
            maximum_aspect_ratio=(
                isolated_debris_compact_unsupported_maximum_aspect_ratio
            ),
            maximum_mean_instance_pixels=(
                isolated_debris_compact_unsupported_maximum_mean_instance_pixels
            ),
            maximum_mean_red_blue_difference=(
                isolated_debris_compact_unsupported_maximum_mean_red_blue_difference
            ),
            strong_red_blue_difference=(
                isolated_debris_compact_unsupported_strong_red_blue_difference
            ),
            maximum_strong_chromatic_fraction=(
                isolated_debris_compact_unsupported_maximum_strong_chromatic_fraction
            ),
            elongation_ratio=(isolated_debris_compact_unsupported_elongation_ratio),
            maximum_elongated_instance_fraction=(
                isolated_debris_compact_unsupported_maximum_elongated_instance_fraction
            ),
            maximum_context_nuclear_supported_fraction=(
                isolated_debris_compact_unsupported_maximum_context_nuclear_fraction
            ),
            minimum_component_fill_fraction=(
                isolated_debris_compact_unsupported_minimum_component_fill_fraction
            ),
            minimum_instance_fraction=(
                isolated_debris_compact_unsupported_minimum_instance_fraction
            ),
        )
        foam_core_candidate = _bright_instance_core_candidates(
            labels,
            read_rgb,
            maximum=maximum,
            instance_areas=areas,
            minimum_core_pixels=isolated_debris_foam_core_minimum_pixels,
            minimum_bright_fraction=(isolated_debris_foam_core_minimum_bright_fraction),
            minimum_intensity=isolated_debris_foam_core_minimum_intensity,
            maximum_instance_area=(isolated_debris_foam_core_maximum_instance_area),
            block_size=block_size,
        )
        foam_core_candidate &= compact_removal_eligible
        foam_unsupported_grid = _compact_unsupported_prediction_grid(
            organized_context_prediction,
            organized_context_pixels,
            organized_context_labels,
            organized_context_red_blue,
            ~foam_core_candidate,
            areas,
            foam_core_candidate,
            bin_size=isolated_debris_organized_bin_size,
            minimum_bin_occupancy_fraction=(
                isolated_debris_organized_minimum_bin_occupancy_fraction
            ),
            minimum_component_pixels=(isolated_debris_foam_core_minimum_component_area),
            maximum_aspect_ratio=(isolated_debris_foam_core_maximum_aspect_ratio),
            maximum_mean_instance_pixels=float(
                isolated_debris_foam_core_maximum_instance_area
            ),
            maximum_mean_red_blue_difference=255.0,
            strong_red_blue_difference=(
                isolated_debris_compact_unsupported_strong_red_blue_difference
            ),
            maximum_strong_chromatic_fraction=1.0,
            elongation_ratio=(isolated_debris_compact_unsupported_elongation_ratio),
            maximum_elongated_instance_fraction=1.0,
            maximum_context_nuclear_supported_fraction=1.0,
            minimum_component_fill_fraction=(
                isolated_debris_compact_unsupported_minimum_component_fill_fraction
            ),
            minimum_instance_fraction=(
                isolated_debris_compact_unsupported_minimum_instance_fraction
            ),
            window_size=isolated_debris_organized_window_size,
            window_overlap=isolated_debris_organized_window_overlap,
            require_removable_instance_area=False,
            include_nuclear_supported_in_connectivity=True,
        )
        organized_necrotic_protection = _bounded_organized_necrotic_protection(
            isolated_organized_keep,
            areas,
            maximum_instance_area=(
                isolated_debris_organized_necrotic_protection_maximum_instance_area
            ),
        )
        protection_context_prediction = organized_context_prediction
        if organized_necrotic_protection_use_eligible_context:
            protection_context_prediction = np.zeros_like(organized_context_prediction)
            protection_context_pixels = np.zeros_like(organized_context_pixels)
            eligible_context_instances = compact_removal_eligible.copy()
            if (
                isolated_debris_organized_necrotic_protection_maximum_instance_area
                is not None
            ):
                maximum_protected_area = (
                    isolated_debris_organized_necrotic_protection_maximum_instance_area
                )
                eligible_context_instances &= ~(
                    isolated_organized_keep & (areas > maximum_protected_area)
                )
            if organized_necrotic_protection_require_independent_nuclear_context:
                independent_nuclear_context = (
                    nuclear_keep
                    | isolated_nuclear_keep
                    | global_nuclear_escape_keep
                    | adaptive_core_keep
                )
                independent_nuclear_context[0] = False
                eligible_context_instances &= independent_nuclear_context
            for y0 in range(0, height, block_size):
                y1 = min(height, y0 + block_size)
                for x0 in range(0, width, block_size):
                    x1 = min(width, x0 + block_size)
                    block = np.asarray(labels[y0:y1, x0:x1])
                    eligible_prediction = (block > 0) & eligible_context_instances[
                        block
                    ]
                    _accumulate_context_grid(
                        protection_context_prediction,
                        protection_context_pixels,
                        eligible_prediction,
                        x0=x0,
                        y0=y0,
                        bin_size=isolated_debris_organized_bin_size,
                    )
            if not np.array_equal(protection_context_pixels, organized_context_pixels):
                raise RuntimeError(
                    "organized-necrotic eligible context geometry changed"
                )
        if organized_necrotic_protection_minimum_local_prediction_fraction is not None:
            organized_necrotic_protection = (
                _locally_supported_organized_necrotic_protection(
                    labels,
                    organized_necrotic_protection,
                    areas,
                    protection_context_prediction,
                    organized_context_pixels,
                    maximum=maximum,
                    bin_size=isolated_debris_organized_bin_size,
                    window_size=256,
                    minimum_local_prediction_fraction=(
                        organized_necrotic_protection_minimum_local_prediction_fraction
                    ),
                    minimum_instance_fraction=(
                        isolated_debris_context_minimum_instance_fraction
                    ),
                    exclude_instance_from_local_prediction=(
                        organized_necrotic_protection_use_eligible_context
                    ),
                    context_instance_areas=(
                        np.where(eligible_context_instances, areas, 0)
                        if organized_necrotic_protection_use_eligible_context
                        else None
                    ),
                    block_size=block_size,
                )
            )
        if organized_necrotic_sparse_glass_shape_gate:
            organized_necrotic_sparse_glass_excluded = (
                _sparse_glass_organized_necrotic_instances(
                    organized_context_labels,
                    organized_context_intensity,
                    organized_necrotic_protection,
                    bin_size=isolated_debris_organized_bin_size,
                    context_window_size=(
                        _ORGANIZED_NECROTIC_SPARSE_GLASS_CONTEXT_WINDOW
                    ),
                    maximum_mean_intensity=(
                        _ORGANIZED_NECROTIC_SPARSE_GLASS_MAXIMUM_MEAN_INTENSITY
                    ),
                    minimum_tissue_fraction=(
                        _ORGANIZED_NECROTIC_SPARSE_GLASS_MINIMUM_TISSUE_FRACTION
                    ),
                    maximum_sampled_fill_fraction=(
                        _ORGANIZED_NECROTIC_SPARSE_GLASS_MAXIMUM_SAMPLED_FILL_FRACTION
                    ),
                    minimum_sampled_elongation=(
                        _ORGANIZED_NECROTIC_SPARSE_GLASS_MINIMUM_SAMPLED_ELONGATION
                    ),
                )
            )
            organized_necrotic_protection &= ~(organized_necrotic_sparse_glass_excluded)
        necrotic_grid = _fragmented_necrotic_prediction_grid(
            organized_context_prediction,
            organized_context_pixels,
            organized_context_labels,
            organized_context_red_blue,
            organized_context_intensity,
            bin_size=isolated_debris_organized_bin_size,
            minimum_component_pixels=(isolated_debris_necrotic_minimum_component_area),
            maximum_aspect_ratio=(isolated_debris_necrotic_maximum_aspect_ratio),
            minimum_fill_fraction=isolated_debris_necrotic_minimum_fill_fraction,
            maximum_fill_fraction=isolated_debris_necrotic_maximum_fill_fraction,
            minimum_mean_red_blue_difference=(
                isolated_debris_necrotic_minimum_mean_red_blue_difference
            ),
            maximum_mean_red_blue_difference=(
                isolated_debris_necrotic_maximum_mean_red_blue_difference
            ),
            minimum_mean_intensity=(isolated_debris_necrotic_minimum_mean_intensity),
            protected_instances=(
                organized_necrotic_protection
                if isolated_debris_protect_organized_from_necrotic
                else None
            ),
            window_size=isolated_debris_organized_window_size,
            window_overlap=isolated_debris_organized_window_overlap,
            context_window_size=256,
            maximum_local_prediction_fraction=0.40,
        )
        if isolated_debris_diffuse_degenerated_gate:
            # A second, deliberately narrow necrotic branch targets pale or
            # moderately brown fields made from many weakly nuclear fragments.
            # Colour alone is unsafe for IHC, so require a compact fragmented
            # topology, low local nuclear-support fraction, small mean
            # instances, and a relatively bright source field together.
            necrotic_grid |= _compact_unsupported_prediction_grid(
                organized_context_prediction,
                organized_context_pixels,
                organized_context_labels,
                organized_context_red_blue,
                isolated_nuclear_keep_for_grid,
                areas,
                compact_removal_eligible & (~isolated_organized_keep),
                bin_size=isolated_debris_organized_bin_size,
                minimum_bin_occupancy_fraction=(
                    isolated_debris_organized_minimum_bin_occupancy_fraction
                ),
                minimum_component_pixels=max(
                    1, isolated_debris_necrotic_minimum_component_area // 5
                ),
                maximum_aspect_ratio=3.0,
                maximum_mean_instance_pixels=600.0,
                maximum_mean_red_blue_difference=60.0,
                strong_red_blue_difference=70.0,
                maximum_strong_chromatic_fraction=0.25,
                elongation_ratio=(isolated_debris_compact_unsupported_elongation_ratio),
                maximum_elongated_instance_fraction=0.65,
                maximum_context_nuclear_supported_fraction=0.10,
                minimum_component_fill_fraction=0.05,
                minimum_instance_fraction=(
                    isolated_debris_necrotic_minimum_instance_fraction
                ),
                sampled_mean_intensity=organized_context_intensity,
                minimum_mean_intensity=150.0,
                window_size=isolated_debris_organized_window_size,
                window_overlap=isolated_debris_organized_window_overlap,
                include_nuclear_supported_in_connectivity=True,
            )
        brown_debris_grid = _fragmented_necrotic_prediction_grid(
            organized_context_prediction,
            organized_context_pixels,
            organized_context_labels,
            organized_context_red_blue,
            organized_context_intensity,
            bin_size=isolated_debris_organized_bin_size,
            minimum_component_pixels=(isolated_debris_brown_minimum_component_area),
            maximum_aspect_ratio=isolated_debris_brown_maximum_aspect_ratio,
            minimum_fill_fraction=isolated_debris_brown_minimum_fill_fraction,
            maximum_fill_fraction=isolated_debris_brown_maximum_fill_fraction,
            minimum_mean_red_blue_difference=(
                isolated_debris_brown_minimum_mean_red_blue_difference
            ),
            maximum_mean_red_blue_difference=255.0,
            minimum_mean_intensity=None,
            maximum_mean_intensity=(isolated_debris_brown_maximum_mean_intensity),
            minimum_bin_red_blue_difference=(
                isolated_debris_brown_minimum_mean_red_blue_difference
            ),
            maximum_bin_intensity=(isolated_debris_brown_maximum_mean_intensity),
            eligible_instances=None,
            window_size=isolated_debris_organized_window_size,
            window_overlap=isolated_debris_organized_window_overlap,
            context_window_size=256,
            maximum_local_prediction_fraction=0.50,
            maximum_source_component_pixels=(
                isolated_debris_brown_maximum_source_component_area
            ),
            source_maximum_mean_intensity=(
                isolated_debris_brown_source_maximum_mean_intensity
            ),
            source_dilation_bins=isolated_debris_brown_source_dilation_bins,
        )
        for y0 in range(0, height, block_size):
            y1 = min(height, y0 + block_size)
            rows = np.arange(y0, y1) // isolated_debris_organized_bin_size
            for x0 in range(0, width, block_size):
                x1 = min(width, x0 + block_size)
                columns = np.arange(x0, x1) // isolated_debris_organized_bin_size
                block = np.asarray(labels[y0:y1, x0:x1])
                compact_unsupported_pixels = compact_unsupported_grid[
                    np.ix_(rows, columns)
                ]
                counts = np.bincount(
                    block[compact_unsupported_pixels].ravel(),
                    minlength=maximum + 1,
                )
                isolated_compact_unsupported += counts[: maximum + 1].astype(np.uint64)
                foam_unsupported_pixels = foam_unsupported_grid[np.ix_(rows, columns)]
                foam_counts = np.bincount(
                    block[foam_unsupported_pixels].ravel(),
                    minlength=maximum + 1,
                )
                isolated_foam_unsupported += foam_counts[: maximum + 1].astype(
                    np.uint64
                )
                necrotic_pixels = necrotic_grid[np.ix_(rows, columns)]
                necrotic_counts = np.bincount(
                    block[necrotic_pixels].ravel(),
                    minlength=maximum + 1,
                )
                isolated_necrotic_supported += necrotic_counts[: maximum + 1].astype(
                    np.uint64
                )
                brown_debris_pixels = brown_debris_grid[np.ix_(rows, columns)]
                brown_debris_counts = np.bincount(
                    block[brown_debris_pixels].ravel(),
                    minlength=maximum + 1,
                )
                isolated_brown_debris_supported += brown_debris_counts[
                    : maximum + 1
                ].astype(np.uint64)
    compact_unsupported_artifact = (areas > 0) & (
        isolated_compact_unsupported
        >= np.ceil(
            areas * isolated_debris_compact_unsupported_minimum_instance_fraction
        ).astype(np.uint64)
    )
    foam_unsupported_artifact = (areas > 0) & (
        isolated_foam_unsupported
        >= np.ceil(
            areas * isolated_debris_compact_unsupported_minimum_instance_fraction
        ).astype(np.uint64)
    )
    necrotic_artifact = (areas > 0) & (
        isolated_necrotic_supported
        >= np.ceil(areas * isolated_debris_necrotic_minimum_instance_fraction).astype(
            np.uint64
        )
    )
    brown_debris_artifact = (areas > 0) & (
        isolated_brown_debris_supported
        >= np.ceil(areas * isolated_debris_brown_minimum_instance_fraction).astype(
            np.uint64
        )
    )
    oversized_chromatic_artifact = (
        areas >= isolated_debris_oversized_chromatic_minimum_area
    ) & (
        isolated_nuclear_supported
        >= np.ceil(areas * isolated_debris_oversized_chromatic_minimum_fraction).astype(
            np.uint64
        )
    )
    fold_artifact = (
        _fold_artifact_instances(
            organized_context_labels,
            organized_context_red_blue,
            organized_context_intensity,
            isolated_nuclear_supported,
            areas,
            minimum_instance_area=isolated_debris_fold_minimum_instance_area,
            minimum_component_area=isolated_debris_fold_minimum_component_area,
            maximum_nuclear_fraction=isolated_debris_nuclear_minimum_fraction,
            maximum_mean_red_blue_difference=(
                isolated_debris_fold_maximum_mean_red_blue_difference
            ),
            maximum_mean_intensity=isolated_debris_fold_maximum_mean_intensity,
            dense_minimum_instances=(isolated_debris_fold_dense_minimum_instances),
            dense_minimum_aspect_ratio=(
                isolated_debris_fold_dense_minimum_aspect_ratio
            ),
            dense_connectivity_dilation_bins=(
                isolated_debris_fold_dense_connectivity_dilation_bins
            ),
            dense_maximum_mean_red_blue_difference=(
                isolated_debris_fold_dense_maximum_mean_red_blue_difference
            ),
            dense_maximum_mean_intensity=(
                isolated_debris_fold_dense_maximum_mean_intensity
            ),
            aneuclear_removal_eligible=compact_removal_eligible,
        )
        if require_isolated_debris_gate
        else np.zeros(maximum + 1, dtype=bool)
    )
    oversized_brown_artifact = np.zeros(maximum + 1, dtype=bool)
    if isolated_debris_oversized_brown_gate:
        if (
            isolated_red_sum is None
            or isolated_green_sum is None
            or isolated_blue_sum is None
        ):
            raise RuntimeError("oversized-brown channel sums are absent")
        oversized_brown_artifact = _oversized_brown_anuclear_instances(
            areas,
            isolated_nuclear_supported,
            isolated_red_sum,
            isolated_green_sum,
            isolated_blue_sum,
            minimum_instance_area=(isolated_debris_oversized_chromatic_minimum_area),
            maximum_mean_intensity=(
                isolated_debris_oversized_brown_maximum_mean_intensity
            ),
        )
    clustered_brown_artifact = np.zeros(maximum + 1, dtype=bool)
    if isolated_debris_clustered_brown_gate:
        if (
            isolated_red_sum is None
            or isolated_green_sum is None
            or isolated_blue_sum is None
        ):
            raise RuntimeError("clustered-brown channel sums are absent")
        clustered_brown_artifact = _clustered_brown_anuclear_instances(
            organized_context_prediction,
            organized_context_pixels,
            organized_context_labels,
            areas,
            isolated_nuclear_supported,
            isolated_red_sum,
            isolated_green_sum,
            isolated_blue_sum,
            bin_size=isolated_debris_organized_bin_size,
            minimum_instance_area=(
                isolated_debris_clustered_brown_minimum_instance_area
            ),
            maximum_instance_area=(
                isolated_debris_clustered_brown_maximum_instance_area
            ),
            minimum_component_area=(
                isolated_debris_clustered_brown_minimum_component_area
            ),
        )
    detached_fragment_artifact = (
        _detached_oversized_fragment_instances(
            organized_context_prediction,
            organized_context_pixels,
            organized_context_labels,
            areas,
            isolated_red_blue_sum,
            isolated_intensity_sum,
            bin_size=isolated_debris_organized_bin_size,
            minimum_instance_area=(isolated_debris_detached_minimum_instance_area),
            minimum_component_area=(isolated_debris_detached_minimum_component_area),
            minimum_instances=isolated_debris_detached_minimum_instances,
            minimum_aspect_ratio=(isolated_debris_detached_minimum_aspect_ratio),
            maximum_mean_red_blue_difference=(
                isolated_debris_detached_maximum_mean_red_blue_difference
            ),
            maximum_mean_intensity=(isolated_debris_detached_maximum_mean_intensity),
            context_window_size=(isolated_debris_detached_context_window_size),
            maximum_prediction_fraction=(
                isolated_debris_detached_maximum_prediction_fraction
            ),
            compact_minimum_instance_area=(
                isolated_debris_detached_compact_minimum_instance_area
            ),
            compact_minimum_component_area=(
                isolated_debris_detached_compact_minimum_component_area
            ),
            compact_minimum_instances=(
                isolated_debris_detached_compact_minimum_instances
            ),
            compact_maximum_mean_red_blue_difference=(
                isolated_debris_detached_compact_maximum_mean_red_blue_difference
            ),
            compact_maximum_mean_intensity=(
                isolated_debris_detached_compact_maximum_mean_intensity
            ),
        )
        if require_isolated_debris_gate
        else np.zeros(maximum + 1, dtype=bool)
    )
    detached_fragment_artifact |= oversized_brown_artifact
    detached_fragment_artifact |= clustered_brown_artifact
    satellite_debris_candidate = (
        _sparse_anuclear_satellite_instances(
            organized_context_prediction,
            organized_context_pixels,
            organized_context_labels,
            areas,
            isolated_nuclear_supported,
            isolated_red_blue_sum,
            isolated_intensity_sum,
            bin_size=isolated_debris_organized_bin_size,
            maximum_instance_area=(isolated_debris_satellite_maximum_instance_area),
            minimum_component_area=(isolated_debris_satellite_minimum_component_area),
        )
        if require_isolated_debris_gate and isolated_debris_satellite_gate
        else np.zeros(maximum + 1, dtype=bool)
    )
    satellite_interior_protected = np.zeros(maximum + 1, dtype=bool)
    if isolated_debris_satellite_minimum_mask_distance_um is not None:
        if (
            isolated_debris_satellite_tissue_mask is None
            or isolated_debris_satellite_mpp_xy is None
        ):
            raise RuntimeError("satellite interior-protection geometry is absent")
        satellite_interior_protected = _interior_tissue_instance_protection(
            organized_context_labels,
            isolated_debris_satellite_tissue_mask,
            maximum + 1,
            content_shape=(height, width),
            bin_size=isolated_debris_organized_bin_size,
            mpp_xy=isolated_debris_satellite_mpp_xy,
            minimum_mask_distance_um=(
                isolated_debris_satellite_minimum_mask_distance_um
            ),
            minimum_sample_fraction=(CELL_SATELLITE_PROTECTION_MINIMUM_SAMPLE_FRACTION),
        )
    satellite_debris_artifact = satellite_debris_candidate & (
        ~satellite_interior_protected
    )
    neutral_precipitate_artifact = (
        _sparse_neutral_dark_precipitate_instances(
            organized_context_prediction,
            organized_context_pixels,
            organized_context_labels,
            areas,
            isolated_nuclear_supported,
            isolated_neutral_dark,
            isolated_very_dark,
            isolated_red_blue_sum,
            isolated_intensity_sum,
            bin_size=isolated_debris_organized_bin_size,
            minimum_instance_area=isolated_debris_glass_minimum_area,
            maximum_instance_area=isolated_debris_satellite_maximum_instance_area * 2,
            minimum_neutral_dark_fraction=0.28,
            minimum_very_dark_fraction=0.20,
            maximum_nuclear_fraction=0.35,
            minimum_mean_red_blue_difference=-10.0,
            maximum_mean_red_blue_difference=15.0,
            maximum_mean_intensity=135.0,
            context_window_size=256,
            maximum_prediction_fraction=0.25,
            minimum_sparse_sample_fraction=0.50,
        )
        if require_isolated_debris_gate and isolated_debris_neutral_precipitate_gate
        else np.zeros(maximum + 1, dtype=bool)
    )
    glass_artifact = (
        _isolated_glass_artifact_instances(
            organized_context_prediction,
            organized_context_pixels,
            organized_context_labels,
            organized_context_red_blue,
            organized_context_intensity,
            areas,
            instance_red_blue_sum=isolated_red_blue_sum,
            instance_intensity_sum=isolated_intensity_sum,
            stain_counts=isolated_context_evidence,
            stain_pixel_counts=isolated_context_pixels,
            stain_bin_size=isolated_debris_context_bin_size,
            maximum_context_stain_fraction=(
                isolated_debris_glass_maximum_context_stain_fraction
            ),
            low_stain_maximum_mean_red_blue_difference=(
                isolated_debris_glass_low_stain_maximum_mean_red_blue_difference
            ),
            low_stain_minimum_mean_intensity=(
                isolated_debris_glass_low_stain_minimum_mean_intensity
            ),
            bin_size=isolated_debris_organized_bin_size,
            context_window_size=isolated_debris_glass_context_window_size,
            minimum_area=isolated_debris_glass_minimum_area,
            maximum_prediction_fraction=(
                isolated_debris_glass_maximum_prediction_fraction
            ),
            minimum_context_fraction=isolated_debris_glass_minimum_context_fraction,
            maximum_mean_red_blue_difference=(
                isolated_debris_glass_maximum_mean_red_blue_difference
            ),
            minimum_mean_intensity=isolated_debris_glass_minimum_mean_intensity,
        )
        if require_isolated_debris_gate
        else np.zeros(maximum + 1, dtype=bool)
    )
    if require_isolated_debris_gate:
        glass_artifact &= compact_removal_eligible & ~global_nuclear_escape_keep
    if isolated_debris_self_dense_glass_gate:
        # Keep this detector independent of the ordinary compact-removal and
        # global-nuclear-escape guards.  The failure mode it targets is a
        # single blurred scanner/glass spot whose own predicted footprint
        # makes its context look dense and whose purple blur can create false
        # nuclear support.  Re-applying those older guards here makes the
        # detector a no-op on precisely that morphology.  Its strict size,
        # neutral-colour, compact-fill, and sparse-ring criteria provide the
        # necessary protection for real cells embedded in tissue.
        glass_artifact |= _self_dense_glass_artifact_instances(
            organized_context_prediction,
            organized_context_pixels,
            organized_context_labels,
            areas,
            isolated_red_blue_sum,
            isolated_intensity_sum,
            bin_size=isolated_debris_organized_bin_size,
            minimum_instance_area=max(1, isolated_debris_glass_minimum_area * 80),
            minimum_mean_red_blue_difference=-10.0,
            maximum_mean_red_blue_difference=10.0,
            minimum_mean_intensity=200.0,
            maximum_aspect_ratio=1.8,
            minimum_sampled_fill_fraction=0.55,
            minimum_sampled_area_fraction=0.40,
            ring_dilation_bins=32,
            maximum_ring_prediction_fraction=0.03,
        )
    isolated_keep = (
        isolated_context_keep
        & (~neutral_dark_artifact)
        & (~very_dark_artifact)
        & (~micro_island_artifact)
        & (~compact_unsupported_artifact)
        & (~foam_unsupported_artifact)
        & (~oversized_chromatic_artifact)
        & (~fold_artifact)
        & (~detached_fragment_artifact)
        & (~satellite_debris_artifact)
        & (~neutral_precipitate_artifact)
        & (~glass_artifact)
        & (~necrotic_artifact)
        & (~brown_debris_artifact)
    )
    isolated_keep[0] = True
    keep = (
        size_keep
        & stain_keep
        & (nuclear_effective_keep if require_nuclear_support else True)
        & (context_keep if require_source_tissue_context else True)
        & (adaptive_core_effective_keep if require_adaptive_nuclear_core else True)
        & (isolated_keep if require_isolated_debris_gate else True)
    )
    keep[0] = True
    small_removed = (areas > 0) & (~size_keep)
    flat_removed = (areas > 0) & size_keep & (~stain_keep)
    nuclear_removed = (
        size_keep & stain_keep & (~nuclear_effective_keep)
        if require_nuclear_support
        else None
    )
    context_removed = (
        size_keep
        & stain_keep
        & (nuclear_effective_keep if require_nuclear_support else True)
        & (~context_keep)
        if require_source_tissue_context
        else None
    )
    adaptive_core_removed = (
        size_keep
        & stain_keep
        & (nuclear_effective_keep if require_nuclear_support else True)
        & (context_keep if require_source_tissue_context else True)
        & (~adaptive_core_effective_keep)
        if require_adaptive_nuclear_core
        else None
    )
    isolated_debris_removed = (
        size_keep
        & stain_keep
        & (nuclear_effective_keep if require_nuclear_support else True)
        & (context_keep if require_source_tissue_context else True)
        & (adaptive_core_effective_keep if require_adaptive_nuclear_core else True)
        & (~isolated_keep)
        if require_isolated_debris_gate
        else None
    )
    micro_island_removed = (
        size_keep
        & stain_keep
        & (nuclear_effective_keep if require_nuclear_support else True)
        & (context_keep if require_source_tissue_context else True)
        & (adaptive_core_effective_keep if require_adaptive_nuclear_core else True)
        & isolated_context_keep
        & (~neutral_dark_artifact)
        & (~very_dark_artifact)
        & micro_island_artifact
        if require_isolated_debris_gate
        else None
    )
    oversized_chromatic_removed = (
        size_keep
        & stain_keep
        & (nuclear_effective_keep if require_nuclear_support else True)
        & (context_keep if require_source_tissue_context else True)
        & (adaptive_core_effective_keep if require_adaptive_nuclear_core else True)
        & isolated_context_keep
        & (~neutral_dark_artifact)
        & (~very_dark_artifact)
        & (~micro_island_artifact)
        & (~compact_unsupported_artifact)
        & (~foam_unsupported_artifact)
        & oversized_chromatic_artifact
        if require_isolated_debris_gate
        else None
    )
    fold_artifact_removed = (
        size_keep
        & stain_keep
        & (nuclear_effective_keep if require_nuclear_support else True)
        & (context_keep if require_source_tissue_context else True)
        & (adaptive_core_effective_keep if require_adaptive_nuclear_core else True)
        & isolated_context_keep
        & (~neutral_dark_artifact)
        & (~very_dark_artifact)
        & (~micro_island_artifact)
        & (~compact_unsupported_artifact)
        & (~foam_unsupported_artifact)
        & (~oversized_chromatic_artifact)
        & fold_artifact
        if require_isolated_debris_gate
        else None
    )
    detached_fragment_removed = (
        size_keep
        & stain_keep
        & (nuclear_effective_keep if require_nuclear_support else True)
        & (context_keep if require_source_tissue_context else True)
        & (adaptive_core_effective_keep if require_adaptive_nuclear_core else True)
        & isolated_context_keep
        & (~neutral_dark_artifact)
        & (~very_dark_artifact)
        & (~micro_island_artifact)
        & (~compact_unsupported_artifact)
        & (~foam_unsupported_artifact)
        & (~oversized_chromatic_artifact)
        & (~fold_artifact)
        & detached_fragment_artifact
        if require_isolated_debris_gate
        else None
    )
    clustered_brown_fragment_removed = (
        detached_fragment_removed & clustered_brown_artifact
        if detached_fragment_removed is not None
        else None
    )
    satellite_debris_removed = (
        size_keep
        & stain_keep
        & (nuclear_effective_keep if require_nuclear_support else True)
        & (context_keep if require_source_tissue_context else True)
        & (adaptive_core_effective_keep if require_adaptive_nuclear_core else True)
        & isolated_context_keep
        & (~neutral_dark_artifact)
        & (~very_dark_artifact)
        & (~micro_island_artifact)
        & (~compact_unsupported_artifact)
        & (~foam_unsupported_artifact)
        & (~oversized_chromatic_artifact)
        & (~fold_artifact)
        & (~detached_fragment_artifact)
        & satellite_debris_artifact
        if require_isolated_debris_gate
        else None
    )
    neutral_precipitate_removed = (
        size_keep
        & stain_keep
        & (nuclear_effective_keep if require_nuclear_support else True)
        & (context_keep if require_source_tissue_context else True)
        & (adaptive_core_effective_keep if require_adaptive_nuclear_core else True)
        & isolated_context_keep
        & (~neutral_dark_artifact)
        & (~very_dark_artifact)
        & (~micro_island_artifact)
        & (~compact_unsupported_artifact)
        & (~foam_unsupported_artifact)
        & (~oversized_chromatic_artifact)
        & (~fold_artifact)
        & (~detached_fragment_artifact)
        & (~satellite_debris_artifact)
        & neutral_precipitate_artifact
        if require_isolated_debris_gate
        else None
    )
    glass_artifact_removed = (
        size_keep
        & stain_keep
        & (nuclear_effective_keep if require_nuclear_support else True)
        & (context_keep if require_source_tissue_context else True)
        & (adaptive_core_effective_keep if require_adaptive_nuclear_core else True)
        & isolated_context_keep
        & (~neutral_dark_artifact)
        & (~very_dark_artifact)
        & (~micro_island_artifact)
        & (~compact_unsupported_artifact)
        & (~foam_unsupported_artifact)
        & (~oversized_chromatic_artifact)
        & (~fold_artifact)
        & (~detached_fragment_artifact)
        & (~satellite_debris_artifact)
        & (~neutral_precipitate_artifact)
        & glass_artifact
        if require_isolated_debris_gate
        else None
    )
    necrotic_artifact_removed = (
        size_keep
        & stain_keep
        & (nuclear_effective_keep if require_nuclear_support else True)
        & (context_keep if require_source_tissue_context else True)
        & (adaptive_core_effective_keep if require_adaptive_nuclear_core else True)
        & isolated_context_keep
        & (~neutral_dark_artifact)
        & (~very_dark_artifact)
        & (~micro_island_artifact)
        & (~compact_unsupported_artifact)
        & (~foam_unsupported_artifact)
        & (~oversized_chromatic_artifact)
        & (~fold_artifact)
        & (~detached_fragment_artifact)
        & (~satellite_debris_artifact)
        & (~neutral_precipitate_artifact)
        & (~glass_artifact)
        & necrotic_artifact
        if require_isolated_debris_gate
        else None
    )
    brown_debris_artifact_removed = (
        size_keep
        & stain_keep
        & (nuclear_effective_keep if require_nuclear_support else True)
        & (context_keep if require_source_tissue_context else True)
        & (adaptive_core_effective_keep if require_adaptive_nuclear_core else True)
        & isolated_context_keep
        & (~neutral_dark_artifact)
        & (~very_dark_artifact)
        & (~micro_island_artifact)
        & (~compact_unsupported_artifact)
        & (~foam_unsupported_artifact)
        & (~oversized_chromatic_artifact)
        & (~fold_artifact)
        & (~detached_fragment_artifact)
        & (~satellite_debris_artifact)
        & (~neutral_precipitate_artifact)
        & (~glass_artifact)
        & (~necrotic_artifact)
        & brown_debris_artifact
        if require_isolated_debris_gate
        else None
    )
    compact_unsupported_removed = (
        size_keep
        & stain_keep
        & (nuclear_effective_keep if require_nuclear_support else True)
        & (context_keep if require_source_tissue_context else True)
        & (adaptive_core_effective_keep if require_adaptive_nuclear_core else True)
        & isolated_context_keep
        & (~neutral_dark_artifact)
        & (~very_dark_artifact)
        & (~micro_island_artifact)
        & compact_unsupported_artifact
        if require_isolated_debris_gate
        else None
    )
    foam_unsupported_removed = (
        size_keep
        & stain_keep
        & (nuclear_effective_keep if require_nuclear_support else True)
        & (context_keep if require_source_tissue_context else True)
        & (adaptive_core_effective_keep if require_adaptive_nuclear_core else True)
        & isolated_context_keep
        & (~neutral_dark_artifact)
        & (~very_dark_artifact)
        & (~micro_island_artifact)
        & (~compact_unsupported_artifact)
        & foam_unsupported_artifact
        if require_isolated_debris_gate
        else None
    )
    organized_tissue_escape = (
        size_keep
        & stain_keep
        & (nuclear_effective_keep if require_nuclear_support else True)
        & (context_keep if require_source_tissue_context else True)
        & (adaptive_core_effective_keep if require_adaptive_nuclear_core else True)
        & isolated_organized_keep
        & (
            (~nuclear_keep)
            | (~adaptive_core_keep)
            | ((~isolated_dense_keep) & (~isolated_nuclear_keep))
        )
        & (~neutral_dark_artifact)
        & (~very_dark_artifact)
        & (~micro_island_artifact)
        & (~compact_unsupported_artifact)
        & (~foam_unsupported_artifact)
        & (~oversized_chromatic_artifact)
        & (~fold_artifact)
        & (~detached_fragment_artifact)
        & (~satellite_debris_artifact)
        & (~neutral_precipitate_artifact)
        & (~glass_artifact)
        & (~necrotic_artifact)
        & (~brown_debris_artifact)
        if require_isolated_debris_gate
        else None
    )
    small_removed_instances = int(np.count_nonzero(small_removed))
    small_removed_pixels = int(areas[small_removed].sum())
    removed_instances = int(np.count_nonzero(flat_removed))
    removed_pixels = int(areas[flat_removed].sum())
    nuclear_removed_instances = (
        int(np.count_nonzero(nuclear_removed)) if nuclear_removed is not None else 0
    )
    nuclear_removed_pixels = (
        int(areas[nuclear_removed].sum()) if nuclear_removed is not None else 0
    )
    context_removed_instances = (
        int(np.count_nonzero(context_removed)) if context_removed is not None else 0
    )
    context_removed_pixels = (
        int(areas[context_removed].sum()) if context_removed is not None else 0
    )
    adaptive_core_removed_instances = (
        int(np.count_nonzero(adaptive_core_removed))
        if adaptive_core_removed is not None
        else 0
    )
    adaptive_core_removed_pixels = (
        int(areas[adaptive_core_removed].sum())
        if adaptive_core_removed is not None
        else 0
    )
    isolated_debris_removed_instances = (
        int(np.count_nonzero(isolated_debris_removed))
        if isolated_debris_removed is not None
        else 0
    )
    isolated_debris_removed_pixels = (
        int(areas[isolated_debris_removed].sum())
        if isolated_debris_removed is not None
        else 0
    )
    micro_island_removed_instances = (
        int(np.count_nonzero(micro_island_removed))
        if micro_island_removed is not None
        else 0
    )
    micro_island_removed_pixels = (
        int(areas[micro_island_removed].sum())
        if micro_island_removed is not None
        else 0
    )
    organized_tissue_escape_instances = (
        int(np.count_nonzero(organized_tissue_escape))
        if organized_tissue_escape is not None
        else 0
    )
    organized_tissue_escape_pixels = (
        int(areas[organized_tissue_escape].sum())
        if organized_tissue_escape is not None
        else 0
    )
    organized_necrotic_sparse_glass_excluded_instances = int(
        np.count_nonzero(organized_necrotic_sparse_glass_excluded)
    )
    organized_necrotic_sparse_glass_excluded_pixels = int(
        areas[organized_necrotic_sparse_glass_excluded].sum()
    )
    oversized_chromatic_removed_instances = (
        int(np.count_nonzero(oversized_chromatic_removed))
        if oversized_chromatic_removed is not None
        else 0
    )
    compact_unsupported_removed_instances = (
        int(np.count_nonzero(compact_unsupported_removed))
        if compact_unsupported_removed is not None
        else 0
    )
    compact_unsupported_removed_pixels = (
        int(areas[compact_unsupported_removed].sum())
        if compact_unsupported_removed is not None
        else 0
    )
    foam_unsupported_removed_instances = (
        int(np.count_nonzero(foam_unsupported_removed))
        if foam_unsupported_removed is not None
        else 0
    )
    foam_unsupported_removed_pixels = (
        int(areas[foam_unsupported_removed].sum())
        if foam_unsupported_removed is not None
        else 0
    )
    oversized_chromatic_removed_pixels = (
        int(areas[oversized_chromatic_removed].sum())
        if oversized_chromatic_removed is not None
        else 0
    )
    fold_artifact_removed_instances = (
        int(np.count_nonzero(fold_artifact_removed))
        if fold_artifact_removed is not None
        else 0
    )
    fold_artifact_removed_pixels = (
        int(areas[fold_artifact_removed].sum())
        if fold_artifact_removed is not None
        else 0
    )
    detached_fragment_removed_instances = (
        int(np.count_nonzero(detached_fragment_removed))
        if detached_fragment_removed is not None
        else 0
    )
    detached_fragment_removed_pixels = (
        int(areas[detached_fragment_removed].sum())
        if detached_fragment_removed is not None
        else 0
    )
    clustered_brown_fragment_removed_instances = (
        int(np.count_nonzero(clustered_brown_fragment_removed))
        if clustered_brown_fragment_removed is not None
        else 0
    )
    clustered_brown_fragment_removed_pixels = (
        int(areas[clustered_brown_fragment_removed].sum())
        if clustered_brown_fragment_removed is not None
        else 0
    )
    satellite_debris_removed_instances = (
        int(np.count_nonzero(satellite_debris_removed))
        if satellite_debris_removed is not None
        else 0
    )
    satellite_debris_removed_pixels = (
        int(areas[satellite_debris_removed].sum())
        if satellite_debris_removed is not None
        else 0
    )
    satellite_interior_protected_candidates = (
        satellite_debris_candidate & satellite_interior_protected
    )
    satellite_interior_protected_instances = int(
        np.count_nonzero(satellite_interior_protected_candidates)
    )
    satellite_interior_protected_pixels = int(
        areas[satellite_interior_protected_candidates].sum()
    )
    neutral_precipitate_removed_instances = (
        int(np.count_nonzero(neutral_precipitate_removed))
        if neutral_precipitate_removed is not None
        else 0
    )
    neutral_precipitate_removed_pixels = (
        int(areas[neutral_precipitate_removed].sum())
        if neutral_precipitate_removed is not None
        else 0
    )
    glass_artifact_removed_instances = (
        int(np.count_nonzero(glass_artifact_removed))
        if glass_artifact_removed is not None
        else 0
    )
    glass_artifact_removed_pixels = (
        int(areas[glass_artifact_removed].sum())
        if glass_artifact_removed is not None
        else 0
    )
    necrotic_artifact_removed_instances = (
        int(np.count_nonzero(necrotic_artifact_removed))
        if necrotic_artifact_removed is not None
        else 0
    )
    necrotic_artifact_removed_pixels = (
        int(areas[necrotic_artifact_removed].sum())
        if necrotic_artifact_removed is not None
        else 0
    )
    brown_debris_artifact_removed_instances = (
        int(np.count_nonzero(brown_debris_artifact_removed))
        if brown_debris_artifact_removed is not None
        else 0
    )
    brown_debris_artifact_removed_pixels = (
        int(areas[brown_debris_artifact_removed].sum())
        if brown_debris_artifact_removed is not None
        else 0
    )
    if (
        small_removed_instances
        or removed_instances
        or nuclear_removed_instances
        or context_removed_instances
        or adaptive_core_removed_instances
        or isolated_debris_removed_instances
        or brown_debris_artifact_removed_instances
    ):
        for y0 in range(0, height, block_size):
            y1 = min(height, y0 + block_size)
            for x0 in range(0, width, block_size):
                x1 = min(width, x0 + block_size)
                block = labels[y0:y1, x0:x1]
                block[~keep[np.asarray(block)]] = 0
    enclosed_cytoplasmic_qc = {
        "enclosed_cytoplasmic_child_instances_merged": 0,
        "enclosed_cytoplasmic_child_pixels_merged": 0,
    }
    if reconcile_enclosed_cytoplasmic_children:
        enclosed_cytoplasmic_qc = _merge_enclosed_cytoplasmic_child_instances(
            labels,
            areas,
            isolated_red_blue_sum,
            isolated_intensity_sum,
            block_size=4096,
            block_overlap=256,
            minimum_child_area=max(1, isolated_debris_glass_minimum_area // 2),
            maximum_child_area=max(1, isolated_debris_glass_minimum_area * 4),
            minimum_parent_area_ratio=2.0,
            minimum_dominant_contact_fraction=0.95,
            maximum_child_neighbors=2,
            minimum_parent_mean_red_blue_difference=45.0,
            maximum_parent_mean_intensity=190.0,
        )
    return {
        "post_constraint_small_instances_removed": small_removed_instances,
        "post_constraint_small_pixels_removed": small_removed_pixels,
        "flat_background_instances_removed": removed_instances,
        "flat_background_pixels_removed": removed_pixels,
        "nuclear_unsupported_instances_removed": nuclear_removed_instances,
        "nuclear_unsupported_pixels_removed": nuclear_removed_pixels,
        "source_context_unsupported_instances_removed": (context_removed_instances),
        "source_context_unsupported_pixels_removed": context_removed_pixels,
        "adaptive_nuclear_core_unsupported_instances_removed": (
            adaptive_core_removed_instances
        ),
        "adaptive_nuclear_core_unsupported_pixels_removed": (
            adaptive_core_removed_pixels
        ),
        "isolated_debris_unsupported_instances_removed": (
            isolated_debris_removed_instances
        ),
        "isolated_debris_unsupported_pixels_removed": (isolated_debris_removed_pixels),
        "micro_island_debris_instances_removed": (micro_island_removed_instances),
        "micro_island_debris_pixels_removed": micro_island_removed_pixels,
        "organized_tissue_escape_instances": organized_tissue_escape_instances,
        "organized_tissue_escape_pixels": organized_tissue_escape_pixels,
        "organized_necrotic_sparse_glass_instances_excluded": (
            organized_necrotic_sparse_glass_excluded_instances
        ),
        "organized_necrotic_sparse_glass_pixels_excluded": (
            organized_necrotic_sparse_glass_excluded_pixels
        ),
        "compact_unsupported_mosaic_instances_removed": (
            compact_unsupported_removed_instances
        ),
        "compact_unsupported_mosaic_pixels_removed": (
            compact_unsupported_removed_pixels
        ),
        "foam_mosaic_instances_removed": foam_unsupported_removed_instances,
        "foam_mosaic_pixels_removed": foam_unsupported_removed_pixels,
        "oversized_chromatic_artifact_instances_removed": (
            oversized_chromatic_removed_instances
        ),
        "oversized_chromatic_artifact_pixels_removed": (
            oversized_chromatic_removed_pixels
        ),
        "fold_artifact_instances_removed": fold_artifact_removed_instances,
        "fold_artifact_pixels_removed": fold_artifact_removed_pixels,
        "detached_fragment_instances_removed": detached_fragment_removed_instances,
        "detached_fragment_pixels_removed": detached_fragment_removed_pixels,
        "clustered_brown_fragment_instances_removed": (
            clustered_brown_fragment_removed_instances
        ),
        "clustered_brown_fragment_pixels_removed": (
            clustered_brown_fragment_removed_pixels
        ),
        "satellite_debris_instances_removed": satellite_debris_removed_instances,
        "satellite_debris_pixels_removed": satellite_debris_removed_pixels,
        "satellite_interior_instances_protected": (
            satellite_interior_protected_instances
        ),
        "satellite_interior_pixels_protected": satellite_interior_protected_pixels,
        "neutral_precipitate_instances_removed": (
            neutral_precipitate_removed_instances
        ),
        "neutral_precipitate_pixels_removed": neutral_precipitate_removed_pixels,
        "glass_artifact_instances_removed": glass_artifact_removed_instances,
        "glass_artifact_pixels_removed": glass_artifact_removed_pixels,
        "necrotic_artifact_instances_removed": (necrotic_artifact_removed_instances),
        "necrotic_artifact_pixels_removed": necrotic_artifact_removed_pixels,
        "brown_debris_artifact_instances_removed": (
            brown_debris_artifact_removed_instances
        ),
        "brown_debris_artifact_pixels_removed": (brown_debris_artifact_removed_pixels),
        **enclosed_cytoplasmic_qc,
        "instance_evidence": {
            "method": "local_optical_density",
            "background_percentile": _EVIDENCE_BACKGROUND_PERCENTILE,
            "minimum_optical_density": _EVIDENCE_MINIMUM_OPTICAL_DENSITY,
            "minimum_instance_fraction": _EVIDENCE_MINIMUM_INSTANCE_FRACTION,
            "minimum_area_pixels": minimum_area,
            "global_nuclear_support": require_nuclear_support,
            "nuclear_minimum_optical_density": nuclear_minimum_optical_density,
            "nuclear_minimum_pixels": nuclear_minimum_pixels,
            "nuclear_minimum_fraction": nuclear_minimum_fraction,
            "source_tissue_context": require_source_tissue_context,
            "source_tissue_context_bin_size_px": source_tissue_context_bin_size,
            "source_tissue_context_minimum_fraction": (
                source_tissue_context_minimum_fraction
            ),
            "source_tissue_context_minimum_instance_fraction": (
                source_tissue_context_minimum_instance_fraction
            ),
            "adaptive_nuclear_core": require_adaptive_nuclear_core,
            "adaptive_nuclear_core_percentile": adaptive_nuclear_core_percentile,
            "adaptive_nuclear_core_minimum_area_pixels": (
                adaptive_nuclear_core_minimum_area
            ),
            "adaptive_nuclear_core_minimum_area_um2": (
                adaptive_nuclear_core_minimum_area_um2
            ),
            "adaptive_nuclear_core_minimum_pixels": (
                adaptive_nuclear_core_minimum_pixels
            ),
            "adaptive_nuclear_core_minimum_fraction": (
                adaptive_nuclear_core_minimum_fraction
            ),
            "adaptive_nuclear_core_minimum_blue_ratio": (
                adaptive_nuclear_core_minimum_blue_ratio
            ),
            "adaptive_nuclear_core_threshold_quantiles": (
                np.quantile(adaptive_core_thresholds, [0.0, 0.5, 1.0]).tolist()
                if adaptive_core_thresholds
                else None
            ),
            "isolated_debris_gate": require_isolated_debris_gate,
            "isolated_debris_context_bin_size_px": (isolated_debris_context_bin_size),
            "isolated_debris_context_minimum_stain_fraction": (
                isolated_debris_context_minimum_stain_fraction
            ),
            "isolated_debris_context_minimum_instance_fraction": (
                isolated_debris_context_minimum_instance_fraction
            ),
            "isolated_debris_component_minimum_nuclear_pixels": (
                isolated_debris_component_minimum_nuclear_pixels
            ),
            "isolated_debris_allow_nuclear_escape": (
                isolated_debris_allow_nuclear_escape
            ),
            "isolated_debris_allow_organized_escape": (
                isolated_debris_allow_organized_escape
            ),
            "isolated_debris_nuclear_minimum_pixels": (
                isolated_debris_nuclear_minimum_pixels
            ),
            "isolated_debris_nuclear_minimum_fraction": (
                isolated_debris_nuclear_minimum_fraction
            ),
            "isolated_debris_nuclear_minimum_blue_ratio": (
                isolated_debris_nuclear_minimum_blue_ratio
            ),
            "isolated_debris_global_nuclear_escape_maximum_fraction": (
                isolated_debris_global_nuclear_escape_maximum_fraction
            ),
            "isolated_debris_neutral_dark_maximum_value": (
                isolated_debris_neutral_dark_maximum_value
            ),
            "isolated_debris_neutral_dark_maximum_chroma": (
                isolated_debris_neutral_dark_maximum_chroma
            ),
            "isolated_debris_neutral_dark_maximum_fraction": (
                isolated_debris_neutral_dark_maximum_fraction
            ),
            "isolated_debris_very_dark_maximum_value": (
                isolated_debris_very_dark_maximum_value
            ),
            "isolated_debris_very_dark_maximum_chroma": (
                isolated_debris_very_dark_maximum_chroma
            ),
            "isolated_debris_very_dark_maximum_fraction": (
                isolated_debris_very_dark_maximum_fraction
            ),
            "isolated_debris_micro_bin_size_px": (isolated_debris_micro_bin_size),
            "isolated_debris_micro_minimum_stain_fraction": (
                isolated_debris_micro_minimum_stain_fraction
            ),
            "isolated_debris_micro_maximum_component_bins": (
                isolated_debris_micro_maximum_component_bins
            ),
            "isolated_debris_micro_minimum_nuclear_fraction": (
                isolated_debris_micro_minimum_nuclear_fraction
            ),
            "isolated_debris_micro_minimum_instance_fraction": (
                isolated_debris_micro_minimum_instance_fraction
            ),
            "isolated_debris_organized_bin_size_px": (
                isolated_debris_organized_bin_size
            ),
            "isolated_debris_organized_window_size_px": (
                isolated_debris_organized_window_size
            ),
            "isolated_debris_organized_window_overlap_px": (
                isolated_debris_organized_window_overlap
            ),
            "isolated_debris_organized_minimum_bin_occupancy_fraction": (
                isolated_debris_organized_minimum_bin_occupancy_fraction
            ),
            "isolated_debris_organized_minimum_component_pixels": (
                isolated_debris_organized_minimum_component_pixels
            ),
            "isolated_debris_organized_compact_minimum_component_pixels": (
                isolated_debris_organized_compact_minimum_component_pixels
            ),
            "isolated_debris_organized_minimum_aspect_ratio": (
                isolated_debris_organized_minimum_aspect_ratio
            ),
            "isolated_debris_organized_minimum_mean_instance_pixels": (
                isolated_debris_organized_minimum_mean_instance_pixels
            ),
            "isolated_debris_organized_minimum_instance_fraction": (
                isolated_debris_organized_minimum_instance_fraction
            ),
            "isolated_debris_organized_strong_red_blue_difference": (
                isolated_debris_organized_strong_red_blue_difference
            ),
            "isolated_debris_organized_minimum_strong_chromatic_fraction": (
                isolated_debris_organized_minimum_strong_chromatic_fraction
            ),
            "isolated_debris_compact_unsupported_minimum_area_pixels": (
                isolated_debris_compact_unsupported_minimum_area
            ),
            "isolated_debris_compact_unsupported_minimum_area_um2": (
                isolated_debris_compact_unsupported_minimum_area_um2
            ),
            "isolated_debris_compact_unsupported_maximum_aspect_ratio": (
                isolated_debris_compact_unsupported_maximum_aspect_ratio
            ),
            "isolated_debris_compact_unsupported_maximum_mean_instance_pixels": (
                isolated_debris_compact_unsupported_maximum_mean_instance_pixels
            ),
            "isolated_debris_compact_unsupported_maximum_mean_red_blue_difference": (
                isolated_debris_compact_unsupported_maximum_mean_red_blue_difference
            ),
            "isolated_debris_compact_unsupported_strong_red_blue_difference": (
                isolated_debris_compact_unsupported_strong_red_blue_difference
            ),
            "isolated_debris_compact_unsupported_maximum_strong_chromatic_fraction": (
                isolated_debris_compact_unsupported_maximum_strong_chromatic_fraction
            ),
            "isolated_debris_compact_unsupported_elongation_ratio": (
                isolated_debris_compact_unsupported_elongation_ratio
            ),
            "isolated_debris_compact_unsupported_maximum_elongated_instance_fraction": (
                isolated_debris_compact_unsupported_maximum_elongated_instance_fraction
            ),
            "isolated_debris_compact_unsupported_maximum_context_nuclear_fraction": (
                isolated_debris_compact_unsupported_maximum_context_nuclear_fraction
            ),
            "isolated_debris_compact_unsupported_minimum_component_fill_fraction": (
                isolated_debris_compact_unsupported_minimum_component_fill_fraction
            ),
            "isolated_debris_compact_unsupported_minimum_instance_fraction": (
                isolated_debris_compact_unsupported_minimum_instance_fraction
            ),
            "isolated_debris_foam_minimum_mean_intensity": (
                isolated_debris_foam_minimum_mean_intensity
            ),
            "isolated_debris_foam_maximum_mean_instance_pixels": (
                isolated_debris_foam_maximum_mean_instance_pixels
            ),
            "isolated_debris_foam_maximum_strong_chromatic_fraction": (
                isolated_debris_foam_maximum_strong_chromatic_fraction
            ),
            "isolated_debris_foam_maximum_context_nuclear_fraction": (
                isolated_debris_foam_maximum_context_nuclear_fraction
            ),
            "isolated_debris_foam_maximum_elongated_instance_fraction": (
                isolated_debris_foam_maximum_elongated_instance_fraction
            ),
            "isolated_debris_foam_core_minimum_pixels": (
                isolated_debris_foam_core_minimum_pixels
            ),
            "isolated_debris_foam_core_minimum_bright_fraction": (
                isolated_debris_foam_core_minimum_bright_fraction
            ),
            "isolated_debris_foam_core_minimum_intensity": (
                isolated_debris_foam_core_minimum_intensity
            ),
            "isolated_debris_foam_core_maximum_instance_area_pixels": (
                isolated_debris_foam_core_maximum_instance_area
            ),
            "isolated_debris_foam_core_maximum_instance_area_um2": (
                isolated_debris_foam_core_maximum_instance_area_um2
            ),
            "isolated_debris_foam_core_minimum_component_area_pixels": (
                isolated_debris_foam_core_minimum_component_area
            ),
            "isolated_debris_foam_core_minimum_component_area_um2": (
                isolated_debris_foam_core_minimum_component_area_um2
            ),
            "isolated_debris_foam_core_maximum_aspect_ratio": (
                isolated_debris_foam_core_maximum_aspect_ratio
            ),
            "isolated_debris_oversized_chromatic_minimum_area_pixels": (
                isolated_debris_oversized_chromatic_minimum_area
            ),
            "isolated_debris_oversized_chromatic_minimum_area_um2": (
                isolated_debris_oversized_chromatic_minimum_area_um2
            ),
            "isolated_debris_oversized_chromatic_minimum_fraction": (
                isolated_debris_oversized_chromatic_minimum_fraction
            ),
            "isolated_debris_fold_minimum_instance_area_pixels": (
                isolated_debris_fold_minimum_instance_area
            ),
            "isolated_debris_fold_minimum_instance_area_um2": (
                isolated_debris_fold_minimum_instance_area_um2
            ),
            "isolated_debris_fold_minimum_component_area_pixels": (
                isolated_debris_fold_minimum_component_area
            ),
            "isolated_debris_fold_minimum_component_area_um2": (
                isolated_debris_fold_minimum_component_area_um2
            ),
            "isolated_debris_fold_maximum_mean_red_blue_difference": (
                isolated_debris_fold_maximum_mean_red_blue_difference
            ),
            "isolated_debris_fold_maximum_mean_intensity": (
                isolated_debris_fold_maximum_mean_intensity
            ),
            "isolated_debris_fold_dense_minimum_instances": (
                isolated_debris_fold_dense_minimum_instances
            ),
            "isolated_debris_fold_dense_minimum_aspect_ratio": (
                isolated_debris_fold_dense_minimum_aspect_ratio
            ),
            "isolated_debris_fold_dense_connectivity_dilation_bins": (
                isolated_debris_fold_dense_connectivity_dilation_bins
            ),
            "isolated_debris_fold_dense_maximum_mean_red_blue_difference": (
                isolated_debris_fold_dense_maximum_mean_red_blue_difference
            ),
            "isolated_debris_fold_dense_maximum_mean_intensity": (
                isolated_debris_fold_dense_maximum_mean_intensity
            ),
            "isolated_debris_detached_minimum_instance_area_pixels": (
                isolated_debris_detached_minimum_instance_area
            ),
            "isolated_debris_detached_minimum_instance_area_um2": (
                isolated_debris_detached_minimum_instance_area_um2
            ),
            "isolated_debris_detached_minimum_component_area_pixels": (
                isolated_debris_detached_minimum_component_area
            ),
            "isolated_debris_detached_minimum_component_area_um2": (
                isolated_debris_detached_minimum_component_area_um2
            ),
            "isolated_debris_detached_minimum_instances": (
                isolated_debris_detached_minimum_instances
            ),
            "isolated_debris_detached_minimum_aspect_ratio": (
                isolated_debris_detached_minimum_aspect_ratio
            ),
            "isolated_debris_detached_maximum_mean_red_blue_difference": (
                isolated_debris_detached_maximum_mean_red_blue_difference
            ),
            "isolated_debris_detached_maximum_mean_intensity": (
                isolated_debris_detached_maximum_mean_intensity
            ),
            "isolated_debris_detached_context_window_size_px": (
                isolated_debris_detached_context_window_size
            ),
            "isolated_debris_detached_maximum_prediction_fraction": (
                isolated_debris_detached_maximum_prediction_fraction
            ),
            "isolated_debris_detached_compact_minimum_instance_area_pixels": (
                isolated_debris_detached_compact_minimum_instance_area
            ),
            "isolated_debris_detached_compact_minimum_instance_area_um2": (
                isolated_debris_detached_compact_minimum_instance_area_um2
            ),
            "isolated_debris_detached_compact_minimum_component_area_pixels": (
                isolated_debris_detached_compact_minimum_component_area
            ),
            "isolated_debris_detached_compact_minimum_component_area_um2": (
                isolated_debris_detached_compact_minimum_component_area_um2
            ),
            "isolated_debris_detached_compact_minimum_instances": (
                isolated_debris_detached_compact_minimum_instances
            ),
            "isolated_debris_detached_compact_maximum_mean_red_blue_difference": (
                isolated_debris_detached_compact_maximum_mean_red_blue_difference
            ),
            "isolated_debris_detached_compact_maximum_mean_intensity": (
                isolated_debris_detached_compact_maximum_mean_intensity
            ),
            "isolated_debris_oversized_brown_gate": (
                isolated_debris_oversized_brown_gate
            ),
            "isolated_debris_oversized_brown_minimum_instance_area_pixels": (
                isolated_debris_oversized_chromatic_minimum_area
            ),
            "isolated_debris_oversized_brown_minimum_instance_area_um2": (
                isolated_debris_oversized_chromatic_minimum_area_um2
            ),
            "isolated_debris_oversized_brown_maximum_nuclear_fraction": 0.01,
            "isolated_debris_oversized_brown_minimum_red_green_difference": 5.0,
            "isolated_debris_oversized_brown_minimum_red_blue_difference": 20.0,
            "isolated_debris_oversized_brown_maximum_mean_intensity": (
                isolated_debris_oversized_brown_maximum_mean_intensity
            ),
            "isolated_debris_clustered_brown_gate": (
                isolated_debris_clustered_brown_gate
            ),
            "isolated_debris_clustered_brown_minimum_instance_area_pixels": (
                isolated_debris_clustered_brown_minimum_instance_area
            ),
            "isolated_debris_clustered_brown_minimum_instance_area_um2": (
                isolated_debris_clustered_brown_minimum_instance_area_um2
            ),
            "isolated_debris_clustered_brown_maximum_instance_area_pixels": (
                isolated_debris_clustered_brown_maximum_instance_area
            ),
            "isolated_debris_clustered_brown_maximum_instance_area_um2": (
                isolated_debris_clustered_brown_maximum_instance_area_um2
            ),
            "isolated_debris_clustered_brown_minimum_component_area_pixels": (
                isolated_debris_clustered_brown_minimum_component_area
            ),
            "isolated_debris_clustered_brown_minimum_component_area_um2": (
                isolated_debris_clustered_brown_minimum_component_area_um2
            ),
            "isolated_debris_clustered_brown_minimum_instances": 4,
            "isolated_debris_clustered_brown_connectivity_dilation_bins": 12,
            "isolated_debris_clustered_brown_maximum_nuclear_fraction": 0.02,
            "isolated_debris_clustered_brown_minimum_red_green_difference": 5.0,
            "isolated_debris_clustered_brown_minimum_red_blue_difference": 30.0,
            "isolated_debris_clustered_brown_maximum_mean_intensity": 175.0,
            "isolated_debris_clustered_brown_context_window_size_px": 256,
            "isolated_debris_clustered_brown_maximum_prediction_fraction": 0.50,
            "isolated_debris_clustered_brown_minimum_sparse_sample_fraction": 0.75,
            "isolated_debris_satellite_maximum_instance_area_pixels": (
                isolated_debris_satellite_maximum_instance_area
            ),
            "isolated_debris_satellite_maximum_instance_area_um2": (
                isolated_debris_satellite_maximum_instance_area_um2
            ),
            "isolated_debris_satellite_minimum_component_area_pixels": (
                isolated_debris_satellite_minimum_component_area
            ),
            "isolated_debris_satellite_minimum_component_area_um2": (
                isolated_debris_satellite_minimum_component_area_um2
            ),
            "isolated_debris_satellite_minimum_instances": 3,
            "isolated_debris_satellite_connectivity_dilation_bins": 8,
            "isolated_debris_satellite_maximum_nuclear_fraction": 0.05,
            "isolated_debris_satellite_minimum_mean_red_blue_difference": 5.0,
            "isolated_debris_satellite_maximum_mean_red_blue_difference": 70.0,
            "isolated_debris_satellite_minimum_mean_intensity": 90.0,
            "isolated_debris_satellite_maximum_mean_intensity": 220.0,
            "isolated_debris_satellite_context_window_size_px": 256,
            "isolated_debris_satellite_maximum_prediction_fraction": 0.35,
            "isolated_debris_satellite_gate_enabled": (isolated_debris_satellite_gate),
            **(
                {
                    "isolated_debris_satellite_minimum_mask_distance_um": (
                        isolated_debris_satellite_minimum_mask_distance_um
                    ),
                    "isolated_debris_satellite_minimum_sample_fraction": (
                        CELL_SATELLITE_PROTECTION_MINIMUM_SAMPLE_FRACTION
                    ),
                    "isolated_debris_satellite_mask_shape": list(
                        isolated_debris_satellite_tissue_mask.shape
                    ),
                    "isolated_debris_satellite_mpp_xy": list(
                        isolated_debris_satellite_mpp_xy
                    ),
                }
                if isolated_debris_satellite_minimum_mask_distance_um is not None
                and isolated_debris_satellite_tissue_mask is not None
                and isolated_debris_satellite_mpp_xy is not None
                else {}
            ),
            "isolated_debris_diffuse_degenerated_gate": (
                isolated_debris_diffuse_degenerated_gate
            ),
            "isolated_debris_neutral_precipitate_gate": (
                isolated_debris_neutral_precipitate_gate
            ),
            "isolated_debris_neutral_precipitate_minimum_instance_area_pixels": (
                isolated_debris_glass_minimum_area
            ),
            "isolated_debris_neutral_precipitate_maximum_instance_area_pixels": (
                isolated_debris_satellite_maximum_instance_area * 2
            ),
            "isolated_debris_neutral_precipitate_minimum_core_fraction": 0.28,
            "isolated_debris_neutral_precipitate_minimum_very_dark_fraction": 0.20,
            "isolated_debris_neutral_precipitate_maximum_nuclear_fraction": 0.35,
            "isolated_debris_neutral_precipitate_mean_red_blue_range": [
                -10.0,
                15.0,
            ],
            "isolated_debris_neutral_precipitate_maximum_mean_intensity": 135.0,
            "isolated_debris_neutral_precipitate_context_window_size_px": 256,
            "isolated_debris_neutral_precipitate_maximum_prediction_fraction": 0.25,
            "isolated_debris_neutral_precipitate_minimum_sparse_sample_fraction": (
                0.50
            ),
            "isolated_debris_glass_minimum_area_pixels": (
                isolated_debris_glass_minimum_area
            ),
            "isolated_debris_glass_minimum_area_um2": (
                isolated_debris_glass_minimum_area_um2
            ),
            "isolated_debris_glass_context_window_size_px": (
                isolated_debris_glass_context_window_size
            ),
            "isolated_debris_glass_maximum_prediction_fraction": (
                isolated_debris_glass_maximum_prediction_fraction
            ),
            "isolated_debris_glass_maximum_context_stain_fraction": (
                isolated_debris_glass_maximum_context_stain_fraction
            ),
            "isolated_debris_glass_low_stain_maximum_mean_red_blue_difference": (
                isolated_debris_glass_low_stain_maximum_mean_red_blue_difference
            ),
            "isolated_debris_glass_low_stain_minimum_mean_intensity": (
                isolated_debris_glass_low_stain_minimum_mean_intensity
            ),
            "isolated_debris_glass_minimum_context_fraction": (
                isolated_debris_glass_minimum_context_fraction
            ),
            "isolated_debris_glass_maximum_mean_red_blue_difference": (
                isolated_debris_glass_maximum_mean_red_blue_difference
            ),
            "isolated_debris_glass_minimum_mean_intensity": (
                isolated_debris_glass_minimum_mean_intensity
            ),
            "isolated_debris_self_dense_glass_gate": (
                isolated_debris_self_dense_glass_gate
            ),
            "isolated_debris_necrotic_minimum_component_area_pixels": (
                isolated_debris_necrotic_minimum_component_area
            ),
            "isolated_debris_necrotic_minimum_component_area_um2": (
                isolated_debris_necrotic_minimum_component_area_um2
            ),
            "isolated_debris_necrotic_maximum_aspect_ratio": (
                isolated_debris_necrotic_maximum_aspect_ratio
            ),
            "isolated_debris_necrotic_minimum_fill_fraction": (
                isolated_debris_necrotic_minimum_fill_fraction
            ),
            "isolated_debris_necrotic_maximum_fill_fraction": (
                isolated_debris_necrotic_maximum_fill_fraction
            ),
            "isolated_debris_necrotic_minimum_mean_red_blue_difference": (
                isolated_debris_necrotic_minimum_mean_red_blue_difference
            ),
            "isolated_debris_necrotic_maximum_mean_red_blue_difference": (
                isolated_debris_necrotic_maximum_mean_red_blue_difference
            ),
            "isolated_debris_necrotic_minimum_mean_intensity": (
                isolated_debris_necrotic_minimum_mean_intensity
            ),
            "isolated_debris_necrotic_minimum_instance_fraction": (
                isolated_debris_necrotic_minimum_instance_fraction
            ),
            "isolated_debris_protect_organized_from_necrotic": (
                isolated_debris_protect_organized_from_necrotic
            ),
            (
                "isolated_debris_organized_necrotic_protection_"
                "maximum_instance_area_pixels"
            ): (isolated_debris_organized_necrotic_protection_maximum_instance_area),
            "isolated_debris_organized_necrotic_protection_maximum_instance_area_um2": (
                isolated_debris_organized_necrotic_protection_maximum_instance_area_um2
            ),
            (
                "isolated_debris_organized_necrotic_protection_"
                "minimum_local_prediction_fraction"
            ): (organized_necrotic_protection_minimum_local_prediction_fraction),
            ("isolated_debris_organized_necrotic_protection_local_prediction_source"): (
                "preartifact_independent_nuclear_external"
                if organized_necrotic_protection_require_independent_nuclear_context
                else (
                    "preartifact_eligible_external"
                    if organized_necrotic_protection_use_eligible_context
                    else "raw"
                )
            ),
            "isolated_debris_organized_necrotic_sparse_glass_shape_gate": (
                organized_necrotic_sparse_glass_shape_gate
            ),
            "isolated_debris_organized_necrotic_sparse_glass_context_window_size_px": (
                _ORGANIZED_NECROTIC_SPARSE_GLASS_CONTEXT_WINDOW
            ),
            "isolated_debris_organized_necrotic_sparse_glass_maximum_mean_intensity": (
                _ORGANIZED_NECROTIC_SPARSE_GLASS_MAXIMUM_MEAN_INTENSITY
            ),
            "isolated_debris_organized_necrotic_sparse_glass_minimum_tissue_fraction": (
                _ORGANIZED_NECROTIC_SPARSE_GLASS_MINIMUM_TISSUE_FRACTION
            ),
            (
                "isolated_debris_organized_necrotic_sparse_glass_"
                "maximum_sampled_fill_fraction"
            ): (_ORGANIZED_NECROTIC_SPARSE_GLASS_MAXIMUM_SAMPLED_FILL_FRACTION),
            (
                "isolated_debris_organized_necrotic_sparse_glass_"
                "minimum_sampled_elongation"
            ): (_ORGANIZED_NECROTIC_SPARSE_GLASS_MINIMUM_SAMPLED_ELONGATION),
            "reconcile_enclosed_cytoplasmic_children": (
                reconcile_enclosed_cytoplasmic_children
            ),
            "isolated_debris_necrotic_window_size_px": (
                isolated_debris_organized_window_size
            ),
            "isolated_debris_necrotic_window_overlap_px": (
                isolated_debris_organized_window_overlap
            ),
            "isolated_debris_necrotic_context_window_size_px": 256,
            "isolated_debris_necrotic_maximum_local_prediction_fraction": 0.40,
            "isolated_debris_brown_minimum_component_area_pixels": (
                isolated_debris_brown_minimum_component_area
            ),
            "isolated_debris_brown_minimum_component_area_um2": (
                isolated_debris_brown_minimum_component_area_um2
            ),
            "isolated_debris_brown_maximum_aspect_ratio": (
                isolated_debris_brown_maximum_aspect_ratio
            ),
            "isolated_debris_brown_minimum_fill_fraction": (
                isolated_debris_brown_minimum_fill_fraction
            ),
            "isolated_debris_brown_maximum_fill_fraction": (
                isolated_debris_brown_maximum_fill_fraction
            ),
            "isolated_debris_brown_minimum_mean_red_blue_difference": (
                isolated_debris_brown_minimum_mean_red_blue_difference
            ),
            "isolated_debris_brown_maximum_mean_intensity": (
                isolated_debris_brown_maximum_mean_intensity
            ),
            "isolated_debris_brown_minimum_instance_fraction": (
                isolated_debris_brown_minimum_instance_fraction
            ),
            "isolated_debris_brown_requires_nuclear_unsupported": False,
            "isolated_debris_brown_expands_to_source_component": True,
            "isolated_debris_brown_window_size_px": (
                isolated_debris_organized_window_size
            ),
            "isolated_debris_brown_window_overlap_px": (
                isolated_debris_organized_window_overlap
            ),
            "isolated_debris_brown_context_window_size_px": 256,
            "isolated_debris_brown_maximum_local_prediction_fraction": 0.50,
            "isolated_debris_brown_maximum_source_component_area_pixels": (
                isolated_debris_brown_maximum_source_component_area
            ),
            "isolated_debris_brown_maximum_source_component_area_um2": (
                isolated_debris_brown_maximum_source_component_area_um2
            ),
            "isolated_debris_brown_source_maximum_mean_intensity": (
                isolated_debris_brown_source_maximum_mean_intensity
            ),
            "isolated_debris_brown_source_dilation_bins": (
                isolated_debris_brown_source_dilation_bins
            ),
            "block_size_px": block_size,
        },
    }


def _isolated_glass_artifact_instances(
    prediction_counts: np.ndarray,
    pixel_counts: np.ndarray,
    sampled_labels: np.ndarray,
    sampled_red_blue_difference: np.ndarray,
    sampled_intensity: np.ndarray,
    instance_areas: np.ndarray,
    *,
    instance_red_blue_sum: np.ndarray | None = None,
    instance_intensity_sum: np.ndarray | None = None,
    stain_counts: np.ndarray | None = None,
    stain_pixel_counts: np.ndarray | None = None,
    stain_bin_size: int = 64,
    maximum_context_stain_fraction: float = 0.20,
    low_stain_maximum_mean_red_blue_difference: float = 35.0,
    low_stain_minimum_mean_intensity: float = 120.0,
    bin_size: int,
    context_window_size: int,
    minimum_area: int,
    maximum_prediction_fraction: float,
    minimum_context_fraction: float,
    maximum_mean_red_blue_difference: float,
    minimum_mean_intensity: float,
) -> np.ndarray:
    """Reject sparse bright pale predictions caused by mask leakage or debris.

    Pale glass fibers and particulate brown debris can contain enough
    chromatic pixels to mimic nuclear support. True pale tissue generally
    forms a locally populated cell field; this gate therefore requires both
    the pale color and sparse prediction context before removing an instance.
    Exact native-pixel color sums avoid the center-sampling bias that is most
    severe for small fragmented objects.
    """

    if not (
        prediction_counts.shape
        == pixel_counts.shape
        == sampled_labels.shape
        == sampled_red_blue_difference.shape
        == sampled_intensity.shape
    ):
        raise ValueError("glass-artifact sample grids must have matching shapes")
    if sampled_labels.size and int(sampled_labels.max()) >= len(instance_areas):
        raise ValueError("glass-artifact sampled labels exceed instance lookup")
    sample_counts = np.bincount(
        sampled_labels.ravel(), minlength=len(instance_areas)
    ).astype(np.uint64)
    if (instance_red_blue_sum is None) != (instance_intensity_sum is None):
        raise ValueError("glass-artifact exact color sums must be provided together")
    if instance_red_blue_sum is not None and instance_intensity_sum is not None:
        if not (
            instance_red_blue_sum.shape
            == instance_intensity_sum.shape
            == instance_areas.shape
        ):
            raise ValueError("glass-artifact exact color sums must match areas")
        mean_red_blue = instance_red_blue_sum / np.maximum(instance_areas, 1)
        mean_intensity = instance_intensity_sum / np.maximum(instance_areas, 1)
    else:
        mean_red_blue = np.divide(
            np.bincount(
                sampled_labels.ravel(),
                weights=sampled_red_blue_difference.ravel(),
                minlength=len(instance_areas),
            ),
            np.maximum(sample_counts, 1),
        )
        mean_intensity = np.divide(
            np.bincount(
                sampled_labels.ravel(),
                weights=sampled_intensity.ravel(),
                minlength=len(instance_areas),
            ),
            np.maximum(sample_counts, 1),
        )
    from scipy import ndimage as ndi

    prediction_fraction = prediction_counts / np.maximum(pixel_counts, 1)
    local_prediction_fraction = ndi.uniform_filter(
        prediction_fraction.astype(np.float32),
        size=context_window_size // bin_size,
        mode="nearest",
    )
    sparse_context = local_prediction_fraction < maximum_prediction_fraction
    low_stain_context = np.zeros_like(sparse_context)
    if (stain_counts is None) != (stain_pixel_counts is None):
        raise ValueError("glass-artifact stain grids must be provided together")
    if stain_counts is not None and stain_pixel_counts is not None:
        if stain_counts.shape != stain_pixel_counts.shape:
            raise ValueError("glass-artifact stain grids must have matching shapes")
        if context_window_size % stain_bin_size:
            raise ValueError("glass-artifact context must divide the stain grid")
        stain_fraction = stain_counts / np.maximum(stain_pixel_counts, 1)
        local_stain_fraction = ndi.uniform_filter(
            stain_fraction.astype(np.float32),
            size=context_window_size // stain_bin_size,
            mode="nearest",
        )
        stain_rows = np.minimum(
            local_stain_fraction.shape[0] - 1,
            np.arange(sampled_labels.shape[0]) * bin_size // stain_bin_size,
        )
        stain_columns = np.minimum(
            local_stain_fraction.shape[1] - 1,
            np.arange(sampled_labels.shape[1]) * bin_size // stain_bin_size,
        )
        low_stain_context = (
            local_stain_fraction[np.ix_(stain_rows, stain_columns)]
            < maximum_context_stain_fraction
        )
    sparse_counts = np.bincount(
        sampled_labels[sparse_context].ravel(), minlength=len(instance_areas)
    ).astype(np.uint64)
    low_stain_counts = np.bincount(
        sampled_labels[low_stain_context].ravel(), minlength=len(instance_areas)
    ).astype(np.uint64)
    sparse_pale = (
        (sparse_counts >= np.ceil(sample_counts * minimum_context_fraction))
        & (mean_red_blue <= maximum_mean_red_blue_difference)
        & (mean_intensity >= minimum_mean_intensity)
    )
    low_stain_muted = (
        (low_stain_counts >= np.ceil(sample_counts * minimum_context_fraction))
        & (mean_red_blue <= low_stain_maximum_mean_red_blue_difference)
        & (mean_intensity >= low_stain_minimum_mean_intensity)
    )
    artifact = (
        (instance_areas >= minimum_area)
        & (sample_counts > 0)
        & (sparse_pale | low_stain_muted)
    )
    artifact[0] = False
    return artifact


def _self_dense_glass_artifact_instances(
    prediction_counts: np.ndarray,
    pixel_counts: np.ndarray,
    sampled_labels: np.ndarray,
    instance_areas: np.ndarray,
    instance_red_blue_sum: np.ndarray,
    instance_intensity_sum: np.ndarray,
    *,
    bin_size: int,
    minimum_instance_area: int,
    minimum_mean_red_blue_difference: float,
    maximum_mean_red_blue_difference: float,
    minimum_mean_intensity: float,
    maximum_aspect_ratio: float,
    minimum_sampled_fill_fraction: float,
    minimum_sampled_area_fraction: float,
    ring_dilation_bins: int,
    maximum_ring_prediction_fraction: float,
) -> np.ndarray:
    """Reject a large pale pseudo-cell that makes its own context look dense.

    A defocused scanner spot can be predicted as one compact object. Its area
    then dominates the ordinary local-density estimate and shields it from the
    sparse-glass gate. This branch requires a pale, nearly neutral, compact
    sampled footprint surrounded by genuinely sparse predictions. Large cells
    embedded in tissue therefore retain their surrounding cellular context.
    """

    if not (prediction_counts.shape == pixel_counts.shape == sampled_labels.shape):
        raise ValueError("self-dense glass grids must have matching shapes")
    if not (
        instance_areas.shape
        == instance_red_blue_sum.shape
        == instance_intensity_sum.shape
    ):
        raise ValueError("self-dense glass instance lookups must match")
    if sampled_labels.size and int(sampled_labels.max()) >= len(instance_areas):
        raise ValueError("self-dense glass labels exceed instance lookups")
    if bin_size <= 0 or minimum_instance_area <= 0 or ring_dilation_bins <= 0:
        raise ValueError("self-dense glass geometry must be positive")
    if not (
        minimum_mean_red_blue_difference < maximum_mean_red_blue_difference
        and 0 < minimum_mean_intensity <= 255
        and maximum_aspect_ratio >= 1
        and 0 < minimum_sampled_fill_fraction <= 1
        and 0 < minimum_sampled_area_fraction <= 1
        and 0 <= maximum_ring_prediction_fraction < 1
    ):
        raise ValueError("self-dense glass thresholds are invalid")

    mean_red_blue = instance_red_blue_sum / np.maximum(instance_areas, 1)
    mean_intensity = instance_intensity_sum / np.maximum(instance_areas, 1)
    candidates = (
        (instance_areas >= minimum_instance_area)
        & (mean_red_blue >= minimum_mean_red_blue_difference)
        & (mean_red_blue <= maximum_mean_red_blue_difference)
        & (mean_intensity >= minimum_mean_intensity)
    )
    candidates[0] = False
    candidate_grid = candidates[sampled_labels] & (sampled_labels > 0)
    artifact = np.zeros_like(candidates, dtype=bool)
    if not np.any(candidate_grid):
        return artifact

    rows, columns = np.nonzero(candidate_grid)
    labels = sampled_labels[rows, columns]
    order = np.argsort(labels, kind="stable")
    rows = rows[order]
    columns = columns[order]
    labels = labels[order]
    starts = np.r_[0, np.flatnonzero(np.diff(labels)) + 1]
    stops = np.r_[starts[1:], len(labels)]
    prediction_fraction = prediction_counts / np.maximum(pixel_counts, 1)

    from scipy import ndimage as ndi

    for start, stop in zip(starts, stops, strict=True):
        instance = int(labels[start])
        instance_rows = rows[start:stop]
        instance_columns = columns[start:stop]
        y0 = int(instance_rows.min())
        y1 = int(instance_rows.max()) + 1
        x0 = int(instance_columns.min())
        x1 = int(instance_columns.max()) + 1
        height = y1 - y0
        width = x1 - x0
        aspect_ratio = max(width / max(height, 1), height / max(width, 1))
        sampled_fill_fraction = len(instance_rows) / max(width * height, 1)
        sampled_area_fraction = (
            len(instance_rows)
            * bin_size
            * bin_size
            / max(int(instance_areas[instance]), 1)
        )
        if (
            aspect_ratio > maximum_aspect_ratio
            or sampled_fill_fraction < minimum_sampled_fill_fraction
            or sampled_area_fraction < minimum_sampled_area_fraction
        ):
            continue
        ey0 = max(0, y0 - ring_dilation_bins)
        ey1 = min(sampled_labels.shape[0], y1 + ring_dilation_bins)
        ex0 = max(0, x0 - ring_dilation_bins)
        ex1 = min(sampled_labels.shape[1], x1 + ring_dilation_bins)
        local_instance = sampled_labels[ey0:ey1, ex0:ex1] == instance
        ring = ndi.binary_dilation(
            local_instance,
            structure=np.ones((3, 3), dtype=np.uint8),
            iterations=ring_dilation_bins,
        ) & (~local_instance)
        if not np.any(ring):
            continue
        ring_prediction = float(prediction_fraction[ey0:ey1, ex0:ex1][ring].mean())
        if ring_prediction <= maximum_ring_prediction_fraction:
            artifact[instance] = True
    artifact[0] = False
    return artifact


def _merge_enclosed_cytoplasmic_child_instances(
    labels: np.ndarray,
    instance_areas: np.ndarray,
    instance_red_blue_sum: np.ndarray,
    instance_intensity_sum: np.ndarray,
    *,
    block_size: int,
    block_overlap: int,
    minimum_child_area: int,
    maximum_child_area: int,
    minimum_parent_area_ratio: float,
    minimum_dominant_contact_fraction: float,
    maximum_child_neighbors: int,
    minimum_parent_mean_red_blue_difference: float,
    maximum_parent_mean_intensity: float,
) -> dict[str, int]:
    """Merge a nested pseudo-cell into its strongly stained parent instance.

    Cytoplasmic DAB can make one biological cell appear as a large brown ring
    plus a second nuclear-sized instance wholly inside it.  A real planar cell
    normally contacts background or several neighbouring cells; therefore a
    child with no background contact and nearly all of its interface against a
    single much larger parent is a conservative double-counting signature.
    Requiring the parent itself to be strongly red-over-blue restricts this
    reconciliation to the IHC failure mode instead of ordinary H&E mosaics.

    Blocks overlap so a compact child touching one processing boundary is
    evaluated completely in a neighbouring block.  Conflicting nominations
    are discarded, and area-ratio ordering prevents merge cycles.
    """

    if labels.ndim != 2:
        raise ValueError("enclosed-child labels must be two-dimensional")
    if not (
        instance_areas.shape
        == instance_red_blue_sum.shape
        == instance_intensity_sum.shape
    ):
        raise ValueError("enclosed-child instance lookups must match")
    if (
        block_size <= 0
        or block_overlap < 0
        or block_overlap >= block_size
        or minimum_child_area <= 0
        or maximum_child_area < minimum_child_area
        or minimum_parent_area_ratio <= 1
        or not 0 < minimum_dominant_contact_fraction <= 1
        or maximum_child_neighbors <= 0
        or not np.isfinite(minimum_parent_mean_red_blue_difference)
        or not 0 < maximum_parent_mean_intensity <= 255
    ):
        raise ValueError("enclosed-child thresholds are invalid")

    maximum = len(instance_areas) - 1
    base = np.uint64(maximum + 1)
    parent_lookup = np.zeros(maximum + 1, dtype=np.uint32)
    ambiguous = np.zeros(maximum + 1, dtype=bool)
    mean_red_blue = instance_red_blue_sum / np.maximum(instance_areas, 1)
    mean_intensity = instance_intensity_sum / np.maximum(instance_areas, 1)
    height, width = labels.shape

    for y0 in tile_starts(height, tile_size=block_size, overlap=block_overlap):
        y1 = min(y0 + block_size, height)
        for x0 in tile_starts(width, tile_size=block_size, overlap=block_overlap):
            x1 = min(x0 + block_size, width)
            block = np.asarray(labels[y0:y1, x0:x1])
            if block.size == 0 or not np.any(block):
                continue
            if int(block.max()) >= len(instance_areas):
                raise ValueError("enclosed-child labels exceed instance lookups")
            boundary_ids = np.unique(
                np.concatenate((block[0], block[-1], block[:, 0], block[:, -1]))
            )
            directed_codes: list[np.ndarray] = []
            background_ids: list[np.ndarray] = []
            for first, second in (
                (block[:, :-1], block[:, 1:]),
                (block[:-1, :], block[1:, :]),
            ):
                different = first != second
                forward = different & (first > 0)
                reverse = different & (second > 0)
                if np.any(forward & (second == 0)):
                    background_ids.append(first[forward & (second == 0)])
                if np.any(reverse & (first == 0)):
                    background_ids.append(second[reverse & (first == 0)])
                nonzero_forward = forward & (second > 0)
                nonzero_reverse = reverse & (first > 0)
                if np.any(nonzero_forward):
                    directed_codes.append(
                        first[nonzero_forward].astype(np.uint64) * base
                        + second[nonzero_forward].astype(np.uint64)
                    )
                if np.any(nonzero_reverse):
                    directed_codes.append(
                        second[nonzero_reverse].astype(np.uint64) * base
                        + first[nonzero_reverse].astype(np.uint64)
                    )
            if not directed_codes:
                continue
            codes, contact_counts = np.unique(
                np.concatenate(directed_codes), return_counts=True
            )
            sources = (codes // base).astype(np.int64)
            neighbours = (codes % base).astype(np.int64)
            unique_sources, starts = np.unique(sources, return_index=True)
            stops = np.r_[starts[1:], len(sources)]
            touches_background = (
                np.unique(np.concatenate(background_ids))
                if background_ids
                else np.empty(0, dtype=block.dtype)
            )
            boundary_lookup = np.zeros(maximum + 1, dtype=bool)
            boundary_lookup[boundary_ids[boundary_ids <= maximum]] = True
            background_lookup = np.zeros(maximum + 1, dtype=bool)
            background_lookup[touches_background[touches_background <= maximum]] = True
            for child, start, stop in zip(unique_sources, starts, stops, strict=True):
                child = int(child)
                if (
                    child <= 0
                    or boundary_lookup[child]
                    or background_lookup[child]
                    or ambiguous[child]
                    or not minimum_child_area
                    <= int(instance_areas[child])
                    <= maximum_child_area
                ):
                    continue
                local_counts = contact_counts[start:stop]
                if len(local_counts) > maximum_child_neighbors:
                    continue
                dominant_index = int(np.argmax(local_counts))
                dominant_count = int(local_counts[dominant_index])
                total_count = int(local_counts.sum())
                if dominant_count < minimum_dominant_contact_fraction * total_count:
                    continue
                parent = int(neighbours[start + dominant_index])
                if (
                    parent <= 0
                    or instance_areas[parent]
                    < instance_areas[child] * minimum_parent_area_ratio
                    or mean_red_blue[parent] < minimum_parent_mean_red_blue_difference
                    or mean_intensity[parent] > maximum_parent_mean_intensity
                ):
                    continue
                previous = int(parent_lookup[child])
                if previous and previous != parent:
                    parent_lookup[child] = 0
                    ambiguous[child] = True
                else:
                    parent_lookup[child] = parent

    children = np.flatnonzero(parent_lookup)
    if children.size == 0:
        return {
            "enclosed_cytoplasmic_child_instances_merged": 0,
            "enclosed_cytoplasmic_child_pixels_merged": 0,
        }
    for child in children:
        root = int(parent_lookup[child])
        while parent_lookup[root]:
            root = int(parent_lookup[root])
        parent_lookup[child] = root
    replacement = np.arange(maximum + 1, dtype=np.uint32)
    replacement[children] = parent_lookup[children]
    for y0 in range(0, height, block_size):
        y1 = min(y0 + block_size, height)
        for x0 in range(0, width, block_size):
            x1 = min(x0 + block_size, width)
            block = labels[y0:y1, x0:x1]
            block[...] = replacement[np.asarray(block)]
    return {
        "enclosed_cytoplasmic_child_instances_merged": int(children.size),
        "enclosed_cytoplasmic_child_pixels_merged": int(instance_areas[children].sum()),
    }


def _sparse_anuclear_satellite_instances(
    prediction_counts: np.ndarray,
    pixel_counts: np.ndarray,
    sampled_labels: np.ndarray,
    instance_areas: np.ndarray,
    nuclear_supported_pixels: np.ndarray,
    instance_red_blue_sum: np.ndarray,
    instance_intensity_sum: np.ndarray,
    *,
    bin_size: int,
    maximum_instance_area: int,
    minimum_component_area: int,
    minimum_instances: int = 3,
    connectivity_dilation_bins: int = 8,
    maximum_nuclear_fraction: float = 0.05,
    minimum_mean_red_blue_difference: float = 5.0,
    maximum_mean_red_blue_difference: float = 70.0,
    minimum_mean_intensity: float = 90.0,
    maximum_mean_intensity: float = 220.0,
    context_window_size: int = 256,
    maximum_prediction_fraction: float = 0.35,
) -> np.ndarray:
    """Reject clusters of small anuclear satellites around detached tissue.

    Mid-intensity brown or gray fragments can evade both the dark-brown debris
    gate and the pale-glass gate.  A fragment is removed here only when several
    physically small, weakly nuclear, muted instances form a nearby cluster in
    sparse prediction context.  The joint topology and context requirements
    preserve isolated viable cells, nucleated detached tissue islands, and
    organized epithelial or stromal fields.
    """

    if not (prediction_counts.shape == pixel_counts.shape == sampled_labels.shape):
        raise ValueError("satellite-debris sample grids must have matching shapes")
    if not (
        instance_areas.shape
        == nuclear_supported_pixels.shape
        == instance_red_blue_sum.shape
        == instance_intensity_sum.shape
    ):
        raise ValueError("satellite-debris instance lookups must match")
    if sampled_labels.size and int(sampled_labels.max()) >= len(instance_areas):
        raise ValueError("satellite-debris sampled labels exceed instance lookup")
    if bin_size <= 0 or context_window_size <= 0 or context_window_size % bin_size:
        raise ValueError("satellite-debris context geometry is invalid")
    if minimum_instances < 2 or connectivity_dilation_bins <= 0:
        raise ValueError("satellite-debris component topology is invalid")
    if maximum_instance_area <= 0 or minimum_component_area <= 0:
        raise ValueError("satellite-debris physical areas are invalid")
    if not 0 <= maximum_nuclear_fraction < 1:
        raise ValueError("satellite-debris nuclear fraction is invalid")
    if not (
        minimum_mean_red_blue_difference < maximum_mean_red_blue_difference
        and 0 < minimum_mean_intensity < maximum_mean_intensity <= 255
        and 0 < maximum_prediction_fraction <= 1
    ):
        raise ValueError("satellite-debris color or context thresholds are invalid")

    mean_red_blue = instance_red_blue_sum / np.maximum(instance_areas, 1)
    mean_intensity = instance_intensity_sum / np.maximum(instance_areas, 1)
    nuclear_fraction = nuclear_supported_pixels / np.maximum(instance_areas, 1)
    candidate = (
        (instance_areas > 0)
        & (instance_areas <= maximum_instance_area)
        & (nuclear_fraction <= maximum_nuclear_fraction)
        & (mean_red_blue >= minimum_mean_red_blue_difference)
        & (mean_red_blue <= maximum_mean_red_blue_difference)
        & (mean_intensity >= minimum_mean_intensity)
        & (mean_intensity <= maximum_mean_intensity)
    )
    candidate[0] = False

    from scipy import ndimage as ndi

    prediction_fraction = prediction_counts / np.maximum(pixel_counts, 1)
    local_prediction_fraction = ndi.uniform_filter(
        prediction_fraction.astype(np.float32),
        size=context_window_size // bin_size,
        mode="nearest",
    )
    # Mask dense tissue before connectivity. Otherwise a broad stain range can
    # bridge true tissue into a satellite component and make the entire mixed
    # component appear dense, shielding the detached fragments around it.
    candidate_grid = (
        candidate[sampled_labels]
        & (sampled_labels > 0)
        & (local_prediction_fraction <= maximum_prediction_fraction)
    )
    connected = ndi.binary_dilation(
        candidate_grid,
        structure=np.ones((3, 3), dtype=np.uint8),
        iterations=connectivity_dilation_bins,
    )
    components, _ = ndi.label(connected, structure=np.ones((3, 3), dtype=np.uint8))
    artifact = np.zeros(len(instance_areas), dtype=bool)
    for component_index, component_slice in enumerate(
        ndi.find_objects(components), start=1
    ):
        if component_slice is None:
            continue
        component_candidates = (
            components[component_slice] == component_index
        ) & candidate_grid[component_slice]
        ids = np.unique(sampled_labels[component_slice][component_candidates])
        ids = ids[ids > 0]
        if (
            ids.size < minimum_instances
            or int(instance_areas[ids].sum()) < minimum_component_area
        ):
            continue
        context = local_prediction_fraction[component_slice][component_candidates]
        if not context.size or float(np.mean(context)) > maximum_prediction_fraction:
            continue
        artifact[ids] = True
    artifact[0] = False
    return artifact


def _interior_tissue_instance_protection(
    sampled_labels: np.ndarray,
    tissue_mask: np.ndarray,
    instance_count: int,
    *,
    content_shape: tuple[int, int],
    bin_size: int,
    mpp_xy: tuple[float, float],
    minimum_mask_distance_um: float,
    minimum_sample_fraction: float = 0.50,
) -> np.ndarray:
    """Protect labels whose sampled footprint lies well inside accepted tissue.

    The accepted registration mask is intentionally evaluated at its native
    thumbnail resolution, with physical sampling supplied to the distance
    transform. A majority rule avoids rescuing cells that merely touch an
    interior region while remaining adjacent to glass, a lumen, or debris.
    """

    labels = np.asarray(sampled_labels)
    mask = np.asarray(tissue_mask, dtype=bool)
    if labels.ndim != 2 or mask.ndim != 2 or not mask.size:
        raise ValueError("satellite interior-protection masks must be 2D")
    if not isinstance(instance_count, int) or instance_count <= 0:
        raise ValueError("satellite interior-protection instance count is invalid")
    if labels.size and int(labels.max()) >= instance_count:
        raise ValueError("satellite interior-protection labels exceed lookup")
    height, width = content_shape
    if height <= 0 or width <= 0 or bin_size <= 0:
        raise ValueError("satellite interior-protection geometry is invalid")
    if len(mpp_xy) != 2 or any(value <= 0 for value in mpp_xy):
        raise ValueError("satellite interior-protection MPP is invalid")
    if minimum_mask_distance_um <= 0 or not np.isfinite(minimum_mask_distance_um):
        raise ValueError("satellite interior-protection distance is invalid")
    if not 0 < minimum_sample_fraction <= 1:
        raise ValueError("satellite interior-protection fraction is invalid")

    from scipy import ndimage as ndi

    distance_um = ndi.distance_transform_edt(
        mask,
        sampling=(
            height * mpp_xy[1] / mask.shape[0],
            width * mpp_xy[0] / mask.shape[1],
        ),
    )
    native_rows = np.minimum(
        np.arange(labels.shape[0], dtype=np.int64) * bin_size + bin_size // 2,
        height - 1,
    )
    native_columns = np.minimum(
        np.arange(labels.shape[1], dtype=np.int64) * bin_size + bin_size // 2,
        width - 1,
    )
    mask_rows = np.floor(
        (native_rows.astype(np.float64) + 0.5) * mask.shape[0] / height
    ).astype(np.int64)
    mask_columns = np.floor(
        (native_columns.astype(np.float64) + 0.5) * mask.shape[1] / width
    ).astype(np.int64)
    np.clip(mask_rows, 0, mask.shape[0] - 1, out=mask_rows)
    np.clip(mask_columns, 0, mask.shape[1] - 1, out=mask_columns)
    interior = distance_um[np.ix_(mask_rows, mask_columns)] >= minimum_mask_distance_um
    sample_counts = np.bincount(labels.ravel(), minlength=instance_count)
    interior_counts = np.bincount(labels[interior].ravel(), minlength=instance_count)
    protected = (sample_counts > 0) & (
        interior_counts / np.maximum(sample_counts, 1) >= minimum_sample_fraction
    )
    protected[0] = False
    return protected


def _sparse_neutral_dark_precipitate_instances(
    prediction_counts: np.ndarray,
    pixel_counts: np.ndarray,
    sampled_labels: np.ndarray,
    instance_areas: np.ndarray,
    nuclear_supported_pixels: np.ndarray,
    neutral_dark_pixels: np.ndarray,
    very_dark_pixels: np.ndarray,
    instance_red_blue_sum: np.ndarray,
    instance_intensity_sum: np.ndarray,
    *,
    bin_size: int,
    minimum_instance_area: int,
    maximum_instance_area: int,
    minimum_neutral_dark_fraction: float,
    minimum_very_dark_fraction: float,
    maximum_nuclear_fraction: float,
    minimum_mean_red_blue_difference: float,
    maximum_mean_red_blue_difference: float,
    maximum_mean_intensity: float,
    context_window_size: int,
    maximum_prediction_fraction: float,
    minimum_sparse_sample_fraction: float,
) -> np.ndarray:
    """Reject neutral-black precipitate pseudo-cells in sparse tissue context.

    Black chromogen precipitate can be segmented as a cell-sized object with a
    pale halo.  Its dark core is therefore too small for the existing 50%
    whole-instance neutral-dark rule.  Extremely dark precipitate can also
    produce a false hematoxylin response, so nuclear support is only a broad
    ceiling here; neutral and very-dark core fractions, near-neutral mean
    colour, low mean intensity, bounded area, and a sparse 256-pixel prediction
    neighbourhood must all agree.  Dense tissue, chromatic blue nuclei, brown
    target-positive cells, and large folds remain outside the admitted failure
    morphology.
    """

    grids = (prediction_counts, pixel_counts, sampled_labels)
    lookups = (
        instance_areas,
        nuclear_supported_pixels,
        neutral_dark_pixels,
        very_dark_pixels,
        instance_red_blue_sum,
        instance_intensity_sum,
    )
    if len({grid.shape for grid in grids}) != 1:
        raise ValueError("neutral-precipitate sample grids must have matching shapes")
    if (
        any(array.ndim != 1 for array in lookups)
        or len({array.shape for array in lookups}) != 1
    ):
        raise ValueError("neutral-precipitate instance lookups must match")
    if sampled_labels.size and int(sampled_labels.max()) >= len(instance_areas):
        raise ValueError("neutral-precipitate labels exceed instance lookups")
    if (
        bin_size <= 0
        or context_window_size <= 0
        or context_window_size % bin_size
        or minimum_instance_area <= 0
        or maximum_instance_area < minimum_instance_area
    ):
        raise ValueError("neutral-precipitate geometry must be positive")
    if not (
        0 < minimum_neutral_dark_fraction <= 1
        and 0 < minimum_very_dark_fraction <= minimum_neutral_dark_fraction
        and 0 <= maximum_nuclear_fraction < 1
        and minimum_mean_red_blue_difference < maximum_mean_red_blue_difference
        and 0 < maximum_mean_intensity <= 255
        and 0 < maximum_prediction_fraction <= 1
        and 0 < minimum_sparse_sample_fraction <= 1
    ):
        raise ValueError("neutral-precipitate thresholds are invalid")

    denominator = np.maximum(instance_areas, 1)
    candidate = (
        (instance_areas >= minimum_instance_area)
        & (instance_areas <= maximum_instance_area)
        & (
            neutral_dark_pixels
            >= np.ceil(instance_areas * minimum_neutral_dark_fraction).astype(np.uint64)
        )
        & (
            very_dark_pixels
            >= np.ceil(instance_areas * minimum_very_dark_fraction).astype(np.uint64)
        )
        & (nuclear_supported_pixels <= instance_areas * maximum_nuclear_fraction)
        & (instance_red_blue_sum / denominator >= minimum_mean_red_blue_difference)
        & (instance_red_blue_sum / denominator <= maximum_mean_red_blue_difference)
        & (instance_intensity_sum / denominator <= maximum_mean_intensity)
    )
    candidate[0] = False
    if not np.any(candidate):
        return candidate

    from scipy import ndimage as ndi

    prediction_fraction = prediction_counts / np.maximum(pixel_counts, 1)
    local_prediction_fraction = ndi.uniform_filter(
        prediction_fraction.astype(np.float32),
        size=context_window_size // bin_size,
        mode="nearest",
    )
    sample_counts = np.bincount(
        sampled_labels.ravel(), minlength=len(instance_areas)
    ).astype(np.uint64)
    sparse_counts = np.bincount(
        sampled_labels[
            local_prediction_fraction <= maximum_prediction_fraction
        ].ravel(),
        minlength=len(instance_areas),
    ).astype(np.uint64)
    artifact = (
        candidate
        & (sample_counts > 0)
        & (
            sparse_counts
            >= np.ceil(sample_counts * minimum_sparse_sample_fraction).astype(np.uint64)
        )
    )
    artifact[0] = False
    return artifact


def _clustered_magenta_anuclear_instances(
    sampled_labels: np.ndarray,
    instance_areas: np.ndarray,
    nuclear_supported_pixels: np.ndarray,
    instance_red_sum: np.ndarray,
    instance_green_sum: np.ndarray,
    instance_blue_sum: np.ndarray,
    *,
    maximum_instance_area: int,
    minimum_component_area: int,
    minimum_instances: int = 8,
    connectivity_dilation_bins: int = 2,
    maximum_nuclear_fraction: float = 0.05,
    minimum_red_green_difference: float = 20.0,
    minimum_red_blue_difference: float = -20.0,
    minimum_blue_green_difference: float = 10.0,
) -> np.ndarray:
    """Identify dense islands of small magenta anucleate objects.

    PAS-positive erythrocytes and similarly colored particulate debris can be
    segmented into highly regular cell-sized objects.  Their dense packing
    intentionally defeats sparse-context debris rules, while their magenta
    color can falsely resemble hematoxylin when only a blue-ratio test is
    used.  This gate therefore requires all of the following: a physically
    small instance, weak nuclear evidence, magenta rather than blue or DAB
    channel ordering, and a multi-instance connected component.  The joint
    requirements preserve blue lymphocyte nuclei, brown DAB-positive cells,
    solitary viable cells, and large PAS-positive tissue structures.
    """

    lookups = (
        instance_areas,
        nuclear_supported_pixels,
        instance_red_sum,
        instance_green_sum,
        instance_blue_sum,
    )
    if any(array.ndim != 1 for array in lookups):
        raise ValueError("magenta-anuclear instance lookups must be one-dimensional")
    if len({array.shape for array in lookups}) != 1:
        raise ValueError("magenta-anuclear instance lookups must match")
    if sampled_labels.ndim != 2:
        raise ValueError("magenta-anuclear sampled labels must be two-dimensional")
    if sampled_labels.size and int(sampled_labels.max()) >= len(instance_areas):
        raise ValueError("magenta-anuclear sampled labels exceed instance lookup")
    if maximum_instance_area <= 0 or minimum_component_area <= 0:
        raise ValueError("magenta-anuclear physical areas must be positive")
    if minimum_instances < 2 or connectivity_dilation_bins <= 0:
        raise ValueError("magenta-anuclear component topology is invalid")
    if not 0 <= maximum_nuclear_fraction < 1:
        raise ValueError("magenta-anuclear nuclear fraction is invalid")
    if not all(
        np.isfinite(value)
        for value in (
            minimum_red_green_difference,
            minimum_red_blue_difference,
            minimum_blue_green_difference,
        )
    ):
        raise ValueError("magenta-anuclear color thresholds must be finite")

    denominator = np.maximum(instance_areas, 1)
    mean_red = instance_red_sum / denominator
    mean_green = instance_green_sum / denominator
    mean_blue = instance_blue_sum / denominator
    nuclear_fraction = nuclear_supported_pixels / denominator
    candidate = (
        (instance_areas > 0)
        & (instance_areas <= maximum_instance_area)
        & (nuclear_fraction <= maximum_nuclear_fraction)
        & ((mean_red - mean_green) >= minimum_red_green_difference)
        & ((mean_red - mean_blue) >= minimum_red_blue_difference)
        & ((mean_blue - mean_green) >= minimum_blue_green_difference)
    )
    candidate[0] = False

    from scipy import ndimage as ndi

    candidate_grid = candidate[sampled_labels] & (sampled_labels > 0)
    connected = ndi.binary_dilation(
        candidate_grid,
        structure=np.ones((3, 3), dtype=np.uint8),
        iterations=connectivity_dilation_bins,
    )
    components, _ = ndi.label(connected, structure=np.ones((3, 3), dtype=np.uint8))
    artifact = np.zeros(len(instance_areas), dtype=bool)
    for component_index, component_slice in enumerate(
        ndi.find_objects(components), start=1
    ):
        if component_slice is None:
            continue
        component_candidates = (
            components[component_slice] == component_index
        ) & candidate_grid[component_slice]
        ids = np.unique(sampled_labels[component_slice][component_candidates])
        ids = ids[ids > 0]
        if (
            ids.size >= minimum_instances
            and int(instance_areas[ids].sum()) >= minimum_component_area
        ):
            artifact[ids] = True
    artifact[0] = False
    return artifact


def _oversized_brown_anuclear_instances(
    instance_areas: np.ndarray,
    nuclear_supported_pixels: np.ndarray,
    instance_red_sum: np.ndarray,
    instance_green_sum: np.ndarray,
    instance_blue_sum: np.ndarray,
    *,
    minimum_instance_area: int,
    maximum_nuclear_fraction: float = 0.01,
    minimum_red_green_difference: float = 5.0,
    minimum_red_blue_difference: float = 20.0,
    maximum_mean_intensity: float = 180.0,
) -> np.ndarray:
    """Identify impossible single-cell labels over large brown fragments.

    A detached DAB-positive fold or tissue fragment can occasionally be traced
    as one label spanning thousands of square microns.  Such a label is not a
    valid cell even when the fragment itself is biological tissue.  The gate
    is deliberately narrow: the instance must be physically oversized,
    brown-channel ordered, non-pale, and almost devoid of blue nuclear support.
    Ordinary DAB-positive cells, nuclear-rich tissue, neutral glass/background,
    and hematoxylin-rich structures therefore remain protected.
    """

    lookups = (
        instance_areas,
        nuclear_supported_pixels,
        instance_red_sum,
        instance_green_sum,
        instance_blue_sum,
    )
    if any(array.ndim != 1 for array in lookups):
        raise ValueError("oversized-brown instance lookups must be one-dimensional")
    if len({array.shape for array in lookups}) != 1:
        raise ValueError("oversized-brown instance lookups must match")
    if minimum_instance_area <= 0:
        raise ValueError("oversized-brown minimum area must be positive")
    if not 0 <= maximum_nuclear_fraction < 1:
        raise ValueError("oversized-brown nuclear fraction is invalid")
    if (
        not all(
            np.isfinite(value)
            for value in (
                minimum_red_green_difference,
                minimum_red_blue_difference,
                maximum_mean_intensity,
            )
        )
        or not 0 < maximum_mean_intensity <= 255
    ):
        raise ValueError("oversized-brown color thresholds are invalid")

    denominator = np.maximum(instance_areas, 1)
    mean_red = instance_red_sum / denominator
    mean_green = instance_green_sum / denominator
    mean_blue = instance_blue_sum / denominator
    mean_intensity = (mean_red + mean_green + mean_blue) / 3.0
    nuclear_fraction = nuclear_supported_pixels / denominator
    artifact = (
        (instance_areas >= minimum_instance_area)
        & (nuclear_fraction <= maximum_nuclear_fraction)
        & ((mean_red - mean_green) >= minimum_red_green_difference)
        & ((mean_red - mean_blue) >= minimum_red_blue_difference)
        & (mean_intensity <= maximum_mean_intensity)
    )
    artifact[0] = False
    return artifact


def _clustered_brown_anuclear_instances(
    prediction_counts: np.ndarray,
    pixel_counts: np.ndarray,
    sampled_labels: np.ndarray,
    instance_areas: np.ndarray,
    nuclear_supported_pixels: np.ndarray,
    instance_red_sum: np.ndarray,
    instance_green_sum: np.ndarray,
    instance_blue_sum: np.ndarray,
    *,
    bin_size: int,
    minimum_instance_area: int,
    maximum_instance_area: int,
    minimum_component_area: int,
    minimum_instances: int = 4,
    connectivity_dilation_bins: int = 12,
    maximum_nuclear_fraction: float = 0.02,
    minimum_red_green_difference: float = 5.0,
    minimum_red_blue_difference: float = 30.0,
    maximum_mean_intensity: float = 175.0,
    context_window_size: int = 256,
    maximum_prediction_fraction: float = 0.50,
    minimum_sparse_sample_fraction: float = 0.75,
) -> np.ndarray:
    """Identify clustered, cell-like brown labels inside sparse luminal context.

    A blurred DAB-positive fragment can be divided into several plausible-size
    polygons even though it contains no resolvable nuclei.  Single-instance
    size filters cannot remove that failure safely.  This gate therefore
    requires a *cluster* of physically enlarged, brown-channel-ordered,
    anuclear labels, with most sampled pixels lying in a locally sparse
    prediction field.  Dense viable DAB-positive tissue, isolated large cells,
    hematoxylin-supported cells, pale glass, and blue tissue each fail at least
    one independent guard.
    """

    grids = (prediction_counts, pixel_counts, sampled_labels)
    lookups = (
        instance_areas,
        nuclear_supported_pixels,
        instance_red_sum,
        instance_green_sum,
        instance_blue_sum,
    )
    if len({grid.shape for grid in grids}) != 1 or any(
        grid.ndim != 2 for grid in grids
    ):
        raise ValueError("clustered-brown sample grids must be matching 2D arrays")
    if (
        any(array.ndim != 1 for array in lookups)
        or len({array.shape for array in lookups}) != 1
    ):
        raise ValueError("clustered-brown instance lookups must match")
    if sampled_labels.size and int(sampled_labels.max()) >= len(instance_areas):
        raise ValueError("clustered-brown labels exceed instance lookups")
    if (
        bin_size <= 0
        or minimum_instance_area <= 0
        or maximum_instance_area < minimum_instance_area
        or minimum_component_area < minimum_instance_area
        or minimum_instances < 2
        or connectivity_dilation_bins <= 0
        or context_window_size <= 0
        or context_window_size % bin_size
    ):
        raise ValueError("clustered-brown geometry must be positive and ordered")
    if not (
        0 <= maximum_nuclear_fraction < 1
        and 0 < maximum_mean_intensity <= 255
        and 0 < maximum_prediction_fraction <= 1
        and 0 < minimum_sparse_sample_fraction <= 1
    ):
        raise ValueError("clustered-brown evidence thresholds are invalid")
    if not all(
        np.isfinite(value)
        for value in (
            minimum_red_green_difference,
            minimum_red_blue_difference,
            maximum_mean_intensity,
        )
    ):
        raise ValueError("clustered-brown color thresholds must be finite")

    denominator = np.maximum(instance_areas, 1)
    mean_red = instance_red_sum / denominator
    mean_green = instance_green_sum / denominator
    mean_blue = instance_blue_sum / denominator
    mean_intensity = (mean_red + mean_green + mean_blue) / 3.0
    candidate = (
        (instance_areas >= minimum_instance_area)
        & (instance_areas <= maximum_instance_area)
        & (nuclear_supported_pixels <= instance_areas * maximum_nuclear_fraction)
        & ((mean_red - mean_green) >= minimum_red_green_difference)
        & ((mean_red - mean_blue) >= minimum_red_blue_difference)
        & (mean_intensity <= maximum_mean_intensity)
    )
    candidate[0] = False
    if not np.any(candidate):
        return candidate

    from scipy import ndimage as ndi

    prediction_fraction = prediction_counts / np.maximum(pixel_counts, 1)
    local_prediction_fraction = ndi.uniform_filter(
        prediction_fraction.astype(np.float32),
        size=context_window_size // bin_size,
        mode="nearest",
    )
    candidate_grid = candidate[sampled_labels] & (sampled_labels > 0)
    sample_counts = np.bincount(
        sampled_labels[candidate_grid].ravel(), minlength=len(instance_areas)
    ).astype(np.uint64)
    sparse_counts = np.bincount(
        sampled_labels[
            candidate_grid & (local_prediction_fraction <= maximum_prediction_fraction)
        ].ravel(),
        minlength=len(instance_areas),
    ).astype(np.uint64)
    sparse_candidate = (
        candidate
        & (sample_counts > 0)
        & (
            sparse_counts
            >= np.ceil(sample_counts * minimum_sparse_sample_fraction).astype(np.uint64)
        )
    )
    sparse_candidate[0] = False
    if not np.any(sparse_candidate):
        return sparse_candidate

    sparse_grid = sparse_candidate[sampled_labels] & (sampled_labels > 0)
    connected = ndi.binary_dilation(
        sparse_grid,
        structure=np.ones((3, 3), dtype=np.uint8),
        iterations=connectivity_dilation_bins,
    )
    components, _ = ndi.label(connected, structure=np.ones((3, 3), dtype=np.uint8))
    artifact = np.zeros(len(instance_areas), dtype=bool)
    for component_index, component_slice in enumerate(
        ndi.find_objects(components), start=1
    ):
        if component_slice is None:
            continue
        component_candidates = (
            components[component_slice] == component_index
        ) & sparse_grid[component_slice]
        ids = np.unique(sampled_labels[component_slice][component_candidates])
        ids = ids[ids > 0]
        if (
            ids.size >= minimum_instances
            and int(instance_areas[ids].sum()) >= minimum_component_area
        ):
            artifact[ids] = True
    artifact[0] = False
    return artifact


def _detached_oversized_fragment_instances(
    prediction_counts: np.ndarray,
    pixel_counts: np.ndarray,
    sampled_labels: np.ndarray,
    instance_areas: np.ndarray,
    instance_red_blue_sum: np.ndarray,
    instance_intensity_sum: np.ndarray,
    *,
    bin_size: int,
    minimum_instance_area: int,
    minimum_component_area: int,
    minimum_instances: int,
    minimum_aspect_ratio: float,
    maximum_mean_red_blue_difference: float,
    maximum_mean_intensity: float,
    context_window_size: int,
    maximum_prediction_fraction: float,
    compact_minimum_instance_area: int = 1200,
    compact_minimum_component_area: int = 3000,
    compact_minimum_instances: int = 2,
    compact_maximum_mean_red_blue_difference: float = 0.0,
    compact_maximum_mean_intensity: float = 180.0,
) -> np.ndarray:
    """Reject sparse elongated clusters split from one detached fragment.

    The rule operates on existing native-space CPSAM labels and exact color
    sums; it never creates or recomputes cells. Multiple oversized muted
    instances must touch on the native-aligned sample grid, form an elongated
    aggregate, and lie in sparse prediction context. These joint constraints
    protect solitary large cells, dense epithelial ribbons, and viable tissue.
    """

    if not (prediction_counts.shape == pixel_counts.shape == sampled_labels.shape):
        raise ValueError("detached-fragment sample grids must have matching shapes")
    if not (
        instance_areas.shape
        == instance_red_blue_sum.shape
        == instance_intensity_sum.shape
    ):
        raise ValueError("detached-fragment instance lookups must match")
    if sampled_labels.size and int(sampled_labels.max()) >= len(instance_areas):
        raise ValueError("detached-fragment sampled labels exceed instance lookup")

    mean_red_blue = instance_red_blue_sum / np.maximum(instance_areas, 1)
    mean_intensity = instance_intensity_sum / np.maximum(instance_areas, 1)
    muted = (mean_red_blue <= maximum_mean_red_blue_difference) & (
        mean_intensity <= maximum_mean_intensity
    )
    muted[0] = False
    from scipy import ndimage as ndi

    prediction_fraction = prediction_counts / np.maximum(pixel_counts, 1)
    local_prediction_fraction = ndi.uniform_filter(
        prediction_fraction.astype(np.float32),
        size=context_window_size // bin_size,
        mode="nearest",
    )
    artifact = np.zeros(len(instance_areas), dtype=bool)

    def qualify(
        candidate: np.ndarray,
        *,
        required_instances: int,
        required_area: int,
        required_aspect_ratio: float | None,
    ) -> None:
        candidate_grid = candidate[sampled_labels] & (sampled_labels > 0)
        connected_grid = ndi.binary_dilation(
            candidate_grid,
            structure=np.ones((3, 3), dtype=np.uint8),
        )
        components, _ = ndi.label(
            connected_grid,
            structure=np.ones((3, 3), dtype=np.uint8),
        )
        for component_index, component_slice in enumerate(
            ndi.find_objects(components), start=1
        ):
            if component_slice is None:
                continue
            component_candidates = (
                components[component_slice] == component_index
            ) & candidate_grid[component_slice]
            ids = np.unique(sampled_labels[component_slice][component_candidates])
            ids = ids[ids > 0]
            if (
                ids.size < required_instances
                or instance_areas[ids].sum() < required_area
            ):
                continue
            if required_aspect_ratio is not None:
                component_rows, component_columns = np.nonzero(component_candidates)
                component_height = int(component_rows.max() - component_rows.min() + 1)
                component_width = int(
                    component_columns.max() - component_columns.min() + 1
                )
                aspect_ratio = max(
                    component_width / max(component_height, 1),
                    component_height / max(component_width, 1),
                )
                if aspect_ratio < required_aspect_ratio:
                    continue
            context = local_prediction_fraction[component_slice][component_candidates]
            if (
                not context.size
                or float(np.mean(context)) > maximum_prediction_fraction
            ):
                continue
            # Once a cluster qualifies, also remove immediately touching muted
            # fragments that the model split below the oversized threshold.
            # They are part of the same detached object but cannot establish
            # the topology by themselves.
            expanded_slice = tuple(
                slice(max(0, axis.start - 1), min(limit, axis.stop + 1))
                for axis, limit in zip(
                    component_slice, sampled_labels.shape, strict=True
                )
            )
            component_nearby = ndi.binary_dilation(
                components[expanded_slice] == component_index,
                structure=np.ones((3, 3), dtype=np.uint8),
            )
            source_ids = np.unique(
                sampled_labels[expanded_slice][
                    component_nearby & muted[sampled_labels[expanded_slice]]
                ]
            )
            artifact[source_ids[source_ids > 0]] = True

    elongated_candidate = (instance_areas >= minimum_instance_area) & muted
    elongated_candidate[0] = False
    qualify(
        elongated_candidate,
        required_instances=minimum_instances,
        required_area=minimum_component_area,
        required_aspect_ratio=minimum_aspect_ratio,
    )
    compact_candidate = (
        (instance_areas >= compact_minimum_instance_area)
        & (mean_red_blue <= compact_maximum_mean_red_blue_difference)
        & (mean_intensity <= compact_maximum_mean_intensity)
    )
    compact_candidate[0] = False
    qualify(
        compact_candidate,
        required_instances=compact_minimum_instances,
        required_area=compact_minimum_component_area,
        required_aspect_ratio=None,
    )
    return artifact


def _fold_artifact_instances(
    sampled_labels: np.ndarray,
    sampled_red_blue_difference: np.ndarray,
    sampled_intensity: np.ndarray,
    nuclear_supported_pixels: np.ndarray,
    instance_areas: np.ndarray,
    *,
    minimum_instance_area: int,
    minimum_component_area: int,
    maximum_nuclear_fraction: float,
    maximum_mean_red_blue_difference: float,
    maximum_mean_intensity: float,
    dense_minimum_instances: int = 3,
    dense_minimum_aspect_ratio: float = 3.0,
    dense_connectivity_dilation_bins: int = 8,
    dense_maximum_mean_red_blue_difference: float = 60.0,
    dense_maximum_mean_intensity: float = 115.0,
    aneuclear_removal_eligible: np.ndarray | None = None,
) -> np.ndarray:
    """Identify clustered oversized pseudo-cells in crushed or folded tissue.

    Fold artifacts are required to be physically oversized, effectively
    aneuclear, muted rather than strongly red/brown, and optically dense.  The
    component-area requirement protects isolated large cells while retaining
    a single exceptionally large false compartment.  Color is sampled on the
    same native-aligned grid used by the topology gates; quantification and
    source labels are never recomputed.
    """

    if not (
        sampled_labels.shape
        == sampled_red_blue_difference.shape
        == sampled_intensity.shape
    ):
        raise ValueError("fold-artifact sample grids must have matching shapes")
    if nuclear_supported_pixels.shape != instance_areas.shape:
        raise ValueError("fold-artifact instance lookups must have matching shapes")
    if (
        aneuclear_removal_eligible is not None
        and aneuclear_removal_eligible.shape != instance_areas.shape
    ):
        raise ValueError("fold-artifact eligibility lookup must match instances")
    if sampled_labels.size and int(sampled_labels.max()) >= len(instance_areas):
        raise ValueError("fold-artifact sampled labels exceed instance lookup")
    sample_counts = np.bincount(
        sampled_labels.ravel(), minlength=len(instance_areas)
    ).astype(np.uint64)
    red_blue_sum = np.bincount(
        sampled_labels.ravel(),
        weights=sampled_red_blue_difference.ravel(),
        minlength=len(instance_areas),
    )
    intensity_sum = np.bincount(
        sampled_labels.ravel(),
        weights=sampled_intensity.ravel(),
        minlength=len(instance_areas),
    )
    mean_red_blue = np.divide(
        red_blue_sum,
        np.maximum(sample_counts, 1),
    )
    mean_intensity = np.divide(
        intensity_sum,
        np.maximum(sample_counts, 1),
    )
    aneuclear_candidate = (
        (instance_areas >= minimum_instance_area)
        & (sample_counts > 0)
        & (
            nuclear_supported_pixels
            < np.maximum(2, np.ceil(instance_areas * maximum_nuclear_fraction))
        )
        & (mean_red_blue <= maximum_mean_red_blue_difference)
        & (mean_intensity <= maximum_mean_intensity)
    )
    aneuclear_candidate[0] = False
    dense_candidate = (
        (instance_areas >= minimum_instance_area)
        & (sample_counts > 0)
        & (mean_red_blue <= dense_maximum_mean_red_blue_difference)
        & (mean_intensity <= dense_maximum_mean_intensity)
    )
    dense_candidate[0] = False

    from scipy import ndimage as ndi

    def mark_clustered(
        candidate: np.ndarray,
        artifact: np.ndarray,
        *,
        minimum_instances: int,
        minimum_aspect_ratio: float | None,
        connectivity_dilation_bins: int,
    ) -> None:
        candidate_grid = candidate[sampled_labels]
        connected_grid = ndi.binary_dilation(
            candidate_grid,
            structure=np.ones((3, 3), dtype=np.uint8),
            iterations=connectivity_dilation_bins,
        )
        components, _ = ndi.label(
            connected_grid,
            structure=np.ones((3, 3), dtype=np.uint8),
        )
        for component_index, component_slice in enumerate(
            ndi.find_objects(components), start=1
        ):
            if component_slice is None:
                continue
            component_candidate = (
                components[component_slice] == component_index
            ) & candidate_grid[component_slice]
            ids = np.unique(sampled_labels[component_slice][component_candidate])
            ids = ids[ids > 0]
            if (
                ids.size < minimum_instances
                or instance_areas[ids].sum() < minimum_component_area
            ):
                continue
            if minimum_aspect_ratio is not None:

                def aspect_ratio(selection: np.ndarray) -> float:
                    rows, columns = np.nonzero(selection)
                    component_height = int(rows.max() - rows.min() + 1)
                    component_width = int(columns.max() - columns.min() + 1)
                    return max(
                        component_width / max(component_height, 1),
                        component_height / max(component_width, 1),
                    )

                if aspect_ratio(component_candidate) < minimum_aspect_ratio:
                    # A nearby oversized outlier can join an otherwise narrow
                    # crushed band through the bounded dilation. Evaluate only
                    # minimum-size subsets for small components and mark the
                    # elongated subset, never the compact outlier.
                    if ids.size <= minimum_instances or ids.size > 12:
                        continue
                    from itertools import combinations

                    selected = np.zeros(len(instance_areas), dtype=bool)
                    local_labels = sampled_labels[component_slice]
                    for subset_tuple in combinations(ids.tolist(), minimum_instances):
                        subset = np.asarray(subset_tuple, dtype=ids.dtype)
                        if instance_areas[subset].sum() < minimum_component_area:
                            continue
                        subset_pixels = component_candidate & np.isin(
                            local_labels, subset
                        )
                        if aspect_ratio(subset_pixels) >= minimum_aspect_ratio:
                            selected[subset] = True
                    if not np.any(selected):
                        continue
                    artifact[selected] = True
                    continue
            artifact[ids] = True

    aneuclear_artifact = np.zeros(len(instance_areas), dtype=bool)
    mark_clustered(
        aneuclear_candidate,
        aneuclear_artifact,
        minimum_instances=1,
        minimum_aspect_ratio=None,
        connectivity_dilation_bins=1,
    )
    if aneuclear_removal_eligible is not None:
        aneuclear_artifact &= aneuclear_removal_eligible
    dense_artifact = np.zeros(len(instance_areas), dtype=bool)
    mark_clustered(
        dense_candidate,
        dense_artifact,
        minimum_instances=dense_minimum_instances,
        minimum_aspect_ratio=dense_minimum_aspect_ratio,
        connectivity_dilation_bins=dense_connectivity_dilation_bins,
    )
    return aneuclear_artifact | dense_artifact


def _bounded_organized_necrotic_protection(
    organized_instances: np.ndarray,
    instance_areas: np.ndarray,
    *,
    maximum_instance_area: int | None,
) -> np.ndarray:
    """Bound organized-necrotic rescue to biologically cell-sized objects.

    The unbounded form is retained for immutable v76 reproduction. Newer
    profiles supply a physical-area-derived pixel ceiling so a large detached
    fold or processing loop cannot inherit a cell-level exemption.
    """

    if (
        organized_instances.ndim != 1
        or instance_areas.ndim != 1
        or organized_instances.shape != instance_areas.shape
    ):
        raise ValueError("organized protection arrays must be matching vectors")
    if maximum_instance_area is not None and maximum_instance_area <= 0:
        raise ValueError("organized protection maximum instance area must be positive")
    protected = np.asarray(organized_instances, dtype=bool).copy()
    if maximum_instance_area is not None:
        protected &= instance_areas <= maximum_instance_area
    if protected.size:
        protected[0] = False
    return protected


def _sparse_glass_organized_necrotic_instances(
    sampled_labels: np.ndarray,
    sampled_intensity: np.ndarray,
    organized_instances: np.ndarray,
    *,
    bin_size: int,
    context_window_size: int,
    maximum_mean_intensity: float,
    minimum_tissue_fraction: float,
    maximum_sampled_fill_fraction: float,
    minimum_sampled_elongation: float,
) -> np.ndarray:
    """Find thin shape-only rescues isolated over mostly glass.

    A small folded processing strand can have enough local organization to
    resemble an epithelial edge.  This gate is deliberately conjunctive: the
    candidate must be sparsely filled, elongated, and surrounded by too little
    substantive source tissue.  Compact isolated cells and elongated cells
    embedded at a real tissue edge therefore remain protected.
    """

    if (
        sampled_labels.ndim != 2
        or sampled_intensity.shape != sampled_labels.shape
        or organized_instances.ndim != 1
    ):
        raise ValueError("sparse-glass organized-necrotic arrays are invalid")
    if sampled_labels.size and int(sampled_labels.max()) >= len(organized_instances):
        raise ValueError("sparse-glass sampled labels exceed instance lookup")
    if bin_size <= 0 or context_window_size <= 0 or context_window_size % bin_size:
        raise ValueError("sparse-glass context geometry is invalid")
    if not (
        0 < maximum_mean_intensity <= 255
        and 0 < minimum_tissue_fraction <= 1
        and 0 < maximum_sampled_fill_fraction <= 1
        and minimum_sampled_elongation > 1
    ):
        raise ValueError("sparse-glass thresholds are invalid")

    artifact = np.zeros_like(organized_instances, dtype=bool)
    candidate_grid = (sampled_labels > 0) & organized_instances[sampled_labels]
    if not np.any(candidate_grid):
        return artifact
    rows, columns = np.nonzero(candidate_grid)
    instance_labels = sampled_labels[rows, columns]
    order = np.argsort(instance_labels, kind="stable")
    rows = rows[order]
    columns = columns[order]
    instance_labels = instance_labels[order]
    starts = np.r_[0, np.flatnonzero(np.diff(instance_labels)) + 1]
    stops = np.r_[starts[1:], len(instance_labels)]
    window_bins = context_window_size // bin_size
    substantive_tissue = sampled_intensity < maximum_mean_intensity

    for start, stop in zip(starts, stops, strict=True):
        if stop - start < 3:
            continue
        instance = int(instance_labels[start])
        instance_rows = rows[start:stop]
        instance_columns = columns[start:stop]
        height = int(instance_rows.max() - instance_rows.min() + 1)
        width = int(instance_columns.max() - instance_columns.min() + 1)
        sampled_fill_fraction = len(instance_rows) / max(height * width, 1)
        if sampled_fill_fraction >= maximum_sampled_fill_fraction:
            continue

        coordinates = np.column_stack(
            (instance_rows * bin_size, instance_columns * bin_size)
        ).astype(np.float64)
        covariance = np.cov(coordinates, rowvar=False)
        eigenvalues = np.linalg.eigvalsh(covariance)
        sampled_elongation = math.sqrt(
            (max(float(eigenvalues[-1]), 0.0) + 1.0)
            / (max(float(eigenvalues[0]), 0.0) + 1.0)
        )
        if sampled_elongation <= minimum_sampled_elongation:
            continue

        center_y = int((instance_rows.min() + instance_rows.max()) // 2)
        center_x = int((instance_columns.min() + instance_columns.max()) // 2)
        y0 = max(
            0,
            min(sampled_labels.shape[0] - window_bins, center_y - window_bins // 2),
        )
        x0 = max(
            0,
            min(sampled_labels.shape[1] - window_bins, center_x - window_bins // 2),
        )
        y1 = min(sampled_labels.shape[0], y0 + window_bins)
        x1 = min(sampled_labels.shape[1], x0 + window_bins)
        local_tissue_fraction = float(substantive_tissue[y0:y1, x0:x1].mean())
        if local_tissue_fraction < minimum_tissue_fraction:
            artifact[instance] = True
    if artifact.size:
        artifact[0] = False
    return artifact


def _locally_supported_organized_necrotic_protection(
    labels: np.ndarray,
    organized_instances: np.ndarray,
    instance_areas: np.ndarray,
    prediction_counts: np.ndarray,
    pixel_counts: np.ndarray,
    *,
    maximum: int,
    bin_size: int,
    window_size: int,
    minimum_local_prediction_fraction: float,
    minimum_instance_fraction: float,
    exclude_instance_from_local_prediction: bool = False,
    context_instance_areas: np.ndarray | None = None,
    block_size: int,
) -> np.ndarray:
    """Keep organized rescues only inside a minimally cellular neighborhood."""

    if labels.ndim != 2:
        raise ValueError("organized protection labels must be two-dimensional")
    if organized_instances.shape != (maximum + 1,) or instance_areas.shape != (
        maximum + 1,
    ):
        raise ValueError("organized protection instance vectors are invalid")
    if prediction_counts.shape != pixel_counts.shape:
        raise ValueError("organized protection context grids must match")
    expected_shape = (
        (labels.shape[0] + bin_size - 1) // bin_size,
        (labels.shape[1] + bin_size - 1) // bin_size,
    )
    if prediction_counts.shape != expected_shape:
        raise ValueError("organized protection context grid geometry is invalid")
    if bin_size <= 0 or window_size <= 0 or window_size % bin_size or block_size <= 0:
        raise ValueError("organized protection context geometry is invalid")
    if not 0 < minimum_local_prediction_fraction <= 1:
        raise ValueError("organized protection local prediction fraction is invalid")
    if not 0 < minimum_instance_fraction <= 1:
        raise ValueError("organized protection instance fraction is invalid")
    if not isinstance(exclude_instance_from_local_prediction, bool):
        raise TypeError("organized protection self-exclusion flag must be boolean")
    if context_instance_areas is not None and context_instance_areas.shape != (
        maximum + 1,
    ):
        raise ValueError("organized protection context-instance areas are invalid")
    if (
        context_instance_areas is not None
        and not exclude_instance_from_local_prediction
    ):
        raise ValueError(
            "organized protection context-instance areas require self-exclusion"
        )

    from scipy import ndimage as ndi

    window_bins = window_size // bin_size
    if exclude_instance_from_local_prediction:
        window_area = float(window_bins * window_bins)
        local_prediction_counts = (
            ndi.uniform_filter(
                prediction_counts.astype(np.float32),
                size=window_bins,
                mode="constant",
                cval=0.0,
            )
            * window_area
        )
        local_pixel_counts = (
            ndi.uniform_filter(
                pixel_counts.astype(np.float32),
                size=window_bins,
                mode="constant",
                cval=0.0,
            )
            * window_area
        )
    else:
        prediction_fraction = prediction_counts / np.maximum(pixel_counts, 1)
        local_prediction_fraction = ndi.uniform_filter(
            prediction_fraction.astype(np.float32),
            size=window_bins,
            mode="nearest",
        )
        supported_grid = local_prediction_fraction >= minimum_local_prediction_fraction
    supported = np.zeros(maximum + 1, dtype=np.uint64)
    height, width = labels.shape
    for y0 in range(0, height, block_size):
        y1 = min(height, y0 + block_size)
        rows = np.arange(y0, y1) // bin_size
        for x0 in range(0, width, block_size):
            x1 = min(width, x0 + block_size)
            columns = np.arange(x0, x1) // bin_size
            block = np.asarray(labels[y0:y1, x0:x1])
            if exclude_instance_from_local_prediction:
                local_counts = local_prediction_counts[np.ix_(rows, columns)]
                local_pixels = local_pixel_counts[np.ix_(rows, columns)]
                external_counts = np.maximum(
                    local_counts
                    - (
                        instance_areas[block]
                        if context_instance_areas is None
                        else context_instance_areas[block]
                    ),
                    0.0,
                )
                context = (
                    external_counts / np.maximum(local_pixels, 1.0)
                    >= minimum_local_prediction_fraction
                )
            else:
                context = supported_grid[np.ix_(rows, columns)]
            counts = np.bincount(block[context].ravel(), minlength=maximum + 1)
            supported += counts[: maximum + 1].astype(np.uint64)
    required = np.ceil(instance_areas * minimum_instance_fraction).astype(np.uint64)
    protected = np.asarray(organized_instances, dtype=bool) & (supported >= required)
    if protected.size:
        protected[0] = False
    return protected


def _fragmented_necrotic_prediction_grid(
    prediction_counts: np.ndarray,
    pixel_counts: np.ndarray,
    sampled_labels: np.ndarray,
    sampled_red_blue_difference: np.ndarray,
    sampled_intensity: np.ndarray,
    *,
    bin_size: int,
    minimum_component_pixels: int,
    maximum_aspect_ratio: float,
    minimum_fill_fraction: float,
    maximum_fill_fraction: float,
    minimum_mean_red_blue_difference: float,
    maximum_mean_red_blue_difference: float,
    minimum_mean_intensity: float | None,
    maximum_mean_intensity: float | None = None,
    minimum_bin_red_blue_difference: float | None = None,
    maximum_bin_intensity: float | None = None,
    eligible_instances: np.ndarray | None = None,
    protected_instances: np.ndarray | None = None,
    window_size: int | None = None,
    window_overlap: int = 0,
    context_window_size: int | None = None,
    maximum_local_prediction_fraction: float | None = None,
    maximum_source_component_pixels: int | None = None,
    source_maximum_mean_intensity: float | None = None,
    source_dilation_bins: int = 0,
) -> np.ndarray:
    """Find compact, sparse, pale-lavender dead-cell/debris fields.

    Necrotic clouds can contain chromatic fragments that individually satisfy
    nuclear support.  They differ from viable tumor by low component fill and
    from normal H&E fragments by their near-neutral red/blue balance.  A
    one-bin dilation connects only immediately neighboring predictions so the
    rule remains local and does not bridge tissue-scale gaps.
    """

    if not (
        prediction_counts.shape
        == pixel_counts.shape
        == sampled_labels.shape
        == sampled_red_blue_difference.shape
        == sampled_intensity.shape
    ):
        raise ValueError("necrotic-field sample grids must have matching shapes")
    prediction_fraction = prediction_counts / np.maximum(pixel_counts, 1)
    predicted = (prediction_fraction >= 0.10) & (sampled_labels > 0)
    if context_window_size is not None or maximum_local_prediction_fraction is not None:
        if (
            context_window_size is None
            or maximum_local_prediction_fraction is None
            or context_window_size <= 0
            or context_window_size % bin_size
            or not 0 < maximum_local_prediction_fraction <= 1
        ):
            raise ValueError("necrotic-field local context is invalid")
        from scipy import ndimage as ndi

        local_prediction_fraction = ndi.uniform_filter(
            prediction_fraction.astype(np.float32),
            size=context_window_size // bin_size,
            mode="nearest",
        )
        predicted &= local_prediction_fraction <= maximum_local_prediction_fraction
    if eligible_instances is not None:
        if eligible_instances.ndim != 1 or (
            sampled_labels.size
            and int(np.max(sampled_labels)) >= eligible_instances.size
        ):
            raise ValueError("necrotic-field eligible instances are invalid")
        predicted &= eligible_instances[sampled_labels]
    if protected_instances is not None and (
        protected_instances.ndim != 1
        or (
            sampled_labels.size
            and int(np.max(sampled_labels)) >= protected_instances.size
        )
    ):
        raise ValueError("necrotic-field protected instances are invalid")
    if minimum_bin_red_blue_difference is not None:
        predicted &= sampled_red_blue_difference >= minimum_bin_red_blue_difference
    if maximum_bin_intensity is not None:
        predicted &= sampled_intensity <= maximum_bin_intensity

    use_source_components = maximum_source_component_pixels is not None
    if use_source_components != (source_maximum_mean_intensity is not None):
        raise ValueError("necrotic-field source component parameters are incomplete")
    if use_source_components and (
        maximum_source_component_pixels <= 0
        or not 0 < source_maximum_mean_intensity <= 255
        or source_dilation_bins < 0
    ):
        raise ValueError("necrotic-field source component parameters are invalid")

    from scipy import ndimage as ndi

    def local_artifact(
        local_predicted: np.ndarray,
        local_prediction_counts: np.ndarray,
        local_red_blue: np.ndarray,
        local_intensity: np.ndarray,
    ) -> np.ndarray:
        source_components: np.ndarray | None = None
        source_component_keep: np.ndarray | None = None
        if use_source_components:
            source_foreground = local_intensity <= source_maximum_mean_intensity
            if source_dilation_bins:
                source_foreground = ndi.binary_dilation(
                    source_foreground,
                    structure=np.ones((3, 3), dtype=np.uint8),
                    iterations=source_dilation_bins,
                )
            source_components, source_component_count = ndi.label(
                source_foreground,
                structure=np.ones((3, 3), dtype=np.uint8),
            )
            source_component_pixels = np.bincount(
                source_components.ravel(),
                minlength=source_component_count + 1,
            ) * (bin_size * bin_size)
            source_component_keep = (
                source_component_pixels <= maximum_source_component_pixels
            )
            source_component_keep[0] = False
            boundary_ids = np.unique(
                np.concatenate(
                    (
                        source_components[0],
                        source_components[-1],
                        source_components[:, 0],
                        source_components[:, -1],
                    )
                )
            )
            source_component_keep[boundary_ids] = False
        connected = ndi.binary_dilation(
            local_predicted,
            structure=np.ones((3, 3), dtype=np.uint8),
        )
        components, component_count = ndi.label(
            connected,
            structure=np.ones((3, 3), dtype=np.uint8),
        )
        component_artifact = np.zeros(component_count + 1, dtype=bool)
        source_component_artifact = (
            np.zeros_like(source_component_keep)
            if source_component_keep is not None
            else None
        )
        for component_index, component_slice in enumerate(
            ndi.find_objects(components), start=1
        ):
            if component_slice is None:
                continue
            component_mask = (
                components[component_slice] == component_index
            ) & local_predicted[component_slice]
            component_pixels = float(
                local_prediction_counts[component_slice][component_mask].sum()
            )
            if component_pixels < minimum_component_pixels:
                continue
            component_height = component_slice[0].stop - component_slice[0].start
            component_width = component_slice[1].stop - component_slice[1].start
            aspect_ratio = max(
                component_width / max(component_height, 1),
                component_height / max(component_width, 1),
            )
            fill_fraction = component_pixels / (
                component_height * component_width * bin_size * bin_size
            )
            mean_red_blue = float(
                local_red_blue[component_slice][component_mask].mean()
            )
            mean_intensity = float(
                local_intensity[component_slice][component_mask].mean()
            )
            source_ids = np.empty(0, dtype=np.int64)
            source_component_isolated = True
            if source_components is not None and source_component_keep is not None:
                source_ids = np.unique(
                    source_components[component_slice][component_mask]
                )
                source_ids = source_ids[source_ids > 0]
                source_component_isolated = bool(
                    source_ids.size and np.all(source_component_keep[source_ids])
                )
            component_artifact[component_index] = (
                aspect_ratio <= maximum_aspect_ratio
                and minimum_fill_fraction <= fill_fraction <= maximum_fill_fraction
                and minimum_mean_red_blue_difference
                <= mean_red_blue
                <= maximum_mean_red_blue_difference
                and (
                    minimum_mean_intensity is None
                    or mean_intensity >= minimum_mean_intensity
                )
                and (
                    maximum_mean_intensity is None
                    or mean_intensity <= maximum_mean_intensity
                )
                and source_component_isolated
            )
            if (
                component_artifact[component_index]
                and source_component_artifact is not None
            ):
                source_component_artifact[source_ids] = True
        component_artifact[0] = False
        if source_components is not None and source_component_artifact is not None:
            source_component_artifact[0] = False
            return source_component_artifact[source_components]
        return connected & component_artifact[components]

    if window_size is None:
        artifact = local_artifact(
            predicted,
            prediction_counts,
            sampled_red_blue_difference,
            sampled_intensity,
        )
    else:
        if (
            window_size <= 0
            or window_overlap < 0
            or window_overlap >= window_size
            or window_size % bin_size
            or window_overlap % bin_size
        ):
            raise ValueError("necrotic-field window geometry is invalid")
        artifact = np.zeros(predicted.shape, dtype=bool)
        window_bins = window_size // bin_size
        overlap_bins = window_overlap // bin_size
        grid_height, grid_width = predicted.shape
        for y0 in tile_starts(grid_height, tile_size=window_bins, overlap=overlap_bins):
            y1 = min(y0 + window_bins, grid_height)
            for x0 in tile_starts(
                grid_width, tile_size=window_bins, overlap=overlap_bins
            ):
                x1 = min(x0 + window_bins, grid_width)
                artifact[y0:y1, x0:x1] |= local_artifact(
                    predicted[y0:y1, x0:x1],
                    prediction_counts[y0:y1, x0:x1],
                    sampled_red_blue_difference[y0:y1, x0:x1],
                    sampled_intensity[y0:y1, x0:x1],
                )
    if protected_instances is not None:
        # Protection is intentionally post-detection and therefore monotonic.
        # Removing organized instances before connected-component analysis can
        # split one non-artifact field into smaller components that newly pass
        # the necrosis thresholds, causing unrelated viable cells to disappear.
        artifact &= ~protected_instances[sampled_labels]
    return artifact


def _bright_instance_core_candidates(
    labels: np.ndarray,
    read_rgb: Callable[[int, int, int, int], np.ndarray],
    *,
    maximum: int,
    instance_areas: np.ndarray,
    minimum_core_pixels: int,
    minimum_bright_fraction: float,
    minimum_intensity: float,
    maximum_instance_area: int,
    block_size: int,
) -> np.ndarray:
    """Return small predictions whose exact eroded interiors are mostly pale.

    The two-pixel native erosion excludes the honeycomb wall that caused gray
    luminal mucus to resemble hematoxylin evidence. Counts are accumulated with
    a halo at block boundaries, so the decision is invariant to block layout.
    This is only an instance nomination stage; a separate large-component gate
    requires the candidates to form a compact mosaic before any label is
    removed.
    """

    from scipy import ndimage as ndi

    if instance_areas.shape != (maximum + 1,):
        raise ValueError("foam core instance-area lookup is invalid")
    height, width = labels.shape
    core_pixels = np.zeros(maximum + 1, dtype=np.uint64)
    bright_pixels = np.zeros(maximum + 1, dtype=np.uint64)
    halo = 2
    for y0 in range(0, height, block_size):
        y1 = min(height, y0 + block_size)
        hy0 = max(0, y0 - halo)
        hy1 = min(height, y1 + halo)
        for x0 in range(0, width, block_size):
            x1 = min(width, x0 + block_size)
            hx0 = max(0, x0 - halo)
            hx1 = min(width, x1 + halo)
            expanded = np.asarray(labels[hy0:hy1, hx0:hx1])
            if not np.any(expanded):
                continue
            rgb = np.asarray(read_rgb(hx0, hy0, hx1 - hx0, hy1 - hy0))
            if rgb.shape != (*expanded.shape, 3):
                raise ValueError("foam core RGB block does not match labels")
            local_min = ndi.minimum_filter(expanded, size=5, mode="nearest")
            local_max = ndi.maximum_filter(expanded, size=5, mode="nearest")
            sy0, sx0 = y0 - hy0, x0 - hx0
            sy1, sx1 = sy0 + (y1 - y0), sx0 + (x1 - x0)
            block = expanded[sy0:sy1, sx0:sx1]
            interior = (
                (block > 0)
                & (local_min[sy0:sy1, sx0:sx1] == block)
                & (local_max[sy0:sy1, sx0:sx1] == block)
            )
            intensity = np.mean(rgb[sy0:sy1, sx0:sx1], axis=2)
            counts = np.bincount(block[interior], minlength=maximum + 1)
            bright = np.bincount(
                block[interior & (intensity >= minimum_intensity)],
                minlength=maximum + 1,
            )
            core_pixels += counts[: maximum + 1].astype(np.uint64)
            bright_pixels += bright[: maximum + 1].astype(np.uint64)
    candidates = (
        (instance_areas > 0)
        & (instance_areas <= maximum_instance_area)
        & (core_pixels >= minimum_core_pixels)
        & (
            bright_pixels
            >= np.ceil(core_pixels * minimum_bright_fraction).astype(np.uint64)
        )
    )
    candidates[0] = False
    return candidates


def _compact_unsupported_prediction_grid(
    prediction_counts: np.ndarray,
    pixel_counts: np.ndarray,
    sampled_labels: np.ndarray,
    sampled_red_blue_difference: np.ndarray,
    nuclear_supported_instances: np.ndarray,
    instance_areas: np.ndarray,
    eligible_instances: np.ndarray,
    *,
    bin_size: int,
    minimum_bin_occupancy_fraction: float,
    minimum_component_pixels: int,
    maximum_aspect_ratio: float,
    maximum_mean_instance_pixels: float,
    maximum_mean_red_blue_difference: float,
    strong_red_blue_difference: float,
    maximum_strong_chromatic_fraction: float,
    elongation_ratio: float,
    maximum_elongated_instance_fraction: float,
    maximum_context_nuclear_supported_fraction: float,
    minimum_component_fill_fraction: float,
    minimum_instance_fraction: float,
    sampled_mean_intensity: np.ndarray | None = None,
    minimum_mean_intensity: float | None = None,
    window_size: int | None = None,
    window_overlap: int = 0,
    require_removable_instance_area: bool = True,
    include_nuclear_supported_in_connectivity: bool = False,
) -> np.ndarray:
    """Find large compact mosaics made from nuclear-unsupported predictions.

    Cellpose can trace regular walls in foreign luminal fragments as hundreds
    of plausible small cells.  Those fields can satisfy stain-density checks,
    unlike a single oversized debris label.  Nuclear-supported instances split
    the unsupported mosaic before a one-bin closing reconnects only very local
    gaps.  A native red-minus-blue ceiling prevents the same topology rule from
    removing red/brown-stained tissue with locally weak nuclei.  The returned
    mask contains unsupported bins only, so genuine nuclei embedded in or next
    to an artifact retain their independent evidence.

    Elongated components are deliberately excluded: thin epithelial ribbons
    may be locally weak in hematoxylin yet remain morphologically coherent.
    Compact fields are also protected when most removable instances themselves
    are elongated. This captures gland-wall organization even when a folded
    epithelial region has a compact overall bounding box.
    The physical-area floor is checked again against the complete areas of
    instances that would actually be removed. This prevents a large connected
    context from lending its size to a much smaller legitimate folded gland.
    """

    if not (
        prediction_counts.shape
        == pixel_counts.shape
        == sampled_labels.shape
        == sampled_red_blue_difference.shape
    ):
        raise ValueError("compact unsupported grids must have matching shapes")
    if (sampled_mean_intensity is None) != (minimum_mean_intensity is None):
        raise ValueError(
            "sampled mean intensity and minimum mean intensity must be supplied "
            "together"
        )
    if sampled_mean_intensity is not None and (
        sampled_mean_intensity.shape != prediction_counts.shape
    ):
        raise ValueError("compact unsupported intensity grid shape is invalid")
    if minimum_mean_intensity is not None and not (
        0 <= minimum_mean_intensity <= 255 and np.isfinite(minimum_mean_intensity)
    ):
        raise ValueError("compact unsupported minimum mean intensity is invalid")
    if (
        nuclear_supported_instances.ndim != 1
        or instance_areas.ndim != 1
        or eligible_instances.ndim != 1
    ):
        raise ValueError("instance evidence must be one-dimensional")
    if not (
        len(nuclear_supported_instances)
        == len(instance_areas)
        == len(eligible_instances)
    ):
        raise ValueError("instance evidence lookups must have matching lengths")
    if sampled_labels.size and int(sampled_labels.max()) >= len(
        nuclear_supported_instances
    ):
        raise ValueError("sampled labels exceed nuclear-support lookup")
    if not isinstance(require_removable_instance_area, bool):
        raise TypeError("require removable instance area must be boolean")
    if not isinstance(include_nuclear_supported_in_connectivity, bool):
        raise TypeError("nuclear-supported connectivity flag must be boolean")
    if window_size is not None:
        if (
            window_size <= 0
            or window_overlap < 0
            or window_overlap >= window_size
            or window_size % bin_size
            or window_overlap % bin_size
        ):
            raise ValueError("compact unsupported window geometry is invalid")
        artifact = np.zeros(prediction_counts.shape, dtype=bool)
        window_bins = window_size // bin_size
        overlap_bins = window_overlap // bin_size
        grid_height, grid_width = prediction_counts.shape
        for y0 in tile_starts(grid_height, tile_size=window_bins, overlap=overlap_bins):
            y1 = min(y0 + window_bins, grid_height)
            for x0 in tile_starts(
                grid_width, tile_size=window_bins, overlap=overlap_bins
            ):
                x1 = min(x0 + window_bins, grid_width)
                artifact[y0:y1, x0:x1] |= _compact_unsupported_prediction_grid(
                    prediction_counts[y0:y1, x0:x1],
                    pixel_counts[y0:y1, x0:x1],
                    sampled_labels[y0:y1, x0:x1],
                    sampled_red_blue_difference[y0:y1, x0:x1],
                    nuclear_supported_instances,
                    instance_areas,
                    eligible_instances,
                    bin_size=bin_size,
                    minimum_bin_occupancy_fraction=(minimum_bin_occupancy_fraction),
                    minimum_component_pixels=minimum_component_pixels,
                    maximum_aspect_ratio=maximum_aspect_ratio,
                    maximum_mean_instance_pixels=maximum_mean_instance_pixels,
                    maximum_mean_red_blue_difference=(maximum_mean_red_blue_difference),
                    strong_red_blue_difference=strong_red_blue_difference,
                    maximum_strong_chromatic_fraction=(
                        maximum_strong_chromatic_fraction
                    ),
                    elongation_ratio=elongation_ratio,
                    maximum_elongated_instance_fraction=(
                        maximum_elongated_instance_fraction
                    ),
                    maximum_context_nuclear_supported_fraction=(
                        maximum_context_nuclear_supported_fraction
                    ),
                    minimum_component_fill_fraction=(minimum_component_fill_fraction),
                    minimum_instance_fraction=minimum_instance_fraction,
                    sampled_mean_intensity=(
                        sampled_mean_intensity[y0:y1, x0:x1]
                        if sampled_mean_intensity is not None
                        else None
                    ),
                    minimum_mean_intensity=minimum_mean_intensity,
                    require_removable_instance_area=(require_removable_instance_area),
                    include_nuclear_supported_in_connectivity=(
                        include_nuclear_supported_in_connectivity
                    ),
                )
        return artifact
    prediction_fraction = prediction_counts / np.maximum(pixel_counts, 1)
    predicted = prediction_fraction >= minimum_bin_occupancy_fraction
    unsupported = (
        predicted
        & (sampled_labels > 0)
        & (~nuclear_supported_instances[sampled_labels])
    )

    from scipy import ndimage as ndi

    # A supported nucleus may carve a small hole through an otherwise
    # continuous pseudo-cell lattice.  Close only one binned pixel so large
    # tissue gaps and separate structures remain separate.
    if include_nuclear_supported_in_connectivity:
        supported = (
            predicted
            & (sampled_labels > 0)
            & nuclear_supported_instances[sampled_labels]
        )
        # Admit only supported bins locally enclosed by the unsupported
        # lattice. This bridges genuine nuclei embedded within luminal foam,
        # without allowing a supported epithelial field to connect the foam
        # component to the rest of the tissue section.
        unsupported_neighbors = ndi.convolve(
            unsupported.astype(np.uint8),
            np.ones((3, 3), dtype=np.uint8),
            mode="constant",
            cval=0,
        )
        connectivity = unsupported | (supported & (unsupported_neighbors >= 2))
    else:
        connectivity = unsupported
    connected = ndi.binary_closing(
        connectivity,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    components, component_count = ndi.label(
        connected,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    unsupported_pixels = prediction_counts * unsupported
    component_pixels = np.bincount(
        components.ravel(),
        weights=unsupported_pixels.ravel(),
        minlength=component_count + 1,
    )
    component_artifact = np.zeros(component_count + 1, dtype=bool)
    for component_index, component_slice in enumerate(
        ndi.find_objects(components), start=1
    ):
        if (
            component_slice is None
            or component_pixels[component_index] < minimum_component_pixels
        ):
            continue
        component_height = component_slice[0].stop - component_slice[0].start
        component_width = component_slice[1].stop - component_slice[1].start
        aspect_ratio = max(
            component_width / max(component_height, 1),
            component_height / max(component_width, 1),
        )
        component_mask = (components[component_slice] == component_index) & unsupported[
            component_slice
        ]
        component_labels = sampled_labels[component_slice][component_mask]
        instance_ids, local_instance_indices = np.unique(
            component_labels,
            return_inverse=True,
        )
        instance_count = np.count_nonzero(instance_ids)
        mean_instance_pixels = component_pixels[component_index] / max(
            instance_count, 1
        )
        component_fill_fraction = component_pixels[component_index] / (
            component_height * component_width * bin_size * bin_size
        )
        component_instance_pixels = np.bincount(
            local_instance_indices,
            weights=unsupported_pixels[component_slice][component_mask].ravel(),
            minlength=len(instance_ids),
        )
        removable_instances = component_instance_pixels >= np.ceil(
            instance_areas[instance_ids] * minimum_instance_fraction
        )
        removable_instances &= instance_ids > 0
        removable_instances &= eligible_instances[instance_ids]
        shape_instances = (
            removable_instances
            if require_removable_instance_area
            else (instance_ids > 0) & eligible_instances[instance_ids]
        )
        removable_component_pixels = instance_areas[
            instance_ids[removable_instances]
        ].sum()
        elongated_instance_fraction = _sampled_elongated_instance_fraction(
            sampled_labels[component_slice],
            component_mask,
            instance_ids,
            shape_instances,
            bin_size=bin_size,
            elongation_ratio=elongation_ratio,
        )
        mean_red_blue_difference = float(
            sampled_red_blue_difference[component_slice][component_mask].mean()
        )
        mean_intensity = (
            float(sampled_mean_intensity[component_slice][component_mask].mean())
            if sampled_mean_intensity is not None
            else None
        )
        strong_chromatic_fraction = float(
            np.mean(
                sampled_red_blue_difference[component_slice][component_mask]
                > strong_red_blue_difference
            )
        )
        context_mask = predicted[component_slice] & (
            sampled_labels[component_slice] > 0
        )
        context_nuclear_supported_fraction = float(
            nuclear_supported_instances[
                sampled_labels[component_slice][context_mask]
            ].mean()
        )
        component_artifact[component_index] = (
            aspect_ratio <= maximum_aspect_ratio
            and mean_instance_pixels <= maximum_mean_instance_pixels
            and mean_red_blue_difference <= maximum_mean_red_blue_difference
            and strong_chromatic_fraction <= maximum_strong_chromatic_fraction
            and elongated_instance_fraction <= maximum_elongated_instance_fraction
            and context_nuclear_supported_fraction
            <= maximum_context_nuclear_supported_fraction
            and component_fill_fraction >= minimum_component_fill_fraction
            and (
                not require_removable_instance_area
                or removable_component_pixels >= minimum_component_pixels
            )
            and (
                minimum_mean_intensity is None
                or (
                    mean_intensity is not None
                    and mean_intensity >= minimum_mean_intensity
                )
            )
        )
    component_artifact[0] = False
    return unsupported & component_artifact[components]


def _sampled_elongated_instance_fraction(
    sampled_labels: np.ndarray,
    component_mask: np.ndarray,
    instance_ids: np.ndarray,
    removable_instances: np.ndarray,
    *,
    bin_size: int,
    elongation_ratio: float,
) -> float:
    """Estimate organized cell shape from native-aligned grid samples.

    Only removable instances with at least three sampled centers contribute.
    A one-pixel native covariance regularizer avoids treating a two-pixel-wide
    prediction as infinitely elongated while preserving native-scale geometry.
    """

    removable_ids = instance_ids[removable_instances]
    if removable_ids.size == 0:
        return 0.0
    rows, columns = np.nonzero(component_mask)
    labels = sampled_labels[component_mask]
    selected = np.isin(labels, removable_ids, assume_unique=False)
    if not np.any(selected):
        return 0.0
    labels = labels[selected]
    rows = rows[selected].astype(np.float64) * bin_size
    columns = columns[selected].astype(np.float64) * bin_size
    local = np.searchsorted(removable_ids, labels)
    count = np.bincount(local, minlength=len(removable_ids)).astype(np.float64)
    sum_x = np.bincount(local, weights=columns, minlength=len(removable_ids))
    sum_y = np.bincount(local, weights=rows, minlength=len(removable_ids))
    sum_xx = np.bincount(local, weights=columns * columns, minlength=len(removable_ids))
    sum_yy = np.bincount(local, weights=rows * rows, minlength=len(removable_ids))
    sum_xy = np.bincount(local, weights=columns * rows, minlength=len(removable_ids))
    usable = count >= 3
    if not np.any(usable):
        return 0.0
    safe_count = np.maximum(count, 1)
    covariance_xx = sum_xx / safe_count - (sum_x / safe_count) ** 2
    covariance_yy = sum_yy / safe_count - (sum_y / safe_count) ** 2
    covariance_xy = sum_xy / safe_count - (sum_x * sum_y / safe_count**2)
    trace = covariance_xx + covariance_yy
    discriminant = np.sqrt(
        np.maximum(
            (covariance_xx - covariance_yy) ** 2 + 4 * covariance_xy**2,
            0,
        )
    )
    major = (trace + discriminant) / 2
    minor = (trace - discriminant) / 2
    ratios = (major + 1.0) / (minor + 1.0)
    return float(np.mean(ratios[usable] > elongation_ratio))


def _locally_organized_prediction_grid(
    prediction_counts: np.ndarray,
    pixel_counts: np.ndarray,
    sampled_labels: np.ndarray,
    sampled_red_blue_difference: np.ndarray,
    *,
    bin_size: int,
    window_size: int,
    window_overlap: int,
    minimum_bin_occupancy_fraction: float,
    minimum_component_pixels: int,
    compact_minimum_component_pixels: int,
    minimum_aspect_ratio: float,
    minimum_mean_instance_pixels: float,
    strong_red_blue_difference: float,
    minimum_strong_chromatic_fraction: float,
) -> np.ndarray:
    """Find coherent CPSAM mosaics within overlapping native-scale windows.

    Whole-slide connectivity is intentionally not used here. Thin epithelial
    ribbons can touch a much larger tissue prediction elsewhere on the slide,
    causing one global component to inherit the small mean instance area of
    unrelated dense tissue. Conversely, a debris chain can bridge otherwise
    separate regions. Bounded overlapping windows preserve the local topology
    that CPSAM saw during inference while avoiding hard seams at window edges.
    """

    if not (
        prediction_counts.shape
        == pixel_counts.shape
        == sampled_labels.shape
        == sampled_red_blue_difference.shape
    ):
        raise ValueError("organized prediction grids must have matching shapes")
    prediction_fraction = prediction_counts / np.maximum(pixel_counts, 1)
    prediction_grid = prediction_fraction >= minimum_bin_occupancy_fraction
    organized_grid = np.zeros(prediction_grid.shape, dtype=bool)
    window_bins = window_size // bin_size
    overlap_bins = window_overlap // bin_size
    grid_height, grid_width = prediction_grid.shape

    from scipy import ndimage as ndi

    for y0 in tile_starts(grid_height, tile_size=window_bins, overlap=overlap_bins):
        y1 = min(y0 + window_bins, grid_height)
        for x0 in tile_starts(grid_width, tile_size=window_bins, overlap=overlap_bins):
            x1 = min(x0 + window_bins, grid_width)
            local_prediction = prediction_grid[y0:y1, x0:x1]
            components, component_count = ndi.label(
                local_prediction,
                structure=np.ones((3, 3), dtype=np.uint8),
            )
            component_pixels = np.bincount(
                components.ravel(),
                weights=prediction_counts[y0:y1, x0:x1].ravel(),
                minlength=component_count + 1,
            )
            component_keep = np.zeros(component_count + 1, dtype=bool)
            local_labels = sampled_labels[y0:y1, x0:x1]
            for component_index, component_slice in enumerate(
                ndi.find_objects(components), start=1
            ):
                if (
                    component_slice is None
                    or component_pixels[component_index] < minimum_component_pixels
                ):
                    continue
                component_height = component_slice[0].stop - component_slice[0].start
                component_width = component_slice[1].stop - component_slice[1].start
                aspect_ratio = max(
                    component_width / max(component_height, 1),
                    component_height / max(component_width, 1),
                )
                if (
                    aspect_ratio < minimum_aspect_ratio
                    and component_pixels[component_index]
                    < compact_minimum_component_pixels
                ):
                    continue
                component_mask = components[component_slice] == component_index
                instance_count = np.count_nonzero(
                    np.unique(local_labels[component_slice][component_mask])
                )
                mean_instance_pixels = component_pixels[component_index] / max(
                    instance_count, 1
                )
                strong_chromatic_fraction = float(
                    np.mean(
                        sampled_red_blue_difference[y0:y1, x0:x1][component_slice][
                            component_mask
                        ]
                        > strong_red_blue_difference
                    )
                )
                component_keep[component_index] = (
                    mean_instance_pixels >= minimum_mean_instance_pixels
                    or strong_chromatic_fraction >= minimum_strong_chromatic_fraction
                )
            component_keep[0] = False
            organized_grid[y0:y1, x0:x1] |= (
                local_prediction & component_keep[components]
            )
    return organized_grid


def _accumulate_context_grid(
    evidence_counts: np.ndarray,
    pixel_counts: np.ndarray,
    evidence: np.ndarray,
    *,
    x0: int,
    y0: int,
    bin_size: int,
) -> None:
    """Accumulate exact source-evidence counts into globally aligned bins."""

    height, width = evidence.shape
    if x0 % bin_size == 0 and y0 % bin_size == 0:
        grid_height = (height + bin_size - 1) // bin_size
        grid_width = (width + bin_size - 1) // bin_size
        padded_height = grid_height * bin_size
        padded_width = grid_width * bin_size
        if padded_height == height and padded_width == width:
            padded = evidence
        else:
            padded = np.pad(
                evidence,
                ((0, padded_height - height), (0, padded_width - width)),
                constant_values=False,
            )
        counts = padded.reshape(
            grid_height,
            bin_size,
            grid_width,
            bin_size,
        ).sum(axis=(1, 3), dtype=np.uint64)
        row_sizes = np.minimum(
            bin_size,
            height - np.arange(grid_height, dtype=np.int64) * bin_size,
        )
        column_sizes = np.minimum(
            bin_size,
            width - np.arange(grid_width, dtype=np.int64) * bin_size,
        )
        gy = y0 // bin_size
        gx = x0 // bin_size
        evidence_counts[gy : gy + grid_height, gx : gx + grid_width] += counts
        pixel_counts[gy : gy + grid_height, gx : gx + grid_width] += (
            row_sizes[:, None] * column_sizes[None, :]
        ).astype(np.uint64)
        return
    y = 0
    while y < height:
        global_y = y0 + y
        take_y = min(height - y, bin_size - (global_y % bin_size))
        x = 0
        while x < width:
            global_x = x0 + x
            take_x = min(width - x, bin_size - (global_x % bin_size))
            block = evidence[y : y + take_y, x : x + take_x]
            grid_y = global_y // bin_size
            grid_x = global_x // bin_size
            evidence_counts[grid_y, grid_x] += int(np.count_nonzero(block))
            pixel_counts[grid_y, grid_x] += block.size
            x += take_x
        y += take_y


def _read_source_rgb(
    source: Any,
    x: int,
    y: int,
    width: int,
    height: int,
) -> np.ndarray:
    crop = source.crop(x, y, width, height)
    return np.frombuffer(crop.write_to_memory(), dtype=np.uint8).reshape(
        height, width, 3
    )


def _copy_labels_to_canvas(
    path: Path,
    destination: np.ndarray,
    *,
    block_size: int,
) -> int:
    """Copy a sealed uint32 label pyramid into a filter-upgrade canvas."""

    image = _import_pyvips().Image.new_from_file(str(path), access="sequential")
    if (
        image.width != destination.shape[1]
        or image.height != destination.shape[0]
        or image.bands != 1
        or image.format != "uint"
    ):
        raise ValueError("legacy label geometry is invalid for filter-only upgrade")
    maximum = 0
    for y0 in range(0, image.height, block_size):
        height = min(block_size, image.height - y0)
        for x0 in range(0, image.width, block_size):
            width = min(block_size, image.width - x0)
            crop = image.crop(x0, y0, width, height)
            block = np.frombuffer(crop.write_to_memory(), dtype=np.uint32).reshape(
                height, width
            )
            destination[y0 : y0 + height, x0 : x0 + width] = block
            maximum = max(maximum, int(block.max(initial=0)))
    return maximum


def _write_cached_tile(path: Path, mask: np.ndarray, fingerprint: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(
            temporary,
            mask=np.asarray(mask, dtype=np.int32),
            fingerprint=np.asarray(fingerprint),
        )
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_pyramidal_labels(
    raw_path: Path, output: Path, *, width: int, height: int
) -> None:
    pyvips = _import_pyvips()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.partial.tiff")
    temporary.unlink(missing_ok=True)
    image = pyvips.Image.rawload(str(raw_path), width, height, 1, format="uint")
    image.tiffsave(
        str(temporary),
        tile=True,
        tile_width=512,
        tile_height=512,
        pyramid=True,
        subifd=True,
        compression="deflate",
        bigtiff=True,
        properties=True,
    )
    os.replace(temporary, output)


def _disk_label_qc(
    labels: np.ndarray,
    maximum: int,
    *,
    tissue_mask: np.ndarray,
    content_shape: tuple[int, int],
    min_size: int,
) -> dict[str, object]:
    """Measure label geometry and exact mask containment in one blockwise pass."""

    if labels.shape != content_shape:
        raise ValueError("label and content shapes must match")
    tissue = np.asarray(tissue_mask, dtype=bool)
    if tissue.ndim != 2 or not tissue.size:
        raise ValueError("tissue mask must be a non-empty 2D array")
    height, width = content_shape
    mask_height, mask_width = tissue.shape
    columns = np.floor(
        (np.arange(width, dtype=np.float64) + 0.5) * mask_width / width
    ).astype(np.int64)
    np.clip(columns, 0, mask_width - 1, out=columns)
    areas = np.zeros(maximum + 1, dtype=np.uint64)
    outside_tissue_pixels = 0
    for y0 in range(0, height, 1024):
        y1 = min(height, y0 + 1024)
        block = np.asarray(labels[y0:y1])
        rows = np.floor(
            (np.arange(y0, y1, dtype=np.float64) + 0.5) * mask_height / height
        ).astype(np.int64)
        np.clip(rows, 0, mask_height - 1, out=rows)
        block_tissue = tissue[np.ix_(rows, columns)]
        outside_tissue_pixels += int(np.count_nonzero((block > 0) & (~block_tissue)))
        counts = np.bincount(block.ravel(), minlength=maximum + 1)
        areas += counts[: maximum + 1].astype(np.uint64)
    foreground = areas[1:]
    foreground = foreground[foreground > 0]
    quantiles = (
        np.quantile(foreground, [0.05, 0.5, 0.95]).tolist()
        if foreground.size
        else [0.0, 0.0, 0.0]
    )
    return {
        "cell_count": int(foreground.size),
        "foreground_pixels": int(foreground.sum()),
        "area_px_quantiles": [round(float(value), 3) for value in quantiles],
        "post_constraint_cells_below_min_size": int(
            np.count_nonzero(foreground < min_size)
        ),
        "outside_tissue_pixels_final": outside_tissue_pixels,
    }


def _request_payload(
    config: CellSegmentationConfig,
    *,
    method_profile: str = CELL_METHOD_REFERENCE_PROFILE,
    dense_recovery_payload: dict[str, object] | None = None,
) -> dict[str, object]:
    organized_necrotic_maximum_area_um2 = (
        config.isolated_debris_organized_necrotic_protection_maximum_instance_area_um2
    )
    organized_necrotic_minimum_density = getattr(
        config, _ORGANIZED_NECROTIC_MINIMUM_DENSITY_FIELD
    )
    organized_necrotic_use_eligible_context = (
        config.isolated_debris_organized_necrotic_protection_use_eligible_context
    )
    organized_necrotic_require_independent_nuclear_context = getattr(
        config, _ORGANIZED_NECROTIC_INDEPENDENT_CONTEXT_FIELD
    )
    organized_necrotic_sparse_glass_shape_gate = (
        config.isolated_debris_organized_necrotic_sparse_glass_shape_gate
    )
    return {
        "sections": list(config.sections),
        "analysis_selection": (
            "all_registered"
            if config.analysis_manifest is None
            else "registered_minus_manifest_exclusions"
        ),
        "model": config.model,
        "method": config.method,
        "method_reference": {
            "name": CELL_METHOD_REFERENCE_NAME,
            "sha256": CELL_METHOD_REFERENCE_SHA256,
            "profile": method_profile,
        },
        **(
            {"dense_small_cell_recovery": dense_recovery_payload}
            if dense_recovery_payload is not None
            else {}
        ),
        "inference_batch_size": config.inference_batch_size,
        "tile_size": config.tile_size,
        "overlap": config.overlap,
        "tissue_fraction_threshold": config.tissue_fraction_threshold,
        "first_cellprob": config.first_cellprob,
        "first_diameter": config.first_diameter,
        "second_cellprob": config.second_cellprob,
        "second_diameter": config.second_diameter,
        "flow_threshold": config.flow_threshold,
        "merge_threshold": config.merge_threshold,
        "match_ios": config.match_ios,
        "min_size": config.min_size,
        "multiscale_nuclear_support": {
            "enabled": config.multiscale_nuclear_support,
            "minimum_optical_density": config.nuclear_minimum_optical_density,
            "minimum_pixels": config.nuclear_minimum_pixels,
            "minimum_fraction": config.nuclear_minimum_fraction,
        },
        "global_nuclear_support": {
            "enabled": config.global_nuclear_support,
            "minimum_optical_density": (config.global_nuclear_minimum_optical_density),
            "minimum_pixels": config.global_nuclear_minimum_pixels,
            "minimum_fraction": config.global_nuclear_minimum_fraction,
        },
        "source_tissue_context": {
            "enabled": config.source_tissue_context,
            "bin_size_px": config.source_tissue_context_bin_size,
            "minimum_stain_fraction": (config.source_tissue_context_minimum_fraction),
            "minimum_instance_fraction": (
                config.source_tissue_context_minimum_instance_fraction
            ),
        },
        "adaptive_nuclear_core": {
            "enabled": config.adaptive_nuclear_core,
            "percentile": config.adaptive_nuclear_core_percentile,
            "minimum_area_um2": config.adaptive_nuclear_core_minimum_area_um2,
            "minimum_pixels": config.adaptive_nuclear_core_minimum_pixels,
            "minimum_fraction": config.adaptive_nuclear_core_minimum_fraction,
            "minimum_blue_ratio": (config.adaptive_nuclear_core_minimum_blue_ratio),
        },
        "isolated_debris_gate": {
            "enabled": config.isolated_debris_gate,
            "context_bin_size_px": config.isolated_debris_context_bin_size,
            "context_minimum_stain_fraction": (
                config.isolated_debris_context_minimum_stain_fraction
            ),
            "context_minimum_instance_fraction": (
                config.isolated_debris_context_minimum_instance_fraction
            ),
            "component_minimum_nuclear_pixels": (
                config.isolated_debris_component_minimum_nuclear_pixels
            ),
            "allow_nuclear_escape": config.isolated_debris_allow_nuclear_escape,
            **(
                {"allow_organized_escape": False}
                if not config.isolated_debris_allow_organized_escape
                else {}
            ),
            "nuclear_minimum_pixels": (config.isolated_debris_nuclear_minimum_pixels),
            "nuclear_minimum_fraction": (
                config.isolated_debris_nuclear_minimum_fraction
            ),
            "nuclear_minimum_blue_ratio": (
                config.isolated_debris_nuclear_minimum_blue_ratio
            ),
            "neutral_dark_maximum_value": (
                config.isolated_debris_neutral_dark_maximum_value
            ),
            "neutral_dark_maximum_chroma": (
                config.isolated_debris_neutral_dark_maximum_chroma
            ),
            "neutral_dark_maximum_fraction": (
                config.isolated_debris_neutral_dark_maximum_fraction
            ),
            "very_dark_maximum_value": (config.isolated_debris_very_dark_maximum_value),
            "very_dark_maximum_chroma": (
                config.isolated_debris_very_dark_maximum_chroma
            ),
            "very_dark_maximum_fraction": (
                config.isolated_debris_very_dark_maximum_fraction
            ),
            "micro_bin_size_px": config.isolated_debris_micro_bin_size,
            "micro_minimum_stain_fraction": (
                config.isolated_debris_micro_minimum_stain_fraction
            ),
            "micro_maximum_component_bins": (
                config.isolated_debris_micro_maximum_component_bins
            ),
            "micro_minimum_nuclear_fraction": (
                config.isolated_debris_micro_minimum_nuclear_fraction
            ),
            "micro_minimum_instance_fraction": (
                config.isolated_debris_micro_minimum_instance_fraction
            ),
            "organized_bin_size_px": config.isolated_debris_organized_bin_size,
            "organized_window_size_px": (config.isolated_debris_organized_window_size),
            "organized_window_overlap_px": (
                config.isolated_debris_organized_window_overlap
            ),
            "organized_minimum_bin_occupancy_fraction": (
                config.isolated_debris_organized_minimum_bin_occupancy_fraction
            ),
            "organized_minimum_component_pixels": (
                config.isolated_debris_organized_minimum_component_pixels
            ),
            "organized_compact_minimum_component_pixels": (
                config.isolated_debris_organized_compact_minimum_component_pixels
            ),
            "organized_minimum_aspect_ratio": (
                config.isolated_debris_organized_minimum_aspect_ratio
            ),
            "organized_minimum_mean_instance_pixels": (
                config.isolated_debris_organized_minimum_mean_instance_pixels
            ),
            "organized_minimum_instance_fraction": (
                config.isolated_debris_organized_minimum_instance_fraction
            ),
            "organized_strong_red_blue_difference": (
                config.isolated_debris_organized_strong_red_blue_difference
            ),
            "organized_minimum_strong_chromatic_fraction": (
                config.isolated_debris_organized_minimum_strong_chromatic_fraction
            ),
            "compact_unsupported_minimum_area_um2": (
                config.isolated_debris_compact_unsupported_minimum_area_um2
            ),
            "compact_unsupported_maximum_aspect_ratio": (
                config.isolated_debris_compact_unsupported_maximum_aspect_ratio
            ),
            "compact_unsupported_maximum_mean_instance_pixels": (
                config.isolated_debris_compact_unsupported_maximum_mean_instance_pixels
            ),
            "compact_unsupported_maximum_mean_red_blue_difference": (
                config.isolated_debris_compact_unsupported_maximum_mean_red_blue_difference
            ),
            "compact_unsupported_strong_red_blue_difference": (
                config.isolated_debris_compact_unsupported_strong_red_blue_difference
            ),
            "compact_unsupported_maximum_strong_chromatic_fraction": (
                config.isolated_debris_compact_unsupported_maximum_strong_chromatic_fraction
            ),
            "compact_unsupported_elongation_ratio": (
                config.isolated_debris_compact_unsupported_elongation_ratio
            ),
            "compact_unsupported_maximum_elongated_instance_fraction": (
                config.isolated_debris_compact_unsupported_maximum_elongated_instance_fraction
            ),
            "compact_unsupported_maximum_context_nuclear_fraction": (
                config.isolated_debris_compact_unsupported_maximum_context_nuclear_fraction
            ),
            "compact_unsupported_minimum_component_fill_fraction": (
                config.isolated_debris_compact_unsupported_minimum_component_fill_fraction
            ),
            "compact_unsupported_minimum_instance_fraction": (
                config.isolated_debris_compact_unsupported_minimum_instance_fraction
            ),
            "foam_minimum_mean_intensity": (
                config.isolated_debris_foam_minimum_mean_intensity
            ),
            "foam_maximum_mean_instance_pixels": (
                config.isolated_debris_foam_maximum_mean_instance_pixels
            ),
            "foam_maximum_strong_chromatic_fraction": (
                config.isolated_debris_foam_maximum_strong_chromatic_fraction
            ),
            "foam_maximum_context_nuclear_fraction": (
                config.isolated_debris_foam_maximum_context_nuclear_fraction
            ),
            "foam_maximum_elongated_instance_fraction": (
                config.isolated_debris_foam_maximum_elongated_instance_fraction
            ),
            "foam_core_minimum_pixels": (
                config.isolated_debris_foam_core_minimum_pixels
            ),
            "foam_core_minimum_bright_fraction": (
                config.isolated_debris_foam_core_minimum_bright_fraction
            ),
            "foam_core_minimum_intensity": (
                config.isolated_debris_foam_core_minimum_intensity
            ),
            "foam_core_maximum_instance_area_um2": (
                config.isolated_debris_foam_core_maximum_instance_area_um2
            ),
            "foam_core_minimum_component_area_um2": (
                config.isolated_debris_foam_core_minimum_component_area_um2
            ),
            "foam_core_maximum_aspect_ratio": (
                config.isolated_debris_foam_core_maximum_aspect_ratio
            ),
            "oversized_chromatic_minimum_area_um2": (
                config.isolated_debris_oversized_chromatic_minimum_area_um2
            ),
            "oversized_chromatic_minimum_fraction": (
                config.isolated_debris_oversized_chromatic_minimum_fraction
            ),
            "fold_minimum_instance_area_um2": (
                config.isolated_debris_fold_minimum_instance_area_um2
            ),
            "fold_minimum_component_area_um2": (
                config.isolated_debris_fold_minimum_component_area_um2
            ),
            "fold_maximum_mean_red_blue_difference": (
                config.isolated_debris_fold_maximum_mean_red_blue_difference
            ),
            "fold_maximum_mean_intensity": (
                config.isolated_debris_fold_maximum_mean_intensity
            ),
            "fold_dense_minimum_instances": (
                config.isolated_debris_fold_dense_minimum_instances
            ),
            "fold_dense_minimum_aspect_ratio": (
                config.isolated_debris_fold_dense_minimum_aspect_ratio
            ),
            "fold_dense_connectivity_dilation_bins": (
                config.isolated_debris_fold_dense_connectivity_dilation_bins
            ),
            "fold_dense_maximum_mean_red_blue_difference": (
                config.isolated_debris_fold_dense_maximum_mean_red_blue_difference
            ),
            "fold_dense_maximum_mean_intensity": (
                config.isolated_debris_fold_dense_maximum_mean_intensity
            ),
            "detached_minimum_instance_area_um2": (
                config.isolated_debris_detached_minimum_instance_area_um2
            ),
            "detached_minimum_component_area_um2": (
                config.isolated_debris_detached_minimum_component_area_um2
            ),
            "detached_minimum_instances": (
                config.isolated_debris_detached_minimum_instances
            ),
            "detached_minimum_aspect_ratio": (
                config.isolated_debris_detached_minimum_aspect_ratio
            ),
            "detached_maximum_mean_red_blue_difference": (
                config.isolated_debris_detached_maximum_mean_red_blue_difference
            ),
            "detached_maximum_mean_intensity": (
                config.isolated_debris_detached_maximum_mean_intensity
            ),
            "detached_context_window_size_px": (
                config.isolated_debris_detached_context_window_size
            ),
            "detached_maximum_prediction_fraction": (
                config.isolated_debris_detached_maximum_prediction_fraction
            ),
            "detached_compact_minimum_instance_area_um2": (
                config.isolated_debris_detached_compact_minimum_instance_area_um2
            ),
            "detached_compact_minimum_component_area_um2": (
                config.isolated_debris_detached_compact_minimum_component_area_um2
            ),
            "detached_compact_minimum_instances": (
                config.isolated_debris_detached_compact_minimum_instances
            ),
            "detached_compact_maximum_mean_red_blue_difference": (
                config.isolated_debris_detached_compact_maximum_mean_red_blue_difference
            ),
            "detached_compact_maximum_mean_intensity": (
                config.isolated_debris_detached_compact_maximum_mean_intensity
            ),
            "glass_minimum_area_um2": (config.isolated_debris_glass_minimum_area_um2),
            "glass_context_window_size_px": (
                config.isolated_debris_glass_context_window_size
            ),
            "glass_maximum_prediction_fraction": (
                config.isolated_debris_glass_maximum_prediction_fraction
            ),
            "glass_maximum_context_stain_fraction": (
                config.isolated_debris_glass_maximum_context_stain_fraction
            ),
            "glass_low_stain_maximum_mean_red_blue_difference": (
                config.isolated_debris_glass_low_stain_maximum_mean_red_blue_difference
            ),
            "glass_low_stain_minimum_mean_intensity": (
                config.isolated_debris_glass_low_stain_minimum_mean_intensity
            ),
            "glass_minimum_context_fraction": (
                config.isolated_debris_glass_minimum_context_fraction
            ),
            "glass_maximum_mean_red_blue_difference": (
                config.isolated_debris_glass_maximum_mean_red_blue_difference
            ),
            "glass_minimum_mean_intensity": (
                config.isolated_debris_glass_minimum_mean_intensity
            ),
            **(
                {"self_dense_glass_gate": True}
                if config.isolated_debris_self_dense_glass_gate
                else {}
            ),
            **(
                {
                    "oversized_brown_gate": True,
                    "oversized_brown_maximum_mean_intensity": (
                        config.isolated_debris_oversized_brown_maximum_mean_intensity
                    ),
                }
                if config.isolated_debris_oversized_brown_gate
                else {}
            ),
            **(
                {"clustered_brown_gate": True}
                if config.isolated_debris_clustered_brown_gate
                else {}
            ),
            **(
                {"diffuse_degenerated_gate": True}
                if config.isolated_debris_diffuse_degenerated_gate
                else {}
            ),
            **(
                {"neutral_precipitate_gate": True}
                if config.isolated_debris_neutral_precipitate_gate
                else {}
            ),
            **(
                {
                    "satellite_gate": {
                        "enabled": True,
                        "section": config.sections[0],
                        "scope": CELL_SATELLITE_PROTECTION_SCOPE,
                        "interior_protection": {
                            "minimum_mask_distance_um": (
                                config.isolated_debris_satellite_minimum_mask_distance_um
                            ),
                            "minimum_sample_fraction": (
                                CELL_SATELLITE_PROTECTION_MINIMUM_SAMPLE_FRACTION
                            ),
                        },
                    }
                }
                if config.isolated_debris_satellite_minimum_mask_distance_um is not None
                else {}
            ),
            **(
                {"reconcile_enclosed_cytoplasmic_children": True}
                if config.reconcile_enclosed_cytoplasmic_children
                else {}
            ),
            "necrotic_minimum_component_area_um2": (
                config.isolated_debris_necrotic_minimum_component_area_um2
            ),
            "necrotic_maximum_aspect_ratio": (
                config.isolated_debris_necrotic_maximum_aspect_ratio
            ),
            "necrotic_minimum_fill_fraction": (
                config.isolated_debris_necrotic_minimum_fill_fraction
            ),
            "necrotic_maximum_fill_fraction": (
                config.isolated_debris_necrotic_maximum_fill_fraction
            ),
            "necrotic_minimum_mean_red_blue_difference": (
                config.isolated_debris_necrotic_minimum_mean_red_blue_difference
            ),
            "necrotic_maximum_mean_red_blue_difference": (
                config.isolated_debris_necrotic_maximum_mean_red_blue_difference
            ),
            "necrotic_minimum_mean_intensity": (
                config.isolated_debris_necrotic_minimum_mean_intensity
            ),
            "necrotic_minimum_instance_fraction": (
                config.isolated_debris_necrotic_minimum_instance_fraction
            ),
            "necrotic_window_size_px": config.isolated_debris_organized_window_size,
            "necrotic_window_overlap_px": (
                config.isolated_debris_organized_window_overlap
            ),
            "necrotic_context_window_size_px": 256,
            "necrotic_maximum_local_prediction_fraction": 0.40,
            **(
                {
                    "protect_organized_from_necrotic": True,
                    **(
                        {
                            "organized_necrotic_protection_maximum_instance_area_um2": (
                                organized_necrotic_maximum_area_um2
                            )
                        }
                        if organized_necrotic_maximum_area_um2 is not None
                        else {}
                    ),
                    **(
                        {
                            (
                                "organized_necrotic_protection_"
                                "minimum_local_prediction_fraction"
                            ): (organized_necrotic_minimum_density)
                        }
                        if organized_necrotic_minimum_density is not None
                        else {}
                    ),
                    **(
                        {
                            "organized_necrotic_protection_local_prediction_source": (
                                "preartifact_independent_nuclear_external"
                                if (
                                    organized_necrotic_require_independent_nuclear_context
                                )
                                else "preartifact_eligible_external"
                            )
                        }
                        if organized_necrotic_use_eligible_context
                        else {}
                    ),
                    **(
                        {
                            "organized_necrotic_sparse_glass_shape_gate": {
                                "context_window_size_px": (
                                    _ORGANIZED_NECROTIC_SPARSE_GLASS_CONTEXT_WINDOW
                                ),
                                "maximum_mean_intensity": (
                                    _ORGANIZED_NECROTIC_SPARSE_GLASS_MAXIMUM_MEAN_INTENSITY
                                ),
                                "minimum_tissue_fraction": (
                                    _ORGANIZED_NECROTIC_SPARSE_GLASS_MINIMUM_TISSUE_FRACTION
                                ),
                                "maximum_sampled_fill_fraction": (
                                    _ORGANIZED_NECROTIC_SPARSE_GLASS_MAXIMUM_SAMPLED_FILL_FRACTION
                                ),
                                "minimum_sampled_elongation": (
                                    _ORGANIZED_NECROTIC_SPARSE_GLASS_MINIMUM_SAMPLED_ELONGATION
                                ),
                            }
                        }
                        if organized_necrotic_sparse_glass_shape_gate
                        else {}
                    ),
                }
                if config.isolated_debris_protect_organized_from_necrotic
                else {}
            ),
            "brown_minimum_component_area_um2": (
                config.isolated_debris_brown_minimum_component_area_um2
            ),
            "brown_maximum_aspect_ratio": (
                config.isolated_debris_brown_maximum_aspect_ratio
            ),
            "brown_minimum_fill_fraction": (
                config.isolated_debris_brown_minimum_fill_fraction
            ),
            "brown_maximum_fill_fraction": (
                config.isolated_debris_brown_maximum_fill_fraction
            ),
            "brown_minimum_mean_red_blue_difference": (
                config.isolated_debris_brown_minimum_mean_red_blue_difference
            ),
            "brown_maximum_mean_intensity": (
                config.isolated_debris_brown_maximum_mean_intensity
            ),
            "brown_minimum_instance_fraction": (
                config.isolated_debris_brown_minimum_instance_fraction
            ),
            "brown_requires_nuclear_unsupported": False,
            "brown_expands_to_source_component": True,
            "brown_window_size_px": config.isolated_debris_organized_window_size,
            "brown_window_overlap_px": (
                config.isolated_debris_organized_window_overlap
            ),
            "brown_context_window_size_px": 256,
            "brown_maximum_local_prediction_fraction": 0.50,
            "brown_maximum_source_component_area_um2": (
                config.isolated_debris_brown_maximum_source_component_area_um2
            ),
            "brown_source_maximum_mean_intensity": (
                config.isolated_debris_brown_source_maximum_mean_intensity
            ),
            "brown_source_dilation_bins": (
                config.isolated_debris_brown_source_dilation_bins
            ),
        },
        "tissue_constraint": "accepted_mask_nearest",
        "instance_evidence": {
            "method": "local_optical_density",
            "background_percentile": _EVIDENCE_BACKGROUND_PERCENTILE,
            "minimum_optical_density": _EVIDENCE_MINIMUM_OPTICAL_DENSITY,
            "minimum_instance_fraction": _EVIDENCE_MINIMUM_INSTANCE_FRACTION,
            "block_size_px": _EVIDENCE_BLOCK_SIZE,
        },
    }


def _progress(callback: Callable[[str], None] | None, message: str) -> None:
    if callback is not None:
        callback(message)


def _baseline_fold_tile_profile_fingerprints(
    *,
    preflight_fingerprint: str,
    model: dict[str, object],
    request: dict[str, object],
    algorithm_version: int = _ALGORITHM_VERSION,
) -> tuple[str, ...]:
    """Return an exact baseline-v75 profile usable only for raw tile reuse.

    The two fold color thresholds are post-inference evidence gates. Their
    validated dark-fold variant therefore may reuse CPSAM tile masks produced
    with the baseline values, while completed section checkpoints remain bound
    to the current full request and can never be mistaken for current output.
    """

    isolated = request.get("isolated_debris_gate")
    if not isinstance(isolated, dict):
        return ()
    current = (
        isolated.get("fold_maximum_mean_red_blue_difference"),
        isolated.get("fold_maximum_mean_intensity"),
    )
    baseline = (40.0, 180.0)
    if current == baseline:
        return ()
    baseline_request = json.loads(json.dumps(request))
    baseline_isolated = baseline_request["isolated_debris_gate"]
    baseline_isolated["fold_maximum_mean_red_blue_difference"] = baseline[0]
    baseline_isolated["fold_maximum_mean_intensity"] = baseline[1]
    return (
        _json_sha256(
            {
                "algorithm_version": algorithm_version,
                "preflight_fingerprint": preflight_fingerprint,
                "model": model,
                "request": baseline_request,
            }
        ),
    )


def _baseline_foam_tile_profile_fingerprints(
    *,
    preflight_fingerprint: str,
    model: dict[str, object],
    request: dict[str, object],
    algorithm_version: int = _ALGORITHM_VERSION,
) -> tuple[str, ...]:
    """Return the exact baseline-v75 profile for foam-only filter probes.

    Foam-core settings are evaluated only after inference from the native CPSAM
    tile masks. A bounded threshold experiment may therefore reuse tiles from
    the validated baseline request, while section checkpoints and final
    artifacts remain bound to the complete current request fingerprint.
    """

    isolated = request.get("isolated_debris_gate")
    if not isinstance(isolated, dict):
        return ()
    keys = (
        "foam_core_minimum_pixels",
        "foam_core_minimum_bright_fraction",
        "foam_core_minimum_intensity",
        "foam_core_maximum_instance_area_um2",
        "foam_core_minimum_component_area_um2",
        "foam_core_maximum_aspect_ratio",
    )
    baseline = (8, 0.80, 180.0, 110.0, 6500.0, 2.50)
    if tuple(isolated.get(key) for key in keys) == baseline:
        return ()
    baseline_request = json.loads(json.dumps(request))
    baseline_isolated = baseline_request["isolated_debris_gate"]
    for key, value in zip(keys, baseline, strict=True):
        baseline_isolated[key] = value
    return (
        _json_sha256(
            {
                "algorithm_version": algorithm_version,
                "preflight_fingerprint": preflight_fingerprint,
                "model": model,
                "request": baseline_request,
            }
        ),
    )


def _json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _import_pyvips() -> Any:
    try:
        import pyvips
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "whole-slide cell segmentation requires Histopia's optional "
            "'cells' dependencies"
        ) from error
    return pyvips
