"""Scientific promotion gate for complete native-space cell runs."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from histopia.cells._bounded_additive_recovery import (
    BOUNDED_ADDITIVE_RECOVERY_ALGORITHM_VERSION,
    BOUNDED_ADDITIVE_RECOVERY_METHOD_PROFILE,
    validate_bounded_additive_recovery_provenance,
    validate_bounded_additive_recovery_qc,
)
from histopia.cells._chromatic import (
    CHROMATIC_MICROCLUSTER_ALGORITHM_VERSIONS,
    CHROMATIC_MICROCLUSTER_POLICIES,
    CHROMATIC_MICROCLUSTER_PROMOTION_PROFILES,
    PAS_MICROCLUSTER_ALGORITHM_VERSION,
    chromatic_microcluster_gate,
    chromatic_microcluster_run_spec,
)
from histopia.cells._dense_small_cell_recovery import (
    DENSE_SMALL_CELL_RECOVERY_ALGORITHM_VERSIONS,
    DENSE_SMALL_CELL_RECOVERY_PROMOTION_PROFILES,
    dense_small_cell_recovery_parameters_for_method,
    dense_small_cell_recovery_profile,
    validate_dense_small_cell_recovery_model,
)
from histopia.cells._foreign_material import (
    SATURATED_CYAN_FOREIGN_MATERIAL_ALGORITHM_VERSION,
    SATURATED_CYAN_FOREIGN_MATERIAL_METHOD_PROFILE,
    SATURATED_CYAN_FOREIGN_MATERIAL_SCOPE,
    saturated_cyan_foreign_material_gate,
)
from histopia.cells._neutral_precipitate import (
    NEUTRAL_PRECIPITATE_ALGORITHM_VERSION,
    NEUTRAL_PRECIPITATE_FORBIDDEN_REFINEMENTS,
    NEUTRAL_PRECIPITATE_METHOD_PROFILE,
    validate_neutral_precipitate_qc,
    validate_neutral_precipitate_source,
)
from histopia.cells._preflight import CellPreflight
from histopia.cells._result import validate_cell_result
from histopia.cells._reviewed_addition import (
    REVIEWED_CACHE_ADDITION_ALGORITHM_VERSION,
    REVIEWED_CACHE_ADDITION_METHOD_PROFILE,
    REVIEWED_CACHE_ADDITION_SCOPE,
    validate_reviewed_cache_addition_manifest,
    validate_reviewed_cache_addition_qc,
)
from histopia.cells._reviewed_exclusion import (
    REVIEWED_ADDITION_EXCLUSION_ALGORITHM_VERSION,
    REVIEWED_ADDITION_EXCLUSION_METHOD_PROFILE,
    REVIEWED_ADDITION_EXCLUSION_SOURCE_ALGORITHM_VERSION,
    REVIEWED_ARTIFACT_EXCLUSION_ALGORITHM_VERSION,
    REVIEWED_ARTIFACT_EXCLUSION_METHOD_PROFILE,
    REVIEWED_ARTIFACT_EXCLUSION_SCOPE,
    REVIEWED_ARTIFACT_EXCLUSION_SOURCE_ALGORITHM_VERSION,
    reviewed_artifact_exclusion_manifest_sha256,
    reviewed_artifact_exclusion_qc_evidence,
    reviewed_artifact_exclusion_section_sha256,
    validate_reviewed_artifact_exclusion_manifest,
)
from histopia.cells._reviewed_input_addition import (
    REVIEWED_INPUT_ADDITION_ALGORITHM_VERSION,
    REVIEWED_INPUT_ADDITION_METHOD_PROFILE,
    validate_reviewed_input_addition_provenance,
    validate_reviewed_input_addition_qc,
)
from histopia.cells._reviewed_multi_addition import (
    REVIEWED_MULTI_ADDITION_ALGORITHM_VERSION,
    REVIEWED_MULTI_ADDITION_METHOD_PROFILE,
    validate_reviewed_multi_addition_provenance,
    validate_reviewed_multi_addition_qc,
)

CELL_METHOD_REFERENCE_NAME = "run_best_method_whole_image.py"
CELL_METHOD_REFERENCE_SHA256 = (
    "d827cea0f33d9cdc08b95a7b26fa715ea805fff1a464c105088277a01bfcd548"
)
CELL_METHOD_REFERENCE_PROFILE = "combined-containment-cpsam-wsi-v71"
CELL_PROMOTION_ALGORITHM_VERSION = 75
CELL_ORGANIZED_ESCAPE_ALGORITHM_VERSION = 87
CELL_ORGANIZED_ESCAPE_METHOD_PROFILE = "combined-containment-cpsam-wsi-v82"
CELL_ORGANIZED_ESCAPE_SCOPE = "post-inference-organized-escape-exclusion-v1"
CELL_SATELLITE_PROTECTION_ALGORITHM_VERSION = 89
CELL_SATELLITE_PROTECTION_METHOD_PROFILE = "combined-containment-cpsam-wsi-v84"
CELL_SATELLITE_PROTECTION_SCOPE = (
    "post-inference-section-satellite-interior-protection-v1"
)
CELL_SATELLITE_PROTECTION_MINIMUM_MASK_DISTANCE_UM = 200.0
CELL_SATELLITE_PROTECTION_MINIMUM_SAMPLE_FRACTION = 0.50
CELL_NO_ESCAPE_ALGORITHM_VERSIONS = frozenset(
    {CELL_ORGANIZED_ESCAPE_ALGORITHM_VERSION, PAS_MICROCLUSTER_ALGORITHM_VERSION}
)
CELL_VALIDATED_PROMOTION_PROFILES = (
    frozenset(
        {
            (75, "combined-containment-cpsam-wsi-v71"),
            (82, "combined-containment-cpsam-wsi-v78"),
            (83, "combined-containment-cpsam-wsi-v79"),
            (84, "combined-containment-cpsam-wsi-v80"),
            (
                REVIEWED_INPUT_ADDITION_ALGORITHM_VERSION,
                REVIEWED_INPUT_ADDITION_METHOD_PROFILE,
            ),
            (
                REVIEWED_MULTI_ADDITION_ALGORITHM_VERSION,
                REVIEWED_MULTI_ADDITION_METHOD_PROFILE,
            ),
            (
                CELL_ORGANIZED_ESCAPE_ALGORITHM_VERSION,
                CELL_ORGANIZED_ESCAPE_METHOD_PROFILE,
            ),
            (
                CELL_SATELLITE_PROTECTION_ALGORITHM_VERSION,
                CELL_SATELLITE_PROTECTION_METHOD_PROFILE,
            ),
            (
                BOUNDED_ADDITIVE_RECOVERY_ALGORITHM_VERSION,
                BOUNDED_ADDITIVE_RECOVERY_METHOD_PROFILE,
            ),
            *DENSE_SMALL_CELL_RECOVERY_PROMOTION_PROFILES,
            (
                SATURATED_CYAN_FOREIGN_MATERIAL_ALGORITHM_VERSION,
                SATURATED_CYAN_FOREIGN_MATERIAL_METHOD_PROFILE,
            ),
            (
                REVIEWED_ARTIFACT_EXCLUSION_ALGORITHM_VERSION,
                REVIEWED_ARTIFACT_EXCLUSION_METHOD_PROFILE,
            ),
            (
                REVIEWED_CACHE_ADDITION_ALGORITHM_VERSION,
                REVIEWED_CACHE_ADDITION_METHOD_PROFILE,
            ),
            (
                REVIEWED_ADDITION_EXCLUSION_ALGORITHM_VERSION,
                REVIEWED_ADDITION_EXCLUSION_METHOD_PROFILE,
            ),
            (
                NEUTRAL_PRECIPITATE_ALGORITHM_VERSION,
                NEUTRAL_PRECIPITATE_METHOD_PROFILE,
            ),
        }
    )
    | CHROMATIC_MICROCLUSTER_PROMOTION_PROFILES
)

# Exact, visually validated foam-core profiles.  The default profile is used
# cohort-wide.  The second profile is the stricter bright-microfoam profile
# validated on 4714 against both pathological bubble lattices and viable-tissue
# controls.  Keeping complete tuples here prevents unreviewed parameter mixing.
CELL_VALIDATED_FOAM_CORE_PROFILES = frozenset(
    {
        (8, 0.80, 180.0, 110.0, 6500.0, 2.50),
        (2, 0.65, 120.0, 180.0, 3000.0, 2.50),
        (1, 0.45, 100.0, 180.0, 1000.0, 2.50),
    }
)

# The focus safeguard is a deterministic post-inference filter.  Its sealed
# runner accepts either the validated v75 campaign result or the v82
# sparse-glass refilter as input and records the exact source result and label
# digests.  Keep the promotion gate in lock-step with that runner so a bounded
# focus rescue does not require an unrelated v82 refilter first.
CELL_VALIDATED_FOCUS_SOURCE_ALGORITHMS = frozenset({75, 82})
_DENSE_SMALL_CELL_TILE_KEY = re.compile(r"^r[0-9]{4}_c[0-9]{4}_y[0-9]+_x[0-9]+\.npz$")


def validate_cell_promotion_candidate(
    run_dir: Path | str,
    *,
    expected_preflight: CellPreflight | None = None,
    expected_model: str = "cpsam",
    expected_method: str = "containment",
    require_complete_preflight: bool = True,
) -> dict[str, object]:
    """Require a complete sealed run using the validated containment profile.

    This gate is deliberately stricter than :func:`validate_cell_result`, which
    proves artifact integrity but permits other experimental algorithms.  A run
    promoted to the stable reviewer must also match the selected registration
    preflight and retain the tissue, debris, and nuclear-support safeguards used
    by the validated whole-slide method.
    """

    root = Path(run_dir)
    result = validate_cell_result(root)
    algorithm_version = result.get("algorithm_version")

    preflight_relative = result.get("preflight")
    if not isinstance(preflight_relative, str):
        raise ValueError("cell promotion requires a sealed preflight")
    preflight = _object(root / preflight_relative, "cell preflight")
    if preflight.get("schema_version") != 2:
        raise ValueError("cell promotion requires preflight schema version 2")
    if result.get("preflight_fingerprint") != preflight.get("fingerprint"):
        raise ValueError("cell promotion preflight fingerprint is stale")
    if expected_preflight is not None and (
        preflight.get("fingerprint") != expected_preflight.fingerprint
    ):
        raise ValueError("cell promotion preflight differs from current selection")

    preflight_rows = _rows(preflight, "slides", "cell preflight")
    result_rows = _rows(result, "slides", "cell result")
    slide_count = preflight.get("slide_count")
    if slide_count != len(preflight_rows):
        raise ValueError("cell promotion does not cover every selected section")
    if require_complete_preflight and len(result_rows) != len(preflight_rows):
        raise ValueError("cell promotion does not cover every selected section")
    expected_pairs = [
        (row.get("section"), row.get("slide_name")) for row in preflight_rows
    ]
    result_pairs = [(row.get("section"), row.get("slide")) for row in result_rows]
    if require_complete_preflight:
        pairs_match = result_pairs == expected_pairs
    else:
        expected_pair_set = set(expected_pairs)
        pairs_match = (
            bool(result_pairs)
            and len(result_pairs) == len(set(result_pairs))
            and all(pair in expected_pair_set for pair in result_pairs)
            and result_pairs
            == [pair for pair in expected_pairs if pair in set(result_pairs)]
        )
    if not pairs_match or any(
        not isinstance(section, str) or not isinstance(slide, str)
        for section, slide in result_pairs
    ):
        raise ValueError("cell promotion sections differ from the sealed preflight")

    if algorithm_version == 85:
        return _validate_composite_candidate(
            root,
            result,
            result_rows=result_rows,
            expected_model=expected_model,
            expected_method=expected_method,
        )

    request = result.get("request")
    if not isinstance(request, dict):
        raise ValueError("cell promotion request is missing")
    if (
        request.get("model") != expected_model
        or request.get("method") != expected_method
    ):
        raise ValueError("cell promotion method profile is not the validated candidate")
    reference = request.get("method_reference")
    if (
        not isinstance(reference, dict)
        or reference.get("name") != CELL_METHOD_REFERENCE_NAME
        or reference.get("sha256") != CELL_METHOD_REFERENCE_SHA256
        or (algorithm_version, reference.get("profile"))
        not in CELL_VALIDATED_PROMOTION_PROFILES
    ):
        raise ValueError("cell promotion method reference is stale")
    method_profile = str(reference["profile"])
    if request.get("tissue_constraint") != "accepted_mask_nearest":
        raise ValueError("cell promotion requires the accepted tissue-mask constraint")
    if request.get("analysis_selection") not in {
        "all_registered",
        "registered_minus_manifest_exclusions",
    }:
        raise ValueError("cell promotion analysis selection is invalid")
    _validate_support(
        request,
        "multiscale_nuclear_support",
        optical_density=0.12,
        pixels=3,
        fraction=0.005,
    )
    _validate_support(
        request,
        "global_nuclear_support",
        optical_density=0.15,
        pixels=8,
        fraction=0.02,
    )
    instance_evidence = request.get("instance_evidence")
    if not isinstance(instance_evidence, dict) or (
        instance_evidence.get("method") != "local_optical_density"
    ):
        raise ValueError("cell promotion requires local debris/background evidence")
    source_context = request.get("source_tissue_context")
    if (
        not isinstance(source_context, dict)
        or source_context.get("enabled") is not True
    ):
        raise ValueError("cell promotion requires source tissue-context evidence")
    if (
        source_context.get("bin_size_px") != 256
        or not _at_least(source_context.get("minimum_stain_fraction"), 0.03)
        or not _at_least(source_context.get("minimum_instance_fraction"), 0.01)
    ):
        raise ValueError("cell promotion source tissue-context thresholds are unsafe")
    adaptive_core = request.get("adaptive_nuclear_core")
    if (
        not isinstance(adaptive_core, dict)
        or adaptive_core.get("enabled") is not True
        or adaptive_core.get("percentile") != 70.0
        or adaptive_core.get("minimum_area_um2") != 225.0
        or adaptive_core.get("minimum_pixels") != 8
        or adaptive_core.get("minimum_fraction") != 0.10
        or adaptive_core.get("minimum_blue_ratio") != 1.08
    ):
        raise ValueError(
            "cell promotion requires the validated adaptive nuclear-core gate"
        )
    isolated_debris = request.get("isolated_debris_gate")
    escape_policy_valid = (
        isolated_debris.get("allow_nuclear_escape") is False
        and isolated_debris.get("allow_organized_escape") is False
        if algorithm_version in CELL_NO_ESCAPE_ALGORITHM_VERSIONS
        and isinstance(isolated_debris, dict)
        else isinstance(isolated_debris, dict)
        and isolated_debris.get("allow_nuclear_escape") is True
        and "allow_organized_escape" not in isolated_debris
    )
    if (
        not isinstance(isolated_debris, dict)
        or isolated_debris.get("enabled") is not True
        or isolated_debris.get("context_bin_size_px") != 64
        or isolated_debris.get("context_minimum_stain_fraction") != 0.80
        or isolated_debris.get("context_minimum_instance_fraction") != 0.75
        or isolated_debris.get("component_minimum_nuclear_pixels") != 8
        or not escape_policy_valid
        or isolated_debris.get("nuclear_minimum_pixels") != 2
        or isolated_debris.get("nuclear_minimum_fraction") != 0.005
        or isolated_debris.get("nuclear_minimum_blue_ratio") != 1.08
        or isolated_debris.get("neutral_dark_maximum_value") != 90
        or isolated_debris.get("neutral_dark_maximum_chroma") != 30
        or isolated_debris.get("neutral_dark_maximum_fraction") != 0.50
        or isolated_debris.get("very_dark_maximum_value") != 60
        or isolated_debris.get("very_dark_maximum_chroma") != 20
        or isolated_debris.get("very_dark_maximum_fraction") != 0.45
        or isolated_debris.get("micro_bin_size_px") != 16
        or isolated_debris.get("micro_minimum_stain_fraction") != 0.80
        or isolated_debris.get("micro_maximum_component_bins") != 512
        or isolated_debris.get("micro_minimum_nuclear_fraction") != 0.01
        or isolated_debris.get("micro_minimum_instance_fraction") != 0.50
        or isolated_debris.get("organized_bin_size_px") != 4
        or isolated_debris.get("organized_window_size_px") != 1024
        or isolated_debris.get("organized_window_overlap_px") != 128
        or isolated_debris.get("organized_minimum_bin_occupancy_fraction") != 0.10
        or isolated_debris.get("organized_minimum_component_pixels") != 10_000
        or isolated_debris.get("organized_compact_minimum_component_pixels") != 20_000
        or isolated_debris.get("organized_minimum_aspect_ratio") != 4.0
        or isolated_debris.get("organized_minimum_mean_instance_pixels") != 350.0
        or isolated_debris.get("organized_minimum_instance_fraction") != 0.50
        or isolated_debris.get("organized_strong_red_blue_difference") != 40.0
        or isolated_debris.get("organized_minimum_strong_chromatic_fraction") != 0.25
        or isolated_debris.get("compact_unsupported_minimum_area_um2") != 10_000.0
        or isolated_debris.get("compact_unsupported_maximum_aspect_ratio") != 1.75
        or isolated_debris.get("compact_unsupported_maximum_mean_instance_pixels")
        != 250.0
        or isolated_debris.get("compact_unsupported_maximum_mean_red_blue_difference")
        != 21.0
        or isolated_debris.get("compact_unsupported_strong_red_blue_difference") != 40.0
        or isolated_debris.get("compact_unsupported_maximum_strong_chromatic_fraction")
        != 0.10
        or isolated_debris.get("compact_unsupported_elongation_ratio") != 2.0
        or isolated_debris.get(
            "compact_unsupported_maximum_elongated_instance_fraction"
        )
        != 0.50
        or isolated_debris.get("compact_unsupported_maximum_context_nuclear_fraction")
        != 0.20
        or isolated_debris.get("compact_unsupported_minimum_component_fill_fraction")
        != 0.20
        or isolated_debris.get("compact_unsupported_minimum_instance_fraction") != 0.50
        or isolated_debris.get("foam_minimum_mean_intensity") != 160.0
        or isolated_debris.get("foam_maximum_mean_instance_pixels") != 250.0
        or isolated_debris.get("foam_maximum_strong_chromatic_fraction") != 0.10
        or isolated_debris.get("foam_maximum_context_nuclear_fraction") != 0.50
        or isolated_debris.get("foam_maximum_elongated_instance_fraction") != 0.50
        or (
            isolated_debris.get("foam_core_minimum_pixels"),
            isolated_debris.get("foam_core_minimum_bright_fraction"),
            isolated_debris.get("foam_core_minimum_intensity"),
            isolated_debris.get("foam_core_maximum_instance_area_um2"),
            isolated_debris.get("foam_core_minimum_component_area_um2"),
            isolated_debris.get("foam_core_maximum_aspect_ratio"),
        )
        not in CELL_VALIDATED_FOAM_CORE_PROFILES
        or isolated_debris.get("oversized_chromatic_minimum_area_um2") != 1000.0
        or isolated_debris.get("oversized_chromatic_minimum_fraction") != 0.50
        or isolated_debris.get("fold_minimum_instance_area_um2") != 300.0
        or isolated_debris.get("fold_minimum_component_area_um2") != 750.0
        or (
            isolated_debris.get("fold_maximum_mean_red_blue_difference"),
            isolated_debris.get("fold_maximum_mean_intensity"),
        )
        not in {
            (40.0, 180.0),
            # Dark, chromatic fold profile: native-pixel visual audits showed
            # that this bounded pair removes extreme anuclear pseudo-cells in
            # crushed DAB tissue while the lower intensity ceiling protects
            # bright large cells and ordinary positive tissue.
            (85.0, 130.0),
        }
        or isolated_debris.get("fold_dense_minimum_instances") != 3
        or isolated_debris.get("fold_dense_minimum_aspect_ratio") != 3.0
        or isolated_debris.get("fold_dense_connectivity_dilation_bins") != 8
        or isolated_debris.get("fold_dense_maximum_mean_red_blue_difference") != 60.0
        or isolated_debris.get("fold_dense_maximum_mean_intensity") != 115.0
        or isolated_debris.get("detached_minimum_instance_area_um2") != 200.0
        or isolated_debris.get("detached_minimum_component_area_um2") != 1000.0
        or isolated_debris.get("detached_minimum_instances") != 3
        or isolated_debris.get("detached_minimum_aspect_ratio") != 3.0
        or isolated_debris.get("detached_maximum_mean_red_blue_difference") != 35.0
        or isolated_debris.get("detached_maximum_mean_intensity") != 220.0
        or isolated_debris.get("detached_context_window_size_px") != 512
        or isolated_debris.get("detached_maximum_prediction_fraction") != 0.20
        or isolated_debris.get("detached_compact_minimum_instance_area_um2") != 300.0
        or isolated_debris.get("detached_compact_minimum_component_area_um2") != 750.0
        or isolated_debris.get("detached_compact_minimum_instances") != 2
        or isolated_debris.get("detached_compact_maximum_mean_red_blue_difference")
        != 0.0
        or isolated_debris.get("detached_compact_maximum_mean_intensity") != 180.0
        or isolated_debris.get("glass_minimum_area_um2") != 25.0
        or isolated_debris.get("glass_context_window_size_px") != 256
        or isolated_debris.get("glass_maximum_prediction_fraction") != 0.40
        or isolated_debris.get("glass_maximum_context_stain_fraction") != 0.20
        or isolated_debris.get("glass_low_stain_maximum_mean_red_blue_difference")
        != 35.0
        or isolated_debris.get("glass_low_stain_minimum_mean_intensity") != 120.0
        or isolated_debris.get("glass_minimum_context_fraction") != 0.50
        or isolated_debris.get("glass_maximum_mean_red_blue_difference") != 25.0
        or isolated_debris.get("glass_minimum_mean_intensity") != 190.0
        or isolated_debris.get("necrotic_minimum_component_area_um2") != 5000.0
        or isolated_debris.get("necrotic_maximum_aspect_ratio") != 2.5
        or isolated_debris.get("necrotic_minimum_fill_fraction") != 0.05
        or isolated_debris.get("necrotic_maximum_fill_fraction") != 0.45
        or isolated_debris.get("necrotic_minimum_mean_red_blue_difference") != -15.0
        or isolated_debris.get("necrotic_maximum_mean_red_blue_difference") != 15.0
        or isolated_debris.get("necrotic_minimum_mean_intensity") != 160.0
        or isolated_debris.get("necrotic_minimum_instance_fraction") != 0.50
        or isolated_debris.get("necrotic_window_size_px") != 1024
        or isolated_debris.get("necrotic_window_overlap_px") != 128
        or isolated_debris.get("necrotic_context_window_size_px") != 256
        or isolated_debris.get("necrotic_maximum_local_prediction_fraction") != 0.40
        or isolated_debris.get("brown_minimum_component_area_um2") != 4.0
        or isolated_debris.get("brown_maximum_aspect_ratio") != 3.0
        or isolated_debris.get("brown_minimum_fill_fraction") != 0.12
        or isolated_debris.get("brown_maximum_fill_fraction") != 0.45
        or isolated_debris.get("brown_minimum_mean_red_blue_difference") != 50.0
        or isolated_debris.get("brown_maximum_mean_intensity") != 120.0
        or isolated_debris.get("brown_minimum_instance_fraction") != 0.50
        or isolated_debris.get("brown_requires_nuclear_unsupported") is not False
        or isolated_debris.get("brown_expands_to_source_component") is not True
        or isolated_debris.get("brown_window_size_px") != 1024
        or isolated_debris.get("brown_window_overlap_px") != 128
        or isolated_debris.get("brown_context_window_size_px") != 256
        or isolated_debris.get("brown_maximum_local_prediction_fraction") != 0.50
        or isolated_debris.get("brown_maximum_source_component_area_um2") != 50_000.0
        or isolated_debris.get("brown_source_maximum_mean_intensity") != 218.0
        or isolated_debris.get("brown_source_dilation_bins") != 0
    ):
        raise ValueError("cell promotion requires the validated isolated-debris gate")

    if algorithm_version == CELL_SATELLITE_PROTECTION_ALGORITHM_VERSION:
        section = str(result_rows[0]["section"]) if len(result_rows) == 1 else None
        expected_satellite_gate = {
            "enabled": True,
            "section": section,
            "scope": CELL_SATELLITE_PROTECTION_SCOPE,
            "interior_protection": {
                "minimum_mask_distance_um": (
                    CELL_SATELLITE_PROTECTION_MINIMUM_MASK_DISTANCE_UM
                ),
                "minimum_sample_fraction": (
                    CELL_SATELLITE_PROTECTION_MINIMUM_SAMPLE_FRACTION
                ),
            },
        }
        forbidden_refinements = (
            "self_dense_glass_gate",
            "oversized_brown_gate",
            "oversized_brown_maximum_mean_intensity",
            "clustered_brown_gate",
            "neutral_precipitate_gate",
            "reconcile_enclosed_cytoplasmic_children",
            "protect_organized_from_necrotic",
            "organized_necrotic_protection_maximum_instance_area_um2",
            "organized_necrotic_protection_minimum_local_prediction_fraction",
            "organized_necrotic_protection_local_prediction_source",
            "organized_necrotic_sparse_glass_shape_gate",
        )
        filter_upgrade = result.get("filter_upgrade")
        if (
            section is None
            or isolated_debris.get("satellite_gate") != expected_satellite_gate
            or any(key in isolated_debris for key in forbidden_refinements)
            or not isinstance(filter_upgrade, dict)
            or filter_upgrade.get("source_algorithm_version") != 75
            or not _is_sha256(filter_upgrade.get("source_result_fingerprint"))
            or filter_upgrade.get("scope") != CELL_SATELLITE_PROTECTION_SCOPE
        ):
            raise ValueError(
                "cell promotion requires the bounded v89 satellite interior protection"
            )
        expected_profile = _json_sha256(
            {
                "algorithm_version": CELL_SATELLITE_PROTECTION_ALGORITHM_VERSION,
                "preflight_fingerprint": result.get("preflight_fingerprint"),
                "model": result.get("model"),
                "request": request,
                "filter_upgrade": filter_upgrade,
            }
        )
        if result.get("profile_fingerprint") != expected_profile:
            raise ValueError("cell promotion v89 satellite profile is stale")
    elif "satellite_gate" in isolated_debris:
        raise ValueError("cell promotion satellite-gate policy is ambiguous")

    neutral_precipitate_source: dict[str, object] | None = None
    if algorithm_version == NEUTRAL_PRECIPITATE_ALGORITHM_VERSION:
        if (
            len(result_rows) != 1
            or isolated_debris.get("neutral_precipitate_gate") is not True
            or any(
                key in isolated_debris
                for key in NEUTRAL_PRECIPITATE_FORBIDDEN_REFINEMENTS
            )
            or "filter_upgrade" in result
        ):
            raise ValueError(
                "cell promotion requires the bounded v93 neutral-precipitate gate"
            )
        neutral_precipitate_source = validate_neutral_precipitate_source(
            result.get("subset_source")
        )
        expected_profile = _json_sha256(
            {
                "algorithm_version": NEUTRAL_PRECIPITATE_ALGORITHM_VERSION,
                "preflight_fingerprint": result.get("preflight_fingerprint"),
                "model": result.get("model"),
                "request": request,
            }
        )
        if result.get("profile_fingerprint") != expected_profile:
            raise ValueError("cell promotion v93 neutral-precipitate profile is stale")
    elif "neutral_precipitate_gate" in isolated_debris:
        raise ValueError("cell promotion neutral-precipitate policy is ambiguous")

    dense_recovery: dict[str, object] | None = None
    if algorithm_version in DENSE_SMALL_CELL_RECOVERY_ALGORITHM_VERSIONS:
        dense_recovery = _validated_dense_small_cell_recovery(
            request.get("dense_small_cell_recovery"),
            result.get("subset_source"),
            result_rows,
            algorithm_version=algorithm_version,
            method_profile=method_profile,
        )
        expected_profile = _json_sha256(
            {
                "algorithm_version": algorithm_version,
                "preflight_fingerprint": result.get("preflight_fingerprint"),
                "model": result.get("model"),
                "request": request,
            }
        )
        if result.get("profile_fingerprint") != expected_profile:
            raise ValueError("cell promotion dense-recovery profile is stale")
    elif "dense_small_cell_recovery" in request:
        raise ValueError("cell promotion dense-recovery policy is ambiguous")

    bounded_additive_recovery: dict[str, object] | None = None
    if algorithm_version == BOUNDED_ADDITIVE_RECOVERY_ALGORITHM_VERSION:
        if len(result_rows) != 1:
            raise ValueError(
                "cell promotion bounded addition must cover exactly one section"
            )
        bounded_additive_recovery = validate_bounded_additive_recovery_provenance(
            request.get("bounded_additive_recovery"),
            result.get("subset_source"),
            section=result_rows[0].get("section"),
            source_identity=result_rows[0].get("source_identity"),
        )
        expected_profile = _json_sha256(
            {
                "algorithm_version": algorithm_version,
                "preflight_fingerprint": result.get("preflight_fingerprint"),
                "model": result.get("model"),
                "request": request,
            }
        )
        if result.get("profile_fingerprint") != expected_profile:
            raise ValueError("cell promotion bounded-additive profile is stale")
    elif "bounded_additive_recovery" in request:
        raise ValueError("cell promotion bounded-additive policy is ambiguous")

    reviewed_cache_addition: dict[str, object] | None = None
    if algorithm_version in {
        REVIEWED_CACHE_ADDITION_ALGORITHM_VERSION,
        REVIEWED_ADDITION_EXCLUSION_ALGORITHM_VERSION,
    }:
        if len(result_rows) != 1:
            raise ValueError(
                "cell promotion reviewed cache addition must cover exactly one section"
            )
        section = str(result_rows[0].get("section"))
        manifest = request.get("reviewed_cache_addition")
        reviewed_rows = validate_reviewed_cache_addition_manifest(
            manifest,
            expected_sections=(section,),
        )
        reviewed_cache_addition = dict(manifest)  # type: ignore[arg-type]
        reviewed_row = reviewed_rows[section]
        subset = result.get("subset_source")
        if (
            not isinstance(subset, dict)
            or set(subset)
            != {
                "scope",
                "source_result_fingerprint",
                "source_preflight_fingerprint",
                "source_profile_fingerprint",
                "source_section_fingerprint",
                "source_labels_sha256",
            }
            or subset.get("scope") != REVIEWED_CACHE_ADDITION_SCOPE
            or subset.get("source_result_fingerprint")
            != reviewed_cache_addition.get("source_result_fingerprint")
            or subset.get("source_preflight_fingerprint")
            != reviewed_cache_addition.get("source_preflight_fingerprint")
            or subset.get("source_labels_sha256")
            != reviewed_row.get("source_labels_sha256")
            or reviewed_row.get("source_identity")
            != result_rows[0].get("source_identity")
            or any(
                not _is_sha256(subset.get(key))
                for key in (
                    "source_profile_fingerprint",
                    "source_section_fingerprint",
                )
            )
        ):
            raise ValueError("cell promotion reviewed cache source seal is stale")
        if algorithm_version == REVIEWED_CACHE_ADDITION_ALGORITHM_VERSION:
            expected_profile = _json_sha256(
                {
                    "algorithm_version": algorithm_version,
                    "preflight_fingerprint": result.get("preflight_fingerprint"),
                    "model": result.get("model"),
                    "request": request,
                }
            )
            if result.get("profile_fingerprint") != expected_profile:
                raise ValueError("cell promotion reviewed cache profile is stale")
    elif "reviewed_cache_addition" in request:
        raise ValueError("cell promotion reviewed cache policy is ambiguous")

    reviewed_multi_addition = None
    if algorithm_version == REVIEWED_MULTI_ADDITION_ALGORITHM_VERSION:
        reviewed_multi_addition = validate_reviewed_multi_addition_provenance(result)
    elif "reviewed_multi_cache_addition" in request:
        raise ValueError("cell promotion reviewed multi-cache policy is ambiguous")

    reviewed_input_addition = None
    if algorithm_version == REVIEWED_INPUT_ADDITION_ALGORITHM_VERSION:
        reviewed_input_addition = validate_reviewed_input_addition_provenance(result)
    elif "reviewed_input_addition" in request:
        raise ValueError("cell promotion reviewed new-input policy is ambiguous")

    focus_gate: dict[str, object] | None = None
    focus_source_algorithm: int | None = None
    if algorithm_version == 83:
        focus_gate = request.get("focus_artifact_gate")  # type: ignore[assignment]
        if focus_gate != _validated_focus_artifact_gate():
            raise ValueError("cell promotion requires the validated v83 focus gate")
        filter_upgrade = result.get("filter_upgrade")
        if (
            not isinstance(filter_upgrade, dict)
            or filter_upgrade.get("source_algorithm_version")
            not in CELL_VALIDATED_FOCUS_SOURCE_ALGORITHMS
            or not _is_sha256(filter_upgrade.get("source_result_fingerprint"))
            or filter_upgrade.get("scope")
            != "post-inference-focus-artifact-exclusion-v1"
        ):
            raise ValueError("cell promotion requires the validated v83 source seal")
        focus_source_algorithm = int(filter_upgrade["source_algorithm_version"])

    chromatic_gates: dict[str, dict[str, object]] = {}
    chromatic_source_algorithm: int | None = None
    if algorithm_version in CHROMATIC_MICROCLUSTER_ALGORITHM_VERSIONS:
        chromatic_gates = _validated_chromatic_gates(
            request,
            result_rows,
            algorithm_version=int(algorithm_version),
        )
        run_spec = chromatic_microcluster_run_spec(
            str(gate["profile"]) for gate in chromatic_gates.values()
        )
        filter_upgrade = result.get("filter_upgrade")
        if (
            not isinstance(filter_upgrade, dict)
            or filter_upgrade.get("source_algorithm_version")
            != run_spec.source_algorithm_version
            or not _is_sha256(filter_upgrade.get("source_result_fingerprint"))
            or filter_upgrade.get("scope") != run_spec.scope
        ):
            raise ValueError(
                "cell promotion requires the validated "
                f"v{run_spec.algorithm_version} chromatic source seal"
            )
        chromatic_source_algorithm = run_spec.source_algorithm_version
        expected_profile = _json_sha256(
            {
                "algorithm_version": run_spec.algorithm_version,
                "preflight_fingerprint": result.get("preflight_fingerprint"),
                "model": result.get("model"),
                "request": request,
                "filter_upgrade": filter_upgrade,
            }
        )
        if result.get("profile_fingerprint") != expected_profile:
            raise ValueError(
                f"cell promotion v{run_spec.algorithm_version} chromatic "
                "profile is stale"
            )

    foreign_material_gates: dict[str, dict[str, object]] = {}
    if algorithm_version == SATURATED_CYAN_FOREIGN_MATERIAL_ALGORITHM_VERSION:
        foreign_material_gates = _validated_foreign_material_gates(
            request,
            result_rows,
        )
        filter_upgrade = result.get("filter_upgrade")
        if (
            not isinstance(filter_upgrade, dict)
            or filter_upgrade.get("source_algorithm_version") != 75
            or not _is_sha256(filter_upgrade.get("source_result_fingerprint"))
            or filter_upgrade.get("scope") != SATURATED_CYAN_FOREIGN_MATERIAL_SCOPE
        ):
            raise ValueError(
                "cell promotion requires the validated v90 foreign-material source seal"
            )
        expected_profile = _json_sha256(
            {
                "algorithm_version": (
                    SATURATED_CYAN_FOREIGN_MATERIAL_ALGORITHM_VERSION
                ),
                "preflight_fingerprint": result.get("preflight_fingerprint"),
                "model": result.get("model"),
                "request": request,
                "filter_upgrade": filter_upgrade,
            }
        )
        if result.get("profile_fingerprint") != expected_profile:
            raise ValueError("cell promotion v90 foreign-material profile is stale")

    reviewed_exclusion_manifest: dict[str, object] | None = None
    reviewed_exclusion_rows: dict[str, dict[str, object]] = {}
    if algorithm_version in {
        REVIEWED_ARTIFACT_EXCLUSION_ALGORITHM_VERSION,
        REVIEWED_ADDITION_EXCLUSION_ALGORITHM_VERSION,
    }:
        payload = request.get("reviewed_artifact_exclusion")
        if not isinstance(payload, dict):
            raise ValueError(
                "cell promotion requires the validated v91 reviewed exclusion"
            )
        reviewed_exclusion_manifest = payload
        reviewed_exclusion_rows = validate_reviewed_artifact_exclusion_manifest(
            payload,
            expected_sections=(str(row.get("section")) for row in result_rows),
        )
        if algorithm_version == REVIEWED_ADDITION_EXCLUSION_ALGORITHM_VERSION:
            if reviewed_cache_addition is None:
                raise ValueError(
                    "combined reviewed output is missing its cache addition"
                )
            _validate_reviewed_addition_exclusion_chain(
                reviewed_cache_addition,
                payload,
            )
        for row in result_rows:
            section = str(row.get("section"))
            if reviewed_exclusion_rows[section].get("source_identity") != row.get(
                "source_identity"
            ):
                raise ValueError(
                    f"section {section}: reviewed source identity is stale"
                )
        filter_upgrade = result.get("filter_upgrade")
        expected_source_algorithm = (
            REVIEWED_ADDITION_EXCLUSION_SOURCE_ALGORITHM_VERSION
            if algorithm_version == REVIEWED_ADDITION_EXCLUSION_ALGORITHM_VERSION
            else REVIEWED_ARTIFACT_EXCLUSION_SOURCE_ALGORITHM_VERSION
        )
        if (
            payload.get("source_preflight_fingerprint")
            != result.get("preflight_fingerprint")
            or not isinstance(filter_upgrade, dict)
            or filter_upgrade.get("source_algorithm_version")
            != expected_source_algorithm
            or filter_upgrade.get("source_result_fingerprint")
            != payload.get("source_result_fingerprint")
            or filter_upgrade.get("scope") != REVIEWED_ARTIFACT_EXCLUSION_SCOPE
        ):
            raise ValueError(
                "cell promotion requires the validated v91 reviewed source seal"
            )
        expected_profile = _json_sha256(
            {
                "algorithm_version": algorithm_version,
                "preflight_fingerprint": result.get("preflight_fingerprint"),
                "model": result.get("model"),
                "request": request,
                "filter_upgrade": filter_upgrade,
            }
        )
        if result.get("profile_fingerprint") != expected_profile:
            raise ValueError(
                f"cell promotion v{algorithm_version} reviewed profile is stale"
            )

    if algorithm_version == CELL_ORGANIZED_ESCAPE_ALGORITHM_VERSION:
        filter_upgrade = result.get("filter_upgrade")
        if (
            not isinstance(filter_upgrade, dict)
            or filter_upgrade.get("source_algorithm_version") != 75
            or not _is_sha256(filter_upgrade.get("source_result_fingerprint"))
            or filter_upgrade.get("scope") != CELL_ORGANIZED_ESCAPE_SCOPE
        ):
            raise ValueError(
                "cell promotion requires the validated v87 raw-cache source seal"
            )

    if algorithm_version in {82, 84} or focus_source_algorithm == 82:
        sparse_glass = isolated_debris.get("organized_necrotic_sparse_glass_shape_gate")
        if (
            isolated_debris.get("self_dense_glass_gate") is not True
            or isolated_debris.get("oversized_brown_gate") is not True
            or isolated_debris.get("oversized_brown_maximum_mean_intensity") != 180.0
            or isolated_debris.get("reconcile_enclosed_cytoplasmic_children")
            is not True
            or isolated_debris.get("protect_organized_from_necrotic") is not True
            or isolated_debris.get(
                "organized_necrotic_protection_maximum_instance_area_um2"
            )
            != 500.0
            or isolated_debris.get(
                "organized_necrotic_protection_minimum_local_prediction_fraction"
            )
            != 0.025
            or isolated_debris.get(
                "organized_necrotic_protection_local_prediction_source"
            )
            != "preartifact_independent_nuclear_external"
            or sparse_glass
            != {
                "context_window_size_px": 128,
                "maximum_mean_intensity": 220.0,
                "minimum_tissue_fraction": 0.14,
                "maximum_sampled_fill_fraction": 0.40,
                "minimum_sampled_elongation": 2.50,
            }
        ):
            raise ValueError("cell promotion requires the validated v82 refilter gate")

    if algorithm_version == 84 and (
        isolated_debris.get("clustered_brown_gate") is not True
    ):
        raise ValueError(
            "cell promotion requires the validated v84 clustered-brown gate"
        )

    for row in result_rows:
        relative = row.get("qc")
        if not isinstance(relative, str):
            raise ValueError("cell promotion QC path is missing")
        qc = _object(root / relative, f"section {row.get('section')} QC")
        _validate_qc(
            row,
            qc,
            profile_fingerprint=result.get("profile_fingerprint"),
            algorithm_version=int(algorithm_version),
            method_profile=method_profile,
        )
        if dense_recovery is not None:
            _validate_dense_small_cell_recovery_qc(
                row,
                qc,
                dense_recovery,
            )
        if bounded_additive_recovery is not None:
            validate_bounded_additive_recovery_qc(
                qc,
                bounded_additive_recovery,
            )
        if reviewed_multi_addition is not None:
            validate_reviewed_multi_addition_qc(qc, reviewed_multi_addition)
        if reviewed_input_addition is not None:
            validate_reviewed_input_addition_qc(qc, reviewed_input_addition)
        if reviewed_cache_addition is not None:
            reviewed_source_labels = None
            reviewed_source_algorithm = None
            if algorithm_version == REVIEWED_ADDITION_EXCLUSION_ALGORITHM_VERSION:
                exclusion_row = reviewed_exclusion_rows.get(str(row.get("section")))
                if not isinstance(exclusion_row, dict):
                    raise ValueError("combined reviewed exclusion source is missing")
                reviewed_source_algorithm = (
                    REVIEWED_ADDITION_EXCLUSION_SOURCE_ALGORITHM_VERSION
                )
                reviewed_source_labels = str(exclusion_row.get("source_labels_sha256"))
            validate_reviewed_cache_addition_qc(
                qc,
                reviewed_cache_addition,
                section=str(row.get("section")),
                latest_filter_source_algorithm_version=(reviewed_source_algorithm),
                latest_filter_source_labels_sha256=reviewed_source_labels,
            )
        if neutral_precipitate_source is not None:
            validate_neutral_precipitate_qc(qc, neutral_precipitate_source)
        if focus_gate is not None:
            _validate_v83_qc(row, qc, focus_gate=focus_gate)
        section = str(row.get("section"))
        if section in chromatic_gates:
            assert chromatic_source_algorithm is not None
            _validate_chromatic_qc(
                row,
                qc,
                gate=chromatic_gates[section],
                source_algorithm_version=chromatic_source_algorithm,
                algorithm_version=int(algorithm_version),
            )
        if section in foreign_material_gates:
            _validate_foreign_material_qc(
                row,
                qc,
                gate=foreign_material_gates[section],
            )
        if section in reviewed_exclusion_rows:
            assert reviewed_exclusion_manifest is not None
            _validate_reviewed_exclusion_qc(
                row,
                qc,
                manifest=reviewed_exclusion_manifest,
                manifest_row=reviewed_exclusion_rows[section],
                source_algorithm_version=(
                    REVIEWED_ADDITION_EXCLUSION_SOURCE_ALGORITHM_VERSION
                    if algorithm_version
                    == REVIEWED_ADDITION_EXCLUSION_ALGORITHM_VERSION
                    else REVIEWED_ARTIFACT_EXCLUSION_SOURCE_ALGORITHM_VERSION
                ),
                algorithm_version=int(algorithm_version),
            )
    return result


def _validate_composite_candidate(
    root: Path,
    result: dict[str, object],
    *,
    result_rows: list[dict[str, object]],
    expected_model: str,
    expected_method: str,
) -> dict[str, object]:
    """Validate a complete result assembled from validated section sources."""

    from histopia.cells._composition import CELL_COMPOSITE_METHOD_PROFILE

    request = result.get("request")
    reference = request.get("method_reference") if isinstance(request, dict) else None
    if (
        not isinstance(request, dict)
        or request.get("model") != expected_model
        or request.get("method") != expected_method
        or request.get("analysis_selection")
        not in {"all_registered", "registered_minus_manifest_exclusions"}
        or request.get("tissue_constraint") != "accepted_mask_nearest"
        or request.get("composition") != "per-section-validated-method-selection-v1"
        or not isinstance(reference, dict)
        or reference.get("name") != CELL_METHOD_REFERENCE_NAME
        or reference.get("sha256") != CELL_METHOD_REFERENCE_SHA256
        or reference.get("profile") != CELL_COMPOSITE_METHOD_PROFILE
    ):
        raise ValueError("cell composite method provenance is stale")
    composition = result.get("composition")
    entries = (
        composition.get("section_sources") if isinstance(composition, dict) else None
    )
    if (
        not isinstance(composition, dict)
        or composition.get("schema_version") != 1
        or composition.get("scope") != "validated-section-composite-v1"
        or not _is_sha256(composition.get("base_result_fingerprint"))
        or not isinstance(entries, list)
        or len(entries) != len(result_rows)
        or any(not isinstance(entry, dict) for entry in entries)
        or not any(entry.get("replacement") is True for entry in entries)
    ):
        raise ValueError("cell composite section-source manifest is invalid")
    expected_profile = _json_sha256(
        {
            "algorithm_version": 85,
            "preflight_fingerprint": result.get("preflight_fingerprint"),
            "model": result.get("model"),
            "request": request,
            "composition": composition,
        }
    )
    if result.get("profile_fingerprint") != expected_profile:
        raise ValueError("cell composite profile fingerprint is stale")
    artifacts = result.get("artifacts")
    model = result.get("model")
    if not isinstance(artifacts, dict) or not isinstance(model, dict):
        raise ValueError("cell composite artifacts or model are invalid")
    by_section: dict[str, dict[str, object]] = {}
    for entry in entries:
        assert isinstance(entry, dict)
        section = entry.get("section")
        if not isinstance(section, str) or section in by_section:
            raise ValueError("cell composite section sources must be unique")
        by_section[section] = entry
    if list(by_section) != [row.get("section") for row in result_rows]:
        raise ValueError("cell composite section-source order is stale")

    base_fingerprint = composition["base_result_fingerprint"]
    for row in result_rows:
        section = str(row["section"])
        entry = by_section[section]
        algorithm_version = entry.get("source_algorithm_version")
        method_profile = entry.get("source_method_profile")
        profile_fingerprint = entry.get("source_profile_fingerprint")
        labels_relative = row.get("labels")
        qc_relative = row.get("qc")
        if (
            not isinstance(algorithm_version, int)
            or isinstance(algorithm_version, bool)
            or (algorithm_version, method_profile)
            not in CELL_VALIDATED_PROMOTION_PROFILES
            or not _is_sha256(profile_fingerprint)
            or not _is_sha256(entry.get("source_result_fingerprint"))
            or not _is_sha256(entry.get("source_preflight_fingerprint"))
            or not _is_sha256(entry.get("source_request_sha256"))
            or not _is_sha256(entry.get("source_model_sha256"))
            or not _is_sha256(entry.get("source_section_fingerprint"))
            or not _is_sha256(entry.get("source_labels_sha256"))
            or not _is_sha256(entry.get("source_qc_sha256"))
            or not isinstance(labels_relative, str)
            or not isinstance(qc_relative, str)
            or artifacts.get(labels_relative) != entry.get("source_labels_sha256")
            or artifacts.get(qc_relative) != entry.get("source_qc_sha256")
            or entry.get("model_weight_name") != model.get("weight_name")
            or entry.get("model_weight_sha256") != model.get("weight_sha256")
        ):
            raise ValueError(f"section {section}: composite source provenance is stale")
        replacement = entry.get("replacement")
        if not isinstance(replacement, bool) or (
            replacement is False
            and entry.get("source_result_fingerprint") != base_fingerprint
        ):
            raise ValueError(f"section {section}: composite base provenance is stale")
        qc = _object(root / qc_relative, f"section {section} QC")
        if qc.get("section_fingerprint") != entry.get("source_section_fingerprint"):
            raise ValueError(
                f"section {section}: composite section fingerprint is stale"
            )
        _validate_qc(
            row,
            qc,
            profile_fingerprint=profile_fingerprint,
            algorithm_version=algorithm_version,
            method_profile=str(method_profile),
        )
        if algorithm_version in DENSE_SMALL_CELL_RECOVERY_ALGORITHM_VERSIONS:
            dense_recovery = _validated_dense_small_cell_recovery(
                entry.get("dense_small_cell_recovery"),
                entry.get("subset_source"),
                [row],
                algorithm_version=algorithm_version,
                method_profile=str(method_profile),
            )
            _validate_dense_small_cell_recovery_qc(
                row,
                qc,
                dense_recovery,
            )
        if algorithm_version == BOUNDED_ADDITIVE_RECOVERY_ALGORITHM_VERSION:
            bounded_additive_recovery = validate_bounded_additive_recovery_provenance(
                entry.get("bounded_additive_recovery"),
                entry.get("subset_source"),
                section=section,
                source_identity=row.get("source_identity"),
            )
            validate_bounded_additive_recovery_qc(
                qc,
                bounded_additive_recovery,
            )
        if algorithm_version == REVIEWED_MULTI_ADDITION_ALGORITHM_VERSION:
            manifest = validate_reviewed_multi_addition_provenance(
                entry,
                composite_entry=True,
                expected_source_fingerprint=base_fingerprint,
            )
            manifest_row = manifest["sections"][0]
            if manifest_row["section"] != section or manifest_row[
                "source_identity"
            ] != row.get("source_identity"):
                raise ValueError("composite reviewed multi-cache section is stale")
            validate_reviewed_multi_addition_qc(qc, manifest)
        if algorithm_version == REVIEWED_INPUT_ADDITION_ALGORITHM_VERSION:
            manifest = validate_reviewed_input_addition_provenance(
                entry,
                composite_entry=True,
                expected_source_fingerprint=base_fingerprint,
            )
            manifest_row = manifest["sections"][0]
            if manifest_row["section"] != section or manifest_row[
                "source_identity"
            ] != row.get("source_identity"):
                raise ValueError("composite reviewed new-input section is stale")
            validate_reviewed_input_addition_qc(qc, manifest)
        if algorithm_version == REVIEWED_CACHE_ADDITION_ALGORITHM_VERSION:
            manifest = entry.get("reviewed_cache_addition")
            rows = validate_reviewed_cache_addition_manifest(
                manifest,
                expected_sections=(section,),
            )
            manifest_row = rows[section]
            subset = entry.get("subset_source")
            if (
                not isinstance(manifest, dict)
                or not isinstance(subset, dict)
                or manifest.get("source_result_fingerprint") != base_fingerprint
                or manifest.get("source_preflight_fingerprint")
                != entry.get("source_preflight_fingerprint")
                or manifest_row.get("source_identity") != row.get("source_identity")
                or subset.get("scope") != REVIEWED_CACHE_ADDITION_SCOPE
                or subset.get("source_result_fingerprint") != base_fingerprint
                or subset.get("source_preflight_fingerprint")
                != manifest.get("source_preflight_fingerprint")
                or subset.get("source_labels_sha256")
                != manifest_row.get("source_labels_sha256")
            ):
                raise ValueError(
                    f"section {section}: composite reviewed-cache provenance is stale"
                )
            exclusion_rows: dict[str, dict[str, object]] = {}
            if algorithm_version == REVIEWED_ADDITION_EXCLUSION_ALGORITHM_VERSION:
                exclusion_rows = validate_reviewed_artifact_exclusion_manifest(
                    entry.get("reviewed_artifact_exclusion"),
                    expected_sections=(section,),
                )
            validate_reviewed_cache_addition_qc(
                qc,
                manifest,
                section=section,
                latest_filter_source_algorithm_version=(
                    REVIEWED_ADDITION_EXCLUSION_SOURCE_ALGORITHM_VERSION
                    if algorithm_version
                    == REVIEWED_ADDITION_EXCLUSION_ALGORITHM_VERSION
                    else None
                ),
                latest_filter_source_labels_sha256=(
                    str(exclusion_rows[section]["source_labels_sha256"])
                    if exclusion_rows
                    else None
                ),
            )
        if algorithm_version == NEUTRAL_PRECIPITATE_ALGORITHM_VERSION:
            neutral_source = validate_neutral_precipitate_source(
                entry.get("subset_source")
            )
            if neutral_source.get("source_result_fingerprint") != base_fingerprint:
                raise ValueError(
                    f"section {section}: composite neutral-precipitate source "
                    "differs from base"
                )
            validate_neutral_precipitate_qc(qc, neutral_source)
        if algorithm_version == 83:
            focus_gate = entry.get("focus_artifact_gate")
            if focus_gate != _validated_focus_artifact_gate():
                raise ValueError(
                    f"section {section}: composite focus provenance is stale"
                )
            filter_upgrade = entry.get("filter_upgrade")
            if (
                not isinstance(filter_upgrade, dict)
                or filter_upgrade.get("source_algorithm_version")
                not in CELL_VALIDATED_FOCUS_SOURCE_ALGORITHMS
                or not _is_sha256(filter_upgrade.get("source_result_fingerprint"))
                or filter_upgrade.get("scope")
                != "post-inference-focus-artifact-exclusion-v1"
            ):
                raise ValueError(
                    f"section {section}: composite focus source seal is stale"
                )
            _validate_v83_qc(row, qc, focus_gate=focus_gate)
        if algorithm_version in CHROMATIC_MICROCLUSTER_ALGORITHM_VERSIONS:
            gate = entry.get("chromatic_microcluster_gate")
            profile = gate.get("profile") if isinstance(gate, dict) else None
            filter_upgrade = entry.get("filter_upgrade")
            run_spec = (
                chromatic_microcluster_run_spec([profile])
                if isinstance(profile, str)
                else None
            )
            if (
                not isinstance(profile, str)
                or profile not in CHROMATIC_MICROCLUSTER_POLICIES
                or gate != chromatic_microcluster_gate(profile)
                or run_spec is None
                or run_spec.algorithm_version != algorithm_version
                or not isinstance(filter_upgrade, dict)
                or filter_upgrade.get("source_algorithm_version")
                != run_spec.source_algorithm_version
                or not _is_sha256(filter_upgrade.get("source_result_fingerprint"))
                or filter_upgrade.get("scope") != run_spec.scope
            ):
                raise ValueError(
                    f"section {section}: composite chromatic provenance is stale"
                )
            _validate_chromatic_qc(
                row,
                qc,
                gate=gate,
                source_algorithm_version=run_spec.source_algorithm_version,
                algorithm_version=algorithm_version,
            )
        if algorithm_version == SATURATED_CYAN_FOREIGN_MATERIAL_ALGORITHM_VERSION:
            gate = entry.get("saturated_cyan_foreign_material_gate")
            filter_upgrade = entry.get("filter_upgrade")
            if (
                gate != saturated_cyan_foreign_material_gate()
                or not isinstance(filter_upgrade, dict)
                or filter_upgrade.get("source_algorithm_version") != 75
                or not _is_sha256(filter_upgrade.get("source_result_fingerprint"))
                or filter_upgrade.get("scope") != SATURATED_CYAN_FOREIGN_MATERIAL_SCOPE
            ):
                raise ValueError(
                    f"section {section}: composite foreign-material provenance is stale"
                )
            _validate_foreign_material_qc(row, qc, gate=gate)
        if algorithm_version in {
            REVIEWED_ARTIFACT_EXCLUSION_ALGORITHM_VERSION,
            REVIEWED_ADDITION_EXCLUSION_ALGORITHM_VERSION,
        }:
            reviewed_manifest = entry.get("reviewed_artifact_exclusion")
            if not isinstance(reviewed_manifest, dict):
                raise ValueError(
                    f"section {section}: composite reviewed provenance is stale"
                )
            reviewed_rows = validate_reviewed_artifact_exclusion_manifest(
                reviewed_manifest,
            )
            if algorithm_version == REVIEWED_ADDITION_EXCLUSION_ALGORITHM_VERSION:
                addition_manifest = entry.get("reviewed_cache_addition")
                if not isinstance(addition_manifest, dict):
                    raise ValueError(
                        f"section {section}: composite combined reviewed "
                        "addition is missing"
                    )
                _validate_reviewed_addition_exclusion_chain(
                    addition_manifest,
                    reviewed_manifest,
                )
            reviewed_row = reviewed_rows.get(section)
            filter_upgrade = entry.get("filter_upgrade")
            expected_source_algorithm = (
                REVIEWED_ADDITION_EXCLUSION_SOURCE_ALGORITHM_VERSION
                if algorithm_version == REVIEWED_ADDITION_EXCLUSION_ALGORITHM_VERSION
                else REVIEWED_ARTIFACT_EXCLUSION_SOURCE_ALGORITHM_VERSION
            )
            if (
                not isinstance(reviewed_row, dict)
                or reviewed_manifest.get("source_preflight_fingerprint")
                != entry.get("source_preflight_fingerprint")
                or reviewed_row.get("source_identity") != row.get("source_identity")
                or not isinstance(filter_upgrade, dict)
                or filter_upgrade.get("source_algorithm_version")
                != expected_source_algorithm
                or filter_upgrade.get("source_result_fingerprint")
                != reviewed_manifest.get("source_result_fingerprint")
                or filter_upgrade.get("scope") != REVIEWED_ARTIFACT_EXCLUSION_SCOPE
            ):
                raise ValueError(
                    f"section {section}: composite reviewed provenance is stale"
                )
            _validate_reviewed_exclusion_qc(
                row,
                qc,
                manifest=reviewed_manifest,
                manifest_row=reviewed_row,
                source_algorithm_version=expected_source_algorithm,
                algorithm_version=algorithm_version,
            )
        if algorithm_version == CELL_ORGANIZED_ESCAPE_ALGORITHM_VERSION:
            filter_upgrade = entry.get("filter_upgrade")
            if (
                not isinstance(filter_upgrade, dict)
                or filter_upgrade.get("source_algorithm_version") != 75
                or not _is_sha256(filter_upgrade.get("source_result_fingerprint"))
                or filter_upgrade.get("scope") != CELL_ORGANIZED_ESCAPE_SCOPE
            ):
                raise ValueError(
                    f"section {section}: composite organized-escape provenance is stale"
                )
        if algorithm_version == CELL_SATELLITE_PROTECTION_ALGORITHM_VERSION:
            filter_upgrade = entry.get("filter_upgrade")
            if (
                entry.get("satellite_gate")
                != {
                    "enabled": True,
                    "section": section,
                    "scope": CELL_SATELLITE_PROTECTION_SCOPE,
                    "interior_protection": {
                        "minimum_mask_distance_um": (
                            CELL_SATELLITE_PROTECTION_MINIMUM_MASK_DISTANCE_UM
                        ),
                        "minimum_sample_fraction": (
                            CELL_SATELLITE_PROTECTION_MINIMUM_SAMPLE_FRACTION
                        ),
                    },
                }
                or not isinstance(filter_upgrade, dict)
                or filter_upgrade.get("source_algorithm_version") != 75
                or not _is_sha256(filter_upgrade.get("source_result_fingerprint"))
                or filter_upgrade.get("scope") != CELL_SATELLITE_PROTECTION_SCOPE
            ):
                raise ValueError(
                    f"section {section}: composite satellite-gate provenance is stale"
                )
    return result


def _validated_chromatic_gates(
    request: dict[str, object],
    result_rows: list[dict[str, object]],
    *,
    algorithm_version: int,
) -> dict[str, dict[str, object]]:
    """Validate an exact per-section chromatic policy manifest."""

    payload = request.get("chromatic_microcluster_gate")
    entries = payload.get("sections") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or not isinstance(entries, list)
        or len(entries) != len(result_rows)
        or any(not isinstance(entry, dict) for entry in entries)
    ):
        raise ValueError(
            f"cell promotion requires the validated v{algorithm_version} chromatic gate"
        )
    output: dict[str, dict[str, object]] = {}
    for entry in entries:
        assert isinstance(entry, dict)
        section = entry.get("section")
        gate = entry.get("gate")
        profile = gate.get("profile") if isinstance(gate, dict) else None
        if (
            not isinstance(section, str)
            or section in output
            or not isinstance(profile, str)
            or profile not in CHROMATIC_MICROCLUSTER_POLICIES
            or gate != chromatic_microcluster_gate(profile)
            or chromatic_microcluster_run_spec([profile]).algorithm_version
            != algorithm_version
        ):
            raise ValueError(
                "cell promotion requires the validated "
                f"v{algorithm_version} chromatic gate"
            )
        output[section] = gate
    if list(output) != [row.get("section") for row in result_rows]:
        raise ValueError(
            f"cell promotion v{algorithm_version} chromatic section manifest is stale"
        )
    return output


def _validated_foreign_material_gates(
    request: dict[str, object],
    result_rows: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    """Validate the exact per-section saturated-cyan policy manifest."""

    payload = request.get("saturated_cyan_foreign_material_gate")
    entries = payload.get("sections") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or not isinstance(entries, list)
        or len(entries) != len(result_rows)
        or any(not isinstance(entry, dict) for entry in entries)
    ):
        raise ValueError(
            "cell promotion requires the validated v90 foreign-material gate"
        )
    expected_gate = saturated_cyan_foreign_material_gate()
    output: dict[str, dict[str, object]] = {}
    for entry in entries:
        assert isinstance(entry, dict)
        section = entry.get("section")
        gate = entry.get("gate")
        if not isinstance(section, str) or section in output or gate != expected_gate:
            raise ValueError(
                "cell promotion requires the validated v90 foreign-material gate"
            )
        assert isinstance(gate, dict)
        output[section] = gate
    if list(output) != [row.get("section") for row in result_rows]:
        raise ValueError("cell promotion v90 foreign-material manifest is stale")
    return output


def _validate_foreign_material_qc(
    row: dict[str, object],
    qc: dict[str, object],
    *,
    gate: dict[str, object],
) -> None:
    """Bind a saturated-cyan output to its source and exact removals."""

    section = row.get("section")
    candidate_count = qc.get("saturated_cyan_foreign_material_candidate_instances")
    removed_count = qc.get("saturated_cyan_foreign_material_instances_removed")
    removed_pixels = qc.get("saturated_cyan_foreign_material_pixels_removed")
    support_pixels = qc.get("saturated_cyan_foreign_material_support_pixels")
    evidence = qc.get("instance_evidence")
    if (
        qc.get("filter_upgrade_from_algorithm_version") != 75
        or not _is_sha256(qc.get("filter_upgrade_source_labels_sha256"))
        or qc.get("tiles_inferred") != 0
        or qc.get("tiles_reused_from_prior_run") != 0
        or not _nonnegative_int(candidate_count)
        or not _nonnegative_int(removed_count)
        or not _nonnegative_int(removed_pixels)
        or not _nonnegative_int(support_pixels)
        or int(removed_count) <= 0
        or int(removed_pixels) <= 0
        or int(support_pixels) <= 0
        or int(candidate_count) != int(removed_count)
        or int(support_pixels) > int(removed_pixels)
        or not isinstance(evidence, dict)
        or evidence.get("saturated_cyan_foreign_material_gate") != gate
    ):
        raise ValueError(f"section {section}: v90 foreign-material provenance is stale")


def _validate_reviewed_exclusion_qc(
    row: dict[str, object],
    qc: dict[str, object],
    *,
    manifest: dict[str, object],
    manifest_row: dict[str, object],
    source_algorithm_version: int = (
        REVIEWED_ARTIFACT_EXCLUSION_SOURCE_ALGORITHM_VERSION
    ),
    algorithm_version: int = REVIEWED_ARTIFACT_EXCLUSION_ALGORITHM_VERSION,
) -> None:
    """Bind an exact reviewed exclusion to its immutable source and evidence."""

    section = str(row.get("section"))
    label_ids = manifest_row.get("label_ids")
    evidence = qc.get("instance_evidence")
    reviewed_evidence = (
        evidence.get("reviewed_artifact_exclusion")
        if isinstance(evidence, dict)
        else None
    )
    expected_manifest_sha256 = reviewed_artifact_exclusion_manifest_sha256(manifest)
    expected_section_sha256 = reviewed_artifact_exclusion_section_sha256(manifest_row)
    expected_evidence = reviewed_artifact_exclusion_qc_evidence(
        manifest,
        section,
    )
    candidate_count = qc.get("reviewed_artifact_candidate_instances")
    removed_count = qc.get("reviewed_artifact_instances_removed")
    removed_pixels = qc.get("reviewed_artifact_pixels_removed")
    if (
        not isinstance(label_ids, list)
        or qc.get("filter_upgrade_from_algorithm_version") != source_algorithm_version
        or qc.get("filter_upgrade_source_labels_sha256")
        != manifest_row.get("source_labels_sha256")
        or qc.get("tiles_inferred") != 0
        or (
            qc.get("tiles_reused_from_prior_run") != 0
            if source_algorithm_version
            == REVIEWED_ARTIFACT_EXCLUSION_SOURCE_ALGORITHM_VERSION
            else (
                not _nonnegative_int(qc.get("tiles_reused_from_prior_run"))
                or int(qc["tiles_reused_from_prior_run"]) <= 0
            )
        )
        or not _nonnegative_int(candidate_count)
        or not _nonnegative_int(removed_count)
        or not _nonnegative_int(removed_pixels)
        or int(candidate_count) != len(label_ids)
        or int(removed_count) != len(label_ids)
        or int(removed_pixels) <= 0
        or qc.get("reviewed_artifact_manifest_sha256") != expected_manifest_sha256
        or qc.get("reviewed_artifact_section_sha256") != expected_section_sha256
        or qc.get("reviewed_artifact_label_ids_sha256")
        != manifest_row.get("label_ids_sha256")
        or reviewed_evidence != expected_evidence
    ):
        raise ValueError(
            f"section {section}: v{algorithm_version} reviewed-exclusion "
            "provenance is stale"
        )


def _validate_reviewed_addition_exclusion_chain(
    addition_manifest: dict[str, object],
    exclusion_manifest: dict[str, object],
) -> None:
    """Require a chained exclusion to preserve every reviewed added label."""

    addition_rows = validate_reviewed_cache_addition_manifest(addition_manifest)
    exclusion_rows = validate_reviewed_artifact_exclusion_manifest(
        exclusion_manifest,
        expected_sections=tuple(addition_rows),
    )
    if exclusion_manifest.get("source_algorithm_version") != (
        REVIEWED_ADDITION_EXCLUSION_SOURCE_ALGORITHM_VERSION
    ):
        raise ValueError("combined reviewed exclusion source algorithm is stale")
    for section, addition_row in addition_rows.items():
        selection = addition_row.get("selection")
        first_output_id = (
            selection.get("first_output_id") if isinstance(selection, dict) else None
        )
        excluded_ids = exclusion_rows[section].get("label_ids")
        if (
            not isinstance(first_output_id, int)
            or isinstance(first_output_id, bool)
            or not isinstance(excluded_ids, list)
            or any(int(label_id) >= first_output_id for label_id in excluded_ids)
        ):
            raise ValueError(
                f"section {section}: combined reviewed exclusion changes an "
                "approved cache addition"
            )


def _validate_chromatic_qc(
    row: dict[str, object],
    qc: dict[str, object],
    *,
    gate: dict[str, object],
    source_algorithm_version: int,
    algorithm_version: int,
) -> None:
    """Bind a chromatic output to its source and deterministic removals."""

    section = row.get("section")
    candidate_count = qc.get("chromatic_microcluster_candidate_instances")
    removed_count = qc.get("chromatic_microcluster_instances_removed")
    removed_pixels = qc.get("chromatic_microcluster_pixels_removed")
    evidence = qc.get("instance_evidence")
    if (
        qc.get("filter_upgrade_from_algorithm_version") != source_algorithm_version
        or not _is_sha256(qc.get("filter_upgrade_source_labels_sha256"))
        or qc.get("tiles_inferred") != 0
        or qc.get("tiles_reused_from_prior_run") != 0
        or not _nonnegative_int(candidate_count)
        or not _nonnegative_int(removed_count)
        or not _nonnegative_int(removed_pixels)
        or int(removed_count) <= 0
        or int(removed_pixels) <= 0
        or int(candidate_count) < int(removed_count)
        or not isinstance(evidence, dict)
        or evidence.get("chromatic_microcluster_gate") != gate
    ):
        raise ValueError(
            f"section {section}: v{algorithm_version} chromatic provenance is stale"
        )


def _validated_focus_artifact_gate() -> dict[str, object]:
    """Return the exact visually validated compact-defocus policy."""

    return {
        "enabled": True,
        "analysis_downsample": 4.0,
        "native_block_size": 128,
        "maximum_mean_intensity": 120.0,
        "minimum_dark_fraction": 0.30,
        "dark_intensity_threshold": 100.0,
        "maximum_block_laplacian_variance": 1500.0,
        "minimum_tissue_fraction": 0.25,
        "minimum_component_blocks": 12,
        "minimum_component_fill_fraction": 0.70,
        "maximum_component_aspect_ratio": 2.0,
        "maximum_median_laplacian_variance": 700.0,
        "growth_maximum_smoothed_intensity": 145.0,
        "growth_gaussian_sigma_px": 8.0,
        "growth_minimum_seed_pixels": 256,
        "instance_action": "remove_whole_labels_touching_native_growth_region",
    }


def _validate_v83_qc(
    row: dict[str, object],
    qc: dict[str, object],
    *,
    focus_gate: dict[str, object],
) -> None:
    """Bind every v83 section to its validated source and exact focus evidence."""

    section = row.get("section")
    components = qc.get("focus_artifact_components")
    component_count = qc.get("focus_artifact_components_excluded")
    instances_removed = qc.get("focus_artifact_instances_removed")
    pixels_removed = qc.get("focus_artifact_pixels_removed")
    evidence = qc.get("instance_evidence")
    if (
        qc.get("filter_upgrade_from_algorithm_version")
        not in CELL_VALIDATED_FOCUS_SOURCE_ALGORITHMS
        or not _is_sha256(qc.get("filter_upgrade_source_labels_sha256"))
        or qc.get("tiles_inferred") != 0
        or qc.get("tiles_reused_from_prior_run") != 0
        or not isinstance(components, list)
        or not _nonnegative_int(component_count)
        or component_count != len(components)
        or not _nonnegative_int(instances_removed)
        or not _nonnegative_int(pixels_removed)
        or not isinstance(evidence, dict)
        or evidence.get("focus_artifact_gate") != focus_gate
    ):
        raise ValueError(f"section {section}: v83 focus provenance is stale")
    for component in components:
        if not isinstance(component, dict):
            raise ValueError(f"section {section}: v83 focus component is invalid")
        blocks = component.get("blocks")
        block_indices = component.get("block_indices_rc")
        if (
            not isinstance(blocks, int)
            or isinstance(blocks, bool)
            or blocks < 12
            or not isinstance(block_indices, list)
            or len(block_indices) != blocks
            or any(not _integer_pair(value) for value in block_indices)
            or not _xyxy(component.get("native_content_bbox_xyxy"))
            or not _xyxy(component.get("native_growth_bbox_xyxy"))
            or not _between(component.get("fill_fraction"), 0.70, 1.0)
            or not _between(component.get("aspect_ratio"), 1.0, 2.0)
            or not _between(component.get("median_laplacian_variance"), 0.0, 700.0)
            or not _at_least(component.get("seed_pixels"), 256)
            or not _at_least(component.get("grown_pixels"), 1)
            or not _nonnegative_int(component.get("touching_labels"))
        ):
            raise ValueError(f"section {section}: v83 focus component is invalid")


def _integer_pair(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(
            isinstance(item, int) and not isinstance(item, bool) and item >= 0
            for item in value
        )
    )


def _xyxy(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 4
        and all(
            isinstance(item, int) and not isinstance(item, bool) and item >= 0
            for item in value
        )
        and value[2] > value[0]
        and value[3] > value[1]
    )


def _between(value: object, minimum: float, maximum: float) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and minimum <= float(value) <= maximum
    )


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _validate_support(
    request: dict[str, object],
    key: str,
    *,
    optical_density: float,
    pixels: int,
    fraction: float,
) -> None:
    support = request.get(key)
    if not isinstance(support, dict) or support.get("enabled") is not True:
        raise ValueError(f"cell promotion requires {key}")
    if (
        not _at_least(support.get("minimum_optical_density"), optical_density)
        or not _at_least(support.get("minimum_pixels"), pixels)
        or not _at_least(support.get("minimum_fraction"), fraction)
    ):
        raise ValueError(f"cell promotion {key} threshold is too permissive")


def _validated_dense_small_cell_recovery(
    value: object,
    subset_source: object,
    result_rows: list[dict[str, object]],
    *,
    algorithm_version: int,
    method_profile: str,
) -> dict[str, object]:
    """Validate path-free provenance for one additive dense-cell recovery."""

    if len(result_rows) != 1 or not isinstance(value, dict):
        raise ValueError("cell promotion dense recovery must cover exactly one section")
    row = result_rows[0]
    section = row.get("section")
    source_identity = row.get("source_identity")
    model = value.get("model")
    parameters = value.get("parameters")
    tiles = value.get("tiles")
    method = value.get("method")
    try:
        expected_profile = dense_small_cell_recovery_profile(method)
        expected_parameters = dense_small_cell_recovery_parameters_for_method(method)
    except ValueError as error:
        raise ValueError("cell promotion dense-recovery provenance is stale") from error
    try:
        validate_dense_small_cell_recovery_model(method, model)
    except ValueError as error:
        raise ValueError("cell promotion dense-recovery provenance is stale") from error
    if (
        value.get("schema_version") != 1
        or expected_profile != (algorithm_version, method_profile)
        or not _is_sha256(value.get("manifest_fingerprint"))
        or not _is_sha256(value.get("source_preflight_fingerprint"))
        or value.get("source_section") != section
        or value.get("source_identity") != source_identity
        or not _is_sha256(value.get("source_section_qc_sha256"))
        or not _is_sha256(value.get("source_labels_sha256"))
        or parameters != expected_parameters
        or not isinstance(model, dict)
        or not isinstance(tiles, list)
        or not tiles
        or any(not isinstance(tile, dict) for tile in tiles)
    ):
        raise ValueError("cell promotion dense-recovery provenance is stale")
    keys: list[str] = []
    for raw_tile in tiles:
        assert isinstance(raw_tile, dict)
        key = raw_tile.get("tile")
        shape = raw_tile.get("shape")
        if (
            not isinstance(key, str)
            or _DENSE_SMALL_CELL_TILE_KEY.fullmatch(key) is None
            or not _is_sha256(raw_tile.get("source_cache_sha256"))
            or not _is_sha256(raw_tile.get("source_tile_fingerprint"))
            or not _is_sha256(raw_tile.get("recovery_fingerprint"))
            or not _is_sha256(raw_tile.get("file_sha256"))
            or not isinstance(shape, list)
            or len(shape) != 2
            or any(
                not isinstance(dimension, int)
                or isinstance(dimension, bool)
                or dimension <= 0
                for dimension in shape
            )
        ):
            raise ValueError("cell promotion dense-recovery tile seal is stale")
        keys.append(key)
    if len(set(keys)) != len(keys) or keys != sorted(keys):
        raise ValueError("cell promotion dense-recovery tile order is stale")
    if not isinstance(subset_source, dict) or (
        subset_source.get("scope") != "cache-only-refilter-v1"
        or subset_source.get("source_labels_sha256")
        != value.get("source_labels_sha256")
        or not _is_sha256(subset_source.get("source_section_fingerprint"))
    ):
        raise ValueError("cell promotion dense-recovery source binding is stale")
    sealed_source = _is_sha256(subset_source.get("source_result_fingerprint"))
    active_source = subset_source.get("source_preflight_fingerprint") == value.get(
        "source_preflight_fingerprint"
    ) and _is_sha256(subset_source.get("source_profile_fingerprint"))
    if not sealed_source and not active_source:
        raise ValueError("cell promotion dense-recovery source seal is stale")
    return dict(value)


def _validate_dense_small_cell_recovery_qc(
    row: dict[str, object],
    qc: dict[str, object],
    recovery: dict[str, object],
) -> None:
    """Require cache-only assembly of every declared recovery tile."""

    tiles = recovery["tiles"]
    assert isinstance(tiles, list)
    reused = qc.get("tiles_reused_from_prior_run")
    if (
        qc.get("tiles_inferred") != 0
        or not _nonnegative_int(reused)
        or int(reused) <= 0
        or qc.get("dense_small_cell_recovery_tiles") != len(tiles)
        or qc.get("dense_small_cell_recovery_manifest_fingerprint")
        != recovery.get("manifest_fingerprint")
    ):
        raise ValueError(
            f"section {row.get('section')}: dense-recovery QC provenance is stale"
        )


def _validate_qc(
    row: dict[str, object],
    qc: dict[str, object],
    *,
    profile_fingerprint: object,
    algorithm_version: int,
    method_profile: str,
) -> None:
    section = row.get("section")
    if (
        qc.get("algorithm_version") != algorithm_version
        or qc.get("method_profile") != method_profile
        or not isinstance(profile_fingerprint, str)
        or qc.get("profile_fingerprint") != profile_fingerprint
    ):
        raise ValueError(f"section {section}: QC method provenance is stale")
    if qc.get("outside_tissue_pixels_final") != 0:
        raise ValueError(f"section {section}: promoted labels leave the tissue mask")
    cell_count = qc.get("cell_count")
    if (
        not isinstance(cell_count, int)
        or isinstance(cell_count, bool)
        or cell_count <= 0
        or row.get("cell_count") != cell_count
    ):
        raise ValueError(f"section {section}: cell count is invalid or stale")
    counters = (
        "tiles_total",
        "tiles_inferred",
        "tiles_reused",
        "tiles_skipped_outside_tissue",
        "post_constraint_small_instances_removed",
        "flat_background_instances_removed",
        "nuclear_unsupported_instances_removed",
        "source_context_unsupported_instances_removed",
        "source_context_unsupported_pixels_removed",
        "adaptive_nuclear_core_unsupported_instances_removed",
        "adaptive_nuclear_core_unsupported_pixels_removed",
        "isolated_debris_unsupported_instances_removed",
        "isolated_debris_unsupported_pixels_removed",
        "micro_island_debris_instances_removed",
        "micro_island_debris_pixels_removed",
        "organized_tissue_escape_instances",
        "organized_tissue_escape_pixels",
        "compact_unsupported_mosaic_instances_removed",
        "compact_unsupported_mosaic_pixels_removed",
        "foam_mosaic_instances_removed",
        "foam_mosaic_pixels_removed",
        "oversized_chromatic_artifact_instances_removed",
        "oversized_chromatic_artifact_pixels_removed",
        "fold_artifact_instances_removed",
        "fold_artifact_pixels_removed",
        "detached_fragment_instances_removed",
        "detached_fragment_pixels_removed",
        "satellite_debris_instances_removed",
        "satellite_debris_pixels_removed",
        "glass_artifact_instances_removed",
        "glass_artifact_pixels_removed",
        "necrotic_artifact_instances_removed",
        "necrotic_artifact_pixels_removed",
        "brown_debris_artifact_instances_removed",
        "brown_debris_artifact_pixels_removed",
    )
    if any(not _nonnegative_int(qc.get(key)) for key in counters):
        raise ValueError(f"section {section}: required QC counters are invalid")
    evidence = qc.get("instance_evidence")
    if not isinstance(evidence, dict) or (
        evidence.get("isolated_debris_satellite_maximum_instance_area_um2") != 300.0
        or evidence.get("isolated_debris_satellite_minimum_component_area_um2") != 500.0
        or evidence.get("isolated_debris_satellite_minimum_instances") != 3
        or evidence.get("isolated_debris_satellite_connectivity_dilation_bins") != 8
        or evidence.get("isolated_debris_satellite_maximum_nuclear_fraction") != 0.05
        or evidence.get("isolated_debris_satellite_minimum_mean_red_blue_difference")
        != 5.0
        or evidence.get("isolated_debris_satellite_maximum_mean_red_blue_difference")
        != 70.0
        or evidence.get("isolated_debris_satellite_minimum_mean_intensity") != 90.0
        or evidence.get("isolated_debris_satellite_maximum_mean_intensity") != 220.0
        or evidence.get("isolated_debris_satellite_context_window_size_px") != 256
        or evidence.get("isolated_debris_satellite_maximum_prediction_fraction") != 0.35
    ):
        raise ValueError(f"section {section}: satellite-debris safeguards are stale")
    if algorithm_version in CELL_NO_ESCAPE_ALGORITHM_VERSIONS and (
        qc.get("tiles_inferred") != 0
        or qc.get("organized_tissue_escape_instances") != 0
        or qc.get("organized_tissue_escape_pixels") != 0
        or evidence.get("isolated_debris_allow_nuclear_escape") is not False
        or evidence.get("isolated_debris_allow_organized_escape") is not False
    ):
        raise ValueError(f"section {section}: no-escape provenance is stale")
    if algorithm_version == CELL_ORGANIZED_ESCAPE_ALGORITHM_VERSION and (
        not _nonnegative_int(qc.get("tiles_reused_from_prior_run"))
        or int(qc["tiles_reused_from_prior_run"]) <= 0
    ):
        raise ValueError(f"section {section}: v87 raw-cache provenance is stale")
    if algorithm_version == CELL_SATELLITE_PROTECTION_ALGORITHM_VERSION:
        mask_shape = evidence.get("isolated_debris_satellite_mask_shape")
        if (
            qc.get("tiles_inferred") != 0
            or not _nonnegative_int(qc.get("tiles_reused_from_prior_run"))
            or int(qc["tiles_reused_from_prior_run"]) <= 0
            or not _nonnegative_int(qc.get("satellite_interior_instances_protected"))
            or int(qc["satellite_interior_instances_protected"]) <= 0
            or not _nonnegative_int(qc.get("satellite_interior_pixels_protected"))
            or int(qc["satellite_interior_pixels_protected"]) <= 0
            or qc.get("satellite_debris_instances_removed", 0) <= 0
            or qc.get("satellite_debris_pixels_removed", 0) <= 0
            or evidence.get("isolated_debris_satellite_gate_enabled") is not True
            or evidence.get("isolated_debris_satellite_minimum_mask_distance_um")
            != CELL_SATELLITE_PROTECTION_MINIMUM_MASK_DISTANCE_UM
            or evidence.get("isolated_debris_satellite_minimum_sample_fraction")
            != CELL_SATELLITE_PROTECTION_MINIMUM_SAMPLE_FRACTION
            or not isinstance(mask_shape, list)
            or len(mask_shape) != 2
            or any(
                not _nonnegative_int(value) or int(value) <= 0 for value in mask_shape
            )
            or evidence.get("isolated_debris_satellite_mpp_xy") != row.get("mpp_xy")
        ):
            raise ValueError(
                f"section {section}: v89 satellite-gate provenance is stale"
            )
    elif evidence.get("isolated_debris_satellite_gate_enabled") not in {None, True}:
        raise ValueError(f"section {section}: satellite-gate provenance is stale")
    if algorithm_version == 84 and (
        not _nonnegative_int(qc.get("clustered_brown_fragment_instances_removed"))
        or not _nonnegative_int(qc.get("clustered_brown_fragment_pixels_removed"))
        or evidence.get("isolated_debris_clustered_brown_gate") is not True
        or evidence.get("isolated_debris_clustered_brown_minimum_instance_area_um2")
        != 225.0
        or evidence.get("isolated_debris_clustered_brown_maximum_instance_area_um2")
        != 1_200.0
        or evidence.get("isolated_debris_clustered_brown_minimum_component_area_um2")
        != 1_500.0
        or evidence.get("isolated_debris_clustered_brown_minimum_instances") != 4
        or evidence.get("isolated_debris_clustered_brown_connectivity_dilation_bins")
        != 12
        or evidence.get("isolated_debris_clustered_brown_maximum_nuclear_fraction")
        != 0.02
        or evidence.get("isolated_debris_clustered_brown_minimum_red_green_difference")
        != 5.0
        or evidence.get("isolated_debris_clustered_brown_minimum_red_blue_difference")
        != 30.0
        or evidence.get("isolated_debris_clustered_brown_maximum_mean_intensity")
        != 175.0
        or evidence.get("isolated_debris_clustered_brown_context_window_size_px") != 256
        or evidence.get("isolated_debris_clustered_brown_maximum_prediction_fraction")
        != 0.50
        or evidence.get(
            "isolated_debris_clustered_brown_minimum_sparse_sample_fraction"
        )
        != 0.75
        or qc["clustered_brown_fragment_instances_removed"]
        > qc["detached_fragment_instances_removed"]
        or qc["clustered_brown_fragment_pixels_removed"]
        > qc["detached_fragment_pixels_removed"]
    ):
        raise ValueError(f"section {section}: clustered-brown safeguards are stale")
    if qc["tiles_total"] != (
        qc["tiles_inferred"] + qc["tiles_reused"] + qc["tiles_skipped_outside_tissue"]
    ):
        raise ValueError(f"section {section}: tile accounting is incomplete")
    if qc.get("post_constraint_cells_below_min_size") != 0:
        raise ValueError(
            f"section {section}: promoted labels retain sub-minimum fragments"
        )


def _object(path: Path, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text())
    except (FileNotFoundError, OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is unavailable") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be an object")
    return payload


def _rows(payload: dict[str, object], key: str, label: str) -> list[dict[str, object]]:
    rows = payload.get(key)
    if (
        not isinstance(rows, list)
        or not rows
        or any(not isinstance(row, dict) for row in rows)
    ):
        raise ValueError(f"{label} slide rows are invalid")
    return rows  # type: ignore[return-value]


def _at_least(value: object, minimum: float) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and float(value) >= minimum
    )


def _nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0
