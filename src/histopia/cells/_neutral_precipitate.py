"""Validated sparse neutral-precipitate cell-filter provenance.

The v93 rescue is deliberately narrow.  It reuses an already sealed native
cell label image and removes only small, very dark, approximately neutral
objects in sparse tissue context.  It does not rerun segmentation, enable the
broader clustered-brown policy, or infer any new cells.
"""

NEUTRAL_PRECIPITATE_ALGORITHM_VERSION = 93
NEUTRAL_PRECIPITATE_METHOD_PROFILE = "combined-containment-cpsam-wsi-v88"
NEUTRAL_PRECIPITATE_SOURCE_ALGORITHM_VERSION = 75
NEUTRAL_PRECIPITATE_SOURCE_SCOPE = "cache-only-refilter-v1"

NEUTRAL_PRECIPITATE_QC_EVIDENCE = {
    "isolated_debris_neutral_precipitate_gate": True,
    "isolated_debris_neutral_precipitate_minimum_core_fraction": 0.28,
    "isolated_debris_neutral_precipitate_minimum_very_dark_fraction": 0.20,
    "isolated_debris_neutral_precipitate_maximum_nuclear_fraction": 0.35,
    "isolated_debris_neutral_precipitate_mean_red_blue_range": [-10.0, 15.0],
    "isolated_debris_neutral_precipitate_maximum_mean_intensity": 135.0,
    "isolated_debris_neutral_precipitate_context_window_size_px": 256,
    "isolated_debris_neutral_precipitate_maximum_prediction_fraction": 0.25,
    "isolated_debris_neutral_precipitate_minimum_sparse_sample_fraction": 0.50,
}

# These refinements belong to other, independently reviewed rescue profiles.
# Their absence makes v93 a neutral-precipitate-only change to the v75 policy.
NEUTRAL_PRECIPITATE_FORBIDDEN_REFINEMENTS = frozenset(
    {
        "self_dense_glass_gate",
        "oversized_brown_gate",
        "oversized_brown_maximum_mean_intensity",
        "clustered_brown_gate",
        "satellite_gate",
        "diffuse_degenerated_gate",
        "reconcile_enclosed_cytoplasmic_children",
        "protect_organized_from_necrotic",
        "organized_necrotic_protection_maximum_instance_area_um2",
        "organized_necrotic_protection_minimum_local_prediction_fraction",
        "organized_necrotic_protection_local_prediction_source",
        "organized_necrotic_sparse_glass_shape_gate",
    }
)


def validate_neutral_precipitate_source(value: object) -> dict[str, object]:
    """Validate path-free binding to the exact sealed source section."""

    if not isinstance(value, dict) or (
        value.get("scope") != NEUTRAL_PRECIPITATE_SOURCE_SCOPE
        or not _is_sha256(value.get("source_result_fingerprint"))
        or not _is_sha256(value.get("source_section_fingerprint"))
        or not _is_sha256(value.get("source_labels_sha256"))
    ):
        raise ValueError("cell promotion neutral-precipitate source seal is stale")
    return dict(value)


def validate_neutral_precipitate_qc(
    qc: dict[str, object],
    source: dict[str, object],
) -> None:
    """Require a cache-only run and the exact reviewed neutral-dark policy."""

    evidence = qc.get("instance_evidence")
    minimum_pixels = (
        evidence.get("isolated_debris_neutral_precipitate_minimum_instance_area_pixels")
        if isinstance(evidence, dict)
        else None
    )
    maximum_pixels = (
        evidence.get("isolated_debris_neutral_precipitate_maximum_instance_area_pixels")
        if isinstance(evidence, dict)
        else None
    )
    glass_minimum = (
        evidence.get("isolated_debris_glass_minimum_area_pixels")
        if isinstance(evidence, dict)
        else None
    )
    satellite_maximum = (
        evidence.get("isolated_debris_satellite_maximum_instance_area_pixels")
        if isinstance(evidence, dict)
        else None
    )
    removed_instances = qc.get("neutral_precipitate_instances_removed")
    removed_pixels = qc.get("neutral_precipitate_pixels_removed")
    if (
        qc.get("tiles_inferred") != 0
        or not _positive_int(qc.get("tiles_reused_from_prior_run"))
        or not _positive_int(removed_instances)
        or not _positive_int(removed_pixels)
        or not isinstance(evidence, dict)
        or any(
            evidence.get(key) != expected
            for key, expected in NEUTRAL_PRECIPITATE_QC_EVIDENCE.items()
        )
        or not _positive_int(minimum_pixels)
        or not _positive_int(maximum_pixels)
        or minimum_pixels != glass_minimum
        or not _positive_int(satellite_maximum)
        or maximum_pixels != int(satellite_maximum) * 2
        or not _is_sha256(qc.get("labels_sha256"))
        or qc.get("filter_upgrade_from_algorithm_version") is not None
        or qc.get("filter_upgrade_source_labels_sha256") is not None
        or source.get("source_labels_sha256") == qc.get("labels_sha256")
    ):
        raise ValueError("cell promotion neutral-precipitate QC provenance is stale")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0
