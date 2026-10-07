"""Conservative chromatic-microobject safeguards for cell boundaries.

Some strongly chromogenic stains can turn compact anucleate material into a
dense set of plausible Cellpose instances. A single small prediction is not
enough evidence for deletion: lymphocytes and small epithelial profiles are
real. This module therefore combines physical size, counterstain support, and
spatial clustering. It is intended as a post-inference safeguard for a
visually audited stain-specific rescue, not as a replacement for segmentation.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

CHROMATIC_MICROCLUSTER_ALGORITHM_VERSION = 86
CHROMATIC_MICROCLUSTER_METHOD_PROFILE = "combined-containment-cpsam-wsi-v81"
CHROMATIC_MICROCLUSTER_SCOPE = "post-inference-chromatic-microcluster-exclusion-v1"
PAS_MICROCLUSTER_ALGORITHM_VERSION = 88
PAS_MICROCLUSTER_METHOD_PROFILE = "combined-containment-cpsam-wsi-v83"
PAS_MICROCLUSTER_SCOPE = "post-inference-pas-anucleate-microcluster-exclusion-v1"
CHROMATIC_MICROCLUSTER_BLUE_RATIO = 1.08


@dataclass(frozen=True, slots=True)
class ChromaticMicroclusterPolicy:
    """Exact thresholds for anucleate chromatic microobject clusters."""

    maximum_instance_area_pixels: int
    minimum_counterstain_fraction: float
    maximum_centroid_distance_pixels: float
    minimum_component_area_pixels: int
    minimum_component_instances: int
    maximum_component_aspect_ratio: float
    minimum_component_fill_fraction: float

    def __post_init__(self) -> None:
        if self.maximum_instance_area_pixels <= 0:
            raise ValueError("maximum instance area must be positive")
        if not 0 < self.minimum_counterstain_fraction <= 1:
            raise ValueError("counterstain fraction must be in (0, 1]")
        if self.maximum_centroid_distance_pixels <= 0:
            raise ValueError("centroid distance must be positive")
        if self.minimum_component_area_pixels <= 0:
            raise ValueError("component area must be positive")
        if self.minimum_component_instances < 2:
            raise ValueError("component instance count must be at least two")
        if self.maximum_component_aspect_ratio < 1:
            raise ValueError("component aspect ratio must be at least one")
        if not 0 < self.minimum_component_fill_fraction <= 1:
            raise ValueError("component fill fraction must be in (0, 1]")


@dataclass(frozen=True, slots=True)
class ChromaticMicroclusterRunSpec:
    """Immutable result provenance for one family of chromatic policies."""

    algorithm_version: int
    method_profile: str
    scope: str
    source_algorithm_version: int


CHROMATIC_MICROCLUSTER_POLICIES = {
    # Visually bounded on Prrx1 section 4714/022 after the strict bright-foam
    # gate. Only very small, essentially counterstain-free objects in populous
    # compact residual clusters are removed.
    "residual-foam-microobjects-v1": ChromaticMicroclusterPolicy(
        maximum_instance_area_pixels=120,
        minimum_counterstain_fraction=0.01,
        maximum_centroid_distance_pixels=40.0,
        minimum_component_area_pixels=300,
        minimum_component_instances=12,
        maximum_component_aspect_ratio=3.0,
        minimum_component_fill_fraction=0.02,
    ),
    # A second immutable residual-foam profile, visually bounded on the same
    # four native-resolution 4714/022 fields.  The wider linkage radius joins
    # the sparse pseudo-cell lattice that v1 left behind, while the unchanged
    # size/counterstain constraints and a still-positive fill floor selected
    # no objects in viable interior, high-OD, or debris-risk controls.
    "residual-foam-microobjects-v2": ChromaticMicroclusterPolicy(
        maximum_instance_area_pixels=120,
        minimum_counterstain_fraction=0.01,
        maximum_centroid_distance_pixels=64.0,
        minimum_component_area_pixels=300,
        minimum_component_instances=12,
        maximum_component_aspect_ratio=3.0,
        minimum_component_fill_fraction=0.01,
    ),
    # Final 4714/022 residual-foam rescue.  Native-resolution review of the
    # highest-impact fields across the whole section showed that a modestly
    # larger microobject ceiling and counterstain allowance recover fragmented
    # foam/debris profiles with weak blue contamination.  Requiring at least
    # 20 linked instances retained zero selections in the viable interior,
    # high-OD tissue, and debris-adjacent viable control fields.
    "residual-foam-microobjects-v3": ChromaticMicroclusterPolicy(
        maximum_instance_area_pixels=150,
        minimum_counterstain_fraction=0.20,
        maximum_centroid_distance_pixels=64.0,
        minimum_component_area_pixels=300,
        minimum_component_instances=20,
        maximum_component_aspect_ratio=3.0,
        minimum_component_fill_fraction=0.01,
    ),
    # PAS-specific follow-up to the no-escape v87 filter. Native-resolution
    # review of the highest-impact fields across 4714/006 showed that compact
    # clusters of <=150-pixel objects with <40% blue-dominant counterstain are
    # anucleate erythrocyte/debris profiles. The 30-instance, 1,500-pixel,
    # compact-component requirements retained the dark nuclei and larger
    # nucleated cells in the same fields and selected no objects in the fixed
    # high-OD, tissue-edge, or debris-adjacent viable controls.
    "pas-anucleate-microobjects-v2": ChromaticMicroclusterPolicy(
        maximum_instance_area_pixels=150,
        minimum_counterstain_fraction=0.40,
        maximum_centroid_distance_pixels=32.0,
        minimum_component_area_pixels=1500,
        minimum_component_instances=30,
        maximum_component_aspect_ratio=3.0,
        minimum_component_fill_fraction=0.05,
    ),
}

_BASE_CHROMATIC_SPEC = ChromaticMicroclusterRunSpec(
    algorithm_version=CHROMATIC_MICROCLUSTER_ALGORITHM_VERSION,
    method_profile=CHROMATIC_MICROCLUSTER_METHOD_PROFILE,
    scope=CHROMATIC_MICROCLUSTER_SCOPE,
    source_algorithm_version=75,
)
_PAS_CHROMATIC_SPEC = ChromaticMicroclusterRunSpec(
    algorithm_version=PAS_MICROCLUSTER_ALGORITHM_VERSION,
    method_profile=PAS_MICROCLUSTER_METHOD_PROFILE,
    scope=PAS_MICROCLUSTER_SCOPE,
    source_algorithm_version=87,
)
CHROMATIC_MICROCLUSTER_RUN_SPECS = {
    "residual-foam-microobjects-v1": _BASE_CHROMATIC_SPEC,
    "residual-foam-microobjects-v2": _BASE_CHROMATIC_SPEC,
    "residual-foam-microobjects-v3": _BASE_CHROMATIC_SPEC,
    "pas-anucleate-microobjects-v2": _PAS_CHROMATIC_SPEC,
}
CHROMATIC_MICROCLUSTER_ALGORITHM_VERSIONS = frozenset(
    spec.algorithm_version for spec in CHROMATIC_MICROCLUSTER_RUN_SPECS.values()
)
CHROMATIC_MICROCLUSTER_PROMOTION_PROFILES = frozenset(
    (spec.algorithm_version, spec.method_profile)
    for spec in CHROMATIC_MICROCLUSTER_RUN_SPECS.values()
)


def chromatic_microcluster_run_spec(
    profiles: Iterable[str],
) -> ChromaticMicroclusterRunSpec:
    """Return the single immutable run family shared by ``profiles``."""

    names = tuple(str(profile) for profile in profiles)
    if not names:
        raise ValueError("at least one chromatic profile is required")
    try:
        specs = {CHROMATIC_MICROCLUSTER_RUN_SPECS[name] for name in names}
    except KeyError as error:
        message = f"unknown chromatic microcluster profile: {error.args[0]}"
        raise ValueError(message) from error
    if len(specs) != 1:
        raise ValueError(
            "chromatic profiles from different provenance families cannot mix"
        )
    return next(iter(specs))


def chromatic_microcluster_gate(profile: str) -> dict[str, object]:
    """Return the immutable request/QC payload for one validated profile."""

    try:
        policy = CHROMATIC_MICROCLUSTER_POLICIES[profile]
    except KeyError as error:
        message = f"unknown chromatic microcluster profile: {profile}"
        raise ValueError(message) from error
    return {
        "enabled": True,
        "profile": profile,
        "minimum_blue_ratio": CHROMATIC_MICROCLUSTER_BLUE_RATIO,
        "maximum_instance_area_pixels": policy.maximum_instance_area_pixels,
        "minimum_counterstain_fraction": policy.minimum_counterstain_fraction,
        "maximum_centroid_distance_pixels": (policy.maximum_centroid_distance_pixels),
        "minimum_component_area_pixels": policy.minimum_component_area_pixels,
        "minimum_component_instances": policy.minimum_component_instances,
        "maximum_component_aspect_ratio": policy.maximum_component_aspect_ratio,
        "minimum_component_fill_fraction": policy.minimum_component_fill_fraction,
        "instance_action": "remove_whole_labels_in_validated_compact_cluster",
    }


def blue_counterstain_core(rgb: np.ndarray, *, minimum_blue_ratio: float) -> np.ndarray:
    """Return pixels whose blue channel dominates both other RGB channels."""

    values = np.asarray(rgb)
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError("RGB evidence must have shape (height, width, 3)")
    if minimum_blue_ratio <= 1 or not np.isfinite(minimum_blue_ratio):
        raise ValueError("minimum blue ratio must be finite and greater than one")
    maximum_other = np.maximum(values[..., 0], values[..., 1]).astype(np.float32)
    return values[..., 2].astype(np.float32) >= maximum_other * minimum_blue_ratio


def clustered_chromatic_microobject_instances(
    instance_areas: np.ndarray,
    instance_x_sums: np.ndarray,
    instance_y_sums: np.ndarray,
    counterstain_pixels: np.ndarray,
    *,
    policy: ChromaticMicroclusterPolicy,
) -> np.ndarray:
    """Identify compact clusters of small predictions lacking counterstain.

    Arrays are indexed by label identifier, including background at index zero.
    Coordinates are area-weighted native-pixel sums accumulated blockwise from
    the sealed label image. Candidate instances are joined only when their
    centroids are close. A connected group must also be large, populous,
    compact, and spatially filled before any member is returned.
    """

    areas = np.asarray(instance_areas)
    x_sums = np.asarray(instance_x_sums)
    y_sums = np.asarray(instance_y_sums)
    counterstain = np.asarray(counterstain_pixels)
    if any(value.ndim != 1 for value in (areas, x_sums, y_sums, counterstain)):
        raise ValueError("instance evidence arrays must be one-dimensional")
    if not (areas.shape == x_sums.shape == y_sums.shape == counterstain.shape):
        raise ValueError("instance evidence arrays must have matching shapes")
    if areas.size == 0:
        raise ValueError("instance evidence arrays must include background")
    if np.any(areas < 0) or np.any(counterstain < 0):
        raise ValueError("instance counts must be nonnegative")
    if np.any(counterstain > areas):
        raise ValueError("counterstain pixels cannot exceed instance area")
    if np.any(~np.isfinite(x_sums)) or np.any(~np.isfinite(y_sums)):
        raise ValueError("instance coordinate sums must be finite")

    fractions = counterstain / np.maximum(areas, 1)
    candidates = np.flatnonzero(
        (areas > 0)
        & (areas <= policy.maximum_instance_area_pixels)
        & (fractions < policy.minimum_counterstain_fraction)
    )
    candidates = candidates[candidates > 0]
    artifact = np.zeros(areas.shape, dtype=bool)
    if candidates.size < policy.minimum_component_instances:
        return artifact

    centroids = np.column_stack(
        (
            x_sums[candidates] / areas[candidates],
            y_sums[candidates] / areas[candidates],
        )
    )
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    pairs = cKDTree(centroids).query_pairs(
        policy.maximum_centroid_distance_pixels,
        output_type="ndarray",
    )
    if pairs.size:
        directed_pairs = np.concatenate((pairs, pairs[:, ::-1]), axis=0)
        adjacency = coo_matrix(
            (
                np.ones(len(directed_pairs), dtype=np.uint8),
                (directed_pairs[:, 0], directed_pairs[:, 1]),
            ),
            shape=(len(candidates), len(candidates)),
        ).tocsr()
    else:
        adjacency = coo_matrix((len(candidates), len(candidates))).tocsr()
    component_count, components = connected_components(adjacency, directed=False)

    padding = 2.0 * np.sqrt(policy.maximum_instance_area_pixels / np.pi)
    for component in range(component_count):
        local = np.flatnonzero(components == component)
        if local.size < policy.minimum_component_instances:
            continue
        component_area = float(areas[candidates[local]].sum())
        if component_area < policy.minimum_component_area_pixels:
            continue
        component_centroids = centroids[local]
        width = float(np.ptp(component_centroids[:, 0]) + padding)
        height = float(np.ptp(component_centroids[:, 1]) + padding)
        aspect_ratio = max(width / max(height, 1.0), height / max(width, 1.0))
        fill_fraction = component_area / max(width * height, 1.0)
        if (
            aspect_ratio <= policy.maximum_component_aspect_ratio
            and fill_fraction >= policy.minimum_component_fill_fraction
        ):
            artifact[candidates[local]] = True
    artifact[0] = False
    return artifact
