"""Conservative detection and geometry for dense-small-cell recovery.

The primary CPSAM workflow can occasionally leave sharply bounded gaps in
otherwise continuous fields of small, hematoxylin-rich cells.  These pure
functions identify that specific mismatch between visible nuclear evidence and
instance coverage.  They do not run an alternate segmentation model and do not
modify a whole-slide result by themselves.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Collection, Mapping
from dataclasses import dataclass

import numpy as np
import scipy.ndimage as ndi

from histopia.cells._algorithms import containment_merge


@dataclass(frozen=True, slots=True)
class DenseSmallCellGapEvidence:
    """Summary of one native-resolution recovery-screening window."""

    triggered: bool
    global_nuclear_fraction: float
    gap_bin_count: int
    bin_count: int
    gap_bin_fraction: float
    gap_bins: np.ndarray
    candidate_gap_bins: np.ndarray


def dense_small_cell_gap_evidence(
    nuclear_evidence: np.ndarray,
    labels: np.ndarray,
    *,
    bin_size: int = 64,
    minimum_global_nuclear_fraction: float = 0.80,
    minimum_bin_nuclear_fraction: float = 0.80,
    maximum_bin_instance_fraction: float = 0.40,
    minimum_gap_bins: int = 8,
) -> DenseSmallCellGapEvidence:
    """Detect dense nuclear bins that lack corresponding cell instances.

    A window is eligible only when nuclear evidence occupies most of the full
    window.  This prevents lumina, tissue edges, sparse stroma, and ordinary
    low-cellularity regions from being mistaken for inference dropouts.  The
    returned bin mask is active only after the minimum number of discrepant
    bins is reached, so downstream recovery cannot act on isolated cells.
    """

    nuclear = np.asarray(nuclear_evidence, dtype=bool)
    instances = np.asarray(labels)
    if nuclear.ndim != 2 or instances.ndim != 2 or nuclear.shape != instances.shape:
        raise ValueError("nuclear evidence and labels must have the same 2D shape")
    if not np.issubdtype(instances.dtype, np.integer):
        raise TypeError("labels must contain integers")
    if bin_size <= 0:
        raise ValueError("bin_size must be positive")
    if minimum_gap_bins <= 0:
        raise ValueError("minimum_gap_bins must be positive")
    for name, value in (
        ("minimum_global_nuclear_fraction", minimum_global_nuclear_fraction),
        ("minimum_bin_nuclear_fraction", minimum_bin_nuclear_fraction),
        ("maximum_bin_instance_fraction", maximum_bin_instance_fraction),
    ):
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be finite and between zero and one")
    if nuclear.size == 0:
        raise ValueError("screening window must not be empty")

    nuclear_bins = _block_fractions(nuclear, bin_size)
    instance_bins = _block_fractions(instances > 0, bin_size)
    raw_gap_bins = (nuclear_bins >= minimum_bin_nuclear_fraction) & (
        instance_bins <= maximum_bin_instance_fraction
    )
    global_nuclear_fraction = float(np.mean(nuclear))
    gap_bin_count = int(np.count_nonzero(raw_gap_bins))
    triggered = (
        global_nuclear_fraction >= minimum_global_nuclear_fraction
        and gap_bin_count >= minimum_gap_bins
    )
    gap_bins = raw_gap_bins if triggered else np.zeros_like(raw_gap_bins)
    bin_count = int(raw_gap_bins.size)
    return DenseSmallCellGapEvidence(
        triggered=triggered,
        global_nuclear_fraction=global_nuclear_fraction,
        gap_bin_count=gap_bin_count,
        bin_count=bin_count,
        gap_bin_fraction=gap_bin_count / bin_count,
        gap_bins=gap_bins,
        candidate_gap_bins=raw_gap_bins,
    )


def connected_gap_tile_scopes(
    candidate_gap_bins_by_tile: Mapping[str, np.ndarray],
    origins_xy_by_tile: Mapping[str, tuple[int, int]],
    strict_seed_tiles: Collection[str],
    *,
    bin_size: int = 64,
    connectivity: int = 8,
) -> dict[str, np.ndarray]:
    """Keep every raw gap-bin component connected to a strict seed tile.

    Whole-tile nuclear-fraction screening is intentionally conservative, but a
    true dense-cell dropout can cross into a neighboring tile whose mixture of
    tissue and background lowers that tile's global fraction.  This helper
    embeds raw candidate bins from overlapping tiles in one global bin lattice
    and returns only components that touch a strictly triggered tile.  It can
    therefore continue a repair across tile boundaries without admitting
    disconnected debris elsewhere on the slide.

    Tile origins must be aligned to ``bin_size``.  Returned masks retain each
    tile's local shape and include only bins in seed-connected components.
    """

    if bin_size <= 0:
        raise ValueError("bin_size must be positive")
    if connectivity not in {4, 8}:
        raise ValueError("connectivity must be 4 or 8")
    keys = set(candidate_gap_bins_by_tile)
    if keys != set(origins_xy_by_tile):
        raise ValueError("candidate gap masks and tile origins must have the same keys")
    seeds = set(strict_seed_tiles)
    if not seeds <= keys:
        raise ValueError("strict seed tiles must be present in candidate gap masks")

    masks: dict[str, np.ndarray] = {}
    global_members: dict[tuple[int, int], list[tuple[str, int, int]]] = {}
    for key in sorted(keys):
        mask = np.asarray(candidate_gap_bins_by_tile[key], dtype=bool)
        if mask.ndim != 2 or not mask.size:
            raise ValueError(f"candidate gap mask must be nonempty and 2D: {key}")
        origin = origins_xy_by_tile[key]
        if (
            not isinstance(origin, tuple)
            or len(origin) != 2
            or any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
                or value % bin_size
                for value in origin
            )
        ):
            raise ValueError(f"tile origin must align to bin_size: {key}")
        masks[key] = mask
        x_bin = origin[0] // bin_size
        y_bin = origin[1] // bin_size
        for row, column in np.argwhere(mask):
            point = (y_bin + int(row), x_bin + int(column))
            global_members.setdefault(point, []).append((key, int(row), int(column)))

    seed_points: set[tuple[int, int]] = set()
    for key in seeds:
        mask = masks[key]
        if not np.any(mask):
            raise ValueError(f"strict seed tile contains no candidate gap bins: {key}")
        x_bin = origins_xy_by_tile[key][0] // bin_size
        y_bin = origins_xy_by_tile[key][1] // bin_size
        seed_points.update(
            (y_bin + int(row), x_bin + int(column)) for row, column in np.argwhere(mask)
        )
    if not seed_points:
        return {}

    offsets = ((-1, 0), (1, 0), (0, -1), (0, 1))
    if connectivity == 8:
        offsets += ((-1, -1), (-1, 1), (1, -1), (1, 1))
    connected = set(seed_points)
    pending = deque(seed_points)
    while pending:
        row, column = pending.popleft()
        for dy, dx in offsets:
            neighbor = (row + dy, column + dx)
            if neighbor in global_members and neighbor not in connected:
                connected.add(neighbor)
                pending.append(neighbor)

    scopes = {key: np.zeros_like(mask) for key, mask in masks.items()}
    for point in connected:
        for key, row, column in global_members[point]:
            scopes[key][row, column] = True
    return {key: scopes[key] for key in sorted(scopes) if np.any(scopes[key])}


def expand_gap_bins(
    gap_bins: np.ndarray,
    image_shape: tuple[int, int],
    *,
    bin_size: int = 64,
    context_bins: int = 1,
) -> np.ndarray:
    """Project qualifying bins to pixels with bounded neighboring context."""

    bins = np.asarray(gap_bins, dtype=bool)
    if bins.ndim != 2:
        raise ValueError("gap_bins must be 2D")
    if len(image_shape) != 2 or any(size <= 0 for size in image_shape):
        raise ValueError("image_shape must contain two positive dimensions")
    if bin_size <= 0:
        raise ValueError("bin_size must be positive")
    if context_bins < 0:
        raise ValueError("context_bins must be nonnegative")
    expected = tuple((size + bin_size - 1) // bin_size for size in image_shape)
    if bins.shape != expected:
        raise ValueError("gap-bin geometry does not match image_shape and bin_size")
    if context_bins:
        bins = ndi.binary_dilation(
            bins,
            structure=np.ones((3, 3), dtype=bool),
            iterations=context_bins,
        )
    pixels = np.repeat(np.repeat(bins, bin_size, axis=0), bin_size, axis=1)
    return pixels[: image_shape[0], : image_shape[1]]


def bounded_instance_territories(
    nuclear_labels: np.ndarray,
    *,
    maximum_growth_pixels: float = 4.0,
    support_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Grow nuclear instances into nearest-seed cell territories.

    Growth is Euclidean, deterministic, and capped in native pixels.  Optional
    support evidence can further restrict proposed cytoplasm, while every
    original nuclear marker is retained.  The function preserves input label
    identifiers so a caller can maintain explicit recovery provenance before
    whole-slide stitching assigns final identifiers.
    """

    markers = np.asarray(nuclear_labels)
    if markers.ndim != 2:
        raise ValueError("nuclear_labels must be 2D")
    if not np.issubdtype(markers.dtype, np.integer):
        raise TypeError("nuclear_labels must contain integers")
    if np.any(markers < 0):
        raise ValueError("nuclear_labels must be nonnegative")
    if not np.isfinite(maximum_growth_pixels) or maximum_growth_pixels < 0:
        raise ValueError("maximum_growth_pixels must be finite and nonnegative")
    if support_mask is None:
        supported = np.ones(markers.shape, dtype=bool)
    else:
        supported = np.asarray(support_mask, dtype=bool)
        if supported.shape != markers.shape:
            raise ValueError("support_mask must match nuclear_labels")
    if not np.any(markers):
        return np.zeros(markers.shape, dtype=np.int32)

    distance, nearest_indices = ndi.distance_transform_edt(
        markers == 0,
        return_indices=True,
    )
    nearest = markers[tuple(nearest_indices)]
    allowed = (distance <= maximum_growth_pixels) & supported
    allowed |= markers > 0
    return np.where(allowed, nearest, 0).astype(np.int32, copy=False)


def complete_dense_small_cell_gaps(
    source_labels: np.ndarray,
    recovered_labels: np.ndarray,
    gap_bins: np.ndarray,
    *,
    bin_size: int = 64,
    context_bins: int = 1,
    maximum_intersection_over_smaller: float = 0.10,
) -> np.ndarray:
    """Add only complementary recovered cells near screened dropout bins.

    Existing CPSAM instances remain authoritative. Recovered instances become
    eligible only when they touch a triggered gap bin or its bounded context,
    and an eligible instance is added only when it has little overlap with an
    existing instance. This repairs missing dense-cell patches without
    replacing good cell boundaries elsewhere in the same native tile.
    """

    source = np.asarray(source_labels)
    recovered = np.asarray(recovered_labels)
    if source.ndim != 2 or recovered.ndim != 2 or source.shape != recovered.shape:
        raise ValueError("source and recovered labels must have the same 2D shape")
    if not np.issubdtype(source.dtype, np.integer) or not np.issubdtype(
        recovered.dtype, np.integer
    ):
        raise TypeError("source and recovered labels must contain integers")
    if np.any(source < 0) or np.any(recovered < 0):
        raise ValueError("source and recovered labels must be nonnegative")
    if (
        not np.isfinite(maximum_intersection_over_smaller)
        or not 0 <= maximum_intersection_over_smaller <= 1
    ):
        raise ValueError(
            "maximum_intersection_over_smaller must be finite and between zero and one"
        )
    recovery_scope = expand_gap_bins(
        gap_bins,
        source.shape,
        bin_size=bin_size,
        context_bins=context_bins,
    )
    selected_ids = np.unique(recovered[recovery_scope])
    selected_ids = selected_ids[selected_ids > 0]
    if not selected_ids.size:
        return source.copy()
    selected = np.where(np.isin(recovered, selected_ids), recovered, 0)
    return containment_merge(
        source,
        selected,
        threshold=maximum_intersection_over_smaller,
    )


def _block_fractions(mask: np.ndarray, bin_size: int) -> np.ndarray:
    """Return fractions for a ceil-divided grid without padding bias."""

    height, width = mask.shape
    rows = (height + bin_size - 1) // bin_size
    cols = (width + bin_size - 1) // bin_size
    output = np.empty((rows, cols), dtype=np.float32)
    for row in range(rows):
        y0 = row * bin_size
        y1 = min(height, y0 + bin_size)
        for col in range(cols):
            x0 = col * bin_size
            x1 = min(width, x0 + bin_size)
            output[row, col] = float(np.mean(mask[y0:y1, x0:x1]))
    return output
