"""Cell-to-region bags for weak supervision at a declared measurement scale.

Region-level supervision does not identify individual cell protein values.
Sampling preserves cell indices; padding and unsupported cells remain explicit.
"""

from __future__ import annotations

import numpy as np


def sample_cell_bags(
    cell_to_bag: np.ndarray,
    eligible_bags: np.ndarray,
    *,
    maximum_bags: int,
    maximum_cells: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select bags and cells without replacement, using -1 only for padding.

    A cell bag of -1 means unsupported. Return selected bag IDs, padded source
    cell indices, and full cell counts before within-bag sampling.
    """
    membership = np.asarray(cell_to_bag)
    eligible = np.asarray(eligible_bags)
    if membership.ndim != 1 or membership.dtype.kind not in "iu":
        raise ValueError("cell membership must be a one-dimensional integer array")
    if np.any(membership < -1):
        raise ValueError("only -1 denotes unsupported membership")
    if eligible.ndim != 1 or eligible.dtype != bool:
        raise ValueError("eligible bags must be a one-dimensional boolean array")
    if np.any(membership >= len(eligible)):
        raise ValueError("cell membership exceeds the bag inventory")
    if maximum_bags < 1 or maximum_cells < 1:
        raise ValueError("sampling limits must be positive")
    membership = membership.astype(np.int64, copy=False)
    counts = np.bincount(membership[membership >= 0], minlength=len(eligible))
    available = np.flatnonzero(eligible & (counts > 0))
    rng = np.random.default_rng(seed)
    selected = np.sort(
        rng.choice(available, min(maximum_bags, len(available)), replace=False)
    )
    indices = np.full((len(selected), maximum_cells), -1, dtype=np.int64)
    order = np.argsort(membership, kind="stable")
    ordered = membership[order]
    for row, bag in enumerate(selected):
        start, stop = (
            np.searchsorted(ordered, bag, side="left"),
            np.searchsorted(ordered, bag, side="right"),
        )
        members = order[start:stop]
        chosen = np.sort(
            rng.choice(members, min(maximum_cells, len(members)), replace=False)
        )
        indices[row, : len(chosen)] = chosen
    return selected, indices, counts[selected]


def mean_cell_vectors(
    cell_to_bag: np.ndarray, values: np.ndarray, *, bag_count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Pool finite cell-vector entries; missing support yields NaN, never zero."""
    membership, vectors = np.asarray(cell_to_bag), np.asarray(values)
    if (
        membership.ndim != 1
        or membership.dtype.kind not in "iu"
        or vectors.ndim != 2
        or len(membership) != len(vectors)
    ):
        raise ValueError("cell membership and vectors must align")
    if bag_count < 0 or np.any(membership < -1) or np.any(membership >= bag_count):
        raise ValueError("cell membership exceeds the bag inventory")
    if np.any(np.isinf(vectors)):
        raise ValueError("missing vector values must use NaN")
    membership = membership.astype(np.int64, copy=False)
    result = np.full((bag_count, vectors.shape[1]), np.nan, dtype=np.float64)
    counts = np.zeros(result.shape, dtype=np.int64)
    for marker in range(vectors.shape[1]):
        keep = (membership >= 0) & np.isfinite(vectors[:, marker])
        counts[:, marker] = np.bincount(membership[keep], minlength=bag_count)
        total = np.bincount(
            membership[keep], weights=vectors[keep, marker], minlength=bag_count
        )
        np.divide(
            total, counts[:, marker], out=result[:, marker], where=counts[:, marker] > 0
        )
    return result, counts
