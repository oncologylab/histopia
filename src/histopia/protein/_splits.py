"""Leakage-resistant spatial and grouped evaluation splits."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class SpatialSplit:
    """Train, validation, test, and excluded cell indices."""

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    excluded: np.ndarray
    block_ids: np.ndarray


def build_spatial_split(
    xy_um: np.ndarray,
    *,
    block_um: float = 1024.0,
    buffer_um: float = 112.0,
    seed: int = 0,
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> SpatialSplit:
    """Assign complete spatial macrotiles and buffer their fold boundaries."""

    points = np.asarray(xy_um, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
        raise ValueError("xy_um must have shape (cells, 2) with finite values")
    if block_um <= 0 or buffer_um <= 0 or buffer_um * 2 >= block_um:
        raise ValueError("block and buffer sizes are incompatible")
    split_fractions = np.asarray(fractions, dtype=np.float64)
    if split_fractions.shape != (3,) or np.any(split_fractions <= 0):
        raise ValueError("fractions must contain three positive values")
    split_fractions /= split_fractions.sum()
    origin = points.min(axis=0) if len(points) else np.zeros(2)
    block_xy = np.floor((points - origin) / block_um).astype(np.int64)
    unique_blocks, inverse = np.unique(block_xy, axis=0, return_inverse=True)
    assignments = np.empty(len(unique_blocks), dtype=np.int8)
    thresholds = np.cumsum(split_fractions)
    for index, block in enumerate(unique_blocks):
        token = f"{seed}:{int(block[0])}:{int(block[1])}".encode()
        draw = int.from_bytes(hashlib.sha256(token).digest()[:8], "big") / 2**64
        assignments[index] = int(np.searchsorted(thresholds, draw, side="right"))
    fold = assignments[inverse] if len(points) else np.empty(0, dtype=np.int8)
    local = (points - origin) % block_um if len(points) else points.copy()
    buffered = np.any((local < buffer_um) | (local > block_um - buffer_um), axis=1)
    indices = np.arange(len(points), dtype=np.int64)
    return SpatialSplit(
        train=indices[(fold == 0) & ~buffered],
        validation=indices[(fold == 1) & ~buffered],
        test=indices[(fold == 2) & ~buffered],
        excluded=indices[buffered],
        block_ids=inverse.astype(np.int32),
    )


def leave_one_group_out(
    groups: np.ndarray,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Return deterministic train/test folds for sections or mice."""

    values = np.asarray(groups)
    if values.ndim != 1:
        raise ValueError("groups must be one-dimensional")
    indices = np.arange(len(values), dtype=np.int64)
    return tuple(
        (indices[values != group], indices[values == group])
        for group in np.unique(values)
    )
