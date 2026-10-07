"""Geometry and tissue support for tile-local candidate instances."""

from __future__ import annotations

import numpy as np


def profile_tile_instances(
    labels: np.ndarray,
    supported_labels: np.ndarray,
    *,
    origin_xy: tuple[float, float] = (0, 0),
    mpp_xy: tuple[float, float] | None = None,
) -> dict[str, np.ndarray]:
    """Preserve raw label IDs and quantify their supported boundary pixels.

    IDs belong to one tile. This function does not stitch overlap, infer cell
    identity, or establish segmentation accuracy. Coordinates use pixel centers
    (the first pixel is at 0.5, 0.5) on the native image canvas. Unsupported
    centroids and unknown physical measurements remain NaN. ``mpp_xy`` must
    describe native pixels, rather than a resized preview's pixel spacing.
    """
    raw = np.asarray(labels)
    supported = np.asarray(supported_labels)
    if raw.ndim != 2 or supported.shape != raw.shape or not raw.size:
        raise ValueError("labels must have the same nonempty two-dimensional shape")
    if not all(np.issubdtype(a.dtype, np.integer) for a in (raw, supported)):
        raise TypeError("labels must contain integers")
    if np.any(raw < 0) or np.any(supported < 0):
        raise ValueError("labels must be nonnegative")
    if np.any((supported > 0) & (supported != raw)):
        raise ValueError("support may remove pixels but must preserve raw label IDs")
    origin = np.asarray(origin_xy, dtype=float)
    if origin.shape != (2,) or not np.isfinite(origin).all():
        raise ValueError("origin_xy must contain two finite coordinates")
    scale = np.full(2, np.nan) if mpp_xy is None else np.asarray(mpp_xy, float)
    if mpp_xy is not None and (
        scale.shape != (2,) or not np.isfinite(scale).all() or np.any(scale <= 0)
    ):
        raise ValueError("mpp_xy must contain two finite positive pixel spacings")
    y, x = np.nonzero(raw)
    ids, inverse, area = np.unique(raw[y, x], return_inverse=True, return_counts=True)
    count = len(ids)
    keep = supported[y, x] > 0
    weights = np.bincount(inverse[keep], minlength=count)
    xy = np.full((count, 2), np.nan)
    has_support = weights > 0
    for axis, coordinate in enumerate((x, y)):
        total = np.bincount(
            inverse[keep], weights=coordinate[keep] + 0.5, minlength=count
        )
        xy[has_support, axis] = total[has_support] / weights[has_support] + origin[axis]
    edge = (x == 0) | (y == 0) | (x == raw.shape[1] - 1) | (y == raw.shape[0] - 1)
    clipped = np.bincount(inverse[edge], minlength=count) > 0
    return {
        "label_id": ids,
        "raw_area_px": area,
        "supported_area_px": weights,
        "supported_fraction": np.divide(weights, area, dtype=float),
        "has_support": has_support,
        "native_xy_px": xy,
        "native_xy_um": xy * scale,
        "supported_area_um2": np.where(has_support, weights * np.prod(scale), np.nan),
        "touches_tile_edge": clipped,
    }
