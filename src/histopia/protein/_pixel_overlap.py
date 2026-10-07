"""Stream native cell boundaries against coarser, unresampled OD pixels."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RasterCellMeans:
    label_ids: np.ndarray
    mean_od: np.ndarray
    effective_pixels: np.ndarray
    coverage: np.ndarray
    measured: np.ndarray


def _axis_shares(coordinates, step):
    base = np.floor(coordinates).astype(np.int64)
    first = np.minimum(coordinates + step, base + 1) - coordinates
    return base, first, np.maximum(step - first, 0)


def native_cell_od_means(
    reader: Callable[[int, int], np.ndarray],
    shape: tuple[int, int],
    label_ids: np.ndarray,
    native_areas: np.ndarray,
    target_od: np.ndarray,
    tissue_mask: np.ndarray,
    *,
    label_origin_xy: tuple[float, float],
    measurement_origin_xy: tuple[float, float],
    native_mpp_xy: tuple[float, float],
    analysis_mpp: float,
    stripe_height: int = 64,
    minimum_effective_pixels: float = 2.0,
    minimum_coverage: float = 0.70,
    progress: Callable[[int, int], None] | None = None,
) -> RasterCellMeans:
    """Integrate exact rectangular pixel overlaps without enlarging the OD map.

    Origins describe native pixel *edges*, independent of a centroid convention.
    Native and measurement rasters share the same unregistered slide frame.
    A native pixel must be no wider than a measurement pixel along either axis.
    Each OD pixel contributes the fraction of its area occupied by a cell;
    splitting that pixel never creates additional measurement resolution.
    Effective pixels are summed area fractions, not independent observations.
    Means below either coverage gate remain missing. Only continuous OD is
    summarized; this function does not infer a positivity threshold.
    """
    ids = np.asarray(label_ids)
    area = np.asarray(native_areas, dtype=float)
    od = np.asarray(target_od)
    mask = np.asarray(tissue_mask)
    scale = np.asarray(native_mpp_xy, dtype=float)
    origins = np.asarray([label_origin_xy, measurement_origin_xy], dtype=float)
    if ids.ndim != 1 or ids.dtype.kind not in "iu" or np.any(ids <= 0):
        raise ValueError("cell IDs must be positive integers")
    if len(np.unique(ids)) != len(ids) or area.shape != ids.shape:
        raise ValueError("unique cell IDs and areas must align")
    if np.any(~np.isfinite(area)) or np.any(area <= 0):
        raise ValueError("native cell areas must be positive and finite")
    if od.ndim != 2 or mask.shape != od.shape or mask.dtype != bool:
        raise ValueError("OD pixels and boolean tissue support must align")
    if np.any(~np.isfinite(od[mask])) or np.any(od[mask] < 0):
        raise ValueError("supported OD must be finite and nonnegative")
    if (
        not np.isfinite(analysis_mpp)
        or analysis_mpp <= 0
        or scale.shape != (2,)
        or np.any(~np.isfinite(scale))
        or np.any(scale <= 0)
        or np.any(scale > analysis_mpp)
        or origins.shape != (2, 2)
        or np.any(~np.isfinite(origins))
    ):
        raise ValueError("invalid physical pixel scale or native origins")
    if (
        len(shape) != 2
        or any(int(v) != v or v < 1 for v in shape)
        or stripe_height < 1
        or int(stripe_height) != stripe_height
        or not np.isfinite(minimum_effective_pixels)
        or minimum_effective_pixels <= 0
        or not 0 <= minimum_coverage <= 1
    ):
        raise ValueError("invalid raster shape, stripe or coverage controls")
    maximum = int(ids.max(initial=0))
    known = np.zeros(maximum + 1, dtype=bool)
    known[ids] = True
    total = np.zeros(maximum + 1)
    weights = np.zeros(maximum + 1)
    seen = np.zeros(maximum + 1, dtype=np.int64)
    height, width = shape
    step = scale / analysis_mpp
    offset = (origins[0] - origins[1]) * step
    xbase, wx0, wx1 = _axis_shares(np.arange(width) * step[0] + offset[0], step[0])
    for top in range(0, height, stripe_height):
        rows = min(stripe_height, height - top)
        labels = np.asarray(reader(top, rows))
        if labels.shape != (rows, width) or labels.dtype.kind not in "iu":
            raise ValueError("native label reader returned an invalid stripe")
        if np.any(labels < 0) or int(labels.max(initial=0)) > maximum:
            raise ValueError("native raster contains an unknown cell ID")
        yy, xx = np.nonzero(labels)
        lid = labels[yy, xx].astype(np.int64)
        if not np.all(known[lid]):
            raise ValueError("native raster contains an unknown cell ID")
        seen += np.bincount(lid, minlength=maximum + 1)
        ybase, wy0, wy1 = _axis_shares((yy + top) * step[1] + offset[1], step[1])
        for dx, wx in enumerate((wx0[xx], wx1[xx])):
            ox = xbase[xx] + dx
            for dy, wy in enumerate((wy0, wy1)):
                oy, weight = ybase + dy, wx * wy
                inside = (
                    (ox >= 0)
                    & (oy >= 0)
                    & (ox < od.shape[1])
                    & (oy < od.shape[0])
                    & (weight > 0)
                )
                selected = np.flatnonzero(inside)
                selected = selected[mask[oy[selected], ox[selected]]]
                w = weight[selected]
                weights += np.bincount(lid[selected], weights=w, minlength=maximum + 1)
                total += np.bincount(
                    lid[selected],
                    weights=w * od[oy[selected], ox[selected]],
                    minlength=maximum + 1,
                )
        if progress is not None and top % (stripe_height * 32) == 0:
            progress(top + rows, height)
    if not np.array_equal(seen[ids], area):
        raise ValueError("cell areas differ from the bound native label raster")
    effective = weights[ids]
    coverage = effective / (area * np.prod(step))
    if np.any(coverage > 1 + 1e-8):
        raise ValueError("overlap area exceeded the native cell area")
    coverage = np.minimum(coverage, 1)
    measured = (effective >= minimum_effective_pixels) & (coverage >= minimum_coverage)
    means = np.divide(
        total[ids], effective, out=np.full(len(ids), np.nan), where=measured
    )
    return RasterCellMeans(ids.copy(), means, effective, coverage, measured)
