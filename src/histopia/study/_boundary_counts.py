"""Exact native cell perimeter counts with an optional CUDA backend."""

from __future__ import annotations

import numpy as np


def boundary_overlap_counts(
    cell_labels,
    region_labels,
    *,
    core_rows: tuple[int, int] | None = None,
    device: str = "cpu",
) -> np.ndarray:
    """Count (cell ID, region index, boundary pixels) on aligned native rasters.

    Region 0 is unsupported and remains in the denominator. A boundary is a
    positive cell pixel with a different four-connected neighbor; raster edges
    have background neighbors. Whole-slide stripes must include a one-row halo
    and specify the nonoverlapping ``core_rows``. Sparse cell IDs are preserved.
    The CUDA branch imports PyTorch only on demand and uses integer operations,
    with the same sorted counts as the NumPy implementation.
    """
    cells, regions = np.asarray(cell_labels), np.asarray(region_labels)
    if (
        cells.ndim != 2
        or regions.shape != cells.shape
        or cells.dtype.kind not in "iu"
        or regions.dtype.kind not in "iu"
        or np.any(cells < 0)
        or np.any(regions < 0)
    ):
        raise ValueError("aligned nonnegative integer cell and region rasters required")
    if not isinstance(device, str) or device not in {"cpu", "cuda", "cuda:0"}:
        raise ValueError("device must be cpu, cuda or cuda:0")
    start, stop = core_rows if core_rows is not None else (0, cells.shape[0])
    if (
        type(start) is not int
        or type(stop) is not int
        or not 0 <= start <= stop <= cells.shape[0]
    ):
        raise ValueError("core_rows must be an in-bounds half-open interval")
    if not cells.size or start == stop:
        return np.empty((0, 3), np.int64)
    stride = int(regions.max()) + 1
    if int(cells.max()) * stride + stride - 1 > np.iinfo(np.int64).max:
        raise ValueError("cell/region pair encoding would overflow int64")
    if device == "cpu":
        padded = np.pad(cells, 1)
        center = cells[start:stop]
        boundary = (center > 0) & (
            (padded[start:stop, 1:-1] != center)
            | (padded[start + 2 : stop + 2, 1:-1] != center)
            | (padded[start + 1 : stop + 1, :-2] != center)
            | (padded[start + 1 : stop + 1, 2:] != center)
        )
        keys = center[boundary].astype(np.int64) * stride + regions[start:stop][
            boundary
        ].astype(np.int64)
        keys, counts = np.unique(keys, return_counts=True)
    else:
        import torch
        import torch.nn.functional as functional

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        tensor = torch.as_tensor(cells.astype(np.int64), device=device)
        padded = functional.pad(tensor, (1, 1, 1, 1))
        center = tensor[start:stop]
        boundary = (center > 0) & (
            (padded[start:stop, 1:-1] != center)
            | (padded[start + 2 : stop + 2, 1:-1] != center)
            | (padded[start + 1 : stop + 1, :-2] != center)
            | (padded[start + 1 : stop + 1, 2:] != center)
        )
        grid = torch.as_tensor(regions[start:stop].astype(np.int64), device=device)
        keys, counts = torch.unique(
            center[boundary] * stride + grid[boundary],
            sorted=True,
            return_counts=True,
        )
        keys, counts = keys.cpu().numpy(), counts.cpu().numpy()
    return np.column_stack((keys // stride, keys % stride, counts)).astype(np.int64)
