"""Physical cell neighborhoods and within-tissue spatial null controls."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from histopia.protein._vector_transfer import _array_hash, _ids
from histopia.study._manifest import fingerprint


@dataclass(frozen=True)
class NeighborhoodResult:
    cell_ids: np.ndarray
    feature_ids: tuple[str, ...]
    radius_um: float
    means: np.ndarray
    support_count: np.ndarray
    neighbor_count: np.ndarray
    fingerprint: str
    coordinate_units: str = "um"
    biological_validation: str = "pending_independent_evidence"


def physical_neighborhoods(
    cell_ids,
    xyz_um,
    values,
    support,
    feature_ids,
    mouse_ids,
    section_indices,
    tissue_ids,
    *,
    upstream_fingerprint: str,
    radius_um: float = 64,
    adjacent_sections: bool = False,
    z_spacing_kind: str = "unknown",
    registration_support=None,
) -> NeighborhoodResult:
    """Leave the center cell out; restrict neighbors to its mouse and tissue.

    Within-section distances use XY. Adjacent-section neighborhoods use true
    XYZ and only index differences <= 1; their spacing must be explicitly
    physical or assumed. Tissue IDs in this mode must describe a supported
    registered tissue component, not unrelated within-section region labels.
    Missing features have separate denominators. No cell class is inferred.
    """
    from scipy.spatial import cKDTree

    ids = _ids(cell_ids, "cell IDs")
    features = tuple(_ids(feature_ids, "feature IDs"))
    xyz, data, valid = (
        np.asarray(xyz_um, float),
        np.asarray(values, float),
        np.asarray(support),
    )
    mice, sections, tissue = (
        np.asarray(mouse_ids, str),
        np.asarray(section_indices),
        np.asarray(tissue_ids, str),
    )
    n = len(ids)
    if xyz.shape != (n, 3) or not np.all(np.isfinite(xyz)):
        raise ValueError("finite registered XYZ coordinates must align with cells")
    if (
        data.shape != (n, len(features))
        or valid.shape != data.shape
        or valid.dtype != bool
    ):
        raise ValueError("neighborhood values and boolean support must align")
    if (
        any(a.shape != (n,) for a in (mice, sections, tissue))
        or sections.dtype.kind not in "iu"
    ):
        raise ValueError("mouse, integer section and tissue identifiers must align")
    if (
        np.any(np.isinf(data))
        or not np.isfinite(radius_um)
        or radius_um <= 0
        or not upstream_fingerprint
    ):
        raise ValueError("finite data, physical radius and provenance are required")
    if adjacent_sections and z_spacing_kind not in {"physical", "assumed"}:
        raise ValueError("adjacent-section neighborhoods require explicit Z spacing")
    geometry = (
        np.ones(n, bool)
        if registration_support is None
        else np.asarray(registration_support)
    )
    if geometry.shape != (n,) or geometry.dtype != bool:
        raise ValueError("registration support must be an aligned boolean mask")
    geometry = geometry & (tissue != "")
    valid = valid & np.isfinite(data)
    means = np.full(data.shape, np.nan)
    counts = np.zeros(data.shape, np.int32)
    total = np.zeros(n, np.int32)
    groups = {}
    for i in np.flatnonzero(geometry):
        key = (mice[i], tissue[i], None if adjacent_sections else int(sections[i]))
        groups.setdefault(key, []).append(i)
    for indices in groups.values():
        ix = np.asarray(indices)
        points = xyz[ix] if adjacent_sections else xyz[ix, :2]
        tree = cKDTree(points)
        for j, i in enumerate(ix):
            neighbors = ix[tree.query_ball_point(points[j], radius_um)]
            neighbors = neighbors[neighbors != i]
            if adjacent_sections:
                neighbors = neighbors[
                    np.abs(sections[neighbors].astype(np.int64) - int(sections[i])) <= 1
                ]
            total[i] = len(neighbors)
            v = valid[neighbors]
            count = v.sum(axis=0)
            counts[i] = count
            means[i] = np.divide(
                np.where(v, data[neighbors], 0).sum(axis=0),
                count,
                out=np.full(len(features), np.nan),
                where=count > 0,
            )
    binding = fingerprint(
        {
            "algorithm": "physical-neighborhood-v1",
            "upstream": upstream_fingerprint,
            "ids": ids.tolist(),
            "features": features,
            "radius_um": radius_um,
            "adjacent": adjacent_sections,
            "z_spacing_kind": z_spacing_kind,
            "arrays": [
                _array_hash(a)
                for a in (xyz, data, valid, mice, sections, tissue, geometry)
            ],
        }
    )
    return NeighborhoodResult(ids, features, radius_um, means, counts, total, binding)


def within_tissue_permutation(
    values, support, mouse_ids, section_ids, tissue_ids, *, seed: int
):
    """Permute whole cell profiles with missingness within mouse/section/tissue.

    Joint profile permutation retains cross-marker covariance and marginal
    support. Coordinates and tissue geometry remain fixed. This is a spatial
    null, not independent biological validation.
    """
    data, mask = np.asarray(values), np.asarray(support)
    if data.ndim != 2 or mask.shape != data.shape or mask.dtype != bool:
        raise ValueError("aligned profile matrix and boolean support are required")
    keys = [np.asarray(x, str) for x in (mouse_ids, section_ids, tissue_ids)]
    if any(x.shape != (len(data),) for x in keys):
        raise ValueError("permutation strata must align with cells")
    groups = {}
    for i, key in enumerate(zip(*keys, strict=True)):
        groups.setdefault(key, []).append(i)
    rng = np.random.default_rng(seed)
    order = np.arange(len(data))
    for key, indices in groups.items():
        if key[2]:
            order[indices] = rng.permutation(indices)
    return data[order].copy(), mask[order].copy(), order


def niche_stability(
    labelings: dict[str, tuple[np.ndarray, np.ndarray]],
) -> list[dict[str, object]]:
    """ARI on shared immutable cell IDs across sampling, scale or seed runs."""
    from sklearn.metrics import adjusted_rand_score

    rows = []
    names = sorted(labelings)
    for i, left in enumerate(names):
        li, ll = labelings[left]
        li = _ids(li, "left cell IDs")
        if np.asarray(ll).shape != li.shape:
            raise ValueError("niche labels must align with cell IDs")
        for right in names[i + 1 :]:
            ri, rl = labelings[right]
            ri = _ids(ri, "right cell IDs")
            if np.asarray(rl).shape != ri.shape:
                raise ValueError("niche labels must align with cell IDs")
            _, left_index, right_index = np.intersect1d(li, ri, return_indices=True)
            rows.append(
                {
                    "left": left,
                    "right": right,
                    "shared_cells": len(left_index),
                    "adjusted_rand_index": float(
                        adjusted_rand_score(
                            np.asarray(ll)[left_index], np.asarray(rl)[right_index]
                        )
                    )
                    if len(left_index) >= 2
                    else None,
                }
            )
    return rows
