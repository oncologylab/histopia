"""Remove detached one-section components from serial semantic atlases."""

from __future__ import annotations

from typing import Any

import numpy as np

from histopia.semantic._features import PatchFeatures, subset_patch_features

_METHOD = "adjacent-component-support-v1"
_PRESERVE_RELATIVE_AREA = 0.5
_NEIGHBOR_RADIUS_PATCH_WIDTHS = 1.5


def filter_stack_supported_components(
    sections: tuple[PatchFeatures, ...],
    *,
    min_neighbor_fraction: float,
) -> tuple[tuple[PatchFeatures, ...], dict[str, Any] | None]:
    """Drop small components without registered support in adjacent sections.

    The dominant component and any component at least half its size are always
    retained. Smaller components must place the requested fraction of their
    patch centroids near tissue in either adjacent section. This removes
    one-slide tears and mounting fragments without eroding recurring small
    lobules or legitimate terminal-section tissue.
    """

    threshold = float(min_neighbor_fraction)
    if not 0 <= threshold <= 1:
        raise ValueError("min_neighbor_fraction must be between zero and one")
    if threshold == 0:
        return sections, None
    if not sections:
        raise ValueError("stack component filtering requires at least one section")
    patch_widths = {
        float(section.patch_size_px * section.analysis_mpp) for section in sections
    }
    if len(patch_widths) != 1:
        raise ValueError("stack component filtering requires one patch width")

    try:
        from scipy import ndimage
        from scipy.spatial import cKDTree
    except ImportError as exc:  # pragma: no cover - semantic extra guard
        raise RuntimeError("stack component filtering requires scipy") from exc

    radius_um = _NEIGHBOR_RADIUS_PATCH_WIDTHS * next(iter(patch_widths))
    trees = tuple(cKDTree(section.reference_um_xy) for section in sections)
    filtered: list[PatchFeatures] = []
    section_rows: list[dict[str, Any]] = []
    total_removed_components = 0
    total_removed_patches = 0

    for index, section in enumerate(sections):
        occupancy = np.zeros(section.grid_shape, dtype=bool)
        occupancy[section.grid_rc[:, 0], section.grid_rc[:, 1]] = True
        component_grid, component_count = ndimage.label(
            occupancy,
            structure=np.ones((3, 3), dtype=np.uint8),
        )
        component_ids = component_grid[section.grid_rc[:, 0], section.grid_rc[:, 1]]
        sizes = np.bincount(component_ids, minlength=component_count + 1)[1:]
        largest = int(sizes.max())
        selected = np.zeros(len(section.features), dtype=bool)
        component_rows: list[dict[str, Any]] = []
        adjacent = tuple(
            trees[neighbor]
            for neighbor in (index - 1, index + 1)
            if 0 <= neighbor < len(sections)
        )
        for component_id, raw_size in enumerate(sizes, start=1):
            component = component_ids == component_id
            size = int(raw_size)
            relative_area = size / largest
            points = section.reference_um_xy[component]
            supported = np.zeros(size, dtype=bool)
            for tree in adjacent:
                distance, _ = tree.query(
                    points,
                    k=1,
                    distance_upper_bound=radius_um,
                )
                supported |= np.isfinite(distance)
            neighbor_fraction = float(supported.mean()) if adjacent else 1.0
            keep = (
                relative_area >= _PRESERVE_RELATIVE_AREA
                or neighbor_fraction >= threshold
            )
            if keep:
                selected |= component
            else:
                total_removed_components += 1
                total_removed_patches += size
            component_rows.append(
                {
                    "patches": size,
                    "relative_area": relative_area,
                    "neighbor_fraction": neighbor_fraction,
                    "retained": keep,
                }
            )
        filtered.append(subset_patch_features(section, selected))
        section_rows.append(
            {
                "slide_id": section.slide_id,
                "source_patches": len(section.features),
                "retained_patches": int(selected.sum()),
                "components": component_rows,
            }
        )

    report: dict[str, Any] = {
        "method": _METHOD,
        "min_neighbor_fraction": threshold,
        "neighbor_radius_um": radius_um,
        "preserve_relative_area": _PRESERVE_RELATIVE_AREA,
        "removed_components": total_removed_components,
        "removed_patches": total_removed_patches,
        "sections": section_rows,
    }
    return tuple(filtered), report
