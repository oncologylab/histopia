"""Pure instance-mask operations used by cell-boundary workflows."""

from __future__ import annotations

import numpy as np
import scipy.ndimage as ndi

_CROSS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
_HEMATOXYLIN_DAB_BASIS = np.asarray(
    ((0.650, 0.704, 0.286), (0.268, 0.570, 0.776)),
    dtype=np.float32,
)


def merge_small_into_large(
    small_mask: np.ndarray, large_mask: np.ndarray
) -> np.ndarray:
    """Keep large-mask instances hit by a small-mask instance centroid."""

    small = np.asarray(small_mask, dtype=np.int32)
    large = np.asarray(large_mask, dtype=np.int32)
    if small.shape != large.shape:
        raise ValueError("small and large masks must have the same shape")
    if not np.any(small) or not np.any(large):
        return np.zeros_like(small, dtype=np.int32)
    rows, cols = np.nonzero(small)
    labels = small[rows, cols].astype(np.int64)
    counts = np.bincount(labels)
    row_sums = np.bincount(labels, weights=rows.astype(float))
    col_sums = np.bincount(labels, weights=cols.astype(float))
    present = np.nonzero(counts[1:])[0] + 1
    centroid_rows = np.clip(
        np.round(row_sums[present] / counts[present]).astype(np.int64),
        0,
        small.shape[0] - 1,
    )
    centroid_cols = np.clip(
        np.round(col_sums[present] / counts[present]).astype(np.int64),
        0,
        small.shape[1] - 1,
    )
    keep = np.unique(large[centroid_rows, centroid_cols])
    keep = keep[keep != 0]
    if keep.size == 0:
        return np.zeros_like(small, dtype=np.int32)
    lookup = np.zeros(int(large.max()) + 1, dtype=bool)
    lookup[keep] = True
    return relabel(np.where(lookup[large], large, 0))


def containment_merge(
    first_mask: np.ndarray,
    second_mask: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Add second-mask cells whose maximum intersection-over-smaller is low."""

    first = clean_mask(first_mask)
    second = clean_mask(second_mask)
    if first.shape != second.shape:
        raise ValueError("first and second masks must have the same shape")
    if int(second.max()) == 0:
        return first
    first_areas = np.bincount(first.ravel())
    second_areas = np.bincount(second.ravel())
    max_ios = max_intersection_over_smaller(first, second, first_areas, second_areas)
    output = first.copy()
    occupied = output > 0
    next_label = int(output.max())
    for second_id, bbox in enumerate(ndi.find_objects(second), start=1):
        if bbox is None or max_ios[second_id] >= float(threshold):
            continue
        region = second[bbox] == second_id
        add_region = region & (~occupied[bbox])
        if not np.any(add_region):
            continue
        components, count = filled_components(add_region)
        view = output[bbox]
        for component_id in range(1, count + 1):
            next_label += 1
            view[components == component_id] = next_label
        occupied[bbox][add_region] = True
    return clean_mask(output)


def constrain_labels_to_tissue(
    labels: np.ndarray,
    tissue_mask: np.ndarray,
) -> np.ndarray:
    """Remove every predicted instance pixel outside an accepted tissue mask."""

    instances = np.asarray(labels)
    tissue = np.asarray(tissue_mask, dtype=bool)
    if instances.ndim != 2 or tissue.shape != instances.shape:
        raise ValueError("labels and tissue mask must have the same 2D shape")
    if not np.issubdtype(instances.dtype, np.integer):
        raise TypeError("labels must contain integers")
    if np.all(tissue):
        return instances
    return np.where(tissue, instances, 0).astype(instances.dtype, copy=False)


def local_stain_evidence(
    image: np.ndarray,
    *,
    background_percentile: float = 98.0,
    minimum_optical_density: float = 0.08,
) -> np.ndarray:
    """Identify pixels darker than a locally estimated brightfield background."""

    rgb = np.asarray(image)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("stain evidence requires an RGB image")
    if not 0 < background_percentile <= 100:
        raise ValueError("background_percentile must be between zero and 100")
    if minimum_optical_density <= 0 or not np.isfinite(minimum_optical_density):
        raise ValueError("minimum_optical_density must be finite and positive")
    if rgb.dtype == np.uint8:
        background = np.percentile(
            rgb.reshape(-1, 3), background_percentile, axis=0
        ).astype(np.float32)
        levels = np.arange(256, dtype=np.float32)
        squared_distance = np.zeros(rgb.shape[:2], dtype=np.float32)
        for channel in range(3):
            lookup = np.log((background[channel] + 1.0) / (levels + 1.0))
            np.maximum(lookup, 0.0, out=lookup)
            squared_distance += np.square(lookup[rgb[:, :, channel]])
        np.sqrt(squared_distance, out=squared_distance)
        return squared_distance >= minimum_optical_density
    values = rgb.astype(np.float32, copy=False)
    background = np.percentile(
        values.reshape(-1, 3), background_percentile, axis=0
    ).astype(np.float32)
    optical_density = np.log((background[None, None, :] + 1.0) / (values + 1.0))
    np.maximum(optical_density, 0.0, out=optical_density)
    return np.linalg.norm(optical_density, axis=2) >= minimum_optical_density


def hematoxylin_evidence(
    image: np.ndarray,
    *,
    background_percentile: float = 98.0,
    minimum_optical_density: float = 0.12,
) -> np.ndarray:
    """Return conservative nuclear-counterstain evidence in brightfield RGB."""

    rgb = np.asarray(image)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("hematoxylin evidence requires an RGB image")
    if not 0 < background_percentile <= 100:
        raise ValueError("background_percentile must be between zero and 100")
    if minimum_optical_density <= 0 or not np.isfinite(minimum_optical_density):
        raise ValueError("minimum_optical_density must be finite and positive")
    concentrations = hematoxylin_concentration(
        rgb, background_percentile=background_percentile
    )
    return concentrations >= minimum_optical_density


def hematoxylin_concentration(
    image: np.ndarray,
    *,
    background_percentile: float = 98.0,
) -> np.ndarray:
    """Estimate nonnegative hematoxylin concentration in brightfield RGB."""

    rgb = np.asarray(image)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("hematoxylin concentration requires an RGB image")
    if not 0 < background_percentile <= 100:
        raise ValueError("background_percentile must be between zero and 100")
    values = rgb.astype(np.float32, copy=False)
    background = np.percentile(
        values.reshape(-1, 3), background_percentile, axis=0
    ).astype(np.float32)
    optical_density = np.log((background[None, None, :] + 1.0) / (values + 1.0))
    np.maximum(optical_density, 0.0, out=optical_density)
    basis = _HEMATOXYLIN_DAB_BASIS
    projection = basis.T @ np.linalg.inv(basis @ basis.T)
    concentrations = optical_density @ projection
    np.maximum(concentrations, 0.0, out=concentrations)
    return concentrations[..., 0]


def retain_nuclear_supported_instances(
    labels: np.ndarray,
    nuclear_evidence: np.ndarray,
    *,
    minimum_pixels: int = 3,
    minimum_fraction: float = 0.005,
) -> np.ndarray:
    """Remove proposed instances without enough enclosed nuclear evidence."""

    instances = clean_mask(labels)
    evidence = np.asarray(nuclear_evidence, dtype=bool)
    if instances.shape != evidence.shape:
        raise ValueError("labels and nuclear evidence must have the same shape")
    if minimum_pixels <= 0:
        raise ValueError("minimum_pixels must be positive")
    if not 0 <= minimum_fraction <= 1:
        raise ValueError("minimum_fraction must be between zero and one")
    if not np.any(instances):
        return instances
    maximum = int(instances.max())
    areas = np.bincount(instances.ravel(), minlength=maximum + 1)
    supported = np.bincount(instances[evidence].ravel(), minlength=maximum + 1)
    required = np.maximum(
        minimum_pixels,
        np.ceil(areas * minimum_fraction).astype(np.int64),
    )
    keep = supported >= required
    keep[0] = False
    return relabel(np.where(keep[instances], instances, 0))


def max_intersection_over_smaller(
    first: np.ndarray,
    second: np.ndarray,
    first_areas: np.ndarray | None = None,
    second_areas: np.ndarray | None = None,
) -> np.ndarray:
    """Return the best pairwise intersection-over-smaller for each second label."""

    first_areas = np.bincount(first.ravel()) if first_areas is None else first_areas
    second_areas = np.bincount(second.ravel()) if second_areas is None else second_areas
    output = np.zeros(len(second_areas), dtype=float)
    overlap = (first > 0) & (second > 0)
    if not np.any(overlap):
        return output
    first_ids = first[overlap].astype(np.int64)
    second_ids = second[overlap].astype(np.int64)
    stride = len(first_areas)
    pair_ids, intersections = np.unique(
        second_ids * stride + first_ids, return_counts=True
    )
    pair_second = pair_ids // stride
    pair_first = pair_ids % stride
    denominators = np.minimum(second_areas[pair_second], first_areas[pair_first])
    ios = np.divide(
        intersections,
        denominators,
        out=np.zeros(intersections.shape, dtype=float),
        where=denominators > 0,
    )
    np.maximum.at(output, pair_second, ios)
    return output


def clean_mask(mask: np.ndarray) -> np.ndarray:
    """Fill instance holes, split disconnected parts, and relabel densely."""

    mask = np.asarray(mask, dtype=np.int32)
    if mask.size == 0 or int(mask.max()) == 0:
        return np.zeros_like(mask, dtype=np.int32)
    output = np.zeros_like(mask, dtype=np.int32)
    next_label = 1
    for label_id, bbox in enumerate(ndi.find_objects(mask), start=1):
        if bbox is None:
            continue
        region = mask[bbox] == label_id
        if not np.any(region):
            continue
        components, count = filled_components(region)
        view = output[bbox]
        for component_id in range(1, count + 1):
            view[components == component_id] = next_label
            next_label += 1
    return output


def filled_components(region: np.ndarray) -> tuple[np.ndarray, int]:
    """Fill internal holes and return four-connected components."""

    filled = ndi.binary_fill_holes(region)
    components, count = ndi.label(filled, structure=_CROSS)
    return components.astype(np.int32, copy=False), int(count)


def relabel(mask: np.ndarray) -> np.ndarray:
    """Map nonzero labels to contiguous positive integers."""

    labels = np.unique(mask)
    labels = labels[labels != 0]
    if labels.size == 0:
        return np.zeros_like(mask, dtype=np.int32)
    mapping = np.zeros(int(labels.max()) + 1, dtype=np.int32)
    mapping[labels] = np.arange(1, labels.size + 1, dtype=np.int32)
    return mapping[np.asarray(mask, dtype=np.int64)]


def label_qc(mask: np.ndarray) -> dict[str, object]:
    """Return compact instance-count and area diagnostics."""

    values = np.asarray(mask)
    areas = np.bincount(values.ravel().astype(np.int64))[1:]
    areas = areas[areas > 0]
    quantiles = (
        np.quantile(areas, [0.05, 0.5, 0.95]).tolist()
        if areas.size
        else [0.0, 0.0, 0.0]
    )
    return {
        "cell_count": int(areas.size),
        "foreground_pixels": int(areas.sum()),
        "area_px_quantiles": [round(float(value), 3) for value in quantiles],
    }
