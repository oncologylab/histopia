"""Conservative detection and exclusion of unrecoverable focus artifacts.

The detector operates on a low-resolution view only to identify candidate
regions.  It deliberately combines darkness, low local detail, component
compactness, and component size: darkness alone is not evidence of an
artifact because valid H&E and chromogen-rich tissue can also be very dark.
Accepted seeds are expanded on native RGB only along a connected dark region,
then whole touching instances can be removed without cutting cell polygons.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi


@dataclass(frozen=True, slots=True)
class FocusArtifactPolicy:
    """Validated thresholds for the compact dark-defocus safeguard."""

    native_block_size: int = 128
    maximum_mean_intensity: float = 120.0
    minimum_dark_fraction: float = 0.30
    dark_intensity_threshold: float = 100.0
    maximum_block_laplacian_variance: float = 1500.0
    minimum_tissue_fraction: float = 0.25
    minimum_component_blocks: int = 12
    minimum_component_fill_fraction: float = 0.70
    maximum_component_aspect_ratio: float = 2.0
    maximum_median_laplacian_variance: float = 700.0
    growth_maximum_smoothed_intensity: float = 145.0
    growth_gaussian_sigma_px: float = 8.0
    growth_minimum_seed_pixels: int = 256


@dataclass(frozen=True, slots=True)
class FocusArtifactComponent:
    """One accepted low-resolution focus-artifact seed component."""

    blocks: int
    block_bbox_rc: tuple[int, int, int, int]
    block_indices_rc: tuple[tuple[int, int], ...]
    fill_fraction: float
    aspect_ratio: float
    median_laplacian_variance: float


def detect_focus_artifact_blocks(
    rgb: np.ndarray,
    tissue: np.ndarray,
    *,
    downsample: float,
    policy: FocusArtifactPolicy | None = None,
) -> tuple[np.ndarray, tuple[FocusArtifactComponent, ...]]:
    """Return accepted block seeds in analysis-pixel coordinates.

    ``rgb`` and ``tissue`` describe the same analysis-resolution view.
    ``downsample`` is its scale relative to native source pixels.  The returned
    mask has the same height and width as the usable, whole-block prefix of
    ``rgb``; callers retain the component block coordinates for provenance.
    """

    active = policy or FocusArtifactPolicy()
    values = np.asarray(rgb)
    tissue_values = np.asarray(tissue, dtype=np.float32)
    if values.ndim != 3 or values.shape[2] < 3:
        raise ValueError("rgb must have shape (height, width, at least 3)")
    if tissue_values.shape != values.shape[:2]:
        raise ValueError("tissue must match the RGB height and width")
    if not np.isfinite(downsample) or downsample <= 0:
        raise ValueError("downsample must be finite and positive")
    if values.size == 0:
        raise ValueError("rgb must be non-empty")
    block = round(active.native_block_size / downsample)
    if block <= 0 or not np.isclose(
        block * downsample, active.native_block_size, atol=0.5
    ):
        raise ValueError("analysis resolution cannot represent native blocks")
    height = (values.shape[0] // block) * block
    width = (values.shape[1] // block) * block
    if height == 0 or width == 0:
        raise ValueError("analysis view is smaller than one focus block")

    values = values[:height, :width, :3].astype(np.float32, copy=False)
    tissue_values = np.clip(tissue_values[:height, :width], 0.0, 1.0)
    gray = values.mean(axis=2)
    laplacian = ndi.laplace(gray)
    block_shape = (height // block, block, width // block, block)
    means = gray.reshape(block_shape).mean(axis=(1, 3))
    dark_fraction = (
        (gray < active.dark_intensity_threshold).reshape(block_shape).mean(axis=(1, 3))
    )
    laplacian_variance = laplacian.reshape(block_shape).var(axis=(1, 3))
    tissue_fraction = tissue_values.reshape(block_shape).mean(axis=(1, 3))
    candidates = (
        (means <= active.maximum_mean_intensity)
        & (dark_fraction >= active.minimum_dark_fraction)
        & (laplacian_variance <= active.maximum_block_laplacian_variance)
        & (tissue_fraction >= active.minimum_tissue_fraction)
    )
    component_labels, _ = ndi.label(candidates, structure=np.ones((3, 3), dtype=bool))
    accepted_mask = np.zeros((height, width), dtype=bool)
    accepted: list[FocusArtifactComponent] = []
    for component_index, component_slice in enumerate(
        ndi.find_objects(component_labels), start=1
    ):
        if component_slice is None:
            continue
        selected = component_labels == component_index
        rows, columns = np.nonzero(selected)
        blocks = int(rows.size)
        if blocks < active.minimum_component_blocks:
            continue
        row_min = int(rows.min())
        row_max = int(rows.max()) + 1
        column_min = int(columns.min())
        column_max = int(columns.max()) + 1
        block_height = row_max - row_min
        block_width = column_max - column_min
        fill = blocks / (block_height * block_width)
        aspect = max(block_height, block_width) / min(block_height, block_width)
        median_laplacian = float(np.median(laplacian_variance[selected]))
        if (
            fill < active.minimum_component_fill_fraction
            or aspect > active.maximum_component_aspect_ratio
            or median_laplacian > active.maximum_median_laplacian_variance
        ):
            continue
        accepted_mask |= np.repeat(np.repeat(selected, block, axis=0), block, axis=1)
        accepted.append(
            FocusArtifactComponent(
                blocks=blocks,
                block_bbox_rc=(
                    row_min,
                    column_min,
                    row_max,
                    column_max,
                ),
                block_indices_rc=tuple(
                    (int(row), int(column))
                    for row, column in zip(rows.tolist(), columns.tolist(), strict=True)
                ),
                fill_fraction=fill,
                aspect_ratio=aspect,
                median_laplacian_variance=median_laplacian,
            )
        )
    return accepted_mask, tuple(accepted)


def grow_focus_artifact_regions(
    rgb: np.ndarray,
    seed_mask: np.ndarray,
    *,
    policy: FocusArtifactPolicy | None = None,
) -> np.ndarray:
    """Grow accepted seeds only through connected dark native RGB regions."""

    active = policy or FocusArtifactPolicy()
    values = np.asarray(rgb)
    seeds = np.asarray(seed_mask, dtype=bool)
    if values.ndim != 3 or values.shape[2] < 3:
        raise ValueError("rgb must have shape (height, width, at least 3)")
    if seeds.shape != values.shape[:2]:
        raise ValueError("seed_mask must match the RGB height and width")
    if not np.any(seeds):
        return np.zeros(seeds.shape, dtype=bool)
    gray = values[..., :3].astype(np.float32, copy=False).mean(axis=2)
    smoothed = ndi.gaussian_filter(gray, active.growth_gaussian_sigma_px)
    growth_candidates = smoothed <= active.growth_maximum_smoothed_intensity
    labels, _ = ndi.label(growth_candidates)
    seed_labels, overlap = np.unique(labels[seeds & (labels > 0)], return_counts=True)
    accepted_labels = seed_labels[overlap >= active.growth_minimum_seed_pixels]
    if accepted_labels.size == 0:
        return np.zeros(seeds.shape, dtype=bool)
    return np.asarray(
        ndi.binary_fill_holes(np.isin(labels, accepted_labels)), dtype=bool
    )


def labels_touching_mask(labels: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return sorted positive instance IDs intersecting ``mask``."""

    values = np.asarray(labels)
    selected = np.asarray(mask, dtype=bool)
    if values.ndim != 2 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("labels must be a 2D integer array")
    if selected.shape != values.shape:
        raise ValueError("mask must match the label geometry")
    return np.unique(values[selected & (values > 0)])


def remove_labels_touching_mask(
    labels: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Remove whole instances touching a focus artifact, preserving dtype."""

    values = np.asarray(labels)
    rejected = labels_touching_mask(values, mask)
    output = values.copy()
    if rejected.size:
        output[np.isin(output, rejected)] = 0
    return output, rejected
