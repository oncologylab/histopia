"""Resolution-faithful aggregation of stain measurements into cells."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class CellMeasurements:
    """Per-cell summaries calculated from weighted analysis-pixel overlaps."""

    label_ids: np.ndarray
    mean_od: np.ndarray
    median_od: np.ndarray
    q90_od: np.ndarray
    positive_area_fraction: np.ndarray
    effective_pixels: np.ndarray
    coverage: np.ndarray
    binary_label: np.ndarray
    measured: np.ndarray


def aggregate_cell_measurements(
    cell_ids: np.ndarray,
    target_od: np.ndarray,
    *,
    overlap_weights: np.ndarray | None = None,
    positive: np.ndarray | None = None,
    valid: np.ndarray | None = None,
    expected_area: np.ndarray | None = None,
    positive_area_fraction: float = 0.25,
    negative_area_fraction: float = 0.05,
    minimum_effective_pixels: float = 2.0,
    minimum_coverage: float = 0.70,
) -> CellMeasurements:
    """Aggregate arbitrary fractional stain-pixel/cell overlaps.

    Each input row represents one overlap between a cell and a ``4 µm/px``
    stain-map pixel. Repeating a pixel for multiple cells with fractional
    ``overlap_weights`` preserves boundaries without upsampling the OD map.
    """

    labels = np.asarray(cell_ids)
    od = np.asarray(target_od, dtype=np.float64)
    if labels.ndim != 1 or od.ndim != 1 or labels.shape != od.shape:
        raise ValueError("cell_ids and target_od must be matching vectors")
    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError("cell_ids must contain integers")
    if np.any(labels < 0) or not np.all(np.isfinite(od)) or np.any(od < 0):
        raise ValueError("cell IDs and OD values must be finite and nonnegative")
    weights = (
        np.ones(labels.shape, dtype=np.float64)
        if overlap_weights is None
        else np.asarray(overlap_weights, dtype=np.float64)
    )
    validity = (
        np.ones(labels.shape, dtype=bool)
        if valid is None
        else np.asarray(valid, dtype=bool)
    )
    if weights.shape != labels.shape or validity.shape != labels.shape:
        raise ValueError("weights and validity must match cell_ids")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("overlap weights must be finite and nonnegative")
    if not 0 <= negative_area_fraction < positive_area_fraction <= 1:
        raise ValueError("binary area-fraction thresholds are invalid")
    if minimum_effective_pixels <= 0 or not 0 <= minimum_coverage <= 1:
        raise ValueError("measurement coverage thresholds are invalid")

    foreground = (labels > 0) & validity & (weights > 0)
    unique = np.unique(labels[labels > 0]).astype(np.uint32)
    count = len(unique)
    means = np.full(count, np.nan, dtype=np.float32)
    medians = np.full(count, np.nan, dtype=np.float32)
    q90 = np.full(count, np.nan, dtype=np.float32)
    positive_fraction = np.full(count, np.nan, dtype=np.float32)
    effective = np.zeros(count, dtype=np.float32)
    coverage = np.zeros(count, dtype=np.float32)
    binary = np.full(count, -1, dtype=np.int8)

    expected_lookup = _expected_area_lookup(unique, expected_area)
    positive_values = None if positive is None else np.asarray(positive, dtype=bool)
    if positive_values is not None and positive_values.shape != labels.shape:
        raise ValueError("positive must match cell_ids")
    for index, label_id in enumerate(unique):
        selected = foreground & (labels == label_id)
        cell_weights = weights[selected]
        total = float(cell_weights.sum())
        effective[index] = total
        denominator = expected_lookup[index]
        coverage[index] = min(1.0, total / denominator) if denominator > 0 else 0.0
        if total <= 0:
            continue
        cell_od = od[selected]
        means[index] = np.average(cell_od, weights=cell_weights)
        medians[index] = _weighted_quantile(cell_od, cell_weights, 0.5)
        q90[index] = _weighted_quantile(cell_od, cell_weights, 0.9)
        if positive_values is not None:
            fraction = float(cell_weights[positive_values[selected]].sum()) / total
            positive_fraction[index] = fraction
            if fraction >= positive_area_fraction:
                binary[index] = 1
            elif fraction <= negative_area_fraction:
                binary[index] = 0
    measured = (effective >= minimum_effective_pixels) & (coverage >= minimum_coverage)
    binary[~measured] = -1
    return CellMeasurements(
        label_ids=unique,
        mean_od=means,
        median_od=medians,
        q90_od=q90,
        positive_area_fraction=positive_fraction,
        effective_pixels=effective,
        coverage=coverage,
        binary_label=binary,
        measured=measured,
    )


def inner_boundary_mask(labels: np.ndarray, *, width_px: int = 1) -> np.ndarray:
    """Return a label-preserving inward boundary band without SciPy."""

    values = np.asarray(labels)
    if values.ndim != 2 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("labels must be a two-dimensional integer array")
    if width_px < 1:
        raise ValueError("width_px must be positive")
    interior = values.copy()
    for _ in range(width_px):
        same = np.ones(values.shape, dtype=bool)
        same[1:, :] &= interior[1:, :] == interior[:-1, :]
        same[:-1, :] &= interior[:-1, :] == interior[1:, :]
        same[:, 1:] &= interior[:, 1:] == interior[:, :-1]
        same[:, :-1] &= interior[:, :-1] == interior[:, 1:]
        interior = np.where(same, interior, 0)
    return np.where((values > 0) & (interior == 0), values, 0)


def _expected_area_lookup(
    unique: np.ndarray, expected_area: np.ndarray | None
) -> np.ndarray:
    if expected_area is None:
        return np.ones(len(unique), dtype=np.float64)
    values = np.asarray(expected_area, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("expected_area must be a positive finite vector")
    if len(values) == len(unique):
        return values
    if len(values) > int(unique.max(initial=0)):
        return values[unique]
    raise ValueError("expected_area must align with label_ids or label index")


def _weighted_quantile(
    values: np.ndarray, weights: np.ndarray, quantile: float
) -> float:
    order = np.argsort(values, kind="stable")
    ordered_values = values[order]
    cumulative = np.cumsum(weights[order])
    position = quantile * cumulative[-1]
    return float(
        ordered_values[min(np.searchsorted(cumulative, position), len(order) - 1)]
    )
