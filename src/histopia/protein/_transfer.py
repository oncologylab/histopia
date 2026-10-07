"""Morphology-first transfer helpers for serial-section protein prediction.

Semantic regions are never primary predictors in this module.  The registered
anchor transfer may use them only as a soft mismatch penalty because a single
semantic region can contain several cell states and types.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class RegisteredAnchorTransfer:
    """Cell-resolved transfer from measured serial-section anchor cells.

    Unsupported query cells have ``NaN`` predictions and zero confidence.  A
    caller can therefore fall back to an inductive model without silently
    extrapolating a measured section across unrelated tissue.
    """

    prediction: np.ndarray
    confidence: np.ndarray
    uncertainty: np.ndarray
    support_count: np.ndarray
    nearest_distance_um: np.ndarray


def fixed_stain_neutral_projection(
    morphology: np.ndarray,
    *,
    components: int = 32,
    seed: int = 0x48495354,
    batch_size: int = 20_000,
) -> np.ndarray:
    """Compress high-resolution morphology without fitting on protein outcomes.

    A fixed Rademacher projection makes per-cell registered matching tractable
    while ensuring that measured OD, marker identity, and semantic labels can
    never influence the projection.  Identical inputs and controls always
    produce bitwise-identical keys.
    """

    values = _matrix("morphology", morphology)
    if components < 1 or batch_size < 1:
        raise ValueError("projection controls must be positive")
    generator = np.random.default_rng(seed)
    projection = generator.choice(
        np.asarray((-1.0, 1.0), dtype=np.float32),
        size=(values.shape[1], components),
    ) / np.sqrt(max(values.shape[1], 1))
    output = np.empty((len(values), components), dtype=np.float32)
    for start in range(0, len(values), batch_size):
        stop = min(start + batch_size, len(values))
        output[start:stop] = values[start:stop] @ projection
    return output


def registered_morphospatial_anchor_transfer(
    query_morphology: np.ndarray,
    query_reference_xyz_um: np.ndarray,
    anchor_morphology: np.ndarray,
    anchor_reference_xyz_um: np.ndarray,
    anchor_od: np.ndarray,
    *,
    query_regions: np.ndarray | None = None,
    anchor_regions: np.ndarray | None = None,
    query_anchor_indices: np.ndarray | None = None,
    spatial_candidates: int = 64,
    neighbors: int = 8,
    xy_bandwidth_um: float = 96.0,
    z_bandwidth_um: float = 32.0,
    maximum_xy_distance_um: float = 384.0,
    minimum_cosine_similarity: float = 0.0,
    morphology_temperature: float = 0.20,
    semantic_mismatch_weight: float = 0.65,
    aggregation: str = "weighted_median",
    allow_signed_outcomes: bool = False,
    exclude_exact_matches: bool = True,
    batch_size: int = 128,
    workers: int = 1,
) -> RegisteredAnchorTransfer:
    """Transfer measured OD using registered location and cell morphology.

    Candidate anchors are found in registered XY space and then reranked by
    anisotropic XYZ distance and stain-neutral morphology.  Semantic regions
    are optional and only down-weight mismatches; they can never create a
    prediction on their own.  Exact same-cell matches are excluded so the
    primitive can also be evaluated on held-out cells from a measured section.

    ``query_anchor_indices`` may contain an aligned anchor index for each query
    cell, or ``-1`` when no identity is shared.  It is the most reliable way to
    exclude self matches.  Coordinate-and-morphology duplicates are also
    excluded automatically when ``exclude_exact_matches`` is true.

    Set ``allow_signed_outcomes`` only when transferring an auditable quantity
    such as measured-minus-model OD residuals.  The default retains the strict
    non-negative validation expected for absolute OD measurements.
    """

    query = _matrix("query_morphology", query_morphology)
    query_xyz = _matrix(
        "query_reference_xyz_um", query_reference_xyz_um, rows=len(query)
    ).astype(np.float64)
    anchor = _matrix("anchor_morphology", anchor_morphology)
    anchor_xyz = _matrix(
        "anchor_reference_xyz_um", anchor_reference_xyz_um, rows=len(anchor)
    ).astype(np.float64)
    outcomes = np.asarray(anchor_od, dtype=np.float64)
    if query.shape[1] != anchor.shape[1] or query.shape[1] == 0:
        raise ValueError("query and anchor morphology widths must match")
    if query_xyz.shape[1] != 3 or anchor_xyz.shape[1] != 3:
        raise ValueError("registered coordinates must have shape (cells, 3)")
    if len(anchor) < 2:
        raise ValueError("at least two measured anchor cells are required")
    if outcomes.ndim != 1 or len(outcomes) != len(anchor):
        raise ValueError("anchor_od must be aligned with anchor cells")
    if not isinstance(allow_signed_outcomes, bool):
        raise ValueError("allow_signed_outcomes must be boolean")
    if not np.all(np.isfinite(outcomes)) or (
        not allow_signed_outcomes and np.any(outcomes < 0)
    ):
        requirement = "finite" if allow_signed_outcomes else "finite and non-negative"
        raise ValueError(f"anchor_od must be {requirement}")

    controls = (
        xy_bandwidth_um,
        z_bandwidth_um,
        maximum_xy_distance_um,
        morphology_temperature,
    )
    if not all(np.isfinite(value) and value > 0 for value in controls):
        raise ValueError("distance and morphology controls must be positive")
    if (
        spatial_candidates < 1
        or neighbors < 1
        or batch_size < 1
        or neighbors > spatial_candidates
        or workers == 0
        or workers < -1
    ):
        raise ValueError("candidate, neighbor, and batch controls are invalid")
    if not -1 <= minimum_cosine_similarity <= 1:
        raise ValueError("minimum_cosine_similarity must be between -1 and 1")
    if not 0 < semantic_mismatch_weight <= 1:
        raise ValueError("semantic_mismatch_weight must be in (0, 1]")
    if aggregation not in {"weighted_mean", "weighted_median"}:
        raise ValueError("aggregation must be weighted_mean or weighted_median")

    query_region, anchor_region = _aligned_regions(
        query_regions,
        anchor_regions,
        query_count=len(query),
        anchor_count=len(anchor),
    )
    self_index = _self_indices(
        query_anchor_indices,
        query_count=len(query),
        anchor_count=len(anchor),
    )

    # Standardize with anchor-only statistics.  Candidate morphology is
    # normalized lazily by batch so 1,536-dimensional UNI2-h matrices do not
    # require another full-size in-memory copy.
    center = np.mean(anchor, axis=0, dtype=np.float64)
    scale = np.std(anchor, axis=0, dtype=np.float64)
    scale = np.where(scale > 1e-6, scale, 1.0)

    from scipy.spatial import cKDTree

    tree = cKDTree(anchor_xyz[:, :2])
    candidate_count = min(int(spatial_candidates), len(anchor))
    prediction = np.full(len(query), np.nan, dtype=np.float32)
    confidence = np.zeros(len(query), dtype=np.float32)
    uncertainty = np.full(len(query), np.nan, dtype=np.float32)
    support_count = np.zeros(len(query), dtype=np.int32)
    nearest_distance = np.full(len(query), np.inf, dtype=np.float32)
    anchor_od_scale = max(
        float(np.subtract(*np.quantile(outcomes, [0.75, 0.25]))),
        1e-6,
    )

    for start in range(0, len(query), batch_size):
        stop = min(start + batch_size, len(query))
        xy_distance, index = tree.query(
            query_xyz[start:stop, :2],
            k=candidate_count,
            workers=workers,
        )
        xy_distance = np.asarray(xy_distance, dtype=np.float64)
        index = np.asarray(index, dtype=np.int64)
        if candidate_count == 1:
            xy_distance = xy_distance[:, None]
            index = index[:, None]

        query_key = _standardized_unit_rows(query[start:stop], center, scale)
        anchor_key = _standardized_unit_rows(anchor[index], center, scale)
        similarity = np.einsum("bd,bkd->bk", query_key, anchor_key)
        delta = anchor_xyz[index] - query_xyz[start:stop, None, :]
        z_distance = np.abs(delta[:, :, 2])
        scaled_distance = np.sqrt(
            (xy_distance / xy_bandwidth_um) ** 2 + (z_distance / z_bandwidth_um) ** 2
        )
        cost = 0.5 * scaled_distance**2 + (1.0 - similarity) / morphology_temperature
        eligible = (xy_distance <= maximum_xy_distance_um) & (
            similarity >= minimum_cosine_similarity
        )

        if query_region is not None and anchor_region is not None:
            mismatch = anchor_region[index] != query_region[start:stop, None]
            cost = cost - mismatch * np.log(semantic_mismatch_weight)

        if self_index is not None:
            eligible &= index != self_index[start:stop, None]
        if exclude_exact_matches:
            same_coordinate = np.all(np.abs(delta) <= 1e-4, axis=2)
            same_morphology = np.all(
                np.isclose(
                    anchor[index],
                    query[start:stop, None, :],
                    rtol=1e-6,
                    atol=1e-7,
                ),
                axis=2,
            )
            eligible &= ~(same_coordinate & same_morphology)
        cost = np.where(eligible, cost, np.inf)

        order = np.argsort(cost, axis=1, kind="stable")[:, :neighbors]
        selected_cost = np.take_along_axis(cost, order, axis=1)
        selected_index = np.take_along_axis(index, order, axis=1)
        selected_similarity = np.take_along_axis(similarity, order, axis=1)
        selected_xy = np.take_along_axis(xy_distance, order, axis=1)
        selected_z = np.take_along_axis(z_distance, order, axis=1)
        valid = np.isfinite(selected_cost)
        counts = np.sum(valid, axis=1)
        supported = counts > 0
        if not np.any(supported):
            continue

        row_min = np.min(selected_cost, axis=1, where=valid, initial=np.inf)
        relative_cost = np.zeros_like(selected_cost)
        np.subtract(
            selected_cost,
            row_min[:, None],
            out=relative_cost,
            where=valid,
        )
        weights = np.exp(
            -np.clip(relative_cost, 0.0, 80.0),
            where=valid,
            out=np.zeros_like(selected_cost),
        )
        weight_sum = np.sum(weights, axis=1)
        weights = np.divide(
            weights,
            weight_sum[:, None],
            out=np.zeros_like(weights),
            where=weight_sum[:, None] > 0,
        )
        selected_od = outcomes[selected_index]
        if aggregation == "weighted_median":
            transferred = _row_weighted_median(selected_od, weights, valid)
        else:
            transferred = np.sum(weights * selected_od, axis=1)
        dispersion = np.sqrt(
            np.sum(weights * (selected_od - transferred[:, None]) ** 2, axis=1)
        )
        selected_distance = np.sqrt(selected_xy**2 + selected_z**2)
        closest = np.min(selected_distance, axis=1, where=valid, initial=np.inf)
        morphology_quality = np.sum(
            weights
            * np.clip(
                (selected_similarity - minimum_cosine_similarity)
                / max(1.0 - minimum_cosine_similarity, 1e-6),
                0.0,
                1.0,
            ),
            axis=1,
        )
        spatial_quality = np.exp(
            -0.5 * np.min(scaled_distance, axis=1, where=eligible, initial=np.inf) ** 2
        )
        effective_support = np.divide(
            1.0,
            np.sum(weights**2, axis=1),
            out=np.zeros(len(weights), dtype=np.float64),
            where=np.sum(weights**2, axis=1) > 0,
        )
        support_quality = np.sqrt(
            np.clip(effective_support / min(neighbors, 4), 0.0, 1.0)
        )
        agreement = 1.0 / (1.0 + dispersion / anchor_od_scale)
        quality = np.clip(
            morphology_quality * spatial_quality * support_quality * agreement,
            0.0,
            1.0,
        )

        batch_rows = np.arange(start, stop)[supported]
        prediction[batch_rows] = transferred[supported].astype(np.float32)
        confidence[batch_rows] = quality[supported].astype(np.float32)
        uncertainty[batch_rows] = dispersion[supported].astype(np.float32)
        support_count[batch_rows] = counts[supported].astype(np.int32)
        nearest_distance[batch_rows] = closest[supported].astype(np.float32)

    return RegisteredAnchorTransfer(
        prediction=prediction,
        confidence=confidence,
        uncertainty=uncertainty,
        support_count=support_count,
        nearest_distance_um=nearest_distance,
    )


def morphospatial_features(
    morphology: np.ndarray,
    reference_xyz_um: np.ndarray,
    *,
    shape_features: np.ndarray | None = None,
    neighborhood_features: np.ndarray | None = None,
    coordinate_scale_um: float = 256.0,
) -> np.ndarray:
    """Combine stain-neutral morphology, cell geometry, neighborhood, and xyz.

    Coordinates are expressed in registered physical space and scaled to keep
    them numerically comparable with standardized morphology features.  The
    function never accepts a semantic label, preventing accidental use of the
    coarse segmentation as a primary feature.
    """

    morphology = _matrix("morphology", morphology)
    xyz = _matrix("reference_xyz_um", reference_xyz_um, rows=len(morphology))
    if xyz.shape[1] != 3:
        raise ValueError("reference_xyz_um must have shape (cells, 3)")
    if not np.isfinite(coordinate_scale_um) or coordinate_scale_um <= 0:
        raise ValueError("coordinate_scale_um must be positive and finite")
    parts = [morphology]
    if shape_features is not None:
        parts.append(_matrix("shape_features", shape_features, rows=len(morphology)))
    if neighborhood_features is not None:
        parts.append(
            _matrix(
                "neighborhood_features",
                neighborhood_features,
                rows=len(morphology),
            )
        )
    parts.append(xyz / float(coordinate_scale_um))
    return np.concatenate(parts, axis=1).astype(np.float32)


def select_bracketing_sections(
    target_z_um: float, measured_section_z_um: dict[str, float]
) -> tuple[str, ...]:
    """Select bracketing measured sections, or the nearest one outside range."""

    if not measured_section_z_um:
        raise ValueError("at least one measured section is required")
    if not np.isfinite(target_z_um) or any(
        not np.isfinite(value) for value in measured_section_z_um.values()
    ):
        raise ValueError("section z coordinates must be finite")
    ordered = sorted(measured_section_z_um.items(), key=lambda item: item[1])
    below = [item for item in ordered if item[1] <= target_z_um]
    above = [item for item in ordered if item[1] >= target_z_um]
    if below and above:
        left, right = below[-1], above[0]
        return (left[0],) if left[0] == right[0] else (left[0], right[0])
    nearest = min(ordered, key=lambda item: abs(item[1] - target_z_um))
    return (nearest[0],)


def morphology_compatible_smoothing(
    values: np.ndarray,
    morphology: np.ndarray,
    reference_xyz_um: np.ndarray,
    *,
    neighbors: int = 8,
    maximum_distance_um: float = 96.0,
    minimum_cosine_similarity: float = 0.75,
    blend: float = 0.35,
) -> np.ndarray:
    """Smooth only across nearby cells with compatible morphology.

    This avoids the biologically unsafe assumption that all spatial neighbors
    share expression.  It is dependency-light and intended for deterministic
    post-processing and candidate-model tests.
    """

    y = np.asarray(values, dtype=np.float64)
    features = _matrix("morphology", morphology, rows=len(y)).astype(np.float64)
    xyz = _matrix("reference_xyz_um", reference_xyz_um, rows=len(y)).astype(np.float64)
    if y.ndim != 1 or not np.all(np.isfinite(y)):
        raise ValueError("values must be a finite vector")
    if neighbors < 1 or maximum_distance_um <= 0:
        raise ValueError("neighbor controls must be positive")
    if not -1 <= minimum_cosine_similarity <= 1 or not 0 <= blend <= 1:
        raise ValueError("similarity and blend controls are outside their range")
    norms = np.linalg.norm(features, axis=1)
    normalized = np.divide(
        features,
        norms[:, None],
        out=np.zeros_like(features),
        where=norms[:, None] > 0,
    )
    output = y.copy()
    for index in range(len(y)):
        distance = np.linalg.norm(xyz - xyz[index], axis=1)
        similarity = normalized @ normalized[index]
        eligible = np.flatnonzero(
            (distance > 0)
            & (distance <= maximum_distance_um)
            & (similarity >= minimum_cosine_similarity)
        )
        if not len(eligible):
            continue
        eligible = eligible[np.argsort(distance[eligible])[:neighbors]]
        weight = similarity[eligible] / np.maximum(distance[eligible], 1.0)
        neighbor_value = np.average(y[eligible], weights=weight)
        output[index] = (1.0 - blend) * y[index] + blend * neighbor_value
    return output.astype(np.float32)


def _aligned_regions(
    query_regions: np.ndarray | None,
    anchor_regions: np.ndarray | None,
    *,
    query_count: int,
    anchor_count: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if (query_regions is None) != (anchor_regions is None):
        raise ValueError("query_regions and anchor_regions must be provided together")
    if query_regions is None or anchor_regions is None:
        return None, None
    query = np.asarray(query_regions)
    anchor = np.asarray(anchor_regions)
    if query.ndim != 1 or len(query) != query_count:
        raise ValueError("query_regions must be aligned with query cells")
    if anchor.ndim != 1 or len(anchor) != anchor_count:
        raise ValueError("anchor_regions must be aligned with anchor cells")
    return query, anchor


def _self_indices(
    value: np.ndarray | None,
    *,
    query_count: int,
    anchor_count: int,
) -> np.ndarray | None:
    if value is None:
        return None
    index = np.asarray(value)
    if (
        index.ndim != 1
        or len(index) != query_count
        or not np.issubdtype(index.dtype, np.integer)
    ):
        raise ValueError("query_anchor_indices must be an aligned integer vector")
    index = index.astype(np.int64, copy=False)
    if np.any(index < -1) or np.any(index >= anchor_count):
        raise ValueError("query_anchor_indices contains an invalid anchor index")
    return index


def _standardized_unit_rows(
    values: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    standardized = (np.asarray(values, dtype=np.float64) - center) / scale
    norms = np.linalg.norm(standardized, axis=-1, keepdims=True)
    return np.divide(
        standardized,
        norms,
        out=np.zeros_like(standardized),
        where=norms > 0,
    )


def _row_weighted_median(
    values: np.ndarray,
    weights: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    order = np.argsort(values, axis=1, kind="stable")
    ordered_value = np.take_along_axis(values, order, axis=1)
    ordered_weight = np.take_along_axis(weights, order, axis=1)
    ordered_valid = np.take_along_axis(valid, order, axis=1)
    cumulative = np.cumsum(np.where(ordered_valid, ordered_weight, 0.0), axis=1)
    median_index = np.argmax(cumulative >= 0.5, axis=1)
    return ordered_value[np.arange(len(values)), median_index]


def _matrix(name: str, value: np.ndarray, *, rows: int | None = None) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.ndim != 2 or (rows is not None and len(matrix) != rows):
        raise ValueError(f"{name} must be a two-dimensional aligned matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be finite")
    return matrix
