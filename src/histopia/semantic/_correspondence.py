"""Deterministic tile correspondence across adjacent registered sections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class CorrespondenceConfig:
    """Controls sparse coarse-to-fine matching in physical coordinates."""

    patch_width_um: float
    search_radii_patch_widths: tuple[float, ...] = (8.0, 4.0, 2.0)
    context_radii_grid: tuple[int, ...] = (1, 2)
    min_feature_similarity: float = 0.55
    min_reciprocal_margin: float = 0.01
    max_field_residual_patch_widths: float = 1.5
    min_neighborhood_consistency: float = 0.25
    min_confidence: float = 0.45
    field_neighbors: int = 12

    def __post_init__(self) -> None:
        if self.patch_width_um <= 0:
            raise ValueError("patch_width_um must be positive")
        if not self.search_radii_patch_widths or any(
            radius <= 0 for radius in self.search_radii_patch_widths
        ):
            raise ValueError("search radii must be non-empty and positive")
        if any(radius <= 0 for radius in self.context_radii_grid):
            raise ValueError("context radii must be positive")
        if self.field_neighbors <= 0:
            raise ValueError("field_neighbors must be positive")


@dataclass(frozen=True, slots=True)
class AdjacentSectionCorrespondence:
    """Accepted reciprocal tile matches and explicit unmatched tile indices."""

    source_section: int
    target_section: int
    source_indices: np.ndarray
    target_indices: np.ndarray
    confidence: np.ndarray
    feature_similarity: np.ndarray
    reciprocal_margin: np.ndarray
    field_residual_um: np.ndarray
    neighborhood_consistency: np.ndarray
    estimated_displacement_um_xy: np.ndarray
    unmatched_source_indices: np.ndarray
    unmatched_target_indices: np.ndarray


def match_adjacent_sections(
    source_grid_rc: np.ndarray,
    source_um_xy: np.ndarray,
    source_features: np.ndarray,
    target_grid_rc: np.ndarray,
    target_um_xy: np.ndarray,
    target_features: np.ndarray,
    *,
    source_section: int,
    target_section: int,
    config: CorrespondenceConfig,
) -> AdjacentSectionCorrespondence:
    """Match tiles from one section only to the immediately following section."""

    if target_section != source_section + 1:
        raise ValueError("correspondence sections must be adjacent and ordered")
    source_grid_rc, source_um_xy, source_features = _validate_section(
        source_grid_rc, source_um_xy, source_features, name="source"
    )
    target_grid_rc, target_um_xy, target_features = _validate_section(
        target_grid_rc, target_um_xy, target_features, name="target"
    )
    if source_features.shape[1] != target_features.shape[1]:
        raise ValueError("source and target feature dimensions must match")
    if not len(source_um_xy) or not len(target_um_xy):
        return _empty_result(
            source_section, target_section, len(source_um_xy), len(target_um_xy)
        )

    source_descriptor = _context_descriptors(
        source_grid_rc, source_features, config.context_radii_grid
    )
    target_descriptor = _context_descriptors(
        target_grid_rc, target_features, config.context_radii_grid
    )
    field = np.zeros_like(source_um_xy, dtype=np.float64)
    matched_source = np.empty(0, dtype=np.int64)
    matched_target = np.empty(0, dtype=np.int64)
    similarity = np.empty(0, dtype=np.float32)
    margin = np.empty(0, dtype=np.float32)

    for radius_in_patches in config.search_radii_patch_widths:
        matched_source, matched_target, similarity, margin = _reciprocal_matches(
            source_um_xy,
            target_um_xy,
            source_descriptor,
            target_descriptor,
            field,
            radius_in_patches * config.patch_width_um,
        )
        seeds = (similarity >= config.min_feature_similarity) & (
            margin >= config.min_reciprocal_margin
        )
        if np.count_nonzero(seeds) >= 2:
            displacement = (
                target_um_xy[matched_target[seeds]]
                - source_um_xy[matched_source[seeds]]
            )
            field = _smooth_displacement_field(
                source_um_xy,
                source_um_xy[matched_source[seeds]],
                displacement,
                config,
            )

    if not len(matched_source):
        return _result_from_matches(
            source_section,
            target_section,
            len(source_um_xy),
            len(target_um_xy),
            matched_source,
            matched_target,
            field,
            similarity,
            margin,
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32),
        )

    observed = target_um_xy[matched_target] - source_um_xy[matched_source]
    field_residual = np.linalg.norm(observed - field[matched_source], axis=1)
    consistency = _neighborhood_consistency(
        source_um_xy[matched_source], observed, config.patch_width_um
    )
    confidence = _confidence(similarity, margin, field_residual, consistency, config)
    accepted = _accepted_matches(
        similarity, margin, field_residual, consistency, confidence, config
    )
    matched_source = matched_source[accepted]
    matched_target = matched_target[accepted]
    similarity = similarity[accepted]
    margin = margin[accepted]
    while len(matched_source):
        observed = target_um_xy[matched_target] - source_um_xy[matched_source]
        field = _smooth_displacement_field(
            source_um_xy,
            source_um_xy[matched_source],
            observed,
            config,
        )
        field_residual = np.linalg.norm(
            observed - field[matched_source], axis=1
        ).astype(np.float32)
        consistency = _neighborhood_consistency(
            source_um_xy[matched_source], observed, config.patch_width_um
        )
        confidence = _confidence(
            similarity, margin, field_residual, consistency, config
        )
        accepted = _accepted_matches(
            similarity, margin, field_residual, consistency, confidence, config
        )
        if np.all(accepted):
            break
        matched_source = matched_source[accepted]
        matched_target = matched_target[accepted]
        similarity = similarity[accepted]
        margin = margin[accepted]
    if not len(matched_source):
        field = np.zeros_like(source_um_xy, dtype=np.float64)
        field_residual = np.empty(0, dtype=np.float32)
        consistency = np.empty(0, dtype=np.float32)
        confidence = np.empty(0, dtype=np.float32)

    return _result_from_matches(
        source_section,
        target_section,
        len(source_um_xy),
        len(target_um_xy),
        matched_source,
        matched_target,
        field,
        similarity,
        margin,
        field_residual,
        consistency,
        confidence,
    )


def _validate_section(
    grid_rc: np.ndarray,
    um_xy: np.ndarray,
    features: np.ndarray,
    *,
    name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid_rc = np.asarray(grid_rc)
    um_xy = np.asarray(um_xy, dtype=np.float64)
    features = np.asarray(features, dtype=np.float32)
    if grid_rc.ndim != 2 or grid_rc.shape[1:] != (2,):
        raise ValueError(f"{name} grid_rc must have shape (n, 2)")
    if um_xy.ndim != 2 or um_xy.shape[1:] != (2,):
        raise ValueError(f"{name} um_xy must have shape (n, 2)")
    if features.ndim != 2:
        raise ValueError(f"{name} features must have shape (n, d)")
    if not (len(grid_rc) == len(um_xy) == len(features)):
        raise ValueError(f"{name} section arrays must have aligned rows")
    if not np.all(np.isfinite(um_xy)) or not np.all(np.isfinite(features)):
        raise ValueError(f"{name} coordinates and features must be finite")
    return grid_rc.astype(np.int64, copy=False), um_xy, features


def _context_descriptors(
    grid_rc: np.ndarray,
    features: np.ndarray,
    radii: tuple[int, ...],
) -> np.ndarray:
    normalized = _normalize_rows(features)
    lookup = {tuple(rc): index for index, rc in enumerate(grid_rc.tolist())}
    parts = [normalized]
    for radius in radii:
        directional = np.zeros((len(grid_rc), 8, normalized.shape[1]), np.float32)
        counts = np.zeros((len(grid_rc), 8), dtype=np.int16)
        for index, (row, column) in enumerate(grid_rc):
            for row_offset in range(-radius, radius + 1):
                for column_offset in range(-radius, radius + 1):
                    if max(abs(row_offset), abs(column_offset)) != radius:
                        continue
                    neighbor = lookup.get(
                        (int(row + row_offset), int(column + column_offset))
                    )
                    if neighbor is None:
                        continue
                    direction = _direction_bin(row_offset, column_offset)
                    directional[index, direction] += normalized[neighbor]
                    counts[index, direction] += 1
        directional /= np.maximum(counts[..., None], 1)
        parts.append(directional.reshape(len(grid_rc), -1))
        parts.append((counts > 0).astype(np.float32))
    return _normalize_rows(np.concatenate(parts, axis=1))


def _direction_bin(row_offset: int, column_offset: int) -> int:
    angle = np.arctan2(-row_offset, column_offset)
    return int(np.floor((angle + np.pi / 8) / (np.pi / 4))) % 8


def _reciprocal_matches(
    source_xy: np.ndarray,
    target_xy: np.ndarray,
    source_descriptor: np.ndarray,
    target_descriptor: np.ndarray,
    field: np.ndarray,
    radius_um: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tree = _ckdtree(target_xy)
    candidates = tree.query_ball_point(source_xy + field, radius_um)
    source_best = np.full(len(source_xy), -1, dtype=np.int64)
    source_margin = np.zeros(len(source_xy), dtype=np.float32)
    best_similarity = np.full(len(source_xy), -1.0, dtype=np.float32)
    target_best = np.full(len(target_xy), -1, dtype=np.int64)
    target_best_score = np.full(len(target_xy), -np.inf, dtype=np.float32)
    target_second_score = np.full(len(target_xy), -np.inf, dtype=np.float32)
    for source_index, candidate_indices in enumerate(candidates):
        indices = np.asarray(sorted(candidate_indices), dtype=np.int64)
        if not len(indices):
            continue
        similarities = target_descriptor[indices] @ source_descriptor[source_index]
        order = np.lexsort((indices, -similarities))
        best = int(order[0])
        source_best[source_index] = indices[best]
        best_similarity[source_index] = similarities[best]
        source_margin[source_index] = (
            float(similarities[best] - similarities[order[1]])
            if len(order) > 1
            else 0.0
        )
        for target_index, score in zip(indices, similarities, strict=True):
            target_index = int(target_index)
            current_score = float(target_best_score[target_index])
            current_source = int(target_best[target_index])
            if score > current_score or (
                score == current_score and source_index < current_source
            ):
                target_second_score[target_index] = current_score
                target_best_score[target_index] = score
                target_best[target_index] = source_index
            elif score > target_second_score[target_index]:
                target_second_score[target_index] = score

    has_runner_up = np.isfinite(target_second_score)
    target_margin = np.zeros(len(target_xy), dtype=np.float32)
    target_margin[has_runner_up] = (
        target_best_score[has_runner_up] - target_second_score[has_runner_up]
    )

    valid = source_best >= 0
    reciprocal = np.zeros(len(source_xy), dtype=bool)
    reciprocal[valid] = target_best[source_best[valid]] == np.flatnonzero(valid)
    matched_source = np.flatnonzero(reciprocal)
    matched_target = source_best[matched_source]
    margin = np.minimum(source_margin[matched_source], target_margin[matched_target])
    return (
        matched_source,
        matched_target,
        best_similarity[matched_source],
        margin.astype(np.float32),
    )


def _smooth_displacement_field(
    query_xy: np.ndarray,
    matched_xy: np.ndarray,
    displacement: np.ndarray,
    config: CorrespondenceConfig,
) -> np.ndarray:
    if not len(matched_xy):
        return np.zeros_like(query_xy, dtype=np.float64)
    neighbor_count = min(config.field_neighbors, len(matched_xy))
    distance, indices = _ckdtree(matched_xy).query(query_xy, k=neighbor_count)
    distance = np.asarray(distance, dtype=np.float64)
    indices = np.asarray(indices, dtype=np.int64)
    if neighbor_count == 1:
        distance = distance[:, None]
        indices = indices[:, None]
    local = displacement[indices]
    center = np.median(local, axis=1)
    deviation = np.linalg.norm(local - center[:, None, :], axis=2)
    spatial_weight = np.exp(-0.5 * (distance / (3.0 * config.patch_width_um)) ** 2)
    robust_weight = np.exp(-0.5 * (deviation / config.patch_width_um) ** 2)
    weight = spatial_weight * robust_weight
    return np.sum(weight[..., None] * local, axis=1) / np.maximum(
        np.sum(weight, axis=1, keepdims=True), np.finfo(float).eps
    )


def _neighborhood_consistency(
    matched_xy: np.ndarray,
    displacement: np.ndarray,
    patch_width_um: float,
) -> np.ndarray:
    neighborhoods = _ckdtree(matched_xy).query_ball_point(
        matched_xy, 3.0 * patch_width_um
    )
    consistency = np.empty(len(matched_xy), dtype=np.float32)
    for index, neighbors in enumerate(neighborhoods):
        neighbors = np.asarray(neighbors, dtype=np.int64)
        neighbors = neighbors[neighbors != index]
        if not len(neighbors):
            consistency[index] = 0.0
            continue
        delta = np.linalg.norm(displacement[neighbors] - displacement[index], axis=1)
        consistency[index] = float(
            np.mean(np.exp(-0.5 * (delta / patch_width_um) ** 2))
        )
    return consistency


def _confidence(
    similarity: np.ndarray,
    margin: np.ndarray,
    field_residual: np.ndarray,
    consistency: np.ndarray,
    config: CorrespondenceConfig,
) -> np.ndarray:
    feature_score = np.clip(similarity, 0.0, 1.0)
    margin_score = np.clip(margin / 0.2, 0.0, 1.0)
    field_score = np.exp(-0.5 * (field_residual / config.patch_width_um) ** 2)
    return np.power(
        feature_score * margin_score * field_score * consistency, 0.25
    ).astype(np.float32)


def _accepted_matches(
    similarity: np.ndarray,
    margin: np.ndarray,
    field_residual: np.ndarray,
    consistency: np.ndarray,
    confidence: np.ndarray,
    config: CorrespondenceConfig,
) -> np.ndarray:
    return (
        (similarity >= config.min_feature_similarity)
        & (margin >= config.min_reciprocal_margin)
        & (
            field_residual
            <= config.max_field_residual_patch_widths * config.patch_width_um
        )
        & (consistency >= config.min_neighborhood_consistency)
        & (confidence >= config.min_confidence)
    )


def _empty_result(
    source_section: int,
    target_section: int,
    source_count: int,
    target_count: int,
) -> AdjacentSectionCorrespondence:
    return _result_from_matches(
        source_section,
        target_section,
        source_count,
        target_count,
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.int64),
        np.zeros((source_count, 2), dtype=np.float32),
        np.empty(0, dtype=np.float32),
        np.empty(0, dtype=np.float32),
        np.empty(0, dtype=np.float32),
        np.empty(0, dtype=np.float32),
        np.empty(0, dtype=np.float32),
    )


def _result_from_matches(
    source_section: int,
    target_section: int,
    source_count: int,
    target_count: int,
    source_indices: np.ndarray,
    target_indices: np.ndarray,
    field: np.ndarray,
    similarity: np.ndarray,
    margin: np.ndarray,
    field_residual: np.ndarray,
    consistency: np.ndarray,
    confidence: np.ndarray,
) -> AdjacentSectionCorrespondence:
    unmatched_source = np.setdiff1d(
        np.arange(source_count, dtype=np.int64), source_indices, assume_unique=True
    )
    unmatched_target = np.setdiff1d(
        np.arange(target_count, dtype=np.int64), target_indices, assume_unique=True
    )
    return AdjacentSectionCorrespondence(
        source_section=source_section,
        target_section=target_section,
        source_indices=source_indices.astype(np.int64, copy=False),
        target_indices=target_indices.astype(np.int64, copy=False),
        confidence=np.asarray(confidence, dtype=np.float32),
        feature_similarity=np.asarray(similarity, dtype=np.float32),
        reciprocal_margin=np.asarray(margin, dtype=np.float32),
        field_residual_um=np.asarray(field_residual, dtype=np.float32),
        neighborhood_consistency=np.asarray(consistency, dtype=np.float32),
        estimated_displacement_um_xy=np.asarray(field, dtype=np.float32),
        unmatched_source_indices=unmatched_source,
        unmatched_target_indices=unmatched_target,
    )


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, np.finfo(np.float32).eps)


def _ckdtree(values: np.ndarray) -> Any:
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise RuntimeError("semantic correspondence requires scipy") from exc
    return cKDTree(values)
