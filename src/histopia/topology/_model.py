"""Dependency-light data structures and transition calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ObservedSection:
    """One selected-K semantic section rasterized in reference coordinates."""

    slide_id: str
    labels: np.ndarray
    membership: np.ndarray
    support: np.ndarray
    tissue_fraction: np.ndarray
    sparse_labels: np.ndarray


@dataclass(frozen=True, slots=True)
class PairEvidence:
    """Comparable evidence for one adjacent observed section pair."""

    source_section: int
    target_section: int
    score: float
    support_dice: float
    semantic_js: float
    matched_label_agreement: float
    correspondence_coverage: float
    median_confidence: float
    displacement_patch_widths: float
    displacement_strain: float


@dataclass(frozen=True, slots=True)
class GapDecision:
    """Auditable z-interval decision for one observed pair."""

    source_section: int
    target_section: int
    intervals: int
    missing_sections: int
    status: str
    confidence: float
    score: float
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MorphologyGapCalibrator:
    """Robust within-stack centers for one-to-many section intervals."""

    centers: np.ndarray
    scale: np.ndarray
    counts: np.ndarray


@dataclass(frozen=True, slots=True)
class ReconstructedPlane:
    """Observed or inferred semantic field at one z coordinate."""

    z_um: float
    segment: int
    source_section: int
    target_section: int
    fraction: float
    observed: bool
    slide_id: str | None
    membership: np.ndarray
    labels: np.ndarray
    support: np.ndarray
    uncertainty: np.ndarray


def pair_evidence(
    source: ObservedSection,
    target: ObservedSection,
    *,
    source_indices: np.ndarray,
    target_indices: np.ndarray,
    confidence: np.ndarray,
    source_xy: np.ndarray,
    target_xy: np.ndarray,
    source_patch_count: int,
    target_patch_count: int,
    patch_width_um: float,
) -> PairEvidence:
    """Summarize shape, semantic, and correspondence change for one pair."""

    support_union = source.support | target.support
    support_intersection = source.support & target.support
    dice = (
        2.0 * float(np.count_nonzero(support_intersection))
        / float(np.count_nonzero(source.support) + np.count_nonzero(target.support))
        if np.any(support_union)
        else 1.0
    )
    source_hist = _label_distribution(source.labels, source.membership.shape[0])
    target_hist = _label_distribution(target.labels, target.membership.shape[0])
    js = _jensen_shannon(source_hist, target_hist)
    link_count = len(confidence)
    coverage = (
        float(link_count / min(source_patch_count, target_patch_count))
        if min(source_patch_count, target_patch_count)
        else 0.0
    )
    median_confidence = float(np.median(confidence)) if link_count else 0.0
    if link_count:
        source_sparse = _sparse_labels(source, source_indices)
        target_sparse = _sparse_labels(target, target_indices)
        agreement = float(np.mean(source_sparse == target_sparse))
        displacement = np.asarray(target_xy, dtype=float) - np.asarray(
            source_xy, dtype=float
        )
        displacement_norm = np.linalg.norm(displacement, axis=1) / patch_width_um
        median_displacement = float(np.median(displacement_norm))
        center = np.median(displacement, axis=0)
        strain = float(
            np.median(np.linalg.norm(displacement - center, axis=1)) / patch_width_um
        )
    else:
        agreement = 0.0
        median_displacement = 4.0
        strain = 4.0
    score = (
        0.28 * (1.0 - dice)
        + 0.18 * js
        + 0.20 * (1.0 - agreement)
        + 0.12 * min(median_displacement / 2.0, 2.0)
        + 0.12 * min(strain / 1.5, 2.0)
        + 0.10 * (1.0 - min(coverage / 0.5, 1.0))
    )
    return PairEvidence(
        source_section=0,
        target_section=1,
        score=max(float(score), np.finfo(float).eps),
        support_dice=dice,
        semantic_js=js,
        matched_label_agreement=agreement,
        correspondence_coverage=coverage,
        median_confidence=median_confidence,
        displacement_patch_widths=median_displacement,
        displacement_strain=strain,
    )


def infer_morphology_gap_decisions(
    sections: tuple[ObservedSection, ...],
    evidence: tuple[PairEvidence, ...],
    *,
    max_span: int,
    max_inferred_missing: int,
) -> tuple[GapDecision, ...]:
    """Classify each adjacent pair against other within-stack morphology spans."""

    maximum = min(max_span, max_inferred_missing + 1, len(sections) - 1)
    decisions: list[GapDecision] = []
    for index, item in enumerate(evidence):
        calibrator = fit_morphology_gap_calibrator(
            sections,
            max_span=maximum,
            excluded_pair=(index, index + 1),
        )
        predicted, confidence, _ = predict_morphology_interval(
            morphology_pair_features(sections[index], sections[index + 1]),
            calibrator,
            maximum_intervals=maximum,
        )
        reasons: list[str] = []
        supported = True
        if item.correspondence_coverage < 0.05:
            supported = False
            reasons.append("low_correspondence_coverage")
        if item.median_confidence < 0.45:
            supported = False
            reasons.append("low_correspondence_confidence")
        if confidence < 0.15:
            supported = False
            reasons.append("ambiguous_morphology_interval")
        if predicted == 1:
            status = "observed"
            intervals = 1
            reasons = []
        elif supported:
            status = "inferred"
            intervals = predicted
        else:
            status = "unresolved"
            intervals = 1
        decisions.append(
            GapDecision(
                source_section=index,
                target_section=index + 1,
                intervals=intervals,
                missing_sections=max(0, intervals - 1),
                status=status,
                confidence=confidence,
                score=item.score,
                reasons=tuple(reasons),
            )
        )
    return tuple(decisions)


def fit_morphology_gap_calibrator(
    sections: tuple[ObservedSection, ...],
    *,
    max_span: int,
    excluded_pair: tuple[int, int] | None = None,
) -> MorphologyGapCalibrator:
    """Fit robust interval centers from known index spans within one stack."""

    feature_rows: list[np.ndarray] = []
    interval_rows: list[int] = []
    for span in range(1, max_span + 1):
        for source_index in range(len(sections) - span):
            target_index = source_index + span
            if excluded_pair == (source_index, target_index):
                continue
            feature_rows.append(
                morphology_pair_features(
                    sections[source_index],
                    sections[target_index],
                )
            )
            interval_rows.append(span)
    if not feature_rows:
        raise ValueError("morphology gap calibration has no section pairs")
    features = np.stack(feature_rows)
    global_median = np.median(features, axis=0)
    mad = 1.4826 * np.median(np.abs(features - global_median), axis=0)
    spread = np.ptp(features, axis=0)
    scale = np.maximum(mad, np.maximum(0.05 * spread, 1e-4))
    centers = np.full((max_span, features.shape[1]), np.nan, dtype=np.float64)
    counts = np.zeros(max_span, dtype=np.int32)
    intervals = np.asarray(interval_rows)
    for span in range(1, max_span + 1):
        selected = features[intervals == span]
        counts[span - 1] = len(selected)
        if len(selected):
            centers[span - 1] = np.median(selected, axis=0)
    return MorphologyGapCalibrator(centers=centers, scale=scale, counts=counts)


def predict_morphology_interval(
    features: np.ndarray,
    calibrator: MorphologyGapCalibrator,
    *,
    maximum_intervals: int,
) -> tuple[int, float, tuple[float, ...]]:
    """Predict an interval and distance-margin confidence."""

    maximum = min(maximum_intervals, len(calibrator.centers))
    centers = calibrator.centers[:maximum]
    valid = np.all(np.isfinite(centers), axis=1) & (
        calibrator.counts[:maximum] >= 2
    )
    distances = np.full(maximum, np.inf, dtype=np.float64)
    distances[valid] = np.linalg.norm(
        (centers[valid] - np.asarray(features, dtype=float)) / calibrator.scale,
        axis=1,
    )
    if not np.any(np.isfinite(distances)):
        return 1, 0.0, tuple(float(value) for value in distances)
    order = np.argsort(distances)
    predicted = int(order[0]) + 1
    best = distances[order[0]]
    second = distances[order[1]] if len(order) > 1 else np.inf
    confidence = (
        float(np.clip((second - best) / max(second, 1e-8), 0, 1))
        if np.isfinite(second)
        else 1.0
    )
    return predicted, confidence, tuple(float(value) for value in distances)


def morphology_pair_features(
    source: ObservedSection,
    target: ObservedSection,
) -> np.ndarray:
    """Return registration-aware shape and semantic differences for two fields."""

    source_support = source.support
    target_support = target.support
    source_area = max(int(np.count_nonzero(source_support)), 1)
    target_area = max(int(np.count_nonzero(target_support)), 1)
    intersection = source_support & target_support
    support_dice = (
        2.0 * np.count_nonzero(intersection) / (source_area + target_area)
    )
    classes = source.membership.shape[0]
    semantic_js = _jensen_shannon(
        _label_distribution(source.labels, classes),
        _label_distribution(target.labels, classes),
    )
    agreement = (
        float(np.mean(source.labels[intersection] == target.labels[intersection]))
        if np.any(intersection)
        else 0.0
    )
    source_rc = np.argwhere(source_support)
    target_rc = np.argwhere(target_support)
    centroid_shift = np.linalg.norm(
        np.mean(source_rc, axis=0) - np.mean(target_rc, axis=0)
    ) / max(np.sqrt(0.5 * (source_area + target_area)), 1.0)
    source_extent = np.ptp(source_rc, axis=0) + 1
    target_extent = np.ptp(target_rc, axis=0) + 1
    extent_change = np.abs(np.log(target_extent / source_extent))
    return np.asarray(
        [
            1.0 - support_dice,
            semantic_js,
            1.0 - agreement,
            abs(np.log(target_area / source_area)),
            centroid_shift,
            extent_change[0],
            extent_change[1],
        ],
        dtype=np.float64,
    )


def _label_distribution(labels: np.ndarray, count: int) -> np.ndarray:
    selected = np.asarray(labels)[np.asarray(labels) >= 0]
    values = np.bincount(selected.astype(np.int64), minlength=count).astype(float)
    values += np.finfo(float).eps
    return values / values.sum()


def _jensen_shannon(left: np.ndarray, right: np.ndarray) -> float:
    midpoint = 0.5 * (left + right)
    divergence = 0.5 * (
        np.sum(left * np.log(left / midpoint))
        + np.sum(right * np.log(right / midpoint))
    )
    return float(np.sqrt(max(0.0, divergence)))


def _sparse_labels(section: ObservedSection, indices: np.ndarray) -> np.ndarray:
    return np.asarray(section.sparse_labels)[np.asarray(indices, dtype=np.int64)]
