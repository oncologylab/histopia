"""Held-out validation for semantic topology interpolation."""

from __future__ import annotations

from dataclasses import asdict, replace

import numpy as np

from histopia.topology._interpolate import (
    interpolate_pair,
    smooth_displacement_field,
)
from histopia.topology._model import (
    ObservedSection,
    PairEvidence,
    fit_morphology_gap_calibrator,
    morphology_pair_features,
    pair_evidence,
    predict_morphology_interval,
)


def run_holdout_benchmark(
    sections: tuple[ObservedSection, ...],
    adjacent_evidence: tuple[PairEvidence, ...],
    *,
    origin_um_xy: tuple[float, float],
    spacing_um: float,
    max_hidden_sections: int,
) -> dict[str, object]:
    """Hide bounded section spans and score endpoint-only reconstruction.

    Endpoint correspondences use only selected-K semantic labels and registered
    reference coordinates. Hidden labels are reserved for evaluation.
    """

    if len(sections) < 3:
        raise ValueError("topology holdout benchmarking requires three sections")
    if len(adjacent_evidence) != len(sections) - 1:
        raise ValueError("holdout benchmark requires every adjacent pair")
    maximum = min(max_hidden_sections, len(sections) - 2)
    maximum_intervals = min(max_hidden_sections + 1, len(sections) - 1)
    cases: list[dict[str, object]] = []
    for hidden_count in range(1, maximum + 1):
        for start in range(1, len(sections) - hidden_count):
            stop = start + hidden_count
            source = sections[start - 1]
            target = sections[stop]
            (
                source_xy,
                target_xy,
                confidence,
                source_link_labels,
                target_link_labels,
            ) = _semantic_endpoint_links(
                source,
                target,
                origin_um_xy=origin_um_xy,
                spacing_um=spacing_um,
            )
            field, field_confidence = smooth_displacement_field(
                source.support.shape,
                source_xy=source_xy,
                target_xy=target_xy,
                origin_um_xy=origin_um_xy,
                spacing_um=spacing_um,
            )
            zero_field = np.zeros_like(field)
            perfect_confidence = np.ones_like(field_confidence)
            flow_metrics: list[dict[str, float]] = []
            zero_metrics: list[dict[str, float]] = []
            nearest_metrics: list[dict[str, float]] = []
            for offset, truth in enumerate(sections[start:stop], start=1):
                fraction = offset / (hidden_count + 1)
                common = {
                    "fraction": fraction,
                    "z_um": float(offset),
                    "segment": 0,
                    "source_section": start - 1,
                    "target_section": stop,
                }
                flow_plane = interpolate_pair(
                    source,
                    target,
                    displacement_rc=field,
                    flow_confidence=field_confidence,
                    **common,
                )
                zero_plane = interpolate_pair(
                    source,
                    target,
                    displacement_rc=zero_field,
                    flow_confidence=perfect_confidence,
                    **common,
                )
                nearest = source if fraction <= 0.5 else target
                flow_metrics.append(_field_metrics(flow_plane.labels, truth.labels))
                zero_metrics.append(_field_metrics(zero_plane.labels, truth.labels))
                nearest_metrics.append(_field_metrics(nearest.labels, truth.labels))

            endpoint_evidence = _endpoint_evidence(
                source,
                target,
                source_xy=source_xy,
                target_xy=target_xy,
                confidence=confidence,
                source_labels=source_link_labels,
                target_labels=target_link_labels,
                spacing_um=spacing_um,
            )
            expected_intervals = hidden_count + 1
            calibrator = fit_morphology_gap_calibrator(
                sections,
                max_span=maximum_intervals,
                excluded_pair=(start - 1, stop),
            )
            (
                predicted_intervals,
                interval_confidence,
                interval_distances,
            ) = predict_morphology_interval(
                morphology_pair_features(source, target),
                calibrator,
                maximum_intervals=maximum_intervals,
            )
            cases.append(
                {
                    "source_section": start - 1,
                    "target_section": stop,
                    "hidden_sections": hidden_count,
                    "expected_intervals": expected_intervals,
                    "predicted_intervals": predicted_intervals,
                    "interval_confidence": interval_confidence,
                    "interval_distances": interval_distances,
                    "interval_error": abs(predicted_intervals - expected_intervals),
                    "endpoint_link_count": len(confidence),
                    "endpoint_evidence": asdict(endpoint_evidence),
                    "flow": _mean_metrics(flow_metrics),
                    "zero_flow": _mean_metrics(zero_metrics),
                    "nearest_section": _mean_metrics(nearest_metrics),
                }
            )
    summary = _summarize_cases(cases)
    return {
        "schema_version": 1,
        "method": "endpoint_only_selected_k_holdout",
        "adjacent_pair_count": len(adjacent_evidence),
        "hidden_section_limit": maximum,
        "gap_calibration": "leave_one_pair_out_robust_morphology_centers",
        "cases": cases,
        "summary": summary,
    }


def _semantic_endpoint_links(
    source: ObservedSection,
    target: ObservedSection,
    *,
    origin_um_xy: tuple[float, float],
    spacing_um: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from scipy.spatial import cKDTree

    source_points: list[np.ndarray] = []
    target_points: list[np.ndarray] = []
    distances: list[np.ndarray] = []
    source_labels: list[np.ndarray] = []
    target_labels: list[np.ndarray] = []
    classes = source.membership.shape[0]
    for class_index in range(classes):
        source_rc = np.argwhere(source.labels == class_index)
        target_rc = np.argwhere(target.labels == class_index)
        if not len(source_rc) or not len(target_rc):
            continue
        forward_distance, forward_index = cKDTree(target_rc).query(source_rc)
        _, reverse_index = cKDTree(source_rc).query(target_rc)
        source_index = np.arange(len(source_rc))
        mutual = reverse_index[forward_index] == source_index
        close = forward_distance <= 4.0
        accepted = mutual & close
        if not np.any(accepted):
            continue
        source_points.append(source_rc[accepted])
        target_points.append(target_rc[forward_index[accepted]])
        distances.append(forward_distance[accepted])
        source_labels.append(
            np.full(np.count_nonzero(accepted), class_index, dtype=np.int16)
        )
        target_labels.append(
            np.full(np.count_nonzero(accepted), class_index, dtype=np.int16)
        )
    if not source_points:
        empty = np.empty((0, 2), dtype=np.float64)
        labels = np.empty(0, dtype=np.int16)
        return empty, empty.copy(), np.empty(0, dtype=np.float32), labels, labels.copy()
    source_rc = np.concatenate(source_points)
    target_rc = np.concatenate(target_points)
    distance = np.concatenate(distances)
    source_xy = _rc_to_xy(
        source_rc,
        origin_um_xy=origin_um_xy,
        spacing_um=spacing_um,
    )
    target_xy = _rc_to_xy(
        target_rc,
        origin_um_xy=origin_um_xy,
        spacing_um=spacing_um,
    )
    confidence = np.exp(-distance / 2.0).astype(np.float32)
    return (
        source_xy,
        target_xy,
        confidence,
        np.concatenate(source_labels),
        np.concatenate(target_labels),
    )


def _endpoint_evidence(
    source: ObservedSection,
    target: ObservedSection,
    *,
    source_xy: np.ndarray,
    target_xy: np.ndarray,
    confidence: np.ndarray,
    source_labels: np.ndarray,
    target_labels: np.ndarray,
    spacing_um: float,
) -> PairEvidence:
    source_for_evidence = replace(source, sparse_labels=source_labels)
    target_for_evidence = replace(target, sparse_labels=target_labels)
    link_indices = np.arange(len(confidence), dtype=np.int64)
    return pair_evidence(
        source_for_evidence,
        target_for_evidence,
        source_indices=link_indices,
        target_indices=link_indices,
        confidence=confidence,
        source_xy=source_xy,
        target_xy=target_xy,
        source_patch_count=int(np.count_nonzero(source.support)),
        target_patch_count=int(np.count_nonzero(target.support)),
        patch_width_um=spacing_um,
    )


def _rc_to_xy(
    coordinates_rc: np.ndarray,
    *,
    origin_um_xy: tuple[float, float],
    spacing_um: float,
) -> np.ndarray:
    return np.column_stack(
        (
            origin_um_xy[0] + coordinates_rc[:, 1] * spacing_um,
            origin_um_xy[1] + coordinates_rc[:, 0] * spacing_um,
        )
    )


def _field_metrics(predicted: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    predicted_support = predicted >= 0
    truth_support = truth >= 0
    classes = sorted(
        set(np.unique(predicted[predicted_support]).tolist())
        | set(np.unique(truth[truth_support]).tolist())
    )
    class_dice = [
        _dice(predicted == class_index, truth == class_index) for class_index in classes
    ]
    return {
        "tissue_dice": _dice(predicted_support, truth_support),
        "macro_class_dice": float(np.mean(class_dice)) if class_dice else 1.0,
        "boundary_f1": _boundary_f1(predicted_support, truth_support),
    }


def _dice(left: np.ndarray, right: np.ndarray) -> float:
    denominator = np.count_nonzero(left) + np.count_nonzero(right)
    return (
        2.0 * float(np.count_nonzero(left & right)) / denominator
        if denominator
        else 1.0
    )


def _boundary_f1(left: np.ndarray, right: np.ndarray) -> float:
    from scipy.ndimage import binary_dilation, binary_erosion

    left_boundary = left ^ binary_erosion(left)
    right_boundary = right ^ binary_erosion(right)
    if not np.any(left_boundary) and not np.any(right_boundary):
        return 1.0
    precision = (
        float(np.mean(binary_dilation(right_boundary)[left_boundary]))
        if np.any(left_boundary)
        else 0.0
    )
    recall = (
        float(np.mean(binary_dilation(left_boundary)[right_boundary]))
        if np.any(right_boundary)
        else 0.0
    )
    return (
        2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    )


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    return {
        name: float(np.mean([row[name] for row in rows]))
        for name in ("tissue_dice", "macro_class_dice", "boundary_f1")
    }


def _summarize_cases(cases: list[dict[str, object]]) -> dict[str, object]:
    flow = [
        float(case["flow"]["macro_class_dice"])  # type: ignore[index]
        for case in cases
    ]
    zero = [
        float(case["zero_flow"]["macro_class_dice"])  # type: ignore[index]
        for case in cases
    ]
    nearest = [
        float(case["nearest_section"]["macro_class_dice"])  # type: ignore[index]
        for case in cases
    ]
    gap_accuracy = float(
        np.mean(
            [
                int(case["predicted_intervals"]) == int(case["expected_intervals"])
                for case in cases
            ]
        )
    )
    return {
        "case_count": len(cases),
        "flow_macro_class_dice": float(np.mean(flow)),
        "zero_flow_macro_class_dice": float(np.mean(zero)),
        "nearest_macro_class_dice": float(np.mean(nearest)),
        "flow_gain_over_zero": float(np.mean(np.asarray(flow) - np.asarray(zero))),
        "flow_gain_over_nearest": float(
            np.mean(np.asarray(flow) - np.asarray(nearest))
        ),
        "gap_interval_accuracy": gap_accuracy,
        "supports_gap_inference": gap_accuracy >= 0.60,
        "supports_flow_interpolation": float(np.mean(flow)) >= float(np.mean(zero)),
    }
