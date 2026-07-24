"""Brightfield/IHC tissue mask generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypeVar

import numpy as np
from scipy import ndimage as ndi

from histopia.registration._config import BrightfieldMaskConfig

MaskKey = TypeVar("MaskKey")


@dataclass(slots=True)
class TissueMaskResult:
    """A tissue mask and its QC metadata."""

    mask: np.ndarray
    method: str
    metrics: dict[str, float]
    accepted: bool
    warnings: list[str]
    candidate_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    candidate_warnings: dict[str, list[str]] = field(default_factory=dict)
    candidate_masks: dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "metrics": self.metrics,
            "accepted": self.accepted,
            "warnings": self.warnings,
            "candidate_metrics": self.candidate_metrics,
            "candidate_warnings": self.candidate_warnings,
        }


def create_tissue_mask(
    image: np.ndarray,
    config: BrightfieldMaskConfig | None = None,
) -> TissueMaskResult:
    """Create a QC-scored tissue mask for a brightfield RGB thumbnail."""

    config = config or BrightfieldMaskConfig()
    rgb = _as_rgb_float(image)
    height, width = rgb.shape[:2]

    if config.mode == "full":
        mask = np.ones((height, width), dtype=bool)
        return TissueMaskResult(
            mask=mask,
            method="full",
            metrics=_mask_metrics(mask),
            accepted=True,
            warnings=["full mask requested explicitly"],
        )

    raw_candidates = {
        "hysteresis_tissue": _hysteresis_tissue_candidate(rgb),
        "background_corrected": _background_corrected_candidate(rgb),
        "edge_texture": _edge_texture_candidate(rgb),
        "optical_density": _od_candidate(rgb),
        "saturation_value": _saturation_value_candidate(rgb),
        "adaptive_brightness": _adaptive_brightness_candidate(rgb),
    }
    evidence = np.sum(np.stack(tuple(raw_candidates.values())), axis=0, dtype=np.uint8)
    optical_density = np.mean(-np.log(np.clip(rgb, 1 / 255, 1.0)), axis=2)
    candidates = {
        method: _retain_tissue_objects(
            _clean_mask(mask, config), evidence, optical_density, config
        )
        for method, mask in raw_candidates.items()
    }
    candidates["group_density_union"] = _group_density_union_candidate(
        raw_candidates["background_corrected"],
        raw_candidates["optical_density"],
        config,
    )
    candidates["group_pale_tissue"] = _pale_tissue_candidate(
        rgb,
        evidence,
        optical_density,
        config,
    )
    candidates["group_pale_tissue"] |= candidates["group_density_union"]
    candidates["object_aware_fusion"] = _object_aware_fusion(
        rgb, raw_candidates, config
    )
    candidates = {
        method: _carve_large_blank_regions(rgb, mask)
        for method, mask in candidates.items()
    }
    candidate_metrics = {
        method: _mask_metrics(mask) for method, mask in candidates.items()
    }
    candidate_warnings = {
        method: _mask_warnings(mask, config) for method, mask in candidates.items()
    }

    fusion = candidates["object_aware_fusion"]
    fusion_warnings = candidate_warnings["object_aware_fusion"]
    candidate_scores = {
        method: _mask_score(mask, config)
        for method, mask in candidates.items()
        if method not in {"group_density_union", "group_pale_tissue"}
    }
    best_method = max(candidate_scores, key=candidate_scores.__getitem__)
    best_metrics = candidate_metrics[best_method]
    fusion_metrics = candidate_metrics["object_aware_fusion"]
    fusion_under_covers = (
        best_method in {"background_corrected", "optical_density"}
        and not candidate_warnings[best_method]
        and candidate_scores[best_method]
        > candidate_scores["object_aware_fusion"] + 0.15
        and fusion_metrics["foreground_fraction"]
        < best_metrics["foreground_fraction"] * 0.70
    )
    if not fusion_warnings and not fusion_under_covers:
        return TissueMaskResult(
            fusion,
            "object_aware_fusion",
            candidate_metrics["object_aware_fusion"],
            True,
            [],
            candidate_metrics,
            candidate_warnings,
            candidates,
        )

    scored = [
        (
            candidate_scores[method],
            method,
            mask,
            candidate_metrics[method],
            candidate_warnings[method],
        )
        for method, mask in candidates.items()
        if method in candidate_scores
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    score, method, mask, metrics, warnings = scored[0]
    accepted = score > 0 and not warnings

    if accepted:
        return TissueMaskResult(
            mask,
            method,
            metrics,
            True,
            [],
            candidate_metrics,
            candidate_warnings,
            candidates,
        )

    if config.allow_full_fallback:
        full = np.ones((height, width), dtype=bool)
        fallback_metrics = _mask_metrics(full)
        return TissueMaskResult(
            mask=full,
            method="full_fallback",
            metrics=fallback_metrics,
            accepted=True,
            warnings=[
                "all auto_tissue candidates failed QC",
                f"best_candidate={method}",
                *warnings,
            ],
            candidate_metrics=candidate_metrics,
            candidate_warnings=candidate_warnings,
            candidate_masks=candidates,
        )

    return TissueMaskResult(
        mask,
        method,
        metrics,
        False,
        warnings,
        candidate_metrics,
        candidate_warnings,
        candidates,
    )


def refine_group_tissue_masks(
    results: dict[MaskKey, TissueMaskResult],
    *,
    physical_pixel_areas: dict[MaskKey, float | None] | None = None,
    normalized_shape: tuple[int, int] = (256, 256),
    min_group_support: float = 0.12,
) -> dict[MaskKey, TissueMaskResult]:
    """Remove slide-specific objects using the cohort's recurring topology.

    Slide canvases are normalized without registration. A component is retained
    when it is the dominant object on its slide or overlaps tissue proposals in
    other sections. This makes masking section-group aware while allowing
    gradual changes in tissue shape and size.
    """

    if len(results) < 3:
        return dict(results)
    keys = list(results)
    proposal_masks = {key: _candidate_union(results[key]) for key in keys}
    proposal_normalized = {
        key: _resize_binary(proposal_masks[key], normalized_shape) for key in keys
    }
    fragment_proposals = {
        key: results[key].candidate_masks.get(
            "group_pale_tissue",
            results[key].mask,
        )
        for key in keys
    }
    fragment_normalized = {
        key: _resize_binary(fragment_proposals[key], normalized_shape) for key in keys
    }
    expected_physical_area = _expected_group_physical_area(
        results, physical_pixel_areas
    )
    selected: dict[MaskKey, TissueMaskResult] = {}
    for key in keys:
        result = results[key]
        peer_support = np.mean(
            np.stack(
                [
                    ndi.binary_dilation(proposal_normalized[peer], iterations=24)
                    for peer in keys
                    if peer != key
                ]
            ),
            axis=0,
        )
        key_index = keys.index(key)
        adjacent_keys = (
            keys[max(0, key_index - 1) : key_index]
            + keys[key_index + 1 : key_index + 2]
        )
        fragment_support = np.mean(
            np.stack(
                [
                    ndi.binary_dilation(fragment_normalized[peer], iterations=5)
                    for peer in adjacent_keys
                ]
            ),
            axis=0,
        )
        ranked = _select_group_supported_candidate(
            result,
            peer_support,
            normalized_shape,
            physical_pixel_area=(physical_pixel_areas or {}).get(key),
            expected_physical_area=expected_physical_area,
        )
        recovery_candidate = np.logical_or(
            result.candidate_masks.get("group_pale_tissue", result.mask),
            proposal_masks[key],
        )
        recovered = _augment_with_group_components(
            recovery_candidate,
            ranked.mask,
            fragment_support,
            normalized_shape,
            small_fragment_support=0.45,
            small_fragments_only=True,
        )
        if not np.array_equal(recovered, ranked.mask):
            recovered_metrics = _mask_metrics(recovered)
            recovered_warnings = _mask_warnings(
                recovered,
                BrightfieldMaskConfig(),
            )
            if not recovered_warnings:
                ranked = TissueMaskResult(
                    mask=recovered,
                    method=f"{ranked.method}+group_fragment_recovery",
                    metrics=recovered_metrics,
                    accepted=True,
                    warnings=[],
                    candidate_metrics=ranked.candidate_metrics,
                    candidate_warnings=ranked.candidate_warnings,
                    candidate_masks=ranked.candidate_masks,
                )
        ranked = _polish_selected_mask(ranked)
        selected[key] = ranked
    normalized = {
        key: _resize_binary(selected[key].mask, normalized_shape) for key in keys
    }
    refined: dict[MaskKey, TissueMaskResult] = {}
    for key in keys:
        result = selected[key]
        labels, count = ndi.label(result.mask)
        if count <= 1:
            refined[key] = result
            continue
        peer_support = np.mean(
            np.stack(
                [
                    ndi.binary_dilation(normalized[peer], iterations=24)
                    for peer in keys
                    if peer != key
                ]
            ),
            axis=0,
        )
        direct_peer_support = np.mean(
            np.stack(
                [
                    ndi.binary_dilation(normalized[peer], iterations=3)
                    for peer in keys
                    if peer != key
                ]
            ),
            axis=0,
        )
        key_index = keys.index(key)
        adjacent_keys = (
            keys[max(0, key_index - 1) : key_index]
            + keys[key_index + 1 : key_index + 2]
        )
        adjacent_support = np.mean(
            np.stack(
                [
                    ndi.binary_dilation(normalized[peer], iterations=5)
                    for peer in adjacent_keys
                ]
            ),
            axis=0,
        )
        keep = np.zeros(count + 1, dtype=bool)
        component_sizes = np.bincount(labels.ravel())
        largest_component = int(component_sizes[1:].max(initial=0))
        main_labels = component_sizes >= largest_component * 0.50
        main_labels[0] = False
        distance_to_main = ndi.distance_transform_edt(~main_labels[labels])
        maximum_fragment_gap = 0.08 * float(np.hypot(*result.mask.shape))
        component_support: dict[int, float] = {}
        for label in range(1, count + 1):
            relative_area = component_sizes[label] / max(largest_component, 1)
            if relative_area < 0.015:
                continue
            component = labels == label
            rows, cols = np.nonzero(component)
            row_span = rows.max() - rows.min() + 1
            col_span = cols.max() - cols.min() + 1
            long_axis = max(
                row_span / component.shape[0], col_span / component.shape[1]
            )
            short_axis = min(
                row_span / component.shape[0], col_span / component.shape[1]
            )
            bbox_fill = component_sizes[label] / max(row_span * col_span, 1)
            max_row_occupancy = (
                np.bincount(rows - rows.min(), minlength=row_span).max(initial=0)
                / col_span
            )
            max_col_occupancy = (
                np.bincount(cols - cols.min(), minlength=col_span).max(initial=0)
                / row_span
            )
            if long_axis > 0.18 and short_axis < 0.08:
                continue
            if long_axis > 0.45 and short_axis < 0.15 and bbox_fill < 0.40:
                continue
            if (
                long_axis > 0.40
                and bbox_fill < 0.25
                and max_row_occupancy > 0.60
                and max_col_occupancy > 0.60
            ):
                continue
            normalized_component = _resize_binary(component, normalized_shape)
            if not normalized_component.any():
                continue
            support = float(np.mean(peer_support[normalized_component]))
            direct_support = float(np.mean(direct_peer_support[normalized_component]))
            neighbor_support = float(np.mean(adjacent_support[normalized_component]))
            component_support[label] = support
            close_supported_fragment = (
                relative_area < 0.10
                and float(np.min(distance_to_main[component])) <= maximum_fragment_gap
                and (
                    neighbor_support >= 0.45
                    or direct_support >= max(0.02, 0.50 / (len(keys) - 1))
                )
            )
            close_native_fragment = (
                0.02 <= relative_area < 0.10
                and float(np.min(distance_to_main[component]))
                <= 0.02 * float(np.hypot(*result.mask.shape))
                and _component_center_fill_ratio(component) >= 0.45
            )
            recurring_substantial_component = (
                support >= min_group_support
                and relative_area >= 0.10
                and (relative_area >= 0.50 or direct_support >= 0.50)
            )
            if (
                recurring_substantial_component
                or close_supported_fragment
                or close_native_fragment
            ):
                keep[label] = True
        if not np.any(keep):
            if not component_support:
                refined[key] = result
                continue
            support_order = sorted(
                component_support,
                key=lambda label: (
                    component_support[label],
                    component_sizes[label],
                ),
                reverse=True,
            )
            best_supported_label = support_order[0]
            best_support = component_support[best_supported_label]
            second_support = (
                component_support[support_order[1]] if len(support_order) > 1 else 0.0
            )
            if best_support < 0.02 or best_support - second_support < 0.05:
                refined[key] = result
                continue
            keep[best_supported_label] = True
        mask = keep[labels]
        if np.count_nonzero(mask) < np.count_nonzero(result.mask) * 0.25:
            refined[key] = result
            continue
        if np.array_equal(mask, result.mask):
            refined[key] = result
            continue
        metrics = _mask_metrics(mask)
        warnings = _mask_warnings(mask, BrightfieldMaskConfig())
        refined[key] = TissueMaskResult(
            mask=mask,
            method=f"{result.method}+group_consensus",
            metrics=metrics,
            accepted=not warnings,
            warnings=warnings,
            candidate_metrics=result.candidate_metrics,
            candidate_warnings=result.candidate_warnings,
            candidate_masks=result.candidate_masks,
        )
    expected_fraction = float(
        np.median([np.mean(mask) for mask in normalized.values()])
    )
    final_physical_areas = [
        float(np.count_nonzero(refined[key].mask) * pixel_area)
        for key, pixel_area in (physical_pixel_areas or {}).items()
        if key in refined and pixel_area is not None and pixel_area > 0
    ]
    final_expected_physical_area = (
        float(np.median(final_physical_areas)) if final_physical_areas else None
    )
    for key, result in refined.items():
        fraction = float(np.mean(_resize_binary(result.mask, normalized_shape)))
        result.metrics["group_expected_foreground_fraction"] = expected_fraction
        result.metrics["group_foreground_fraction_ratio"] = (
            fraction / expected_fraction if expected_fraction else 0.0
        )
        pixel_area = (physical_pixel_areas or {}).get(key)
        if final_expected_physical_area is not None and pixel_area is not None:
            physical_area = float(np.count_nonzero(result.mask) * pixel_area)
            result.metrics["physical_foreground_area_um2"] = physical_area
            result.metrics["group_physical_area_ratio"] = (
                physical_area / final_expected_physical_area
            )
    return refined


def _expected_group_physical_area(
    results: dict[MaskKey, TissueMaskResult],
    physical_pixel_areas: dict[MaskKey, float | None] | None,
) -> float | None:
    if not physical_pixel_areas:
        return None
    areas = [
        float(np.count_nonzero(result.mask) * pixel_area)
        for key, result in results.items()
        if (pixel_area := physical_pixel_areas.get(key)) is not None and pixel_area > 0
    ]
    return float(np.median(areas)) if areas else None


def _candidate_union(result: TissueMaskResult) -> np.ndarray:
    accepted = [
        mask
        for method, mask in result.candidate_masks.items()
        if method
        in {
            "object_aware_fusion",
            "background_corrected",
            "optical_density",
            "adaptive_brightness",
        }
        and not result.candidate_warnings.get(method, [])
        and mask.any()
    ]
    if not accepted:
        return result.mask
    return np.logical_or.reduce(accepted)


def _select_group_supported_candidate(
    result: TissueMaskResult,
    peer_support: np.ndarray,
    normalized_shape: tuple[int, int],
    *,
    physical_pixel_area: float | None,
    expected_physical_area: float | None,
) -> TissueMaskResult:
    allowed_methods = {
        "object_aware_fusion",
        "background_corrected",
        "optical_density",
        "adaptive_brightness",
        "group_density_union",
        "group_pale_tissue",
    }
    candidates: list[tuple[str, np.ndarray]] = []
    for method, candidate in result.candidate_masks.items():
        if method not in allowed_methods or not candidate.any():
            continue
        if method in {"group_density_union", "group_pale_tissue"}:
            candidate = _augment_with_group_components(
                candidate,
                result.mask,
                peer_support,
                normalized_shape,
            )
            if not candidate.any() or _mask_warnings(
                candidate, BrightfieldMaskConfig()
            ):
                continue
        elif result.candidate_warnings.get(method, []):
            continue
        candidates.append((method, candidate))
    if not candidates:
        return result
    baseline_fraction = max(
        (
            float(np.mean(mask))
            for method, mask in candidates
            if method not in {"group_density_union", "group_pale_tissue"}
        ),
        default=0.0,
    )
    candidates = [
        (method, mask)
        for method, mask in candidates
        if method not in {"group_density_union", "group_pale_tissue"}
        or float(np.mean(mask)) >= baseline_fraction * 0.60
    ]
    if (
        physical_pixel_area is not None
        and expected_physical_area is not None
        and baseline_fraction * result.mask.size * physical_pixel_area
        >= expected_physical_area * 0.90
    ):
        candidates = [
            (method, mask)
            for method, mask in candidates
            if method not in {"group_density_union", "group_pale_tissue"}
        ]
    candidate_fractions = np.array([np.mean(mask) for _, mask in candidates])
    typical_fraction = float(np.median(candidate_fractions[candidate_fractions > 0]))
    support_total = float(np.sum(peer_support))
    scored: list[tuple[float, str, np.ndarray]] = []
    for method, mask in candidates:
        normalized = _resize_binary(mask, normalized_shape)
        covered = float(np.sum(peer_support[normalized])) / max(support_total, 1.0)
        unsupported = float(np.mean(peer_support[normalized] < 0.05))
        score = _mask_score(mask, BrightfieldMaskConfig())
        score += 2.0 * covered - 1.5 * unsupported
        if method == "adaptive_brightness":
            score -= 0.30
        relative_fraction = float(np.mean(mask)) / max(typical_fraction, 1e-6)
        if (
            method not in {"group_density_union", "group_pale_tissue"}
            and relative_fraction > 1.5
        ):
            score -= 8.0 * float(np.log(relative_fraction / 1.5))
        if (
            physical_pixel_area is not None
            and expected_physical_area is not None
            and expected_physical_area > 0
        ):
            area = float(np.count_nonzero(mask) * physical_pixel_area)
            ratio = max(area / expected_physical_area, 1e-6)
            score -= 0.35 * abs(float(np.log(ratio)))
        scored.append((score, method, mask))
    _, method, mask = max(scored, key=lambda item: item[0])
    if method == result.method:
        return result
    metrics = _mask_metrics(mask)
    warnings = _mask_warnings(mask, BrightfieldMaskConfig())
    return TissueMaskResult(
        mask=mask,
        method=f"{method}+group_ranked",
        metrics=metrics,
        accepted=not warnings,
        warnings=warnings,
        candidate_metrics=result.candidate_metrics,
        candidate_warnings=result.candidate_warnings,
        candidate_masks=result.candidate_masks,
    )


def _polish_selected_mask(result: TissueMaskResult) -> TissueMaskResult:
    if not result.candidate_masks:
        return result
    cleaned = _remove_straight_border_rails(
        _remove_hollow_detached_artifacts(result.mask)
    )
    if result.method.startswith(("group_density_union", "group_pale_tissue")):
        if np.array_equal(cleaned, result.mask):
            return result
        metrics = _mask_metrics(cleaned)
        warnings = _mask_warnings(cleaned, BrightfieldMaskConfig())
        return TissueMaskResult(
            mask=cleaned,
            method=f"{result.method}+satellite_cleanup",
            metrics=metrics,
            accepted=not warnings,
            warnings=warnings,
            candidate_metrics=result.candidate_metrics,
            candidate_warnings=result.candidate_warnings,
            candidate_masks=result.candidate_masks,
        )
    polished = ndi.binary_dilation(cleaned, iterations=2)
    polished = _fill_small_holes(
        polished,
        max_area=max(64, int(polished.size * 0.004)),
    )
    polished = _remove_straight_border_rails(
        _remove_hollow_detached_artifacts(polished)
    )
    if np.array_equal(polished, result.mask):
        return result
    metrics = _mask_metrics(polished)
    warnings = _mask_warnings(polished, BrightfieldMaskConfig())
    return TissueMaskResult(
        mask=polished,
        method=f"{result.method}+polished",
        metrics=metrics,
        accepted=not warnings,
        warnings=warnings,
        candidate_metrics=result.candidate_metrics,
        candidate_warnings=result.candidate_warnings,
        candidate_masks=result.candidate_masks,
    )


def _remove_hollow_detached_artifacts(mask: np.ndarray) -> np.ndarray:
    """Remove detached ring-like objects while retaining solid tissue lobules."""

    labels, count = ndi.label(mask)
    if count < 2:
        return mask
    sizes = np.bincount(labels.ravel())
    largest = int(sizes[1:].max(initial=0))
    keep = np.ones(count + 1, dtype=bool)
    keep[0] = False
    for label in range(1, count + 1):
        area = int(sizes[label])
        ratio = area / max(largest, 1)
        if not 0.015 <= ratio <= 0.15:
            continue
        component = labels == label
        rows, cols = np.nonzero(component)
        height = rows.max() - rows.min() + 1
        width = cols.max() - cols.min() + 1
        aspect = width / height
        if 0.55 <= aspect <= 1.80 and _component_center_fill_ratio(component) < 0.35:
            keep[label] = False
    return keep[labels]


def _component_center_fill_ratio(component: np.ndarray) -> float:
    rows, cols = np.nonzero(component)
    if not rows.size:
        return 0.0
    top, bottom = int(rows.min()), int(rows.max()) + 1
    left, right = int(cols.min()), int(cols.max()) + 1
    height = bottom - top
    width = right - left
    center_top = top + int(round(height * 0.30))
    center_bottom = top + int(round(height * 0.70))
    center_left = left + int(round(width * 0.30))
    center_right = left + int(round(width * 0.70))
    center = component[center_top:center_bottom, center_left:center_right]
    outer_area = height * width - center.size
    outer_count = int(np.count_nonzero(component[top:bottom, left:right])) - int(
        np.count_nonzero(center)
    )
    center_fill = float(np.mean(center)) if center.size else 0.0
    outer_fill = outer_count / max(outer_area, 1)
    return center_fill / max(outer_fill, 1e-6)


def _augment_with_group_components(
    candidate: np.ndarray,
    trusted: np.ndarray,
    peer_support: np.ndarray,
    normalized_shape: tuple[int, int],
    *,
    minimum_support: float = 0.12,
    small_fragment_support: float = 0.30,
    small_fragments_only: bool = False,
) -> np.ndarray:
    labels, count = ndi.label(candidate)
    if count == 0:
        return trusted
    augmented = trusted.copy()
    trusted_neighborhood = ndi.binary_dilation(trusted, iterations=12)
    trusted_labels, trusted_count = ndi.label(trusted)
    trusted_sizes = np.bincount(trusted_labels.ravel())
    largest_trusted = int(np.max(trusted_sizes[1:])) if trusted_count else 64
    minimum_component_area = max(64, int(largest_trusted * 0.015))
    distance_to_trusted = ndi.distance_transform_edt(~trusted)
    maximum_fragment_gap = (0.15 if small_fragments_only else 0.08) * float(
        np.hypot(*candidate.shape)
    )
    strongly_supported_native = _resize_binary(
        peer_support >= small_fragment_support,
        candidate.shape,
    )
    for label in range(1, count + 1):
        component = labels == label
        if np.any(component & trusted):
            augmented |= component & trusted_neighborhood
            augmented |= component & strongly_supported_native
            continue
        if np.count_nonzero(component) < minimum_component_area:
            continue
        rows, cols = np.nonzero(component)
        row_span = rows.max() - rows.min() + 1
        col_span = cols.max() - cols.min() + 1
        long_axis = max(row_span / candidate.shape[0], col_span / candidate.shape[1])
        short_axis = min(row_span / candidate.shape[0], col_span / candidate.shape[1])
        bbox_fill = np.count_nonzero(component) / max(row_span * col_span, 1)
        near_border = (
            rows.min() < candidate.shape[0] * 0.03
            or rows.max() >= candidate.shape[0] * 0.97
            or cols.min() < candidate.shape[1] * 0.03
            or cols.max() >= candidate.shape[1] * 0.97
        )
        if long_axis > 0.20 and short_axis < 0.10:
            continue
        if long_axis > 0.45 and short_axis < 0.15 and bbox_fill < 0.40:
            continue
        if near_border and long_axis > 0.25 and short_axis < 0.15:
            continue
        component_area = np.count_nonzero(component)
        fragment_ratio_limit = 0.20 if small_fragments_only else 0.10
        if (
            small_fragments_only
            and component_area >= largest_trusted * fragment_ratio_limit
        ):
            continue
        if component_area >= largest_trusted * 0.30:
            augmented |= component
            continue
        normalized = _resize_binary(component, normalized_shape)
        support = float(np.mean(peer_support[normalized])) if normalized.any() else 0.0
        is_small_fragment = component_area < largest_trusted * fragment_ratio_limit
        close_to_trusted = (
            float(np.min(distance_to_trusted[component])) <= maximum_fragment_gap
        )
        native_continuation = (
            is_small_fragment
            and component_area >= largest_trusted * 0.02
            and float(np.min(distance_to_trusted[component]))
            <= 0.02 * float(np.hypot(*candidate.shape))
            and _component_center_fill_ratio(component) >= 0.45
        )
        required_support = (
            small_fragment_support if is_small_fragment else minimum_support
        )
        if (
            support >= required_support and (not is_small_fragment or close_to_trusted)
        ) or native_continuation:
            augmented |= component
    return augmented


def _group_density_union_candidate(
    background: np.ndarray,
    optical_density: np.ndarray,
    config: BrightfieldMaskConfig,
) -> np.ndarray:
    candidate = np.logical_or(background, optical_density)
    candidate = _remove_scanner_edges(candidate)
    candidate = clean_external_tissue_mask(_clean_mask(candidate, config), config)
    candidate = ndi.binary_dilation(candidate, iterations=2)
    return _fill_small_holes(candidate, max_area=max(64, int(candidate.size * 0.004)))


def _remove_scanner_edges(candidate: np.ndarray) -> np.ndarray:
    """Disconnect scanner-bed borders and dense rails from tissue proposals."""

    candidate = _remove_straight_border_rails(candidate)
    height, width = candidate.shape
    edge_rows = max(2, int(round(height * 0.005)))
    edge_cols = max(2, int(round(width * 0.005)))
    candidate[:edge_rows] = False
    candidate[-edge_rows:] = False
    candidate[:, :edge_cols] = False
    candidate[:, -edge_cols:] = False
    row_positions = np.arange(height)
    col_positions = np.arange(width)
    dense_rows = (np.mean(candidate, axis=1) > 0.65) & (
        (row_positions < height * 0.12) | (row_positions >= height * 0.88)
    )
    dense_cols = (np.mean(candidate, axis=0) > 0.65) & (
        (col_positions < width * 0.06) | (col_positions >= width * 0.94)
    )
    if np.any(dense_rows):
        dense_rows = ndi.binary_dilation(dense_rows, iterations=edge_rows)
        candidate[dense_rows] = False
    if np.any(dense_cols):
        dense_cols = ndi.binary_dilation(dense_cols, iterations=edge_cols)
        candidate[:, dense_cols] = False
    return candidate


def _remove_straight_border_rails(candidate: np.ndarray) -> np.ndarray:
    """Remove long axis-aligned runs near the slide boundary."""

    candidate = candidate.copy()
    height, width = candidate.shape
    edge_rows = max(2, int(round(height * 0.005)))
    edge_cols = max(2, int(round(width * 0.005)))
    distance = ndi.distance_transform_edt(candidate)
    thickness = max(
        edge_rows,
        edge_cols,
        int(round(min(height, width) * 0.035)),
    )
    thin = distance <= thickness
    tissue_core = distance > thickness
    protected_perimeter = (
        ndi.distance_transform_cdt(~tissue_core, metric="taxicab") <= thickness * 2
        if tissue_core.any()
        else np.zeros_like(candidate)
    )
    horizontal_zone = np.zeros_like(candidate)
    zone_rows = max(edge_rows, int(round(height * 0.12)))
    horizontal_zone[:zone_rows] = True
    horizontal_zone[-zone_rows:] = True
    vertical_zone = np.zeros_like(candidate)
    zone_cols = max(edge_cols, int(round(width * 0.04)))
    vertical_zone[:, :zone_cols] = True
    vertical_zone[:, -zone_cols:] = True
    horizontal_seed = _axis_binary_opening(
        candidate & thin & ~protected_perimeter & horizontal_zone,
        max(12, int(round(width * 0.12))),
        axis=1,
    )
    vertical_seed = _axis_binary_opening(
        candidate & thin & ~protected_perimeter & vertical_zone,
        max(12, int(round(height * 0.12))),
        axis=0,
    )
    horizontal = _axis_binary_dilation(
        horizontal_seed,
        max(3, int(round(width * 0.25))),
        axis=1,
    ) & (candidate & thin & horizontal_zone)
    vertical = _axis_binary_dilation(
        vertical_seed,
        max(3, int(round(height * 0.25))),
        axis=0,
    ) & (candidate & thin & vertical_zone)
    rails = ndi.binary_dilation(
        horizontal | vertical,
        iterations=max(edge_rows, edge_cols) * 2,
    )
    candidate &= ~rails
    return _remove_border_bar_components(candidate)


def _axis_binary_opening(mask: np.ndarray, size: int, *, axis: int) -> np.ndarray:
    """Open a binary mask along one axis in linear time."""

    eroded = ndi.minimum_filter1d(
        mask, size=size, axis=axis, mode="constant", cval=False
    )
    return ndi.maximum_filter1d(
        eroded,
        size=size,
        axis=axis,
        mode="constant",
        cval=False,
        origin=-1 if size % 2 == 0 else 0,
    )


def _axis_binary_dilation(mask: np.ndarray, size: int, *, axis: int) -> np.ndarray:
    """Dilate a binary mask along one axis in linear time."""

    return ndi.maximum_filter1d(
        mask,
        size=size,
        axis=axis,
        mode="constant",
        cval=False,
        origin=-1 if size % 2 == 0 else 0,
    )


def _remove_border_bar_components(candidate: np.ndarray) -> np.ndarray:
    """Remove detached elongated components that run along the scanner edge."""

    labels, count = ndi.label(candidate)
    if count < 2:
        return candidate
    height, width = candidate.shape
    keep = np.ones(count + 1, dtype=bool)
    keep[0] = False
    component_sizes = np.bincount(labels.ravel(), minlength=count + 1)
    for label, bounds in enumerate(ndi.find_objects(labels), start=1):
        if bounds is None:
            continue
        row_bounds, col_bounds = bounds
        row_span = row_bounds.stop - row_bounds.start
        col_span = col_bounds.stop - col_bounds.start
        long_axis = max(row_span / height, col_span / width)
        short_axis = min(row_span / height, col_span / width)
        effective_short_axis = min(
            short_axis,
            (
                component_sizes[label] / (row_span * width)
                if row_span >= col_span
                else component_sizes[label] / (col_span * height)
            ),
        )
        touches_edge = (
            row_bounds.start < height * 0.02
            or row_bounds.stop - 1 >= height * 0.98
            or col_bounds.start < width * 0.02
            or col_bounds.stop - 1 >= width * 0.98
        )
        if touches_edge and long_axis > 0.10 and effective_short_axis < 0.08:
            keep[label] = False
    return keep[labels]


def _pale_tissue_candidate(
    rgb: np.ndarray,
    evidence: np.ndarray,
    optical_density: np.ndarray,
    config: BrightfieldMaskConfig,
) -> np.ndarray:
    """Propose low-contrast tissue only for group-gated recovery."""

    background = _estimate_background_rgb(rgb)
    color_delta = np.linalg.norm(rgb - background, axis=2)
    candidate = _clean_mask(_remove_scanner_edges(color_delta > 0.025), config)
    # Do not rank disconnected pieces here. The group augmentation step has
    # adjacent-section support that can distinguish pale tissue from debris.
    return clean_external_tissue_mask(candidate, config)


def _fill_small_holes(mask: np.ndarray, *, max_area: int) -> np.ndarray:
    background_labels, count = ndi.label(~mask)
    if count == 0:
        return mask
    sizes = np.bincount(background_labels.ravel())
    border_labels = np.unique(
        np.concatenate(
            [
                background_labels[0],
                background_labels[-1],
                background_labels[:, 0],
                background_labels[:, -1],
            ]
        )
    )
    fill = sizes <= max_area
    fill[border_labels] = False
    fill[0] = False
    return mask | fill[background_labels]


def _carve_large_blank_regions(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Remove large low-texture glass regions absorbed into a tissue mask."""

    if not mask.any():
        return mask
    brightness = np.mean(rgb, axis=2)
    local_mean = ndi.uniform_filter(brightness, size=15, mode="nearest")
    local_square_mean = ndi.uniform_filter(
        brightness * brightness,
        size=15,
        mode="nearest",
    )
    local_std = np.sqrt(np.maximum(local_square_mean - local_mean * local_mean, 0))
    background = _estimate_background_rgb(rgb)
    color_delta = np.linalg.norm(rgb - background, axis=2)
    low_texture = local_std < 0.025
    neutral_plate_proposal = (
        low_texture
        & (_saturation(rgb) < 0.025)
        & (brightness > 0.65)
        & (brightness < 0.96)
    )
    neutral_plate_labels, _ = ndi.label(neutral_plate_proposal)
    neutral_plate_sizes = np.bincount(neutral_plate_labels.ravel())
    fragmented_plate_ids = np.flatnonzero(neutral_plate_sizes >= mask.size * 0.005)
    fragmented_plate_ids = fragmented_plate_ids[fragmented_plate_ids != 0]
    white = np.all(rgb > 0.995, axis=2)
    separating_rows = np.mean(white, axis=1) > 0.95
    separating_cols = np.mean(white, axis=0) > 0.95
    fragmented_canvas = (
        np.mean(separating_rows) >= 0.08 or np.mean(separating_cols) >= 0.08
    )
    neutral_plate = (
        np.isin(neutral_plate_labels, fragmented_plate_ids)
        if fragmented_canvas and fragmented_plate_ids.size >= 3
        else np.zeros_like(mask, dtype=bool)
    )
    labels, count = ndi.label((low_texture & (color_delta < 0.15)) | neutral_plate)
    if count == 0:
        return mask
    sizes = np.bincount(labels.ravel())
    large = sizes >= mask.size * 0.01
    large[0] = False
    blank = ndi.binary_closing(large[labels], iterations=6)
    neutral_labels = np.unique(labels[neutral_plate])
    neutral_labels = neutral_labels[large[neutral_labels]]
    if neutral_labels.size:
        neutral_blank = np.isin(labels, neutral_labels)
        blank |= ndi.binary_dilation(neutral_blank, iterations=12)
    removed = mask & blank
    if np.count_nonzero(removed) < np.count_nonzero(mask) * 0.05:
        return mask
    carved = mask & ~blank
    carved_labels, carved_count = ndi.label(carved)
    if carved_count == 0:
        return mask
    carved_sizes = np.bincount(carved_labels.ravel())
    largest = int(carved_sizes[1:].max(initial=0))
    keep = carved_sizes >= largest * 0.01
    keep[0] = False
    result = keep[carved_labels]
    return result if result.any() else mask


def _resize_binary(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    zoom = (shape[0] / mask.shape[0], shape[1] / mask.shape[1])
    resized = ndi.zoom(np.asarray(mask, dtype=np.uint8), zoom, order=0)
    return resized[: shape[0], : shape[1]].astype(bool)


def _object_aware_fusion(
    rgb: np.ndarray,
    candidates: dict[str, np.ndarray],
    config: BrightfieldMaskConfig,
) -> np.ndarray:
    """Fuse complementary evidence, then retain tissue-like objects.

    Strong consensus seeds are expanded only through pixels supported by at
    least one independent brightfield signal. Object filtering is separate so
    pale tissue recovery does not also retain scanner rails and debris.
    """

    evidence = np.sum(np.stack(tuple(candidates.values())), axis=0, dtype=np.uint8)
    optical_density = np.mean(-np.log(np.clip(rgb, 1 / 255, 1.0)), axis=2)
    strong = (evidence >= 3) & (optical_density > 0.018)
    if config.close_radius_px > 0:
        strong = ndi.binary_closing(strong, iterations=config.close_radius_px)
    if config.open_radius_px > 0:
        strong = ndi.binary_opening(strong, iterations=config.open_radius_px)
    return _retain_tissue_objects(strong, evidence, optical_density, config)


def _retain_tissue_objects(
    mask: np.ndarray,
    evidence: np.ndarray,
    optical_density: np.ndarray,
    config: BrightfieldMaskConfig,
) -> np.ndarray:
    """Reject geometric artifacts while preserving multiple tissue pieces."""

    labels, label_count = ndi.label(mask)
    if label_count == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = np.bincount(labels.ravel())
    eligible = np.zeros(label_count + 1, dtype=bool)
    centers: dict[int, tuple[float, float]] = {}
    radii: dict[int, float] = {}
    valid_labels: list[int] = []
    height, width = mask.shape
    border_rows = max(1, int(height * 0.05))
    border_cols = max(1, int(width * 0.05))
    for label in range(1, label_count + 1):
        component = labels == label
        rows, cols = np.nonzero(component)
        area = rows.size
        if area < config.min_object_area_px:
            continue
        row_span = rows.max() - rows.min() + 1
        col_span = cols.max() - cols.min() + 1
        centers[label] = (float(np.mean(rows)), float(np.mean(cols)))
        radii[label] = float(np.sqrt(area / np.pi))
        long_axis = max(row_span / height, col_span / width)
        short_axis = min(row_span / height, col_span / width)
        fill = area / max(row_span * col_span, 1)
        near_border = (
            rows.min() < border_rows
            or rows.max() >= height - border_rows
            or cols.min() < border_cols
            or cols.max() >= width - border_cols
        )
        line_like = long_axis > 0.55 and short_axis < 0.18
        frame_like = (
            row_span / height > 0.75 and col_span / width > 0.75 and fill < 0.35
        )
        lattice_like = (
            near_border and long_axis > 0.30 and short_axis > 0.20 and fill < 0.32
        )
        if frame_like or lattice_like or (near_border and line_like):
            continue
        valid_labels.append(label)

    if not valid_labels:
        return np.zeros_like(mask, dtype=bool)
    largest = int(max(sizes[label] for label in valid_labels))
    for label in valid_labels:
        component = labels == label
        area = int(sizes[label])

        agreement = float(np.mean(evidence[component]))
        density = float(np.mean(optical_density[component]))
        strongly_supported = (
            area >= max(config.min_object_area_px, largest * 0.05)
            and agreement >= 2.75
            and density >= 0.022
        )
        substantial = (
            area >= max(config.min_object_area_px, largest * 0.40)
            and agreement >= 1.65
            and density >= 0.012
        )
        eligible[label] = strongly_supported or substantial

    if not np.any(eligible):
        return np.zeros_like(mask, dtype=bool)
    attachable = [int(label) for label in np.flatnonzero(eligible)]
    max_gap = 0.12 * float(np.hypot(height, width))
    unseen = set(attachable)
    clusters: list[set[int]] = []
    while unseen:
        cluster = {unseen.pop()}
        pending = list(cluster)
        while pending:
            current = pending.pop()
            for candidate in list(unseen):
                if (
                    _component_gap(
                        centers[current],
                        radii[current],
                        centers[candidate],
                        radii[candidate],
                    )
                    > max_gap
                ):
                    continue
                unseen.remove(candidate)
                cluster.add(candidate)
                pending.append(candidate)
        clusters.append(cluster)
    cluster_areas = [
        sum(int(sizes[label]) for label in cluster) for cluster in clusters
    ]
    largest_cluster = max(cluster_areas)
    retained: set[int] = set()
    for cluster, area in zip(clusters, cluster_areas, strict=True):
        if area < largest_cluster * 0.40:
            continue
        retained.update(cluster)
    keep = np.zeros(label_count + 1, dtype=bool)
    keep[list(retained)] = True
    return keep[labels]


def _component_gap(
    first_center: tuple[float, float],
    first_radius: float,
    second_center: tuple[float, float],
    second_radius: float,
) -> float:
    """Approximate edge separation without irregular bounding-box overlap."""

    center_distance = float(
        np.hypot(
            first_center[0] - second_center[0],
            first_center[1] - second_center[1],
        )
    )
    return max(0.0, center_distance - first_radius - second_radius)


def evaluate_tissue_mask(
    mask: np.ndarray,
    config: BrightfieldMaskConfig | None = None,
) -> tuple[dict[str, float], list[str]]:
    """Return the standard metrics and warnings for an externally supplied mask."""

    config = config or BrightfieldMaskConfig()
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError("mask must be a two-dimensional array")
    return _mask_metrics(binary), _mask_warnings(binary, config)


def clean_external_tissue_mask(
    mask: np.ndarray,
    config: BrightfieldMaskConfig | None = None,
) -> np.ndarray:
    """Apply the artifact cleanup used for automatic mask candidates."""

    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError("mask must be a two-dimensional array")
    resolved = config or BrightfieldMaskConfig()
    cleaned = _remove_border_dominated_components(binary, resolved)
    return _remove_long_thin_components(cleaned, resolved)


def _as_rgb_float(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, np.newaxis], 3, axis=2)
    if arr.ndim != 3 or arr.shape[2] < 3:
        msg = "image must be a grayscale or RGB-like array"
        raise ValueError(msg)
    arr = arr[:, :, :3].astype(np.float32, copy=False)
    if arr.max(initial=0) > 1.5:
        arr /= 255.0
    return np.clip(arr, 0.0, 1.0)


def _od_candidate(rgb: np.ndarray) -> np.ndarray:
    od = -np.log(np.clip(rgb, 1 / 255, 1.0))
    od_signal = np.mean(od, axis=2)
    threshold = max(_otsu_threshold(od_signal), 0.035)
    return od_signal > threshold


def _background_corrected_candidate(rgb: np.ndarray) -> np.ndarray:
    background_rgb = _estimate_background_rgb(rgb)
    color_delta = np.linalg.norm(rgb - background_rgb, axis=2)
    brightness = np.mean(rgb, axis=2)
    background_brightness = float(np.mean(background_rgb))
    dark_delta = background_brightness - brightness

    border_color_delta = _border_values(color_delta)
    border_dark_delta = _border_values(dark_delta)
    color_threshold = max(
        float(np.percentile(border_color_delta, 99.5)) + 0.025,
        0.055,
    )
    dark_threshold = max(
        float(np.percentile(border_dark_delta, 99.5)) + 0.015,
        0.035,
    )
    saturation = _saturation(rgb)
    candidate = (
        (color_delta > color_threshold)
        | (dark_delta > dark_threshold)
        | ((saturation > 0.10) & (brightness < 0.94))
    )
    return candidate & (brightness < 0.985)


def _hysteresis_tissue_candidate(rgb: np.ndarray) -> np.ndarray:
    """Grow pale tissue from stain-rich or textured tissue seeds."""

    background_rgb = _estimate_background_rgb(rgb)
    color_delta = np.linalg.norm(rgb - background_rgb, axis=2)
    brightness = np.mean(rgb, axis=2)
    optical_density = np.mean(-np.log(np.clip(rgb, 1 / 255, 1.0)), axis=2)
    gradient = np.hypot(
        ndi.sobel(brightness, axis=0),
        ndi.sobel(brightness, axis=1),
    )
    strong = (
        (color_delta > max(0.075, _otsu_threshold(color_delta)))
        | (optical_density > max(0.055, _otsu_threshold(optical_density)))
        | ((gradient > np.percentile(gradient, 94)) & (brightness < 0.97))
    )
    weak = (
        (color_delta > 0.025)
        | (optical_density > 0.018)
        | ((gradient > np.percentile(gradient, 82)) & (brightness < 0.985))
    ) & (brightness < 0.992)
    seeds = ndi.binary_dilation(strong, iterations=2)
    return ndi.binary_propagation(seeds, mask=weak | seeds)


def _saturation_value_candidate(rgb: np.ndarray) -> np.ndarray:
    max_channel = np.max(rgb, axis=2)
    min_channel = np.min(rgb, axis=2)
    chroma = max_channel - min_channel
    saturation = chroma / np.maximum(max_channel, 1e-6)
    value = max_channel
    return ((saturation > 0.035) | (value < 0.90)) & (value < 0.985)


def _adaptive_brightness_candidate(rgb: np.ndarray) -> np.ndarray:
    brightness = np.mean(rgb, axis=2)
    inverted = 1.0 - brightness
    window = max(15, int(min(rgb.shape[:2]) / 16))
    local_mean = ndi.uniform_filter(inverted, size=window, mode="nearest")
    local_sq_mean = ndi.uniform_filter(inverted * inverted, size=window, mode="nearest")
    local_std = np.sqrt(np.maximum(local_sq_mean - local_mean * local_mean, 0))
    threshold = local_mean + 0.25 * local_std
    return (inverted > threshold) & (brightness < 0.985)


def _edge_texture_candidate(rgb: np.ndarray) -> np.ndarray:
    brightness = np.mean(rgb, axis=2)
    gradient = np.hypot(
        ndi.sobel(brightness, axis=0),
        ndi.sobel(brightness, axis=1),
    )
    threshold = max(float(np.percentile(gradient, 92)), 0.015)
    return (gradient > threshold) & (brightness < 0.985)


def _estimate_background_rgb(rgb: np.ndarray) -> np.ndarray:
    border = np.concatenate(
        [rgb[0, :, :], rgb[-1, :, :], rgb[:, 0, :], rgb[:, -1, :]],
        axis=0,
    )
    pixels = rgb.reshape(-1, 3)
    sample_step = max(1, pixels.shape[0] // 100_000)
    pixels = pixels[::sample_step]
    candidates = np.concatenate([border, pixels], axis=0)
    brightness = np.mean(candidates, axis=1)
    saturation = _saturation(candidates[:, np.newaxis, :])[:, 0]
    bright_neutral = candidates[
        (brightness >= np.percentile(brightness, 70))
        & (saturation <= np.percentile(saturation, 60))
    ]
    if bright_neutral.size == 0:
        bright_neutral = candidates[brightness >= np.percentile(brightness, 80)]
    return np.median(bright_neutral, axis=0)


def _saturation(rgb: np.ndarray) -> np.ndarray:
    max_channel = np.max(rgb, axis=2)
    min_channel = np.min(rgb, axis=2)
    return (max_channel - min_channel) / np.maximum(max_channel, 1e-6)


def _border_values(values: np.ndarray) -> np.ndarray:
    return np.concatenate([values[0, :], values[-1, :], values[:, 0], values[:, -1]])


def _clean_mask(mask: np.ndarray, config: BrightfieldMaskConfig) -> np.ndarray:
    cleaned = np.asarray(mask, dtype=bool)
    if config.close_radius_px > 0:
        cleaned = ndi.binary_closing(cleaned, iterations=config.close_radius_px)
    if config.open_radius_px > 0:
        cleaned = ndi.binary_opening(cleaned, iterations=config.open_radius_px)
    labels, label_count = ndi.label(cleaned)
    if label_count == 0:
        return np.zeros_like(cleaned, dtype=bool)

    sizes = np.bincount(labels.ravel())
    keep = sizes >= config.min_object_area_px
    keep[0] = False
    cleaned = keep[labels]
    cleaned = _remove_border_dominated_components(cleaned, config)
    return _remove_long_thin_components(cleaned, config)


def _remove_long_thin_components(
    mask: np.ndarray,
    config: BrightfieldMaskConfig,
) -> np.ndarray:
    labels, label_count = ndi.label(mask)
    if label_count == 0:
        return mask
    keep = np.ones(label_count + 1, dtype=bool)
    keep[0] = False
    for label in range(1, label_count + 1):
        rows, cols = np.nonzero(labels == label)
        row_span = rows.max() - rows.min() + 1
        col_span = cols.max() - cols.min() + 1
        bbox_area = row_span * col_span
        fill = rows.size / max(bbox_area, 1)
        long_axis = max(row_span / mask.shape[0], col_span / mask.shape[1])
        short_axis = min(row_span / mask.shape[0], col_span / mask.shape[1])
        border_rows = max(1, int(mask.shape[0] * 0.05))
        border_cols = max(1, int(mask.shape[1] * 0.05))
        lies_near_border = (
            rows.min() < border_rows
            or rows.max() >= mask.shape[0] - border_rows
            or cols.min() < border_cols
            or cols.max() >= mask.shape[1] - border_cols
        )
        if long_axis > 0.65 and short_axis < 0.08 and fill < 0.35:
            keep[label] = False
        if lies_near_border and long_axis > 0.65 and short_axis < 0.06:
            keep[label] = False
        if rows.size < config.min_object_area_px:
            keep[label] = False
    return keep[labels]


def _remove_border_dominated_components(
    mask: np.ndarray,
    config: BrightfieldMaskConfig,
) -> np.ndarray:
    labels, label_count = ndi.label(mask)
    if label_count == 0:
        return np.zeros_like(mask, dtype=bool)

    border = np.zeros_like(mask, dtype=bool)
    strip_rows = max(1, int(mask.shape[0] * 0.05))
    strip_cols = max(1, int(mask.shape[1] * 0.05))
    border[:strip_rows, :] = True
    border[-strip_rows:, :] = True
    border[:, :strip_cols] = True
    border[:, -strip_cols:] = True

    sizes = np.bincount(labels.ravel())
    border_sizes = np.bincount(labels[border].ravel(), minlength=sizes.size)
    keep = np.ones(sizes.shape[0], dtype=bool)
    keep[0] = False
    border_fraction = border_sizes / np.maximum(sizes, 1)
    keep &= border_fraction <= config.max_component_border_fraction
    for label in range(1, label_count + 1):
        component_rows, component_cols = np.nonzero(labels == label)
        row_span = (component_rows.max() - component_rows.min() + 1) / mask.shape[0]
        col_span = (component_cols.max() - component_cols.min() + 1) / mask.shape[1]
        is_frame_like = row_span > 0.80 and col_span > 0.80
        if (
            is_frame_like
            and border_fraction[label] > config.max_frame_component_border_fraction
        ):
            keep[label] = False
    keep &= sizes >= config.min_object_area_px
    return keep[labels]


def _mask_metrics(mask: np.ndarray) -> dict[str, float]:
    mask = np.asarray(mask, dtype=bool)
    total = float(mask.size)
    area = float(mask.sum())
    labels, label_count = ndi.label(mask)
    sizes = np.bincount(labels.ravel()) if label_count else np.array([0])
    component_sizes = sizes[1:] if sizes.size > 1 else np.array([], dtype=np.int64)
    largest = float(component_sizes.max(initial=0))

    if area:
        extent_mask = _dominant_component_mask(mask)
        rows, cols = np.nonzero(extent_mask)
        bbox_area = float((rows.max() - rows.min() + 1) * (cols.max() - cols.min() + 1))
        border_pixels = np.concatenate(
            [mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]]
        )
        border_touch = float(border_pixels.mean())
        strip_rows = max(1, int(mask.shape[0] * 0.05))
        strip_cols = max(1, int(mask.shape[1] * 0.05))
        top_strip = float(mask[:strip_rows, :].mean())
        bottom_strip = float(mask[-strip_rows:, :].mean())
        left_strip = float(mask[:, :strip_cols].mean())
        right_strip = float(mask[:, -strip_cols:].mean())
        filled = ndi.binary_fill_holes(mask)
        hole_fraction = float((filled.sum() - mask.sum()) / max(filled.sum(), 1))
    else:
        bbox_area = 0.0
        border_touch = 0.0
        top_strip = 0.0
        bottom_strip = 0.0
        left_strip = 0.0
        right_strip = 0.0
        hole_fraction = 0.0

    return {
        "foreground_fraction": area / total,
        "component_count": float(label_count),
        "largest_component_fraction": largest / max(area, 1.0),
        "bbox_fraction": bbox_area / total,
        "border_touch_fraction": border_touch,
        "top_strip_foreground_fraction": top_strip,
        "bottom_strip_foreground_fraction": bottom_strip,
        "left_strip_foreground_fraction": left_strip,
        "right_strip_foreground_fraction": right_strip,
        "max_border_strip_foreground_fraction": max(
            top_strip,
            bottom_strip,
            left_strip,
            right_strip,
        ),
        "hole_fraction": hole_fraction,
    }


def _dominant_component_mask(
    mask: np.ndarray,
    min_relative_area: float = 0.01,
) -> np.ndarray:
    """Keep components large enough to define robust tissue crop bounds."""

    mask_bool = np.asarray(mask, dtype=bool)
    labels, label_count = ndi.label(mask_bool)
    if label_count == 0:
        return np.zeros_like(mask_bool)
    sizes = np.bincount(labels.ravel())
    largest = int(sizes[1:].max(initial=0))
    keep = sizes >= max(1, int(np.ceil(largest * min_relative_area)))
    keep[0] = False
    return keep[labels]


def _mask_warnings(mask: np.ndarray, config: BrightfieldMaskConfig) -> list[str]:
    metrics = _mask_metrics(mask)
    warnings: list[str] = []
    if metrics["foreground_fraction"] < config.min_foreground_fraction:
        warnings.append("foreground fraction is too small")
    if metrics["foreground_fraction"] > config.max_foreground_fraction:
        warnings.append("foreground fraction is too large")
    if metrics["largest_component_fraction"] < config.min_largest_component_fraction:
        warnings.append("largest component fraction is too small")
    if metrics["bbox_fraction"] < config.min_bbox_fraction:
        warnings.append("tissue bounding box is too small")
    if (
        metrics["foreground_fraction"] > 0.10
        and metrics["max_border_strip_foreground_fraction"]
        > config.max_border_strip_fraction
    ):
        warnings.append("mask includes broad border foreground")
    return warnings


def _mask_score(mask: np.ndarray, config: BrightfieldMaskConfig) -> float:
    warnings = _mask_warnings(mask, config)
    if warnings:
        return -float(len(warnings))
    metrics = _mask_metrics(mask)
    area = metrics["foreground_fraction"]
    bbox = metrics["bbox_fraction"]
    largest = metrics["largest_component_fraction"]
    border_penalty = metrics["max_border_strip_foreground_fraction"]
    fragmentation_penalty = min(metrics["component_count"] * 0.005, 0.25)
    return (
        1.0 + bbox + largest - abs(area - 0.25) - border_penalty - fragmentation_penalty
    )


def _otsu_threshold(values: np.ndarray) -> float:
    finite = np.asarray(values[np.isfinite(values)], dtype=np.float32)
    if finite.size == 0:
        return 0.0
    hist, edges = np.histogram(finite, bins=256)
    centers = (edges[:-1] + edges[1:]) / 2
    total = hist.sum()
    if total == 0:
        return 0.0
    weight_bg = np.cumsum(hist)
    weight_fg = total - weight_bg
    mean_bg = np.cumsum(hist * centers) / np.maximum(weight_bg, 1)
    mean_fg = (
        np.cumsum((hist * centers)[::-1]) / np.maximum(np.cumsum(hist[::-1]), 1)
    )[::-1]
    variance = weight_bg[:-1] * weight_fg[:-1] * (mean_bg[:-1] - mean_fg[1:]) ** 2
    if variance.size == 0:
        return float(np.mean(finite))
    return float(centers[int(np.argmax(variance))])
