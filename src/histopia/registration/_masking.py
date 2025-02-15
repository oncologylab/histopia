"""Brightfield/IHC tissue mask generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import ndimage as ndi

from histopia.registration._config import BrightfieldMaskConfig


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

    candidates = {
        "background_corrected": _clean_mask(
            _background_corrected_candidate(rgb),
            config,
        ),
        "edge_texture": _clean_mask(_edge_texture_candidate(rgb), config),
        "optical_density": _clean_mask(_od_candidate(rgb), config),
        "saturation_value": _clean_mask(_saturation_value_candidate(rgb), config),
        "adaptive_brightness": _clean_mask(_adaptive_brightness_candidate(rgb), config),
    }
    candidate_metrics = {
        method: _mask_metrics(mask) for method, mask in candidates.items()
    }
    candidate_warnings = {
        method: _mask_warnings(mask, config) for method, mask in candidates.items()
    }

    scored = [
        (
            _mask_score(mask, config),
            method,
            mask,
            candidate_metrics[method],
            candidate_warnings[method],
        )
        for method, mask in candidates.items()
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
    border_brightness = np.mean(border, axis=1)
    bright_border = border[border_brightness >= np.percentile(border_brightness, 60)]
    if bright_border.size == 0:
        bright_border = border
    return np.median(bright_border, axis=0)


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
    cleaned = ndi.binary_fill_holes(cleaned)
    labels, label_count = ndi.label(cleaned)
    if label_count == 0:
        return np.zeros_like(cleaned, dtype=bool)

    sizes = np.bincount(labels.ravel())
    keep = sizes >= config.min_object_area_px
    keep[0] = False
    cleaned = keep[labels]
    return _remove_border_dominated_components(cleaned, config)


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
