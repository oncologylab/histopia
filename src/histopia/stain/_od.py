"""Optical-density conversion and guarded illumination correction."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class BackgroundModel:
    """Robust blank-glass white reference and quadratic shading surface."""

    white_reference: np.ndarray
    shading_coefficients: np.ndarray
    glass_pixels: int
    fallback_used: bool
    before_spatial_cv: float
    after_spatial_cv: float

    def __post_init__(self) -> None:
        white = np.asarray(self.white_reference, dtype=np.float64)
        coefficients = np.asarray(self.shading_coefficients, dtype=np.float64)
        if white.shape != (3,) or not np.all(np.isfinite(white)):
            raise ValueError("white_reference must contain three finite values")
        if np.any(white <= 0):
            raise ValueError("white_reference values must be positive")
        if coefficients.shape != (3, 6) or not np.all(np.isfinite(coefficients)):
            raise ValueError("shading_coefficients must have shape (3, 6)")
        object.__setattr__(self, "white_reference", white)
        object.__setattr__(self, "shading_coefficients", coefficients)

    def to_json_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["white_reference"] = self.white_reference.tolist()
        payload["shading_coefficients"] = self.shading_coefficients.tolist()
        return payload

    @classmethod
    def from_json_dict(cls, payload: dict[str, object]) -> BackgroundModel:
        return cls(
            white_reference=np.asarray(payload["white_reference"], dtype=float),
            shading_coefficients=np.asarray(
                payload["shading_coefficients"], dtype=float
            ),
            glass_pixels=int(payload["glass_pixels"]),
            fallback_used=bool(payload["fallback_used"]),
            before_spatial_cv=float(payload["before_spatial_cv"]),
            after_spatial_cv=float(payload["after_spatial_cv"]),
        )


def rgb_to_od(rgb: np.ndarray, white_reference: np.ndarray) -> np.ndarray:
    """Convert RGB transmission into nonnegative natural-log optical density."""

    values = np.asarray(rgb, dtype=np.float32)
    white = np.asarray(white_reference, dtype=np.float32)
    if values.shape[-1] != 3 or white.shape != (3,):
        raise ValueError("RGB values and white reference must have three channels")
    ratio = (values + 1.0) / (white + 1.0)
    return np.maximum(-np.log(np.clip(ratio, 1e-6, None)), 0.0).astype(np.float32)


def od_to_rgb(od: np.ndarray, white_reference: np.ndarray) -> np.ndarray:
    """Invert :func:`rgb_to_od` for tests and reconstruction diagnostics."""

    values = np.asarray(od, dtype=np.float32)
    white = np.asarray(white_reference, dtype=np.float32)
    if values.shape[-1] != 3 or white.shape != (3,):
        raise ValueError("OD values and white reference must have three channels")
    return np.clip((white + 1.0) * np.exp(-values) - 1.0, 0, 255).astype(np.uint8)


def estimate_background_model(
    rgb: np.ndarray,
    tissue_mask: np.ndarray,
    *,
    max_samples: int,
    seed: int,
) -> BackgroundModel:
    """Estimate blank glass and smooth scanner illumination from one slide."""

    image = np.asarray(rgb, dtype=np.uint8)
    tissue = np.asarray(tissue_mask, dtype=bool)
    if image.ndim != 3 or image.shape[2] != 3 or tissue.shape != image.shape[:2]:
        raise ValueError("RGB image and tissue mask shapes are inconsistent")
    brightness = image.mean(axis=2)
    chroma = image.max(axis=2) - image.min(axis=2)
    glass = (~tissue) & (brightness >= 160) & (chroma <= 90)
    fallback = int(np.count_nonzero(glass)) < 256
    if fallback:
        cutoff = float(np.quantile(brightness, 0.90))
        glass = (~tissue) & (brightness >= cutoff)
    if int(np.count_nonzero(glass)) < 64:
        cutoff = float(np.quantile(brightness, 0.95))
        glass = brightness >= cutoff
        fallback = True
    coordinates = np.argwhere(glass)
    if not len(coordinates):
        raise ValueError("no blank-glass candidates were found")
    rng = np.random.default_rng(seed)
    if len(coordinates) > max_samples:
        coordinates = coordinates[
            rng.choice(len(coordinates), size=max_samples, replace=False)
        ]
    sampled = image[coordinates[:, 0], coordinates[:, 1]].astype(np.float64)
    white = np.clip(np.quantile(sampled, 0.995, axis=0), 1.0, 255.0)
    design = _polynomial_design(
        coordinates[:, 1],
        coordinates[:, 0],
        image.shape[1],
        image.shape[0],
    )
    coefficients = np.zeros((3, 6), dtype=np.float64)
    for channel in range(3):
        response = sampled[:, channel] / white[channel]
        keep = np.ones(len(response), dtype=bool)
        fit = np.array([1, 0, 0, 0, 0, 0], dtype=np.float64)
        for _ in range(3):
            if np.count_nonzero(keep) < 32:
                break
            fit, *_ = np.linalg.lstsq(design[keep], response[keep], rcond=None)
            residual = np.abs(response - design @ fit)
            limit = float(np.quantile(residual[keep], 0.90))
            keep = residual <= max(limit, 1e-4)
        coefficients[channel] = fit
    before = _spatial_cv(sampled / white)
    local = _evaluate_polynomial_at(
        coordinates[:, 1],
        coordinates[:, 0],
        image.shape[1],
        image.shape[0],
        coefficients,
    )
    adjusted = sampled / np.clip(local, 0.65, 1.35)
    after = _spatial_cv(adjusted / white)
    return BackgroundModel(
        white_reference=white,
        shading_coefficients=coefficients,
        glass_pixels=int(np.count_nonzero(glass)),
        fallback_used=fallback,
        before_spatial_cv=before,
        after_spatial_cv=after,
    )


def apply_shading_correction(
    rgb: np.ndarray,
    model: BackgroundModel,
    *,
    normalized_bounds_xyxy: tuple[float, float, float, float] | None = None,
) -> np.ndarray:
    """Remove the smooth field at full-image or explicit tile coordinates."""

    image = np.asarray(rgb, dtype=np.float32)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("RGB image must have shape (height, width, 3)")
    height, width = image.shape[:2]
    bounds = (
        (-1.0, -1.0, 1.0, 1.0)
        if normalized_bounds_xyxy is None
        else tuple(float(value) for value in normalized_bounds_xyxy)
    )
    if len(bounds) != 4 or not np.all(np.isfinite(bounds)):
        raise ValueError("normalized tile bounds must contain four finite values")
    left, top, right, bottom = bounds
    if right < left or bottom < top:
        raise ValueError("normalized tile bounds must be ordered")
    x = np.linspace(left, right, width, dtype=np.float32)[None, :]
    y = np.linspace(top, bottom, height, dtype=np.float32)[:, None]
    output = np.empty_like(image, dtype=np.float32)
    for channel, coefficients in enumerate(model.shading_coefficients):
        c = coefficients.astype(np.float32)
        field = c[0] + c[1] * x + c[2] * y + c[3] * x * x + c[4] * x * y + c[5] * y * y
        output[..., channel] = image[..., channel] / np.clip(field, 0.65, 1.35)
    return np.clip(output, 0, 255).astype(np.uint8)


def _polynomial_design(
    x: np.ndarray,
    y: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    normalized_x = 2.0 * np.asarray(x, dtype=np.float64) / max(width - 1, 1) - 1.0
    normalized_y = 2.0 * np.asarray(y, dtype=np.float64) / max(height - 1, 1) - 1.0
    return np.column_stack(
        [
            np.ones(len(normalized_x)),
            normalized_x,
            normalized_y,
            normalized_x**2,
            normalized_x * normalized_y,
            normalized_y**2,
        ]
    )


def _evaluate_polynomial_at(
    x: np.ndarray,
    y: np.ndarray,
    width: int,
    height: int,
    coefficients: np.ndarray,
) -> np.ndarray:
    design = _polynomial_design(x, y, width, height)
    return design @ coefficients.T


def _spatial_cv(values: np.ndarray) -> float:
    luminance = np.asarray(values, dtype=float).mean(axis=1)
    mean = float(np.mean(luminance))
    return float(np.std(luminance) / mean) if mean > 0 else float("inf")
