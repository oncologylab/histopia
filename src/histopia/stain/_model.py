"""Stain-vector estimation and reusable optical-density models."""

from __future__ import annotations

import math
import warnings
from dataclasses import asdict, dataclass
from itertools import permutations
from typing import Any

import numpy as np

from histopia.stain._assays import StainFamily
from histopia.stain._od import (
    BackgroundModel,
    apply_shading_correction,
    rgb_to_od,
)

_HEMATOXYLIN = (0.650, 0.704, 0.286)
_DAB = (0.268, 0.570, 0.776)
_SIRIUS_COUNTER = (0.355, 0.695, 0.625)
_SIRIUS_RED = (0.000, 0.976, 0.217)
_PAS = (0.175, 0.972, 0.155)
_ALCIAN_COUNTER = (0.214, 0.851, 0.480)
_ALCIAN_BLUE = (0.874, 0.457, 0.167)
_LEGACY_COUNTER = (0.5674455450, 0.6022160264, 0.5615526787)
_LEGACY_DAB = (0.3292281557, 0.5167694067, 0.7902899479)

_PRIORS = {
    StainFamily.H_DAB: (_HEMATOXYLIN, _DAB),
    StainFamily.SIRIUS_RED: (_SIRIUS_COUNTER, _SIRIUS_RED),
    StainFamily.PAS: (_HEMATOXYLIN, _PAS),
    StainFamily.ALCIAN_BLUE: (_ALCIAN_COUNTER, _ALCIAN_BLUE),
}


@dataclass(frozen=True, slots=True)
class CandidateFit:
    """One deterministic stain-vector candidate and its diagnostics."""

    method: str
    vectors: np.ndarray
    reconstruction_nrmse: float
    glass_leakage: float
    prior_angle_degrees: float
    bootstrap_angle_degrees: float
    target_q95: float
    converged: bool = True
    optimization_iterations: int = 0
    target_rank_correlation: float = 1.0

    def __post_init__(self) -> None:
        vectors = _validated_vectors(self.vectors)
        object.__setattr__(self, "vectors", vectors)
        for name in (
            "reconstruction_nrmse",
            "glass_leakage",
            "prior_angle_degrees",
            "bootstrap_angle_degrees",
            "target_q95",
            "target_rank_correlation",
        ):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        if not isinstance(self.converged, bool):
            raise TypeError("converged must be a bool")
        if (
            isinstance(self.optimization_iterations, bool)
            or not isinstance(self.optimization_iterations, (int, np.integer))
            or self.optimization_iterations < 0
        ):
            raise ValueError("optimization_iterations must be a nonnegative integer")
        object.__setattr__(
            self,
            "optimization_iterations",
            int(self.optimization_iterations),
        )

    def to_json_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["vectors"] = self.vectors.tolist()
        return payload

    @classmethod
    def from_json_dict(cls, payload: dict[str, object]) -> CandidateFit:
        return cls(
            method=str(payload["method"]),
            vectors=np.asarray(payload["vectors"], dtype=float),
            reconstruction_nrmse=float(payload["reconstruction_nrmse"]),
            glass_leakage=float(payload["glass_leakage"]),
            prior_angle_degrees=float(payload["prior_angle_degrees"]),
            bootstrap_angle_degrees=float(payload["bootstrap_angle_degrees"]),
            target_q95=float(payload["target_q95"]),
            converged=bool(payload.get("converged", True)),
            optimization_iterations=int(payload.get("optimization_iterations", 0)),
            target_rank_correlation=float(payload.get("target_rank_correlation", 1.0)),
        )


@dataclass(frozen=True, slots=True)
class StainConcentrations:
    """Raw and nuisance-corrected concentration maps."""

    raw_target_od: np.ndarray
    corrected_target_od: np.ndarray
    counterstain_od: np.ndarray
    reconstruction_residual: np.ndarray


@dataclass(frozen=True, slots=True)
class StainModel:
    """A fitted model that can quantify future native-resolution RGB tiles."""

    family: StainFamily
    marker: str
    method: str
    background: BackgroundModel
    raw_vectors: np.ndarray
    corrected_vectors: np.ndarray
    correction_accepted: bool
    correction_rank_correlation: float
    raw_glass_leakage: float
    corrected_glass_leakage: float
    content_bbox_native_xywh: tuple[int, int, int, int] | None = None
    positive_threshold_od: float | None = None
    threshold_accepted: bool = False

    def __post_init__(self) -> None:
        if self.family is StainFamily.CONTEXT_HE:
            raise ValueError("context H&E slides do not use a quantitative model")
        if not self.marker.strip() or not self.method.strip():
            raise ValueError("marker and method must not be blank")
        object.__setattr__(self, "raw_vectors", _validated_vectors(self.raw_vectors))
        object.__setattr__(
            self,
            "corrected_vectors",
            _validated_vectors(self.corrected_vectors),
        )
        for name in (
            "correction_rank_correlation",
            "raw_glass_leakage",
            "corrected_glass_leakage",
        ):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        if self.positive_threshold_od is not None:
            value = float(self.positive_threshold_od)
            if not math.isfinite(value) or value < 0:
                raise ValueError("positive_threshold_od must be finite and nonnegative")
        if self.content_bbox_native_xywh is not None:
            bounds = tuple(int(value) for value in self.content_bbox_native_xywh)
            if (
                len(bounds) != 4
                or bounds[2] <= 0
                or bounds[3] <= 0
                or any(value < 0 for value in bounds)
            ):
                raise ValueError("native content bounds must be a positive XYWH tuple")
            object.__setattr__(self, "content_bbox_native_xywh", bounds)

    def transform_rgb(
        self,
        rgb: np.ndarray,
        *,
        normalized_bounds_xyxy: tuple[float, float, float, float] | None = None,
    ) -> StainConcentrations:
        """Quantify RGB at full-image or explicit normalized coordinates."""

        raw_od = rgb_to_od(rgb, self.background.white_reference)
        raw_concentrations, _ = unmix_od(raw_od, self.raw_vectors)
        corrected_rgb = (
            apply_shading_correction(
                rgb,
                self.background,
                normalized_bounds_xyxy=normalized_bounds_xyxy,
            )
            if self.correction_accepted
            else np.asarray(rgb, dtype=np.uint8)
        )
        corrected_od = rgb_to_od(corrected_rgb, self.background.white_reference)
        vectors = (
            self.corrected_vectors if self.correction_accepted else self.raw_vectors
        )
        corrected_concentrations, residual = unmix_od(corrected_od, vectors)
        return StainConcentrations(
            raw_target_od=raw_concentrations[..., 1],
            corrected_target_od=corrected_concentrations[..., 1],
            counterstain_od=corrected_concentrations[..., 0],
            reconstruction_residual=residual,
        )

    def transform_native_tile(
        self,
        rgb: np.ndarray,
        *,
        tile_bbox_native_xywh: tuple[int, int, int, int],
    ) -> StainConcentrations:
        """Quantify an exact-resolution native tile in saved slide coordinates."""

        if self.content_bbox_native_xywh is None:
            raise ValueError("stain model does not contain native content bounds")
        image = np.asarray(rgb)
        tile_x, tile_y, tile_width, tile_height = (
            int(value) for value in tile_bbox_native_xywh
        )
        if (
            tile_width <= 0
            or tile_height <= 0
            or image.shape[:2] != (tile_height, tile_width)
        ):
            raise ValueError("native tile pixels and XYWH dimensions must match")
        x, y, width, height = self.content_bbox_native_xywh
        if (
            tile_x < x
            or tile_y < y
            or tile_x + tile_width > x + width
            or tile_y + tile_height > y + height
        ):
            raise ValueError("native tile must stay inside saved content bounds")
        left = 2.0 * (tile_x - x) / max(width - 1, 1) - 1.0
        right = 2.0 * (tile_x + tile_width - 1 - x) / max(width - 1, 1) - 1.0
        top = 2.0 * (tile_y - y) / max(height - 1, 1) - 1.0
        bottom = 2.0 * (tile_y + tile_height - 1 - y) / max(height - 1, 1) - 1.0
        return self.transform_rgb(
            image,
            normalized_bounds_xyxy=(left, top, right, bottom),
        )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": 2,
            "family": self.family.value,
            "marker": self.marker,
            "method": self.method,
            "background": self.background.to_json_dict(),
            "raw_vectors": self.raw_vectors.tolist(),
            "corrected_vectors": self.corrected_vectors.tolist(),
            "correction_accepted": self.correction_accepted,
            "correction_rank_correlation": self.correction_rank_correlation,
            "raw_glass_leakage": self.raw_glass_leakage,
            "corrected_glass_leakage": self.corrected_glass_leakage,
            "content_bbox_native_xywh": (
                list(self.content_bbox_native_xywh)
                if self.content_bbox_native_xywh is not None
                else None
            ),
            "positive_threshold_od": self.positive_threshold_od,
            "threshold_accepted": self.threshold_accepted,
        }

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> StainModel:
        if payload.get("schema_version") not in {1, 2}:
            raise ValueError("unsupported stain model schema")
        return cls(
            family=StainFamily(str(payload["family"])),
            marker=str(payload["marker"]),
            method=str(payload["method"]),
            background=BackgroundModel.from_json_dict(payload["background"]),
            raw_vectors=np.asarray(payload["raw_vectors"], dtype=float),
            corrected_vectors=np.asarray(payload["corrected_vectors"], dtype=float),
            correction_accepted=bool(payload["correction_accepted"]),
            correction_rank_correlation=float(payload["correction_rank_correlation"]),
            raw_glass_leakage=float(payload["raw_glass_leakage"]),
            corrected_glass_leakage=float(payload["corrected_glass_leakage"]),
            content_bbox_native_xywh=(
                tuple(int(value) for value in payload["content_bbox_native_xywh"])
                if payload.get("content_bbox_native_xywh") is not None
                else None
            ),
            positive_threshold_od=(
                float(payload["positive_threshold_od"])
                if payload.get("positive_threshold_od") is not None
                else None
            ),
            threshold_accepted=bool(payload.get("threshold_accepted", False)),
        )


def canonical_vectors(family: StainFamily) -> np.ndarray:
    """Return versioned counterstain and target vectors for one family."""

    if family is StainFamily.CONTEXT_HE:
        raise ValueError("H&E context slides have no target stain vector")
    return _validated_vectors(np.asarray(_PRIORS[family], dtype=np.float64))


def legacy_vectors(family: StainFamily) -> np.ndarray:
    """Return the historical Yi H-DAB baseline or the family fixed baseline."""

    if family is StainFamily.H_DAB:
        return _validated_vectors(
            np.asarray((_LEGACY_COUNTER, _LEGACY_DAB), dtype=np.float64)
        )
    return canonical_vectors(family)


def unmix_od(
    optical_density: np.ndarray,
    vectors: np.ndarray,
    *,
    chunk_pixels: int = 500_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve exact two-component nonnegative least squares in bounded chunks."""

    od = np.asarray(optical_density, dtype=np.float32)
    basis = _validated_vectors(vectors).astype(np.float32)
    if od.shape[-1] != 3:
        raise ValueError("optical density must have three channels")
    flat = od.reshape(-1, 3)
    concentrations = np.empty((len(flat), 2), dtype=np.float32)
    residual = np.empty(len(flat), dtype=np.float32)
    gram_inverse = np.linalg.inv(basis @ basis.T)
    projection = basis.T @ gram_inverse
    for start in range(0, len(flat), chunk_pixels):
        stop = min(start + chunk_pixels, len(flat))
        values = flat[start:stop]
        unconstrained = values @ projection
        valid = np.all(unconstrained >= 0, axis=1)
        selected = np.maximum(unconstrained, 0)
        if np.any(~valid):
            subset = values[~valid]
            only0 = np.maximum(subset @ basis[0], 0)
            only1 = np.maximum(subset @ basis[1], 0)
            residual0 = np.sum(
                (subset - only0[:, None] * basis[0]) ** 2,
                axis=1,
            )
            residual1 = np.sum(
                (subset - only1[:, None] * basis[1]) ** 2,
                axis=1,
            )
            use0 = residual0 <= residual1
            replacement = np.zeros((len(subset), 2), dtype=np.float32)
            replacement[use0, 0] = only0[use0]
            replacement[~use0, 1] = only1[~use0]
            selected[~valid] = replacement
        reconstruction = selected @ basis
        concentrations[start:stop] = selected
        residual[start:stop] = np.sqrt(np.mean((values - reconstruction) ** 2, axis=1))
    shape = od.shape[:-1]
    return concentrations.reshape(*shape, 2), residual.reshape(shape)


def fit_candidate(
    tissue_od: np.ndarray,
    glass_od: np.ndarray,
    family: StainFamily,
    method: str,
    *,
    seed: int,
) -> CandidateFit:
    """Fit one method and calculate directly comparable diagnostics."""

    tissue = _valid_od_rows(tissue_od)
    glass = _valid_od_rows(glass_od, minimum_norm=0.0)
    prior = canonical_vectors(family)
    converged = True
    optimization_iterations = 0
    if method == "legacy":
        vectors = legacy_vectors(family)
    elif method == "fixed":
        vectors = prior
    elif method == "macenko":
        vectors = estimate_macenko_vectors(tissue, prior)
    elif method == "nmf":
        vectors, converged, optimization_iterations = _fit_nmf_vectors(
            tissue,
            prior,
            seed=seed,
        )
    else:
        raise ValueError(f"unsupported stain method: {method}")
    concentrations, residual = unmix_od(tissue, vectors)
    reference_concentrations, _ = unmix_od(tissue, prior)
    glass_concentrations, _ = unmix_od(glass, vectors)
    denominator = max(float(np.sqrt(np.mean(tissue**2))), 1e-8)
    bootstrap, bootstrap_converged = _bootstrap_angle(
        tissue,
        prior,
        method,
        vectors,
        seed=seed,
    )
    return CandidateFit(
        method=method,
        vectors=vectors,
        reconstruction_nrmse=float(np.sqrt(np.mean(residual**2)) / denominator),
        glass_leakage=float(np.quantile(glass_concentrations[:, 1], 0.95)),
        prior_angle_degrees=float(np.mean(_row_angles(vectors, prior))),
        bootstrap_angle_degrees=bootstrap,
        target_q95=float(np.quantile(concentrations[:, 1], 0.95)),
        converged=converged and bootstrap_converged,
        optimization_iterations=optimization_iterations,
        target_rank_correlation=_rank_correlation(
            reference_concentrations[:, 1],
            concentrations[:, 1],
        ),
    )


def estimate_macenko_vectors(
    optical_density: np.ndarray,
    priors: np.ndarray,
) -> np.ndarray:
    """Estimate robust OD-plane endpoints and match them to family priors."""

    values = _valid_od_rows(optical_density)
    if len(values) < 64:
        return _validated_vectors(priors)
    covariance = np.cov(values, rowvar=False)
    _, eigenvectors = np.linalg.eigh(covariance)
    plane = eigenvectors[:, -2:]
    projected = values @ plane
    angles = np.arctan2(projected[:, 1], projected[:, 0])
    low, high = np.quantile(angles, [0.01, 0.99])
    endpoints = np.vstack(
        [
            plane @ np.array([np.cos(low), np.sin(low)]),
            plane @ np.array([np.cos(high), np.sin(high)]),
        ]
    )
    endpoints = np.abs(endpoints)
    return _match_vectors(endpoints, priors)


def estimate_nmf_vectors(
    optical_density: np.ndarray,
    priors: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    """Estimate nonnegative components initialized by family priors."""

    vectors, _, _ = _fit_nmf_vectors(optical_density, priors, seed=seed)
    return vectors


def _fit_nmf_vectors(
    optical_density: np.ndarray,
    priors: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, bool, int]:
    values = _valid_od_rows(optical_density)
    if len(values) < 64:
        return _validated_vectors(priors), True, 0
    try:
        from sklearn.decomposition import NMF
        from sklearn.exceptions import ConvergenceWarning
    except ImportError as exc:
        raise RuntimeError("NMF stain fitting requires the 'stain' extra") from exc
    initial = _validated_vectors(priors).astype(values.dtype, copy=False)
    weights, _ = unmix_od(values, initial)
    estimator = NMF(
        n_components=2,
        init="custom",
        solver="mu",
        beta_loss="frobenius",
        max_iter=300,
        tol=1e-4,
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        estimator.fit_transform(
            np.maximum(values, 0),
            W=np.maximum(weights, 1e-6),
            H=np.maximum(initial, 1e-6),
        )
    iterations = int(estimator.n_iter_)
    return (
        _match_vectors(estimator.components_, priors),
        iterations < estimator.max_iter,
        iterations,
    )


def shrink_vectors(
    slide_vectors: np.ndarray,
    cohort_vectors: np.ndarray,
    amount: float,
) -> np.ndarray:
    """Shrink slide-specific vector directions toward a cohort template."""

    slide = _validated_vectors(slide_vectors)
    cohort = _validated_vectors(cohort_vectors)
    return _validated_vectors((1.0 - amount) * slide + amount * cohort)


def cohort_vector_template(vectors: list[np.ndarray]) -> np.ndarray:
    """Return a robust row-wise cohort stain-vector template."""

    if not vectors:
        raise ValueError("cohort template requires at least one slide")
    return _validated_vectors(np.median(np.stack(vectors), axis=0))


def select_family_method(
    candidates: list[list[CandidateFit]],
    *,
    minimum_target_rank: float = 0.98,
    minimum_rank_guard_fraction: float = 0.90,
) -> tuple[str, dict[str, dict[str, float]]]:
    """Select a converged, rank-preserving method across nuisance diagnostics."""

    if not candidates:
        raise ValueError("method selection requires candidate fits")
    if not 0 <= minimum_target_rank <= 1:
        raise ValueError("minimum_target_rank must be between zero and one")
    if not 0 <= minimum_rank_guard_fraction <= 1:
        raise ValueError("minimum_rank_guard_fraction must be between zero and one")
    methods = sorted(
        set.intersection(*(set(row.method for row in slide) for slide in candidates))
    )
    if not methods:
        raise ValueError("slides have no common candidate methods")
    metrics = (
        "reconstruction_nrmse",
        "glass_leakage",
        "prior_angle_degrees",
        "bootstrap_angle_degrees",
    )
    summary: dict[str, dict[str, float]] = {}
    for method in methods:
        rows = [
            next(row for row in slide if row.method == method) for slide in candidates
        ]
        summary[method] = {
            metric: float(np.median([getattr(row, metric) for row in rows]))
            for metric in metrics
        }
        summary[method]["convergence_fraction"] = float(
            np.mean([row.converged for row in rows])
        )
        summary[method]["target_rank_correlation"] = float(
            np.median([row.target_rank_correlation for row in rows])
        )
        summary[method]["rank_guard_fraction"] = float(
            np.mean(
                [row.target_rank_correlation >= minimum_target_rank for row in rows]
            )
        )
    rank_totals = {method: 0.0 for method in methods}
    for metric in metrics:
        ordered = sorted(methods, key=lambda method: (summary[method][metric], method))
        for rank, method in enumerate(ordered):
            rank_totals[method] += rank
    for method in methods:
        summary[method]["rank_total"] = rank_totals[method]
    eligible = [
        method
        for method in methods
        if summary[method]["convergence_fraction"] == 1.0
        and summary[method]["rank_guard_fraction"] >= minimum_rank_guard_fraction
    ]
    if not eligible:
        raise ValueError(
            "stain methods contain no converged, rank-preserving candidate"
        )
    selected = min(
        eligible,
        key=lambda method: (
            rank_totals[method],
            summary[method]["glass_leakage"],
            summary[method]["reconstruction_nrmse"],
            method,
        ),
    )
    return selected, summary


def _bootstrap_angle(
    tissue: np.ndarray,
    priors: np.ndarray,
    method: str,
    vectors: np.ndarray,
    *,
    seed: int,
) -> tuple[float, bool]:
    if method in {"fixed", "legacy"} or len(tissue) < 256:
        return 0.0, True
    rng = np.random.default_rng(seed)
    angles = []
    converged = True
    count = min(len(tissue), 5_000)
    for index in range(2):
        sample = tissue[rng.choice(len(tissue), size=count, replace=True)]
        if method == "macenko":
            estimated = estimate_macenko_vectors(sample, priors)
        else:
            estimated, stable, _ = _fit_nmf_vectors(
                sample,
                priors,
                seed=seed + index + 1,
            )
            converged = converged and stable
        angles.extend(_row_angles(estimated, vectors))
    return float(np.quantile(angles, 0.95)), converged


def _valid_od_rows(
    values: np.ndarray,
    *,
    minimum_norm: float = 0.05,
) -> np.ndarray:
    rows = np.asarray(values, dtype=np.float32).reshape(-1, 3)
    finite = np.all(np.isfinite(rows), axis=1)
    strong = np.linalg.norm(rows, axis=1) >= minimum_norm
    selected = rows[finite & strong]
    if not len(selected):
        return np.zeros((1, 3), dtype=np.float32)
    return selected


def _validated_vectors(vectors: np.ndarray) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float64)
    if values.shape != (2, 3) or not np.all(np.isfinite(values)):
        raise ValueError("stain vectors must have shape (2, 3)")
    values = np.maximum(values, 0)
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms <= 1e-8):
        raise ValueError("stain vectors must have nonzero rows")
    normalized = values / norms[:, None]
    if abs(float(normalized[0] @ normalized[1])) > 0.995:
        raise ValueError("stain vectors are too nearly collinear")
    return normalized


def _match_vectors(vectors: np.ndarray, priors: np.ndarray) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float64)
    prior = _validated_vectors(priors)
    values = np.maximum(values, 0)
    values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)
    best = min(
        permutations(range(2)),
        key=lambda order: sum(
            1.0 - float(values[source] @ prior[target])
            for target, source in enumerate(order)
        ),
    )
    return _validated_vectors(values[list(best)])


def _row_angles(left: np.ndarray, right: np.ndarray) -> list[float]:
    first = _validated_vectors(left)
    second = _validated_vectors(right)
    cosine = np.clip(np.sum(first * second, axis=1), -1, 1)
    return [float(value) for value in np.degrees(np.arccos(cosine))]


def _rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    first = np.asarray(left, dtype=float)
    second = np.asarray(right, dtype=float)
    if len(first) != len(second) or len(first) < 2:
        return 0.0
    first_ranks = _average_ranks(first)
    second_ranks = _average_ranks(second)
    if np.ptp(first_ranks) == 0 or np.ptp(second_ranks) == 0:
        return 0.0
    correlation = np.corrcoef(first_ranks, second_ranks)[0, 1]
    return float(correlation) if math.isfinite(float(correlation)) else 0.0


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    boundaries = np.concatenate(
        (
            np.array([0]),
            np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1,
            np.array([len(values)]),
        )
    )
    ranks = np.empty(len(values), dtype=float)
    for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
        ranks[order[start:stop]] = (start + stop - 1) / 2
    return ranks
