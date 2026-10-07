"""Conservative tissue-only adaptive background subtraction for target OD."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from histopia.stain._model import _rank_correlation, counterstain_only_mask

_MIN_SUPPORT_PIXELS = 256
_MIN_SUPPORT_FRACTION = 0.02
_LOW_FRACTION_SUPPORT_PIXELS = 4_096
_MIN_SUPPRESSION_FRACTION = 0.10
_MAX_SUPPRESSION_FRACTION = 0.75
_MIN_Q95_RETENTION = 0.45
_COUNTERSTAIN_METHOD = "counterstain-conditioned-v3"


@dataclass(frozen=True, slots=True)
class AdaptiveBackgroundResult:
    """Diagnostics for one inferred target-OD floor."""

    method: str
    accepted: bool
    floor_od: float
    support_pixels: int
    tissue_pixels: int
    support_fraction: float
    bootstrap_width_od: float
    suppression_fraction: float
    q95_retention: float
    rank_correlation: float
    post_support_q95_od: float
    rejection_reasons: tuple[str, ...]

    def to_json_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["rejection_reasons"] = list(self.rejection_reasons)
        return payload

    @classmethod
    def from_json_dict(cls, payload: dict[str, object]) -> AdaptiveBackgroundResult:
        return cls(
            method=str(payload["method"]),
            accepted=bool(payload["accepted"]),
            floor_od=float(payload["floor_od"]),
            support_pixels=int(payload["support_pixels"]),
            tissue_pixels=int(payload["tissue_pixels"]),
            support_fraction=float(payload["support_fraction"]),
            bootstrap_width_od=float(payload["bootstrap_width_od"]),
            suppression_fraction=float(payload["suppression_fraction"]),
            q95_retention=float(payload["q95_retention"]),
            rank_correlation=float(payload["rank_correlation"]),
            post_support_q95_od=float(payload["post_support_q95_od"]),
            rejection_reasons=tuple(str(v) for v in payload["rejection_reasons"]),
        )


@dataclass(frozen=True, slots=True)
class CounterstainAdaptiveResult:
    """Auditable parameters for tissue-only counterstain-conditioned OD.

    The model removes only a bounded nuisance envelope.  It does not rescale
    the surviving signal or force different sections to share a marginal
    distribution, so biological prevalence remains available to downstream
    models.
    """

    method: str
    accepted: bool
    intercept_od: float
    counterstain_slope: float
    counterstain_center_od: float
    minimum_nuisance_od: float
    maximum_nuisance_od: float
    legacy_floor_od: float
    blend_fraction: float
    mixture_components: int
    negative_components: int
    mixture_separation: float
    support_pixels: int
    tissue_pixels: int
    support_fraction: float
    bootstrap_width_od: float
    suppression_fraction: float
    q95_retention: float
    physical_tail_rank_correlation: float
    residual_rank_correlation: float
    post_support_q95_od: float
    counterstain_correlation_before: float
    counterstain_correlation_after: float
    spatial_field_used: bool
    rejection_reasons: tuple[str, ...]

    def to_json_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["rejection_reasons"] = list(self.rejection_reasons)
        return payload

    @classmethod
    def from_json_dict(cls, payload: dict[str, object]) -> CounterstainAdaptiveResult:
        if payload.get("method") != _COUNTERSTAIN_METHOD:
            raise ValueError("counterstain adaptive background method is unsupported")
        return cls(
            method=str(payload["method"]),
            accepted=bool(payload["accepted"]),
            intercept_od=float(payload["intercept_od"]),
            counterstain_slope=float(payload["counterstain_slope"]),
            counterstain_center_od=float(payload["counterstain_center_od"]),
            minimum_nuisance_od=float(payload["minimum_nuisance_od"]),
            maximum_nuisance_od=float(payload["maximum_nuisance_od"]),
            legacy_floor_od=float(payload["legacy_floor_od"]),
            blend_fraction=float(payload["blend_fraction"]),
            mixture_components=int(payload["mixture_components"]),
            negative_components=int(payload["negative_components"]),
            mixture_separation=float(payload["mixture_separation"]),
            support_pixels=int(payload["support_pixels"]),
            tissue_pixels=int(payload["tissue_pixels"]),
            support_fraction=float(payload["support_fraction"]),
            bootstrap_width_od=float(payload["bootstrap_width_od"]),
            suppression_fraction=float(payload["suppression_fraction"]),
            q95_retention=float(payload["q95_retention"]),
            physical_tail_rank_correlation=float(
                payload["physical_tail_rank_correlation"]
            ),
            residual_rank_correlation=float(payload["residual_rank_correlation"]),
            post_support_q95_od=float(payload["post_support_q95_od"]),
            counterstain_correlation_before=float(
                payload["counterstain_correlation_before"]
            ),
            counterstain_correlation_after=float(
                payload["counterstain_correlation_after"]
            ),
            spatial_field_used=bool(payload["spatial_field_used"]),
            rejection_reasons=tuple(
                str(value) for value in payload["rejection_reasons"]
            ),
        )


def infer_counterstain_conditioned_background(
    target_od: np.ndarray,
    counterstain_od: np.ndarray,
    tissue_mask: np.ndarray,
    *,
    confidence: np.ndarray | None = None,
    seed: int = 0,
) -> CounterstainAdaptiveResult:
    """Infer a bounded per-slide nuisance model from negative tissue.

    A three-component mixture on ``log1p(target OD)`` separates the broad
    negative population from the positive tail.  A robust nonnegative slope
    then removes target-channel leakage that rises with counterstain OD.  The
    candidate is blended back toward the validated v2 scalar floor whenever
    necessary to retain at least 45% of q95 signal and avoid suppressing more
    than 75% of integrated OD.
    """

    target, counter, tissue, confidence_values = _validated_counterstain_inputs(
        target_od,
        counterstain_od,
        tissue_mask,
        confidence,
    )
    tissue_values = target[tissue]
    counter_values = counter[tissue]
    confidence_tissue = confidence_values[tissue]
    tissue_pixels = int(len(tissue_values))
    fit_mask = confidence_tissue >= 0.25
    fit_indices = np.flatnonzero(fit_mask)
    rng = np.random.default_rng(seed)
    if len(fit_indices) > 200_000:
        fit_indices = np.sort(rng.choice(fit_indices, size=200_000, replace=False))
    fit_target = tissue_values[fit_indices]
    fit_counter = counter_values[fit_indices]
    labels, probabilities, means, variances, weights = _fit_log_mixture(
        fit_target,
        seed=seed,
    )
    order = np.argsort(means)
    cumulative = np.cumsum(weights[order])
    negative_components = min(
        2,
        max(1, int(np.searchsorted(cumulative, 0.55)) + 1),
    )
    negative_ids = order[:negative_components]
    positive_id = int(order[negative_components])
    last_negative = int(order[negative_components - 1])
    mixture_separation = float(
        (means[positive_id] - means[last_negative])
        / max(
            np.sqrt((variances[positive_id] + variances[last_negative]) / 2.0),
            1e-8,
        )
    )
    negative_probability = probabilities[:, negative_ids].sum(axis=1)
    assigned_negative = np.isin(labels, negative_ids)
    lowest = labels == int(order[0])
    high_confidence_negative = negative_probability >= 0.90

    legacy = infer_adaptive_background(target, counter, tissue, seed=seed)
    legacy_floor = float(legacy.floor_od if legacy.accepted else 0.0)
    lowest_values = fit_target[lowest]
    negative_values = fit_target[assigned_negative]
    candidate_floor = max(
        legacy_floor,
        float(np.quantile(lowest_values, 0.95)) if len(lowest_values) else 0.0,
        float(np.quantile(negative_values, 0.80)) if len(negative_values) else 0.0,
    )

    regression_support = high_confidence_negative
    slope, center = _counterstain_leakage_slope(
        fit_target[regression_support],
        fit_counter[regression_support],
        candidate_floor,
    )
    blend = _bounded_correction_blend(
        tissue_values,
        counter_values,
        legacy_floor=legacy_floor,
        candidate_intercept=candidate_floor,
        candidate_slope=slope,
        center=center,
    )
    intercept = legacy_floor + blend * (candidate_floor - legacy_floor)
    slope *= blend
    minimum = max(legacy_floor, 0.50 * intercept)
    maximum = max(minimum, 1.50 * intercept)
    nuisance = np.clip(
        intercept + slope * (counter_values - center),
        minimum,
        maximum,
    )
    adaptive_values = np.maximum(tissue_values - nuisance, 0.0)

    sample_nuisance = np.clip(
        intercept + slope * (fit_counter - center),
        minimum,
        maximum,
    )
    inferred_support = high_confidence_negative & (fit_target <= sample_nuisance + 0.02)
    support_values = np.maximum(
        fit_target[inferred_support] - sample_nuisance[inferred_support],
        0.0,
    )
    support_pixels = int(np.count_nonzero(inferred_support))
    support_fraction = support_pixels / max(len(fit_target), 1)
    post_support_q95 = (
        float(np.quantile(support_values, 0.95))
        if len(support_values)
        else float("inf")
    )
    total_before = float(np.sum(tissue_values, dtype=np.float64))
    total_after = float(np.sum(adaptive_values, dtype=np.float64))
    suppression = 1.0 - total_after / max(total_before, 1e-12)
    before_q95 = float(np.quantile(tissue_values, 0.95))
    after_q95 = float(np.quantile(adaptive_values, 0.95))
    retention = after_q95 / max(before_q95, 1e-12)

    tail = tissue_values >= np.quantile(tissue_values, 0.90)
    physical_rank = _safe_rank(
        tissue_values[tail & (adaptive_values > 0)],
        adaptive_values[tail & (adaptive_values > 0)],
    )
    residual_before_floor = np.maximum(
        tissue_values - slope * (counter_values - center),
        0.0,
    )
    retained = adaptive_values > 0
    residual_rank = _safe_rank(
        residual_before_floor[retained], adaptive_values[retained]
    )
    before_corr = _safe_correlation(
        fit_target[high_confidence_negative],
        fit_counter[high_confidence_negative],
    )
    after_corr = _safe_correlation(
        np.maximum(
            fit_target[high_confidence_negative]
            - sample_nuisance[high_confidence_negative],
            0.0,
        ),
        fit_counter[high_confidence_negative],
    )
    bootstrap_width = _bootstrap_negative_floor_width(
        negative_values,
        legacy_floor=legacy_floor,
        seed=seed,
    )

    reasons: list[str] = []
    minimum_support = min(
        tissue_pixels,
        max(_LOW_FRACTION_SUPPORT_PIXELS, int(0.02 * tissue_pixels)),
    )
    if support_pixels < minimum_support:
        reasons.append("insufficient_negative_component_support")
    if mixture_separation < 0.75:
        reasons.append("negative_positive_mixture_not_separated")
    if bootstrap_width > max(0.02, 0.20 * max(intercept, 1e-6)):
        reasons.append("nuisance_bootstrap_unstable")
    if not _MIN_SUPPRESSION_FRACTION <= suppression <= _MAX_SUPPRESSION_FRACTION + 1e-5:
        reasons.append("suppression_outside_guard")
    if retention < _MIN_Q95_RETENTION - 1e-7:
        reasons.append("q95_retention_below_guard")
    if residual_rank < 0.99:
        reasons.append("residual_rank_not_preserved")
    if post_support_q95 > 0.02 + 1e-7:
        reasons.append("negative_component_background_remains")
    return CounterstainAdaptiveResult(
        method=_COUNTERSTAIN_METHOD,
        accepted=not reasons,
        intercept_od=float(intercept),
        counterstain_slope=float(slope),
        counterstain_center_od=float(center),
        minimum_nuisance_od=float(minimum),
        maximum_nuisance_od=float(maximum),
        legacy_floor_od=legacy_floor,
        blend_fraction=float(blend),
        mixture_components=3,
        negative_components=int(negative_components),
        mixture_separation=mixture_separation,
        support_pixels=support_pixels,
        tissue_pixels=tissue_pixels,
        support_fraction=float(support_fraction),
        bootstrap_width_od=float(bootstrap_width),
        suppression_fraction=float(suppression),
        q95_retention=float(retention),
        physical_tail_rank_correlation=float(physical_rank),
        residual_rank_correlation=float(residual_rank),
        post_support_q95_od=post_support_q95,
        counterstain_correlation_before=float(before_corr),
        counterstain_correlation_after=float(after_corr),
        spatial_field_used=False,
        rejection_reasons=tuple(reasons),
    )


def apply_counterstain_conditioned_background(
    target_od: np.ndarray,
    counterstain_od: np.ndarray,
    tissue_mask: np.ndarray,
    result: CounterstainAdaptiveResult,
) -> np.ndarray:
    """Apply one accepted-or-candidate v3 nuisance model inside tissue only."""

    target, counter, tissue, _confidence = _validated_counterstain_inputs(
        target_od,
        counterstain_od,
        tissue_mask,
        None,
    )
    nuisance = _nuisance_values(
        counter[tissue],
        intercept=result.intercept_od,
        slope=result.counterstain_slope,
        center=result.counterstain_center_od,
        minimum=result.minimum_nuisance_od,
        maximum=result.maximum_nuisance_od,
    )
    output = np.zeros(target.shape, dtype=np.float32)
    output[tissue] = np.maximum(target[tissue] - nuisance, 0.0)
    return output


def infer_adaptive_background(
    target_od: np.ndarray,
    counterstain_od: np.ndarray,
    tissue_mask: np.ndarray,
    *,
    seed: int = 0,
) -> AdaptiveBackgroundResult:
    """Infer and gate a scalar floor without changing physical OD arrays."""

    target = np.asarray(target_od, dtype=np.float32)
    counter = np.asarray(counterstain_od, dtype=np.float32)
    tissue = np.asarray(tissue_mask, dtype=bool)
    if target.shape != counter.shape or target.shape != tissue.shape:
        raise ValueError("adaptive OD inputs must have the same shape")
    tissue_values = target[tissue]
    counter_values = counter[tissue]
    if not len(tissue_values):
        raise ValueError("adaptive OD requires tissue pixels")
    support_local = counterstain_only_mask(
        np.column_stack([counter_values, tissue_values])
    )
    support_values = tissue_values[support_local]
    support_pixels = int(len(support_values))
    tissue_pixels = int(len(tissue_values))
    support_fraction = support_pixels / tissue_pixels
    floor = float(
        min(
            np.quantile(support_values, 0.95),
            np.quantile(tissue_values, 0.35),
        )
    )
    adaptive_values = np.maximum(tissue_values - floor, 0)
    total_before = float(np.sum(tissue_values, dtype=np.float64))
    total_after = float(np.sum(adaptive_values, dtype=np.float64))
    suppression = 1.0 - total_after / max(total_before, 1e-12)
    before_q95 = float(np.quantile(tissue_values, 0.95))
    after_q95 = float(np.quantile(adaptive_values, 0.95))
    retention = after_q95 / max(before_q95, 1e-12)
    retained = adaptive_values > 0
    rank = _rank_correlation(tissue_values[retained], adaptive_values[retained])
    post_support_q95 = float(np.quantile(np.maximum(support_values - floor, 0), 0.95))
    bootstrap_width = _bootstrap_floor_width(
        support_values,
        tissue_values,
        seed=seed,
    )

    reasons: list[str] = []
    if support_pixels < _MIN_SUPPORT_PIXELS:
        reasons.append("insufficient_support_pixels")
    # A genuinely broad stain can leave only a small fraction of
    # counterstain-dominant tissue.  In that case an absolute support of at
    # least 4k pixels, together with the bootstrap and rank/retention guards
    # below, is more informative than rejecting solely on area fraction.
    if (
        support_fraction < _MIN_SUPPORT_FRACTION
        and support_pixels < _LOW_FRACTION_SUPPORT_PIXELS
    ):
        reasons.append("insufficient_support_fraction")
    if floor < 0.02:
        reasons.append("floor_below_minimum")
    if bootstrap_width > max(0.02, 0.20 * floor):
        reasons.append("floor_bootstrap_unstable")
    # Total-OD suppression can be high on an over-stained slide because most
    # tissue pixels carry the same nuisance floor.  Permit that case only when
    # the upper-tail signal remains substantial and its ordering is preserved.
    if not _MIN_SUPPRESSION_FRACTION <= suppression <= _MAX_SUPPRESSION_FRACTION:
        reasons.append("suppression_outside_guard")
    if retention < _MIN_Q95_RETENTION:
        reasons.append("q95_retention_below_guard")
    if rank < 0.995:
        reasons.append("rank_not_preserved")
    if post_support_q95 > 0.02 + 1e-7:
        reasons.append("support_background_remains")
    return AdaptiveBackgroundResult(
        method="inferred_floor-v2",
        accepted=not reasons,
        floor_od=floor,
        support_pixels=support_pixels,
        tissue_pixels=tissue_pixels,
        support_fraction=float(support_fraction),
        bootstrap_width_od=bootstrap_width,
        suppression_fraction=float(suppression),
        q95_retention=float(retention),
        rank_correlation=float(rank),
        post_support_q95_od=post_support_q95,
        rejection_reasons=tuple(reasons),
    )


def apply_adaptive_background(
    values: np.ndarray,
    tissue_mask: np.ndarray,
    result: AdaptiveBackgroundResult,
) -> np.ndarray:
    """Return the accepted derived map, or the unchanged physical fallback."""

    signal = np.asarray(values, dtype=np.float32)
    tissue = np.asarray(tissue_mask, dtype=bool)
    if signal.shape != tissue.shape:
        raise ValueError("adaptive OD values and tissue mask must have the same shape")
    floor = result.floor_od if result.accepted else 0.0
    output = np.zeros(signal.shape, dtype=np.float32)
    output[tissue] = np.maximum(signal[tissue] - floor, 0)
    return output


def _bootstrap_floor_width(
    support: np.ndarray,
    tissue: np.ndarray,
    *,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    support_values = _bounded_sample(support, rng, 20_000)
    tissue_values = _bounded_sample(tissue, rng, 50_000)
    if len(support_values) < 2 or len(tissue_values) < 2:
        return float("inf")
    estimates = np.empty(100, dtype=np.float64)
    for index in range(len(estimates)):
        sampled_support = rng.choice(support_values, len(support_values), replace=True)
        sampled_tissue = rng.choice(tissue_values, len(tissue_values), replace=True)
        estimates[index] = min(
            np.quantile(sampled_support, 0.95),
            np.quantile(sampled_tissue, 0.35),
        )
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(high - low)


def _bounded_sample(
    values: np.ndarray,
    rng: np.random.Generator,
    maximum: int,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if len(array) <= maximum:
        return array
    return array[rng.choice(len(array), maximum, replace=False)]


def _validated_counterstain_inputs(
    target_od: np.ndarray,
    counterstain_od: np.ndarray,
    tissue_mask: np.ndarray,
    confidence: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    target = np.asarray(target_od, dtype=np.float32)
    counter = np.asarray(counterstain_od, dtype=np.float32)
    tissue = np.asarray(tissue_mask, dtype=bool)
    confidence_values = (
        np.ones(target.shape, dtype=np.float32)
        if confidence is None
        else np.asarray(confidence, dtype=np.float32)
    )
    if (
        target.ndim != 2
        or target.shape != counter.shape
        or target.shape != tissue.shape
        or target.shape != confidence_values.shape
    ):
        raise ValueError("counterstain adaptive inputs must share one 2D shape")
    for name, values in (
        ("target OD", target),
        ("counterstain OD", counter),
        ("confidence", confidence_values),
    ):
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError(f"counterstain adaptive {name} must be nonnegative")
    if not np.any(tissue):
        raise ValueError("counterstain adaptive correction requires tissue")
    return target, counter, tissue, confidence_values


def _fit_log_mixture(
    values: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    signal = np.asarray(values, dtype=np.float32)
    if len(signal) < 256 or float(np.ptp(signal)) <= 1e-8:
        raise ValueError("counterstain adaptive correction needs variable tissue OD")
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError as exc:
        raise RuntimeError(
            "counterstain-conditioned correction requires Histopia's stain extra"
        ) from exc
    transformed = np.log1p(signal).reshape(-1, 1)
    model = GaussianMixture(
        n_components=3,
        covariance_type="full",
        random_state=seed,
        reg_covar=1e-6,
        n_init=3,
        max_iter=200,
    ).fit(transformed)
    labels = model.predict(transformed)
    probabilities = model.predict_proba(transformed)
    return (
        labels,
        probabilities,
        np.asarray(model.means_[:, 0], dtype=np.float64),
        np.asarray(model.covariances_[:, 0, 0], dtype=np.float64),
        np.asarray(model.weights_, dtype=np.float64),
    )


def _counterstain_leakage_slope(
    target: np.ndarray,
    counterstain: np.ndarray,
    intercept: float,
) -> tuple[float, float]:
    values = np.asarray(target, dtype=np.float64)
    counter = np.asarray(counterstain, dtype=np.float64)
    if len(values) < 256 or float(np.ptp(counter)) <= 1e-8 or intercept <= 0:
        return 0.0, float(np.median(counter)) if len(counter) else 0.0
    center = float(np.median(counter))
    x = counter - center
    y = values - np.median(values)
    slope = max(float(np.dot(x, y) / max(np.dot(x, x), 1e-12)), 0.0)
    for _ in range(10):
        residual = y - slope * x
        scale = 1.4826 * float(np.median(np.abs(residual - np.median(residual)))) + 1e-8
        weights = np.minimum(
            1.0,
            1.345 * scale / np.maximum(np.abs(residual), 1e-12),
        )
        denominator = float(np.dot(weights * x, x))
        if denominator <= 1e-12:
            slope = 0.0
            break
        slope = max(float(np.dot(weights * x, y) / denominator), 0.0)
    counter_span = float(np.quantile(counter, 0.95) - np.quantile(counter, 0.05))
    if counter_span > 0:
        slope = min(slope, 0.25 * intercept / counter_span)
    return float(slope), center


def _nuisance_values(
    counterstain: np.ndarray,
    *,
    intercept: float,
    slope: float,
    center: float,
    minimum: float | None = None,
    maximum: float | None = None,
) -> np.ndarray:
    low = max(0.0, 0.50 * intercept) if minimum is None else float(minimum)
    high = max(low, 1.50 * intercept) if maximum is None else float(maximum)
    values = intercept + slope * (np.asarray(counterstain) - center)
    return np.clip(values, low, high).astype(np.float32)


def _bounded_correction_blend(
    target: np.ndarray,
    counterstain: np.ndarray,
    *,
    legacy_floor: float,
    candidate_intercept: float,
    candidate_slope: float,
    center: float,
) -> float:
    values = np.asarray(target, dtype=np.float64)
    counter = np.asarray(counterstain, dtype=np.float64)
    before_q95 = float(np.quantile(values, 0.95))
    total_before = float(np.sum(values, dtype=np.float64))

    def accepted(fraction: float) -> bool:
        intercept = legacy_floor + fraction * (candidate_intercept - legacy_floor)
        slope = candidate_slope * fraction
        minimum = max(legacy_floor, 0.50 * intercept)
        maximum = max(minimum, 1.50 * intercept)
        nuisance = np.clip(
            intercept + slope * (counter - center),
            minimum,
            maximum,
        )
        output = np.maximum(values - nuisance, 0.0)
        retention = float(np.quantile(output, 0.95)) / max(before_q95, 1e-12)
        suppression = 1.0 - float(np.sum(output)) / max(total_before, 1e-12)
        return (
            retention >= _MIN_Q95_RETENTION - 1e-8
            and suppression <= _MAX_SUPPRESSION_FRACTION - 1e-4
        )

    if accepted(1.0):
        return 1.0
    low, high = 0.0, 1.0
    if not accepted(low):
        return 0.0
    for _ in range(30):
        middle = (low + high) / 2.0
        if accepted(middle):
            low = middle
        else:
            high = middle
    return float(low)


def _bootstrap_negative_floor_width(
    values: np.ndarray,
    *,
    legacy_floor: float,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    sample = _bounded_sample(np.asarray(values, dtype=np.float32), rng, 20_000)
    if len(sample) < 256:
        return float("inf")
    estimates = np.empty(50, dtype=np.float64)
    for index in range(len(estimates)):
        drawn = rng.choice(sample, len(sample), replace=True)
        estimates[index] = max(legacy_floor, float(np.quantile(drawn, 0.80)))
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(high - low)


def _safe_rank(left: np.ndarray, right: np.ndarray) -> float:
    first = np.asarray(left, dtype=np.float64)
    second = np.asarray(right, dtype=np.float64)
    if len(first) < 3 or first.shape != second.shape:
        return 1.0
    return float(_rank_correlation(first, second))


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    first = np.asarray(left, dtype=np.float64)
    second = np.asarray(right, dtype=np.float64)
    if (
        len(first) < 3
        or first.shape != second.shape
        or float(np.ptp(first)) <= 1e-12
        or float(np.ptp(second)) <= 1e-12
    ):
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])
