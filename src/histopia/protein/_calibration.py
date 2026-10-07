"""Auditable monotonic calibration for repeated target-stain measurements."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np


def confidence_weighted_registered_residual_correction(
    baseline_od: np.ndarray,
    transferred_residual_od: np.ndarray,
    anchor_confidence: np.ndarray,
    *,
    maximum_weight: float,
    confidence_power: float = 1.0,
    directional_tail_guard: float = 1.0,
    lower_quantile: float = 0.25,
    upper_quantile: float = 0.90,
    maximum_absolute_correction_od: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply registered measured-minus-baseline residuals to a query slide.

    The transferred quantity is an error correction learned only from measured
    anchor cells: ``observed OD - frozen inductive prediction``.  This differs
    from blending absolute OD because a negative local residual can directly
    remove diffuse background and a positive residual can restore a compressed
    bright tail.  Unsupported residuals fall back exactly to ``baseline_od``.

    The directional guard suppresses uncertain corrections that would raise a
    low-ranked baseline cell or lower a high-ranked one.  It does not suppress
    the two desired repair directions.  An optional absolute cap bounds the OD
    change after confidence weighting.
    """

    baseline = np.asarray(baseline_od, dtype=np.float64)
    residual = np.asarray(transferred_residual_od, dtype=np.float64)
    confidence = np.asarray(anchor_confidence, dtype=np.float64)
    if (
        baseline.ndim != 1
        or residual.shape != baseline.shape
        or confidence.shape != baseline.shape
    ):
        raise ValueError("registered residual correction inputs must align in 1D")
    if (
        not np.all(np.isfinite(baseline))
        or np.any(baseline < 0)
        or not np.all(np.isfinite(confidence))
        or np.any((confidence < 0) | (confidence > 1))
    ):
        raise ValueError("registered residual correction inputs are invalid")
    if not np.isfinite(maximum_weight) or not 0 <= maximum_weight <= 1:
        raise ValueError("maximum_weight must be between zero and one")
    if not np.isfinite(confidence_power) or confidence_power <= 0:
        raise ValueError("confidence_power must be positive")
    if not np.isfinite(directional_tail_guard) or not 0 <= directional_tail_guard <= 1:
        raise ValueError("directional_tail_guard must be between zero and one")
    if (
        not np.isfinite(lower_quantile)
        or not np.isfinite(upper_quantile)
        or not 0 <= lower_quantile < upper_quantile <= 1
    ):
        raise ValueError("tail quantiles must be ordered within [0, 1]")
    if maximum_absolute_correction_od is not None and (
        not np.isfinite(maximum_absolute_correction_od)
        or maximum_absolute_correction_od <= 0
    ):
        raise ValueError("maximum_absolute_correction_od must be positive")

    supported = np.isfinite(residual)
    effective = np.zeros(len(baseline), dtype=np.float64)
    effective[supported] = maximum_weight * confidence[supported] ** confidence_power
    lower = np.quantile(baseline, lower_quantile)
    upper = np.quantile(baseline, upper_quantile)
    toward_center = supported & (
        ((baseline <= lower) & (residual > 0)) | ((baseline >= upper) & (residual < 0))
    )
    effective[toward_center] *= 1.0 - directional_tail_guard * (
        1.0 - confidence[toward_center]
    )
    correction = np.zeros(len(baseline), dtype=np.float64)
    correction[supported] = effective[supported] * residual[supported]
    if maximum_absolute_correction_od is not None:
        correction = np.clip(
            correction,
            -maximum_absolute_correction_od,
            maximum_absolute_correction_od,
        )
    refined = np.maximum(baseline + correction, 0.0)
    return refined.astype(np.float32), effective.astype(np.float32)


def tail_guarded_registered_anchor_blend(
    baseline_od: np.ndarray,
    transferred_od: np.ndarray,
    anchor_confidence: np.ndarray,
    *,
    maximum_weight: float,
    confidence_power: float = 1.0,
    directional_tail_guard: float = 1.0,
    lower_quantile: float = 0.25,
    upper_quantile: float = 0.90,
) -> tuple[np.ndarray, np.ndarray]:
    """Blend registered anchors while guarding low background and bright tails.

    The guard uses only the inductive baseline rank and anchor-match
    confidence.  It suppresses uncertain moves from low baseline cells upward
    and high baseline cells downward, the two directions that create diffuse
    background and compressed expression tails.  High-confidence registered
    matches remain able to override the baseline.  Unsupported ``NaN`` anchor
    predictions fall back exactly to the inductive baseline.
    """

    baseline = np.asarray(baseline_od, dtype=np.float64)
    transferred = np.asarray(transferred_od, dtype=np.float64)
    confidence = np.asarray(anchor_confidence, dtype=np.float64)
    if (
        baseline.ndim != 1
        or transferred.shape != baseline.shape
        or confidence.shape != baseline.shape
    ):
        raise ValueError("registered anchor blend inputs must align in 1D")
    if (
        not np.all(np.isfinite(baseline))
        or np.any(baseline < 0)
        or not np.all(np.isfinite(confidence))
        or np.any((confidence < 0) | (confidence > 1))
        or np.any(np.isfinite(transferred) & (transferred < 0))
    ):
        raise ValueError("registered anchor blend inputs are invalid")
    if not np.isfinite(maximum_weight) or not 0 <= maximum_weight <= 1:
        raise ValueError("maximum_weight must be between zero and one")
    if not np.isfinite(confidence_power) or confidence_power <= 0:
        raise ValueError("confidence_power must be positive")
    if not np.isfinite(directional_tail_guard) or not 0 <= directional_tail_guard <= 1:
        raise ValueError("directional_tail_guard must be between zero and one")
    if (
        not np.isfinite(lower_quantile)
        or not np.isfinite(upper_quantile)
        or not 0 <= lower_quantile < upper_quantile <= 1
    ):
        raise ValueError("tail quantiles must be ordered within [0, 1]")

    supported = np.isfinite(transferred)
    effective = np.zeros(len(baseline), dtype=np.float64)
    effective[supported] = maximum_weight * confidence[supported] ** confidence_power
    lower = np.quantile(baseline, lower_quantile)
    upper = np.quantile(baseline, upper_quantile)
    toward_center = supported & (
        ((baseline <= lower) & (transferred > baseline))
        | ((baseline >= upper) & (transferred < baseline))
    )
    effective[toward_center] *= 1.0 - directional_tail_guard * (
        1.0 - confidence[toward_center]
    )
    refined = baseline.copy()
    refined[supported] += effective[supported] * (
        transferred[supported] - baseline[supported]
    )
    return refined.astype(np.float32), effective.astype(np.float32)


def confidence_adaptive_morphology_transfer(
    baseline_od: np.ndarray,
    transferred_od: np.ndarray,
    baseline_relative_expression: np.ndarray,
    *,
    base_weight: float,
    gate_power: float = 0.5,
    gate_floor: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Blend a morphology outcome bank without flattening confident tails.

    ``baseline_relative_expression`` must be the neural model's percentile in
    its training-only OD reference.  The symmetric gate peaks at the middle of
    that reference and tapers toward zero at confident low and high values::

        gate = floor + (1 - floor) * (4 * r * (1 - r)) ** power

    This keeps the transfer target stain-free at inference, while limiting the
    regression-to-the-mean that a fixed nearest-neighbor blend can introduce.
    The returned effective weights make the adaptive policy directly auditable.
    """

    baseline = np.asarray(baseline_od, dtype=np.float64)
    transferred = np.asarray(transferred_od, dtype=np.float64)
    relative = np.asarray(baseline_relative_expression, dtype=np.float64)
    if (
        baseline.ndim != 1
        or transferred.shape != baseline.shape
        or relative.shape != baseline.shape
    ):
        raise ValueError("confidence-adaptive transfer inputs must align in 1D")
    if (
        not np.all(np.isfinite(baseline))
        or not np.all(np.isfinite(transferred))
        or not np.all(np.isfinite(relative))
        or np.any(baseline < 0)
        or np.any(transferred < 0)
        or np.any((relative < 0) | (relative > 1))
    ):
        raise ValueError("confidence-adaptive transfer inputs are invalid")
    if not np.isfinite(base_weight) or not 0 <= base_weight <= 1:
        raise ValueError("confidence-adaptive base_weight must be between zero and one")
    if not np.isfinite(gate_power) or gate_power <= 0:
        raise ValueError("confidence-adaptive gate_power must be positive")
    if not np.isfinite(gate_floor) or not 0 <= gate_floor <= 1:
        raise ValueError("confidence-adaptive gate_floor must be between zero and one")

    confidence_gate = np.power(
        np.clip(4.0 * relative * (1.0 - relative), 0.0, 1.0),
        gate_power,
    )
    confidence_gate = gate_floor + (1.0 - gate_floor) * confidence_gate
    effective_weight = base_weight * confidence_gate
    refined = baseline + effective_weight * (transferred - baseline)
    return (
        np.maximum(refined, 0.0).astype(np.float32),
        effective_weight.astype(np.float32),
    )


@dataclass(frozen=True, slots=True)
class SectionOdCalibration:
    """A portable, zero-anchored piecewise-linear OD calibration."""

    section: str
    source_knots: np.ndarray
    reference_knots: np.ndarray
    anchor_count: int
    status: str
    scale: float

    def __post_init__(self) -> None:
        source = np.asarray(self.source_knots, dtype=np.float64)
        reference = np.asarray(self.reference_knots, dtype=np.float64)
        if (
            source.ndim != 1
            or source.shape != reference.shape
            or len(source) < 2
            or source[0] != 0
            or reference[0] != 0
            or np.any(np.diff(source) <= 0)
            or np.any(np.diff(reference) < 0)
            or not np.all(np.isfinite(source))
            or not np.all(np.isfinite(reference))
        ):
            raise ValueError("OD calibration knots must be finite and monotonic")
        if self.anchor_count < 0 or self.status not in {"accepted", "identity"}:
            raise ValueError("OD calibration status is invalid")
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("OD calibration scale must be positive")

    def apply(self, values: np.ndarray) -> np.ndarray:
        """Map nonnegative values to the common assay reference."""

        raw = np.asarray(values, dtype=np.float64)
        output = np.full(raw.shape, np.nan, dtype=np.float64)
        valid = np.isfinite(raw) & (raw >= 0)
        if not np.any(valid):
            return output.astype(np.float32)
        source = np.asarray(self.source_knots, dtype=np.float64)
        reference = np.asarray(self.reference_knots, dtype=np.float64)
        mapped = np.interp(raw[valid], source, reference)
        high = raw[valid] > source[-1]
        if np.any(high):
            tail = max(
                (reference[-1] - reference[-2]) / (source[-1] - source[-2]),
                0.0,
            )
            mapped[high] = reference[-1] + tail * (raw[valid][high] - source[-1])
        output[valid] = np.maximum(mapped, 0.0)
        return output.astype(np.float32)

    def as_dict(self) -> dict[str, object]:
        return {
            "section": self.section,
            "source_knots": np.asarray(self.source_knots).tolist(),
            "reference_knots": np.asarray(self.reference_knots).tolist(),
            "anchor_count": self.anchor_count,
            "status": self.status,
            "scale": self.scale,
        }


@dataclass(frozen=True, slots=True)
class RegisteredOdCalibration:
    """Calibration collection bound to registered spatial anchors."""

    sections: tuple[SectionOdCalibration, ...]
    bin_um: float
    minimum_anchors: int
    reference_section: str
    fingerprint: str

    def for_section(self, section: str) -> SectionOdCalibration:
        try:
            return next(row for row in self.sections if row.section == section)
        except StopIteration as error:
            raise KeyError(f"no OD calibration for section {section}") from error

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "method": "registered-bin-monotonic-v1",
            "bin_um": self.bin_um,
            "minimum_anchors": self.minimum_anchors,
            "reference_section": self.reference_section,
            "sections": [row.as_dict() for row in self.sections],
            "fingerprint": self.fingerprint,
        }


def fit_registered_od_calibration(
    section_ids: np.ndarray,
    reference_um_xy: np.ndarray,
    measured_od: np.ndarray,
    semantic_region: np.ndarray,
    support: np.ndarray,
    *,
    bin_um: float = 64.0,
    minimum_cells_per_bin: int = 4,
    minimum_anchors: int = 128,
    scale_bounds: tuple[float, float] = (0.5, 2.5),
    knot_quantiles: int = 32,
) -> RegisteredOdCalibration:
    """Harmonize repeated stains using shared registered tissue neighborhoods.

    Only one global monotonic mapping is learned per stained section.  The
    mapping cannot create local expression patterns or change within-section
    ordering, and identity is used when registered support is insufficient.
    """

    sections = np.asarray(section_ids, dtype=str)
    xy = np.asarray(reference_um_xy, dtype=np.float64)
    od = np.asarray(measured_od, dtype=np.float64)
    regions = np.asarray(semantic_region, dtype=np.int64)
    supported = np.asarray(support, dtype=bool)
    count = len(sections)
    if (
        xy.shape != (count, 2)
        or od.shape != (count,)
        or regions.shape != (count,)
        or supported.shape != (count,)
    ):
        raise ValueError("registered OD calibration inputs do not align")
    if not np.isfinite(bin_um) or bin_um <= 0:
        raise ValueError("bin_um must be positive")
    if minimum_cells_per_bin < 1 or minimum_anchors < 2 or knot_quantiles < 2:
        raise ValueError("calibration support controls must be positive")
    low, high = (float(value) for value in scale_bounds)
    if not 0 < low <= high:
        raise ValueError("calibration scale bounds are invalid")

    unique_sections = tuple(sorted(np.unique(sections)))
    if len(unique_sections) < 2:
        raise ValueError("registered OD calibration requires repeated stains")
    bins = np.floor(xy / bin_um).astype(np.int64)
    summaries: dict[str, dict[tuple[int, int, int], float]] = {}
    for section in unique_sections:
        selected = (
            (sections == section)
            & supported
            & np.isfinite(od)
            & (od >= 0)
            & np.all(np.isfinite(xy), axis=1)
        )
        groups: dict[tuple[int, int, int], list[float]] = {}
        for (bx, by), region, value in zip(
            bins[selected], regions[selected], od[selected], strict=True
        ):
            groups.setdefault((int(bx), int(by), int(region)), []).append(float(value))
        summaries[section] = {
            key: float(np.median(values))
            for key, values in groups.items()
            if len(values) >= minimum_cells_per_bin
        }

    common = set.intersection(*(set(summaries[section]) for section in unique_sections))
    reference_section = max(
        unique_sections,
        key=lambda section: (
            float(np.median([summaries[section][key] for key in common]))
            if common
            else float("-inf"),
            section,
        ),
    )
    pooled = {key: summaries[reference_section][key] for key in common}
    calibrations = tuple(
        _fit_section_calibration(
            section,
            summaries[section],
            pooled,
            minimum_anchors=minimum_anchors,
            scale_bounds=(low, high),
            knot_quantiles=knot_quantiles,
        )
        for section in unique_sections
    )
    core = {
        "schema_version": 1,
        "method": "registered-bin-monotonic-v1",
        "bin_um": float(bin_um),
        "minimum_anchors": int(minimum_anchors),
        "reference_section": reference_section,
        "sections": [row.as_dict() for row in calibrations],
    }
    fingerprint = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return RegisteredOdCalibration(
        calibrations,
        float(bin_um),
        int(minimum_anchors),
        reference_section,
        fingerprint,
    )


def _fit_section_calibration(
    section: str,
    summary: dict[tuple[int, int, int], float],
    pooled: dict[tuple[int, int, int], float],
    *,
    minimum_anchors: int,
    scale_bounds: tuple[float, float],
    knot_quantiles: int,
) -> SectionOdCalibration:
    keys = sorted(set(summary) & set(pooled))
    source = np.asarray([summary[key] for key in keys], dtype=np.float64)
    target = np.asarray([pooled[key] for key in keys], dtype=np.float64)
    positive = np.isfinite(source) & np.isfinite(target) & (source > 0) & (target >= 0)
    source, target = source[positive], target[positive]
    if len(source) < minimum_anchors:
        maximum = max(float(source.max(initial=1.0)), 1.0)
        return SectionOdCalibration(
            section,
            np.asarray([0.0, maximum]),
            np.asarray([0.0, maximum]),
            int(len(source)),
            "identity",
            1.0,
        )
    ratio = np.median(target[source > 1e-8] / source[source > 1e-8])
    ratio = float(np.clip(ratio, *scale_bounds))
    quantiles = np.linspace(0.0, 1.0, min(knot_quantiles, len(source)) + 1)[1:]
    source_knots = np.quantile(source, quantiles)
    target_knots = np.quantile(target, quantiles)
    keep = np.r_[True, np.diff(source_knots) > 1e-8]
    source_knots, target_knots = source_knots[keep], target_knots[keep]
    target_knots = np.maximum.accumulate(target_knots)
    target_knots = np.clip(
        target_knots,
        source_knots * scale_bounds[0],
        source_knots * scale_bounds[1],
    )
    source_knots = np.r_[0.0, source_knots]
    target_knots = np.r_[0.0, target_knots]
    if len(source_knots) < 2:
        maximum = max(float(source.max(initial=1.0)), 1.0)
        source_knots = np.asarray([0.0, maximum])
        target_knots = np.asarray([0.0, maximum * ratio])
    return SectionOdCalibration(
        section,
        source_knots.astype(np.float32),
        target_knots.astype(np.float32),
        int(len(source)),
        "accepted",
        ratio,
    )


@dataclass(frozen=True, slots=True)
class CohortSectionOdHarmonization:
    """One cohort/section mapping onto a common repeated-assay OD scale."""

    cohort: str
    calibration: SectionOdCalibration

    def __post_init__(self) -> None:
        if not self.cohort:
            raise ValueError("OD harmonization cohort must be non-empty")

    @property
    def section(self) -> str:
        return self.calibration.section

    @property
    def scale(self) -> float:
        return self.calibration.scale

    @property
    def status(self) -> str:
        return self.calibration.status

    def apply(self, values: np.ndarray) -> np.ndarray:
        """Apply the section's zero-preserving monotonic transform."""

        return self.calibration.apply(values)

    def as_dict(self) -> dict[str, object]:
        return {"cohort": self.cohort, **self.calibration.as_dict()}


@dataclass(frozen=True, slots=True)
class EqualSectionOdHarmonization:
    """Equal-section reference for repeated adaptive OD assays.

    This is a technical assay-scale normalization. It cannot create spatial
    expression patterns and preserves the within-section ordering of every
    nonnegative measurement. It must not be interpreted as recovery of an
    absolute OD scale across biologically different specimens.
    """

    sections: tuple[CohortSectionOdHarmonization, ...]
    quantiles: np.ndarray
    reference_quantiles: np.ndarray
    minimum_cells: int
    scale_bounds: tuple[float, float]
    fingerprint: str
    reference_cohorts: tuple[str, ...] | None = None

    def for_section(self, cohort: str, section: str) -> CohortSectionOdHarmonization:
        try:
            return next(
                row
                for row in self.sections
                if row.cohort == cohort and row.section == section
            )
        except StopIteration as error:
            raise KeyError(f"no OD harmonization for {cohort}/{section}") from error

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": 2,
            "method": "equal-section-adaptive-quantile-v2",
            "quantiles": np.asarray(self.quantiles).tolist(),
            "reference_quantiles": np.asarray(self.reference_quantiles).tolist(),
            "minimum_cells": self.minimum_cells,
            "scale_bounds": list(self.scale_bounds),
            "sections": [row.as_dict() for row in self.sections],
            "fingerprint": self.fingerprint,
        }
        if self.reference_cohorts is not None:
            payload["reference_cohorts"] = list(self.reference_cohorts)
        return payload


def fit_equal_section_od_harmonization(
    cohort_ids: np.ndarray,
    section_ids: np.ndarray,
    measured_od: np.ndarray,
    support: np.ndarray,
    *,
    minimum_cells: int = 128,
    scale_bounds: tuple[float, float] = (0.5, 2.5),
    quantiles: tuple[float, ...] = (
        0.0,
        0.01,
        0.05,
        0.1,
        0.25,
        0.5,
        0.75,
        0.9,
        0.95,
        0.99,
        1.0,
    ),
    reference_cohorts: tuple[str, ...] | None = None,
) -> EqualSectionOdHarmonization:
    """Map repeated tissue-only adaptive OD sections to an equal-weight scale.

    A reference quantile curve is the pointwise median of qualifying stain
    sections, with each section contributing once regardless of its cell count.
    ``reference_cohorts`` can freeze that curve to training mice while still
    mapping external holdout sections onto the same assay scale. Each section
    receives one bounded, zero-anchored monotonic mapping. The mapping changes
    only assay intensity, never support or spatial structure.
    """

    cohorts = np.asarray(cohort_ids, dtype=str)
    sections = np.asarray(section_ids, dtype=str)
    values = np.asarray(measured_od, dtype=np.float64)
    supported = np.asarray(support, dtype=bool)
    count = len(cohorts)
    if (
        sections.shape != (count,)
        or values.shape != (count,)
        or supported.shape != (count,)
    ):
        raise ValueError("study OD harmonization inputs do not align")
    if minimum_cells < 4:
        raise ValueError("study OD harmonization requires at least four cells")
    low, high = (float(value) for value in scale_bounds)
    if not 0 < low <= 1 <= high:
        raise ValueError("study OD harmonization scale bounds are invalid")
    probabilities = np.asarray(quantiles, dtype=np.float64)
    if (
        probabilities.ndim != 1
        or len(probabilities) < 3
        or probabilities[0] != 0
        or probabilities[-1] != 1
        or np.any(np.diff(probabilities) <= 0)
    ):
        raise ValueError("study OD harmonization quantiles are invalid")

    selected_reference_cohorts: tuple[str, ...] | None = None
    if reference_cohorts is not None:
        selected_reference_cohorts = tuple(sorted(set(reference_cohorts)))
        available_cohorts = set(cohorts.tolist())
        if (
            not selected_reference_cohorts
            or set(selected_reference_cohorts) - available_cohorts
        ):
            raise ValueError("study OD harmonization reference cohorts are invalid")

    keys = tuple(sorted(set(zip(cohorts.tolist(), sections.tolist(), strict=True))))
    if len(keys) < 2:
        raise ValueError("study OD harmonization requires repeated stain sections")
    distributions: dict[tuple[str, str], np.ndarray] = {}
    curves: dict[tuple[str, str], np.ndarray] = {}
    for cohort, section in keys:
        selected = (
            (cohorts == cohort)
            & (sections == section)
            & supported
            & np.isfinite(values)
            & (values >= 0)
        )
        distribution = values[selected]
        distributions[(cohort, section)] = distribution
        if len(distribution) >= minimum_cells:
            curve = np.maximum.accumulate(np.quantile(distribution, probabilities))
            curve[0] = 0.0
            curves[(cohort, section)] = curve
    if len(curves) < 2:
        raise ValueError(
            "study OD harmonization requires two adequately sampled sections"
        )
    reference_curves = tuple(
        curve
        for (cohort, _section), curve in curves.items()
        if selected_reference_cohorts is None or cohort in selected_reference_cohorts
    )
    if len(reference_curves) < 2:
        raise ValueError(
            "study OD harmonization requires two adequately sampled reference sections"
        )
    reference = np.maximum.accumulate(np.median(np.stack(reference_curves), axis=0))
    reference[0] = 0.0

    calibrations: list[CohortSectionOdHarmonization] = []
    for cohort, section in keys:
        distribution = distributions[(cohort, section)]
        curve = curves.get((cohort, section))
        if curve is None:
            maximum = max(float(distribution.max(initial=1.0)), 1.0)
            calibration = SectionOdCalibration(
                section=section,
                source_knots=np.asarray([0.0, maximum], dtype=np.float32),
                reference_knots=np.asarray([0.0, maximum], dtype=np.float32),
                anchor_count=int(len(distribution)),
                status="identity",
                scale=1.0,
            )
        else:
            calibration = _quantile_section_harmonization(
                section,
                curve,
                reference,
                anchor_count=len(distribution),
                scale_bounds=(low, high),
            )
        calibrations.append(CohortSectionOdHarmonization(cohort, calibration))

    portable_probabilities = probabilities.astype(np.float32)
    portable_reference = reference.astype(np.float32)
    core = {
        "schema_version": 2,
        "method": "equal-section-adaptive-quantile-v2",
        "quantiles": portable_probabilities.tolist(),
        "reference_quantiles": portable_reference.tolist(),
        "minimum_cells": int(minimum_cells),
        "scale_bounds": [low, high],
        "sections": [row.as_dict() for row in calibrations],
    }
    if selected_reference_cohorts is not None:
        core["reference_cohorts"] = list(selected_reference_cohorts)
    fingerprint = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return EqualSectionOdHarmonization(
        tuple(calibrations),
        portable_probabilities,
        portable_reference,
        int(minimum_cells),
        (low, high),
        fingerprint,
        selected_reference_cohorts,
    )


def _quantile_section_harmonization(
    section: str,
    source_curve: np.ndarray,
    reference_curve: np.ndarray,
    *,
    anchor_count: int,
    scale_bounds: tuple[float, float],
) -> SectionOdCalibration:
    """Build a bounded, strictly sourced, zero-anchored quantile mapping."""

    source = np.asarray(source_curve, dtype=np.float64)
    target = np.asarray(reference_curve, dtype=np.float64)
    positive = source > 1e-8
    source, target = source[positive], target[positive]
    keep = np.r_[True, np.diff(source) > 1e-8] if len(source) else np.asarray([])
    source, target = source[keep], target[keep]
    if not len(source):
        return SectionOdCalibration(
            section,
            np.asarray([0.0, 1.0], dtype=np.float32),
            np.asarray([0.0, 1.0], dtype=np.float32),
            anchor_count,
            "identity",
            1.0,
        )
    target = np.clip(
        np.maximum.accumulate(target),
        source * scale_bounds[0],
        source * scale_bounds[1],
    )
    target = np.maximum.accumulate(target)
    ratios = target / source
    scale = float(np.median(ratios[np.isfinite(ratios)]))
    return SectionOdCalibration(
        section,
        np.r_[0.0, source].astype(np.float32),
        np.r_[0.0, target].astype(np.float32),
        anchor_count,
        "accepted",
        scale,
    )
