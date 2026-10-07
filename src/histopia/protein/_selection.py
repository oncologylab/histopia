"""Conservative promotion gates for protein-transfer candidates."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProteinPromotionDecision:
    """Machine-readable decision for an atomic review-site promotion."""

    accepted: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ProteinCandidateScore:
    """Comparable held-out score for one frozen architecture candidate."""

    candidate: str
    eligible: bool
    reasons: tuple[str, ...]
    held_out_cohorts: tuple[str, ...]
    median_spearman_64um: float | None
    minimum_spearman_64um: float | None
    mean_mae_improvement: float | None
    worst_absolute_mean_bias_fraction: float | None


@dataclass(frozen=True, slots=True)
class ProteinCandidateSelection:
    """Deterministic development-only architecture selection."""

    selected: str | None
    candidates: tuple[ProteinCandidateScore, ...]


def evaluate_protein_promotion(
    metrics: dict[str, object],
    *,
    minimum_spearman: float = 0.40,
    minimum_spearman_64um: float = 0.55,
    maximum_mae: float = 0.535,
    minimum_fold_improvement: float = -0.01,
    maximum_mean_bias_fraction: float = 0.10,
    maximum_median_bias_fraction: float = 0.15,
) -> ProteinPromotionDecision:
    """Require held-out accuracy, baseline parity, and calibrated OD bias limits."""

    reasons: list[str] = []
    for key, limit in (
        ("spearman", minimum_spearman),
        ("spearman_64um", minimum_spearman_64um),
    ):
        value = metrics.get(key)
        if not isinstance(value, (int, float)) or float(value) < limit:
            reasons.append(f"{key} is below {limit:.3f}")
    mae = metrics.get("mae")
    if not isinstance(mae, (int, float)) or float(mae) > maximum_mae:
        reasons.append(f"mae exceeds {maximum_mae:.3f}")
    folds = metrics.get("folds")
    if not isinstance(folds, list) or not folds:
        reasons.append("per-section held-out folds are missing")
    else:
        for fold in folds:
            if not isinstance(fold, dict):
                reasons.append("held-out fold metadata is malformed")
                continue
            improvement = fold.get("mae_improvement")
            if (
                not isinstance(improvement, (int, float))
                or float(improvement) < minimum_fold_improvement
            ):
                reasons.append(
                    f"fold {fold.get('held_out', '?')} is more than 1% worse "
                    "than its semantic baseline"
                )
            for key, limit in (
                ("mean_bias_fraction", maximum_mean_bias_fraction),
                ("median_bias_fraction", maximum_median_bias_fraction),
            ):
                value = fold.get(key)
                if not isinstance(value, (int, float)) or abs(float(value)) > limit:
                    reasons.append(
                        f"fold {fold.get('held_out', '?')} absolute {key} "
                        f"exceeds {limit:.3f}"
                    )
    for key, limit in (
        ("mean_bias_fraction", maximum_mean_bias_fraction),
        ("median_bias_fraction", maximum_median_bias_fraction),
    ):
        value = metrics.get(key)
        if not isinstance(value, (int, float)) or abs(float(value)) > limit:
            reasons.append(f"absolute {key} exceeds {limit:.3f}")
    return ProteinPromotionDecision(not reasons, tuple(reasons))


def select_protein_architecture_candidate(
    candidates: Mapping[str, Mapping[str, object]],
    *,
    expected_holdouts: Sequence[str],
    maximum_absolute_mean_bias_fraction: float = 0.25,
) -> ProteinCandidateSelection:
    """Select a frozen architecture using identical leave-one-mouse-out folds.

    Selection is intentionally restricted to development folds. Every candidate
    must cover the exact expected mouse set, improve over the semantic baseline
    on every mouse, have positive mean MAE improvement, and remain within the
    configured OD-bias guardrail. Eligible candidates are ranked by median and
    then minimum 64-µm Spearman, mean MAE improvement, worst absolute bias, and
    finally their stable identifier.
    """

    if isinstance(expected_holdouts, (str, bytes)):
        raise ValueError("expected_holdouts must be a sequence of identifiers")
    holdouts = tuple(str(value) for value in expected_holdouts)
    if not holdouts or any(not value for value in holdouts):
        raise ValueError("expected_holdouts must contain non-empty identifiers")
    if len(set(holdouts)) != len(holdouts):
        raise ValueError("expected_holdouts must be unique")
    if (
        not _finite_number(maximum_absolute_mean_bias_fraction)
        or float(maximum_absolute_mean_bias_fraction) < 0
    ):
        raise ValueError("maximum_absolute_mean_bias_fraction must be finite")
    if not isinstance(candidates, Mapping) or not candidates:
        raise ValueError("at least one architecture candidate is required")
    if any(not isinstance(name, str) or not name for name in candidates):
        raise ValueError("candidate identifiers must be non-empty strings")

    scores = tuple(
        _score_architecture_candidate(
            name,
            metrics,
            expected_holdouts=holdouts,
            maximum_absolute_mean_bias_fraction=float(
                maximum_absolute_mean_bias_fraction
            ),
        )
        for name, metrics in sorted(candidates.items())
    )
    eligible = [score for score in scores if score.eligible]
    eligible.sort(
        key=lambda score: (
            -float(score.median_spearman_64um),
            -float(score.minimum_spearman_64um),
            -float(score.mean_mae_improvement),
            float(score.worst_absolute_mean_bias_fraction),
            score.candidate,
        )
    )
    return ProteinCandidateSelection(
        selected=eligible[0].candidate if eligible else None,
        candidates=scores,
    )


def _score_architecture_candidate(
    candidate: object,
    metrics: object,
    *,
    expected_holdouts: tuple[str, ...],
    maximum_absolute_mean_bias_fraction: float,
) -> ProteinCandidateScore:
    name = str(candidate)
    reasons: list[str] = []
    if not isinstance(candidate, str) or not candidate:
        reasons.append("candidate identifier is invalid")
    if not isinstance(metrics, Mapping):
        reasons.append("metrics are malformed")
        return _invalid_candidate_score(name, reasons)
    if metrics.get("evaluation") != "leave_one_mouse_out":
        reasons.append("evaluation is not leave-one-mouse-out")
    raw_folds = metrics.get("folds")
    if not isinstance(raw_folds, list):
        reasons.append("held-out folds are missing")
        return _invalid_candidate_score(name, reasons)

    rows: dict[str, tuple[float, float, float]] = {}
    for fold in raw_folds:
        if not isinstance(fold, Mapping):
            reasons.append("held-out fold metadata is malformed")
            continue
        held_out = fold.get("held_out")
        if not isinstance(held_out, str) or not held_out:
            reasons.append("held-out cohort identifier is invalid")
            continue
        if held_out in rows:
            reasons.append(f"held-out cohort {held_out} is duplicated")
            continue
        values = tuple(
            fold.get(key)
            for key in ("spearman_64um", "mae_improvement", "mean_bias_fraction")
        )
        if not all(_finite_number(value) for value in values):
            reasons.append(f"held-out cohort {held_out} has non-finite metrics")
            continue
        spearman, improvement, bias = (float(value) for value in values)
        rows[held_out] = (spearman, improvement, bias)
        if fold.get("beats_semantic_baseline") is not True or improvement <= 0:
            reasons.append(
                f"held-out cohort {held_out} does not beat the semantic baseline"
            )
        if abs(bias) > maximum_absolute_mean_bias_fraction:
            reasons.append(
                f"held-out cohort {held_out} absolute mean bias exceeds "
                f"{maximum_absolute_mean_bias_fraction:.3f}"
            )

    observed = tuple(sorted(rows))
    expected = tuple(sorted(expected_holdouts))
    if observed != expected:
        reasons.append("held-out cohort coverage differs from the frozen design")
    if not rows:
        return _invalid_candidate_score(name, reasons)

    spearman = tuple(row[0] for row in rows.values())
    improvement = tuple(row[1] for row in rows.values())
    bias = tuple(abs(row[2]) for row in rows.values())
    median_spearman = _median(spearman)
    mean_improvement = sum(improvement) / len(improvement)
    if mean_improvement <= 0:
        reasons.append("mean MAE improvement is not positive")
    return ProteinCandidateScore(
        candidate=name,
        eligible=not reasons,
        reasons=tuple(dict.fromkeys(reasons)),
        held_out_cohorts=observed,
        median_spearman_64um=median_spearman,
        minimum_spearman_64um=min(spearman),
        mean_mae_improvement=mean_improvement,
        worst_absolute_mean_bias_fraction=max(bias),
    )


def _invalid_candidate_score(
    candidate: str,
    reasons: list[str],
) -> ProteinCandidateScore:
    return ProteinCandidateScore(
        candidate=candidate,
        eligible=False,
        reasons=tuple(dict.fromkeys(reasons)),
        held_out_cohorts=(),
        median_spearman_64um=None,
        minimum_spearman_64um=None,
        mean_mae_improvement=None,
        worst_absolute_mean_bias_fraction=None,
    )


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _median(values: tuple[float, ...]) -> float:
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2
