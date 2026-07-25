"""Constrained, reviewable ordering of serial tissue sections."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from histopia._atomic import write_json_atomic


@dataclass(frozen=True, slots=True)
class CavityContinuitySummary:
    """Describe substantial internal-cavity continuity along a proposed stack."""

    blocks: tuple[tuple[int, int], ...]
    weak_threshold: float
    strong_threshold: float
    bridge_gap: int

    @property
    def review_recommended(self) -> bool:
        """Return whether substantial cavities form multiple separated blocks."""

        return len(self.blocks) > 1

    def to_json_dict(self) -> dict[str, object]:
        """Return one-based block bounds suitable for review metadata."""

        return {
            "blocks": [
                {"start_order": start, "end_order": end} for start, end in self.blocks
            ],
            "block_count": len(self.blocks),
            "review_recommended": self.review_recommended,
            "weak_threshold": self.weak_threshold,
            "strong_threshold": self.strong_threshold,
            "bridge_gap": self.bridge_gap,
        }


@dataclass(frozen=True, slots=True)
class PhysicalAreaContinuitySummary:
    """Describe whether calibrated tissue area changes smoothly through a stack."""

    available: bool
    trend: str | None
    normalized_rmse: float | None
    max_residual_fraction: float | None
    max_adjacent_relative_change: float | None
    max_adjacent_orders: tuple[int, int] | None
    residual_threshold: float
    adjacent_threshold: float

    @property
    def review_recommended(self) -> bool:
        """Return whether the area trajectory has a strong reversal or jump."""

        return bool(
            self.available
            and (
                (
                    self.max_residual_fraction is not None
                    and self.max_residual_fraction >= self.residual_threshold
                )
                or (
                    self.max_adjacent_relative_change is not None
                    and self.max_adjacent_relative_change >= self.adjacent_threshold
                )
            )
        )

    def to_json_dict(self) -> dict[str, object]:
        """Return calibrated area-continuity diagnostics for a review manifest."""

        return {
            "available": self.available,
            "trend": self.trend,
            "normalized_rmse": self.normalized_rmse,
            "max_residual_fraction": self.max_residual_fraction,
            "max_adjacent_relative_change": self.max_adjacent_relative_change,
            "max_adjacent_orders": (
                {
                    "from_order": self.max_adjacent_orders[0],
                    "to_order": self.max_adjacent_orders[1],
                }
                if self.max_adjacent_orders is not None
                else None
            ),
            "review_recommended": self.review_recommended,
            "residual_threshold": self.residual_threshold,
            "adjacent_threshold": self.adjacent_threshold,
        }


@dataclass(frozen=True, slots=True)
class SectionOrderProposal:
    """A deterministic proposal that preserves explicitly anchored slots."""

    slides: tuple[str, ...]
    fixed_positions: dict[str, int]
    fingerprint: str
    objective: float
    runner_up_objective: float | None = None
    adjacent_distances: tuple[float, ...] = ()
    physical_areas_um2: dict[str, float | None] | None = None
    input_fingerprints: dict[str, str] | None = None
    orientation_quarter_turns: dict[str, int] | None = None
    cavity_fractions: dict[str, float] | None = None

    def to_json_dict(self, *, approved: bool = False) -> dict[str, object]:
        cavity_summary = summarize_cavity_continuity(
            self.slides, self.cavity_fractions or {}
        )
        area_summary = summarize_physical_area_continuity(
            self.slides, self.physical_areas_um2
        )
        return {
            "schema_version": 3,
            "algorithm": "anchored-morphology-v3",
            "approved": approved,
            "fingerprint": self.fingerprint,
            "objective": self.objective,
            "runner_up_objective": self.runner_up_objective,
            "confidence_margin": (
                self.runner_up_objective - self.objective
                if self.runner_up_objective is not None
                else None
            ),
            "fixed_positions": self.fixed_positions,
            "input_fingerprints": self.input_fingerprints or {},
            "physically_calibrated": bool(self.physical_areas_um2)
            and all(area is not None for area in self.physical_areas_um2.values()),
            "physical_area_continuity": area_summary.to_json_dict(),
            "cavity_continuity": cavity_summary.to_json_dict(),
            "slides": [
                {
                    "order": index + 1,
                    "slide": slide,
                    "fixed": self.fixed_positions.get(slide) == index + 1,
                    "distance_from_previous": (
                        self.adjacent_distances[index - 1] if index else None
                    ),
                    "physical_tissue_area_um2": (
                        self.physical_areas_um2.get(slide)
                        if self.physical_areas_um2 is not None
                        else None
                    ),
                    "quarter_turns_ccw": (self.orientation_quarter_turns or {}).get(
                        slide, 0
                    ),
                    "largest_internal_cavity_fraction": (
                        self.cavity_fractions.get(slide)
                        if self.cavity_fractions is not None
                        else None
                    ),
                }
                for index, slide in enumerate(self.slides)
            ],
        }


def propose_anchored_order(
    slide_names: tuple[str, ...],
    distances: np.ndarray,
    fixed_positions: dict[str, int],
    *,
    beam_width: int = 4096,
    physical_areas_um2: dict[str, float | None] | None = None,
    input_fingerprints: dict[str, str] | None = None,
    orientation_quarter_turns: dict[str, int] | None = None,
    cavity_fractions: dict[str, float] | None = None,
) -> SectionOrderProposal:
    """Optimize morphology continuity without moving fixed sequence slots."""

    count = len(slide_names)
    matrix = np.asarray(distances, dtype=float)
    if matrix.shape != (count, count):
        raise ValueError("distance matrix shape does not match slide count")
    if not np.allclose(matrix, matrix.T) or np.any(matrix < 0):
        raise ValueError("distance matrix must be symmetric and non-negative")
    unknown = set(fixed_positions) - set(slide_names)
    if unknown:
        raise ValueError(f"fixed positions contain unknown slides: {sorted(unknown)}")
    positions = list(fixed_positions.values())
    if any(position < 1 or position > count for position in positions):
        raise ValueError("fixed positions must be within the slide sequence")
    if len(positions) != len(set(positions)):
        raise ValueError("fixed positions must be unique")
    if beam_width <= 0:
        raise ValueError("beam_width must be positive")
    if input_fingerprints is not None:
        missing = set(slide_names) - set(input_fingerprints)
        extra = set(input_fingerprints) - set(slide_names)
        if missing or extra:
            raise ValueError(
                "input fingerprints must exactly match slides "
                f"(missing={sorted(missing)}, extra={sorted(extra)})"
            )
    if cavity_fractions is not None:
        missing = set(slide_names) - set(cavity_fractions)
        extra = set(cavity_fractions) - set(slide_names)
        if missing or extra:
            raise ValueError(
                "cavity fractions must exactly match slides "
                f"(missing={sorted(missing)}, extra={sorted(extra)})"
            )
        if any(
            not np.isfinite(value) or value < 0 or value > 1
            for value in cavity_fractions.values()
        ):
            raise ValueError("cavity fractions must be between zero and one")

    index = {name: offset for offset, name in enumerate(slide_names)}
    fixed_by_position = {position: name for name, position in fixed_positions.items()}
    free = tuple(sorted(set(slide_names) - set(fixed_positions)))
    beam: list[tuple[float, tuple[str, ...], tuple[str, ...]]] = [(0.0, (), free)]
    for position in range(1, count + 1):
        expanded: list[tuple[float, tuple[str, ...], tuple[str, ...]]] = []
        for cost, sequence, remaining in beam:
            candidates = (
                (fixed_by_position[position],)
                if position in fixed_by_position
                else remaining
            )
            for candidate in candidates:
                edge = (
                    matrix[index[sequence[-1]], index[candidate]] if sequence else 0.0
                )
                next_remaining = (
                    remaining
                    if position in fixed_by_position
                    else tuple(item for item in remaining if item != candidate)
                )
                expanded.append(
                    (cost + float(edge), (*sequence, candidate), next_remaining)
                )
        expanded.sort(key=lambda item: (item[0], item[1]))
        beam = expanded[:beam_width]

    sequence = list(beam[0][1])
    movable = [offset for offset in range(count) if offset + 1 not in positions]
    improved = True
    while improved:
        improved = False
        baseline = _path_objective(sequence, matrix, index)
        for first_index, first in enumerate(movable):
            for second in movable[first_index + 1 :]:
                sequence[first], sequence[second] = sequence[second], sequence[first]
                candidate_cost = _path_objective(sequence, matrix, index)
                if candidate_cost + 1e-12 < baseline:
                    baseline = candidate_cost
                    improved = True
                else:
                    sequence[first], sequence[second] = (
                        sequence[second],
                        sequence[first],
                    )

    ordered = tuple(sequence)
    alternative_costs = sorted(
        cost for cost, candidate, _ in beam if candidate != ordered
    )
    runner_up = alternative_costs[0] if alternative_costs else None
    return _build_section_order_proposal(
        ordered,
        slide_names,
        matrix,
        fixed_positions,
        runner_up_objective=runner_up,
        physical_areas_um2=physical_areas_um2,
        input_fingerprints=input_fingerprints,
        orientation_quarter_turns=orientation_quarter_turns,
        cavity_fractions=cavity_fractions,
    )


def _build_section_order_proposal(
    ordered: tuple[str, ...],
    slide_names: tuple[str, ...],
    distances: np.ndarray,
    fixed_positions: dict[str, int],
    *,
    runner_up_objective: float | None,
    physical_areas_um2: dict[str, float | None] | None,
    input_fingerprints: dict[str, str] | None,
    orientation_quarter_turns: dict[str, int] | None,
    cavity_fractions: dict[str, float] | None,
) -> SectionOrderProposal:
    """Rebuild derived proposal fields from one exact cached slide order."""

    if len(ordered) != len(slide_names) or set(ordered) != set(slide_names):
        raise ValueError("cached section order must be an exact slide permutation")
    for slide, position in fixed_positions.items():
        if (
            slide not in ordered
            or position < 1
            or position > len(ordered)
            or ordered[position - 1] != slide
        ):
            raise ValueError("cached section order violates a fixed position")
    matrix = np.asarray(distances, dtype=float)
    if matrix.shape != (len(slide_names), len(slide_names)):
        raise ValueError("distance matrix shape does not match slide count")
    index = {name: offset for offset, name in enumerate(slide_names)}
    objective = _path_objective(list(ordered), matrix, index)
    if runner_up_objective is not None and (
        not np.isfinite(runner_up_objective) or runner_up_objective + 1e-12 < objective
    ):
        raise ValueError("cached runner-up objective is invalid")
    fingerprint = _fingerprint(
        ordered,
        fixed_positions,
        matrix,
        physical_areas_um2=physical_areas_um2,
        input_fingerprints=input_fingerprints,
        orientation_quarter_turns=orientation_quarter_turns,
        cavity_fractions=cavity_fractions,
    )
    adjacent_distances = tuple(
        float(matrix[index[first], index[second]])
        for first, second in zip(ordered, ordered[1:], strict=False)
    )
    return SectionOrderProposal(
        slides=ordered,
        fixed_positions=dict(fixed_positions),
        fingerprint=fingerprint,
        objective=objective,
        runner_up_objective=runner_up_objective,
        adjacent_distances=adjacent_distances,
        physical_areas_um2=(
            dict(physical_areas_um2) if physical_areas_um2 is not None else None
        ),
        input_fingerprints=(
            dict(input_fingerprints) if input_fingerprints is not None else None
        ),
        orientation_quarter_turns=(
            dict(orientation_quarter_turns)
            if orientation_quarter_turns is not None
            else None
        ),
        cavity_fractions=(
            dict(cavity_fractions) if cavity_fractions is not None else None
        ),
    )


def summarize_cavity_continuity(
    slides: tuple[str, ...],
    cavity_fractions: dict[str, float],
    *,
    weak_threshold: float = 0.015,
    strong_threshold: float = 0.04,
    bridge_gap: int = 1,
) -> CavityContinuitySummary:
    """Find graded cavity blocks while tolerating borderline single-slide gaps."""

    if not cavity_fractions:
        return CavityContinuitySummary((), weak_threshold, strong_threshold, bridge_gap)
    if set(slides) != set(cavity_fractions):
        raise ValueError("cavity fractions must exactly match slides")
    if not 0 <= weak_threshold <= strong_threshold <= 1:
        raise ValueError("cavity thresholds must satisfy 0 <= weak <= strong <= 1")
    if bridge_gap < 0:
        raise ValueError("bridge_gap must be non-negative")
    if any(
        not np.isfinite(value) or value < 0 or value > 1
        for value in cavity_fractions.values()
    ):
        raise ValueError("cavity fractions must be between zero and one")

    values = [cavity_fractions[slide] for slide in slides]
    has_strong_cavity = any(value >= strong_threshold for value in values)
    active = [value >= weak_threshold for value in values]
    active_indices = [index for index, value in enumerate(active) if value]
    for first, second in zip(active_indices, active_indices[1:], strict=False):
        if second - first - 1 <= bridge_gap:
            active[first : second + 1] = [True] * (second - first + 1)

    blocks: list[tuple[int, int]] = []
    start: int | None = None
    for index, is_active in enumerate((*active, False)):
        if is_active and start is None:
            start = index
        elif not is_active and start is not None:
            if not has_strong_cavity or any(
                value >= strong_threshold for value in values[start:index]
            ):
                blocks.append((start + 1, index))
            start = None
    return CavityContinuitySummary(
        tuple(blocks), weak_threshold, strong_threshold, bridge_gap
    )


def summarize_physical_area_continuity(
    slides: tuple[str, ...],
    physical_areas_um2: dict[str, float | None] | None,
    *,
    residual_threshold: float = 0.15,
    adjacent_threshold: float = 0.20,
) -> PhysicalAreaContinuitySummary:
    """Measure strong reversals and jumps in calibrated tissue area.

    The best monotonic isotonic fit is used only as a review diagnostic. It does
    not alter the proposed order, because biological area can change
    non-monotonically and similarity ordering is not a measured physical z-axis.
    """

    if (
        not np.isfinite(residual_threshold)
        or not np.isfinite(adjacent_threshold)
        or residual_threshold < 0
        or adjacent_threshold < 0
    ):
        raise ValueError("area-continuity thresholds must be finite and non-negative")
    unavailable = PhysicalAreaContinuitySummary(
        available=False,
        trend=None,
        normalized_rmse=None,
        max_residual_fraction=None,
        max_adjacent_relative_change=None,
        max_adjacent_orders=None,
        residual_threshold=residual_threshold,
        adjacent_threshold=adjacent_threshold,
    )
    if not physical_areas_um2:
        return unavailable
    if set(slides) != set(physical_areas_um2):
        raise ValueError("physical areas must exactly match slides")
    values = [physical_areas_um2[slide] for slide in slides]
    if any(value is None or not np.isfinite(value) or value <= 0 for value in values):
        return unavailable

    areas = np.asarray(values, dtype=float)
    normalized = areas / float(np.median(areas))
    increasing = _isotonic_non_decreasing(normalized)
    decreasing = -_isotonic_non_decreasing(-normalized)
    increasing_rmse = float(np.sqrt(np.mean(np.square(normalized - increasing))))
    decreasing_rmse = float(np.sqrt(np.mean(np.square(normalized - decreasing))))
    if increasing_rmse < decreasing_rmse or (
        np.isclose(increasing_rmse, decreasing_rmse) and areas[-1] >= areas[0]
    ):
        trend = "nondecreasing"
        fit = increasing
        normalized_rmse = increasing_rmse
    else:
        trend = "nonincreasing"
        fit = decreasing
        normalized_rmse = decreasing_rmse

    residual = np.abs(normalized - fit)
    if len(areas) >= 2:
        adjacent_changes = np.abs(np.diff(areas)) / (0.5 * (areas[:-1] + areas[1:]))
        adjacent_index = int(np.argmax(adjacent_changes))
        max_adjacent_change = float(adjacent_changes[adjacent_index])
        max_adjacent_orders = (adjacent_index + 1, adjacent_index + 2)
    else:
        max_adjacent_change = 0.0
        max_adjacent_orders = None
    return PhysicalAreaContinuitySummary(
        available=True,
        trend=trend,
        normalized_rmse=normalized_rmse,
        max_residual_fraction=float(np.max(residual)),
        max_adjacent_relative_change=max_adjacent_change,
        max_adjacent_orders=max_adjacent_orders,
        residual_threshold=residual_threshold,
        adjacent_threshold=adjacent_threshold,
    )


def _isotonic_non_decreasing(values: np.ndarray) -> np.ndarray:
    """Return an equal-weight non-decreasing fit using pooled adjacent violators."""

    block_values: list[float] = []
    block_weights: list[int] = []
    block_starts: list[int] = []
    block_ends: list[int] = []
    for index, value in enumerate(np.asarray(values, dtype=float)):
        block_values.append(float(value))
        block_weights.append(1)
        block_starts.append(index)
        block_ends.append(index + 1)
        while len(block_values) >= 2 and block_values[-2] > block_values[-1]:
            weight = block_weights[-2] + block_weights[-1]
            pooled = (
                block_values[-2] * block_weights[-2]
                + block_values[-1] * block_weights[-1]
            ) / weight
            block_values[-2:] = [pooled]
            block_weights[-2:] = [weight]
            block_ends[-2:] = [block_ends[-1]]
            block_starts.pop()

    fitted = np.empty(len(values), dtype=float)
    for value, start, end in zip(block_values, block_starts, block_ends, strict=True):
        fitted[start:end] = value
    return fitted


def write_order_proposal(path: Path, proposal: SectionOrderProposal) -> None:
    """Write a proposal while retaining approval only for the same fingerprint."""

    approved = False
    review_metadata: dict[str, object] = {}
    if path.exists():
        previous = json.loads(path.read_text())
        approved = bool(previous.get("approved")) and (
            previous.get("fingerprint") == proposal.fingerprint
        )
        if approved:
            review_metadata = {
                key: previous[key]
                for key in ("reviewer", "reviewed_at", "notes")
                if key in previous
            }
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = proposal.to_json_dict(approved=approved)
    payload.update(review_metadata)
    write_json_atomic(path, payload)


def order_is_approved(path: Path, fingerprint: str) -> bool:
    """Return whether a human approved the exact current proposal."""

    if not path.exists():
        return False
    payload = json.loads(path.read_text())
    return bool(payload.get("approved")) and payload.get("fingerprint") == fingerprint


def _path_objective(
    sequence: list[str | None], matrix: np.ndarray, index: dict[str, int]
) -> float:
    names = [value for value in sequence if value is not None]
    return float(
        sum(
            matrix[index[first], index[second]]
            for first, second in zip(names, names[1:], strict=False)
        )
    )


def _fingerprint(
    slides: tuple[str, ...],
    fixed_positions: dict[str, int],
    matrix: np.ndarray,
    *,
    physical_areas_um2: dict[str, float | None] | None,
    input_fingerprints: dict[str, str] | None,
    orientation_quarter_turns: dict[str, int] | None,
    cavity_fractions: dict[str, float] | None,
) -> str:
    payload = {
        "algorithm": "anchored-morphology-v3",
        "slides": slides,
        "fixed_positions": sorted(fixed_positions.items()),
        "distances": np.round(matrix, 8).tolist(),
        "physical_areas_um2": (
            sorted(physical_areas_um2.items()) if physical_areas_um2 else []
        ),
        "input_fingerprints": (
            sorted(input_fingerprints.items()) if input_fingerprints else []
        ),
        "orientation_quarter_turns": sorted((orientation_quarter_turns or {}).items()),
        "cavity_fractions": sorted((cavity_fractions or {}).items()),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
