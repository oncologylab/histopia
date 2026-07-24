"""Constrained, reviewable ordering of serial tissue sections."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


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
    objective = _path_objective(list(ordered), matrix, index)
    alternative_costs = sorted(
        cost for cost, candidate, _ in beam if candidate != ordered
    )
    runner_up = alternative_costs[0] if alternative_costs else None
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
        runner_up_objective=runner_up,
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


def write_order_proposal(path: Path, proposal: SectionOrderProposal) -> None:
    """Write a proposal while retaining approval only for the same fingerprint."""

    approved = False
    if path.exists():
        payload = json.loads(path.read_text())
        approved = bool(payload.get("approved")) and (
            payload.get("fingerprint") == proposal.fingerprint
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(proposal.to_json_dict(approved=approved), indent=2)
    path.write_text(payload + "\n")


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
