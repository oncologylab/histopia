"""Constrained, reviewable ordering of serial tissue sections."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


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

    def to_json_dict(self, *, approved: bool = False) -> dict[str, object]:
        return {
            "schema_version": 1,
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
            "physically_calibrated": bool(self.physical_areas_um2) and all(
                area is not None for area in self.physical_areas_um2.values()
            ),
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
    fingerprint = _fingerprint(ordered, fixed_positions, matrix)
    adjacent_distances = tuple(
        float(matrix[index[first], index[second]])
        for first, second in zip(ordered, ordered[1:], strict=False)
    )
    return SectionOrderProposal(
        ordered,
        dict(fixed_positions),
        fingerprint,
        objective,
        runner_up,
        adjacent_distances,
        dict(physical_areas_um2) if physical_areas_um2 is not None else None,
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
    slides: tuple[str, ...], fixed_positions: dict[str, int], matrix: np.ndarray
) -> str:
    payload = {
        "slides": slides,
        "fixed_positions": sorted(fixed_positions.items()),
        "distances": np.round(matrix, 8).tolist(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
