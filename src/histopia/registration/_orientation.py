"""Cohort-consistent orientation of serial-section tissue masks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TypeVar

import numpy as np
from scipy import ndimage as ndi

OrientationKey = TypeVar("OrientationKey", bound=str)


@dataclass(frozen=True, slots=True)
class OrientationDecision:
    """A reviewable quarter-turn selected from dominant tissue morphology."""

    quarter_turns_ccw: int
    score: float
    confidence_margin: float

    @property
    def degrees_ccw(self) -> int:
        return self.quarter_turns_ccw * 90


@dataclass(frozen=True, slots=True)
class GroupOrientation:
    """Orientation decisions tied to an exact set of input masks."""

    anchor: str
    decisions: dict[str, OrientationDecision]
    fingerprint: str

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "anchor": self.anchor,
            "fingerprint": self.fingerprint,
            "slides": {
                key: {
                    "quarter_turns_ccw": decision.quarter_turns_ccw,
                    "degrees_ccw": decision.degrees_ccw,
                    "score": decision.score,
                    "confidence_margin": decision.confidence_margin,
                }
                for key, decision in sorted(self.decisions.items())
            },
        }


def orient_section_group(
    masks: dict[str, np.ndarray],
    *,
    anchor: str | None = None,
    normalized_size: int = 192,
    minimum_confidence_margin: float = 0.05,
) -> GroupOrientation:
    """Match dominant tissue directions using reviewable quarter-turns.

    The anchor remains at zero degrees. When no anchor is supplied, the medoid
    mask under rotation-invariant Dice similarity is used. Source arrays are
    never modified.
    """

    if not masks:
        raise ValueError("at least one mask is required")
    if normalized_size < 32:
        raise ValueError("normalized_size must be at least 32")
    if minimum_confidence_margin < 0:
        raise ValueError("minimum_confidence_margin must be non-negative")
    if anchor is not None and anchor not in masks:
        raise ValueError(f"unknown orientation anchor: {anchor}")
    normalized = {
        key: _normalize_dominant_object(mask, normalized_size)
        for key, mask in masks.items()
    }
    if anchor is None:
        anchor = _rotation_invariant_medoid(normalized)
    reference = normalized[anchor]
    reference_aspect = _main_topology_log_aspect(masks[anchor])
    decisions: dict[str, OrientationDecision] = {}
    for key, mask in normalized.items():
        scores = tuple(_dice(np.rot90(mask, turns), reference) for turns in range(4))
        candidates = (0, 1, 2, 3)
        if key == anchor:
            best_turn = 0
        else:
            candidates = _aspect_compatible_turns(
                _main_topology_log_aspect(masks[key]), reference_aspect
            )
            best_turn = max(candidates, key=lambda turns: (scores[turns], -turns))
        ordered_scores = sorted((scores[turn] for turn in candidates), reverse=True)
        margin = ordered_scores[0] - ordered_scores[1]
        if key != anchor and margin < minimum_confidence_margin:
            best_turn = 0
        decisions[key] = OrientationDecision(
            quarter_turns_ccw=best_turn,
            score=float(scores[best_turn]),
            confidence_margin=float(margin),
        )
    fingerprint = _orientation_fingerprint(
        masks,
        anchor,
        decisions,
        normalized_size,
        minimum_confidence_margin,
    )
    return GroupOrientation(anchor, decisions, fingerprint)


def apply_quarter_turn(array: np.ndarray, quarter_turns_ccw: int) -> np.ndarray:
    """Return an array rotated counterclockwise by a multiple of 90 degrees."""

    return np.rot90(np.asarray(array), int(quarter_turns_ccw) % 4).copy()


def _main_topology_log_aspect(mask: np.ndarray) -> float:
    binary = np.asarray(mask, dtype=bool)
    labels, count = ndi.label(binary)
    if count == 0:
        raise ValueError("orientation masks must contain foreground")
    sizes = np.bincount(labels.ravel())
    largest = int(sizes[1:].max(initial=0))
    keep = sizes >= largest * 0.10
    keep[0] = False
    rows, cols = np.nonzero(keep[labels])
    height = int(rows.max() - rows.min() + 1)
    width = int(cols.max() - cols.min() + 1)
    topology_aspect = float(np.log(width / height))
    if abs(topology_aspect) >= 0.12:
        return topology_aspect
    return float(np.log(binary.shape[1] / binary.shape[0]))


def _aspect_compatible_turns(
    moving_log_aspect: float,
    reference_log_aspect: float,
    *,
    square_tolerance: float = 0.12,
) -> tuple[int, ...]:
    if (
        abs(moving_log_aspect) < square_tolerance
        or abs(reference_log_aspect) < square_tolerance
    ):
        return (0, 1, 2, 3)
    if np.sign(moving_log_aspect) == np.sign(reference_log_aspect):
        return (0, 2)
    return (1, 3)


def _normalize_dominant_object(mask: np.ndarray, size: int) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError("orientation masks must be two-dimensional")
    labels, count = ndi.label(binary)
    if count == 0:
        raise ValueError("orientation masks must contain foreground")
    sizes = np.bincount(labels.ravel())
    largest = int(sizes[1:].max(initial=0))
    keep = sizes >= largest * 0.10
    keep[0] = False
    main_topology = keep[labels]
    rows, cols = np.nonzero(main_topology)
    crop = main_topology[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]
    side = max(crop.shape)
    square = np.zeros((side, side), dtype=bool)
    row_offset = (side - crop.shape[0]) // 2
    col_offset = (side - crop.shape[1]) // 2
    square[
        row_offset : row_offset + crop.shape[0],
        col_offset : col_offset + crop.shape[1],
    ] = crop
    zoom = (size / side, size / side)
    resized = ndi.zoom(square.astype(np.uint8), zoom, order=0)
    output = np.zeros((size, size), dtype=bool)
    height = min(size, resized.shape[0])
    width = min(size, resized.shape[1])
    output[:height, :width] = resized[:height, :width] > 0
    return output


def _rotation_invariant_medoid(masks: dict[str, np.ndarray]) -> str:
    keys = sorted(masks)
    costs: dict[str, float] = {}
    for candidate in keys:
        reference = masks[candidate]
        costs[candidate] = sum(
            1.0
            - max(_dice(np.rot90(masks[peer], turns), reference) for turns in range(4))
            for peer in keys
            if peer != candidate
        )
    return min(keys, key=lambda key: (costs[key], key))


def _dice(first: np.ndarray, second: np.ndarray) -> float:
    denominator = int(np.count_nonzero(first) + np.count_nonzero(second))
    if denominator == 0:
        return 1.0
    return 2.0 * float(np.count_nonzero(first & second)) / denominator


def _orientation_fingerprint(
    masks: dict[str, np.ndarray],
    anchor: str,
    decisions: dict[str, OrientationDecision],
    normalized_size: int,
    minimum_confidence_margin: float,
) -> str:
    digest = hashlib.sha256()
    metadata = {
        "algorithm": "dominant-object-quarter-turn-v1",
        "anchor": anchor,
        "normalized_size": normalized_size,
        "minimum_confidence_margin": minimum_confidence_margin,
        "turns": {
            key: decision.quarter_turns_ccw
            for key, decision in sorted(decisions.items())
        },
    }
    digest.update(json.dumps(metadata, sort_keys=True).encode())
    for key, mask in sorted(masks.items()):
        digest.update(key.encode())
        digest.update(np.ascontiguousarray(mask, dtype=np.uint8).tobytes())
    return digest.hexdigest()
