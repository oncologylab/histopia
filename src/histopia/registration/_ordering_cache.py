"""Exact, corruption-tolerant cache for expensive morphology distances."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from histopia._atomic import write_json_atomic

_PROPOSAL_CACHE_SCHEMA = "histopia-section-order-proposal-v1"


def ordering_cache_fingerprint(
    slide_names: tuple[str, ...],
    input_fingerprints: dict[str, str],
    settings: dict[str, object],
) -> str:
    """Fingerprint every input that can affect pairwise ordering distances."""

    payload = {
        "schema_version": 1,
        "algorithm": "section-distance-v1",
        "slides": list(slide_names),
        "inputs": [[name, input_fingerprints[name]] for name in slide_names],
        "settings": settings,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_ordering_distance_cache(
    path: Path | str,
    *,
    expected_fingerprint: str,
    expected_size: int,
) -> np.ndarray | None:
    """Load a valid exact cache, returning ``None`` for stale/corrupt data."""

    try:
        with np.load(Path(path), allow_pickle=False) as data:
            fingerprint = str(data["fingerprint"].item())
            matrix = np.asarray(data["distances"], dtype=np.float64)
            checksum = str(data["matrix_sha256"].item())
    except (KeyError, OSError, TypeError, ValueError):
        return None
    if fingerprint != expected_fingerprint:
        return None
    if matrix.shape != (expected_size, expected_size):
        return None
    if (
        not np.all(np.isfinite(matrix))
        or np.any(matrix < 0)
        or not np.allclose(matrix, matrix.T)
        or not np.allclose(np.diag(matrix), 0)
    ):
        return None
    if checksum != _matrix_checksum(matrix):
        return None
    return matrix


def write_ordering_distance_cache(
    path: Path | str,
    distances: np.ndarray,
    *,
    fingerprint: str,
) -> Path:
    """Atomically write a distance matrix after validating its invariants."""

    path = Path(path)
    matrix = np.ascontiguousarray(distances, dtype=np.float64)
    size = matrix.shape[0] if matrix.ndim == 2 else -1
    loadable = (
        matrix.shape == (size, size)
        and np.all(np.isfinite(matrix))
        and np.all(matrix >= 0)
        and np.allclose(matrix, matrix.T)
        and np.allclose(np.diag(matrix), 0)
    )
    if not loadable:
        raise ValueError("ordering distance matrix must be finite and symmetric")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                fingerprint=np.asarray(fingerprint),
                distances=matrix,
                matrix_sha256=np.asarray(_matrix_checksum(matrix)),
            )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def ordering_proposal_cache_fingerprint(
    distance_fingerprint: str,
    distances: np.ndarray,
    fixed_positions: dict[str, int],
    *,
    physical_areas_um2: dict[str, float | None],
    input_fingerprints: dict[str, str],
    orientation_quarter_turns: dict[str, int],
    cavity_fractions: dict[str, float],
    beam_width: int = 4096,
) -> str:
    """Fingerprint every input to the deterministic anchored-order search."""

    payload = {
        "schema": _PROPOSAL_CACHE_SCHEMA,
        "algorithm": "anchored-morphology-v3",
        "distance_fingerprint": distance_fingerprint,
        "distance_sha256": _matrix_checksum(np.asarray(distances)),
        "fixed_positions": sorted(fixed_positions.items()),
        "physical_areas_um2": sorted(physical_areas_um2.items()),
        "input_fingerprints": sorted(input_fingerprints.items()),
        "orientation_quarter_turns": sorted(orientation_quarter_turns.items()),
        "cavity_fractions": sorted(cavity_fractions.items()),
        "beam_width": beam_width,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_ordering_proposal_cache(
    path: Path | str,
    *,
    expected_fingerprint: str,
) -> tuple[tuple[str, ...], float | None] | None:
    """Load one exact cached order and runner-up score."""

    cache_path = Path(path)
    if cache_path.is_symlink():
        return None
    try:
        payload = json.loads(cache_path.read_text())
    except (json.JSONDecodeError, OSError, TypeError):
        return None
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != _PROPOSAL_CACHE_SCHEMA
        or payload.get("fingerprint") != expected_fingerprint
    ):
        return None
    proposal = payload.get("proposal")
    checksum = payload.get("proposal_sha256")
    if not isinstance(proposal, dict) or not isinstance(checksum, str):
        return None
    if _proposal_checksum(proposal) != checksum:
        return None
    slides = proposal.get("slides")
    runner_up = proposal.get("runner_up_objective")
    if (
        not isinstance(slides, list)
        or not slides
        or any(not isinstance(slide, str) or not slide for slide in slides)
        or len(slides) != len(set(slides))
        or (
            runner_up is not None
            and (
                not isinstance(runner_up, (int, float))
                or isinstance(runner_up, bool)
                or not math.isfinite(runner_up)
            )
        )
    ):
        return None
    return tuple(slides), float(runner_up) if runner_up is not None else None


def write_ordering_proposal_cache(
    path: Path | str,
    *,
    fingerprint: str,
    slides: tuple[str, ...],
    runner_up_objective: float | None,
) -> Path:
    """Atomically cache one deterministic anchored-order search result."""

    if (
        not slides
        or any(not isinstance(slide, str) or not slide for slide in slides)
        or len(slides) != len(set(slides))
    ):
        raise ValueError("cached section order must contain unique slide names")
    if runner_up_objective is not None and (
        isinstance(runner_up_objective, bool)
        or not isinstance(runner_up_objective, (int, float))
        or not math.isfinite(runner_up_objective)
    ):
        raise ValueError("cached runner-up objective must be finite")
    proposal = {
        "slides": list(slides),
        "runner_up_objective": runner_up_objective,
    }
    return write_json_atomic(
        path,
        {
            "schema": _PROPOSAL_CACHE_SCHEMA,
            "fingerprint": fingerprint,
            "proposal": proposal,
            "proposal_sha256": _proposal_checksum(proposal),
        },
        sort_keys=True,
    )


def _matrix_checksum(matrix: np.ndarray) -> str:
    canonical = np.ascontiguousarray(matrix, dtype="<f8")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _proposal_checksum(proposal: dict[str, object]) -> str:
    encoded = json.dumps(
        proposal,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
