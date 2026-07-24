"""Exact, corruption-tolerant cache for expensive morphology distances."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np


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


def _matrix_checksum(matrix: np.ndarray) -> str:
    canonical = np.ascontiguousarray(matrix, dtype="<f8")
    return hashlib.sha256(canonical.tobytes()).hexdigest()
