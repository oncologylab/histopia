"""Exact, corruption-tolerant cache for rigid pair estimates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Protocol

import numpy as np

from histopia._atomic import write_json_atomic
from histopia.registration._rigid import RigidTransformResult

_CACHE_SCHEMA = "histopia-rigid-pair-cache-v1"
_ALGORITHM = "rigid-pair-estimate-v1"


class _Digest(Protocol):
    def update(self, value: bytes) -> object: ...


def rigid_crop_fingerprint(
    slide_path: Path | str,
    image: np.ndarray,
    mask: np.ndarray,
    offset_xy: np.ndarray,
    scale: float,
) -> str:
    """Fingerprint every oriented crop value used for one pair estimate."""

    digest = hashlib.sha256(_CACHE_SCHEMA.encode())
    digest.update(str(Path(slide_path).resolve()).encode())
    _update_array_digest(digest, image)
    _update_array_digest(digest, np.asarray(mask, dtype=bool))
    _update_array_digest(digest, np.asarray(offset_xy, dtype="<f8"))
    digest.update(np.asarray(float(scale), dtype="<f8").tobytes())
    return digest.hexdigest()


def rigid_pair_fingerprint(
    fixed_crop_fingerprint: str,
    moving_crop_fingerprint: str,
    settings: dict[str, object],
) -> str:
    """Fingerprint one directed pair and every algorithm setting."""

    payload = {
        "schema": _CACHE_SCHEMA,
        "algorithm": _ALGORITHM,
        "fixed_crop": fixed_crop_fingerprint,
        "moving_crop": moving_crop_fingerprint,
        "settings": settings,
    }
    return _json_digest(payload)


def load_rigid_pair_cache(
    cache_dir: Path | str,
    fingerprint: str,
) -> tuple[RigidTransformResult, RigidTransformResult] | None:
    """Load an exact full/crop transform pair, or reject stale corruption."""

    path = Path(cache_dir) / f"{fingerprint}.json"
    if path.is_symlink() or not path.is_file():
        return None
    try:
        record = json.loads(path.read_text())
        payload = record["payload"]
        if (
            record.get("schema") != _CACHE_SCHEMA
            or record.get("fingerprint") != fingerprint
            or record.get("payload_sha256") != _json_digest(payload)
        ):
            return None
        full = _transform_from_json(payload["full_transform"])
        crop = _transform_from_json(payload["crop_transform"])
    except (
        json.JSONDecodeError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
    ):
        return None
    return full, crop


def write_rigid_pair_cache(
    cache_dir: Path | str,
    fingerprint: str,
    full_transform: RigidTransformResult,
    crop_transform: RigidTransformResult,
) -> Path:
    """Atomically write one validated directed pair estimate."""

    payload = {
        "full_transform": _validated_transform_dict(full_transform),
        "crop_transform": _validated_transform_dict(crop_transform),
    }
    path = Path(cache_dir) / f"{fingerprint}.json"
    return write_json_atomic(
        path,
        {
            "schema": _CACHE_SCHEMA,
            "fingerprint": fingerprint,
            "payload_sha256": _json_digest(payload),
            "payload": payload,
        },
    )


def _validated_transform_dict(
    transform: RigidTransformResult,
) -> dict[str, object]:
    payload = transform.to_json_dict()
    _transform_from_json(payload)
    return payload


def _transform_from_json(payload: object) -> RigidTransformResult:
    if not isinstance(payload, dict):
        raise TypeError("rigid transform cache payload must be an object")
    matrix = np.asarray(payload.get("matrix"), dtype=float)
    method = payload.get("method")
    match_count = payload.get("match_count")
    inlier_count = payload.get("inlier_count")
    warnings = payload.get("warnings")
    if (
        matrix.shape != (3, 3)
        or not np.all(np.isfinite(matrix))
        or not np.allclose(matrix[2], (0.0, 0.0, 1.0), rtol=0.0, atol=1e-12)
    ):
        raise ValueError("rigid transform cache matrix is invalid")
    if not isinstance(method, str) or not method:
        raise ValueError("rigid transform cache method is invalid")
    if (
        not isinstance(match_count, int)
        or isinstance(match_count, bool)
        or match_count < 0
        or not isinstance(inlier_count, int)
        or isinstance(inlier_count, bool)
        or inlier_count < 0
    ):
        raise ValueError("rigid transform cache counts are invalid")
    if not isinstance(warnings, list) or any(
        not isinstance(value, str) for value in warnings
    ):
        raise ValueError("rigid transform cache warnings are invalid")
    return RigidTransformResult(
        matrix=matrix,
        method=method,
        match_count=match_count,
        inlier_count=inlier_count,
        warnings=list(warnings),
    )


def _update_array_digest(digest: _Digest, values: np.ndarray) -> None:
    array = np.asarray(values)
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(np.ascontiguousarray(array).tobytes())


def _json_digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
