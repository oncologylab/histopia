"""Validated local caches for deterministic registration preprocessing."""

from __future__ import annotations

import hashlib
import json
import os
import zipfile
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from histopia.registration._config import BrightfieldMaskConfig
from histopia.registration._masking import TissueMaskResult
from histopia.registration._slides import SlideGeometry

_THUMBNAIL_CACHE_SCHEMA = "histopia-registration-thumbnail-v1"
_MASK_CACHE_SCHEMA = "histopia-registration-independent-mask-v1"
_GROUP_MASK_CACHE_SCHEMA = "histopia-registration-group-mask-v1"


def load_or_create_thumbnail(
    source: Path,
    max_dim_px: int,
    cache_dir: Path | None,
    loader: Callable[[Path, int], tuple[np.ndarray, SlideGeometry]],
) -> tuple[np.ndarray, SlideGeometry]:
    """Load a source-bound thumbnail cache entry or decode and persist it."""

    if cache_dir is None:
        return loader(source, max_dim_px)
    fingerprint = _thumbnail_fingerprint(source, max_dim_px)
    entry = cache_dir / "thumbnails" / fingerprint
    cached = _load_thumbnail_entry(entry, fingerprint)
    if cached is not None:
        return cached
    image, geometry = loader(source, max_dim_px)
    _write_thumbnail_entry(entry, fingerprint, image, geometry)
    return image, geometry


def load_or_create_independent_mask(
    source: Path,
    image: np.ndarray,
    config: BrightfieldMaskConfig,
    cache_dir: Path | None,
    creator: Callable[[np.ndarray, BrightfieldMaskConfig], TissueMaskResult],
) -> TissueMaskResult:
    """Load a thumbnail/config-bound mask cache entry or compute it."""

    if cache_dir is None:
        return creator(image, config)
    fingerprint = _mask_fingerprint(source, image, config)
    entry = cache_dir / "masks" / fingerprint
    cached = _load_mask_entry(entry, fingerprint, image.shape[:2])
    if cached is not None:
        return cached
    result = creator(image, config)
    _write_mask_entry(entry, fingerprint, result)
    return result


def load_or_create_group_masks(
    results: dict[Path, TissueMaskResult],
    images: dict[Path, np.ndarray],
    physical_pixel_areas: dict[Path, float | None],
    cache_dir: Path | None,
    creator: Callable[[], dict[Path, TissueMaskResult]],
) -> dict[Path, TissueMaskResult]:
    """Load an exact cohort-refinement cache entry or compute and persist it."""

    if cache_dir is None:
        return creator()
    fingerprint = _group_mask_fingerprint(
        results,
        images,
        physical_pixel_areas,
    )
    entry = cache_dir / "group_masks" / fingerprint
    cached = _load_group_mask_entry(
        entry,
        fingerprint,
        tuple(results),
        {path: image.shape[:2] for path, image in images.items()},
    )
    if cached is not None:
        return cached
    refined = creator()
    _write_group_mask_entry(entry, fingerprint, refined)
    return refined


def _thumbnail_fingerprint(source: Path, max_dim_px: int) -> str:
    stat = source.stat()
    payload = {
        "schema": _THUMBNAIL_CACHE_SCHEMA,
        "source": str(source.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "max_dim_px": max_dim_px,
    }
    return _json_digest(payload)


def _mask_fingerprint(
    source: Path,
    image: np.ndarray,
    config: BrightfieldMaskConfig,
) -> str:
    digest = hashlib.sha256()
    digest.update(_MASK_CACHE_SCHEMA.encode())
    digest.update(str(source.resolve()).encode())
    digest.update(json.dumps(asdict(config), sort_keys=True).encode())
    digest.update(str(image.dtype).encode())
    digest.update(np.asarray(image.shape, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(image).tobytes())
    return digest.hexdigest()


def _group_mask_fingerprint(
    results: dict[Path, TissueMaskResult],
    images: dict[Path, np.ndarray],
    physical_pixel_areas: dict[Path, float | None],
) -> str:
    digest = hashlib.sha256()
    digest.update(_GROUP_MASK_CACHE_SCHEMA.encode())
    for path in results:
        digest.update(str(path.resolve()).encode())
        result = results[path]
        digest.update(result.method.encode())
        digest.update(np.ascontiguousarray(result.mask).tobytes())
        for name in sorted(result.candidate_masks):
            digest.update(name.encode())
            digest.update(np.ascontiguousarray(result.candidate_masks[name]).tobytes())
        image = images[path]
        digest.update(str(image.dtype).encode())
        digest.update(np.asarray(image.shape, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(image).tobytes())
        pixel_area = physical_pixel_areas.get(path)
        digest.update(repr(pixel_area).encode())
    return digest.hexdigest()


def _load_thumbnail_entry(
    entry: Path,
    fingerprint: str,
) -> tuple[np.ndarray, SlideGeometry] | None:
    try:
        metadata = json.loads((entry / "metadata.json").read_text())
        if metadata.get("fingerprint") != fingerprint:
            return None
        image = np.load(entry / "image.npy", allow_pickle=False)
        geometry = _geometry_from_json(metadata["geometry"])
        if (
            image.dtype != np.uint8
            or image.ndim != 3
            or image.shape[2] != 3
            or tuple(image.shape[:2]) != geometry.thumbnail_shape
            or _array_digest(image) != metadata.get("image_sha256")
        ):
            return None
        return image, geometry
    except (
        EOFError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
    ):
        return None


def _write_thumbnail_entry(
    entry: Path,
    fingerprint: str,
    image: np.ndarray,
    geometry: SlideGeometry,
) -> None:
    entry.mkdir(parents=True, exist_ok=True)
    stored_image = np.asarray(image, dtype=np.uint8)
    _write_npy_atomic(entry / "image.npy", stored_image)
    _write_json_atomic(
        entry / "metadata.json",
        {
            "fingerprint": fingerprint,
            "image_sha256": _array_digest(stored_image),
            "geometry": geometry.to_json_dict(),
        },
    )


def _load_mask_entry(
    entry: Path,
    fingerprint: str,
    expected_shape: tuple[int, int],
) -> TissueMaskResult | None:
    try:
        metadata = json.loads((entry / "metadata.json").read_text())
        if metadata.get("fingerprint") != fingerprint:
            return None
        with np.load(entry / "masks.npz", allow_pickle=False) as arrays:
            mask = np.asarray(arrays["mask"], dtype=bool)
            names = metadata["candidate_names"]
            candidates = {
                name: np.asarray(arrays[f"candidate_{index}"], dtype=bool)
                for index, name in enumerate(names)
            }
        if mask.shape != expected_shape or any(
            candidate.shape != expected_shape for candidate in candidates.values()
        ):
            return None
        result = TissueMaskResult(
            mask=mask,
            method=str(metadata["method"]),
            metrics=_float_mapping(metadata["metrics"]),
            accepted=bool(metadata["accepted"]),
            warnings=[str(value) for value in metadata["warnings"]],
            candidate_metrics={
                str(name): _float_mapping(values)
                for name, values in metadata["candidate_metrics"].items()
            },
            candidate_warnings={
                str(name): [str(value) for value in values]
                for name, values in metadata["candidate_warnings"].items()
            },
            candidate_masks=candidates,
        )
        if _mask_result_digest(result) != metadata.get("result_sha256"):
            return None
        return result
    except (
        EOFError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        zipfile.BadZipFile,
        json.JSONDecodeError,
    ):
        return None


def _write_mask_entry(
    entry: Path,
    fingerprint: str,
    result: TissueMaskResult,
) -> None:
    entry.mkdir(parents=True, exist_ok=True)
    candidate_names = sorted(result.candidate_masks)
    arrays = {"mask": np.asarray(result.mask, dtype=bool)}
    arrays.update(
        {
            f"candidate_{index}": np.asarray(
                result.candidate_masks[name],
                dtype=bool,
            )
            for index, name in enumerate(candidate_names)
        }
    )
    _write_npz_atomic(entry / "masks.npz", arrays)
    _write_json_atomic(
        entry / "metadata.json",
        {
            "fingerprint": fingerprint,
            "result_sha256": _mask_result_digest(result),
            "method": result.method,
            "metrics": result.metrics,
            "accepted": result.accepted,
            "warnings": result.warnings,
            "candidate_metrics": result.candidate_metrics,
            "candidate_warnings": result.candidate_warnings,
            "candidate_names": candidate_names,
        },
    )


def _load_group_mask_entry(
    entry: Path,
    fingerprint: str,
    expected_paths: tuple[Path, ...],
    expected_shapes: dict[Path, tuple[int, int]],
) -> dict[Path, TissueMaskResult] | None:
    try:
        metadata = json.loads((entry / "metadata.json").read_text())
        if metadata.get("fingerprint") != fingerprint:
            return None
        names = metadata["slides"]
        if names != [str(path.resolve()) for path in expected_paths]:
            return None
        loaded: dict[Path, TissueMaskResult] = {}
        for index, path in enumerate(expected_paths):
            result = _load_mask_entry(
                entry / f"{index:04d}",
                f"{fingerprint}:{index}",
                expected_shapes[path],
            )
            if result is None:
                return None
            loaded[path] = result
        return loaded
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def _write_group_mask_entry(
    entry: Path,
    fingerprint: str,
    results: dict[Path, TissueMaskResult],
) -> None:
    entry.mkdir(parents=True, exist_ok=True)
    paths = tuple(results)
    for index, path in enumerate(paths):
        _write_mask_entry(
            entry / f"{index:04d}",
            f"{fingerprint}:{index}",
            results[path],
        )
    _write_json_atomic(
        entry / "metadata.json",
        {
            "fingerprint": fingerprint,
            "slides": [str(path.resolve()) for path in paths],
        },
    )


def _geometry_from_json(payload: dict[str, Any]) -> SlideGeometry:
    mpp = payload.get("mpp_xy")
    return SlideGeometry(
        native_shape=tuple(int(value) for value in payload["native_shape"]),
        content_bbox_xywh=tuple(int(value) for value in payload["content_bbox_xywh"]),
        thumbnail_shape=tuple(int(value) for value in payload["thumbnail_shape"]),
        bounds_source=str(payload["bounds_source"]),
        mpp_xy=tuple(float(value) for value in mpp) if mpp is not None else None,
        mpp_source=str(payload.get("mpp_source", "unavailable")),
    )


def _float_mapping(payload: dict[str, Any]) -> dict[str, float]:
    return {str(key): float(value) for key, value in payload.items()}


def _json_digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _array_digest(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _mask_result_digest(result: TissueMaskResult) -> str:
    digest = hashlib.sha256()
    digest.update(result.method.encode())
    digest.update(_array_digest(result.mask).encode())
    digest.update(
        json.dumps(
            {
                "metrics": result.metrics,
                "accepted": result.accepted,
                "warnings": result.warnings,
                "candidate_metrics": result.candidate_metrics,
                "candidate_warnings": result.candidate_warnings,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )
    for name in sorted(result.candidate_masks):
        digest.update(name.encode())
        digest.update(_array_digest(result.candidate_masks[name]).encode())
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    temporary.replace(path)


def _write_npy_atomic(path: Path, array: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    temporary.replace(path)


def _write_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)
