"""Validated, bounded stain assets for the registered-section viewer."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from histopia._atomic import write_binary_atomic
from histopia.registration._errors import OptionalDependencyError
from histopia.stain._artifacts import StainMap
from histopia.stain._result_validation import validate_stain_result

_PROBE_MAX_DIMENSION = 256
_PROBE_NODATA = np.iinfo(np.uint16).max
_PROBE_MAX_VALUE = _PROBE_NODATA - 1
_HEATMAP_STOPS = np.asarray(
    [
        [246, 247, 244],
        [39, 128, 126],
        [238, 190, 70],
        [181, 49, 48],
    ],
    dtype=np.float32,
)


@dataclass(frozen=True, slots=True)
class StainViewerRun:
    """One sealed stain result bound to an exact registration result."""

    root: Path
    payload: dict[str, object]
    slides: dict[str, dict[str, object]]
    display_max_od: float
    review: dict[str, object]


@dataclass(frozen=True, slots=True)
class StainViewerAssets:
    """Display rasters and a compact linked-probe grid for one section."""

    raw_rgba: np.ndarray
    corrected_rgba: np.ndarray
    raw_overlay_rgba: np.ndarray
    corrected_overlay_rgba: np.ndarray
    probe_values: np.ndarray
    probe_width: int
    probe_height: int
    probe_scale_od: float


def load_stain_viewer_run(
    registration_run: Path,
    registration_payload: dict[str, object],
    stain_run: Path,
) -> StainViewerRun:
    """Validate a stain result and its exact registration/order binding."""

    payload = validate_stain_result(stain_run)
    registration_sha = _file_sha256(registration_run / "registration_result.json")
    if payload.get("registration_result_sha256") != registration_sha:
        raise ValueError("stain result belongs to a different registration result")
    registration_ids = [
        Path(str(row["path"])).name for row in registration_payload["slides"]
    ]
    stain_rows = payload.get("slides")
    if not isinstance(stain_rows, list):
        raise ValueError("stain result slides must be a list")
    stain_ids = [str(row.get("id", "")) for row in stain_rows]
    if stain_ids != registration_ids:
        raise ValueError("stain result slide order does not match registration")
    if len(set(stain_ids)) != len(stain_ids):
        raise ValueError("stain result slide identities must be unique")
    rows = {str(row["id"]): row for row in stain_rows if isinstance(row, dict)}
    q99 = [
        float(row["quantiles"]["0.99"])
        for row in stain_rows
        if row.get("quantified")
        and isinstance(row.get("quantiles"), dict)
        and row["quantiles"].get("0.99") is not None
    ]
    if not q99 or not np.all(np.isfinite(q99)):
        raise ValueError("stain result has no finite quantified display range")
    review = _load_review(stain_run, str(payload["fingerprint"]))
    return StainViewerRun(
        root=stain_run,
        payload=payload,
        slides=rows,
        display_max_od=max(float(max(q99)), 1e-4),
        review=review,
    )


def build_stain_viewer_assets(
    stain_map: StainMap,
    *,
    source_shape: tuple[int, int],
    matrix: np.ndarray,
    output_shape: tuple[int, int],
    registered_rgb: np.ndarray,
    registered_mask: np.ndarray,
    display_max_od: float,
) -> StainViewerAssets:
    """Warp one source-space map and derive bounded viewer artifacts."""

    raw = _resize_and_warp_float(
        stain_map.raw_target_od,
        source_shape,
        matrix,
        output_shape,
    )
    corrected = _resize_and_warp_float(
        stain_map.corrected_target_od,
        source_shape,
        matrix,
        output_shape,
    )
    map_tissue = _resize_and_warp_mask(
        stain_map.tissue_mask,
        source_shape,
        matrix,
        output_shape,
    )
    tissue = np.asarray(registered_mask, dtype=bool) & map_tissue
    raw = np.where(tissue, np.maximum(raw, 0), 0).astype(np.float32)
    corrected = np.where(tissue, np.maximum(corrected, 0), 0).astype(np.float32)
    raw_rgba = _heatmap_rgba(raw, tissue, display_max_od)
    corrected_rgba = _heatmap_rgba(corrected, tissue, display_max_od)
    probe, probe_width, probe_height, probe_scale = _probe_grid(
        raw,
        corrected,
        tissue,
        display_max_od,
    )
    return StainViewerAssets(
        raw_rgba=raw_rgba,
        corrected_rgba=corrected_rgba,
        raw_overlay_rgba=_overlay_rgba(
            registered_rgb,
            registered_mask,
            tissue,
            raw_rgba,
            raw,
            display_max_od,
        ),
        corrected_overlay_rgba=_overlay_rgba(
            registered_rgb,
            registered_mask,
            tissue,
            corrected_rgba,
            corrected,
            display_max_od,
        ),
        probe_values=probe,
        probe_width=probe_width,
        probe_height=probe_height,
        probe_scale_od=probe_scale,
    )


def write_stain_probe(path: Path, values: np.ndarray) -> Path:
    """Atomically write planar little-endian uint16 raw/corrected grids."""

    array = np.ascontiguousarray(values, dtype="<u2")
    return write_binary_atomic(path, lambda stream: stream.write(array.tobytes()))


def _resize_and_warp_float(
    values: np.ndarray,
    source_shape: tuple[int, int],
    matrix: np.ndarray,
    output_shape: tuple[int, int],
) -> np.ndarray:
    try:
        import cv2
        from PIL import Image
    except ImportError as exc:
        raise OptionalDependencyError(
            "opencv-contrib-python-headless and pillow",
            "visualization",
        ) from exc
    source_height, source_width = source_shape
    resized = np.asarray(
        Image.fromarray(np.asarray(values, dtype=np.float32), mode="F").resize(
            (source_width, source_height),
            resample=Image.Resampling.BILINEAR,
        ),
        dtype=np.float32,
    )
    output_height, output_width = output_shape
    return cv2.warpAffine(
        resized,
        np.asarray(matrix, dtype=np.float32)[:2, :],
        dsize=(output_width, output_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _resize_and_warp_mask(
    values: np.ndarray,
    source_shape: tuple[int, int],
    matrix: np.ndarray,
    output_shape: tuple[int, int],
) -> np.ndarray:
    try:
        import cv2
        from PIL import Image
    except ImportError as exc:
        raise OptionalDependencyError(
            "opencv-contrib-python-headless and pillow",
            "visualization",
        ) from exc
    source_height, source_width = source_shape
    resized = np.asarray(
        Image.fromarray(np.asarray(values, dtype=np.uint8) * 255).resize(
            (source_width, source_height),
            resample=Image.Resampling.NEAREST,
        )
    )
    output_height, output_width = output_shape
    return (
        cv2.warpAffine(
            resized,
            np.asarray(matrix, dtype=np.float32)[:2, :],
            dsize=(output_width, output_height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        > 127
    )


def _heatmap_rgba(
    values: np.ndarray,
    tissue: np.ndarray,
    maximum: float,
) -> np.ndarray:
    normalized = np.clip(np.asarray(values, dtype=np.float32) / maximum, 0, 1)
    segment = np.minimum(
        (normalized * (len(_HEATMAP_STOPS) - 1)).astype(np.intp),
        len(_HEATMAP_STOPS) - 2,
    )
    local = normalized * (len(_HEATMAP_STOPS) - 1) - segment
    rgb = (
        _HEATMAP_STOPS[segment] * (1 - local[..., None])
        + _HEATMAP_STOPS[segment + 1] * local[..., None]
    )
    return np.dstack(
        [
            np.clip(rgb, 0, 255).astype(np.uint8),
            np.where(tissue, 238, 0).astype(np.uint8),
        ]
    )


def _overlay_rgba(
    registered_rgb: np.ndarray,
    registered_mask: np.ndarray,
    signal_mask: np.ndarray,
    heatmap: np.ndarray,
    values: np.ndarray,
    maximum: float,
) -> np.ndarray:
    normalized = np.sqrt(np.clip(values / maximum, 0, 1))[..., None]
    alpha = np.where(
        np.asarray(signal_mask)[..., None],
        0.12 + 0.68 * normalized,
        0,
    )
    rgb = (
        np.asarray(registered_rgb, dtype=np.float32) * (1 - alpha)
        + heatmap[..., :3].astype(np.float32) * alpha
    )
    return np.dstack(
        [
            np.clip(rgb, 0, 255).astype(np.uint8),
            (np.asarray(registered_mask, dtype=np.uint8) * 255),
        ]
    )


def _probe_grid(
    raw: np.ndarray,
    corrected: np.ndarray,
    tissue: np.ndarray,
    maximum: float,
) -> tuple[np.ndarray, int, int, float]:
    from PIL import Image

    height, width = tissue.shape
    scale = min(1.0, _PROBE_MAX_DIMENSION / max(height, width))
    probe_width = max(1, round(width * scale))
    probe_height = max(1, round(height * scale))

    def resize_float(values: np.ndarray) -> np.ndarray:
        return np.asarray(
            Image.fromarray(np.asarray(values, dtype=np.float32), mode="F").resize(
                (probe_width, probe_height),
                resample=Image.Resampling.BILINEAR,
            ),
            dtype=np.float32,
        )

    probe_tissue = (
        np.asarray(
            Image.fromarray(np.asarray(tissue, dtype=np.uint8) * 255).resize(
                (probe_width, probe_height),
                resample=Image.Resampling.NEAREST,
            )
        )
        > 127
    )
    quantization_scale = maximum / _PROBE_MAX_VALUE
    channels = []
    for values in (raw, corrected):
        resized = resize_float(values)
        encoded = np.rint(np.clip(resized, 0, maximum) / quantization_scale).astype(
            np.uint16
        )
        encoded[~probe_tissue] = _PROBE_NODATA
        channels.append(encoded)
    return (
        np.stack(channels),
        probe_width,
        probe_height,
        quantization_scale,
    )


def _load_review(root: Path, fingerprint: str) -> dict[str, object]:
    try:
        review = json.loads((root / "stain_review.json").read_text())
    except (OSError, json.JSONDecodeError):
        return {"approved": False, "fingerprint_matches": False}
    return {
        "approved": bool(review.get("approved")),
        "fingerprint_matches": review.get("fingerprint") == fingerprint,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
