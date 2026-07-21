"""Compact patch-feature artifacts and registration-aware coordinates."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from histopia.registration._slides import SlideGeometry


class PatchEncoder(Protocol):
    """Minimal interface implemented by UNI2-h and test encoders."""

    def encode(self, images: np.ndarray) -> np.ndarray:
        """Return one feature row per RGB image."""


PatchReader = Callable[[int, int, int, int, int], np.ndarray]


@dataclass(frozen=True, slots=True)
class PatchFeatures:
    """One feature vector per tissue patch in a source whole-slide image."""

    slide_id: str
    features: np.ndarray
    grid_rc: np.ndarray
    native_xy: np.ndarray
    reference_um_xy: np.ndarray
    tissue_fraction: np.ndarray
    grid_shape: tuple[int, int]
    patch_size_px: int
    analysis_mpp: float

    def __post_init__(self) -> None:
        arrays = (
            self.features,
            self.grid_rc,
            self.native_xy,
            self.reference_um_xy,
            self.tissue_fraction,
        )
        if len({array.shape[0] for array in arrays}) != 1:
            raise ValueError("feature arrays must contain the same number of patches")
        if self.features.ndim != 2:
            raise ValueError("features must be a two-dimensional array")
        for name, array in (
            ("grid_rc", self.grid_rc),
            ("native_xy", self.native_xy),
            ("reference_um_xy", self.reference_um_xy),
        ):
            if array.ndim != 2 or array.shape[1] != 2:
                raise ValueError(f"{name} must have shape (patches, 2)")
        if self.tissue_fraction.ndim != 1:
            raise ValueError("tissue_fraction must be one-dimensional")
        if min(self.grid_shape) <= 0 or self.patch_size_px <= 0:
            raise ValueError("grid and patch dimensions must be positive")
        if self.analysis_mpp <= 0:
            raise ValueError("analysis_mpp must be positive")

    def save(self, path: Path | str) -> Path:
        """Write a compressed, portable artifact without repeated tile vectors."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            schema_version=np.int16(1),
            slide_id=np.asarray(self.slide_id),
            features=np.asarray(self.features, dtype=np.float16),
            grid_rc=np.asarray(self.grid_rc, dtype=np.int32),
            native_xy=np.asarray(self.native_xy, dtype=np.float64),
            reference_um_xy=np.asarray(self.reference_um_xy, dtype=np.float64),
            tissue_fraction=np.asarray(self.tissue_fraction, dtype=np.float32),
            grid_shape=np.asarray(self.grid_shape, dtype=np.int32),
            patch_size_px=np.int32(self.patch_size_px),
            analysis_mpp=np.float64(self.analysis_mpp),
        )
        return path

    @classmethod
    def load(cls, path: Path | str) -> PatchFeatures:
        """Load and validate a compact feature artifact."""

        with np.load(Path(path), allow_pickle=False) as data:
            if int(data["schema_version"]) != 1:
                raise ValueError("unsupported patch feature schema")
            return cls(
                slide_id=str(data["slide_id"]),
                features=data["features"],
                grid_rc=data["grid_rc"],
                native_xy=data["native_xy"],
                reference_um_xy=data["reference_um_xy"],
                tissue_fraction=data["tissue_fraction"],
                grid_shape=tuple(int(value) for value in data["grid_shape"]),
                patch_size_px=int(data["patch_size_px"]),
                analysis_mpp=float(data["analysis_mpp"]),
            )


def map_native_to_reference_um(
    native_xy: np.ndarray,
    *,
    native_to_thumbnail: np.ndarray,
    moving_to_reference_thumbnail: np.ndarray,
    reference_thumbnail_to_native: np.ndarray,
    reference_mpp_xy: tuple[float, float],
) -> np.ndarray:
    """Map source native-pixel coordinates into reference micrometres."""

    points = np.asarray(native_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("native_xy must have shape (points, 2)")
    homogeneous = np.column_stack([points, np.ones(points.shape[0])])
    matrix = (
        np.diag([reference_mpp_xy[0], reference_mpp_xy[1], 1.0])
        @ np.asarray(reference_thumbnail_to_native, dtype=np.float64)
        @ np.asarray(moving_to_reference_thumbnail, dtype=np.float64)
        @ np.asarray(native_to_thumbnail, dtype=np.float64)
    )
    mapped = (matrix @ homogeneous.T).T
    return mapped[:, :2] / mapped[:, 2, None]


def extract_patch_features(
    *,
    slide_id: str,
    geometry: SlideGeometry,
    tissue_mask: np.ndarray,
    moving_to_reference_thumbnail: np.ndarray,
    reference_geometry: SlideGeometry,
    reader: PatchReader,
    encoder: PatchEncoder,
    analysis_mpp: float = 0.5,
    patch_size_px: int = 224,
    min_tissue_fraction: float = 0.5,
    batch_size: int = 64,
) -> PatchFeatures:
    """Read and encode tissue patches on a calibrated, non-overlapping grid."""

    if geometry.mpp_xy is None or reference_geometry.mpp_xy is None:
        raise ValueError("feature extraction requires calibrated slide MPP")
    mask = np.asarray(tissue_mask, dtype=bool)
    if mask.shape != geometry.thumbnail_shape:
        raise ValueError("tissue mask must match the registration thumbnail")
    patch_um = patch_size_px * analysis_mpp
    native_width = max(1, int(round(patch_um / geometry.mpp_xy[0])))
    native_height = max(1, int(round(patch_um / geometry.mpp_xy[1])))
    x0, y0, content_width, content_height = geometry.content_bbox_xywh
    rows = content_height // native_height
    cols = content_width // native_width
    accepted: list[tuple[int, int, int, int, float]] = []
    for row in range(rows):
        top = y0 + row * native_height
        for col in range(cols):
            left = x0 + col * native_width
            fraction = _mask_coverage(
                mask,
                geometry.native_to_thumbnail,
                left,
                top,
                native_width,
                native_height,
            )
            if fraction >= min_tissue_fraction:
                accepted.append((row, col, left, top, fraction))
    if not accepted:
        raise ValueError(f"no tissue patches passed coverage for {slide_id}")

    feature_batches: list[np.ndarray] = []
    for start in range(0, len(accepted), batch_size):
        batch_rows = accepted[start : start + batch_size]
        images = np.stack(
            [
                reader(left, top, native_width, native_height, patch_size_px)
                for _, _, left, top, _ in batch_rows
            ]
        )
        if images.shape[1:] != (patch_size_px, patch_size_px, 3):
            raise ValueError("patch reader must return output_px square RGB arrays")
        encoded = np.asarray(encoder.encode(images), dtype=np.float32)
        if encoded.ndim != 2 or encoded.shape[0] != len(images):
            raise ValueError("encoder must return one feature vector per image")
        feature_batches.append(encoded)

    grid_rc = np.asarray([(row, col) for row, col, *_ in accepted], dtype=np.int32)
    native_xy = np.asarray(
        [
            (left + native_width / 2, top + native_height / 2)
            for _, _, left, top, _ in accepted
        ],
        dtype=np.float64,
    )
    reference_xy = map_native_to_reference_um(
        native_xy,
        native_to_thumbnail=geometry.native_to_thumbnail,
        moving_to_reference_thumbnail=moving_to_reference_thumbnail,
        reference_thumbnail_to_native=reference_geometry.thumbnail_to_native,
        reference_mpp_xy=reference_geometry.mpp_xy,
    )
    return PatchFeatures(
        slide_id=slide_id,
        features=np.concatenate(feature_batches),
        grid_rc=grid_rc,
        native_xy=native_xy,
        reference_um_xy=reference_xy,
        tissue_fraction=np.asarray([row[-1] for row in accepted], dtype=np.float32),
        grid_shape=(rows, cols),
        patch_size_px=patch_size_px,
        analysis_mpp=analysis_mpp,
    )


def _mask_coverage(
    mask: np.ndarray,
    native_to_thumbnail: np.ndarray,
    left: int,
    top: int,
    width: int,
    height: int,
) -> float:
    corners = np.array([[left, top], [left + width, top + height]], dtype=float)
    mapped = _apply_homogeneous(corners, native_to_thumbnail)
    x0, y0 = np.floor(mapped[0]).astype(int)
    x1, y1 = np.ceil(mapped[1]).astype(int)
    x0, x1 = np.clip((x0, x1), 0, mask.shape[1])
    y0, y1 = np.clip((y0, y1), 0, mask.shape[0])
    return float(np.mean(mask[y0:y1, x0:x1])) if x1 > x0 and y1 > y0 else 0.0


def _apply_homogeneous(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack([points, np.ones(len(points))])
    mapped = (np.asarray(matrix, dtype=float) @ homogeneous.T).T
    return mapped[:, :2] / mapped[:, 2, None]
