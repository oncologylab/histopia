"""Compact patch-feature artifacts and registration-aware coordinates."""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections import deque
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

import numpy as np

from histopia.registration._slides import SlideGeometry


class PatchEncoder(Protocol):
    """Minimal interface implemented by UNI2-h and test encoders."""

    def encode(self, images: np.ndarray) -> np.ndarray:
        """Return one feature row per RGB image."""


PatchReader = Callable[[int, int, int, int, int], np.ndarray]
PatchRequest = tuple[int, int, int, int, int]


class BatchPatchReader(Protocol):
    """Optional optimized interface for reading an ordered patch batch."""

    def read_many(self, requests: Sequence[PatchRequest]) -> Sequence[np.ndarray]:
        """Return one RGB array for each request without changing its order."""


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
    provenance: dict[str, object] | None = None
    fingerprint: str | None = None
    content_fingerprint: str | None = None

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
        expected = _provenance_fingerprint(self.provenance)
        if self.fingerprint is not None and self.fingerprint != expected:
            raise ValueError("feature provenance fingerprint does not match")
        object.__setattr__(self, "fingerprint", expected)
        if self.content_fingerprint is not None:
            expected_content = _content_fingerprint(self)
            if self.content_fingerprint != expected_content:
                raise ValueError("feature content fingerprint does not match")

    def save(self, path: Path | str) -> Path:
        """Write a portable artifact without repeated tile vectors."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        stored = _stored_arrays(self)
        content_fingerprint = _content_fingerprint(self)
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp.npz", delete=False
        ) as stream:
            temporary = Path(stream.name)
        try:
            np.savez(
                temporary,
                schema_version=np.int16(3),
                slide_id=np.asarray(self.slide_id),
                **stored,
                grid_shape=np.asarray(self.grid_shape, dtype=np.int32),
                patch_size_px=np.int32(self.patch_size_px),
                analysis_mpp=np.float64(self.analysis_mpp),
                provenance_json=np.asarray(_canonical_json(self.provenance)),
                fingerprint=np.asarray(self.fingerprint or ""),
                content_fingerprint=np.asarray(content_fingerprint),
            )
            self.load(temporary)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return path

    @classmethod
    def load(cls, path: Path | str) -> PatchFeatures:
        """Load and validate a compact feature artifact."""

        with np.load(Path(path), allow_pickle=False) as data:
            schema_version = int(data["schema_version"])
            if schema_version not in {1, 2, 3}:
                raise ValueError("unsupported patch feature schema")
            provenance = (
                json.loads(str(data["provenance_json"]))
                if schema_version >= 2
                else None
            )
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
                provenance=provenance,
                fingerprint=(str(data["fingerprint"]) or None)
                if schema_version >= 2
                else None,
                content_fingerprint=str(data["content_fingerprint"])
                if schema_version == 3
                else None,
            )


def subset_patch_features(
    artifact: PatchFeatures,
    selected: np.ndarray,
) -> PatchFeatures:
    """Return a content-sealed subset while preserving source provenance."""

    mask = np.asarray(selected, dtype=bool)
    if mask.shape != (len(artifact.features),):
        raise ValueError("selected must contain one boolean per patch")
    if not np.any(mask):
        raise ValueError("patch feature subset must not be empty")
    candidate = PatchFeatures(
        slide_id=artifact.slide_id,
        features=artifact.features[mask],
        grid_rc=artifact.grid_rc[mask],
        native_xy=artifact.native_xy[mask],
        reference_um_xy=artifact.reference_um_xy[mask],
        tissue_fraction=artifact.tissue_fraction[mask],
        grid_shape=artifact.grid_shape,
        patch_size_px=artifact.patch_size_px,
        analysis_mpp=artifact.analysis_mpp,
        provenance=artifact.provenance,
    )
    return replace(
        candidate,
        content_fingerprint=_content_fingerprint(candidate),
    )


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _provenance_fingerprint(provenance: dict[str, object] | None) -> str | None:
    if provenance is None:
        return None
    return hashlib.sha256(_canonical_json(provenance).encode()).hexdigest()


def _stored_arrays(features: PatchFeatures) -> dict[str, np.ndarray]:
    return {
        "features": np.asarray(features.features, dtype=np.float16),
        "grid_rc": np.asarray(features.grid_rc, dtype=np.int32),
        "native_xy": np.asarray(features.native_xy, dtype=np.float64),
        "reference_um_xy": np.asarray(
            features.reference_um_xy,
            dtype=np.float64,
        ),
        "tissue_fraction": np.asarray(
            features.tissue_fraction,
            dtype=np.float32,
        ),
    }


def _content_fingerprint(features: PatchFeatures) -> str:
    digest = hashlib.sha256(b"histopia-patch-features-content-v1\0")
    metadata = {
        "slide_id": features.slide_id,
        "grid_shape": list(features.grid_shape),
        "patch_size_px": features.patch_size_px,
        "analysis_mpp": features.analysis_mpp,
        "provenance": features.provenance,
    }
    digest.update(_canonical_json(metadata).encode())
    canonical_dtypes = {
        "features": "<f2",
        "grid_rc": "<i4",
        "native_xy": "<f8",
        "reference_um_xy": "<f8",
        "tissue_fraction": "<f4",
    }
    for name, stored in _stored_arrays(features).items():
        array = np.ascontiguousarray(stored, dtype=canonical_dtypes[name])
        digest.update(name.encode())
        digest.update(array.dtype.str.encode())
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


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
    patch_workers: int = 1,
    provenance: dict[str, object] | None = None,
    reusable_features: PatchFeatures | None = None,
) -> PatchFeatures:
    """Read and encode tissue patches on a calibrated, non-overlapping grid.

    A sealed feature artifact from the same native slide grid may be supplied to
    avoid encoding unchanged patches again. Tissue coverage and registered
    coordinates are always recalculated from the current mask and transform.
    """

    if patch_workers <= 0:
        raise ValueError("patch_workers must be positive")
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
    coverage = _grid_mask_coverage(
        mask,
        native_to_thumbnail=geometry.native_to_thumbnail,
        content_origin_xy=(x0, y0),
        grid_shape=(rows, cols),
        patch_shape=(native_height, native_width),
    )
    accepted = [
        (
            int(row),
            int(col),
            x0 + int(col) * native_width,
            y0 + int(row) * native_height,
            float(coverage[row, col]),
        )
        for row, col in np.argwhere(coverage >= min_tissue_fraction)
    ]
    if not accepted:
        raise ValueError(f"no tissue patches passed coverage for {slide_id}")

    reusable_by_grid = _reusable_features_by_grid(
        reusable_features,
        slide_id=slide_id,
        geometry=geometry,
        grid_shape=(rows, cols),
        native_patch_shape=(native_height, native_width),
        patch_size_px=patch_size_px,
        analysis_mpp=analysis_mpp,
    )
    feature_rows: list[np.ndarray | None] = []
    missing: list[tuple[int, tuple[int, int, int, int, float]]] = []
    for index, row in enumerate(accepted):
        reused = reusable_by_grid.get((row[0], row[1]))
        feature_rows.append(reused)
        if reused is None:
            missing.append((index, row))

    def request(row: tuple[int, int, int, int, float]) -> PatchRequest:
        _, _, left, top, _ = row
        return left, top, native_width, native_height, patch_size_px

    def read_patch(row: tuple[int, int, int, int, float]) -> np.ndarray:
        return reader(*request(row))

    patch_executor = (
        ThreadPoolExecutor(max_workers=patch_workers) if patch_workers > 1 else None
    )
    read_many = getattr(reader, "read_many", None)

    def read_batch(
        batch_rows: Sequence[tuple[int, int, int, int, float]],
    ) -> tuple[np.ndarray, ...]:
        if callable(read_many):
            return tuple(read_many(tuple(request(row) for row in batch_rows)))
        patches = (
            patch_executor.map(read_patch, batch_rows)
            if patch_executor is not None
            else map(read_patch, batch_rows)
        )
        return tuple(patches)

    batches = tuple(
        missing[start : start + batch_size]
        for start in range(0, len(missing), batch_size)
    )
    if batches:
        prefetch_depth = patch_workers if callable(read_many) else 1
        prefetch_executor = ThreadPoolExecutor(max_workers=prefetch_depth)
        pending: deque[Future[tuple[np.ndarray, ...]]] = deque(
            prefetch_executor.submit(
                read_batch,
                tuple(row for _, row in batch),
            )
            for batch in batches[:prefetch_depth]
        )
        try:
            for batch_index, batch in enumerate(batches):
                patches = pending.popleft().result()
                next_index = batch_index + prefetch_depth
                if next_index < len(batches):
                    pending.append(
                        prefetch_executor.submit(
                            read_batch,
                            tuple(row for _, row in batches[next_index]),
                        )
                    )
                images = np.stack(patches)
                if images.shape[1:] != (patch_size_px, patch_size_px, 3):
                    raise ValueError(
                        "patch reader must return output_px square RGB arrays"
                    )
                encoded = np.asarray(encoder.encode(images), dtype=np.float32)
                if encoded.ndim != 2 or encoded.shape[0] != len(images):
                    raise ValueError("encoder must return one feature vector per image")
                if reusable_by_grid:
                    reusable_width = next(iter(reusable_by_grid.values())).shape[0]
                    if encoded.shape[1] != reusable_width:
                        raise ValueError(
                            "reusable feature width does not match encoder output"
                        )
                for (output_index, _), encoded_row in zip(
                    batch,
                    encoded,
                    strict=True,
                ):
                    feature_rows[output_index] = encoded_row
        finally:
            for future in pending:
                future.cancel()
            prefetch_executor.shutdown(cancel_futures=True)
            if patch_executor is not None:
                patch_executor.shutdown(cancel_futures=True)
    elif patch_executor is not None:
        patch_executor.shutdown(cancel_futures=True)

    if any(row is None for row in feature_rows):
        raise RuntimeError("feature extraction did not fill every accepted patch")
    features = np.stack([row for row in feature_rows if row is not None]).astype(
        np.float32,
        copy=False,
    )

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
        features=features,
        grid_rc=grid_rc,
        native_xy=native_xy,
        reference_um_xy=reference_xy,
        tissue_fraction=np.asarray([row[-1] for row in accepted], dtype=np.float32),
        grid_shape=(rows, cols),
        patch_size_px=patch_size_px,
        analysis_mpp=analysis_mpp,
        provenance=provenance,
    )


def _reusable_features_by_grid(
    artifact: PatchFeatures | None,
    *,
    slide_id: str,
    geometry: SlideGeometry,
    grid_shape: tuple[int, int],
    native_patch_shape: tuple[int, int],
    patch_size_px: int,
    analysis_mpp: float,
) -> dict[tuple[int, int], np.ndarray]:
    """Validate a reusable native-grid artifact and index its feature rows."""

    if artifact is None:
        return {}
    if artifact.slide_id != slide_id:
        raise ValueError("reusable features belong to a different slide")
    if artifact.grid_shape != grid_shape:
        raise ValueError("reusable features use a different native grid shape")
    if artifact.patch_size_px != patch_size_px:
        raise ValueError("reusable features use a different patch size")
    if artifact.analysis_mpp != analysis_mpp:
        raise ValueError("reusable features use a different analysis MPP")
    if artifact.features.shape[1] <= 0:
        raise ValueError("reusable features must have a positive feature width")

    grid = np.asarray(artifact.grid_rc, dtype=np.int64)
    if (
        np.any(grid < 0)
        or np.any(grid[:, 0] >= grid_shape[0])
        or np.any(grid[:, 1] >= grid_shape[1])
    ):
        raise ValueError("reusable features contain out-of-bounds grid cells")
    keys = [tuple(int(value) for value in row) for row in grid]
    if len(set(keys)) != len(keys):
        raise ValueError("reusable features contain duplicate grid cells")

    native_height, native_width = native_patch_shape
    x0, y0, _, _ = geometry.content_bbox_xywh
    expected_native_xy = np.column_stack(
        [
            x0 + grid[:, 1] * native_width + native_width / 2,
            y0 + grid[:, 0] * native_height + native_height / 2,
        ]
    )
    if not np.allclose(
        artifact.native_xy,
        expected_native_xy,
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError("reusable features use different native patch centers")
    return {
        key: np.asarray(feature, dtype=np.float32)
        for key, feature in zip(keys, artifact.features, strict=True)
    }


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


def _grid_mask_coverage(
    mask: np.ndarray,
    *,
    native_to_thumbnail: np.ndarray,
    content_origin_xy: tuple[int, int],
    grid_shape: tuple[int, int],
    patch_shape: tuple[int, int],
) -> np.ndarray:
    """Compute exact mask fractions for a regular native-pixel patch grid."""

    rows, cols = grid_shape
    patch_height, patch_width = patch_shape
    if rows <= 0 or cols <= 0:
        return np.zeros((max(0, rows), max(0, cols)), dtype=np.float64)
    origin_x, origin_y = content_origin_xy
    grid_rows, grid_cols = np.indices((rows, cols), dtype=np.int64)
    top_left = np.column_stack(
        [
            (origin_x + grid_cols.ravel() * patch_width),
            (origin_y + grid_rows.ravel() * patch_height),
        ]
    )
    bottom_right = top_left + np.array([patch_width, patch_height])
    mapped_start = _apply_homogeneous(top_left, native_to_thumbnail)
    mapped_end = _apply_homogeneous(bottom_right, native_to_thumbnail)
    x0 = np.floor(mapped_start[:, 0]).astype(np.int64)
    y0 = np.floor(mapped_start[:, 1]).astype(np.int64)
    x1 = np.ceil(mapped_end[:, 0]).astype(np.int64)
    y1 = np.ceil(mapped_end[:, 1]).astype(np.int64)
    x0 = np.clip(x0, 0, mask.shape[1])
    x1 = np.clip(x1, 0, mask.shape[1])
    y0 = np.clip(y0, 0, mask.shape[0])
    y1 = np.clip(y1, 0, mask.shape[0])

    integral = np.pad(
        np.asarray(mask, dtype=np.int64).cumsum(axis=0).cumsum(axis=1),
        ((1, 0), (1, 0)),
    )
    tissue = integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]
    area = (x1 - x0) * (y1 - y0)
    coverage = np.divide(
        tissue,
        area,
        out=np.zeros(tissue.shape, dtype=np.float64),
        where=area > 0,
    )
    return coverage.reshape(rows, cols)


def _apply_homogeneous(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack([points, np.ones(len(points))])
    mapped = (np.asarray(matrix, dtype=float) @ homogeneous.T).T
    return mapped[:, :2] / mapped[:, 2, None]
