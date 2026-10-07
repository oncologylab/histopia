"""Interoperability with portable patch-feature tables."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from histopia.semantic._features import PatchFeatures


@dataclass(frozen=True, slots=True)
class StampFeatureTable:
    """Feature rows and native ``x, y`` coordinates loaded from STAMP HDF5."""

    features: np.ndarray
    coordinates_xy: np.ndarray
    source_sha256: str
    feature_dataset: str = "feats"
    coordinate_dataset: str = "coords"

    def __post_init__(self) -> None:
        if self.features.ndim != 2 or not len(self.features):
            raise ValueError("STAMP features must be a non-empty two-dimensional array")
        if self.coordinates_xy.shape != (len(self.features), 2):
            raise ValueError("STAMP coordinates must have shape (patches, 2)")
        if not np.all(np.isfinite(self.features)):
            raise ValueError("STAMP features must contain only finite values")
        if not np.all(np.isfinite(self.coordinates_xy)):
            raise ValueError("STAMP coordinates must contain only finite values")
        if len(self.source_sha256) != 64:
            raise ValueError("source_sha256 must be a SHA-256 digest")

    def to_patch_features(
        self,
        *,
        slide_id: str,
        native_to_reference_um: np.ndarray,
        native_patch_size_px: float,
        encoder_patch_size_px: int = 224,
        analysis_mpp: float = 0.5,
        coordinates_are_centers: bool = False,
        tissue_fraction: np.ndarray | None = None,
    ) -> PatchFeatures:
        """Bind imported rows to registered physical coordinates explicitly."""

        matrix = np.asarray(native_to_reference_um, dtype=np.float64)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise ValueError("native_to_reference_um must be a finite 3x3 matrix")
        if native_patch_size_px <= 0:
            raise ValueError("native_patch_size_px must be positive")
        if encoder_patch_size_px <= 0 or analysis_mpp <= 0:
            raise ValueError("encoder patch size and analysis MPP must be positive")
        native_xy = np.asarray(self.coordinates_xy, dtype=np.float64).copy()
        if not coordinates_are_centers:
            native_xy += native_patch_size_px / 2.0
        homogeneous = np.column_stack([native_xy, np.ones(len(native_xy))])
        mapped = (matrix @ homogeneous.T).T
        if np.any(np.isclose(mapped[:, 2], 0)):
            raise ValueError("native_to_reference_um maps coordinates to infinity")
        reference_um_xy = mapped[:, :2] / mapped[:, 2, None]
        grid_rc, grid_shape = _coordinate_grid(self.coordinates_xy)
        fractions = (
            np.ones(len(native_xy), dtype=np.float32)
            if tissue_fraction is None
            else np.asarray(tissue_fraction, dtype=np.float32)
        )
        if fractions.shape != (len(native_xy),):
            raise ValueError("tissue_fraction must contain one value per patch")
        return PatchFeatures(
            slide_id=slide_id,
            features=np.asarray(self.features, dtype=np.float32),
            grid_rc=grid_rc,
            native_xy=native_xy,
            reference_um_xy=reference_um_xy,
            tissue_fraction=fractions,
            grid_shape=grid_shape,
            patch_size_px=encoder_patch_size_px,
            analysis_mpp=analysis_mpp,
            provenance={
                "source_format": "stamp-hdf5",
                "source_sha256": self.source_sha256,
                "feature_dataset": self.feature_dataset,
                "coordinate_dataset": self.coordinate_dataset,
                "coordinate_origin": (
                    "center" if coordinates_are_centers else "top-left"
                ),
                "native_patch_size_px": float(native_patch_size_px),
            },
        )


def load_stamp_features(
    path: Path | str,
    *,
    feature_dataset: str = "feats",
    coordinate_dataset: str = "coords",
) -> StampFeatureTable:
    """Load the common STAMP HDF5 feature/coordinate layout.

    The function reads data only. Registration and physical coordinate binding
    remain explicit through :meth:`StampFeatureTable.to_patch_features`.
    """

    source = Path(path)
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise RuntimeError(
            "STAMP HDF5 import requires h5py from the 'semantic' extra"
        ) from exc
    with h5py.File(source, "r") as handle:
        if feature_dataset not in handle or coordinate_dataset not in handle:
            raise ValueError(
                "STAMP HDF5 must contain the configured feature and coordinate datasets"
            )
        features = np.asarray(handle[feature_dataset], dtype=np.float32)
        coordinates = np.asarray(handle[coordinate_dataset], dtype=np.float64)
    return StampFeatureTable(
        features=features,
        coordinates_xy=coordinates,
        source_sha256=_file_sha256(source),
        feature_dataset=feature_dataset,
        coordinate_dataset=coordinate_dataset,
    )


def _coordinate_grid(coordinates_xy: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    coordinates = np.asarray(coordinates_xy, dtype=np.float64)
    unique_x = np.unique(coordinates[:, 0])
    unique_y = np.unique(coordinates[:, 1])
    cols = np.searchsorted(unique_x, coordinates[:, 0])
    rows = np.searchsorted(unique_y, coordinates[:, 1])
    return (
        np.column_stack([rows, cols]).astype(np.int32),
        (len(unique_y), len(unique_x)),
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
