"""Target-free, cell-resolved multiscale morphology features.

The feature schema in this module is deliberately independent of protein
outcomes.  It combines the existing stain-neutral UNI2-h token assigned to
each native cell with measurements from the accepted cell boundary, local
cell neighborhoods, and registered physical coordinates.  Target chromogen,
marker identity, semantic class, and measured OD are not accepted inputs.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from histopia._atomic import write_binary_atomic

FEATURE_SCHEMA_ID = "native-hdab-neutral-cell-multiscale-v3"
DEFAULT_RADII_UM = (16.0, 32.0, 64.0, 128.0)
SHAPE_FEATURE_NAMES = (
    "log_area_um2",
    "log_perimeter_um",
    "compactness",
    "extent",
    "bbox_aspect",
    "eccentricity",
    "moment_aspect",
    "orientation_sin",
    "orientation_cos",
)
TEXTURE_FEATURE_NAMES = (
    "hematoxylin_mean",
    "hematoxylin_std",
    "hematoxylin_q25",
    "hematoxylin_q50",
    "hematoxylin_q75",
    "hematoxylin_q90",
    "hematoxylin_entropy",
    "boundary_hematoxylin_mean",
    "boundary_hematoxylin_std",
    "boundary_hematoxylin_q90",
)


@dataclass(frozen=True, slots=True)
class CellFeatureSet:
    """One target-free, fingerprinted feature matrix per native cell."""

    slide_id: str
    label_ids: np.ndarray
    native_xy: np.ndarray
    reference_um_xyz: np.ndarray
    morphology: np.ndarray
    phenotype: np.ndarray
    neighborhood: np.ndarray
    position: np.ndarray
    supported: np.ndarray
    feature_names: dict[str, tuple[str, ...]]
    provenance: dict[str, object]
    fingerprint: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.slide_id, str) or not self.slide_id:
            raise ValueError("cell feature slide_id must be non-empty text")
        labels = np.asarray(self.label_ids)
        count = len(labels)
        if (
            labels.ndim != 1
            or not np.issubdtype(labels.dtype, np.integer)
            or np.any(labels <= 0)
            or len(np.unique(labels)) != count
        ):
            raise ValueError("cell feature labels must be unique positive integers")
        if np.asarray(self.native_xy).shape != (count, 2):
            raise ValueError("native_xy must have shape (cells, 2)")
        if np.asarray(self.reference_um_xyz).shape != (count, 3):
            raise ValueError("reference_um_xyz must have shape (cells, 3)")
        matrices = {
            "morphology": self.morphology,
            "phenotype": self.phenotype,
            "neighborhood": self.neighborhood,
            "position": self.position,
        }
        if set(self.feature_names) != set(matrices):
            raise ValueError("cell feature names must describe every feature group")
        for name, value in matrices.items():
            matrix = np.asarray(value)
            names = tuple(self.feature_names[name])
            if matrix.ndim != 2 or matrix.shape != (count, len(names)):
                raise ValueError(f"{name} feature names and matrix do not align")
            if not np.all(np.isfinite(matrix)):
                raise ValueError(f"{name} features must be finite")
        if np.asarray(self.supported).shape != (count,):
            raise ValueError("cell feature support must align with labels")
        if not np.all(np.isfinite(self.native_xy)) or not np.all(
            np.isfinite(self.reference_um_xyz)
        ):
            raise ValueError("cell feature coordinates must be finite")
        _validate_path_free(self.provenance)
        expected = _feature_fingerprint(self)
        if self.fingerprint is not None and self.fingerprint != expected:
            raise ValueError("cell feature fingerprint does not match")
        object.__setattr__(self, "fingerprint", expected)

    @property
    def features(self) -> np.ndarray:
        """Return all feature groups in their stable schema order."""

        return np.concatenate(
            (self.morphology, self.phenotype, self.neighborhood, self.position),
            axis=1,
        ).astype(np.float32, copy=False)

    def save(self, path: Path | str) -> Path:
        target = Path(path)
        metadata = {
            "schema_version": 1,
            "feature_schema_id": FEATURE_SCHEMA_ID,
            "slide_id": self.slide_id,
            "feature_names": {
                key: list(value) for key, value in sorted(self.feature_names.items())
            },
            "provenance": self.provenance,
            "fingerprint": self.fingerprint,
        }

        def writer(stream) -> None:
            np.savez_compressed(
                stream,
                metadata_json=np.asarray(_canonical_json(metadata)),
                **_feature_arrays(self),
            )

        return write_binary_atomic(target, writer)

    @classmethod
    def load(cls, path: Path | str) -> CellFeatureSet:
        with np.load(Path(path), allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"]))
            if (
                metadata.get("schema_version") != 1
                or metadata.get("feature_schema_id") != FEATURE_SCHEMA_ID
            ):
                raise ValueError("unsupported multiscale cell feature schema")
            return cls(
                slide_id=str(metadata["slide_id"]),
                feature_names={
                    str(key): tuple(str(name) for name in value)
                    for key, value in metadata["feature_names"].items()
                },
                provenance=dict(metadata["provenance"]),
                fingerprint=str(metadata["fingerprint"]),
                **{name: data[name] for name in _FEATURE_ARRAY_NAMES},
            )


def build_cell_feature_set(
    *,
    slide_id: str,
    label_ids: np.ndarray,
    native_xy: np.ndarray,
    reference_um_xyz: np.ndarray,
    morphology: np.ndarray,
    phenotype: np.ndarray,
    supported: np.ndarray,
    radii_um: tuple[float, ...] = DEFAULT_RADII_UM,
    coordinate_scale_um: float = 256.0,
    provenance: dict[str, object],
) -> CellFeatureSet:
    """Assemble the v3 target-free feature artifact for one section."""

    morph = robust_section_standardize(morphology)
    neighborhood, neighborhood_names = multiscale_neighborhood_features(
        np.asarray(reference_um_xyz, dtype=np.float64)[:, :2],
        morph,
        radii_um=radii_um,
    )
    position, position_names = registered_position_features(
        reference_um_xyz,
        coordinate_scale_um=coordinate_scale_um,
    )
    phenotype_names = SHAPE_FEATURE_NAMES + TEXTURE_FEATURE_NAMES
    if np.asarray(phenotype).shape != (len(label_ids), len(phenotype_names)):
        raise ValueError("cell phenotype matrix does not use the v3 schema")
    morphology_names = tuple(f"uni2h_{index:04d}" for index in range(morph.shape[1]))
    return CellFeatureSet(
        slide_id=slide_id,
        label_ids=np.asarray(label_ids, dtype=np.uint32),
        native_xy=np.asarray(native_xy, dtype=np.float64),
        reference_um_xyz=np.asarray(reference_um_xyz, dtype=np.float64),
        morphology=morph,
        phenotype=np.asarray(phenotype, dtype=np.float32),
        neighborhood=neighborhood,
        position=position,
        supported=np.asarray(supported, dtype=bool),
        feature_names={
            "morphology": morphology_names,
            "phenotype": phenotype_names,
            "neighborhood": neighborhood_names,
            "position": position_names,
        },
        provenance={"feature_schema_id": FEATURE_SCHEMA_ID, **provenance},
    )


def robust_section_standardize(features: np.ndarray) -> np.ndarray:
    """Robustly normalize a stain-neutral section without target outcomes."""

    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("morphology features must be a finite matrix")
    center = np.median(values, axis=0)
    lower, upper = np.quantile(values, (0.25, 0.75), axis=0)
    scale = np.asarray(upper - lower, dtype=np.float32) / 1.349
    fallback = values.std(axis=0)
    scale = np.where(scale >= 1e-6, scale, fallback)
    scale = np.where(scale >= 1e-6, scale, 1.0)
    return ((values - center) / scale).astype(np.float32)


def multiscale_neighborhood_features(
    reference_um_xy: np.ndarray,
    morphology: np.ndarray,
    *,
    radii_um: tuple[float, ...] = DEFAULT_RADII_UM,
    neighbors: int = 16,
    projection_components: int = 32,
    chunk_size: int = 20_000,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Summarize nearby stain-neutral cells at several physical scales.

    A fixed random projection bounds memory and is identical for every section.
    It is not fitted on target outcomes or on a held-out cohort.
    """

    from scipy.spatial import cKDTree

    xy = np.asarray(reference_um_xy, dtype=np.float64)
    values = np.asarray(morphology, dtype=np.float32)
    radii = tuple(float(value) for value in radii_um)
    if (
        xy.ndim != 2
        or xy.shape != (len(values), 2)
        or values.ndim != 2
        or not np.all(np.isfinite(xy))
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("neighborhood coordinates and morphology must align")
    if not radii or any(not math.isfinite(value) or value <= 0 for value in radii):
        raise ValueError("neighborhood radii must be positive and finite")
    if tuple(sorted(set(radii))) != radii:
        raise ValueError("neighborhood radii must be unique and increasing")
    if neighbors < 1 or projection_components < 1 or chunk_size < 1:
        raise ValueError("neighborhood controls must be positive")
    projected = _fixed_projection(values, min(projection_components, values.shape[1]))
    names: list[str] = []
    for radius in radii:
        label = _radius_label(radius)
        names.extend(
            [f"r{label}_mean_{index:02d}" for index in range(projected.shape[1])]
        )
        names.extend(
            [f"r{label}_std_{index:02d}" for index in range(projected.shape[1])]
        )
        names.extend(
            (
                f"r{label}_log_count",
                f"r{label}_log_nearest_um",
                f"r{label}_log_mean_distance_um",
            )
        )
    output = np.zeros((len(values), len(names)), dtype=np.float32)
    if len(values) < 2:
        return output, tuple(names)
    tree = cKDTree(xy)
    query_count = min(neighbors + 1, len(values))
    width = projected.shape[1]
    group_width = 2 * width + 3
    for radius_index, radius in enumerate(radii):
        offset = radius_index * group_width
        for start in range(0, len(values), chunk_size):
            stop = min(start + chunk_size, len(values))
            distance, index = tree.query(
                xy[start:stop],
                k=query_count,
                distance_upper_bound=radius,
            )
            distance = np.asarray(distance, dtype=np.float64)
            index = np.asarray(index, dtype=np.int64)
            if distance.ndim == 1:
                distance, index = distance[:, None], index[:, None]
            valid = np.isfinite(distance) & (distance > 0) & (index < len(values))
            safe = np.where(valid, index, 0)
            selected = projected[safe]
            count = valid.sum(axis=1)
            divisor = np.maximum(count, 1)[:, None]
            mean = (selected * valid[..., None]).sum(axis=1) / divisor
            variance = (((selected - mean[:, None, :]) ** 2) * valid[..., None]).sum(
                axis=1
            ) / divisor
            nearest = np.min(np.where(valid, distance, np.inf), axis=1, initial=np.inf)
            nearest[~np.isfinite(nearest)] = 0
            mean_distance = np.divide(
                np.where(valid, distance, 0).sum(axis=1),
                np.maximum(count, 1),
            )
            output[start:stop, offset : offset + width] = mean
            output[start:stop, offset + width : offset + 2 * width] = np.sqrt(
                np.maximum(variance, 0)
            )
            output[start:stop, offset + 2 * width : offset + group_width] = (
                np.column_stack(
                    (np.log1p(count), np.log1p(nearest), np.log1p(mean_distance))
                )
            )
    return output, tuple(names)


def registered_position_features(
    reference_um_xyz: np.ndarray,
    *,
    coordinate_scale_um: float = 256.0,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Encode absolute registered position and tissue-relative XY location."""

    xyz = np.asarray(reference_um_xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.all(np.isfinite(xyz)):
        raise ValueError("registered position must have shape (cells, 3)")
    if not math.isfinite(coordinate_scale_um) or coordinate_scale_um <= 0:
        raise ValueError("coordinate scale must be positive and finite")
    if not len(xyz):
        return np.empty((0, 21), np.float32), _position_feature_names()
    lower = np.quantile(xyz[:, :2], 0.01, axis=0)
    upper = np.quantile(xyz[:, :2], 0.99, axis=0)
    span = np.maximum(upper - lower, 1.0)
    relative = np.clip((xyz[:, :2] - lower) / span, 0, 1) * 2 - 1
    parts = [xyz / coordinate_scale_um, relative]
    for frequency in (1.0, 2.0, 4.0, 8.0):
        angle = math.pi * frequency * relative
        parts.extend((np.sin(angle), np.cos(angle)))
    return np.concatenate(parts, axis=1).astype(np.float32), _position_feature_names()


def cell_phenotype_features(
    labels: np.ndarray,
    rgb: np.ndarray,
    *,
    pixel_size_um_xy: tuple[float, float],
    label_ids: np.ndarray | None = None,
    histogram_bins: int = 32,
    histogram_max_od: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return cell-boundary shape and target-free hematoxylin texture.

    This array implementation is used for deterministic tests and modest
    images.  Whole-slide callers should use
    :func:`stream_cell_phenotype_features`.
    """

    values = np.asarray(labels)
    image = np.asarray(rgb)
    if values.ndim != 2 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("cell labels must be a two-dimensional integer image")
    if image.shape != (*values.shape, 3):
        raise ValueError("RGB image and cell labels must share geometry")
    maximum = int(values.max(initial=0))
    state = _PhenotypeAccumulator(
        maximum,
        pixel_size_um_xy=pixel_size_um_xy,
        histogram_bins=histogram_bins,
        histogram_max_od=histogram_max_od,
    )
    state.add(values.astype(np.uint32, copy=False), image, top=0)
    ids = (
        np.asarray(label_ids, dtype=np.uint32)
        if label_ids is not None
        else np.flatnonzero(state.counts[1:] > 0).astype(np.uint32) + 1
    )
    return ids, state.finish(ids)


def stream_cell_phenotype_features(
    labels_path: Path | str,
    source_path: Path | str,
    *,
    pixel_size_um_xy: tuple[float, float],
    label_ids: np.ndarray,
    content_bbox_native_xywh: tuple[int, int, int, int] | None = None,
    stripe_height: int = 256,
) -> np.ndarray:
    """Stream native WSI and label stripes into per-cell phenotype features."""

    try:
        import pyvips
    except ImportError as error:
        raise RuntimeError(
            "whole-slide phenotype extraction requires the 'wsi' extra"
        ) from error
    if stripe_height < 2:
        raise ValueError("phenotype stripe height must be at least two pixels")
    ids = np.asarray(label_ids, dtype=np.uint32)
    if ids.ndim != 1 or np.any(ids <= 0) or len(np.unique(ids)) != len(ids):
        raise ValueError("phenotype label IDs must be unique and positive")
    labels = pyvips.Image.new_from_file(str(labels_path), access="random")
    if labels.bands > 1:
        labels = labels[0]
    source = pyvips.Image.new_from_file(str(source_path), access="random")
    source = _content_aligned_source(
        labels,
        source,
        content_bbox_native_xywh=content_bbox_native_xywh,
    )
    state = _PhenotypeAccumulator(
        int(ids.max(initial=0)),
        pixel_size_um_xy=pixel_size_um_xy,
    )
    from histopia._vips_image import normalize_vips_rgb_uchar

    for top in range(0, labels.height, stripe_height):
        height = min(stripe_height, labels.height - top)
        halo_top = max(top - 1, 0)
        halo_bottom = min(top + height + 1, labels.height)
        label_region = labels.crop(0, halo_top, labels.width, halo_bottom - halo_top)
        if label_region.format != "uint":
            label_region = label_region.cast("uint")
        label_array = np.frombuffer(
            label_region.write_to_memory(), dtype=np.uint32
        ).reshape(label_region.height, label_region.width)
        start = top - halo_top
        core_labels = label_array[start : start + height]
        rgb_region = normalize_vips_rgb_uchar(source.crop(0, top, source.width, height))
        rgb = np.frombuffer(rgb_region.write_to_memory(), dtype=np.uint8).reshape(
            height, source.width, rgb_region.bands
        )
        boundary = _boundary_mask(label_array)[start : start + height]
        state.add(core_labels, rgb, top=top, boundary=boundary)
    missing = ids[state.counts[ids] == 0]
    if len(missing):
        raise ValueError("phenotype extraction did not observe every cell label")
    return state.finish(ids)


def _content_aligned_source(
    labels: object,
    source: object,
    *,
    content_bbox_native_xywh: tuple[int, int, int, int] | None,
) -> object:
    """Return the source view in the content-local cell-label coordinates."""

    label_shape = (int(labels.width), int(labels.height))
    source_shape = (int(source.width), int(source.height))
    if label_shape == source_shape:
        return source
    if content_bbox_native_xywh is None:
        raise ValueError("native source and cell labels have different geometry")
    if len(content_bbox_native_xywh) != 4 or any(
        isinstance(value, bool) for value in content_bbox_native_xywh
    ):
        raise ValueError("native content bounding box is invalid")
    x, y, width, height = (int(value) for value in content_bbox_native_xywh)
    if (
        x < 0
        or y < 0
        or width <= 0
        or height <= 0
        or (width, height) != label_shape
        or x + width > source_shape[0]
        or y + height > source_shape[1]
    ):
        raise ValueError("native content bounding box does not match cell labels")
    return source.crop(x, y, width, height)


class _PhenotypeAccumulator:
    def __init__(
        self,
        maximum_label: int,
        *,
        pixel_size_um_xy: tuple[float, float],
        histogram_bins: int = 32,
        histogram_max_od: float = 2.0,
    ) -> None:
        if maximum_label < 0 or histogram_bins < 4 or histogram_max_od <= 0:
            raise ValueError("phenotype accumulator controls are invalid")
        if any(not math.isfinite(float(v)) or v <= 0 for v in pixel_size_um_xy):
            raise ValueError("phenotype pixel sizes must be positive and finite")
        size = maximum_label + 1
        self.pixel_size_um_xy = tuple(float(v) for v in pixel_size_um_xy)
        self.histogram_bins = histogram_bins
        self.histogram_max_od = float(histogram_max_od)
        self.counts = np.zeros(size, dtype=np.uint64)
        self.sum_h = np.zeros(size, dtype=np.float64)
        self.sum_h2 = np.zeros(size, dtype=np.float64)
        self.boundary_counts = np.zeros(size, dtype=np.uint64)
        self.boundary_sum_h = np.zeros(size, dtype=np.float64)
        self.boundary_sum_h2 = np.zeros(size, dtype=np.float64)
        self.histogram = np.zeros((size, histogram_bins), dtype=np.uint64)
        self.boundary_histogram = np.zeros((size, histogram_bins), dtype=np.uint64)
        self.perimeter_px = np.zeros(size, dtype=np.uint64)
        self.min_x = np.full(size, np.iinfo(np.int64).max, dtype=np.int64)
        self.max_x = np.full(size, -1, dtype=np.int64)
        self.min_y = np.full(size, np.iinfo(np.int64).max, dtype=np.int64)
        self.max_y = np.full(size, -1, dtype=np.int64)
        self.sum_x = np.zeros(size, dtype=np.float64)
        self.sum_y = np.zeros(size, dtype=np.float64)
        self.sum_x2 = np.zeros(size, dtype=np.float64)
        self.sum_y2 = np.zeros(size, dtype=np.float64)
        self.sum_xy = np.zeros(size, dtype=np.float64)

    def add(
        self,
        labels: np.ndarray,
        rgb: np.ndarray,
        *,
        top: int,
        boundary: np.ndarray | None = None,
    ) -> None:
        values = np.asarray(labels, dtype=np.uint32)
        image = np.asarray(rgb, dtype=np.uint8)
        if image.shape != (*values.shape, 3):
            raise ValueError("phenotype RGB stripe does not match labels")
        if int(values.max(initial=0)) >= len(self.counts):
            raise ValueError("phenotype stripe contains an undeclared label")
        boundary_mask = _boundary_mask(values) if boundary is None else boundary
        if np.asarray(boundary_mask).shape != values.shape:
            raise ValueError("phenotype boundary mask does not match labels")
        hematoxylin = _hematoxylin_concentration(image)
        flat = values.ravel().astype(np.int64, copy=False)
        self.counts += np.bincount(flat, minlength=len(self.counts)).astype(np.uint64)
        self.sum_h += np.bincount(
            flat, weights=hematoxylin.ravel(), minlength=len(self.counts)
        )
        self.sum_h2 += np.bincount(
            flat, weights=np.square(hematoxylin).ravel(), minlength=len(self.counts)
        )
        bins = np.minimum(
            np.floor(
                np.clip(hematoxylin, 0, self.histogram_max_od)
                / self.histogram_max_od
                * self.histogram_bins
            ).astype(np.int64),
            self.histogram_bins - 1,
        )
        encoded = flat * self.histogram_bins + bins.ravel()
        self.histogram += (
            np.bincount(encoded, minlength=self.histogram.size)
            .reshape(self.histogram.shape)
            .astype(np.uint64)
        )
        selected = np.asarray(boundary_mask, dtype=bool) & (values > 0)
        boundary_labels = values[selected].astype(np.int64, copy=False)
        boundary_h = hematoxylin[selected]
        self.boundary_counts += np.bincount(
            boundary_labels, minlength=len(self.counts)
        ).astype(np.uint64)
        self.perimeter_px += np.bincount(
            boundary_labels, minlength=len(self.counts)
        ).astype(np.uint64)
        self.boundary_sum_h += np.bincount(
            boundary_labels,
            weights=boundary_h,
            minlength=len(self.counts),
        )
        self.boundary_sum_h2 += np.bincount(
            boundary_labels,
            weights=np.square(boundary_h),
            minlength=len(self.counts),
        )
        boundary_bins = bins[selected]
        boundary_encoded = boundary_labels * self.histogram_bins + boundary_bins
        self.boundary_histogram += (
            np.bincount(boundary_encoded, minlength=self.boundary_histogram.size)
            .reshape(self.boundary_histogram.shape)
            .astype(np.uint64)
        )
        yy, xx = np.nonzero(values)
        if not len(xx):
            return
        labels_present = values[yy, xx].astype(np.int64, copy=False)
        absolute_y = yy.astype(np.float64) + float(top)
        x = xx.astype(np.float64)
        self.sum_x += np.bincount(labels_present, weights=x, minlength=len(self.counts))
        self.sum_y += np.bincount(
            labels_present, weights=absolute_y, minlength=len(self.counts)
        )
        self.sum_x2 += np.bincount(
            labels_present, weights=x * x, minlength=len(self.counts)
        )
        self.sum_y2 += np.bincount(
            labels_present, weights=absolute_y * absolute_y, minlength=len(self.counts)
        )
        self.sum_xy += np.bincount(
            labels_present, weights=x * absolute_y, minlength=len(self.counts)
        )
        by, bx = np.nonzero(selected)
        if len(bx):
            boundary_present = values[by, bx].astype(np.int64, copy=False)
            np.minimum.at(self.min_x, boundary_present, bx)
            np.maximum.at(self.max_x, boundary_present, bx)
            np.minimum.at(self.min_y, boundary_present, by + top)
            np.maximum.at(self.max_y, boundary_present, by + top)

    def finish(self, label_ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(label_ids, dtype=np.int64)
        count = np.maximum(self.counts[ids].astype(np.float64), 1.0)
        sx, sy = self.pixel_size_um_xy
        area = count * sx * sy
        perimeter = self.perimeter_px[ids].astype(np.float64) * math.sqrt(sx * sy)
        compactness = 4 * math.pi * area / np.maximum(perimeter**2, 1e-8)
        width = np.maximum(self.max_x[ids] - self.min_x[ids] + 1, 1)
        height = np.maximum(self.max_y[ids] - self.min_y[ids] + 1, 1)
        extent = count / np.maximum(width * height, 1)
        bbox_aspect = np.maximum(width, height) / np.maximum(
            np.minimum(width, height), 1
        )
        mean_x, mean_y = self.sum_x[ids] / count, self.sum_y[ids] / count
        var_x = np.maximum(self.sum_x2[ids] / count - mean_x**2, 0)
        var_y = np.maximum(self.sum_y2[ids] / count - mean_y**2, 0)
        covariance = self.sum_xy[ids] / count - mean_x * mean_y
        root = np.sqrt(np.maximum((var_x - var_y) ** 2 + 4 * covariance**2, 0))
        major = np.maximum((var_x + var_y + root) / 2, 1e-8)
        minor = np.maximum((var_x + var_y - root) / 2, 1e-8)
        eccentricity = np.sqrt(np.maximum(1 - minor / major, 0))
        moment_aspect = np.sqrt(major / minor)
        orientation = 0.5 * np.arctan2(2 * covariance, var_x - var_y)
        shape = np.column_stack(
            (
                np.log1p(area),
                np.log1p(perimeter),
                compactness,
                extent,
                bbox_aspect,
                eccentricity,
                moment_aspect,
                np.sin(orientation),
                np.cos(orientation),
            )
        )
        mean_h = self.sum_h[ids] / count
        std_h = np.sqrt(np.maximum(self.sum_h2[ids] / count - mean_h**2, 0))
        quantiles = _histogram_quantiles(
            self.histogram[ids],
            (0.25, 0.5, 0.75, 0.9),
            maximum=self.histogram_max_od,
        )
        probabilities = self.histogram[ids] / count[:, None]
        logarithm = np.zeros_like(probabilities)
        np.log(probabilities, out=logarithm, where=probabilities > 0)
        entropy = -np.sum(probabilities * logarithm, axis=1)
        boundary_count = np.maximum(self.boundary_counts[ids].astype(np.float64), 1.0)
        boundary_mean = self.boundary_sum_h[ids] / boundary_count
        boundary_std = np.sqrt(
            np.maximum(
                self.boundary_sum_h2[ids] / boundary_count - boundary_mean**2,
                0,
            )
        )
        boundary_q90 = _histogram_quantiles(
            self.boundary_histogram[ids],
            (0.9,),
            maximum=self.histogram_max_od,
        )[:, 0]
        texture = np.column_stack(
            (
                mean_h,
                std_h,
                quantiles,
                entropy,
                boundary_mean,
                boundary_std,
                boundary_q90,
            )
        )
        result = np.concatenate((shape, texture), axis=1).astype(np.float32)
        if not np.all(np.isfinite(result)):
            raise ValueError("cell phenotype extraction produced non-finite values")
        return result


def _hematoxylin_concentration(rgb: np.ndarray) -> np.ndarray:
    values = np.asarray(rgb, dtype=np.float32)
    optical_density = -np.log((values + 1.0) / 256.0)
    basis = np.asarray(
        [[0.650, 0.268], [0.704, 0.570], [0.286, 0.776]], dtype=np.float32
    )
    coefficient = np.linalg.pinv(basis)[0]
    return np.maximum(optical_density @ coefficient, 0).astype(np.float32)


def _boundary_mask(labels: np.ndarray) -> np.ndarray:
    values = np.asarray(labels)
    if values.ndim != 2:
        raise ValueError("boundary labels must be two-dimensional")
    padded = np.pad(values, 1, mode="constant")
    center = padded[1:-1, 1:-1]
    return (center > 0) & (
        (center != padded[:-2, 1:-1])
        | (center != padded[2:, 1:-1])
        | (center != padded[1:-1, :-2])
        | (center != padded[1:-1, 2:])
    )


def _histogram_quantiles(
    histogram: np.ndarray,
    quantiles: tuple[float, ...],
    *,
    maximum: float,
) -> np.ndarray:
    values = np.asarray(histogram, dtype=np.uint64)
    cumulative = np.cumsum(values, axis=1)
    total = np.maximum(cumulative[:, -1], 1)
    output = np.empty((len(values), len(quantiles)), dtype=np.float64)
    for column, quantile in enumerate(quantiles):
        threshold = np.ceil(total * quantile).astype(np.uint64)
        index = np.argmax(cumulative >= threshold[:, None], axis=1)
        output[:, column] = (index + 0.5) / values.shape[1] * maximum
    return output


def _fixed_projection(features: np.ndarray, components: int) -> np.ndarray:
    if components == features.shape[1]:
        projected = np.asarray(features, dtype=np.float32)
    else:
        rng = np.random.default_rng(0x48495354)
        matrix = rng.normal(
            0,
            1 / math.sqrt(components),
            size=(features.shape[1], components),
        ).astype(np.float32)
        projected = np.asarray(features, dtype=np.float32) @ matrix
    center = projected.mean(axis=0)
    scale = projected.std(axis=0)
    scale[scale < 1e-6] = 1
    return ((projected - center) / scale).astype(np.float32)


def _position_feature_names() -> tuple[str, ...]:
    names = ["registered_x", "registered_y", "registered_z", "relative_x", "relative_y"]
    for frequency in (1, 2, 4, 8):
        names.extend(
            (
                f"relative_x_sin_f{frequency}",
                f"relative_y_sin_f{frequency}",
                f"relative_x_cos_f{frequency}",
                f"relative_y_cos_f{frequency}",
            )
        )
    return tuple(names)


def _radius_label(value: float) -> str:
    return str(int(value)) if value.is_integer() else str(value).replace(".", "p")


_FEATURE_ARRAY_NAMES = (
    "label_ids",
    "native_xy",
    "reference_um_xyz",
    "morphology",
    "phenotype",
    "neighborhood",
    "position",
    "supported",
)


def _feature_arrays(features: CellFeatureSet) -> dict[str, np.ndarray]:
    dtypes = {
        "label_ids": np.uint32,
        "native_xy": np.float64,
        "reference_um_xyz": np.float64,
        "morphology": np.float32,
        "phenotype": np.float32,
        "neighborhood": np.float32,
        "position": np.float32,
        "supported": np.bool_,
    }
    return {
        name: np.asarray(getattr(features, name), dtype=dtypes[name])
        for name in _FEATURE_ARRAY_NAMES
    }


def _feature_fingerprint(features: CellFeatureSet) -> str:
    digest = hashlib.sha256(b"histopia-cell-multiscale-features-v3\0")
    metadata = {
        "feature_schema_id": FEATURE_SCHEMA_ID,
        "slide_id": features.slide_id,
        "feature_names": {
            key: list(value) for key, value in sorted(features.feature_names.items())
        },
        "provenance": features.provenance,
    }
    digest.update(_canonical_json(metadata).encode())
    for name, value in sorted(_feature_arrays(features).items()):
        array = np.ascontiguousarray(value)
        digest.update(name.encode())
        digest.update(array.dtype.str.encode())
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _validate_path_free(value: Any, *, key: str = "provenance") -> None:
    if isinstance(value, Path):
        raise ValueError(f"{key} must not contain filesystem paths")
    if isinstance(value, dict):
        for child_key, child in value.items():
            text = str(child_key)
            if text.lower().endswith(("_path", "_dir", "_root")):
                raise ValueError(f"{key} must not contain filesystem path fields")
            _validate_path_free(child, key=f"{key}.{text}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_path_free(child, key=f"{key}[{index}]")
    elif isinstance(value, str) and (value.startswith("/") or ":\\" in value):
        raise ValueError(f"{key} must not contain absolute filesystem paths")


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))
