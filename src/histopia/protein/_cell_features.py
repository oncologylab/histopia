"""Native-resolution, stain-neutral UNI2-h features for individual cells."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from histopia._atomic import write_binary_atomic


class SpatialTokenEncoder(Protocol):
    """Minimal interface required from a dense pathology encoder."""

    model_fingerprint: str

    def encode_tokens(self, images: np.ndarray) -> np.ndarray: ...


PatchRequest = tuple[int, int, int, int, int]
PatchReader = Callable[[int, int, int, int, int], np.ndarray]
Neutralizer = Callable[[np.ndarray, tuple[int, int, int, int]], np.ndarray]


class VipsPatchReader:
    """Read bounded native WSI regions and resize them to encoder input."""

    provenance_id = "pyvips-native-random-v1"

    def __init__(self, path: Path | str) -> None:
        try:
            import pyvips
        except ImportError as error:
            raise RuntimeError(
                "cell token extraction requires the 'wsi' extra"
            ) from error
        self._image = pyvips.Image.new_from_file(str(path), access="random")

    def __call__(
        self, x: int, y: int, width: int, height: int, output_px: int
    ) -> np.ndarray:
        from histopia._vips_image import normalize_vips_rgb_uchar

        tile = self._image.crop(x, y, width, height).resize(
            output_px / width,
            vscale=output_px / height,
        )
        tile = normalize_vips_rgb_uchar(tile)
        return np.frombuffer(tile.write_to_memory(), dtype=np.uint8).reshape(
            tile.height, tile.width, tile.bands
        )

    def read_many(self, requests: Sequence[PatchRequest]) -> tuple[np.ndarray, ...]:
        """Decode neighboring crops from one resized row strip."""

        groups: dict[tuple[int, int, int, int], list[tuple[int, PatchRequest]]] = {}
        for index, request in enumerate(requests):
            x, y, width, height, output_px = request
            groups.setdefault((y, width, height, output_px), []).append(
                (index, request)
            )
        patches: list[np.ndarray | None] = [None] * len(requests)
        for (y, width, height, output_px), items in groups.items():
            minimum = min(request[0] for _, request in items)
            maximum = max(request[0] + width for _, request in items)
            strip = self._image.crop(minimum, y, maximum - minimum, height).resize(
                output_px / width,
                vscale=output_px / height,
            )
            from histopia._vips_image import normalize_vips_rgb_uchar

            strip = normalize_vips_rgb_uchar(strip)
            array = np.frombuffer(strip.write_to_memory(), dtype=np.uint8).reshape(
                strip.height, strip.width, strip.bands
            )
            for index, request in items:
                start = round((request[0] - minimum) * output_px / width)
                patch = array[:, start : start + output_px]
                patches[index] = (
                    patch
                    if patch.shape == (output_px, output_px, 3)
                    else self(*request)
                )
        if any(patch is None for patch in patches):
            raise RuntimeError("batch WSI reader did not fill every crop")
        return tuple(patch for patch in patches if patch is not None)


def neutralize_hdab_morphology(
    image: np.ndarray, _crop: tuple[int, int, int, int]
) -> np.ndarray:
    """Render high-resolution hematoxylin morphology without target DAB OD.

    The transform uses a fixed H-DAB optical-density basis and returns only
    the hematoxylin component in a fixed palette.  Per-patch robust scaling
    suppresses slide-wide intensity and color differences; the target DAB
    component is never included in the encoder image.
    """

    rgb = np.asarray(image, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("morphology neutralization requires RGB input")
    optical_density = -np.log((rgb + 1.0) / 256.0)
    # Columns are normalized Ruifrok hematoxylin and DAB stain vectors.
    basis = np.asarray(
        [[0.650, 0.268], [0.704, 0.570], [0.286, 0.776]], dtype=np.float32
    )
    concentration = optical_density.reshape(-1, 3) @ np.linalg.pinv(basis).T
    hematoxylin = np.maximum(concentration[:, 0], 0).reshape(rgb.shape[:2])
    positive = hematoxylin[hematoxylin > 0]
    maximum = float(np.quantile(positive, 0.99)) if positive.size else 1.0
    normalized = np.clip(hematoxylin / max(maximum, 1e-6), 0, 1)[..., None]
    white = np.asarray([255.0, 255.0, 255.0], dtype=np.float32)
    purple = np.asarray([72.0, 54.0, 122.0], dtype=np.float32)
    return np.rint(white * (1 - normalized) + purple * normalized).astype(np.uint8)


def label_centroids_from_tiff(
    path: Path | str, *, stripe_height: int = 512
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stream exact native label areas and centroids without loading the WSI."""

    if stripe_height <= 0:
        raise ValueError("stripe_height must be positive")
    try:
        import pyvips
    except ImportError as error:
        raise RuntimeError("label geometry requires the 'wsi' extra") from error
    image = pyvips.Image.new_from_file(str(path), access="sequential")
    if image.bands > 1:
        image = image[0]
    maximum = int(image.max())
    image = pyvips.Image.new_from_file(str(path), access="sequential")
    if image.bands > 1:
        image = image[0]
    counts = np.zeros(maximum + 1, dtype=np.uint64)
    sum_x = np.zeros(maximum + 1, dtype=np.float64)
    sum_y = np.zeros(maximum + 1, dtype=np.float64)
    x_coordinates = np.arange(image.width, dtype=np.float64)
    for top in range(0, image.height, stripe_height):
        height = min(stripe_height, image.height - top)
        stripe = image.crop(0, top, image.width, height)
        if stripe.format != "uint":
            stripe = stripe.cast("uint")
        labels = np.frombuffer(stripe.write_to_memory(), dtype=np.uint32).reshape(
            height, image.width
        )
        flat = labels.ravel().astype(np.int64, copy=False)
        counts += np.bincount(flat, minlength=maximum + 1).astype(np.uint64)
        sum_x += np.bincount(
            flat,
            weights=np.broadcast_to(x_coordinates, labels.shape).ravel(),
            minlength=maximum + 1,
        )
        sum_y += np.bincount(
            flat,
            weights=np.broadcast_to(
                np.arange(top, top + height, dtype=np.float64)[:, None],
                labels.shape,
            ).ravel(),
            minlength=maximum + 1,
        )
    present = np.flatnonzero(counts[1:] > 0) + 1
    xy = np.column_stack(
        (sum_x[present] / counts[present], sum_y[present] / counts[present])
    )
    return present.astype(np.uint32), xy, counts[present]


@dataclass(frozen=True, slots=True)
class CellTokenFeatures:
    """One genuinely spatial UNI2-h representation per native cell."""

    slide_id: str
    label_ids: np.ndarray
    native_xy: np.ndarray
    features: np.ndarray
    support_weight: np.ndarray
    analysis_mpp: float
    token_spacing_um: float
    provenance: dict[str, object]
    fingerprint: str | None = None

    def __post_init__(self) -> None:
        labels = np.asarray(self.label_ids)
        count = len(labels)
        if labels.shape != (count,) or not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("cell token label_ids must be an integer vector")
        if np.any(labels <= 0) or len(np.unique(labels)) != count:
            raise ValueError("cell token label_ids must be unique and positive")
        if np.asarray(self.native_xy).shape != (count, 2):
            raise ValueError("cell token native_xy must have shape (cells, 2)")
        values = np.asarray(self.features)
        if values.ndim != 2 or values.shape[0] != count:
            raise ValueError("cell token features must have shape (cells, features)")
        weights = np.asarray(self.support_weight, dtype=float)
        if weights.shape != (count,) or np.any(weights < 0):
            raise ValueError("cell token support weights must be nonnegative")
        if not np.all(np.isfinite(self.native_xy)) or not np.all(np.isfinite(values)):
            raise ValueError("cell token coordinates and features must be finite")
        for name in ("analysis_mpp", "token_spacing_um"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        expected = _feature_fingerprint(self)
        if self.fingerprint is not None and self.fingerprint != expected:
            raise ValueError("cell token feature fingerprint does not match")
        object.__setattr__(self, "fingerprint", expected)

    @property
    def supported(self) -> np.ndarray:
        return np.asarray(self.support_weight) > 0

    def save(self, path: Path | str) -> Path:
        target = Path(path)
        metadata = {
            "schema_version": 1,
            "slide_id": self.slide_id,
            "analysis_mpp": self.analysis_mpp,
            "token_spacing_um": self.token_spacing_um,
            "provenance": self.provenance,
            "fingerprint": self.fingerprint,
        }

        def writer(stream) -> None:
            np.savez_compressed(
                stream,
                metadata_json=np.asarray(_canonical_json(metadata)),
                label_ids=np.asarray(self.label_ids, dtype=np.uint32),
                native_xy=np.asarray(self.native_xy, dtype=np.float64),
                features=np.asarray(self.features, dtype=np.float16),
                support_weight=np.asarray(self.support_weight, dtype=np.float32),
            )

        return write_binary_atomic(target, writer)

    @classmethod
    def load(cls, path: Path | str) -> CellTokenFeatures:
        with np.load(Path(path), allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"]))
            if metadata.get("schema_version") != 1:
                raise ValueError("unsupported cell token feature schema")
            return cls(
                slide_id=str(metadata["slide_id"]),
                label_ids=data["label_ids"],
                native_xy=data["native_xy"],
                features=np.asarray(data["features"], dtype=np.float32),
                support_weight=data["support_weight"],
                analysis_mpp=float(metadata["analysis_mpp"]),
                token_spacing_um=float(metadata["token_spacing_um"]),
                provenance=dict(metadata["provenance"]),
                fingerprint=str(metadata["fingerprint"]),
            )


@dataclass(frozen=True, slots=True)
class CellTokenFeatureIndex:
    """Small, array-free index for an immutable cell-token artifact."""

    slide_id: str
    analysis_mpp: float
    token_spacing_um: float
    provenance: dict[str, object]
    fingerprint: str


def cell_token_feature_index(path: Path | str) -> CellTokenFeatureIndex:
    """Read and validate token metadata without inflating feature arrays.

    This index is intended for cache scheduling and coverage audits. Scientific
    consumers must continue to use :meth:`CellTokenFeatures.load`, which also
    recomputes the content fingerprint over every array.
    """

    with np.load(Path(path), allow_pickle=False) as data:
        if "metadata_json" not in data.files:
            raise ValueError("cell token feature metadata is missing")
        metadata = json.loads(str(data["metadata_json"]))
    if not isinstance(metadata, dict) or metadata.get("schema_version") != 1:
        raise ValueError("unsupported cell token feature schema")
    required = {
        "schema_version",
        "slide_id",
        "analysis_mpp",
        "token_spacing_um",
        "provenance",
        "fingerprint",
    }
    if set(metadata) != required:
        raise ValueError("cell token feature metadata fields differ")
    slide_id = metadata["slide_id"]
    provenance = metadata["provenance"]
    fingerprint = metadata["fingerprint"]
    if not isinstance(slide_id, str) or not slide_id:
        raise ValueError("cell token feature slide ID is invalid")
    if not isinstance(provenance, dict):
        raise ValueError("cell token feature provenance is invalid")
    if (
        not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in fingerprint)
    ):
        raise ValueError("cell token feature fingerprint is invalid")
    analysis_mpp = float(metadata["analysis_mpp"])
    token_spacing_um = float(metadata["token_spacing_um"])
    if any(
        not math.isfinite(value) or value <= 0
        for value in (analysis_mpp, token_spacing_um)
    ):
        raise ValueError("cell token feature sampling is invalid")
    return CellTokenFeatureIndex(
        slide_id=slide_id,
        analysis_mpp=analysis_mpp,
        token_spacing_um=token_spacing_um,
        provenance=dict(provenance),
        fingerprint=fingerprint,
    )


def validate_cell_token_feature_binding(
    path: Path | str,
    *,
    slide_id: str,
    cell_result_fingerprint: str,
    source_identity: str,
    mask_sha256: str,
    analysis_mpp: float = 0.5,
) -> CellTokenFeatureIndex:
    """Require a cached token artifact to match its exact current inputs."""

    index = cell_token_feature_index(path)
    expected = {
        "source_identity": source_identity,
        "mask_sha256": mask_sha256,
        "cell_result_fingerprint": cell_result_fingerprint,
        "feature_view": "native-hdab-neutral-spatial-uni2h-v2",
        "view": "native-stain-neutral-uni2h-spatial-tokens-v1",
        "crop_size_px": 224,
        "crop_stride_px": 112,
        "tissue_masked": True,
    }
    if index.slide_id != slide_id:
        raise ValueError("cell token feature is bound to a different slide")
    if not np.isclose(index.analysis_mpp, analysis_mpp, rtol=0, atol=1e-9):
        raise ValueError("cell token feature analysis resolution differs")
    for key, value in expected.items():
        if index.provenance.get(key) != value:
            raise ValueError(f"cell token feature {key} binding differs")
    model_fingerprint = index.provenance.get("model_fingerprint")
    if not isinstance(model_fingerprint, str) or not model_fingerprint:
        raise ValueError("cell token feature model fingerprint is missing")
    return index


def extract_cell_token_features(
    *,
    slide_id: str,
    content_bbox_native_xywh: tuple[int, int, int, int],
    source_mpp_xy: tuple[float, float],
    tissue_mask: np.ndarray,
    native_to_mask: np.ndarray,
    label_ids: np.ndarray,
    cell_native_xy: np.ndarray,
    reader: PatchReader,
    neutralizer: Neutralizer,
    encoder: SpatialTokenEncoder,
    analysis_mpp: float = 0.5,
    crop_size_px: int = 224,
    crop_stride_px: int = 112,
    minimum_tissue_fraction: float = 0.10,
    batch_size: int = 16,
    provenance: dict[str, object] | None = None,
) -> CellTokenFeatures:
    """Extract overlapping dense tokens and interpolate them into cells.

    Each cell receives position-specific patch tokens from every overlapping
    crop that contains its centroid.  A Hann weight blends crop edges, which
    prevents the block discontinuities created by nearest global embeddings.
    """

    if analysis_mpp <= 0 or crop_size_px <= 0 or crop_stride_px <= 0:
        raise ValueError("feature extraction sizes must be positive")
    if crop_stride_px > crop_size_px:
        raise ValueError("crop stride cannot exceed crop size")
    if batch_size <= 0 or not 0 <= minimum_tissue_fraction <= 1:
        raise ValueError("batch size and tissue fraction are invalid")
    labels = np.asarray(label_ids, dtype=np.uint32)
    cells = np.asarray(cell_native_xy, dtype=np.float64)
    if labels.ndim != 1 or cells.shape != (len(labels), 2):
        raise ValueError("cell labels and native coordinates must align")
    if len(np.unique(labels)) != len(labels) or np.any(labels <= 0):
        raise ValueError("cell labels must be unique and positive")
    mask = np.asarray(tissue_mask, dtype=bool)
    matrix = np.asarray(native_to_mask, dtype=np.float64)
    if mask.ndim != 2 or matrix.shape != (3, 3):
        raise ValueError("tissue mask geometry is invalid")
    if not len(cells):
        raise ValueError("cell token extraction requires at least one cell")

    crop_um = crop_size_px * analysis_mpp
    stride_um = crop_stride_px * analysis_mpp
    crop_width = max(1, int(round(crop_um / source_mpp_xy[0])))
    crop_height = max(1, int(round(crop_um / source_mpp_xy[1])))
    stride_x = max(1, int(round(stride_um / source_mpp_xy[0])))
    stride_y = max(1, int(round(stride_um / source_mpp_xy[1])))
    requests = tuple(
        _accepted_crop_requests(
            content_bbox_native_xywh,
            crop_shape=(crop_width, crop_height),
            stride_xy=(stride_x, stride_y),
            mask=mask,
            native_to_mask=matrix,
            minimum_tissue_fraction=minimum_tissue_fraction,
            output_px=crop_size_px,
        )
    )
    if not requests:
        raise ValueError(f"no tissue crops passed coverage for {slide_id}")

    from scipy.spatial import cKDTree

    tree = cKDTree(cells)
    sums: np.ndarray | None = None
    weights = np.zeros(len(cells), dtype=np.float32)
    read_many = getattr(reader, "read_many", None)
    for start in range(0, len(requests), batch_size):
        batch_requests = requests[start : start + batch_size]
        raw = (
            tuple(read_many(batch_requests))
            if callable(read_many)
            else tuple(reader(*request) for request in batch_requests)
        )
        images = np.stack(
            [
                _validated_neutral_image(
                    neutralizer(
                        image,
                        (request[0], request[1], request[2], request[3]),
                    ),
                    crop_size_px,
                )
                for image, request in zip(raw, batch_requests, strict=True)
            ]
        )
        token_batch = np.asarray(encoder.encode_tokens(images), dtype=np.float32)
        if token_batch.ndim != 4 or token_batch.shape[0] != len(batch_requests):
            raise ValueError("encoder must return one spatial token grid per crop")
        if token_batch.shape[1] != token_batch.shape[2]:
            raise ValueError("encoder token lattice must be square")
        if sums is None:
            sums = np.zeros((len(cells), token_batch.shape[-1]), dtype=np.float32)
        elif sums.shape[1] != token_batch.shape[-1]:
            raise ValueError("encoder token width changed between batches")
        for request, token_grid in zip(batch_requests, token_batch, strict=True):
            crop = (request[0], request[1], request[2], request[3])
            indices = _cells_inside_crop(tree, cells, crop)
            if not len(indices):
                continue
            sampled, blend = interpolate_token_grid(
                token_grid, crop_bbox_native_xywh=crop, cell_native_xy=cells[indices]
            )
            sums[indices] += sampled * blend[:, None]
            weights[indices] += blend
    if sums is None:
        raise RuntimeError("cell token extraction produced no encoded crops")
    features = np.divide(
        sums,
        weights[:, None],
        out=np.zeros_like(sums),
        where=weights[:, None] > 0,
    )
    token_spacing_um = crop_um / token_batch.shape[1]
    return CellTokenFeatures(
        slide_id=slide_id,
        label_ids=labels,
        native_xy=cells,
        features=features,
        support_weight=weights,
        analysis_mpp=analysis_mpp,
        token_spacing_um=token_spacing_um,
        provenance={
            **(provenance or {}),
            "model_fingerprint": encoder.model_fingerprint,
            "view": "native-stain-neutral-uni2h-spatial-tokens-v1",
            "crop_size_px": crop_size_px,
            "crop_stride_px": crop_stride_px,
            "crop_count": len(requests),
            "tissue_masked": True,
        },
    )


def interpolate_token_grid(
    token_grid: np.ndarray,
    *,
    crop_bbox_native_xywh: tuple[int, int, int, int],
    cell_native_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Bilinearly sample spatial tokens and return seam-suppressing weights."""

    tokens = np.asarray(token_grid, dtype=np.float32)
    cells = np.asarray(cell_native_xy, dtype=np.float64)
    if tokens.ndim != 3 or tokens.shape[0] != tokens.shape[1]:
        raise ValueError("token_grid must have shape (side, side, features)")
    if cells.ndim != 2 or cells.shape[1] != 2:
        raise ValueError("cell_native_xy must have shape (cells, 2)")
    left, top, width, height = crop_bbox_native_xywh
    if width <= 0 or height <= 0:
        raise ValueError("crop dimensions must be positive")
    side_y, side_x = tokens.shape[:2]
    ux = (cells[:, 0] - left) / width
    uy = (cells[:, 1] - top) / height
    if np.any((ux < 0) | (ux > 1) | (uy < 0) | (uy > 1)):
        raise ValueError("cell coordinates must stay inside their crop")
    tx = np.clip(ux * side_x - 0.5, 0, side_x - 1)
    ty = np.clip(uy * side_y - 0.5, 0, side_y - 1)
    x0 = np.floor(tx).astype(np.int64)
    y0 = np.floor(ty).astype(np.int64)
    x1 = np.minimum(x0 + 1, side_x - 1)
    y1 = np.minimum(y0 + 1, side_y - 1)
    wx = (tx - x0).astype(np.float32)
    wy = (ty - y0).astype(np.float32)
    sampled = (
        tokens[y0, x0] * ((1 - wx) * (1 - wy))[:, None]
        + tokens[y0, x1] * (wx * (1 - wy))[:, None]
        + tokens[y1, x0] * ((1 - wx) * wy)[:, None]
        + tokens[y1, x1] * (wx * wy)[:, None]
    )
    blend = np.maximum(
        np.sin(np.pi * np.clip(ux, 0, 1)) ** 2 * np.sin(np.pi * np.clip(uy, 0, 1)) ** 2,
        1e-3,
    ).astype(np.float32)
    return sampled.astype(np.float32), blend


def cell_shape_features(
    labels: np.ndarray, *, pixel_size_um_xy: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Return label IDs and compact physical shape descriptors."""

    from scipy import ndimage as ndi

    values = np.asarray(labels)
    if values.ndim != 2 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("cell labels must be a two-dimensional integer image")
    if any(not math.isfinite(float(v)) or float(v) <= 0 for v in pixel_size_um_xy):
        raise ValueError("pixel sizes must be positive and finite")
    maximum = int(values.max(initial=0))
    if maximum == 0:
        return np.empty(0, np.uint32), np.empty((0, 5), np.float32)
    area_px = np.bincount(values.ravel(), minlength=maximum + 1).astype(float)
    present = np.flatnonzero(area_px[1:] > 0) + 1
    boundary = (values > 0) & (values != ndi.grey_erosion(values, size=(3, 3)))
    perimeter_px = np.bincount(
        values[boundary].astype(np.int64), minlength=maximum + 1
    ).astype(float)
    boxes = ndi.find_objects(values)
    width = np.zeros(maximum + 1, dtype=float)
    height = np.zeros(maximum + 1, dtype=float)
    for label, box in enumerate(boxes, start=1):
        if box is None:
            continue
        height[label] = box[0].stop - box[0].start
        width[label] = box[1].stop - box[1].start
    px_area = float(pixel_size_um_xy[0] * pixel_size_um_xy[1])
    area_um2 = area_px[present] * px_area
    perimeter_um = perimeter_px[present] * math.sqrt(px_area)
    compactness = np.divide(
        4 * math.pi * area_um2,
        np.maximum(perimeter_um**2, 1e-8),
    )
    bbox_area = width[present] * height[present]
    extent = np.divide(area_px[present], np.maximum(bbox_area, 1))
    aspect = np.maximum(width[present], height[present]) / np.maximum(
        np.minimum(width[present], height[present]), 1
    )
    features = np.column_stack(
        (np.log1p(area_um2), np.log1p(perimeter_um), compactness, extent, aspect)
    )
    return present.astype(np.uint32), features.astype(np.float32)


def cell_neighborhood_features(
    reference_um_xy: np.ndarray,
    shape_features: np.ndarray,
    *,
    neighbors: int = 16,
) -> np.ndarray:
    """Summarize nearby cell density and morphology without target outcomes."""

    from scipy.spatial import cKDTree

    xy = np.asarray(reference_um_xy, dtype=np.float64)
    shape = np.asarray(shape_features, dtype=np.float32)
    if xy.ndim != 2 or xy.shape[1] != 2 or shape.ndim != 2 or len(shape) != len(xy):
        raise ValueError("cell coordinates and shape features must align")
    if neighbors < 1:
        raise ValueError("neighbors must be positive")
    if not len(xy):
        return np.empty((0, shape.shape[1] + 3), dtype=np.float32)
    count = min(neighbors + 1, len(xy))
    distance, index = cKDTree(xy).query(xy, k=count)
    if count == 1:
        distance = np.zeros((len(xy), 1))
        index = np.arange(len(xy))[:, None]
    distance = np.asarray(distance)[:, 1:]
    index = np.asarray(index)[:, 1:]
    if not distance.shape[1]:
        return np.column_stack(
            (np.zeros((len(xy), 3), dtype=np.float32), shape)
        ).astype(np.float32)
    radius = distance[:, -1]
    density = distance.shape[1] / (math.pi * np.maximum(radius, 1.0) ** 2)
    mean_distance = distance.mean(axis=1)
    nearest_distance = distance[:, 0]
    neighborhood_shape = shape[index].mean(axis=1)
    return np.column_stack(
        (
            np.log1p(density),
            np.log1p(nearest_distance),
            np.log1p(mean_distance),
            neighborhood_shape,
        )
    ).astype(np.float32)


def _accepted_crop_requests(
    content_bbox: tuple[int, int, int, int],
    *,
    crop_shape: tuple[int, int],
    stride_xy: tuple[int, int],
    mask: np.ndarray,
    native_to_mask: np.ndarray,
    minimum_tissue_fraction: float,
    output_px: int,
) -> Sequence[PatchRequest]:
    x0, y0, content_width, content_height = content_bbox
    crop_width, crop_height = crop_shape
    if crop_width > content_width or crop_height > content_height:
        return ()
    xs = _covering_starts(x0, content_width, crop_width, stride_xy[0])
    ys = _covering_starts(y0, content_height, crop_height, stride_xy[1])
    requests = []
    for top in ys:
        for left in xs:
            if (
                _mask_fraction(
                    mask,
                    native_to_mask,
                    left,
                    top,
                    crop_width,
                    crop_height,
                )
                >= minimum_tissue_fraction
            ):
                requests.append((left, top, crop_width, crop_height, output_px))
    return requests


def _covering_starts(
    origin: int, length: int, window: int, stride: int
) -> tuple[int, ...]:
    last = origin + length - window
    starts = list(range(origin, last + 1, stride))
    if not starts or starts[-1] != last:
        starts.append(last)
    return tuple(starts)


def _mask_fraction(
    mask: np.ndarray,
    matrix: np.ndarray,
    left: int,
    top: int,
    width: int,
    height: int,
) -> float:
    corners = np.asarray([[left, top, 1.0], [left + width, top + height, 1.0]])
    mapped = (matrix @ corners.T).T
    mapped = mapped[:, :2] / mapped[:, 2, None]
    x0, y0 = np.floor(mapped[0]).astype(int)
    x1, y1 = np.ceil(mapped[1]).astype(int)
    x0, x1 = np.clip((x0, x1), 0, mask.shape[1])
    y0, y1 = np.clip((y0, y1), 0, mask.shape[0])
    return float(mask[y0:y1, x0:x1].mean()) if x1 > x0 and y1 > y0 else 0.0


def _cells_inside_crop(
    tree, cells: np.ndarray, crop: tuple[int, int, int, int]
) -> np.ndarray:
    left, top, width, height = crop
    center = (left + width / 2, top + height / 2)
    candidates = np.asarray(
        tree.query_ball_point(center, math.hypot(width, height) / 2), dtype=np.int64
    )
    if not len(candidates):
        return candidates
    selected = cells[candidates]
    inside = (
        (selected[:, 0] >= left)
        & (selected[:, 0] <= left + width)
        & (selected[:, 1] >= top)
        & (selected[:, 1] <= top + height)
    )
    return candidates[inside]


def _validated_neutral_image(image: np.ndarray, crop_size_px: int) -> np.ndarray:
    values = np.asarray(image)
    if values.shape != (crop_size_px, crop_size_px, 3) or values.dtype != np.uint8:
        raise ValueError("neutralizer must return a uint8 RGB crop at encoder size")
    return values


def _feature_fingerprint(features: CellTokenFeatures) -> str:
    digest = hashlib.sha256(b"histopia-cell-token-features-v1\0")
    digest.update(
        _canonical_json(
            {
                "slide_id": features.slide_id,
                "analysis_mpp": features.analysis_mpp,
                "token_spacing_um": features.token_spacing_um,
                "provenance": features.provenance,
            }
        ).encode()
    )
    for name, value, dtype in (
        ("label_ids", features.label_ids, np.uint32),
        ("native_xy", features.native_xy, np.float64),
        ("features", features.features, np.float16),
        ("support_weight", features.support_weight, np.float32),
    ):
        array = np.ascontiguousarray(value, dtype=dtype)
        digest.update(name.encode())
        digest.update(array.dtype.str.encode())
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))
