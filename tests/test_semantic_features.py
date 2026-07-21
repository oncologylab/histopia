from __future__ import annotations

from pathlib import Path

import numpy as np

from histopia.registration import SlideGeometry
from histopia.semantic import PatchFeatures
from histopia.semantic._features import (
    extract_patch_features,
    map_native_to_reference_um,
)


def test_patch_features_round_trip_compact_npz(tmp_path: Path) -> None:
    artifact = PatchFeatures(
        slide_id="section.ndpi",
        features=np.arange(12, dtype=np.float32).reshape(3, 4),
        grid_rc=np.array([[0, 0], [0, 1], [1, 0]], dtype=np.int32),
        native_xy=np.array([[10, 20], [30, 20], [10, 40]], dtype=np.float64),
        reference_um_xy=np.array([[5, 10], [15, 10], [5, 20]], dtype=np.float64),
        tissue_fraction=np.array([0.9, 0.8, 0.7], dtype=np.float32),
        grid_shape=(2, 2),
        patch_size_px=224,
        analysis_mpp=0.5,
    )

    path = artifact.save(tmp_path / "features.npz")
    loaded = PatchFeatures.load(path)

    assert loaded.slide_id == artifact.slide_id
    assert loaded.grid_shape == (2, 2)
    assert loaded.features.dtype == np.float16
    np.testing.assert_allclose(loaded.features, artifact.features)
    np.testing.assert_array_equal(loaded.grid_rc, artifact.grid_rc)
    np.testing.assert_allclose(loaded.reference_um_xy, artifact.reference_um_xy)


def test_native_centers_map_through_thumbnail_registration_to_reference_um() -> None:
    native_xy = np.array([[100.0, 200.0], [300.0, 400.0]])
    native_to_thumbnail = np.array(
        [[0.1, 0.0, -5.0], [0.0, 0.2, -10.0], [0.0, 0.0, 1.0]]
    )
    moving_to_reference_thumbnail = np.array(
        [[2.0, 0.0, 3.0], [0.0, 0.5, 7.0], [0.0, 0.0, 1.0]]
    )
    reference_thumbnail_to_native = np.array(
        [[4.0, 0.0, 20.0], [0.0, 5.0, 30.0], [0.0, 0.0, 1.0]]
    )

    result = map_native_to_reference_um(
        native_xy,
        native_to_thumbnail=native_to_thumbnail,
        moving_to_reference_thumbnail=moving_to_reference_thumbnail,
        reference_thumbnail_to_native=reference_thumbnail_to_native,
        reference_mpp_xy=(0.5, 0.25),
    )

    expected = np.array([[36.0, 35.0], [116.0, 60.0]])
    np.testing.assert_allclose(result, expected)


def test_patch_features_rejects_misaligned_rows() -> None:
    with np.testing.assert_raises_regex(ValueError, "same number of patches"):
        PatchFeatures(
            slide_id="bad",
            features=np.zeros((2, 4), dtype=np.float32),
            grid_rc=np.zeros((1, 2), dtype=np.int32),
            native_xy=np.zeros((2, 2), dtype=np.float64),
            reference_um_xy=np.zeros((2, 2), dtype=np.float64),
            tissue_fraction=np.ones(2, dtype=np.float32),
            grid_shape=(1, 2),
            patch_size_px=224,
            analysis_mpp=0.5,
        )


def test_extract_patch_features_filters_grid_with_registered_tissue_mask() -> None:
    geometry = SlideGeometry(
        native_shape=(448, 448),
        content_bbox_xywh=(0, 0, 448, 448),
        thumbnail_shape=(4, 4),
        bounds_source="test",
        mpp_xy=(0.5, 0.5),
    )
    mask = np.zeros((4, 4), dtype=bool)
    mask[:2, :] = True
    calls: list[tuple[int, int, int, int, int]] = []

    def reader(x: int, y: int, width: int, height: int, output_px: int) -> np.ndarray:
        calls.append((x, y, width, height, output_px))
        return np.full((output_px, output_px, 3), x + y, dtype=np.uint8)

    class MeanEncoder:
        def encode(self, images: np.ndarray) -> np.ndarray:
            means = images.mean(axis=(1, 2, 3), dtype=np.float32)
            return np.column_stack([means, np.ones(len(images), dtype=np.float32)])

    result = extract_patch_features(
        slide_id="section.ndpi",
        geometry=geometry,
        tissue_mask=mask,
        moving_to_reference_thumbnail=np.eye(3),
        reference_geometry=geometry,
        reader=reader,
        encoder=MeanEncoder(),
        batch_size=8,
    )

    assert result.grid_shape == (2, 2)
    np.testing.assert_array_equal(result.grid_rc, [[0, 0], [0, 1]])
    np.testing.assert_allclose(result.native_xy, [[112, 112], [336, 112]])
    np.testing.assert_allclose(result.reference_um_xy, [[56, 56], [168, 56]])
    assert calls == [(0, 0, 224, 224, 224), (224, 0, 224, 224, 224)]
