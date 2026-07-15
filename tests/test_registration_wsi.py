from pathlib import Path

import numpy as np
import pytest

from histopia.registration import (
    SlideGeometry,
    calculate_thumbnail_overlap_bbox,
    thumbnail_to_full_resolution_matrix,
    warp_slide_to_reference,
)


def test_slide_geometry_maps_thumbnail_pixels_to_micrometres() -> None:
    geometry = SlideGeometry(
        (1000, 2000),
        (100, 200, 1000, 500),
        (100, 200),
        "openslide.bounds",
        (0.5, 0.25),
        "openslide.mpp",
    )

    point = geometry.thumbnail_to_physical @ np.array([20.0, 40.0, 1.0])

    assert np.allclose(point, [100.0, 100.0, 1.0])


def test_slide_geometry_rejects_uncalibrated_physical_mapping() -> None:
    geometry = SlideGeometry((10, 10), (0, 0, 10, 10), (10, 10), "full")

    with pytest.raises(ValueError, match="spacing is unavailable"):
        _ = geometry.thumbnail_to_physical


def test_thumbnail_matrix_scales_to_full_resolution_coordinates() -> None:
    thumbnail_matrix = np.eye(3, dtype=float)
    thumbnail_matrix[:2, 2] = [5.0, -3.0]

    full_matrix = thumbnail_to_full_resolution_matrix(
        thumbnail_matrix,
        moving_thumbnail_shape=(50, 100),
        moving_full_shape=(200, 400),
        reference_thumbnail_shape=(40, 80),
        reference_full_shape=(200, 400),
    )

    assert np.allclose(full_matrix[:2, :2], 1.25 * np.eye(2))
    assert np.allclose(full_matrix[:2, 2], [25.0, -15.0])


def test_thumbnail_overlap_bbox_intersects_transformed_canvases() -> None:
    identity = np.eye(3)
    translated = np.eye(3)
    translated[0, 2] = 10

    bbox = calculate_thumbnail_overlap_bbox(
        [((80, 100), identity), ((80, 100), translated)],
        (80, 100),
    )

    assert bbox == (10, 0, 90, 80)


@pytest.mark.integration
def test_warp_slide_to_reference_places_pixels_in_reference_canvas(
    tmp_path: Path,
) -> None:
    pyvips = pytest.importorskip("pyvips")
    moving_array = np.full((30, 40, 3), 255, dtype=np.uint8)
    moving_array[8:15, 10:18] = [255, 0, 0]
    moving = pyvips.Image.new_from_memory(
        moving_array.tobytes(),
        40,
        30,
        3,
        "uchar",
    )
    reference = pyvips.Image.black(50, 40, bands=3).invert()
    moving_path = tmp_path / "moving.tiff"
    reference_path = tmp_path / "reference.tiff"
    output_path = tmp_path / "warped.tiff"
    moving.tiffsave(str(moving_path))
    reference.tiffsave(str(reference_path))
    matrix = np.eye(3, dtype=float)
    matrix[:2, 2] = [5.0, 3.0]

    result = warp_slide_to_reference(
        moving_path,
        reference_path,
        output_path,
        matrix,
        moving_thumbnail_shape=(30, 40),
        reference_thumbnail_shape=(40, 50),
        compression="lzw",
        pyramid=False,
        tile_size=16,
    )

    output = pyvips.Image.new_from_file(str(output_path))
    array = np.frombuffer(output.write_to_memory(), dtype=np.uint8).reshape(
        output.height,
        output.width,
        output.bands,
    )
    red = (array[:, :, 0] == 255) & (array[:, :, 1] == 0) & (array[:, :, 2] == 0)
    rows, cols = np.nonzero(red)
    assert result.reference_shape == (40, 50)
    assert (rows.min(), rows.max()) == (11, 17)
    assert (cols.min(), cols.max()) == (15, 22)
    assert not output_path.with_name(f".{output_path.name}.tmp").exists()


def test_pyramidal_warp_rejects_unvalidated_compression(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires validated JPEG"):
        warp_slide_to_reference(
            tmp_path / "moving.tiff",
            tmp_path / "reference.tiff",
            tmp_path / "warped.tiff",
            np.eye(3),
            moving_thumbnail_shape=(10, 10),
            reference_thumbnail_shape=(10, 10),
            compression="lzw",
            pyramid=True,
        )


@pytest.mark.integration
def test_wsi_warp_composes_reference_to_moving_displacement(
    tmp_path: Path,
) -> None:
    pyvips = pytest.importorskip("pyvips")
    moving_array = np.full((24, 32, 3), 255, dtype=np.uint8)
    moving_array[8:14, 12:18] = [0, 0, 255]
    moving = pyvips.Image.new_from_memory(
        moving_array.tobytes(),
        32,
        24,
        3,
        "uchar",
    )
    moving_path = tmp_path / "moving.tiff"
    reference_path = tmp_path / "reference.tiff"
    output_path = tmp_path / "warped.tiff"
    moving.tiffsave(str(moving_path))
    pyvips.Image.black(32, 24, bands=3).invert().tiffsave(str(reference_path))
    displacement = np.zeros((24, 32, 2), dtype=np.float32)
    displacement[:, :, 0] = 2.0

    result = warp_slide_to_reference(
        moving_path,
        reference_path,
        output_path,
        np.eye(3),
        moving_thumbnail_shape=(24, 32),
        reference_thumbnail_shape=(24, 32),
        compression="lzw",
        pyramid=False,
        reference_to_rigid_moving_displacement=displacement,
    )

    output = pyvips.Image.new_from_file(str(output_path))
    array = np.frombuffer(output.write_to_memory(), dtype=np.uint8).reshape(
        output.height,
        output.width,
        output.bands,
    )
    blue = (array[:, :, 0] == 0) & (array[:, :, 1] == 0) & (array[:, :, 2] == 255)
    rows, cols = np.nonzero(blue)
    assert result.non_rigid_applied
    assert (rows.min(), rows.max()) == (8, 13)
    assert (cols.min(), cols.max()) == (10, 15)


@pytest.mark.integration
def test_wsi_warp_applies_reference_thumbnail_crop(tmp_path: Path) -> None:
    pyvips = pytest.importorskip("pyvips")
    image = pyvips.Image.black(40, 32, bands=3).invert()
    moving_path = tmp_path / "moving.tiff"
    reference_path = tmp_path / "reference.tiff"
    output_path = tmp_path / "warped.tiff"
    image.tiffsave(str(moving_path))
    image.tiffsave(str(reference_path))

    result = warp_slide_to_reference(
        moving_path,
        reference_path,
        output_path,
        np.eye(3),
        moving_thumbnail_shape=(32, 40),
        reference_thumbnail_shape=(32, 40),
        compression="lzw",
        pyramid=False,
        reference_thumbnail_bbox=(5, 4, 20, 16),
    )

    output = pyvips.Image.new_from_file(str(output_path))
    assert (output.width, output.height) == (20, 16)
    assert result.output_shape == (16, 20)
    assert result.reference_offset_xy == (5, 4)
    assert np.allclose(result.matrix[:2, 2], [-5, -4])
