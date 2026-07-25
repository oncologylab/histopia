import json
import os
from pathlib import Path

import numpy as np
import pytest

import histopia.registration._wsi as wsi_module
from histopia.registration import (
    SlideGeometry,
    calculate_thumbnail_overlap_bbox,
    geometry_thumbnail_to_native_matrix,
    thumbnail_to_full_resolution_matrix,
    warp_saved_registration,
    warp_slide_to_reference,
)
from histopia.registration._wsi import _as_rgb_uchar, read_slide_shape


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


@pytest.mark.integration
def test_full_resolution_reader_preserves_raw_scanner_orientation(
    tmp_path: Path,
) -> None:
    image_module = pytest.importorskip("PIL.Image")
    pytest.importorskip("pyvips")
    path = tmp_path / "oriented.jpg"
    pixels = np.full((10, 20, 3), 255, dtype=np.uint8)
    pixels[1:4, 2:6] = [255, 0, 0]
    exif = image_module.Exif()
    exif[274] = 6
    image_module.fromarray(pixels).save(path, exif=exif)

    assert read_slide_shape(path) == (10, 20)


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
def test_full_resolution_reader_normalizes_grayscale_alpha() -> None:
    pyvips = pytest.importorskip("pyvips")
    pixels = np.array([[[0, 0], [64, 128], [200, 255]]], dtype=np.uint8)
    image = pyvips.Image.new_from_memory(
        pixels.tobytes(),
        3,
        1,
        2,
        "uchar",
    ).copy(interpretation="b-w")

    rgb = _as_rgb_uchar(image)
    array = np.frombuffer(rgb.write_to_memory(), dtype=np.uint8).reshape(1, 3, 3)

    assert array.tolist() == [[[255, 255, 255], [159, 159, 159], [200, 200, 200]]]


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


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"compression": "zstd"}, ValueError, "compression must be one of"),
        ({"jpeg_quality": 101}, ValueError, "at most 100"),
        ({"tile_size": 0}, ValueError, "tile_size must be positive"),
        ({"pyramid": 1}, TypeError, "pyramid must be a boolean"),
    ],
)
def test_wsi_warp_validates_writer_settings_before_io(
    tmp_path: Path,
    kwargs: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        warp_slide_to_reference(
            tmp_path / "moving.tiff",
            tmp_path / "reference.tiff",
            tmp_path / "warped.tiff",
            np.eye(3),
            moving_thumbnail_shape=(10, 10),
            reference_thumbnail_shape=(10, 10),
            **kwargs,
        )


def test_wsi_warp_configures_vips_before_source_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configured: list[int | None] = []
    monkeypatch.setattr(
        wsi_module,
        "configure_vips_threads",
        configured.append,
    )

    with pytest.raises(FileNotFoundError):
        warp_slide_to_reference(
            tmp_path / "moving.tiff",
            tmp_path / "reference.tiff",
            tmp_path / "warped.tiff",
            np.eye(3),
            moving_thumbnail_shape=(10, 10),
            reference_thumbnail_shape=(10, 10),
            vips_threads=5,
        )

    assert configured == [5]


def test_wsi_warp_rejects_source_output_collision(tmp_path: Path) -> None:
    moving = tmp_path / "moving.tiff"

    with pytest.raises(ValueError, match="must not replace a source"):
        warp_slide_to_reference(
            moving,
            tmp_path / "reference.tiff",
            moving,
            np.eye(3),
            moving_thumbnail_shape=(10, 10),
            reference_thumbnail_shape=(10, 10),
        )


@pytest.mark.parametrize("name", ["overwrite", "accepted_non_rigid_only"])
def test_saved_wsi_warp_rejects_integer_boolean_controls(
    tmp_path: Path,
    name: str,
) -> None:
    with pytest.raises(TypeError, match="must be a boolean"):
        warp_saved_registration(tmp_path, **{name: 1})


@pytest.mark.parametrize(
    ("slide_names", "error", "message"),
    [
        ("moving", TypeError, "iterable of slide names"),
        ((1,), TypeError, "strings or paths"),
        ((), ValueError, "must not be empty"),
        (("moving", "moving"), ValueError, "must not contain duplicates"),
    ],
)
def test_saved_wsi_warp_validates_slide_selectors_before_reading_run(
    tmp_path: Path,
    slide_names: object,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        warp_saved_registration(tmp_path, slide_names=slide_names)


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
def test_wsi_non_rigid_flow_uses_scanner_content_geometry(tmp_path: Path) -> None:
    pyvips = pytest.importorskip("pyvips")
    moving_array = np.full((32, 40, 3), 255, dtype=np.uint8)
    moving_array[12:18, 16:22] = [0, 0, 255]
    moving_path = tmp_path / "moving.tiff"
    reference_path = tmp_path / "reference.tiff"
    output_path = tmp_path / "warped.tiff"
    pyvips.Image.new_from_memory(
        moving_array.tobytes(),
        40,
        32,
        3,
        "uchar",
    ).tiffsave(str(moving_path))
    pyvips.Image.black(40, 32, bands=3).invert().tiffsave(str(reference_path))
    geometry = SlideGeometry(
        native_shape=(32, 40),
        content_bbox_xywh=(8, 6, 24, 16),
        thumbnail_shape=(8, 12),
        bounds_source="test.bounds",
    )
    displacement = np.zeros((8, 12, 2), dtype=np.float32)
    displacement[:, :, 0] = 2.0

    warp_slide_to_reference(
        moving_path,
        reference_path,
        output_path,
        np.eye(3),
        moving_thumbnail_shape=(8, 12),
        reference_thumbnail_shape=(8, 12),
        moving_geometry=geometry,
        reference_geometry=geometry,
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
    assert (rows.min(), rows.max()) == (12, 17)
    assert (cols.min(), cols.max()) == (12, 17)


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


@pytest.mark.integration
def test_saved_wsi_warp_uses_geometry_and_resumes_exact_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run, output, moving_geometry, reference_geometry = _saved_wsi_run(tmp_path)

    first = warp_saved_registration(run, output)
    mtimes = {
        result.output_path: result.output_path.stat().st_mtime_ns for result in first
    }
    summary_path = output / "full_resolution_warps.json"
    summary_stat = summary_path.stat()
    monkeypatch.setattr(wsi_module, "configure_vips_threads", lambda _value: None)
    second = warp_saved_registration(run, output, vips_threads=7)

    expected = geometry_thumbnail_to_native_matrix(
        np.eye(3),
        moving_geometry=moving_geometry,
        reference_geometry=reference_geometry,
    )
    assert np.allclose(first[0].matrix, expected)
    assert len(second) == 2
    assert {
        result.output_path: result.output_path.stat().st_mtime_ns for result in second
    } == mtimes
    assert summary_path.stat().st_mtime_ns == summary_stat.st_mtime_ns
    assert summary_path.stat().st_ino == summary_stat.st_ino
    summary = json.loads(summary_path.read_text())
    assert len(summary) == 2
    assert all(row["provenance"]["schema_version"] == 1 for row in summary)
    assert all(row["provenance"]["export_fingerprint"] for row in summary)
    assert all(
        row["provenance"]["execution"]["requested_vips_threads"] is None
        for row in summary
    )


@pytest.mark.integration
def test_saved_wsi_warp_selects_named_slides_incrementally(
    tmp_path: Path,
) -> None:
    run, output, _, _ = _saved_wsi_run(tmp_path)

    moving = warp_saved_registration(
        run,
        output,
        slide_names=("moving.tiff",),
    )
    assert [result.output_path.name for result in moving] == ["moving.registered.tiff"]
    assert not (output / "reference.registered.tiff").exists()
    first_summary = json.loads((output / "full_resolution_warps.json").read_text())
    assert [Path(row["output_path"]).name for row in first_summary] == [
        "moving.registered.tiff"
    ]

    reference = warp_saved_registration(
        run,
        output,
        slide_names=(Path("reference"),),
    )
    assert [result.output_path.name for result in reference] == [
        "reference.registered.tiff"
    ]
    final_summary = json.loads((output / "full_resolution_warps.json").read_text())
    assert [Path(row["output_path"]).name for row in final_summary] == [
        "moving.registered.tiff",
        "reference.registered.tiff",
    ]
    reference_output = output / "reference.registered.tiff"
    reference_mtime = reference_output.stat().st_mtime_ns
    reference_record = final_summary[1]

    warp_saved_registration(
        run,
        output,
        slide_names=("moving",),
        overwrite=True,
    )

    rewritten_summary = json.loads((output / "full_resolution_warps.json").read_text())
    assert reference_output.stat().st_mtime_ns == reference_mtime
    assert rewritten_summary[1] == reference_record


@pytest.mark.integration
def test_saved_wsi_warp_rejects_unknown_or_overlapping_slide_selectors(
    tmp_path: Path,
) -> None:
    run, output, _, _ = _saved_wsi_run(tmp_path)

    with pytest.raises(ValueError, match="not present"):
        warp_saved_registration(run, output, slide_names=("unknown",))
    with pytest.raises(ValueError, match="same slide more than once"):
        warp_saved_registration(
            run,
            output,
            slide_names=("moving", "moving.tiff"),
        )


@pytest.mark.integration
def test_saved_wsi_warp_rejects_changed_registration_request(
    tmp_path: Path,
) -> None:
    run, output, _, _ = _saved_wsi_run(tmp_path)
    warp_saved_registration(run, output)
    result_path = run / "registration_result.json"
    payload = json.loads(result_path.read_text())
    payload["slides"][0]["transform"]["matrix"][0][2] = 1.0
    result_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="provenance differs"):
        warp_saved_registration(run, output)


@pytest.mark.integration
def test_saved_wsi_warp_rejects_modified_output(tmp_path: Path) -> None:
    run, output, _, _ = _saved_wsi_run(tmp_path)
    results = warp_saved_registration(run, output)
    changed = results[0].output_path
    stat = changed.stat()
    os.utime(changed, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))

    with pytest.raises(ValueError, match="registered output changed"):
        warp_saved_registration(run, output)


@pytest.mark.integration
def test_saved_wsi_warp_preserves_prior_output_when_run_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run, output, _, _ = _saved_wsi_run(tmp_path)
    results = warp_saved_registration(run, output)
    prior = results[0].output_path.read_bytes()
    result_path = run / "registration_result.json"

    def mutate_run(
        moving_path: Path,
        reference_path: Path,
        output_path: Path,
        matrix: np.ndarray,
        **kwargs: object,
    ) -> wsi_module.WsiWarpResult:
        del moving_path, reference_path, matrix, kwargs
        Path(output_path).write_bytes(b"incomplete replacement")
        payload = json.loads(result_path.read_text())
        payload["warnings"] = ["changed concurrently"]
        result_path.write_text(json.dumps(payload))
        return wsi_module.WsiWarpResult(
            Path(output_path),
            np.eye(3),
            (50, 60),
            (60, 70),
        )

    monkeypatch.setattr(wsi_module, "warp_slide_to_reference", mutate_run)

    with pytest.raises(ValueError, match="registration result changed"):
        warp_saved_registration(run, output, overwrite=True)

    assert results[0].output_path.read_bytes() == prior
    assert (
        not results[0]
        .output_path.with_name(f".{results[0].output_path.name}.pending")
        .exists()
    )


def _saved_wsi_run(
    tmp_path: Path,
) -> tuple[Path, Path, SlideGeometry, SlideGeometry]:
    pyvips = pytest.importorskip("pyvips")
    pillow = pytest.importorskip("PIL.Image")
    source = tmp_path / "source"
    run = tmp_path / "run"
    processed = run / "processed"
    output = tmp_path / "registered"
    source.mkdir()
    processed.mkdir(parents=True)
    moving_path = source / "moving.tiff"
    reference_path = source / "reference.tiff"
    pyvips.Image.black(60, 50, bands=3).invert().tiffsave(str(moving_path))
    pyvips.Image.black(70, 60, bands=3).invert().tiffsave(str(reference_path))

    moving_geometry = SlideGeometry(
        native_shape=(50, 60),
        content_bbox_xywh=(10, 8, 30, 20),
        thumbnail_shape=(20, 30),
        bounds_source="test.bounds",
    )
    reference_geometry = SlideGeometry(
        native_shape=(60, 70),
        content_bbox_xywh=(15, 12, 35, 25),
        thumbnail_shape=(25, 35),
        bounds_source="test.bounds",
    )
    pillow.fromarray(np.full((20, 30, 3), 255, dtype=np.uint8)).save(
        processed / "moving.thumbnail.png"
    )
    pillow.fromarray(np.full((25, 35, 3), 255, dtype=np.uint8)).save(
        processed / "reference.thumbnail.png"
    )
    identity = np.eye(3).tolist()
    payload = {
        "reference_slide": str(reference_path),
        "slides": [
            {
                "path": str(moving_path),
                "geometry": moving_geometry.to_json_dict(),
                "transform": {"matrix": identity},
                "non_rigid_transform": None,
            },
            {
                "path": str(reference_path),
                "geometry": reference_geometry.to_json_dict(),
                "transform": {"matrix": identity},
                "non_rigid_transform": None,
            },
        ],
    }
    (run / "registration_result.json").write_text(json.dumps(payload))
    return run, output, moving_geometry, reference_geometry
