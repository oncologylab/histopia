from __future__ import annotations

import json
import sys
from io import BytesIO
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from histopia.cells._cellpose import (
    CellposeRuntime,
    load_cellpose_runtime,
    segment_tile,
    segment_tiles,
)
from histopia.cells._config import CellSegmentationConfig
from histopia.cells._dense_small_cell_recovery import (
    DENSE_SMALL_CELL_RECOVERY_METHOD,
)
from histopia.cells._pipeline import (
    _accumulate_context_grid,
    _active_method_profile,
    _baseline_foam_tile_profile_fingerprints,
    _baseline_fold_tile_profile_fingerprints,
    _bounded_organized_necrotic_protection,
    _bright_instance_core_candidates,
    _clustered_brown_anuclear_instances,
    _clustered_magenta_anuclear_instances,
    _compact_unsupported_prediction_grid,
    _detached_oversized_fragment_instances,
    _disk_label_qc,
    _file_sha256,
    _filter_flat_background_instances,
    _fold_artifact_instances,
    _fragmented_necrotic_prediction_grid,
    _interior_tissue_instance_protection,
    _isolated_glass_artifact_instances,
    _json_sha256,
    _load_cached_tile,
    _load_completed_slide,
    _load_reusable_tile,
    _locally_organized_prediction_grid,
    _locally_supported_organized_necrotic_protection,
    _merge_enclosed_cytoplasmic_child_instances,
    _oversized_brown_anuclear_instances,
    _prepare_tile_cache_reuse,
    _raw_tile_inference_request,
    _replace_reusable_tile_with_dense_recovery,
    _request_payload,
    _run_slide,
    _section_fingerprint,
    _select_processing_slides,
    _self_dense_glass_artifact_instances,
    _sparse_anuclear_satellite_instances,
    _sparse_glass_organized_necrotic_instances,
    _sparse_neutral_dark_precipitate_instances,
    _tile_fingerprint,
    _tile_tissue_mask,
    _version_75_postfilter_request,
    _write_cached_tile,
    run_cell_segmentation,
)
from histopia.cells._preflight import (
    CellPreflight,
    CellPreflightSlide,
    load_cell_preflight,
    write_cell_preflight,
)
from histopia.cells._qupath import _cell_features
from histopia.cells._result import write_cell_result
from histopia.cells._tiles import CellTile
from histopia.compute import ComputeDevice


def test_bright_instance_core_candidates_use_eroded_exact_interiors() -> None:
    labels = np.zeros((24, 48), dtype=np.uint32)
    # Equal-size adjacent instances: the first has a pale interior and dark
    # walls; the second has a dark interior. A third pale object exceeds the
    # bounded instance area and must remain ineligible.
    labels[4:16, 4:16] = 1
    labels[4:16, 20:32] = 2
    labels[2:22, 36:47] = 3
    rgb = np.full((*labels.shape, 3), 80, dtype=np.uint8)
    rgb[6:14, 6:14] = 210
    rgb[4:16, 20:32] = 130
    rgb[4:20, 38:45] = 210
    areas = np.bincount(labels.ravel(), minlength=4).astype(np.uint64)

    def read_rgb(x: int, y: int, width: int, height: int) -> np.ndarray:
        return rgb[y : y + height, x : x + width]

    expected = np.array([False, True, False, False])
    for block_size in (12, 24, 64):
        observed = _bright_instance_core_candidates(
            labels,
            read_rgb,
            maximum=3,
            instance_areas=areas,
            minimum_core_pixels=8,
            minimum_bright_fraction=0.80,
            minimum_intensity=180.0,
            maximum_instance_area=180,
            block_size=block_size,
        )
        assert np.array_equal(observed, expected)


def test_bright_instance_core_candidates_skip_distant_empty_blocks() -> None:
    labels = np.zeros((64, 64), dtype=np.uint32)
    labels[4:12, 4:12] = 1
    rgb = np.full((*labels.shape, 3), 210, dtype=np.uint8)
    areas = np.bincount(labels.ravel(), minlength=2).astype(np.uint64)
    reads: list[tuple[int, int, int, int]] = []

    def read_rgb(x: int, y: int, width: int, height: int) -> np.ndarray:
        reads.append((x, y, width, height))
        return rgb[y : y + height, x : x + width]

    observed = _bright_instance_core_candidates(
        labels,
        read_rgb,
        maximum=1,
        instance_areas=areas,
        minimum_core_pixels=8,
        minimum_bright_fraction=0.80,
        minimum_intensity=180.0,
        maximum_instance_area=180,
        block_size=16,
    )

    assert np.array_equal(observed, np.array([False, True]))
    assert reads == [(0, 0, 18, 18)]


def _necrotic_grid(
    *, fill: float, red_blue: float, intensity: float, eligible: bool = True
) -> np.ndarray:
    shape = (50, 50)
    pixels = np.full(shape, 16, dtype=np.uint32)
    prediction = np.zeros(shape, dtype=np.uint32)
    if fill >= 1.0:
        prediction[5:45, 5:45] = 16
    else:
        for row in range(5, 45, 4):
            for column in range(5, 45, 4):
                prediction[row : row + 2, column : column + 2] = 16
    labels = (prediction > 0).astype(np.uint32)
    colors = np.full(shape, red_blue, dtype=np.float32)
    values = np.full(shape, intensity, dtype=np.float32)
    return _fragmented_necrotic_prediction_grid(
        prediction,
        pixels,
        labels,
        colors,
        values,
        bin_size=4,
        minimum_component_pixels=5000,
        maximum_aspect_ratio=2.5,
        minimum_fill_fraction=0.05,
        maximum_fill_fraction=0.45,
        minimum_mean_red_blue_difference=-15.0,
        maximum_mean_red_blue_difference=15.0,
        minimum_mean_intensity=160.0,
        eligible_instances=np.array([False, eligible]),
    )


def test_necrotic_gate_rejects_compact_sparse_pale_lavender_field() -> None:
    assert (
        np.count_nonzero(_necrotic_grid(fill=0.25, red_blue=-6.5, intensity=164.0)) > 0
    )


def test_necrotic_gate_preserves_dense_viable_tissue() -> None:
    assert not np.any(_necrotic_grid(fill=1.0, red_blue=-6.5, intensity=164.0))


def test_necrotic_gate_preserves_normal_blue_he_fragment() -> None:
    assert not np.any(_necrotic_grid(fill=0.25, red_blue=-25.0, intensity=164.0))


def test_necrotic_gate_preserves_nuclear_supported_instances() -> None:
    assert not np.any(
        _necrotic_grid(
            fill=0.25,
            red_blue=-6.5,
            intensity=164.0,
            eligible=False,
        )
    )


def test_necrotic_gate_can_exclude_locally_organized_tissue() -> None:
    shape = (50, 50)
    pixel_counts = np.full(shape, 16, dtype=np.uint32)
    prediction_counts = np.zeros(shape, dtype=np.uint32)
    # A connected one-bin lattice occupies 43.75% of its bounding box. It is
    # sparse enough to resemble a necrotic cloud by fill alone, but one large,
    # coherent prediction establishes the organized-tissue safeguard.
    for row in range(5, 45):
        for column in range(5, 45):
            if (row - 5) % 4 == 0 or (column - 5) % 4 == 0:
                prediction_counts[row, column] = 16
    sampled_labels = (prediction_counts > 0).astype(np.uint32)
    red_blue = np.full(shape, -6.5, dtype=np.float32)
    intensity = np.full(shape, 164.0, dtype=np.float32)
    organized = _locally_organized_prediction_grid(
        prediction_counts,
        pixel_counts,
        sampled_labels,
        red_blue,
        bin_size=4,
        window_size=200,
        window_overlap=0,
        minimum_bin_occupancy_fraction=0.10,
        minimum_component_pixels=5_000,
        compact_minimum_component_pixels=5_000,
        minimum_aspect_ratio=4.0,
        minimum_mean_instance_pixels=350.0,
        strong_red_blue_difference=40.0,
        minimum_strong_chromatic_fraction=0.25,
    )
    organized_instances = np.array(
        [
            False,
            int(prediction_counts[organized & (sampled_labels == 1)].sum())
            >= int(np.ceil(prediction_counts.sum() * 0.50)),
        ]
    )
    common = {
        "bin_size": 4,
        "minimum_component_pixels": 5_000,
        "maximum_aspect_ratio": 2.5,
        "minimum_fill_fraction": 0.05,
        "maximum_fill_fraction": 0.45,
        "minimum_mean_red_blue_difference": -15.0,
        "maximum_mean_red_blue_difference": 15.0,
        "minimum_mean_intensity": 160.0,
    }

    unprotected = _fragmented_necrotic_prediction_grid(
        prediction_counts,
        pixel_counts,
        sampled_labels,
        red_blue,
        intensity,
        **common,
    )
    protected = _fragmented_necrotic_prediction_grid(
        prediction_counts,
        pixel_counts,
        sampled_labels,
        red_blue,
        intensity,
        eligible_instances=~organized_instances,
        **common,
    )

    assert organized_instances.tolist() == [False, True]
    assert np.any(unprotected)
    assert not np.any(protected)


def test_necrotic_protection_is_post_detection_and_monotonic() -> None:
    shape = (50, 50)
    pixel_counts = np.full(shape, 16, dtype=np.uint32)
    prediction_counts = np.zeros(shape, dtype=np.uint32)
    for row in range(5, 45, 4):
        for column in range(5, 45, 4):
            prediction_counts[row : row + 2, column : column + 2] = 16
    sampled_labels = (prediction_counts > 0).astype(np.uint32)
    sampled_labels[:, 25:] *= 2
    red_blue = np.full(shape, -6.5, dtype=np.float32)
    intensity = np.full(shape, 164.0, dtype=np.float32)
    common = {
        "bin_size": 4,
        "minimum_component_pixels": 5_000,
        "maximum_aspect_ratio": 2.5,
        "minimum_fill_fraction": 0.05,
        "maximum_fill_fraction": 0.45,
        "minimum_mean_red_blue_difference": -15.0,
        "maximum_mean_red_blue_difference": 15.0,
        "minimum_mean_intensity": 160.0,
    }

    baseline = _fragmented_necrotic_prediction_grid(
        prediction_counts,
        pixel_counts,
        sampled_labels,
        red_blue,
        intensity,
        **common,
    )
    protected = _fragmented_necrotic_prediction_grid(
        prediction_counts,
        pixel_counts,
        sampled_labels,
        red_blue,
        intensity,
        protected_instances=np.array([False, True, False]),
        **common,
    )

    assert np.any(baseline & (sampled_labels == 1))
    assert np.any(baseline & (sampled_labels == 2))
    assert np.array_equal(
        protected,
        baseline & (sampled_labels != 1),
    )


def test_necrotic_gate_requires_locally_sparse_prediction_context() -> None:
    shape = (80, 80)
    pixels = np.full(shape, 16, dtype=np.uint32)
    sparse = np.zeros(shape, dtype=np.uint32)
    for row in range(20, 60, 4):
        for column in range(20, 60, 4):
            sparse[row : row + 2, column : column + 2] = 16
    labels = (sparse > 0).astype(np.uint32)
    red_blue = np.zeros(shape, dtype=np.float32)
    intensity = np.full(shape, 170.0, dtype=np.float32)
    common = {
        "bin_size": 4,
        "minimum_component_pixels": 5000,
        "maximum_aspect_ratio": 2.5,
        "minimum_fill_fraction": 0.05,
        "maximum_fill_fraction": 0.45,
        "minimum_mean_red_blue_difference": -15.0,
        "maximum_mean_red_blue_difference": 15.0,
        "minimum_mean_intensity": 160.0,
        "window_size": 256,
        "window_overlap": 32,
        "context_window_size": 64,
        "maximum_local_prediction_fraction": 0.40,
    }
    sparse_grid = _fragmented_necrotic_prediction_grid(
        sparse,
        pixels,
        labels,
        red_blue,
        intensity,
        **common,
    )
    dense_context = sparse.copy()
    dense_context[8:72, 8:72] = 16
    dense_labels = (dense_context > 0).astype(np.uint32)
    dense_red_blue = np.full(shape, 30.0, dtype=np.float32)
    dense_red_blue[20:60, 20:60] = 0.0
    dense_grid = _fragmented_necrotic_prediction_grid(
        dense_context,
        pixels,
        dense_labels,
        dense_red_blue,
        intensity,
        **common,
    )

    assert np.any(sparse_grid)
    assert not np.any(dense_grid)


def _brown_fragment_grid(
    *,
    dense: bool = False,
    red_blue: float = 60.0,
    elongated: bool = False,
    eligible: bool = True,
    minimum_component_pixels: int = 5_000,
) -> np.ndarray:
    shape = (40, 100) if elongated else (50, 50)
    pixels = np.full(shape, 16, dtype=np.uint32)
    prediction = np.zeros(shape, dtype=np.uint32)
    row_stop = 25 if elongated else 45
    column_stop = 95 if elongated else 45
    if dense:
        prediction[5:row_stop, 5:column_stop] = 16
    else:
        for row in range(5, row_stop, 4):
            for column in range(5, column_stop, 4):
                prediction[row : row + 2, column : column + 2] = 16
    labels = (prediction > 0).astype(np.uint32)
    return _fragmented_necrotic_prediction_grid(
        prediction,
        pixels,
        labels,
        np.full(shape, red_blue, dtype=np.float32),
        np.full(shape, 105.0, dtype=np.float32),
        bin_size=4,
        minimum_component_pixels=minimum_component_pixels,
        maximum_aspect_ratio=3.0,
        minimum_fill_fraction=0.22,
        maximum_fill_fraction=0.45,
        minimum_mean_red_blue_difference=50.0,
        maximum_mean_red_blue_difference=255.0,
        minimum_mean_intensity=None,
        maximum_mean_intensity=120.0,
        minimum_bin_red_blue_difference=50.0,
        maximum_bin_intensity=120.0,
        eligible_instances=np.array([False, eligible]),
    )


def test_brown_debris_gate_rejects_compact_dark_chromatic_fragments() -> None:
    assert np.any(_brown_fragment_grid())


def test_brown_debris_gate_reaches_small_cpsam_fragment_components() -> None:
    assert np.any(_brown_fragment_grid(minimum_component_pixels=5_000))
    assert not np.any(_brown_fragment_grid(minimum_component_pixels=7_000))


def test_brown_debris_gate_preserves_dense_viable_brown_tissue() -> None:
    assert not np.any(_brown_fragment_grid(dense=True))


def test_brown_debris_gate_preserves_muted_or_elongated_tissue() -> None:
    assert not np.any(_brown_fragment_grid(red_blue=42.0))
    assert not np.any(_brown_fragment_grid(elongated=True))


def test_brown_debris_gate_preserves_nuclear_supported_instances() -> None:
    assert not np.any(_brown_fragment_grid(eligible=False))


def test_brown_debris_gate_uses_bounded_windows_to_break_slide_scale_bridge() -> None:
    shape = (300, 600)
    pixels = np.full(shape, 16, dtype=np.uint32)
    prediction = np.zeros(shape, dtype=np.uint32)
    for row in range(20, 180, 4):
        for column in range(20, 180, 4):
            prediction[row : row + 3, column : column + 2] = 16
    prediction[98:102, 180:590] = 16
    labels = (prediction > 0).astype(np.uint32)
    common = {
        "bin_size": 4,
        "minimum_component_pixels": 5000,
        "maximum_aspect_ratio": 3.0,
        "minimum_fill_fraction": 0.22,
        "maximum_fill_fraction": 0.45,
        "minimum_mean_red_blue_difference": 50.0,
        "maximum_mean_red_blue_difference": 255.0,
        "minimum_mean_intensity": None,
        "maximum_mean_intensity": 120.0,
        "minimum_bin_red_blue_difference": 50.0,
        "maximum_bin_intensity": 120.0,
        "eligible_instances": np.array([False, True]),
    }
    global_grid = _fragmented_necrotic_prediction_grid(
        prediction,
        pixels,
        labels,
        np.full(shape, 60.0, dtype=np.float32),
        np.full(shape, 105.0, dtype=np.float32),
        **common,
    )
    windowed_grid = _fragmented_necrotic_prediction_grid(
        prediction,
        pixels,
        labels,
        np.full(shape, 60.0, dtype=np.float32),
        np.full(shape, 105.0, dtype=np.float32),
        window_size=1024,
        window_overlap=128,
        **common,
    )

    assert not np.any(global_grid)
    assert np.any(windowed_grid[20:180, 20:180])


def test_brown_debris_gate_requires_bounded_source_component() -> None:
    shape = (80, 160)
    pixels = np.full(shape, 16, dtype=np.uint32)
    prediction = np.zeros(shape, dtype=np.uint32)
    for row in range(20, 40, 4):
        for column in range(20, 40, 4):
            prediction[row : row + 2, column : column + 2] = 16
        for column in range(100, 120, 4):
            prediction[row : row + 2, column : column + 2] = 16
    labels = (prediction > 0).astype(np.uint32)
    intensity = np.full(shape, 250.0, dtype=np.float32)
    intensity[16:44, 16:44] = 210.0
    intensity[16:44, 96:] = 210.0
    intensity[prediction > 0] = 105.0

    artifact = _fragmented_necrotic_prediction_grid(
        prediction,
        pixels,
        labels,
        np.full(shape, 60.0, dtype=np.float32),
        intensity,
        bin_size=4,
        minimum_component_pixels=200,
        maximum_aspect_ratio=3.0,
        minimum_fill_fraction=0.12,
        maximum_fill_fraction=0.45,
        minimum_mean_red_blue_difference=50.0,
        maximum_mean_red_blue_difference=255.0,
        minimum_mean_intensity=None,
        maximum_mean_intensity=120.0,
        minimum_bin_red_blue_difference=50.0,
        maximum_bin_intensity=120.0,
        eligible_instances=np.array([False, True]),
        maximum_source_component_pixels=30_000,
        source_maximum_mean_intensity=220.0,
        source_dilation_bins=2,
    )

    assert np.any(artifact[20:40, 20:40])
    assert artifact[17, 17]
    assert not np.any(artifact[20:40, 100:120])


def test_brown_debris_gate_breaks_pale_glass_bridge_to_sloughed_cluster() -> None:
    shape = (80, 160)
    pixels = np.full(shape, 16, dtype=np.uint32)
    prediction = np.zeros(shape, dtype=np.uint32)
    for row in range(20, 40, 4):
        for column in range(20, 40, 4):
            prediction[row : row + 2, column : column + 2] = 16
    labels = (prediction > 0).astype(np.uint32)
    intensity = np.full(shape, 250.0, dtype=np.float32)
    intensity[16:44, 16:44] = 210.0
    # A barely visible glass-adjacent trail must not join the compact cluster
    # to a slide-edge source component. This is the measured 6180/009 failure
    # mode; pixels at 219 were admitted by the former ceiling of 220.
    intensity[28:30, 44:] = 219.0
    intensity[prediction > 0] = 105.0
    common = {
        "bin_size": 4,
        "minimum_component_pixels": 200,
        "maximum_aspect_ratio": 3.0,
        "minimum_fill_fraction": 0.12,
        "maximum_fill_fraction": 0.45,
        "minimum_mean_red_blue_difference": 50.0,
        "maximum_mean_red_blue_difference": 255.0,
        "minimum_mean_intensity": None,
        "maximum_mean_intensity": 120.0,
        "minimum_bin_red_blue_difference": 50.0,
        "maximum_bin_intensity": 120.0,
        "eligible_instances": np.array([False, True]),
        "maximum_source_component_pixels": 30_000,
        "source_dilation_bins": 0,
    }

    former = _fragmented_necrotic_prediction_grid(
        prediction,
        pixels,
        labels,
        np.full(shape, 60.0, dtype=np.float32),
        intensity,
        source_maximum_mean_intensity=220.0,
        **common,
    )
    corrected = _fragmented_necrotic_prediction_grid(
        prediction,
        pixels,
        labels,
        np.full(shape, 60.0, dtype=np.float32),
        intensity,
        source_maximum_mean_intensity=218.0,
        **common,
    )

    assert not np.any(former)
    assert np.any(corrected[20:40, 20:40])
    assert not np.any(corrected[:, 44:])


def _fake_cellpose_modules(monkeypatch: pytest.MonkeyPatch, cache: Path) -> ModuleType:
    models = ModuleType("cellpose.models")
    models.MODEL_DIR = cache

    class FakeModel:
        def __init__(self, **kwargs) -> None:
            self.pretrained_model = str(cache / str(kwargs["pretrained_model"]))
            self.device = kwargs["device"]

    models.CellposeModel = FakeModel
    cellpose = ModuleType("cellpose")
    cellpose.models = models
    torch = ModuleType("torch")
    torch.device = lambda value: value
    monkeypatch.setitem(sys.modules, "cellpose", cellpose)
    monkeypatch.setitem(sys.modules, "cellpose.models", models)
    monkeypatch.setitem(sys.modules, "torch", torch)
    return models


def test_cellpose_runtime_refuses_implicit_model_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_cellpose_modules(monkeypatch, tmp_path)
    monkeypatch.setattr("histopia.cells._cellpose.version", lambda name: "4.2.1")
    monkeypatch.setattr(
        "histopia.cells._cellpose.resolve_compute_device",
        lambda *args, **kwargs: ComputeDevice("cpu", "cpu", "cpu"),
    )
    config = CellSegmentationConfig(
        registration_run=tmp_path,
        output_dir=tmp_path,
        model_cache=tmp_path,
    )

    with pytest.raises(FileNotFoundError, match="cache-model"):
        load_cellpose_runtime(config)


def test_cellpose_runtime_records_cached_weight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_cellpose_modules(monkeypatch, tmp_path)
    weight = tmp_path / "cpsam"
    weight.write_bytes(b"model")
    monkeypatch.setattr(
        "histopia.cells._cellpose.resolve_compute_device",
        lambda *args, **kwargs: ComputeDevice("cpu", "cpu", "cpu"),
    )
    monkeypatch.setattr("histopia.cells._cellpose.version", lambda name: "4.2.1")
    runtime = load_cellpose_runtime(
        CellSegmentationConfig(
            registration_run=tmp_path,
            output_dir=tmp_path,
            model_cache=tmp_path,
        )
    )

    assert runtime.weight_path == str(weight)
    assert len(runtime.weight_sha256) == 64


def test_cellpose_runtime_rejects_legacy_version_for_cpsam(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_cellpose_modules(monkeypatch, tmp_path)
    (tmp_path / "cpsam").write_bytes(b"model")
    monkeypatch.setattr("histopia.cells._cellpose.version", lambda name: "3.1.1.1")

    with pytest.raises(RuntimeError, match="requires Cellpose 4 or newer"):
        load_cellpose_runtime(
            CellSegmentationConfig(
                registration_run=tmp_path,
                output_dir=tmp_path,
                model_cache=tmp_path,
            )
        )


def test_cellpose_runtime_rejects_silent_weight_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models = _fake_cellpose_modules(monkeypatch, tmp_path)
    (tmp_path / "cpsam").write_bytes(b"requested")
    fallback = tmp_path / "cyto3"
    fallback.write_bytes(b"fallback")

    class FallbackModel:
        def __init__(self, **kwargs) -> None:
            self.pretrained_model = str(fallback)
            self.device = kwargs["device"]

    models.CellposeModel = FallbackModel
    monkeypatch.setattr(
        "histopia.cells._cellpose.resolve_compute_device",
        lambda *args, **kwargs: ComputeDevice("cpu", "cpu", "cpu"),
    )
    monkeypatch.setattr("histopia.cells._cellpose.version", lambda name: "4.2.1")

    with pytest.raises(RuntimeError, match="Refusing a silent model fallback"):
        load_cellpose_runtime(
            CellSegmentationConfig(
                registration_run=tmp_path,
                output_dir=tmp_path,
                model_cache=tmp_path,
            )
        )


def test_multiscale_method_adds_complementary_small_cells(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = np.zeros((8, 8), dtype=np.int32)
    first[1:4, 1:4] = 1
    second = np.zeros_like(first)
    second[1:3, 1:3] = 1
    second[5:7, 5:7] = 2
    calls: list[tuple[int, int, float]] = []

    def fake_direct(
        _images, _model, *, diameter, batch_size, flow_threshold, **_kwargs
    ):
        calls.append((diameter, batch_size, flow_threshold))
        return [first if diameter == 26 else second]

    monkeypatch.setattr("histopia.cells._cellpose._direct_masks", fake_direct)
    runtime = CellposeRuntime(
        model=object(),
        device=ComputeDevice("cpu", "cpu", "cpu"),
        cellpose_version="test",
        weight_path=str(tmp_path / "cpsam"),
        weight_sha256="a" * 64,
    )
    config = CellSegmentationConfig(
        tmp_path,
        tmp_path / "out",
        method="multiscale",
        first_diameter=26,
        second_diameter=15,
        inference_batch_size=32,
        flow_threshold=0.35,
        multiscale_nuclear_support=False,
    )

    result = segment_tile(np.zeros((8, 8, 3), dtype=np.uint8), runtime, config)

    assert calls == [(26, 32, 0.35), (15, 32, 0.35)]
    assert set(np.unique(result)) == {0, 1, 2}


def test_containment_batch_matches_serial_tiles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_combined(images, _model, *, diameter, **_kwargs):
        output = []
        for image in images:
            mask = np.zeros(image.shape[:2], dtype=np.int32)
            if diameter == 26:
                mask[1:4, 1:4] = 1
            elif int(image[0, 0, 0]) > 0:
                mask[5:7, 5:7] = 1
            output.append(mask)
        return output

    monkeypatch.setattr("histopia.cells._cellpose._combined_masks", fake_combined)
    runtime = CellposeRuntime(
        model=object(),
        device=ComputeDevice("cpu", "cpu", "cpu"),
        cellpose_version="test",
        weight_path=str(tmp_path / "cpsam"),
        weight_sha256="a" * 64,
    )
    config = CellSegmentationConfig(
        tmp_path,
        tmp_path / "out",
        method="containment",
        multiscale_nuclear_support=False,
    )
    images = [np.full((8, 8, 3), value, dtype=np.uint8) for value in (0, 1, 2)]

    batched = segment_tiles(images, runtime, config)
    serial = [segment_tile(image, runtime, config) for image in images]

    for observed, expected in zip(batched, serial, strict=True):
        np.testing.assert_array_equal(observed, expected)


def test_native_tile_mask_uses_nearest_thumbnail_pixels() -> None:
    thumbnail = np.array([[True, False], [False, True]])
    tile = CellTile(0, 0, 0, 4, 0, 4)

    sampled = _tile_tissue_mask(thumbnail, tile, content_shape=(4, 4))

    assert np.array_equal(
        sampled,
        np.array(
            [
                [True, True, False, False],
                [True, True, False, False],
                [False, False, True, True],
                [False, False, True, True],
            ]
        ),
    )


def test_aligned_context_grid_accumulation_matches_exact_bins() -> None:
    evidence = np.zeros((19, 23), dtype=bool)
    evidence[1:17:2, 3:21:3] = True
    counts = np.zeros((8, 8), dtype=np.uint64)
    pixels = np.zeros_like(counts)

    _accumulate_context_grid(
        counts,
        pixels,
        evidence,
        x0=8,
        y0=12,
        bin_size=4,
    )

    for row in range(5):
        for column in range(6):
            block = evidence[row * 4 : (row + 1) * 4, column * 4 : (column + 1) * 4]
            assert counts[row + 3, column + 2] == np.count_nonzero(block)
            assert pixels[row + 3, column + 2] == block.size


def test_disk_label_qc_audits_exact_mask_containment_and_small_fragments() -> None:
    labels = np.zeros((4, 4), dtype=np.uint32)
    labels[:2, :2] = 1
    labels[2, 2] = 2
    tissue = np.array([[True, False], [False, False]])

    qc = _disk_label_qc(
        labels,
        2,
        tissue_mask=tissue,
        content_shape=labels.shape,
        min_size=3,
    )

    assert qc["cell_count"] == 2
    assert qc["outside_tissue_pixels_final"] == 1
    assert qc["post_constraint_cells_below_min_size"] == 1


def test_flat_background_filter_keeps_only_stain_supported_instances() -> None:
    labels = np.zeros((8, 12), dtype=np.uint32)
    labels[1:5, 1:5] = 1
    labels[2:6, 7:11] = 2
    image = np.full((8, 12, 3), (235, 233, 234), dtype=np.uint8)
    image[2:5, 2:5] = (135, 110, 155)

    qc = _filter_flat_background_instances(
        labels,
        2,
        read_rgb=lambda x, y, width, height: image[y : y + height, x : x + width],
        block_size=5,
    )

    assert set(np.unique(labels)) == {0, 1}
    assert qc["flat_background_instances_removed"] == 1
    assert qc["flat_background_pixels_removed"] == 16
    assert qc["instance_evidence"]["block_size_px"] == 5


def test_global_evidence_filter_removes_post_constraint_small_instances() -> None:
    labels = np.zeros((8, 12), dtype=np.uint32)
    labels[1:5, 1:5] = 1
    labels[6, 10:12] = 2
    image = np.full((8, 12, 3), (235, 233, 234), dtype=np.uint8)
    image[1:5, 1:5] = (72, 58, 128)
    image[6, 10:12] = (72, 58, 128)

    qc = _filter_flat_background_instances(
        labels,
        2,
        read_rgb=lambda x, y, width, height: image[y : y + height, x : x + width],
        block_size=5,
        minimum_area=3,
    )

    assert set(np.unique(labels)) == {0, 1}
    assert qc["post_constraint_small_instances_removed"] == 1
    assert qc["post_constraint_small_pixels_removed"] == 2
    assert qc["flat_background_instances_removed"] == 0
    assert qc["instance_evidence"]["minimum_area_pixels"] == 3


def test_global_nuclear_filter_rejects_stained_brown_debris() -> None:
    labels = np.zeros((12, 18), dtype=np.uint32)
    labels[2:8, 2:8] = 1
    labels[2:9, 11:17] = 2
    image = np.full((12, 18, 3), (242, 240, 238), dtype=np.uint8)
    image[3:6, 3:6] = (72, 58, 128)
    image[3:8, 12:16] = (112, 70, 34)

    qc = _filter_flat_background_instances(
        labels,
        2,
        read_rgb=lambda x, y, width, height: image[y : y + height, x : x + width],
        block_size=18,
        require_nuclear_support=True,
        nuclear_minimum_optical_density=0.08,
        nuclear_minimum_pixels=3,
        nuclear_minimum_fraction=0.01,
    )

    assert set(np.unique(labels)) == {0, 1}
    assert qc["flat_background_instances_removed"] == 0
    assert qc["nuclear_unsupported_instances_removed"] == 1
    assert qc["nuclear_unsupported_pixels_removed"] == 42


def test_source_tissue_context_rejects_isolated_nucleus_like_debris() -> None:
    labels = np.zeros((512, 512), dtype=np.uint32)
    labels[80:100, 80:100] = 1
    labels[300:320, 340:360] = 2
    image = np.full((512, 512, 3), (242, 240, 238), dtype=np.uint8)
    image[80:100, 80:100] = (72, 58, 128)
    image[256:, 256:] = (215, 205, 210)
    image[256::4, 256:] = (150, 125, 155)
    image[300:320, 340:360] = (72, 58, 128)

    qc = _filter_flat_background_instances(
        labels,
        2,
        read_rgb=lambda x, y, width, height: image[y : y + height, x : x + width],
        block_size=300,
        require_source_tissue_context=True,
        source_tissue_context_bin_size=256,
        source_tissue_context_minimum_fraction=0.03,
        source_tissue_context_minimum_instance_fraction=0.01,
    )

    assert set(np.unique(labels)) == {0, 2}
    assert qc["source_context_unsupported_instances_removed"] == 1
    assert qc["source_context_unsupported_pixels_removed"] == 400
    evidence = qc["instance_evidence"]
    assert evidence["source_tissue_context"] is True
    assert evidence["source_tissue_context_bin_size_px"] == 256


def test_adaptive_nuclear_core_rejects_large_diffuse_luminal_object() -> None:
    labels = np.zeros((80, 120), dtype=np.uint32)
    labels[10:50, 5:35] = 1
    labels[10:50, 45:75] = 2
    labels[60:70, 90:100] = 3
    image = np.full((80, 120, 3), (242, 240, 238), dtype=np.uint8)
    # Neutral-dark debris projects onto the hematoxylin basis but is not a
    # blue/purple nuclear core. The second object contains a compact blue core.
    image[10:50, 5:35] = (38, 29, 31)
    image[10:50, 45:75] = (205, 188, 211)
    image[20:50, 50:75] = (72, 58, 128)
    image[60:70, 90:100] = (110, 85, 145)

    qc = _filter_flat_background_instances(
        labels,
        3,
        read_rgb=lambda x, y, width, height: image[y : y + height, x : x + width],
        block_size=120,
        require_adaptive_nuclear_core=True,
        adaptive_nuclear_core_percentile=70.0,
        adaptive_nuclear_core_minimum_area=1000,
        adaptive_nuclear_core_minimum_area_um2=225.0,
        adaptive_nuclear_core_minimum_pixels=8,
        adaptive_nuclear_core_minimum_fraction=0.10,
        adaptive_nuclear_core_minimum_blue_ratio=1.08,
    )

    assert set(np.unique(labels)) == {0, 2, 3}
    assert qc["adaptive_nuclear_core_unsupported_instances_removed"] == 1
    assert qc["adaptive_nuclear_core_unsupported_pixels_removed"] == 1200
    evidence = qc["instance_evidence"]
    assert evidence["adaptive_nuclear_core"] is True
    assert evidence["adaptive_nuclear_core_minimum_area_pixels"] == 1000
    assert evidence["adaptive_nuclear_core_minimum_area_um2"] == 225.0
    assert evidence["adaptive_nuclear_core_minimum_fraction"] == 0.10
    assert evidence["adaptive_nuclear_core_minimum_blue_ratio"] == 1.08


def test_isolated_debris_gate_requires_supported_non_dark_tissue_context() -> None:
    labels = np.zeros((192, 192), dtype=np.uint32)
    labels[8:24, 8:24] = 1
    labels[80:100, 80:100] = 2
    labels[140:160, 140:160] = 3
    labels[32:48, 32:48] = 4
    labels[32:48, 80:96] = 5
    labels[32:48, 112:128] = 6
    image = np.full((192, 192, 3), (245, 243, 241), dtype=np.uint8)
    # A connected stain-dense component containing blue nuclear pixels admits
    # a brown cell even when that individual instance has no chromatic nucleus.
    image[:64, :128] = (125, 82, 42)
    image[8:24, 8:24] = (125, 82, 42)
    image[2:14, 70:82] = (72, 58, 128)
    # An isolated brown/neutral object in a white lumen is debris.
    image[80:100, 80:100] = (95, 70, 45)
    # Nuclear evidence cannot let a disconnected candidate validate itself.
    image[140:160, 140:160] = (205, 188, 211)
    image[146:154, 146:154] = (72, 58, 128)
    # Neutral-black objects are rejected even when adjacent tissue gives their
    # grid component valid nuclear context. Chromatic hematoxylin-rich tissue
    # must remain eligible despite being extremely dark.
    image[32:48, 32:48] = (40, 42, 41)
    image[32:48, 80:96] = (55, 40, 30)
    image[32:48, 112:128] = (20, 20, 55)

    qc = _filter_flat_background_instances(
        labels,
        6,
        read_rgb=lambda x, y, width, height: image[y : y + height, x : x + width],
        block_size=192,
        require_isolated_debris_gate=True,
        isolated_debris_context_bin_size=64,
        isolated_debris_context_minimum_stain_fraction=0.80,
        isolated_debris_context_minimum_instance_fraction=0.75,
        isolated_debris_component_minimum_nuclear_pixels=8,
        isolated_debris_allow_nuclear_escape=False,
        isolated_debris_nuclear_minimum_pixels=2,
        isolated_debris_nuclear_minimum_fraction=0.005,
        isolated_debris_nuclear_minimum_blue_ratio=1.08,
        isolated_debris_neutral_dark_maximum_value=90,
        isolated_debris_neutral_dark_maximum_chroma=30,
        isolated_debris_neutral_dark_maximum_fraction=0.50,
        isolated_debris_very_dark_maximum_value=60,
        isolated_debris_very_dark_maximum_chroma=20,
        isolated_debris_very_dark_maximum_fraction=0.45,
    )

    assert set(np.unique(labels)) == {0, 1, 6}
    assert qc["isolated_debris_unsupported_instances_removed"] == 4
    assert qc["isolated_debris_unsupported_pixels_removed"] == 1312
    evidence = qc["instance_evidence"]
    assert evidence["isolated_debris_gate"] is True
    assert evidence["isolated_debris_context_bin_size_px"] == 64
    assert evidence["isolated_debris_context_minimum_stain_fraction"] == 0.80
    assert evidence["isolated_debris_component_minimum_nuclear_pixels"] == 8
    assert evidence["isolated_debris_allow_nuclear_escape"] is False
    assert evidence["isolated_debris_neutral_dark_maximum_chroma"] == 30
    assert evidence["isolated_debris_neutral_dark_maximum_fraction"] == 0.50
    assert evidence["isolated_debris_very_dark_maximum_fraction"] == 0.45
    assert evidence["isolated_debris_very_dark_maximum_chroma"] == 20
    assert evidence["isolated_debris_nuclear_minimum_blue_ratio"] == 1.08
    assert evidence["isolated_debris_micro_bin_size_px"] == 16
    assert evidence["isolated_debris_micro_minimum_nuclear_fraction"] == 0.01


def test_micro_island_gate_separates_debris_merged_by_coarse_grid() -> None:
    labels = np.zeros((128, 128), dtype=np.uint32)
    labels[12:28, 12:28] = 1
    labels[76:100, 88:112] = 2
    image = np.full((128, 128, 3), (245, 243, 241), dtype=np.uint8)
    # At 64 px, the viable tissue and detached DAB island occupy diagonally
    # adjacent dense bins, so the coarse 8-connected support joins them. At
    # 16 px, a blank column separates the weakly nuclear island.
    image[:64, :64] = (125, 82, 42)
    image[4:20:2, 4:60:2] = (72, 58, 128)
    image[64:128, 76:128] = (125, 82, 42)

    qc = _filter_flat_background_instances(
        labels,
        2,
        read_rgb=lambda x, y, width, height: image[y : y + height, x : x + width],
        block_size=128,
        require_isolated_debris_gate=True,
    )

    assert set(np.unique(labels)) == {0, 1}
    assert qc["micro_island_debris_instances_removed"] == 1
    assert qc["micro_island_debris_pixels_removed"] == 576


def test_nuclear_escape_preserves_thin_tissue_but_micro_gate_removes_debris() -> None:
    labels = np.zeros((160, 160), dtype=np.uint32)
    labels[8:56, 8:56] = 1
    labels[16:112, 80:88] = 2
    labels[128:144, 128:144] = 3
    image = np.full((160, 160, 3), (245, 243, 241), dtype=np.uint8)
    image[8:56, 8:56] = (125, 82, 42)
    image[12:52:8, 12:52:8] = (72, 58, 128)
    # An organized, nuclear-rich epithelial strip is too thin to fill 80% of
    # any coarse context bin. Per-instance nuclear escape must preserve it.
    image[16:112, 80:88] = (125, 82, 42)
    image[20:108:8, 82:86] = (72, 58, 128)
    # A compact brown island has just enough chromatic pixels to pass the
    # per-instance escape, but less than 1% component-level nuclear support.
    # The 16 px micro-island backstop must still reject it.
    image[128:144, 128:144] = (125, 82, 42)
    image[130, 130:132] = (72, 58, 128)

    qc = _filter_flat_background_instances(
        labels,
        3,
        read_rgb=lambda x, y, width, height: image[y : y + height, x : x + width],
        block_size=160,
        require_isolated_debris_gate=True,
    )

    assert set(np.unique(labels)) == {0, 1, 2}
    assert qc["micro_island_debris_instances_removed"] == 1
    assert qc["micro_island_debris_pixels_removed"] == 256
    assert qc["instance_evidence"]["isolated_debris_allow_nuclear_escape"] is True


def test_adaptive_nuclear_escape_rescues_pale_nucleated_stroma() -> None:
    labels = np.zeros((80, 160), dtype=np.uint32)
    labels[16:48, 16:48] = 1
    labels[16:48, 96:128] = 2
    image = np.full((80, 160, 3), (245, 243, 241), dtype=np.uint8)
    # Both instances contain pale cytoplasm and therefore need 21 pixels to
    # pass the fixed 2% global nuclear gate.  Twelve chromatic nuclear pixels
    # validate the stromal cell through the adaptive 0.5% escape, while the
    # aneuclear brown fragment remains rejected.
    image[16:48, 16:48] = (224, 218, 220)
    image[20:23, 20:24] = (105, 82, 148)
    image[16:48, 96:128] = (125, 82, 42)

    qc = _filter_flat_background_instances(
        labels,
        2,
        read_rgb=lambda x, y, width, height: image[y : y + height, x : x + width],
        block_size=160,
        require_nuclear_support=True,
        nuclear_minimum_optical_density=0.15,
        nuclear_minimum_pixels=8,
        nuclear_minimum_fraction=0.02,
        require_isolated_debris_gate=True,
        isolated_debris_micro_maximum_component_bins=1,
        # This fixture isolates the nuclear-gate interaction; sparse pale-glass
        # rejection has independent coverage below.
        isolated_debris_glass_minimum_area=10_000,
    )

    assert set(np.unique(labels)) == {0, 1}
    assert qc["nuclear_unsupported_instances_removed"] == 1
    assert qc["nuclear_unsupported_pixels_removed"] == 1_024


def test_global_nuclear_escape_preserves_nonblue_ihc_nuclei_only() -> None:
    labels = np.zeros((96, 192), dtype=np.uint32)
    labels[16:48, 16:48] = 1
    labels[16:48, 80:112] = 2
    labels[16:48, 144:176] = 3
    image = np.full((96, 192, 3), (245, 243, 241), dtype=np.uint8)
    # Weak-counterstain/brown nuclear IHC can leave a real nucleus gray-brown,
    # so it passes hematoxylin unmixing without satisfying a blue-dominance
    # rule.  The isolated-context gate must not delete that CPSAM cell.
    image[16:48, 16:48] = (224, 218, 220)
    image[20:28, 20:24] = (160, 145, 135)
    # DAB-only debris has no hematoxylin evidence and must remain rejected.
    image[16:48, 80:112] = (125, 82, 42)
    # Neutral-dark debris can project onto the hematoxylin basis, but the
    # existing neutral-dark backstop must still reject it after the escape.
    image[16:48, 144:176] = (80, 70, 75)

    qc = _filter_flat_background_instances(
        labels,
        3,
        read_rgb=lambda x, y, width, height: image[y : y + height, x : x + width],
        block_size=192,
        require_nuclear_support=True,
        nuclear_minimum_optical_density=0.12,
        nuclear_minimum_pixels=8,
        nuclear_minimum_fraction=0.02,
        require_isolated_debris_gate=True,
    )

    assert set(np.unique(labels)) == {0, 1}
    assert qc["nuclear_unsupported_instances_removed"] == 1
    assert qc["isolated_debris_unsupported_instances_removed"] == 1


def test_organized_component_escape_preserves_brown_tissue_not_small_debris() -> None:
    labels = np.zeros((128, 128), dtype=np.uint32)
    labels[8:24, 8:120] = 1
    labels[96:112, 96:112] = 2
    image = np.full((128, 128, 3), (245, 243, 241), dtype=np.uint8)
    # Strong DAB can obscure the blue chromatic signal in a real epithelial
    # strip. Its connected CPSAM prediction is large enough to establish
    # organized tissue without allowing a small detached island to self-rescue.
    image[8:24, 8:120] = (125, 82, 42)
    image[96:112, 96:112] = (125, 82, 42)

    qc = _filter_flat_background_instances(
        labels,
        2,
        read_rgb=lambda x, y, width, height: image[y : y + height, x : x + width],
        block_size=128,
        require_isolated_debris_gate=True,
        isolated_debris_micro_maximum_component_bins=1,
        isolated_debris_organized_minimum_component_pixels=1_000,
    )

    assert set(np.unique(labels)) == {0, 1}
    assert qc["organized_tissue_escape_instances"] == 1
    assert qc["organized_tissue_escape_pixels"] == 1_792
    assert qc["isolated_debris_unsupported_instances_removed"] == 1
    assert qc["isolated_debris_unsupported_pixels_removed"] == 256


def test_organized_component_escape_bypasses_adaptive_blue_core_gate() -> None:
    labels = np.zeros((128, 128), dtype=np.uint32)
    labels[8:24, 8:120] = 1
    labels[96:112, 96:112] = 2
    image = np.full((128, 128, 3), (245, 243, 241), dtype=np.uint8)
    image[labels > 0] = (125, 82, 42)

    qc = _filter_flat_background_instances(
        labels,
        2,
        read_rgb=lambda x, y, width, height: image[y : y + height, x : x + width],
        block_size=128,
        require_adaptive_nuclear_core=True,
        adaptive_nuclear_core_minimum_area=100,
        require_isolated_debris_gate=True,
        isolated_debris_micro_maximum_component_bins=1,
        isolated_debris_organized_minimum_component_pixels=1_000,
    )

    assert set(np.unique(labels)) == {0, 1}
    assert qc["organized_tissue_escape_instances"] == 1
    assert qc["organized_tissue_escape_pixels"] == 1_792
    assert qc["adaptive_nuclear_core_unsupported_instances_removed"] == 1
    assert qc["adaptive_nuclear_core_unsupported_pixels_removed"] == 256


def test_organized_component_escape_bypasses_global_nuclear_gate() -> None:
    labels = np.zeros((128, 128), dtype=np.uint32)
    labels[8:24, 8:120] = 1
    labels[96:112, 96:112] = 2
    image = np.full((128, 128, 3), (245, 243, 241), dtype=np.uint8)
    image[labels > 0] = (125, 82, 42)

    qc = _filter_flat_background_instances(
        labels,
        2,
        read_rgb=lambda x, y, width, height: image[y : y + height, x : x + width],
        block_size=128,
        require_nuclear_support=True,
        nuclear_minimum_optical_density=0.15,
        nuclear_minimum_pixels=8,
        nuclear_minimum_fraction=0.02,
        require_isolated_debris_gate=True,
        isolated_debris_micro_maximum_component_bins=1,
        isolated_debris_organized_minimum_component_pixels=1_000,
    )

    assert set(np.unique(labels)) == {0, 1}
    assert qc["organized_tissue_escape_instances"] == 1
    assert qc["organized_tissue_escape_pixels"] == 1_792
    assert qc["nuclear_unsupported_instances_removed"] == 1
    assert qc["nuclear_unsupported_pixels_removed"] == 256


def test_organized_component_escape_rejects_fragmented_cell_mosaic() -> None:
    labels = np.zeros((80, 128), dtype=np.uint32)
    labels[8:24, 8:40] = 1
    labels[8:24, 40:72] = 2
    for index, x0 in enumerate(range(8, 72, 16), start=3):
        labels[48:64, x0 : x0 + 16] = index
    image = np.full((80, 128, 3), (245, 243, 241), dtype=np.uint8)
    image[labels > 0] = (125, 116, 113)

    qc = _filter_flat_background_instances(
        labels,
        6,
        read_rgb=lambda x, y, width, height: image[y : y + height, x : x + width],
        block_size=128,
        require_isolated_debris_gate=True,
        isolated_debris_organized_minimum_component_pixels=1_000,
        isolated_debris_organized_minimum_mean_instance_pixels=350,
    )

    assert set(np.unique(labels)) == {0, 1, 2}
    assert qc["organized_tissue_escape_instances"] == 2
    assert qc["organized_tissue_escape_pixels"] == 1_024
    assert qc["isolated_debris_unsupported_instances_removed"] == 4
    assert qc["isolated_debris_unsupported_pixels_removed"] == 1_024


def test_organized_escape_uses_bounded_overlapping_windows() -> None:
    labels = np.zeros((64, 144), dtype=np.uint32)
    labels[8:24, 0:32] = 1
    labels[8:24, 32:64] = 2
    for index, x0 in enumerate(range(64, 128, 16), start=3):
        labels[8:24, x0 : x0 + 16] = index
    image = np.full((64, 144, 3), (245, 243, 241), dtype=np.uint8)
    image[labels > 0] = (125, 116, 113)

    qc = _filter_flat_background_instances(
        labels,
        6,
        read_rgb=lambda x, y, width, height: image[y : y + height, x : x + width],
        block_size=144,
        require_isolated_debris_gate=True,
        isolated_debris_micro_maximum_component_bins=1,
        isolated_debris_organized_window_size=64,
        isolated_debris_organized_window_overlap=16,
        isolated_debris_organized_minimum_component_pixels=800,
        isolated_debris_organized_compact_minimum_component_pixels=2_000,
        isolated_debris_organized_minimum_mean_instance_pixels=350,
    )

    assert set(np.unique(labels)) == {0, 1, 2}
    assert qc["organized_tissue_escape_instances"] == 2
    assert qc["organized_tissue_escape_pixels"] == 1_024
    assert qc["isolated_debris_unsupported_instances_removed"] == 4
    evidence = qc["instance_evidence"]
    assert evidence["isolated_debris_organized_window_size_px"] == 64
    assert evidence["isolated_debris_organized_window_overlap_px"] == 16


def test_oversized_chromatic_artifact_gate_rejects_giant_blue_blob() -> None:
    labels = np.zeros((120, 120), dtype=np.uint32)
    labels[0:50, 0:100] = 1
    labels[80:100, 0:50] = 2
    image = np.full((120, 120, 3), (245, 243, 241), dtype=np.uint8)
    image[labels > 0] = (85, 55, 145)

    qc = _filter_flat_background_instances(
        labels,
        2,
        read_rgb=lambda x, y, width, height: image[y : y + height, x : x + width],
        block_size=120,
        require_isolated_debris_gate=True,
        isolated_debris_micro_maximum_component_bins=1,
        isolated_debris_oversized_chromatic_minimum_area=4_000,
        isolated_debris_oversized_chromatic_minimum_area_um2=1_000.0,
        isolated_debris_oversized_chromatic_minimum_fraction=0.50,
    )

    assert set(np.unique(labels)) == {0, 2}
    assert qc["oversized_chromatic_artifact_instances_removed"] == 1
    assert qc["oversized_chromatic_artifact_pixels_removed"] == 5_000
    evidence = qc["instance_evidence"]
    assert evidence["isolated_debris_oversized_chromatic_minimum_area_pixels"] == 4_000
    assert evidence["isolated_debris_oversized_chromatic_minimum_area_um2"] == 1_000.0


def test_oversized_brown_filter_rejects_one_impossible_fragment() -> None:
    labels = np.zeros((128, 128), dtype=np.uint32)
    labels[8:88, 8:88] = 1
    labels[96:116, 96:116] = 2
    image = np.full((128, 128, 3), (245, 243, 241), dtype=np.uint8)
    image[labels > 0] = (150, 135, 115)
    # Enough blue pixels establish genuine tissue context and avoid the older
    # fold criterion, but remain far below one percent of the giant fragment.
    image[12:17, 12:20] = (72, 58, 128)
    image[100:104, 100:104] = (72, 58, 128)

    qc = _filter_flat_background_instances(
        labels,
        2,
        read_rgb=lambda x, y, width, height: image[y : y + height, x : x + width],
        block_size=128,
        require_isolated_debris_gate=True,
        isolated_debris_micro_maximum_component_bins=1,
        isolated_debris_oversized_chromatic_minimum_area=4_000,
        isolated_debris_oversized_chromatic_minimum_area_um2=1_000.0,
        isolated_debris_oversized_brown_gate=True,
    )

    assert set(np.unique(labels)) == {0, 2}
    assert qc["detached_fragment_instances_removed"] == 1
    assert qc["detached_fragment_pixels_removed"] == 6_400
    evidence = qc["instance_evidence"]
    assert evidence["isolated_debris_oversized_brown_gate"] is True
    assert (
        evidence["isolated_debris_oversized_brown_minimum_instance_area_um2"] == 1_000.0
    )
    assert evidence["isolated_debris_oversized_brown_maximum_mean_intensity"] == 180.0
    assert evidence["isolated_debris_self_dense_glass_gate"] is False
    assert evidence["isolated_debris_diffuse_degenerated_gate"] is False
    assert evidence["isolated_debris_neutral_precipitate_gate"] is False
    assert evidence["isolated_debris_protect_organized_from_necrotic"] is False
    assert evidence["reconcile_enclosed_cytoplasmic_children"] is False


def test_clustered_brown_filter_removes_anuclear_cluster_and_preserves_cell() -> None:
    labels = np.zeros((160, 160), dtype=np.uint32)
    locations = ((48, 48), (48, 82), (82, 48), (82, 82))
    for index, (y0, x0) in enumerate(locations, start=1):
        labels[y0 : y0 + 30, x0 : x0 + 30] = index
    labels[8:28, 8:28] = 5
    image = np.full((160, 160, 3), (245, 243, 241), dtype=np.uint8)
    image[labels > 0] = (150, 130, 110)
    image[labels == 5] = (72, 58, 128)
    # A tiny amount of blue support mirrors the real blurred fragment: enough
    # to survive the broad context gate, but well below this cluster detector's
    # two-percent nuclear ceiling.
    for y0, x0 in locations:
        image[y0, x0 : x0 + 6] = (72, 58, 128)

    qc = _filter_flat_background_instances(
        labels,
        5,
        read_rgb=lambda x, y, width, height: image[y : y + height, x : x + width],
        block_size=160,
        require_isolated_debris_gate=True,
        isolated_debris_micro_maximum_component_bins=1,
        isolated_debris_clustered_brown_gate=True,
        isolated_debris_clustered_brown_minimum_instance_area=800,
        isolated_debris_clustered_brown_minimum_instance_area_um2=200.0,
        isolated_debris_clustered_brown_maximum_instance_area=1_200,
        isolated_debris_clustered_brown_maximum_instance_area_um2=300.0,
        isolated_debris_clustered_brown_minimum_component_area=3_000,
        isolated_debris_clustered_brown_minimum_component_area_um2=750.0,
    )

    assert set(np.unique(labels)) == {0, 5}
    assert qc["clustered_brown_fragment_instances_removed"] == 4
    assert qc["clustered_brown_fragment_pixels_removed"] == 3_600
    evidence = qc["instance_evidence"]
    assert evidence["isolated_debris_clustered_brown_gate"] is True
    assert (
        evidence["isolated_debris_clustered_brown_minimum_instance_area_um2"] == 200.0
    )
    assert (
        evidence["isolated_debris_clustered_brown_maximum_instance_area_um2"] == 300.0
    )
    assert (
        evidence["isolated_debris_clustered_brown_minimum_component_area_um2"] == 750.0
    )


def test_fold_artifact_gate_rejects_clustered_muted_aneuclear_compartments() -> None:
    labels = np.zeros((12, 12), dtype=np.uint32)
    labels[2:7, 2:6] = 1
    labels[2:7, 6:10] = 2
    red_blue = np.full(labels.shape, 25, dtype=np.int16)
    intensity = np.full(labels.shape, 110, dtype=np.uint8)
    areas = np.array([0, 1700, 1900], dtype=np.uint64)
    nuclear = np.array([0, 0, 1], dtype=np.uint64)

    artifact = _fold_artifact_instances(
        labels,
        red_blue,
        intensity,
        nuclear,
        areas,
        minimum_instance_area=1400,
        minimum_component_area=3500,
        maximum_nuclear_fraction=0.005,
        maximum_mean_red_blue_difference=40.0,
        maximum_mean_intensity=180.0,
    )

    assert artifact.tolist() == [False, True, True]


def test_fold_artifact_gate_rejects_elongated_dense_nuclear_supported_fold() -> None:
    labels = np.zeros((24, 30), dtype=np.uint32)
    labels[3:7, 2:6] = 1
    labels[3:7, 6:10] = 2
    # A bounded gap represents the measured sampling discontinuity across the
    # 4312/008 crushed band; the 8-bin dilation can bridge at most 64 native px.
    labels[3:7, 22:26] = 3
    # A fourth dark giant nearby makes the full dilated component compact. It
    # must not hide the three-member horizontal fold or be removed with it.
    labels[15:19, 6:10] = 4
    red_blue = np.full(labels.shape, 40, dtype=np.int16)
    intensity = np.full(labels.shape, 90, dtype=np.uint8)
    areas = np.array([0, 1700, 1900, 1800, 1700], dtype=np.uint64)
    # All three compartments have enough blue-core evidence to evade the
    # original aneuclear fold branch.
    nuclear = np.array([0, 100, 100, 100, 100], dtype=np.uint64)

    artifact = _fold_artifact_instances(
        labels,
        red_blue,
        intensity,
        nuclear,
        areas,
        minimum_instance_area=1400,
        minimum_component_area=3500,
        maximum_nuclear_fraction=0.005,
        maximum_mean_red_blue_difference=40.0,
        maximum_mean_intensity=180.0,
        # Dense attached folds intentionally bypass the compact-removal lookup;
        # their geometry/color gate supplies the independent safeguard.
        aneuclear_removal_eligible=np.zeros(5, dtype=bool),
    )

    assert artifact.tolist() == [False, True, True, True, False]


@pytest.mark.parametrize(
    ("layout", "red_blue_value", "intensity_value"),
    (
        ("compact", 40, 90),
        ("two_instances", 40, 90),
        ("elongated", 70, 90),
        ("elongated", 40, 130),
    ),
)
def test_dense_fold_guard_preserves_nonmatching_large_cell_groups(
    layout: str,
    red_blue_value: int,
    intensity_value: int,
) -> None:
    labels = np.zeros((12, 18), dtype=np.uint32)
    if layout == "compact":
        labels[3:7, 4:8] = 1
        labels[3:7, 8:12] = 2
        labels[7:11, 4:8] = 3
    else:
        labels[3:7, 2:6] = 1
        labels[3:7, 6:10] = 2
        if layout == "elongated":
            labels[3:7, 10:14] = 3
    red_blue = np.full(labels.shape, red_blue_value, dtype=np.int16)
    intensity = np.full(labels.shape, intensity_value, dtype=np.uint8)
    areas = np.array([0, 1700, 1900, 1800], dtype=np.uint64)
    nuclear = np.array([0, 100, 100, 100], dtype=np.uint64)

    artifact = _fold_artifact_instances(
        labels,
        red_blue,
        intensity,
        nuclear,
        areas,
        minimum_instance_area=1400,
        minimum_component_area=3500,
        maximum_nuclear_fraction=0.005,
        maximum_mean_red_blue_difference=40.0,
        maximum_mean_intensity=180.0,
    )

    assert not np.any(artifact)


def test_detached_fragment_gate_rejects_sparse_elongated_oversized_cluster() -> None:
    labels = np.zeros((128, 256), dtype=np.uint32)
    for index in range(5):
        labels[60:66, 60 + index * 13 : 72 + index * 13] = index + 1
    labels[61:65, 124:128] = 6
    prediction = (labels > 0).astype(np.uint32) * 16
    pixels = np.full(labels.shape, 16, dtype=np.uint32)
    areas = np.array([0, 1200, 1300, 1400, 1500, 1600, 200], dtype=np.uint64)
    red_blue_sum = areas.astype(np.float64) * 12.0
    intensity_sum = areas.astype(np.float64) * 155.0

    artifact = _detached_oversized_fragment_instances(
        prediction,
        pixels,
        labels,
        areas,
        red_blue_sum,
        intensity_sum,
        bin_size=4,
        minimum_instance_area=1000,
        minimum_component_area=5000,
        minimum_instances=3,
        minimum_aspect_ratio=3.0,
        maximum_mean_red_blue_difference=35.0,
        maximum_mean_intensity=220.0,
        context_window_size=128,
        maximum_prediction_fraction=0.20,
    )

    assert artifact.tolist() == [False, True, True, True, True, True, True]


def test_self_dense_glass_gate_rejects_only_isolated_compact_pseudo_cell() -> None:
    labels = np.zeros((100, 220), dtype=np.uint32)
    labels[30:70, 20:60] = 1
    labels[30:70, 150:190] = 2
    pixels = np.full(labels.shape, 16, dtype=np.uint32)
    prediction = (labels > 0).astype(np.uint32) * 16
    # The second equally pale object is embedded in a predicted tissue field.
    prediction[10:90, 125:215] = np.maximum(prediction[10:90, 125:215], 8)
    areas = np.array([0, 25_600, 25_600], dtype=np.uint64)
    red_blue_sum = np.zeros(3, dtype=np.float64)
    intensity_sum = areas.astype(np.float64) * 210.0

    artifact = _self_dense_glass_artifact_instances(
        prediction,
        pixels,
        labels,
        areas,
        red_blue_sum,
        intensity_sum,
        bin_size=4,
        minimum_instance_area=10_000,
        minimum_mean_red_blue_difference=-10.0,
        maximum_mean_red_blue_difference=10.0,
        minimum_mean_intensity=200.0,
        maximum_aspect_ratio=1.8,
        minimum_sampled_fill_fraction=0.55,
        minimum_sampled_area_fraction=0.40,
        ring_dilation_bins=16,
        maximum_ring_prediction_fraction=0.03,
    )

    assert artifact.tolist() == [False, True, False]


def test_enclosed_cytoplasmic_child_reconciliation_is_stain_gated() -> None:
    labels = np.zeros((48, 72), dtype=np.uint32)
    labels[4:24, 4:28] = 1
    labels[10:15, 12:17] = 2
    # The same nested topology under a weakly red-over-blue parent is a
    # negative control and must remain separate.
    labels[4:24, 36:60] = 3
    labels[10:15, 44:49] = 4
    # An ordinary adjacent child has background contact and is not enclosed.
    labels[32:42, 4:14] = 5
    labels[34:40, 14:20] = 6
    areas = np.bincount(labels.ravel(), minlength=7).astype(np.uint64)
    red_blue_sum = np.zeros(7, dtype=np.float64)
    red_blue_sum[1] = areas[1] * 70.0
    red_blue_sum[2] = areas[2] * 20.0
    red_blue_sum[3] = areas[3] * 15.0
    intensity_sum = areas.astype(np.float64) * 120.0

    qc = _merge_enclosed_cytoplasmic_child_instances(
        labels,
        areas,
        red_blue_sum,
        intensity_sum,
        block_size=40,
        block_overlap=12,
        minimum_child_area=20,
        maximum_child_area=80,
        minimum_parent_area_ratio=2.0,
        minimum_dominant_contact_fraction=0.95,
        maximum_child_neighbors=2,
        minimum_parent_mean_red_blue_difference=45.0,
        maximum_parent_mean_intensity=190.0,
    )

    assert qc == {
        "enclosed_cytoplasmic_child_instances_merged": 1,
        "enclosed_cytoplasmic_child_pixels_merged": 25,
    }
    assert np.all(labels[10:15, 12:17] == 1)
    assert np.all(labels[10:15, 44:49] == 4)
    assert np.all(labels[34:40, 14:20] == 6)


def test_satellite_debris_gate_rejects_clustered_mid_intensity_fragments() -> None:
    labels = np.zeros((128, 256), dtype=np.uint32)
    for index in range(4):
        labels[60:66, 60 + index * 18 : 72 + index * 18] = index + 1
    prediction = (labels > 0).astype(np.uint32) * 16
    pixels = np.full(labels.shape, 16, dtype=np.uint32)
    areas = np.array([0, 900, 1000, 1100, 1200], dtype=np.uint64)
    nuclear = np.array([0, 0, 2, 5, 8], dtype=np.uint64)
    red_blue_sum = areas.astype(np.float64) * 25.0
    intensity_sum = areas.astype(np.float64) * 175.0

    artifact = _sparse_anuclear_satellite_instances(
        prediction,
        pixels,
        labels,
        areas,
        nuclear,
        red_blue_sum,
        intensity_sum,
        bin_size=4,
        maximum_instance_area=1500,
        minimum_component_area=2500,
        context_window_size=256,
    )

    assert artifact.tolist() == [False, True, True, True, True]


def test_satellite_debris_gate_preserves_dense_nucleated_and_solitary_cells() -> None:
    labels = np.zeros((128, 256), dtype=np.uint32)
    for index in range(4):
        labels[60:66, 60 + index * 18 : 72 + index * 18] = index + 1
    labels[10:20, 220:230] = 5
    pixels = np.full(labels.shape, 16, dtype=np.uint32)
    sparse_prediction = (labels > 0).astype(np.uint32) * 16
    dense_prediction = np.full(labels.shape, 8, dtype=np.uint32)
    dense_prediction[labels > 0] = 16
    areas = np.array([0, 900, 1000, 1100, 1200, 900], dtype=np.uint64)
    nuclear = np.array([0, 60, 60, 60, 60, 0], dtype=np.uint64)
    red_blue_sum = areas.astype(np.float64) * 25.0
    intensity_sum = areas.astype(np.float64) * 175.0
    arguments = {
        "bin_size": 4,
        "maximum_instance_area": 1500,
        "minimum_component_area": 2500,
        "context_window_size": 256,
    }

    nucleated = _sparse_anuclear_satellite_instances(
        sparse_prediction,
        pixels,
        labels,
        areas,
        nuclear,
        red_blue_sum,
        intensity_sum,
        **arguments,
    )
    dense = _sparse_anuclear_satellite_instances(
        dense_prediction,
        pixels,
        labels,
        areas,
        np.zeros_like(nuclear),
        red_blue_sum,
        intensity_sum,
        **arguments,
    )
    solitary_labels = np.where(labels == 5, labels, 0)
    solitary = _sparse_anuclear_satellite_instances(
        sparse_prediction,
        pixels,
        solitary_labels,
        areas,
        np.zeros_like(nuclear),
        red_blue_sum,
        intensity_sum,
        **arguments,
    )

    assert not np.any(nucleated)
    assert not np.any(dense)
    assert not np.any(solitary)


def test_neutral_precipitate_gate_requires_all_sparse_dark_evidence() -> None:
    labels = np.zeros((64, 128), dtype=np.uint32)
    labels[10:12, 10:12] = 1
    labels[10:12, 92:94] = 2
    labels[38:40, 10:12] = 3
    labels[38:40, 42:44] = 4
    labels[38:40, 74:76] = 5
    labels[38:40, 106:108] = 6
    pixels = np.full(labels.shape, 16, dtype=np.uint32)
    prediction = (labels > 0).astype(np.uint32) * 16
    # The second otherwise identical object lies in a dense predicted field.
    prediction[:28, 78:108] = np.maximum(prediction[:28, 78:108], 8)
    areas = np.array([0, 800, 800, 800, 800, 4000, 800], dtype=np.uint64)
    nuclear = np.array([0, 0, 0, 20, 0, 0, 0], dtype=np.uint64)
    neutral_dark = np.array([0, 300, 300, 300, 300, 1500, 300], dtype=np.uint64)
    very_dark = np.array([0, 220, 220, 220, 220, 1100, 100], dtype=np.uint64)
    red_blue_sum = areas.astype(np.float64) * np.array(
        [0.0, 2.0, 2.0, -30.0, 40.0, 2.0, 2.0]
    )
    intensity_sum = areas.astype(np.float64) * 100.0

    artifact = _sparse_neutral_dark_precipitate_instances(
        prediction,
        pixels,
        labels,
        areas,
        nuclear,
        neutral_dark,
        very_dark,
        red_blue_sum,
        intensity_sum,
        bin_size=4,
        minimum_instance_area=100,
        maximum_instance_area=3000,
        minimum_neutral_dark_fraction=0.28,
        minimum_very_dark_fraction=0.20,
        maximum_nuclear_fraction=0.35,
        minimum_mean_red_blue_difference=-10.0,
        maximum_mean_red_blue_difference=15.0,
        maximum_mean_intensity=135.0,
        context_window_size=32,
        maximum_prediction_fraction=0.25,
        minimum_sparse_sample_fraction=0.50,
    )

    assert artifact.tolist() == [False, True, False, False, False, False, False]


def test_magenta_anuclear_gate_rejects_clustered_small_objects() -> None:
    labels = np.zeros((64, 128), dtype=np.uint32)
    for index in range(10):
        y = 24 + (index // 5) * 6
        x = 30 + (index % 5) * 8
        labels[y : y + 3, x : x + 4] = index + 1
    areas = np.array([0, *([100] * 10)], dtype=np.uint64)
    nuclear = np.zeros_like(areas)
    red = areas.astype(np.float64) * 170.0
    green = areas.astype(np.float64) * 130.0
    blue = areas.astype(np.float64) * 175.0

    artifact = _clustered_magenta_anuclear_instances(
        labels,
        areas,
        nuclear,
        red,
        green,
        blue,
        maximum_instance_area=250,
        minimum_component_area=800,
    )

    assert artifact.tolist() == [False, *([True] * 10)]


def test_magenta_anuclear_gate_preserves_biological_and_solitary_controls() -> None:
    labels = np.zeros((96, 192), dtype=np.uint32)
    # Four nearby groups share topology but fail a different scientific guard:
    # blue nuclei, brown DAB, nuclear support, and excessive physical size.
    groups = ((1, 20), (9, 55), (17, 90), (25, 125))
    for first_id, base_x in groups:
        for offset in range(8):
            y = 36 + (offset // 4) * 6
            x = base_x + (offset % 4) * 8
            labels[y : y + 3, x : x + 4] = first_id + offset
    labels[10:13, 175:179] = 33
    areas = np.array([0, *([100] * 32), 100], dtype=np.uint64)
    nuclear = np.zeros_like(areas)
    red = areas.astype(np.float64) * 170.0
    green = areas.astype(np.float64) * 130.0
    blue = areas.astype(np.float64) * 175.0
    # Blue cells are not magenta; DAB cells have blue below green; the third
    # group has convincing nuclear support; the fourth is physically large.
    red[1:9] = areas[1:9] * 105.0
    green[1:9] = areas[1:9] * 90.0
    blue[1:9] = areas[1:9] * 165.0
    red[9:17] = areas[9:17] * 155.0
    green[9:17] = areas[9:17] * 105.0
    blue[9:17] = areas[9:17] * 65.0
    nuclear[17:25] = 10
    areas[25:33] = 400
    red[25:33] = areas[25:33] * 170.0
    green[25:33] = areas[25:33] * 130.0
    blue[25:33] = areas[25:33] * 175.0

    artifact = _clustered_magenta_anuclear_instances(
        labels,
        areas,
        nuclear,
        red,
        green,
        blue,
        maximum_instance_area=250,
        minimum_component_area=600,
    )

    assert not np.any(artifact)


def test_oversized_brown_gate_rejects_giant_anuclear_fragment() -> None:
    areas = np.array([0, 30_000], dtype=np.uint64)
    nuclear = np.array([0, 10], dtype=np.uint64)
    red = areas.astype(np.float64) * 150.0
    green = areas.astype(np.float64) * 135.0
    blue = areas.astype(np.float64) * 115.0

    artifact = _oversized_brown_anuclear_instances(
        areas,
        nuclear,
        red,
        green,
        blue,
        minimum_instance_area=6_000,
    )

    assert artifact.tolist() == [False, True]


def test_oversized_brown_gate_preserves_scientific_controls() -> None:
    areas = np.array([0, 500, 30_000, 30_000, 30_000], dtype=np.uint64)
    nuclear = np.array([0, 0, 6_000, 0, 0], dtype=np.uint64)
    red = areas.astype(np.float64) * 150.0
    green = areas.astype(np.float64) * 135.0
    blue = areas.astype(np.float64) * 115.0
    # A nuclear-rich brown region, a neutral pale region, and a blue region
    # each fail a distinct guard. The first object is a normal cell-sized DAB
    # prediction and remains below the physical-area floor.
    red[3] = green[3] = blue[3] = areas[3] * 225.0
    red[4] = areas[4] * 120.0
    green[4] = areas[4] * 130.0
    blue[4] = areas[4] * 170.0

    artifact = _oversized_brown_anuclear_instances(
        areas,
        nuclear,
        red,
        green,
        blue,
        minimum_instance_area=6_000,
    )

    assert artifact.tolist() == [False, False, False, False, False]


def test_oversized_brown_gate_preserves_large_clear_brown_rim_profile() -> None:
    areas = np.array([0, 6_383], dtype=np.uint64)
    nuclear = np.array([0, 0], dtype=np.uint64)
    red = areas.astype(np.float64) * 206.96
    green = areas.astype(np.float64) * 194.92
    blue = areas.astype(np.float64) * 183.30

    artifact = _oversized_brown_anuclear_instances(
        areas,
        nuclear,
        red,
        green,
        blue,
        minimum_instance_area=4_721,
    )

    assert artifact.tolist() == [False, False]


def test_clustered_brown_gate_rejects_sparse_anuclear_fragment_cluster() -> None:
    labels = np.zeros((96, 96), dtype=np.uint32)
    labels[35:38, 30:33] = 1
    labels[37:40, 36:39] = 2
    labels[42:45, 32:35] = 3
    labels[44:47, 39:42] = 4
    prediction = (labels > 0).astype(np.uint32) * 16
    pixels = np.full(labels.shape, 16, dtype=np.uint32)
    areas = np.array([0, 950, 1100, 1200, 1300], dtype=np.uint64)
    nuclear = np.zeros(5, dtype=np.uint64)
    red = areas.astype(np.float64) * 120.0
    green = areas.astype(np.float64) * 90.0
    blue = areas.astype(np.float64) * 65.0

    artifact = _clustered_brown_anuclear_instances(
        prediction,
        pixels,
        labels,
        areas,
        nuclear,
        red,
        green,
        blue,
        bin_size=4,
        minimum_instance_area=900,
        maximum_instance_area=4000,
        minimum_component_area=4000,
        context_window_size=64,
    )

    assert artifact.tolist() == [False, True, True, True, True]


@pytest.mark.parametrize(
    "control",
    ("dense tissue", "nuclear support", "blue tissue"),
)
def test_clustered_brown_gate_preserves_independent_controls(
    control: str,
) -> None:
    labels = np.zeros((96, 96), dtype=np.uint32)
    labels[35:38, 30:33] = 1
    labels[37:40, 36:39] = 2
    labels[42:45, 32:35] = 3
    labels[44:47, 39:42] = 4
    prediction = (labels > 0).astype(np.uint32) * 16
    pixels = np.full(labels.shape, 16, dtype=np.uint32)
    areas = np.array([0, 950, 1100, 1200, 1300], dtype=np.uint64)
    nuclear = np.zeros(5, dtype=np.uint64)
    red = areas.astype(np.float64) * 120.0
    green = areas.astype(np.float64) * 90.0
    blue = areas.astype(np.float64) * 65.0
    if control == "dense tissue":
        prediction.fill(16)
    elif control == "nuclear support":
        nuclear[1:] = 100
    elif control == "blue tissue":
        blue[1:] = red[1:] + 20

    artifact = _clustered_brown_anuclear_instances(
        prediction,
        pixels,
        labels,
        areas,
        nuclear,
        red,
        green,
        blue,
        bin_size=4,
        minimum_instance_area=900,
        maximum_instance_area=4000,
        minimum_component_area=4000,
        context_window_size=64,
    )

    assert not np.any(artifact)


def test_detached_fragment_gate_preserves_dense_tissue_and_single_cell() -> None:
    labels = np.zeros((128, 256), dtype=np.uint32)
    for index in range(5):
        labels[60:66, 60 + index * 13 : 72 + index * 13] = index + 1
    labels[10:20, 220:230] = 6
    sparse_prediction = (labels > 0).astype(np.uint32) * 16
    dense_prediction = np.full(labels.shape, 8, dtype=np.uint32)
    dense_prediction[labels > 0] = 16
    pixels = np.full(labels.shape, 16, dtype=np.uint32)
    areas = np.array([0, 1200, 1300, 1400, 1500, 1600, 5000], dtype=np.uint64)
    red_blue_sum = areas.astype(np.float64) * 12.0
    intensity_sum = areas.astype(np.float64) * 155.0
    arguments = {
        "bin_size": 4,
        "minimum_instance_area": 1000,
        "minimum_component_area": 5000,
        "minimum_instances": 3,
        "minimum_aspect_ratio": 3.0,
        "maximum_mean_red_blue_difference": 35.0,
        "maximum_mean_intensity": 220.0,
        "context_window_size": 512,
        "maximum_prediction_fraction": 0.20,
    }

    dense_artifact = _detached_oversized_fragment_instances(
        dense_prediction,
        pixels,
        labels,
        areas,
        red_blue_sum,
        intensity_sum,
        **arguments,
    )
    solitary_labels = np.where(labels == 6, labels, 0)
    solitary_artifact = _detached_oversized_fragment_instances(
        sparse_prediction,
        pixels,
        solitary_labels,
        areas,
        red_blue_sum,
        intensity_sum,
        **arguments,
    )

    assert not np.any(dense_artifact)
    assert not np.any(solitary_artifact)


def test_detached_fragment_gate_rejects_compact_crushed_blue_cluster() -> None:
    labels = np.zeros((128, 128), dtype=np.uint32)
    labels[52:66, 52:66] = 1
    labels[58:72, 67:81] = 2
    labels[62:68, 82:88] = 3
    prediction = (labels > 0).astype(np.uint32) * 16
    pixels = np.full(labels.shape, 16, dtype=np.uint32)
    areas = np.array([0, 1700, 1900, 250], dtype=np.uint64)
    red_blue_sum = areas.astype(np.float64) * -8.0
    intensity_sum = areas.astype(np.float64) * 150.0

    artifact = _detached_oversized_fragment_instances(
        prediction,
        pixels,
        labels,
        areas,
        red_blue_sum,
        intensity_sum,
        bin_size=4,
        minimum_instance_area=1000,
        minimum_component_area=5000,
        minimum_instances=3,
        minimum_aspect_ratio=3.0,
        maximum_mean_red_blue_difference=35.0,
        maximum_mean_intensity=220.0,
        context_window_size=512,
        maximum_prediction_fraction=0.20,
        compact_minimum_instance_area=1400,
        compact_minimum_component_area=3500,
        compact_minimum_instances=2,
        compact_maximum_mean_red_blue_difference=0.0,
        compact_maximum_mean_intensity=180.0,
    )

    assert artifact.tolist() == [False, True, True, True]


def test_detached_compact_gate_preserves_dense_and_warm_large_cells() -> None:
    labels = np.zeros((128, 128), dtype=np.uint32)
    labels[52:66, 52:66] = 1
    labels[58:72, 67:81] = 2
    pixels = np.full(labels.shape, 16, dtype=np.uint32)
    sparse_prediction = (labels > 0).astype(np.uint32) * 16
    dense_prediction = np.full(labels.shape, 8, dtype=np.uint32)
    dense_prediction[labels > 0] = 16
    areas = np.array([0, 1700, 1900], dtype=np.uint64)
    intensity_sum = areas.astype(np.float64) * 150.0
    arguments = {
        "bin_size": 4,
        "minimum_instance_area": 1000,
        "minimum_component_area": 5000,
        "minimum_instances": 3,
        "minimum_aspect_ratio": 3.0,
        "maximum_mean_red_blue_difference": 35.0,
        "maximum_mean_intensity": 220.0,
        "context_window_size": 512,
        "maximum_prediction_fraction": 0.20,
        "compact_minimum_instance_area": 1400,
        "compact_minimum_component_area": 3500,
        "compact_minimum_instances": 2,
        "compact_maximum_mean_red_blue_difference": 0.0,
        "compact_maximum_mean_intensity": 180.0,
    }

    dense = _detached_oversized_fragment_instances(
        dense_prediction,
        pixels,
        labels,
        areas,
        areas.astype(np.float64) * -8.0,
        intensity_sum,
        **arguments,
    )
    warm = _detached_oversized_fragment_instances(
        sparse_prediction,
        pixels,
        labels,
        areas,
        areas.astype(np.float64) * 8.0,
        intensity_sum,
        **arguments,
    )

    assert not np.any(dense)
    assert not np.any(warm)


def test_fold_artifact_gate_preserves_brown_bright_and_nuclear_large_cells() -> None:
    labels = np.zeros((12, 16), dtype=np.uint32)
    labels[2:7, 1:5] = 1
    labels[2:7, 6:10] = 2
    labels[2:7, 11:15] = 3
    red_blue = np.full(labels.shape, 25, dtype=np.int16)
    red_blue[labels == 1] = 75
    intensity = np.full(labels.shape, 110, dtype=np.uint8)
    intensity[labels == 2] = 205
    areas = np.array([0, 4000, 4000, 4000], dtype=np.uint64)
    nuclear = np.array([0, 0, 0, 40], dtype=np.uint64)

    artifact = _fold_artifact_instances(
        labels,
        red_blue,
        intensity,
        nuclear,
        areas,
        minimum_instance_area=1400,
        minimum_component_area=3500,
        maximum_nuclear_fraction=0.005,
        maximum_mean_red_blue_difference=40.0,
        maximum_mean_intensity=180.0,
    )

    assert artifact.tolist() == [False, False, False, False]


def test_glass_artifact_gate_rejects_sparse_bright_lavender_prediction() -> None:
    shape = (64, 64)
    labels = np.zeros(shape, dtype=np.uint32)
    labels[28:36, 28:36] = 1
    # A fragmented field occupies about 20% locally: not empty glass, but far
    # below the coverage of a viable cell sheet.
    prediction = np.full(shape, 3, dtype=np.uint32)
    prediction[labels > 0] = 16
    pixels = np.full(shape, 16, dtype=np.uint32)
    red_blue = np.zeros(shape, dtype=np.int16)
    red_blue[labels == 1] = -8
    intensity = np.full(shape, 245, dtype=np.uint8)
    intensity[labels == 1] = 210

    artifact = _isolated_glass_artifact_instances(
        prediction,
        pixels,
        labels,
        red_blue,
        intensity,
        np.array([0, 500], dtype=np.uint64),
        bin_size=4,
        context_window_size=256,
        minimum_area=100,
        maximum_prediction_fraction=0.30,
        minimum_context_fraction=0.50,
        maximum_mean_red_blue_difference=0.0,
        minimum_mean_intensity=190.0,
    )

    assert artifact.tolist() == [False, True]


def test_glass_artifact_gate_preserves_dense_pale_blue_tissue() -> None:
    shape = (64, 64)
    labels = np.arange(1, 1 + np.prod(shape), dtype=np.uint32).reshape(shape)
    prediction = np.full(shape, 16, dtype=np.uint32)
    pixels = np.full(shape, 16, dtype=np.uint32)
    red_blue = np.full(shape, -8, dtype=np.int16)
    intensity = np.full(shape, 210, dtype=np.uint8)
    areas = np.full(labels.max() + 1, 500, dtype=np.uint64)
    areas[0] = 0

    artifact = _isolated_glass_artifact_instances(
        prediction,
        pixels,
        labels,
        red_blue,
        intensity,
        areas,
        bin_size=4,
        context_window_size=256,
        minimum_area=100,
        maximum_prediction_fraction=0.30,
        minimum_context_fraction=0.50,
        maximum_mean_red_blue_difference=0.0,
        minimum_mean_intensity=190.0,
    )

    assert not np.any(artifact)


def test_glass_artifact_gate_uses_exact_color_for_small_pale_debris() -> None:
    shape = (64, 64)
    labels = np.zeros(shape, dtype=np.uint32)
    labels[28:36, 28:36] = 1
    prediction = (labels > 0).astype(np.uint32) * 16
    pixels = np.full(shape, 16, dtype=np.uint32)
    # The 4 px center sample is unrepresentative of the small object.
    red_blue = np.zeros(shape, dtype=np.int16)
    red_blue[labels == 1] = 75
    intensity = np.full(shape, 245, dtype=np.uint8)
    intensity[labels == 1] = 100
    areas = np.array([0, 500], dtype=np.uint64)

    artifact = _isolated_glass_artifact_instances(
        prediction,
        pixels,
        labels,
        red_blue,
        intensity,
        areas,
        instance_red_blue_sum=np.array([0.0, 10.0 * 500]),
        instance_intensity_sum=np.array([0.0, 210.0 * 500]),
        bin_size=4,
        context_window_size=256,
        minimum_area=100,
        maximum_prediction_fraction=0.30,
        minimum_context_fraction=0.50,
        maximum_mean_red_blue_difference=25.0,
        minimum_mean_intensity=190.0,
    )

    assert artifact.tolist() == [False, True]


def test_glass_artifact_gate_uses_low_stain_context_in_populated_debris() -> None:
    shape = (64, 64)
    labels = np.zeros(shape, dtype=np.uint32)
    labels[28:36, 28:36] = 1
    prediction = np.full(shape, 12, dtype=np.uint32)
    pixels = np.full(shape, 16, dtype=np.uint32)
    red_blue = np.zeros(shape, dtype=np.int16)
    intensity = np.full(shape, 210, dtype=np.uint8)
    areas = np.array([0, 500], dtype=np.uint64)
    stain_pixels = np.full((16, 16), 256, dtype=np.uint64)
    low_stain = np.full((16, 16), 20, dtype=np.uint64)

    artifact = _isolated_glass_artifact_instances(
        prediction,
        pixels,
        labels,
        red_blue,
        intensity,
        areas,
        instance_red_blue_sum=np.array([0.0, 30.0 * 500]),
        instance_intensity_sum=np.array([0.0, 150.0 * 500]),
        stain_counts=low_stain,
        stain_pixel_counts=stain_pixels,
        stain_bin_size=16,
        maximum_context_stain_fraction=0.20,
        low_stain_maximum_mean_red_blue_difference=35.0,
        low_stain_minimum_mean_intensity=120.0,
        bin_size=4,
        context_window_size=64,
        minimum_area=100,
        maximum_prediction_fraction=0.40,
        minimum_context_fraction=0.50,
        maximum_mean_red_blue_difference=25.0,
        minimum_mean_intensity=190.0,
    )

    assert artifact.tolist() == [False, True]


def test_compact_unsupported_gate_rejects_pseudocell_mosaic() -> None:
    labels = np.zeros((384, 384), dtype=np.uint32)
    next_label = 1
    for y0 in range(20, 340, 16):
        for x0 in range(20, 340, 16):
            labels[y0 : y0 + 14, x0 : x0 + 14] = next_label
            next_label += 1
    image = np.full((384, 384, 3), (245, 243, 241), dtype=np.uint8)
    image[labels > 0] = (125, 116, 113)

    qc = _filter_flat_background_instances(
        labels,
        next_label - 1,
        read_rgb=lambda x, y, width, height: image[y : y + height, x : x + width],
        block_size=384,
        require_isolated_debris_gate=True,
        isolated_debris_organized_minimum_component_pixels=25_000,
        isolated_debris_organized_compact_minimum_component_pixels=50_000,
        isolated_debris_organized_minimum_mean_instance_pixels=150,
        isolated_debris_compact_unsupported_minimum_area=50_000,
        isolated_debris_compact_unsupported_minimum_area_um2=10_000.0,
        isolated_debris_compact_unsupported_maximum_aspect_ratio=1.75,
        isolated_debris_compact_unsupported_maximum_mean_instance_pixels=250.0,
        isolated_debris_compact_unsupported_maximum_mean_red_blue_difference=21.0,
        isolated_debris_compact_unsupported_maximum_context_nuclear_fraction=0.425,
        isolated_debris_compact_unsupported_minimum_component_fill_fraction=0.20,
        isolated_debris_compact_unsupported_minimum_instance_fraction=0.50,
    )

    assert set(np.unique(labels)) == {0}
    assert qc["compact_unsupported_mosaic_instances_removed"] == 400
    assert qc["compact_unsupported_mosaic_pixels_removed"] == 78_400
    evidence = qc["instance_evidence"]
    assert evidence["isolated_debris_compact_unsupported_minimum_area_pixels"] == 50_000
    assert evidence["isolated_debris_compact_unsupported_minimum_area_um2"] == 10_000.0


def test_compact_unsupported_gate_preserves_red_brown_tissue_mosaic() -> None:
    labels = np.zeros((384, 384), dtype=np.uint32)
    next_label = 1
    for y0 in range(20, 340, 16):
        for x0 in range(20, 340, 16):
            labels[y0 : y0 + 14, x0 : x0 + 14] = next_label
            next_label += 1
    image = np.full((384, 384, 3), (245, 243, 241), dtype=np.uint8)
    image[labels > 0] = (125, 82, 42)

    qc = _filter_flat_background_instances(
        labels,
        next_label - 1,
        read_rgb=lambda x, y, width, height: image[y : y + height, x : x + width],
        block_size=384,
        require_isolated_debris_gate=True,
        isolated_debris_organized_minimum_component_pixels=25_000,
        isolated_debris_organized_compact_minimum_component_pixels=50_000,
        isolated_debris_organized_minimum_mean_instance_pixels=150,
        isolated_debris_compact_unsupported_minimum_area=50_000,
        isolated_debris_compact_unsupported_minimum_area_um2=10_000.0,
        isolated_debris_compact_unsupported_maximum_aspect_ratio=1.75,
        isolated_debris_compact_unsupported_maximum_mean_instance_pixels=250.0,
        isolated_debris_compact_unsupported_maximum_mean_red_blue_difference=21.0,
        isolated_debris_compact_unsupported_maximum_context_nuclear_fraction=0.425,
        isolated_debris_compact_unsupported_minimum_component_fill_fraction=0.20,
        isolated_debris_compact_unsupported_minimum_instance_fraction=0.50,
    )

    assert set(np.unique(labels)) != {0}
    assert qc["compact_unsupported_mosaic_instances_removed"] == 0


def test_compact_unsupported_gate_preserves_sparse_folded_ring() -> None:
    shape = (256, 256)
    ring = np.zeros(shape, dtype=bool)
    ring[8:248, 8:248] = True
    ring[18:238, 18:238] = False
    prediction_counts = ring.astype(np.uint32) * 16
    pixel_counts = np.full(shape, 16, dtype=np.uint32)
    sampled_labels = np.arange(1, 1 + np.prod(shape), dtype=np.uint32).reshape(shape)
    nuclear_supported = np.zeros(sampled_labels.max() + 1, dtype=bool)
    instance_areas = np.full(sampled_labels.max() + 1, 16, dtype=np.uint64)
    instance_areas[0] = 0
    eligible_instances = np.ones(sampled_labels.max() + 1, dtype=bool)
    eligible_instances[0] = False
    red_blue = np.zeros(shape, dtype=np.int16)

    artifact = _compact_unsupported_prediction_grid(
        prediction_counts,
        pixel_counts,
        sampled_labels,
        red_blue,
        nuclear_supported,
        instance_areas,
        eligible_instances,
        bin_size=4,
        minimum_bin_occupancy_fraction=0.25,
        minimum_component_pixels=50_000,
        maximum_aspect_ratio=1.75,
        maximum_mean_instance_pixels=250.0,
        maximum_mean_red_blue_difference=21.0,
        strong_red_blue_difference=40.0,
        maximum_strong_chromatic_fraction=0.10,
        elongation_ratio=2.0,
        maximum_elongated_instance_fraction=0.50,
        maximum_context_nuclear_supported_fraction=0.425,
        minimum_component_fill_fraction=0.20,
        minimum_instance_fraction=0.50,
    )

    assert not np.any(artifact)


def test_sparse_organized_gate_preserves_large_cell_tissue_fragment() -> None:
    shape = (64, 256)
    prediction_counts = np.full(shape, 2, dtype=np.uint32)
    pixel_counts = np.full(shape, 16, dtype=np.uint32)
    sampled_labels = np.zeros(shape, dtype=np.uint32)
    next_label = 1
    for y0 in range(0, shape[0], 16):
        for x0 in range(0, shape[1], 16):
            sampled_labels[y0 : y0 + 16, x0 : x0 + 16] = next_label
            next_label += 1

    sparse_organized = _locally_organized_prediction_grid(
        prediction_counts,
        pixel_counts,
        sampled_labels,
        np.zeros(shape, dtype=np.int16),
        bin_size=4,
        window_size=1024,
        window_overlap=128,
        minimum_bin_occupancy_fraction=0.10,
        minimum_component_pixels=25_000,
        compact_minimum_component_pixels=50_000,
        minimum_aspect_ratio=4.0,
        minimum_mean_instance_pixels=350.0,
        strong_red_blue_difference=40.0,
        minimum_strong_chromatic_fraction=0.25,
    )
    legacy_organized = _locally_organized_prediction_grid(
        prediction_counts,
        pixel_counts,
        sampled_labels,
        np.zeros(shape, dtype=np.int16),
        bin_size=4,
        window_size=1024,
        window_overlap=128,
        minimum_bin_occupancy_fraction=0.25,
        minimum_component_pixels=25_000,
        compact_minimum_component_pixels=50_000,
        minimum_aspect_ratio=4.0,
        minimum_mean_instance_pixels=350.0,
        strong_red_blue_difference=40.0,
        minimum_strong_chromatic_fraction=0.25,
    )

    assert np.all(sparse_organized)
    assert not np.any(legacy_organized)


def test_sparse_organized_gate_does_not_protect_small_cell_debris_mosaic() -> None:
    shape = (64, 256)
    prediction_counts = np.full(shape, 2, dtype=np.uint32)
    pixel_counts = np.full(shape, 16, dtype=np.uint32)
    sampled_labels = np.arange(1, 1 + np.prod(shape), dtype=np.uint32).reshape(shape)

    organized = _locally_organized_prediction_grid(
        prediction_counts,
        pixel_counts,
        sampled_labels,
        np.zeros(shape, dtype=np.int16),
        bin_size=4,
        window_size=1024,
        window_overlap=128,
        minimum_bin_occupancy_fraction=0.10,
        minimum_component_pixels=25_000,
        compact_minimum_component_pixels=50_000,
        minimum_aspect_ratio=4.0,
        minimum_mean_instance_pixels=350.0,
        strong_red_blue_difference=40.0,
        minimum_strong_chromatic_fraction=0.25,
    )

    assert not np.any(organized)


def test_organized_gate_protects_compact_large_cell_tissue_fragment() -> None:
    shape = (40, 40)
    prediction_counts = np.full(shape, 16, dtype=np.uint32)
    pixel_counts = np.full(shape, 16, dtype=np.uint32)
    sampled_labels = np.zeros(shape, dtype=np.uint32)
    next_label = 1
    for y0 in range(0, shape[0], 5):
        for x0 in range(0, shape[1], 5):
            sampled_labels[y0 : y0 + 5, x0 : x0 + 5] = next_label
            next_label += 1

    organized = _locally_organized_prediction_grid(
        prediction_counts,
        pixel_counts,
        sampled_labels,
        np.zeros(shape, dtype=np.int16),
        bin_size=4,
        window_size=1024,
        window_overlap=128,
        minimum_bin_occupancy_fraction=0.10,
        minimum_component_pixels=10_000,
        compact_minimum_component_pixels=20_000,
        minimum_aspect_ratio=4.0,
        minimum_mean_instance_pixels=350.0,
        strong_red_blue_difference=40.0,
        minimum_strong_chromatic_fraction=0.25,
    )

    assert np.all(organized)


def test_compact_unsupported_gate_preserves_strongly_chromatic_tissue() -> None:
    shape = (96, 96)
    prediction_counts = np.full(shape, 16, dtype=np.uint32)
    pixel_counts = np.full(shape, 16, dtype=np.uint32)
    sampled_labels = np.arange(1, 1 + np.prod(shape), dtype=np.uint32).reshape(shape)
    nuclear_supported = np.zeros(sampled_labels.max() + 1, dtype=bool)
    instance_areas = np.full(sampled_labels.max() + 1, 16, dtype=np.uint64)
    instance_areas[0] = 0
    eligible_instances = np.ones(sampled_labels.max() + 1, dtype=bool)
    eligible_instances[0] = False
    # The legacy component mean remains below 21, while a target-positive
    # subpopulation has the strong brown chromaticity seen in real cJun tissue.
    red_blue = np.full(shape, 8, dtype=np.int16)
    red_blue[:, :20] = 60

    protected = _compact_unsupported_prediction_grid(
        prediction_counts,
        pixel_counts,
        sampled_labels,
        red_blue,
        nuclear_supported,
        instance_areas,
        eligible_instances,
        bin_size=4,
        minimum_bin_occupancy_fraction=0.25,
        minimum_component_pixels=50_000,
        maximum_aspect_ratio=1.75,
        maximum_mean_instance_pixels=250.0,
        maximum_mean_red_blue_difference=21.0,
        strong_red_blue_difference=40.0,
        maximum_strong_chromatic_fraction=0.10,
        elongation_ratio=2.0,
        maximum_elongated_instance_fraction=0.50,
        maximum_context_nuclear_supported_fraction=0.425,
        minimum_component_fill_fraction=0.20,
        minimum_instance_fraction=0.50,
    )
    unprotected = _compact_unsupported_prediction_grid(
        prediction_counts,
        pixel_counts,
        sampled_labels,
        red_blue,
        nuclear_supported,
        instance_areas,
        eligible_instances,
        bin_size=4,
        minimum_bin_occupancy_fraction=0.25,
        minimum_component_pixels=50_000,
        maximum_aspect_ratio=1.75,
        maximum_mean_instance_pixels=250.0,
        maximum_mean_red_blue_difference=21.0,
        strong_red_blue_difference=40.0,
        maximum_strong_chromatic_fraction=0.25,
        elongation_ratio=2.0,
        maximum_elongated_instance_fraction=0.50,
        maximum_context_nuclear_supported_fraction=0.425,
        minimum_component_fill_fraction=0.20,
        minimum_instance_fraction=0.50,
    )

    assert not np.any(protected)
    assert np.any(unprotected)


def test_compact_unsupported_gate_preserves_mixed_nuclear_tissue() -> None:
    shape = (96, 96)
    prediction_counts = np.full(shape, 16, dtype=np.uint32)
    pixel_counts = np.full(shape, 16, dtype=np.uint32)
    sampled_labels = np.arange(1, 1 + np.prod(shape), dtype=np.uint32).reshape(shape)
    nuclear_supported = np.zeros(sampled_labels.max() + 1, dtype=bool)
    # A compact viable IHC fragment can contain weak/neutral cells intermixed
    # with a substantial supported nuclear population.  The former 42.5%
    # ceiling erased the weak cells even though one quarter of their immediate
    # predicted context supplied independent nuclear evidence.
    supported_grid = np.zeros(shape, dtype=bool)
    supported_grid[::2, ::2] = True
    nuclear_supported[sampled_labels[supported_grid]] = True
    instance_areas = np.full(sampled_labels.max() + 1, 16, dtype=np.uint64)
    instance_areas[0] = 0
    eligible_instances = np.ones(sampled_labels.max() + 1, dtype=bool)
    eligible_instances[0] = False
    red_blue = np.zeros(shape, dtype=np.int16)

    protected = _compact_unsupported_prediction_grid(
        prediction_counts,
        pixel_counts,
        sampled_labels,
        red_blue,
        nuclear_supported,
        instance_areas,
        eligible_instances,
        bin_size=4,
        minimum_bin_occupancy_fraction=0.25,
        minimum_component_pixels=50_000,
        maximum_aspect_ratio=1.75,
        maximum_mean_instance_pixels=250.0,
        maximum_mean_red_blue_difference=21.0,
        strong_red_blue_difference=40.0,
        maximum_strong_chromatic_fraction=0.10,
        elongation_ratio=2.0,
        maximum_elongated_instance_fraction=0.50,
        maximum_context_nuclear_supported_fraction=0.20,
        minimum_component_fill_fraction=0.20,
        minimum_instance_fraction=0.50,
    )
    legacy = _compact_unsupported_prediction_grid(
        prediction_counts,
        pixel_counts,
        sampled_labels,
        red_blue,
        nuclear_supported,
        instance_areas,
        eligible_instances,
        bin_size=4,
        minimum_bin_occupancy_fraction=0.25,
        minimum_component_pixels=50_000,
        maximum_aspect_ratio=1.75,
        maximum_mean_instance_pixels=250.0,
        maximum_mean_red_blue_difference=21.0,
        strong_red_blue_difference=40.0,
        maximum_strong_chromatic_fraction=0.10,
        elongation_ratio=2.0,
        maximum_elongated_instance_fraction=0.50,
        maximum_context_nuclear_supported_fraction=0.425,
        minimum_component_fill_fraction=0.20,
        minimum_instance_fraction=0.50,
    )

    assert not np.any(protected)
    assert np.any(legacy)


def test_foam_mosaic_gate_requires_bright_source_and_bounded_nuclear_context() -> None:
    shape = (100, 100)
    prediction_counts = np.full(shape, 16, dtype=np.uint32)
    pixel_counts = np.full(shape, 16, dtype=np.uint32)
    sampled_labels = np.arange(1, 1 + np.prod(shape), dtype=np.uint32).reshape(shape)
    nuclear_supported = np.zeros(sampled_labels.max() + 1, dtype=bool)
    supported_grid = np.zeros(shape, dtype=bool)
    supported_grid[::2, ::2] = True
    nuclear_supported[sampled_labels[supported_grid]] = True
    instance_areas = np.full(sampled_labels.max() + 1, 16, dtype=np.uint64)
    instance_areas[0] = 0
    eligible_instances = np.ones(sampled_labels.max() + 1, dtype=bool)
    eligible_instances[0] = False
    red_blue = np.zeros(shape, dtype=np.int16)

    def classify(intensity: int, nuclear_ceiling: float) -> np.ndarray:
        return _compact_unsupported_prediction_grid(
            prediction_counts,
            pixel_counts,
            sampled_labels,
            red_blue,
            nuclear_supported,
            instance_areas,
            eligible_instances,
            bin_size=4,
            minimum_bin_occupancy_fraction=0.25,
            minimum_component_pixels=50_000,
            maximum_aspect_ratio=1.75,
            maximum_mean_instance_pixels=250.0,
            maximum_mean_red_blue_difference=21.0,
            strong_red_blue_difference=40.0,
            maximum_strong_chromatic_fraction=0.10,
            elongation_ratio=2.0,
            maximum_elongated_instance_fraction=0.45,
            maximum_context_nuclear_supported_fraction=nuclear_ceiling,
            minimum_component_fill_fraction=0.20,
            minimum_instance_fraction=0.50,
            sampled_mean_intensity=np.full(shape, intensity, dtype=np.float32),
            minimum_mean_intensity=160.0,
        )

    # The 25%-supported, pale, round pseudo-cell lattice matches the native
    # luminal foam failure.  Either dark tissue or the stricter general
    # nuclear-context ceiling protects the same topology.
    assert np.any(classify(170, 0.30))
    assert not np.any(classify(150, 0.30))
    assert not np.any(classify(170, 0.20))


def test_foam_mosaic_window_recovers_local_compact_field_from_long_bridge() -> None:
    shape = (160, 320)
    prediction_counts = np.full(shape, 16, dtype=np.uint32)
    pixel_counts = np.full(shape, 16, dtype=np.uint32)
    sampled_labels = np.arange(1, 1 + np.prod(shape), dtype=np.uint32).reshape(shape)
    lookup_size = int(sampled_labels.max()) + 1
    nuclear_supported = np.zeros(lookup_size, dtype=bool)
    instance_areas = np.full(lookup_size, 16, dtype=np.uint64)
    instance_areas[0] = 0
    eligible_instances = np.ones(lookup_size, dtype=bool)
    eligible_instances[0] = False
    common = dict(
        bin_size=4,
        minimum_bin_occupancy_fraction=0.10,
        minimum_component_pixels=50_000,
        maximum_aspect_ratio=1.75,
        maximum_mean_instance_pixels=250.0,
        maximum_mean_red_blue_difference=21.0,
        strong_red_blue_difference=40.0,
        maximum_strong_chromatic_fraction=0.10,
        elongation_ratio=2.0,
        maximum_elongated_instance_fraction=0.45,
        maximum_context_nuclear_supported_fraction=0.30,
        minimum_component_fill_fraction=0.20,
        minimum_instance_fraction=0.50,
        sampled_mean_intensity=np.full(shape, 170, dtype=np.float32),
        minimum_mean_intensity=160.0,
    )

    global_result = _compact_unsupported_prediction_grid(
        prediction_counts,
        pixel_counts,
        sampled_labels,
        np.zeros(shape, dtype=np.int16),
        nuclear_supported,
        instance_areas,
        eligible_instances,
        **common,
    )
    windowed_result = _compact_unsupported_prediction_grid(
        prediction_counts,
        pixel_counts,
        sampled_labels,
        np.zeros(shape, dtype=np.int16),
        nuclear_supported,
        instance_areas,
        eligible_instances,
        **common,
        window_size=640,
        window_overlap=64,
    )

    assert not np.any(global_result)
    assert np.any(windowed_result)


def test_compact_unsupported_gate_preserves_organized_elongated_instances() -> None:
    shape = (100, 100)
    prediction_counts = np.full(shape, 16, dtype=np.uint32)
    pixel_counts = np.full(shape, 16, dtype=np.uint32)
    # Four consecutive sampled centers belong to each short gland-wall cell.
    # The component remains compact overall, but its constituent cells have a
    # coherent elongated shape that is absent from a pseudo-cell debris field.
    sampled_labels = np.empty(shape, dtype=np.uint32)
    next_label = 1
    for row in range(shape[0]):
        for column in range(0, shape[1], 4):
            sampled_labels[row, column : column + 4] = next_label
            next_label += 1
    nuclear_supported = np.zeros(next_label, dtype=bool)
    instance_areas = np.full(next_label, 64, dtype=np.uint64)
    instance_areas[0] = 0
    eligible_instances = np.ones(next_label, dtype=bool)
    eligible_instances[0] = False
    red_blue = np.zeros(shape, dtype=np.int16)

    protected = _compact_unsupported_prediction_grid(
        prediction_counts,
        pixel_counts,
        sampled_labels,
        red_blue,
        nuclear_supported,
        instance_areas,
        eligible_instances,
        bin_size=4,
        minimum_bin_occupancy_fraction=0.25,
        minimum_component_pixels=50_000,
        maximum_aspect_ratio=1.75,
        maximum_mean_instance_pixels=250.0,
        maximum_mean_red_blue_difference=21.0,
        strong_red_blue_difference=40.0,
        maximum_strong_chromatic_fraction=0.10,
        elongation_ratio=2.0,
        maximum_elongated_instance_fraction=0.50,
        maximum_context_nuclear_supported_fraction=0.425,
        minimum_component_fill_fraction=0.20,
        minimum_instance_fraction=0.50,
    )

    assert not np.any(protected)


def test_compact_unsupported_gate_requires_physical_area_of_removable_cells() -> None:
    shape = (96, 96)
    prediction_counts = np.full(shape, 16, dtype=np.uint32)
    pixel_counts = np.full(shape, 16, dtype=np.uint32)
    sampled_labels = np.arange(1, 1 + np.prod(shape), dtype=np.uint32).reshape(shape)
    nuclear_supported = np.zeros(sampled_labels.max() + 1, dtype=bool)
    # The candidate field is large and dense, but every sampled instance has
    # too little overlap to qualify for removal. Its context must not lend a
    # physical-area credential to cells the gate would leave behind.
    instance_areas = np.full(sampled_labels.max() + 1, 64, dtype=np.uint64)
    instance_areas[0] = 0
    eligible_instances = np.zeros(sampled_labels.max() + 1, dtype=bool)
    red_blue = np.zeros(shape, dtype=np.int16)

    artifact = _compact_unsupported_prediction_grid(
        prediction_counts,
        pixel_counts,
        sampled_labels,
        red_blue,
        nuclear_supported,
        instance_areas,
        eligible_instances,
        bin_size=4,
        minimum_bin_occupancy_fraction=0.25,
        minimum_component_pixels=50_000,
        maximum_aspect_ratio=1.75,
        maximum_mean_instance_pixels=250.0,
        maximum_mean_red_blue_difference=21.0,
        strong_red_blue_difference=40.0,
        maximum_strong_chromatic_fraction=0.10,
        elongation_ratio=2.0,
        maximum_elongated_instance_fraction=0.50,
        maximum_context_nuclear_supported_fraction=0.425,
        minimum_component_fill_fraction=0.20,
        minimum_instance_fraction=0.50,
    )

    assert not np.any(artifact)


def test_foam_topology_defers_exact_instance_overlap_to_native_pixel_gate() -> None:
    shape = (96, 96)
    prediction_counts = np.full(shape, 16, dtype=np.uint32)
    pixel_counts = np.full(shape, 16, dtype=np.uint32)
    sampled_labels = np.arange(1, 1 + np.prod(shape), dtype=np.uint32).reshape(shape)
    lookup_size = int(sampled_labels.max()) + 1
    nuclear_supported = np.zeros(lookup_size, dtype=bool)
    # Each center sample represents only 16 aggregate predicted pixels, while
    # the stitched native instance has a larger exact area. The topology stage
    # must not pretend this center assignment is an exact overlap measurement.
    instance_areas = np.full(lookup_size, 200, dtype=np.uint64)
    instance_areas[0] = 0
    eligible_instances = np.ones(lookup_size, dtype=bool)
    eligible_instances[0] = False
    common = dict(
        bin_size=4,
        minimum_bin_occupancy_fraction=0.10,
        minimum_component_pixels=50_000,
        maximum_aspect_ratio=1.75,
        maximum_mean_instance_pixels=250.0,
        maximum_mean_red_blue_difference=21.0,
        strong_red_blue_difference=40.0,
        maximum_strong_chromatic_fraction=0.10,
        elongation_ratio=2.0,
        maximum_elongated_instance_fraction=0.45,
        maximum_context_nuclear_supported_fraction=0.30,
        minimum_component_fill_fraction=0.20,
        minimum_instance_fraction=0.50,
        sampled_mean_intensity=np.full(shape, 170, dtype=np.float32),
        minimum_mean_intensity=160.0,
    )
    guarded = _compact_unsupported_prediction_grid(
        prediction_counts,
        pixel_counts,
        sampled_labels,
        np.zeros(shape, dtype=np.int16),
        nuclear_supported,
        instance_areas,
        eligible_instances,
        **common,
    )
    nominated = _compact_unsupported_prediction_grid(
        prediction_counts,
        pixel_counts,
        sampled_labels,
        np.zeros(shape, dtype=np.int16),
        nuclear_supported,
        instance_areas,
        eligible_instances,
        **common,
        require_removable_instance_area=False,
    )

    assert not np.any(guarded)
    assert np.any(nominated)


def test_foam_supported_bridge_is_bounded_to_unsupported_lattice() -> None:
    shape = (100, 160)
    prediction_counts = np.full(shape, 16, dtype=np.uint32)
    pixel_counts = np.full(shape, 16, dtype=np.uint32)
    sampled_labels = np.arange(1, 1 + np.prod(shape), dtype=np.uint32).reshape(shape)
    lookup_size = int(sampled_labels.max()) + 1
    nuclear_supported = np.zeros(lookup_size, dtype=bool)
    # A thin supported seam crosses a compact foam field, while the supported
    # half-plane represents adjacent viable tissue. Only the enclosed seam may
    # bridge the unsupported component; the tissue must not enlarge it.
    supported_grid = np.zeros(shape, dtype=bool)
    supported_grid[:, 50] = True
    supported_grid[:, 100:] = True
    nuclear_supported[sampled_labels[supported_grid]] = True
    instance_areas = np.full(lookup_size, 16, dtype=np.uint64)
    instance_areas[0] = 0
    eligible_instances = np.ones(lookup_size, dtype=bool)
    eligible_instances[0] = False

    artifact = _compact_unsupported_prediction_grid(
        prediction_counts,
        pixel_counts,
        sampled_labels,
        np.zeros(shape, dtype=np.int16),
        nuclear_supported,
        instance_areas,
        eligible_instances,
        bin_size=4,
        minimum_bin_occupancy_fraction=0.10,
        minimum_component_pixels=50_000,
        maximum_aspect_ratio=1.75,
        maximum_mean_instance_pixels=250.0,
        maximum_mean_red_blue_difference=21.0,
        strong_red_blue_difference=40.0,
        maximum_strong_chromatic_fraction=0.10,
        elongation_ratio=2.0,
        maximum_elongated_instance_fraction=0.50,
        maximum_context_nuclear_supported_fraction=0.50,
        minimum_component_fill_fraction=0.20,
        minimum_instance_fraction=0.50,
        sampled_mean_intensity=np.full(shape, 170, dtype=np.float32),
        minimum_mean_intensity=160.0,
        require_removable_instance_area=False,
        include_nuclear_supported_in_connectivity=True,
    )

    assert np.any(artifact[:, :100])
    assert not np.any(artifact[:, 100:])


def test_tile_cache_accepts_current_or_legacy_fingerprint(tmp_path: Path) -> None:
    path = tmp_path / "tile.npz"
    expected = np.arange(9, dtype=np.int32).reshape(3, 3)
    _write_cached_tile(path, expected, "legacy")

    assert np.array_equal(
        _load_cached_tile(path, ("current", "legacy")),
        expected,
    )
    assert _load_cached_tile(path, ("current",)) is None


def _cell_preflight_for_cache_reuse(
    root: Path,
    *,
    mask: str,
    transform: str,
    source: str = "s" * 64,
) -> CellPreflight:
    slide = CellPreflightSlide(
        section="001",
        slide_name="section.ndpi",
        source_path=str(root / "section.ndpi"),
        source_identity=source,
        mask_path=str(root / "section.mask.png"),
        mask_sha256=mask,
        transform_sha256=transform,
        native_shape=(4, 4),
        content_bbox_xywh=(0, 0, 4, 4),
        thumbnail_shape=(2, 2),
        mpp_xy=(0.5, 0.5),
        is_reference=True,
    )
    core = {
        "schema_version": 2,
        "registration_result_sha256": "r" * 64,
        "registration_approval_sha256": "a" * 64,
        "analysis_manifest_sha256": None,
        "excluded_slides": [],
        "slides": [slide.portable_json_dict()],
    }
    return CellPreflight(
        schema_version=2,
        registration_run=str(root),
        registration_result_sha256="r" * 64,
        registration_approval_sha256="a" * 64,
        analysis_manifest_sha256=None,
        excluded_slides=(),
        slides=(slide,),
        fingerprint=_json_sha256(core),
    )


def test_processing_section_selection_preserves_complete_preflight_order(
    tmp_path: Path,
) -> None:
    first = _cell_preflight_for_cache_reuse(
        tmp_path,
        mask="m" * 64,
        transform="t" * 64,
    ).slides[0]
    second = CellPreflightSlide(
        section="002",
        slide_name="second.ndpi",
        source_path=str(tmp_path / "second.ndpi"),
        source_identity="q" * 64,
        mask_path=str(tmp_path / "second.mask.png"),
        mask_sha256="n" * 64,
        transform_sha256="u" * 64,
        native_shape=(4, 4),
        content_bbox_xywh=(0, 0, 4, 4),
        thumbnail_shape=(2, 2),
        mpp_xy=(0.5, 0.5),
        is_reference=False,
    )
    preflight = CellPreflight(
        schema_version=2,
        registration_run=str(tmp_path),
        registration_result_sha256="r" * 64,
        registration_approval_sha256="a" * 64,
        analysis_manifest_sha256=None,
        excluded_slides=(),
        slides=(first, second),
        fingerprint="p" * 64,
    )

    selected = _select_processing_slides(
        preflight,
        ("second", "001", "second.ndpi"),
    )

    assert tuple(slide.section for slide in selected) == ("001", "002")
    assert _select_processing_slides(preflight, None) == preflight.slides


@pytest.mark.parametrize("selectors", [(), ("",), ("missing",)])
def test_processing_section_selection_rejects_invalid_workers(
    tmp_path: Path,
    selectors: tuple[str, ...],
) -> None:
    preflight = _cell_preflight_for_cache_reuse(
        tmp_path,
        mask="m" * 64,
        transform="t" * 64,
    )

    with pytest.raises(ValueError):
        _select_processing_slides(preflight, selectors)


def test_processing_worker_writes_checkpoint_without_sealing_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preflight = _cell_preflight_for_cache_reuse(
        tmp_path / "registration",
        mask="m" * 64,
        transform="t" * 64,
    )
    output = tmp_path / "run"
    config = CellSegmentationConfig(tmp_path / "registration", output)
    monkeypatch.setattr(
        "histopia.cells._pipeline.preflight_cell_run", lambda _config: preflight
    )
    monkeypatch.setattr(
        "histopia.cells._pipeline.load_cellpose_runtime", lambda _config: object()
    )
    monkeypatch.setattr(
        "histopia.cells._pipeline.runtime_provenance",
        lambda _runtime: {
            "weight_sha256": "w" * 64,
            "device": {"resolved": "cpu"},
        },
    )
    row = {"section": "001", "slide": "section.ndpi", "cell_count": 7}
    monkeypatch.setattr(
        "histopia.cells._pipeline._run_slide", lambda *_args, **_kwargs: row
    )

    checkpoint = run_cell_segmentation(
        config,
        processing_sections=("001",),
    )

    payload = json.loads(checkpoint.read_text())
    assert checkpoint.parent == output / "worker-checkpoints"
    assert payload["complete_result"] is False
    assert payload["requested_sections"] == ["001"]
    assert payload["slides"] == [row]
    assert not (output / "cell_result.json").exists()
    assert not (output / "cell_performance.json").exists()


def test_cache_only_worker_inherits_sealed_model_without_loading_cellpose(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "candidate"
    preflight = _cell_preflight_for_cache_reuse(
        source,
        mask="m" * 64,
        transform="t" * 64,
    )
    write_cell_preflight(preflight, source / "preflight.json")
    (source / "labels").mkdir()
    (source / "qc").mkdir()
    (source / "labels/001.cells.tiff").write_bytes(b"labels")
    (source / "qc/001.json").write_text("{}")
    model = {
        "weight_sha256": "w" * 64,
        "device": {"resolved": "cuda:0", "accelerator_name": "source-gpu"},
    }
    request = _request_payload(CellSegmentationConfig(source, source / "unused"))
    profile = _json_sha256(
        {
            "algorithm_version": 75,
            "preflight_fingerprint": preflight.fingerprint,
            "model": model,
            "request": request,
        }
    )
    write_cell_result(
        source,
        {
            "schema_version": 1,
            "algorithm_version": 75,
            "preflight": "preflight.json",
            "preflight_fingerprint": preflight.fingerprint,
            "model": model,
            "request": request,
            "profile_fingerprint": profile,
            "slides": [
                {
                    "section": "001",
                    "labels": "labels/001.cells.tiff",
                    "qc": "qc/001.json",
                }
            ],
        },
    )
    config = CellSegmentationConfig(
        source,
        output,
        tile_cache_reuse_run=source,
        require_complete_tile_cache_reuse=True,
        isolated_debris_self_dense_glass_gate=True,
    )
    monkeypatch.setattr(
        "histopia.cells._pipeline.preflight_cell_run", lambda _config: preflight
    )

    def fail_runtime(_config: CellSegmentationConfig) -> object:
        raise AssertionError("cache-only worker loaded Cellpose")

    monkeypatch.setattr("histopia.cells._pipeline.load_cellpose_runtime", fail_runtime)
    row = {"section": "001", "slide": "section.ndpi", "cell_count": 7}

    def capture(*args: object, **_kwargs: object) -> dict[str, object]:
        assert args[3] is None
        return row

    monkeypatch.setattr("histopia.cells._pipeline._run_slide", capture)
    checkpoint = run_cell_segmentation(config, processing_sections=("001",))
    payload = json.loads(checkpoint.read_text())

    assert payload["slides"] == [row]


def test_dense_recovery_worker_is_cache_only_and_path_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "candidate"
    preflight = _cell_preflight_for_cache_reuse(
        tmp_path / "registration",
        mask="m" * 64,
        transform="t" * 64,
    )
    model = {
        "weight_sha256": "w" * 64,
        "device": {"resolved": "cuda:0", "accelerator_name": "source-gpu"},
    }
    source_result = {
        "algorithm_version": 75,
        "preflight_fingerprint": preflight.fingerprint,
        "model": model,
    }
    recovery_payload = {
        "schema_version": 1,
        "method": DENSE_SMALL_CELL_RECOVERY_METHOD,
        "manifest_fingerprint": "f" * 64,
        "source_section": "001",
        "tiles": [{"tile": "r0000_c0000_y0_x0.npz"}],
    }
    recovery = SimpleNamespace(
        fingerprint="f" * 64,
        method=DENSE_SMALL_CELL_RECOVERY_METHOD,
        tiles={},
        request_payload=lambda: recovery_payload,
    )
    config = CellSegmentationConfig(
        tmp_path / "registration",
        output,
        tile_cache_reuse_run=source,
        sections=("001",),
        require_complete_tile_cache_reuse=True,
        dense_small_cell_recovery_manifest=tmp_path / "recovery" / "manifest.json",
    )
    monkeypatch.setattr(
        "histopia.cells._pipeline.preflight_cell_run", lambda _config: preflight
    )
    monkeypatch.setattr(
        "histopia.cells._pipeline.validate_cell_result_index",
        lambda _source: source_result,
    )
    monkeypatch.setattr(
        "histopia.cells._pipeline.load_dense_small_cell_recovery_manifest",
        lambda *_args, **_kwargs: recovery,
    )
    monkeypatch.setattr(
        "histopia.cells._pipeline.dense_small_cell_recovery_manifest_profile",
        lambda _path: (
            94,
            "combined-containment-cpsam-stardist-dense-recovery-v1",
        ),
    )
    monkeypatch.setattr(
        "histopia.cells._pipeline._prepare_tile_cache_reuse",
        lambda *_args, **_kwargs: {"section.ndpi": object()},
    )

    def fail_runtime(_config: CellSegmentationConfig) -> object:
        raise AssertionError("dense recovery loaded Cellpose")

    monkeypatch.setattr("histopia.cells._pipeline.load_cellpose_runtime", fail_runtime)
    captured: dict[str, object] = {}

    def capture(*args: object, **kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        assert args[3] is None
        return {"section": "001", "slide": "section.ndpi", "cell_count": 7}

    monkeypatch.setattr("histopia.cells._pipeline._run_slide", capture)
    run_cell_segmentation(config, processing_sections=("001",))

    assert captured["dense_recovery"] is recovery
    request = _request_payload(
        config,
        method_profile="combined-containment-cpsam-stardist-dense-recovery-v1",
        dense_recovery_payload=recovery_payload,
    )
    assert request["dense_small_cell_recovery"] == recovery_payload
    assert str(tmp_path) not in json.dumps(request)


def test_dense_recovery_reconstructs_exact_v75_tile_profile() -> None:
    config = CellSegmentationConfig(
        Path("registration"),
        Path("candidate"),
        tile_cache_reuse_run=Path("source"),
        sections=("014",),
        require_complete_tile_cache_reuse=True,
        dense_small_cell_recovery_manifest=Path("recovery/manifest.json"),
    )
    recovery_payload = {
        "schema_version": 1,
        "method": DENSE_SMALL_CELL_RECOVERY_METHOD,
        "manifest_fingerprint": "f" * 64,
        "source_section": "014",
        "tiles": [{"tile": "r0011_c0040_y9856_x35840.npz"}],
    }
    recovered = _request_payload(
        config,
        method_profile="combined-containment-cpsam-stardist-dense-recovery-v1",
        dense_recovery_payload=recovery_payload,
    )
    baseline = _request_payload(
        CellSegmentationConfig(
            Path("registration"),
            Path("source"),
            sections=("014",),
        )
    )

    assert _version_75_postfilter_request(recovered) == baseline


def test_dense_recovery_replaces_only_exact_target_tile() -> None:
    tile = CellTile(0, 0, 0, 4, 0, 4)
    source = np.ones((4, 4), dtype=np.int32)
    replacement = np.full((4, 4), 7, dtype=np.int32)
    calls = 0

    def load_mask() -> np.ndarray:
        nonlocal calls
        calls += 1
        return replacement

    recovery = SimpleNamespace(
        tiles={f"{tile.key}.npz": SimpleNamespace(load_mask=load_mask)}
    )

    observed, replaced = _replace_reusable_tile_with_dense_recovery(
        recovery, tile, source
    )
    untouched, untouched_replaced = _replace_reusable_tile_with_dense_recovery(
        SimpleNamespace(tiles={}), tile, source
    )

    np.testing.assert_array_equal(observed, replacement)
    assert replaced is True
    assert untouched is source
    assert untouched_replaced is False
    assert calls == 1


def test_dense_recovery_rejects_changed_native_tile_geometry() -> None:
    tile = CellTile(0, 0, 0, 4, 0, 4)
    recovery = SimpleNamespace(
        tiles={
            f"{tile.key}.npz": SimpleNamespace(
                load_mask=lambda: np.ones((3, 4), dtype=np.int32)
            )
        }
    )

    with pytest.raises(ValueError, match="geometry changed"):
        _replace_reusable_tile_with_dense_recovery(
            recovery, tile, np.ones((4, 4), dtype=np.int32)
        )


def test_cross_registration_tile_reuse_is_exact_and_mask_independent(
    tmp_path: Path,
) -> None:
    old_root = tmp_path / "old"
    output = tmp_path / "new"
    old = _cell_preflight_for_cache_reuse(
        old_root,
        mask="m" * 64,
        transform="t" * 64,
    )
    current = _cell_preflight_for_cache_reuse(
        tmp_path / "registration",
        mask="n" * 64,
        transform="u" * 64,
    )
    write_cell_preflight(old, old_root / "preflight.json")
    config = CellSegmentationConfig(
        tmp_path / "registration",
        output,
        tile_cache_reuse_run=old_root,
    )
    model = {"weight_sha256": "w" * 64, "device": {"resolved": "cuda:0"}}
    request = _request_payload(config)
    reusable = _prepare_tile_cache_reuse(
        config,
        current,
        model=model,
        request=request,
    )["section.ndpi"]
    tile = CellTile(0, 0, 0, 4, 0, 4)
    expected = np.arange(16, dtype=np.int32).reshape(4, 4)
    fingerprint = _tile_fingerprint(
        old,
        old.slides[0],
        reusable.profile_fingerprints[0],
        tile,
    )
    _write_cached_tile(
        reusable.cache_root / f"{tile.key}.npz",
        expected,
        fingerprint,
    )

    np.testing.assert_array_equal(_load_reusable_tile(reusable, tile), expected)
    _write_cached_tile(
        reusable.cache_root / f"{tile.key}.npz",
        expected,
        "wrong-profile",
    )
    assert _load_reusable_tile(reusable, tile) is None


def test_cross_registration_tile_reuse_rejects_changed_source_identity(
    tmp_path: Path,
) -> None:
    old_root = tmp_path / "old"
    old = _cell_preflight_for_cache_reuse(
        old_root,
        mask="m" * 64,
        transform="t" * 64,
    )
    changed = _cell_preflight_for_cache_reuse(
        tmp_path / "registration",
        mask="n" * 64,
        transform="u" * 64,
        source="x" * 64,
    )
    write_cell_preflight(old, old_root / "preflight.json")
    config = CellSegmentationConfig(
        tmp_path / "registration",
        tmp_path / "new",
        tile_cache_reuse_run=old_root,
    )

    with pytest.raises(ValueError, match="geometry or identity changed"):
        _prepare_tile_cache_reuse(
            config,
            changed,
            model={"weight_sha256": "w" * 64},
            request=_request_payload(config),
        )


def test_cross_run_foam_probe_reuses_exact_baseline_tile_profile(
    tmp_path: Path,
) -> None:
    old_root = tmp_path / "old"
    old = _cell_preflight_for_cache_reuse(
        old_root,
        mask="m" * 64,
        transform="t" * 64,
    )
    current = _cell_preflight_for_cache_reuse(
        tmp_path / "registration",
        mask="n" * 64,
        transform="u" * 64,
    )
    write_cell_preflight(old, old_root / "preflight.json")
    config = CellSegmentationConfig(
        tmp_path / "registration",
        tmp_path / "new",
        tile_cache_reuse_run=old_root,
        isolated_debris_foam_core_minimum_intensity=160.0,
    )
    model = {"weight_sha256": "w" * 64, "device": {"resolved": "cuda:0"}}
    request = _request_payload(config)
    reusable = _prepare_tile_cache_reuse(
        config,
        current,
        model=model,
        request=request,
    )["section.ndpi"]
    assert len(reusable.profile_fingerprints) == 2

    tile = CellTile(0, 0, 0, 4, 0, 4)
    expected = np.arange(16, dtype=np.int32).reshape(4, 4)
    baseline_profile = reusable.profile_fingerprints[1]
    _write_cached_tile(
        reusable.cache_root / f"{tile.key}.npz",
        expected,
        _tile_fingerprint(old, old.slides[0], baseline_profile, tile),
    )

    np.testing.assert_array_equal(_load_reusable_tile(reusable, tile), expected)


def test_cross_run_subset_reuses_profile_sealed_by_compatible_source_result(
    tmp_path: Path,
) -> None:
    old_root = tmp_path / "old"
    old = _cell_preflight_for_cache_reuse(
        old_root,
        mask="m" * 64,
        transform="t" * 64,
    )
    current = _cell_preflight_for_cache_reuse(
        tmp_path / "registration",
        mask="n" * 64,
        transform="u" * 64,
    )
    write_cell_preflight(old, old_root / "preflight.json")
    model = {"weight_sha256": "w" * 64, "device": {"resolved": "cuda:0"}}
    source_config = CellSegmentationConfig(old_root, old_root / "unused")
    source_request = _request_payload(source_config)
    source_profile = _json_sha256(
        {
            "algorithm_version": 75,
            "preflight_fingerprint": old.fingerprint,
            "model": model,
            "request": source_request,
        }
    )
    (old_root / "labels").mkdir()
    (old_root / "qc").mkdir()
    (old_root / "labels/001.cells.tiff").write_bytes(b"labels")
    (old_root / "qc/001.json").write_text("{}")
    write_cell_result(
        old_root,
        {
            "schema_version": 1,
            "algorithm_version": 75,
            "coordinate_space": "native_content_bbox",
            "preflight": "preflight.json",
            "preflight_fingerprint": old.fingerprint,
            "registration_result_sha256": old.registration_result_sha256,
            "registration_approval_sha256": old.registration_approval_sha256,
            "model": model,
            "request": source_request,
            "profile_fingerprint": source_profile,
            "slides": [
                {
                    "section": "001",
                    "labels": "labels/001.cells.tiff",
                    "qc": "qc/001.json",
                }
            ],
        },
    )
    config = CellSegmentationConfig(
        tmp_path / "registration",
        tmp_path / "new",
        tile_cache_reuse_run=old_root,
        sections=("001",),
        isolated_debris_foam_core_minimum_intensity=160.0,
    )
    request = _request_payload(config)
    assert source_request["sections"] == []
    assert request["sections"] == ["001"]
    assert _raw_tile_inference_request(source_request) == _raw_tile_inference_request(
        request
    )

    reusable = _prepare_tile_cache_reuse(
        config,
        current,
        model=model,
        request=request,
    )["section.ndpi"]
    assert source_profile in reusable.profile_fingerprints

    tile = CellTile(0, 0, 0, 4, 0, 4)
    expected = np.arange(16, dtype=np.int32).reshape(4, 4)
    _write_cached_tile(
        reusable.cache_root / f"{tile.key}.npz",
        expected,
        _tile_fingerprint(old, old.slides[0], source_profile, tile),
    )
    np.testing.assert_array_equal(_load_reusable_tile(reusable, tile), expected)


def test_v76_postfilter_reuses_exact_sealed_v75_raw_tiles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_root = tmp_path / "old"
    old = _cell_preflight_for_cache_reuse(
        old_root,
        mask="m" * 64,
        transform="t" * 64,
    )
    current = _cell_preflight_for_cache_reuse(
        tmp_path / "registration",
        mask="n" * 64,
        transform="u" * 64,
    )
    write_cell_preflight(old, old_root / "preflight.json")
    model = {"weight_sha256": "w" * 64, "device": {"resolved": "cuda:0"}}
    source_request = _request_payload(CellSegmentationConfig(old_root, tmp_path / "x"))
    source_profile = _json_sha256(
        {
            "algorithm_version": 75,
            "preflight_fingerprint": old.fingerprint,
            "model": model,
            "request": source_request,
        }
    )
    (old_root / "labels").mkdir()
    (old_root / "qc").mkdir()
    (old_root / "labels/001.cells.tiff").write_bytes(b"labels")
    (old_root / "qc/001.json").write_text("{}")
    write_cell_result(
        old_root,
        {
            "schema_version": 1,
            "algorithm_version": 75,
            "preflight": "preflight.json",
            "preflight_fingerprint": old.fingerprint,
            "model": model,
            "request": source_request,
            "profile_fingerprint": source_profile,
            "slides": [
                {
                    "section": "001",
                    "labels": "labels/001.cells.tiff",
                    "qc": "qc/001.json",
                }
            ],
        },
    )
    monkeypatch.setattr("histopia.cells._pipeline._ALGORITHM_VERSION", 76)
    config = CellSegmentationConfig(
        tmp_path / "registration",
        tmp_path / "new",
        tile_cache_reuse_run=old_root,
        isolated_debris_self_dense_glass_gate=True,
        isolated_debris_oversized_brown_gate=True,
        reconcile_enclosed_cytoplasmic_children=True,
        isolated_debris_protect_organized_from_necrotic=True,
    )
    legacy_profile = _json_sha256({"algorithm_version": 67, "profile": "legacy"})
    reusable = _prepare_tile_cache_reuse(
        config,
        current,
        model=model,
        request=_request_payload(config),
        compatible_profile_fingerprints=(legacy_profile,),
    )["section.ndpi"]
    assert source_profile in reusable.profile_fingerprints
    assert legacy_profile in reusable.profile_fingerprints

    tile = CellTile(0, 0, 0, 4, 0, 4)
    expected = np.arange(16, dtype=np.int32).reshape(4, 4)
    _write_cached_tile(
        reusable.cache_root / f"{tile.key}.npz",
        expected,
        _tile_fingerprint(old, old.slides[0], legacy_profile, tile),
    )
    np.testing.assert_array_equal(_load_reusable_tile(reusable, tile), expected)


def test_cross_run_sealed_profile_rejects_changed_inference_controls(
    tmp_path: Path,
) -> None:
    old_root = tmp_path / "old"
    old = _cell_preflight_for_cache_reuse(
        old_root,
        mask="m" * 64,
        transform="t" * 64,
    )
    current = _cell_preflight_for_cache_reuse(
        tmp_path / "registration",
        mask="n" * 64,
        transform="u" * 64,
    )
    write_cell_preflight(old, old_root / "preflight.json")
    model = {"weight_sha256": "w" * 64}
    source_request = _request_payload(CellSegmentationConfig(old_root, tmp_path / "x"))
    source_profile = _json_sha256(
        {
            "algorithm_version": 75,
            "preflight_fingerprint": old.fingerprint,
            "model": model,
            "request": source_request,
        }
    )
    (old_root / "labels").mkdir()
    (old_root / "qc").mkdir()
    (old_root / "labels/001.cells.tiff").write_bytes(b"labels")
    (old_root / "qc/001.json").write_text("{}")
    write_cell_result(
        old_root,
        {
            "schema_version": 1,
            "algorithm_version": 75,
            "preflight": "preflight.json",
            "preflight_fingerprint": old.fingerprint,
            "model": model,
            "request": source_request,
            "profile_fingerprint": source_profile,
            "slides": [
                {
                    "section": "001",
                    "labels": "labels/001.cells.tiff",
                    "qc": "qc/001.json",
                }
            ],
        },
    )
    changed = CellSegmentationConfig(
        tmp_path / "registration",
        tmp_path / "new",
        tile_cache_reuse_run=old_root,
        first_cellprob=-1.5,
    )

    with pytest.raises(ValueError, match="inference controls changed"):
        _prepare_tile_cache_reuse(
            changed,
            current,
            model=model,
            request=_request_payload(changed),
        )


def test_persisted_cell_preflight_rejects_tampering(tmp_path: Path) -> None:
    preflight = _cell_preflight_for_cache_reuse(
        tmp_path,
        mask="m" * 64,
        transform="t" * 64,
    )
    path = write_cell_preflight(preflight, tmp_path / "preflight.json")

    assert load_cell_preflight(path) == preflight
    payload = json.loads(path.read_text())
    payload["slides"][0]["mask_sha256"] = "changed"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="fingerprint is stale"):
        load_cell_preflight(path)


def test_dark_fold_profile_reuses_only_exact_baseline_raw_tile_profile() -> None:
    config = CellSegmentationConfig(Path("registration"), Path("output"))
    request = _request_payload(config)
    model = {"name": "cpsam", "device": "cuda:0"}

    assert not _baseline_fold_tile_profile_fingerprints(
        preflight_fingerprint="p" * 64,
        model=model,
        request=request,
    )

    tuned = json.loads(json.dumps(request))
    tuned["isolated_debris_gate"]["fold_maximum_mean_red_blue_difference"] = 85.0
    tuned["isolated_debris_gate"]["fold_maximum_mean_intensity"] = 130.0
    fingerprints = _baseline_fold_tile_profile_fingerprints(
        preflight_fingerprint="p" * 64,
        model=model,
        request=tuned,
    )

    assert len(fingerprints) == 1
    assert fingerprints[0] == _json_sha256(
        {
            "algorithm_version": 75,
            "preflight_fingerprint": "p" * 64,
            "model": model,
            "request": request,
        }
    )


def test_validated_postfilter_bundle_is_explicit_and_v75_default_compatible() -> None:
    baseline = _request_payload(
        CellSegmentationConfig(Path("registration"), Path("output"))
    )
    baseline_gate = baseline["isolated_debris_gate"]
    assert "satellite_gate" not in baseline_gate
    assert "self_dense_glass_gate" not in baseline_gate
    assert "oversized_brown_gate" not in baseline_gate
    assert "clustered_brown_gate" not in baseline_gate
    assert "diffuse_degenerated_gate" not in baseline_gate
    assert "neutral_precipitate_gate" not in baseline_gate
    assert "reconcile_enclosed_cytoplasmic_children" not in baseline_gate
    assert "protect_organized_from_necrotic" not in baseline_gate
    assert (
        "organized_necrotic_protection_maximum_instance_area_um2" not in baseline_gate
    )
    assert (
        "organized_necrotic_protection_minimum_local_prediction_fraction"
        not in baseline_gate
    )
    assert "organized_necrotic_protection_local_prediction_source" not in baseline_gate

    tuned = _request_payload(
        CellSegmentationConfig(
            Path("registration"),
            Path("output"),
            isolated_debris_self_dense_glass_gate=True,
            isolated_debris_oversized_brown_gate=True,
            isolated_debris_oversized_brown_maximum_mean_intensity=180.0,
            isolated_debris_clustered_brown_gate=True,
            isolated_debris_diffuse_degenerated_gate=True,
            isolated_debris_neutral_precipitate_gate=True,
            reconcile_enclosed_cytoplasmic_children=True,
            isolated_debris_protect_organized_from_necrotic=True,
            isolated_debris_organized_necrotic_protection_maximum_instance_area_um2=(
                500.0
            ),
            isolated_debris_organized_necrotic_protection_minimum_local_prediction_fraction=(
                0.025
            ),
            isolated_debris_organized_necrotic_protection_use_eligible_context=True,
            isolated_debris_organized_necrotic_protection_require_independent_nuclear_context=(
                True
            ),
            isolated_debris_organized_necrotic_sparse_glass_shape_gate=True,
        )
    )
    gate = tuned["isolated_debris_gate"]
    assert gate["self_dense_glass_gate"] is True
    assert gate["oversized_brown_gate"] is True
    assert gate["oversized_brown_maximum_mean_intensity"] == 180.0
    assert gate["clustered_brown_gate"] is True
    assert gate["diffuse_degenerated_gate"] is True
    assert gate["neutral_precipitate_gate"] is True
    assert gate["reconcile_enclosed_cytoplasmic_children"] is True
    assert gate["protect_organized_from_necrotic"] is True
    assert gate["organized_necrotic_protection_maximum_instance_area_um2"] == 500.0
    assert (
        gate["organized_necrotic_protection_minimum_local_prediction_fraction"] == 0.025
    )
    assert (
        gate["organized_necrotic_protection_local_prediction_source"]
        == "preartifact_independent_nuclear_external"
    )
    sparse_shape = gate["organized_necrotic_sparse_glass_shape_gate"]
    assert sparse_shape["context_window_size_px"] == 128
    assert sparse_shape["minimum_tissue_fraction"] == 0.14
    assert _version_75_postfilter_request(tuned) == baseline
    future = json.loads(json.dumps(tuned))
    future["method_reference"]["profile"] = "combined-containment-cpsam-wsi-v72"
    assert _version_75_postfilter_request(future) == baseline
    assert _raw_tile_inference_request(future) == _raw_tile_inference_request(baseline)

    with pytest.raises(ValueError, match="refinements require isolated_debris_gate"):
        CellSegmentationConfig(
            Path("registration"),
            Path("output"),
            isolated_debris_gate=False,
            isolated_debris_self_dense_glass_gate=True,
        )

    with pytest.raises(ValueError, match="maximum instance area must be positive"):
        CellSegmentationConfig(
            Path("registration"),
            Path("output"),
            isolated_debris_organized_necrotic_protection_maximum_instance_area_um2=(
                0.0
            ),
        )

    with pytest.raises(ValueError, match="minimum local prediction fraction"):
        CellSegmentationConfig(
            Path("registration"),
            Path("output"),
            isolated_debris_organized_necrotic_protection_minimum_local_prediction_fraction=(
                1.1
            ),
        )

    with pytest.raises(ValueError, match="eligible organized-necrotic context"):
        CellSegmentationConfig(
            Path("registration"),
            Path("output"),
            isolated_debris_protect_organized_from_necrotic=True,
            isolated_debris_organized_necrotic_protection_use_eligible_context=True,
        )

    with pytest.raises(ValueError, match="requires organized-necrotic protection"):
        CellSegmentationConfig(
            Path("registration"),
            Path("output"),
            isolated_debris_organized_necrotic_protection_minimum_local_prediction_fraction=(
                0.025
            ),
            isolated_debris_organized_necrotic_protection_use_eligible_context=True,
        )

    with pytest.raises(ValueError, match="requires eligible context"):
        CellSegmentationConfig(
            Path("registration"),
            Path("output"),
            isolated_debris_gate=True,
            isolated_debris_protect_organized_from_necrotic=True,
            isolated_debris_organized_necrotic_protection_require_independent_nuclear_context=(
                True
            ),
        )

    with pytest.raises(ValueError, match="shape gate requires"):
        CellSegmentationConfig(
            Path("registration"),
            Path("output"),
            isolated_debris_gate=True,
            isolated_debris_organized_necrotic_sparse_glass_shape_gate=True,
        )


def test_satellite_gate_exemption_is_single_section_cache_only_v89() -> None:
    source = Path("sealed-v75")
    config = CellSegmentationConfig(
        Path("registration"),
        Path("output"),
        tile_cache_reuse_run=source,
        sections=("004",),
        require_complete_tile_cache_reuse=True,
        isolated_debris_satellite_minimum_mask_distance_um=200.0,
    )

    assert _active_method_profile(config) == (
        89,
        "combined-containment-cpsam-wsi-v84",
    )
    request = _request_payload(
        config,
        method_profile="combined-containment-cpsam-wsi-v84",
    )
    assert request["isolated_debris_gate"]["satellite_gate"] == {
        "enabled": True,
        "section": "004",
        "scope": "post-inference-section-satellite-interior-protection-v1",
        "interior_protection": {
            "minimum_mask_distance_um": 200.0,
            "minimum_sample_fraction": 0.50,
        },
    }
    baseline = _request_payload(
        CellSegmentationConfig(
            Path("registration"),
            Path("baseline"),
            sections=("004",),
        )
    )
    assert _version_75_postfilter_request(request) == baseline


@pytest.mark.parametrize(
    "kwargs",
    (
        {},
        {"sections": ("004", "005")},
        {"sections": ("004",), "isolated_debris_self_dense_glass_gate": True},
    ),
)
def test_satellite_gate_exemption_rejects_unbounded_policies(
    kwargs: dict[str, object],
) -> None:
    config = CellSegmentationConfig(
        Path("registration"),
        Path("output"),
        tile_cache_reuse_run=Path("sealed-v75"),
        require_complete_tile_cache_reuse=True,
        isolated_debris_satellite_minimum_mask_distance_um=200.0,
        **kwargs,
    )

    with pytest.raises(ValueError, match="one explicit section"):
        _active_method_profile(config)


def test_satellite_gate_cannot_be_disabled_or_arbitrarily_tuned() -> None:
    with pytest.raises(ValueError, match="cannot be disabled"):
        CellSegmentationConfig(
            Path("registration"),
            Path("output"),
            isolated_debris_satellite_gate=False,
        )

    config = CellSegmentationConfig(
        Path("registration"),
        Path("output"),
        tile_cache_reuse_run=Path("sealed-v75"),
        sections=("004",),
        require_complete_tile_cache_reuse=True,
        isolated_debris_satellite_minimum_mask_distance_um=199.0,
    )
    with pytest.raises(ValueError, match="validated 200-um"):
        _active_method_profile(config)


def test_satellite_interior_protection_uses_physical_mask_distance() -> None:
    sampled_labels = np.zeros((10, 10), dtype=np.uint32)
    sampled_labels[4:6, 4:6] = 1
    sampled_labels[1, 4:6] = 2
    tissue_mask = np.zeros((10, 10), dtype=bool)
    tissue_mask[1:9, 1:9] = True

    protected = _interior_tissue_instance_protection(
        sampled_labels,
        tissue_mask,
        3,
        content_shape=(40, 40),
        bin_size=4,
        mpp_xy=(1.0, 1.0),
        minimum_mask_distance_um=8.0,
    )

    np.testing.assert_array_equal(protected, [False, True, False])


def test_organized_necrotic_protection_excludes_oversized_pseudo_cells() -> None:
    organized = np.array([True, True, True, False], dtype=bool)
    areas = np.array([0, 1_819, 3_299, 538], dtype=np.uint64)

    unbounded = _bounded_organized_necrotic_protection(
        organized,
        areas,
        maximum_instance_area=None,
    )
    bounded = _bounded_organized_necrotic_protection(
        organized,
        areas,
        maximum_instance_area=2_357,
    )

    np.testing.assert_array_equal(unbounded, [False, True, True, False])
    np.testing.assert_array_equal(bounded, [False, True, False, False])


def test_organized_necrotic_protection_excludes_sparse_glass_fragments() -> None:
    labels = np.zeros((4, 4), dtype=np.uint32)
    labels[:, :2] = 1
    labels[:, 2:] = 2
    organized = np.array([False, True, True], dtype=bool)
    areas = np.array([0, 8, 8], dtype=np.uint64)
    prediction_counts = np.array([[8, 0], [8, 0]], dtype=np.uint32)
    pixel_counts = np.full((2, 2), 16, dtype=np.uint32)

    protected = _locally_supported_organized_necrotic_protection(
        labels,
        organized,
        areas,
        prediction_counts,
        pixel_counts,
        maximum=2,
        bin_size=2,
        window_size=2,
        minimum_local_prediction_fraction=0.025,
        minimum_instance_fraction=0.01,
        block_size=4,
    )

    np.testing.assert_array_equal(protected, [False, True, False])


def test_organized_necrotic_protection_cannot_validate_itself() -> None:
    isolated_labels = np.zeros((8, 8), dtype=np.uint32)
    isolated_labels[3:5, 2:4] = 1
    isolated = np.array([False, True], dtype=bool)
    isolated_areas = np.array([0, 4], dtype=np.uint64)
    isolated_prediction_counts = (isolated_labels > 0).astype(np.uint32)
    pixel_counts = np.ones((8, 8), dtype=np.uint32)

    isolated_protected = _locally_supported_organized_necrotic_protection(
        isolated_labels,
        isolated,
        isolated_areas,
        isolated_prediction_counts,
        pixel_counts,
        maximum=1,
        bin_size=1,
        window_size=3,
        minimum_local_prediction_fraction=0.025,
        minimum_instance_fraction=0.01,
        exclude_instance_from_local_prediction=True,
        block_size=8,
    )
    np.testing.assert_array_equal(isolated_protected, [False, False])

    neighboring_labels = isolated_labels.copy()
    neighboring_labels[3:5, 4:6] = 2
    neighboring = np.array([False, True, True], dtype=bool)
    neighboring_areas = np.array([0, 4, 4], dtype=np.uint64)
    neighboring_prediction_counts = (neighboring_labels > 0).astype(np.uint32)
    neighboring_protected = _locally_supported_organized_necrotic_protection(
        neighboring_labels,
        neighboring,
        neighboring_areas,
        neighboring_prediction_counts,
        pixel_counts,
        maximum=2,
        bin_size=1,
        window_size=3,
        minimum_local_prediction_fraction=0.025,
        minimum_instance_fraction=0.01,
        exclude_instance_from_local_prediction=True,
        block_size=8,
    )
    np.testing.assert_array_equal(neighboring_protected, [False, True, True])

    independently_supported_areas = np.array([0, 0, 4], dtype=np.uint64)
    independently_supported = _locally_supported_organized_necrotic_protection(
        neighboring_labels,
        neighboring,
        neighboring_areas,
        (neighboring_labels == 2).astype(np.uint32),
        pixel_counts,
        maximum=2,
        bin_size=1,
        window_size=3,
        minimum_local_prediction_fraction=0.025,
        minimum_instance_fraction=0.01,
        exclude_instance_from_local_prediction=True,
        context_instance_areas=independently_supported_areas,
        block_size=8,
    )
    np.testing.assert_array_equal(independently_supported, [False, True, False])


def test_sparse_glass_shape_gate_is_conjunctive() -> None:
    labels = np.zeros((64, 64), dtype=np.uint32)
    for offset in range(18):
        labels[8 + offset, 4 + offset] = 1
        labels[8 + offset, 40 + offset] = 3
    labels[35:41, 8:14] = 2
    intensity = np.full_like(labels, 245, dtype=np.uint8)
    intensity[:32, 32:] = 150
    organized = np.array([False, True, True, True], dtype=bool)

    artifact = _sparse_glass_organized_necrotic_instances(
        labels,
        intensity,
        organized,
        bin_size=4,
        context_window_size=128,
        maximum_mean_intensity=220.0,
        minimum_tissue_fraction=0.14,
        maximum_sampled_fill_fraction=0.40,
        minimum_sampled_elongation=2.50,
    )

    np.testing.assert_array_equal(artifact, [False, True, False, False])


def test_foam_probe_reuses_only_exact_baseline_raw_tile_profile() -> None:
    config = CellSegmentationConfig(Path("registration"), Path("output"))
    request = _request_payload(config)
    model = {"name": "cpsam", "device": "cuda:0"}

    assert not _baseline_foam_tile_profile_fingerprints(
        preflight_fingerprint="p" * 64,
        model=model,
        request=request,
    )

    tuned = json.loads(json.dumps(request))
    tuned["isolated_debris_gate"]["foam_core_minimum_pixels"] = 4
    tuned["isolated_debris_gate"]["foam_core_minimum_intensity"] = 170.0
    fingerprints = _baseline_foam_tile_profile_fingerprints(
        preflight_fingerprint="p" * 64,
        model=model,
        request=tuned,
    )

    assert len(fingerprints) == 1
    assert fingerprints[0] == _json_sha256(
        {
            "algorithm_version": 75,
            "preflight_fingerprint": "p" * 64,
            "model": model,
            "request": request,
        }
    )


def test_completed_section_checkpoint_requires_current_digest_and_geometry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "run"
    labels = output / "labels" / "001.cells.tiff"
    qc_path = output / "qc" / "001.json"
    labels.parent.mkdir(parents=True)
    qc_path.parent.mkdir()
    labels.write_bytes(b"sealed labels")
    preflight = type("Preflight", (), {"fingerprint": "p" * 64})()
    slide = type(
        "Slide",
        (),
        {
            "section": "001",
            "slide_name": "slide.ndpi",
            "source_identity": "s" * 64,
            "mask_sha256": "m" * 64,
            "is_reference": True,
            "mpp_xy": (0.5, 0.5),
            "native_shape": (3, 4),
            "content_bbox_xywh": (0, 0, 4, 3),
            "transform_sha256": "t" * 64,
        },
    )()
    profile = "r" * 64
    qc_path.write_text(
        json.dumps(
            {
                "checkpoint_schema_version": 1,
                "section_fingerprint": _section_fingerprint(preflight, slide, profile),
                "labels_sha256": _file_sha256(labels),
                "cell_count": 7,
                "outside_tissue_pixels_final": 0,
                "tiles_total": 5,
                "tiles_inferred": 2,
                "tiles_reused": 1,
                "tiles_skipped_outside_tissue": 2,
            }
        )
    )

    image = type(
        "Image",
        (),
        {"width": 4, "height": 3, "bands": 1, "format": "uint"},
    )()
    image_type = type(
        "ImageType",
        (),
        {"new_from_file": staticmethod(lambda *_args, **_kwargs: image)},
    )
    monkeypatch.setattr(
        "histopia.cells._pipeline._import_pyvips",
        lambda: type("Vips", (), {"Image": image_type})(),
    )
    config = CellSegmentationConfig(tmp_path, output)

    resumed = _load_completed_slide(config, preflight, slide, profile)

    assert resumed is not None
    assert resumed["section"] == "001"
    assert resumed["cell_count"] == 7
    image.width = 5
    assert _load_completed_slide(config, preflight, slide, profile) is None
    image.width = 4
    labels.write_bytes(b"changed labels")
    assert _load_completed_slide(config, preflight, slide, profile) is None


def test_run_slide_returns_checkpoint_before_opening_wsi(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = {"section": "001", "cell_count": 7}
    monkeypatch.setattr(
        "histopia.cells._pipeline._load_completed_slide",
        lambda *_args, **_kwargs: expected,
    )
    monkeypatch.setattr(
        "histopia.cells._pipeline._import_pyvips",
        lambda: pytest.fail("a sealed checkpoint must not reopen the WSI"),
    )
    progress: list[str] = []

    result = _run_slide(
        CellSegmentationConfig(tmp_path, tmp_path / "run"),
        object(),
        object(),
        object(),
        ("profile",),
        filter_upgrade_profile_fingerprints=(),
        progress=progress.append,
    )

    assert result is expected
    assert progress == ["  reusing sealed completed section"]


def test_qupath_features_use_native_pixel_offsets() -> None:
    labels = np.zeros((10, 12), dtype=np.int32)
    labels[2:7, 3:9] = 1

    features = _cell_features(
        labels,
        np.array([1]),
        offset_xy=(100, 200),
        section="001",
        simplify_tolerance_px=0.5,
    )

    assert len(features) == 1
    feature = features[0]
    assert feature["id"] == "histopia-cell-001-1"
    ring = feature["geometry"]["coordinates"][0]
    assert ring[0] == ring[-1]
    assert min(point[0] for point in ring) >= 102
    assert min(point[1] for point in ring) >= 201


@pytest.mark.cells
def test_pyramidal_labels_render_as_transparent_boundaries(tmp_path: Path) -> None:
    pytest.importorskip("pyvips")
    from PIL import Image

    from histopia.cells._pipeline import _write_pyramidal_labels
    from histopia.visualization._wsi_tiles import (
        WsiLayer,
        _discover_levels,
        _render_cell_boundaries,
        _render_layer_tile,
    )

    shape = (900, 1200)
    raw = tmp_path / "labels.uint32.memmap"
    labels = np.memmap(raw, dtype=np.uint32, mode="w+", shape=shape)
    labels[:] = 0
    labels[100:500, 120:620] = 1
    labels[520:820, 700:1120] = 2
    labels.flush()
    del labels
    output = tmp_path / "labels.tiff"

    _write_pyramidal_labels(raw, output, width=shape[1], height=shape[0])
    levels, _mpp = _discover_levels(output)
    assert (levels[-1].width, levels[-1].height) == (shape[1], shape[0])
    assert len(levels) > 1
    assert any(level.source_kind == "subifd" for level in levels)

    import pyvips

    image = pyvips.Image.new_from_file(str(output), access="random")
    rendered = _render_cell_boundaries(image.crop(0, 0, image.width, image.height))
    rgba = np.asarray(Image.open(BytesIO(rendered)))
    assert rgba.shape == (shape[0], shape[1], 4)
    assert np.count_nonzero(rgba[..., 3]) > 0
    layer = WsiLayer(
        name="cells",
        path=output,
        digest="a" * 64,
        levels=levels,
        tile_size=512,
        microns_per_pixel=None,
        source_shape=(shape[1], shape[0]),
    )
    smallest = np.asarray(Image.open(BytesIO(_render_layer_tile(layer, 0, 0, 0))))
    assert smallest.shape[:2] == (levels[0].height, levels[0].width)
