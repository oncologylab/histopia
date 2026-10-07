from pathlib import Path

import numpy as np
import pytest

from histopia.cells._algorithms import (
    clean_mask,
    constrain_labels_to_tissue,
    containment_merge,
    hematoxylin_evidence,
    local_stain_evidence,
    merge_small_into_large,
    retain_nuclear_supported_instances,
)
from histopia.cells._config import CellSegmentationConfig
from histopia.cells._dense_small_cell import (
    bounded_instance_territories,
    complete_dense_small_cell_gaps,
    connected_gap_tile_scopes,
    dense_small_cell_gap_evidence,
    expand_gap_bins,
)
from histopia.cells._tiles import DiskLabelCanvas, make_tiles, mutual_best_mapping


def test_combined_mask_keeps_large_instances_hit_by_small_centroids() -> None:
    small = np.zeros((8, 12), dtype=np.int32)
    small[2:4, 2:4] = 1
    large = np.zeros_like(small)
    large[1:5, 1:5] = 7
    large[1:5, 7:11] = 9

    combined = merge_small_into_large(small, large)

    assert set(np.unique(combined)) == {0, 1}
    assert np.all(combined[1:5, 1:5] == 1)
    assert np.all(combined[:, 7:11] == 0)


def test_containment_merge_adds_only_complementary_instances() -> None:
    first = np.zeros((10, 12), dtype=np.int32)
    first[1:5, 1:5] = 1
    second = np.zeros_like(first)
    second[2:4, 2:4] = 1
    second[5:9, 7:11] = 2

    merged = containment_merge(first, second, threshold=0.1)

    assert set(np.unique(merged)) == {0, 1, 2}
    assert merged[2, 2] == 1
    assert merged[6, 8] == 2


def test_clean_mask_fills_holes_and_splits_disconnected_labels() -> None:
    mask = np.zeros((8, 8), dtype=np.int32)
    mask[1:4, 1:4] = 4
    mask[2, 2] = 0
    mask[6, 6] = 4

    cleaned = clean_mask(mask)

    assert cleaned[2, 2] > 0
    assert cleaned[6, 6] != cleaned[1, 1]
    assert set(np.unique(cleaned)) == {0, 1, 2}


def test_tissue_constraint_removes_every_outside_label_pixel() -> None:
    labels = np.zeros((5, 6), dtype=np.int32)
    labels[1:4, 1:5] = 7
    tissue = np.zeros_like(labels, dtype=bool)
    tissue[:3, :4] = True

    constrained = constrain_labels_to_tissue(labels, tissue)

    assert constrained.dtype == labels.dtype
    assert np.all(constrained[~tissue] == 0)
    assert np.array_equal(constrained[tissue], labels[tissue])


def test_local_stain_evidence_rejects_flat_bright_background() -> None:
    image = np.full((8, 10, 3), (235, 233, 234), dtype=np.uint8)
    image[2:6, 2:5] = (145, 120, 160)

    evidence = local_stain_evidence(image)

    assert np.all(evidence[2:6, 2:5])
    assert not np.any(evidence[:, 6:])


def test_uint8_stain_evidence_matches_direct_optical_density() -> None:
    image = np.random.default_rng(17).integers(80, 256, (13, 17, 3), dtype=np.uint8)
    background = np.percentile(image.reshape(-1, 3), 98, axis=0).astype(np.float32)
    values = image.astype(np.float32)
    optical_density = np.log((background[None, None, :] + 1) / (values + 1))
    np.maximum(optical_density, 0, out=optical_density)
    expected = np.linalg.norm(optical_density, axis=2) >= 0.08

    assert np.array_equal(local_stain_evidence(image), expected)


def test_nuclear_support_rejects_brown_debris_and_keeps_blue_nucleus() -> None:
    image = np.full((12, 18, 3), (242, 240, 238), dtype=np.uint8)
    image[3:6, 3:6] = (72, 58, 128)
    image[3:8, 12:16] = (112, 70, 34)
    labels = np.zeros(image.shape[:2], dtype=np.int32)
    labels[2:8, 2:8] = 1
    labels[2:9, 11:17] = 2

    evidence = hematoxylin_evidence(image, minimum_optical_density=0.08)
    retained = retain_nuclear_supported_instances(
        labels,
        evidence,
        minimum_pixels=3,
        minimum_fraction=0.01,
    )

    assert np.any(evidence[3:6, 3:6])
    assert not np.any(evidence[3:8, 12:16])
    assert set(np.unique(retained)) == {0, 1}
    assert retained[4, 4] == 1
    assert retained[4, 13] == 0


def test_dense_small_cell_gap_requires_dense_global_nuclear_context() -> None:
    nuclear = np.ones((256, 256), dtype=bool)
    labels = np.ones((256, 256), dtype=np.int32)
    labels[:64, :128] = 0

    evidence = dense_small_cell_gap_evidence(
        nuclear,
        labels,
        bin_size=64,
        minimum_gap_bins=2,
    )

    assert evidence.triggered is True
    assert evidence.global_nuclear_fraction == 1.0
    assert evidence.gap_bin_count == 2
    assert evidence.bin_count == 16
    assert evidence.gap_bin_fraction == pytest.approx(0.125)
    assert np.array_equal(evidence.gap_bins[0], [True, True, False, False])

    sparse = nuclear.copy()
    sparse[128:] = False
    control = dense_small_cell_gap_evidence(
        sparse,
        labels,
        bin_size=64,
        minimum_gap_bins=2,
    )
    assert control.triggered is False
    assert not np.any(control.gap_bins)


def test_dense_small_cell_gap_projection_adds_bounded_context() -> None:
    bins = np.zeros((3, 4), dtype=bool)
    bins[1, 1] = True

    pixels = expand_gap_bins(
        bins,
        (130, 195),
        bin_size=64,
        context_bins=1,
    )

    assert pixels.shape == (130, 195)
    assert np.all(pixels[:128, :192])
    assert not np.any(pixels[128:, 192:])


def test_connected_gap_tile_scopes_continue_only_seeded_global_component() -> None:
    seed = np.asarray([[False, True], [False, False]])
    continuation = np.asarray([[True, True], [False, False]])
    distant = np.asarray([[False, True], [False, False]])

    scopes = connected_gap_tile_scopes(
        {
            "seed": seed,
            "continuation": continuation,
            "distant": distant,
        },
        {
            "seed": (0, 0),
            "continuation": (64, 0),
            "distant": (512, 0),
        },
        {"seed"},
        bin_size=64,
    )

    assert set(scopes) == {"seed", "continuation"}
    np.testing.assert_array_equal(scopes["seed"], seed)
    np.testing.assert_array_equal(scopes["continuation"], continuation)


def test_connected_gap_tile_scopes_reject_misaligned_geometry() -> None:
    with pytest.raises(ValueError, match="align"):
        connected_gap_tile_scopes(
            {"seed": np.ones((1, 1), dtype=bool)},
            {"seed": (1, 0)},
            {"seed"},
            bin_size=64,
        )


def test_bounded_instance_territories_preserve_seeds_and_growth_limit() -> None:
    markers = np.zeros((9, 13), dtype=np.int32)
    markers[4, 3] = 7
    markers[4, 9] = 11
    support = np.ones_like(markers, dtype=bool)
    support[:3, :] = False

    cells = bounded_instance_territories(
        markers,
        maximum_growth_pixels=2,
        support_mask=support,
    )

    assert cells[4, 3] == 7
    assert cells[4, 9] == 11
    assert cells[4, 5] == 7
    assert cells[4, 7] == 11
    assert cells[4, 6] == 0
    assert not np.any(cells[:3])
    assert not np.any(cells[0, :])


def test_dense_small_cell_completion_preserves_source_and_adds_only_gap_cells() -> None:
    source = np.zeros((128, 128), dtype=np.int32)
    source[18:34, 18:34] = 4
    recovered = np.zeros_like(source)
    recovered[18:34, 18:34] = 1
    recovered[80:90, 80:90] = 2
    recovered[80:90, 20:30] = 3
    gap_bins = np.zeros((2, 2), dtype=bool)
    gap_bins[1, 1] = True

    completed = complete_dense_small_cell_gaps(
        source,
        recovered,
        gap_bins,
        context_bins=0,
    )

    assert np.all(completed[18:34, 18:34] > 0)
    assert np.all(completed[80:90, 80:90] > 0)
    assert not np.any(completed[80:90, 20:30])
    assert len(np.unique(completed[completed > 0])) == 2


def test_dense_small_cell_completion_rejects_invalid_overlap() -> None:
    labels = np.zeros((64, 64), dtype=np.int32)
    bins = np.ones((1, 1), dtype=bool)

    with pytest.raises(ValueError, match="between zero and one"):
        complete_dense_small_cell_gaps(
            labels,
            labels,
            bins,
            maximum_intersection_over_smaller=1.1,
        )


def test_dense_small_cell_helpers_reject_invalid_geometry() -> None:
    with pytest.raises(ValueError, match="same 2D shape"):
        dense_small_cell_gap_evidence(
            np.ones((4, 4), dtype=bool),
            np.ones((3, 4), dtype=np.int32),
        )
    with pytest.raises(ValueError, match="does not match"):
        expand_gap_bins(np.ones((3, 2), dtype=bool), (65, 65), bin_size=64)
    with pytest.raises(ValueError, match="nonnegative"):
        bounded_instance_territories(
            np.asarray([[0, -1]], dtype=np.int32),
        )


def test_tile_grid_is_end_aligned_and_mutual_mapping_is_stable() -> None:
    tiles = make_tiles(1500, 1700, tile_size=1024, overlap=128)
    assert tiles[0].y0 == 0 and tiles[0].x0 == 0
    assert tiles[-1].y1 == 1500 and tiles[-1].x1 == 1700

    local = np.zeros((5, 5), dtype=np.int32)
    existing = np.zeros_like(local)
    local[1:4, 1:4] = 2
    existing[1:4, 1:4] = 11
    assert mutual_best_mapping(local, existing, 0.5) == {2: 11}


def test_disk_canvas_reuses_instance_across_overlapping_tiles(tmp_path: Path) -> None:
    canvas = DiskLabelCanvas(tmp_path, (4, 6), resume=False)
    left, right = make_tiles(4, 6, tile_size=4, overlap=2)
    left_mask = np.zeros((4, 4), dtype=np.int32)
    right_mask = np.zeros((4, 4), dtype=np.int32)
    left_mask[1:3, 2:4] = 1
    right_mask[1:3, 0:2] = 1

    canvas.merge(left, left_mask, match_ios=0.5)
    canvas.merge(right, right_mask, match_ios=0.5)

    assert np.unique(canvas.labels[1:3, 2:4]).tolist() == [1]


def test_disk_canvas_accepts_already_sanitized_tile(tmp_path: Path) -> None:
    canvas = DiskLabelCanvas(tmp_path, (4, 4), resume=False)
    tile = make_tiles(4, 4, tile_size=4, overlap=2)[0]
    local = np.zeros((4, 4), dtype=np.int32)
    local[1:3, 1:3] = 8

    canvas.merge(tile, local, match_ios=0.5, sanitize=False)

    assert np.unique(canvas.labels).tolist() == [0, 1]


def test_new_disk_canvas_starts_with_zero_next_label(tmp_path: Path) -> None:
    canvas = DiskLabelCanvas(tmp_path, (7, 9), resume=False)

    assert canvas.next_label == 0
    assert not np.any(canvas.labels)

    canvas.labels[:] = 7
    canvas.weights[:] = 4
    canvas.flush()
    del canvas.labels
    del canvas.weights

    replacement = DiskLabelCanvas(tmp_path, (7, 9), resume=False)
    assert not np.any(replacement.labels)
    assert not np.any(replacement.weights)


def test_disk_canvas_recreates_incomplete_resume_pair(tmp_path: Path) -> None:
    shape = (7, 9)
    canvas = DiskLabelCanvas(tmp_path, shape, resume=False)
    canvas.labels[:] = 7
    canvas.weights[:] = 4
    canvas.flush()
    del canvas.labels
    del canvas.weights
    (tmp_path / "weights.uint16.memmap").write_bytes(b"")

    replacement = DiskLabelCanvas(tmp_path, shape, resume=True)

    assert not np.any(replacement.labels)
    assert not np.any(replacement.weights)
    assert (tmp_path / "labels.uint32.memmap").stat().st_size == 7 * 9 * 4
    assert (tmp_path / "weights.uint16.memmap").stat().st_size == 7 * 9 * 2


def test_cell_config_rejects_implicit_or_unbounded_controls(tmp_path: Path) -> None:
    config = CellSegmentationConfig(tmp_path, tmp_path / "out")
    assert config.model == "cpsam"
    assert config.tile_cache_reuse_run is None
    assert config.dense_small_cell_recovery_manifest is None
    assert config.method == "containment"
    assert config.flow_threshold == 0.0
    assert config.multiscale_nuclear_support is True
    assert config.global_nuclear_support is True
    assert config.global_nuclear_minimum_pixels == 8
    assert config.global_nuclear_minimum_fraction == 0.02
    assert config.source_tissue_context is True
    assert config.source_tissue_context_bin_size == 256
    assert config.source_tissue_context_minimum_fraction == 0.03
    assert config.source_tissue_context_minimum_instance_fraction == 0.01
    assert config.adaptive_nuclear_core is True
    assert config.adaptive_nuclear_core_percentile == 70.0
    assert config.adaptive_nuclear_core_minimum_area_um2 == 225.0
    assert config.adaptive_nuclear_core_minimum_pixels == 8
    assert config.adaptive_nuclear_core_minimum_fraction == 0.10
    assert config.adaptive_nuclear_core_minimum_blue_ratio == 1.08
    assert config.isolated_debris_gate is True
    assert config.isolated_debris_context_bin_size == 64
    assert config.isolated_debris_context_minimum_stain_fraction == 0.80
    assert config.isolated_debris_context_minimum_instance_fraction == 0.75
    assert config.isolated_debris_component_minimum_nuclear_pixels == 8
    assert config.isolated_debris_allow_nuclear_escape is True
    assert config.isolated_debris_allow_organized_escape is True
    assert config.isolated_debris_nuclear_minimum_pixels == 2
    assert config.isolated_debris_nuclear_minimum_fraction == 0.005
    assert config.isolated_debris_nuclear_minimum_blue_ratio == 1.08
    assert config.isolated_debris_neutral_dark_maximum_value == 90
    assert config.isolated_debris_neutral_dark_maximum_chroma == 30
    assert config.isolated_debris_neutral_dark_maximum_fraction == 0.50
    assert config.isolated_debris_very_dark_maximum_value == 60
    assert config.isolated_debris_very_dark_maximum_chroma == 20
    assert config.isolated_debris_very_dark_maximum_fraction == 0.45
    assert config.isolated_debris_micro_bin_size == 16
    assert config.isolated_debris_micro_minimum_stain_fraction == 0.80
    assert config.isolated_debris_micro_maximum_component_bins == 512
    assert config.isolated_debris_micro_minimum_nuclear_fraction == 0.01
    assert config.isolated_debris_micro_minimum_instance_fraction == 0.50
    assert config.isolated_debris_organized_bin_size == 4
    assert config.isolated_debris_organized_window_size == 1024
    assert config.isolated_debris_organized_window_overlap == 128
    assert config.isolated_debris_organized_minimum_bin_occupancy_fraction == 0.10
    assert config.isolated_debris_organized_minimum_component_pixels == 10_000
    assert config.isolated_debris_organized_compact_minimum_component_pixels == 20_000
    assert config.isolated_debris_organized_minimum_aspect_ratio == 4.0
    assert config.isolated_debris_organized_minimum_mean_instance_pixels == 350.0
    assert config.isolated_debris_organized_minimum_instance_fraction == 0.50
    assert config.isolated_debris_compact_unsupported_minimum_area_um2 == 10_000.0
    assert config.isolated_debris_compact_unsupported_maximum_aspect_ratio == 1.75
    assert (
        config.isolated_debris_compact_unsupported_maximum_mean_instance_pixels == 250.0
    )
    assert (
        config.isolated_debris_compact_unsupported_maximum_mean_red_blue_difference
        == 21.0
    )
    assert config.isolated_debris_compact_unsupported_strong_red_blue_difference == 40.0
    assert (
        config.isolated_debris_compact_unsupported_maximum_strong_chromatic_fraction
        == 0.10
    )
    assert config.isolated_debris_compact_unsupported_elongation_ratio == 2.0
    assert (
        config.isolated_debris_compact_unsupported_maximum_elongated_instance_fraction
        == 0.50
    )
    assert (
        config.isolated_debris_compact_unsupported_maximum_context_nuclear_fraction
        == 0.20
    )
    assert (
        config.isolated_debris_compact_unsupported_minimum_component_fill_fraction
        == 0.20
    )
    assert config.isolated_debris_compact_unsupported_minimum_instance_fraction == 0.50
    assert config.isolated_debris_foam_minimum_mean_intensity == 160.0
    assert config.isolated_debris_foam_maximum_mean_instance_pixels == 250.0
    assert config.isolated_debris_foam_maximum_strong_chromatic_fraction == 0.10
    assert config.isolated_debris_foam_maximum_context_nuclear_fraction == 0.50
    assert config.isolated_debris_foam_maximum_elongated_instance_fraction == 0.50
    assert config.isolated_debris_foam_core_minimum_pixels == 8
    assert config.isolated_debris_foam_core_minimum_bright_fraction == 0.80
    assert config.isolated_debris_foam_core_minimum_intensity == 180.0
    assert config.isolated_debris_foam_core_maximum_instance_area_um2 == 110.0
    assert config.isolated_debris_foam_core_minimum_component_area_um2 == 6500.0
    assert config.isolated_debris_foam_core_maximum_aspect_ratio == 2.50
    assert config.isolated_debris_oversized_chromatic_minimum_area_um2 == 1000.0
    assert config.isolated_debris_oversized_chromatic_minimum_fraction == 0.50
    assert config.isolated_debris_fold_minimum_instance_area_um2 == 300.0
    assert config.isolated_debris_fold_minimum_component_area_um2 == 750.0
    assert config.isolated_debris_fold_maximum_mean_red_blue_difference == 40.0
    assert config.isolated_debris_fold_maximum_mean_intensity == 180.0
    assert config.isolated_debris_fold_dense_minimum_instances == 3
    assert config.isolated_debris_fold_dense_minimum_aspect_ratio == 3.0
    assert config.isolated_debris_fold_dense_connectivity_dilation_bins == 8
    assert config.isolated_debris_fold_dense_maximum_mean_red_blue_difference == 60.0
    assert config.isolated_debris_fold_dense_maximum_mean_intensity == 115.0
    assert config.isolated_debris_detached_minimum_instance_area_um2 == 200.0
    assert config.isolated_debris_detached_minimum_component_area_um2 == 1000.0
    assert config.isolated_debris_detached_minimum_instances == 3
    assert config.isolated_debris_detached_minimum_aspect_ratio == 3.0
    assert config.isolated_debris_detached_maximum_mean_red_blue_difference == 35.0
    assert config.isolated_debris_detached_maximum_mean_intensity == 220.0
    assert config.isolated_debris_detached_context_window_size == 512
    assert config.isolated_debris_detached_maximum_prediction_fraction == 0.20
    assert config.isolated_debris_detached_compact_minimum_instance_area_um2 == 300.0
    assert config.isolated_debris_detached_compact_minimum_component_area_um2 == 750.0
    assert config.isolated_debris_detached_compact_minimum_instances == 2
    assert (
        config.isolated_debris_detached_compact_maximum_mean_red_blue_difference == 0.0
    )
    assert config.isolated_debris_detached_compact_maximum_mean_intensity == 180.0
    assert config.isolated_debris_glass_minimum_area_um2 == 25.0
    assert config.isolated_debris_glass_context_window_size == 256
    assert config.isolated_debris_glass_maximum_prediction_fraction == 0.40
    assert config.isolated_debris_glass_maximum_context_stain_fraction == 0.20
    assert (
        config.isolated_debris_glass_low_stain_maximum_mean_red_blue_difference == 35.0
    )
    assert config.isolated_debris_glass_low_stain_minimum_mean_intensity == 120.0
    assert config.isolated_debris_glass_minimum_context_fraction == 0.50
    assert config.isolated_debris_glass_maximum_mean_red_blue_difference == 25.0
    assert config.isolated_debris_glass_minimum_mean_intensity == 190.0
    assert config.isolated_debris_necrotic_minimum_component_area_um2 == 5000.0
    assert config.isolated_debris_necrotic_maximum_aspect_ratio == 2.5
    assert config.isolated_debris_necrotic_minimum_fill_fraction == 0.05
    assert config.isolated_debris_necrotic_maximum_fill_fraction == 0.45
    assert config.isolated_debris_necrotic_minimum_mean_red_blue_difference == -15.0
    assert config.isolated_debris_necrotic_maximum_mean_red_blue_difference == 15.0
    assert config.isolated_debris_necrotic_minimum_mean_intensity == 160.0
    assert config.isolated_debris_necrotic_minimum_instance_fraction == 0.50
    assert config.isolated_debris_brown_minimum_component_area_um2 == 4.0
    assert config.isolated_debris_brown_maximum_aspect_ratio == 3.0
    assert config.isolated_debris_brown_minimum_fill_fraction == 0.12
    assert config.isolated_debris_brown_maximum_fill_fraction == 0.45
    assert config.isolated_debris_brown_minimum_mean_red_blue_difference == 50.0
    assert config.isolated_debris_brown_maximum_mean_intensity == 120.0
    assert config.isolated_debris_brown_minimum_instance_fraction == 0.50
    assert config.isolated_debris_brown_maximum_source_component_area_um2 == 50_000.0
    assert config.isolated_debris_brown_source_maximum_mean_intensity == 218.0
    assert config.isolated_debris_brown_source_dilation_bins == 0
    assert config.isolated_debris_organized_strong_red_blue_difference == 40.0
    assert config.isolated_debris_organized_minimum_strong_chromatic_fraction == 0.25
    with pytest.raises(ValueError, match="model must be"):
        CellSegmentationConfig(tmp_path, tmp_path / "out", model="default")
    with pytest.raises(ValueError, match="requires tile_cache_reuse_run"):
        CellSegmentationConfig(
            tmp_path,
            tmp_path / "out",
            require_complete_tile_cache_reuse=True,
        )
    with pytest.raises(ValueError, match="one explicit section"):
        CellSegmentationConfig(
            tmp_path,
            tmp_path / "out",
            tile_cache_reuse_run=tmp_path / "source",
            require_complete_tile_cache_reuse=True,
            dense_small_cell_recovery_manifest=tmp_path / "manifest.json",
        )
    with pytest.raises(ValueError, match="smaller"):
        CellSegmentationConfig(tmp_path, tmp_path / "out", overlap=1024)
    with pytest.raises(ValueError, match="source_tissue_context_minimum_fraction"):
        CellSegmentationConfig(
            tmp_path,
            tmp_path / "out",
            source_tissue_context_minimum_fraction=0,
        )
    with pytest.raises(ValueError, match="necrotic_minimum_fill_fraction"):
        CellSegmentationConfig(
            tmp_path,
            tmp_path / "out",
            isolated_debris_necrotic_minimum_fill_fraction=0.50,
        )
    with pytest.raises(ValueError, match="brown_minimum_fill_fraction"):
        CellSegmentationConfig(
            tmp_path,
            tmp_path / "out",
            isolated_debris_brown_minimum_fill_fraction=0.50,
        )
    with pytest.raises(ValueError, match="brown_source_dilation_bins"):
        CellSegmentationConfig(
            tmp_path,
            tmp_path / "out",
            isolated_debris_brown_source_dilation_bins=-1,
        )
    with pytest.raises(ValueError, match="organized_window_overlap"):
        CellSegmentationConfig(
            tmp_path,
            tmp_path / "out",
            isolated_debris_organized_window_size=128,
            isolated_debris_organized_window_overlap=128,
        )
    with pytest.raises(ValueError, match="multiples"):
        CellSegmentationConfig(
            tmp_path,
            tmp_path / "out",
            isolated_debris_organized_window_overlap=130,
        )
    with pytest.raises(ValueError, match="oversized_chromatic_minimum_area_um2"):
        CellSegmentationConfig(
            tmp_path,
            tmp_path / "out",
            isolated_debris_oversized_chromatic_minimum_area_um2=0,
        )
    with pytest.raises(ValueError, match="compact_unsupported_minimum_area_um2"):
        CellSegmentationConfig(
            tmp_path,
            tmp_path / "out",
            isolated_debris_compact_unsupported_minimum_area_um2=0,
        )
    with pytest.raises(ValueError, match="compact_unsupported_maximum_aspect_ratio"):
        CellSegmentationConfig(
            tmp_path,
            tmp_path / "out",
            isolated_debris_compact_unsupported_maximum_aspect_ratio=0.5,
        )
    with pytest.raises(
        ValueError, match="compact_unsupported_minimum_instance_fraction"
    ):
        CellSegmentationConfig(
            tmp_path,
            tmp_path / "out",
            isolated_debris_compact_unsupported_minimum_instance_fraction=0,
        )
    with pytest.raises(
        ValueError, match="compact_unsupported_maximum_mean_instance_pixels"
    ):
        CellSegmentationConfig(
            tmp_path,
            tmp_path / "out",
            isolated_debris_compact_unsupported_maximum_mean_instance_pixels=0,
        )
    with pytest.raises(
        ValueError, match="compact_unsupported_maximum_mean_red_blue_difference"
    ):
        CellSegmentationConfig(
            tmp_path,
            tmp_path / "out",
            isolated_debris_compact_unsupported_maximum_mean_red_blue_difference=(
                float("nan")
            ),
        )
    with pytest.raises(
        ValueError,
        match="compact_unsupported_maximum_context_nuclear_fraction",
    ):
        CellSegmentationConfig(
            tmp_path,
            tmp_path / "out",
            isolated_debris_compact_unsupported_maximum_context_nuclear_fraction=2,
        )
    with pytest.raises(
        ValueError,
        match="compact_unsupported_maximum_strong_chromatic_fraction",
    ):
        CellSegmentationConfig(
            tmp_path,
            tmp_path / "out",
            isolated_debris_compact_unsupported_maximum_strong_chromatic_fraction=2,
        )
    with pytest.raises(ValueError, match="compact_unsupported_elongation_ratio"):
        CellSegmentationConfig(
            tmp_path,
            tmp_path / "out",
            isolated_debris_compact_unsupported_elongation_ratio=1,
        )
    with pytest.raises(
        ValueError,
        match="compact_unsupported_maximum_elongated_instance_fraction",
    ):
        CellSegmentationConfig(
            tmp_path,
            tmp_path / "out",
            isolated_debris_compact_unsupported_maximum_elongated_instance_fraction=2,
        )
    with pytest.raises(
        ValueError,
        match="compact_unsupported_minimum_component_fill_fraction",
    ):
        CellSegmentationConfig(
            tmp_path,
            tmp_path / "out",
            isolated_debris_compact_unsupported_minimum_component_fill_fraction=0,
        )
    with pytest.raises(ValueError, match="foam_minimum_mean_intensity"):
        CellSegmentationConfig(
            tmp_path,
            tmp_path / "out",
            isolated_debris_foam_minimum_mean_intensity=300,
        )
    with pytest.raises(ValueError, match="foam_maximum_context_nuclear_fraction"):
        CellSegmentationConfig(
            tmp_path,
            tmp_path / "out",
            isolated_debris_foam_maximum_context_nuclear_fraction=2,
        )
    with pytest.raises(ValueError, match="foam_maximum_mean_instance_pixels"):
        CellSegmentationConfig(
            tmp_path,
            tmp_path / "out",
            isolated_debris_foam_maximum_mean_instance_pixels=0,
        )
    with pytest.raises(ValueError, match="foam_maximum_strong_chromatic_fraction"):
        CellSegmentationConfig(
            tmp_path,
            tmp_path / "out",
            isolated_debris_foam_maximum_strong_chromatic_fraction=2,
        )
    with pytest.raises(ValueError, match="foam_maximum_elongated_instance_fraction"):
        CellSegmentationConfig(
            tmp_path,
            tmp_path / "out",
            isolated_debris_foam_maximum_elongated_instance_fraction=-1,
        )
    with pytest.raises(ValueError, match="oversized_chromatic_minimum_fraction"):
        CellSegmentationConfig(
            tmp_path,
            tmp_path / "out",
            isolated_debris_oversized_chromatic_minimum_fraction=0,
        )
    with pytest.raises(ValueError, match="adaptive_nuclear_core_percentile"):
        CellSegmentationConfig(
            tmp_path,
            tmp_path / "out",
            adaptive_nuclear_core_percentile=100,
        )
    with pytest.raises(ValueError, match="adaptive_nuclear_core_minimum_blue_ratio"):
        CellSegmentationConfig(
            tmp_path,
            tmp_path / "out",
            adaptive_nuclear_core_minimum_blue_ratio=1,
        )
    with pytest.raises(
        ValueError, match="isolated_debris_context_minimum_stain_fraction"
    ):
        CellSegmentationConfig(
            tmp_path,
            tmp_path / "out",
            isolated_debris_context_minimum_stain_fraction=0,
        )
    with pytest.raises(ValueError, match="isolated_debris_nuclear_minimum_blue_ratio"):
        CellSegmentationConfig(
            tmp_path,
            tmp_path / "out",
            isolated_debris_nuclear_minimum_blue_ratio=1,
        )
