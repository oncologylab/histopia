from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from histopia.topology._model import ObservedSection
from histopia.topology._volume import (
    DenseVolume,
    _smooth_viewer_mesh,
    benchmark_envelope_methods,
    filter_persistent_components,
    load_registered_mask_stack,
    reconstruct_dense_volume,
    regularize_semantic_core,
    regularize_semantic_partition,
    write_connected_meshes,
)


def test_registered_masks_are_warped_to_one_physical_grid(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    processed.mkdir()
    first = np.zeros((20, 24), dtype=np.uint8)
    first[5:15, 5:13] = 255
    second = np.zeros_like(first)
    second[5:15, 7:15] = 255
    Image.fromarray(first).save(processed / "first.mask.png")
    Image.fromarray(second).save(processed / "second.mask.png")
    rows = [
        _registration_row("first.ndpi", np.eye(3), reference=True),
        _registration_row(
            "second.ndpi",
            np.asarray([[1, 0, -2], [0, 1, 0], [0, 0, 1]]),
            reference=False,
        ),
    ]
    rows[1]["mask_review"] = {"status": "pending"}
    (tmp_path / "registration_result.json").write_text(json.dumps({"slides": rows}))

    with pytest.raises(ValueError, match="not approved"):
        load_registered_mask_stack(
            tmp_path,
            ("first.ndpi", "second.ndpi"),
            target_spacing_um=1,
            max_xy_dim_px=64,
        )
    stack = load_registered_mask_stack(
        tmp_path,
        ("first.ndpi", "second.ndpi"),
        target_spacing_um=1,
        max_xy_dim_px=64,
        require_review=False,
    )

    assert stack.masks.shape[0] == 2
    assert max(stack.masks.shape[1:]) <= 64
    assert np.array_equal(stack.masks[0], stack.masks[1])
    assert all(row["mask_sha256"] for row in stack.provenance)


def test_dense_samples_are_not_reported_as_observed_sections() -> None:
    sections = tuple(_section(index) for index in range(3))
    masks = np.stack([section.support for section in sections])
    empty_link = {
        "source_indices": np.empty(0, dtype=np.int64),
        "target_indices": np.empty(0, dtype=np.int64),
        "source_um_xy": np.empty((0, 2)),
        "target_um_xy": np.empty((0, 2)),
        "confidence": np.empty(0, dtype=np.float32),
    }

    volume = reconstruct_dense_volume(
        masks,
        sections,
        (empty_link, empty_link),
        (0.0, 5.0, 10.0),
        (1, 1),
        (0, 0, 0),
        origin_um_xy=(0, 0),
        spacing_um=1,
        section_thickness_um=5,
        samples_per_interval=8,
        envelope_method="linear_sdf",
    )

    assert volume.envelope_sdf.shape[0] == 17
    assert np.count_nonzero(volume.observed_section_indices >= 0) == 3
    assert np.allclose(np.diff(volume.z_positions_um), 0.625)


def test_envelope_benchmark_prefers_guarded_baseline_for_smooth_change() -> None:
    sections = tuple(_section(index) for index in range(5))
    masks = np.stack([section.support for section in sections])

    result = benchmark_envelope_methods(
        masks,
        sections,
        tuple(float(index * 5) for index in range(5)),
        origin_um_xy=(0, 0),
        spacing_um=1,
    )

    assert result["selected_method"] == "linear_sdf"
    assert result["status"] == "passed"


def test_viewer_component_filter_rejects_large_single_section_regions() -> None:
    occupancy = np.zeros((5, 14, 14), dtype=bool)
    occupancy[0:5, 2:5, 2:5] = True
    occupancy[2, 13, 0] = True
    occupancy[2, 7:12, 7:12] = True

    filtered, before, after = filter_persistent_components(
        occupancy,
        observed_dense_indices=np.asarray([0, 2, 4]),
        voxel_volume_um3=1,
        minimum_component_volume_um3=5,
        minimum_observed_sections=2,
    )

    assert before == 3
    assert after == 1
    assert filtered[2, 3, 3]
    assert not filtered[2, 9, 9]
    assert not filtered[2, 13, 0]


def test_viewer_component_filter_can_omit_unsupported_semantic_region() -> None:
    occupancy = np.zeros((5, 14, 14), dtype=bool)
    occupancy[2, 3:11, 3:11] = True

    filtered, before, after = filter_persistent_components(
        occupancy,
        observed_dense_indices=np.asarray([0, 2, 4]),
        voxel_volume_um3=1,
        minimum_component_volume_um3=5,
        minimum_observed_sections=2,
        keep_largest_if_empty=False,
    )

    assert before == 1
    assert after == 0
    assert not filtered.any()


def test_viewer_component_filter_caps_scattered_regions_by_volume() -> None:
    occupancy = np.zeros((5, 36, 36), dtype=bool)
    anchors = ((2, 2), (2, 12), (2, 22), (12, 2), (12, 12))
    for index, (row, column) in enumerate(anchors):
        size = index + 2
        occupancy[:, row : row + size, column : column + size] = True

    filtered, before, after = filter_persistent_components(
        occupancy,
        observed_dense_indices=np.asarray([0, 2, 4]),
        voxel_volume_um3=1,
        minimum_component_volume_um3=1,
        minimum_observed_sections=3,
        max_components=3,
    )

    assert before == 5
    assert after == 3
    assert not filtered[:, 2:4, 2:4].any()
    assert filtered[:, 12:17, 12:17].all()


def test_viewer_mesh_smoothing_reduces_display_surface_roughness() -> None:
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 2.0],
        ],
        dtype=np.float32,
    )
    faces = np.asarray(
        [
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3],
        ],
        dtype=np.uint32,
    )

    smoothed = _smooth_viewer_mesh(
        vertices,
        faces,
        z_scale=12,
        iterations=5,
    )

    assert smoothed.shape == vertices.shape
    assert np.all(np.isfinite(smoothed))
    assert np.ptp(smoothed[:, 2]) < np.ptp(vertices[:, 2])


def test_semantic_core_regularization_closes_only_subpatch_gaps() -> None:
    occupancy = np.zeros((25, 15, 24), dtype=bool)
    occupancy[3:22, 3:12, 2:9] = True
    occupancy[3:22, 3:12, 10:17] = True
    occupancy[3:22, 3:12, 21:24] = True

    regularized = regularize_semantic_core(
        occupancy,
        spacing_um=56,
        z_spacing_um=0.625,
        source_patch_width_um=112,
    )

    assert regularized[12, 7, 9]
    assert not regularized[12, 7, 18:21].any()


def test_connected_meshes_include_persistent_cores_and_complete_partitions(
    tmp_path: Path,
) -> None:
    support = np.zeros((9, 20, 24), dtype=bool)
    support[:, 2:18, 2:22] = True
    membership = np.zeros((9, 2, 20, 24), dtype=np.float32)
    membership[:, 0, 2:18, 2:12] = 0.86
    membership[:, 0, 2:18, 12:22] = 0.14
    membership[:, 1, 2:18, 2:12] = 0.14
    membership[:, 1, 2:18, 12:22] = 0.86
    volume = DenseVolume(
        envelope_sdf=np.where(support, 1.0, -1.0).astype(np.float32),
        membership=membership,
        uncertainty=np.zeros_like(support, dtype=np.float32),
        z_positions_um=np.arange(9, dtype=float),
        observed_section_indices=np.asarray([0, -1, -1, -1, 1, -1, -1, -1, 2]),
        segments=np.zeros(9, dtype=np.int16),
    )

    envelope, cores, partitions, classes, uncertainty = write_connected_meshes(
        tmp_path,
        volume,
        palette=("#d73027", "#4575b4"),
        origin_um_xy=(0, 0),
        spacing_um=8,
        z_spacing_um=1,
        source_patch_width_um=16,
        section_thickness_um=4,
    )

    assert envelope["role"] == "envelope"
    assert uncertainty is None
    assert {row["class_index"] for row in cores} == {0, 1}
    assert {row["class_index"] for row in partitions} == {0, 1}
    assert sum(
        row["viewer_partition_volume_fraction_of_tissue"] for row in classes
    ) == pytest.approx(1.0)
    assert all(
        row["viewer_regularization"]["partition_is_exhaustive"] for row in partitions
    )
    assert all((tmp_path / row["viewer_asset"]).is_file() for row in partitions)


def test_semantic_partition_reassigns_only_small_unobserved_components() -> None:
    support = np.ones((3, 6, 7), dtype=bool)
    membership = np.zeros((3, 2, 6, 7), dtype=np.float32)
    membership[:, 0] = 0.8
    membership[:, 1] = 0.2
    membership[:, 1, 2:5, 4:7] = 0.9
    membership[:, 0, 2:5, 4:7] = 0.1
    membership[1, 1, 0, 0] = 0.95
    membership[1, 0, 0, 0] = 0.05

    labels, removed = regularize_semantic_partition(
        membership,
        support,
        observed_dense_indices=np.asarray([0, 2]),
        voxel_volume_um3=1,
        minimum_component_volume_um3=2,
    )

    assert np.all(labels[:, 2:5, 4:7] == 1)
    assert labels[1, 0, 0] == 0
    assert removed == (0, 1)
    assert np.all(labels >= 0)


def _section(offset: int) -> ObservedSection:
    labels = np.full((18, 20), -1, dtype=np.int16)
    labels[4:14, 3 + offset : 11 + offset] = 0
    labels[7:11, 6 + offset : 9 + offset] = 1
    membership = np.stack((labels == 0, labels == 1)).astype(np.float32)
    return ObservedSection(
        slide_id=f"section-{offset}",
        labels=labels,
        membership=membership,
        support=labels >= 0,
        tissue_fraction=(labels >= 0).astype(np.float32),
        sparse_labels=labels[labels >= 0],
    )


def _registration_row(
    name: str,
    transform: np.ndarray,
    *,
    reference: bool,
) -> dict[str, object]:
    return {
        "path": str(Path("/data") / name),
        "is_reference": reference,
        "transform": {"matrix": transform.tolist()},
        "geometry": {
            "thumbnail_shape": [20, 24],
            "thumbnail_to_physical": np.eye(3).tolist(),
        },
        "mask": {"accepted": True, "method": "test"},
        "mask_review": {"status": "auto_pass"},
    }
