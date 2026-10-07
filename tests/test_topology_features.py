from __future__ import annotations

import numpy as np
import pytest

from histopia.topology import (
    SpatialFeatureConfig,
    extract_spatial_feature_spectrum,
)


def _record(spectrum, feature: str, *, scope: str, class_name: str | None = None):
    matches = [
        item
        for item in spectrum.records
        if item.feature == feature
        and item.scope == scope
        and item.class_name == class_name
    ]
    assert len(matches) == 1
    return matches[0]


def test_two_dimensional_composition_entropy_and_graph_features() -> None:
    labels = np.array(
        [
            [0, 0, -1, 1],
            [0, 0, -1, 1],
            [-1, -1, -1, 1],
            [0, -1, 1, 1],
        ],
        dtype=np.int16,
    )
    spectrum = extract_spatial_feature_spectrum(
        labels,
        spacing_um_xy=(10, 10),
        class_names=("epithelium", "stroma"),
        scopes=("2d",),
        config=SpatialFeatureConfig(
            entropy_window_um=(20,),
            contact_distance_um=50,
            large_region_area_um2=10_000,
        ),
    )

    assert _record(
        spectrum,
        "area_fraction",
        scope="2d",
        class_name="epithelium",
    ).value == pytest.approx(5 / 10)
    assert _record(spectrum, "semantic_entropy", scope="2d").value == pytest.approx(1)
    assert (
        _record(
            spectrum,
            "region_nodes",
            scope="2d",
            class_name="epithelium",
        ).value
        == 2
    )
    assert any(item.class_pair == ("epithelium", "stroma") for item in spectrum.records)


def test_large_region_is_split_using_physical_blocks() -> None:
    labels = np.zeros((10, 10), dtype=np.int16)
    spectrum = extract_spatial_feature_spectrum(
        labels,
        spacing_um_xy=(100, 100),
        class_names=("tissue",),
        scopes=("2d",),
        config=SpatialFeatureConfig(
            entropy_window_um=(200,),
            contact_distance_um=500,
            large_region_area_um2=100_000,
            region_block_width_um=500,
        ),
    )

    assert (
        _record(
            spectrum,
            "region_nodes",
            scope="2d",
            class_name="tissue",
        ).value
        == 4
    )


def test_adjacent_and_volume_features_use_physical_z() -> None:
    stack = np.full((3, 4, 4), -1, dtype=np.int16)
    stack[0, 1:3, 1:3] = 0
    stack[1, 1:3, 1:4] = 0
    stack[2, 1:3, 2:4] = 0
    stack[:, 0, 0] = 1
    spectrum = extract_spatial_feature_spectrum(
        stack,
        spacing_um_xy=(100, 100),
        z_positions_um=np.array([0, 10, 30]),
        class_names=("epithelium", "immune"),
        config=SpatialFeatureConfig(
            entropy_window_um=(200,),
            section_thickness_um=4,
        ),
    )

    assert _record(spectrum, "mean_support_dice", scope="2.5d").value > 0
    assert (
        _record(
            spectrum,
            "z_span",
            scope="3d",
            class_name="epithelium",
        ).value
        == 30
    )
    assert (
        _record(
            spectrum,
            "volume",
            scope="3d",
            class_name="epithelium",
        ).value
        > 0
    )
    assert any(
        item.scope == "3d"
        and item.feature == "interface_area"
        and item.class_pair == ("epithelium", "immune")
        for item in spectrum.records
    )


def test_spatial_features_are_invariant_to_label_translation() -> None:
    first = np.full((8, 8), -1, dtype=np.int16)
    first[1:3, 1:3] = 0
    first[5:7, 5:7] = 0
    second = np.full((12, 12), -1, dtype=np.int16)
    second[3:5, 4:6] = 0
    second[7:9, 8:10] = 0
    config = SpatialFeatureConfig(
        entropy_window_um=(20,),
        contact_distance_um=100,
        large_region_area_um2=10_000,
    )

    left = extract_spatial_feature_spectrum(
        first,
        spacing_um_xy=(10, 10),
        class_names=("tissue",),
        scopes=("2d",),
        config=config,
    )
    right = extract_spatial_feature_spectrum(
        second,
        spacing_um_xy=(10, 10),
        class_names=("tissue",),
        scopes=("2d",),
        config=config,
    )

    for feature in ("region_nodes", "region_edges", "mean_edge_distance"):
        assert _record(
            left,
            feature,
            scope="2d",
            class_name="tissue",
        ).value == pytest.approx(
            _record(
                right,
                feature,
                scope="2d",
                class_name="tissue",
            ).value
        )


def test_spatial_features_validate_dimensions_and_z_order() -> None:
    with pytest.raises(ValueError, match="require a label stack"):
        extract_spatial_feature_spectrum(
            np.zeros((3, 3), dtype=np.int16),
            spacing_um_xy=(1, 1),
            scopes=("3d",),
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        extract_spatial_feature_spectrum(
            np.zeros((2, 3, 3), dtype=np.int16),
            spacing_um_xy=(1, 1),
            z_positions_um=np.array([1, 1]),
            scopes=("3d",),
        )
    with pytest.raises(ValueError, match="at least two sections"):
        extract_spatial_feature_spectrum(
            np.zeros((1, 3, 3), dtype=np.int16),
            spacing_um_xy=(1, 1),
            scopes=("2.5d",),
        )


def test_all_background_section_has_defined_zero_entropy() -> None:
    spectrum = extract_spatial_feature_spectrum(
        np.full((3, 3), -1, dtype=np.int16),
        spacing_um_xy=(10, 10),
        scopes=("2d",),
        config=SpatialFeatureConfig(entropy_window_um=(20,)),
    )

    assert spectrum.class_names == ()
    assert _record(spectrum, "semantic_entropy", scope="2d").value == 0
    assert _record(spectrum, "local_entropy_20um", scope="2d").value == 0
