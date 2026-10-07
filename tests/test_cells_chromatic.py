import numpy as np
import pytest

from histopia.cells import (
    ChromaticMicroclusterPolicy,
    blue_counterstain_core,
    chromatic_microcluster_gate,
    clustered_chromatic_microobject_instances,
)


def _policy() -> ChromaticMicroclusterPolicy:
    return ChromaticMicroclusterPolicy(
        maximum_instance_area_pixels=300,
        minimum_counterstain_fraction=0.8,
        maximum_centroid_distance_pixels=32.0,
        minimum_component_area_pixels=2_000,
        minimum_component_instances=8,
        maximum_component_aspect_ratio=3.0,
        minimum_component_fill_fraction=0.10,
    )


def test_blue_counterstain_core_requires_blue_dominance() -> None:
    rgb = np.array([[[80, 90, 110], [120, 80, 100], [50, 50, 53]]], dtype=np.uint8)

    core = blue_counterstain_core(rgb, minimum_blue_ratio=1.08)

    assert core.tolist() == [[True, False, False]]


def test_residual_foam_v2_gate_is_an_exact_immutable_profile() -> None:
    gate = chromatic_microcluster_gate("residual-foam-microobjects-v2")

    assert gate == {
        "enabled": True,
        "profile": "residual-foam-microobjects-v2",
        "minimum_blue_ratio": 1.08,
        "maximum_instance_area_pixels": 120,
        "minimum_counterstain_fraction": 0.01,
        "maximum_centroid_distance_pixels": 64.0,
        "minimum_component_area_pixels": 300,
        "minimum_component_instances": 12,
        "maximum_component_aspect_ratio": 3.0,
        "minimum_component_fill_fraction": 0.01,
        "instance_action": "remove_whole_labels_in_validated_compact_cluster",
    }


def test_residual_foam_v3_gate_is_an_exact_immutable_profile() -> None:
    gate = chromatic_microcluster_gate("residual-foam-microobjects-v3")

    assert gate == {
        "enabled": True,
        "profile": "residual-foam-microobjects-v3",
        "minimum_blue_ratio": 1.08,
        "maximum_instance_area_pixels": 150,
        "minimum_counterstain_fraction": 0.20,
        "maximum_centroid_distance_pixels": 64.0,
        "minimum_component_area_pixels": 300,
        "minimum_component_instances": 20,
        "maximum_component_aspect_ratio": 3.0,
        "minimum_component_fill_fraction": 0.01,
        "instance_action": "remove_whole_labels_in_validated_compact_cluster",
    }


def test_pas_anucleate_v2_gate_is_an_exact_immutable_profile() -> None:
    gate = chromatic_microcluster_gate("pas-anucleate-microobjects-v2")

    assert gate == {
        "enabled": True,
        "profile": "pas-anucleate-microobjects-v2",
        "minimum_blue_ratio": 1.08,
        "maximum_instance_area_pixels": 150,
        "minimum_counterstain_fraction": 0.40,
        "maximum_centroid_distance_pixels": 32.0,
        "minimum_component_area_pixels": 1500,
        "minimum_component_instances": 30,
        "maximum_component_aspect_ratio": 3.0,
        "minimum_component_fill_fraction": 0.05,
        "instance_action": "remove_whole_labels_in_validated_compact_cluster",
    }


def test_retired_pas_profile_cannot_be_promoted() -> None:
    with pytest.raises(ValueError, match="unknown chromatic microcluster profile"):
        chromatic_microcluster_gate("pas-magenta-anuclear-v1")


def test_clustered_chromatic_microobjects_select_compact_anucleate_group() -> None:
    count = 13
    areas = np.zeros(count, dtype=np.int64)
    areas[1:] = 250
    x = np.zeros(count, dtype=np.float64)
    y = np.zeros(count, dtype=np.float64)
    cluster_x = np.array([0, 20, 40, 0, 20, 40, 0, 20, 40, 20], dtype=float)
    cluster_y = np.array([0, 0, 0, 20, 20, 20, 40, 40, 40, 60], dtype=float)
    x[1:11] = cluster_x * areas[1:11]
    y[1:11] = cluster_y * areas[1:11]
    x[11:] = np.array([400, 500]) * areas[11:]
    y[11:] = np.array([400, 500]) * areas[11:]
    counterstain = np.zeros(count, dtype=np.int64)

    selected = clustered_chromatic_microobject_instances(
        areas,
        x,
        y,
        counterstain,
        policy=_policy(),
    )

    assert selected[1:11].all()
    assert not selected[[0, 11, 12]].any()


def test_clustered_chromatic_microobjects_protect_supported_or_linear_cells() -> None:
    count = 19
    areas = np.zeros(count, dtype=np.int64)
    areas[1:] = 250
    x = np.zeros(count, dtype=np.float64)
    y = np.zeros(count, dtype=np.float64)
    x[1:10] = np.arange(9) * 20 * areas[1:10]
    y[1:10] = 20 * areas[1:10]
    x[10:] = (np.arange(9) % 3) * 20 * areas[10:]
    y[10:] = (np.arange(9) // 3) * 20 * areas[10:]
    counterstain = np.zeros(count, dtype=np.int64)
    counterstain[10:] = 225

    selected = clustered_chromatic_microobject_instances(
        areas,
        x,
        y,
        counterstain,
        policy=_policy(),
    )

    assert not selected.any()


@pytest.mark.parametrize(
    "mutation",
    [
        {"maximum_instance_area_pixels": 0},
        {"minimum_counterstain_fraction": 0.0},
        {"maximum_centroid_distance_pixels": 0.0},
        {"minimum_component_instances": 1},
    ],
)
def test_chromatic_policy_rejects_invalid_thresholds(
    mutation: dict[str, float],
) -> None:
    values: dict[str, float | int] = {
        "maximum_instance_area_pixels": 300,
        "minimum_counterstain_fraction": 0.8,
        "maximum_centroid_distance_pixels": 32.0,
        "minimum_component_area_pixels": 2_000,
        "minimum_component_instances": 8,
        "maximum_component_aspect_ratio": 3.0,
        "minimum_component_fill_fraction": 0.10,
    }
    values.update(mutation)

    with pytest.raises(ValueError):
        ChromaticMicroclusterPolicy(**values)  # type: ignore[arg-type]
