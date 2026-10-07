import numpy as np
import pytest

from histopia.cells import (
    SaturatedCyanForeignMaterialPolicy,
    saturated_cyan_foreign_material_core,
    saturated_cyan_foreign_material_gate,
    saturated_cyan_foreign_material_instances,
)


def test_saturated_cyan_core_requires_joint_channel_order_and_darkness() -> None:
    rgb = np.array(
        [
            [
                [60, 75, 90],
                [60, 67, 90],
                [60, 80, 90],
                [100, 120, 160],
            ]
        ],
        dtype=np.uint8,
    )

    core = saturated_cyan_foreign_material_core(rgb)

    assert core.tolist() == [[True, False, False, False]]


def test_saturated_cyan_gate_is_exact_and_immutable() -> None:
    assert saturated_cyan_foreign_material_gate() == {
        "enabled": True,
        "minimum_blue_green_difference": 12,
        "minimum_green_red_difference": 8,
        "maximum_blue_value": 149,
        "minimum_instance_pixels": 16,
        "minimum_instance_fraction": 0.30,
        "instance_action": "remove_whole_labels_with_validated_color_support",
    }


def test_saturated_cyan_instances_require_pixel_count_and_fraction() -> None:
    areas = np.array([0, 100, 100, 40, 200], dtype=np.uint64)
    cyan = np.array([0, 30, 29, 15, 60], dtype=np.uint64)

    selected = saturated_cyan_foreign_material_instances(areas, cyan)

    assert selected.tolist() == [False, True, False, False, True]


@pytest.mark.parametrize(
    "mutation",
    [
        {"minimum_blue_green_difference": 0},
        {"minimum_green_red_difference": 0},
        {"maximum_blue_value": 256},
        {"minimum_instance_pixels": 0},
        {"minimum_instance_fraction": 0.0},
    ],
)
def test_saturated_cyan_policy_rejects_invalid_thresholds(
    mutation: dict[str, float],
) -> None:
    with pytest.raises(ValueError):
        SaturatedCyanForeignMaterialPolicy(**mutation)  # type: ignore[arg-type]


def test_saturated_cyan_instances_reject_impossible_evidence() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        saturated_cyan_foreign_material_instances(
            np.array([0, 10]),
            np.array([0, 11]),
        )
