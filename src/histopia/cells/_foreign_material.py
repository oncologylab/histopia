"""Conservative detection of saturated-cyan foreign material in cell labels.

Rare slide-processing material can form a long blue-green strand that a cell
model divides into plausible-looking instances.  Ordinary hematoxylin is blue
but does not normally satisfy the joint blue-over-green, green-over-red, and
low-blue-value constraints over a large fraction of a whole cell label.  The
gate in this module therefore acts on sealed native-pixel evidence and removes
only labels with substantial support from that distinctive color ordering.

The thresholds are intentionally immutable.  They were bounded on native
Ki67 section 6389/007: the artifact field contained 26 qualifying labels,
while three spatially separated viable-tissue controls contained none.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SATURATED_CYAN_FOREIGN_MATERIAL_ALGORITHM_VERSION = 90
SATURATED_CYAN_FOREIGN_MATERIAL_METHOD_PROFILE = "combined-containment-cpsam-wsi-v85"
SATURATED_CYAN_FOREIGN_MATERIAL_SCOPE = (
    "post-inference-saturated-cyan-foreign-material-exclusion-v1"
)


@dataclass(frozen=True, slots=True)
class SaturatedCyanForeignMaterialPolicy:
    """Exact native-pixel thresholds for the saturated-cyan safeguard."""

    minimum_blue_green_difference: int = 12
    minimum_green_red_difference: int = 8
    maximum_blue_value: int = 149
    minimum_instance_pixels: int = 16
    minimum_instance_fraction: float = 0.30

    def __post_init__(self) -> None:
        if self.minimum_blue_green_difference <= 0:
            raise ValueError("blue-green difference must be positive")
        if self.minimum_green_red_difference <= 0:
            raise ValueError("green-red difference must be positive")
        if not 0 <= self.maximum_blue_value <= 255:
            raise ValueError("maximum blue value must be in [0, 255]")
        if self.minimum_instance_pixels <= 0:
            raise ValueError("minimum instance pixels must be positive")
        if not 0 < self.minimum_instance_fraction <= 1:
            raise ValueError("minimum instance fraction must be in (0, 1]")


SATURATED_CYAN_FOREIGN_MATERIAL_POLICY = SaturatedCyanForeignMaterialPolicy()


def saturated_cyan_foreign_material_core(
    rgb: np.ndarray,
    *,
    policy: SaturatedCyanForeignMaterialPolicy = (
        SATURATED_CYAN_FOREIGN_MATERIAL_POLICY
    ),
) -> np.ndarray:
    """Return native pixels with the validated blue-green-red ordering."""

    values = np.asarray(rgb)
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError("RGB evidence must have shape (height, width, 3)")
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError("RGB evidence must be numeric")
    red = values[..., 0].astype(np.int16)
    green = values[..., 1].astype(np.int16)
    blue = values[..., 2].astype(np.int16)
    return (
        (blue - green >= policy.minimum_blue_green_difference)
        & (green - red >= policy.minimum_green_red_difference)
        & (blue <= policy.maximum_blue_value)
    )


def saturated_cyan_foreign_material_instances(
    instance_areas: np.ndarray,
    saturated_cyan_pixels: np.ndarray,
    *,
    policy: SaturatedCyanForeignMaterialPolicy = (
        SATURATED_CYAN_FOREIGN_MATERIAL_POLICY
    ),
) -> np.ndarray:
    """Select whole labels substantially supported by saturated-cyan pixels."""

    areas = np.asarray(instance_areas)
    cyan = np.asarray(saturated_cyan_pixels)
    if areas.ndim != 1 or cyan.ndim != 1 or areas.shape != cyan.shape:
        raise ValueError("instance evidence arrays must be matching vectors")
    if areas.size == 0:
        raise ValueError("instance evidence arrays must include background")
    if np.any(areas < 0) or np.any(cyan < 0):
        raise ValueError("instance evidence counts must be nonnegative")
    if np.any(cyan > areas):
        raise ValueError("saturated-cyan pixels cannot exceed instance area")
    minimum_fraction_pixels = np.ceil(
        areas.astype(np.float64) * policy.minimum_instance_fraction
    ).astype(np.uint64)
    selected = (
        (areas > 0)
        & (cyan >= policy.minimum_instance_pixels)
        & (cyan >= minimum_fraction_pixels)
    )
    selected = np.asarray(selected, dtype=bool)
    selected[0] = False
    return selected


def saturated_cyan_foreign_material_gate() -> dict[str, object]:
    """Return the exact request/QC payload for the validated safeguard."""

    policy = SATURATED_CYAN_FOREIGN_MATERIAL_POLICY
    return {
        "enabled": True,
        "minimum_blue_green_difference": policy.minimum_blue_green_difference,
        "minimum_green_red_difference": policy.minimum_green_red_difference,
        "maximum_blue_value": policy.maximum_blue_value,
        "minimum_instance_pixels": policy.minimum_instance_pixels,
        "minimum_instance_fraction": policy.minimum_instance_fraction,
        "instance_action": "remove_whole_labels_with_validated_color_support",
    }
