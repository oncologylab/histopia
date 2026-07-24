"""Dependency-light libvips image normalization."""

from __future__ import annotations

from typing import Any


def normalize_vips_rgb_uchar(image: Any) -> Any:
    """Return a three-band uchar image for Histopia brightfield workflows.

    Two-band inputs are interpreted as luminance plus alpha and composited onto
    white before the luminance band is expanded. This avoids turning transparent
    scanner background into false dark tissue.
    """

    bands = int(image.bands)
    if bands < 1:
        raise ValueError("libvips image must contain at least one band")
    if bands > 3:
        image = image[:3]
    elif bands == 2:
        image = image.flatten(background=[_alpha_max(image)])
    if image.bands == 1:
        image = image.bandjoin([image, image])
    if image.bands != 3:
        raise ValueError("libvips image could not be normalized to RGB")
    if image.format != "uchar":
        image = image.cast("uchar")
    return image


def _alpha_max(image: Any) -> float:
    interpretation = str(getattr(image, "interpretation", ""))
    if interpretation in {"grey16", "rgb16"}:
        return 65_535.0
    if interpretation == "scrgb":
        return 1.0
    return 255.0
