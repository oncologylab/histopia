from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage as ndi

from histopia.registration._io import (
    _mask_boundary,
    _prepare_mask_overlay,
    overlay_mask,
)


def _reference_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    source = np.asarray(image, dtype=np.uint8)
    mask_bool = np.asarray(mask, dtype=bool)
    luminance = np.mean(source, axis=2, keepdims=True)
    rgb = np.repeat(luminance, 3, axis=2)
    rgb = np.clip(0.65 * rgb + 70, 0, 255).astype(np.uint8)
    rgb[mask_bool] = source[mask_bool]
    rgb[mask_bool, 0] = np.maximum(rgb[mask_bool, 0], 220)
    rgb[mask_bool, 1] = (0.65 * rgb[mask_bool, 1]).astype(np.uint8)
    rgb[mask_bool, 2] = (0.65 * rgb[mask_bool, 2]).astype(np.uint8)
    boundary = mask_bool & ~ndi.binary_erosion(mask_bool)
    rgb[boundary] = np.array([0, 210, 230], dtype=np.uint8)
    return rgb


@pytest.mark.parametrize(
    "shape",
    [(1, 1), (1, 7), (7, 1), (2, 5), (5, 2), (12, 17)],
)
def test_mask_boundary_matches_scipy_four_connected_erosion(
    shape: tuple[int, int],
) -> None:
    rng = np.random.default_rng(20260726)
    mask = rng.random(shape) > 0.35

    expected = mask & ~ndi.binary_erosion(mask)

    assert np.array_equal(_mask_boundary(mask), expected)


def test_prepared_mask_overlays_match_previous_rendering_exactly() -> None:
    rng = np.random.default_rng(20260726)
    image = rng.integers(0, 256, size=(51, 67, 3), dtype=np.uint8)
    context = _prepare_mask_overlay(image)

    for threshold in (0.15, 0.35, 0.55, 0.75):
        mask = rng.random(image.shape[:2]) > threshold
        expected = _reference_overlay(image, mask)

        assert np.array_equal(overlay_mask(image, mask), expected)
        assert np.array_equal(
            overlay_mask(image, mask, context=context),
            expected,
        )


def test_overlay_mask_rejects_mismatched_prepared_context() -> None:
    image = np.zeros((12, 16, 3), dtype=np.uint8)
    context = _prepare_mask_overlay(image)

    with pytest.raises(ValueError, match="shapes must match"):
        overlay_mask(image, np.zeros((11, 16), dtype=bool), context=context)
