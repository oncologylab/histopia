import numpy as np

from histopia.registration._dense_matching import (
    balanced_correspondences,
    normalized_to_pixel_centres,
)


def test_normalized_grid_is_pixel_centred_with_anisotropic_dimensions():
    points = np.array([[-1 + 1 / 200, -1 + 1 / 100], [1 - 1 / 200, 1 - 1 / 100]])
    np.testing.assert_allclose(
        normalized_to_pixel_centres(points, (100, 200)), [[0, 0], [199, 99]], atol=1e-12
    )


def test_correspondences_exclude_missing_support_and_balance_both_frames():
    a = np.array([[1, 1], [2, 2], [30, 30], [60, 60], [99, 99], [np.nan, 4]])
    b = np.array([[1, 1], [2, 2], [1, 1], [60, 60], [99, 99], [4, 4]])
    mask = np.ones((100, 100), bool)
    mask[99, 99] = False
    aa, bb, confidence = balanced_correspondences(
        a, b, np.arange(6, 0, -1), mask, mask, bins=4, per_bin=1
    )
    np.testing.assert_array_equal(aa, [[1, 1], [60, 60]])
    np.testing.assert_array_equal(bb, [[1, 1], [60, 60]])
    np.testing.assert_array_equal(confidence, [6, 3])
