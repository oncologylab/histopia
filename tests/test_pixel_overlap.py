import numpy as np
import pytest

from histopia.protein._pixel_overlap import native_cell_od_means


def test_fractional_overlap_matches_independent_rectangle_integration():
    labels = np.array([[7, 7, 0, 2, 2], [7, 7, 2, 2, 2], [0, 7, 2, 2, 0]], np.uint32)
    ids = np.array([7, 2], np.uint32)  # Deliberately not sorted.
    areas = np.array([(labels == i).sum() for i in ids])
    od = np.array([[0.2, 0.6, 0.3], [0.9, 0.1, 0.8]])
    tissue = np.array([[True, False, True], [True, True, True]])
    scale = (0.6, 0.8)
    origin = (-0.3, 0.2)
    observed = native_cell_od_means(
        lambda top, n: labels[top : top + n],
        labels.shape,
        ids,
        areas,
        od,
        tissue,
        label_origin_xy=origin,
        measurement_origin_xy=(0, 0),
        native_mpp_xy=scale,
        analysis_mpp=1,
        stripe_height=1,
        minimum_effective_pixels=0.01,
        minimum_coverage=0,
    )
    totals, weights = [], []
    for cell in ids:
        total = weight = 0.0
        for y, x in np.argwhere(labels == cell):
            left, right = (x + origin[0]) * scale[0], (x + origin[0] + 1) * scale[0]
            top, bottom = (y + origin[1]) * scale[1], (y + origin[1] + 1) * scale[1]
            for j, i in np.argwhere(tissue):
                overlap = max(0, min(right, i + 1) - max(left, i)) * max(
                    0, min(bottom, j + 1) - max(top, j)
                )
                weight += overlap
                total += overlap * od[j, i]
        totals.append(total)
        weights.append(weight)
    np.testing.assert_array_equal(observed.label_ids, ids)
    np.testing.assert_allclose(observed.effective_pixels, weights)
    np.testing.assert_allclose(observed.mean_od, np.array(totals) / weights)
    np.testing.assert_allclose(
        observed.coverage, np.array(weights) / (areas * np.prod(scale))
    )


def test_shared_od_pixel_does_not_create_independent_measurements():
    labels = np.array([[1, 1, 2, 2]] * 4, np.uint32)
    result = native_cell_od_means(
        lambda top, n: labels[top : top + n],
        labels.shape,
        np.array([1, 2]),
        np.array([8, 8]),
        np.array([[0.75]]),
        np.array([[True]]),
        label_origin_xy=(0, 0),
        measurement_origin_xy=(0, 0),
        native_mpp_xy=(1, 1),
        analysis_mpp=4,
    )
    np.testing.assert_allclose(result.effective_pixels, [0.5, 0.5])
    np.testing.assert_allclose(result.coverage, [1, 1])
    assert not result.measured.any()
    assert np.isnan(result.mean_od).all()


@pytest.mark.parametrize("error", ["unknown_id", "area", "unsupported"])
def test_binding_and_missing_support(error):
    labels = np.full((4, 4), 3, np.uint32)
    kwargs = dict(
        label_origin_xy=(0, 0),
        measurement_origin_xy=(0, 0),
        native_mpp_xy=(1, 1),
        analysis_mpp=2,
        stripe_height=2,
    )
    ids = np.array([2 if error == "unknown_id" else 3])
    areas = np.array([15 if error == "area" else 16])
    mask = np.full((2, 2), error != "unsupported")
    if error != "unsupported":
        with pytest.raises(ValueError):
            native_cell_od_means(
                lambda t, n: labels[t : t + n],
                labels.shape,
                ids,
                areas,
                np.ones((2, 2)),
                mask,
                **kwargs,
            )
    else:
        result = native_cell_od_means(
            lambda t, n: labels[t : t + n],
            labels.shape,
            ids,
            areas,
            np.full((2, 2), np.nan),
            mask,
            **kwargs,
        )
        assert not result.measured.any()
        assert np.isnan(result.mean_od).all()
        assert result.coverage[0] == 0
