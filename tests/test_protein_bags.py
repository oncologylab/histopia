import numpy as np
import pytest

from histopia.protein._bags import mean_cell_vectors, sample_cell_bags


def test_sampling_preserves_membership_and_never_repeats_cells():
    membership = np.array([2, 0, 2, -1, 1, 2, 0, 1, 2])
    eligible = np.array([True, False, True, True])
    chosen, indices, counts = sample_cell_bags(
        membership, eligible, maximum_bags=3, maximum_cells=3, seed=8
    )
    np.testing.assert_array_equal(chosen, [0, 2])
    np.testing.assert_array_equal(counts, [2, 4])
    for bag, ix in zip(chosen, indices, strict=True):
        valid = ix[ix >= 0]
        assert len(np.unique(valid)) == len(valid)
        assert np.all(membership[valid] == bag)
    assert indices[0, -1] == -1
    for a, b in zip(
        (chosen, indices, counts),
        sample_cell_bags(membership, eligible, maximum_bags=3, maximum_cells=3, seed=8),
        strict=True,
    ):
        np.testing.assert_array_equal(a, b)


def test_means_respect_cell_order_and_per_marker_missing_support():
    membership = np.array([1, -1, 0, 1])
    values = np.array([[2, np.nan], [100, 100], [4, 8], [6, 3]])
    means, counts = mean_cell_vectors(membership, values, bag_count=3)
    np.testing.assert_allclose(
        means, [[4, 8], [4, 3], [np.nan, np.nan]], equal_nan=True
    )
    np.testing.assert_array_equal(counts, [[1, 1], [2, 1], [0, 0]])
    order = np.array([3, 0, 2, 1])
    permuted, _ = mean_cell_vectors(membership[order], values[order], bag_count=3)
    np.testing.assert_allclose(permuted, means, equal_nan=True)


def test_invalid_membership_and_empty_support():
    with pytest.raises(ValueError, match="inventory"):
        sample_cell_bags(
            np.array([3]), np.array([True]), maximum_bags=1, maximum_cells=1, seed=0
        )
    with pytest.raises(ValueError, match="NaN"):
        mean_cell_vectors(np.array([0]), np.array([[np.inf]]), bag_count=1)
    chosen, indices, counts = sample_cell_bags(
        np.array([-1]), np.array([True]), maximum_bags=1, maximum_cells=3, seed=0
    )
    assert len(chosen) == len(counts) == 0
    assert indices.shape == (0, 3)


def test_unsigned_membership_preserves_valid_ids_and_rejects_overflow():
    membership = np.array([1, 0, 1], dtype=np.uint64)
    means, counts = mean_cell_vectors(
        membership, np.array([[2.0], [5.0], [4.0]]), bag_count=2
    )
    np.testing.assert_allclose(means, [[5], [3]])
    np.testing.assert_array_equal(counts, [[1], [2]])
    chosen, indices, sizes = sample_cell_bags(
        membership, np.array([True, True]), maximum_bags=2, maximum_cells=2, seed=0
    )
    np.testing.assert_array_equal(chosen, [0, 1])
    np.testing.assert_array_equal(indices, [[1, -1], [0, 2]])
    np.testing.assert_array_equal(sizes, [1, 2])
    with pytest.raises(ValueError, match="inventory"):
        mean_cell_vectors(
            np.array([2**64 - 1], dtype=np.uint64), np.array([[1.0]]), bag_count=2
        )
