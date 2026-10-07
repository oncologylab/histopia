import numpy as np
import pytest

from histopia.study._neighborhood_graph import (
    NeighborhoodReducer,
    cluster_profiles,
    physical_radius_graph,
    settle_profile_groups,
)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_physical_graph_support_and_joint_profile_permutation(device):
    if device != "cpu":
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
    xy = np.array([[0, 0], [3, 4], [4, 3], [6, 8], [1, 1], [2, 2]], float)
    tissue = np.array([1, 1, 1, 1, 2, 0])
    centers = np.array([0, 1, 4, 5])
    graph = physical_radius_graph(xy, tissue, centers, 5)
    expected = np.zeros((4, 6), np.float32)
    for row, center in enumerate(centers):
        for cell in range(len(xy)):
            if (
                cell != center
                and tissue[center] > 0
                and tissue[cell] == tissue[center]
                and np.linalg.norm(xy[cell] - xy[center]) <= 5
            ):
                expected[row, cell] = 1
    np.testing.assert_array_equal(graph.toarray(), expected)
    values = np.array(
        [[1, 10], [2, np.nan], [4, 40], [8, 80], [100, 1000], [200, 2000]]
    )
    support = np.isfinite(values)
    support[0, 1] = False  # finite-but-unsupported must also be omitted
    reducer = NeighborhoodReducer(graph, values, support, minimum=1, device=device)
    for order in [None, np.array([3, 2, 0, 1, 4, 5])]:
        for keep in [None, np.array([True, False, True, True, True, True])]:
            result, counts = reducer.summarize(order=order, keep=keep)
            rows = values if order is None else values[order]
            valid = support if order is None else support[order]
            valid = valid & np.isfinite(rows)
            if keep is not None:
                valid &= keep[:, None]
            for i in range(len(centers)):
                for j in range(2):
                    chosen = rows[(expected[i] > 0) & valid[:, j], j]
                    assert counts[i, j] == len(chosen)
                    if len(chosen):
                        assert result[i, j] == pytest.approx(chosen.mean())
                    else:
                        assert np.isnan(result[i, j])


def test_missing_support_and_invalid_geometry_fail_explicitly():
    graph = physical_radius_graph([[0, 0], [1, 0]], [1, 1], [0], 2)
    reducer = NeighborhoodReducer(graph, [[1], [2]], [[True], [False]])
    mean, count = reducer.summarize()
    assert np.isnan(mean).all() and count[0, 0] == 0
    with pytest.raises(ValueError, match="permutation"):
        reducer.summarize(order=[0, 0])
    with pytest.raises(ValueError, match="sampling"):
        reducer.summarize(keep=[1, 0])
    with pytest.raises(ValueError, match="unique center"):
        physical_radius_graph([[0, 0]], [1], [0, 0], 1)
    with pytest.raises(ValueError, match="physical XY"):
        physical_radius_graph([[np.nan, 0]], [1], [0], 1)
    with pytest.raises(ValueError, match="positive physical"):
        physical_radius_graph([[0, 0]], [1], [0], -1)


def test_cluster_reference_cuda_and_degenerate_profiles():
    array = np.array([[0, 0], [0.01, 0.02], [10, 10], [10.01, 10.02]])
    labels, centers = cluster_profiles(array, clusters=2, seed=3)
    assert labels[0] == labels[1] and labels[2] == labels[3]
    assert labels[0] != labels[2]
    same, centroid = cluster_profiles(np.ones((10, 4)), clusters=6)
    assert np.all(same == 0) and centroid.shape == (1, 4)
    with pytest.raises(ValueError, match="complete finite"):
        cluster_profiles([[np.nan, 0]])
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        gpu_labels, gpu_centers = cluster_profiles(
            array, clusters=2, seed=3, device="cuda:0"
        )
        np.testing.assert_array_equal(gpu_labels, labels)
        np.testing.assert_allclose(gpu_centers, centers, rtol=1e-10, atol=1e-10)


def test_saved_fit_continues_until_labels_are_stationary():
    x = np.random.default_rng(120).normal(size=(300, 4))
    initial, _ = cluster_profiles(x, clusters=6, seed=8, iterations=1)
    frozen = initial.copy()
    labels, centers, detail = settle_profile_groups(x, initial)
    np.testing.assert_array_equal(initial, frozen)
    assert detail["stationary"] and detail["iterations"] > 1
    for group in range(len(centers)):
        np.testing.assert_allclose(centers[group], x[labels == group].mean(axis=0))
    np.testing.assert_array_equal(
        labels, ((x[:, None] - centers[None]) ** 2).sum(axis=2).argmin(axis=1)
    )
    with pytest.raises(ValueError, match="did not settle"):
        settle_profile_groups(x, initial, maximum_iterations=1)
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        gpu_labels, gpu_centers, gpu_detail = settle_profile_groups(
            x, initial, device="cuda:0"
        )
        np.testing.assert_array_equal(gpu_labels, labels)
        np.testing.assert_allclose(gpu_centers, centers)
        assert gpu_detail["stationary"]
