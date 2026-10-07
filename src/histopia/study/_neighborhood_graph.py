"""Sparse physical neighborhoods and optional CUDA profile aggregation.

Graphs are built from geometry alone. Joint permutations move complete feature
profiles together with their support masks; they never move coordinates.
"""

from __future__ import annotations

import numpy as np


def physical_radius_graph(xy_um, tissue_components, centers, radius_um: float):
    """Return a center-by-cell CSR graph, excluding each center itself.

    Components must identify connected tissue within a single physical section.
    Zero denotes unsupported tissue. XY is in physical micrometers, including
    distinct calibrated scanner X/Y pixel sizes when applicable.
    """
    from scipy.sparse import csr_matrix
    from scipy.spatial import cKDTree

    xy = np.asarray(xy_um, dtype=float)
    groups = np.asarray(tissue_components)
    selected = np.asarray(centers)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.all(np.isfinite(xy)):
        raise ValueError("finite physical XY coordinates are required")
    if groups.shape != (len(xy),) or groups.dtype.kind not in "iu":
        raise ValueError("integer tissue components must align with cells")
    if np.any(groups < 0):
        raise ValueError("tissue components must be nonnegative")
    if (
        selected.ndim != 1
        or selected.dtype.kind not in "iu"
        or np.any(selected < 0)
        or np.any(selected >= len(xy))
        or len(np.unique(selected)) != len(selected)
    ):
        raise ValueError("unique center indices must refer to input cells")
    if not np.isfinite(radius_um) or radius_um <= 0:
        raise ValueError("a positive physical radius is required")
    neighbors = [np.empty(0, np.int64) for _ in selected]
    for component in np.unique(groups[selected]):
        if component == 0:
            continue
        indices = np.flatnonzero(groups == component)
        rows = np.flatnonzero(groups[selected] == component)
        tree = cKDTree(xy[indices])
        found = tree.query_ball_point(
            xy[selected[rows]], radius_um, workers=2, return_sorted=True
        )
        for row, local in zip(rows, found, strict=True):
            ids = indices[np.asarray(local, np.int64)]
            neighbors[row] = ids[ids != selected[row]]
    counts = np.array([len(row) for row in neighbors], np.int64)
    indptr = np.r_[0, np.cumsum(counts)]
    indices = np.concatenate(neighbors) if neighbors else np.empty(0, np.int64)
    return csr_matrix(
        (np.ones(len(indices), np.float32), indices, indptr),
        shape=(len(selected), len(xy)),
    )


class NeighborhoodReducer:
    """Cache one graph and its supported profiles on CPU or an optional GPU."""

    def __init__(self, graph, values, support, *, minimum=5, device="cpu"):
        from scipy.sparse import csr_matrix

        self.graph = csr_matrix(graph, copy=True)
        self.graph.sum_duplicates()
        self.graph.sort_indices()
        array = np.asarray(values, dtype=np.float32)
        valid = np.asarray(support)
        if (
            array.ndim != 2
            or array.shape[0] != self.graph.shape[1]
            or valid.shape != array.shape
            or valid.dtype != bool
            or np.any(np.isinf(array))
        ):
            raise ValueError(
                "finite-or-missing profiles and boolean support must align"
            )
        if type(minimum) is not int or minimum < 1:
            raise ValueError("minimum support must be a positive integer")
        if not np.all(self.graph.data == 1):
            raise ValueError("the graph must contain unique unweighted neighbor edges")
        if device != "cpu" and not str(device).startswith("cuda"):
            raise ValueError("device must be cpu or cuda")
        self.minimum = minimum
        self.device = device
        valid = valid & np.isfinite(array)
        self.values = np.where(valid, array, 0)
        self.valid = valid.astype(np.float32)
        if device != "cpu":
            import torch

            self._torch = torch
            self._graph = torch.sparse_csr_tensor(
                torch.as_tensor(self.graph.indptr, dtype=torch.int64, device=device),
                torch.as_tensor(self.graph.indices, dtype=torch.int64, device=device),
                torch.as_tensor(self.graph.data, dtype=torch.float32, device=device),
                size=self.graph.shape,
                device=device,
                check_invariants=True,
            )
            self._values = torch.as_tensor(self.values, device=device)
            self._valid = torch.as_tensor(self.valid, device=device)

    def summarize(self, *, order=None, keep=None):
        """Return means and support counts; unsupported means remain NaN.

        ``order`` must be a complete permutation. The caller restricts it to
        declared tissue strata. ``keep`` subsamples cells without changing the
        graph. The same ordering and sampling always apply to every feature.
        """
        n = len(self.values)
        if order is not None:
            order = np.asarray(order)
            if (
                order.shape != (n,)
                or order.dtype.kind not in "iu"
                or np.any(order < 0)
                or np.any(order >= n)
                or np.any(np.bincount(order.astype(np.int64), minlength=n) != 1)
            ):
                raise ValueError("order must be a complete profile permutation")
        if keep is not None:
            keep = np.asarray(keep)
            if keep.shape != (n,) or keep.dtype != bool:
                raise ValueError("sampling support must be an aligned boolean mask")
        if self.device == "cpu":
            values, valid = self.values, self.valid
            if order is not None:
                values, valid = values[order], valid[order]
            if keep is not None:
                values, valid = values * keep[:, None], valid * keep[:, None]
            count = self.graph @ valid
            total = self.graph @ values
            mean = np.divide(
                total,
                count,
                out=np.full(total.shape, np.nan, np.float32),
                where=count >= self.minimum,
            )
            return mean, count.astype(np.int64)
        torch = self._torch
        values, valid = self._values, self._valid
        if order is not None:
            index = torch.as_tensor(order.astype(np.int64), device=self.device)
            values, valid = values[index], valid[index]
        if keep is not None:
            mask = torch.as_tensor(keep[:, None], device=self.device)
            values, valid = values * mask, valid * mask
        count = torch.sparse.mm(self._graph, valid)
        total = torch.sparse.mm(self._graph, values)
        mean = torch.where(count >= self.minimum, total / count, float("nan"))
        return mean.cpu().numpy(), count.to(torch.int64).cpu().numpy()


def cluster_profiles(values, *, clusters=6, seed=0, iterations=60, device="cpu"):
    """Seeded Lloyd clustering of complete, pre-standardized profiles.

    This defines exploratory profile groups, never cell types or validated
    biological niches. Fewer distinct profiles yield fewer effective groups.
    No measured outcomes or specimen identifiers enter the fit.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or not len(array) or not np.all(np.isfinite(array)):
        raise ValueError("complete finite profiles are required")
    if type(clusters) is not int or clusters < 1 or iterations < 1:
        raise ValueError("positive cluster and iteration counts are required")
    unique = np.unique(array, axis=0)
    k = min(clusters, len(unique))
    initial = unique[np.random.default_rng(seed).choice(len(unique), k, replace=False)]
    if device == "cpu":
        centroids = initial.copy()
        for _ in range(iterations):
            distances = np.sum((array[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
            labels = distances.argmin(axis=1)
            updated = centroids.copy()
            for group in range(k):
                member = labels == group
                if member.any():
                    updated[group] = array[member].mean(axis=0)
                else:
                    farthest = distances.min(axis=1).argmax()
                    updated[group] = array[farthest]
                    distances[farthest] = 0
            delta = np.max(np.abs(updated - centroids))
            centroids = updated
            if delta <= 1e-7:
                break
        labels = np.sum(
            (array[:, None, :] - centroids[None, :, :]) ** 2, axis=2
        ).argmin(axis=1)
        return labels, centroids
    if not str(device).startswith("cuda"):
        raise ValueError("device must be cpu or cuda")
    import torch

    tensor = torch.as_tensor(array, device=device)
    centroids = torch.as_tensor(initial, device=device)
    for _ in range(iterations):
        distances = ((tensor[:, None, :] - centroids[None, :, :]) ** 2).sum(dim=2)
        labels = distances.argmin(dim=1)
        counts = torch.bincount(labels, minlength=k)
        updated = torch.zeros_like(centroids)
        updated.index_add_(0, labels, tensor)
        updated /= counts.clamp(min=1)[:, None]
        for group in torch.nonzero(counts == 0).flatten().tolist():
            farthest = distances.min(dim=1).values.argmax()
            updated[group] = tensor[farthest]
            distances[farthest] = 0
        delta = (updated - centroids).abs().max().item()
        centroids = updated
        if delta <= 1e-7:
            break
    labels = (
        ((tensor[:, None, :] - centroids[None, :, :]) ** 2).sum(dim=2).argmin(dim=1)
    )
    return labels.cpu().numpy(), centroids.cpu().numpy()


def settle_profile_groups(
    values, initial_labels, *, maximum_iterations=1000, device="cpu"
):
    """Continue a saved fit until a centroid update leaves assignments unchanged.

    Inputs are complete profiles and their previous labels. Empty groups are
    discarded, never assigned invented observations. The iteration budget is a
    failure bound, not a convergence claim. Previous results remain immutable.
    """
    array = np.asarray(values, dtype=np.float64)
    initial = np.asarray(initial_labels)
    if (
        array.ndim != 2
        or not len(array)
        or not np.all(np.isfinite(array))
        or initial.shape != (len(array),)
        or initial.dtype.kind not in "iu"
        or np.any(initial < 0)
    ):
        raise ValueError("finite profiles and nonnegative group labels must align")
    if type(maximum_iterations) is not int or maximum_iterations < 1:
        raise ValueError("a positive convergence budget is required")
    original, labels = np.unique(initial, return_inverse=True)
    initial_groups = len(original)
    if device == "cpu":
        for iteration in range(1, maximum_iterations + 1):
            _, labels = np.unique(labels, return_inverse=True)
            centers = np.stack(
                [array[labels == j].mean(axis=0) for j in range(labels.max() + 1)]
            )
            updated = ((array[:, None] - centers[None]) ** 2).sum(axis=2).argmin(axis=1)
            if np.array_equal(updated, labels):
                return (
                    labels,
                    centers,
                    dict(
                        iterations=iteration,
                        stationary=True,
                        initial_groups=initial_groups,
                        effective_groups=len(centers),
                    ),
                )
            labels = updated
        raise ValueError(
            "profile assignments did not settle within the convergence budget"
        )
    if not str(device).startswith("cuda"):
        raise ValueError("device must be cpu or cuda")
    import torch

    tensor = torch.as_tensor(array, device=device)
    labels = torch.as_tensor(labels, dtype=torch.int64, device=device)
    for iteration in range(1, maximum_iterations + 1):
        _, labels = torch.unique(labels, sorted=True, return_inverse=True)
        counts = torch.bincount(labels)
        centers = torch.zeros(
            (len(counts), array.shape[1]), dtype=tensor.dtype, device=device
        )
        centers.index_add_(0, labels, tensor)
        centers /= counts[:, None]
        updated = ((tensor[:, None] - centers[None]) ** 2).sum(dim=2).argmin(dim=1)
        if torch.equal(updated, labels):
            return (
                labels.cpu().numpy(),
                centers.cpu().numpy(),
                dict(
                    iterations=iteration,
                    stationary=True,
                    initial_groups=initial_groups,
                    effective_groups=len(centers),
                ),
            )
        labels = updated
    raise ValueError("profile assignments did not settle within the convergence budget")
