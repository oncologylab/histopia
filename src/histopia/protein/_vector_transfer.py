"""Outcome-independent cell correspondence and missing-aware protein transfer.

This complements the frozen scalar transfer without changing its numerics.
Correspondences describe uncertain local support, never confirmed cell identity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from histopia.study._manifest import fingerprint


def _ids(values, name: str) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim == 1 and result.size == 0:
        return result.astype("U1")
    if result.ndim != 1 or result.dtype.kind not in "USiu":
        raise ValueError(f"{name} must be a one-dimensional ID array")
    result = result.astype(str)
    if len(set(result)) != len(result) or np.any(result == ""):
        raise ValueError(f"{name} must be unique and nonempty")
    return result


@dataclass(frozen=True)
class CellCorrespondence:
    source_cell_ids: np.ndarray
    target_cell_ids: np.ndarray
    source_indices: np.ndarray  # target rows × neighbors; -1 means unsupported
    weights: np.ndarray
    distances_um: np.ndarray
    confidence: np.ndarray  # correspondence score, not calibrated probability
    fingerprint: str
    coordinate_units: str = "um"

    def __post_init__(self):
        source = _ids(self.source_cell_ids, "source IDs")
        target = _ids(self.target_cell_ids, "target IDs")
        ix = np.asarray(self.source_indices)
        weights = np.asarray(self.weights, float)
        distance = np.asarray(self.distances_um, float)
        confidence = np.asarray(self.confidence, float)
        if (
            ix.ndim != 2
            or len(ix) != len(target)
            or ix.dtype.kind not in "iu"
            or weights.shape != ix.shape
            or distance.shape != ix.shape
            or confidence.shape != (len(target),)
        ):
            raise ValueError("correspondence arrays must align with target IDs")
        if np.any(ix < -1) or np.any(ix >= len(source)):
            raise ValueError("correspondence contains an unknown source cell")
        valid = ix >= 0
        if (
            not np.all(np.isfinite(weights))
            or np.any(weights < 0)
            or np.any(weights[~valid] != 0)
            or not np.all(np.isnan(distance[~valid]))
            or not np.all(np.isfinite(distance[valid]))
            or np.any(distance[valid] < 0)
        ):
            raise ValueError("invalid correspondence support, weights or distance")
        if not np.allclose(weights.sum(axis=1), valid.any(axis=1)):
            raise ValueError("supported correspondence weights must sum to one")
        if (
            not np.all(np.isfinite(confidence))
            or np.any(confidence < 0)
            or np.any(confidence > 1)
            or not self.fingerprint
            or self.coordinate_units != "um"
        ):
            raise ValueError(
                "correspondence requires confidence, provenance and um units"
            )
        for name, array in (
            ("source_cell_ids", source),
            ("target_cell_ids", target),
            ("source_indices", ix),
            ("weights", weights),
            ("distances_um", distance),
            ("confidence", confidence),
        ):
            copied = array.copy()
            copied.flags.writeable = False
            object.__setattr__(self, name, copied)


@dataclass(frozen=True)
class ProteinVectorTransfer:
    correspondence: CellCorrespondence
    protein_ids: tuple[str, ...]
    values: np.ndarray
    support: np.ndarray
    support_count: np.ndarray
    uncertainty: np.ndarray
    fingerprint: str


def match_registered_cells(
    source_cell_ids,
    target_cell_ids,
    source_xyz_um,
    target_xyz_um,
    source_morphology,
    target_morphology,
    *,
    upstream_fingerprints: dict[str, str],
    source_support=None,
    target_support=None,
    neighbors: int = 8,
    candidates: int = 64,
    maximum_distance_um: float = 128,
    xy_bandwidth_um: float = 64,
    z_bandwidth_um: float = 32,
    minimum_cosine_similarity: float = 0,
    morphology_temperature: float = 0.2,
    mode: str = "combined",
    batch_size: int = 256,
) -> CellCorrespondence:
    """Match in a common registered frame, with no target protein argument.

    Combined/spatial controls search XY; the morphology control searches
    source-standardized unit embeddings. All controls retain the same hard
    physical radius and supplied tissue support. Reuse this graph for every
    protein, so missing markers cannot change correspondence selection.
    """
    from scipy.spatial import cKDTree

    sid, tid = _ids(source_cell_ids, "source IDs"), _ids(target_cell_ids, "target IDs")
    sx, tx = np.asarray(source_xyz_um, float), np.asarray(target_xyz_um, float)
    sm, tm = np.asarray(source_morphology, float), np.asarray(target_morphology, float)
    if sx.shape != (len(sid), 3) or tx.shape != (len(tid), 3):
        raise ValueError("coordinates must be aligned (cells, 3) in micrometres")
    if sm.ndim != 2 or tm.shape != (len(tid), sm.shape[1]) or len(sm) != len(sid):
        raise ValueError("morphology must align with cell IDs and feature columns")
    if not sm.shape[1] or not all(np.all(np.isfinite(a)) for a in (sx, tx, sm, tm)):
        raise ValueError("coordinates and morphology must be finite")
    if mode not in {"combined", "spatial", "morphology"}:
        raise ValueError("mode must be combined, spatial or morphology")
    if not upstream_fingerprints or any(not v for v in upstream_fingerprints.values()):
        raise ValueError("upstream fingerprints are required")
    for value in (neighbors, candidates, batch_size):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("neighbor and batch controls must be positive integers")
    if neighbors > candidates or not -1 <= minimum_cosine_similarity <= 1:
        raise ValueError("invalid correspondence controls")
    for value in (
        maximum_distance_um,
        xy_bandwidth_um,
        z_bandwidth_um,
        morphology_temperature,
    ):
        if not np.isfinite(value) or value <= 0:
            raise ValueError("distance and temperature controls must be positive")
    ss = (
        np.ones(len(sid), bool)
        if source_support is None
        else np.asarray(source_support)
    )
    ts = (
        np.ones(len(tid), bool)
        if target_support is None
        else np.asarray(target_support)
    )
    if (
        ss.shape != sid.shape
        or ts.shape != tid.shape
        or ss.dtype != bool
        or ts.dtype != bool
    ):
        raise ValueError("support masks must be aligned boolean arrays")
    index = np.full((len(tid), neighbors), -1, np.int64)
    weights = np.zeros(index.shape, float)
    distance = np.full(index.shape, np.nan)
    confidence = np.zeros(len(tid))
    eligible_source = np.flatnonzero(ss)
    if len(eligible_source):
        center = sm[ss].mean(axis=0)
        scale = sm[ss].std(axis=0)
        scale[scale < 1e-6] = 1
        sk, tk = (sm - center) / scale, (tm - center) / scale
        sk /= np.maximum(np.linalg.norm(sk, axis=1, keepdims=True), 1e-12)
        tk /= np.maximum(np.linalg.norm(tk, axis=1, keepdims=True), 1e-12)
        tree = cKDTree(sk[ss] if mode == "morphology" else sx[ss, :2])
        count = min(candidates, len(eligible_source))
        for start in range(0, len(tid), batch_size):
            stop = min(start + batch_size, len(tid))
            query = tk[start:stop] if mode == "morphology" else tx[start:stop, :2]
            _, nearest = tree.query(query, k=count)
            nearest = eligible_source[np.asarray(nearest).reshape(-1, count)]
            delta = sx[nearest] - tx[start:stop, None]
            d = np.linalg.norm(delta, axis=2)
            similarity = np.einsum("nd,nkd->nk", tk[start:stop], sk[nearest])
            spatial_cost = 0.5 * (
                (delta[..., :2] ** 2).sum(axis=2) / xy_bandwidth_um**2
                + delta[..., 2] ** 2 / z_bandwidth_um**2
            )
            morph_cost = (1 - np.clip(similarity, -1, 1)) / morphology_temperature
            cost = spatial_cost if mode == "spatial" else morph_cost
            if mode == "combined":
                cost = cost + spatial_cost
            valid = (d <= maximum_distance_um) & ts[start:stop, None]
            if mode != "spatial":
                valid &= similarity >= minimum_cosine_similarity
            # IDs must be section-qualified; shared IDs are true self matches.
            valid &= sid[nearest] != tid[start:stop, None]
            cost = np.where(valid, cost, np.inf)
            order = np.argsort(cost, axis=1, kind="stable")[:, :neighbors]
            chosen = np.take_along_axis(nearest, order, axis=1)
            chosen_cost = np.take_along_axis(cost, order, axis=1)
            chosen_distance = np.take_along_axis(d, order, axis=1)
            ok = np.isfinite(chosen_cost)
            quality = np.exp(-np.minimum(chosen_cost, 700)) * ok
            total = quality.sum(axis=1, keepdims=True)
            w = np.divide(quality, total, out=np.zeros_like(quality), where=total > 0)
            width = chosen.shape[1]
            index[start:stop, :width] = np.where(ok, chosen, -1)
            weights[start:stop, :width] = w
            distance[start:stop, :width] = np.where(ok, chosen_distance, np.nan)
            confidence[start:stop] = quality.max(axis=1)
    binding = {
        "algorithm": "registered-vector-correspondence-v1",
        "upstream": upstream_fingerprints,
        "source_ids": sid.tolist(),
        "target_ids": tid.tolist(),
        "mode": mode,
        "controls": [
            neighbors,
            candidates,
            maximum_distance_um,
            xy_bandwidth_um,
            z_bandwidth_um,
            minimum_cosine_similarity,
            morphology_temperature,
        ],
        "array_hashes": [
            _array_hash(a) for a in (sx, tx, sm, tm, ss, ts, index, weights)
        ],
    }
    return CellCorrespondence(
        sid, tid, index, weights, distance, confidence, fingerprint(binding)
    )


def _array_hash(array: np.ndarray) -> str:
    import hashlib

    return hashlib.sha256(
        str((array.shape, array.dtype.str)).encode()
        + np.ascontiguousarray(array).tobytes()
    ).hexdigest()


def transfer_protein_vectors(
    correspondence: CellCorrespondence,
    source_cell_ids,
    protein_ids,
    source_values,
    *,
    source_fingerprint: str,
    source_uncertainty=None,
) -> ProteinVectorTransfer:
    """Weighted vector means with per-marker missing support and total variance.

    Uncertainty is sqrt(weighted between-anchor variance + weighted source
    variance), not a calibrated confidence interval. It must be validated on
    held-out mice. Omitted source uncertainty remains unknown; pass explicit
    zeros only when scientifically justified. Unsupported entries remain NaN.
    """
    sid = _ids(source_cell_ids, "source IDs")
    if not np.array_equal(sid, correspondence.source_cell_ids):
        raise ValueError("source cell ordering differs from correspondence")
    proteins = tuple(_ids(protein_ids, "protein IDs"))
    values = np.asarray(source_values, float)
    if values.shape != (len(sid), len(proteins)) or np.any(np.isinf(values)):
        raise ValueError("source values must align with cell IDs and proteins")
    if np.any(values[np.isfinite(values)] < 0) or not source_fingerprint:
        raise ValueError("protein values must be nonnegative and fingerprinted")
    variance = np.full_like(values, np.nan)
    if source_uncertainty is not None:
        sd = np.asarray(source_uncertainty, float)
        if (
            sd.shape != values.shape
            or np.any(sd[np.isfinite(sd)] < 0)
            or np.any(np.isinf(sd))
        ):
            raise ValueError("source uncertainty must be nonnegative and aligned")
        variance = sd**2
    shape = (len(correspondence.target_cell_ids), len(proteins))
    output, uncertainty = np.full(shape, np.nan), np.full(shape, np.nan)
    counts = np.zeros(shape, np.int32)
    # Bounded batches avoid a whole-slide cells × neighbors × proteins tensor.
    for start in range(0, shape[0], 2048):
        stop = min(start + 2048, shape[0])
        ix = correspondence.source_indices[start:stop]
        if not len(sid):
            break
        sampled = values[np.maximum(ix, 0)]
        valid = (ix[..., None] >= 0) & np.isfinite(sampled)
        w = correspondence.weights[start:stop, :, None] * valid
        total = w.sum(axis=1)
        w = np.divide(w, total[:, None], out=np.zeros_like(w), where=total[:, None] > 0)
        mean = (w * np.where(valid, sampled, 0)).sum(axis=1)
        spread = (w * np.where(valid, (sampled - mean[:, None]) ** 2, 0)).sum(axis=1)
        var = variance[np.maximum(ix, 0)]
        unknown = np.any(valid & ~np.isfinite(var) & (w > 0), axis=1)
        spread += (w * np.where(valid & np.isfinite(var), var, 0)).sum(axis=1)
        supported = total > 0
        output[start:stop] = np.where(supported, mean, np.nan)
        uncertainty[start:stop] = np.where(
            supported & ~unknown, np.sqrt(spread), np.nan
        )
        counts[start:stop] = (valid & (w > 0)).sum(axis=1)
    binding = fingerprint(
        {
            "correspondence": correspondence.fingerprint,
            "source": source_fingerprint,
            "proteins": proteins,
            "values": _array_hash(values),
            "variance": _array_hash(variance),
        }
    )
    return ProteinVectorTransfer(
        correspondence, proteins, output, counts > 0, counts, uncertainty, binding
    )
