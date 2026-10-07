"""Graph composition in explicitly calibrated physical coordinates.

Graph closure is an engineering diagnostic. It is not long-range anatomical TRE
and cannot establish physical section order or measured section spacing.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StackEdge:
    source: str
    target: str
    matrix: np.ndarray
    confidence: float
    source_probes_xy: np.ndarray


def compose_stack_paths(image_ids, reference, edges):
    """Compose paths into the reference; disconnected planes stay None."""
    ids = set(image_ids)
    if reference not in ids:
        raise ValueError("reference is not a stack member")
    graph = {i: [] for i in ids}
    for edge in edges:
        h = np.asarray(edge.matrix, float)
        if edge.source not in ids or edge.target not in ids:
            raise ValueError("edge crosses the declared reconstruction")
        if (
            h.shape != (3, 3)
            or not np.isfinite(h).all()
            or not np.allclose(h[2], [0, 0, 1])
            or np.linalg.det(h[:2, :2]) <= 0
        ):
            raise ValueError("finite orientation-preserving affine edge required")
        if not np.isfinite(edge.confidence) or not 0 < edge.confidence <= 1:
            raise ValueError("edge confidence must be in (0, 1]")
        cost = -np.log(edge.confidence) + 0.01
        graph[edge.target].append((edge.source, h, cost))
        graph[edge.source].append((edge.target, np.linalg.inv(h), cost))
    distance = {reference: 0.0}
    transforms = {i: None for i in ids}
    transforms[reference] = np.eye(3)
    queue = [(0.0, reference)]
    paths = {reference: [reference]}
    while queue:
        value, current = heapq.heappop(queue)
        if value > distance[current]:
            continue
        for neighbor, neighbor_to_current, cost in graph[current]:
            new = value + cost
            if new < distance.get(neighbor, float("inf")):
                distance[neighbor] = new
                transforms[neighbor] = transforms[current] @ neighbor_to_current
                paths[neighbor] = [neighbor, *paths[current]]
                heapq.heappush(queue, (new, neighbor))
    return transforms, paths


def optimize_stack_corrections(
    initial, reference, edges, *, max_rotation_degrees=15.0, max_translation_um=500.0
):
    """Robust bounded pose correction; preserve each plane's shape and dimensions.

    Corrections are rotations/translations in the reference physical frame.
    They do not remove any pre-existing affine deformation. Input probes measure
    edge consistency only and must never be described as independent landmarks.
    """
    from scipy.optimize import least_squares

    if initial.get(reference) is None:
        raise ValueError("reference pose missing")
    ids = sorted(i for i, h in initial.items() if h is not None and i != reference)
    supported = [
        e
        for e in edges
        if initial.get(e.source) is not None and initial.get(e.target) is not None
    ]
    if not ids or not supported:
        return dict(initial), {
            "status": "no_supported_optimization",
            "independent_accuracy": False,
        }
    centres = {}
    for i in ids:
        points = [np.asarray(e.source_probes_xy) for e in supported if e.source == i]
        centre = np.concatenate(points).mean(0) if points else np.zeros(2)
        centres[i] = (initial[i] @ [*centre, 1])[:2]

    def poses(parameters):
        out = dict(initial)
        for i, (angle, dx, dy) in zip(ids, parameters.reshape(-1, 3), strict=True):
            rotation = np.array(
                [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
            )
            correction = np.eye(3)
            correction[:2, :2] = rotation
            correction[:2, 2] = centres[i] - rotation @ centres[i] + [dx, dy]
            out[i] = correction @ initial[i]
        return out

    def residual(parameters):
        output = poses(parameters)
        values = []
        for e in supported:
            points = np.asarray(e.source_probes_xy, float)
            if (
                points.ndim != 2
                or points.shape[1] != 2
                or not len(points)
                or not np.isfinite(points).all()
            ):
                raise ValueError("finite physical consistency probes required")
            p = np.c_[points, np.ones(len(points))]
            discrepancy = (p @ (output[e.source] - output[e.target] @ e.matrix).T)[
                :, :2
            ]
            values.extend((discrepancy * np.sqrt(e.confidence / len(points))).ravel())
        return np.asarray(values)

    bounds = np.tile(
        [np.deg2rad(max_rotation_degrees), max_translation_um, max_translation_um],
        len(ids),
    )
    start = np.zeros(3 * len(ids))
    fit = least_squares(
        residual,
        start,
        bounds=(-bounds, bounds),
        loss="soft_l1",
        f_scale=64.0,
        max_nfev=200,
    )
    applied = fit.x if fit.success else start
    return poses(applied), {
        "status": "complete" if fit.success else "optimizer_limit",
        "optimizer_message": fit.message,
        "initial_rms_consistency_um": float(np.sqrt(np.mean(residual(start) ** 2))),
        "final_rms_consistency_um": float(np.sqrt(np.mean(residual(applied) ** 2))),
        "candidate_rms_consistency_um": float(np.sqrt(np.mean(residual(fit.x) ** 2))),
        "corrections_applied": bool(fit.success),
        "independent_accuracy": False,
        "corrections": {
            i: applied.reshape(-1, 3)[j].tolist() for j, i in enumerate(ids)
        },
    }
