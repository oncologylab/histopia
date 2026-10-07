"""Mouse-level protein-vector evaluation without changing protein promotion gates."""

from __future__ import annotations

import numpy as np

from histopia.protein._vector_transfer import _ids
from histopia.study._manifest import validate_fit_scope


def _metrics(measured, predicted):
    measured = np.asarray(measured, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if measured.shape != predicted.shape or measured.ndim != 1:
        raise ValueError("metric inputs must be aligned one-dimensional arrays")
    if not np.all(np.isfinite(measured)) or not np.all(np.isfinite(predicted)):
        raise ValueError("metric inputs must contain only supported finite values")
    if not len(measured):
        return {"n": 0, "mae": None, "rmse": None, "pearson_r": None}
    delta = predicted - measured
    r = None
    if len(measured) > 1 and np.ptp(measured) > 0 and np.ptp(predicted) > 0:
        r = float(np.corrcoef(measured, predicted)[0, 1])
    return {
        "n": len(measured),
        "mae": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta**2))),
        "pearson_r": r,
    }


def evaluate_protein_vectors(
    study,
    cell_ids,
    mouse_ids,
    section_ids,
    xy_um,
    protein_ids,
    measured,
    predicted,
    support,
    *,
    fitting_mice_by_holdout: dict[str, list[str]],
    protocol: str,
    measurement_mpp: float,
    aggregate_um: float = 64,
) -> dict[str, object]:
    """Report each held-out mouse separately, with equal-weight 64-µm bins.

    Bin comparison uses identical supported cells for observed and predicted
    means. Coverage reports the remaining measured cells separately. Keep
    direct H&E, transferred H&E and correspondence controls in separate calls;
    these diagnostics never constitute a protein promotion decision.
    """
    ids, proteins = _ids(cell_ids, "cell IDs"), _ids(protein_ids, "protein IDs")
    mice, sections = np.asarray(mouse_ids, str), np.asarray(section_ids, str)
    xy = np.asarray(xy_um, float)
    obs, pred, valid = (
        np.asarray(measured, float),
        np.asarray(predicted, float),
        np.asarray(support),
    )
    if (
        mice.shape != ids.shape
        or sections.shape != ids.shape
        or xy.shape != (len(ids), 2)
    ):
        raise ValueError("evaluation cell coordinates and groups must align")
    if (
        obs.shape != (len(ids), len(proteins))
        or pred.shape != obs.shape
        or valid.shape != obs.shape
        or valid.dtype != bool
    ):
        raise ValueError("protein values and boolean support must align")
    if not np.all(np.isfinite(xy)) or np.any(np.isinf(obs)) or np.any(np.isinf(pred)):
        raise ValueError(
            "evaluation coordinates must be finite; missing outcomes use NaN"
        )
    if (
        not np.isfinite(aggregate_um)
        or aggregate_um <= 0
        or not np.isfinite(measurement_mpp)
        or measurement_mpp <= 0
    ):
        raise ValueError("measurement and aggregation resolution must be explicit")
    if protocol not in {
        "direct_he_prediction",
        "he_prediction_then_section_transfer",
        "spatial_control",
        "morphology_control",
    }:
        raise ValueError(
            "direct prediction and transfer require separate named protocols"
        )
    rows, spatial = [], []
    bins = np.floor(xy / aggregate_um).astype(np.int64)
    for mouse in sorted(set(mice)):
        validate_fit_scope(
            study, fitting_mice_by_holdout.get(mouse, []), evaluation_mice=[mouse]
        )
        for k, protein in enumerate(proteins):
            observed = (mice == mouse) & np.isfinite(obs[:, k])
            keep = observed & valid[:, k] & np.isfinite(pred[:, k])
            groups = {}
            for i in np.flatnonzero(keep):
                groups.setdefault((sections[i], *bins[i]), []).append(i)
            om, pm = [], []
            for (section, bx, by), indices in groups.items():
                o, p = float(obs[indices, k].mean()), float(pred[indices, k].mean())
                om.append(o)
                pm.append(p)
                spatial.append(
                    {
                        "mouse_id": mouse,
                        "section_id": section,
                        "protein_id": protein,
                        "x_um": int(bx) * aggregate_um,
                        "y_um": int(by) * aggregate_um,
                        "supported_cells": len(indices),
                        "measured_mean": o,
                        "predicted_mean": p,
                        "signed_error": p - o,
                    }
                )
            rows.append(
                {
                    "mouse_id": mouse,
                    "protein_id": protein,
                    "measured_cells": int(observed.sum()),
                    "supported_coverage": float(keep.sum() / observed.sum())
                    if observed.any()
                    else None,
                    "supported_cell_metrics": _metrics(obs[keep, k], pred[keep, k]),
                    "aggregate_metrics": _metrics(np.asarray(om), np.asarray(pm)),
                }
            )
    return {
        "protocol": protocol,
        "measurement_mpp": measurement_mpp,
        "aggregate_um": aggregate_um,
        "mouse_metrics": rows,
        "spatial_errors": spatial,
        "promotion_status": (
            "not_assessed; existing protein guardrails and native QC required"
        ),
    }
