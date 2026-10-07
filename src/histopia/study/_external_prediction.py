"""Donor-weighted ridge models and donor-level staining benchmark metrics.

Callers must provide training data only to fit. Scalers are part of the returned
model; prediction never estimates normalization from the evaluated cohort.
"""

from __future__ import annotations

import numpy as np


def donor_weights(donors: np.ndarray) -> np.ndarray:
    """Give each donor equal total weight and keep mean patch weight at one."""
    _, inverse, counts = np.unique(donors, return_inverse=True, return_counts=True)
    if not len(inverse):
        raise ValueError("training donors cannot be empty")
    return len(inverse) / (len(counts) * counts[inverse])


def fit_ridge_path(
    x: np.ndarray, y: np.ndarray, donors: np.ndarray, alphas: list[float]
) -> list[dict]:
    """Fit log1p staining targets with training-only weighted standardization."""
    from scipy.linalg import eigh

    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.shape != (len(x),) or len(donors) != len(x):
        raise ValueError("features, targets and donors must have matching rows")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(y < 0):
        raise ValueError("training inputs must be finite with nonnegative targets")
    if not alphas or any(a <= 0 or not np.isfinite(a) for a in alphas):
        raise ValueError("ridge strengths must be finite and positive")
    weights = donor_weights(donors)
    center = np.average(x, axis=0, weights=weights)
    scale = np.sqrt(np.average((x - center) ** 2, axis=0, weights=weights))
    scale[scale < 1e-6] = 1.0
    z = (x - center) / scale
    target = np.log1p(y)
    intercept = float(np.average(target, weights=weights))
    gram = (z.T * weights) @ z
    rhs = (z.T * weights) @ (target - intercept)
    values, vectors = eigh(gram, check_finite=False)
    projection = vectors.T @ rhs
    return [
        dict(
            center=center,
            scale=scale,
            coefficients=vectors @ (projection / (np.maximum(values, 0) + alpha)),
            intercept=np.array(intercept),
            alpha=np.array(alpha),
        )
        for alpha in alphas
    ]


def predict_ridge(model: dict, x: np.ndarray) -> np.ndarray:
    """Apply frozen parameters, preserving the input row order."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(model["center"]) or not np.isfinite(x).all():
        raise ValueError("prediction features do not match the frozen model")
    log = ((x - model["center"]) / model["scale"]) @ model["coefficients"] + model[
        "intercept"
    ]
    # Floating point overflow is an error, not a silently clipped success.
    with np.errstate(over="raise", invalid="raise"):
        return np.maximum(np.expm1(log), 0)


def donor_metrics(y: np.ndarray, predicted: np.ndarray, donors: np.ndarray) -> dict:
    """Report equal-donor errors and within-donor patch correlations separately."""
    y, predicted = np.asarray(y), np.asarray(predicted)
    if y.shape != predicted.shape or y.ndim != 1 or len(donors) != len(y):
        raise ValueError("metrics require aligned rows")
    if not np.isfinite(y).all() or not np.isfinite(predicted).all() or not len(y):
        raise ValueError("metrics require finite supported predictions")
    rows = []
    for donor in np.unique(donors):
        mask = donors == donor
        actual, estimate = y[mask], predicted[mask]
        corr = (
            float(np.corrcoef(actual, estimate)[0, 1])
            if (len(actual) >= 3 and actual.std() > 1e-8 and estimate.std() > 1e-8)
            else None
        )
        rows.append(
            dict(
                donor=str(donor),
                patches=int(mask.sum()),
                mae=float(np.mean(np.abs(actual - estimate))),
                rmse=float(np.sqrt(np.mean((actual - estimate) ** 2))),
                bias=float(np.mean(estimate - actual)),
                observed_mean=float(actual.mean()),
                predicted_mean=float(estimate.mean()),
                within_donor_patch_r=corr,
            )
        )
    return dict(
        donors=len(rows),
        patches=len(y),
        donor_mean_mae=float(np.mean([r["mae"] for r in rows])),
        donor_mean_rmse=float(np.mean([r["rmse"] for r in rows])),
        donor_mean_bias=float(np.mean([r["bias"] for r in rows])),
        rows=rows,
    )
