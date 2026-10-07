"""Small, explicit contracts for landmark registration benchmarks.

Coordinates are (x, y) pixel centres. Matrices map source to target, never the
pull maps used by image resamplers. Image geometry is supplied by the caller;
no pixel calibration, crop offset, orientation, or section order is inferred.
This module imports no image reader or registration engine.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a homogeneous forward mapping to N two-dimensional points."""
    xy = np.asarray(points, dtype=float)
    matrix = np.asarray(matrix, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or matrix.shape != (3, 3):
        raise ValueError("Expected N by 2 points and a 3 by 3 matrix")
    if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12:
        raise ValueError("Mapping must be finite and invertible")
    out = np.c_[xy, np.ones(len(xy))] @ matrix.T
    with np.errstate(divide="ignore", invalid="ignore"):
        return out[:, :2] / out[:, 2, None]


def pixel_geometry(
    downsample_xy: tuple[float, float] = (1.0, 1.0),
    crop_xy: tuple[float, float] = (0.0, 0.0),
    mpp_xy: tuple[float, float] | None = None,
    orientation: np.ndarray | None = None,
) -> np.ndarray:
    """Map resized pixel centres through a crop into native or physical space.

    ``orientation`` is an explicit cropped-native to full-native mapping after
    the crop translation. Its translation must encode the chosen rotation or
    reflection origin. Positive downsampling and calibration are mandatory.
    """
    scale = np.asarray(downsample_xy, dtype=float)
    offset = np.asarray(crop_xy, dtype=float)
    if scale.shape != (2,) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Downsampling must contain two positive finite values")
    if offset.shape != (2,) or not np.isfinite(offset).all():
        raise ValueError("Crop origin must contain two finite values")
    result = np.eye(3)
    result[0, 0], result[1, 1] = scale
    result[:2, 2] = offset + (scale - 1) / 2
    if orientation is not None:
        result = np.asarray(orientation, dtype=float) @ result
    if mpp_xy is not None:
        mpp = np.asarray(mpp_xy, dtype=float)
        if mpp.shape != (2,) or not np.isfinite(mpp).all() or np.any(mpp <= 0):
            raise ValueError("Calibration must contain two positive finite values")
        result = np.diag([*mpp, 1.0]) @ result
    transform_points(np.empty((0, 2)), result)  # validate orientation as well
    return result


def mean_annotator_tre(predicted: np.ndarray, annotations: np.ndarray) -> np.ndarray:
    """ACROBAT-style point errors: average distance to each reviewer position.

    ``annotations`` has shape (points, reviewers, 2), in the same units as
    ``predicted``. Missing-prediction fallback and image-bound clipping belong
    to the challenge input adapter, before this calculation. The mean of
    distances is intentionally different from distance to the mean position.
    """
    p, a = np.asarray(predicted, float), np.asarray(annotations, float)
    if (
        p.ndim != 2
        or p.shape[1] != 2
        or a.ndim != 3
        or a.shape[0] != len(p)
        or a.shape[2] != 2
        or a.shape[1] < 1
    ):
        raise ValueError(
            "Expected (N, 2) predictions and (N, reviewers, 2) annotations"
        )
    if not np.isfinite(p).all() or not np.isfinite(a).all():
        raise ValueError(
            "Resolve missing predictions/references explicitly before scoring"
        )
    return np.linalg.norm(p[:, None, :] - a, axis=2).mean(axis=1)


def change_frame(
    matrix: np.ndarray, source_geometry: np.ndarray, target_geometry: np.ndarray
) -> np.ndarray:
    """Convert a forward analysis-grid mapping to the supplied coordinate frame."""
    return (
        np.asarray(target_geometry)
        @ np.asarray(matrix)
        @ np.linalg.inv(source_geometry)
    )


def invert_pull_points(
    source_points: np.ndarray,
    pull_displacement: np.ndarray,
    initial_target_points: np.ndarray,
    *,
    tolerance_px: float = 1e-3,
    max_iterations: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Invert a target-grid pull map at source points with checked residuals.

    The pull map is ``source = target + displacement[target_y, target_x]``.
    Initial target guesses normally come from the inverse global pull matrix.
    Returned validity requires an in-grid, locally nonsingular solution whose
    residual meets tolerance. Unsupported points remain NaN. This does not
    establish global invertibility; folding must also be audited separately.
    """
    from scipy.ndimage import map_coordinates

    source = np.asarray(source_points, dtype=float)
    guess = np.asarray(initial_target_points, dtype=float).copy()
    field = np.asarray(pull_displacement, dtype=float)
    if source.ndim != 2 or source.shape[1] != 2 or guess.shape != source.shape:
        raise ValueError("Source and initial target points must both have shape (N, 2)")
    if field.ndim != 3 or field.shape[2] != 2 or min(field.shape[:2]) < 2:
        raise ValueError("Pull displacement must have shape (H, W, 2), H and W >= 2")
    if tolerance_px <= 0 or max_iterations < 1:
        raise ValueError("Tolerance and iteration count must be positive")
    height, width = field.shape[:2]
    dy_x, dx_x = np.gradient(field[:, :, 0])
    dy_y, dx_y = np.gradient(field[:, :, 1])
    gradients = (dx_x + 1, dy_x, dx_y, dy_y + 1)

    def sample(array, points):
        return map_coordinates(
            array,
            [points[:, 1], points[:, 0]],
            order=1,
            mode="constant",
            cval=np.nan,
            prefilter=False,
        )

    alive = np.isfinite(source).all(axis=1) & np.isfinite(guess).all(axis=1)
    residual = np.full(len(source), np.inf)
    for _ in range(max_iterations):
        alive &= (
            (guess[:, 0] >= 0)
            & (guess[:, 0] <= width - 1)
            & (guess[:, 1] >= 0)
            & (guess[:, 1] <= height - 1)
        )
        indices = np.flatnonzero(alive)
        if not len(indices):
            break
        q = guess[indices]
        value = (
            q
            + np.column_stack([sample(field[:, :, k], q) for k in range(2)])
            - source[indices]
        )
        a, b, c, d = (sample(g, q) for g in gradients)
        determinant = a * d - b * c
        nonsingular = (
            np.isfinite(value).all(axis=1)
            & np.isfinite(determinant)
            & (np.abs(determinant) > 1e-8)
        )
        alive[indices[~nonsingular]] = False
        residual[indices] = np.linalg.norm(value, axis=1)
        update = nonsingular & (residual[indices] > tolerance_px)
        if not np.any(update):
            break
        step = np.column_stack(
            [
                (d[update] * value[update, 0] - b[update] * value[update, 1])
                / determinant[update],
                (-c[update] * value[update, 0] + a[update] * value[update, 1])
                / determinant[update],
            ]
        )
        guess[indices[update]] -= step
    # Recheck after the last Newton step, including points that left the field.
    inside = (
        alive
        & (guess[:, 0] >= 0)
        & (guess[:, 0] <= width - 1)
        & (guess[:, 1] >= 0)
        & (guess[:, 1] <= height - 1)
    )
    idx = np.flatnonzero(inside)
    if len(idx):
        residual[idx] = np.linalg.norm(
            guess[idx]
            + np.column_stack([sample(field[:, :, k], guess[idx]) for k in range(2)])
            - source[idx],
            axis=1,
        )
    valid = inside & np.isfinite(residual) & (residual <= tolerance_px)
    guess[~valid] = np.nan
    return guess, valid, residual


def match_landmarks(
    source: Mapping[str, Sequence[float]],
    target: Mapping[str, Sequence[float]],
) -> tuple[list[str], np.ndarray, np.ndarray, dict[str, list[str]]]:
    """Match explicit IDs, exposing missing/invalid points rather than zipping rows."""
    ids = sorted(source.keys() & target.keys())
    valid, invalid = [], []
    for key in ids:
        a, b = np.asarray(source[key]), np.asarray(target[key])
        (
            valid
            if a.shape == b.shape == (2,)
            and np.isfinite(a).all()
            and np.isfinite(b).all()
            else invalid
        ).append(key)
    return (
        valid,
        np.array([source[k] for k in valid]).reshape(-1, 2),
        np.array([target[k] for k in valid]).reshape(-1, 2),
        {
            "missing_source": sorted(target.keys() - source.keys()),
            "missing_target": sorted(source.keys() - target.keys()),
            "invalid": invalid,
        },
    )


def landmark_metrics(
    source: np.ndarray,
    target: np.ndarray,
    predicted: np.ndarray,
    target_shape_hw: tuple[int, int],
    *,
    mpp_xy: tuple[float, float] | None = None,
) -> dict:
    """TRE in target pixels, rTRE by target diagonal, optional physical TRE.

    Non-finite predictions count as failures and use the unchanged source
    coordinate for challenge-compatible error calculations. Reference points
    must already be ID-matched and finite. Initial error uses source numeric
    coordinates in the target frame, as in ANHIR (no fitted initialization).
    """
    a, b, p = (np.asarray(x, dtype=float) for x in (source, target, predicted))
    if (
        a.shape != b.shape
        or a.shape != p.shape
        or a.ndim != 2
        or a.shape[1] != 2
        or not len(a)
    ):
        raise ValueError("Nonempty matching N by 2 arrays required")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Missing references must be resolved before scoring")
    if len(target_shape_hw) != 2 or min(target_shape_hw) <= 0:
        raise ValueError("Positive target image shape required")
    valid = np.isfinite(p).all(axis=1)
    p = np.where(valid[:, None], p, a)
    before = np.linalg.norm(a - b, axis=1)
    after = np.linalg.norm(p - b, axis=1)
    diagonal = float(np.hypot(*target_shape_hw))
    nonzero = before > 0
    out = {
        "n_landmarks": len(a),
        "n_missing_predictions": int((~valid).sum()),
        "tre_px": after.tolist(),
        "initial_tre_px": before.tolist(),
        "median_tre_px": float(np.median(after)),
        "p90_tre_px": float(np.percentile(after, 90)),
        "p95_tre_px": float(np.percentile(after, 95)),
        "max_tre_px": float(after.max()),
        "median_rtre": float(np.median(after) / diagonal),
        "max_rtre": float(after.max() / diagonal),
        "robustness": float(np.mean(after < before)),
        "mean_distance_reduction_px": float(np.mean(before - after)),
        "median_fractional_reduction": float(
            np.median(1 - after[nonzero] / before[nonzero])
        )
        if nonzero.any()
        else None,
    }
    if mpp_xy is not None:
        pixel_geometry(mpp_xy=mpp_xy)
        physical = np.linalg.norm((p - b) * np.asarray(mpp_xy), axis=1)
        out.update(
            tre_um=physical.tolist(),
            median_tre_um=float(np.median(physical)),
            p90_tre_um=float(np.percentile(physical, 90)),
        )
    return out


def acrobat_2022_score(errors_by_pair: Mapping[str, Sequence[float]]) -> float:
    """Median of per-pair 90th percentiles; errors average annotator distances.

    Input excludes adjudicated invalid landmarks according to the challenge
    protocol. This does not provide access to the private challenge references.
    """
    if not errors_by_pair or any(not len(x) for x in errors_by_pair.values()):
        raise ValueError("Each evaluated pair needs errors")
    return float(np.median([np.percentile(x, 90) for x in errors_by_pair.values()]))


def block_bootstrap(
    values: Sequence[float],
    blocks: Sequence[str],
    *,
    resamples: int = 2000,
    seed: int = 20261002,
) -> dict:
    """Percentile CI for an equally weighted block mean (not a landmark CI)."""
    v = np.asarray(values, float)
    b = np.asarray(blocks)
    if v.ndim != 1 or len(v) != len(b) or not len(v) or not np.isfinite(v).all():
        raise ValueError("Matching finite values and block identifiers required")
    if resamples < 1:
        raise ValueError("Positive resample count required")
    means = np.array([v[b == k].mean() for k in np.unique(b)])
    estimates = means[
        np.random.default_rng(seed).integers(0, len(means), (resamples, len(means)))
    ].mean(axis=1)
    return {
        "mean": float(means.mean()),
        "ci95": np.percentile(estimates, [2.5, 97.5]).tolist(),
        "n_blocks": len(means),
        "n_values": len(v),
        "resamples": resamples,
        "seed": seed,
        "ci_informative": len(means) > 1,
    }


def order_metrics(
    truth: Sequence[str], predicted: Sequence[str], *, reversal_allowed: bool = True
) -> dict:
    """Compare complete permutations, optionally treating reversal as equivalent.

    Using an algorithm output as ``truth`` measures stability, not accuracy.
    Physical-order accuracy requires independent acquisition metadata.
    """
    if (
        len(truth) < 2
        or len(set(truth)) != len(truth)
        or len(predicted) != len(truth)
        or set(predicted) != set(truth)
    ):
        raise ValueError("Complete unique permutations of at least two planes required")
    position = {k: i for i, k in enumerate(truth)}
    p = np.array([position[k] for k in predicted])
    n = len(p)
    adjacent = np.abs(np.diff(p)) == 1 if reversal_allowed else np.diff(p) == 1
    rho = 1 - 6 * np.sum((p - np.arange(n)) ** 2) / (n * (n * n - 1))
    exact = list(predicted) == list(truth) or (
        reversal_allowed and list(predicted) == list(truth)[::-1]
    )
    return {
        "adjacency_recovery": float(adjacent.mean()),
        "rank_agreement": float(abs(rho) if reversal_allowed else rho),
        "complete_recovery": bool(exact),
        "reversal_allowed": reversal_allowed,
    }


def affine_shape_metrics(physical_forward_matrix: np.ndarray) -> dict:
    """Directional and area changes, explicitly assessed in physical coordinates."""
    a = np.asarray(physical_forward_matrix, float)[:2, :2]
    singular = np.linalg.svd(a, compute_uv=False)
    determinant = float(np.linalg.det(a))
    return {
        "min_directional_scale": float(singular.min()),
        "max_directional_scale": float(singular.max()),
        "signed_area_ratio": determinant,
        "area_ratio": abs(determinant),
        "reflection": determinant < 0,
    }


def mask_reference_metrics(
    reference: np.ndarray,
    predicted: np.ndarray,
    *,
    spacing_xy: tuple[float, float] = (1, 1),
) -> dict:
    """Compare same-section masks against independent reference outlines.

    Distances use the supplied units. This must not be interpreted as masking
    accuracy when the masks belong to different physical tissue sections.
    Empty boundaries have undefined distances, represented as None.
    """
    from scipy import ndimage

    a, b = np.asarray(reference, bool), np.asarray(predicted, bool)
    if a.ndim != 2 or a.shape != b.shape:
        raise ValueError("Masks must share a two-dimensional image frame")
    pixel_geometry(mpp_xy=spacing_xy)
    intersection = int(np.sum(a & b))
    total = int(a.sum() + b.sum())
    result = {
        "dice": 2 * intersection / total if total else 1.0,
        "retained_reference_tissue": intersection / int(a.sum()) if a.any() else None,
        "included_background_fraction": int(np.sum(b & ~a)) / int(b.sum())
        if b.any()
        else None,
        "mean_boundary_distance": None,
        "p95_boundary_distance": None,
    }
    if a.any() and b.any():
        ab = a & ~ndimage.binary_erosion(a, border_value=0)
        bb = b & ~ndimage.binary_erosion(b, border_value=0)
        sampling = tuple(reversed(spacing_xy))
        distances = np.r_[
            ndimage.distance_transform_edt(~ab, sampling=sampling)[bb],
            ndimage.distance_transform_edt(~bb, sampling=sampling)[ab],
        ]
        result.update(
            mean_boundary_distance=float(distances.mean()),
            p95_boundary_distance=float(np.percentile(distances, 95)),
        )
    return result


def displacement_jacobian_metrics(
    displacement_xy: np.ndarray, support: np.ndarray | None = None
) -> dict:
    """Jacobian of a pixel-grid mapping x + displacement(x), with explicit support.

    The caller must identify whether this is a forward or pull field. A pull
    field determinant describes that mapping, not its inverse deformation.
    Pixel calibration must be incorporated separately when frames differ.
    """
    d = np.asarray(displacement_xy, float)
    if d.ndim != 3 or d.shape[-1] != 2 or min(d.shape[:2]) < 2:
        raise ValueError("Expected an H by W by 2 displacement field")
    use = np.ones(d.shape[:2], bool) if support is None else np.asarray(support, bool)
    if use.shape != d.shape[:2] or not use.any() or not np.isfinite(d).all():
        raise ValueError("Finite displacement and nonempty matching support required")
    ux_y, ux_x = np.gradient(d[..., 0])
    uy_y, uy_x = np.gradient(d[..., 1])
    determinant = ((1 + ux_x) * (1 + uy_y) - ux_y * uy_x)[use]
    return {
        "n_supported_pixels": int(use.sum()),
        "fold_fraction": float(np.mean(determinant <= 0)),
        "jacobian_p01": float(np.percentile(determinant, 1)),
        "jacobian_median": float(np.median(determinant)),
        "jacobian_p99": float(np.percentile(determinant, 99)),
    }
