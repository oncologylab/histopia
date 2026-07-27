"""Flow-guided interpolation of selected-K semantic fields."""

from __future__ import annotations

import numpy as np

from histopia.topology._model import ObservedSection, ReconstructedPlane


def smooth_displacement_field(
    shape: tuple[int, int],
    *,
    source_xy: np.ndarray,
    target_xy: np.ndarray,
    origin_um_xy: tuple[float, float],
    spacing_um: float,
    sigma_cells: float = 1.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize sparse physical displacements with normalized convolution."""

    try:
        from scipy.ndimage import gaussian_filter
    except ImportError as exc:
        raise RuntimeError("topology interpolation requires scipy") from exc
    height, width = shape
    weighted = np.zeros((2, height, width), dtype=np.float64)
    support = np.zeros((height, width), dtype=np.float64)
    source = np.asarray(source_xy, dtype=float)
    target = np.asarray(target_xy, dtype=float)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2:
        raise ValueError("topology link coordinates must have shape (links, 2)")
    if not len(source):
        return weighted.astype(np.float32), support.astype(np.float32)
    cols = np.rint((source[:, 0] - origin_um_xy[0]) / spacing_um).astype(int)
    rows = np.rint((source[:, 1] - origin_um_xy[1]) / spacing_um).astype(int)
    inside = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    rows = rows[inside]
    cols = cols[inside]
    displacement = (target[inside] - source[inside]) / spacing_um
    np.add.at(support, (rows, cols), 1.0)
    np.add.at(weighted[0], (rows, cols), displacement[:, 1])
    np.add.at(weighted[1], (rows, cols), displacement[:, 0])
    blurred_support = gaussian_filter(support, sigma=sigma_cells, mode="nearest")
    field = np.zeros_like(weighted)
    for axis in range(2):
        numerator = gaussian_filter(weighted[axis], sigma=sigma_cells, mode="nearest")
        np.divide(
            numerator,
            np.maximum(blurred_support, np.finfo(float).eps),
            out=field[axis],
            where=blurred_support > np.finfo(float).eps,
        )
    if np.any(blurred_support <= np.finfo(float).eps):
        from scipy.ndimage import distance_transform_edt

        missing = blurred_support <= np.finfo(float).eps
        indices = distance_transform_edt(
            missing, return_distances=False, return_indices=True
        )
        field[:, missing] = field[:, indices[0][missing], indices[1][missing]]
    confidence = blurred_support / max(float(blurred_support.max()), 1.0)
    return field.astype(np.float32), confidence.astype(np.float32)


def interpolate_pair(
    source: ObservedSection,
    target: ObservedSection,
    *,
    fraction: float,
    z_um: float,
    segment: int,
    source_section: int,
    target_section: int,
    displacement_rc: np.ndarray,
    flow_confidence: np.ndarray,
) -> ReconstructedPlane:
    """Interpolate one virtual plane while preserving endpoint class support."""

    if not 0 < fraction < 1:
        raise ValueError("interpolation fraction must be in (0, 1)")
    if source.membership.shape != target.membership.shape:
        raise ValueError("semantic endpoint fields must have equal shape")
    try:
        from scipy.ndimage import distance_transform_edt, map_coordinates
    except ImportError as exc:
        raise RuntimeError("topology interpolation requires scipy") from exc
    classes, height, width = source.membership.shape
    rows, cols = np.indices((height, width), dtype=np.float32)
    source_coords = np.asarray(
        [rows - fraction * displacement_rc[0], cols - fraction * displacement_rc[1]]
    )
    target_coords = np.asarray(
        [
            rows + (1.0 - fraction) * displacement_rc[0],
            cols + (1.0 - fraction) * displacement_rc[1],
        ]
    )
    source_membership = np.stack(
        [
            map_coordinates(
                source.membership[index],
                source_coords,
                order=1,
                mode="constant",
                cval=0.0,
            )
            for index in range(classes)
        ]
    )
    target_membership = np.stack(
        [
            map_coordinates(
                target.membership[index],
                target_coords,
                order=1,
                mode="constant",
                cval=0.0,
            )
            for index in range(classes)
        ]
    )
    source_sdf = _signed_distance(source.support, distance_transform_edt)
    target_sdf = _signed_distance(target.support, distance_transform_edt)
    warped_source_sdf = map_coordinates(
        source_sdf, source_coords, order=1, mode="constant", cval=-1.0
    )
    warped_target_sdf = map_coordinates(
        target_sdf, target_coords, order=1, mode="constant", cval=-1.0
    )
    sdf = (1.0 - fraction) * warped_source_sdf + fraction * warped_target_sdf
    support = sdf >= 0
    membership = (
        (1.0 - fraction) * source_membership + fraction * target_membership
    )
    total = membership.sum(axis=0)
    np.divide(
        membership,
        np.maximum(total, np.finfo(np.float32).eps),
        out=membership,
        where=total > np.finfo(np.float32).eps,
    )
    membership[:, ~support] = 0.0
    labels = np.argmax(membership, axis=0).astype(np.int16)
    labels[~support] = -1
    entropy = np.zeros((height, width), dtype=np.float32)
    positive = membership > 0
    entropy -= np.sum(
        np.where(positive, membership * np.log(np.maximum(membership, 1e-7)), 0.0),
        axis=0,
    )
    entropy /= max(float(np.log(classes)), np.finfo(float).eps)
    distance_uncertainty = 4.0 * fraction * (1.0 - fraction)
    uncertainty = np.clip(
        0.45 * entropy
        + 0.35 * distance_uncertainty
        + 0.20 * (1.0 - np.asarray(flow_confidence, dtype=float)),
        0,
        1,
    ).astype(np.float32)
    uncertainty[~support] = 1.0
    return ReconstructedPlane(
        z_um=z_um,
        segment=segment,
        source_section=source_section,
        target_section=target_section,
        fraction=fraction,
        observed=False,
        slide_id=None,
        membership=membership.astype(np.float32),
        labels=labels,
        support=support,
        uncertainty=uncertainty,
    )


def observed_plane(
    section: ObservedSection,
    *,
    z_um: float,
    segment: int,
    section_index: int,
) -> ReconstructedPlane:
    """Wrap an observed field as an exact, zero-reconstruction-uncertainty plane."""

    return ReconstructedPlane(
        z_um=z_um,
        segment=segment,
        source_section=section_index,
        target_section=section_index,
        fraction=0.0,
        observed=True,
        slide_id=section.slide_id,
        membership=section.membership,
        labels=section.labels,
        support=section.support,
        uncertainty=np.zeros(section.support.shape, dtype=np.float32),
    )


def _signed_distance(mask: np.ndarray, distance_transform_edt) -> np.ndarray:
    inside = distance_transform_edt(mask)
    outside = distance_transform_edt(~mask)
    return inside - outside
