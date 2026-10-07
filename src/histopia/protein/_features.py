"""Stain-neutral morphology features and cross-section semantic context."""

from __future__ import annotations

import numpy as np


def render_neutral_morphology(
    counterstain_od: np.ndarray,
    tissue_mask: np.ndarray,
    *,
    display_max: float | None = None,
    palette_rgb: tuple[int, int, int] = (74, 54, 122),
) -> np.ndarray:
    """Render one counterstain channel in a fixed synthetic RGB palette."""

    od = np.asarray(counterstain_od, dtype=np.float64)
    tissue = np.asarray(tissue_mask, dtype=bool)
    if od.ndim != 2 or tissue.shape != od.shape:
        raise ValueError("counterstain OD and tissue mask must share a 2D shape")
    if not np.all(np.isfinite(od)) or np.any(od < 0):
        raise ValueError("counterstain OD must be finite and nonnegative")
    if display_max is None:
        values = od[tissue & (od > 0)]
        display_max = float(np.quantile(values, 0.995)) if values.size else 1.0
    if not np.isfinite(display_max) or display_max <= 0:
        raise ValueError("display_max must be positive and finite")
    palette = np.asarray(palette_rgb, dtype=np.float64)
    if palette.shape != (3,) or np.any((palette < 0) | (palette > 255)):
        raise ValueError("palette_rgb must contain three bytes")
    density = np.clip(od / display_max, 0.0, 1.0)[..., None]
    rgb = 255.0 * (1.0 - density) + palette * density
    rgb[~tissue] = 255.0
    return np.rint(rgb).astype(np.uint8)


def pool_patch_features_to_cells(
    patch_xy_um: np.ndarray,
    patch_features: np.ndarray,
    cell_xy_um: np.ndarray,
    *,
    neighbors: int = 4,
    maximum_distance_um: float = 112.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Distance-weight nearby stain-neutral patch features into cells."""

    patches = np.asarray(patch_xy_um, dtype=np.float64)
    features = np.asarray(patch_features, dtype=np.float64)
    cells = np.asarray(cell_xy_um, dtype=np.float64)
    if patches.ndim != 2 or patches.shape[1] != 2:
        raise ValueError("patch_xy_um must have shape (patches, 2)")
    if cells.ndim != 2 or cells.shape[1] != 2:
        raise ValueError("cell_xy_um must have shape (cells, 2)")
    if features.ndim != 2 or features.shape[0] != patches.shape[0]:
        raise ValueError("patch features must align with patch coordinates")
    if neighbors < 1 or maximum_distance_um <= 0:
        raise ValueError("pooling controls must be positive")
    output = np.zeros((len(cells), features.shape[1]), dtype=np.float32)
    supported = np.zeros(len(cells), dtype=bool)
    if not len(patches):
        return output, supported
    count = min(neighbors, len(patches))
    for start in range(0, len(cells), 4096):
        stop = min(start + 4096, len(cells))
        distances2 = np.sum(
            (cells[start:stop, None, :] - patches[None, :, :]) ** 2, axis=2
        )
        indices = np.argpartition(distances2, count - 1, axis=1)[:, :count]
        selected2 = np.take_along_axis(distances2, indices, axis=1)
        valid = selected2 <= maximum_distance_um**2
        weights = np.where(valid, 1.0 / np.maximum(np.sqrt(selected2), 1.0), 0.0)
        denominator = weights.sum(axis=1)
        supported[start:stop] = denominator > 0
        pooled = np.sum(features[indices] * weights[..., None], axis=1)
        output[start:stop] = np.divide(
            pooled,
            denominator[:, None],
            out=np.zeros_like(pooled),
            where=denominator[:, None] > 0,
        )
    return output, supported


def leave_one_section_out_consensus(
    section_probabilities: np.ndarray,
    section_index: int,
    *,
    section_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Exclude one section and calculate a normalized 3D semantic consensus."""

    values = np.asarray(section_probabilities, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError(
            "section probabilities must have shape (sections, cells, regions)"
        )
    if not 0 <= section_index < values.shape[0]:
        raise IndexError("section_index is outside section probabilities")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("semantic probabilities must be finite and nonnegative")
    weights = (
        np.ones(values.shape[:2], dtype=np.float64)
        if section_weights is None
        else np.asarray(section_weights, dtype=np.float64)
    )
    if weights.shape != values.shape[:2] or np.any(weights < 0):
        raise ValueError("section weights must match sections and cells")
    weights = weights.copy()
    weights[section_index] = 0.0
    total_weight = weights.sum(axis=0)
    consensus = np.sum(values * weights[..., None], axis=0)
    consensus = np.divide(
        consensus,
        total_weight[:, None],
        out=np.zeros_like(consensus),
        where=total_weight[:, None] > 0,
    )
    probability_sum = consensus.sum(axis=1)
    consensus = np.divide(
        consensus,
        probability_sum[:, None],
        out=np.zeros_like(consensus),
        where=probability_sum[:, None] > 0,
    )
    return consensus.astype(np.float32), (total_weight > 0)
