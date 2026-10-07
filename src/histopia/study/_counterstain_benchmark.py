"""Fixed H/DAB separation and native-pixel patch targets for a 2D pilot.

The hematoxylin rendering is an IHC counterstain input, never synthetic H&E.
Optical density is a staining proxy, not an absolute protein concentration.
"""

from __future__ import annotations

import numpy as np

H_DAB_BASIS = np.array([[0.650, 0.704, 0.286], [0.268, 0.570, 0.776]], dtype=np.float32)
H_DAB_INVERSE = np.linalg.pinv(H_DAB_BASIS)


def separate_fixed_hdab(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return nonnegative H, DAB and residual OD using a fixed white point."""
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("stain separation requires uint8 RGB")
    od = -np.log((image.astype(np.float32) + 1) / 256)
    concentrations = np.maximum(od @ H_DAB_INVERSE, 0)
    residual = np.linalg.norm(od - concentrations @ H_DAB_BASIS, axis=2)
    return concentrations[..., 0], concentrations[..., 1], residual


def counterstain_patch_inputs(rgb: np.ndarray, *, patch_size: int = 224) -> dict:
    """Select nonoverlapping patches using only H-derived morphology support.

    A fixed H threshold and morphology operations determine support. No DAB
    score enters patch selection, neutral rendering or the returned nuisance
    features. The fixed display range of 2 OD is not fitted to test images.
    Incomplete edge patches are omitted, and retained coverage is explicit.
    """
    from scipy.ndimage import binary_closing, binary_fill_holes

    from histopia.protein._features import render_neutral_morphology

    if patch_size < 1:
        raise ValueError("patch size must be positive")
    h, dab, residual = separate_fixed_hdab(rgb)
    # 0.12 OD was qualified on development images: 0.04 OD included the
    # scanner's off-white glass background. This value is frozen before test.
    tissue = binary_fill_holes(binary_closing(h > 0.12, iterations=3))
    neutral = render_neutral_morphology(h, tissue, display_max=2.0)
    images, coordinates, targets, stats, support = [], [], [], [], []
    for y in range(0, h.shape[0] - patch_size + 1, patch_size):
        for x in range(0, h.shape[1] - patch_size + 1, patch_size):
            bounds = np.s_[y : y + patch_size, x : x + patch_size]
            mask = tissue[bounds]
            if mask.mean() < 0.25:
                continue
            hv = h[bounds][mask]
            images.append(neutral[bounds])
            coordinates.append((x, y))
            targets.append(float(dab[bounds][mask].mean()))
            stats.append(
                [
                    float(hv.mean()),
                    float(hv.std()),
                    *np.quantile(hv, [0.25, 0.5, 0.75]).tolist(),
                    float(mask.mean()),
                ]
            )
            support.append(int(mask.sum()))
    return dict(
        images=np.asarray(images, dtype=np.uint8).reshape(
            -1, patch_size, patch_size, 3
        ),
        xy_px=np.asarray(coordinates, dtype=np.int32).reshape(-1, 2),
        target_dab_od=np.asarray(targets, dtype=np.float32),
        h_statistics=np.asarray(stats, dtype=np.float32).reshape(-1, 6),
        supported_pixels=np.asarray(support, dtype=np.int32),
        h_od=h,
        dab_od=dab,
        residual_od=residual,
        morphology_rgb=neutral,
        tissue=tissue,
    )
