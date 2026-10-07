"""Opt-in recovery of pale tissue from existing engineered mask proposals."""

from __future__ import annotations

import numpy as np

from ._adaptive import isolated_annotation_exclusion


def recover_supported_mask(image, base_mask, proposals):
    """Recover proposal pixels with chromatic or textured native-image evidence.

    This is engineered image processing, not a trained object detector. Supplied
    group proposals may incorporate neighboring-section priors; callers must
    retain their provenance. No evaluated transform or landmark is used here.
    Neutral uniform shading alone cannot justify an addition. Original masks,
    additions, and proposed/excluded artifacts must remain separately traceable.
    """
    import cv2

    rgb = np.asarray(image)
    base = np.asarray(base_mask, bool)
    if rgb.dtype != np.uint8 or rgb.shape != (*base.shape, 3):
        raise ValueError("RGB uint8 input and aligned mask required")
    gray = rgb.astype("float32").mean(2)
    mean = cv2.GaussianBlur(gray, (0, 0), 2)
    variance = np.maximum(0, cv2.GaussianBlur(gray**2, (0, 0), 2) - mean**2)
    od = -np.log(np.clip((rgb.astype("float32") + 1) / 256, 1e-5, 1)).mean(2)
    support = (od > 0.035) & (
        (np.ptp(rgb.astype(float), axis=2) > 6) | (variance > 6.25)
    )
    union = np.zeros(base.shape, bool)
    used = []
    for name in [
        "hysteresis_tissue",
        "background_corrected",
        "group_pale_tissue",
        "group_density_union",
    ]:
        if name in proposals:
            value = np.asarray(proposals[name], bool)
            if value.shape != base.shape:
                raise ValueError("proposal geometry mismatch")
            union |= value
            used.append(name)
    addition = union & support & ~base
    _, labels, stats, _ = cv2.connectedComponentsWithStats(addition.astype("uint8"))
    indices = np.flatnonzero(
        stats[:, cv2.CC_STAT_AREA] >= max(32, round(base.size * 0.00002))
    )
    addition = np.isin(labels, indices[indices != 0])
    proposal, exclusion = isolated_annotation_exclusion(rgb, base | addition)
    result = (base | addition) & ~exclusion
    return result, {
        "original_pixels": int(base.sum()),
        "added_pixels": int(addition.sum()),
        "excluded_pixels": int(((base | addition) & exclusion).sum()),
        "annotation_proposal_pixels": int(proposal.sum()),
        "recovered_pixels": int(result.sum()),
        "proposal_names": used,
        "trained_detector": False,
        "independent_mask_accuracy": "pending outlines",
    }
