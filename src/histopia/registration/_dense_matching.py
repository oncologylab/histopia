"""Optional dense matchers used for global hypotheses, never automatic dense warps."""

from __future__ import annotations

import numpy as np

from ._guarded import candidate_from_points, pixel_center_resize_map, structure_channels
from ._learned_matching import quarter_turn_matrix


def normalized_to_pixel_centres(points, shape_hw):
    """Convert align_corners=False coordinates to zero-based pixel centres.

    RoMa's public to_pixel helper returns edge-based coordinates, which require
    subtracting 0.5 before composing with Histopia's pixel-centre transforms.
    """
    xy = np.asarray(points, float)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError("finite normalized XY pairs required")
    h, w = shape_hw
    if min(h, w) <= 0:
        raise ValueError("positive image dimensions required")
    return (xy + 1) * np.array([w, h]) / 2 - 0.5


def balanced_correspondences(
    source_xy, target_xy, confidence, source_mask, target_mask, *, bins=8, per_bin=64
):
    """Filter tissue support and cap density in both images, independent of outcomes."""
    a, b, scores = np.asarray(source_xy), np.asarray(target_xy), np.asarray(confidence)
    if (
        a.shape != b.shape
        or a.ndim != 2
        or a.shape[1] != 2
        or scores.shape != (len(a),)
    ):
        raise ValueError("correspondence arrays differ")
    valid = np.isfinite(a).all(1) & np.isfinite(b).all(1) & np.isfinite(scores)
    aa, bb = np.zeros(a.shape, int), np.zeros(b.shape, int)
    aa[valid], bb[valid] = np.rint(a[valid]).astype(int), np.rint(b[valid]).astype(int)
    for xy, mask in [(aa, source_mask), (bb, target_mask)]:
        valid &= (
            (xy[:, 0] >= 0)
            & (xy[:, 0] < mask.shape[1])
            & (xy[:, 1] >= 0)
            & (xy[:, 1] < mask.shape[0])
        )
        indices = np.flatnonzero(valid)
        valid[indices] &= mask[xy[indices, 1], xy[indices, 0]]
    counts = [np.zeros((bins, bins), int), np.zeros((bins, bins), int)]
    selected = []
    for index in np.argsort(-scores, kind="stable"):
        if not valid[index]:
            continue
        cells = []
        for xy, mask in [(aa, source_mask), (bb, target_mask)]:
            x, y = np.minimum(
                bins - 1, xy[index] * bins // np.array([mask.shape[1], mask.shape[0]])
            )
            cells.append((y, x))
        if any(
            count[cell] >= per_bin for count, cell in zip(counts, cells, strict=True)
        ):
            continue
        for count, cell in zip(counts, cells, strict=True):
            count[cell] += 1
        selected.append(index)
    selected = np.asarray(selected, int)
    return a[selected], b[selected], scores[selected]


def dense_candidates(
    source,
    target,
    *,
    backend,
    model,
    config,
    source_modality="unknown",
    target_modality="unknown",
):
    """Frozen matcher proposals across structural channels and four orientations.

    The caller owns model loading, provenance, and the shared GPU lock. All
    candidates are checked later using identical physical/support constraints.
    """
    import cv2
    import torch
    from PIL import Image

    if backend not in {"loftr", "romav2"}:
        raise ValueError("unsupported dense matcher")
    torch.manual_seed(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    channels_a = structure_channels(source.image, source.mask, source_modality)
    channels_b = structure_channels(target.image, target.mask, target_modality)
    if backend == "romav2":
        channels_a["rgb"], channels_b["rgb"] = source.image, target.image
    candidates, evidence = [], []
    for name in channels_a:
        for turns in range(4):
            a = np.ascontiguousarray(np.rot90(channels_a[name], turns))
            b = channels_b[name]
            with torch.inference_mode():
                if backend == "romav2":
                    rgb_a = np.repeat(a[..., None], 3, 2) if a.ndim == 2 else a
                    rgb_b = np.repeat(b[..., None], 3, 2) if b.ndim == 2 else b
                    prediction = model.match(
                        Image.fromarray(rgb_a), Image.fromarray(rgb_b)
                    )
                    matches, overlap, *_ = model.sample(prediction, 5000)
                    matches = matches.detach().cpu().numpy()
                    confidence = overlap.detach().cpu().numpy().reshape(-1)
                    keep = confidence >= 0.1
                    ax = normalized_to_pixel_centres(matches[keep, :2], a.shape[:2])
                    bx = normalized_to_pixel_centres(matches[keep, 2:], b.shape[:2])
                    confidence = confidence[keep]
                else:
                    tensors, maps = [], []
                    for gray in [a, b]:
                        scale = min(1.0, 640 / max(gray.shape))
                        h, w = [max(8, int(n * scale) // 8 * 8) for n in gray.shape]
                        small = cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA)
                        tensors.append(
                            torch.from_numpy(small.copy())[None, None]
                            .float()
                            .to(device)
                            / 255
                        )
                        maps.append(pixel_center_resize_map(gray.shape, (h, w)))
                    prediction = model({"image0": tensors[0], "image1": tensors[1]})
                    ax, bx = [
                        prediction[key].detach().cpu().numpy()
                        for key in ["keypoints0", "keypoints1"]
                    ]
                    confidence = prediction["confidence"].detach().cpu().numpy()
                    ax = (np.c_[ax, np.ones(len(ax))] @ maps[0].T)[:, :2]
                    bx = (np.c_[bx, np.ones(len(bx))] @ maps[1].T)[:, :2]
                    keep = confidence >= 0.2
                    ax, bx, confidence = ax[keep], bx[keep], confidence[keep]
            rotation = quarter_turn_matrix(source.mask.shape, turns)
            ax = (np.c_[ax, np.ones(len(ax))] @ np.linalg.inv(rotation).T)[:, :2]
            ax, bx, confidence = balanced_correspondences(
                ax, bx, confidence, source.mask, target.mask
            )
            c = candidate_from_points(
                ax,
                bx,
                stage=f"{backend}_{name}_{turns * 90}",
                target_shape=target.mask.shape,
                config=config,
            )
            evidence.append(
                {
                    "channel": name,
                    "rotation": turns * 90,
                    "supported_matches": len(ax),
                    "inliers": c.feature_inlier_count if c else 0,
                }
            )
            if c is not None:
                candidates.append(c)
    return candidates, evidence
