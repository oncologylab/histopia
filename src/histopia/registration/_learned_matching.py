"""Optional frozen DISK/LightGlue hypotheses; model imports occur only on use."""

from __future__ import annotations

import numpy as np

from histopia.registration._guarded import (
    GuardedRegistrationConfig,
    ImageFrame,
    RegistrationCandidate,
    candidate_from_points,
    structure_channels,
)


def quarter_turn_matrix(shape_hw: tuple[int, int], turns: int) -> np.ndarray:
    """Original pixel centers into ``numpy.rot90(image, turns)`` coordinates."""
    h, w = shape_hw
    if min(h, w) <= 0:
        raise ValueError("positive image dimensions required")
    return np.asarray(
        [
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[0, 1, 0], [-1, 0, w - 1], [0, 0, 1]],
            [[-1, 0, w - 1], [0, -1, h - 1], [0, 0, 1]],
            [[0, -1, h - 1], [1, 0, 0], [0, 0, 1]],
        ][turns % 4],
        dtype=float,
    )


def learned_candidates(
    source: ImageFrame,
    target: ImageFrame,
    *,
    config: GuardedRegistrationConfig,
    source_modality: str = "unknown",
    target_modality: str = "unknown",
    maximum_features: int = 2048,
) -> tuple[list[RegistrationCandidate], dict]:
    """Frozen pretrained features, four exact orientations, geometric RANSAC.

    The caller supplies a GPU resource claim if needed. No correspondence labels
    or protein targets are consumed. These generic pretrained models are not a
    trained Histopia tissue detector.
    """
    import kornia.feature as kf
    import torch

    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    disk = kf.DISK.from_pretrained("depth").eval().to(device)
    matcher = kf.LightGlueMatcher("disk").eval().to(device)
    a = structure_channels(source.image, source.mask, source_modality)["hematoxylin"]
    b = structure_channels(target.image, target.mask, target_modality)["hematoxylin"]

    def features(gray, mask):
        if max(gray.shape) > config.max_dim_px:
            raise ValueError("learned detector image exceeds frozen ceiling")
        rgb = np.repeat(gray[..., None], 3, axis=2).copy()
        tensor = torch.from_numpy(rgb).permute(2, 0, 1)[None].float().to(device) / 255
        with torch.inference_mode():
            raw = disk(tensor, n=maximum_features, pad_if_not_divisible=True)[0]
        xy = raw.keypoints.detach().cpu().numpy()
        idx = np.rint(xy).astype(int)
        idx[:, 0] = np.clip(idx[:, 0], 0, mask.shape[1] - 1)
        idx[:, 1] = np.clip(idx[:, 1], 0, mask.shape[0] - 1)
        keep = torch.from_numpy(mask[idx[:, 1], idx[:, 0]]).to(device)
        return raw.keypoints[keep], raw.descriptors[keep]

    fixed_xy, fixed_desc = features(b, target.mask)
    candidates = []
    diagnostics = {
        "device": str(device),
        "torch": torch.__version__,
        "maximum_features": maximum_features,
        "orientations": [],
    }
    for turns in range(4):
        gray = np.ascontiguousarray(np.rot90(a, turns))
        mask = np.ascontiguousarray(np.rot90(source.mask, turns))
        moving_xy, moving_desc = features(gray, mask)
        if min(len(moving_xy), len(fixed_xy)) < config.min_feature_inliers:
            diagnostics["orientations"].append({"turns": turns, "matches": 0})
            continue
        with torch.inference_mode():
            la = kf.laf_from_center_scale_ori(moving_xy[None])
            lb = kf.laf_from_center_scale_ori(fixed_xy[None])
            _, indices = matcher(
                moving_desc, fixed_desc, la, lb, hw1=gray.shape, hw2=b.shape
            )
        idx = indices.detach().cpu().numpy()
        src = moving_xy.detach().cpu().numpy()[idx[:, 0]]
        T = quarter_turn_matrix(source.mask.shape, turns)
        src = (np.c_[src, np.ones(len(src))] @ np.linalg.inv(T).T)[:, :2]
        tgt = fixed_xy.detach().cpu().numpy()[idx[:, 1]]
        candidate = candidate_from_points(
            src,
            tgt,
            stage=f"disk_lightglue_rotation_{turns * 90}",
            target_shape=target.mask.shape,
            config=config,
        )
        diagnostics["orientations"].append(
            {
                "turns": turns,
                "matches": len(src),
                "inliers": candidate.feature_inlier_count if candidate else 0,
            }
        )
        if candidate is not None:
            candidates.append(candidate)
    return candidates, diagnostics
