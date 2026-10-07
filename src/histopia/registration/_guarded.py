"""Opt-in, image-only registration candidates with separate geometric evidence.

This experimental module does not change the historical registration defaults.
Image resampling uses pixel centers and the *actual* rounded X/Y output sizes.
Mask agreement is never represented as a number of feature correspondences.
Evaluation landmarks are deliberately absent from the fitting interface.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class GuardedRegistrationConfig:
    max_dim_px: int = 2048
    mask_mode: str = "supported_components"
    matcher: str = "structural"
    refine: bool = True
    seed: int = 20261005
    max_features: int = 6000
    match_ratio: float = 0.75
    min_feature_inliers: int = 10
    max_relative_scale_change: float = 0.35
    max_relative_anisotropy: float = 1.30
    min_structure_gain: float = 0.002

    def __post_init__(self):
        if self.max_dim_px < 64 or self.max_features < 16:
            raise ValueError("working size and feature budget are too small")
        if self.mask_mode not in {"dominant", "all_components", "supported_components"}:
            raise ValueError("unknown tissue crop mode")
        if self.matcher not in {"legacy", "structural", "learned"}:
            raise ValueError("unknown matcher")
        if not 0 < self.match_ratio < 1 or self.min_feature_inliers < 3:
            raise ValueError("invalid matching constraints")
        values = [
            self.max_relative_scale_change,
            self.max_relative_anisotropy,
            self.min_structure_gain,
        ]
        if (
            not np.isfinite(values).all()
            or not 0 < values[0] < 1
            or values[1] < 1
            or values[2] < 0
        ):
            raise ValueError("invalid shape or selection constraints")


@dataclass(slots=True)
class ImageFrame:
    image: np.ndarray
    mask: np.ndarray
    working_to_input: np.ndarray
    retained_mask_fraction: float
    bbox_xywh: tuple[int, int, int, int]


@dataclass(slots=True)
class RegistrationCandidate:
    matrix: np.ndarray
    stage: str
    feature_match_count: int = 0
    feature_inlier_count: int = 0
    feature_spatial_coverage: float = 0.0
    diagnostics: dict[str, Any] = field(default_factory=dict)
    accepted: bool = True
    rejection_reason: str | None = None

    def to_json_dict(self):
        return {
            "matrix": self.matrix.tolist(),
            "stage": self.stage,
            "feature_match_count": self.feature_match_count,
            "feature_inlier_count": self.feature_inlier_count,
            "feature_spatial_coverage": self.feature_spatial_coverage,
            "diagnostics": self.diagnostics,
            "accepted": self.accepted,
            "rejection_reason": self.rejection_reason,
        }


@dataclass(slots=True)
class GuardedRegistrationResult:
    matrix: np.ndarray
    selected_stage: str
    candidates: list[RegistrationCandidate]
    source_frame: ImageFrame
    target_frame: ImageFrame
    settings: GuardedRegistrationConfig

    def to_json_dict(self):
        return {
            "source_to_target_matrix": self.matrix.tolist(),
            "selected_stage": self.selected_stage,
            "direction": "source_to_target",
            "coordinate_units": "input_pixel_centres",
            "settings": asdict(self.settings),
            "candidates": [c.to_json_dict() for c in self.candidates],
            "frames": {
                label: {
                    "working_to_input": frame.working_to_input.tolist(),
                    "retained_mask_fraction": frame.retained_mask_fraction,
                    "bbox_xywh": list(frame.bbox_xywh),
                    "shape_hw": list(frame.mask.shape),
                }
                for label, frame in [
                    ("source", self.source_frame),
                    ("target", self.target_frame),
                ]
            },
        }


def pixel_center_resize_map(
    input_shape: tuple[int, int],
    output_shape: tuple[int, int],
    offset_xy: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Output pixel-center coordinates to the original uncropped input frame."""
    if (
        len(input_shape) != 2
        or len(output_shape) != 2
        or min(*input_shape, *output_shape) <= 0
    ):
        raise ValueError("positive two-dimensional image shapes required")
    sy, sx = np.asarray(input_shape, dtype=float) / np.asarray(
        output_shape, dtype=float
    )
    x, y = offset_xy
    return np.array(
        [[sx, 0, x + (sx - 1) / 2], [0, sy, y + (sy - 1) / 2], [0, 0, 1.0]], dtype=float
    )


def component_frame(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    ceiling: int = 2048,
    mode: str = "all_components",
    padding: float = 0.08,
) -> ImageFrame:
    """Crop supported components and bind the realized, anisotropic resize exactly."""
    import cv2

    rgb = np.asarray(image)
    m = np.asarray(mask, dtype=bool)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.shape[:2] != m.shape:
        raise ValueError("RGB image and two-dimensional tissue mask must agree")
    if not m.any():
        raise ValueError("No supported tissue")
    if mode not in {"dominant", "all_components", "supported_components"}:
        raise ValueError("unknown crop mode")
    extent = m
    if mode == "dominant":
        n, labels, stats, _ = cv2.connectedComponentsWithStats(m.astype("uint8"))
        extent = labels == (1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    yy, xx = np.nonzero(extent)
    pad = int(round(max(np.ptp(xx), np.ptp(yy)) * padding))
    x0 = max(0, int(xx.min()) - pad)
    y0 = max(0, int(yy.min()) - pad)
    x1 = min(m.shape[1], int(xx.max()) + pad + 1)
    y1 = min(m.shape[0], int(yy.max()) + pad + 1)
    h, w = y1 - y0, x1 - x0
    scale = min(1.0, ceiling / max(h, w))
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    arr = cv2.resize(rgb[y0:y1, x0:x1], (nw, nh), interpolation=cv2.INTER_AREA)
    cm = (
        cv2.resize(
            m[y0:y1, x0:x1].astype("uint8"), (nw, nh), interpolation=cv2.INTER_NEAREST
        )
        > 0
    )
    return ImageFrame(
        arr,
        cm,
        pixel_center_resize_map((h, w), (nh, nw), (x0, y0)),
        float(m[y0:y1, x0:x1].sum() / m.sum()),
        (x0, y0, w, h),
    )


def supported_tissue_mask(
    image: np.ndarray,
    base_mask: np.ndarray,
    proposals: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Admit pale additions only with native optical-density/texture support.

    This is a deterministic tissue proposal, not an independently validated mask.
    It never copies unsupported tissue from another section.
    """
    import cv2

    base = np.asarray(base_mask, dtype=bool)
    if not proposals:
        return base.copy()
    rgb = np.asarray(image, dtype="float32")
    gray = rgb.mean(axis=2)
    od = -np.log(np.maximum((rgb + 1) / 256, 1e-5)).mean(axis=2)
    mean = cv2.GaussianBlur(gray, (0, 0), 2)
    var = np.maximum(0, cv2.GaussianBlur(gray**2, (0, 0), 2) - mean**2)
    native = (od > 0.035) | ((np.sqrt(var) > 2.5) & (gray < 250))
    addition = np.zeros_like(base)
    for name in [
        "group_pale_tissue",
        "group_density_union",
        "background_corrected",
        "object_aware_fusion",
    ]:
        if name in proposals:
            arr = np.asarray(proposals[name], bool)
            if arr.shape != base.shape:
                raise ValueError("proposal geometry mismatch")
            addition |= arr & native
    result = base | addition
    n, labels, stats, _ = cv2.connectedComponentsWithStats(result.astype("uint8"))
    minimum = max(32, round(result.size * 0.00002))
    keep = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] >= minimum)
    keep = keep[keep != 0]
    return base | np.isin(labels, keep)


def structure_channels(
    image: np.ndarray, mask: np.ndarray, modality: str = "unknown"
) -> dict[str, np.ndarray]:
    """Fixed structural projections, never fitted to protein or landmark outcomes."""
    import cv2

    rgb = np.asarray(image, dtype="float32")
    m = np.asarray(mask, bool)
    od = -np.log(np.clip((rgb + 1) / 256, 1e-5, 1))
    h = np.array([0.650, 0.704, 0.286])
    second = (
        np.array([0.072, 0.990, 0.105])
        if modality.lower() in {"he", "h&e"}
        else np.array([0.268, 0.570, 0.776])
    )
    h /= np.linalg.norm(h)
    second /= np.linalg.norm(second)
    third = np.cross(h, second)
    third /= np.linalg.norm(third)
    basis = np.stack([h, second, third], axis=1)
    hematox = (od @ np.linalg.inv(basis).T)[..., 0]
    gray = cv2.cvtColor(rgb.astype("uint8"), cv2.COLOR_RGB2GRAY).astype("float32")

    def scaled(a):
        vals = a[m]
        if not len(vals):
            return np.zeros(m.shape, "uint8")
        low, high = np.percentile(vals, [1, 99])
        v = np.clip((a - low) / max(high - low, 1e-6) * 255, 0, 255).astype("uint8")
        return v

    g = scaled(255 - gray)
    hh = scaled(hematox)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return {"gray": clahe.apply(g), "hematoxylin": clahe.apply(hh)}


def prepare_structural_features(
    frame: ImageFrame, modality: str, config: GuardedRegistrationConfig
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    import cv2

    detector = cv2.SIFT_create(
        nfeatures=config.max_features, contrastThreshold=0.015, edgeThreshold=12
    )
    features = {}
    for name, channel in structure_channels(frame.image, frame.mask, modality).items():
        keypoints, desc = detector.detectAndCompute(
            channel, frame.mask.astype("uint8") * 255
        )
        xy = np.asarray([k.pt for k in keypoints], dtype="float32").reshape(-1, 2)
        if desc is None:
            desc = np.empty((0, 128), dtype="float32")
        # RootSIFT uses a Hellinger embedding, retaining the original SIFT keypoints.
        desc = np.sqrt(desc / (np.abs(desc).sum(axis=1, keepdims=True) + 1e-12)).astype(
            "float32"
        )
        features[name] = (xy, desc)
    return features


def mutual_ratio_matches(
    a: np.ndarray, b: np.ndarray, ratio: float = 0.75
) -> np.ndarray:
    """Descriptor-index correspondences with reciprocal ratio-test support."""
    import cv2

    if len(a) < 2 or len(b) < 2:
        return np.empty((0, 2), dtype=int)
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=64))
    ab = matcher.knnMatch(a, b, k=2)
    ba = matcher.knnMatch(b, a, k=2)
    reverse = {
        p[0].queryIdx: p[0].trainIdx
        for p in ba
        if len(p) == 2 and p[0].distance < ratio * p[1].distance
    }
    pairs = [
        (p[0].queryIdx, p[0].trainIdx)
        for p in ab
        if len(p) == 2
        and p[0].distance < ratio * p[1].distance
        and reverse.get(p[0].trainIdx) == p[0].queryIdx
    ]
    return np.asarray(pairs, dtype=int).reshape(-1, 2)


def candidate_from_points(
    source_xy: np.ndarray,
    target_xy: np.ndarray,
    *,
    stage: str,
    target_shape: tuple[int, int],
    config: GuardedRegistrationConfig,
) -> RegistrationCandidate | None:
    import cv2

    src = np.asarray(source_xy, dtype="float32")
    tgt = np.asarray(target_xy, dtype="float32")
    if src.shape != tgt.shape or src.ndim != 2 or src.shape[1] != 2:
        raise ValueError("paired XY arrays required")
    if len(src) < config.min_feature_inliers:
        return None
    M, inliers = cv2.estimateAffinePartial2D(
        src,
        tgt,
        method=cv2.RANSAC,
        ransacReprojThreshold=4.0,
        maxIters=5000,
        confidence=0.999,
        refineIters=10,
    )
    if M is None or inliers is None:
        return None
    n = int(inliers.sum())
    if n < config.min_feature_inliers:
        return None
    H = np.vstack([M, [0, 0, 1.0]])
    if not np.isfinite(H).all() or np.linalg.det(H[:2, :2]) <= 0:
        return None
    points = tgt[inliers[:, 0] > 0]
    bins = np.floor(points / np.array([target_shape[1], target_shape[0]]) * 4).astype(
        int
    )
    coverage = len(np.unique(np.clip(bins, 0, 3), axis=0)) / 16
    return RegistrationCandidate(H, stage, len(src), n, float(coverage))


def _nmi(a, b):
    hist = np.histogram2d(a, b, bins=32, range=[[0, 256], [0, 256]])[0]
    hist /= max(1, hist.sum())
    px = hist.sum(axis=1)
    py = hist.sum(axis=0)
    ha = -np.sum(px[px > 0] * np.log(px[px > 0]))
    hb = -np.sum(py[py > 0] * np.log(py[py > 0]))
    nz = hist > 0
    independent = px[:, None] * py[None, :]
    mi = np.sum(hist[nz] * np.log(hist[nz] / np.maximum(independent[nz], 1e-15)))
    return float(2 * mi / max(ha + hb, 1e-12))


def evaluate_candidate(
    candidate: RegistrationCandidate,
    source: ImageFrame,
    target: ImageFrame,
    source_channels: dict[str, np.ndarray],
    target_channels: dict[str, np.ndarray],
    source_physical: np.ndarray | None = None,
    target_physical: np.ndarray | None = None,
) -> None:
    """Image-only evidence; mask overlap and feature counts stay separate."""
    import cv2

    H = candidate.matrix
    if not np.isfinite(H).all() or abs(np.linalg.det(H[:2, :2])) < 1e-8:
        candidate.accepted = False
        candidate.rejection_reason = "invalid mapping"
        return
    native = target.working_to_input @ H @ np.linalg.inv(source.working_to_input)
    physical = (
        (target_physical @ native @ np.linalg.inv(source_physical))
        if source_physical is not None and target_physical is not None
        else native
    )
    singular = np.linalg.svd(physical[:2, :2], compute_uv=False)
    if (
        np.linalg.det(physical[:2, :2]) <= 0
        or singular.min() < 0.5
        or singular.max() > 2
        or singular.max() / singular.min() > 1.3
    ):
        candidate.accepted = False
        candidate.rejection_reason = "physical shape guard"
        return
    # Bound scoring cost independently of native image dimensions.
    th, tw = target.mask.shape
    factor = min(1.0, 600 / max(th, tw))
    nh, nw = max(1, round(th * factor)), max(1, round(tw * factor))
    G = pixel_center_resize_map((th, tw), (nh, nw))
    M = np.linalg.inv(G) @ H
    tm = (
        cv2.resize(
            target.mask.astype("uint8"), (nw, nh), interpolation=cv2.INTER_NEAREST
        )
        > 0
    )
    wm = (
        cv2.warpAffine(
            source.mask.astype("uint8"), M[:2], (nw, nh), flags=cv2.INTER_NEAREST
        )
        > 0
    )
    support = tm & wm
    overlap = int(support.sum())
    denom = int(tm.sum() + wm.sum())
    dice = 2 * overlap / max(1, denom)
    if overlap < 64:
        candidate.accepted = False
        candidate.rejection_reason = "insufficient common tissue"
        return
    values = []
    for name in ["gray", "hematoxylin"]:
        a = cv2.resize(target_channels[name], (nw, nh), interpolation=cv2.INTER_AREA)
        b = cv2.warpAffine(
            source_channels[name], M[:2], (nw, nh), flags=cv2.INTER_LINEAR
        )
        values.append(_nmi(a[support], b[support]))
    structure = float(np.mean(values))
    feature = min(1.0, np.log1p(candidate.feature_inlier_count) / np.log1p(100))
    score = (
        0.70 * structure
        + 0.15 * dice
        + 0.10 * candidate.feature_spatial_coverage
        + 0.05 * feature
    )
    candidate.diagnostics.update(
        mask_overlap_dice=float(dice),
        structural_nmi=structure,
        target_tissue_coverage=float(overlap / max(1, tm.sum())),
        physical_directional_scales=singular.tolist(),
        physical_area_ratio=float(np.linalg.det(physical[:2, :2])),
        selection_score=float(score),
    )


def estimate_guarded_registration(
    source: np.ndarray,
    target: np.ndarray,
    *,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    source_modality: str = "unknown",
    target_modality: str = "unknown",
    config: GuardedRegistrationConfig | None = None,
    source_proposals: dict[str, np.ndarray] | None = None,
    target_proposals: dict[str, np.ndarray] | None = None,
    source_physical: np.ndarray | None = None,
    target_physical: np.ndarray | None = None,
    extra_candidates: list[RegistrationCandidate] | None = None,
    prepared_source: dict | None = None,
    prepared_target: dict | None = None,
) -> GuardedRegistrationResult:
    """Estimate source-to-target alignment without consuming evaluation annotations."""
    import cv2

    from histopia.registration._rigid import (
        estimate_rigid_transform,
        refine_rigid_transform,
    )

    cfg = config or GuardedRegistrationConfig()
    cv2.setRNGSeed(cfg.seed)
    if cfg.matcher == "learned" and extra_candidates is None:
        raise ValueError(
            "Learned matching requires explicit pretrained matcher candidates"
        )
    sm = np.asarray(source_mask, bool)
    tm = np.asarray(target_mask, bool)
    if cfg.mask_mode == "supported_components":
        sm = supported_tissue_mask(source, sm, source_proposals)
        tm = supported_tissue_mask(target, tm, target_proposals)
    sf = component_frame(source, sm, ceiling=cfg.max_dim_px, mode=cfg.mask_mode)
    tf = component_frame(target, tm, ceiling=cfg.max_dim_px, mode=cfg.mask_mode)
    sc = structure_channels(sf.image, sf.mask, source_modality)
    tc = structure_channels(tf.image, tf.mask, target_modality)
    initial = estimate_rigid_transform(
        tf.image, sf.image, fixed_mask=tf.mask, moving_mask=sf.mask, refine=False
    )
    actual = initial.inlier_count if initial.method.startswith("feature:") else 0
    base = RegistrationCandidate(
        initial.matrix, "legacy_initializer", initial.match_count, actual
    )
    candidates = [base]
    legacy = refine_rigid_transform(tf.mask, sf.mask, initial)
    candidates.append(
        RegistrationCandidate(
            legacy.matrix, "legacy_mask_refinement", initial.match_count, actual
        )
    )
    if cfg.matcher != "legacy":
        fs = prepared_source or prepare_structural_features(sf, source_modality, cfg)
        ft = prepared_target or prepare_structural_features(tf, target_modality, cfg)
        for name in ["gray", "hematoxylin"]:
            idx = mutual_ratio_matches(fs[name][1], ft[name][1], cfg.match_ratio)
            c = candidate_from_points(
                fs[name][0][idx[:, 0]],
                ft[name][0][idx[:, 1]],
                stage="reciprocal_rootsift_" + name,
                target_shape=tf.mask.shape,
                config=cfg,
            )
            if c is not None:
                candidates.append(c)
    if extra_candidates:
        candidates.extend(extra_candidates)
    for c in candidates:
        evaluate_candidate(c, sf, tf, sc, tc, source_physical, target_physical)
    accepted = [c for c in candidates if c.accepted]
    if not accepted:
        raise ValueError(
            "No candidate has supported tissue and plausible physical geometry"
        )
    # Legacy ablations evaluate the legacy result without selection by stronger signals.
    selected = (
        next((c for c in reversed(candidates[:2]) if c.accepted), accepted[0])
        if cfg.matcher == "legacy"
        else max(accepted, key=lambda c: c.diagnostics["selection_score"])
    )
    if cfg.refine:
        ref = _structural_refinement(selected, sf, tf, sc, tc, cfg)
        if ref is not None:
            evaluate_candidate(ref, sf, tf, sc, tc, source_physical, target_physical)
            if (
                ref.accepted
                and ref.diagnostics["structural_nmi"]
                >= selected.diagnostics["structural_nmi"] + cfg.min_structure_gain
                and ref.diagnostics["mask_overlap_dice"]
                >= selected.diagnostics["mask_overlap_dice"] - 0.01
            ):
                selected = ref
            else:
                ref.accepted = False
                ref.rejection_reason = (
                    ref.rejection_reason or "no structural gain without tissue loss"
                )
            candidates.append(ref)
    M = tf.working_to_input @ selected.matrix @ np.linalg.inv(sf.working_to_input)
    return GuardedRegistrationResult(M, selected.stage, candidates, sf, tf, cfg)


def _structural_refinement(initial, source, target, sc, tc, cfg, *, max_dim=700):
    import cv2

    th, tw = target.mask.shape
    scale = min(1.0, max_dim / max(th, tw))
    nh, nw = max(1, round(th * scale)), max(1, round(tw * scale))
    G = pixel_center_resize_map((th, tw), (nh, nw))
    M = np.linalg.inv(G) @ initial.matrix
    fixed = (
        cv2.resize(tc["hematoxylin"], (nw, nh), interpolation=cv2.INTER_AREA).astype(
            "float32"
        )
        / 255
    )
    moving = cv2.warpAffine(sc["hematoxylin"], M[:2], (nw, nh)).astype("float32") / 255
    tm = cv2.resize(
        target.mask.astype("uint8"), (nw, nh), interpolation=cv2.INTER_NEAREST
    )
    wm = cv2.warpAffine(
        source.mask.astype("uint8"), M[:2], (nw, nh), flags=cv2.INTER_NEAREST
    )
    support = cv2.erode((tm & wm).astype("uint8") * 255, np.ones((3, 3), "uint8"))
    if np.count_nonzero(support) < 64:
        return None
    try:
        _, pull = cv2.findTransformECC(
            fixed,
            moving,
            np.eye(2, 3, dtype="float32"),
            cv2.MOTION_AFFINE,
            (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-5),
            support,
            5,
        )
        delta = np.linalg.inv(np.vstack([pull, [0, 0, 1.0]]))
        singular = np.linalg.svd(delta[:2, :2], compute_uv=False)
        if (
            singular.min() < 1 - cfg.max_relative_scale_change
            or singular.max() > 1 + cfg.max_relative_scale_change
            or singular.max() / singular.min() > cfg.max_relative_anisotropy
        ):
            return None
        refined = G @ delta @ M
    except (cv2.error, np.linalg.LinAlgError):
        return None
    return RegistrationCandidate(
        refined,
        initial.stage + "+structural_affine",
        initial.feature_match_count,
        initial.feature_inlier_count,
        initial.feature_spatial_coverage,
    )
