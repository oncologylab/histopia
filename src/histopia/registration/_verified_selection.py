"""Experimental, image-only selection with spatially separated feature checks.

Historical registration defaults are unchanged. Imported hypotheses may have
seen these features; verification is not independent anatomical validation.
Unknown stains use a generic gradient channel, never assumed DAB chemistry.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._adaptive import UnsupportedRegistration, support_evidence
from ._candidate_refinement import AlignmentSeed
from ._guarded import (
    GuardedRegistrationConfig,
    GuardedRegistrationResult,
    RegistrationCandidate,
    candidate_from_points,
    component_frame,
    evaluate_candidate,
    mutual_ratio_matches,
    structure_channels,
)


@dataclass(frozen=True)
class VerifiedSelectionConfig:
    """Frozen image-only settings; no evaluation annotations are accepted."""

    corrected_modality: bool = True
    verified_selection: bool = True
    generate_features: bool = True
    abstain_on_incoherent_features: bool = False
    minimum_matches: int = 8
    minimum_cells: int = 3
    tolerance_px: float = 6.0
    minimum_gain: float = 0.01
    seed: int = 20261010

    def __post_init__(self):
        if self.minimum_matches < 3 or not 1 <= self.minimum_cells <= 16:
            raise ValueError("invalid verification support")
        if not np.isfinite([self.tolerance_px, self.minimum_gain]).all():
            raise ValueError("finite selection settings required")
        if self.tolerance_px <= 0 or self.minimum_gain < 0:
            raise ValueError("invalid selection tolerances")


def modality_channels(image, mask, modality, *, corrected=True):
    """Use known H&E/IHC chemistry or generic intensity and gradient evidence.

    The second key is retained for compatibility with the common-frame scorer;
    for unknown modality it holds gradient magnitude, not hematoxylin concentration.
    """
    import cv2

    known = modality.lower() in {"he", "h&e", "ihc", "h-dab", "hdab"}
    channels = structure_channels(image, mask, modality if corrected else "unknown")
    if corrected and not known:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype("float32")
        gradient = cv2.magnitude(
            cv2.Sobel(gray, cv2.CV_32F, 1, 0),
            cv2.Sobel(gray, cv2.CV_32F, 0, 1),
        )
        values = gradient[np.asarray(mask, bool)]
        high = float(np.percentile(values, 99)) if len(values) else 1.0
        channels["hematoxylin"] = np.clip(
            gradient / max(high, 1e-6) * 255, 0, 255
        ).astype("uint8")
    return channels


def spatial_partition(points, shape_hw):
    """Checkerboard cells for fitting versus verification; independent of names."""
    xy = np.asarray(points, float)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError("finite XY features required")
    cells = np.clip(np.floor(xy / np.array(shape_hw[::-1]) * 4).astype(int), 0, 3)
    return (cells[:, 0] + cells[:, 1]) % 2, cells


def correspondence_evidence(
    matrix, source_xy, target_xy, source_shape, target_shape, *, tolerance_px=6.0
):
    """Symmetric residual and coverage for image features, not annotated landmarks."""
    a, b = np.asarray(source_xy, float), np.asarray(target_xy, float)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 2:
        raise ValueError("paired XY arrays required")
    h = np.asarray(matrix, float)
    if (
        h.shape != (3, 3)
        or not np.isfinite(h).all()
        or not np.allclose(h[2], [0, 0, 1])
    ):
        raise ValueError("finite affine mapping required")
    inverse = np.linalg.inv(h)
    forward = np.linalg.norm((np.c_[a, np.ones(len(a))] @ h.T)[:, :2] - b, axis=1)
    reverse = np.linalg.norm((np.c_[b, np.ones(len(b))] @ inverse.T)[:, :2] - a, axis=1)
    # Scale each direction in its own working frame, so anisotropic image sizes
    # do not make reverse verification spuriously easier or harder.
    reverse *= np.hypot(*target_shape) / np.hypot(*source_shape)
    supported = np.maximum(forward, reverse) <= tolerance_px
    _, ac = spatial_partition(a, source_shape)
    _, bc = spatial_partition(b, target_shape)
    occupied = min(
        len(np.unique(ac[supported], axis=0)), len(np.unique(bc[supported], axis=0))
    )
    return {
        "verification_matches": len(a),
        "verification_supported": int(supported.sum()),
        "verification_fraction": float(supported.mean()) if len(a) else None,
        "verification_cells": occupied,
        "verification_median_residual_px": float(
            np.median(np.maximum(forward, reverse))
        )
        if len(a)
        else None,
    }


def _matches(source, target, sc, tc, config):
    import cv2

    detector = cv2.SIFT_create(
        nfeatures=6000, contrastThreshold=0.015, edgeThreshold=12
    )
    fits, checks = [], []
    for name in sc:
        features = []
        for frame, channels in [(source, sc), (target, tc)]:
            kp, desc = detector.detectAndCompute(
                channels[name], frame.mask.astype("uint8") * 255
            )
            xy = np.asarray([k.pt for k in kp], "float32").reshape(-1, 2)
            if desc is None:
                desc = np.empty((0, 128), "float32")
            desc = np.sqrt(desc / (np.abs(desc).sum(1, keepdims=True) + 1e-12)).astype(
                "float32"
            )
            features.append((xy, desc))
        pairs = mutual_ratio_matches(features[0][1], features[1][1])
        a, b = features[0][0][pairs[:, 0]], features[1][0][pairs[:, 1]]
        pa, _ = spatial_partition(a, source.mask.shape)
        pb, _ = spatial_partition(b, target.mask.shape)
        fitting, checking = (pa == 0) & (pb == 0), (pa == 1) & (pb == 1)
        fits.append((name, a[fitting], b[fitting]))
        checks.append((a[checking], b[checking]))
    return (
        fits,
        np.concatenate([x[0] for x in checks]),
        np.concatenate([x[1] for x in checks]),
    )


def select_verified_registration(
    source,
    target,
    seeds: list[AlignmentSeed],
    *,
    source_mask,
    target_mask,
    source_physical,
    target_physical,
    source_modality="unknown",
    target_modality="unknown",
    predecessor="previous",
    config=None,
):
    """Rank a common candidate pool, preserving a supported predecessor on ambiguity.

    Feature fitting and verification occupy disjoint cells in both images. Seeds
    from earlier pipelines are explicitly not claimed to be independent of checks.
    No outcome-dependent per-specimen thresholds or evaluation landmarks enter here.
    """
    import cv2

    cfg = config or VerifiedSelectionConfig()
    cv2.setRNGSeed(cfg.seed)
    guarded = GuardedRegistrationConfig(seed=cfg.seed, mask_mode="all_components")
    sf, tf = component_frame(source, source_mask), component_frame(target, target_mask)
    sc, tc = [
        modality_channels(f.image, f.mask, m, corrected=cfg.corrected_modality)
        for f, m in [(sf, source_modality), (tf, target_modality)]
    ]
    fits, va, vb = _matches(sf, tf, sc, tc, cfg)
    candidates = []
    for seed in seeds:
        matrix = np.linalg.inv(tf.working_to_input) @ seed.matrix @ sf.working_to_input
        candidates.append(RegistrationCandidate(matrix, seed.name))
    if cfg.generate_features:
        for name, a, b in fits:
            c = candidate_from_points(
                a,
                b,
                stage="spatial_fit_" + name,
                target_shape=tf.mask.shape,
                config=guarded,
            )
            if c is not None:
                candidates.append(c)
    for c in candidates:
        evaluate_candidate(c, sf, tf, sc, tc, source_physical, target_physical)
        if not c.accepted:
            continue
        c.diagnostics.update(support_evidence(c.matrix, sf.mask, tf.mask))
        c.diagnostics.update(
            correspondence_evidence(
                c.matrix,
                va,
                vb,
                sf.mask.shape,
                tf.mask.shape,
                tolerance_px=cfg.tolerance_px,
            )
        )
        d = c.diagnostics
        d["verification_independent_of_legacy_seeds"] = False
        if not cfg.verified_selection:
            continue
        scales = np.asarray(d["physical_directional_scales"])
        if scales.min() < 0.65 or scales.max() > 1.35:
            c.accepted, c.rejection_reason = False, "excessive physical scale change"
            continue
        sufficient = d["verification_matches"] >= cfg.minimum_matches
        d["distributed_support"] = (
            d["verification_supported"] >= cfg.minimum_matches
            and d["verification_cells"] >= cfg.minimum_cells
        )
        geometric = d["verification_fraction"] if sufficient else 0.0
        # Strong distributed checks take precedence over tissue silhouette.
        d["selection_score"] = (
            0.60 * geometric
            + 0.30 * d["structural_nmi"]
            + 0.10 * d["full_extent_dice"]
            - 0.05 * float(np.abs(np.log(scales)).sum())
        )
        if sufficient and d["verification_supported"] == 0:
            c.accepted, c.rejection_reason = (
                False,
                "contradicted by verification features",
            )
    incoherent = cfg.abstain_on_incoherent_features and not any(
        c.diagnostics.get("distributed_support", False) for c in candidates
    )
    if cfg.verified_selection and incoherent:
        # Unmatched descriptors are not evidence that every geometric hypothesis
        # is wrong. Only restore candidates that passed all other guardrails.
        for c in candidates:
            if c.rejection_reason == "contradicted by verification features":
                c.accepted, c.rejection_reason = True, None
    accepted = [c for c in candidates if c.accepted]
    diagnostics = {
        "modality": {"source": source_modality, "target": target_modality},
        "evaluation_landmarks_used": False,
        "independent_anatomical_accuracy": False,
    }
    if not accepted:
        raise UnsupportedRegistration(
            "no supported verified candidate", candidates, diagnostics
        )
    best = max(accepted, key=lambda c: c.diagnostics["selection_score"])
    previous = next((c for c in accepted if c.stage == predecessor), None)
    if cfg.verified_selection and previous is not None and best is not previous:
        gain = (
            best.diagnostics["selection_score"]
            - previous.diagnostics["selection_score"]
        )
        if (
            incoherent
            or gain < cfg.minimum_gain
            or (
                not best.diagnostics["distributed_support"]
                and best.diagnostics["structural_nmi"]
                <= previous.diagnostics["structural_nmi"]
            )
        ):
            best = previous
            diagnostics["predecessor_retention"] = "insufficient supported improvement"
    if incoherent:
        diagnostics["feature_evidence"] = "inconclusive; no distributed consensus"
    diagnostics["verification_scope"] = (
        "image features; imported seeds may have used them"
    )
    native = tf.working_to_input @ best.matrix @ np.linalg.inv(sf.working_to_input)
    return GuardedRegistrationResult(
        native, best.stage, candidates, sf, tf, guarded
    ), diagnostics
