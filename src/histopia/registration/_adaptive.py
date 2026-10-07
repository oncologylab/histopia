"""Experimental image-only recovery with explicit support and abstention.

No evaluation annotations enter this module. Geometric feature verification is
an algorithmic diagnostic, not independent anatomical validation. Historical
registration defaults are unchanged. Heavy backends are imported only on use.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ._candidate_refinement import AlignmentSeed, dark_annotation_proposals
from ._guarded import (
    GuardedRegistrationConfig,
    GuardedRegistrationResult,
    RegistrationCandidate,
    _structural_refinement,
    candidate_from_points,
    component_frame,
    evaluate_candidate,
    mutual_ratio_matches,
    pixel_center_resize_map,
    prepare_structural_features,
    structure_channels,
)


@dataclass(frozen=True, slots=True)
class AdaptiveConfig:
    exclude_isolated_marks: bool = False
    balanced_selection: bool = True
    recover_initialization: bool = True
    rescue_mode: str = "adaptive"
    refinement_sizes: tuple[int, ...] = (700, 1400)
    seed: int = 20261006

    def __post_init__(self):
        if self.rescue_mode not in {"off", "adaptive", "always"}:
            raise ValueError("unknown rescue mode")
        if not self.refinement_sizes or any(
            not 64 <= x <= 2048 for x in self.refinement_sizes
        ):
            raise ValueError("refinement sizes must be within 64..2048")


class UnsupportedRegistration(ValueError):
    """Unsuccessful registration retaining every attempted hypothesis."""

    def __init__(self, message, candidates, diagnostics):
        super().__init__(message)
        self.candidates = candidates
        self.diagnostics = diagnostics


def isolated_annotation_exclusion(image, tissue_mask):
    """Exclude only dark neutral proposals surrounded predominantly by glass.

    Proposals remain separate from exclusions. This heuristic is experimental:
    it does not claim to recognize all ink or distinguish every dark pigment.
    """
    import cv2

    rgb = np.asarray(image)
    mask = np.asarray(tissue_mask, bool)
    if mask.shape != rgb.shape[:2]:
        raise ValueError("mask geometry differs from image")
    proposal = dark_annotation_proposals(rgb)
    _, labels = cv2.connectedComponents(proposal.astype("uint8"))
    excluded = np.zeros(mask.shape, bool)
    # Ring tests use native image evidence, not a mask already polluted by ink.
    od = -np.log(np.clip((rgb.astype(float) + 1) / 256, 1e-5, 1)).mean(axis=2)
    chroma = np.ptp(rgb.astype(float), axis=2)
    tissue_evidence = (od > 0.12) | (chroma > 15)
    for index in range(1, int(labels.max()) + 1):
        component = labels == index
        ring = cv2.dilate(component.astype("uint8"), np.ones((19, 19), "uint8")) > 0
        ring &= ~proposal
        if ring.sum() >= 64 and float(tissue_evidence[ring].mean()) < 0.10:
            excluded |= component
    return proposal, excluded


def support_evidence(matrix, source_mask, target_mask):
    """Measure overlap without discarding source tissue outside the target canvas."""
    import cv2

    a, b = np.asarray(source_mask, bool), np.asarray(target_mask, bool)
    h = np.asarray(matrix, float)
    if (
        h.shape != (3, 3)
        or not np.isfinite(h).all()
        or not np.allclose(h[2], [0, 0, 1])
    ):
        raise ValueError("finite affine source-to-target matrix required")
    area = abs(float(np.linalg.det(h[:2, :2])))
    if area < 1e-10:
        raise ValueError("singular transform")
    warped = (
        cv2.warpAffine(
            a.astype("uint8"), h[:2], (b.shape[1], b.shape[0]), flags=cv2.INTER_NEAREST
        )
        > 0
    )
    overlap = int(np.count_nonzero(warped & b))
    expected_source = max(1.0, float(a.sum()) * area)
    source_coverage = min(1.0, overlap / expected_source)
    target_coverage = overlap / max(1, int(b.sum()))
    return {
        "source_tissue_coverage": float(source_coverage),
        "target_tissue_coverage": float(target_coverage),
        "source_canvas_retention": min(1.0, float(warped.sum()) / expected_source),
        "full_extent_dice": min(1.0, 2 * overlap / max(1.0, expected_source + b.sum())),
        "common_pixels": overlap,
    }


def balanced_evaluate(candidate, sf, tf, sc, tc, sp, tp, verification):
    """Symmetric structural evidence and descriptor consistency.

    Verification descriptors are excluded from new RootSIFT hypotheses only;
    imported legacy hypotheses may have used them. This is not independent QC.
    """

    evaluate_candidate(candidate, sf, tf, sc, tc, sp, tp)
    if not candidate.accepted:
        return
    reverse = RegistrationCandidate(np.linalg.inv(candidate.matrix), "reverse_check")
    evaluate_candidate(reverse, tf, sf, tc, sc, tp, sp)
    if not reverse.accepted:
        candidate.accepted = False
        candidate.rejection_reason = "reverse support: " + str(reverse.rejection_reason)
        return
    evidence = support_evidence(candidate.matrix, sf.mask, tf.mask)
    # Both directional image comparisons use their own supported original frame.
    structure = (
        candidate.diagnostics["structural_nmi"] + reverse.diagnostics["structural_nmi"]
    ) / 2
    num, supported = 0, 0
    for src, tgt in verification:
        pred = (np.c_[src, np.ones(len(src))] @ candidate.matrix.T)[:, :2]
        num += len(src)
        supported += int(np.count_nonzero(np.linalg.norm(pred - tgt, axis=1) <= 6.0))
    fraction = supported / max(1, num)
    # The geometric term is omitted, not treated as a zero, when unavailable.
    score = 0.65 * structure + 0.20 * evidence["full_extent_dice"]
    score = score + 0.15 * fraction if num >= 6 else score / 0.85
    candidate.diagnostics.update(
        evidence,
        symmetric_structural_nmi=float(structure),
        verification_matches=num,
        verification_supported=supported,
        verification_fraction=float(fraction),
        selection_score=float(score),
    )


def orientation_seeds(sf, tf, source_physical, target_physical):
    """Twelve physical rotations with tissue-centroid translation, no reflections."""
    ps = source_physical @ sf.working_to_input
    pt = target_physical @ tf.working_to_input
    sy, sx = np.nonzero(sf.mask)
    ty, tx = np.nonzero(tf.mask)
    cs = ps @ [sx.mean(), sy.mean(), 1]
    ct = pt @ [tx.mean(), ty.mean(), 1]
    candidates = []
    for degrees in range(0, 360, 30):
        theta = np.deg2rad(degrees)
        r = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        physical = np.eye(3)
        physical[:2, :2] = r
        physical[:2, 2] = ct[:2] - r @ cs[:2]
        candidates.append(
            RegistrationCandidate(
                np.linalg.inv(pt) @ physical @ ps, f"physical_orientation_{degrees}"
            )
        )
    return candidates


def mutual_information_refine(candidate, sf, tf, sc, tc, *, seed=20261006):
    """SimpleITK Mattes-MI Euler residual after a physical global initialization.

    SimpleITK estimates a target-to-moving pull mapping; inversion and the exact
    pixel-center resize map are required before composing with the initializer.
    """
    import cv2
    import SimpleITK as sitk

    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(6)
    th, tw = tf.mask.shape
    scale = min(1.0, 384 / max(th, tw))
    nh, nw = max(1, round(th * scale)), max(1, round(tw * scale))
    g = pixel_center_resize_map((th, tw), (nh, nw))
    m = np.linalg.inv(g) @ candidate.matrix
    fixed = cv2.resize(tc["gray"], (nw, nh)).astype("float32") / 255
    moving = cv2.warpAffine(sc["gray"], m[:2], (nw, nh)).astype("float32") / 255
    tm = cv2.resize(tf.mask.astype("uint8"), (nw, nh), interpolation=cv2.INTER_NEAREST)
    sm = cv2.warpAffine(
        sf.mask.astype("uint8"), m[:2], (nw, nh), flags=cv2.INTER_NEAREST
    )
    if np.count_nonzero(tm & sm) < 64:
        return None
    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(32)
    registration.SetMetricSamplingStrategy(registration.NONE)
    registration.SetMetricFixedMask(sitk.GetImageFromArray(tm))
    registration.SetMetricMovingMask(sitk.GetImageFromArray(sm))
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsRegularStepGradientDescent(
        2.0, 0.02, 120, gradientMagnitudeTolerance=1e-6
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    initial = sitk.Euler2DTransform()
    initial.SetCenter(((nw - 1) / 2, (nh - 1) / 2))
    registration.SetInitialTransform(initial)
    fitted = registration.Execute(
        sitk.GetImageFromArray(fixed), sitk.GetImageFromArray(moving)
    )
    origin = np.array(fitted.TransformPoint((0.0, 0.0)))
    pull = np.eye(3)
    pull[:2, 2] = origin
    pull[:2, 0] = np.array(fitted.TransformPoint((1.0, 0.0))) - origin
    pull[:2, 1] = np.array(fitted.TransformPoint((0.0, 1.0))) - origin
    result = RegistrationCandidate(
        g @ np.linalg.inv(pull) @ m, candidate.stage + "+simpleitk_mi"
    )
    result.diagnostics["mi_optimizer_stop"] = (
        registration.GetOptimizerStopConditionDescription()
    )
    return result


def estimate_adaptive_registration(
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
    config=None,
):
    """Preserve alternatives, recover from empty seeds, and retain failed evidence."""
    import cv2

    cfg = config or AdaptiveConfig()
    cv2.setRNGSeed(cfg.seed)
    guarded = GuardedRegistrationConfig(mask_mode="all_components", seed=cfg.seed)
    diagnostics = {"settings": asdict(cfg), "escalation": [], "mask_evidence": {}}
    arrays, masks = [], []
    for name, rgb, mask in [
        ("source", source, source_mask),
        ("target", target, target_mask),
    ]:
        rgb, mask = np.asarray(rgb).copy(), np.asarray(mask, bool).copy()
        proposal, exclude = isolated_annotation_exclusion(rgb, mask)
        diagnostics["mask_evidence"][name] = {
            "proposal_pixels": int(proposal.sum()),
            "excluded_pixels": int(exclude.sum()) if cfg.exclude_isolated_marks else 0,
            "original_tissue_pixels": int(mask.sum()),
            "automatic_exclusions_are_experimental": True,
        }
        if cfg.exclude_isolated_marks:
            mask &= ~exclude
            rgb[exclude] = 255
        arrays.append(rgb)
        masks.append(mask)
    try:
        sf, tf = [
            component_frame(a, m, ceiling=2048)
            for a, m in zip(arrays, masks, strict=True)
        ]
    except ValueError as e:
        raise UnsupportedRegistration(str(e), [], diagnostics) from e
    sc, tc = [
        structure_channels(f.image, f.mask, mod)
        for f, mod in [(sf, source_modality), (tf, target_modality)]
    ]
    candidates, verification = [], []
    if cfg.recover_initialization or cfg.balanced_selection:
        features = [
            prepare_structural_features(f, mod, guarded)
            for f, mod in [(sf, source_modality), (tf, target_modality)]
        ]
        for channel in ["gray", "hematoxylin"]:
            a, b = features[0][channel], features[1][channel]
            indices = mutual_ratio_matches(a[1], b[1])
            fit, verify = indices[::2], indices[1::2]
            verification.append((a[0][verify[:, 0]], b[0][verify[:, 1]]))
            if cfg.recover_initialization:
                c = candidate_from_points(
                    a[0][fit[:, 0]],
                    b[0][fit[:, 1]],
                    stage="rootsift_fit_half_" + channel,
                    target_shape=tf.mask.shape,
                    config=guarded,
                )
                if c is not None:
                    candidates.append(c)
    for seed in seeds:
        m = np.asarray(seed.matrix, float)
        if (
            m.shape != (3, 3)
            or not np.isfinite(m).all()
            or not np.allclose(m[2], [0, 0, 1])
        ):
            raise ValueError("finite affine seed required")
        h = np.linalg.inv(tf.working_to_input) @ m @ sf.working_to_input
        if not any(np.allclose(h, c.matrix, atol=1e-5) for c in candidates):
            candidates.append(RegistrationCandidate(h, seed.name))

    def score(c):
        if cfg.balanced_selection:
            balanced_evaluate(
                c, sf, tf, sc, tc, source_physical, target_physical, verification
            )
        else:
            evaluate_candidate(c, sf, tf, sc, tc, source_physical, target_physical)

    for c in candidates:
        score(c)
    accepted = sorted(
        [c for c in candidates if c.accepted],
        key=lambda c: c.diagnostics["selection_score"],
        reverse=True,
    )
    reasons = []
    if not accepted:
        reasons.append("no_accepted_initialization")
    else:
        top = accepted[0]
        if top.diagnostics["target_tissue_coverage"] < 0.35:
            reasons.append("low_target_support")
        if top.diagnostics["structural_nmi"] < 0.08:
            reasons.append("weak_structural_agreement")
        if (
            top.diagnostics.get("verification_matches", 0) >= 6
            and top.diagnostics["verification_fraction"] < 0.5
        ):
            reasons.append("weak_distributed_feature_agreement")
        for other in accepted[1:]:
            pts = np.array([[0, 0, 1], [tf.mask.shape[1], tf.mask.shape[0], 1]], float)
            difference = np.linalg.norm(
                (pts @ (top.matrix - other.matrix).T)[:, :2], axis=1
            ).mean()
            if (
                difference > 0.02 * np.hypot(*tf.mask.shape)
                and top.diagnostics["selection_score"]
                - other.diagnostics["selection_score"]
                < 0.02
            ):
                reasons.append("competing_geometry")
                break
    diagnostics["rescue_reasons"] = reasons
    if cfg.rescue_mode == "always" or (cfg.rescue_mode == "adaptive" and reasons):
        diagnostics["escalation"].append("physical_multiangle_simpleitk_mi")
        for c in orientation_seeds(sf, tf, source_physical, target_physical):
            score(c)
            candidates.append(c)
            try:
                refined = mutual_information_refine(c, sf, tf, sc, tc, seed=cfg.seed)
                if refined is not None:
                    score(refined)
                    candidates.append(refined)
            except (RuntimeError, ImportError, np.linalg.LinAlgError) as e:
                c.diagnostics["mi_failure"] = f"{type(e).__name__}: {e}"
    # Refine only the three best global hypotheses, preserving their predecessors.
    accepted = sorted(
        [c for c in candidates if c.accepted],
        key=lambda c: c.diagnostics["selection_score"],
        reverse=True,
    )
    for initial in accepted[:3]:
        current = initial
        for size in cfg.refinement_sizes:
            refined = _structural_refinement(
                current, sf, tf, sc, tc, guarded, max_dim=size
            )
            if refined is None:
                current.diagnostics.setdefault(
                    "unavailable_refinement_sizes", []
                ).append(size)
                continue
            refined.stage += f"@{size}px"
            score(refined)
            candidates.append(refined)
            if refined.accepted:
                gain = (
                    refined.diagnostics["selection_score"]
                    - current.diagnostics["selection_score"]
                )
                retained = refined.diagnostics.get(
                    "full_extent_dice", refined.diagnostics["mask_overlap_dice"]
                )
                previous = current.diagnostics.get(
                    "full_extent_dice", current.diagnostics["mask_overlap_dice"]
                )
                if gain >= guarded.min_structure_gain and retained >= previous - 0.01:
                    current = refined
                else:
                    refined.accepted = False
                    refined.rejection_reason = (
                        "no material structural gain without tissue loss"
                    )
    accepted = [c for c in candidates if c.accepted]
    if not accepted:
        raise UnsupportedRegistration(
            "no supported, physically plausible candidate", candidates, diagnostics
        )
    best = max(accepted, key=lambda c: c.diagnostics["selection_score"])
    matrix = tf.working_to_input @ best.matrix @ np.linalg.inv(sf.working_to_input)
    result = GuardedRegistrationResult(matrix, best.stage, candidates, sf, tf, guarded)
    return result, diagnostics
