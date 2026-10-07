"""Image-only selection and refinement of independently initialized alignments.

Candidate transforms must be obtained without evaluation annotations. This
experimental interface deliberately has no landmark or protein input. It uses
one common image frame so that candidate scores are directly comparable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._guarded import (
    GuardedRegistrationConfig,
    GuardedRegistrationResult,
    RegistrationCandidate,
    _structural_refinement,
    component_frame,
    evaluate_candidate,
    structure_channels,
)


@dataclass(frozen=True, slots=True)
class AlignmentSeed:
    name: str
    matrix: np.ndarray


def dark_annotation_proposals(image: np.ndarray) -> np.ndarray:
    """Flag large, nearly black, neutral components as possible scan annotations.

    This conservative flag is only a proposal. Dark biological pigment can also
    be flagged, so callers must preserve the original mask and this exclusion
    separately. It is not an independently validated artifact detector.
    """
    import cv2

    rgb = np.asarray(image)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("RGB uint8 image required")
    dark = (rgb.max(axis=2) < 65) & (np.ptp(rgb.astype(float), axis=2) < 18)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(dark.astype("uint8"))
    minimum = max(64, int(rgb.shape[0] * rgb.shape[1] * 0.00015))
    valid = []
    for i in range(1, len(stats)):
        _, _, w, h, area = stats[i]
        if area >= minimum and min(w, h) >= 8:
            valid.append(i)
    mask = np.isin(labels, valid).astype("uint8")
    return cv2.dilate(mask, np.ones((7, 7), "uint8")) > 0


def refine_registration_candidates(
    source: np.ndarray,
    target: np.ndarray,
    seeds: list[AlignmentSeed],
    *,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    source_modality: str = "unknown",
    target_modality: str = "unknown",
    source_physical: np.ndarray | None = None,
    target_physical: np.ndarray | None = None,
    config: GuardedRegistrationConfig | None = None,
    refinement_sizes: tuple[int, ...] = (700,),
) -> GuardedRegistrationResult:
    """Refine all distinct starts, selecting by common-frame image evidence.

    Feature counts do not enter selection because seeds from different methods
    may report incompatible correspondence statistics. Every accepted candidate
    is scored on the same structural projections and original tissue masks.
    """
    cfg = config or GuardedRegistrationConfig(mask_mode="all_components")
    if not refinement_sizes or any(
        size < 64 or size > cfg.max_dim_px for size in refinement_sizes
    ):
        raise ValueError("refinement sizes must stay within the working-image ceiling")
    if not seeds:
        raise ValueError("at least one image-derived initialization is required")
    sf = component_frame(source, source_mask, ceiling=cfg.max_dim_px)
    tf = component_frame(target, target_mask, ceiling=cfg.max_dim_px)
    sc = structure_channels(sf.image, sf.mask, source_modality)
    tc = structure_channels(tf.image, tf.mask, target_modality)
    candidates: list[RegistrationCandidate] = []
    for seed in seeds:
        matrix = np.asarray(seed.matrix, dtype=float)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError("seeds require finite homogeneous 3 by 3 matrices")
        if not np.allclose(matrix[2], [0, 0, 1]):
            raise ValueError("only affine seed matrices are supported")
        working = np.linalg.inv(tf.working_to_input) @ matrix @ sf.working_to_input
        if any(np.allclose(working, c.matrix, atol=1e-5) for c in candidates):
            continue
        candidate = RegistrationCandidate(working, seed.name)
        evaluate_candidate(candidate, sf, tf, sc, tc, source_physical, target_physical)
        candidates.append(candidate)
        if candidate.accepted and cfg.refine:
            current = candidate
            for size in refinement_sizes:
                ref = _structural_refinement(current, sf, tf, sc, tc, cfg, max_dim=size)
                if ref is not None:
                    ref.stage += f"@{size}px"
                    evaluate_candidate(
                        ref, sf, tf, sc, tc, source_physical, target_physical
                    )
                    # Preserve each earlier alternative. Refinement is not forced.
                    candidates.append(ref)
                    if (
                        ref.accepted
                        and ref.diagnostics["selection_score"]
                        > current.diagnostics["selection_score"]
                    ):
                        current = ref
    accepted = [c for c in candidates if c.accepted]
    if not accepted:
        raise ValueError("no supported, physically plausible candidate")
    # A fixed Dice weight provides weak support regularization, not a surrogate
    # for anatomical accuracy. Neither landmark errors nor tissue outlines enter.
    selected = max(accepted, key=lambda c: c.diagnostics["selection_score"])
    matrix = tf.working_to_input @ selected.matrix @ np.linalg.inv(sf.working_to_input)
    return GuardedRegistrationResult(matrix, selected.stage, candidates, sf, tf, cfg)
