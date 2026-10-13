"""Opt-in agreement between separately fitted matcher families and directions.

Checks use image features, not anatomical reference labels. Imported hypotheses
may have seen checking features. A reverse registration is fitted from a new
matcher call with reversed inputs; an inverse is never counted as a reverse fit.
"""

from __future__ import annotations

import numpy as np

from ._adaptive import UnsupportedRegistration, support_evidence
from ._candidate_refinement import AlignmentSeed
from ._dense_matching import dense_candidates
from ._guarded import (
    GuardedRegistrationConfig,
    GuardedRegistrationResult,
    RegistrationCandidate,
    candidate_from_points,
    component_frame,
    evaluate_candidate,
)
from ._verified_selection import (
    VerifiedSelectionConfig,
    _matches,
    modality_channels,
    partition_correspondences,
    predecessor_binding,
    spatial_partition,
    transform_fingerprint,
)


def _mapped(matrix, points):
    return (np.c_[points, np.ones(len(points))] @ matrix.T)[:, :2]


def _affine(matrix):
    value = np.asarray(matrix, float)
    if (
        value.shape != (3, 3)
        or not np.isfinite(value).all()
        or not np.allclose(value[2], [0, 0, 1])
        or abs(np.linalg.det(value[:2, :2])) < 1e-12
    ):
        raise ValueError("finite, nonsingular affine mapping required")
    return value


def directional_correspondence_evidence(
    matrix, source_xy, target_xy, source_shape, target_shape, *, tolerance_px=6.0
):
    """Distinct feature support, using each residual in its own working frame.

    The inverse is used only for residual measurement, never as independent
    reverse-fit evidence. Duplicate endpoints occupy one working-pixel bin.
    """
    if not np.isfinite(tolerance_px) or tolerance_px <= 0:
        raise ValueError("positive finite residual tolerance required")
    fa, fb, ca, cb = partition_correspondences(
        source_xy, target_xy, source_shape, target_shape, policy="source_spatial"
    )
    a, b = np.concatenate([fa, ca]), np.concatenate([fb, cb])
    h = _affine(matrix)
    forward = np.linalg.norm(_mapped(h, a) - b, axis=1)
    reverse = np.linalg.norm(_mapped(np.linalg.inv(h), b) - a, axis=1)
    supported = (forward <= tolerance_px) & (reverse <= tolerance_px)
    _, ac = spatial_partition(a, source_shape)
    _, bc = spatial_partition(b, target_shape)
    return {
        "verification_matches": len(a),
        "verification_supported": int(supported.sum()),
        "verification_cells": min(
            len(np.unique(ac[supported], axis=0)),
            len(np.unique(bc[supported], axis=0)),
        ),
        "verification_fraction": float(supported.mean()) if len(a) else None,
        "forward_median_residual_px": float(np.median(forward)) if len(a) else None,
        "reverse_median_residual_px": float(np.median(reverse)) if len(a) else None,
        "residual_frames": "each direction's destination working frame",
    }


def independent_direction_agreement(
    forward,
    reverse,
    source_xy,
    target_xy,
    *,
    source_working_to_physical,
    target_working_to_physical,
    tolerance_px=6.0,
):
    """Round-trip agreement of separately estimated maps in both original frames.

    Caller must supply an independently fitted reverse map. Calibration converts
    the full XY residual, preserving anisotropy; physical units follow supplied
    calibration. Agreement alone never establishes anatomical accuracy.
    """
    f, r = _affine(forward), _affine(reverse)
    sp, tp = _affine(source_working_to_physical), _affine(target_working_to_physical)
    if not np.isfinite(tolerance_px) or tolerance_px <= 0:
        raise ValueError("positive finite agreement tolerance required")
    values = {}
    supported = True
    for name, points, cycle, calibration in [
        ("source", source_xy, r @ f, sp),
        ("target", target_xy, f @ r, tp),
    ]:
        xy = np.asarray(points, float)
        if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
            raise ValueError("finite XY checks required")
        delta = _mapped(cycle, xy) - xy
        pixel = np.linalg.norm(delta, axis=1)
        physical = np.linalg.norm(delta @ calibration[:2, :2].T, axis=1)
        p90 = float(np.percentile(pixel, 90)) if len(xy) else None
        values[name] = {
            "checks": len(xy),
            "p90_cycle_error_px": p90,
            "p90_cycle_error_physical": (
                float(np.percentile(physical, 90)) if len(xy) else None
            ),
        }
        supported &= p90 is not None and p90 <= tolerance_px
    return {"agrees": bool(supported), "frames": values}


def _supported(evidence, config):
    return (
        evidence["verification_supported"] >= config.minimum_matches
        and evidence["verification_cells"] >= config.minimum_cells
    )


def _exclude_fitting_endpoints(source_xy, target_xy, fitting_pairs):
    """Exclude checks sharing either endpoint bin with any fit in one direction.

    A feature can straddle a spatial-cell edge through subpixel jitter in another
    channel, orientation, or matcher. Exclusion across all fitting pools prevents
    it from checking a hypothesis that already fitted the same feature location.
    """
    occupied = [set(), set()]
    for pair in fitting_pairs:
        for index, xy in enumerate(pair):
            occupied[index].update(map(tuple, np.floor(xy + 0.5).astype(np.int64)))
    keep = np.ones(len(source_xy), bool)
    for index, xy in enumerate([source_xy, target_xy]):
        bins = np.floor(xy + 0.5).astype(np.int64)
        keep &= np.array([tuple(point) not in occupied[index] for point in bins], bool)
    return source_xy[keep], target_xy[keep]


def _direction_matches(
    source, target, sc, tc, *, model, guarded, config, sm, tm, direction, collector
):
    fits, va, vb = _matches(source, target, sc, tc, config)
    candidates, records, dense_checks = [], [], []
    fitting_pairs = [(a, b) for _, a, b in fits]
    checks = {"rootsift": (va, vb)}
    for channel, a, b in fits:
        c = candidate_from_points(
            a,
            b,
            stage=f"{direction}_rootsift_{channel}",
            target_shape=target.mask.shape,
            config=guarded,
        )
        if c is not None:
            c.diagnostics.update(backend="rootsift", direction=direction)
            candidates.append(c)
        if collector is not None:
            collector(
                {
                    "backend": "rootsift",
                    "channel": channel,
                    "direction": direction,
                    "partition": "source_spatial",
                    "role": "fitting",
                    "coordinate_units": "working_pixel_centres",
                    "source_xy": a.copy(),
                    "target_xy": b.copy(),
                }
            )
    if collector is not None:
        collector(
            {
                "backend": "rootsift",
                "channel": "combined",
                "direction": direction,
                "partition": "source_spatial",
                "role": "checking_raw",
                "coordinate_units": "working_pixel_centres",
                "source_xy": va.copy(),
                "target_xy": vb.copy(),
            }
        )
    records.append(
        {
            "backend": "rootsift",
            "direction": direction,
            "fitting_matches": sum(len(a) for _, a, _ in fits),
            "checking_matches": len(va),
            "partition": "source_spatial",
        }
    )

    def collect(record):
        a, b = record["source_xy"], record["target_xy"]
        fa, fb, ca, cb = partition_correspondences(
            a,
            b,
            source.mask.shape,
            target.mask.shape,
            policy="source_spatial",
        )
        fitting_pairs.append((fa, fb))
        dense_checks.append((ca, cb))
        stage = f"{direction}_romav2_{record['channel']}_{record['rotation']}"
        c = candidate_from_points(
            fa, fb, stage=stage, target_shape=target.mask.shape, config=guarded
        )
        if c is not None:
            c.diagnostics.update(
                backend="romav2",
                direction=direction,
                channel=record["channel"],
                rotation=record["rotation"],
            )
            candidates.append(c)
        records.append(
            {
                "backend": "romav2",
                "direction": direction,
                "channel": record["channel"],
                "rotation": record["rotation"],
                "fitting_matches": len(fa),
                "checking_matches": len(ca),
                "partition": "source_spatial",
            }
        )
        if collector is not None:
            collector(
                {
                    **record,
                    "direction": direction,
                    "partition": "source_spatial",
                    "role": "matching_raw",
                }
            )

    dense_candidates(
        source,
        target,
        backend="romav2",
        model=model,
        config=guarded,
        source_modality=sm,
        target_modality=tm,
        generic_unknown=True,
        correspondence_collector=collect,
        fit_correspondences=False,
    )
    checks["romav2"] = (
        np.concatenate([pair[0] for pair in dense_checks])
        if dense_checks
        else np.empty((0, 2)),
        np.concatenate([pair[1] for pair in dense_checks])
        if dense_checks
        else np.empty((0, 2)),
    )
    for family, (a, b) in checks.items():
        ca, cb = _exclude_fitting_endpoints(a, b, fitting_pairs)
        checks[family] = ca, cb
        records.append(
            {
                "backend": family,
                "direction": direction,
                "checking_before_fit_exclusion": len(a),
                "checking_matches": len(ca),
                "fit_endpoint_exclusions": len(a) - len(ca),
                "partition": "source_spatial",
                "endpoint_bin_px": 1,
            }
        )
        if collector is not None:
            collector(
                {
                    "backend": family,
                    "channel": "combined",
                    "direction": direction,
                    "partition": "source_spatial",
                    "role": "checking",
                    "coordinate_units": "working_pixel_centres",
                    "source_xy": ca.copy(),
                    "target_xy": cb.copy(),
                }
            )
    return candidates, checks, records


def _score(candidate, sf, tf, sc, tc, sp, tp, checks, config):
    evaluate_candidate(candidate, sf, tf, sc, tc, sp, tp)
    if not candidate.accepted:
        return
    scales = np.asarray(candidate.diagnostics["physical_directional_scales"])
    if scales.min() < 0.65 or scales.max() > 1.35:
        candidate.accepted = False
        candidate.rejection_reason = "excessive physical scale change"
        return
    candidate.diagnostics.update(support_evidence(candidate.matrix, sf.mask, tf.mask))
    families = {
        family: directional_correspondence_evidence(
            candidate.matrix,
            a,
            b,
            sf.mask.shape,
            tf.mask.shape,
            tolerance_px=config.tolerance_px,
        )
        for family, (a, b) in checks.items()
    }
    candidate.diagnostics["matcher_checks"] = families
    candidate.diagnostics["both_matchers_supported"] = all(
        _supported(evidence, config) for evidence in families.values()
    )
    fraction = min(e["verification_fraction"] or 0.0 for e in families.values())
    d = candidate.diagnostics
    d["selection_score"] = (
        0.60 * fraction
        + 0.30 * d["structural_nmi"]
        + 0.10 * d["full_extent_dice"]
        - 0.05 * float(np.abs(np.log(scales)).sum())
    )


def select_consensus_registration(
    source,
    target,
    seeds: list[AlignmentSeed],
    *,
    model,
    source_mask,
    target_mask,
    source_physical,
    target_physical,
    source_modality="unknown",
    target_modality="unknown",
    predecessor="adaptive_previous",
    config=None,
    correspondence_collector=None,
):
    """Require two matcher families and separately fitted reverse agreement.

    RoMa's model is provided by the caller, which owns model provenance and the
    GPU lock. Every direction searches four dense-matcher orientations. Fits use
    even source spatial cells and checks use odd cells. Imported seeds are not
    claimed independent of checks. Disjointness holds within each direction;
    forward and reverse features need not be mutually independent. No evaluation
    annotations enter this API.
    """
    import cv2

    cfg = config or VerifiedSelectionConfig(partition_policy="source_spatial")
    if cfg.partition_policy != "source_spatial":
        raise ValueError("matcher agreement requires source_spatial partition")
    binding = predecessor_binding(seeds, predecessor)
    cv2.setRNGSeed(cfg.seed)
    guarded = GuardedRegistrationConfig(seed=cfg.seed, mask_mode="all_components")
    sf, tf = component_frame(source, source_mask), component_frame(target, target_mask)
    sp, tp = _affine(source_physical), _affine(target_physical)
    sc = modality_channels(sf.image, sf.mask, source_modality)
    tc = modality_channels(tf.image, tf.mask, target_modality)
    forward, checks, records = _direction_matches(
        sf,
        tf,
        sc,
        tc,
        model=model,
        guarded=guarded,
        config=cfg,
        sm=source_modality,
        tm=target_modality,
        direction="forward",
        collector=correspondence_collector,
    )
    # These are new matcher calls with swapped images, not inverted forward fits.
    reverse, reverse_checks, reverse_records = _direction_matches(
        tf,
        sf,
        tc,
        sc,
        model=model,
        guarded=guarded,
        config=cfg,
        sm=target_modality,
        tm=source_modality,
        direction="reverse",
        collector=correspondence_collector,
    )
    for seed in seeds:
        matrix = (
            np.linalg.inv(tf.working_to_input)
            @ _affine(seed.matrix)
            @ sf.working_to_input
        )
        forward.append(
            RegistrationCandidate(
                matrix,
                seed.name,
                diagnostics={
                    "backend": "imported",
                    "direction": "forward",
                    "checking_features_independent_of_seed": False,
                },
            )
        )
    for c in forward:
        _score(c, sf, tf, sc, tc, sp, tp, checks, cfg)
    for c in reverse:
        _score(c, tf, sf, tc, sc, tp, sp, reverse_checks, cfg)
    reverse_supported = [
        c for c in reverse if c.accepted and c.diagnostics["both_matchers_supported"]
    ]
    eligible = []
    # Cycle checks use distributed held-out source and target features. The reverse
    # map also has its own separate checking pool and both-family support gate.
    a, b = checks["rootsift"]
    for c in forward:
        if not c.accepted:
            continue
        if not c.diagnostics["both_matchers_supported"]:
            c.diagnostics["consensus_rejection_reason"] = (
                "insufficient distributed support from both matchers"
            )
            continue
        agreements = [
            (
                r,
                independent_direction_agreement(
                    c.matrix,
                    r.matrix,
                    a,
                    b,
                    source_working_to_physical=sp @ sf.working_to_input,
                    target_working_to_physical=tp @ tf.working_to_input,
                    tolerance_px=cfg.tolerance_px,
                ),
            )
            for r in reverse_supported
        ]
        matched = [(r, evidence) for r, evidence in agreements if evidence["agrees"]]
        c.diagnostics["reverse_agreement"] = [
            {"stage": r.stage, **evidence} for r, evidence in agreements
        ]
        if matched:
            r, evidence = max(
                matched, key=lambda item: item[0].diagnostics["selection_score"]
            )
            c.diagnostics["independently_fitted_reverse_stage"] = r.stage
            eligible.append(c)
        else:
            c.diagnostics["consensus_rejection_reason"] = (
                "no independently fitted reverse consensus"
            )
    previous = next((c for c in forward if c.stage == predecessor and c.accepted), None)
    diagnostics = {
        **binding,
        "backend": ["rootsift", "romav2"],
        "partition": "source_spatial",
        "matches": records + reverse_records,
        "reverse_candidates": [c.to_json_dict() for c in reverse],
        "independent_reverse_fitting": True,
        "evaluation_landmarks_used": False,
        "independent_anatomical_accuracy": False,
        "coordinate_units": "input_pixel_centres",
        "verification_scope": "image features; imported seeds may have used checks",
        "partition_scope": (
            "disjoint within each direction; directions can share features"
        ),
        "working_frames": {
            "source_to_input": sf.working_to_input.tolist(),
            "target_to_input": tf.working_to_input.tolist(),
            "source_input_to_physical": sp.tolist(),
            "target_input_to_physical": tp.tolist(),
        },
    }
    best = (
        max(eligible, key=lambda c: c.diagnostics["selection_score"])
        if eligible
        else None
    )
    if previous is not None and (
        best is None
        or best.diagnostics["selection_score"] - previous.diagnostics["selection_score"]
        < cfg.minimum_gain
    ):
        best = previous
        diagnostics["predecessor_retention"] = (
            "insufficient two-matcher and reverse-fit improvement"
        )
    if best is None:
        raise UnsupportedRegistration(
            "no supported matcher agreement or predecessor", forward, diagnostics
        )
    diagnostics["selected_stage"] = best.stage
    if best is previous:
        native = np.asarray(
            next(seed.matrix for seed in seeds if seed.name == predecessor), float
        ).copy()
        diagnostics["retained_predecessor_id"] = binding["predecessor_id"]
        diagnostics["retained_predecessor_transform_sha256"] = binding[
            "predecessor_transform_sha256"
        ]
    else:
        native = tf.working_to_input @ best.matrix @ np.linalg.inv(sf.working_to_input)
    diagnostics["selected_transform_sha256"] = transform_fingerprint(native)
    return GuardedRegistrationResult(
        native, best.stage, forward, sf, tf, guarded
    ), diagnostics
