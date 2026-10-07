from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from histopia.stain import StainFamily
from histopia.stain._adaptive import (
    apply_adaptive_background,
    apply_counterstain_conditioned_background,
    infer_adaptive_background,
    infer_counterstain_conditioned_background,
)
from histopia.stain._model import (
    CandidateFit,
    StainModel,
    _rank_correlation,
    canonical_vectors,
    cohort_vector_template,
    counterstain_only_mask,
    estimate_nmf_vectors,
    fit_candidate,
    select_family_method,
    shrink_vectors,
    unmix_od,
)
from histopia.stain._od import (
    apply_shading_correction,
    estimate_background_model,
    od_to_rgb,
    rgb_to_od,
)
from histopia.stain._pipeline import _equivalent_fit_provenance


def test_adaptive_background_removes_stable_tissue_floor_only() -> None:
    rng = np.random.default_rng(17)
    tissue = np.ones((100, 100), dtype=bool)
    tissue[:5, :] = False
    counterstain = rng.gamma(2.0, 0.15, tissue.shape).astype(np.float32)
    target = np.zeros(tissue.shape, dtype=np.float32)
    values = 0.04 + rng.gamma(1.5, 0.08, np.count_nonzero(tissue))
    strongly_stained = rng.random(len(values)) < 0.20
    values[strongly_stained] += rng.gamma(2.0, 0.30, np.count_nonzero(strongly_stained))
    target[tissue] = values

    result = infer_adaptive_background(target, counterstain, tissue, seed=4)
    derived = apply_adaptive_background(target, tissue, result)

    assert result.accepted is True
    assert result.floor_od >= 0.02
    assert result.rank_correlation >= 0.995
    assert 0.10 <= result.suppression_fraction <= 0.75
    assert np.count_nonzero(derived[~tissue]) == 0
    np.testing.assert_array_equal(target, target.copy())


def test_adaptive_background_falls_back_when_floor_is_too_small() -> None:
    rng = np.random.default_rng(19)
    tissue = np.ones((40, 40), dtype=bool)
    target = rng.uniform(0.0, 0.015, tissue.shape).astype(np.float32)
    counterstain = rng.uniform(0.0, 1.0, tissue.shape).astype(np.float32)

    result = infer_adaptive_background(target, counterstain, tissue, seed=3)
    derived = apply_adaptive_background(target, tissue, result)

    assert result.accepted is False
    assert "floor_below_minimum" in result.rejection_reasons
    np.testing.assert_array_equal(derived, target)


def test_adaptive_background_accepts_stable_overstain_floor() -> None:
    rng = np.random.default_rng(23)
    tissue = np.ones((160, 160), dtype=bool)
    counterstain = rng.gamma(2.0, 0.18, tissue.shape).astype(np.float32)
    target = np.zeros(tissue.shape, dtype=np.float32)
    values = 0.32 + rng.gamma(1.2, 0.035, np.count_nonzero(tissue))
    stained = rng.random(len(values)) < 0.12
    values[stained] += rng.gamma(2.0, 0.45, np.count_nonzero(stained))
    target[tissue] = values

    result = infer_adaptive_background(target, counterstain, tissue, seed=5)
    derived = apply_adaptive_background(target, tissue, result)

    assert result.accepted is True
    assert result.method == "inferred_floor-v2"
    assert 0.45 < result.suppression_fraction <= 0.75
    assert result.q95_retention >= 0.45
    assert result.rank_correlation >= 0.995
    assert float(np.quantile(derived[tissue], 0.95)) > 0


def test_adaptive_background_allows_large_stable_low_fraction_support() -> None:
    rng = np.random.default_rng(29)
    tissue = np.ones((512, 512), dtype=bool)
    target = 0.25 + rng.gamma(1.3, 0.05, tissue.shape).astype(np.float32)
    counterstain = target.copy()
    support = np.zeros(tissue.shape, dtype=bool)
    support.ravel()[:5_000] = True
    counterstain[support] = 1.2
    target[support] = 0.25 + rng.uniform(0, 0.01, np.count_nonzero(support))

    result = infer_adaptive_background(target, counterstain, tissue, seed=8)

    assert result.support_fraction < 0.02
    assert result.support_pixels >= 4_096
    assert "insufficient_support_fraction" not in result.rejection_reasons


def test_counterstain_conditioned_background_removes_diffuse_nuisance_only() -> None:
    rng = np.random.default_rng(31)
    tissue = np.ones((200, 200), dtype=bool)
    tissue[:10] = False
    counterstain = rng.gamma(2.0, 0.18, tissue.shape).astype(np.float32)
    values = (
        0.25
        + 0.12 * (counterstain[tissue] - np.median(counterstain[tissue]))
        + rng.normal(0.0, 0.018, np.count_nonzero(tissue))
    )
    positive = rng.random(len(values)) < 0.22
    values[positive] += rng.gamma(2.0, 0.35, np.count_nonzero(positive))
    target = np.zeros(tissue.shape, dtype=np.float32)
    target[tissue] = np.maximum(values, 0)
    confidence = tissue.astype(np.float32) * 0.9

    result = infer_counterstain_conditioned_background(
        target,
        counterstain,
        tissue,
        confidence=confidence,
        seed=31,
    )
    derived = apply_counterstain_conditioned_background(
        target,
        counterstain,
        tissue,
        result,
    )

    assert result.accepted is True
    assert result.method == "counterstain-conditioned-v3"
    assert result.spatial_field_used is False
    assert 0.10 <= result.suppression_fraction <= 0.75001
    assert result.q95_retention >= 0.45
    assert result.residual_rank_correlation >= 0.99
    assert result.minimum_nuisance_od <= result.maximum_nuisance_od
    assert np.count_nonzero(derived[~tissue]) == 0
    assert np.mean(derived[tissue] > 0.02) < 0.30
    np.testing.assert_array_equal(target[~tissue], 0)


def test_fit_cache_reuse_ignores_only_campaign_preflight_fingerprint() -> None:
    expected = {
        "preflight_fingerprint": "new",
        "source_sha256": "source",
        "mask_sha256": "mask",
        "analysis_mpp": 4.0,
    }
    observed = {**expected, "preflight_fingerprint": "old"}

    assert _equivalent_fit_provenance(observed, expected)
    assert not _equivalent_fit_provenance(
        {**observed, "mask_sha256": "different"}, expected
    )


def test_nonnegative_unmixing_recovers_known_concentrations() -> None:
    rng = np.random.default_rng(3)
    vectors = canonical_vectors(StainFamily.H_DAB)
    expected = rng.gamma(1.5, 0.2, size=(4096, 2)).astype(np.float32)
    white = np.array([250.0, 248.0, 246.0])
    rgb = od_to_rgb(expected @ vectors, white)

    observed, residual = unmix_od(rgb_to_od(rgb, white), vectors)

    assert float(np.mean(np.abs(observed - expected))) < 0.006
    assert float(np.quantile(residual, 0.99)) < 0.004
    assert np.all(observed >= 0)


@pytest.mark.parametrize(
    "family",
    [
        StainFamily.H_DAB,
        StainFamily.SIRIUS_RED,
        StainFamily.PAS,
        StainFamily.ALCIAN_BLUE,
    ],
)
def test_macenko_candidate_tracks_every_family_prior(family: StainFamily) -> None:
    rng = np.random.default_rng(7)
    priors = canonical_vectors(family)
    concentrations = rng.gamma(1.2, 0.25, size=(5000, 2))
    tissue = concentrations @ priors

    fit = fit_candidate(
        tissue,
        np.zeros((100, 3)),
        family,
        "macenko",
        seed=7,
    )

    assert fit.reconstruction_nrmse < 0.02
    assert fit.prior_angle_degrees < 5
    assert fit.glass_leakage == 0


def test_nmf_accepts_float32_optical_density() -> None:
    rng = np.random.default_rng(11)
    priors = canonical_vectors(StainFamily.H_DAB)
    concentrations = rng.gamma(1.2, 0.25, size=(512, 2)).astype(np.float32)
    optical_density = (concentrations @ priors).astype(np.float32)

    vectors = estimate_nmf_vectors(optical_density, priors, seed=11)

    assert vectors.shape == (2, 3)
    assert np.all(np.isfinite(vectors))
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1)


def test_background_model_reduces_blank_glass_spatial_variation() -> None:
    height, width = 120, 160
    x = np.linspace(0.78, 1.0, width)[None, :, None]
    white = np.array([248.0, 246.0, 244.0])[None, None, :]
    rgb = np.broadcast_to(white * x, (height, width, 3)).astype(np.uint8).copy()
    tissue = np.zeros((height, width), dtype=bool)
    tissue[30:90, 45:120] = True
    vectors = canonical_vectors(StainFamily.H_DAB)
    signal = np.zeros((height, width, 2), dtype=np.float32)
    signal[30:90, 45:120, 0] = 0.15
    signal[45:75, 65:100, 1] = 0.55
    rgb[tissue] = od_to_rgb(
        (signal @ vectors)[tissue],
        np.array([248.0, 246.0, 244.0]),
    )

    model = estimate_background_model(rgb, tissue, max_samples=10_000, seed=1)
    corrected = apply_shading_correction(rgb, model)
    before = rgb[~tissue].mean(axis=1)
    after = corrected[~tissue].mean(axis=1)

    assert model.after_spatial_cv < model.before_spatial_cv
    assert float(np.std(after) / np.mean(after)) < float(
        np.std(before) / np.mean(before)
    )

    stain_model = StainModel(
        family=StainFamily.H_DAB,
        marker="DAB",
        method="fixed",
        background=model,
        raw_vectors=vectors,
        corrected_vectors=vectors,
        correction_accepted=True,
        correction_rank_correlation=1.0,
        raw_glass_leakage=0.02,
        corrected_glass_leakage=0.01,
        content_bbox_native_xywh=(10, 20, width, height),
    )
    full = stain_model.transform_rgb(rgb)
    left = stain_model.transform_native_tile(
        rgb[:, :80],
        tile_bbox_native_xywh=(10, 20, 80, height),
    )
    right = stain_model.transform_native_tile(
        rgb[:, 80:],
        tile_bbox_native_xywh=(90, 20, 80, height),
    )

    np.testing.assert_allclose(
        np.column_stack([left.corrected_target_od, right.corrected_target_od]),
        full.corrected_target_od,
        atol=2e-6,
    )


def test_cohort_method_selection_and_shrinkage_are_deterministic() -> None:
    vectors = canonical_vectors(StainFamily.H_DAB)
    candidates = []
    for offset in (0.0, 0.01):
        tissue = np.array([[0.2, 0.4], [0.5, 0.1], [0.3, 0.3]]) @ vectors + offset
        candidates.append(
            [
                fit_candidate(
                    tissue,
                    np.zeros((20, 3)),
                    StainFamily.H_DAB,
                    method,
                    seed=2,
                )
                for method in ("legacy", "fixed")
            ]
        )

    selected, metrics = select_family_method(candidates)
    template = cohort_vector_template(
        [
            next(row.vectors for row in slide if row.method == selected)
            for slide in candidates
        ]
    )
    shrunk = shrink_vectors(candidates[0][0].vectors, template, 0.25)

    assert selected in {"fixed", "legacy"}
    assert set(metrics) == {"fixed", "legacy"}
    assert all("counterstain_leakage" in row for row in metrics.values())
    np.testing.assert_allclose(np.linalg.norm(shrunk, axis=1), 1)


def test_counterstain_only_leakage_contributes_to_method_selection() -> None:
    vectors = canonical_vectors(StainFamily.H_DAB)
    tissue = np.array([[0.5, 0.02], [0.4, 0.01], [0.3, 0.03]]) @ vectors
    baseline = fit_candidate(
        tissue,
        np.zeros((20, 3)),
        StainFamily.H_DAB,
        "fixed",
        seed=5,
    )
    clean = replace(baseline, method="clean", counterstain_leakage=0.01)
    leaky = replace(baseline, method="leaky", counterstain_leakage=0.5)

    selected, metrics = select_family_method([[leaky, clean]])

    assert selected == "clean"
    assert (
        metrics["clean"]["counterstain_leakage"]
        < metrics["leaky"]["counterstain_leakage"]
    )


def test_counterstain_only_mask_selects_strong_counterstain_low_target() -> None:
    concentrations = np.array(
        [[0.8, 0.01], [0.7, 0.02], [0.1, 0.8], [0.2, 0.7]] * 4,
        dtype=np.float32,
    )

    selected = counterstain_only_mask(concentrations)

    assert selected.shape == (16,)
    assert np.all(concentrations[selected, 0] > concentrations[selected, 1])


def test_nonconverged_adaptive_method_is_ineligible() -> None:
    vectors = canonical_vectors(StainFamily.H_DAB)
    tissue = np.array([[0.2, 0.4], [0.5, 0.1], [0.3, 0.3]]) @ vectors
    fixed = fit_candidate(
        tissue,
        np.zeros((20, 3)),
        StainFamily.H_DAB,
        "fixed",
        seed=3,
    )
    nmf = replace(
        fixed,
        method="nmf",
        reconstruction_nrmse=0.0,
        glass_leakage=0.0,
        converged=False,
        optimization_iterations=300,
    )

    selected, metrics = select_family_method([[fixed, nmf]])

    assert selected == "fixed"
    assert metrics["fixed"]["convergence_fraction"] == 1.0
    assert metrics["nmf"]["convergence_fraction"] == 0.0
    restored = CandidateFit.from_json_dict(nmf.to_json_dict())
    assert restored.method == "nmf"
    assert restored.converged is False
    assert restored.optimization_iterations == 300
    np.testing.assert_allclose(restored.vectors, nmf.vectors)


def test_low_rank_preservation_makes_method_ineligible() -> None:
    vectors = canonical_vectors(StainFamily.H_DAB)
    tissue = np.array([[0.2, 0.4], [0.5, 0.1], [0.3, 0.3]]) @ vectors
    fixed = fit_candidate(
        tissue,
        np.zeros((20, 3)),
        StainFamily.H_DAB,
        "fixed",
        seed=4,
    )
    adaptive = replace(
        fixed,
        method="adaptive",
        reconstruction_nrmse=0.0,
        glass_leakage=0.0,
        target_rank_correlation=0.5,
    )

    selected, metrics = select_family_method([[fixed, adaptive]])

    assert selected == "fixed"
    assert metrics["fixed"]["rank_guard_fraction"] == 1.0
    assert metrics["adaptive"]["rank_guard_fraction"] == 0.0


def test_rank_correlation_uses_average_ranks_for_ties() -> None:
    left = np.array([0.0, 0.0, 1.0, 2.0])
    right = np.array([0.0, 1.0, 1.0, 2.0])

    assert _rank_correlation(left, left) == pytest.approx(1.0)
    assert _rank_correlation(left, right) == pytest.approx(5 / 6)
    assert _rank_correlation(np.ones(4), np.ones(4)) == 0.0
