from __future__ import annotations

import numpy as np
import pytest

from histopia.stain import StainFamily
from histopia.stain._model import (
    StainModel,
    canonical_vectors,
    cohort_vector_template,
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
    np.testing.assert_allclose(np.linalg.norm(shrunk, axis=1), 1)
