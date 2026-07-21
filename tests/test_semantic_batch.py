from __future__ import annotations

import numpy as np

from histopia.semantic._batch import (
    BatchAcceptanceGuard,
    BatchCorrectionResult,
    BatchDiagnosticStage,
    correct_batch_offsets,
)


def _batch_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(41)
    first_type = rng.normal([-2.0, 0.5, 1.0, -0.5], 0.18, size=(40, 4))
    second_type = rng.normal([2.0, -0.5, -1.0, 0.5], 0.18, size=(40, 4))
    biology = np.vstack([first_type, second_type])
    shift = np.array([3.0, -2.0, 1.5, 0.75])
    features = np.vstack([biology, biology + shift, biology])
    section_offsets = np.array([0, 80, 160, 240], dtype=np.int64)
    good_pairs = np.column_stack([np.arange(80), np.arange(80, 160)])
    bad_pairs = np.array([[0, 150], [2, 152], [70, 81]], dtype=np.int64)
    anchor_pairs = np.vstack([good_pairs, bad_pairs])
    anchor_weights = np.concatenate([np.ones(80), np.full(3, 0.01)])
    return features, section_offsets, anchor_pairs, anchor_weights


def test_batch_correction_recovers_offsets_without_changing_local_geometry() -> None:
    features, section_offsets, anchor_pairs, anchor_weights = _batch_fixture()

    result = correct_batch_offsets(
        features,
        section_offsets,
        anchor_pairs,
        anchor_weights,
        seed=17,
    )

    assert isinstance(result, BatchCorrectionResult)
    assert isinstance(result.guard, BatchAcceptanceGuard)
    assert isinstance(result.raw_diagnostics, BatchDiagnosticStage)
    assert result.guard.accepted
    assert result.guard.reasons == ()
    np.testing.assert_allclose(result.section_corrections[0], 0.0, atol=1e-10)
    np.testing.assert_allclose(
        result.section_corrections[1],
        np.array([-3.0, 2.0, -1.5, -0.75]),
        atol=0.03,
    )
    np.testing.assert_allclose(result.section_corrections[2], 0.0, atol=1e-10)
    np.testing.assert_allclose(result.corrected_features[:80], features[:80])
    np.testing.assert_allclose(result.corrected_features[160:], features[160:])
    before = np.linalg.norm(features[80:120] - features[120:160], axis=1)
    after = np.linalg.norm(
        result.corrected_features[80:120] - result.corrected_features[120:160],
        axis=1,
    )
    np.testing.assert_allclose(after, before, atol=1e-10)

    assert result.unsupported_sections == (2,)
    assert result.raw_diagnostics.stage == "raw"
    assert result.legacy_diagnostics.stage == "legacy"
    assert result.corrected_diagnostics.stage == "anchor_corrected"
    assert (
        result.corrected_diagnostics.median_anchor_cosine_distance
        < result.raw_diagnostics.median_anchor_cosine_distance
    )
    assert (
        result.corrected_diagnostics.slide_variance_fraction
        < result.raw_diagnostics.slide_variance_fraction
    )
    assert result.corrected_diagnostics.within_slide_knn_preservation >= 0.99
    assert result.corrected_diagnostics.correction_magnitude > 0.0
    assert 0.0 < result.corrected_diagnostics.anchor_coverage < 1.0
    assert 0.0 <= result.corrected_diagnostics.slide_prediction_accuracy <= 1.0


def test_batch_correction_is_reproducible() -> None:
    arguments = _batch_fixture()

    first = correct_batch_offsets(*arguments, seed=9)
    second = correct_batch_offsets(*arguments, seed=9)

    np.testing.assert_array_equal(first.corrected_features, second.corrected_features)
    np.testing.assert_array_equal(first.section_corrections, second.section_corrections)
    assert first.raw_diagnostics == second.raw_diagnostics
    assert first.corrected_diagnostics == second.corrected_diagnostics
    assert first.guard == second.guard


def test_batch_guard_rejects_anchor_alignment_that_creates_slide_variance() -> None:
    first = np.vstack(
        [
            np.repeat([[1.0, 1.0]], 30, axis=0),
            np.repeat([[6.0, 1.0]], 30, axis=0),
        ]
    )
    features = np.vstack([first, first])
    section_offsets = np.array([0, 60, 120], dtype=np.int64)
    misleading_pairs = np.column_stack([np.arange(30), np.arange(90, 120)])

    result = correct_batch_offsets(
        features,
        section_offsets,
        misleading_pairs,
        np.ones(30),
        seed=5,
    )

    assert not result.guard.accepted
    assert "slide_variance_fraction" in result.guard.reasons
    np.testing.assert_array_equal(result.corrected_features, features)
    assert not np.array_equal(result.proposed_features, features)
    assert (
        result.corrected_diagnostics.slide_prediction_accuracy
        != result.raw_diagnostics.slide_prediction_accuracy
    )
