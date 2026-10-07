import numpy as np
import pytest

from histopia.registration._adaptive import (
    AdaptiveConfig,
    UnsupportedRegistration,
    estimate_adaptive_registration,
    isolated_annotation_exclusion,
    orientation_seeds,
    support_evidence,
)
from histopia.registration._candidate_refinement import AlignmentSeed
from histopia.registration._guarded import component_frame


def test_support_counts_source_outside_target_canvas():
    pytest.importorskip("cv2")
    source = np.ones((100, 200), bool)
    target = np.ones((100, 100), bool)
    evidence = support_evidence(np.eye(3), source, target)
    assert evidence["source_canvas_retention"] == 0.5
    assert evidence["source_tissue_coverage"] == 0.5
    assert evidence["target_tissue_coverage"] == 1
    assert evidence["full_extent_dice"] == pytest.approx(2 / 3)


def test_scale_is_accounted_for_in_support_area():
    pytest.importorskip("cv2")
    source = np.ones((100, 100), bool)
    target = np.ones((50, 50), bool)
    evidence = support_evidence(np.diag([0.5, 0.5, 1]), source, target)
    assert evidence["full_extent_dice"] == pytest.approx(1)


def test_artifact_exclusion_preserves_dark_biology_and_brown_dab():
    pytest.importorskip("cv2")
    rgb = np.full((300, 300, 3), 255, "uint8")
    rgb[20:50, 30:40] = 5
    rgb[30:40, 20:50] = 5
    rgb[90:220, 90:220] = [190, 130, 170]
    rgb[130:160, 140:150] = 5
    rgb[140:150, 130:160] = 5
    rgb[240:280, 240:280] = [60, 35, 10]
    proposal, excluded = isolated_annotation_exclusion(rgb, np.any(rgb < 240, axis=2))
    assert proposal[145, 145]
    assert excluded[35, 35]
    assert not excluded[145, 145]
    assert not excluded[250, 250]


def test_orientation_initializers_preserve_anisotropic_physical_geometry():
    pytest.importorskip("cv2")
    rgb = np.full((80, 100, 3), 150, "uint8")
    mask = np.ones(rgb.shape[:2], bool)
    sf = component_frame(rgb, mask)
    tf = component_frame(rgb, mask)
    sp, tp = np.diag([0.5, 0.8, 1]), np.diag([0.7, 0.3, 1])
    for c in orientation_seeds(sf, tf, sp, tp):
        physical = (
            tp
            @ tf.working_to_input
            @ c.matrix
            @ np.linalg.inv(sf.working_to_input)
            @ np.linalg.inv(sp)
        )
        np.testing.assert_allclose(
            np.linalg.svd(physical[:2, :2], compute_uv=False), [1, 1], atol=1e-10
        )
        assert np.linalg.det(physical[:2, :2]) == pytest.approx(1)


def test_failed_registration_preserves_rejected_candidates():
    pytest.importorskip("cv2")
    rgb = np.full((100, 100, 3), 150, "uint8")
    mask = np.ones((100, 100), bool)
    reflection = np.array([[-1, 0, 99], [0, 1, 0], [0, 0, 1]])
    with pytest.raises(UnsupportedRegistration) as err:
        estimate_adaptive_registration(
            rgb,
            rgb,
            [AlignmentSeed("reflection", reflection)],
            source_mask=mask,
            target_mask=mask,
            source_physical=np.eye(3),
            target_physical=np.eye(3),
            config=AdaptiveConfig(recover_initialization=False, rescue_mode="off"),
        )
    assert err.value.candidates[0].rejection_reason == "physical shape guard"
    assert "no_accepted_initialization" in err.value.diagnostics["rescue_reasons"]


def test_translation_is_selected_without_evaluation_landmarks():
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(20261006)
    rgb = np.full((180, 220, 3), 255, "uint8")
    rgb[30:140, 40:170] = rng.integers(40, 220, (110, 130, 3), dtype="uint8")
    mask = np.any(rgb < 240, axis=2)
    true = np.array([[1, 0, 15], [0, 1, -8], [0, 0, 1.0]])
    target = cv2.warpAffine(rgb, true[:2], (220, 180), borderValue=(255, 255, 255))
    tm = cv2.warpAffine(mask.astype("uint8"), true[:2], (220, 180)) > 0
    fit, diagnostics = estimate_adaptive_registration(
        rgb,
        target,
        [AlignmentSeed("wrong", np.eye(3)), AlignmentSeed("right", true)],
        source_mask=mask,
        target_mask=tm,
        source_physical=np.eye(3),
        target_physical=np.eye(3),
        config=AdaptiveConfig(rescue_mode="off", refinement_sizes=(128,)),
    )
    np.testing.assert_allclose(fit.matrix, true, atol=0.1)
    assert diagnostics["escalation"] == []


def test_empty_support_retains_failure_evidence():
    rgb = np.full((80, 80, 3), 255, "uint8")
    with pytest.raises(UnsupportedRegistration) as err:
        estimate_adaptive_registration(
            rgb,
            rgb,
            [],
            source_mask=np.zeros((80, 80), bool),
            target_mask=np.ones((80, 80), bool),
            source_physical=np.eye(3),
            target_physical=np.eye(3),
        )
    assert "mask_evidence" in err.value.diagnostics


def test_configuration_rejects_out_of_budget_refinement():
    with pytest.raises(ValueError):
        AdaptiveConfig(refinement_sizes=(4096,))
