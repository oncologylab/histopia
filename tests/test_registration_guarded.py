import subprocess
import sys

import numpy as np
import pytest

from histopia.registration._guarded import (
    GuardedRegistrationConfig,
    RegistrationCandidate,
    component_frame,
    estimate_guarded_registration,
    evaluate_candidate,
    pixel_center_resize_map,
    structure_channels,
    supported_tissue_mask,
)


def test_actual_anisotropic_resize_and_offset_preserve_image_extent():
    G = pixel_center_resize_map((97, 203), (38, 80), (14, 29))
    edges = np.array([[-0.5, -0.5, 1], [79.5, 37.5, 1]]) @ G.T
    assert np.allclose(edges[:, :2], [[13.5, 28.5], [216.5, 125.5]])
    points = np.array([[0, 0, 1], [25, 31, 1], [79, 37, 1]])
    assert np.allclose((points @ G.T) @ np.linalg.inv(G).T, points)


def test_all_component_crop_keeps_disconnected_tissue():
    image = np.full((200, 300, 3), 255, np.uint8)
    mask = np.zeros((200, 300), bool)
    mask[30:150, 20:120] = True
    mask[175:190, 280:295] = True
    all_frame = component_frame(image, mask, ceiling=128)
    dominant = component_frame(image, mask, ceiling=128, mode="dominant")
    assert all_frame.retained_mask_fraction == 1
    assert dominant.retained_mask_fraction < 1
    assert max(all_frame.image.shape[:2]) <= 128


def test_pale_mask_does_not_add_blank_glass():
    image = np.full((80, 100, 3), 255, np.uint8)
    mask = np.zeros((80, 100), bool)
    mask[20:60, 25:70] = True
    image[mask] = [190, 175, 190]
    proposed = np.ones_like(mask)
    added = supported_tissue_mask(image, mask, {"group_pale_tissue": proposed})
    assert np.array_equal(added, mask)


def test_mask_evidence_never_becomes_feature_inliers():

    rng = np.random.default_rng(11)
    image = rng.integers(0, 220, (80, 90, 3), dtype="uint8")
    mask = np.ones((80, 90), bool)
    frame = component_frame(image, mask)
    channels = structure_channels(image, mask)
    candidate = RegistrationCandidate(np.eye(3), "mask_only")
    evaluate_candidate(candidate, frame, frame, channels, channels)
    assert candidate.accepted
    assert candidate.diagnostics["mask_overlap_dice"] == 1
    assert candidate.feature_inlier_count == 0
    assert candidate.feature_match_count == 0


def test_reflection_rejected_even_with_good_mask_overlap():
    image = np.full((70, 80, 3), 170, np.uint8)
    mask = np.ones((70, 80), bool)
    frame = component_frame(image, mask)
    channels = structure_channels(image, mask)
    candidate = RegistrationCandidate(
        np.array([[-1, 0, 79], [0, 1, 0], [0, 0, 1.0]]), "reflection"
    )
    evaluate_candidate(candidate, frame, frame, channels, channels)
    assert not candidate.accepted
    assert candidate.rejection_reason == "physical shape guard"


def test_synthetic_cross_color_translation_and_separate_evidence():
    import cv2

    rng = np.random.default_rng(4)
    source = np.full((240, 280, 3), 250, np.uint8)
    for _ in range(120):
        xy = tuple(rng.integers([35, 35], [235, 200]).tolist())
        color = tuple(rng.integers(40, 180, 3).tolist())
        cv2.circle(source, xy, int(rng.integers(2, 7)), color, -1)
    H = np.array([[1.0, 0, 12], [0, 1, -8], [0, 0, 1]])
    target = cv2.warpAffine(source, H[:2], (280, 240), borderValue=(250, 250, 250))
    sm = source.mean(axis=2) < 230
    tm = target.mean(axis=2) < 230
    r = estimate_guarded_registration(
        source,
        target,
        source_mask=sm,
        target_mask=tm,
        config=GuardedRegistrationConfig(
            max_dim_px=256, max_features=1000, refine=True, mask_mode="all_components"
        ),
    )
    points = np.array([[70, 70, 1], [140, 130, 1], [210, 180, 1]])
    assert np.max(np.linalg.norm((points @ (r.matrix - H).T)[:, :2], axis=1)) < 1
    assert all(c.feature_inlier_count <= c.feature_match_count for c in r.candidates)


def test_empty_mask_has_explicit_missing_support():
    with pytest.raises(ValueError, match="No supported tissue"):
        component_frame(np.zeros((80, 80, 3), np.uint8), np.zeros((80, 80), bool))


def test_guarded_config_is_lazy_and_rejects_invalid_limits():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from histopia.registration import GuardedRegistrationConfig; "
            'assert "cv2" not in sys.modules; assert "torch" not in sys.modules',
        ],
        check=True,
    )
    with pytest.raises(ValueError):
        GuardedRegistrationConfig(max_relative_scale_change=float("nan"))


def test_quarter_turn_mapping_matches_actual_pixel_locations():
    from histopia.registration._learned_matching import quarter_turn_matrix

    original = np.arange(35).reshape(5, 7)
    for turns in range(4):
        rotated = np.rot90(original, turns)
        T = quarter_turn_matrix(original.shape, turns)
        for y, x in [(0, 0), (2, 4), (4, 6)]:
            xx, yy, _ = (T @ [x, y, 1]).astype(int)
            assert rotated[yy, xx] == original[y, x]
