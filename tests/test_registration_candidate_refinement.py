import numpy as np
import pytest

from histopia.registration._candidate_refinement import (
    AlignmentSeed,
    dark_annotation_proposals,
    refine_registration_candidates,
)
from histopia.registration._guarded import GuardedRegistrationConfig


def test_annotation_proposal_does_not_remove_small_nuclei():
    image = np.full((200, 200, 3), 255, dtype=np.uint8)
    image[20:50, 28:38] = 5
    image[28:38, 20:50] = 5
    image[130:133, 120:123] = 5
    proposal = dark_annotation_proposals(image)
    assert proposal[30, 30]
    assert not proposal[131, 121]
    assert not proposal[180, 180]


def test_annotation_proposal_rejects_ambiguous_intensity_units():
    with pytest.raises(ValueError, match="uint8"):
        dark_annotation_proposals(np.ones((20, 20, 3), dtype=float))


def test_common_frame_selection_rejects_wrong_initialization():
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(81)
    image = np.full((180, 200, 3), 255, dtype=np.uint8)
    image[25:150, 35:165] = rng.integers(50, 210, (125, 130, 3), dtype=np.uint8)
    mask = np.any(image < 240, axis=2)
    true = np.array([[1, 0, 11], [0, 1, -7], [0, 0, 1]], dtype=float)
    target = cv2.warpAffine(image, true[:2], (200, 180), borderValue=(255, 255, 255))
    target_mask = cv2.warpAffine(mask.astype("uint8"), true[:2], (200, 180)) > 0
    reflection = np.array([[-1, 0, 190], [0, 1, 0], [0, 0, 1]], dtype=float)
    fit = refine_registration_candidates(
        image,
        target,
        [
            AlignmentSeed("wrong", np.eye(3)),
            AlignmentSeed("right", true),
            AlignmentSeed("reflected", reflection),
        ],
        source_mask=mask,
        target_mask=target_mask,
        config=GuardedRegistrationConfig(refine=False),
    )
    np.testing.assert_allclose(fit.matrix, true, atol=1e-8)
    assert fit.selected_stage == "right"
    assert not fit.candidates[-1].accepted
    assert all(c.feature_inlier_count == 0 for c in fit.candidates)


def test_seed_validation():
    image = np.full((32, 32, 3), 140, dtype=np.uint8)
    mask = np.ones((32, 32), dtype=bool)
    with pytest.raises(ValueError, match="at least one"):
        refine_registration_candidates(
            image, image, [], source_mask=mask, target_mask=mask
        )
    with pytest.raises(ValueError, match="affine"):
        refine_registration_candidates(
            image,
            image,
            [AlignmentSeed("perspective", np.ones((3, 3)))],
            source_mask=mask,
            target_mask=mask,
        )
