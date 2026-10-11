import numpy as np
import pytest

from histopia.registration._candidate_refinement import AlignmentSeed
from histopia.registration._verified_selection import (
    VerifiedSelectionConfig,
    correspondence_evidence,
    modality_channels,
    select_verified_registration,
    spatial_partition,
)


def test_unknown_modality_uses_generic_gradient_not_dab():
    pytest.importorskip("cv2")
    image = np.full((64, 64, 3), [220, 160, 190], dtype="uint8")
    mask = np.ones((64, 64), bool)
    image[20:40, 20:40] = [130, 90, 80]
    generic = modality_channels(image, mask, "unknown")
    legacy = modality_channels(image, mask, "unknown", corrected=False)
    assert not np.array_equal(generic["hematoxylin"], legacy["hematoxylin"])
    assert generic["hematoxylin"][30, 30] == 0


def test_he_projection_differs_from_ihc():
    pytest.importorskip("cv2")
    image = np.random.default_rng(19).integers(50, 250, (60, 70, 3), dtype="uint8")
    mask = np.ones(image.shape[:2], bool)
    he = modality_channels(image, mask, "he")
    ihc = modality_channels(image, mask, "ihc")
    assert not np.array_equal(he["hematoxylin"], ihc["hematoxylin"])


def test_partition_separates_spatial_cells():
    xy = np.array([[5, 5], [10, 10], [30, 5], [90, 90]])
    partition, cells = spatial_partition(xy, (100, 100))
    assert partition.tolist() == [0, 0, 1, 0]
    np.testing.assert_array_equal(cells[0], cells[1])


def test_verification_detects_wrong_rotation_despite_equal_silhouette():
    a = np.array([[10, 10], [30, 10], [70, 30], [90, 70], [30, 90]])
    right = correspondence_evidence(np.eye(3), a, a, (100, 100), (100, 100))
    wrong = correspondence_evidence(
        np.array([[-1, 0, 99], [0, -1, 99], [0, 0, 1]]),
        a,
        a,
        (100, 100),
        (100, 100),
    )
    assert right["verification_supported"] == 5
    assert right["verification_cells"] == 5
    assert wrong["verification_supported"] == 0


def test_missing_verification_stays_missing():
    empty = np.empty((0, 2))
    result = correspondence_evidence(np.eye(3), empty, empty, (100, 100), (100, 100))
    assert result["verification_fraction"] is None
    assert result["verification_median_residual_px"] is None


def test_verified_selection_recovers_translation_and_rejects_enlargement():
    cv2 = pytest.importorskip("cv2")
    image = np.random.default_rng(72).integers(30, 230, (200, 240, 3), dtype="uint8")
    mask = np.ones(image.shape[:2], bool)
    true = np.array([[1, 0, 12], [0, 1, -7], [0, 0, 1.0]])
    target = cv2.warpAffine(image, true[:2], (240, 200), borderValue=(255, 255, 255))
    tm = cv2.warpAffine(mask.astype("uint8"), true[:2], (240, 200)) > 0
    fit, _ = select_verified_registration(
        image,
        target,
        [
            AlignmentSeed("previous", true),
            AlignmentSeed("enlarged", np.diag([1.6, 1.6, 1])),
        ],
        source_mask=mask,
        target_mask=tm,
        source_physical=np.eye(3),
        target_physical=np.eye(3),
        source_modality="he",
        target_modality="he",
        config=VerifiedSelectionConfig(generate_features=False),
    )
    np.testing.assert_allclose(fit.matrix, true, atol=1e-9)
    assert not fit.candidates[1].accepted
    assert fit.candidates[1].rejection_reason == "excessive physical scale change"


def test_invalid_selection_tolerances_rejected():
    with pytest.raises(ValueError):
        VerifiedSelectionConfig(tolerance_px=float("nan"))
    with pytest.raises(ValueError):
        VerifiedSelectionConfig(minimum_cells=17)


def test_incoherent_features_do_not_override_supported_predecessor(monkeypatch):
    pytest.importorskip("cv2")
    from histopia.registration import _verified_selection as module
    from histopia.registration._adaptive import UnsupportedRegistration

    image = np.random.default_rng(72).integers(30, 230, (200, 240, 3), dtype="uint8")
    mask = np.ones(image.shape[:2], bool)
    a = np.array([[20 + 20 * i, 25] for i in range(8)], dtype=float)
    b = a + [0, 90]
    monkeypatch.setattr(module, "_matches", lambda *args: ([], a, b))
    kwargs = dict(
        source_mask=mask,
        target_mask=mask,
        source_physical=np.eye(3),
        target_physical=np.eye(3),
        source_modality="he",
        target_modality="he",
    )
    seeds = [
        AlignmentSeed("previous", np.eye(3)),
        AlignmentSeed("enlarged", np.diag([1.6, 1.6, 1])),
    ]
    with pytest.raises(UnsupportedRegistration):
        select_verified_registration(image, image, seeds, **kwargs)
    result, evidence = select_verified_registration(
        image,
        image,
        seeds,
        **kwargs,
        config=VerifiedSelectionConfig(abstain_on_incoherent_features=True),
    )
    np.testing.assert_array_equal(result.matrix, np.eye(3))
    assert evidence["feature_evidence"].startswith("inconclusive")
    assert result.candidates[0].diagnostics["verification_supported"] == 0
    assert not result.candidates[1].accepted
