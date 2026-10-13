import numpy as np
import pytest

from histopia.registration._candidate_refinement import AlignmentSeed
from histopia.registration._verified_selection import (
    VerifiedSelectionConfig,
    _matches,
    correspondence_evidence,
    modality_channels,
    partition_correspondences,
    predecessor_binding,
    select_verified_registration,
    spatial_partition,
    transform_fingerprint,
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
    with pytest.raises(ValueError, match="partition policy"):
        VerifiedSelectionConfig(partition_policy="target")


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


@pytest.mark.parametrize(
    "matrix",
    [
        np.array([[0, -1, 119], [1, 0, 0], [0, 0, 1]]),
        np.array([[0, 1, 0], [-1, 0, 199], [0, 0, 1]]),
    ],
)
def test_source_partition_preserves_quarter_turn_matches_with_unequal_sizes(matrix):
    source = np.array([(x, y) for x in [25, 75, 125, 175] for y in [15, 45, 75, 105]])
    target = (np.c_[source, np.ones(len(source))] @ matrix.T)[:, :2]
    old = partition_correspondences(
        source, target, (120, 200), (200, 120), policy="legacy_both"
    )
    assert len(old[0]) == len(old[2]) == 0
    fa, fb, ca, cb = partition_correspondences(
        source, target, (120, 200), (200, 120), policy="source_spatial"
    )
    assert len(fa) == len(ca) == 8
    assert len(fb) == len(cb) == 8
    evidence = correspondence_evidence(matrix, ca, cb, (120, 200), (200, 120))
    assert evidence["verification_supported"] == 8
    assert evidence["verification_source_cells"] == 8
    assert evidence["verification_target_cells"] == 8
    assert not set(map(tuple, spatial_partition(fa, (120, 200))[1])) & set(
        map(tuple, spatial_partition(ca, (120, 200))[1])
    )


def test_source_partition_deduplicates_each_feature_endpoint_deterministically():
    source = np.array([[30, 5], [30.1, 5.1], [5, 30], [8, 31], [80, 5]], float)
    target = np.array([[30, 5], [30.2, 5.2], [5, 30], [5.1, 30.1], [80, 5]], float)
    result = partition_correspondences(source, target, (100, 100), (100, 100))
    shuffled = partition_correspondences(
        source[::-1], target[::-1], (100, 100), (100, 100)
    )
    assert len(result[0]) + len(result[2]) == 3
    for first, second in zip(result, shuffled, strict=True):
        np.testing.assert_array_equal(first, second)


def test_matches_deduplicate_across_channels_before_evidence(monkeypatch):
    from types import SimpleNamespace

    cv2 = pytest.importorskip("cv2")
    from histopia.registration import _verified_selection as module

    xy = np.array([(x, y) for x in [25, 75, 125, 175] for y in [15, 45, 75, 105]])

    class Detector:
        def detectAndCompute(self, *args):
            return [SimpleNamespace(pt=point) for point in xy], np.ones((16, 128))

    monkeypatch.setattr(cv2, "SIFT_create", lambda **kwargs: Detector())
    monkeypatch.setattr(
        module,
        "mutual_ratio_matches",
        lambda *args: np.c_[np.arange(16), np.arange(16)],
    )
    frame = SimpleNamespace(mask=np.ones((120, 200), dtype=bool))
    channels = {"gray": np.zeros((120, 200)), "hematoxylin": np.zeros((120, 200))}
    fits, a, b = _matches(
        frame,
        frame,
        channels,
        channels,
        VerifiedSelectionConfig(partition_policy="source_spatial"),
    )
    assert sum(len(fit[1]) for fit in fits) == 8
    assert len(a) == len(b) == 8
    legacy_fits, legacy_a, _ = _matches(
        frame, frame, channels, channels, VerifiedSelectionConfig()
    )
    assert sum(len(fit[1]) for fit in legacy_fits) == 16
    assert len(legacy_a) == 16


def test_spatial_distribution_requires_both_frames():
    source = np.array([[30, 5], [5, 30], [80, 5], [5, 80]])
    target = source * 0.1
    evidence = correspondence_evidence(
        np.diag([0.1, 0.1, 1]), source, target, (100, 100), (100, 100)
    )
    assert evidence["verification_supported"] == 4
    assert evidence["verification_source_cells"] == 4
    assert evidence["verification_target_cells"] == evidence["verification_cells"] == 1


@pytest.mark.parametrize(
    "seeds",
    [
        [],
        [AlignmentSeed("h12", np.eye(3))],
        [AlignmentSeed("v3", np.eye(3)), AlignmentSeed("v3", np.eye(3))],
    ],
)
def test_predecessor_must_resolve_uniquely(seeds):
    with pytest.raises(ValueError, match="exactly one"):
        predecessor_binding(seeds, "v3")


def test_predecessor_binding_can_be_disabled_and_fingerprints_native_matrix():
    matrix = np.array([[1.0, 0, 17], [0, 1, -9], [0, 0, 1]])
    seeds = [AlignmentSeed("v3", matrix)]
    binding = predecessor_binding(seeds, "v3")
    assert binding["predecessor_id"] == "v3"
    assert binding["predecessor_transform_sha256"] == transform_fingerprint(matrix)
    assert binding["predecessor_transform_sha256"] != transform_fingerprint(np.eye(3))
    assert transform_fingerprint(matrix.astype(">f8")) == transform_fingerprint(matrix)
    assert predecessor_binding(seeds, None)["predecessor_id"] is None


def test_retention_preserves_native_transform_with_crop_and_anisotropic_calibration(
    monkeypatch,
):
    cv2 = pytest.importorskip("cv2")
    from histopia.registration import _verified_selection as module

    image = np.random.default_rng(15).integers(30, 230, (240, 320, 3), dtype="uint8")
    mask = np.zeros(image.shape[:2], bool)
    mask[45:195, 50:265] = True
    matrix = np.array([[1.2, 0, 20], [0, 0.8, 13], [0, 0, 1.0]])
    target = cv2.warpAffine(image, matrix[:2], (420, 230))
    target_mask = cv2.warpAffine(mask.astype("uint8"), matrix[:2], (420, 230)) > 0

    def matches(source_frame, target_frame, *args):
        native = np.array(
            [(x, y) for x in [65, 125, 185, 245] for y in [60, 100, 140, 180]]
        )
        source = (
            np.c_[native, np.ones(16)] @ np.linalg.inv(source_frame.working_to_input).T
        )[:, :2]
        target = (
            np.c_[native, np.ones(16)]
            @ (np.linalg.inv(target_frame.working_to_input) @ matrix).T
        )[:, :2]
        return [], source, target

    monkeypatch.setattr(module, "_matches", matches)
    result, diagnostics = select_verified_registration(
        image,
        target,
        [AlignmentSeed("v3", matrix)],
        source_mask=mask,
        target_mask=target_mask,
        source_physical=np.eye(3),
        target_physical=np.diag([1 / 1.2, 1 / 0.8, 1]),
        predecessor="v3",
        config=VerifiedSelectionConfig(
            generate_features=False, partition_policy="source_spatial"
        ),
    )
    np.testing.assert_array_equal(result.matrix, matrix)
    assert result.source_frame.bbox_xywh[:2] != (0, 0)
    assert diagnostics["retained_predecessor_id"] == "v3"
    assert diagnostics[
        "retained_predecessor_transform_sha256"
    ] == transform_fingerprint(matrix)
    assert diagnostics["selected_transform_sha256"] == transform_fingerprint(matrix)
    np.testing.assert_allclose(
        result.candidates[0].diagnostics["physical_directional_scales"], [1, 1]
    )


@pytest.mark.parametrize("shape", [(0, 100), (100, float("nan")), (100,)])
def test_partition_rejects_invalid_coordinate_frame(shape):
    with pytest.raises(ValueError, match="height and width"):
        spatial_partition(np.empty((0, 2)), shape)
