import numpy as np
import pytest

from histopia.registration import _agreement as module
from histopia.registration._adaptive import UnsupportedRegistration
from histopia.registration._candidate_refinement import AlignmentSeed
from histopia.registration._guarded import RegistrationCandidate
from histopia.registration._verified_selection import VerifiedSelectionConfig


def _points():
    return np.array(
        [
            [60, 10],
            [70, 15],
            [160, 10],
            [170, 20],
            [10, 60],
            [20, 70],
            [110, 60],
            [120, 70],
            [60, 110],
            [70, 120],
            [160, 110],
            [170, 120],
        ],
        dtype=float,
    )


def _translation(x, y=0):
    return np.array([[1, 0, x], [0, 1, y], [0, 0, 1]], dtype=float)


def test_distinct_endpoints_cannot_inflate_support():
    a = np.repeat(_points()[:2], 20, axis=0)
    result = module.directional_correspondence_evidence(
        np.eye(3), a, a, (200, 200), (200, 200)
    )
    assert result["verification_supported"] == 2
    assert result["verification_cells"] == 1


def test_correspondence_tolerance_belongs_to_each_working_frame():
    # A 5-pixel target residual corresponds to 10 pixels in the source frame.
    # Image diagonal normalization would incorrectly make this pass.
    a = _points()
    matrix = np.diag([0.5, 0.5, 1])
    b = a * 0.5 + [5, 0]
    result = module.directional_correspondence_evidence(
        matrix, a, b, (200, 200), (100, 100)
    )
    assert result["verification_supported"] == 0
    assert result["forward_median_residual_px"] == 5
    assert result["reverse_median_residual_px"] == 10


def test_independent_reverse_disagreement_and_anisotropic_calibration():
    forward = _translation(20)
    reverse = _translation(-15)
    evidence = module.independent_direction_agreement(
        forward,
        reverse,
        _points(),
        _points() + [20, 0],
        source_working_to_physical=np.diag([2, 5, 1]),
        target_working_to_physical=np.diag([3, 7, 1]),
    )
    assert evidence["agrees"]
    assert evidence["frames"]["source"]["p90_cycle_error_physical"] == 10
    assert evidence["frames"]["target"]["p90_cycle_error_physical"] == 15
    wrong = module.independent_direction_agreement(
        forward,
        _translation(-10),
        _points(),
        _points() + [20, 0],
        source_working_to_physical=np.eye(3),
        target_working_to_physical=np.eye(3),
    )
    assert not wrong["agrees"]


def test_calibrated_roundtrip_with_rotated_unequal_frames_and_offsets():
    f = np.array([[0, -0.5, 73], [2, 0, 9], [0, 0, 1.0]])
    a = _points()
    b = module._mapped(f, a)
    # Reverse is supplied as a separately obtained mapping, with known values.
    r = np.array([[0, 0.5, -4.5], [-2, 0, 146], [0, 0, 1.0]])
    evidence = module.independent_direction_agreement(
        f,
        r,
        a,
        b,
        source_working_to_physical=np.array([[0.25, 0, 12], [0, 0.5, 7], [0, 0, 1]]),
        target_working_to_physical=np.array([[1, 0, 5], [0, 0.25, 3], [0, 0, 1]]),
    )
    assert evidence["agrees"]
    assert evidence["frames"]["source"]["p90_cycle_error_px"] == 0


def test_missing_agreement_points_never_establish_support():
    empty = np.empty((0, 2))
    evidence = module.independent_direction_agreement(
        np.eye(3),
        np.eye(3),
        empty,
        empty,
        source_working_to_physical=np.eye(3),
        target_working_to_physical=np.eye(3),
    )
    assert not evidence["agrees"]


def _run(monkeypatch, *, reverse_offset=0, absent_family=False, prior=None):
    pytest.importorskip("cv2")
    calls = []
    image = np.random.default_rng(72).integers(30, 230, (200, 200, 3), dtype="uint8")
    mask = np.ones((200, 200), bool)

    def direction(sf, tf, sc, tc, **kwargs):
        direction = kwargs["direction"]
        calls.append(direction)
        h = _translation(reverse_offset if direction == "reverse" else 0)
        a = _points()
        b = module._mapped(h, a)
        checks = {"rootsift": (a, b), "romav2": (a, b)}
        if absent_family:
            checks["romav2"] = (a[:2], b[:2])
        candidate = RegistrationCandidate(h, direction + "_fit")
        return [candidate], checks, [{"direction": direction}]

    monkeypatch.setattr(module, "_direction_matches", direction)
    prior = _translation(10) if prior is None else prior
    result, diagnostics = module.select_consensus_registration(
        image,
        image,
        [AlignmentSeed("adaptive_previous", prior)],
        model=object(),
        source_mask=mask,
        target_mask=mask,
        source_physical=np.eye(3),
        target_physical=np.eye(3),
    )
    return result, diagnostics, calls


def test_replace_predecessor_only_with_both_families_and_separate_reverse(monkeypatch):
    result, diagnostics, calls = _run(monkeypatch)
    assert calls == ["forward", "reverse"]
    assert result.selected_stage == "forward_fit"
    assert (
        result.candidates[0].diagnostics["independently_fitted_reverse_stage"]
        == "reverse_fit"
    )
    assert diagnostics["evaluation_landmarks_used"] is False
    assert diagnostics["independent_anatomical_accuracy"] is False
    assert len(diagnostics["predecessor_transform_sha256"]) == 64


def test_conflicting_independent_reverse_retains_predecessor(monkeypatch):
    result, diagnostics, calls = _run(monkeypatch, reverse_offset=15)
    assert calls == ["forward", "reverse"]
    assert result.selected_stage == "adaptive_previous"
    assert diagnostics["predecessor_retention"]
    assert result.candidates[0].diagnostics["reverse_agreement"][0]["agrees"] is False


def test_missing_matcher_family_retains_predecessor(monkeypatch):
    result, diagnostics, _ = _run(monkeypatch, absent_family=True)
    assert result.selected_stage == "adaptive_previous"
    assert diagnostics["predecessor_retention"]


def test_unacceptable_predecessor_and_conflict_are_unsupported(monkeypatch):
    with pytest.raises(UnsupportedRegistration, match="no supported matcher"):
        _run(monkeypatch, reverse_offset=15, prior=np.diag([1.6, 1.6, 1]))


def test_consensus_requires_rotation_safe_partition():
    pytest.importorskip("cv2")
    with pytest.raises(ValueError, match="source_spatial"):
        module.select_consensus_registration(
            None,
            None,
            [],
            model=object(),
            source_mask=None,
            target_mask=None,
            source_physical=np.eye(3),
            target_physical=np.eye(3),
            config=VerifiedSelectionConfig(),
        )


def test_missing_or_duplicate_predecessor_rejected_before_matcher_calls():
    pytest.importorskip("cv2")
    kwargs = dict(
        model=object(),
        source_mask=None,
        target_mask=None,
        source_physical=np.eye(3),
        target_physical=np.eye(3),
    )
    with pytest.raises(ValueError):
        module.select_consensus_registration(None, None, [], **kwargs)
    with pytest.raises(ValueError):
        module.select_consensus_registration(
            None, None, [AlignmentSeed("adaptive_previous", np.eye(3))] * 2, **kwargs
        )


def test_dense_correspondences_are_partitioned_before_fitting(monkeypatch):
    pytest.importorskip("cv2")
    from histopia.registration._guarded import GuardedRegistrationConfig, ImageFrame

    fitting = _points() + [50, 0]  # Opposite parity, while retaining valid coordinates.
    fitting = fitting[fitting[:, 0] < 200]
    checking = _points()
    all_points = np.concatenate([fitting, checking])
    frame = ImageFrame(
        np.zeros((200, 200, 3), "uint8"),
        np.ones((200, 200), bool),
        np.eye(3),
        1.0,
        (0, 0, 200, 200),
    )
    calls, artifacts = [], []
    monkeypatch.setattr(module, "_matches", lambda *args: ([], checking, checking))

    def fake_dense(*args, **kwargs):
        assert kwargs["fit_correspondences"] is False
        assert kwargs["generic_unknown"] is True
        kwargs["correspondence_collector"](
            {
                "backend": "romav2",
                "channel": "rgb",
                "rotation": 90,
                "source_xy": all_points,
                "target_xy": all_points,
                "confidence": np.ones(len(all_points)),
            }
        )
        return [], []

    def capture_fit(a, b, **kwargs):
        calls.append((a.copy(), b.copy()))
        return RegistrationCandidate(np.eye(3), kwargs["stage"])

    monkeypatch.setattr(module, "dense_candidates", fake_dense)
    monkeypatch.setattr(module, "candidate_from_points", capture_fit)
    candidates, checks, _ = module._direction_matches(
        frame,
        frame,
        {},
        {},
        model=object(),
        guarded=GuardedRegistrationConfig(),
        config=VerifiedSelectionConfig(partition_policy="source_spatial"),
        sm="unknown",
        tm="unknown",
        direction="reverse",
        collector=artifacts.append,
    )
    assert len(candidates) == len(calls) == 1
    parity, _ = module.spatial_partition(calls[0][0], (200, 200))
    assert (parity == 0).all()
    parity, _ = module.spatial_partition(checks["romav2"][0], (200, 200))
    assert (parity == 1).all()
    assert artifacts[-1]["direction"] == "reverse"


def test_cross_channel_boundary_jitter_cannot_leak_fitting_endpoints(monkeypatch):
    pytest.importorskip("cv2")
    from histopia.registration._guarded import GuardedRegistrationConfig, ImageFrame

    fit_a = np.array([[49.9, 10], [25, 10]])
    fit_b = np.array([[40, 10], [99.9, 15]])
    check_a = np.array([[50.1, 10], [60, 20], [160, 10]])
    check_b = np.array([[140, 10], [100.1, 15], [160, 10]])
    frame = ImageFrame(
        np.zeros((200, 200, 3), "uint8"),
        np.ones((200, 200), bool),
        np.eye(3),
        1.0,
        (0, 0, 200, 200),
    )
    artifacts = []
    monkeypatch.setattr(module, "_matches", lambda *args: ([], check_a, check_b))
    monkeypatch.setattr(module, "candidate_from_points", lambda *args, **kwargs: None)

    def fake_dense(*args, **kwargs):
        for rotation, a, b in [(0, fit_a, fit_b), (90, check_a, check_b)]:
            kwargs["correspondence_collector"](
                {
                    "backend": "romav2",
                    "channel": "rgb",
                    "rotation": rotation,
                    "source_xy": a,
                    "target_xy": b,
                    "confidence": np.ones(len(a)),
                }
            )
        return [], []

    monkeypatch.setattr(module, "dense_candidates", fake_dense)
    _, checks, records = module._direction_matches(
        frame,
        frame,
        {},
        {},
        model=object(),
        guarded=GuardedRegistrationConfig(),
        config=VerifiedSelectionConfig(partition_policy="source_spatial"),
        sm="unknown",
        tm="unknown",
        direction="forward",
        collector=artifacts.append,
    )
    # Neither matching source nor matching target endpoint bins can cross pools.
    for family in ["rootsift", "romav2"]:
        np.testing.assert_array_equal(checks[family][0], [[160, 10]])
        np.testing.assert_array_equal(checks[family][1], [[160, 10]])
        summary = next(
            r
            for r in records
            if r["backend"] == family and "fit_endpoint_exclusions" in r
        )
        assert summary["fit_endpoint_exclusions"] == 2
        saved = next(
            r for r in artifacts if r["backend"] == family and r["role"] == "checking"
        )
        np.testing.assert_array_equal(saved["source_xy"], checks[family][0])


def test_retention_preserves_native_transform_bytes_with_noninteger_frame(monkeypatch):
    pytest.importorskip("cv2")
    from histopia.registration._guarded import ImageFrame
    from histopia.registration._verified_selection import transform_fingerprint

    image = np.random.default_rng(41).integers(30, 230, (200, 200, 3), dtype="uint8")
    frame = ImageFrame(
        image,
        np.ones((200, 200), bool),
        np.array([[1.3, 0, 23.7], [0, 1.1, 16.2], [0, 0, 1]]),
        1.0,
        (0, 0, 200, 200),
    )
    monkeypatch.setattr(module, "component_frame", lambda *args: frame)
    prior = np.array([[0.998, -0.004, 7.321], [0.004, 0.998, -2.678], [0, 0, 1.0]])
    result, diagnostics, _ = _run(monkeypatch, absent_family=True, prior=prior)
    np.testing.assert_array_equal(result.matrix, prior)
    assert diagnostics["retained_predecessor_id"] == "adaptive_previous"
    assert diagnostics[
        "retained_predecessor_transform_sha256"
    ] == transform_fingerprint(prior)
    assert diagnostics["selected_transform_sha256"] == transform_fingerprint(prior)
