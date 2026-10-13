import numpy as np

from histopia.registration._dense_matching import (
    balanced_correspondences,
    normalized_to_pixel_centres,
)


def test_normalized_grid_is_pixel_centred_with_anisotropic_dimensions():
    points = np.array([[-1 + 1 / 200, -1 + 1 / 100], [1 - 1 / 200, 1 - 1 / 100]])
    np.testing.assert_allclose(
        normalized_to_pixel_centres(points, (100, 200)), [[0, 0], [199, 99]], atol=1e-12
    )


def test_correspondences_exclude_missing_support_and_balance_both_frames():
    a = np.array([[1, 1], [2, 2], [30, 30], [60, 60], [99, 99], [np.nan, 4]])
    b = np.array([[1, 1], [2, 2], [1, 1], [60, 60], [99, 99], [4, 4]])
    mask = np.ones((100, 100), bool)
    mask[99, 99] = False
    aa, bb, confidence = balanced_correspondences(
        a, b, np.arange(6, 0, -1), mask, mask, bins=4, per_bin=1
    )
    np.testing.assert_array_equal(aa, [[1, 1], [60, 60]])
    np.testing.assert_array_equal(bb, [[1, 1], [60, 60]])
    np.testing.assert_array_equal(confidence, [6, 3])


def test_optional_collector_skips_fitting_and_keeps_four_orientations(monkeypatch):
    import contextlib
    import sys
    import types

    import pytest

    pytest.importorskip("cv2")
    pytest.importorskip("PIL")
    from histopia.registration import _dense_matching as module
    from histopia.registration._guarded import GuardedRegistrationConfig, ImageFrame

    # Exercise the adapter without importing or requiring torch/model checkpoints.
    fake_torch = types.SimpleNamespace(
        manual_seed=lambda seed: None,
        cuda=types.SimpleNamespace(is_available=lambda: False),
        inference_mode=contextlib.nullcontext,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class Tensor:
        def __init__(self, value):
            self.value = value

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.value

    class Model:
        def match(self, a, b):
            return None

        def sample(self, prediction, count):
            return Tensor(np.zeros((1, 4))), Tensor(np.ones(1))

    def never_fit(*args, **kwargs):
        raise AssertionError("unpartitioned dense coordinates must not enter fitting")

    monkeypatch.setattr(module, "candidate_from_points", never_fit)
    image = np.full((100, 160, 3), 127, "uint8")
    frame = ImageFrame(
        image, np.ones((100, 160), bool), np.eye(3), 1.0, (0, 0, 160, 100)
    )
    collected = []
    candidates, evidence = module.dense_candidates(
        frame,
        frame,
        backend="romav2",
        model=Model(),
        config=GuardedRegistrationConfig(),
        generic_unknown=True,
        correspondence_collector=collected.append,
        fit_correspondences=False,
    )
    assert not candidates
    assert len(evidence) == len(collected) == 12
    for name in {record["channel"] for record in collected}:
        assert {
            record["rotation"] for record in collected if record["channel"] == name
        } == {0, 90, 180, 270}
    for record in collected:
        np.testing.assert_allclose(record["source_xy"], [[79.5, 49.5]])
        assert record["coordinate_units"] == "working_pixel_centres"
