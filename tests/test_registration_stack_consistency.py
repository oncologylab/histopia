import numpy as np
import pytest

from histopia.registration._stack_consistency import (
    StackEdge,
    compose_stack_paths,
    optimize_stack_corrections,
)


def shift(x):
    return np.array([[1.0, 0, x], [0, 1, 0], [0, 0, 1]])


def test_serial_composition_direction_and_missing_planes():
    probes = np.array([[0.0, 0], [100, 0], [0, 100]])
    edges = [
        StackEdge("b", "a", shift(10), 0.9, probes),
        StackEdge("c", "b", shift(20), 0.9, probes),
    ]
    poses, paths = compose_stack_paths(["a", "b", "c", "missing"], "a", edges)
    np.testing.assert_allclose(poses["c"], shift(30))
    assert poses["missing"] is None
    assert paths["c"] == ["c", "b", "a"]


def test_pose_corrections_preserve_existing_area_and_reference():
    probes = np.array([[0.0, 0], [100, 0], [0, 100]])
    edges = [StackEdge("b", "a", shift(10), 1, probes)]
    initial = {"a": np.eye(3), "b": shift(15), "missing": None}
    output, evidence = optimize_stack_corrections(initial, "a", edges)
    np.testing.assert_allclose(output["a"], np.eye(3))
    np.testing.assert_allclose(output["b"], shift(10), atol=1e-4)
    assert output["missing"] is None
    assert np.linalg.det(output["b"][:2, :2]) == pytest.approx(1)
    assert not evidence["independent_accuracy"]


def test_cross_stack_edge_is_rejected():
    with pytest.raises(ValueError, match="crosses"):
        compose_stack_paths(
            ["a", "b"], "a", [StackEdge("foreign", "a", np.eye(3), 1, np.zeros((1, 2)))]
        )


def test_nonconverged_optimizer_retains_initial_poses(monkeypatch):
    from types import SimpleNamespace

    import scipy.optimize

    monkeypatch.setattr(
        scipy.optimize,
        "least_squares",
        lambda *args, **kwargs: SimpleNamespace(
            x=np.array([0.0, 100.0, 100.0]), success=False, message="limit"
        ),
    )
    initial = {"a": np.eye(3), "b": shift(15)}
    edges = [StackEdge("b", "a", shift(10), 1, np.array([[0.0, 0.0], [100.0, 100.0]]))]
    output, evidence = optimize_stack_corrections(initial, "a", edges)
    np.testing.assert_array_equal(output["b"], initial["b"])
    assert not evidence["corrections_applied"]
    assert evidence["status"] == "optimizer_limit"
