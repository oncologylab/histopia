import numpy as np
import pytest

from histopia.registration._stack_consistency import (
    StackEdge,
    compose_stack_paths,
    optimize_stack_corrections,
    propose_stack_references,
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


def edge(source, target, confidence=1.0):
    return StackEdge(source, target, shift(0), confidence, np.zeros((1, 2)))


def test_reference_proposals_preserve_components_and_explicit_isolation():
    result = propose_stack_references(
        ["section-1", "section-3", "section-4", "other", "reference"],
        [edge("section-1", "section-3"), edge("section-3", "section-4")],
        reference="reference",
    )
    assert result.explicit_reference == "reference"
    assert result.explicit_reference_status == "isolated"
    assert result.components[0].image_ids == ("section-1", "section-3", "section-4")
    assert result.components[0].proposed_reference == "section-3"
    assert result.components[0].total_path_cost == pytest.approx(0.02)
    assert result.components[0].supported_degree == 2
    assert [c.image_ids for c in result.components[1:]] == [("other",), ("reference",)]
    assert "isolated" in result.components[-1].reason


def test_reference_minimizes_weighted_cost_before_degree():
    edges = [edge("a", "b", 0.01), edge("b", "c"), edge("c", "d"), edge("d", "e")]
    result = propose_stack_references(list("abcde"), edges, reference="a")
    assert result.components[0].proposed_reference == "c"
    assert result.explicit_reference_status == "connected"
    assert result.explicit_reference == "a"


def test_reference_is_deterministic_and_parallel_edges_do_not_inflate_degree():
    edges = [edge("a", "b"), edge("a", "b", 0.4), edge("b", "c"), edge("c", "d")]
    first = propose_stack_references(list("abcd"), edges)
    second = propose_stack_references(list("dcba"), reversed(edges))
    assert first == second
    assert first.components[0].proposed_reference == "b"
    assert first.components[0].supported_degree == 2
    assert first.explicit_reference_status == "not_requested"


def test_reference_cost_tie_prefers_unique_degree_before_identifier():
    # Equal shortest-path sums at a and b, but b also has a redundant direct edge.
    edges = [edge("a", "b"), edge("a", "c"), edge("b", "d"), edge("b", "c", 0.5)]
    result = propose_stack_references(list("abcd"), edges)
    assert result.components[0].proposed_reference == "b"
    assert result.components[0].supported_degree == 3


@pytest.mark.parametrize("function", [compose_stack_paths, propose_stack_references])
def test_all_graph_consumers_reject_cross_reconstruction_edges(function):
    with pytest.raises(ValueError, match="crosses"):
        function(
            ["a", "b"],
            *(["a"] if function is compose_stack_paths else []),
            [edge("a", "foreign")],
        )


@pytest.mark.parametrize(
    "bad_edge, message",
    [
        (StackEdge("a", "b", np.zeros((3, 3)), 1, np.zeros((1, 2))), "affine"),
        (StackEdge("a", "b", np.eye(3) * np.nan, 1, np.zeros((1, 2))), "affine"),
        (StackEdge("a", "b", np.eye(3), 0, np.zeros((1, 2))), "confidence"),
        (StackEdge("a", "b", np.eye(3), np.nan, np.zeros((1, 2))), "confidence"),
        (StackEdge("a", "b", np.eye(3), 1, np.zeros((0, 2))), "probes"),
        (StackEdge("a", "b", np.eye(3), 1, np.array([[np.nan, 0]])), "probes"),
        (StackEdge("a", "a", np.eye(3), 1, np.zeros((1, 2))), "distinct"),
    ],
)
def test_reference_rejects_invalid_evidence(bad_edge, message):
    with pytest.raises(ValueError, match=message):
        propose_stack_references(["a", "b"], [bad_edge])


def test_reference_empty_duplicate_and_unknown_ids():
    assert propose_stack_references([], []).components == ()
    with pytest.raises(ValueError, match="unique"):
        propose_stack_references(["a", "a"], [])
    with pytest.raises(ValueError, match="member"):
        propose_stack_references(["a"], [], reference="other")
    with pytest.raises(ValueError, match="string"):
        propose_stack_references(["a", ""], [])


def test_optimizer_validates_poses_even_without_supported_edges():
    with pytest.raises(ValueError, match="affine"):
        optimize_stack_corrections({"a": np.zeros((3, 3))}, "a", [])
    with pytest.raises(ValueError, match="bounds"):
        optimize_stack_corrections({"a": np.eye(3)}, "a", [], max_translation_um=np.inf)
