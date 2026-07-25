from __future__ import annotations

import json

import numpy as np

from histopia.registration._ordering_cache import (
    load_ordering_distance_cache,
    load_ordering_proposal_cache,
    ordering_cache_fingerprint,
    ordering_proposal_cache_fingerprint,
    write_ordering_distance_cache,
    write_ordering_proposal_cache,
)


def test_ordering_distance_cache_round_trip_and_stale_rejection(tmp_path) -> None:
    path = tmp_path / "cache" / "distances.npz"
    matrix = np.array(
        [
            [0.0, 0.2, 0.4],
            [0.2, 0.0, 0.3],
            [0.4, 0.3, 0.0],
        ]
    )
    fingerprint = ordering_cache_fingerprint(
        ("a", "b", "c"),
        {"a": "1", "b": "2", "c": "3"},
        {"method": "feature"},
    )

    write_ordering_distance_cache(path, matrix, fingerprint=fingerprint)

    loaded = load_ordering_distance_cache(
        path,
        expected_fingerprint=fingerprint,
        expected_size=3,
    )
    assert loaded is not None
    assert np.array_equal(loaded, matrix)
    assert (
        load_ordering_distance_cache(
            path,
            expected_fingerprint="stale",
            expected_size=3,
        )
        is None
    )


def test_ordering_distance_cache_rejects_corruption(tmp_path) -> None:
    path = tmp_path / "distances.npz"
    with path.open("wb") as stream:
        np.savez(
            stream,
            fingerprint=np.asarray("expected"),
            distances=np.array([[0.0, 0.2], [0.2, 0.0]]),
            matrix_sha256=np.asarray("wrong"),
        )

    assert (
        load_ordering_distance_cache(
            path,
            expected_fingerprint="expected",
            expected_size=2,
        )
        is None
    )


def test_ordering_proposal_cache_round_trip_and_exact_invalidation(tmp_path) -> None:
    path = tmp_path / "proposal.json"
    matrix = np.array(
        [
            [0.0, 0.2, 0.4],
            [0.2, 0.0, 0.3],
            [0.4, 0.3, 0.0],
        ]
    )
    inputs = {
        "distance_fingerprint": "distance-v1",
        "distances": matrix,
        "fixed_positions": {"a": 1},
        "physical_areas_um2": {"a": 1.0, "b": 2.0, "c": None},
        "input_fingerprints": {"a": "1", "b": "2", "c": "3"},
        "orientation_quarter_turns": {"a": 0, "b": 1, "c": 0},
        "cavity_fractions": {"a": 0.0, "b": 0.1, "c": 0.2},
    }
    fingerprint = ordering_proposal_cache_fingerprint(**inputs)
    write_ordering_proposal_cache(
        path,
        fingerprint=fingerprint,
        slides=("a", "c", "b"),
        runner_up_objective=0.75,
    )

    assert load_ordering_proposal_cache(
        path,
        expected_fingerprint=fingerprint,
    ) == (("a", "c", "b"), 0.75)
    assert (
        load_ordering_proposal_cache(
            path,
            expected_fingerprint="stale",
        )
        is None
    )
    changed = dict(inputs)
    changed["distances"] = matrix + np.eye(3)
    assert ordering_proposal_cache_fingerprint(**changed) != fingerprint


def test_ordering_proposal_cache_rejects_corruption_and_symlinks(tmp_path) -> None:
    path = tmp_path / "proposal.json"
    write_ordering_proposal_cache(
        path,
        fingerprint="current",
        slides=("a", "b"),
        runner_up_objective=None,
    )
    payload = json.loads(path.read_text())
    payload["proposal"]["slides"].reverse()
    path.write_text(json.dumps(payload))

    assert load_ordering_proposal_cache(path, expected_fingerprint="current") is None

    target = tmp_path / "target.json"
    path.replace(target)
    path.symlink_to(target)
    assert load_ordering_proposal_cache(path, expected_fingerprint="current") is None
