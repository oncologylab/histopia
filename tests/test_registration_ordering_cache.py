from __future__ import annotations

import numpy as np

from histopia.registration._ordering_cache import (
    load_ordering_distance_cache,
    ordering_cache_fingerprint,
    write_ordering_distance_cache,
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
