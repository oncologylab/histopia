import json
from pathlib import Path

import numpy as np

from histopia.registration._ordering import (
    order_is_approved,
    propose_anchored_order,
    write_order_proposal,
)
from histopia.registration._pipeline import _mask_shape_distance, _read_fixed_positions


def test_anchored_order_preserves_fixed_slots() -> None:
    names = ("HE.ndpi", "A.ndpi", "B.ndpi", "C.ndpi")
    distances = np.array(
        [
            [0.0, 0.1, 0.8, 0.9],
            [0.1, 0.0, 0.2, 0.8],
            [0.8, 0.2, 0.0, 0.1],
            [0.9, 0.8, 0.1, 0.0],
        ]
    )

    proposal = propose_anchored_order(names, distances, {"HE.ndpi": 1, "C.ndpi": 4})

    assert proposal.slides == names
    assert proposal.fixed_positions == {"HE.ndpi": 1, "C.ndpi": 4}


def test_order_approval_is_bound_to_fingerprint(tmp_path: Path) -> None:
    proposal = propose_anchored_order(
        ("HE.ndpi", "IHC.ndpi"),
        np.array([[0.0, 0.2], [0.2, 0.0]]),
        {"HE.ndpi": 1},
    )
    path = tmp_path / "order.json"
    write_order_proposal(path, proposal)
    payload = json.loads(path.read_text())
    payload["approved"] = True
    path.write_text(json.dumps(payload))

    assert order_is_approved(path, proposal.fingerprint)
    assert not order_is_approved(path, "different")

    write_order_proposal(path, proposal)
    assert json.loads(path.read_text())["approved"] is True


def test_anchored_order_optimizes_across_fixed_middle_slot() -> None:
    names = ("A", "B", "ANCHOR", "C", "D")
    positions = {name: index for index, name in enumerate(names)}
    distances = np.full((5, 5), 9.0)
    np.fill_diagonal(distances, 0.0)
    for first, second in zip(names, names[1:], strict=False):
        i, j = positions[first], positions[second]
        distances[i, j] = distances[j, i] = 0.1

    proposal = propose_anchored_order(
        tuple(reversed(names)), distances, {"ANCHOR": 3}, beam_width=128
    )

    assert proposal.slides in {names, tuple(reversed(names))}
    assert proposal.slides[2] == "ANCHOR"
    assert proposal.runner_up_objective is not None


def test_anchored_order_only_assigns_unfixed_slots() -> None:
    names = ("HE", "A", "B", "C", "D", "END")
    distances = np.ones((6, 6), dtype=float)
    np.fill_diagonal(distances, 0.0)

    proposal = propose_anchored_order(
        names,
        distances,
        {"HE": 1, "C": 3, "END": 6},
    )

    assert proposal.slides[0] == "HE"
    assert proposal.slides[2] == "C"
    assert proposal.slides[5] == "END"
    assert set(proposal.slides[index] for index in (1, 3, 4)) == {"A", "B", "D"}


def test_order_fingerprint_changes_with_anchor_or_morphology() -> None:
    names = ("HE", "A", "B")
    distances = np.array(
        [[0.0, 0.2, 0.8], [0.2, 0.0, 0.3], [0.8, 0.3, 0.0]]
    )

    baseline = propose_anchored_order(names, distances, {"HE": 1})
    changed_anchor = propose_anchored_order(names, distances, {"HE": 1, "B": 3})
    changed_distances = propose_anchored_order(
        names,
        distances + np.array([[0.0, 0.1, 0.0], [0.1, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        {"HE": 1},
    )

    assert baseline.fingerprint != changed_anchor.fingerprint
    assert baseline.fingerprint != changed_distances.fingerprint


def test_fixed_position_reader_rejects_unknown_anchor(tmp_path: Path) -> None:
    order_path = tmp_path / "anchors.csv"
    order_path.write_text("slide,order\nmissing.ndpi,1\n")

    with np.testing.assert_raises_regex(ValueError, "do not match"):
        _read_fixed_positions((tmp_path / "HE.ndpi",), order_path)


def test_shape_distance_penalizes_topology_jump() -> None:
    one_piece = np.zeros((100, 120), dtype=bool)
    one_piece[20:80, 25:95] = True
    two_pieces = one_piece.copy()
    two_pieces[:, 57:63] = False

    assert _mask_shape_distance(one_piece, one_piece) == 0
    assert _mask_shape_distance(one_piece, two_pieces) > 0
