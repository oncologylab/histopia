import json
from pathlib import Path

import numpy as np

from histopia.registration._ordering import (
    order_is_approved,
    propose_anchored_order,
    summarize_cavity_continuity,
    write_order_proposal,
)
from histopia.registration._pipeline import (
    _mask_hole_topology_distance,
    _mask_shape_distance,
    _read_fixed_positions,
)


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
    assert proposal.adjacent_distances == (0.1, 0.2, 0.1)


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


def test_order_proposal_preserves_review_metadata_only_for_same_fingerprint(
    tmp_path: Path,
) -> None:
    first = propose_anchored_order(
        ("HE.ndpi", "IHC.ndpi"),
        np.array([[0.0, 0.2], [0.2, 0.0]]),
        {"HE.ndpi": 1},
    )
    path = tmp_path / "order.json"
    write_order_proposal(path, first)
    payload = json.loads(path.read_text())
    payload.update(
        {
            "approved": True,
            "reviewer": "Reviewer",
            "reviewed_at": "2026-07-24T10:00:00+00:00",
            "notes": "Reviewed.",
        }
    )
    path.write_text(json.dumps(payload))

    write_order_proposal(path, first)
    preserved = json.loads(path.read_text())
    assert preserved["approved"] is True
    assert preserved["reviewer"] == "Reviewer"

    changed = propose_anchored_order(
        ("HE.ndpi", "IHC.ndpi"),
        np.array([[0.0, 0.3], [0.3, 0.0]]),
        {"HE.ndpi": 1},
    )
    write_order_proposal(path, changed)
    invalidated = json.loads(path.read_text())
    assert invalidated["approved"] is False
    assert "reviewer" not in invalidated


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
    distances = np.array([[0.0, 0.2, 0.8], [0.2, 0.0, 0.3], [0.8, 0.3, 0.0]])

    baseline = propose_anchored_order(names, distances, {"HE": 1})
    changed_anchor = propose_anchored_order(names, distances, {"HE": 1, "B": 3})
    changed_distances = propose_anchored_order(
        names,
        distances + np.array([[0.0, 0.1, 0.0], [0.1, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        {"HE": 1},
    )

    assert baseline.fingerprint != changed_anchor.fingerprint
    assert baseline.fingerprint != changed_distances.fingerprint


def test_order_fingerprint_changes_with_accepted_mask_input() -> None:
    names = ("HE", "A")
    distances = np.array([[0.0, 0.2], [0.2, 0.0]])
    baseline = propose_anchored_order(
        names,
        distances,
        {"HE": 1},
        input_fingerprints={"HE": "mask-a", "A": "mask-b"},
    )
    changed = propose_anchored_order(
        names,
        distances,
        {"HE": 1},
        input_fingerprints={"HE": "mask-a", "A": "mask-c"},
    )

    assert baseline.fingerprint != changed.fingerprint
    assert baseline.to_json_dict()["schema_version"] == 3
    assert baseline.to_json_dict()["input_fingerprints"]["A"] == "mask-b"


def test_order_proposal_records_physical_calibration() -> None:
    proposal = propose_anchored_order(
        ("HE", "A"),
        np.array([[0.0, 0.2], [0.2, 0.0]]),
        {"HE": 1},
        physical_areas_um2={"HE": 2_000_000.0, "A": 1_800_000.0},
    )

    payload = proposal.to_json_dict()

    assert payload["physically_calibrated"] is True
    assert payload["slides"][1]["distance_from_previous"] == 0.2
    assert payload["slides"][1]["physical_tissue_area_um2"] == 1_800_000.0


def test_order_proposal_records_cavity_continuity() -> None:
    names = ("HE", "A", "B", "C", "D", "E")
    fractions = {
        "HE": 0.05,
        "A": 0.03,
        "B": 0.0,
        "C": 0.0,
        "D": 0.02,
        "E": 0.06,
    }
    proposal = propose_anchored_order(
        names,
        np.zeros((6, 6)),
        {"HE": 1, "A": 2, "B": 3, "C": 4, "D": 5, "E": 6},
        cavity_fractions=fractions,
    )

    payload = proposal.to_json_dict()

    assert payload["schema_version"] == 3
    assert payload["cavity_continuity"]["blocks"] == [
        {"start_order": 1, "end_order": 2},
        {"start_order": 5, "end_order": 6},
    ]
    assert payload["cavity_continuity"]["review_recommended"] is True
    assert payload["slides"][0]["largest_internal_cavity_fraction"] == 0.05


def test_cavity_continuity_bridges_borderline_single_slide_gaps() -> None:
    names = tuple(str(index) for index in range(1, 9))
    fractions = {
        "1": 0.02,
        "2": 0.05,
        "3": 0.02,
        "4": 0.0,
        "5": 0.03,
        "6": 0.05,
        "7": 0.02,
        "8": 0.0,
    }

    summary = summarize_cavity_continuity(names, fractions)

    assert summary.blocks == ((1, 7),)
    assert summary.review_recommended is False


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


def test_hole_topology_distance_penalizes_significant_cavity_jump() -> None:
    solid = np.zeros((100, 120), dtype=bool)
    solid[10:90, 15:105] = True
    small_hole = solid.copy()
    small_hole[48:52, 58:62] = False
    large_hole = solid.copy()
    large_hole[35:65, 45:75] = False
    similar_large_hole = solid.copy()
    similar_large_hole[34:66, 44:76] = False

    assert _mask_hole_topology_distance(solid, small_hole) == 0
    assert _mask_hole_topology_distance(solid, large_hole) == 1
    assert 0 < _mask_hole_topology_distance(large_hole, similar_large_hole) < 1


def test_hole_topology_distance_is_continuous_near_review_threshold() -> None:
    tissue = np.zeros((100, 100), dtype=bool)
    tissue[10:90, 10:90] = True
    below = tissue.copy()
    below[45:54, 45:55] = False
    above = tissue.copy()
    above[45:55, 45:55] = False

    distance = _mask_hole_topology_distance(below, above)

    assert 0 < distance < 0.05
