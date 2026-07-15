import numpy as np
import pytest

from histopia.registration import apply_quarter_turn, orient_section_group


def _asymmetric_mask() -> np.ndarray:
    mask = np.zeros((120, 120), dtype=bool)
    mask[20:95, 25:50] = True
    mask[70:95, 25:95] = True
    mask[30:45, 50:65] = True
    return mask


def test_orientation_matches_quarter_turns_to_fixed_anchor() -> None:
    anchor = _asymmetric_mask()
    masks = {
        "anchor": anchor,
        "rotated-90": np.rot90(anchor, 1),
        "rotated-180": np.rot90(anchor, 2),
    }

    result = orient_section_group(masks, anchor="anchor")

    assert result.anchor == "anchor"
    assert result.decisions["anchor"].quarter_turns_ccw == 0
    assert result.decisions["rotated-90"].quarter_turns_ccw == 3
    assert result.decisions["rotated-180"].quarter_turns_ccw == 2
    assert result.decisions["rotated-90"].score > 0.95


def test_orientation_uses_dominant_object_instead_of_debris() -> None:
    anchor = _asymmetric_mask()
    rotated = np.rot90(anchor, 1)
    rotated[2:8, 2:8] = True

    result = orient_section_group(
        {"anchor": anchor, "with-debris": rotated},
        anchor="anchor",
    )

    assert result.decisions["with-debris"].quarter_turns_ccw == 3


def test_orientation_uses_multi_object_topology() -> None:
    anchor = _asymmetric_mask()
    anchor[15:35, 85:105] = True
    rotated = np.rot90(anchor, 2)

    result = orient_section_group(
        {"anchor": anchor, "rotated": rotated},
        anchor="anchor",
    )

    assert result.decisions["rotated"].quarter_turns_ccw == 2


def test_low_confidence_orientation_remains_unrotated() -> None:
    symmetric = np.zeros((120, 120), dtype=bool)
    symmetric[30:90, 30:90] = True

    result = orient_section_group(
        {"anchor": symmetric, "peer": np.rot90(symmetric)},
        anchor="anchor",
    )

    assert result.decisions["peer"].quarter_turns_ccw == 0


def test_clear_matching_aspects_exclude_quarter_turns() -> None:
    anchor = np.zeros((80, 120), dtype=bool)
    anchor[20:55, 10:105] = True
    peer = anchor.copy()
    peer[20:55, 10:55] = False
    peer[25:60, 65:110] = True

    result = orient_section_group(
        {"anchor": anchor, "peer": peer},
        anchor="anchor",
        minimum_confidence_margin=0,
    )

    assert result.decisions["peer"].quarter_turns_ccw in (0, 2)
    assert result.decisions["peer"].confidence_margin == 0


def test_orientation_fingerprint_changes_with_mask() -> None:
    mask = _asymmetric_mask()
    first = orient_section_group({"a": mask, "b": mask}, anchor="a")
    changed = mask.copy()
    changed[100:105, 100:105] = True
    second = orient_section_group({"a": mask, "b": changed}, anchor="a")

    assert first.fingerprint != second.fingerprint


def test_apply_quarter_turn_preserves_channels() -> None:
    image = np.arange(3 * 4 * 3).reshape(3, 4, 3)

    rotated = apply_quarter_turn(image, 1)

    assert rotated.shape == (4, 3, 3)
    assert np.array_equal(rotated, np.rot90(image, 1))


def test_orientation_rejects_unknown_anchor() -> None:
    with pytest.raises(ValueError, match="unknown orientation anchor"):
        orient_section_group({"a": _asymmetric_mask()}, anchor="missing")
