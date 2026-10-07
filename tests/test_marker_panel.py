import pytest

from histopia.study._marker_panel import supported_marker_panel


def observation(marker, subject, **changes):
    return dict(
        marker_id=marker,
        subject_id=subject,
        modality="IHC",
        detection="brightfield",
        measurement_accepted=True,
        **changes,
    )


def test_repeated_sections_do_not_increase_subject_support():
    rows = [observation("MYC", "one")] * 20
    assert supported_marker_panel(rows, minimum_subjects=2) == {}
    rows.append(observation("MYC", "two"))
    assert supported_marker_panel(rows, minimum_subjects=2) == {"MYC": ["one", "two"]}


def test_modality_and_acceptance_fail_closed():
    rows = [observation("SMA", "one")]
    for change in (
        {"modality": "H&E"},
        {"modality": "IF"},
        {"detection": "fluorescence"},
        {"measurement_accepted": False},
        {"measurement_accepted": None},
        {"modality": None},
    ):
        candidate = observation("SMA", "two")
        candidate.update(change)
        assert supported_marker_panel(rows + [candidate], minimum_subjects=2) == {}


def test_antibody_and_phospho_identities_remain_separate():
    rows = [
        observation(m, s)
        for m in ["MYC", "MYC(CST)", "ERK", "pERK"]
        for s in ["a", "b"]
    ]
    panel = supported_marker_panel(rows, minimum_subjects=2, exclude=["ERK"])
    assert list(panel) == ["MYC", "MYC(CST)", "pERK"]


@pytest.mark.parametrize("minimum", [True, 1, 0, 2.5])
def test_invalid_subject_threshold(minimum):
    with pytest.raises(ValueError):
        supported_marker_panel([], minimum_subjects=minimum)
