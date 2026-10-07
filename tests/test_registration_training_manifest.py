import pytest

from histopia.registration._training_manifest import validate_reviewed_training_rows


def rows():
    return [
        dict(
            image_id=str(i),
            source="internal",
            subject_id=str(i),
            block_id=f"pancreas-{i}",
            split=split,
            review_status="reviewed",
            annotations_complete=True,
            annotation_path=f"annotations/{i}.json",
            role="development_exposed",
        )
        for i, split in [(1, "train"), (2, "validation")]
    ]


def test_reviewed_training_and_validation():
    assert validate_reviewed_training_rows(rows()) == {"train": 1, "validation": 1}


def test_mouse_cannot_change_split_with_organ():
    items = rows()
    items[1].update(subject_id="1", block_id="lung-1")
    with pytest.raises(ValueError, match="multiple"):
        validate_reviewed_training_rows(items)


@pytest.mark.parametrize(
    "update",
    [
        {"review_status": "pending"},
        {"annotations_complete": False},
        {"annotation_path": None},
        {"is_independent_evaluation": True},
        {"role": "unassigned"},
    ],
)
def test_unreviewed_or_evaluation_labels_are_rejected(update):
    items = rows()
    items[0].update(update)
    with pytest.raises(ValueError):
        validate_reviewed_training_rows(items)


def test_public_specimen_group_cannot_leak_across_blocks():
    items = rows()
    for item in items:
        item.update(source="ANHIR", specimen_group="same-known-mouse")
    with pytest.raises(ValueError, match="multiple"):
        validate_reviewed_training_rows(items)
