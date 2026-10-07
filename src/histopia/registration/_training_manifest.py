"""Small validation gates for reviewed tissue-model training inputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def validate_reviewed_training_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    """Reject unreviewed masks, evaluation references, and split leakage.

    An absent annotation is not an empty/background training target. Explicit
    reviewed-empty annotations are permitted when annotations_complete is true.
    Internal mouse identity takes precedence over organ or block identity.
    """
    if not rows:
        raise ValueError("no training images")
    seen: set[str] = set()
    group_splits: dict[str, str] = {}
    counts = {"train": 0, "validation": 0}
    for row in rows:
        image_id = str(row["image_id"])
        if image_id in seen:
            raise ValueError("duplicate training image identity")
        seen.add(image_id)
        split = row["split"]
        if split not in counts:
            raise ValueError("training inputs must be train or validation")
        if row.get("is_independent_evaluation"):
            raise ValueError("independent evaluation annotations cannot enter training")
        if row.get("review_status") not in {"reviewed", "adjudicated"}:
            raise ValueError("training annotations await review")
        if not row.get("annotations_complete") or not row.get("annotation_path"):
            raise ValueError("missing annotation must not become a background target")
        if row.get("protected") or row.get("role") in {
            "test",
            "protected",
            "unassigned",
        }:
            raise ValueError("ineligible training subject")
        source = str(row["source"])
        identity = (
            row.get("subject_id")
            if source == "internal"
            else row.get("specimen_group", row.get("block_id"))
        )
        if not identity:
            raise ValueError("verified mouse/block grouping required")
        group = f"{source}:{identity}"
        if group in group_splits and group_splits[group] != split:
            raise ValueError("one mouse/block appears in multiple training splits")
        group_splits[group] = split
        counts[split] += 1
    if not all(counts.values()):
        raise ValueError("both training and validation groups required")
    return counts
