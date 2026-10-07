"""Select a marker panel from accepted, explicitly typed measurements.

Distinct subjects, rather than images or repeated sections, determine support.
Marker identifiers must already distinguish antibody and phospho specificity.
This inventory gate does not establish predictive accuracy.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping


def supported_marker_panel(
    observations: Iterable[Mapping],
    *,
    minimum_subjects: int = 6,
    exclude: Iterable[str] = (),
) -> dict[str, list[str]]:
    """Return sorted marker-to-subject support for chromogenic IHC only.

    Missing modality, detection or acceptance evidence excludes an observation.
    Repeated scans or sections cannot increase the independent subject count.
    """
    if type(minimum_subjects) is not int or minimum_subjects < 2:
        raise ValueError("at least two distinct subjects are required")
    excluded = set(exclude)
    subjects: dict[str, set[str]] = defaultdict(set)
    for row in observations:
        if (
            row.get("modality") != "IHC"
            or row.get("detection") != "brightfield"
            or row.get("measurement_accepted") is not True
        ):
            continue
        marker, subject = row.get("marker_id"), row.get("subject_id")
        if not isinstance(marker, str) or not marker:
            raise ValueError("accepted observations require a marker identity")
        if not isinstance(subject, str) or not subject:
            raise ValueError("accepted observations require a subject identity")
        if marker not in excluded:
            subjects[marker].add(subject)
    return {
        marker: sorted(ids)
        for marker, ids in sorted(subjects.items())
        if len(ids) >= minimum_subjects
    }
