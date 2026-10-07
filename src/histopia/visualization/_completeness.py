"""Path-free completeness audit for configured review cohorts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from histopia._atomic import write_json_atomic
from histopia.visualization._review_api import ReviewRuns


@dataclass(frozen=True, slots=True)
class CohortCompleteness:
    """Coverage of registration, stain maps, and selected cell-boundary runs."""

    cohort_id: str
    expected_sections: int | None
    registration_sections: int
    registration_complete: bool
    registration_approved: bool
    stain_expected: int | None
    stain_quantified: int
    stain_missing_orders: tuple[int, ...]
    stain_configured: bool
    stain_fingerprint: str | None
    cell_expected: int
    cell_segmented: int
    cell_missing_sections: tuple[str, ...]
    cell_configured: bool
    cell_approved: bool
    cell_fingerprint: str | None

    @property
    def complete(self) -> bool:
        """Return whether every required coverage unit is present."""

        return (
            self.registration_complete
            and self.registration_approved
            and self.stain_expected is not None
            and self.stain_quantified == self.stain_expected
            and self.cell_segmented == self.cell_expected
            and self.cell_approved
        )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "id": self.cohort_id,
            "complete": self.complete,
            "registration": {
                "expected_sections": self.expected_sections,
                "completed_sections": self.registration_sections,
                "complete": self.registration_complete,
                "approved": self.registration_approved,
            },
            "stain": {
                "configured": self.stain_configured,
                "expected_non_reference_sections": self.stain_expected,
                "quantified_sections": self.stain_quantified,
                "missing_orders": list(self.stain_missing_orders),
                "fingerprint": self.stain_fingerprint,
            },
            "cells": {
                "configured": self.cell_configured,
                "expected_selected_sections": self.cell_expected,
                "segmented_selected_sections": self.cell_segmented,
                "missing_sections": list(self.cell_missing_sections),
                "approved": self.cell_approved,
                "fingerprint": self.cell_fingerprint,
            },
        }


@dataclass(frozen=True, slots=True)
class AnalysisCompleteness:
    """Path-free completeness report across an explicit cohort scope."""

    cohorts: tuple[CohortCompleteness, ...]

    @property
    def complete(self) -> bool:
        return all(row.complete for row in self.cohorts)

    def to_json_dict(self) -> dict[str, object]:
        known_expected = [
            row.expected_sections
            for row in self.cohorts
            if row.expected_sections is not None
        ]
        known_stain = [
            row.stain_expected for row in self.cohorts if row.stain_expected is not None
        ]
        return {
            "schema_version": 1,
            "complete": self.complete,
            "summary": {
                "cohort_count": len(self.cohorts),
                "complete_cohorts": sum(row.complete for row in self.cohorts),
                "expected_sections": sum(known_expected),
                "registered_sections": sum(
                    row.registration_sections for row in self.cohorts
                ),
                "approved_registration_cohorts": sum(
                    row.registration_approved for row in self.cohorts
                ),
                "expected_stain_sections": sum(known_stain),
                "quantified_stain_sections": sum(
                    row.stain_quantified for row in self.cohorts
                ),
                "expected_cell_sections": sum(
                    row.cell_expected for row in self.cohorts
                ),
                "segmented_cell_sections": sum(
                    row.cell_segmented for row in self.cohorts
                ),
                "approved_cell_cohorts": sum(row.cell_approved for row in self.cohorts),
            },
            "cohorts": [row.to_json_dict() for row in self.cohorts],
        }


def audit_analysis_completeness(
    runs: dict[str, ReviewRuns],
) -> AnalysisCompleteness:
    """Audit all configured cohorts without returning local filesystem paths."""

    if not runs:
        raise ValueError("completeness audit requires at least one cohort")
    return AnalysisCompleteness(
        tuple(_audit_cohort(name, run) for name, run in sorted(runs.items()))
    )


def write_analysis_completeness(
    report: AnalysisCompleteness,
    output: Path | str,
) -> Path:
    """Atomically write a path-free completeness report."""

    return write_json_atomic(output, report.to_json_dict(), sort_keys=True)


def _audit_cohort(cohort_id: str, runs: ReviewRuns) -> CohortCompleteness:
    registration = _load_object(runs.registration / "registration_result.json")
    registration_rows = _rows(registration, "slides")
    expected_sections = (
        len(registration_rows) if registration_rows else _mask_count(runs.registration)
    )
    registration_complete = bool(registration_rows) and (
        expected_sections == len(registration_rows)
    )
    registration_approved = _registration_is_approved(runs.registration)
    context_sections = tuple(
        f"{order:03d}"
        for order, row in enumerate(registration_rows, start=1)
        if _is_context_he(row)
    )
    reference_sections = tuple(
        f"{order:03d}"
        for order, row in enumerate(registration_rows, start=1)
        if row.get("is_reference") is True
    )
    if registration_rows and (
        len(context_sections) != 1 or len(reference_sections) != 1
    ):
        registration_complete = False

    stain = _load_object(runs.stain / "stain_result.json") if runs.stain else None
    stain_rows = _rows(stain, "slides")
    excluded_stain_orders = {
        int(row["order"])
        for row in stain_rows
        if row.get("analysis_included") is False and isinstance(row.get("order"), int)
    }
    stain_expected = (
        sum(
            not _is_context_he(row) and order not in excluded_stain_orders
            for order, row in enumerate(registration_rows, start=1)
        )
        if registration_rows
        else max(expected_sections - 1, 0)
        if expected_sections is not None
        else None
    )
    quantified_orders = {
        int(row["order"])
        for row in stain_rows
        if row.get("quantified") is True and isinstance(row.get("order"), int)
    }
    eligible_orders = (
        {
            order
            for order, row in enumerate(registration_rows, start=1)
            if not _is_context_he(row) and order not in excluded_stain_orders
        }
        if registration_rows
        else set(range(2, (expected_sections or 1) + 1))
    )

    cells = _load_object(runs.cells / "cell_result.json") if runs.cells else None
    cell_rows = _rows(cells, "slides")
    cell_sections = {
        str(row["section"]) for row in cell_rows if isinstance(row.get("section"), str)
    }
    cell_preflight = _load_object(runs.cells / "preflight.json") if runs.cells else None
    if cell_preflight is not None and cell_preflight.get("schema_version") == 2:
        expected_cells = tuple(
            str(row["section"])
            for row in _rows(cell_preflight, "slides")
            if isinstance(row.get("section"), str)
        )
    else:
        # Legacy partial results are never allowed to define their own scope.
        # Until a schema-v2 preflight seals inclusion/exclusion, every
        # registered section is expected, except sections explicitly excluded
        # from the cross-workflow analysis by the sealed stain result.
        expected_cells = tuple(
            f"{order:03d}"
            for order in range(1, len(registration_rows) + 1)
            if order not in excluded_stain_orders
        )
    completed_cells = set(expected_cells) & cell_sections
    cell_review = _load_object(runs.cells / "cell_review.json") if runs.cells else None
    cell_approved = bool(
        cells is not None
        and isinstance(cells.get("fingerprint"), str)
        and cell_review is not None
        and cell_review.get("approved") is True
        and cell_review.get("fingerprint") == cells.get("fingerprint")
    )

    return CohortCompleteness(
        cohort_id=cohort_id,
        expected_sections=expected_sections,
        registration_sections=len(registration_rows),
        registration_complete=registration_complete,
        registration_approved=registration_approved,
        stain_expected=stain_expected,
        stain_quantified=len(quantified_orders & eligible_orders),
        stain_missing_orders=tuple(sorted(eligible_orders - quantified_orders)),
        stain_configured=runs.stain is not None,
        stain_fingerprint=_text(stain, "fingerprint"),
        cell_expected=len(expected_cells),
        cell_segmented=len(completed_cells),
        cell_missing_sections=tuple(
            section for section in expected_cells if section not in completed_cells
        ),
        cell_configured=runs.cells is not None,
        cell_approved=cell_approved,
        cell_fingerprint=_text(cells, "fingerprint"),
    )


def _load_object(path: Path) -> dict[str, object] | None:
    try:
        payload = json.loads(path.read_text())
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _rows(payload: dict[str, object] | None, key: str) -> list[dict[str, object]]:
    if payload is None or not isinstance(payload.get(key), list):
        return []
    return [row for row in payload[key] if isinstance(row, dict)]  # type: ignore[index]


def _mask_count(root: Path) -> int | None:
    rows = _rows(_load_object(root / "mask_review.json"), "slides")
    return len(rows) if rows else None


def _text(payload: dict[str, object] | None, key: str) -> str | None:
    if payload is None:
        return None
    value = payload.get(key)
    return value if isinstance(value, str) else None


def _is_context_he(row: dict[str, object]) -> bool:
    value = row.get("path")
    if not isinstance(value, str) or not value:
        return False
    from histopia.stain._assays import StainFamily, infer_slide_assay

    return (
        infer_slide_assay(
            Path(value).name,
            default_family=StainFamily.H_DAB,
        ).family
        is StainFamily.CONTEXT_HE
    )


def _registration_is_approved(root: Path) -> bool:
    try:
        from histopia.registration._approval import validate_registration_approval

        validate_registration_approval(root)
    except (FileNotFoundError, OSError, ValueError):
        return False
    return True
