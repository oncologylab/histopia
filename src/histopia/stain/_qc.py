"""Quantitative QC summaries for stain runs and cohorts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from histopia._atomic import write_json_atomic, write_text_atomic
from histopia.stain._result_validation import validate_stain_result


@dataclass(frozen=True, slots=True)
class StainRunQc:
    """Comparable metrics for one sealed stain result."""

    fingerprint: str
    slide_count: int
    quantified_slides: int
    context_slides: int
    correction_acceptance_fraction: float
    threshold_acceptance_fraction: float
    median_rank_correlation: float
    median_raw_glass_leakage: float
    median_corrected_glass_leakage: float
    median_background_cv_before: float
    median_background_cv_after: float
    median_reconstruction_residual: float
    review_approved: bool
    flags: tuple[str, ...]


def summarize_stain_run(run_dir: Path | str) -> StainRunQc:
    """Validate and summarize correction, threshold, and residual diagnostics."""

    root = Path(run_dir)
    payload = validate_stain_result(root)
    slides = payload.get("slides")
    if not isinstance(slides, list) or not slides:
        raise ValueError("stain result contains no slides")
    quantified = [row for row in slides if row.get("quantified")]
    qcs = [row.get("qc") for row in quantified]
    if any(not isinstance(qc, dict) for qc in qcs):
        raise ValueError("quantified stain slides must contain QC records")
    flags = sorted(
        {
            str(flag)
            for qc in qcs
            for flag in qc.get("flags", [])
            if isinstance(flag, str)
        }
    )
    if (
        quantified
        and sum(bool(qc["correction_accepted"]) for qc in qcs) / len(quantified) < 0.90
    ):
        flags.append("low_correction_acceptance")
    if (
        quantified
        and sum(bool(qc["threshold_accepted"]) for qc in qcs) / len(quantified) < 0.75
    ):
        flags.append("low_threshold_acceptance")
    review = _load_review(root, str(payload["fingerprint"]))
    return StainRunQc(
        fingerprint=str(payload["fingerprint"]),
        slide_count=len(slides),
        quantified_slides=len(quantified),
        context_slides=len(slides) - len(quantified),
        correction_acceptance_fraction=_fraction(qcs, "correction_accepted"),
        threshold_acceptance_fraction=_fraction(qcs, "threshold_accepted"),
        median_rank_correlation=_median(qcs, "rank_correlation"),
        median_raw_glass_leakage=_median(qcs, "raw_glass_leakage"),
        median_corrected_glass_leakage=_median(qcs, "corrected_glass_leakage"),
        median_background_cv_before=_median(qcs, "background_spatial_cv_before"),
        median_background_cv_after=_median(qcs, "background_spatial_cv_after"),
        median_reconstruction_residual=_median(qcs, "median_reconstruction_residual"),
        review_approved=review,
        flags=tuple(sorted(set(flags))),
    )


def write_stain_cohort_qc(
    runs: Mapping[str, Path | str],
    output_path: Path | str,
) -> Path:
    """Write deterministic JSON and TSV summaries for several cohorts."""

    output = Path(output_path)
    rows = [
        {"mouse_id": mouse_id, **asdict(summarize_stain_run(runs[mouse_id]))}
        for mouse_id in sorted(runs)
    ]
    write_json_atomic(
        output,
        {
            "schema_version": 1,
            "measurement": "relative_chromogen_optical_density",
            "mice": rows,
        },
    )
    columns = tuple(rows[0]) if rows else ("mouse_id",)
    lines = ["\t".join(columns)]
    lines.extend("\t".join(str(row[column]) for column in columns) for row in rows)
    write_text_atomic(output.with_suffix(".tsv"), "\n".join(lines) + "\n")
    return output


def _median(rows: list[dict[str, object]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return float(np.median(values)) if values else 0.0


def _fraction(rows: list[dict[str, object]], key: str) -> float:
    return float(sum(bool(row[key]) for row in rows) / len(rows)) if rows else 0.0


def _load_review(root: Path, fingerprint: str) -> bool:
    try:
        review = json.loads((root / "stain_review.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    return bool(review.get("approved")) and review.get("fingerprint") == fingerprint
