"""Dependency-light quality summaries for semantic atlas cohorts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from histopia.semantic._result import validate_semantic_result

COHORT_QC_THRESHOLDS = {
    "minimum_cluster_fraction": 0.005,
    "minimum_topology_confidence": 0.50,
    "minimum_topology_coverage": 0.05,
    "cohort_modified_z": 3.5,
}


@dataclass(frozen=True, slots=True)
class SemanticRunQc:
    """Comparable measurements derived from one portable semantic result."""

    fingerprint: str
    selected_k: int
    slide_count: int
    patch_count: int
    median_tissue_fraction: float
    accepted_topology_links: int
    median_topology_confidence: float
    topology_coverage: float
    zero_link_pairs: int
    selected_k_stability: float
    selected_k_score: float
    minimum_cluster_fraction: float
    batch_correction_accepted: bool
    raw_slide_variance_fraction: float
    corrected_slide_variance_fraction: float
    raw_slide_prediction_accuracy: float
    corrected_slide_prediction_accuracy: float
    unsupported_sections: tuple[int, ...]
    review_approved: bool
    flags: tuple[str, ...]


def summarize_semantic_run(run_dir: Path | str) -> SemanticRunQc:
    """Summarize labels, links, batch diagnostics, and review state."""

    root = Path(run_dir)
    payload = validate_semantic_result(root)
    _validate_provenance(payload)
    selected_k = int(
        payload["selected_k"]
        if payload.get("selected_k") is not None
        else payload["primary_clusters"]
    )
    cluster_counts = tuple(int(value) for value in payload["cluster_counts"])
    if (
        selected_k not in cluster_counts
        or len(set(cluster_counts)) != len(cluster_counts)
        or any(value < 2 for value in cluster_counts)
    ):
        raise ValueError("semantic cluster counts or selected K are invalid")
    slides = payload["slides"]
    slide_ids = [str(slide.get("id", "")) for slide in slides]
    if not slide_ids or any(not value for value in slide_ids):
        raise ValueError("semantic result must contain named slides")
    if len(set(slide_ids)) != len(slide_ids):
        raise ValueError("semantic result contains duplicate slides")
    if payload["feature_provenance"]["expected_slide_ids"] != slide_ids:
        raise ValueError("semantic slides differ from preflight slide order")
    patch_count = 0
    patch_counts: list[int] = []
    selected_labels: list[np.ndarray] = []
    tissue: list[np.ndarray] = []
    for slide in slides:
        labels_by_k = slide.get("labels")
        if not isinstance(labels_by_k, dict) or set(labels_by_k) != {
            str(value) for value in cluster_counts
        }:
            raise ValueError("semantic slide labels are incomplete for fitted K values")
        label_path = root / slide["labels"][str(selected_k)]
        with np.load(label_path, allow_pickle=False) as data:
            required = {"labels", "tissue_fraction", "grid_rc", "reference_um_xy"}
            if not required.issubset(data.files):
                raise ValueError("semantic label artifact is missing required arrays")
            labels = np.asarray(data["labels"], dtype=np.int64)
            tissue_fraction = np.asarray(data["tissue_fraction"], dtype=float)
            grid = np.asarray(data["grid_rc"])
            coordinates = np.asarray(data["reference_um_xy"], dtype=float)
            if (
                labels.ndim != 1
                or not len(labels)
                or np.any(labels < 0)
                or np.any(labels >= selected_k)
            ):
                raise ValueError("semantic selected-K labels are invalid")
            if (
                tissue_fraction.shape != labels.shape
                or not np.all(np.isfinite(tissue_fraction))
                or np.any(tissue_fraction < 0)
                or np.any(tissue_fraction > 1)
            ):
                raise ValueError("semantic tissue fractions are invalid")
            if (
                grid.shape != (len(labels), 2)
                or len(np.unique(grid, axis=0)) != len(grid)
                or coordinates.shape != (len(labels), 2)
                or not np.all(np.isfinite(coordinates))
            ):
                raise ValueError("semantic patch coordinates are invalid")
            patch_count += len(labels)
            patch_counts.append(len(labels))
            selected_labels.append(labels)
            tissue.append(tissue_fraction)
    confidences: list[np.ndarray] = []
    accepted_links = 0
    topology_capacity = 0
    zero_link_pairs = 0
    topology_pairs = payload.get("topology_pairs", [])
    if len(topology_pairs) != max(0, len(slides) - 1):
        raise ValueError("semantic result must contain every adjacent topology pair")
    for pair_index, pair in enumerate(topology_pairs):
        pair_links = int(pair["accepted_links"])
        if (
            int(pair["source_section"]) != pair_index
            or int(pair["target_section"]) != pair_index + 1
            or pair_links < 0
        ):
            raise ValueError("semantic topology pairs must follow adjacent slide order")
        accepted_links += pair_links
        zero_link_pairs += pair_links == 0
        source = int(pair["source_section"])
        target = int(pair["target_section"])
        topology_capacity += min(patch_counts[source], patch_counts[target])
        with np.load(root / pair["artifact"], allow_pickle=False) as data:
            required = {
                "confidence",
                "source_indices",
                "target_indices",
                "source_um_xy",
                "target_um_xy",
            }
            if not required.issubset(data.files):
                raise ValueError(
                    "semantic topology artifact is missing required arrays"
                )
            confidence = np.asarray(data["confidence"], dtype=float)
            source_indices = np.asarray(data["source_indices"], dtype=np.int64)
            target_indices = np.asarray(data["target_indices"], dtype=np.int64)
            source_xy = np.asarray(data["source_um_xy"], dtype=float)
            target_xy = np.asarray(data["target_um_xy"], dtype=float)
            if (
                confidence.shape != (pair_links,)
                or source_indices.shape != (pair_links,)
                or target_indices.shape != (pair_links,)
                or source_xy.shape != (pair_links, 2)
                or target_xy.shape != (pair_links, 2)
                or not np.all(np.isfinite(confidence))
                or np.any(confidence < 0)
                or np.any(confidence > 1)
                or not np.all(np.isfinite(source_xy))
                or not np.all(np.isfinite(target_xy))
                or np.any(source_indices < 0)
                or np.any(source_indices >= patch_counts[source])
                or len(np.unique(source_indices)) != pair_links
                or np.any(target_indices < 0)
                or np.any(target_indices >= patch_counts[target])
                or len(np.unique(target_indices)) != pair_links
            ):
                raise ValueError("semantic topology artifact is inconsistent")
            confidences.append(confidence)
    batch = payload.get("batch_correction") or {}
    raw_batch = batch.get("raw") or {}
    corrected_batch = batch.get("corrected") or {}
    unsupported_sections = tuple(
        int(value) for value in batch.get("unsupported_sections", ())
    )
    if any(value < 0 or value >= len(slides) for value in unsupported_sections):
        raise ValueError("semantic batch diagnostics reference invalid sections")
    selected_evaluation = next(
        (
            row
            for row in payload.get("k_selection") or ()
            if int(row["k"]) == selected_k
        ),
        {},
    )
    if not selected_evaluation:
        raise ValueError("semantic K-selection metrics are missing for selected K")
    counts = np.bincount(np.concatenate(selected_labels), minlength=selected_k)
    minimum_cluster_fraction = float(np.min(counts) / patch_count)
    topology_coverage = (
        float(accepted_links / topology_capacity) if topology_capacity else 0.0
    )
    median_confidence = _median(confidences)
    flags: list[str] = []
    if zero_link_pairs:
        flags.append("zero_link_pairs")
    if median_confidence < COHORT_QC_THRESHOLDS["minimum_topology_confidence"]:
        flags.append("low_topology_confidence")
    if topology_coverage < COHORT_QC_THRESHOLDS["minimum_topology_coverage"]:
        flags.append("low_topology_coverage")
    if minimum_cluster_fraction < COHORT_QC_THRESHOLDS["minimum_cluster_fraction"]:
        flags.append("small_selected_cluster")
    if not bool(batch.get("accepted")):
        flags.append("batch_correction_rejected")
    if unsupported_sections:
        flags.append("unsupported_sections")
    review = json.loads((root / "semantic_review.json").read_text())
    approved = bool(review.get("approved")) and (
        review.get("fingerprint") == payload.get("fingerprint")
    )
    return SemanticRunQc(
        fingerprint=str(payload["fingerprint"]),
        selected_k=selected_k,
        slide_count=len(payload["slides"]),
        patch_count=patch_count,
        median_tissue_fraction=_median(tissue),
        accepted_topology_links=accepted_links,
        median_topology_confidence=median_confidence,
        topology_coverage=topology_coverage,
        zero_link_pairs=zero_link_pairs,
        selected_k_stability=float(selected_evaluation.get("stability_ari", 0.0)),
        selected_k_score=float(selected_evaluation.get("composite_score", 0.0)),
        minimum_cluster_fraction=minimum_cluster_fraction,
        batch_correction_accepted=bool(batch.get("accepted")),
        raw_slide_variance_fraction=float(
            raw_batch.get("slide_variance_fraction", 0.0)
        ),
        corrected_slide_variance_fraction=float(
            corrected_batch.get("slide_variance_fraction", 0.0)
        ),
        raw_slide_prediction_accuracy=float(
            raw_batch.get("slide_prediction_accuracy", 0.0)
        ),
        corrected_slide_prediction_accuracy=float(
            corrected_batch.get("slide_prediction_accuracy", 0.0)
        ),
        unsupported_sections=unsupported_sections,
        review_approved=approved,
        flags=tuple(flags),
    )


def write_cohort_qc(runs: Mapping[str, Path | str], output_path: Path | str) -> Path:
    """Write deterministic JSON and TSV summaries for multiple mice."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ordered = tuple(sorted(runs))
    summaries = [summarize_semantic_run(runs[mouse_id]) for mouse_id in ordered]
    outlier_flags = _cohort_outlier_flags(summaries)
    rows = []
    for mouse_id, summary, cohort_flags in zip(
        ordered, summaries, outlier_flags, strict=True
    ):
        row = {"mouse_id": mouse_id, **asdict(summary)}
        row["flags"] = sorted(set(row["flags"]) | cohort_flags)
        rows.append(row)
    output.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "thresholds": COHORT_QC_THRESHOLDS,
                "mice": rows,
            },
            indent=2,
        )
        + "\n"
    )
    columns = tuple(rows[0]) if rows else ("mouse_id",)
    tsv = output.with_suffix(".tsv")
    tsv.write_text(
        "\t".join(columns)
        + "\n"
        + "".join(
            "\t".join(str(row[column]) for column in columns) + "\n" for row in rows
        )
    )
    return output


def _median(arrays: list[np.ndarray]) -> float:
    values = np.concatenate(arrays) if arrays else np.empty(0)
    return float(np.median(values)) if len(values) else 0.0


def _validate_provenance(payload: dict[str, object]) -> None:
    provenance = payload.get("feature_provenance")
    required = {
        "preflight_fingerprint",
        "expected_slide_ids",
        "model_fingerprint",
        "analysis_mpp",
        "patch_size_px",
        "min_tissue_fraction",
    }
    if not isinstance(provenance, dict) or not required.issubset(provenance):
        raise ValueError("schema-3 semantic feature provenance is incomplete")
    execution = {
        "batch_size",
        "encoder_runtime",
        "extraction_method",
        "patch_reader",
    }
    if execution.intersection(provenance) and not execution.issubset(provenance):
        raise ValueError("schema-3 semantic execution provenance is incomplete")


def _cohort_outlier_flags(
    summaries: list[SemanticRunQc],
) -> list[set[str]]:
    flags = [set() for _ in summaries]
    metrics = (
        "patch_count",
        "median_tissue_fraction",
        "topology_coverage",
        "median_topology_confidence",
        "selected_k_stability",
        "selected_k_score",
        "minimum_cluster_fraction",
        "corrected_slide_variance_fraction",
        "corrected_slide_prediction_accuracy",
    )
    threshold = COHORT_QC_THRESHOLDS["cohort_modified_z"]
    for metric in metrics:
        values = np.asarray(
            [float(getattr(summary, metric)) for summary in summaries],
            dtype=float,
        )
        if len(values) < 3 or not np.all(np.isfinite(values)):
            continue
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        if mad <= np.finfo(float).eps:
            at_median = np.isclose(values, median, rtol=0.0, atol=np.finfo(float).eps)
            if np.count_nonzero(at_median) <= len(values) / 2:
                continue
            for index in np.flatnonzero(~at_median):
                direction = "high" if values[index] > median else "low"
                flags[int(index)].add(f"cohort_{direction}_{metric}")
            continue
        modified_z = 0.67448975 * (values - median) / mad
        for index in np.flatnonzero(np.abs(modified_z) > threshold):
            direction = "high" if modified_z[index] > 0 else "low"
            flags[int(index)].add(f"cohort_{direction}_{metric}")
    return flags
