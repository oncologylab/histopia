"""Dependency-light quality summaries for semantic atlas cohorts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


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
    batch_correction_accepted: bool
    unsupported_sections: tuple[int, ...]
    review_approved: bool


def summarize_semantic_run(run_dir: Path | str) -> SemanticRunQc:
    """Summarize labels, links, batch diagnostics, and review state."""

    root = Path(run_dir)
    payload = json.loads((root / "semantic_result.json").read_text())
    selected_k = int(
        payload["selected_k"]
        if payload.get("selected_k") is not None
        else payload["primary_clusters"]
    )
    patch_count = 0
    tissue: list[np.ndarray] = []
    for slide in payload["slides"]:
        label_path = root / slide["labels"][str(selected_k)]
        with np.load(label_path, allow_pickle=False) as data:
            patch_count += len(data["labels"])
            tissue.append(np.asarray(data["tissue_fraction"], dtype=float))
    confidences: list[np.ndarray] = []
    accepted_links = 0
    for pair in payload.get("topology_pairs", []):
        accepted_links += int(pair["accepted_links"])
        with np.load(root / pair["artifact"], allow_pickle=False) as data:
            confidences.append(np.asarray(data["confidence"], dtype=float))
    batch = payload.get("batch_correction") or {}
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
        median_topology_confidence=_median(confidences),
        batch_correction_accepted=bool(batch.get("accepted")),
        unsupported_sections=tuple(
            int(x) for x in batch.get("unsupported_sections", ())
        ),
        review_approved=approved,
    )


def write_cohort_qc(runs: Mapping[str, Path | str], output_path: Path | str) -> Path:
    """Write deterministic JSON and TSV summaries for multiple mice."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"mouse_id": mouse_id, **asdict(summarize_semantic_run(runs[mouse_id]))}
        for mouse_id in sorted(runs)
    ]
    output.write_text(json.dumps({"schema_version": 1, "mice": rows}, indent=2) + "\n")
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
