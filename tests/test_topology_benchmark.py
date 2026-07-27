from __future__ import annotations

import numpy as np

from histopia.topology._benchmark import run_holdout_benchmark
from histopia.topology._model import ObservedSection, PairEvidence


def test_holdout_benchmark_scores_hidden_semantic_section() -> None:
    sections = tuple(_section(index) for index in range(5))
    evidence = tuple(_evidence(index) for index in range(4))

    benchmark = run_holdout_benchmark(
        sections,
        evidence,
        origin_um_xy=(0, 0),
        spacing_um=1,
        max_hidden_sections=2,
    )

    summary = benchmark["summary"]
    assert summary["case_count"] == 5
    assert 0 <= summary["flow_macro_class_dice"] <= 1
    assert 0 <= summary["gap_interval_accuracy"] <= 1
    assert {row["hidden_sections"] for row in benchmark["cases"]} == {1, 2}


def _section(offset: int) -> ObservedSection:
    labels = np.full((12, 14), -1, dtype=np.int16)
    labels[3:9, 2 + offset : 7 + offset] = 0
    labels[5:8, 4 + offset : 6 + offset] = 1
    membership = np.stack((labels == 0, labels == 1)).astype(np.float32)
    return ObservedSection(
        slide_id=f"section-{offset}",
        labels=labels,
        membership=membership,
        support=labels >= 0,
        tissue_fraction=(labels >= 0).astype(np.float32),
        sparse_labels=labels[labels >= 0],
    )


def _evidence(index: int) -> PairEvidence:
    return PairEvidence(
        source_section=index,
        target_section=index + 1,
        score=1.0,
        support_dice=0.85,
        semantic_js=0.02,
        matched_label_agreement=0.9,
        correspondence_coverage=0.5,
        median_confidence=0.8,
        displacement_patch_widths=1.0,
        displacement_strain=0.1,
    )
