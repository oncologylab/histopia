from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from histopia.topology import TopologyConfig
from histopia.topology._config import load_topology_config
from histopia.topology._model import ObservedSection, PairEvidence
from histopia.topology._pipeline import _failed_calibration_decisions


def test_topology_config_requires_explicit_physical_thickness(tmp_path: Path) -> None:
    path = tmp_path / "topology.toml"
    path.write_text(
        """
registration_run = "registration"
semantic_run = "semantic"
output_dir = "topology"
section_thickness_um = 5.0
"""
    )

    config = load_topology_config(path)

    assert config.section_thickness_um == 5.0
    assert config.max_inferred_missing == 3
    assert config.reconstruction_samples_per_interval == 8
    assert config.envelope_max_xy_dim_px == 384
    with pytest.raises(ValueError, match="positive"):
        TopologyConfig(
            registration_run=tmp_path,
            semantic_run=tmp_path,
            output_dir=tmp_path,
            section_thickness_um=0,
        )


def test_topology_config_loads_json(
    tmp_path: Path,
) -> None:
    path = tmp_path / "topology.json"
    path.write_text(
        json.dumps(
            {
                "registration_run": "registration",
                "semantic_run": "semantic",
                "output_dir": "topology",
                "section_thickness_um": 5,
            }
        )
    )

    assert load_topology_config(path).max_inferred_missing == 3


def test_failed_gap_calibration_preserves_robust_discontinuities() -> None:
    rows = [
        (0.10 + 0.005 * (index % 9), 0.94 + 0.01 * (index % 3), 0.72)
        for index in range(19)
    ]
    rows.extend(
        (
            (0.263, 0.77, 0.43),
            (0.241, 0.82, 0.42),
            (0.242, 0.83, 0.41),
            (0.241, 0.82, 0.42),
        )
    )
    evidence = tuple(
        PairEvidence(
            source_section=index,
            target_section=index + 1,
            score=score,
            support_dice=support_dice,
            semantic_js=0.05,
            matched_label_agreement=agreement,
            correspondence_coverage=0.7,
            median_confidence=0.7,
            displacement_patch_widths=0.3,
            displacement_strain=0.3,
        )
        for index, (score, support_dice, agreement) in enumerate(rows)
    )

    decisions = _failed_calibration_decisions(evidence)

    assert [row.status for row in decisions[:19]] == ["assumed"] * 19
    assert [row.status for row in decisions[19:]] == ["unresolved"] * 4
    assert "robust_morphology_score_outlier" in decisions[-1].reasons


def test_failed_gap_calibration_retains_nested_terminal_partial_sections() -> None:
    evidence = tuple(
        PairEvidence(
            source_section=index,
            target_section=index + 1,
            score=0.27 if index == 4 else 0.10,
            support_dice=0.65 if index == 4 else 0.95,
            semantic_js=0.05,
            matched_label_agreement=0.35 if index == 4 else 0.75,
            correspondence_coverage=0.70,
            median_confidence=0.70,
            displacement_patch_widths=0.3,
            displacement_strain=0.3,
        )
        for index in range(5)
    )
    full = np.ones((8, 8), dtype=bool)
    nested = np.zeros((8, 8), dtype=bool)
    nested[2:6, 2:6] = True
    sections = tuple(
        ObservedSection(
            slide_id=str(index),
            labels=np.zeros((8, 8), dtype=np.int16),
            membership=np.ones((1, 1), dtype=np.float32),
            support=nested if index == 5 else full,
            tissue_fraction=np.ones((8, 8), dtype=np.float32),
            sparse_labels=np.zeros(1, dtype=np.int16),
        )
        for index in range(6)
    )

    decisions = _failed_calibration_decisions(evidence, sections)

    assert decisions[-1].status == "assumed"
    assert "terminal_nested_partial_section_continuity" in decisions[-1].reasons

    disconnected = list(sections)
    left = np.zeros((8, 8), dtype=bool)
    left[:, :4] = True
    separated = np.zeros((8, 8), dtype=bool)
    separated[:, 4:] = True
    disconnected[-2] = ObservedSection(
        slide_id="left",
        labels=np.zeros((8, 8), dtype=np.int16),
        membership=np.ones((1, 1), dtype=np.float32),
        support=left,
        tissue_fraction=np.ones((8, 8), dtype=np.float32),
        sparse_labels=np.zeros(1, dtype=np.int16),
    )
    disconnected[-1] = ObservedSection(
        slide_id="separated",
        labels=np.zeros((8, 8), dtype=np.int16),
        membership=np.ones((1, 1), dtype=np.float32),
        support=separated,
        tissue_fraction=np.ones((8, 8), dtype=np.float32),
        sparse_labels=np.zeros(1, dtype=np.int16),
    )
    rejected = _failed_calibration_decisions(evidence, tuple(disconnected))
    assert rejected[-1].status == "unresolved"
