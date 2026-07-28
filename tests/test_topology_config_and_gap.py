from __future__ import annotations

import json
from pathlib import Path

import pytest

from histopia.topology import TopologyConfig
from histopia.topology._config import load_topology_config


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
