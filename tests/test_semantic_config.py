from __future__ import annotations

import json

import pytest

from histopia.semantic._config import SemanticAtlasConfig, load_semantic_config


def test_semantic_config_defaults_to_automatic_k_range(tmp_path) -> None:
    config = SemanticAtlasConfig(
        registration_run=tmp_path / "registration",
        output_dir=tmp_path / "semantic",
    )

    assert config.cluster_counts == tuple(range(5, 16))
    assert config.selected_clusters is None
    assert config.device == "auto"


def test_semantic_config_loads_legacy_explicit_cluster_counts(tmp_path) -> None:
    path = tmp_path / "legacy.json"
    path.write_text(
        json.dumps(
            {
                "registration_run": "registration",
                "output_dir": "semantic",
                "primary_clusters": 7,
                "sensitivity_clusters": [5, 10],
            }
        )
    )

    assert load_semantic_config(path).cluster_counts == (7, 5, 10)


def test_semantic_config_rejects_selected_k_outside_generated_counts(tmp_path) -> None:
    with pytest.raises(ValueError, match="selected_clusters must be generated"):
        SemanticAtlasConfig(
            registration_run=tmp_path / "registration",
            output_dir=tmp_path / "semantic",
            cluster_min=5,
            cluster_max=10,
            selected_clusters=11,
        )


def test_semantic_config_validates_and_normalizes_device(tmp_path) -> None:
    config = SemanticAtlasConfig(
        registration_run=tmp_path / "registration",
        output_dir=tmp_path / "semantic",
        device=" CUDA:1 ",
    )

    assert config.device == "cuda:1"
    with pytest.raises(ValueError, match="device must be"):
        SemanticAtlasConfig(
            registration_run=tmp_path / "registration",
            output_dir=tmp_path / "semantic",
            device="gpu",
        )
