from __future__ import annotations

import json
import os
import sys

import pytest

from histopia.semantic._config import SemanticAtlasConfig, load_semantic_config
from histopia.semantic._vips import configure_vips_threads


def test_semantic_config_defaults_to_automatic_k_range(tmp_path) -> None:
    config = SemanticAtlasConfig(
        registration_run=tmp_path / "registration",
        output_dir=tmp_path / "semantic",
    )

    assert config.cluster_counts == tuple(range(5, 16))
    assert config.selected_clusters is None
    assert config.device == "auto"
    assert config.patch_workers == 1
    assert config.vips_threads is None
    assert config.fit_threads == 4


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


@pytest.mark.parametrize(
    "invalid_device",
    ("gpu", "cuda:", "cuda:-1", "cuda:foo", "cuda:1:2", "cuda:١"),
)
def test_semantic_config_validates_and_normalizes_device(
    tmp_path, invalid_device
) -> None:
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
            device=invalid_device,
        )


def test_semantic_config_rejects_nonpositive_patch_workers(tmp_path) -> None:
    with pytest.raises(ValueError, match="patch_workers must be positive"):
        SemanticAtlasConfig(
            registration_run=tmp_path / "registration",
            output_dir=tmp_path / "semantic",
            patch_workers=0,
        )


def test_semantic_config_rejects_nonpositive_vips_threads(tmp_path) -> None:
    with pytest.raises(ValueError, match="vips_threads must be positive"):
        SemanticAtlasConfig(
            registration_run=tmp_path / "registration",
            output_dir=tmp_path / "semantic",
            vips_threads=0,
        )


def test_semantic_config_rejects_nonpositive_fit_threads(tmp_path) -> None:
    with pytest.raises(ValueError, match="fit_threads must be positive"):
        SemanticAtlasConfig(
            registration_run=tmp_path / "registration",
            output_dir=tmp_path / "semantic",
            fit_threads=0,
        )


def test_vips_thread_cap_is_set_before_import(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "pyvips", raising=False)
    monkeypatch.delenv("VIPS_CONCURRENCY", raising=False)

    configure_vips_threads(6)

    assert os.environ["VIPS_CONCURRENCY"] == "6"


def test_vips_thread_cap_cannot_change_after_import(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "pyvips", object())
    monkeypatch.setenv("VIPS_CONCURRENCY", "4")

    with pytest.raises(RuntimeError, match="cannot change after pyvips is imported"):
        configure_vips_threads(8)


@pytest.mark.parametrize(
    ("field", "value", "error", "message"),
    (
        ("analysis_mpp", float("nan"), ValueError, "finite"),
        ("analysis_mpp", 0, ValueError, "positive"),
        ("patch_size_px", True, TypeError, "integer"),
        ("batch_size", 1.5, TypeError, "integer"),
        ("pca_components", 0, ValueError, "positive"),
        ("balanced_patch_cap", -1, ValueError, "positive"),
        ("max_cross_section_distance_um", float("inf"), ValueError, "finite"),
        ("seed", -1, ValueError, "non-negative"),
        ("seed", 2**32, ValueError, "must not exceed"),
        ("device", 1, TypeError, "string"),
    ),
)
def test_semantic_config_rejects_invalid_scientific_controls(
    tmp_path,
    field,
    value,
    error,
    message,
) -> None:
    with pytest.raises(error, match=message):
        SemanticAtlasConfig(
            registration_run=tmp_path / "registration",
            output_dir=tmp_path / "semantic",
            **{field: value},
        )


def test_semantic_config_does_not_truncate_cluster_counts(tmp_path) -> None:
    path = tmp_path / "invalid.json"
    path.write_text(
        json.dumps(
            {
                "registration_run": "registration",
                "output_dir": "semantic",
                "sensitivity_clusters": [5.5],
            }
        )
    )

    with pytest.raises(TypeError, match="must be an integer"):
        load_semantic_config(path)
