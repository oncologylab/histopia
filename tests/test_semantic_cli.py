from __future__ import annotations

from pathlib import Path

import pytest

from histopia.semantic._cli import _build_parser
from histopia.semantic._config import (
    SemanticAtlasConfig,
    override_compute_config,
)


def test_extract_cli_accepts_compute_overrides() -> None:
    args = _build_parser().parse_args(
        [
            "extract",
            "--config",
            "atlas.toml",
            "--device",
            "cuda:1",
            "--batch-size",
            "128",
            "--patch-workers",
            "4",
            "--vips-threads",
            "8",
        ]
    )

    assert args.device == "cuda:1"
    assert args.batch_size == 128
    assert args.patch_workers == 4
    assert args.vips_threads == 8


def test_compute_overrides_are_validated_without_mutating_source(
    tmp_path: Path,
) -> None:
    source = SemanticAtlasConfig(
        registration_run=tmp_path / "registration",
        output_dir=tmp_path / "semantic",
    )

    overridden = override_compute_config(
        source,
        device=" CPU ",
        batch_size=8,
        patch_workers=2,
        vips_threads=3,
    )

    assert source.device == "auto"
    assert source.batch_size == 64
    assert overridden.device == "cpu"
    assert overridden.batch_size == 8
    assert overridden.patch_workers == 2
    assert overridden.vips_threads == 3


def test_cli_rejects_nonpositive_compute_override() -> None:
    with pytest.raises(SystemExit):
        _build_parser().parse_args(
            ["run", "--config", "atlas.toml", "--batch-size", "0"]
        )
