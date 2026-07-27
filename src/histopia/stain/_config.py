"""Configuration for quantitative brightfield stain workflows."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from histopia._validation import (
    finite_float,
    nonnegative_int,
    positive_float,
    positive_int,
    require_bool,
)
from histopia.stain._assays import StainFamily

STAIN_METHODS = ("legacy", "fixed", "macenko", "nmf")


@dataclass(slots=True)
class StainQuantificationConfig:
    """Inputs, scientific controls, and bounded runtime settings."""

    registration_run: Path
    output_dir: Path
    assay_manifest: Path | None = None
    analysis_mpp: float = 4.0
    default_family: str | None = "h-dab"
    methods: tuple[str, ...] = STAIN_METHODS
    sample_pixels: int = 100_000
    white_sample_pixels: int = 50_000
    vector_shrinkage: float = 0.25
    correction_rank_guard: float = 0.98
    seed: int = 0
    workers: int = 1
    vips_threads: int | None = None
    require_registration_approval: bool = False

    def __post_init__(self) -> None:
        self.registration_run = Path(self.registration_run)
        self.output_dir = Path(self.output_dir)
        if self.assay_manifest is not None:
            self.assay_manifest = Path(self.assay_manifest)
        self.analysis_mpp = positive_float("analysis_mpp", self.analysis_mpp)
        if self.default_family is not None:
            self.default_family = StainFamily(self.default_family.strip().lower()).value
        self.methods = tuple(
            dict.fromkeys(str(value).strip().lower() for value in self.methods)
        )
        if not self.methods or any(
            value not in STAIN_METHODS for value in self.methods
        ):
            raise ValueError(
                "methods must contain one or more of: " + ", ".join(STAIN_METHODS)
            )
        self.sample_pixels = positive_int("sample_pixels", self.sample_pixels)
        self.white_sample_pixels = positive_int(
            "white_sample_pixels", self.white_sample_pixels
        )
        self.vector_shrinkage = finite_float("vector_shrinkage", self.vector_shrinkage)
        if not 0 <= self.vector_shrinkage <= 1:
            raise ValueError("vector_shrinkage must be between zero and one")
        self.correction_rank_guard = finite_float(
            "correction_rank_guard", self.correction_rank_guard
        )
        if not 0 <= self.correction_rank_guard <= 1:
            raise ValueError("correction_rank_guard must be between zero and one")
        self.seed = nonnegative_int("seed", self.seed)
        self.workers = positive_int("workers", self.workers)
        if self.vips_threads is not None:
            self.vips_threads = positive_int("vips_threads", self.vips_threads)
        require_bool(
            "require_registration_approval", self.require_registration_approval
        )

    @property
    def resolved_default_family(self) -> StainFamily | None:
        return (
            StainFamily(self.default_family)
            if self.default_family is not None
            else None
        )


def load_stain_config(path: Path | str) -> StainQuantificationConfig:
    """Load JSON or TOML without importing WSI or numerical dependencies."""

    source = Path(path)
    if source.suffix.lower() == ".json":
        payload: dict[str, Any] = json.loads(source.read_text())
    elif source.suffix.lower() in {".toml", ".tml"}:
        if sys.version_info >= (3, 11):
            import tomllib
        else:
            import tomli as tomllib
        payload = tomllib.loads(source.read_text())
    else:
        raise ValueError("stain config must be JSON or TOML")
    values = dict(payload)
    values["methods"] = tuple(values.get("methods", STAIN_METHODS))
    return StainQuantificationConfig(**values)
