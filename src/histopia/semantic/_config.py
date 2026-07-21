"""Configuration for global serial-section semantic atlases."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class SemanticAtlasConfig:
    """Inputs and reproducibility controls for one semantic-atlas run."""

    registration_run: Path
    output_dir: Path
    model_cache_dir: Path | None = None
    analysis_mpp: float = 0.5
    patch_size_px: int = 224
    min_tissue_fraction: float = 0.5
    batch_size: int = 64
    primary_clusters: int = 7
    sensitivity_clusters: tuple[int, ...] = (5, 10)
    pca_components: int = 64
    balanced_patch_cap: int = 4096
    max_cross_section_distance_um: float = 112.0
    seed: int = 0
    device: str = "cuda"

    def __post_init__(self) -> None:
        self.registration_run = Path(self.registration_run)
        self.output_dir = Path(self.output_dir)
        if self.model_cache_dir is not None:
            self.model_cache_dir = Path(self.model_cache_dir)
        if self.analysis_mpp <= 0 or self.patch_size_px <= 0 or self.batch_size <= 0:
            raise ValueError(
                "analysis scale, patch size, and batch size must be positive"
            )
        if not 0 <= self.min_tissue_fraction <= 1:
            raise ValueError("min_tissue_fraction must be between 0 and 1")
        if self.primary_clusters <= 1 or any(x <= 1 for x in self.sensitivity_clusters):
            raise ValueError("cluster counts must be greater than one")

    @property
    def cluster_counts(self) -> tuple[int, ...]:
        return tuple(dict.fromkeys((self.primary_clusters, *self.sensitivity_clusters)))


def load_semantic_config(path: Path | str) -> SemanticAtlasConfig:
    """Load JSON or TOML configuration without importing heavy dependencies."""

    path = Path(path)
    if path.suffix.lower() == ".json":
        data: dict[str, Any] = json.loads(path.read_text())
    elif path.suffix.lower() in {".toml", ".tml"}:
        if sys.version_info >= (3, 11):
            import tomllib
        else:
            import tomli as tomllib
        data = tomllib.loads(path.read_text())
    else:
        raise ValueError("config must be JSON or TOML")
    sensitivity = data.pop("sensitivity_clusters", (5, 10))
    return SemanticAtlasConfig(
        **data,
        sensitivity_clusters=tuple(int(value) for value in sensitivity),
    )
