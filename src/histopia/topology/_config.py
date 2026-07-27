"""Configuration for adaptive semantic topology reconstruction."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from histopia._validation import positive_float, positive_int, require_bool


@dataclass(slots=True)
class TopologyConfig:
    """Inputs and bounded scientific controls for one topology run."""

    registration_run: Path
    semantic_run: Path
    output_dir: Path
    section_thickness_um: float
    z_manifest: Path | None = None
    calibration_max_span: int = 4
    max_inferred_missing: int = 3
    require_approvals: bool = True

    def __post_init__(self) -> None:
        self.registration_run = Path(self.registration_run)
        self.semantic_run = Path(self.semantic_run)
        self.output_dir = Path(self.output_dir)
        if self.z_manifest is not None:
            self.z_manifest = Path(self.z_manifest)
        self.section_thickness_um = positive_float(
            "section_thickness_um", self.section_thickness_um
        )
        self.calibration_max_span = positive_int(
            "calibration_max_span", self.calibration_max_span
        )
        self.max_inferred_missing = positive_int(
            "max_inferred_missing", self.max_inferred_missing
        )
        require_bool("require_approvals", self.require_approvals)


def load_topology_config(path: Path | str) -> TopologyConfig:
    """Load a JSON or TOML topology configuration."""

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
        raise ValueError("topology config must be JSON or TOML")
    return TopologyConfig(**payload)
