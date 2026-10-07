"""Dependency-light configuration for protein-expression prediction."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from histopia._validation import finite_float, nonnegative_int, positive_int

COMPARTMENTS = ("whole_cell", "inner_boundary")
MEASUREMENT_STATISTICS = ("mean", "q90")
OD_HARMONIZATION_METHODS = ("equal_section_quantile_v2", "none")
MODEL_CANDIDATES = (
    "morphospatial_knn",
    "linear",
    "tree",
    "hurdle_mlp",
    "graphsage",
    "gatv2",
    "cross_attention",
    "token_decoder",
    "multi_tower",
    "dual_bank_attention",
    "graph_transformer",
    "shared_multitask",
)
FEATURE_SCHEMAS = (
    "native-hdab-neutral-spatial-uni2h-v2",
    "native-hdab-neutral-cell-multiscale-v3",
)
RELATIONAL_LOSS_PROFILES = ("huber_v1", "tail_rank_v1")

_ALIASES = {
    "ecad": "ecad",
    "ecadherin": "ecad",
    "e-cad": "ecad",
    "e-cadherin": "ecad",
    "cdh1": "ecad",
    "ncad": "ncad",
    "n-cad": "ncad",
    "ncadherin": "ncad",
    "n-cadherin": "ncad",
    "cdh2": "ncad",
    "perk": "perk",
    "p-erk": "perk",
    "phosphoerk": "perk",
    "phospho-erk": "perk",
}


def normalize_target_id(value: str) -> str:
    """Normalize spelling aliases without erasing assay-domain provenance."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError("protein target must be non-empty text")
    compact = re.sub(r"[^a-z0-9-]+", "", value.strip().lower())
    normalized = _ALIASES.get(compact, compact.replace("-", ""))
    if not normalized or not re.fullmatch(r"[a-z][a-z0-9]*", normalized):
        raise ValueError(f"invalid normalized protein target: {value!r}")
    return normalized


@dataclass(frozen=True, slots=True)
class ProteinTarget:
    """One biological target with explicit measurement semantics."""

    target_id: str
    aliases: tuple[str, ...] = ()
    assay_domain: str = "default"
    compartment: str = "whole_cell"
    measurement_statistic: str = "mean"
    binary_enabled: bool = True
    display_name: str | None = None

    def __post_init__(self) -> None:
        raw_target = self.target_id.strip()
        normalized = normalize_target_id(raw_target)
        object.__setattr__(self, "target_id", normalized)
        display = raw_target if self.display_name is None else self.display_name.strip()
        if (
            not display
            or len(display) > 80
            or any(ord(character) < 32 for character in display)
        ):
            raise ValueError("protein display_name must be short printable text")
        object.__setattr__(self, "display_name", display)
        aliases = tuple(dict.fromkeys(str(value).strip() for value in self.aliases))
        if any(not value for value in aliases):
            raise ValueError("protein aliases must be non-empty")
        if any(normalize_target_id(value) != normalized for value in aliases):
            raise ValueError("protein aliases must normalize to target_id")
        object.__setattr__(self, "aliases", aliases)
        domain = self.assay_domain.strip()
        if not domain or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", domain):
            raise ValueError("assay_domain must be a portable identifier")
        object.__setattr__(self, "assay_domain", domain)
        if self.compartment not in COMPARTMENTS:
            raise ValueError("compartment must be one of: " + ", ".join(COMPARTMENTS))
        if self.measurement_statistic not in MEASUREMENT_STATISTICS:
            raise ValueError(
                "measurement_statistic must be one of: "
                + ", ".join(MEASUREMENT_STATISTICS)
            )
        if not isinstance(self.binary_enabled, bool):
            raise TypeError("binary_enabled must be a boolean")


@dataclass(slots=True)
class ProteinPredictionConfig:
    """Scientific controls for leakage-safe protein model fitting."""

    output_dir: Path
    target: ProteinTarget
    seed: int = 0
    pca_components: int = 128
    spatial_block_um: float = 1024.0
    exclusion_buffer_um: float = 112.0
    positive_area_fraction: float = 0.25
    negative_area_fraction: float = 0.05
    minimum_effective_pixels: float = 2.0
    minimum_coverage: float = 0.70
    hidden_units: int = 256
    hidden_layers: int = 4
    ensemble_seeds: int = 3
    model_candidates: tuple[str, ...] = (
        "morphospatial_knn",
        "linear",
        "tree",
        "hurdle_mlp",
    )
    coordinate_scale_um: float = 256.0
    section_spacing_um: float = 5.0
    calibration_enabled: bool = True
    calibration_bin_um: float = 64.0
    calibration_minimum_cells_per_bin: int = 4
    calibration_minimum_anchors: int = 128
    calibration_scale_min: float = 0.5
    calibration_scale_max: float = 2.5
    semantic_blend_weight: float = 0.5
    relational_morphology_transfer_weight: float = 0.0
    relational_morphology_transfer_neighbors: int = 16
    relational_loss_profile: str = "huber_v1"
    od_harmonization: str = "equal_section_quantile_v2"
    feature_schema_id: str = "native-hdab-neutral-spatial-uni2h-v2"
    neighborhood_radii_um: tuple[float, ...] = (16.0, 32.0, 64.0, 128.0)

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        if isinstance(self.target, dict):
            self.target = ProteinTarget(**self.target)
        if not isinstance(self.target, ProteinTarget):
            raise TypeError("target must be a ProteinTarget")
        self.seed = nonnegative_int("seed", self.seed)
        self.pca_components = positive_int("pca_components", self.pca_components)
        self.hidden_units = positive_int("hidden_units", self.hidden_units)
        self.hidden_layers = positive_int("hidden_layers", self.hidden_layers)
        self.ensemble_seeds = positive_int("ensemble_seeds", self.ensemble_seeds)
        self.calibration_minimum_cells_per_bin = positive_int(
            "calibration_minimum_cells_per_bin",
            self.calibration_minimum_cells_per_bin,
        )
        self.calibration_minimum_anchors = positive_int(
            "calibration_minimum_anchors", self.calibration_minimum_anchors
        )
        self.relational_morphology_transfer_neighbors = positive_int(
            "relational_morphology_transfer_neighbors",
            self.relational_morphology_transfer_neighbors,
        )
        self.relational_loss_profile = self.relational_loss_profile.strip().lower()
        if self.relational_loss_profile not in RELATIONAL_LOSS_PROFILES:
            raise ValueError(
                "relational_loss_profile must be one of: "
                + ", ".join(RELATIONAL_LOSS_PROFILES)
            )
        if not isinstance(self.calibration_enabled, bool):
            raise TypeError("calibration_enabled must be a boolean")
        self.od_harmonization = self.od_harmonization.strip().lower()
        if self.od_harmonization not in OD_HARMONIZATION_METHODS:
            raise ValueError(
                "od_harmonization must be one of: "
                + ", ".join(OD_HARMONIZATION_METHODS)
            )
        self.feature_schema_id = self.feature_schema_id.strip()
        if self.feature_schema_id not in FEATURE_SCHEMAS:
            raise ValueError(
                "feature_schema_id must be one of: " + ", ".join(FEATURE_SCHEMAS)
            )
        self.neighborhood_radii_um = tuple(
            finite_float("neighborhood_radii_um", value)
            for value in self.neighborhood_radii_um
        )
        if (
            not self.neighborhood_radii_um
            or any(value <= 0 for value in self.neighborhood_radii_um)
            or tuple(sorted(set(self.neighborhood_radii_um)))
            != self.neighborhood_radii_um
        ):
            raise ValueError(
                "neighborhood_radii_um must be unique, positive, and increasing"
            )
        self.model_candidates = tuple(dict.fromkeys(self.model_candidates))
        invalid = set(self.model_candidates) - set(MODEL_CANDIDATES)
        if not self.model_candidates or invalid:
            raise ValueError(
                "model_candidates must be selected from: " + ", ".join(MODEL_CANDIDATES)
            )
        for name in (
            "spatial_block_um",
            "exclusion_buffer_um",
            "coordinate_scale_um",
            "section_spacing_um",
            "calibration_bin_um",
            "calibration_scale_min",
            "calibration_scale_max",
        ):
            value = finite_float(name, getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            setattr(self, name, value)
        self.semantic_blend_weight = finite_float(
            "semantic_blend_weight", self.semantic_blend_weight
        )
        if not 0 <= self.semantic_blend_weight <= 1:
            raise ValueError("semantic_blend_weight must be between zero and one")
        self.relational_morphology_transfer_weight = finite_float(
            "relational_morphology_transfer_weight",
            self.relational_morphology_transfer_weight,
        )
        if not 0 <= self.relational_morphology_transfer_weight <= 1:
            raise ValueError(
                "relational_morphology_transfer_weight must be between zero and one"
            )
        if self.calibration_scale_min > self.calibration_scale_max:
            raise ValueError(
                "calibration_scale_min must not exceed calibration_scale_max"
            )
        for name in (
            "positive_area_fraction",
            "negative_area_fraction",
            "minimum_coverage",
        ):
            value = finite_float(name, getattr(self, name))
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between zero and one")
            setattr(self, name, value)
        if self.negative_area_fraction >= self.positive_area_fraction:
            raise ValueError(
                "negative_area_fraction must be below positive_area_fraction"
            )
        self.minimum_effective_pixels = finite_float(
            "minimum_effective_pixels", self.minimum_effective_pixels
        )
        if self.minimum_effective_pixels <= 0:
            raise ValueError("minimum_effective_pixels must be positive")


def load_protein_config(path: Path | str) -> ProteinPredictionConfig:
    """Load JSON or TOML without importing numerical or GPU dependencies."""

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
        raise ValueError("protein config must be JSON or TOML")
    values = dict(payload)
    target = values.get("target")
    if not isinstance(target, dict):
        raise ValueError("protein config requires a [target] object")
    values["target"] = ProteinTarget(**target)
    return ProteinPredictionConfig(**values)
