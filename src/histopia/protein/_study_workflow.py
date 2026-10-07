"""Multi-mouse, cell-resolved protein-transfer study workflow.

This workflow is deliberately downstream of validated registration, cell,
stain, and semantic artifacts.  It never recomputes stain quantification and
accepts only tissue-masked adaptive corrected target OD at 4 microns/px.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_ADAPTIVE_MEASUREMENT_VIEW = "tissue-masked-adaptive-corrected-target-od-4um-v1"
_MEASUREMENT_VIEW = "tissue-masked-harmonized-adaptive-corrected-target-od-4um-v2"
_BACKGROUND_MEASUREMENT_VIEW = "tissue-masked-counterstain-conditioned-target-od-4um-v3"
_SUPPORTED_MEASUREMENT_VIEWS = frozenset(
    {
        _ADAPTIVE_MEASUREMENT_VIEW,
        _MEASUREMENT_VIEW,
        _BACKGROUND_MEASUREMENT_VIEW,
    }
)
_ARCHITECTURES = frozenset(
    {
        "cross_attention",
        "hurdle_mlp",
        "extra_trees",
        "multi_tower",
        "dual_bank_attention",
        "graph_transformer",
        "shared_multitask",
    }
)
_PREDICTION_PROTOCOLS = frozenset({"leave-one-mouse-out", "training-visible"})
_MORPHOLOGY_TRANSFER_METHOD_V1 = (
    "calibrated-neural-plus-training-only-morphology-od-transfer-v1"
)
_MORPHOLOGY_TRANSFER_METHOD_V2 = "neural-plus-training-only-morphology-od-transfer-v2"
_MORPHOLOGY_TRANSFER_METHODS = frozenset(
    {_MORPHOLOGY_TRANSFER_METHOD_V1, _MORPHOLOGY_TRANSFER_METHOD_V2}
)


@dataclass(frozen=True, slots=True)
class RealProteinStudyCohort:
    """Exact upstream runs and reusable neutral cell features for one mouse."""

    registration_run: Path
    cell_run: Path
    stain_run: Path
    semantic_run: Path
    feature_dir: Path


def load_real_protein_study_manifest(
    path: Path | str,
) -> dict[str, RealProteinStudyCohort]:
    """Load an explicit local multi-mouse study registry.

    The registry is operational input: paths are resolved locally and are never
    copied into public review metadata.
    """

    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text())
    raw_cohorts = payload.get("cohorts") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or not isinstance(raw_cohorts, dict)
    ):
        raise ValueError("protein study registry must use schema version 1")
    output: dict[str, RealProteinStudyCohort] = {}
    required = (
        "registration_run",
        "cell_run",
        "stain_run",
        "semantic_run",
        "feature_dir",
    )
    for cohort, row in sorted(raw_cohorts.items()):
        if not isinstance(cohort, str) or not cohort:
            raise ValueError("protein study cohort IDs must be non-empty text")
        if not isinstance(row, dict) or set(row) != set(required):
            raise ValueError(f"protein study cohort is incomplete: {cohort}")
        paths: dict[str, Path] = {}
        for key in required:
            value = row[key]
            if not isinstance(value, str) or not value:
                raise ValueError(f"protein study cohort path is invalid: {cohort}")
            candidate = Path(value).expanduser()
            paths[key] = (
                candidate.resolve()
                if candidate.is_absolute()
                else (source.parent / candidate).resolve()
            )
        output[cohort] = RealProteinStudyCohort(**paths)
    if len(output) < 2:
        raise ValueError("protein study registry requires at least two cohorts")
    return output


@dataclass(slots=True)
class _Context:
    mouse_id: str
    source: RealProteinStudyCohort
    bindings: Any
    target_sections: tuple[str, ...]
    stain_by_slide: dict[str, dict[str, object]]
    semantic_by_slide: dict[str, dict[str, object]]
    cell_by_section: dict[str, dict[str, object]]
    preflight_by_slide: dict[str, dict[str, object]]
    registration_by_slide: dict[str, dict[str, object]]
    reference_slide: dict[str, object]
    region_count: int


@dataclass(slots=True)
class _Design:
    label_ids: np.ndarray
    native_xy: np.ndarray
    reference_um_xy: np.ndarray
    reference_um_xyz: np.ndarray
    features: np.ndarray
    supported: np.ndarray
    regions: np.ndarray


@dataclass(slots=True)
class _Candidate:
    architecture: str
    estimator: object
    source_knots: np.ndarray
    target_knots: np.ndarray
    reference_od: np.ndarray
    fingerprint: str
    morphology_transfer_weight: float = 0.0
    morphology_transfer_neighbors: int = 16

    def __post_init__(self) -> None:
        if not math.isfinite(self.morphology_transfer_weight) or not (
            0 <= self.morphology_transfer_weight <= 1
        ):
            raise ValueError("morphology-transfer weight must be between zero and one")
        if self.morphology_transfer_neighbors < 1:
            raise ValueError("morphology-transfer neighbors must be positive")
        if self.morphology_transfer_weight and self.architecture not in {
            "dual_bank_attention",
            "graph_transformer",
            "multi_tower",
        }:
            raise ValueError(
                "morphology transfer requires a compatible neural architecture"
            )
        if self.morphology_transfer_weight and not callable(
            getattr(self.estimator, "predict_morphology_transfer", None)
        ):
            raise ValueError("morphology-transfer outcome bank is unavailable")

    def predict(
        self,
        features: np.ndarray,
        *,
        reference_um_xyz: np.ndarray | None = None,
        device: str,
    ) -> tuple[np.ndarray | None, np.ndarray, np.ndarray, np.ndarray]:
        """Return calibrated probability, percentile, OD, and uncertainty."""

        if self.architecture == "hurdle_mlp":
            probability, _relative, raw, uncertainty = self.estimator.predict(features)
        elif self.architecture in {
            "cross_attention",
            "multi_tower",
            "shared_multitask",
        }:
            raw, uncertainty = self.estimator.predict_accelerated(
                features,
                device=device,
            )
            probability = None
        elif self.architecture in {
            "dual_bank_attention",
            "graph_transformer",
        }:
            if reference_um_xyz is None:
                raise ValueError(
                    "relational prediction requires registered coordinates"
                )
            raw, uncertainty = self.estimator.predict_accelerated(
                features,
                reference_um_xyz,
                device=device,
            )
            probability = None
        else:
            raw, uncertainty = _predict_extra_trees(self.estimator, features)
            probability = None
        od = _apply_quantile_calibration(raw, self.source_knots, self.target_knots)
        lower = _apply_quantile_calibration(
            np.maximum(np.asarray(raw) - uncertainty, 0),
            self.source_knots,
            self.target_knots,
        )
        upper = _apply_quantile_calibration(
            np.asarray(raw) + uncertainty,
            self.source_knots,
            self.target_knots,
        )
        calibrated_uncertainty = np.maximum(upper - lower, 0) / 2.0
        if self.morphology_transfer_weight:
            transfer, transfer_uncertainty = self.estimator.predict_morphology_transfer(
                features,
                neighbors=self.morphology_transfer_neighbors,
            )
            weight = self.morphology_transfer_weight
            base_od = np.asarray(od, dtype=np.float64)
            transferred = np.asarray(transfer, dtype=np.float64)
            od = (1.0 - weight) * base_od + weight * transferred
            calibrated_uncertainty = np.sqrt(
                np.square(1.0 - weight) * np.square(calibrated_uncertainty)
                + np.square(weight) * np.square(transfer_uncertainty)
                + weight * (1.0 - weight) * np.square(base_od - transferred)
            )
        reference = self.reference_od
        relative = np.searchsorted(reference, od, side="right") / len(reference)
        return (
            probability,
            relative.astype(np.float32),
            od.astype(np.float32),
            np.asarray(calibrated_uncertainty, dtype=np.float32),
        )


def prepare_real_protein_study_features(
    config: object,
    cohorts: dict[str, RealProteinStudyCohort],
    *,
    geometry_cache: Path | str,
    cohort_ids: tuple[str, ...] | None = None,
    sections: tuple[str, ...] | None = None,
) -> dict[str, object]:
    """Materialize reusable target-free v3 features for disjoint sections."""

    feature_view = str(
        getattr(
            config,
            "feature_schema_id",
            "native-hdab-neutral-spatial-uni2h-v2",
        )
    )
    if feature_view != "native-hdab-neutral-cell-multiscale-v3":
        raise ValueError("study feature preparation requires the multiscale v3 schema")
    selected_cohorts = tuple(cohort_ids or sorted(cohorts))
    unknown = set(selected_cohorts) - set(cohorts)
    if not selected_cohorts or unknown:
        raise ValueError(
            "study feature cohort selection is invalid: " + ", ".join(sorted(unknown))
        )
    selected_sections = None if sections is None else set(sections)
    if selected_sections is not None and (
        not selected_sections or any(not value.isdigit() for value in selected_sections)
    ):
        raise ValueError("study feature sections must be numeric identifiers")
    # Feature preparation consumes only sealed neutral token artifacts and,
    # when a derived geometry cache is absent, validates the exact cell label
    # immediately before reading it.  Avoid an eager all-artifact digest sweep,
    # which is both unnecessary and fragile on read-only network mounts.
    selected_sources = {cohort: cohorts[cohort] for cohort in selected_cohorts}
    contexts = _contexts(config, selected_sources, verify_artifacts=False)
    rows: list[dict[str, object]] = []
    for cohort in selected_cohorts:
        context = contexts[cohort]
        available = set(context.cell_by_section)
        if selected_sections is not None and not selected_sections <= available:
            missing = selected_sections - available
            raise ValueError(
                f"study feature sections are unavailable for {cohort}: "
                + ", ".join(sorted(missing))
            )
        for section in sorted(selected_sections or available):
            design = _section_design(
                context,
                section,
                config,
                geometry_root=Path(geometry_cache),
                include_semantic=False,
            )
            rows.append(
                {
                    "cohort": cohort,
                    "section": section,
                    "cells": len(design.label_ids),
                    "features": design.features.shape[1],
                }
            )
    return {
        "schema_version": 1,
        "feature_schema_id": feature_view,
        "sections": rows,
        "section_count": len(rows),
        "cell_count": sum(int(row["cells"]) for row in rows),
    }


def prepare_real_protein_study_table(
    config: object,
    cohorts: dict[str, RealProteinStudyCohort],
    output: Path | str,
    *,
    geometry_cache: Path | str,
    max_training_cells_per_section: int = 8_000,
    harmonization_reference_cohorts: tuple[str, ...] | None = None,
) -> Path:
    """Assemble a pooled, target-specific training table without stain leakage."""

    from histopia.protein._calibration import (
        fit_equal_section_od_harmonization,
    )
    from histopia.protein._real_data import (
        aggregate_map_to_sampled_labels,
        compartment_sampled_labels,
        sampled_label_expected_analysis_pixels,
        sampled_label_geometry,
        validated_adaptive_target_measurement,
    )
    from histopia.protein._result import CellExpressionTable
    from histopia.stain._artifacts import AdaptiveStainMap, StainMap

    if len(cohorts) < 2:
        raise ValueError("a protein transfer study requires at least two mice")
    if max_training_cells_per_section < 4:
        raise ValueError("training cells per section must be at least four")
    contexts = _contexts(config, cohorts)
    reference_cohorts: tuple[str, ...] | None = None
    if harmonization_reference_cohorts is not None:
        reference_cohorts = tuple(sorted(set(harmonization_reference_cohorts)))
        if not reference_cohorts or set(reference_cohorts) - set(contexts):
            raise ValueError("protein harmonization reference cohorts are invalid")
        if config.od_harmonization != "equal_section_quantile_v2":
            raise ValueError(
                "protein harmonization reference cohorts require OD harmonization"
            )
    geometry_root = Path(geometry_cache)
    parts: list[dict[str, np.ndarray]] = []
    measurement_rows: list[dict[str, object]] = []
    for mouse_id, context in contexts.items():
        for section in context.target_sections:
            cell_row = context.cell_by_section[section]
            design = _section_design(
                context,
                section,
                config,
                geometry_root=geometry_root,
            )
            sampled, _present, _xy, _counts = sampled_label_geometry(
                context.source.cell_run / str(cell_row["labels"]),
                subifd=2,
                cell_count=int(cell_row["cell_count"]),
            )
            stain_row = context.stain_by_slide[str(cell_row["slide"])]
            stain_map = StainMap.load(context.source.stain_run / str(stain_row["map"]))
            adaptive_map = (
                AdaptiveStainMap.load(
                    context.source.stain_run / str(stain_row["adaptive_map"])
                )
                if isinstance(stain_row.get("adaptive_map"), str)
                else None
            )
            measurement = validated_adaptive_target_measurement(
                stain_map,
                stain_row,
                adaptive_stain_map=adaptive_map,
            )
            sampled = compartment_sampled_labels(
                sampled,
                config.target.compartment,
                source_mpp_xy=stain_map.source_mpp_xy,
                pyramid_scale=4.0,
            )
            mean, effective, fraction = aggregate_map_to_sampled_labels(
                sampled,
                measurement.target_od,
                measurement.tissue_mask,
                measurement.positive_mask,
                source_mpp_xy=stain_map.source_mpp_xy,
                analysis_mpp=float(stain_map.analysis_mpp),
                content_origin_native_xy=stain_map.content_origin_native_xy,
                pyramid_scale=4.0,
                cell_count=int(np.max(design.label_ids, initial=0)),
                statistic=config.target.measurement_statistic,
            )
            expected = sampled_label_expected_analysis_pixels(
                sampled,
                cell_count=int(np.max(design.label_ids, initial=0)),
                source_mpp_xy=stain_map.source_mpp_xy,
                pyramid_scale=4.0,
                analysis_mpp=float(stain_map.analysis_mpp),
            )
            coverage = np.divide(
                effective,
                expected,
                out=np.zeros_like(effective),
                where=expected > 0,
            )
            ids = design.label_ids.astype(np.int64)
            eligible = np.flatnonzero(
                design.supported
                & (effective[ids] >= float(config.minimum_effective_pixels))
                & (coverage[ids] >= float(config.minimum_coverage))
            )
            if len(eligible) > max_training_cells_per_section:
                eligible = _stratified_training_sample(
                    eligible,
                    design.reference_um_xy,
                    mean[ids],
                    size=max_training_cells_per_section,
                    block_um=max(float(config.spatial_block_um) / 2.0, 64.0),
                    rng=np.random.default_rng(
                        _study_section_sampling_seed(
                            int(config.seed),
                            config.target.target_id,
                            mouse_id,
                            section,
                        )
                    ),
                )
            binary = np.full(len(eligible), -1, dtype=np.int8)
            qc = dict(stain_row.get("qc", {}))
            if config.target.binary_enabled and qc.get("threshold_accepted") is True:
                selected_fraction = fraction[ids[eligible]]
                binary[selected_fraction <= config.negative_area_fraction] = 0
                binary[selected_fraction >= config.positive_area_fraction] = 1
            parts.append(
                {
                    "label_ids": design.label_ids[eligible],
                    "section_ids": np.full(len(eligible), section),
                    "mouse_ids": np.full(len(eligible), mouse_id),
                    "native_xy": design.native_xy[eligible],
                    "reference_um_xy": design.reference_um_xy[eligible],
                    "features": design.features[eligible],
                    "measured_od": mean[ids[eligible]],
                    "binary_label": binary,
                    "measurement_coverage": np.clip(coverage[ids[eligible]], 0, 1),
                    "semantic_region": design.regions[eligible],
                    "semantic_support": design.supported[eligible],
                }
            )
            measurement_rows.append(
                {
                    "cohort": mouse_id,
                    "section": section,
                    "measurement_fingerprint": measurement.fingerprint,
                    "measurement_view": measurement.measurement_view,
                    "measurement_statistic": config.target.measurement_statistic,
                    "adaptive_floor_od": measurement.floor_od,
                    "positive_threshold_od": measurement.positive_threshold_od,
                }
            )
    if not parts:
        raise ValueError("protein study contains no accepted target measurements")
    combined = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    raw_measured = np.asarray(combined["measured_od"], dtype=np.float32)
    source_views = {str(row["measurement_view"]) for row in measurement_rows}
    if len(source_views) != 1:
        raise ValueError("protein study target measurement views differ")
    source_view = source_views.pop()
    if config.od_harmonization == "equal_section_quantile_v2":
        harmonization = fit_equal_section_od_harmonization(
            combined["mouse_ids"],
            combined["section_ids"],
            raw_measured,
            np.isfinite(raw_measured),
            reference_cohorts=reference_cohorts,
        )
        harmonized = np.full(raw_measured.shape, np.nan, dtype=np.float32)
        for section_calibration in harmonization.sections:
            selected = (combined["mouse_ids"] == section_calibration.cohort) & (
                combined["section_ids"] == section_calibration.section
            )
            harmonized[selected] = section_calibration.apply(raw_measured[selected])
        combined["measured_od"] = harmonized
        calibration_by_key = {
            (row.cohort, row.section): row for row in harmonization.sections
        }
        for row in measurement_rows:
            calibration = calibration_by_key[(str(row["cohort"]), str(row["section"]))]
            row.update(
                {
                    "harmonization_status": calibration.calibration.status,
                    "harmonization_scale": calibration.calibration.scale,
                    "harmonization_fingerprint": harmonization.fingerprint,
                }
            )
        measurement_view = _MEASUREMENT_VIEW
        od_calibration: dict[str, object] | None = harmonization.as_dict()
        harmonization_scope = (
            "technical repeated-assay scale; within-section rank and spatial "
            "structure preserved; not absolute cross-specimen OD"
        )
    else:
        combined["measured_od"] = raw_measured
        measurement_view = source_view
        od_calibration = None
        harmonization_scope = "none; validated per-section OD is unchanged"
    bindings = _cohort_bindings(contexts)
    table = CellExpressionTable(
        target_id=config.target.target_id,
        **combined,
        provenance={
            "schema_version": 3,
            "measurement_view": measurement_view,
            "measurement_source_view": source_view,
            "measurement_compartment": config.target.compartment,
            "measurement_statistic": config.target.measurement_statistic,
            "coverage_definition": "tissue_supported_compartment_area_fraction",
            "feature_view": str(
                getattr(
                    config,
                    "feature_schema_id",
                    "native-hdab-neutral-spatial-uni2h-v2",
                )
            ),
            "cohort_bindings": bindings,
            "target_sections": measurement_rows,
            "od_calibration": od_calibration,
            "harmonization_scope": harmonization_scope,
            **(
                {"harmonization_reference_cohorts": list(reference_cohorts)}
                if reference_cohorts is not None
                else {}
            ),
            "semantic_role": "soft_region_prior_not_cell_type_bound",
        },
    )
    return table.save(output)


def _load_shared_auxiliary_tables(
    paths: tuple[Path | str, ...],
    *,
    primary: object,
    architecture: str,
) -> tuple[object, ...]:
    """Load exact target tables used only to regularize a shared trunk."""

    if architecture != "shared_multitask":
        if paths:
            raise ValueError(
                "auxiliary protein tables require the shared_multitask architecture"
            )
        return ()
    if not paths:
        raise ValueError("shared_multitask requires at least one auxiliary table")
    from histopia.protein._result import CellExpressionTable

    resolved = tuple(Path(value).expanduser().resolve() for value in paths)
    if len(set(resolved)) != len(resolved):
        raise ValueError("shared multitask auxiliary tables must be unique")
    loaded = tuple(CellExpressionTable.load(value) for value in resolved)
    primary_target = str(primary.target_id)
    targets = tuple(str(value.target_id) for value in loaded)
    if primary_target in targets or len(set(targets)) != len(targets):
        raise ValueError("shared multitask target identities must be unique")
    feature_view = primary.provenance.get("feature_view")
    cohort_bindings = primary.provenance.get("cohort_bindings")
    for value in loaded:
        if (
            value.features.ndim != 2
            or value.features.shape[1] != primary.features.shape[1]
            or value.provenance.get("feature_view") != feature_view
            or value.provenance.get("cohort_bindings") != cohort_bindings
            or value.provenance.get("measurement_view")
            not in _SUPPORTED_MEASUREMENT_VIEWS
        ):
            raise ValueError(
                f"shared multitask auxiliary scope differs for {value.target_id}"
            )
    return tuple(sorted(loaded, key=lambda value: value.target_id))


def _resolve_shared_auxiliary_scope(
    paths: tuple[Path | str, ...],
    *,
    primary: object,
    architecture: str,
    reuse_models_from: Path | str | None,
) -> tuple[tuple[object, ...], dict[str, str]]:
    """Resolve training-only auxiliary lineage for fitting or model reuse.

    A shared-multitask estimator consumes auxiliary outcomes only while its
    trunk is fitted.  Reusing an immutable estimator on a newly added cohort
    must therefore preserve the source tables' fingerprints without requiring
    destination copies whose cohort bindings would necessarily differ.  New
    fits still require the complete, binding-matched auxiliary tables.
    """

    if paths or architecture != "shared_multitask" or reuse_models_from is None:
        loaded = _load_shared_auxiliary_tables(
            paths,
            primary=primary,
            architecture=architecture,
        )
        return loaded, {
            str(value.target_id): str(value.fingerprint) for value in loaded
        }

    from histopia.protein._manifest import validate_protein_result_index

    source = validate_protein_result_index(reuse_models_from)
    raw = source.get("auxiliary_table_fingerprints")
    primary_target = str(primary.target_id)
    if (
        not isinstance(raw, dict)
        or not raw
        or primary_target in raw
        or any(
            not isinstance(target, str)
            or not target
            or not isinstance(fingerprint, str)
            or len(fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in fingerprint)
            for target, fingerprint in raw.items()
        )
    ):
        raise ValueError("reused shared_multitask auxiliary lineage is malformed")
    return (), {str(target): str(raw[target]) for target in sorted(raw)}


def fit_real_protein_study_variant(
    config: object,
    table_path: Path | str,
    cohorts: dict[str, RealProteinStudyCohort],
    output_dir: Path | str,
    *,
    geometry_cache: Path | str,
    architecture: str,
    training_cohorts: tuple[str, ...] | None = None,
    prediction_cohorts: tuple[str, ...] | None = None,
    prediction_protocol: str = "leave-one-mouse-out",
    device: str = "auto",
    epochs: int | None = None,
    reuse_models_from: Path | str | None = None,
    reuse_refinement_evidence: Path | str | None = None,
    auxiliary_table_paths: tuple[Path | str, ...] = (),
) -> Path:
    """Fit and seal one multi-mouse variant for selected available slides.

    ``leave-one-mouse-out`` remains the generalization estimate and default.
    ``training-visible`` uses the final all-training model for training cohorts;
    it is an explicitly labelled capacity/reconstruction upper bound and is
    never eligible for production promotion. ``prediction_cohorts`` limits only
    streamed whole-slide predictions; model fitting, cohort bindings, and
    sampled-table evaluation retain the complete declared study scope.
    """

    from histopia._atomic import write_json_atomic
    from histopia.protein._manifest import write_protein_result
    from histopia.protein._result import CellExpressionTable, ProteinPredictions

    if architecture not in _ARCHITECTURES:
        raise ValueError(f"unsupported study architecture: {architecture}")
    if prediction_protocol not in _PREDICTION_PROTOCOLS:
        raise ValueError(
            f"unsupported study prediction protocol: {prediction_protocol}"
        )
    if epochs is not None and epochs < 1:
        raise ValueError("protein study epochs must be positive")
    if reuse_models_from is not None and epochs is not None:
        raise ValueError("reused protein models already have sealed training epochs")
    if reuse_refinement_evidence is not None and reuse_models_from is None:
        raise ValueError("post-fit refinement requires a sealed reused model")
    contexts = _contexts(config, cohorts, verify_artifacts=False)
    table = CellExpressionTable.load(table_path)
    if table.target_id != config.target.target_id:
        raise ValueError("study table target differs from model config")
    if reuse_models_from is None:
        refinement = None
        refinement_already_applied = False
    else:
        refinement, refinement_already_applied = _load_morphology_transfer_refinement(
            reuse_refinement_evidence,
            source_run=reuse_models_from,
            architecture=architecture,
            target_id=table.target_id,
            weight=float(getattr(config, "relational_morphology_transfer_weight", 0.0)),
            neighbors=int(
                getattr(config, "relational_morphology_transfer_neighbors", 16)
            ),
        )
    auxiliary_tables, auxiliary_bindings = _resolve_shared_auxiliary_scope(
        auxiliary_table_paths,
        primary=table,
        architecture=architecture,
        reuse_models_from=reuse_models_from,
    )
    measurement_view = str(table.provenance.get("measurement_view", ""))
    if measurement_view not in _SUPPORTED_MEASUREMENT_VIEWS:
        raise ValueError("study table does not use strict adaptive target OD")
    od_calibration = table.provenance.get("od_calibration")
    if measurement_view == _MEASUREMENT_VIEW:
        study_calibrations = _validate_study_od_calibration(od_calibration)
    elif od_calibration is not None:
        raise ValueError("unharmonized study table declares OD calibration")
    else:
        study_calibrations = {}
    feature_view = str(table.provenance.get("feature_view", ""))
    if feature_view not in {
        "native-hdab-neutral-spatial-uni2h-v2",
        "native-hdab-neutral-cell-multiscale-v3",
    }:
        raise ValueError("study table does not use neutral cell-resolved features")
    configured_feature_view = str(
        getattr(
            config,
            "feature_schema_id",
            "native-hdab-neutral-spatial-uni2h-v2",
        )
    )
    if feature_view != configured_feature_view:
        raise ValueError("study table feature schema differs from model config")
    if table.provenance.get("cohort_bindings") != _cohort_bindings(contexts):
        raise ValueError("study table upstream cohort bindings are stale")
    if table.provenance.get("measurement_statistic", "mean") != (
        config.target.measurement_statistic
    ):
        raise ValueError("study table measurement statistic differs from config")
    available_mice = tuple(sorted(contexts))
    if prediction_cohorts is None:
        prediction_mice = available_mice
    else:
        requested_prediction_mice = tuple(str(value) for value in prediction_cohorts)
        if (
            not requested_prediction_mice
            or len(set(requested_prediction_mice)) != len(requested_prediction_mice)
            or set(requested_prediction_mice) - set(available_mice)
        ):
            raise ValueError("protein prediction cohort selection is invalid")
        prediction_mice = tuple(sorted(requested_prediction_mice))
    mouse_values = np.asarray(table.mouse_ids, dtype=str)
    measured_mice = set(
        mouse_values[np.isfinite(np.asarray(table.measured_od, dtype=np.float64))]
    )
    training_mice = tuple(training_cohorts or sorted(measured_mice))
    if not training_mice or set(training_mice) - set(available_mice):
        raise ValueError("protein training cohort selection is invalid")
    root = Path(output_dir)
    model_root = root / "models"
    prediction_root = root / "predictions"
    model_root.mkdir(parents=True, exist_ok=True)
    prediction_root.mkdir(parents=True, exist_ok=True)
    table_copy = root / "cell_expression_table.npz"
    shutil.copyfile(table_path, table_copy)

    missing_training_truth = set(training_mice) - measured_mice
    if missing_training_truth:
        raise ValueError(
            "protein training cohorts lack accepted target measurements: "
            + ", ".join(sorted(missing_training_truth))
        )
    external_holdout_mice = tuple(sorted(measured_mice - set(training_mice)))
    if measurement_view == _MEASUREMENT_VIEW and external_holdout_mice:
        _training_only_harmonization_references(
            table.provenance.get("harmonization_reference_cohorts"),
            training_mice,
        )
    all_training = np.flatnonzero(np.isin(mouse_values, training_mice))
    reused_result_fingerprint: str | None = None
    if reuse_models_from is None:
        resumed = _resume_partial_study_candidates(
            architecture,
            config,
            table,
            all_training=all_training,
            available_mice=available_mice,
            training_mice=training_mice,
            mouse_values=mouse_values,
            root=root,
            model_root=model_root,
            device=device,
            epochs=epochs,
            prediction_protocol=prediction_protocol,
            measurement_view=measurement_view,
            feature_view=feature_view,
            auxiliary_bindings=auxiliary_bindings,
            prediction_mice=(
                prediction_mice if prediction_cohorts is not None else None
            ),
        )
        if resumed is None:
            fitted, final_candidate, model_artifacts, fold_metrics = (
                _fit_study_candidates(
                    architecture,
                    config,
                    table,
                    all_training=all_training,
                    available_mice=available_mice,
                    training_mice=training_mice,
                    mouse_values=mouse_values,
                    root=root,
                    model_root=model_root,
                    device=device,
                    epochs=epochs,
                    auxiliary_tables=auxiliary_tables,
                )
            )
        else:
            fitted, final_candidate, model_artifacts, fold_metrics = resumed
        training_epochs = epochs or _default_training_epochs(architecture)
    else:
        (
            fitted,
            final_candidate,
            model_artifacts,
            fold_metrics,
            reused_result_fingerprint,
            training_epochs,
        ) = _reuse_study_candidates(
            reuse_models_from,
            architecture=architecture,
            table=table,
            available_mice=available_mice,
            training_mice=training_mice,
            root=root,
            model_root=model_root,
            measurement_view=measurement_view,
            feature_view=feature_view,
            cohort_bindings=_cohort_bindings(contexts),
            auxiliary_bindings=auxiliary_bindings,
            postfit_refinement=refinement,
            postfit_refinement_already_applied=refinement_already_applied,
        )
        if refinement is not None and not refinement_already_applied:
            fold_metrics = _evaluate_reused_fold_candidates(
                fitted,
                config=config,
                table=table,
                all_training=all_training,
                training_mice=training_mice,
                mouse_values=mouse_values,
                device=device,
            )
    protocol = {
        "schema_version": 1,
        "target_id": table.target_id,
        "architecture": architecture,
        "prediction_protocol": prediction_protocol,
        "training_cohorts": list(training_mice),
        "table_fingerprint": table.fingerprint,
        "feature_schema_id": feature_view,
        "fold_fingerprints": {
            mouse: candidate.fingerprint for mouse, candidate in sorted(fitted.items())
        },
        "final_fingerprint": final_candidate.fingerprint,
        "measurement_view": measurement_view,
        "measurement_compartment": config.target.compartment,
        "measurement_statistic": config.target.measurement_statistic,
        "training_epochs": training_epochs,
        "auxiliary_table_fingerprints": auxiliary_bindings,
    }
    relational_loss_profile = str(
        getattr(config, "relational_loss_profile", "huber_v1")
    )
    if relational_loss_profile != "huber_v1":
        protocol["relational_loss_profile"] = relational_loss_profile
    implementation = _study_runtime_metadata(architecture)
    if implementation is not None:
        protocol["implementation"] = implementation
    if external_holdout_mice:
        protocol["external_holdout_cohorts"] = list(external_holdout_mice)
    if prediction_cohorts is not None:
        protocol["prediction_cohorts"] = list(prediction_mice)
    if reused_result_fingerprint is not None:
        protocol["reused_model_result_fingerprint"] = reused_result_fingerprint
    if refinement is not None:
        protocol["postfit_refinement"] = refinement
    protocol_fingerprint = _fingerprint_json(protocol)
    model_metadata = write_json_atomic(
        root / "model_bundle.json",
        {**protocol, "fingerprint": protocol_fingerprint},
    )

    rows: list[dict[str, object]] = []
    geometry_root = Path(geometry_cache)
    measurement_audit = list(table.provenance.get("target_sections", []))
    for mouse_id in prediction_mice:
        context = contexts[mouse_id]
        for section, cell_row in sorted(context.cell_by_section.items()):
            evaluation_role = _study_prediction_role(
                mouse_id=mouse_id,
                section=section,
                training_mice=training_mice,
                target_sections=context.target_sections,
                prediction_protocol=prediction_protocol,
            )
            training_visible = (
                prediction_protocol == "training-visible" and mouse_id in training_mice
            )
            candidate = final_candidate if training_visible else fitted[mouse_id]
            train = (
                all_training
                if training_visible
                else all_training[mouse_values[all_training] != mouse_id]
                if mouse_id in training_mice and len(training_mice) > 1
                else all_training
            )
            prediction_path = prediction_root / mouse_id / f"{section}.npz"
            if prediction_path.is_file():
                prediction = ProteinPredictions.load(prediction_path)
                _validate_resumable_prediction(
                    prediction,
                    target_id=table.target_id,
                    model_fingerprint=protocol_fingerprint,
                    fold_model_fingerprint=candidate.fingerprint,
                    architecture=architecture,
                    cohort=mouse_id,
                    section=section,
                    cell_count=int(cell_row["cell_count"]),
                    evaluation_role=evaluation_role,
                    measurement_view=measurement_view,
                    feature_view=feature_view,
                    prediction_protocol=prediction_protocol,
                )
                rows.append(
                    {
                        "cohort": mouse_id,
                        "section": section,
                        "predictions": prediction_path.relative_to(root).as_posix(),
                        "prediction_fingerprint": prediction.fingerprint,
                        "cells": int(cell_row["cell_count"]),
                        "measured_cells": int(
                            np.isfinite(prediction.measured_od).sum()
                        ),
                        "has_probability": bool(
                            np.any(np.isfinite(prediction.expression_probability))
                        ),
                        "section_display_max_od": _section_display_max(prediction),
                        "evaluation_role": evaluation_role,
                    }
                )
                continue
            design = _section_design(
                context,
                section,
                config,
                geometry_root=geometry_root,
                include_semantic=(
                    feature_view == "native-hdab-neutral-spatial-uni2h-v2"
                ),
            )
            probability, relative, od, uncertainty = candidate.predict(
                design.features,
                reference_um_xyz=design.reference_um_xyz,
                device=device,
            )
            if (
                architecture == "cross_attention"
                and feature_view == "native-hdab-neutral-spatial-uni2h-v2"
            ):
                from histopia.protein._cli import _region_centered_blend
                from histopia.protein._real_workflow import (
                    _semantic_reference_for_regions,
                )

                semantic = _semantic_reference_for_regions(
                    table,
                    design.regions,
                    train=train,
                )
                od = _region_centered_blend(
                    od,
                    semantic,
                    design.regions,
                    weight=float(config.semantic_blend_weight),
                ).astype(np.float32)
                uncertainty = np.asarray(uncertainty, dtype=np.float32) * float(
                    config.semantic_blend_weight
                )
                reference = np.sort(
                    np.asarray(table.measured_od, dtype=np.float32)[train]
                )
                reference = reference[np.isfinite(reference)]
                relative = (
                    np.searchsorted(reference, od, side="right")
                    / max(len(reference), 1)
                ).astype(np.float32)
            measured = _full_section_measurement(
                context,
                section,
                design,
                minimum_effective_pixels=float(config.minimum_effective_pixels),
                minimum_coverage=float(config.minimum_coverage),
                compartment=config.target.compartment,
                statistic=config.target.measurement_statistic,
                calibration=study_calibrations.get((mouse_id, section)),
            )
            supported = np.asarray(design.supported, dtype=bool)
            prediction = ProteinPredictions(
                target_id=table.target_id,
                model_fingerprint=protocol_fingerprint,
                label_ids=design.label_ids,
                section_ids=np.full(len(design.label_ids), section),
                expression_probability=(
                    np.full(len(od), np.nan, dtype=np.float32)
                    if probability is None
                    else probability
                ),
                relative_expression=relative,
                predicted_od_reference=od,
                measured_od=measured,
                uncertainty=uncertainty,
                supported=supported,
                provenance={
                    "feature_view": feature_view,
                    "measurement_view": measurement_view,
                    "cohort": mouse_id,
                    "architecture": architecture,
                    "fold_model_fingerprint": candidate.fingerprint,
                    "evaluation_role": evaluation_role,
                    "prediction_protocol": prediction_protocol,
                },
            )
            path = prediction.save(prediction_path)
            rows.append(
                {
                    "cohort": mouse_id,
                    "section": section,
                    "predictions": path.relative_to(root).as_posix(),
                    "prediction_fingerprint": prediction.fingerprint,
                    "cells": int(cell_row["cell_count"]),
                    "measured_cells": int(np.isfinite(measured).sum()),
                    "has_probability": bool(
                        np.any(np.isfinite(prediction.expression_probability))
                    ),
                    "section_display_max_od": _section_display_max(prediction),
                    "evaluation_role": prediction.provenance["evaluation_role"],
                }
            )
    measured_scale = np.asarray(table.measured_od, dtype=np.float64)
    measured_scale = measured_scale[np.isfinite(measured_scale)]
    display_max = max(float(np.quantile(measured_scale, 0.99)), 1e-8)
    audit_path = write_json_atomic(
        root / "measurement_audit.json",
        {
            "schema_version": 1,
            "measurement_view": measurement_view,
            "measurement_compartment": config.target.compartment,
            "measurement_statistic": config.target.measurement_statistic,
            "sections": measurement_audit,
            "od_calibration": od_calibration,
            "prediction_protocol": prediction_protocol,
        },
    )
    held_out_metrics = _aggregate_study_metrics(fold_metrics)
    external_holdout_metrics = _evaluate_external_holdouts(
        final_candidate,
        architecture=architecture,
        config=config,
        table=table,
        training=all_training,
        holdout_mice=external_holdout_mice,
        device=device,
    )
    if prediction_protocol == "training-visible":
        training_predictions = _predict_table_candidate(
            final_candidate,
            table,
            all_training,
            all_training,
            config,
            device=device,
        )
        metrics = _training_visible_metrics(
            table,
            all_training,
            training_predictions,
            held_out_reference=held_out_metrics,
        )
    else:
        metrics = held_out_metrics
    from histopia.protein._selection import evaluate_protein_promotion

    held_out_promotion = evaluate_protein_promotion(held_out_metrics)
    promotion_accepted = (
        held_out_promotion.accepted
        if prediction_protocol == "leave-one-mouse-out"
        else False
    )
    promotion_reasons = (
        held_out_promotion.reasons
        if prediction_protocol == "leave-one-mouse-out"
        else (
            "training-visible fit is a diagnostic upper bound, not an "
            "independent generalization estimate",
        )
    )
    suffix = (
        f"all{len(available_mice)}"
        if training_mice == available_mice
        else "-".join(training_mice)
    )
    harmonized = measurement_view == _MEASUREMENT_VIEW
    counterstain_v3 = measurement_view == _BACKGROUND_MEASUREMENT_VIEW
    base_model_version = (
        "adaptive-harmonized-v2"
        if harmonized
        else "counterstain-conditioned-v3"
        if counterstain_v3
        else "adaptive-v1"
    )
    model_version = (
        f"{base_model_version}-training-visible"
        if prediction_protocol == "training-visible"
        else base_model_version
    )
    if feature_view == "native-hdab-neutral-cell-multiscale-v3":
        model_version += "-cell-multiscale-v3"
    if refinement is not None:
        model_version += "-morphology-transfer-v1"
    model_id = (
        f"{table.target_id}-{architecture.replace('_', '-')}-{suffix}-{model_version}"
    )
    protocol_label = (
        " · measured-fit upper bound"
        if prediction_protocol == "training-visible"
        else ""
    )
    return write_protein_result(
        root,
        {
            "schema_version": 4,
            "model_id": model_id,
            "model_label": (
                f"{table.target_id.upper()} · {architecture.replace('_', ' ')} · "
                f"trained {'+'.join(training_mice)}"
                f"{' · harmonized adaptive OD' if harmonized else ''}"
                f"{' · counterstain-conditioned OD' if counterstain_v3 else ''}"
                f" · {config.target.measurement_statistic} per cell"
                f"{protocol_label}"
            ),
            "architecture": architecture,
            **(
                {"implementation": implementation} if implementation is not None else {}
            ),
            "model_version": model_version,
            "feature_schema_id": feature_view,
            "prediction_protocol": prediction_protocol,
            "training_cohorts": list(training_mice),
            **(
                {"prediction_cohorts": list(prediction_mice)}
                if prediction_cohorts is not None
                else {}
            ),
            "auxiliary_targets": list(auxiliary_bindings),
            "auxiliary_table_fingerprints": auxiliary_bindings,
            "cohort_bindings": _cohort_bindings(contexts),
            "measurement_view": measurement_view,
            "measurement_source_view": table.provenance.get("measurement_source_view"),
            "measurement_statistic": config.target.measurement_statistic,
            "od_calibration": od_calibration,
            "target_id": table.target_id,
            "target_label": config.target.display_name,
            "assay_domain": config.target.assay_domain,
            "model": model_metadata.relative_to(root).as_posix(),
            "model_artifacts": model_artifacts,
            "training_table": table_copy.name,
            "measurement_audit": audit_path.name,
            "slides": rows,
            "model_fingerprint": protocol_fingerprint,
            "table_fingerprint": table.fingerprint,
            **(
                {
                    "model_reuse": {
                        "scope": "identical-training-scope-v1",
                        "source_result_fingerprint": reused_result_fingerprint,
                    }
                }
                if reused_result_fingerprint is not None
                else {}
            ),
            **({"postfit_refinement": refinement} if refinement is not None else {}),
            "metrics": metrics,
            **(
                {"external_holdout_metrics": external_holdout_metrics}
                if external_holdout_mice
                else {}
            ),
            "binary_enabled": bool(config.target.binary_enabled),
            "status": "promoted" if promotion_accepted else "candidate",
            "candidate_promotion": {
                "accepted": promotion_accepted,
                "reasons": list(promotion_reasons),
                "policy": (
                    "held-out-accuracy-baseline-parity-and-od-bias-v1"
                    if prediction_protocol == "leave-one-mouse-out"
                    else "training-visible-capacity-diagnostic-v1"
                ),
                "held_out_reference_accepted": held_out_promotion.accepted,
                "held_out_reference_reasons": list(held_out_promotion.reasons),
            },
            "target_global_display_max_od": display_max,
            "prediction_scope": (
                "training-visible measured-section reconstruction and "
                "within-mouse transfer; held-out reference retained"
                if prediction_protocol == "training-visible"
                else "leave-mouse-out harmonized adaptive target OD prediction"
                if harmonized
                else "leave-mouse-out adaptive target OD prediction"
            ),
            "semantic_role": (
                "auxiliary_reporting_only"
                if feature_view == "native-hdab-neutral-cell-multiscale-v3"
                else "soft_region_prior_not_cell_type_bound"
            ),
        },
    )


def ensemble_protein_study_results(
    parent_runs: tuple[Path | str, ...],
    output_dir: Path | str,
) -> Path:
    """Seal an equal-weight ensemble of independent schema-v4 study models.

    Weights are deliberately fixed rather than optimized on held-out outcomes.
    Parent predictions must cover the same cells, measurement view, cohorts,
    and sections. The ensemble's uncertainty combines member uncertainty with
    between-model disagreement, and all parent model artifacts are copied into
    the portable result.
    """

    from histopia._atomic import write_json_atomic
    from histopia.protein._manifest import (
        validate_protein_result,
        write_protein_result,
    )
    from histopia.protein._model import evaluate_predictions
    from histopia.protein._result import CellExpressionTable, ProteinPredictions

    if len(parent_runs) < 2:
        raise ValueError("protein ensemble requires at least two parent models")
    roots = tuple(Path(path).expanduser().resolve() for path in parent_runs)
    parents = tuple(validate_protein_result(root) for root in roots)
    if any(parent.get("schema_version") != 4 for parent in parents):
        raise ValueError("protein ensemble requires schema-v4 study parents")
    if len({str(parent["fingerprint"]) for parent in parents}) != len(parents):
        raise ValueError("protein ensemble parents must be distinct")
    shared_keys = (
        "target_id",
        "training_cohorts",
        "cohort_bindings",
        "measurement_view",
        "measurement_source_view",
        "measurement_statistic",
        "od_calibration",
        "table_fingerprint",
        "assay_domain",
        "binary_enabled",
    )
    first = parents[0]
    for parent in parents[1:]:
        if any(parent.get(key) != first.get(key) for key in shared_keys):
            raise ValueError("protein ensemble parent scientific scopes differ")
    measurement_view = str(first.get("measurement_view", ""))
    if measurement_view not in _SUPPORTED_MEASUREMENT_VIEWS:
        raise ValueError("protein ensemble parent measurement view is unsupported")
    target_id = str(first["target_id"])
    training_cohorts = tuple(str(value) for value in first["training_cohorts"])
    table_source = roots[0] / str(first["training_table"])
    table = CellExpressionTable.load(table_source)
    if table.fingerprint != first.get("table_fingerprint"):
        raise ValueError("protein ensemble training table fingerprint differs")

    indexed_slides: list[dict[tuple[str, str], dict[str, object]]] = []
    for parent in parents:
        rows = parent.get("slides")
        if not isinstance(rows, list):
            raise ValueError("protein ensemble parent slides are missing")
        indexed: dict[tuple[str, str], dict[str, object]] = {}
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("protein ensemble parent slide is invalid")
            key = (str(row.get("cohort", "")), str(row.get("section", "")))
            if not all(key) or key in indexed:
                raise ValueError("protein ensemble parent slide identity is invalid")
            indexed[key] = row
        indexed_slides.append(indexed)
    slide_keys = tuple(sorted(indexed_slides[0]))
    if any(tuple(sorted(rows)) != slide_keys for rows in indexed_slides[1:]):
        raise ValueError("protein ensemble parent slide coverage differs")

    root = Path(output_dir)
    prediction_root = root / "predictions"
    prediction_root.mkdir(parents=True, exist_ok=True)
    table_copy = root / "cell_expression_table.npz"
    shutil.copyfile(table_source, table_copy)
    audit_copy = root / "measurement_audit.json"
    shutil.copyfile(roots[0] / str(first["measurement_audit"]), audit_copy)
    weights = np.full(len(parents), 1.0 / len(parents), dtype=np.float64)
    protocol = {
        "schema_version": 1,
        "method": "equal-weight-independent-model-ensemble-v1",
        "target_id": target_id,
        "training_cohorts": list(training_cohorts),
        "measurement_view": measurement_view,
        "table_fingerprint": table.fingerprint,
        "parents": [
            {
                "model_id": parent["model_id"],
                "architecture": parent["architecture"],
                "result_fingerprint": parent["fingerprint"],
                "model_fingerprint": parent["model_fingerprint"],
                "weight": float(weight),
            }
            for parent, weight in zip(parents, weights, strict=True)
        ],
    }
    model_fingerprint = _fingerprint_json(protocol)
    model_metadata = write_json_atomic(
        root / "ensemble_bundle.json",
        {**protocol, "fingerprint": model_fingerprint},
    )
    copied_model_artifacts: list[str] = []
    for index, (parent_root, parent) in enumerate(zip(roots, parents, strict=True)):
        sources = [parent["model"], *parent.get("model_artifacts", [])]
        for value in sources:
            relative = Path(str(value))
            destination = root / "parent_models" / str(index) / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(parent_root / relative, destination)
            copied_model_artifacts.append(destination.relative_to(root).as_posix())

    combined_predictions: dict[tuple[str, str], ProteinPredictions] = {}
    rows: list[dict[str, object]] = []
    table_mice = np.asarray(table.mouse_ids, dtype=str)
    for cohort, section in slide_keys:
        members = tuple(
            ProteinPredictions.load(
                parent_root / str(parent_rows[(cohort, section)]["predictions"])
            )
            for parent_root, parent_rows in zip(roots, indexed_slides, strict=True)
        )
        _validate_ensemble_members(members, cohort=cohort, section=section)
        member_od = np.stack(
            [np.asarray(member.predicted_od_reference) for member in members]
        )
        predicted = np.average(member_od, axis=0, weights=weights).astype(np.float32)
        member_uncertainty = np.stack(
            [np.asarray(member.uncertainty) for member in members]
        )
        within = np.sum((weights[:, None] * member_uncertainty) ** 2, axis=0)
        disagreement = np.average(
            (member_od - predicted[None, :]) ** 2,
            axis=0,
            weights=weights,
        )
        uncertainty = np.sqrt(np.maximum(within + disagreement, 0)).astype(np.float32)
        training = np.flatnonzero(table_mice != cohort)
        if cohort not in training_cohorts or len(training_cohorts) == 1:
            training = np.flatnonzero(np.isin(table_mice, training_cohorts))
        reference = np.sort(np.asarray(table.measured_od)[training])
        reference = reference[np.isfinite(reference)]
        relative = (
            np.searchsorted(reference, predicted, side="right") / max(len(reference), 1)
        ).astype(np.float32)
        probabilities = np.stack(
            [np.asarray(member.expression_probability) for member in members]
        )
        probability = (
            np.average(probabilities, axis=0, weights=weights).astype(np.float32)
            if np.all(np.isfinite(probabilities))
            else np.full(len(predicted), np.nan, dtype=np.float32)
        )
        roles = {str(member.provenance.get("evaluation_role")) for member in members}
        if len(roles) != 1:
            raise ValueError("protein ensemble parent evaluation roles differ")
        supported = np.logical_and.reduce(
            [np.asarray(member.supported, dtype=bool) for member in members]
        )
        combined = ProteinPredictions(
            target_id=target_id,
            model_fingerprint=model_fingerprint,
            label_ids=members[0].label_ids,
            section_ids=members[0].section_ids,
            expression_probability=probability,
            relative_expression=relative,
            predicted_od_reference=predicted,
            measured_od=members[0].measured_od,
            uncertainty=uncertainty,
            supported=supported,
            provenance={
                "feature_view": "native-hdab-neutral-spatial-uni2h-v2",
                "measurement_view": measurement_view,
                "cohort": cohort,
                "architecture": "equal_ensemble",
                "parent_model_fingerprints": [
                    parent["model_fingerprint"] for parent in parents
                ],
                "evaluation_role": roles.pop(),
            },
        )
        path = combined.save(prediction_root / cohort / f"{section}.npz")
        combined_predictions[(cohort, section)] = combined
        rows.append(
            {
                "cohort": cohort,
                "section": section,
                "predictions": path.relative_to(root).as_posix(),
                "prediction_fingerprint": combined.fingerprint,
                "cells": len(combined.label_ids),
                "measured_cells": int(np.isfinite(combined.measured_od).sum()),
                "evaluation_role": combined.provenance["evaluation_role"],
            }
        )

    predicted_table = np.full(len(table.label_ids), np.nan, dtype=np.float32)
    for cohort, section in slide_keys:
        selected = np.flatnonzero(
            (table_mice == cohort)
            & (np.asarray(table.section_ids, dtype=str) == section)
        )
        if not len(selected):
            continue
        prediction = combined_predictions[(cohort, section)]
        positions = np.searchsorted(prediction.label_ids, table.label_ids[selected])
        if np.any(positions >= len(prediction.label_ids)) or not np.array_equal(
            prediction.label_ids[positions], table.label_ids[selected]
        ):
            raise ValueError("protein ensemble table cells differ from predictions")
        predicted_table[selected] = prediction.predicted_od_reference[positions]
    fold_metrics: list[dict[str, object]] = []
    from histopia.protein._cli import _compare_with_baseline

    for held_out in training_cohorts:
        test = np.flatnonzero(table_mice == held_out)
        train = np.flatnonzero(
            np.isin(table_mice, training_cohorts) & (table_mice != held_out)
        )
        base = evaluate_predictions(
            table.measured_od[test],
            predicted_table[test],
            binary_label=table.binary_label[test],
        )
        metrics = _compare_with_baseline(
            base,
            table,
            train,
            test,
            predicted_table[test],
        )
        fold_metrics.append({"held_out": held_out, **metrics})
    metrics = _aggregate_study_metrics(fold_metrics)
    from histopia.protein._selection import evaluate_protein_promotion

    promotion = evaluate_protein_promotion(metrics)
    display_values = np.asarray(table.measured_od, dtype=np.float64)
    display_values = display_values[np.isfinite(display_values)]
    display_max = max(float(np.quantile(display_values, 0.99)), 1e-8)
    model_version = (
        "adaptive-harmonized-v2"
        if measurement_view == _MEASUREMENT_VIEW
        else "counterstain-conditioned-v3"
        if measurement_view == _BACKGROUND_MEASUREMENT_VIEW
        else "adaptive-v1"
    )
    model_id = f"{target_id}-equal-ensemble-all{len(training_cohorts)}-{model_version}"
    harmonized_label = (
        " · harmonized adaptive OD"
        if model_version == "adaptive-harmonized-v2"
        else " · counterstain-conditioned OD"
        if model_version == "counterstain-conditioned-v3"
        else ""
    )
    return write_protein_result(
        root,
        {
            "schema_version": 4,
            "model_id": model_id,
            "model_label": (
                f"{target_id.upper()} · equal ExtraTrees + attention ensemble · "
                f"trained {'+'.join(training_cohorts)}"
                f"{harmonized_label}"
            ),
            "architecture": "equal_ensemble",
            "model_version": model_version,
            "training_cohorts": list(training_cohorts),
            "cohort_bindings": first["cohort_bindings"],
            "measurement_view": measurement_view,
            "measurement_source_view": first.get("measurement_source_view"),
            "measurement_statistic": first.get("measurement_statistic", "mean"),
            "od_calibration": first.get("od_calibration"),
            "target_id": target_id,
            "assay_domain": first["assay_domain"],
            "model": model_metadata.relative_to(root).as_posix(),
            "model_artifacts": copied_model_artifacts,
            "training_table": table_copy.name,
            "measurement_audit": audit_copy.name,
            "slides": rows,
            "model_fingerprint": model_fingerprint,
            "table_fingerprint": table.fingerprint,
            "metrics": metrics,
            "binary_enabled": bool(first.get("binary_enabled")),
            "status": "promoted" if promotion.accepted else "candidate",
            "candidate_promotion": {
                "accepted": promotion.accepted,
                "reasons": list(promotion.reasons),
                "policy": "held-out-accuracy-baseline-parity-and-od-bias-v1",
            },
            "target_global_display_max_od": display_max,
            "prediction_scope": (
                "leave-mouse-out equal-model ensemble on "
                f"{'harmonized ' if model_version == 'adaptive-harmonized-v2' else ''}"
                "adaptive target OD"
            ),
            "semantic_role": "soft_region_prior_not_cell_type_bound",
        },
    )


def _validate_ensemble_members(
    members: tuple[object, ...],
    *,
    cohort: str,
    section: str,
) -> None:
    """Require identical cells, support, and truth across ensemble parents."""

    first = members[0]
    expected_sections = np.asarray(first.section_ids, dtype=str)
    if set(expected_sections) != {section}:
        raise ValueError("protein ensemble parent section identity differs")
    for member in members[1:]:
        if (
            member.target_id != first.target_id
            or not np.array_equal(member.label_ids, first.label_ids)
            or not np.array_equal(
                np.asarray(member.section_ids, dtype=str), expected_sections
            )
            or not np.array_equal(member.supported, first.supported)
            or not np.allclose(
                member.measured_od,
                first.measured_od,
                rtol=0,
                atol=0,
                equal_nan=True,
            )
            or member.provenance.get("cohort") != cohort
            or member.provenance.get("measurement_view")
            != first.provenance.get("measurement_view")
        ):
            raise ValueError("protein ensemble parent predictions differ in scope")


def _validate_resumable_prediction(
    prediction: object,
    *,
    target_id: str,
    model_fingerprint: str,
    fold_model_fingerprint: str,
    architecture: str,
    cohort: str,
    section: str,
    cell_count: int,
    evaluation_role: str,
    measurement_view: str = _ADAPTIVE_MEASUREMENT_VIEW,
    feature_view: str = "native-hdab-neutral-spatial-uni2h-v2",
    prediction_protocol: str | None = None,
) -> None:
    """Require a complete partial artifact to match the exact resumed run."""

    values = np.asarray(prediction.section_ids, dtype=str)
    provenance = prediction.provenance
    expected = {
        "feature_view": feature_view,
        "measurement_view": measurement_view,
        "cohort": cohort,
        "architecture": architecture,
        "fold_model_fingerprint": fold_model_fingerprint,
        "evaluation_role": evaluation_role,
    }
    if prediction_protocol is not None:
        expected["prediction_protocol"] = prediction_protocol
    if (
        prediction.target_id != target_id
        or prediction.model_fingerprint != model_fingerprint
        or len(prediction.label_ids) != cell_count
        or set(values) != {section}
        or any(provenance.get(key) != value for key, value in expected.items())
    ):
        raise ValueError(f"existing protein prediction is stale for {cohort}/{section}")


def _contexts(
    config: object,
    cohorts: dict[str, RealProteinStudyCohort],
    *,
    verify_artifacts: bool = True,
) -> dict[str, _Context]:
    from histopia.protein._real_data import target_sections, validate_real_run_bindings

    contexts: dict[str, _Context] = {}
    for mouse_id, source in sorted(cohorts.items()):
        bindings = validate_real_run_bindings(
            source.registration_run,
            source.cell_run,
            source.stain_run,
            source.semantic_run,
            verify_artifacts=verify_artifacts,
        )
        cell_by_section = {str(row["section"]): row for row in bindings.cells["slides"]}
        sections = tuple(
            section
            for section in target_sections(
                bindings.stain,
                config.target.target_id,
                require_threshold=bool(config.target.binary_enabled),
                minimum_sections=0,
            )
            if section in cell_by_section
        )
        preflight_path = source.cell_run / str(bindings.cells["preflight"])
        from histopia.cells._result import validate_cell_artifact

        cell_artifacts = bindings.cells.get("artifacts")
        if not isinstance(cell_artifacts, dict):
            raise ValueError("cell result artifact manifest is missing")
        validate_cell_artifact(
            preflight_path,
            str(cell_artifacts[str(bindings.cells["preflight"])]),
        )
        preflight = json.loads(preflight_path.read_text())
        registration_slides = bindings.registration["slides"]
        reference_id = bindings.registration["reference_slide"]
        contexts[mouse_id] = _Context(
            mouse_id=mouse_id,
            source=source,
            bindings=bindings,
            target_sections=sections,
            stain_by_slide={str(row["id"]): row for row in bindings.stain["slides"]},
            semantic_by_slide={
                str(row["id"]): row for row in bindings.semantic["slides"]
            },
            cell_by_section=cell_by_section,
            preflight_by_slide={
                str(row["slide_name"]): row for row in preflight["slides"]
            },
            registration_by_slide={
                Path(str(row["path"])).name: row for row in registration_slides
            },
            reference_slide=next(
                row for row in registration_slides if str(row["path"]) == reference_id
            ),
            region_count=int(bindings.semantic["selected_k"]),
        )
    return contexts


def _section_design(
    context: _Context,
    section: str,
    config: object,
    *,
    geometry_root: Path,
    include_semantic: bool = True,
) -> _Design:
    from histopia.protein._cell_features import (
        CellTokenFeatures,
        label_centroids_from_tiff,
    )
    from histopia.protein._real_data import map_native_to_reference
    from histopia.protein._real_workflow import (
        _cell_context_features,
        _semantic_regions_for_cells,
        _standardize_patch_features,
    )
    from histopia.protein._transfer import morphospatial_features

    cell_row = context.cell_by_section[section]
    token_path = context.source.feature_dir / f"{section}.npz"
    token = CellTokenFeatures.load(token_path)
    if token.provenance.get("cell_result_fingerprint") != context.bindings.cells.get(
        "fingerprint"
    ):
        raise ValueError("neutral cell features are bound to a different cell run")
    cache = geometry_root / context.mouse_id / f"{section}.npz"
    cache_key = {
        "schema_version": 1,
        "registration_result_sha256": context.bindings.cells[
            "registration_result_sha256"
        ],
        "cell_result_fingerprint": context.bindings.cells["fingerprint"],
        "token_fingerprint": token.fingerprint,
    }
    cached = False
    if cache.is_file():
        with np.load(cache, allow_pickle=False) as data:
            metadata = (
                json.loads(str(data["metadata_json"]))
                if "metadata_json" in data.files
                else None
            )
            if metadata == cache_key:
                present = data["present"]
                native_xy = data["native_xy"]
                reference_xy = data["reference_um_xy"]
                counts = data["counts"]
                cached = True
    if not cached:
        from histopia.cells._result import validate_cell_artifact

        label_relative = str(cell_row["labels"])
        cell_artifacts = context.bindings.cells.get("artifacts")
        if not isinstance(cell_artifacts, dict):
            raise ValueError("cell result artifact manifest is missing")
        validate_cell_artifact(
            context.source.cell_run / label_relative,
            str(cell_artifacts[label_relative]),
        )
        area_ids, _centroids, counts = label_centroids_from_tiff(
            context.source.cell_run / label_relative
        )
        if not np.array_equal(area_ids, token.label_ids):
            raise ValueError("cell geometry differs from neutral token labels")
        present = token.label_ids
        native_xy = token.native_xy
        registration_row = context.registration_by_slide[
            Path(str(cell_row["slide"])).name
        ]
        reference_xy = map_native_to_reference(
            native_xy,
            registration_row,
            context.reference_slide,
        )
        from histopia._atomic import write_binary_atomic

        write_binary_atomic(
            cache,
            lambda stream: np.savez_compressed(
                stream,
                metadata_json=np.asarray(json.dumps(cache_key, sort_keys=True)),
                present=present,
                native_xy=native_xy,
                reference_um_xy=reference_xy,
                counts=counts,
            ),
        )
    if not np.array_equal(present, token.label_ids) or not np.allclose(
        native_xy,
        token.native_xy,
        rtol=0,
        atol=1e-6,
    ):
        raise ValueError("cached cell geometry differs from neutral token features")
    geometry = {
        "present": present,
        "native_xy": native_xy,
        "reference_um_xy": reference_xy,
        "counts": counts,
    }
    xyz = np.column_stack(
        (
            reference_xy,
            np.full(len(present), int(section) * float(config.section_spacing_um)),
        )
    )
    feature_view = str(
        getattr(
            config,
            "feature_schema_id",
            "native-hdab-neutral-spatial-uni2h-v2",
        )
    )
    if feature_view == "native-hdab-neutral-cell-multiscale-v3":
        from histopia.protein._multiscale import (
            CellFeatureSet,
            build_cell_feature_set,
            stream_cell_phenotype_features,
        )

        feature_cache = (
            geometry_root / context.mouse_id / "multiscale-v3" / f"{section}.npz"
        )
        provenance = {
            "schema_version": 1,
            "registration_result_sha256": context.bindings.cells[
                "registration_result_sha256"
            ],
            "cell_result_fingerprint": context.bindings.cells["fingerprint"],
            "source_identity": str(cell_row["source_identity"]),
            "token_fingerprint": token.fingerprint,
            "neighborhood_radii_um": list(config.neighborhood_radii_um),
            "coordinate_scale_um": float(config.coordinate_scale_um),
        }
        feature_artifact: CellFeatureSet | None = None
        if feature_cache.is_file():
            loaded = CellFeatureSet.load(feature_cache)
            if loaded.provenance == {
                "feature_schema_id": feature_view,
                **provenance,
            }:
                feature_artifact = loaded
        if feature_artifact is None:
            registration_row = context.registration_by_slide[
                Path(str(cell_row["slide"])).name
            ]
            geometry_metadata = dict(registration_row.get("geometry", {}))
            source_mpp = cell_row.get("mpp_xy", geometry_metadata.get("mpp_xy"))
            if not isinstance(source_mpp, list | tuple) or len(source_mpp) != 2:
                raise ValueError("multiscale features require native pixel calibration")
            raw_bbox = geometry_metadata.get("content_bbox_xywh")
            if not isinstance(raw_bbox, list | tuple) or len(raw_bbox) != 4:
                raise ValueError("multiscale features require a native content box")
            phenotype = stream_cell_phenotype_features(
                context.source.cell_run / str(cell_row["labels"]),
                Path(str(registration_row["path"])),
                pixel_size_um_xy=(float(source_mpp[0]), float(source_mpp[1])),
                label_ids=present,
                content_bbox_native_xywh=tuple(int(value) for value in raw_bbox),
            )
            feature_artifact = build_cell_feature_set(
                slide_id=f"{context.mouse_id}-{section}",
                label_ids=present,
                native_xy=native_xy,
                reference_um_xyz=xyz,
                morphology=token.features,
                phenotype=phenotype,
                supported=token.supported,
                radii_um=tuple(float(value) for value in config.neighborhood_radii_um),
                coordinate_scale_um=float(config.coordinate_scale_um),
                provenance=provenance,
            )
            feature_artifact.save(feature_cache)
        if (
            not np.array_equal(feature_artifact.label_ids, present)
            or not np.allclose(feature_artifact.native_xy, native_xy, atol=1e-6)
            or not np.allclose(feature_artifact.reference_um_xyz, xyz, atol=1e-6)
        ):
            raise ValueError("multiscale features differ from current cell geometry")
        features = feature_artifact.features
    elif feature_view == "native-hdab-neutral-spatial-uni2h-v2":
        shape, neighborhood = _cell_context_features(
            geometry,
            float(config.section_spacing_um),
        )
        features = morphospatial_features(
            _standardize_patch_features(token.features),
            xyz,
            shape_features=shape,
            neighborhood_features=neighborhood,
            coordinate_scale_um=float(config.coordinate_scale_um),
        )
    else:
        raise ValueError(f"unsupported protein feature schema: {feature_view}")
    regions = (
        _semantic_regions_for_cells(
            context.source.semantic_run,
            context.semantic_by_slide,
            cell_row_by_section=context.cell_by_section,
            section=section,
            cell_native_xy=native_xy,
            region_count=context.region_count,
        )
        if include_semantic
        else np.zeros(len(present), dtype=np.int16)
    )
    return _Design(
        label_ids=np.asarray(present, dtype=np.uint32),
        native_xy=np.asarray(native_xy, dtype=np.float64),
        reference_um_xy=np.asarray(reference_xy, dtype=np.float64),
        reference_um_xyz=np.asarray(xyz, dtype=np.float64),
        features=np.asarray(features, dtype=np.float32),
        supported=np.asarray(token.supported, dtype=bool),
        regions=np.asarray(regions, dtype=np.int16),
    )


def _full_section_measurement(
    context: _Context,
    section: str,
    design: _Design,
    *,
    minimum_effective_pixels: float,
    minimum_coverage: float,
    compartment: str,
    statistic: str = "mean",
    calibration: object | None = None,
) -> np.ndarray:
    from histopia.protein._real_data import (
        aggregate_map_to_sampled_labels,
        compartment_sampled_labels,
        sampled_label_expected_analysis_pixels,
        sampled_label_geometry,
        validated_adaptive_target_measurement,
    )
    from histopia.stain._artifacts import AdaptiveStainMap, StainMap

    measured = np.full(len(design.label_ids), np.nan, dtype=np.float32)
    if section not in context.target_sections:
        return measured
    cell_row = context.cell_by_section[section]
    from histopia.cells._result import validate_cell_artifact

    label_relative = str(cell_row["labels"])
    cell_artifacts = context.bindings.cells.get("artifacts")
    if not isinstance(cell_artifacts, dict):
        raise ValueError("cell result artifact manifest is missing")
    validate_cell_artifact(
        context.source.cell_run / label_relative,
        str(cell_artifacts[label_relative]),
    )
    sampled, _present, _xy, _counts = sampled_label_geometry(
        context.source.cell_run / label_relative,
        subifd=2,
        cell_count=int(cell_row["cell_count"]),
    )
    stain_row = context.stain_by_slide[str(cell_row["slide"])]
    from histopia.stain._result_validation import validate_stain_artifact

    stain_artifacts = context.bindings.stain.get("artifacts")
    if not isinstance(stain_artifacts, dict):
        raise ValueError("stain result artifact manifest is missing")
    map_relative = str(stain_row["map"])
    map_path = validate_stain_artifact(
        context.source.stain_run / map_relative,
        str(stain_artifacts[map_relative]),
    )
    stain_map = StainMap.load(map_path)
    adaptive_relative = stain_row.get("adaptive_map")
    adaptive_map = (
        AdaptiveStainMap.load(
            validate_stain_artifact(
                context.source.stain_run / adaptive_relative,
                str(stain_artifacts[adaptive_relative]),
            )
        )
        if isinstance(adaptive_relative, str)
        else None
    )
    measurement = validated_adaptive_target_measurement(
        stain_map,
        stain_row,
        adaptive_stain_map=adaptive_map,
    )
    sampled = compartment_sampled_labels(
        sampled,
        compartment,
        source_mpp_xy=stain_map.source_mpp_xy,
        pyramid_scale=4.0,
    )
    mean, effective, _fraction = aggregate_map_to_sampled_labels(
        sampled,
        measurement.target_od,
        measurement.tissue_mask,
        measurement.positive_mask,
        source_mpp_xy=stain_map.source_mpp_xy,
        analysis_mpp=float(stain_map.analysis_mpp),
        content_origin_native_xy=stain_map.content_origin_native_xy,
        pyramid_scale=4.0,
        cell_count=int(np.max(design.label_ids, initial=0)),
        statistic=statistic,
    )
    expected = sampled_label_expected_analysis_pixels(
        sampled,
        cell_count=int(np.max(design.label_ids, initial=0)),
        source_mpp_xy=stain_map.source_mpp_xy,
        pyramid_scale=4.0,
        analysis_mpp=float(stain_map.analysis_mpp),
    )
    coverage = np.divide(
        effective,
        expected,
        out=np.zeros_like(effective),
        where=expected > 0,
    )
    ids = design.label_ids.astype(np.int64)
    accepted = (effective[ids] >= minimum_effective_pixels) & (
        coverage[ids] >= minimum_coverage
    )
    values = mean[ids[accepted]]
    if calibration is not None:
        values = calibration.apply(values)
    measured[accepted] = values
    return measured


def _validate_study_od_calibration(
    raw: object,
) -> dict[tuple[str, str], object]:
    """Validate and load the path-free cohort-aware assay harmonization."""

    if not isinstance(raw, dict):
        raise ValueError("harmonized study table requires OD calibration")
    if raw.get("schema_version") != 2 or raw.get("method") != (
        "equal-section-adaptive-quantile-v2"
    ):
        raise ValueError("study OD calibration schema is unsupported")
    fingerprint = raw.get("fingerprint")
    sections = raw.get("sections")
    if (
        not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in fingerprint)
        or not isinstance(sections, list)
        or not sections
    ):
        raise ValueError("study OD calibration identity is incomplete")
    core = {key: value for key, value in raw.items() if key != "fingerprint"}
    if _fingerprint_json(core) != fingerprint:
        raise ValueError("study OD calibration fingerprint does not match")

    from histopia.protein._calibration import SectionOdCalibration

    loaded: dict[tuple[str, str], object] = {}
    for value in sections:
        if not isinstance(value, dict):
            raise ValueError("study OD calibration sections must be objects")
        cohort = value.get("cohort")
        section = value.get("section")
        if (
            not isinstance(cohort, str)
            or not cohort
            or not isinstance(section, str)
            or not section
            or (cohort, section) in loaded
        ):
            raise ValueError("study OD calibration section identity is invalid")
        calibration = SectionOdCalibration(
            section=section,
            source_knots=np.asarray(value.get("source_knots"), dtype=np.float64),
            reference_knots=np.asarray(value.get("reference_knots"), dtype=np.float64),
            anchor_count=int(value.get("anchor_count", -1)),
            status=str(value.get("status", "")),
            scale=float(value.get("scale", float("nan"))),
        )
        loaded[(cohort, section)] = calibration
    return loaded


def _shared_auxiliary_training(
    tables: tuple[object, ...],
    *,
    held_out: str | None,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...]:
    """Return auxiliary rows with the primary fold's mouse removed."""

    output = []
    for table in tables:
        mice = np.asarray(table.mouse_ids, dtype=str)
        indices = (
            np.flatnonzero(mice != held_out)
            if held_out is not None
            else np.arange(len(mice), dtype=np.int64)
        )
        output.append((table.features, table.measured_od, indices))
    return tuple(output)


def _fit_study_candidates(
    architecture: str,
    config: object,
    table: object,
    *,
    all_training: np.ndarray,
    available_mice: tuple[str, ...],
    training_mice: tuple[str, ...],
    mouse_values: np.ndarray,
    root: Path,
    model_root: Path,
    device: str,
    epochs: int | None,
    auxiliary_tables: tuple[object, ...],
) -> tuple[
    dict[str, _Candidate],
    _Candidate,
    list[str],
    list[dict[str, object]],
]:
    """Fit each unique fold and the final model exactly once."""

    fitted: dict[str, _Candidate] = {}
    model_artifacts: list[str] = []
    fold_metrics: list[dict[str, object]] = []
    for held_out in available_mice:
        if held_out not in training_mice or len(training_mice) <= 1:
            continue
        train = all_training[mouse_values[all_training] != held_out]
        candidate = _fit_candidate(
            architecture,
            config,
            table,
            train,
            seed=int(config.seed) + int(held_out),
            epochs=epochs,
            auxiliary=_shared_auxiliary_training(
                auxiliary_tables,
                held_out=held_out,
            ),
            auxiliary_target_ids=tuple(value.target_id for value in auxiliary_tables),
        )
        fitted[held_out] = candidate
        model_paths = _save_candidate(
            candidate,
            model_root / f"leave-one-mouse-out-{held_out}",
        )
        model_artifacts.extend(
            path.relative_to(root).as_posix() for path in model_paths
        )
        test = np.flatnonzero(mouse_values == held_out)
        if not len(test):
            continue
        predicted = _predict_table_candidate(
            candidate,
            table,
            train,
            test,
            config,
            device=device,
        )
        from histopia.protein._cli import _compare_with_baseline
        from histopia.protein._model import evaluate_predictions

        probability = (
            candidate.predict(table.features[test], device=device)[0]
            if architecture == "hurdle_mlp"
            else None
        )
        base_metrics = evaluate_predictions(
            table.measured_od[test],
            predicted,
            binary_label=table.binary_label[test],
            probability=probability,
        )
        metrics = _compare_with_baseline(
            base_metrics,
            table,
            train,
            test,
            predicted,
        )
        fold_metrics.append({"held_out": held_out, **metrics})

    final_candidate = _fit_candidate(
        architecture,
        config,
        table,
        all_training,
        seed=int(config.seed),
        epochs=epochs,
        auxiliary=_shared_auxiliary_training(auxiliary_tables, held_out=None),
        auxiliary_target_ids=tuple(value.target_id for value in auxiliary_tables),
    )
    final_paths = _save_candidate(final_candidate, model_root / "final")
    model_artifacts.extend(path.relative_to(root).as_posix() for path in final_paths)
    for mouse_id in available_mice:
        fitted.setdefault(mouse_id, final_candidate)
    return fitted, final_candidate, model_artifacts, fold_metrics


def _resume_partial_study_candidates(
    architecture: str,
    config: object,
    table: object,
    *,
    all_training: np.ndarray,
    available_mice: tuple[str, ...],
    training_mice: tuple[str, ...],
    mouse_values: np.ndarray,
    root: Path,
    model_root: Path,
    device: str,
    epochs: int | None,
    prediction_protocol: str,
    measurement_view: str,
    feature_view: str,
    auxiliary_bindings: dict[str, str] | None = None,
    prediction_mice: tuple[str, ...] | None = None,
) -> (
    tuple[
        dict[str, _Candidate],
        _Candidate,
        list[str],
        list[dict[str, object]],
    ]
    | None
):
    """Resume sealed model weights when prediction streaming was interrupted."""

    auxiliary_bindings = auxiliary_bindings or {}
    bundle_path = root / "model_bundle.json"
    if not bundle_path.is_file():
        return None
    bundle = json.loads(bundle_path.read_text())
    fingerprint = bundle.get("fingerprint")
    if (
        not isinstance(fingerprint, str)
        or _fingerprint_json(
            {key: value for key, value in bundle.items() if key != "fingerprint"}
        )
        != fingerprint
    ):
        raise ValueError("partial protein model bundle fingerprint is stale")
    expected = {
        "schema_version": 1,
        "target_id": table.target_id,
        "architecture": architecture,
        "prediction_protocol": prediction_protocol,
        "training_cohorts": list(training_mice),
        "table_fingerprint": table.fingerprint,
        "feature_schema_id": feature_view,
        "measurement_view": measurement_view,
        "measurement_compartment": config.target.compartment,
        "measurement_statistic": config.target.measurement_statistic,
        "training_epochs": epochs or _default_training_epochs(architecture),
    }
    relational_loss_profile = str(
        getattr(config, "relational_loss_profile", "huber_v1")
    )
    if relational_loss_profile != "huber_v1":
        expected["relational_loss_profile"] = relational_loss_profile
    if hasattr(table, "mouse_ids") and hasattr(table, "measured_od"):
        measured_mice = set(
            np.asarray(table.mouse_ids, dtype=str)[
                np.isfinite(np.asarray(table.measured_od, dtype=np.float64))
            ]
        )
        external_holdout_mice = tuple(sorted(measured_mice - set(training_mice)))
        if external_holdout_mice:
            expected["external_holdout_cohorts"] = list(external_holdout_mice)
    if architecture == "shared_multitask" or auxiliary_bindings:
        expected["auxiliary_table_fingerprints"] = auxiliary_bindings
    if prediction_mice is not None:
        expected["prediction_cohorts"] = list(prediction_mice)
    if any(bundle.get(key) != value for key, value in expected.items()):
        raise ValueError("partial protein model bundle differs from resumed run")
    fold_fingerprints = bundle.get("fold_fingerprints")
    if not isinstance(fold_fingerprints, dict):
        raise ValueError("partial protein fold identities are missing")
    artifacts: list[str] = []
    fitted: dict[str, _Candidate] = {}
    fold_metrics: list[dict[str, object]] = []
    if len(training_mice) > 1:
        for held_out in training_mice:
            stem = model_root / f"leave-one-mouse-out-{held_out}"
            candidate = _load_candidate(stem, architecture)
            if candidate.fingerprint != fold_fingerprints.get(held_out):
                raise ValueError("partial protein fold fingerprint differs")
            fitted[held_out] = candidate
            artifacts.extend(
                path.relative_to(root).as_posix()
                for path in _candidate_artifact_paths(stem, architecture)
            )
            train = all_training[mouse_values[all_training] != held_out]
            test = np.flatnonzero(mouse_values == held_out)
            if not len(test):
                continue
            predicted = _predict_table_candidate(
                candidate,
                table,
                train,
                test,
                config,
                device=device,
            )
            from histopia.protein._cli import _compare_with_baseline
            from histopia.protein._model import evaluate_predictions

            probability = (
                candidate.predict(table.features[test], device=device)[0]
                if architecture == "hurdle_mlp"
                else None
            )
            base_metrics = evaluate_predictions(
                table.measured_od[test],
                predicted,
                binary_label=table.binary_label[test],
                probability=probability,
            )
            metrics = _compare_with_baseline(
                base_metrics,
                table,
                train,
                test,
                predicted,
            )
            fold_metrics.append({"held_out": held_out, **metrics})
    final_stem = model_root / "final"
    final_candidate = _load_candidate(final_stem, architecture)
    if final_candidate.fingerprint != bundle.get("final_fingerprint"):
        raise ValueError("partial final protein model fingerprint differs")
    artifacts.extend(
        path.relative_to(root).as_posix()
        for path in _candidate_artifact_paths(final_stem, architecture)
    )
    for mouse_id in available_mice:
        fitted.setdefault(mouse_id, final_candidate)
    return fitted, final_candidate, artifacts, fold_metrics


def _load_morphology_transfer_refinement(
    evidence_path: Path | str | None,
    *,
    source_run: Path | str | None,
    architecture: str,
    target_id: str,
    weight: float,
    neighbors: int,
) -> tuple[dict[str, object] | None, bool]:
    """Validate a frozen development/confirmation decision for model reuse.

    The evidence binds the unchanged portable neural model by both its internal
    fingerprint and file digest.  Only the post-calibration training-bank
    transfer is added; fitting is never repeated and the evidence path itself
    is not exposed in portable metadata.
    """

    if evidence_path is None:
        if weight and not _source_embeds_morphology_transfer(
            source_run,
            architecture=architecture,
            weight=weight,
            neighbors=neighbors,
        ):
            raise ValueError(
                "reused morphology transfer requires frozen refinement evidence"
            )
        return None, False
    if source_run is None:
        raise ValueError("post-fit refinement requires a sealed reused model")
    if architecture not in {
        "dual_bank_attention",
        "graph_transformer",
        "multi_tower",
    }:
        raise ValueError(
            "morphology-transfer refinement requires a compatible neural model"
        )
    if not math.isfinite(weight) or not 0 < weight <= 1 or neighbors < 1:
        raise ValueError("morphology-transfer refinement controls are invalid")

    from histopia.protein._advanced import PortableMultiTowerRegressor
    from histopia.protein._manifest import (
        validate_protein_artifact,
        validate_protein_result_index,
    )
    from histopia.protein._relational import PortableRelationalRegressor

    source_root = Path(source_run).expanduser().resolve()
    source_result = validate_protein_result_index(source_root)
    evidence_source = Path(evidence_path).expanduser().resolve()
    evidence = json.loads(evidence_source.read_text())
    method = evidence.get("method") if isinstance(evidence, dict) else None
    if (
        not isinstance(evidence, dict)
        or evidence.get("schema_version") != 1
        or evidence.get("target_id") != target_id
        or method not in _MORPHOLOGY_TRANSFER_METHODS
    ):
        raise ValueError("morphology-transfer refinement evidence is incompatible")
    policy = evidence.get("development_policy")
    gate = evidence.get("confirmation_gate")
    rows = evidence.get("cohorts")
    if (
        not isinstance(policy, dict)
        or float(policy.get("weight", -1)) != weight
        or int(policy.get("neighbors", -1)) != neighbors
        or not isinstance(gate, dict)
        or gate.get("accepted") is not True
        or not isinstance(rows, list)
        or not rows
    ):
        raise ValueError("morphology-transfer refinement was not frozen and accepted")
    development = sorted(
        str(row.get("mouse"))
        for row in rows
        if isinstance(row, dict) and row.get("role") == "development"
    )
    confirmation = sorted(
        str(row.get("mouse"))
        for row in rows
        if isinstance(row, dict) and row.get("role") == "untouched_confirmation"
    )
    if not development or not confirmation or set(development) & set(confirmation):
        raise ValueError("morphology-transfer evidence lacks disjoint validation roles")
    evidence_results = sorted(
        str(row.get("result_fingerprint"))
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("result_fingerprint"), str)
    )
    source_fingerprint = str(source_result.get("fingerprint", ""))
    reuse = source_result.get("model_reuse")
    source_lineage = {source_fingerprint}
    if isinstance(reuse, dict):
        source_lineage.add(str(reuse.get("source_result_fingerprint", "")))
    if source_lineage.isdisjoint(evidence_results):
        raise ValueError("refinement evidence does not bind the reused result lineage")

    normalized = {
        "schema_version": 1,
        "method": str(method),
        "weight": weight,
        "neighbors": neighbors,
        "evidence_sha256": _sha256_file(evidence_source),
        "base_model_fingerprint": str(evidence.get("model_fingerprint", "")),
        "base_model_sha256": str(evidence.get("model_sha256", "")),
        "development_cohorts": development,
        "confirmation_cohorts": confirmation,
        "confirmation_gate": "accepted-once-after-development-freeze",
        "evidence_source_result_fingerprints": evidence_results,
    }

    # A sealed source may already contain the accepted post-fit transfer bank.
    # Recomputing that bank is not byte stable across BLAS/GPU runtimes because
    # its normalized float32 projection can differ by a few ULPs.  In that case
    # copy the fingerprint-bound artifacts exactly and propagate their already
    # validated refinement provenance instead of applying the transform twice.
    source_refinement = source_result.get("postfit_refinement")
    if source_refinement is not None:
        if source_refinement != normalized:
            raise ValueError("reused model post-fit refinement evidence differs")
        return normalized, True

    artifacts = source_result.get("artifacts")
    final_relative = "models/final.npz"
    if not isinstance(artifacts, dict) or not isinstance(
        artifacts.get(final_relative), str
    ):
        raise ValueError("reused relational final model is missing")
    model_path = validate_protein_artifact(
        source_root / final_relative,
        str(artifacts[final_relative]),
    )
    model = (
        PortableMultiTowerRegressor.load(model_path)
        if architecture == "multi_tower"
        else PortableRelationalRegressor.load(model_path)
    )
    model_sha256 = _sha256_file(model_path)
    if (
        evidence.get("model_fingerprint") != model.fingerprint
        or evidence.get("model_sha256") != model_sha256
    ):
        raise ValueError("refinement evidence differs from the reused neural model")

    if architecture == "multi_tower" and method == _MORPHOLOGY_TRANSFER_METHOD_V1:
        policy_protocol_sha = policy.get("protocol_sha256")
        protocol_path = evidence_source.parent / f"{target_id}-refinement-protocol.json"
        if (
            not isinstance(policy_protocol_sha, str)
            or not protocol_path.is_file()
            or _sha256_file(protocol_path) != policy_protocol_sha
        ):
            raise ValueError("multi-tower refinement protocol binding is unavailable")
        refinement_protocol = json.loads(protocol_path.read_text())
        training_table_relative = source_result.get("training_table")
        training_cohorts = source_result.get("training_cohorts")
        morphology_slice = model.group_slices.get("morphology")
        if (
            not isinstance(refinement_protocol, dict)
            or refinement_protocol.get("schema_version") != 1
            or refinement_protocol.get("target_id") != target_id
            or refinement_protocol.get("method") != evidence.get("method")
            or refinement_protocol.get("architecture") != architecture
            or refinement_protocol.get("model_fingerprint") != model.fingerprint
            or refinement_protocol.get("model_sha256") != model_sha256
            or not isinstance(training_table_relative, str)
            or not isinstance(training_cohorts, list)
            or not training_cohorts
            or morphology_slice is None
        ):
            raise ValueError("multi-tower refinement protocol is incompatible")
        training_table_path = validate_protein_artifact(
            source_root / training_table_relative,
            str(artifacts.get(training_table_relative, "")),
        )
        from histopia.protein._result import CellExpressionTable

        training_table = CellExpressionTable.load(training_table_path)
        if (
            refinement_protocol.get("source_result_fingerprint") not in source_lineage
            or refinement_protocol.get("training_table_sha256")
            != _sha256_file(training_table_path)
            or refinement_protocol.get("training_table_fingerprint")
            != training_table.fingerprint
            or refinement_protocol.get("transfer_training_cohorts") != training_cohorts
            or refinement_protocol.get("transfer_morphology_slice")
            != list(morphology_slice)
        ):
            raise ValueError("multi-tower refinement training bank differs")

    if method == _MORPHOLOGY_TRANSFER_METHOD_V2:
        _validate_expanded_morphology_transfer_evidence(
            evidence,
            source_result=source_result,
            source_root=source_root,
            architecture=architecture,
            model=model,
        )

    return normalized, False


def _validate_expanded_morphology_transfer_evidence(
    evidence: dict[str, object],
    *,
    source_result: dict[str, object],
    source_root: Path,
    architecture: str,
    model: object,
) -> None:
    """Bind expanded prospective evidence to its exact frozen outcome bank."""

    training_cohorts = source_result.get("training_cohorts")
    transfer_bank = evidence.get("transfer_bank")
    policy = evidence.get("development_policy")
    if (
        not isinstance(training_cohorts, list)
        or not training_cohorts
        or evidence.get("training_cohorts") != training_cohorts
        or not isinstance(transfer_bank, dict)
        or not isinstance(policy, dict)
        or policy.get("selection_scope")
        != "expanded-development-before-4714-8567-outcomes-v1"
        or not isinstance(policy.get("selection_fingerprint"), str)
        or not isinstance(policy.get("selection_sha256"), str)
    ):
        raise ValueError("expanded morphology-transfer evidence is incomplete")
    for key in ("selection_fingerprint", "selection_sha256"):
        value = str(policy[key])
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError(
                "expanded morphology-transfer selection binding is invalid"
            )

    rows = transfer_bank.get("rows")
    if not isinstance(rows, int) or isinstance(rows, bool) or rows < 2:
        raise ValueError("expanded morphology-transfer bank size is invalid")
    model_fingerprint = str(getattr(model, "fingerprint", ""))
    if architecture in {"dual_bank_attention", "graph_transformer"}:
        bank_od = np.asarray(getattr(model, "bank_od", np.asarray([])))
        if transfer_bank != {
            "scope": "sealed-relational-training-bank-v1",
            "rows": len(bank_od),
            "transfer_model_fingerprint": model_fingerprint,
        }:
            raise ValueError("expanded relational transfer bank differs")
        return

    if architecture != "multi_tower":
        raise ValueError("expanded morphology transfer requires a compatible model")
    from histopia.protein._advanced import PortableMultiTowerRegressor
    from histopia.protein._manifest import validate_protein_artifact
    from histopia.protein._result import CellExpressionTable

    if not isinstance(model, PortableMultiTowerRegressor):
        raise ValueError("expanded multi-tower transfer model is incompatible")
    artifacts = source_result.get("artifacts")
    training_table_relative = source_result.get("training_table")
    morphology_slice = model.group_slices.get("morphology")
    if (
        not isinstance(artifacts, dict)
        or not isinstance(training_table_relative, str)
        or not isinstance(artifacts.get(training_table_relative), str)
        or morphology_slice is None
    ):
        raise ValueError("expanded multi-tower training bank is unavailable")
    training_table_path = validate_protein_artifact(
        source_root / training_table_relative,
        str(artifacts[training_table_relative]),
    )
    training_table = CellExpressionTable.load(training_table_path)
    mouse_ids = np.asarray(training_table.mouse_ids, dtype=str)
    measured_od = np.asarray(training_table.measured_od, dtype=np.float32)
    features = np.asarray(training_table.features, dtype=np.float32)
    selected = np.isin(
        mouse_ids, np.asarray(training_cohorts, dtype=str)
    ) & np.isfinite(measured_od)
    if set(mouse_ids[selected]) != set(training_cohorts):
        raise ValueError("expanded multi-tower training rows are incomplete")
    transfer_model = model.with_morphology_transfer_bank(
        features[selected],
        measured_od[selected],
    )
    if transfer_bank != {
        "scope": "fingerprinted-training-table-morphology-bank-v1",
        "rows": int(np.count_nonzero(selected)),
        "training_cohorts": training_cohorts,
        "training_table_fingerprint": training_table.fingerprint,
        "training_table_sha256": _sha256_file(training_table_path),
        "morphology_slice": list(morphology_slice),
        "transfer_model_fingerprint": transfer_model.fingerprint,
    }:
        raise ValueError("expanded multi-tower transfer bank differs")


def _source_embeds_morphology_transfer(
    source_run: Path | str | None,
    *,
    architecture: str,
    weight: float,
    neighbors: int,
) -> bool:
    """Prove that every reused candidate already seals the requested transfer.

    Relational candidates fitted with a nonzero morphology-transfer weight
    include that policy in their candidate fingerprints and calibration
    sidecars.  Reusing those exact artifacts is not a new post-fit refinement
    and therefore does not require a second evidence decision.  A transfer
    absent from any final or leave-one-mouse-out candidate remains fail-closed
    behind the explicit refinement-evidence path.
    """

    if source_run is None or architecture not in {
        "dual_bank_attention",
        "graph_transformer",
        "multi_tower",
    }:
        return False
    from histopia.protein._manifest import (
        validate_protein_artifact,
        validate_protein_result_index,
    )

    root = Path(source_run).expanduser().resolve()
    result = validate_protein_result_index(root)
    if result.get("architecture") != architecture:
        return False
    artifacts = result.get("artifacts")
    model_artifacts = result.get("model_artifacts")
    model_relative = result.get("model")
    training_cohorts = result.get("training_cohorts")
    if (
        not isinstance(artifacts, dict)
        or not isinstance(model_artifacts, list)
        or not isinstance(model_relative, str)
        or not isinstance(training_cohorts, list)
        or not training_cohorts
    ):
        return False
    bundle_path = validate_protein_artifact(
        root / model_relative,
        str(artifacts.get(model_relative, "")),
    )
    bundle = json.loads(bundle_path.read_text())
    if (
        not isinstance(bundle, dict)
        or bundle.get("fingerprint")
        != _fingerprint_json(
            {key: value for key, value in bundle.items() if key != "fingerprint"}
        )
        or bundle.get("training_cohorts") != training_cohorts
    ):
        return False
    expected = {"models/final.calibration.json": bundle.get("final_fingerprint")}
    if len(training_cohorts) > 1:
        folds = bundle.get("fold_fingerprints")
        if not isinstance(folds, dict):
            return False
        expected.update(
            {
                f"models/leave-one-mouse-out-{mouse}.calibration.json": folds.get(mouse)
                for mouse in training_cohorts
            }
        )
    declared = {str(value) for value in model_artifacts}
    calibrations: dict[str, dict[str, object]] = {}
    for relative in expected:
        if relative not in declared or not isinstance(artifacts.get(relative), str):
            return False
        path = validate_protein_artifact(root / relative, str(artifacts[relative]))
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            return False
        calibrations[relative] = payload
    return _embedded_transfer_controls_match(
        calibrations,
        expected,
        weight=weight,
        neighbors=neighbors,
    )


def _embedded_transfer_controls_match(
    calibrations: dict[str, dict[str, object]],
    expected_fingerprints: dict[str, object],
    *,
    weight: float,
    neighbors: int,
) -> bool:
    """Return whether all expected candidates seal identical transfer controls."""

    return set(calibrations) == set(expected_fingerprints) and all(
        payload.get("schema_version") == 1
        and payload.get("fingerprint") == expected_fingerprints[relative]
        and payload.get("morphology_transfer_weight") == weight
        and payload.get("morphology_transfer_neighbors") == neighbors
        for relative, payload in calibrations.items()
    )


def _refine_reused_candidate(
    candidate: _Candidate,
    refinement: dict[str, object],
    *,
    table: object | None = None,
    bank_mice: tuple[str, ...] | None = None,
) -> _Candidate:
    """Apply one fingerprinted post-calibration refinement without refitting."""

    weight = float(refinement["weight"])
    neighbors = int(refinement["neighbors"])
    if candidate.architecture not in {
        "dual_bank_attention",
        "graph_transformer",
        "multi_tower",
    }:
        raise ValueError(
            "morphology-transfer refinement requires a compatible neural model"
        )
    if candidate.morphology_transfer_weight:
        if (
            candidate.morphology_transfer_weight == weight
            and candidate.morphology_transfer_neighbors == neighbors
        ):
            return candidate
        raise ValueError("reused candidate already has a different post-fit refinement")
    estimator = candidate.estimator
    bank_scope: list[str] | None = None
    if candidate.architecture == "multi_tower":
        if table is None or not bank_mice:
            raise ValueError("multi-tower refinement requires a sealed training scope")
        from histopia.protein._advanced import PortableMultiTowerRegressor

        if not isinstance(estimator, PortableMultiTowerRegressor):
            raise ValueError("multi-tower refinement model is incompatible")
        mouse_ids = np.asarray(table.mouse_ids, dtype=str)
        measured_od = np.asarray(table.measured_od, dtype=np.float32)
        features = np.asarray(table.features, dtype=np.float32)
        selected = np.isin(mouse_ids, np.asarray(bank_mice, dtype=str)) & np.isfinite(
            measured_od
        )
        if (
            features.ndim != 2
            or features.shape[0] != len(mouse_ids)
            or measured_od.shape != (len(mouse_ids),)
            or set(mouse_ids[selected]) != set(bank_mice)
        ):
            raise ValueError("multi-tower refinement training rows are incomplete")
        estimator = estimator.with_morphology_transfer_bank(
            features[selected],
            measured_od[selected],
        )
        bank_scope = list(bank_mice)
    fingerprint = _fingerprint_json(
        {
            "schema_version": 1,
            "base_candidate_fingerprint": candidate.fingerprint,
            "method": refinement["method"],
            "weight": weight,
            "neighbors": neighbors,
            "evidence_sha256": refinement["evidence_sha256"],
            "transfer_estimator_fingerprint": getattr(estimator, "fingerprint", None),
            "transfer_training_cohorts": bank_scope,
        }
    )
    return _Candidate(
        architecture=candidate.architecture,
        estimator=estimator,
        source_knots=candidate.source_knots,
        target_knots=candidate.target_knots,
        reference_od=candidate.reference_od,
        fingerprint=fingerprint,
        morphology_transfer_weight=weight,
        morphology_transfer_neighbors=neighbors,
    )


def _validate_existing_postfit_refinement(
    candidate: _Candidate,
    refinement: dict[str, object],
) -> None:
    """Confirm that a copied candidate already contains the sealed refinement."""

    weight = float(refinement["weight"])
    neighbors = int(refinement["neighbors"])
    if (
        candidate.morphology_transfer_weight != weight
        or candidate.morphology_transfer_neighbors != neighbors
    ):
        raise ValueError("reused candidate post-fit refinement controls differ")
    if candidate.architecture == "multi_tower":
        from histopia.protein._advanced import PortableMultiTowerRegressor

        estimator = candidate.estimator
        if (
            not isinstance(estimator, PortableMultiTowerRegressor)
            or estimator.morphology_bank_key is None
            or estimator.morphology_bank_od is None
            or not len(estimator.morphology_bank_od)
        ):
            raise ValueError("reused candidate post-fit morphology bank is missing")


def _evaluate_reused_fold_candidates(
    fitted: dict[str, _Candidate],
    *,
    config: object,
    table: object,
    all_training: np.ndarray,
    training_mice: tuple[str, ...],
    mouse_values: np.ndarray,
    device: str,
) -> list[dict[str, object]]:
    """Recompute LOOCV metrics after a post-fit candidate transformation."""

    if len(training_mice) <= 1:
        return []
    from histopia.protein._cli import _compare_with_baseline
    from histopia.protein._model import evaluate_predictions

    output: list[dict[str, object]] = []
    for held_out in training_mice:
        test = np.flatnonzero(mouse_values == held_out)
        if not len(test):
            continue
        train = all_training[mouse_values[all_training] != held_out]
        predicted = _predict_table_candidate(
            fitted[held_out],
            table,
            train,
            test,
            config,
            device=device,
        )
        metrics = evaluate_predictions(
            table.measured_od[test],
            predicted,
            binary_label=table.binary_label[test],
            probability=None,
        )
        output.append(
            {
                "held_out": held_out,
                **_compare_with_baseline(
                    metrics,
                    table,
                    train,
                    test,
                    predicted,
                ),
            }
        )
    return output


def _reuse_study_candidates(
    source_run: Path | str,
    *,
    architecture: str,
    table: object,
    available_mice: tuple[str, ...],
    training_mice: tuple[str, ...],
    root: Path,
    model_root: Path,
    measurement_view: str,
    feature_view: str,
    cohort_bindings: dict[str, object],
    auxiliary_bindings: dict[str, str] | None = None,
    postfit_refinement: dict[str, object] | None = None,
    postfit_refinement_already_applied: bool = False,
) -> tuple[
    dict[str, _Candidate],
    _Candidate,
    list[str],
    list[dict[str, object]],
    str,
    int | None,
]:
    """Copy and load a sealed fit without repeating optimization.

    Reuse may add or replace external-holdout cohorts, but the complete
    training scope must remain identical: target, feature/measurement views,
    training-cohort upstream bindings, and every training-row array are bound
    before any model artifact is copied.
    """

    auxiliary_bindings = auxiliary_bindings or {}
    if postfit_refinement_already_applied and postfit_refinement is None:
        raise ValueError("already-applied post-fit refinement metadata is missing")
    from histopia.protein._manifest import (
        validate_protein_artifact,
        validate_protein_result_index,
    )

    source_root = Path(source_run).expanduser().resolve()
    if source_root == root.resolve():
        raise ValueError("reused protein model source and destination must differ")
    result = validate_protein_result_index(source_root)
    expected = {
        "schema_version": 4,
        "target_id": table.target_id,
        "architecture": architecture,
        "prediction_protocol": "leave-one-mouse-out",
        "training_cohorts": list(training_mice),
        "feature_schema_id": feature_view,
        "measurement_view": measurement_view,
    }
    if architecture == "shared_multitask" or auxiliary_bindings:
        expected["auxiliary_table_fingerprints"] = auxiliary_bindings
    if any(result.get(key) != value for key, value in expected.items()):
        raise ValueError("reused protein model scientific scope differs")
    artifacts = result["artifacts"]
    assert isinstance(artifacts, dict)

    source_bindings = result.get("cohort_bindings")
    if not isinstance(source_bindings, dict) or any(
        source_bindings.get(mouse_id) != cohort_bindings.get(mouse_id)
        for mouse_id in training_mice
    ):
        raise ValueError("reused protein model training-cohort bindings differ")

    source_table_fingerprint = result.get("table_fingerprint")
    if source_table_fingerprint != table.fingerprint:
        from histopia.protein._result import CellExpressionTable

        source_table_relative = result.get("training_table")
        if not isinstance(source_table_relative, str):
            raise ValueError("reused protein model training table is missing")
        source_table_path = validate_protein_artifact(
            source_root / source_table_relative,
            str(artifacts.get(source_table_relative, "")),
        )
        source_table = CellExpressionTable.load(source_table_path)
        if source_table.fingerprint != source_table_fingerprint:
            raise ValueError("reused protein model training table differs")
        try:
            source_training_fingerprint = _training_scope_fingerprint(
                source_table,
                training_mice,
            )
            destination_training_fingerprint = _training_scope_fingerprint(
                table,
                training_mice,
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError(
                "reused protein model training table cannot be compared"
            ) from error
        if source_training_fingerprint != destination_training_fingerprint:
            raise ValueError("reused protein model training rows differ")

    model_relative = str(result["model"])
    validate_protein_artifact(
        source_root / model_relative,
        str(artifacts[model_relative]),
    )
    bundle = json.loads((source_root / model_relative).read_text())
    bundle_fingerprint = bundle.get("fingerprint")
    if (
        not isinstance(bundle_fingerprint, str)
        or _fingerprint_json(
            {key: value for key, value in bundle.items() if key != "fingerprint"}
        )
        != bundle_fingerprint
    ):
        raise ValueError("reused protein model bundle fingerprint is stale")
    copied: list[str] = []
    raw_model_artifacts = result.get("model_artifacts")
    if not isinstance(raw_model_artifacts, list) or not raw_model_artifacts:
        raise ValueError("reused protein model artifacts are missing")
    for raw_relative in raw_model_artifacts:
        relative = str(raw_relative)
        source = validate_protein_artifact(
            source_root / relative,
            str(artifacts[relative]),
        )
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        copied.append(relative)

    final_stem = model_root / "final"
    final_candidate = _load_candidate(final_stem, architecture)
    if final_candidate.fingerprint != bundle.get("final_fingerprint"):
        raise ValueError("reused final protein model fingerprint differs")
    if postfit_refinement is not None:
        if postfit_refinement_already_applied:
            _validate_existing_postfit_refinement(
                final_candidate,
                postfit_refinement,
            )
        else:
            final_candidate = _refine_reused_candidate(
                final_candidate,
                postfit_refinement,
                table=table,
                bank_mice=training_mice,
            )
            _save_candidate(final_candidate, final_stem)
    fold_fingerprints = bundle.get("fold_fingerprints")
    if not isinstance(fold_fingerprints, dict):
        raise ValueError("reused protein fold identities are missing")
    fitted: dict[str, _Candidate] = {}
    if len(training_mice) > 1:
        for held_out in training_mice:
            stem = model_root / f"leave-one-mouse-out-{held_out}"
            candidate = _load_candidate(stem, architecture)
            if candidate.fingerprint != fold_fingerprints.get(held_out):
                raise ValueError("reused protein fold fingerprint differs")
            if postfit_refinement is not None:
                if postfit_refinement_already_applied:
                    _validate_existing_postfit_refinement(
                        candidate,
                        postfit_refinement,
                    )
                else:
                    candidate = _refine_reused_candidate(
                        candidate,
                        postfit_refinement,
                        table=table,
                        bank_mice=tuple(
                            mouse for mouse in training_mice if mouse != held_out
                        ),
                    )
                    _save_candidate(candidate, stem)
            fitted[held_out] = candidate
    for mouse_id in available_mice:
        fitted.setdefault(mouse_id, final_candidate)
    metrics = result.get("metrics")
    raw_folds = metrics.get("folds") if isinstance(metrics, dict) else None
    if not isinstance(raw_folds, list):
        raise ValueError("reused protein model held-out metrics are missing")
    folds = [dict(row) for row in raw_folds if isinstance(row, dict)]
    if len(folds) != len(raw_folds):
        raise ValueError("reused protein model held-out metrics are malformed")
    raw_epochs = bundle.get("training_epochs")
    training_epochs = int(raw_epochs) if isinstance(raw_epochs, int) else None
    return (
        fitted,
        final_candidate,
        copied,
        folds,
        str(result["fingerprint"]),
        training_epochs,
    )


def _training_scope_fingerprint(
    table: object,
    training_mice: tuple[str, ...],
) -> str:
    """Fingerprint canonical training rows independently of holdout cohorts."""

    from histopia.protein._result import _fingerprint, _table_arrays

    arrays = _table_arrays(table)  # type: ignore[arg-type]
    mouse_ids = arrays["mouse_ids"]
    selected = np.flatnonzero(np.isin(mouse_ids, training_mice))
    if not len(selected) or set(np.asarray(mouse_ids[selected], dtype=str)) != set(
        training_mice
    ):
        raise ValueError("protein training scope is incomplete")
    order = np.lexsort(
        (
            arrays["label_ids"][selected],
            arrays["section_ids"][selected],
            arrays["mouse_ids"][selected],
        )
    )
    canonical = selected[order]
    return _fingerprint(
        b"histopia-protein-training-scope-v1\0",
        {
            "target_id": table.target_id,
            "training_cohorts": list(training_mice),
        },
        {name: value[canonical] for name, value in arrays.items()},
    )


def _training_only_harmonization_references(
    raw_references: object,
    training_mice: tuple[str, ...],
) -> tuple[str, ...]:
    """Validate that an external-holdout OD scale contains no holdout data.

    A frozen reference scale may deliberately use a representative subset of
    the training mice. Requiring every training mouse would reject that valid
    design; the scientific invariant is that references are non-empty, unique,
    and drawn exclusively from the selected training cohort.
    """

    if (
        not isinstance(raw_references, list)
        or not raw_references
        or any(not isinstance(value, str) or not value for value in raw_references)
        or len(set(raw_references)) != len(raw_references)
        or set(raw_references) - set(training_mice)
    ):
        raise ValueError(
            "harmonized external holdout evaluation requires a reference "
            "scale fitted only from the selected training cohorts"
        )
    return tuple(sorted(raw_references))


def _study_section_sampling_seed(
    seed: int,
    target_id: str,
    mouse_id: str,
    section: str,
) -> int:
    """Derive a stable sampling seed unaffected by other study cohorts."""

    digest = hashlib.sha256(b"histopia-protein-study-section-sampling-v1\0")
    for value in (str(seed), target_id, mouse_id, section):
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
    return int.from_bytes(digest.digest()[:8], "little")


def _load_candidate(stem: Path, architecture: str) -> _Candidate:
    """Load one portable candidate and its sealed calibration sidecar."""

    if architecture == "hurdle_mlp":
        from histopia.protein._model import ProteinModel

        estimator = ProteinModel.load(stem.with_suffix(".npz"))
    elif architecture == "cross_attention":
        from histopia.protein._attention import PortableCrossAttentionRegressor

        estimator = PortableCrossAttentionRegressor.load(stem.with_suffix(".npz"))
    elif architecture == "multi_tower":
        from histopia.protein._advanced import PortableMultiTowerRegressor

        estimator = PortableMultiTowerRegressor.load(stem.with_suffix(".npz"))
    elif architecture == "shared_multitask":
        from histopia.protein._deep import PortableSharedMultitaskRegressor

        estimator = PortableSharedMultitaskRegressor.load(stem.with_suffix(".npz"))
    elif architecture in {"dual_bank_attention", "graph_transformer"}:
        from histopia.protein._relational import PortableRelationalRegressor

        estimator = PortableRelationalRegressor.load(stem.with_suffix(".npz"))
    else:
        estimator = _load_sklearn_estimator(stem.with_suffix(".joblib"))
    calibration = json.loads(stem.with_suffix(".calibration.json").read_text())
    if calibration.get("schema_version") != 1:
        raise ValueError("reused protein calibration schema is unsupported")
    candidate = _Candidate(
        architecture=architecture,
        estimator=estimator,
        source_knots=np.asarray(calibration.get("source_knots"), dtype=np.float64),
        target_knots=np.asarray(calibration.get("target_knots"), dtype=np.float64),
        reference_od=np.asarray(calibration.get("reference_od"), dtype=np.float32),
        fingerprint=str(calibration.get("fingerprint", "")),
        morphology_transfer_weight=float(
            calibration.get("morphology_transfer_weight", 0.0)
        ),
        morphology_transfer_neighbors=int(
            calibration.get("morphology_transfer_neighbors", 16)
        ),
    )
    if (
        candidate.source_knots.ndim != 1
        or candidate.source_knots.shape != candidate.target_knots.shape
        or not len(candidate.source_knots)
        or candidate.reference_od.ndim != 1
        or not len(candidate.reference_od)
        or not np.all(np.isfinite(candidate.source_knots))
        or not np.all(np.isfinite(candidate.target_knots))
        or not np.all(np.isfinite(candidate.reference_od))
    ):
        raise ValueError("reused protein calibration arrays are malformed")
    return candidate


def _load_sklearn_estimator(path: Path) -> object:
    """Load pickle-based trees only under their exact scikit-learn runtime."""

    import joblib
    from sklearn.exceptions import InconsistentVersionWarning

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        estimator = joblib.load(path)
    mismatch = next(
        (
            warning.message
            for warning in caught
            if isinstance(warning.message, InconsistentVersionWarning)
        ),
        None,
    )
    if mismatch is not None:
        raise ValueError(
            "scikit-learn model runtime differs from the sealed training runtime: "
            f"{mismatch.original_sklearn_version} required, "
            f"{mismatch.current_sklearn_version} active"
        )
    for warning in caught:
        warnings.warn(warning.message, warning.category, stacklevel=2)
    return estimator


def _fit_candidate(
    architecture: str,
    config: object,
    table: object,
    train: np.ndarray,
    *,
    seed: int,
    epochs: int | None = None,
    auxiliary: tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...] = (),
    auxiliary_target_ids: tuple[str, ...] = (),
) -> _Candidate:
    valid = np.asarray(train, dtype=np.int64)
    valid = valid[np.isfinite(table.measured_od[valid])]
    if len(valid) < 4:
        raise ValueError("protein candidate requires four measured training cells")
    if architecture == "hurdle_mlp":
        from histopia.protein._cli import _fit_model

        estimator = _fit_model(config, table, valid)
        _probability, _relative, raw, _uncertainty = estimator.predict(
            table.features[valid]
        )
        identity = str(estimator.fingerprint)
    elif architecture == "cross_attention":
        from histopia.protein._attention import fit_portable_cross_attention

        estimator = fit_portable_cross_attention(
            table.features,
            table.measured_od,
            valid,
            seed=seed,
            **({"epochs": epochs} if epochs is not None else {}),
        )
        raw, _uncertainty = estimator.predict_accelerated(table.features[valid])
        identity = str(estimator.fingerprint)
    elif architecture == "multi_tower":
        from histopia.protein._advanced import (
            feature_group_slices,
            fit_portable_multi_tower,
        )

        feature_view = str(table.provenance.get("feature_view", ""))
        groups = feature_group_slices(
            feature_view,
            table.features.shape[1],
            radii_count=len(
                getattr(config, "neighborhood_radii_um", (16, 32, 64, 128))
            ),
        )
        estimator = fit_portable_multi_tower(
            table.features,
            table.measured_od,
            valid,
            group_slices=groups,
            section_groups=np.asarray(table.section_ids),
            seed=seed,
            **({"epochs": epochs} if epochs is not None else {}),
        )
        raw, _uncertainty = estimator.predict_accelerated(table.features[valid])
        identity = str(estimator.fingerprint)
    elif architecture in {"dual_bank_attention", "graph_transformer"}:
        from histopia.protein._relational import fit_portable_relational

        estimator = fit_portable_relational(
            architecture,
            table.features,
            table.measured_od,
            valid,
            reference_um_xyz=_table_reference_um_xyz(table, config),
            seed=seed,
            loss_profile=str(getattr(config, "relational_loss_profile", "huber_v1")),
            **({"epochs": epochs} if epochs is not None else {}),
        )
        raw, _uncertainty = estimator.predict_training_accelerated()
        identity = str(estimator.fingerprint)
    elif architecture == "shared_multitask":
        from histopia.protein._deep import fit_portable_shared_multitask

        estimator = fit_portable_shared_multitask(
            table.features,
            table.measured_od,
            valid,
            auxiliary=auxiliary,
            auxiliary_target_ids=auxiliary_target_ids,
            seed=seed,
            **({"epochs": epochs} if epochs is not None else {}),
        )
        raw, _uncertainty = estimator.predict_accelerated(table.features[valid])
        identity = str(estimator.fingerprint)
    else:
        import sklearn
        from sklearn.ensemble import ExtraTreesRegressor

        estimator = ExtraTreesRegressor(
            n_estimators=192,
            min_samples_leaf=max(2, len(valid) // 2_000),
            max_features="sqrt",
            random_state=seed,
            n_jobs=-1,
        )
        estimator.fit(table.features[valid], table.measured_od[valid])
        raw = np.maximum(estimator.predict(table.features[valid]), 0)
        rows_digest = hashlib.sha256(
            np.ascontiguousarray(valid, dtype=np.int64).tobytes()
        ).hexdigest()
        identity = _fingerprint_json(
            {
                "architecture": architecture,
                "seed": seed,
                "rows_sha256": rows_digest,
                "table": table.fingerprint,
                "implementation": {
                    "scikit_learn": sklearn.__version__,
                    "numpy": np.__version__,
                },
            }
        )
    source_knots, target_knots = _fit_quantile_calibration(
        raw,
        table.measured_od[valid],
    )
    morphology_transfer_weight = (
        float(getattr(config, "relational_morphology_transfer_weight", 0.0))
        if architecture in {"dual_bank_attention", "graph_transformer"}
        else 0.0
    )
    morphology_transfer_neighbors = int(
        getattr(config, "relational_morphology_transfer_neighbors", 16)
    )
    fingerprint = _fingerprint_json(
        {
            "base": identity,
            "source_knots": source_knots.tolist(),
            "target_knots": target_knots.tolist(),
            "reference_od_sha256": hashlib.sha256(
                np.ascontiguousarray(
                    np.sort(table.measured_od[valid]).astype(np.float32)
                ).tobytes()
            ).hexdigest(),
            "morphology_transfer_weight": morphology_transfer_weight,
            "morphology_transfer_neighbors": morphology_transfer_neighbors,
        }
    )
    return _Candidate(
        architecture=architecture,
        estimator=estimator,
        source_knots=source_knots,
        target_knots=target_knots,
        reference_od=np.sort(table.measured_od[valid]).astype(np.float32),
        fingerprint=fingerprint,
        morphology_transfer_weight=morphology_transfer_weight,
        morphology_transfer_neighbors=morphology_transfer_neighbors,
    )


def _predict_table_candidate(
    candidate: _Candidate,
    table: object,
    train: np.ndarray,
    test: np.ndarray,
    config: object,
    *,
    device: str,
) -> np.ndarray:
    _probability, _relative, predicted, _uncertainty = candidate.predict(
        table.features[test],
        reference_um_xyz=_table_reference_um_xyz(table, config)[test],
        device=device,
    )
    if (
        candidate.architecture == "cross_attention"
        and table.provenance.get("feature_view")
        == "native-hdab-neutral-spatial-uni2h-v2"
    ):
        from histopia.protein._cli import _region_centered_blend, _semantic_baseline

        semantic, _probability = _semantic_baseline(table, train, test)
        predicted = _region_centered_blend(
            predicted,
            semantic,
            np.asarray(table.semantic_region)[test],
            weight=float(config.semantic_blend_weight),
        )
    return predicted


def _table_reference_um_xyz(table: object, config: object) -> np.ndarray:
    """Return the registered physical cell coordinates used by relational models."""

    xy = np.asarray(table.reference_um_xy, dtype=np.float64)
    sections = np.asarray(table.section_ids, dtype=np.float64)
    xyz = np.column_stack((xy, sections * float(config.section_spacing_um)))
    if xyz.shape != (len(table.features), 3) or not np.all(np.isfinite(xyz)):
        raise ValueError("protein table registered coordinates are malformed")
    return xyz


def _default_training_epochs(architecture: str) -> int | None:
    return {
        "cross_attention": 40,
        "multi_tower": 80,
        "dual_bank_attention": 40,
        "graph_transformer": 40,
        "shared_multitask": 40,
    }.get(architecture)


def _study_runtime_metadata(architecture: str) -> dict[str, object] | None:
    """Record the exact runtime required by pickle-backed candidates."""

    if architecture != "extra_trees":
        return None
    import joblib
    import sklearn

    return {
        "serialization": "joblib-pickle-exact-runtime-v1",
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
        "numpy": np.__version__,
    }


def _save_candidate(candidate: _Candidate, stem: Path) -> tuple[Path, Path]:
    if candidate.architecture == "hurdle_mlp":
        path = candidate.estimator.save(stem.with_suffix(".npz"))
    elif candidate.architecture in {
        "cross_attention",
        "multi_tower",
        "dual_bank_attention",
        "graph_transformer",
        "shared_multitask",
    }:
        path = candidate.estimator.save(stem.with_suffix(".npz"))
    else:
        import joblib

        path = stem.with_suffix(".joblib")
        joblib.dump(candidate.estimator, path, compress=3)
    calibration = _write_candidate_calibration(candidate, stem)
    return Path(path), calibration


def _write_candidate_calibration(candidate: _Candidate, stem: Path) -> Path:
    """Write only the fingerprint-bound calibration/post-fit sidecar."""

    calibration = stem.with_suffix(".calibration.json")
    from histopia._atomic import write_json_atomic

    write_json_atomic(
        calibration,
        {
            "schema_version": 1,
            "fingerprint": candidate.fingerprint,
            "source_knots": candidate.source_knots.tolist(),
            "target_knots": candidate.target_knots.tolist(),
            "reference_od": candidate.reference_od.tolist(),
            "morphology_transfer_weight": candidate.morphology_transfer_weight,
            "morphology_transfer_neighbors": candidate.morphology_transfer_neighbors,
        },
    )
    return calibration


def _candidate_artifact_paths(
    stem: Path,
    architecture: str,
) -> tuple[Path, Path]:
    model_suffix = ".joblib" if architecture == "extra_trees" else ".npz"
    paths = (
        stem.with_suffix(model_suffix),
        stem.with_suffix(".calibration.json"),
    )
    if any(not path.is_file() for path in paths):
        raise ValueError("partial protein model artifacts are incomplete")
    return paths


def _predict_extra_trees(
    estimator: object,
    features: np.ndarray,
    *,
    batch_size: int = 100_000,
) -> tuple[np.ndarray, np.ndarray]:
    from joblib import Parallel, delayed

    output = np.empty(len(features), dtype=np.float32)
    uncertainty = np.empty(len(features), dtype=np.float32)
    trees = tuple(estimator.estimators_)
    for start in range(0, len(features), batch_size):
        stop = min(start + batch_size, len(features))
        chunk = np.ascontiguousarray(features[start:stop], dtype=np.float32)
        members = np.stack(
            Parallel(n_jobs=-1, prefer="threads")(
                delayed(tree.predict)(chunk, check_input=False) for tree in trees
            ),
            axis=0,
        )
        output[start:stop] = np.maximum(members.mean(axis=0), 0)
        uncertainty[start:stop] = members.std(axis=0)
    return output, uncertainty


def _fit_quantile_calibration(
    predicted: np.ndarray,
    measured: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(predicted) & np.isfinite(measured)
    if valid.sum() < 4:
        raise ValueError("OD calibration requires four finite training pairs")
    quantiles = np.asarray([0.0, 0.05, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0])
    source = np.maximum.accumulate(np.quantile(predicted[valid], quantiles))
    target = np.maximum.accumulate(np.quantile(measured[valid], quantiles))
    source[0] = target[0] = 0.0
    keep = np.r_[True, np.diff(source) > 1e-8]
    source, target = source[keep], target[keep]
    if len(source) < 2:
        source = np.asarray([0.0, max(float(np.max(predicted[valid])), 1e-6)])
        target = np.asarray([0.0, max(float(np.max(measured[valid])), 1e-6)])
    return source.astype(np.float32), target.astype(np.float32)


def _apply_quantile_calibration(
    values: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    raw = np.asarray(values, dtype=np.float64)
    mapped = np.interp(raw, source, target)
    high = raw > source[-1]
    if np.any(high):
        slope = max(
            float((target[-1] - target[-2]) / (source[-1] - source[-2])),
            0.0,
        )
        mapped[high] = target[-1] + slope * (raw[high] - source[-1])
    return np.maximum(mapped, 0).astype(np.float32)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stratified_training_sample(
    eligible: np.ndarray,
    reference_um_xy: np.ndarray,
    measured_od: np.ndarray,
    *,
    size: int,
    block_um: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample cells with inverse spatial-block and OD-quantile frequency."""

    selected = np.asarray(eligible, dtype=np.int64)
    xy = np.asarray(reference_um_xy, dtype=np.float64)
    measured = np.asarray(measured_od, dtype=np.float64)
    if (
        selected.ndim != 1
        or xy.ndim != 2
        or xy.shape != (len(measured), 2)
        or np.any(selected < 0)
        or np.any(selected >= len(measured))
        or not np.all(np.isfinite(measured[selected]))
    ):
        raise ValueError("stratified protein sample inputs do not align")
    if size < 1 or not np.isfinite(block_um) or block_um <= 0:
        raise ValueError("stratified protein sample controls are invalid")
    if len(selected) <= size:
        return np.sort(selected)
    values = measured[selected]
    quantiles = np.unique(np.quantile(values, np.linspace(0, 1, 9)[1:-1]))
    od_bin = np.searchsorted(quantiles, values, side="right")
    origin = np.min(xy[selected], axis=0)
    spatial_bin = np.floor((xy[selected] - origin) / block_um).astype(np.int64)
    strata = np.column_stack((spatial_bin, od_bin))
    _unique, inverse, counts = np.unique(
        strata, axis=0, return_inverse=True, return_counts=True
    )
    weight = 1.0 / counts[inverse]
    weight /= weight.sum()
    return np.sort(rng.choice(selected, size=size, replace=False, p=weight))


def _aggregate_study_metrics(
    folds: list[dict[str, object]],
) -> dict[str, object]:
    metrics: dict[str, object] = {
        "schema_version": 3,
        "evaluation": "leave_one_mouse_out",
        "available": bool(folds),
        "fold_count": len(folds),
        "folds": folds,
    }
    for name in (
        "mae",
        "spearman",
        "spearman_64um",
        "mae_improvement",
        "mean_bias_fraction",
        "median_bias_fraction",
        "auroc",
        "average_precision",
    ):
        values = [float(row[name]) for row in folds if row.get(name) is not None]
        metrics[name] = float(np.median(values)) if values else None
    metrics["folds_beating_semantic_fraction"] = (
        float(np.mean([bool(row.get("beats_semantic_baseline")) for row in folds]))
        if folds
        else None
    )
    return metrics


def _evaluate_external_holdouts(
    candidate: _Candidate,
    *,
    architecture: str,
    config: object,
    table: object,
    training: np.ndarray,
    holdout_mice: tuple[str, ...],
    device: str,
) -> dict[str, object]:
    """Evaluate a frozen all-training model on never-fit external mice.

    These rows are deliberately separate from leave-one-training-mouse-out
    model selection. Once a mouse is used to choose or tune a candidate it is
    no longer an untouched holdout and must enter a later training/evaluation
    round under a new fingerprint.
    """

    from histopia.protein._cli import _compare_with_baseline
    from histopia.protein._model import evaluate_predictions

    mouse_values = np.asarray(table.mouse_ids, dtype=str)
    folds: list[dict[str, object]] = []
    for mouse_id in holdout_mice:
        test = np.flatnonzero(mouse_values == mouse_id)
        measured = np.asarray(table.measured_od, dtype=np.float64)[test]
        test = test[np.isfinite(measured)]
        if not len(test):
            continue
        predicted = _predict_table_candidate(
            candidate,
            table,
            np.asarray(training, dtype=np.int64),
            test,
            config,
            device=device,
        )
        probability = (
            candidate.predict(table.features[test], device=device)[0]
            if architecture == "hurdle_mlp"
            else None
        )
        base = evaluate_predictions(
            table.measured_od[test],
            predicted,
            binary_label=table.binary_label[test],
            probability=probability,
        )
        metrics = _compare_with_baseline(
            base,
            table,
            np.asarray(training, dtype=np.int64),
            test,
            predicted,
        )
        folds.append({"held_out": mouse_id, **metrics})
    aggregate = _aggregate_study_metrics(folds)
    aggregate.update(
        {
            "schema_version": 1,
            "evaluation": "frozen_external_mouse_holdout",
            "training_cohorts": sorted(set(mouse_values[training].tolist())),
            "holdout_cohorts": list(holdout_mice),
            "model_selection_eligible": False,
        }
    )
    return aggregate


def _training_visible_metrics(
    table: object,
    training: np.ndarray,
    predicted_od: np.ndarray,
    *,
    held_out_reference: dict[str, object],
) -> dict[str, object]:
    """Summarize an explicitly non-independent fit without hiding LOMO scores."""

    from histopia.protein._cli import _compare_with_baseline
    from histopia.protein._model import evaluate_predictions

    selected = np.asarray(training, dtype=np.int64)
    predicted = np.asarray(predicted_od, dtype=np.float64)
    if predicted.shape != selected.shape:
        raise ValueError("training-visible predictions do not align with rows")
    base = evaluate_predictions(
        np.asarray(table.measured_od)[selected],
        predicted,
    )
    visible = _compare_with_baseline(
        base,
        table,
        selected,
        selected,
        predicted,
    )
    return {
        **visible,
        "schema_version": 4,
        "evaluation": "training_visible_fit",
        "available": bool(len(selected)),
        "training_cells": int(len(selected)),
        "held_out_reference": held_out_reference,
    }


def _study_prediction_role(
    *,
    mouse_id: str,
    section: str,
    training_mice: tuple[str, ...],
    target_sections: tuple[str, ...],
    prediction_protocol: str,
) -> str:
    """Return a leakage-explicit role for one published slide prediction."""

    if prediction_protocol not in _PREDICTION_PROTOCOLS:
        raise ValueError(
            f"unsupported study prediction protocol: {prediction_protocol}"
        )
    if mouse_id not in training_mice:
        return "external-transfer"
    if prediction_protocol == "training-visible":
        return (
            "training-visible-fit"
            if section in target_sections
            else "training-visible-transfer"
        )
    return "leave-one-mouse-out" if len(training_mice) > 1 else "fit-visible"


def _section_display_max(prediction: Any) -> float:
    """Return one robust OD scale shared by measured and predicted cells."""

    supported = np.asarray(prediction.supported, dtype=bool)
    parts = []
    for values in (prediction.predicted_od_reference, prediction.measured_od):
        array = np.asarray(values, dtype=np.float64)
        valid = array[supported & np.isfinite(array)]
        if valid.size:
            parts.append(valid)
    if not parts:
        return 1.0
    return max(float(np.quantile(np.concatenate(parts), 0.99)), 1e-8)


def _cohort_bindings(contexts: dict[str, _Context]) -> dict[str, object]:
    return {
        mouse_id: {
            "registration_result_sha256": context.bindings.cells[
                "registration_result_sha256"
            ],
            "cell_result_fingerprint": context.bindings.cells["fingerprint"],
            "stain_result_fingerprint": context.bindings.stain["fingerprint"],
            "semantic_result_fingerprint": context.bindings.semantic["fingerprint"],
        }
        for mouse_id, context in contexts.items()
    }


def _fingerprint_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
