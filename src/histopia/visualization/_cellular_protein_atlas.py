"""Static, cell-resolved protein atlas for registered serial sections.

The generated application is deliberately server independent.  It contains
only fingerprinted binary assets, a path-free manifest, and a locally vendored
Three.js runtime.  Protein values are never inferred or re-quantified here:
the builder only packages sealed :class:`ProteinPredictions` arrays on the
exact registered cell geometry used by the prediction workflow.
"""

# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import struct
import tempfile
import zlib
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import BinaryIO, Literal, TypeAlias

import numpy as np

from histopia._atomic import write_binary_atomic
from histopia.visualization._protein_review import (
    ProteinRunDescriptor,
    _configured_run_root,
    _model_sort_key,
    _target_display_name,
)
from histopia.visualization._review_theme import themed_review_css

CellGeometryRun: TypeAlias = Path | str
ProteinAtlasModels: TypeAlias = Mapping[str, Mapping[str, ProteinRunDescriptor]]

_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_MAGIC = b"HCPA1"
_BOUNDARY_GEOMETRY_HEADER_SIZE = 160
_VALUES_HEADER_SIZE = 32
_DEFAULT_MAX_BYTES = 650 * 1024 * 1024
_DEFAULT_OVERVIEW_CELLS = 500_000
_DEFAULT_LAYER = "protein"
_DEFAULT_PRESENTATION_TARGETS = ("yap", "ck19", "ecad", "sma")
_DEFAULT_EXTRA_CUTOFF = 0.25
_DISPLAY_LOWER_PERCENTILE = 65.0
_DISPLAY_UPPER_PERCENTILE = 99.5
_DISPLAY_GAMMA = 0.72
_BOUNDARY_ALGORITHM = "accepted-label-radial-boundary-v1"
_DETAIL_BOUNDARY_VERTICES = 16
_OVERVIEW_BOUNDARY_VERTICES = 8
_BOUNDARY_STRIPE_HEIGHT = 384
_BOUNDARY_QC_CELLS = 128

_CELL_IDENTITY_PALETTE = (
    "#ff5ca8",
    "#2dd4ff",
    "#ffd43b",
    "#73e36b",
    "#a78bfa",
    "#ff875f",
    "#55f0c3",
    "#60a5fa",
    "#f472b6",
    "#facc15",
    "#34d399",
    "#c084fc",
    "#fb7185",
    "#22d3ee",
    "#bef264",
    "#f59e0b",
)

_TARGET_COLORS = {
    "yap": "#f14ed3",
    "ck19": "#ffd43b",
    "ecad": "#25c7f5",
    "ncad": "#ff7f9f",
    "sma": "#76e06a",
    "ki67": "#ff5d57",
    "perk": "#43d7ff",
    "cjun": "#ff9b50",
    "junb": "#e8ff6a",
    "lamp1": "#b575ff",
    "ck18": "#ffe088",
    "gfp": "#67f26f",
    "myc": "#ff6fd8",
    "myccst": "#ff8fab",
    "casp3": "#ff835c",
    "erp44": "#69e7e1",
    "glucagon": "#ffba52",
    "insulin": "#5d8cff",
    "nr2f2": "#9da5ff",
    "ph2ax": "#ff5ca8",
    "prrx1": "#50e6a5",
    "sox2": "#bf8cff",
    "sox5": "#76a9ff",
    "twist2": "#ff936f",
}

_TARGET_LABELS = {
    "yap": "YAP",
    "ecad": "E-Cad",
    "ncad": "N-Cad",
    "perk": "pERK",
    "cjun": "cJun",
    "junb": "JunB",
    "ki67": "Ki67",
    "ck18": "CK18",
    "ck19": "CK19",
    "sma": "SMA",
    "lamp1": "Lamp1",
    "myc": "MYC",
    "myccst": "MYC (CST)",
    "casp3": "Casp3",
    "erp44": "ERp44",
    "gfp": "GFP",
    "nr2f2": "Nr2f2",
    "ph2ax": "pH2AX",
    "prrx1": "Prrx1",
    "sox2": "Sox2",
    "sox5": "Sox5",
    "twist2": "Twist2",
    "glucagon": "Glucagon",
    "insulin": "Insulin",
}

_STATUS_LABELS = {
    "promoted": "Promoted held-out transfer",
    "exploratory_calibration_failed": "Exploratory · OD calibration gate failed",
    "exploratory_no_held_out_fold": "Exploratory · no held-out fold",
    "failed_transfer_diagnostic": "Failed transfer diagnostic",
}


def build_cellular_protein_atlas(
    registration_runs: Mapping[str, Path | str],
    cell_runs: Mapping[str, Path | str],
    protein_models: ProteinAtlasModels,
    cell_geometry_runs: Mapping[str, CellGeometryRun],
    output_dir: Path | str,
    *,
    topology_runs: Mapping[str, Path | str] | None = None,
    max_overview_cells: int = _DEFAULT_OVERVIEW_CELLS,
    max_bytes: int = _DEFAULT_MAX_BYTES,
    default_layer: Literal["cells", "protein"] = _DEFAULT_LAYER,
    default_targets: Sequence[str] | None = None,
    workers: int = 1,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Build a browser-only cellular protein atlas.

    One leave-mouse-out model is selected per target and cohort.  A target's
    training-visible reconstruction is never eligible, including on measured
    sections, because switching protocols inside a volume would create a
    scientifically misleading discontinuity.

    ``cell_geometry_runs`` points to the reusable protein geometry-cache root.
    The builder accepts either ``ROOT/COHORT/multiscale-v3/SECTION.npz`` or a
    cohort-local ``ROOT/multiscale-v3/SECTION.npz`` layout.
    """

    if not registration_runs:
        raise ValueError("cellular protein atlas requires at least one cohort")
    _positive_integer(max_overview_cells, "maximum overview cells")
    _positive_integer(max_bytes, "cellular protein atlas byte budget")
    _positive_integer(workers, "cellular protein atlas workers")
    if default_layer not in {"cells", "protein"}:
        raise ValueError(
            "default cellular protein atlas layer must be cells or protein"
        )
    requested_targets = _validated_default_targets(default_targets)
    cohort_ids = tuple(sorted(registration_runs))
    required = set(cohort_ids)
    for name, values in (
        ("cell", cell_runs),
        ("protein model", protein_models),
        ("cell geometry", cell_geometry_runs),
    ):
        missing = required - set(values)
        extra = set(values) - required
        if missing:
            raise ValueError(
                f"cellular protein atlas has no {name} input for {sorted(missing)[0]}"
            )
        if extra:
            raise ValueError(
                f"cellular protein atlas has unknown {name} cohort {sorted(extra)[0]}"
            )
    topology_runs = topology_runs or {}
    unknown_topology = set(topology_runs) - required
    if unknown_topology:
        raise ValueError(
            "cellular protein atlas has unknown topology cohort "
            f"{sorted(unknown_topology)[0]}"
        )
    for cohort in cohort_ids:
        if not _NAME_RE.fullmatch(cohort):
            raise ValueError(f"invalid cellular protein atlas cohort: {cohort!r}")

    output = Path(output_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    reuse_root = output if output.is_dir() else None
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.atlas-", dir=output.parent))
    try:
        index = _build_atlas(
            registration_runs,
            cell_runs,
            protein_models,
            cell_geometry_runs,
            staging,
            topology_runs=topology_runs,
            max_overview_cells=max_overview_cells,
            max_bytes=max_bytes,
            default_layer=default_layer,
            default_targets=requested_targets,
            reuse_root=reuse_root,
            workers=workers,
            progress=progress,
        )
        _publish_directory(staging, output)
        return output / index.name
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def refresh_cellular_depth_models(output_dir: Path | str) -> int:
    """Add current boundary-derived depth summaries to an existing atlas.

    This supports safe in-place upgrades of static presentation bundles: only
    manifest metadata is refreshed, while every fingerprinted geometry and
    protein-value asset remains byte-for-byte unchanged.
    """

    output = Path(output_dir).resolve()
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    cohorts = manifest.get("cohorts")
    if not isinstance(cohorts, list):
        raise ValueError("cellular protein atlas manifest has no cohorts")
    refreshed = 0
    for cohort in cohorts:
        if not isinstance(cohort, dict) or not isinstance(cohort.get("sections"), list):
            continue
        rows: list[tuple[float, int]] = []
        for section in cohort["sections"]:
            geometry = (
                section.get("overview_geometry") if isinstance(section, dict) else None
            )
            if not isinstance(section, dict) or not isinstance(geometry, dict):
                continue
            relative = Path(str(geometry.get("asset", "")))
            asset = (output / relative).resolve()
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or not asset.is_relative_to(output)
            ):
                raise ValueError("cellular depth model asset path is invalid")
            value = _boundary_median_diameter_um(asset)
            section["median_boundary_diameter_um"] = value
            rows.append((value, int(section.get("overview_cell_count", 0))))
        if not rows:
            continue
        physical_gap_um = _median_section_spacing(cohort["sections"])
        median_diameter_um = _weighted_median(rows)
        visual_spacing_um = float(
            np.clip(
                median_diameter_um * 1.05,
                physical_gap_um * 1.1,
                physical_gap_um * 3.2,
            )
        )
        identity = cohort.get("cell_identity")
        if not isinstance(identity, dict):
            identity = {}
            cohort["cell_identity"] = identity
        model = identity.get("morphology_aware_z")
        if not isinstance(model, dict):
            model = {}
            identity["morphology_aware_z"] = model
        model.update(
            {
                "method": "boundary-size-local-packing-display-v1",
                "median_cell_diameter_um": median_diameter_um,
                "physical_section_spacing_um": physical_gap_um,
                "visual_section_spacing_um": visual_spacing_um,
                "visual_z_scale": visual_spacing_um / physical_gap_um,
            }
        )
        refreshed += 1
    _reject_local_paths(manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "manifest-data.js").write_text(
        "globalThis.HISTOPIA_CELLULAR_PROTEIN_ATLAS="
        + json.dumps(manifest, separators=(",", ":"))
        + ";\n"
    )
    return refreshed


def _build_atlas(
    registration_runs: Mapping[str, Path | str],
    cell_runs: Mapping[str, Path | str],
    protein_models: ProteinAtlasModels,
    cell_geometry_runs: Mapping[str, CellGeometryRun],
    output: Path,
    *,
    topology_runs: Mapping[str, Path | str],
    max_overview_cells: int,
    max_bytes: int,
    default_layer: Literal["cells", "protein"],
    default_targets: tuple[str, ...] | None,
    reuse_root: Path | None,
    workers: int,
    progress: Callable[[str], None] | None,
) -> Path:
    from histopia.cells._result import (
        validate_cell_artifact,
        validate_cell_result_index,
    )

    output.mkdir(parents=True, exist_ok=True)
    reusable_sections = _load_reusable_boundary_sections(reuse_root)
    cohorts: list[dict[str, object]] = []
    display_histograms: dict[str, np.ndarray] = {}
    preview_payloads: dict[str, dict[str, object]] = {}
    for cohort in sorted(registration_runs):
        if progress is not None:
            progress(f"Preparing cellular protein atlas cohort {cohort}")
        registration_root = Path(registration_runs[cohort]).expanduser().resolve()
        registration_path = registration_root / "registration_result.json"
        registration_bytes = registration_path.read_bytes()
        registration = json.loads(registration_bytes)
        registration_sha = hashlib.sha256(registration_bytes).hexdigest()
        registration_slides = registration.get("slides")
        if not isinstance(registration_slides, list) or not registration_slides:
            raise ValueError(f"registration result has no slides for {cohort}")

        cell_root = Path(cell_runs[cohort]).expanduser().resolve()
        cell_result = validate_cell_result_index(cell_root)
        if cell_result.get("registration_result_sha256") != registration_sha:
            raise ValueError(f"cell run is not bound to registration for {cohort}")
        cell_slides = cell_result.get("slides")
        if not isinstance(cell_slides, list) or not cell_slides:
            raise ValueError(f"cell result has no slides for {cohort}")
        section_ids = tuple(str(row["section"]) for row in cell_slides)
        if len(set(section_ids)) != len(section_ids):
            raise ValueError(f"cell result contains duplicate sections for {cohort}")
        validated_label_paths = _validate_label_artifacts(
            cell_root,
            cell_result,
            cell_slides,
            workers=workers,
            validator=validate_cell_artifact,
        )

        selected_models = _select_models(
            cohort,
            protein_models[cohort],
            registration_sha=registration_sha,
            cell_fingerprint=str(cell_result["fingerprint"]),
            section_ids=section_ids,
        )
        if not selected_models:
            raise ValueError(
                f"cellular protein atlas has no held-out models for {cohort}"
            )
        cohort_default_targets = _cohort_default_targets(
            selected_models,
            requested=default_targets,
        )
        default_target = cohort_default_targets[0]
        targets = {
            target: _target_manifest(model)
            for target, model in sorted(selected_models.items())
        }
        rows_by_target = {
            target: {
                str(row["section"]): row
                for row in model["result"]["slides"]
                if row.get("cohort") == cohort
            }
            for target, model in selected_models.items()
        }
        for target, rows in rows_by_target.items():
            if tuple(rows) != section_ids:
                raise ValueError(
                    f"protein slide order differs from cell sections: {cohort}/{target}"
                )

        counts = np.asarray(
            [int(row.get("cells", row.get("cell_count", 0))) for row in cell_slides],
            dtype=np.int64,
        )
        if np.any(counts <= 0):
            raise ValueError(f"cell result has invalid section counts for {cohort}")
        overview_counts = _overview_allocations(counts, max_overview_cells)
        section_manifests: list[dict[str, object]] = []
        preview_xyz: list[np.ndarray] = []
        preview_labels: list[np.ndarray] = []
        preview_values: dict[str, list[np.ndarray]] = {
            target: [] for target in selected_models
        }
        preview_supported: dict[str, list[np.ndarray]] = {
            target: [] for target in selected_models
        }
        all_bounds: list[tuple[float, float, float, float, float]] = []
        boundary_diameter_rows: list[tuple[float, int]] = []

        for slide_index, (section, _cell_row) in enumerate(
            zip(section_ids, cell_slides, strict=True)
        ):
            if not isinstance(_cell_row, dict):
                raise ValueError(
                    f"cell result slide row is invalid: {cohort}/{section}"
                )
            if progress is not None:
                progress(
                    f"Packaging {cohort} section {section} "
                    f"({slide_index + 1}/{len(section_ids)})"
                )
            geometry_path = _geometry_path(
                Path(cell_geometry_runs[cohort]).expanduser().resolve(),
                cohort,
                section,
            )
            geometry = _read_geometry(
                geometry_path,
                cohort=cohort,
                section=section,
                registration_sha=registration_sha,
                cell_fingerprint=str(cell_result["fingerprint"]),
            )
            labels = geometry["label_ids"]
            native_xy = geometry["native_xy"]
            xyz = geometry["reference_um_xyz"]
            if len(labels) != counts[slide_index]:
                raise ValueError(f"cell geometry count differs for {cohort}/{section}")
            sample = _deterministic_sample_indices(
                labels, int(overview_counts[slide_index])
            )
            z_values = np.asarray(xyz[:, 2], dtype=np.float64)
            z_um = float(np.median(z_values))
            if np.max(np.abs(z_values - z_um), initial=0.0) > 1e-4:
                raise ValueError(
                    f"cell geometry has mixed Z values for {cohort}/{section}"
                )
            bounds = (
                float(np.min(xyz[:, 0])),
                float(np.min(xyz[:, 1])),
                float(np.max(xyz[:, 0])),
                float(np.max(xyz[:, 1])),
                z_um,
            )
            all_bounds.append(bounds)
            binding = _registration_geometry_binding(
                registration,
                _cell_row,
                cohort=cohort,
                section=section,
            )
            matrix = np.asarray(binding["matrix"], dtype=np.float64)
            bbox = binding["bbox"]
            source_mpp = binding["source_mpp"]
            assert isinstance(bbox, tuple)
            assert isinstance(source_mpp, tuple)
            label_relative = str(_cell_row.get("labels", ""))
            artifacts = cell_result.get("artifacts")
            label_digest = (
                artifacts.get(label_relative) if isinstance(artifacts, dict) else None
            )
            if not isinstance(label_digest, str):
                raise ValueError(f"cell label is not sealed: {cohort}/{section}")
            label_path = validated_label_paths[section]
            detail_fingerprint = _boundary_source_fingerprint(
                registration_sha=registration_sha,
                cell_fingerprint=str(cell_result["fingerprint"]),
                label_digest=label_digest,
                transform_digest=str(binding["transform_digest"]),
                bbox=bbox,
                matrix=matrix,
                labels=labels,
                sectors=_DETAIL_BOUNDARY_VERTICES,
                overview=False,
            )
            overview_fingerprint = _boundary_source_fingerprint(
                registration_sha=registration_sha,
                cell_fingerprint=str(cell_result["fingerprint"]),
                label_digest=label_digest,
                transform_digest=str(binding["transform_digest"]),
                bbox=bbox,
                matrix=matrix,
                labels=labels[sample],
                sectors=_OVERVIEW_BOUNDARY_VERTICES,
                overview=True,
            )
            reused = _reuse_boundary_pair(
                reusable_sections.get((cohort, section)),
                reuse_root,
                output,
                detail_fingerprint=detail_fingerprint,
                overview_fingerprint=overview_fingerprint,
                detail_count=len(labels),
                overview_count=len(sample),
            )
            if reused is None:
                profiles, profile_offsets, geometry_qc = _extract_boundary_profiles(
                    label_path,
                    labels,
                    native_xy,
                    bbox=bbox,
                    matrix=matrix,
                    reference_xy=xyz[:, :2],
                )
                overview_profiles, overview_offsets = _overview_boundary_profiles(
                    profiles[sample], profile_offsets[sample]
                )
                geometry_asset = _write_boundary_geometry_asset(
                    output,
                    cohort,
                    section,
                    labels,
                    native_xy,
                    xyz,
                    profiles,
                    profile_offsets,
                    matrix=matrix,
                    source_mpp=source_mpp,
                    source_fingerprint=detail_fingerprint,
                    quality_control=geometry_qc,
                    overview=False,
                )
                overview_geometry = _write_boundary_geometry_asset(
                    output,
                    cohort,
                    section,
                    labels[sample],
                    native_xy[sample],
                    xyz[sample],
                    overview_profiles,
                    overview_offsets,
                    matrix=matrix,
                    source_mpp=source_mpp,
                    source_fingerprint=overview_fingerprint,
                    quality_control=geometry_qc,
                    overview=True,
                )
            else:
                geometry_asset, overview_geometry = reused
                if progress is not None:
                    progress(f"Reused validated cell boundaries for {cohort}/{section}")
            median_boundary_diameter_um = _boundary_median_diameter_um(
                output / str(overview_geometry["asset"])
            )
            boundary_diameter_rows.append(
                (median_boundary_diameter_um, int(len(sample)))
            )
            _enforce_budget(output, max_bytes)
            section_targets: dict[str, dict[str, object]] = {}
            for target, model in sorted(selected_models.items()):
                row = rows_by_target[target][section]
                prediction_path = model["root"] / str(row["predictions"])
                prediction = _load_prediction(model, row, prediction_path)
                if not np.array_equal(prediction.label_ids, labels):
                    raise ValueError(
                        f"protein labels differ from geometry: {cohort}/{section}/{target}"
                    )
                if not np.all(np.asarray(prediction.section_ids, dtype=str) == section):
                    raise ValueError(
                        f"protein prediction contains another section: {cohort}/{section}/{target}"
                    )
                display_max = float(model["display_max_od"])
                quantized, value_support = _quantize_od(
                    prediction.predicted_od_reference,
                    prediction.supported,
                    display_max,
                )
                histogram = display_histograms.setdefault(
                    target,
                    np.zeros(256, dtype=np.int64),
                )
                histogram += np.bincount(
                    quantized[value_support],
                    minlength=256,
                ).astype(np.int64)
                predicted_asset = _write_values_asset(
                    output,
                    cohort,
                    section,
                    target,
                    quantized,
                    display_max=display_max,
                    kind=3,
                    overview=False,
                )
                overview_predicted = _write_values_asset(
                    output,
                    cohort,
                    section,
                    target,
                    quantized[sample],
                    display_max=display_max,
                    kind=3,
                    overview=True,
                )
                section_target: dict[str, object] = {
                    "predicted": predicted_asset,
                    "overview_predicted": overview_predicted,
                    "supported_cells": int(value_support.sum()),
                    "evaluation_role": row.get("evaluation_role", "unknown"),
                }
                measured = np.asarray(prediction.measured_od, dtype=np.float32)
                measured_support = np.isfinite(measured) & np.asarray(
                    prediction.supported, dtype=bool
                )
                if np.any(measured_support):
                    measured_quantized, _ = _quantize_od(
                        measured,
                        measured_support,
                        display_max,
                    )
                    section_target["observed"] = _write_values_asset(
                        output,
                        cohort,
                        section,
                        target,
                        measured_quantized,
                        display_max=display_max,
                        kind=4,
                        overview=False,
                    )
                    section_target["overview_observed"] = _write_values_asset(
                        output,
                        cohort,
                        section,
                        target,
                        measured_quantized[sample],
                        display_max=display_max,
                        kind=4,
                        overview=True,
                    )
                    section_target["measured_cells"] = int(measured_support.sum())
                section_targets[target] = section_target
                preview_values[target].append(quantized[sample])
                preview_supported[target].append(value_support[sample])
                _enforce_budget(output, max_bytes)
            preview_xyz.append(np.asarray(xyz[sample], dtype=np.float32))
            preview_labels.append(np.asarray(labels[sample], dtype=np.uint32))
            section_manifests.append(
                {
                    "id": section,
                    "z_um": z_um,
                    "cell_count": int(len(labels)),
                    "overview_cell_count": int(len(sample)),
                    "median_boundary_diameter_um": median_boundary_diameter_um,
                    "bounds_um": {
                        "x": [bounds[0], bounds[2]],
                        "y": [bounds[1], bounds[3]],
                    },
                    "geometry": geometry_asset,
                    "overview_geometry": overview_geometry,
                    "targets": section_targets,
                }
            )

        x_min = min(row[0] for row in all_bounds)
        y_min = min(row[1] for row in all_bounds)
        x_max = max(row[2] for row in all_bounds)
        y_max = max(row[3] for row in all_bounds)
        z_min = min(row[4] for row in all_bounds)
        z_max = max(row[4] for row in all_bounds)
        topology = (
            _copy_topology_envelope(
                Path(topology_runs[cohort]).expanduser().resolve(),
                output,
                cohort,
                registration_sha=registration_sha,
            )
            if cohort in topology_runs
            else None
        )
        topology_z_source = (
            str(topology["z_source"])
            if isinstance(topology, dict)
            else "uniform_assumed_after_failed_gap_calibration"
        )
        thickness = (
            float(topology["section_thickness_um"])
            if isinstance(topology, dict)
            else _median_section_spacing(section_manifests)
        )
        physical_gap_um = _median_section_spacing(section_manifests)
        median_boundary_diameter_um = _weighted_median(boundary_diameter_rows)
        visual_section_spacing_um = float(
            np.clip(
                median_boundary_diameter_um * 1.05,
                physical_gap_um * 1.1,
                physical_gap_um * 3.2,
            )
        )
        visual_z_scale = visual_section_spacing_um / physical_gap_um
        cohorts.append(
            {
                "id": cohort,
                "registration_result_sha256": registration_sha,
                "cell_result_fingerprint": cell_result["fingerprint"],
                "cell_count": int(counts.sum()),
                "overview_cell_count": int(overview_counts.sum()),
                "section_count": len(section_ids),
                "bounds_um": {
                    "x": [x_min, x_max],
                    "y": [y_min, y_max],
                    "z": [z_min, z_max],
                },
                "z_source": topology_z_source,
                "z_scope": "uniform 5 µm serial-section spacing; not calibrated section tracking",
                "section_thickness_um": thickness,
                "z_display_scales": [1, 25],
                "z_display_modes": ["physical", "morphology_adaptive", "exploded"],
                "default_target": default_target,
                "default_targets": list(cohort_default_targets),
                "default_section": section_ids[len(section_ids) // 2],
                "cell_identity": {
                    "representation": "accepted cell-label boundary vectors",
                    "geometry_source": "accepted native-resolution cell-label exterior",
                    "color_method": "deterministic section-and-label hash",
                    "overview_sampling": "deterministic bounded sample",
                    "boundary_derived": True,
                    "compact_boundary": True,
                    "exact_boundary": False,
                    "detail_boundary_vertices": _DETAIL_BOUNDARY_VERTICES,
                    "overview_boundary_vertices": _OVERVIEW_BOUNDARY_VERTICES,
                    "z_shape": "detected XY footprint with physical section depth or deterministic morphology- and neighborhood-aware display depth",
                    "scope": "detected XY footprints from independent 2D sections; adaptive mode uses cell-size-derived display spacing and per-cell display offsets but does not create cross-section cell tracks or measurements",
                    "morphology_aware_z": {
                        "method": "boundary-size-local-packing-display-v1",
                        "section_spacing": "robust median accepted-boundary equivalent diameter",
                        "cell_depth": "equivalent boundary diameter, radial compactness, and local size context",
                        "cell_offset": "deterministic neighborhood-balanced micro-offset with morphology-scaled amplitude",
                        "scope": "display only; not a measured Z coordinate or inferred cell lineage",
                        "median_cell_diameter_um": median_boundary_diameter_um,
                        "physical_section_spacing_um": physical_gap_um,
                        "visual_section_spacing_um": visual_section_spacing_um,
                        "visual_z_scale": visual_z_scale,
                    },
                },
                "targets": list(targets.values()),
                "sections": section_manifests,
                "topology_envelope": topology,
            }
        )
        preview_payloads[cohort] = {
            "xyz": np.concatenate(preview_xyz),
            "labels": np.concatenate(preview_labels),
            "values": {
                target: np.concatenate(rows) for target, rows in preview_values.items()
            },
            "supported": {
                target: np.concatenate(rows)
                for target, rows in preview_supported.items()
            },
            "bounds": (x_min, y_min, z_min, x_max, y_max, z_max),
        }

    display_transforms = _display_transforms(display_histograms)
    for cohort_manifest in cohorts:
        cohort_id = str(cohort_manifest["id"])
        targets_by_id = {str(row["id"]): row for row in cohort_manifest["targets"]}
        for target, row in targets_by_id.items():
            transform = dict(display_transforms[target])
            display_max = float(row["display_max_od"])
            transform["lower_od"] = transform["lower_fraction"] * display_max
            transform["upper_od"] = transform["upper_fraction"] * display_max
            row["display_transform"] = transform
        payload = preview_payloads[cohort_id]
        bounds = payload["bounds"]
        assert isinstance(bounds, tuple)
        cell_preview = _write_cell_identity_preview(
            output,
            cohort_id,
            payload["xyz"],
            payload["labels"],
            bounds=bounds,
        )
        protein_preview = _write_protein_preview(
            output,
            cohort_id,
            payload["xyz"],
            payload["values"],
            payload["supported"],
            targets=tuple(str(value) for value in cohort_manifest["default_targets"]),
            target_rows=targets_by_id,
            bounds=bounds,
        )
        cohort_manifest["fallback_previews"] = {
            "cells": cell_preview,
            "protein": protein_preview,
        }
        cohort_manifest["fallback_preview"] = (
            cell_preview if default_layer == "cells" else protein_preview
        )
        _enforce_budget(output, max_bytes)

    manifest = {
        "schema_version": 3,
        "binary_format": "HCPA1",
        "geometry_kind": 2,
        "max_channels": 10,
        "cell_identity_palette": list(_CELL_IDENTITY_PALETTE),
        "coordinate_scope": "accepted native cell boundaries mapped into registered source-section micrometres",
        "measurement_scope": "observed adaptive corrected target OD at 4 µm/px aggregated per cell",
        "prediction_scope": "one held-out transfer model per target across each complete volume",
        "cross_marker_scope": "within-antibody normalized display; not absolute cross-antibody abundance",
        "default": {
            "cohort": cohorts[0]["id"],
            "target": cohorts[0]["default_target"],
            "targets": cohorts[0]["default_targets"],
            "layer": default_layer,
            "mode": "3d",
            "depth_mode": "adaptive",
            "composite": "dominant",
            "extra_cutoff": _DEFAULT_EXTRA_CUTOFF,
            "z_scale": 2,
        },
        "presets": _presets(),
        "cohorts": cohorts,
    }
    _reject_local_paths(manifest)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "manifest-data.js").write_text(
        "globalThis.HISTOPIA_CELLULAR_PROTEIN_ATLAS="
        + json.dumps(manifest, separators=(",", ":"))
        + ";\n"
    )
    vendor = Path(__file__).with_name("_vendor")
    (output / "vendor").mkdir()
    for name in ("three.module.min.js", "OrbitControls.js", "LICENSE-three.txt"):
        shutil.copyfile(vendor / name, output / "vendor" / name)
    (output / "index.html").write_text(_HTML)
    (output / "protein-atlas.css").write_text(
        themed_review_css(_CSS_V2) + _LIGHT_PANEL_OVERRIDES_V2
    )
    atlas_assets = Path(__file__).with_name("_cellular_protein_atlas_assets")
    for name in ("protein-atlas.js", "protein-atlas-worker.js"):
        shutil.copyfile(atlas_assets / name, output / name)
    (output / ".nojekyll").touch()
    _enforce_budget(output, max_bytes)
    inventory = {
        "schema_version": 1,
        "cohort_ids": [str(row["id"]) for row in cohorts],
        "cell_count": sum(int(row["cell_count"]) for row in cohorts),
        "files": _file_inventory(output),
    }
    (output / "atlas-inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    _enforce_budget(output, max_bytes)
    if progress is not None:
        progress(
            "Cellular protein atlas complete: "
            f"{len(cohorts)} cohorts, {inventory['cell_count']:,} cells"
        )
    return output / "index.html"


def _select_models(
    cohort: str,
    configured: Mapping[str, ProteinRunDescriptor],
    *,
    registration_sha: str,
    cell_fingerprint: str,
    section_ids: tuple[str, ...],
) -> dict[str, dict[str, object]]:
    from histopia.protein._manifest import validate_protein_result_index

    candidates: dict[str, list[dict[str, object]]] = {}
    seen: set[str] = set()
    for configured_id, descriptor in sorted(configured.items()):
        if not _NAME_RE.fullmatch(configured_id):
            raise ValueError(f"invalid configured protein model: {configured_id!r}")
        root = _configured_run_root(configured_id, descriptor)
        result = validate_protein_result_index(root)
        fingerprint = str(result["fingerprint"])
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        if result.get("schema_version") != 4:
            continue
        if result.get("prediction_protocol") == "training-visible":
            continue
        if result.get("prediction_protocol") != "leave-one-mouse-out":
            raise ValueError(f"unsupported protein transfer protocol: {configured_id}")
        binding = result.get("cohort_bindings", {}).get(cohort)
        if not isinstance(binding, dict):
            continue
        if binding.get("registration_result_sha256") != registration_sha:
            raise ValueError(
                f"protein model registration binding differs: {cohort}/{configured_id}"
            )
        if binding.get("cell_result_fingerprint") != cell_fingerprint:
            raise ValueError(
                f"protein model cell binding differs: {cohort}/{configured_id}"
            )
        rows = [row for row in result["slides"] if row.get("cohort") == cohort]
        if tuple(str(row["section"]) for row in rows) != section_ids:
            raise ValueError(
                f"protein model does not cover the complete cohort: {cohort}/{configured_id}"
            )
        declared_id = result.get("model_id")
        if declared_id is not None and declared_id != configured_id:
            raise ValueError(
                f"configured protein model {configured_id!r} differs from its sealed model ID"
            )
        approval = _current_approval(root, fingerprint)
        promoted = bool(
            result.get("status") == "promoted" and approval.get("accepted") is True
        )
        status = _model_status(result, promoted=promoted)
        display_max = result.get("target_global_display_max_od")
        if (
            isinstance(display_max, bool)
            or not isinstance(display_max, int | float)
            or not math.isfinite(float(display_max))
            or float(display_max) <= 0
        ):
            raise ValueError(
                f"protein model has no valid target-global OD scale: {configured_id}"
            )
        target = str(result["target_id"]).lower()
        model = {
            "id": str(declared_id or configured_id),
            "target_id": target,
            "root": root,
            "result": result,
            "fingerprint": fingerprint,
            "model_fingerprint": str(result["model_fingerprint"]),
            "approved": approval.get("accepted") is True,
            "promoted": promoted,
            "status": status,
            "display_max_od": float(display_max),
            "metrics": result.get("metrics", {}),
        }
        candidates.setdefault(target, []).append(model)
    selected: dict[str, dict[str, object]] = {}
    for target, rows in candidates.items():
        rows.sort(
            key=lambda row: _model_sort_key(
                {
                    "promoted": row["promoted"],
                    "approved": row["approved"],
                    "metrics": row["metrics"],
                    "target_id": target,
                    "label": row["id"],
                    "id": row["id"],
                }
            )
        )
        selected[target] = rows[0]
    return selected


def _current_approval(root: Path, fingerprint: str) -> dict[str, object]:
    try:
        payload = json.loads((root / "protein_approval.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    if payload.get("schema_version") == 1 and payload.get("fingerprint") == fingerprint:
        return payload
    return {}


def _model_status(result: Mapping[str, object], *, promoted: bool) -> str:
    if promoted:
        return "promoted"
    metrics = result.get("metrics")
    fold_count = (
        len(metrics.get("folds", []))
        if isinstance(metrics, dict) and isinstance(metrics.get("folds", []), list)
        else 0
    )
    promotion = result.get("candidate_promotion")
    reasons = (
        [str(value).lower() for value in promotion.get("reasons", [])]
        if isinstance(promotion, dict)
        and isinstance(promotion.get("reasons", []), list)
        else []
    )
    if fold_count == 0 or any("folds are missing" in reason for reason in reasons):
        return "exploratory_no_held_out_fold"
    if any(
        "worse than" in reason
        or "spearman is below" in reason
        or "average_precision is below" in reason
        for reason in reasons
    ):
        return "failed_transfer_diagnostic"
    return "exploratory_calibration_failed"


def _target_manifest(model: Mapping[str, object]) -> dict[str, object]:
    result = model["result"]
    assert isinstance(result, dict)
    target = str(model["target_id"])
    display = float(model["display_max_od"])
    metrics = result.get("metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    portable_metrics = {
        key: metrics[key]
        for key in (
            "auroc",
            "average_precision",
            "spearman",
            "spearman_64um",
            "mae",
            "mean_bias_fraction",
            "median_bias_fraction",
        )
        if isinstance(metrics.get(key), int | float)
        and not isinstance(metrics.get(key), bool)
        and math.isfinite(float(metrics[key]))
    }
    folds = metrics.get("folds", [])
    status = str(model["status"])
    return {
        "id": target,
        "label": _TARGET_LABELS.get(target, _target_display_name(target)),
        "color": _TARGET_COLORS.get(target, "#8de4ff"),
        "status": status,
        "status_label": _STATUS_LABELS[status],
        "exploratory": status != "promoted",
        "model_id": model["id"],
        "result_fingerprint": model["fingerprint"],
        "model_fingerprint": model["model_fingerprint"],
        "prediction_protocol": "leave-one-mouse-out",
        "measurement_view": result.get("measurement_view"),
        "measurement_statistic": result.get("measurement_statistic", "mean"),
        "display_max_od": display,
        "quantization_error_od": display / 508.0,
        "metrics": portable_metrics,
        "held_out_fold_count": len(folds) if isinstance(folds, list) else 0,
    }


def _geometry_path(root: Path, cohort: str, section: str) -> Path:
    candidates = (
        root / cohort / "multiscale-v3" / f"{section}.npz",
        root / "multiscale-v3" / f"{section}.npz",
        root / cohort / f"{section}.npz",
        root / f"{section}.npz",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"multiscale cell geometry is missing: {cohort}/{section}")


def _validate_label_artifacts(
    root: Path,
    cell_result: Mapping[str, object],
    slides: Sequence[object],
    *,
    workers: int,
    validator: Callable[[Path, str], Path],
) -> dict[str, Path]:
    """Hash sealed label TIFFs with bounded I/O concurrency."""

    artifacts = cell_result.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("cell result artifact manifest is missing")
    tasks: list[tuple[str, Path, str]] = []
    for raw in slides:
        if not isinstance(raw, dict):
            raise ValueError("cell result slide row is invalid")
        section = str(raw.get("section", ""))
        relative = str(raw.get("labels", ""))
        digest = artifacts.get(relative)
        if not section or not isinstance(digest, str):
            raise ValueError(f"cell label is not sealed: {section or 'unknown'}")
        tasks.append((section, root / relative, digest))

    def validate(task: tuple[str, Path, str]) -> tuple[str, Path]:
        section, path, digest = task
        return section, validator(path, digest)

    if workers == 1:
        return dict(validate(task) for task in tasks)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return dict(executor.map(validate, tasks))


def _registration_geometry_binding(
    registration: Mapping[str, object],
    cell_row: Mapping[str, object],
    *,
    cohort: str,
    section: str,
) -> dict[str, object]:
    """Resolve and validate the native-to-reference transform for one section."""

    raw_slides = registration.get("slides")
    if not isinstance(raw_slides, list) or not raw_slides:
        raise ValueError(f"registration result has no slides for {cohort}")
    slide_name = Path(str(cell_row.get("slide", ""))).name
    matches = [
        row
        for row in raw_slides
        if isinstance(row, dict) and Path(str(row.get("path", ""))).name == slide_name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"cell slide does not resolve uniquely in registration: {cohort}/{section}"
        )
    reference_name = Path(str(registration.get("reference_slide", ""))).name
    references = [
        row
        for row in raw_slides
        if isinstance(row, dict)
        and Path(str(row.get("path", ""))).name == reference_name
    ]
    if len(references) != 1:
        raise ValueError(f"registration reference slide is invalid for {cohort}")
    slide = matches[0]
    reference = references[0]
    geometry = slide.get("geometry")
    reference_geometry = reference.get("geometry")
    transform = slide.get("transform")
    if (
        not isinstance(geometry, dict)
        or not isinstance(reference_geometry, dict)
        or not isinstance(transform, dict)
    ):
        raise ValueError(f"registration geometry is incomplete: {cohort}/{section}")
    try:
        bbox = tuple(int(value) for value in geometry["content_bbox_xywh"])
        cell_bbox = tuple(int(value) for value in cell_row["content_bbox_xywh"])
        thumbnail_to_native = np.asarray(
            geometry["thumbnail_to_native"], dtype=np.float64
        )
        reference_thumbnail_to_native = np.asarray(
            reference_geometry["thumbnail_to_native"], dtype=np.float64
        )
        moving_to_reference = np.asarray(transform["matrix"], dtype=np.float64)
        reference_mpp = tuple(float(value) for value in reference_geometry["mpp_xy"])
        source_mpp = tuple(float(value) for value in geometry["mpp_xy"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"registration geometry is malformed: {cohort}/{section}"
        ) from error
    matrices = (thumbnail_to_native, reference_thumbnail_to_native, moving_to_reference)
    if (
        bbox != cell_bbox
        or len(bbox) != 4
        or bbox[2] <= 0
        or bbox[3] <= 0
        or any(
            matrix.shape != (3, 3) or not np.all(np.isfinite(matrix))
            for matrix in matrices
        )
        or len(reference_mpp) != 2
        or len(source_mpp) != 2
        or min(*reference_mpp, *source_mpp) <= 0
    ):
        raise ValueError(f"cell and registration geometry differ: {cohort}/{section}")
    transform_digest = hashlib.sha256(
        json.dumps(transform["matrix"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if cell_row.get("transform_sha256") != transform_digest:
        raise ValueError(f"cell transform binding differs: {cohort}/{section}")
    matrix = (
        np.diag([reference_mpp[0], reference_mpp[1], 1.0])
        @ reference_thumbnail_to_native
        @ moving_to_reference
        @ np.linalg.inv(thumbnail_to_native)
    )
    if (
        not np.all(np.isfinite(matrix))
        or abs(float(np.linalg.det(matrix))) < 1e-12
        or not np.allclose(matrix[2], (0.0, 0.0, 1.0), atol=1e-10)
    ):
        raise ValueError(
            f"native-to-reference transform is invalid: {cohort}/{section}"
        )
    return {
        "bbox": bbox,
        "matrix": matrix,
        "source_mpp": source_mpp,
        "transform_digest": transform_digest,
    }


def _apply_homography(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    homogeneous = np.column_stack([values, np.ones(len(values), dtype=np.float64)])
    mapped = (np.asarray(matrix, dtype=np.float64) @ homogeneous.T).T
    if np.any(np.abs(mapped[:, 2]) < 1e-12):
        raise ValueError("native-to-reference transform maps cells to infinity")
    return mapped[:, :2] / mapped[:, 2, None]


def _boundary_source_fingerprint(
    *,
    registration_sha: str,
    cell_fingerprint: str,
    label_digest: str,
    transform_digest: str,
    bbox: tuple[int, int, int, int],
    matrix: np.ndarray,
    labels: np.ndarray,
    sectors: int,
    overview: bool,
) -> str:
    label_array = np.ascontiguousarray(labels, dtype="<u4")
    payload = {
        "algorithm": _BOUNDARY_ALGORITHM,
        "registration_result_sha256": registration_sha,
        "cell_result_fingerprint": cell_fingerprint,
        "label_artifact_sha256": label_digest,
        "transform_sha256": transform_digest,
        "content_bbox_xywh": list(bbox),
        "native_to_reference_um": np.asarray(matrix, dtype=np.float64).tolist(),
        "label_order_sha256": hashlib.sha256(label_array.tobytes()).hexdigest(),
        "sectors": sectors,
        "overview": overview,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _extract_boundary_profiles(
    path: Path,
    labels: np.ndarray,
    native_xy: np.ndarray,
    *,
    bbox: tuple[int, int, int, int],
    matrix: np.ndarray,
    reference_xy: np.ndarray,
    sectors: int = _DETAIL_BOUNDARY_VERTICES,
    stripe_height: int = _BOUNDARY_STRIPE_HEIGHT,
    qc_cells: int = _BOUNDARY_QC_CELLS,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Stream accepted labels into a compact angular boundary representation."""

    if sectors <= 2 or sectors % 2:
        raise ValueError("boundary sectors must be an even integer above two")
    if stripe_height <= 0 or qc_cells <= 0:
        raise ValueError("boundary extraction bounds must be positive")
    try:
        import pyvips
    except ImportError as error:
        raise RuntimeError(
            "cell-boundary atlas extraction requires the 'wsi' optional extra"
        ) from error

    label_ids = np.asarray(labels, dtype=np.uint32)
    centers = np.asarray(native_xy, dtype=np.float64)
    if centers.shape != (len(label_ids), 2):
        raise ValueError("native cell centers must have shape (cells, 2)")
    origin_x, origin_y, expected_width, expected_height = bbox
    local_centers = centers - np.asarray([origin_x, origin_y], dtype=np.float64)
    if (
        np.any(local_centers[:, 0] < -0.5)
        or np.any(local_centers[:, 0] > expected_width - 0.5)
        or np.any(local_centers[:, 1] < -0.5)
        or np.any(local_centers[:, 1] > expected_height - 0.5)
    ):
        raise ValueError("native cell centers fall outside the accepted content box")

    image = pyvips.Image.new_from_file(str(path), access="random")
    if image.bands > 1:
        image = image[0]
    if image.width != expected_width or image.height != expected_height:
        raise ValueError("accepted cell-label geometry differs from its content box")
    maximum_label = int(label_ids.max(initial=0))
    lookup = np.full(maximum_label + 1, -1, dtype=np.int32)
    lookup[label_ids] = np.arange(len(label_ids), dtype=np.int32)
    best = np.zeros((len(label_ids), sectors), dtype=np.uint64)
    counts_by_label = np.zeros(maximum_label + 1, dtype=np.uint64)

    sample_count = min(qc_cells, len(label_ids))
    sample_indices = _deterministic_sample_indices(label_ids, sample_count)
    sample_lookup = np.full(maximum_label + 1, -1, dtype=np.int16)
    sample_lookup[label_ids[sample_indices]] = np.arange(sample_count, dtype=np.int16)
    sample_x: list[list[np.ndarray]] = [[] for _ in range(sample_count)]
    sample_y: list[list[np.ndarray]] = [[] for _ in range(sample_count)]
    bin_width = 2.0 * math.pi / sectors

    for top in range(0, expected_height, stripe_height):
        height = min(stripe_height, expected_height - top)
        read_top = max(0, top - 1)
        read_bottom = min(expected_height, top + height + 1)
        stripe = image.crop(0, read_top, expected_width, read_bottom - read_top)
        if stripe.format != "uint":
            stripe = stripe.cast("uint")
        block = np.frombuffer(stripe.write_to_memory(), dtype=np.uint32).reshape(
            read_bottom - read_top, expected_width
        )
        offset = top - read_top
        core = block[offset : offset + height]
        stripe_max = int(core.max(initial=0))
        if stripe_max > maximum_label:
            raise ValueError("accepted label TIFF contains an undeclared cell label")
        flat = core.ravel().astype(np.int64, copy=False)
        counts_by_label += np.bincount(flat, minlength=maximum_label + 1).astype(
            np.uint64
        )

        sample_slots = sample_lookup[core]
        sample_rows, sample_columns = np.nonzero(sample_slots >= 0)
        if len(sample_rows):
            slots = sample_slots[sample_rows, sample_columns]
            for slot in np.unique(slots):
                selected = slots == slot
                sample_x[int(slot)].append(sample_columns[selected].astype(np.int32))
                sample_y[int(slot)].append(
                    (sample_rows[selected] + top).astype(np.int32)
                )

        above = (
            block[offset - 1 : offset - 1 + height]
            if top > 0
            else np.vstack((np.zeros((1, expected_width), dtype=np.uint32), core[:-1]))
        )
        below = (
            block[offset + 1 : offset + 1 + height]
            if top + height < expected_height
            else np.vstack((core[1:], np.zeros((1, expected_width), dtype=np.uint32)))
        )
        boundary = (core > 0) & ((core != above) | (core != below))
        boundary[:, 0] |= core[:, 0] > 0
        boundary[:, -1] |= core[:, -1] > 0
        boundary[:, 1:] |= (core[:, 1:] > 0) & (core[:, 1:] != core[:, :-1])
        boundary[:, :-1] |= (core[:, :-1] > 0) & (core[:, :-1] != core[:, 1:])
        boundary[:, 1:] |= (core[:, 1:] > 0) & (core[:, 1:] != above[:, :-1])
        boundary[:, :-1] |= (core[:, :-1] > 0) & (core[:, :-1] != above[:, 1:])
        boundary[:, 1:] |= (core[:, 1:] > 0) & (core[:, 1:] != below[:, :-1])
        boundary[:, :-1] |= (core[:, :-1] > 0) & (core[:, :-1] != below[:, 1:])
        rows, columns = np.nonzero(boundary)
        if not len(rows):
            continue
        ids = core[rows, columns]
        indices = lookup[ids]
        if np.any(indices < 0):
            raise ValueError("accepted label TIFF and geometry labels differ")
        dx = columns.astype(np.float64) - local_centers[indices, 0]
        dy = rows.astype(np.float64) + top - local_centers[indices, 1]
        angles = np.mod(np.arctan2(dy, dx), 2.0 * math.pi)
        bins = np.minimum((angles / bin_width).astype(np.int64), sectors - 1)
        fractions = angles / bin_width - bins
        offsets = np.minimum((fractions * 16.0).astype(np.uint64), 15)
        radius_key = np.maximum(
            np.rint((dx * dx + dy * dy) * 1024.0).astype(np.uint64), 1
        )
        packed = (radius_key << np.uint64(4)) | offsets
        flat_bins = indices.astype(np.int64) * sectors + bins
        np.maximum.at(best.ravel(), flat_bins, packed)

    present_counts = counts_by_label[label_ids]
    if np.any(present_counts == 0):
        raise ValueError("accepted label TIFF is missing declared cell labels")
    if int(np.count_nonzero(counts_by_label[1:])) != len(label_ids):
        raise ValueError("accepted label TIFF contains undeclared cell labels")
    raw_radius = best >> np.uint64(4)
    radii = np.sqrt(raw_radius.astype(np.float64) / 1024.0).astype(np.float32)
    radii[raw_radius > 0] += np.float32(0.5)
    angle_offsets = (best & np.uint64(15)).astype(np.uint8)
    radii, angle_offsets = _fill_missing_boundary_bins(radii, angle_offsets)

    mapped = _apply_homography(centers, matrix)
    registration_error = np.linalg.norm(
        mapped - np.asarray(reference_xy, dtype=np.float64), axis=1
    )
    maximum_registration_error = float(registration_error.max(initial=0.0))
    if maximum_registration_error > 0.25:
        raise ValueError(
            "native cell centers and registered geometry differ by more than 0.25 µm"
        )
    samples = [
        (
            (np.concatenate(sample_x[index]) + origin_x)
            if sample_x[index]
            else np.empty(0, np.int32),
            (np.concatenate(sample_y[index]) + origin_y)
            if sample_y[index]
            else np.empty(0, np.int32),
        )
        for index in range(sample_count)
    ]
    qc = _boundary_quality_control(
        radii[sample_indices],
        angle_offsets[sample_indices],
        centers[sample_indices],
        samples,
        matrix=matrix,
        reference_xy=np.asarray(reference_xy, dtype=np.float64)[sample_indices],
        sectors=sectors,
        maximum_registration_error_um=maximum_registration_error,
    )
    return radii, angle_offsets, qc


def _fill_missing_boundary_bins(
    radii: np.ndarray, offsets: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(radii, dtype=np.float32).copy()
    angles = np.asarray(offsets, dtype=np.uint8).copy()
    present = values > 0
    if np.any(~np.any(present, axis=1)):
        raise ValueError("one or more accepted cells have no exterior boundary")
    sectors = values.shape[1]
    for sector in range(sectors):
        missing = ~present[:, sector]
        if not np.any(missing):
            continue
        previous_distance = np.full(len(values), sectors, dtype=np.int16)
        next_distance = np.full(len(values), sectors, dtype=np.int16)
        previous_value = np.zeros(len(values), dtype=np.float32)
        next_value = np.zeros(len(values), dtype=np.float32)
        for distance in range(1, sectors + 1):
            previous = (sector - distance) % sectors
            take_previous = (
                missing & (previous_distance == sectors) & present[:, previous]
            )
            previous_distance[take_previous] = distance
            previous_value[take_previous] = values[take_previous, previous]
            following = (sector + distance) % sectors
            take_next = missing & (next_distance == sectors) & present[:, following]
            next_distance[take_next] = distance
            next_value[take_next] = values[take_next, following]
        denominator = previous_distance.astype(np.float32) + next_distance.astype(
            np.float32
        )
        interpolated = (
            previous_value * next_distance + next_value * previous_distance
        ) / np.maximum(denominator, 1)
        values[missing, sector] = interpolated[missing]
        angles[missing, sector] = 7
    if np.any(~np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("accepted cell boundary interpolation failed")
    return values, angles


def _boundary_vertices(
    radii: np.ndarray, offsets: np.ndarray, sectors: int
) -> np.ndarray:
    angles = (
        np.arange(sectors, dtype=np.float64)[None, :]
        + (np.asarray(offsets, dtype=np.float64) + 0.5) / 16.0
    ) * (2.0 * math.pi / sectors)
    values = np.asarray(radii, dtype=np.float64)
    return np.stack((values * np.cos(angles), values * np.sin(angles)), axis=2)


def _points_in_polygon(
    x: np.ndarray, y: np.ndarray, vertices: np.ndarray
) -> np.ndarray:
    inside = np.zeros(np.asarray(x).shape, dtype=bool)
    previous = vertices[-1]
    for current in vertices:
        crosses = (current[1] > y) != (previous[1] > y)
        boundary_x = (previous[0] - current[0]) * (y - current[1]) / (
            previous[1] - current[1] + 1e-300
        ) + current[0]
        inside ^= crosses & (x < boundary_x)
        previous = current
    return inside


def _boundary_quality_control(
    radii: np.ndarray,
    offsets: np.ndarray,
    native_xy: np.ndarray,
    samples: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    matrix: np.ndarray,
    reference_xy: np.ndarray,
    sectors: int,
    maximum_registration_error_um: float,
) -> dict[str, object]:
    vertices = _boundary_vertices(radii, offsets, sectors)
    ious: list[float] = []
    area_errors: list[float] = []
    centroid_errors: list[float] = []
    for index, (pixel_x, pixel_y) in enumerate(samples):
        if not len(pixel_x):
            raise ValueError("boundary QC sample is absent from the accepted labels")
        center = np.asarray(native_xy[index], dtype=np.float64)
        polygon = vertices[index] + center
        left = (
            int(math.floor(min(float(polygon[:, 0].min()), float(pixel_x.min())))) - 1
        )
        right = (
            int(math.ceil(max(float(polygon[:, 0].max()), float(pixel_x.max())))) + 1
        )
        top = int(math.floor(min(float(polygon[:, 1].min()), float(pixel_y.min())))) - 1
        bottom = (
            int(math.ceil(max(float(polygon[:, 1].max()), float(pixel_y.max())))) + 1
        )
        grid_y, grid_x = np.mgrid[top : bottom + 1, left : right + 1]
        predicted = _points_in_polygon(grid_x + 0.0, grid_y + 0.0, polygon)
        actual = np.zeros(grid_x.shape, dtype=bool)
        actual[pixel_y - top, pixel_x - left] = True
        intersection = int(np.count_nonzero(predicted & actual))
        union = int(np.count_nonzero(predicted | actual))
        ious.append(intersection / max(union, 1))
        polygon_area = 0.5 * abs(
            float(
                np.dot(polygon[:, 0], np.roll(polygon[:, 1], -1))
                - np.dot(polygon[:, 1], np.roll(polygon[:, 0], -1))
            )
        )
        area_errors.append(abs(polygon_area - len(pixel_x)) / max(len(pixel_x), 1))
        exact_center = np.asarray([pixel_x.mean(), pixel_y.mean()], dtype=np.float64)
        mapped = _apply_homography(exact_center[None, :], matrix)[0]
        centroid_errors.append(float(np.linalg.norm(mapped - reference_xy[index])))
    median_iou = float(np.median(ious))
    p05_iou = float(np.quantile(ious, 0.05))
    median_area_error = float(np.median(area_errors))
    maximum_centroid_error = float(max(centroid_errors, default=0.0))
    if median_iou < 0.82 or p05_iou < 0.60:
        raise ValueError(
            "accepted cell boundary vectors failed the mask-overlap quality gate"
        )
    if median_area_error > 0.12:
        raise ValueError(
            "accepted cell boundary vectors failed the area-error quality gate"
        )
    if maximum_centroid_error > 0.25:
        raise ValueError(
            "accepted label centroids and registered geometry differ by more than 0.25 µm"
        )
    return {
        "sample_cells": len(samples),
        "median_mask_vector_iou": median_iou,
        "p05_mask_vector_iou": p05_iou,
        "median_absolute_area_error_fraction": median_area_error,
        "maximum_centroid_error_um": maximum_centroid_error,
        "maximum_registration_center_error_um": maximum_registration_error_um,
        "invalid_or_self_intersecting": 0,
        "thresholds": {
            "median_mask_vector_iou_min": 0.82,
            "p05_mask_vector_iou_min": 0.60,
            "median_absolute_area_error_fraction_max": 0.12,
            "maximum_centroid_error_um": 0.25,
        },
    }


def _overview_boundary_profiles(
    radii: np.ndarray, offsets: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    detail_sectors = radii.shape[1]
    if detail_sectors % 2:
        raise ValueError("detail boundary sectors must be even")
    overview_sectors = detail_sectors // 2
    detail_angles = (
        np.arange(detail_sectors, dtype=np.float64)[None, :]
        + (np.asarray(offsets, dtype=np.float64) + 0.5) / 16.0
    ) / detail_sectors
    paired_radii = np.asarray(radii, dtype=np.float32).reshape(
        len(radii), overview_sectors, 2
    )
    paired_angles = detail_angles.reshape(len(radii), overview_sectors, 2)
    choice = np.argmax(paired_radii, axis=2)
    rows = np.arange(len(radii))[:, None]
    columns = np.arange(overview_sectors)[None, :]
    output_radii = paired_radii[rows, columns, choice]
    chosen_angles = paired_angles[rows, columns, choice]
    fractions = chosen_angles * overview_sectors - columns
    output_offsets = np.minimum(
        np.floor(np.clip(fractions, 0, 1 - np.finfo(float).eps) * 16), 15
    ).astype(np.uint8)
    return output_radii.astype(np.float32), output_offsets


def _read_geometry(
    path: Path,
    *,
    cohort: str,
    section: str,
    registration_sha: str,
    cell_fingerprint: str,
) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        required = {
            "metadata_json",
            "label_ids",
            "native_xy",
            "reference_um_xyz",
        }
        if not required.issubset(data.files):
            raise ValueError(f"cell geometry schema is incomplete: {cohort}/{section}")
        metadata = json.loads(str(data["metadata_json"]))
        if (
            metadata.get("schema_version") != 1
            or metadata.get("feature_schema_id")
            != "native-hdab-neutral-cell-multiscale-v3"
        ):
            raise ValueError(f"unsupported cell geometry schema: {cohort}/{section}")
        provenance = metadata.get("provenance")
        if not isinstance(provenance, dict):
            raise ValueError(f"cell geometry provenance is missing: {cohort}/{section}")
        if provenance.get("registration_result_sha256") != registration_sha:
            raise ValueError(
                f"cell geometry registration binding differs: {cohort}/{section}"
            )
        if provenance.get("cell_result_fingerprint") != cell_fingerprint:
            raise ValueError(f"cell geometry cell binding differs: {cohort}/{section}")
        slide_id = str(metadata.get("slide_id", ""))
        if slide_id not in {section, f"{cohort}-{section}"}:
            raise ValueError(
                f"cell geometry slide identity differs: {cohort}/{section}"
            )
        labels = np.asarray(data["label_ids"], dtype=np.uint32)
        native_xy = np.asarray(data["native_xy"], dtype=np.float64)
        xyz = np.asarray(data["reference_um_xyz"], dtype=np.float64)
    count = len(labels)
    if (
        labels.shape != (count,)
        or native_xy.shape != (count, 2)
        or xyz.shape != (count, 3)
        or np.any(labels <= 0)
        or len(np.unique(labels)) != count
        or not np.all(np.isfinite(native_xy))
        or not np.all(np.isfinite(xyz))
    ):
        raise ValueError(f"cell geometry arrays are invalid: {cohort}/{section}")
    return {
        "label_ids": labels,
        "native_xy": native_xy,
        "reference_um_xyz": xyz,
    }


def _load_prediction(
    model: Mapping[str, object], row: Mapping[str, object], path: Path
):
    from histopia.protein._manifest import validate_protein_artifact
    from histopia.protein._result import ProteinPredictions

    result = model["result"]
    assert isinstance(result, dict)
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError(f"protein artifact manifest is missing: {model['id']}")
    relative = str(row["predictions"])
    digest = artifacts.get(relative)
    if not isinstance(digest, str):
        raise ValueError(f"protein prediction is not sealed: {model['id']}/{relative}")
    validate_protein_artifact(path, digest)
    prediction = ProteinPredictions.load(path)
    if prediction.fingerprint != row.get("prediction_fingerprint"):
        raise ValueError(
            f"protein prediction fingerprint differs: {model['id']}/{relative}"
        )
    if prediction.model_fingerprint != model["model_fingerprint"]:
        raise ValueError(
            f"protein prediction model binding differs: {model['id']}/{relative}"
        )
    return prediction


def _overview_allocations(counts: np.ndarray, maximum: int) -> np.ndarray:
    if maximum < int(np.count_nonzero(counts)):
        raise ValueError(
            "maximum overview cells must retain at least one cell per section"
        )
    total = int(counts.sum())
    if total <= maximum:
        return counts.copy()
    raw = counts.astype(np.float64) * maximum / total
    allocated = np.maximum(np.floor(raw).astype(np.int64), 1)
    allocated = np.minimum(allocated, counts)
    remainder = maximum - int(allocated.sum())
    if remainder > 0:
        order = np.argsort(-(raw - np.floor(raw)), kind="stable")
        for index in order:
            if remainder <= 0:
                break
            if allocated[index] < counts[index]:
                allocated[index] += 1
                remainder -= 1
    elif remainder < 0:
        order = np.argsort(raw - np.floor(raw), kind="stable")
        for index in order:
            if remainder >= 0:
                break
            if allocated[index] > 1:
                allocated[index] -= 1
                remainder += 1
    if int(allocated.sum()) > maximum:
        raise RuntimeError("overview allocation exceeded its deterministic bound")
    return allocated


def _deterministic_sample_indices(labels: np.ndarray, count: int) -> np.ndarray:
    if count >= len(labels):
        return np.arange(len(labels), dtype=np.int64)
    values = np.asarray(labels, dtype=np.uint64)
    scores = values * np.uint64(11400714819323198485)
    chosen = np.argpartition(scores, count - 1)[:count]
    return np.sort(chosen.astype(np.int64))


def _quantize_od(
    values: np.ndarray,
    supported: np.ndarray,
    display_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(values, dtype=np.float64)
    mask = np.asarray(supported, dtype=bool) & np.isfinite(raw) & (raw >= 0)
    output = np.zeros(raw.shape, dtype=np.uint8)
    output[mask] = 1 + np.rint(np.clip(raw[mask] / display_max, 0, 1) * 254).astype(
        np.uint8
    )
    return output, mask


def _write_boundary_geometry_asset(
    output: Path,
    cohort: str,
    section: str,
    labels: np.ndarray,
    native_xy: np.ndarray,
    xyz: np.ndarray,
    radii: np.ndarray,
    angle_offsets: np.ndarray,
    *,
    matrix: np.ndarray,
    source_mpp: tuple[float, float],
    source_fingerprint: str,
    quality_control: Mapping[str, object],
    overview: bool,
) -> dict[str, object]:
    """Write HCPA1 kind-2 accepted-boundary geometry."""

    count = len(labels)
    centers = np.asarray(native_xy, dtype=np.float64)
    profiles = np.asarray(radii, dtype=np.float32)
    offsets = np.asarray(angle_offsets, dtype=np.uint8)
    sectors = profiles.shape[1] if profiles.ndim == 2 else 0
    if (
        centers.shape != (count, 2)
        or np.asarray(xyz).shape != (count, 3)
        or sectors not in {_DETAIL_BOUNDARY_VERTICES, _OVERVIEW_BOUNDARY_VERTICES}
        or offsets.shape != profiles.shape
        or np.any(~np.isfinite(profiles))
        or np.any(profiles <= 0)
        or np.any(offsets > 15)
    ):
        raise ValueError(f"boundary geometry arrays are invalid: {cohort}/{section}")
    x = centers[:, 0]
    y = centers[:, 1]
    x_min, y_min = float(x.min()), float(y.min())
    x_span = max(float(x.max()) - x_min, 1e-9)
    y_span = max(float(y.max()) - y_min, 1e-9)
    qx = np.rint((x - x_min) / x_span * 65535).astype("<u2")
    qy = np.rint((y - y_min) / y_span * 65535).astype("<u2")
    maximum_by_cell = np.max(profiles, axis=1)
    radius_max = max(float(maximum_by_cell.max(initial=0.0)), 1e-9)
    qmax = np.rint(maximum_by_cell / radius_max * 65535).astype("<u2")
    normalized_radii = np.rint(profiles / maximum_by_cell[:, None] * 255).astype(
        np.uint8
    )
    normalized_radii = np.maximum(normalized_radii, 1).astype(np.uint8)
    packed_offsets = (offsets[:, 0::2] | (offsets[:, 1::2] << np.uint8(4))).astype(
        np.uint8
    )
    flags = 1 if overview else 0
    z_um = float(np.median(xyz[:, 2]))
    transform = np.asarray(matrix, dtype="<f8")
    decoded_centers = np.column_stack(
        (
            x_min + qx.astype(np.float64) / 65535 * x_span,
            y_min + qy.astype(np.float64) / 65535 * y_span,
        )
    )
    decoded_reference = _apply_homography(decoded_centers, transform)
    reference_error = np.linalg.norm(
        decoded_reference - np.asarray(xyz[:, :2], dtype=np.float64), axis=1
    )
    maximum_coordinate_error = float(reference_error.max(initial=0.0))
    if maximum_coordinate_error > 0.25:
        raise ValueError(
            f"boundary geometry quantization exceeds 0.25 µm: {cohort}/{section}"
        )

    def writer(stream: BinaryIO) -> None:
        header = bytearray(_BOUNDARY_GEOMETRY_HEADER_SIZE)
        header[:5] = _MAGIC
        struct.pack_into(
            "<BBBII",
            header,
            5,
            2,
            1,
            flags,
            count,
            _BOUNDARY_GEOMETRY_HEADER_SIZE,
        )
        struct.pack_into("<dddd", header, 16, x_min, y_min, x_span, y_span)
        struct.pack_into("<ffHH", header, 48, radius_max, z_um, sectors, 0)
        struct.pack_into("<9d", header, 64, *transform.ravel())
        struct.pack_into(
            "<ffff",
            header,
            136,
            float(source_mpp[0]),
            float(source_mpp[1]),
            radius_max / 131070,
            0.5 / 255,
        )
        stream.write(header)
        for array in (
            np.asarray(labels, dtype="<u4"),
            qx,
            qy,
            qmax,
            normalized_radii,
            packed_offsets,
        ):
            stream.write(memoryview(np.ascontiguousarray(array)).cast("B"))

    entry = _write_fingerprinted(
        output / "assets" / cohort / "geometry",
        f"{section}-{'overview' if overview else 'cells'}",
        ".hcpa",
        writer,
        output,
    )
    return {
        **entry,
        "count": count,
        "format_kind": 2,
        "representation": _BOUNDARY_ALGORITHM,
        "boundary_vertices": sectors,
        "source_fingerprint": source_fingerprint,
        "coordinate_quantization_error_um": maximum_coordinate_error,
        "maximum_radius_quantization_error_native_px": radius_max / 131070,
        "radial_fraction_quantization_error": 0.5 / 255,
        "angle_quantization_error_degrees": 180 / (sectors * 16),
        "quality_control": dict(quality_control),
    }


def _boundary_median_diameter_um(path: Path) -> float:
    """Read the robust cell-size display statistic from one HCPA1 asset.

    This deliberately mirrors the browser's equivalent-diameter calculation:
    the accepted radial boundary profile supplies cell area, while the sealed
    native-to-reference affine supplies the physical micrometre scale.
    """

    payload = path.read_bytes()
    if len(payload) < _BOUNDARY_GEOMETRY_HEADER_SIZE or payload[:5] != _MAGIC:
        raise ValueError(f"invalid cellular boundary asset: {path.name}")
    kind, version, _flags, count, header = struct.unpack_from("<BBBII", payload, 5)
    radius_max, _z_um, sectors, _reserved = struct.unpack_from("<ffHH", payload, 48)
    if (
        kind != 2
        or version != 1
        or count <= 0
        or header < _BOUNDARY_GEOMETRY_HEADER_SIZE
        or sectors not in {_DETAIL_BOUNDARY_VERTICES, _OVERVIEW_BOUNDARY_VERTICES}
        or not math.isfinite(radius_max)
        or radius_max <= 0
    ):
        raise ValueError(f"invalid cellular boundary header: {path.name}")
    transform = np.asarray(struct.unpack_from("<9d", payload, 64)).reshape(3, 3)
    affine_scale = float(transform[2, 2])
    if (
        abs(float(transform[2, 0])) > 1e-10
        or abs(float(transform[2, 1])) > 1e-10
        or abs(affine_scale) < 1e-12
    ):
        raise ValueError(f"cellular boundary transform is not affine: {path.name}")
    qmax_offset = header + count * 4 + count * 2 + count * 2
    profiles_offset = qmax_offset + count * 2
    profiles_end = profiles_offset + count * sectors
    packed_angles_end = profiles_end + count * sectors // 2
    if packed_angles_end != len(payload):
        raise ValueError(f"invalid cellular boundary payload: {path.name}")
    qmax = np.frombuffer(payload, dtype="<u2", count=count, offset=qmax_offset)
    profiles = np.frombuffer(
        payload,
        dtype=np.uint8,
        count=count * sectors,
        offset=profiles_offset,
    ).reshape(count, sectors)
    maximum_radii = qmax.astype(np.float64) / 65535 * float(radius_max)
    radial_compactness = np.sqrt(
        np.mean(np.square(profiles.astype(np.float64) / 255), axis=1)
    )
    linear = transform[:2, :2] / affine_scale
    area_radius_scale = math.sqrt(max(abs(float(np.linalg.det(linear))), 1e-8))
    diameters = np.maximum(
        0.5,
        2 * maximum_radii * radial_compactness * area_radius_scale,
    )
    median_diameter = float(np.median(diameters))
    if not math.isfinite(median_diameter) or median_diameter <= 0:
        raise ValueError(f"invalid cellular boundary diameter: {path.name}")
    return median_diameter


def _weighted_median(rows: Sequence[tuple[float, int]]) -> float:
    """Return a deterministic count-weighted median for positive values."""

    ordered = sorted(
        (float(value), int(weight))
        for value, weight in rows
        if math.isfinite(float(value)) and value > 0 and weight > 0
    )
    if not ordered:
        raise ValueError("cellular depth model has no valid boundary diameters")
    threshold = sum(weight for _value, weight in ordered) / 2
    cumulative = 0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= threshold:
            return value
    return ordered[-1][0]


def _write_values_asset(
    output: Path,
    cohort: str,
    section: str,
    target: str,
    values: np.ndarray,
    *,
    display_max: float,
    kind: int,
    overview: bool,
) -> dict[str, object]:
    values = np.asarray(values, dtype=np.uint8)
    flags = 1 if overview else 0

    def writer(stream: BinaryIO) -> None:
        header = bytearray(_VALUES_HEADER_SIZE)
        header[:5] = _MAGIC
        struct.pack_into(
            "<BBBII", header, 5, kind, 1, flags, len(values), _VALUES_HEADER_SIZE
        )
        struct.pack_into("<ffII", header, 16, display_max, display_max / 508, 0, 0)
        stream.write(header)
        stream.write(memoryview(np.ascontiguousarray(values)).cast("B"))

    role = "observed" if kind == 4 else "predicted"
    return _write_fingerprinted(
        output / "assets" / cohort / "values" / target,
        f"{section}-{role}{'-overview' if overview else ''}",
        ".hcpa",
        writer,
        output,
    )


def _write_fingerprinted(
    directory: Path,
    stem: str,
    suffix: str,
    writer,
    output: Path,
) -> dict[str, object]:
    directory.mkdir(parents=True, exist_ok=True)
    pending = directory / f".{stem}.pending"
    write_binary_atomic(pending, writer)
    digest = _sha256_file(pending)
    destination = directory / f"{stem}-{digest[:16]}{suffix}"
    os.replace(pending, destination)
    return {
        "asset": destination.relative_to(output).as_posix(),
        "digest": digest,
        "size_bytes": destination.stat().st_size,
    }


def _load_reusable_boundary_sections(
    root: Path | None,
) -> dict[tuple[str, str], dict[str, object]]:
    if root is None:
        return {}
    try:
        manifest = json.loads((root / "manifest.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    cohorts = manifest.get("cohorts") if isinstance(manifest, dict) else None
    if not isinstance(cohorts, list):
        return {}
    output: dict[tuple[str, str], dict[str, object]] = {}
    for cohort in cohorts:
        if not isinstance(cohort, dict) or not isinstance(cohort.get("sections"), list):
            continue
        cohort_id = str(cohort.get("id", ""))
        for section in cohort["sections"]:
            if isinstance(section, dict):
                output[(cohort_id, str(section.get("id", "")))] = section
    return output


def _reuse_boundary_pair(
    section: Mapping[str, object] | None,
    reuse_root: Path | None,
    output: Path,
    *,
    detail_fingerprint: str,
    overview_fingerprint: str,
    detail_count: int,
    overview_count: int,
) -> tuple[dict[str, object], dict[str, object]] | None:
    if section is None or reuse_root is None:
        return None
    candidates = (
        (section.get("geometry"), detail_fingerprint, detail_count),
        (section.get("overview_geometry"), overview_fingerprint, overview_count),
    )
    validated: list[tuple[dict[str, object], Path, Path]] = []
    root_resolved = reuse_root.resolve()
    for raw, fingerprint, count in candidates:
        if (
            not isinstance(raw, dict)
            or raw.get("format_kind") != 2
            or raw.get("representation") != _BOUNDARY_ALGORITHM
            or raw.get("source_fingerprint") != fingerprint
            or raw.get("count") != count
            or not isinstance(raw.get("asset"), str)
            or not isinstance(raw.get("digest"), str)
        ):
            return None
        relative = Path(str(raw["asset"]))
        source = (root_resolved / relative).resolve()
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not source.is_relative_to(root_resolved)
            or not source.is_file()
            or _sha256_file(source) != raw["digest"]
            or source.stat().st_size != raw.get("size_bytes")
        ):
            return None
        destination = output / relative
        validated.append((dict(raw), source, destination))
    for _entry, source, destination in validated:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    return validated[0][0], validated[1][0]


def _copy_topology_envelope(
    root: Path,
    output: Path,
    cohort: str,
    *,
    registration_sha: str,
) -> dict[str, object]:
    from histopia.topology import validate_topology_result

    result = validate_topology_result(root)
    bound = result.get("registration_result_sha256")
    if bound is not None and bound != registration_sha:
        raise ValueError(f"topology registration binding differs: {cohort}")
    row = (
        result.get("envelope")
        if result.get("schema_version") == 2
        else (result.get("meshes") or [None])[0]
    )
    if not isinstance(row, dict):
        raise ValueError(f"topology envelope is missing: {cohort}")
    relative = str(row["viewer_asset"])
    source = root / relative
    digest = _sha256_file(source)
    destination = (
        output / "assets" / cohort / "topology" / f"envelope-{digest[:16]}.bin"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return {
        "asset": destination.relative_to(output).as_posix(),
        "digest": digest,
        "size_bytes": destination.stat().st_size,
        "topology_fingerprint": result["fingerprint"],
        "z_source": result.get("z_source", "unknown"),
        "section_thickness_um": result.get("section_thickness_um", 5.0),
        "scope": "coarse semantic tissue context; not cell type",
    }


def _write_cell_identity_preview(
    output: Path,
    cohort: str,
    xyz: np.ndarray,
    labels: np.ndarray,
    *,
    bounds: tuple[float, float, float, float, float, float],
) -> dict[str, object]:
    labels = np.asarray(labels, dtype=np.uint64)
    z_key = np.rint(np.asarray(xyz)[:, 2] * 1000).astype(np.int64).astype(np.uint64)
    hashed = labels * np.uint64(2654435761)
    hashed ^= z_key * np.uint64(2246822519)
    hashed ^= hashed >> np.uint64(16)
    palette = np.asarray(
        [
            [int(color[index : index + 2], 16) for index in (1, 3, 5)]
            for color in _CELL_IDENTITY_PALETTE
        ],
        dtype=np.uint8,
    )
    colors = palette[np.asarray(hashed % len(palette), dtype=np.int64)]
    return _write_projection_preview(
        output,
        cohort,
        "fallback-cells",
        xyz,
        colors,
        np.ones(len(labels), dtype=np.float32),
        bounds=bounds,
    )


def _write_protein_preview(
    output: Path,
    cohort: str,
    xyz: np.ndarray,
    values: Mapping[str, np.ndarray],
    supported: Mapping[str, np.ndarray],
    *,
    targets: tuple[str, ...],
    target_rows: Mapping[str, Mapping[str, object]],
    bounds: tuple[float, float, float, float, float, float],
) -> dict[str, object]:
    channels: list[np.ndarray] = []
    palette: list[list[int]] = []
    for target in targets:
        raw = np.asarray(values[target], dtype=np.float32)
        raw = np.maximum((raw - 1) / 254, 0)
        mask = np.asarray(supported[target], dtype=bool)
        transform = target_rows[target]["display_transform"]
        assert isinstance(transform, dict)
        lower = float(transform["lower_fraction"])
        upper = float(transform["upper_fraction"])
        gamma = float(transform["gamma"])
        intensity = np.zeros(raw.shape, dtype=np.float32)
        intensity[mask] = np.power(
            np.clip((raw[mask] - lower) / max(upper - lower, 1e-6), 0, 1),
            gamma,
        )
        channels.append(intensity)
        color = str(target_rows[target]["color"])
        palette.append([int(color[index : index + 2], 16) for index in (1, 3, 5)])
    matrix = np.stack(channels, axis=1)
    winners = np.argmax(matrix, axis=1)
    intensity = matrix[np.arange(len(matrix)), winners]
    colors = np.asarray(palette, dtype=np.uint8)[winners]
    return _write_projection_preview(
        output,
        cohort,
        "fallback-protein",
        xyz,
        colors,
        intensity,
        bounds=bounds,
    )


def _write_projection_preview(
    output: Path,
    cohort: str,
    stem: str,
    xyz: np.ndarray,
    colors: np.ndarray,
    intensity: np.ndarray,
    *,
    bounds: tuple[float, float, float, float, float, float],
) -> dict[str, object]:
    width, height = 1260, 720
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = (2, 7, 14)
    xyz = np.asarray(xyz, dtype=np.float64)
    colors = np.asarray(colors, dtype=np.uint8)
    intensity = np.asarray(intensity, dtype=np.float32)
    panels = (
        (0, 0, 840, 720, (0, 1)),
        (860, 0, 1260, 350, (0, 2)),
        (860, 370, 1260, 720, (1, 2)),
    )
    ranges = (
        (bounds[0], bounds[3]),
        (bounds[1], bounds[4]),
        (bounds[2], bounds[5]),
    )
    valid = np.isfinite(intensity) & (intensity > 0)
    shaded = np.rint(
        colors.astype(np.float32) * (0.24 + 0.76 * intensity[:, None])
    ).astype(np.uint8)
    for left, top, right, bottom, axes in panels:
        panel_width = right - left
        panel_height = bottom - top
        ax, ay = axes
        low_x, high_x = ranges[ax]
        low_y, high_y = ranges[ay]
        px = (
            left
            + 12
            + np.rint(
                (xyz[:, ax] - low_x) / max(high_x - low_x, 1e-9) * (panel_width - 25)
            ).astype(np.int64)
        )
        py = (
            bottom
            - 13
            - np.rint(
                (xyz[:, ay] - low_y) / max(high_y - low_y, 1e-9) * (panel_height - 25)
            ).astype(np.int64)
        )
        keep = valid & (px >= left) & (px < right) & (py >= top) & (py < bottom)
        chosen = np.flatnonzero(keep)
        chosen = chosen[np.argsort(intensity[chosen], kind="stable")]
        for dx, dy in ((-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)):
            x = np.clip(px[chosen] + dx, left, right - 1)
            y = np.clip(py[chosen] + dy, top, bottom - 1)
            image[y, x] = shaded[chosen]
    payload = _encode_png(image)
    digest = hashlib.sha256(payload).hexdigest()
    destination = output / "assets" / cohort / f"{stem}-{digest[:16]}.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    return {
        "asset": destination.relative_to(output).as_posix(),
        "digest": digest,
        "size_bytes": len(payload),
    }


def _encode_png(image: np.ndarray) -> bytes:
    height, width, _channels = image.shape

    def chunk(name: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + name
            + payload
            + struct.pack(">I", zlib.crc32(name + payload) & 0xFFFFFFFF)
        )

    rows = b"".join(b"\0" + image[row].tobytes() for row in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows, 9))
        + chunk(b"IEND", b"")
    )


def _median_section_spacing(sections: list[dict[str, object]]) -> float:
    z = np.asarray([float(row["z_um"]) for row in sections], dtype=np.float64)
    differences = np.diff(z)
    positive = differences[differences > 0]
    return float(np.median(positive)) if len(positive) else 5.0


def _presets() -> list[dict[str, object]]:
    return [
        {
            "id": "mechanotransduction-epithelial-stromal",
            "label": (
                "Mechanotransduction & epithelial–stromal state · "
                "YAP / CK19 / E-Cad / SMA"
            ),
            "targets": ["yap", "ck19", "ecad", "sma"],
        },
        {
            "id": "yap-mechanotransduction",
            "label": "YAP mechanotransduction · YAP",
            "targets": ["yap"],
        },
        {
            "id": "epithelial-differentiation",
            "label": "Epithelial differentiation · CK18 / CK19 / E-Cad",
            "targets": ["ck18", "ck19", "ecad"],
        },
        {
            "id": "epithelial-mesenchymal-transition",
            "label": ("Epithelial–mesenchymal transition · E-Cad / N-Cad / SMA / CK19"),
            "targets": ["ecad", "ncad", "sma", "ck19"],
        },
        {
            "id": "proliferation-mapk-response",
            "label": "Proliferation & MAPK response · Ki67 / pERK / cJun / YAP",
            "targets": ["ki67", "perk", "cjun", "yap"],
        },
        {
            "id": "oncogenic-growth",
            "label": "Oncogenic growth programs · MYC / YAP / pERK / Ki67",
            "targets": ["myc", "yap", "perk", "ki67"],
        },
        {
            "id": "reporter-epithelial-identity",
            "label": "GFP reporter & epithelial identity · GFP / CK18 / CK19 / E-Cad",
            "targets": ["gfp", "ck18", "ck19", "ecad"],
        },
    ]


def _publish_directory(staging: Path, output: Path) -> None:
    backup: Path | None = None
    if output.exists():
        backup = output.with_name(f".{output.name}.previous-{os.getpid()}")
        if backup.exists():
            shutil.rmtree(backup)
        os.replace(output, backup)
    try:
        os.replace(staging, output)
    except BaseException:
        if backup is not None and backup.exists() and not output.exists():
            os.replace(backup, output)
        raise
    if backup is not None:
        shutil.rmtree(backup, ignore_errors=True)


def _positive_integer(value: object, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")


def _validated_default_targets(
    values: Sequence[str] | None,
) -> tuple[str, ...] | None:
    if values is None:
        return None
    if isinstance(values, str):
        raise ValueError("default protein targets must be a sequence")
    normalized = tuple(str(value).strip().lower() for value in values)
    if not normalized:
        raise ValueError("default protein targets cannot be empty")
    if len(set(normalized)) != len(normalized):
        raise ValueError("default protein targets cannot contain duplicates")
    invalid = next(
        (value for value in normalized if not _NAME_RE.fullmatch(value)),
        None,
    )
    if invalid is not None:
        raise ValueError(f"invalid default protein target: {invalid!r}")
    return normalized


def _cohort_default_targets(
    selected: Mapping[str, Mapping[str, object]],
    *,
    requested: tuple[str, ...] | None,
) -> tuple[str, ...]:
    if requested is not None:
        missing = [target for target in requested if target not in selected]
        if missing:
            raise ValueError(f"default protein target is unavailable: {missing[0]}")
        return requested
    ordered = [target for target in _DEFAULT_PRESENTATION_TARGETS if target in selected]
    ordered.extend(
        target
        for target, model in sorted(selected.items())
        if model.get("status") == "promoted" and target not in ordered
    )
    ordered.extend(target for target in sorted(selected) if target not in ordered)
    return tuple(ordered[: min(4, len(ordered))])


def _histogram_percentile(histogram: np.ndarray, percentile: float) -> int:
    values = np.asarray(histogram, dtype=np.int64).copy()
    if values.shape != (256,) or np.any(values < 0):
        raise ValueError("protein display histogram must contain 256 counts")
    values[0] = 0
    total = int(values.sum())
    if total <= 0:
        return 1
    rank = int(math.floor((percentile / 100.0) * max(total - 1, 0))) + 1
    return int(np.searchsorted(np.cumsum(values), rank, side="left"))


def _display_transforms(
    histograms: Mapping[str, np.ndarray],
) -> dict[str, dict[str, float | int | str]]:
    transforms: dict[str, dict[str, float | int | str]] = {}
    for target, histogram in sorted(histograms.items()):
        lower = _histogram_percentile(histogram, _DISPLAY_LOWER_PERCENTILE)
        upper = _histogram_percentile(histogram, _DISPLAY_UPPER_PERCENTILE)
        if upper <= lower:
            lower = max(1, lower - 1)
            upper = min(255, max(lower + 1, upper + 1))
        lower_fraction = float(np.clip((lower - 1) / 254.0, 0.0, 1.0))
        upper_fraction = float(np.clip((upper - 1) / 254.0, 0.0, 1.0))
        if upper_fraction <= lower_fraction:
            upper_fraction = min(1.0, lower_fraction + 1.0 / 254.0)
        transforms[target] = {
            "method": "atlas-supported-predicted-percentile-v1",
            "lower_percentile": _DISPLAY_LOWER_PERCENTILE,
            "upper_percentile": _DISPLAY_UPPER_PERCENTILE,
            "lower_fraction": lower_fraction,
            "upper_fraction": upper_fraction,
            "gamma": _DISPLAY_GAMMA,
            "supported_cells": int(np.asarray(histogram, dtype=np.int64)[1:].sum()),
            "scope": "display only; original target OD values are unchanged",
        }
    return transforms


def _enforce_budget(root: Path, maximum: int) -> None:
    size = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    if size > maximum:
        raise ValueError(
            f"cellular protein atlas size {size} exceeds max_bytes {maximum}"
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_local_paths(value: object) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower().endswith(("_path", "_dir", "_root")):
                raise ValueError(
                    "cellular protein atlas manifest contains a path field"
                )
            _reject_local_paths(child)
    elif isinstance(value, list):
        for child in value:
            _reject_local_paths(child)
    elif isinstance(value, str) and (
        value.startswith(("/", "file://")) or re.match(r"^[A-Za-z]:[\\/]", value)
    ):
        raise ValueError(
            "cellular protein atlas manifest contains a local absolute path"
        )


def _file_inventory(root: Path) -> dict[str, dict[str, int | str]]:
    return {
        path.relative_to(root).as_posix(): {
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    }


_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <meta name="theme-color" content="#07111f">
  <title>Histopia cellular protein atlas</title>
  <link rel="icon" href="data:,">
  <link rel="stylesheet" href="protein-atlas.css">
</head>
<body>
  <div class="app" data-layer="protein">
    <main class="workspace">
      <header>
        <div class="brand"><span class="brand-mark">H</span><span><strong>Cellular protein atlas</strong><small>Histopia · section-resolved cells and predicted protein landscapes</small></span></div>
        <button id="panel-toggle" type="button" aria-expanded="false">Controls</button>
        <div class="segments layer-controls" aria-label="Atlas layer">
          <button data-layer="cells">Cell identity</button>
          <button data-layer="protein" class="active">Proteins</button>
        </div>
        <div class="segments mode-controls" aria-label="Atlas view mode">
          <button data-mode="3d" class="active">3D</button>
          <button data-mode="orthogonal">Orthogonal</button>
          <button data-mode="section">Section</button>
        </div>
        <div class="view-controls">
          <button data-view="home">Home</button><button data-view="top">Top</button><button data-view="front">Front</button><button data-view="side">Side</button>
          <button id="fullscreen" aria-label="Toggle fullscreen">Fullscreen</button>
        </div>
      </header>
      <section id="viewport" aria-label="Interactive cell-resolved protein atlas">
        <div id="view-labels" aria-hidden="true">
          <span data-panel="top"><b>XY · section plane</b><i><em></em></i></span>
          <span data-panel="front"><b>XZ · stack depth</b><i><em></em></i></span>
          <span data-panel="side"><b>YZ · stack depth</b><i><em></em></i></span>
        </div>
        <output id="scene-badge">Predicted · YAP / CK19 / E-Cad / SMA · dominant</output>
        <output id="depth-badge">Morphology-aware 3D</output>
        <output id="loading">Preparing cellular atlas…</output>
        <div id="compatibility" hidden><img alt="Static cellular atlas preview"><strong>Interactive WebGL is unavailable</strong><span>A fingerprinted preview of the selected atlas layer is shown.</span></div>
        <output id="cell-detail" hidden></output>
      </section>
      <footer><span id="scope">Predicted per cell · atlas-global within-antibody display</span><span id="cell-count"></span></footer>
    </main>
    <aside>
      <section class="dataset">
        <label>Mouse<select id="cohort" aria-label="Mouse"></select></label>
        <label>Section<select id="section" aria-label="Section"></select></label>
        <output id="provenance"></output>
      </section>
      <section class="target-section">
        <div class="section-heading"><span>Prepared protein composite</span><b id="channel-count">4 / 10</b></div>
        <label>Preset<select id="preset"></select></label>
        <div id="target-list"></div>
        <div id="target-status" aria-live="polite"></div>
      </section>
      <section class="controls display-controls">
        <div class="section-heading"><span>Display</span></div>
        <div class="segments sidebar-modes" aria-label="Projection"><button data-sidebar-mode="3d" class="active">3D</button><button data-sidebar-mode="orthogonal">Ortho</button><button data-sidebar-mode="section">Section</button></div>
        <div class="protein-only expression-block">
          <div class="segments expression-view" aria-label="Expression source"><button data-expression="predicted" class="active">Predicted</button><button data-expression="observed">Observed</button><button data-expression="residual">Residual</button></div>
          <label>Composite<select id="composite"><option value="dominant">Dominant standardized marker</option><option value="overlap">Co-expression blend</option></select></label>
          <label>Pattern focus cutoff<input id="threshold" type="range" min="0" max="50" value="25"><output id="threshold-value">25%</output></label>
        </div>
        <label>Cell footprint size<input id="glyph-size" type="range" min="55" max="220" value="100"><output id="glyph-value">1.00×</output></label>
        <label>Tissue envelope<input id="envelope-opacity" type="range" min="0" max="40" value="14"><output id="envelope-value">14%</output></label>
        <label>Section range<input id="section-range" type="range" min="0" max="100" value="100"><output id="range-value">All</output></label>
        <label class="check"><input id="detail" type="checkbox">Full cell set near selected section</label>
        <label class="check"><input id="cutaway" type="checkbox">Cutaway</label>
        <label id="cut-row" hidden>Cut position<input id="cut-position" type="range" min="-100" max="100" value="0"></label>
      </section>
      <section class="z-section">
        <div class="section-heading"><span>3D depth model</span></div>
        <div class="segments z-controls"><button data-z="physical">Physical</button><button data-z="adaptive" class="active">Morphology-aware</button><button data-z="exploded">Exploded</button></div>
        <p id="z-warning"></p>
      </section>
    </aside>
  </div>
  <script src="manifest-data.js"></script>
  <script type="importmap">{"imports":{"three":"./vendor/three.module.min.js"}}</script>
  <script type="module" src="protein-atlas.js"></script>
</body>
</html>
"""


_CSS_V2 = r"""
:root{color-scheme:dark;--atlas-bg:#02070e;--atlas-line:#1d3446;--atlas-muted:#8da9bc;--atlas-cyan:#25c7f5;--atlas-mint:#72e0b2;font-family:Inter,ui-sans-serif,system-ui,sans-serif}
*{box-sizing:border-box}html,body{width:100%;height:100%;margin:0;overflow:hidden}body{background:var(--atlas-bg)}
.app{height:100dvh;display:grid;grid-template-columns:minmax(0,1fr) clamp(350px,23vw,430px)}
.workspace{min-width:0;min-height:0;display:grid;grid-template-rows:58px minmax(0,1fr) 30px;background:var(--atlas-bg)}
header{display:flex;align-items:center;gap:8px;min-width:0;padding:0 13px;border-bottom:1px solid var(--atlas-line);background:#081725;z-index:4}
.brand{display:flex;align-items:center;gap:9px;min-width:220px;margin-right:auto}.brand-mark{display:grid;place-items:center;width:30px;height:30px;flex:0 0 auto;border-radius:8px;color:#07111f;background:linear-gradient(145deg,#72e0b2,#25c7f5);font-weight:900;box-shadow:0 0 22px #25c7f533}.brand>span:last-child{display:grid;min-width:0}.brand strong{font-size:13px;line-height:1.1}.brand small{margin-top:3px;color:var(--atlas-muted);font-size:9px;white-space:nowrap}
.segments{display:flex;gap:4px}.segments button{height:32px;padding:0 9px;white-space:nowrap}.layer-controls button{font-size:10px}.mode-controls button,.view-controls button{font-size:9px}.view-controls{display:flex;gap:4px}button,select,input{font:inherit}button{cursor:pointer}button:disabled{cursor:not-allowed}#panel-toggle{display:none}
#viewport{position:relative;min-width:0;min-height:0;overflow:hidden;background:radial-gradient(circle at 48% 44%,#0b1b2b 0,#030a13 53%,#01050a 100%)}#viewport canvas{position:relative;z-index:1;display:block;touch-action:none;outline:0}
#viewport.orthogonal:after{content:"";position:absolute;inset:0;z-index:2;pointer-events:none;background:linear-gradient(90deg,transparent calc(68% - 1px),#193246 68%,transparent calc(68% + 1px)),linear-gradient(90deg,transparent 68%,#193246 68%,#193246 100%) 0 50%/100% 1px no-repeat}
#view-labels{display:none;position:absolute;inset:0;z-index:4;pointer-events:none;grid-template-columns:68% 32%;grid-template-rows:50% 50%;color:#b9d7e8}#viewport.orthogonal #view-labels{display:grid}#view-labels span{position:relative;display:block;padding:14px 16px;font-size:9px;letter-spacing:.1em;text-transform:uppercase;text-shadow:0 1px 8px #000}#view-labels span[data-panel=top]{grid-row:1/3;grid-column:1}#view-labels span[data-panel=front]{grid-row:1;grid-column:2}#view-labels span[data-panel=side]{grid-row:2;grid-column:2}#view-labels b{font-weight:700}#view-labels i{position:absolute;left:16px;bottom:18px;height:5px;border:1px solid #d9edf6;border-top:0;font-style:normal;filter:drop-shadow(0 1px 3px #000)}#view-labels i em{position:absolute;left:50%;bottom:7px;transform:translateX(-50%);white-space:nowrap;color:#d9edf6;font-size:8px;font-style:normal;letter-spacing:.03em;text-transform:none}
#scene-badge,#depth-badge{position:absolute;top:13px;z-index:5;padding:6px 9px;border:1px solid #29475d;border-radius:999px;background:#071522dc;color:#dcecf5;font-size:9px;box-shadow:0 6px 20px #0006}#scene-badge{left:15px;max-width:calc(100% - 250px);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}#depth-badge{right:15px}#viewport.orthogonal #scene-badge,#viewport.orthogonal #depth-badge{display:none}
#loading{position:absolute;left:18px;bottom:18px;z-index:7;padding:8px 11px;border:1px solid #31506a;border-radius:7px;background:#071522eb;color:#c9dce8;font-size:10px;box-shadow:0 8px 24px #0008}
#compatibility{position:absolute;inset:0;z-index:8;align-content:center;justify-items:center;gap:9px;padding:24px;text-align:center;background:#02070e}#compatibility:not([hidden]){display:grid}#compatibility img{width:min(96%,1260px);max-height:76vh;object-fit:contain;border:1px solid #1e3a4e;border-radius:10px;box-shadow:0 20px 70px #000b}#compatibility strong{color:#eef7ff;font-size:13px}#compatibility span{color:var(--atlas-muted);font-size:10px}
#cell-detail{position:absolute;right:16px;top:52px;z-index:7;width:min(310px,calc(100% - 32px));padding:10px 12px;border:1px solid #4c7189;border-radius:8px;background:#06131fea;color:#dcecf5;font-size:9px;line-height:1.55;box-shadow:0 12px 30px #0009}#cell-detail strong{display:block;margin-bottom:3px;color:#fff;font-size:10px}#cell-detail .detail-values{display:grid;grid-template-columns:1fr auto;gap:2px 12px;margin-top:6px}#cell-detail i{display:inline-block;width:7px;height:7px;margin-right:5px;border-radius:50%}
footer{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:0 13px;border-top:1px solid var(--atlas-line);background:#07131f;color:var(--atlas-muted);font-size:9px}
aside{min-width:0;min-height:0;overflow:auto;border-left:1px solid #dbe3ec;background:#fff}aside>section{padding:11px 13px;border-bottom:1px solid #dbe3ec}
.dataset{display:grid;grid-template-columns:1fr 1fr;gap:8px}.dataset label{display:grid;gap:4px;color:#4b5563;font-size:10px}.dataset select{width:100%}.dataset output{grid-column:1/-1;color:#64748b;font-size:9px;line-height:1.5}
.section-heading{display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;color:#64748b;font-size:9px;font-weight:800;letter-spacing:.08em;text-transform:uppercase}.section-heading b{color:#176b45;font-size:8px}
.target-section>label,.controls>label,.expression-block>label{display:flex;align-items:center;gap:8px;min-height:26px;color:#4b5563;font-size:10px}.target-section>label{margin-bottom:8px}.target-section>label select,.controls label>select,.controls label>input[type=range],.expression-block label>select,.expression-block label>input[type=range]{flex:1;min-width:0}
.check input{width:14px;height:14px;margin:0}
#target-list{display:grid;grid-template-columns:1fr 1fr;gap:5px;max-height:220px;overflow:auto;padding-right:2px}.target{display:flex;align-items:center;gap:6px;min-width:0;padding:6px;border:1px solid #dbe3ec;border-radius:7px;background:#f9fafb;color:#111827;font-size:9px}.target:hover{border-color:#94a3b8;background:#f3f6fa}.target input{width:13px;height:13px;margin:0}.target i{width:8px;height:8px;flex:0 0 auto;border-radius:50%;box-shadow:0 0 7px currentColor}.target span{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
#target-status{display:flex;flex-wrap:wrap;gap:5px;min-height:28px;margin-top:8px;padding:7px;border-radius:7px;background:#0f172a;color:#d9eef9;font-size:8px;line-height:1.35}.status-chip{display:inline-flex;align-items:center;gap:4px;padding:3px 6px;border:1px solid #334155;border-radius:999px;background:#172033}.status-chip i{width:6px;height:6px;border-radius:50%}.status-message{align-self:center;color:#c9d9e4}
.controls{display:grid;gap:8px}.controls .section-heading{margin:0}.sidebar-modes{display:none}.sidebar-modes button,.expression-view button{flex:1;font-size:9px}.expression-block{display:grid;gap:8px}.expression-view{width:100%}.controls output{width:38px;text-align:right;color:#334155;font-size:9px}.z-controls{width:100%}.z-controls button{flex:1;font-size:8px}.z-section p,.science-note p{margin:8px 0 0;color:#64748b;font-size:9px;line-height:1.55}.science-note strong{color:#334155;font-size:9px;letter-spacing:.08em;text-transform:uppercase}.science-note p+p{color:#7890a1}
.app[data-layer=cells] .protein-only{display:none}.app[data-layer=cells] .target-section{background:linear-gradient(#fff,#fbfdff)}
@media(max-width:1320px){.app{grid-template-columns:minmax(0,1fr) 350px}.brand small{display:none}.view-controls button:not(#fullscreen):not([data-view=home]){display:none}.mode-controls button{padding:0 6px}}
@media(max-height:760px){aside>section{padding:8px 10px}#target-list{max-height:145px}.science-note{display:none}.controls{gap:5px}.target{padding:4px}.workspace{grid-template-rows:54px minmax(0,1fr) 28px}}
@media(max-width:780px){.app{display:block;position:relative}.workspace{height:100dvh;grid-template-rows:52px minmax(0,1fr) 28px}header{padding:0 8px}.brand{min-width:0;margin-right:auto}.brand strong{white-space:nowrap}.brand-mark{width:27px;height:27px}.mode-controls,.view-controls{display:none}.layer-controls button{padding:0 7px;font-size:8px}#panel-toggle{display:block;padding:0 8px;font-size:8px}aside{display:none;position:absolute;inset:52px 0 0;z-index:20;border:0;border-top:1px solid #dbe3ec;background:#fffffff7}.app.panel-open aside{display:block}.dataset{position:sticky;top:0;z-index:2;background:#fff}.sidebar-modes{display:flex}#target-list{max-height:none}footer #scope{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}footer #cell-count{display:none}#view-labels{grid-template-columns:66% 34%}#viewport.orthogonal:after{background:linear-gradient(90deg,transparent calc(66% - 1px),#193246 66%,transparent calc(66% + 1px)),linear-gradient(90deg,transparent 66%,#193246 66%,#193246 100%) 0 50%/100% 1px no-repeat}}
@media(max-width:460px){.brand-mark{display:none}.brand strong{font-size:10px}.layer-controls button{max-width:72px;overflow:hidden;text-overflow:ellipsis}#target-list{grid-template-columns:1fr}#depth-badge{top:auto;right:10px;bottom:10px}#scene-badge{top:10px;left:10px}}
"""

_LIGHT_PANEL_OVERRIDES_V2 = r"""
/* Keep the scientific viewport dark while matching the Home-page controls. */
header{color:#eef7ff!important;background:#081725!important;border-color:#1d3446!important}header strong{color:#eef7ff!important}header .brand small{color:#8da9bc!important}
header button{color:#111827;background:#fff;border-color:#cfd8e3}header button:hover:not(:disabled){color:#111827;background:#f8fafc;border-color:#94a3b8}header .segments button.active{color:#fff;background:#1d4ed8;border-color:#2563eb;box-shadow:inset 0 -3px #72e0b2}
aside{color:#111827!important;background:#fff!important}aside button,aside select{color:#111827;background:#fff;border-color:#cfd8e3}aside .segments button.active{color:#fff;background:#1d4ed8;border-color:#2563eb}
footer,footer #scope{color:#8da9bc!important;background:#07131f!important;border-color:#1d3446!important}
"""
