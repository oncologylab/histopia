"""End-to-end adaptive reconstruction on compact semantic fields."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from histopia._atomic import (
    write_binary_atomic,
    write_json_atomic,
    write_json_atomic_if_changed,
)
from histopia.semantic._approval import validate_semantic_approval
from histopia.semantic._registration_binding import (
    validate_semantic_registration_binding,
)
from histopia.semantic._result_validation import validate_semantic_result
from histopia.topology._benchmark import run_holdout_benchmark
from histopia.topology._config import TopologyConfig
from histopia.topology._interpolate import (
    interpolate_pair,
    observed_plane,
    smooth_displacement_field,
)
from histopia.topology._model import (
    GapDecision,
    ObservedSection,
    PairEvidence,
    ReconstructedPlane,
    infer_morphology_gap_decisions,
    pair_evidence,
)
from histopia.topology._result import (
    validate_topology_result,
    write_topology_result,
)
from histopia.topology._volume import (
    benchmark_envelope_methods,
    load_registered_mask_stack,
    reconstruct_dense_volume,
    write_connected_meshes,
    write_dense_volume,
)

Progress = Callable[[str], None]
TOPOLOGY_ALGORITHM_VERSION = 9
_LINK_COVERAGE_GATE = 0.05
_LINK_CONFIDENCE_GATE = 0.45
_VIEWER_FACE_TARGET = 200_000


def preflight_topology(config: TopologyConfig) -> dict[str, object]:
    """Validate source identities and return a portable reconstruction contract."""

    registration_root = config.registration_run.expanduser().resolve()
    semantic_root = config.semantic_run.expanduser().resolve()
    registration_path = registration_root / "registration_result.json"
    semantic_path = semantic_root / "semantic_result.json"
    semantic = validate_semantic_result(semantic_root)
    binding = validate_semantic_registration_binding(
        registration_root,
        semantic_root,
        semantic_payload=semantic,
    )
    approval: dict[str, object] | None = None
    if config.require_approvals:
        semantic_approval = validate_semantic_approval(semantic_root)
        if binding.registration_approval is None:
            from histopia.registration._approval import validate_registration_approval

            registration_approval = validate_registration_approval(registration_root)
        else:
            registration_approval = binding.registration_approval
        approval = {
            "semantic_fingerprint": semantic_approval.fingerprint,
            "semantic_reviewer": semantic_approval.reviewer,
            "registration_result_sha256": (
                registration_approval.registration_result_sha256
            ),
        }
    selected_k = int(
        semantic["selected_k"]
        if semantic.get("selected_k") is not None
        else semantic["primary_clusters"]
    )
    slides = semantic.get("slides")
    if not isinstance(slides, list) or len(slides) < 2:
        raise ValueError("topology reconstruction requires at least two sections")
    slide_ids = [str(row.get("id", "")) for row in slides if isinstance(row, dict)]
    if len(slide_ids) != len(slides) or any(not value for value in slide_ids):
        raise ValueError("semantic sections must have unique non-empty IDs")
    if len(set(slide_ids)) != len(slide_ids):
        raise ValueError("semantic sections contain duplicate IDs")
    manifest = _load_z_manifest(config, tuple(slide_ids))
    core = {
        "schema_version": 2,
        "algorithm_version": TOPOLOGY_ALGORITHM_VERSION,
        "registration_result_sha256": _sha256_file(registration_path),
        "semantic_result_sha256": _sha256_file(semantic_path),
        "semantic_fingerprint": semantic["fingerprint"],
        "semantic_approval": approval,
        "selected_k": selected_k,
        "slide_ids": slide_ids,
        "section_thickness_um": config.section_thickness_um,
        "z_manifest_sha256": (
            _sha256_file(config.z_manifest) if config.z_manifest is not None else None
        ),
        "z_positions_um": list(manifest[0]) if manifest is not None else None,
        "z_manifest_source": manifest[1] if manifest is not None else None,
        "controls": {
            "calibration_max_span": config.calibration_max_span,
            "max_inferred_missing": config.max_inferred_missing,
            "reconstruction_samples_per_interval": (
                config.reconstruction_samples_per_interval
            ),
            "envelope_max_xy_dim_px": config.envelope_max_xy_dim_px,
        },
    }
    return {**core, "fingerprint": _json_sha256(core)}


def build_topology(
    config: TopologyConfig,
    *,
    progress: Progress | None = None,
) -> Path:
    """Build a sealed selected-K semantic volume and surface meshes."""

    report = progress or (lambda _message: None)
    output = config.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    report("Validating approved registration and semantic inputs")
    preflight = preflight_topology(config)
    preflight_path = output / "preflight.json"
    write_json_atomic_if_changed(preflight_path, preflight)
    existing = output / "topology_result.json"
    if existing.is_file():
        try:
            current = validate_topology_result(output)
            if current.get("preflight_fingerprint") == preflight["fingerprint"]:
                report("Reusing fingerprint-identical topology result")
                return existing
        except (OSError, ValueError):
            pass

    semantic_root = config.semantic_run.expanduser().resolve()
    semantic = validate_semantic_result(semantic_root)
    selected_k = int(preflight["selected_k"])
    report("Loading selected-K semantic labels and topology links")
    raw_sections, patch_width_um = _load_sparse_sections(
        semantic_root,
        semantic,
        selected_k,
    )
    report("Resampling approved tissue masks into reference physical coordinates")
    mask_stack = load_registered_mask_stack(
        config.registration_run,
        tuple(str(row["id"]) for row in raw_sections),
        target_spacing_um=0.5 * patch_width_um,
        max_xy_dim_px=config.envelope_max_xy_dim_px,
        require_review=config.require_approvals,
    )
    origin = mask_stack.origin_um_xy
    grid_shape = tuple(int(value) for value in mask_stack.masks.shape[1:])
    sections = tuple(
        _apply_registered_mask(
            _rasterize_section(
                row,
                origin_um_xy=origin,
                grid_shape=grid_shape,
                spacing_um=mask_stack.spacing_um,
                classes=selected_k,
                acceptance_um=0.8 * patch_width_um,
            ),
            mask_stack.masks[index],
        )
        for index, row in enumerate(raw_sections)
    )
    links = _load_links(semantic_root, semantic)
    evidence = tuple(
        replace(
            pair_evidence(
                sections[index],
                sections[index + 1],
                source_indices=link["source_indices"],
                target_indices=link["target_indices"],
                confidence=link["confidence"],
                source_xy=link["source_um_xy"],
                target_xy=link["target_um_xy"],
                source_patch_count=len(raw_sections[index]["labels"]),
                target_patch_count=len(raw_sections[index + 1]["labels"]),
                patch_width_um=patch_width_um,
            ),
            source_section=index,
            target_section=index + 1,
        )
        for index, link in enumerate(links)
    )
    report("Running endpoint-only held-out section benchmark")
    benchmark = _benchmark_payload(
        sections,
        evidence,
        preflight_fingerprint=str(preflight["fingerprint"]),
        origin_um_xy=origin,
        spacing_um=mask_stack.spacing_um,
        max_hidden_sections=config.max_inferred_missing,
    )
    benchmark_path = output / "benchmark.json"
    write_json_atomic(benchmark_path, benchmark)
    manifest_z = preflight.get("z_positions_um")
    if manifest_z is not None:
        decisions = _manifest_decisions(
            tuple(float(value) for value in manifest_z),
            evidence,
            thickness_um=config.section_thickness_um,
        )
        z_positions = tuple(float(value) for value in manifest_z)
        z_source = (
            "manifest_measured"
            if preflight.get("z_manifest_source") == "measured"
            else "manifest_assumed"
        )
    else:
        if benchmark["summary"]["supports_gap_inference"]:
            decisions = infer_morphology_gap_decisions(
                sections,
                evidence,
                max_span=config.calibration_max_span,
                max_inferred_missing=config.max_inferred_missing,
            )
            z_source = "morphology_inferred"
        else:
            decisions = _uniform_assumed_decisions(evidence)
            z_source = "uniform_assumed_after_failed_gap_calibration"
        z_positions = _inferred_z_positions(
            decisions,
            thickness_um=config.section_thickness_um,
        )
    report(
        "Gap decisions: "
        f"{sum(row.status == 'inferred' for row in decisions)} inferred, "
        f"{sum(row.status == 'unresolved' for row in decisions)} unresolved"
    )
    planes = _reconstruct_planes(
        sections,
        links,
        decisions,
        z_positions,
        origin_um_xy=origin,
        spacing_um=mask_stack.spacing_um,
    )
    report(
        f"Writing {sum(plane.observed for plane in planes)} observed and "
        f"{sum(not plane.observed for plane in planes)} virtual semantic planes"
    )
    plane_rows = _write_planes(output, planes)
    report("Benchmarking registered-mask envelope reconstruction")
    envelope_qc = benchmark_envelope_methods(
        mask_stack.masks,
        sections,
        z_positions,
        origin_um_xy=origin,
        spacing_um=mask_stack.spacing_um,
    )
    selected_envelope_method = str(envelope_qc["selected_method"])
    report(f"Envelope method: {selected_envelope_method} ({envelope_qc['status']})")
    section_segments = _section_segments(decisions)
    report("Reconstructing dense numerical topology fields")
    dense = reconstruct_dense_volume(
        mask_stack.masks,
        sections,
        links,
        z_positions,
        tuple(row.intervals for row in decisions),
        section_segments,
        origin_um_xy=origin,
        spacing_um=mask_stack.spacing_um,
        section_thickness_um=config.section_thickness_um,
        samples_per_interval=config.reconstruction_samples_per_interval,
        envelope_method=selected_envelope_method,
    )
    dense_row = write_dense_volume(output, dense)
    report("Extracting connected envelope and semantic-region surfaces")
    envelope_row, region_rows, class_rows, uncertainty_row = write_connected_meshes(
        output,
        palette=tuple(str(value) for value in semantic["palette"][:selected_k]),
        origin_um_xy=origin,
        spacing_um=mask_stack.spacing_um,
        z_spacing_um=(
            config.section_thickness_um / config.reconstruction_samples_per_interval
        ),
        source_patch_width_um=patch_width_um,
        section_thickness_um=config.section_thickness_um,
        volume=dense,
    )
    core = {
        "schema_version": 2,
        "algorithm_version": TOPOLOGY_ALGORITHM_VERSION,
        "preflight": preflight_path.relative_to(output).as_posix(),
        "benchmark": benchmark_path.relative_to(output).as_posix(),
        "benchmark_fingerprint": benchmark["fingerprint"],
        "preflight_fingerprint": preflight["fingerprint"],
        "registration_result_sha256": preflight["registration_result_sha256"],
        "semantic_fingerprint": semantic["fingerprint"],
        "selected_k": selected_k,
        "palette": semantic["palette"][:selected_k],
        "section_thickness_um": config.section_thickness_um,
        "z_source": z_source,
        "source_patch_width_um": patch_width_um,
        "reference_grid": {
            "origin_um_xy": list(origin),
            "shape_rc": list(grid_shape),
            "spacing_um": mask_stack.spacing_um,
        },
        "observed_section_count": len(sections),
        "virtual_section_count": sum(not plane.observed for plane in planes),
        "segment_count": 1 + sum(row.status == "unresolved" for row in decisions),
        "pair_evidence": [asdict(row) for row in evidence],
        "gap_decisions": [asdict(row) for row in decisions],
        "planes": plane_rows,
        "envelope": envelope_row,
        "semantic_regions": region_rows,
        "uncertainty": uncertainty_row,
        "classes": class_rows,
        "reconstruction_grid": {
            **dense_row,
            "origin_um_xy": list(origin),
            "spacing_um_xy": mask_stack.spacing_um,
            "spacing_um_z": (
                config.section_thickness_um / config.reconstruction_samples_per_interval
            ),
            "samples_per_physical_interval": (
                config.reconstruction_samples_per_interval
            ),
            "sample_semantics": "numerical_reconstruction_not_biological_sections",
        },
        "reconstruction_qc": envelope_qc,
        "registered_masks": list(mask_stack.provenance),
        "measurement": {
            "z_geometry": (
                "measured" if z_source == "manifest_measured" else "estimated"
            ),
            "membership": "correspondence_interpolated_selected_k_soft_field",
            "reconstruction_uncertainty": "unit_interval",
            "full_resolution_images_synthesized": False,
            "viewer_component_filter_changes_measurement": False,
        },
    }
    result = write_topology_result(output, core)
    validate_topology_result(output)
    report(f"Wrote sealed topology result: {result}")
    return result


def benchmark_topology(
    config: TopologyConfig,
    *,
    progress: Progress | None = None,
) -> Path:
    """Run endpoint-only held-out validation without producing a volume."""

    report = progress or (lambda _message: None)
    output = config.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    preflight = preflight_topology(config)
    write_json_atomic_if_changed(output / "preflight.json", preflight)
    semantic_root = config.semantic_run.expanduser().resolve()
    semantic = validate_semantic_result(semantic_root)
    selected_k = int(preflight["selected_k"])
    report("Loading selected-K sections and adjacent correspondence evidence")
    raw_sections, patch_width_um = _load_sparse_sections(
        semantic_root,
        semantic,
        selected_k,
    )
    origin, grid_shape = _common_reference_grid(raw_sections, patch_width_um)
    sections = tuple(
        _rasterize_section(
            row,
            origin_um_xy=origin,
            grid_shape=grid_shape,
            spacing_um=patch_width_um,
            classes=selected_k,
        )
        for row in raw_sections
    )
    links = _load_links(semantic_root, semantic)
    evidence = tuple(
        replace(
            pair_evidence(
                sections[index],
                sections[index + 1],
                source_indices=link["source_indices"],
                target_indices=link["target_indices"],
                confidence=link["confidence"],
                source_xy=link["source_um_xy"],
                target_xy=link["target_um_xy"],
                source_patch_count=len(raw_sections[index]["labels"]),
                target_patch_count=len(raw_sections[index + 1]["labels"]),
                patch_width_um=patch_width_um,
            ),
            source_section=index,
            target_section=index + 1,
        )
        for index, link in enumerate(links)
    )
    report("Running one-to-three-section endpoint holdouts")
    benchmark = _benchmark_payload(
        sections,
        evidence,
        preflight_fingerprint=str(preflight["fingerprint"]),
        origin_um_xy=origin,
        spacing_um=patch_width_um,
        max_hidden_sections=config.max_inferred_missing,
    )
    path = output / "benchmark.json"
    write_json_atomic(path, benchmark)
    report(f"Wrote topology benchmark: {path}")
    return path


def _benchmark_payload(
    sections: tuple[ObservedSection, ...],
    evidence: tuple[PairEvidence, ...],
    *,
    preflight_fingerprint: str,
    origin_um_xy: tuple[float, float],
    spacing_um: float,
    max_hidden_sections: int,
) -> dict[str, object]:
    core = {
        **run_holdout_benchmark(
            sections,
            evidence,
            origin_um_xy=origin_um_xy,
            spacing_um=spacing_um,
            max_hidden_sections=max_hidden_sections,
        ),
        "preflight_fingerprint": preflight_fingerprint,
    }
    return {**core, "fingerprint": _json_sha256(core)}


def _load_sparse_sections(
    root: Path,
    semantic: dict[str, object],
    selected_k: int,
) -> tuple[tuple[dict[str, object], ...], float]:
    rows: list[dict[str, object]] = []
    widths: list[float] = []
    for slide in semantic["slides"]:
        path = root / slide["labels"][str(selected_k)]
        with np.load(path, allow_pickle=False) as data:
            labels = np.asarray(data["labels"], dtype=np.int16)
            coordinates = np.asarray(data["reference_um_xy"], dtype=np.float64)
            tissue = np.asarray(data["tissue_fraction"], dtype=np.float32)
            patch_size = int(data["patch_size_px"])
            analysis_mpp = float(data["analysis_mpp"])
        if (
            labels.ndim != 1
            or coordinates.shape != (len(labels), 2)
            or tissue.shape != labels.shape
            or not len(labels)
            or np.any(labels < 0)
            or np.any(labels >= selected_k)
            or not np.all(np.isfinite(coordinates))
        ):
            raise ValueError(f"invalid selected-K labels for {slide['id']}")
        widths.append(patch_size * analysis_mpp)
        rows.append(
            {
                "id": str(slide["id"]),
                "labels": labels,
                "coordinates": coordinates,
                "tissue_fraction": tissue,
            }
        )
    if not np.allclose(widths, widths[0], rtol=1e-6, atol=1e-6):
        raise ValueError("topology sections must use one physical patch width")
    return tuple(rows), float(widths[0])


def _common_reference_grid(
    rows: tuple[dict[str, object], ...],
    spacing_um: float,
) -> tuple[tuple[float, float], tuple[int, int]]:
    coordinates = np.concatenate(
        [np.asarray(row["coordinates"], dtype=float) for row in rows]
    )
    minimum = coordinates.min(axis=0) - 0.5 * spacing_um
    maximum = coordinates.max(axis=0) + 0.5 * spacing_um
    width, height = np.ceil((maximum - minimum) / spacing_um).astype(int) + 1
    return (
        (float(minimum[0]), float(minimum[1])),
        (int(height), int(width)),
    )


def _rasterize_section(
    row: dict[str, object],
    *,
    origin_um_xy: tuple[float, float],
    grid_shape: tuple[int, int],
    spacing_um: float,
    classes: int,
    acceptance_um: float | None = None,
) -> ObservedSection:
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise RuntimeError("topology reconstruction requires scipy") from exc
    labels = np.asarray(row["labels"], dtype=np.int16)
    coordinates = np.asarray(row["coordinates"], dtype=float)
    sparse_tissue = np.asarray(row["tissue_fraction"], dtype=np.float32)
    grid_rows, grid_cols = np.indices(grid_shape)
    centers = np.column_stack(
        [
            origin_um_xy[0] + grid_cols.ravel() * spacing_um,
            origin_um_xy[1] + grid_rows.ravel() * spacing_um,
        ]
    )
    distances, nearest = cKDTree(coordinates).query(centers, k=1)
    accepted = distances <= (
        float(acceptance_um) if acceptance_um is not None else 0.80 * spacing_um
    )
    dense_labels = np.full(centers.shape[0], -1, dtype=np.int16)
    dense_tissue = np.zeros(centers.shape[0], dtype=np.float32)
    dense_labels[accepted] = labels[nearest[accepted]]
    dense_tissue[accepted] = sparse_tissue[nearest[accepted]]
    dense_labels = dense_labels.reshape(grid_shape)
    dense_tissue = dense_tissue.reshape(grid_shape)
    support = dense_labels >= 0
    membership = np.zeros((classes, *grid_shape), dtype=np.float32)
    for class_index in range(classes):
        membership[class_index] = dense_labels == class_index
    return ObservedSection(
        slide_id=str(row["id"]),
        labels=dense_labels,
        membership=membership,
        support=support,
        tissue_fraction=dense_tissue,
        sparse_labels=labels,
    )


def _apply_registered_mask(
    section: ObservedSection,
    mask: np.ndarray,
) -> ObservedSection:
    """Use approved morphology support without changing sparse semantic evidence."""

    support = np.asarray(mask, dtype=bool)
    if support.shape != section.support.shape:
        raise ValueError("registered mask and semantic grid shapes differ")
    membership = np.asarray(section.membership, dtype=np.float32).copy()
    membership[:, ~support] = 0
    labels = np.asarray(section.labels, dtype=np.int16).copy()
    labels[~support] = -1
    return replace(
        section,
        labels=labels,
        membership=membership,
        support=support,
        tissue_fraction=support.astype(np.float32),
    )


def _section_segments(
    decisions: tuple[GapDecision, ...],
) -> tuple[int, ...]:
    segment = 0
    values = [segment]
    for decision in decisions:
        if decision.status == "unresolved":
            segment += 1
        values.append(segment)
    return tuple(values)


def _load_links(
    root: Path,
    semantic: dict[str, object],
) -> tuple[dict[str, np.ndarray], ...]:
    rows = semantic.get("topology_pairs")
    slides = semantic["slides"]
    if not isinstance(rows, list) or len(rows) != len(slides) - 1:
        raise ValueError("semantic result must contain every adjacent topology pair")
    result: list[dict[str, np.ndarray]] = []
    for index, row in enumerate(rows):
        if (
            int(row["source_section"]) != index
            or int(row["target_section"]) != index + 1
        ):
            raise ValueError("semantic topology pairs are not ordered")
        with np.load(root / row["artifact"], allow_pickle=False) as data:
            result.append(
                {
                    name: np.asarray(data[name]).copy()
                    for name in (
                        "source_indices",
                        "target_indices",
                        "source_um_xy",
                        "target_um_xy",
                        "confidence",
                    )
                }
            )
    return tuple(result)


def _manifest_decisions(
    z_positions: tuple[float, ...],
    evidence: tuple[PairEvidence, ...],
    *,
    thickness_um: float,
) -> tuple[GapDecision, ...]:
    decisions: list[GapDecision] = []
    for index, item in enumerate(evidence):
        ratio = (z_positions[index + 1] - z_positions[index]) / thickness_um
        intervals = int(round(ratio))
        supported = (
            item.correspondence_coverage >= _LINK_COVERAGE_GATE
            and item.median_confidence >= _LINK_CONFIDENCE_GATE
        )
        status = "manifest" if intervals == 1 or supported else "unresolved"
        reasons = (
            ()
            if status != "unresolved"
            else ("manifest_gap_has_insufficient_correspondence",)
        )
        decisions.append(
            GapDecision(
                index,
                index + 1,
                intervals if status != "unresolved" else 1,
                max(0, intervals - 1) if status != "unresolved" else 0,
                status,
                1.0 if supported else 0.0,
                item.score,
                reasons,
            )
        )
    return tuple(decisions)


def _uniform_assumed_decisions(
    evidence: tuple[PairEvidence, ...],
) -> tuple[GapDecision, ...]:
    return tuple(
        GapDecision(
            source_section=index,
            target_section=index + 1,
            intervals=1,
            missing_sections=0,
            status="assumed",
            confidence=0.0,
            score=item.score,
            reasons=("holdout_gap_calibration_failed",),
        )
        for index, item in enumerate(evidence)
    )


def _inferred_z_positions(
    decisions: tuple[GapDecision, ...],
    *,
    thickness_um: float,
) -> tuple[float, ...]:
    positions = [0.0]
    for decision in decisions:
        positions.append(positions[-1] + decision.intervals * thickness_um)
    return tuple(positions)


def _reconstruct_planes(
    sections: tuple[ObservedSection, ...],
    links: tuple[dict[str, np.ndarray], ...],
    decisions: tuple[GapDecision, ...],
    z_positions: tuple[float, ...],
    *,
    origin_um_xy: tuple[float, float],
    spacing_um: float,
) -> tuple[ReconstructedPlane, ...]:
    planes: list[ReconstructedPlane] = [
        observed_plane(sections[0], z_um=z_positions[0], segment=0, section_index=0)
    ]
    segment = 0
    for index, decision in enumerate(decisions):
        if decision.status == "unresolved":
            segment += 1
        elif decision.missing_sections:
            field, field_confidence = smooth_displacement_field(
                sections[index].support.shape,
                source_xy=links[index]["source_um_xy"],
                target_xy=links[index]["target_um_xy"],
                origin_um_xy=origin_um_xy,
                spacing_um=spacing_um,
            )
            for missing_index in range(1, decision.intervals):
                fraction = missing_index / decision.intervals
                planes.append(
                    interpolate_pair(
                        sections[index],
                        sections[index + 1],
                        fraction=fraction,
                        z_um=(
                            z_positions[index]
                            + fraction * (z_positions[index + 1] - z_positions[index])
                        ),
                        segment=segment,
                        source_section=index,
                        target_section=index + 1,
                        displacement_rc=field,
                        flow_confidence=field_confidence,
                    )
                )
        planes.append(
            observed_plane(
                sections[index + 1],
                z_um=z_positions[index + 1],
                segment=segment,
                section_index=index + 1,
            )
        )
    return tuple(planes)


def _write_planes(
    root: Path,
    planes: tuple[ReconstructedPlane, ...],
) -> list[dict[str, object]]:
    directory = root / "planes"
    directory.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for index, plane in enumerate(planes):
        path = directory / f"{index + 1:04d}.npz"
        _savez_atomic(
            path,
            membership=plane.membership.astype(np.float16),
            labels=plane.labels.astype(np.int16),
            support=plane.support.astype(np.uint8),
            uncertainty=plane.uncertainty.astype(np.float16),
        )
        rows.append(
            {
                "index": index,
                "z_um": plane.z_um,
                "segment": plane.segment,
                "source_section": plane.source_section,
                "target_section": plane.target_section,
                "fraction": plane.fraction,
                "observed": plane.observed,
                "slide_id": plane.slide_id,
                "artifact": path.relative_to(root).as_posix(),
                "support_fraction": float(np.mean(plane.support)),
                "median_uncertainty": float(
                    np.median(plane.uncertainty[plane.support])
                    if np.any(plane.support)
                    else 1.0
                ),
            }
        )
    return rows


def _write_meshes(
    root: Path,
    planes: tuple[ReconstructedPlane, ...],
    *,
    class_count: int,
    palette: tuple[str, ...],
    origin_um_xy: tuple[float, float],
    spacing_um: float,
    thickness_um: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    try:
        from skimage.measure import marching_cubes, mesh_surface_area
    except ImportError as exc:
        raise RuntimeError(
            "topology surface extraction requires the 'topology' extra"
        ) from exc
    mesh_dir = root / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    segments = sorted({plane.segment for plane in planes})
    mesh_rows: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []
    for class_index in range(class_count):
        total_volume_um3 = 0.0
        total_area_um2 = 0.0
        inferred_volume_um3 = 0.0
        component_count = 0
        for segment in segments:
            selected = tuple(plane for plane in planes if plane.segment == segment)
            occupancy = np.stack(
                [plane.labels == class_index for plane in selected]
            ).astype(np.uint8)
            total_volume_um3 += _trapezoid_volume(
                occupancy,
                tuple(plane.z_um for plane in selected),
                spacing_um=spacing_um,
                fallback_thickness_um=thickness_um,
            )
            inferred_volume_um3 += _inferred_plane_volume(
                occupancy,
                selected,
                spacing_um=spacing_um,
                thickness_um=thickness_um,
            )
            if not np.any(occupancy):
                continue
            if len(occupancy) == 1:
                occupancy = np.repeat(occupancy, 2, axis=0)
            padded = np.pad(occupancy, 1, mode="constant")
            vertices_zyx, faces, _, _ = marching_cubes(
                padded,
                level=0.5,
                spacing=(thickness_um, spacing_um, spacing_um),
                allow_degenerate=False,
            )
            first_z = selected[0].z_um
            vertices = np.column_stack(
                [
                    origin_um_xy[0] - spacing_um + vertices_zyx[:, 2],
                    origin_um_xy[1] - spacing_um + vertices_zyx[:, 1],
                    first_z - thickness_um + vertices_zyx[:, 0],
                ]
            ).astype(np.float32)
            faces = np.asarray(faces, dtype=np.uint32)
            area = float(mesh_surface_area(vertices, faces))
            total_area_um2 += area
            component_count += _component_count(occupancy)
            stem = f"class-{class_index:02d}-segment-{segment:02d}"
            npz_path = mesh_dir / f"{stem}.npz"
            bin_path = mesh_dir / f"{stem}.bin"
            _savez_atomic(
                npz_path,
                vertices_um_xyz=vertices,
                faces=faces,
                class_index=np.int16(class_index),
                segment=np.int16(segment),
            )
            viewer_vertices = vertices
            viewer_faces = faces
            viewer_downsample = 1
            if len(faces) > _VIEWER_FACE_TARGET:
                viewer_downsample = min(
                    4,
                    max(2, int(math.ceil(math.sqrt(len(faces) / _VIEWER_FACE_TARGET)))),
                )
                viewer_occupancy = occupancy[
                    :, ::viewer_downsample, ::viewer_downsample
                ]
                viewer_padded = np.pad(viewer_occupancy, 1, mode="constant")
                viewer_zyx, viewer_faces, _, _ = marching_cubes(
                    viewer_padded,
                    level=0.5,
                    spacing=(
                        thickness_um,
                        spacing_um * viewer_downsample,
                        spacing_um * viewer_downsample,
                    ),
                    allow_degenerate=False,
                )
                viewer_spacing = spacing_um * viewer_downsample
                viewer_vertices = np.column_stack(
                    [
                        origin_um_xy[0] - viewer_spacing + viewer_zyx[:, 2],
                        origin_um_xy[1] - viewer_spacing + viewer_zyx[:, 1],
                        first_z - thickness_um + viewer_zyx[:, 0],
                    ]
                ).astype(np.float32)
                viewer_faces = np.asarray(viewer_faces, dtype=np.uint32)
            _write_mesh_binary(bin_path, viewer_vertices, viewer_faces)
            mesh_rows.append(
                {
                    "class_index": class_index,
                    "segment": segment,
                    "color": palette[class_index],
                    "vertex_count": len(vertices),
                    "face_count": len(faces),
                    "viewer_face_count": len(viewer_faces),
                    "viewer_downsample_xy": viewer_downsample,
                    "surface_area_mm2": area / 1_000_000.0,
                    "artifact": npz_path.relative_to(root).as_posix(),
                    "viewer_asset": bin_path.relative_to(root).as_posix(),
                }
            )
        class_rows.append(
            {
                "class_index": class_index,
                "color": palette[class_index],
                "estimated_volume_mm3": total_volume_um3 / 1_000_000_000.0,
                "inferred_volume_mm3": inferred_volume_um3 / 1_000_000_000.0,
                "inferred_volume_fraction": (
                    inferred_volume_um3 / total_volume_um3
                    if total_volume_um3 > 0
                    else 0.0
                ),
                "surface_area_mm2": total_area_um2 / 1_000_000.0,
                "component_count": component_count,
            }
        )
    return mesh_rows, class_rows


def _trapezoid_volume(
    occupancy: np.ndarray,
    z_positions: tuple[float, ...],
    *,
    spacing_um: float,
    fallback_thickness_um: float,
) -> float:
    areas = np.count_nonzero(occupancy, axis=(1, 2)) * spacing_um**2
    if len(areas) == 1:
        return float(areas[0] * fallback_thickness_um)
    return float(
        sum(
            0.5
            * (areas[index] + areas[index + 1])
            * (z_positions[index + 1] - z_positions[index])
            for index in range(len(areas) - 1)
        )
    )


def _inferred_plane_volume(
    occupancy: np.ndarray,
    planes: tuple[ReconstructedPlane, ...],
    *,
    spacing_um: float,
    thickness_um: float,
) -> float:
    return float(
        sum(
            np.count_nonzero(occupancy[index]) * spacing_um**2 * thickness_um
            for index, plane in enumerate(planes)
            if not plane.observed
        )
    )


def _component_count(occupancy: np.ndarray) -> int:
    from scipy.ndimage import label

    return int(label(occupancy, structure=np.ones((3, 3, 3), dtype=np.uint8))[1])


def _write_mesh_binary(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
) -> Path:
    def write(stream) -> None:
        stream.write(b"HTM1")
        stream.write(struct.pack("<II", len(vertices), len(faces)))
        stream.write(np.asarray(vertices, dtype="<f4").tobytes())
        stream.write(np.asarray(faces, dtype="<u4").tobytes())

    return write_binary_atomic(path, write)


def _savez_atomic(path: Path, **arrays: np.ndarray) -> Path:
    def write(stream) -> None:
        np.savez_compressed(stream, **arrays)

    return write_binary_atomic(path, write)


def _load_z_manifest(
    config: TopologyConfig,
    slide_ids: tuple[str, ...],
) -> tuple[tuple[float, ...], str] | None:
    if config.z_manifest is None:
        return None
    payload = json.loads(config.z_manifest.read_text())
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("z manifest must use schema version 1")
    manifest_thickness = float(payload.get("section_thickness_um", math.nan))
    if not math.isclose(
        manifest_thickness,
        config.section_thickness_um,
        rel_tol=1e-6,
        abs_tol=1e-6,
    ):
        raise ValueError("z manifest section thickness differs from topology config")
    rows = payload.get("sections")
    if not isinstance(rows, list) or len(rows) != len(slide_ids):
        raise ValueError("z manifest must contain every semantic section")
    ids = tuple(str(row.get("id", "")) for row in rows if isinstance(row, dict))
    if ids != slide_ids:
        raise ValueError("z manifest section order differs from semantic result")
    positions = tuple(float(row.get("z_um", math.nan)) for row in rows)
    if not all(math.isfinite(value) for value in positions) or any(
        right <= left for left, right in zip(positions, positions[1:], strict=True)
    ):
        raise ValueError("z manifest coordinates must be finite and increasing")
    for left, right in zip(positions, positions[1:], strict=True):
        intervals = (right - left) / config.section_thickness_um
        if not math.isclose(intervals, round(intervals), abs_tol=0.05):
            raise ValueError("z manifest gaps must be integer section intervals")
    source = payload.get("source", "assumed")
    if source not in {"assumed", "measured"}:
        raise ValueError("z manifest source must be assumed or measured")
    return positions, str(source)


def _sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
