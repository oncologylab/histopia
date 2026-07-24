"""Portable semantic-atlas results and explicit review state."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from histopia.semantic._atlas import JointAtlas
from histopia.semantic._correspondence import CorrespondenceConfig
from histopia.semantic._features import PatchFeatures

SEMANTIC_PALETTE = (
    "#d73027",
    "#1a9850",
    "#4575b4",
    "#fee08b",
    "#984ea3",
    "#00a6a6",
    "#f46d43",
    "#7f8c8d",
    "#66bd63",
    "#3288bd",
    "#e6ab02",
    "#a6761d",
    "#e7298a",
    "#1b9e77",
    "#666666",
)


def write_atlas_result(
    atlas: JointAtlas,
    sections: tuple[PatchFeatures, ...],
    output_dir: Path | str,
    *,
    primary_clusters: int,
) -> Path:
    """Write labels, model metadata, and an unapproved review record."""

    output_dir = Path(output_dir)
    label_root = output_dir / "labels"
    label_root.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "atlas_model.npz"
    arrays: dict[str, np.ndarray] = {
        "pca_mean": atlas.pca_mean,
        "pca_basis": atlas.pca_basis,
    }
    for count, clustering in atlas.clusterings.items():
        arrays[f"centroids_k{count}"] = clustering.centroids
    np.savez_compressed(model_path, **arrays)

    slide_rows: list[dict[str, object]] = []
    for index, section in enumerate(sections):
        start, stop = atlas.section_offsets[index : index + 2]
        labels_by_count: dict[str, str] = {}
        for count, clustering in atlas.clusterings.items():
            directory = label_root / f"k-{count}"
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"{index + 1:03d}.npz"
            np.savez_compressed(
                path,
                labels=clustering.labels[start:stop].astype(np.int16),
                joint_labels=clustering.joint_labels[start:stop].astype(np.int16),
                grid_rc=section.grid_rc,
                reference_um_xy=section.reference_um_xy,
                tissue_fraction=section.tissue_fraction,
                grid_shape=np.asarray(section.grid_shape, dtype=np.int32),
                patch_size_px=np.int32(section.patch_size_px),
                analysis_mpp=np.float64(section.analysis_mpp),
            )
            labels_by_count[str(count)] = str(path.relative_to(output_dir))
        slide_rows.append({"id": section.slide_id, "labels": labels_by_count})

    clustering_rows = {}
    for count, clustering in atlas.clusterings.items():
        guard = clustering.diffusion_guard
        clustering_rows[str(count)] = {
            "graph_regularization_accepted": guard.accepted if guard else False,
            "changed_fraction": guard.changed_fraction if guard else 0.0,
            "guard_reasons": list(guard.reasons) if guard else [],
        }
    selected_k = primary_clusters
    batch = None
    if atlas.batch_correction is not None:
        correction = atlas.batch_correction
        batch = {
            "accepted": correction.guard.accepted,
            "guard_reasons": list(correction.guard.reasons),
            "unsupported_sections": list(correction.unsupported_sections),
            "raw": asdict(correction.raw_diagnostics),
            "legacy": asdict(correction.legacy_diagnostics),
            "corrected": asdict(correction.corrected_diagnostics),
        }
    k_selection = None
    if atlas.cluster_selection is not None:
        k_selection = [asdict(item) for item in atlas.cluster_selection.evaluations]

    topology_root = output_dir / "topology"
    topology_rows: list[dict[str, object]] = []
    for correspondence in atlas.correspondences:
        topology_root.mkdir(parents=True, exist_ok=True)
        source = correspondence.source_section
        target = correspondence.target_section
        path = topology_root / f"{source + 1:03d}-{target + 1:03d}.npz"
        np.savez_compressed(
            path,
            source_indices=correspondence.source_indices,
            target_indices=correspondence.target_indices,
            source_um_xy=sections[source].reference_um_xy[
                correspondence.source_indices
            ],
            target_um_xy=sections[target].reference_um_xy[
                correspondence.target_indices
            ],
            confidence=correspondence.confidence,
            feature_similarity=correspondence.feature_similarity,
            field_residual_um=correspondence.field_residual_um,
            neighborhood_consistency=correspondence.neighborhood_consistency,
        )
        topology_rows.append(
            {
                "source_section": source,
                "target_section": target,
                "accepted_links": int(len(correspondence.confidence)),
                "artifact": str(path.relative_to(output_dir)),
            }
        )

    patch_widths = {
        float(section.patch_size_px * section.analysis_mpp) for section in sections
    }
    if len(patch_widths) != 1:
        raise ValueError("semantic sections must use one physical patch width")
    correspondence = asdict(
        CorrespondenceConfig(patch_width_um=next(iter(patch_widths)))
    )
    common_provenance = _common_feature_provenance(sections, output_dir)
    core = {
        "schema_version": 3,
        "primary_clusters": primary_clusters,
        "cluster_counts": list(atlas.clusterings),
        "pca_components": atlas.pca_components,
        "feature_normalization": "patch_l2_v2",
        "feature_provenance": common_provenance,
        "fit_runtime": {
            package: _package_version(package)
            for package in ("numpy", "scikit-learn", "scipy")
        },
        "correspondence": correspondence,
        "selected_k": selected_k,
        "batch_correction": batch,
        "k_selection": k_selection,
        "model": model_path.name,
        "palette": list(SEMANTIC_PALETTE),
        "clusterings": clustering_rows,
        "slides": slide_rows,
        "topology_pairs": topology_rows,
    }
    payload = _seal_semantic_result(output_dir, core)
    fingerprint = str(payload["fingerprint"])
    result_path = output_dir / "semantic_result.json"
    result_path.write_text(json.dumps(payload, indent=2) + "\n")
    review = {
        "schema_version": 3,
        "approved": False,
        "fingerprint": fingerprint,
        "reviewer": None,
        "notes": "",
    }
    (output_dir / "semantic_review.json").write_text(
        json.dumps(review, indent=2) + "\n"
    )
    return result_path


def validate_semantic_result(
    run_dir: Path | str,
    payload: dict[str, object] | None = None,
) -> dict[str, object]:
    """Load and verify a schema-3 result and every referenced artifact."""

    root = Path(run_dir)
    loaded = (
        json.loads((root / "semantic_result.json").read_text())
        if payload is None
        else dict(payload)
    )
    if loaded.get("schema_version") != 3:
        raise ValueError("semantic result must use schema version 3")
    references = _referenced_artifacts(root, loaded)
    declared = loaded.get("artifacts")
    if not isinstance(declared, dict) or set(declared) != set(references):
        raise ValueError("semantic result artifact manifest is incomplete or stale")
    for relative, path in references.items():
        if not path.is_file():
            raise ValueError(f"semantic result artifact is missing: {relative}")
        if declared[relative] != _sha256_file(path):
            raise ValueError(f"semantic result artifact digest mismatch: {relative}")
    fingerprint = loaded.get("fingerprint")
    core = {key: value for key, value in loaded.items() if key != "fingerprint"}
    if fingerprint != _fingerprint_core(core):
        raise ValueError("semantic result fingerprint is stale")
    return loaded


def _seal_semantic_result(
    root: Path,
    core: dict[str, object],
) -> dict[str, object]:
    sealed = dict(core)
    references = _referenced_artifacts(root, sealed)
    sealed["artifacts"] = {
        relative: _sha256_file(path) for relative, path in sorted(references.items())
    }
    return {**sealed, "fingerprint": _fingerprint_core(sealed)}


def _referenced_artifacts(
    root: Path,
    payload: dict[str, object],
) -> dict[str, Path]:
    raw_paths: list[object] = [payload.get("model")]
    for slide in payload.get("slides", []):
        raw_paths.extend(slide.get("labels", {}).values())
    raw_paths.extend(pair.get("artifact") for pair in payload.get("topology_pairs", []))
    references: dict[str, Path] = {}
    for raw_path in raw_paths:
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError("semantic artifact paths must be non-empty relative paths")
        relative, resolved = _safe_artifact_path(root, raw_path)
        if relative in references:
            raise ValueError(
                f"semantic artifact is referenced more than once: {relative}"
            )
        references[relative] = resolved
    return references


def _safe_artifact_path(root: Path, value: str) -> tuple[str, Path]:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(
            "semantic artifact paths must be relative to the run directory"
        )
    root_resolved = root.resolve()
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root_resolved):
        raise ValueError(
            "semantic artifact paths must be relative to the run directory"
        )
    return relative.as_posix(), resolved


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint_core(core: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "unavailable"


def _common_feature_provenance(
    sections: tuple[PatchFeatures, ...],
    output_dir: Path,
) -> dict[str, object] | None:
    if not sections or any(section.provenance is None for section in sections):
        return None
    required_keys = (
        "preflight_fingerprint",
        "model_fingerprint",
        "analysis_mpp",
        "patch_size_px",
        "min_tissue_fraction",
    )
    execution_keys = (
        "batch_size",
        "encoder_runtime",
        "extraction_method",
        "patch_reader",
    )
    provenance_rows = tuple(section.provenance for section in sections)
    if any(
        any(key in provenance for key in execution_keys)
        and not all(key in provenance for key in execution_keys)
        for provenance in provenance_rows
    ):
        raise ValueError("semantic execution provenance is incomplete")
    include_execution = all(
        all(key in provenance for key in execution_keys)
        for provenance in provenance_rows
    )
    keys = required_keys + (execution_keys if include_execution else ())
    common: dict[str, object] = {}
    for key in keys:
        values = {
            json.dumps(section.provenance[key], sort_keys=True) for section in sections
        }
        if len(values) != 1:
            raise ValueError(f"semantic feature provenance differs for {key}")
        common[key] = sections[0].provenance[key]
    preflight_path = output_dir / "preflight.json"
    if not preflight_path.is_file():
        raise ValueError("feature-backed semantic results require preflight.json")
    preflight = json.loads(preflight_path.read_text())
    if preflight.get("fingerprint") != common["preflight_fingerprint"]:
        raise ValueError("preflight fingerprint differs from feature provenance")
    raw_slides = preflight.get("slides")
    if not isinstance(raw_slides, list):
        raise ValueError("semantic preflight contains no slide order")
    expected_slide_ids = [str(row.get("slide_name", "")) for row in raw_slides]
    actual_slide_ids = [section.slide_id for section in sections]
    if (
        any(not slide_id for slide_id in expected_slide_ids)
        or len(set(expected_slide_ids)) != len(expected_slide_ids)
        or expected_slide_ids != actual_slide_ids
    ):
        raise ValueError("semantic sections differ from preflight slide order")
    common["expected_slide_ids"] = expected_slide_ids
    return common
