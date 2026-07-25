"""Portable semantic-atlas results and explicit review state."""

from __future__ import annotations

import json
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from histopia._atomic import write_binary_atomic, write_json_atomic
from histopia.semantic import _result_validation
from histopia.semantic._atlas import JointAtlas
from histopia.semantic._correspondence import CorrespondenceConfig
from histopia.semantic._features import PatchFeatures

_seal_semantic_result = _result_validation._seal_semantic_result
validate_semantic_result = _result_validation.validate_semantic_result

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
    """Write sealed atlas artifacts and fingerprint-bound review state."""

    output_dir = Path(output_dir)
    common_provenance = _common_feature_provenance(sections, output_dir)
    patch_widths = {
        float(section.patch_size_px * section.analysis_mpp) for section in sections
    }
    if len(patch_widths) != 1:
        raise ValueError("semantic sections must use one physical patch width")
    if primary_clusters not in atlas.clusterings:
        raise ValueError("primary cluster count is missing from the atlas")
    expected_slide_ids = tuple(section.slide_id for section in sections)
    if atlas.slide_ids != expected_slide_ids:
        raise ValueError("atlas slide order differs from semantic sections")
    expected_offsets = np.concatenate(
        [[0], np.cumsum([len(section.features) for section in sections])]
    )
    if not np.array_equal(atlas.section_offsets, expected_offsets):
        raise ValueError("atlas section offsets differ from semantic sections")

    label_root = output_dir / "labels"
    label_root.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "atlas_model.npz"
    arrays: dict[str, np.ndarray] = {
        "pca_mean": atlas.pca_mean,
        "pca_basis": atlas.pca_basis,
    }
    for count, clustering in atlas.clusterings.items():
        arrays[f"centroids_k{count}"] = clustering.centroids
    _savez_compressed_atomic(model_path, **arrays)

    slide_rows: list[dict[str, object]] = []
    for index, section in enumerate(sections):
        start, stop = atlas.section_offsets[index : index + 2]
        labels_by_count: dict[str, str] = {}
        for count, clustering in atlas.clusterings.items():
            directory = label_root / f"k-{count}"
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"{index + 1:03d}.npz"
            _savez_compressed_atomic(
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
        _savez_compressed_atomic(
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

    correspondence = asdict(
        CorrespondenceConfig(patch_width_um=next(iter(patch_widths)))
    )
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
    write_json_atomic(result_path, payload)
    review_path = output_dir / "semantic_review.json"
    review = _review_for_current_result(output_dir, review_path, fingerprint)
    write_json_atomic(review_path, review)
    return result_path


def _review_for_current_result(
    output_dir: Path,
    review_path: Path,
    fingerprint: str,
) -> dict[str, object]:
    default: dict[str, object] = {
        "schema_version": 3,
        "approved": False,
        "fingerprint": fingerprint,
        "reviewer": None,
        "notes": "",
    }
    try:
        from histopia.semantic._approval import validate_semantic_approval

        validate_semantic_approval(output_dir)
        previous = json.loads(review_path.read_text())
    except (FileNotFoundError, OSError, TypeError, ValueError):
        return default
    return previous if isinstance(previous, dict) else default


def _savez_compressed_atomic(path: Path, **arrays: np.ndarray) -> Path:
    def write(stream) -> None:
        np.savez_compressed(stream, **arrays)

    return write_binary_atomic(path, write)


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
        any(key not in provenance for key in required_keys)
        for provenance in provenance_rows
    ):
        raise ValueError("semantic feature provenance is incomplete")
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
    content_fingerprints = tuple(section.content_fingerprint for section in sections)
    if any(content_fingerprints) and not all(content_fingerprints):
        raise ValueError("semantic feature content sealing is inconsistent")
    if all(content_fingerprints):
        common["feature_integrity"] = "content-sha256-v1"
        common["feature_content_fingerprints"] = list(content_fingerprints)
    else:
        common["feature_integrity"] = "legacy-unsealed"
    common["expected_slide_ids"] = expected_slide_ids
    return common
