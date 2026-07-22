"""Portable semantic-atlas results and explicit review state."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from histopia.semantic._atlas import JointAtlas
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

    core = {
        "schema_version": 2,
        "primary_clusters": primary_clusters,
        "cluster_counts": list(atlas.clusterings),
        "pca_components": atlas.pca_components,
        "selected_k": selected_k,
        "batch_correction": batch,
        "k_selection": k_selection,
        "model": model_path.name,
        "palette": list(SEMANTIC_PALETTE),
        "clusterings": clustering_rows,
        "slides": slide_rows,
        "topology_pairs": topology_rows,
    }
    fingerprint = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    payload = {**core, "fingerprint": fingerprint}
    result_path = output_dir / "semantic_result.json"
    result_path.write_text(json.dumps(payload, indent=2) + "\n")
    review = {
        "schema_version": 2,
        "approved": False,
        "fingerprint": fingerprint,
        "reviewer": None,
        "notes": "",
    }
    (output_dir / "semantic_review.json").write_text(
        json.dumps(review, indent=2) + "\n"
    )
    return result_path
