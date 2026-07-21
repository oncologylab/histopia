"""Portable semantic-atlas results and explicit review state."""

from __future__ import annotations

import hashlib
import json
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
    core = {
        "schema_version": 1,
        "primary_clusters": primary_clusters,
        "cluster_counts": list(atlas.clusterings),
        "pca_components": atlas.pca_components,
        "model": model_path.name,
        "palette": list(SEMANTIC_PALETTE),
        "clusterings": clustering_rows,
        "slides": slide_rows,
    }
    fingerprint = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    payload = {**core, "fingerprint": fingerprint}
    result_path = output_dir / "semantic_result.json"
    result_path.write_text(json.dumps(payload, indent=2) + "\n")
    review = {
        "schema_version": 1,
        "approved": False,
        "fingerprint": fingerprint,
        "reviewer": None,
        "notes": "",
    }
    (output_dir / "semantic_review.json").write_text(
        json.dumps(review, indent=2) + "\n"
    )
    return result_path
