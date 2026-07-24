from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from histopia.qupath import export_qupath_bundle
from histopia.semantic._result import _seal_semantic_result


def test_qupath_bundle_exports_native_semantic_geojson(tmp_path: Path) -> None:
    registration, semantic = _write_runs(tmp_path)

    manifest_path = export_qupath_bundle(
        registration,
        tmp_path / "bundle",
        semantic_run=semantic,
        clusters=2,
    )

    manifest = json.loads(manifest_path.read_text())
    assert manifest["format"] == "histopia-qupath-bundle"
    assert manifest["semantic_clusters"] == 2
    assert manifest["coordinate_conventions"]["semantic_annotations"] == (
        "source_native_pixels"
    )
    slide = manifest["slides"][0]
    assert slide["source_uri"].startswith("file://")
    annotations = json.loads(
        (manifest_path.parent / slide["semantic_annotations"]).read_text()
    )
    assert len(annotations["features"]) == 2
    first = annotations["features"][0]
    assert first["properties"]["objectType"] == "annotation"
    assert first["properties"]["classification"]["color"] == [215, 48, 39]
    assert first["geometry"]["type"] == "MultiPolygon"
    assert first["geometry"]["coordinates"][0][0][0] == [88.0, 88.0]


def test_qupath_bundle_rejects_unavailable_k(tmp_path: Path) -> None:
    registration, semantic = _write_runs(tmp_path)

    with pytest.raises(ValueError, match="K=5 is unavailable"):
        export_qupath_bundle(
            registration,
            tmp_path / "bundle",
            semantic_run=semantic,
            clusters=5,
        )


def _write_runs(root: Path) -> tuple[Path, Path]:
    registration = root / "registration"
    registration.mkdir()
    source = root / "section.ndpi"
    geometry = {
        "native_shape": [1000, 1200],
        "content_bbox_xywh": [0, 0, 1200, 1000],
        "thumbnail_shape": [100, 120],
        "bounds_source": "test",
        "mpp_xy": [0.5, 0.5],
        "mpp_source": "test",
    }
    (registration / "registration_result.json").write_text(
        json.dumps(
            {
                "reference_slide": str(source),
                "slides": [
                    {
                        "path": str(source),
                        "is_reference": True,
                        "aligned_to": None,
                        "geometry": geometry,
                        "transform": {"matrix": np.eye(3).tolist()},
                        "alignment_metrics": {"dice": 1.0},
                    }
                ],
            }
        )
    )
    semantic = root / "semantic"
    labels_dir = semantic / "labels" / "k-2"
    features_dir = semantic / "features"
    labels_dir.mkdir(parents=True)
    features_dir.mkdir()
    np.savez_compressed(
        labels_dir / "001.npz",
        labels=np.array([0, 1], dtype=np.int16),
        joint_labels=np.array([0, 1], dtype=np.int16),
        grid_rc=np.array([[0, 0], [0, 1]], dtype=np.int32),
        reference_um_xy=np.array([[50.0, 50.0], [150.0, 50.0]]),
        tissue_fraction=np.ones(2, dtype=np.float32),
        grid_shape=np.array([1, 2], dtype=np.int32),
        patch_size_px=np.int32(224),
        analysis_mpp=np.float64(0.5),
    )
    np.savez_compressed(
        features_dir / "001-section.npz",
        slide_id=np.asarray(source.name),
        native_xy=np.array([[200.0, 200.0], [400.0, 200.0]]),
    )
    np.savez_compressed(semantic / "atlas_model.npz", pca_mean=np.zeros(2))
    core = {
        "schema_version": 3,
        "primary_clusters": 2,
        "selected_k": 2,
        "cluster_counts": [2],
        "palette": ["#d73027", "#1a9850"],
        "model": "atlas_model.npz",
        "slides": [
            {
                "id": source.name,
                "labels": {"2": "labels/k-2/001.npz"},
            }
        ],
        "topology_pairs": [],
    }
    payload = _seal_semantic_result(semantic, core)
    (semantic / "semantic_result.json").write_text(json.dumps(payload))
    return registration, semantic
