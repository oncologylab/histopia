from __future__ import annotations

import hashlib
import json
import subprocess
import sys
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
    assert manifest["schema_version"] == 2
    assert manifest["format"] == "histopia-qupath-bundle"
    assert manifest["semantic_clusters"] == 2
    assert manifest["semantic_geometry"] == "regions"
    assert manifest["semantic_geometry_version"] == "regions-v1"
    assert manifest["coordinate_conventions"]["semantic_annotations"] == (
        "source_native_pixels"
    )
    slide = manifest["slides"][0]
    assert slide["source_uri"].startswith("file://")
    annotation_path = manifest_path.parent / slide["semantic_annotations"]
    annotations = json.loads(annotation_path.read_text())
    assert (
        slide["semantic_annotations_sha256"]
        == hashlib.sha256(annotation_path.read_bytes()).hexdigest()
    )
    assert slide["semantic_annotations_bytes"] == annotation_path.stat().st_size
    assert slide["semantic_annotation_classes"] == 2
    assert slide["semantic_annotation_regions"] == 2
    assert slide["semantic_patch_count"] == 2
    assert len(annotations["features"]) == 2
    assert annotations["histopia"]["geometry"] == "regions"
    assert annotations["histopia"]["geometry_version"] == "regions-v1"
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


def test_qupath_bundle_coalesces_adjacent_tiles_and_preserves_audit_mode(
    tmp_path: Path,
) -> None:
    registration, semantic = _write_runs(
        tmp_path,
        grid_rc=np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int32),
        native_xy=np.array(
            [[200.0, 200.0], [424.0, 200.0], [200.0, 424.0], [424.0, 424.0]]
        ),
        labels=np.zeros(4, dtype=np.int16),
    )

    region_manifest_path = export_qupath_bundle(
        registration,
        tmp_path / "regions",
        semantic_run=semantic,
        clusters=2,
    )
    tile_manifest_path = export_qupath_bundle(
        registration,
        tmp_path / "tiles",
        semantic_run=semantic,
        clusters=2,
        semantic_geometry="tiles",
    )

    region_manifest = json.loads(region_manifest_path.read_text())
    tile_manifest = json.loads(tile_manifest_path.read_text())
    region_slide = region_manifest["slides"][0]
    tile_slide = tile_manifest["slides"][0]
    assert region_slide["semantic_annotation_regions"] == 1
    assert tile_slide["semantic_annotation_regions"] == 4
    assert (
        region_slide["semantic_annotations_bytes"]
        < (tile_slide["semantic_annotations_bytes"])
    )
    region_geojson = json.loads(
        (region_manifest_path.parent / region_slide["semantic_annotations"]).read_text()
    )
    polygon = region_geojson["features"][0]["geometry"]["coordinates"][0][0]
    assert polygon == [
        [88, 88],
        [536, 88],
        [536, 536],
        [88, 536],
        [88, 88],
    ]


def test_qupath_bundle_rejects_feature_and_label_grid_mismatch(
    tmp_path: Path,
) -> None:
    registration, semantic = _write_runs(tmp_path)
    feature_path = semantic / "features" / "001-section.npz"
    with np.load(feature_path, allow_pickle=False) as feature:
        slide_id = feature["slide_id"]
        native_xy = feature["native_xy"]
    np.savez_compressed(
        feature_path,
        slide_id=slide_id,
        native_xy=native_xy,
        grid_rc=np.array([[0, 0], [1, 0]], dtype=np.int32),
    )

    with pytest.raises(ValueError, match="coordinates do not match labels"):
        export_qupath_bundle(
            registration,
            tmp_path / "bundle",
            semantic_run=semantic,
            clusters=2,
        )


def test_qupath_bundle_rejects_unknown_semantic_geometry(tmp_path: Path) -> None:
    registration, semantic = _write_runs(tmp_path)

    with pytest.raises(ValueError, match="semantic_geometry must be"):
        export_qupath_bundle(
            registration,
            tmp_path / "bundle",
            semantic_run=semantic,
            semantic_geometry="contours",
        )


def _write_runs(
    root: Path,
    *,
    grid_rc: np.ndarray | None = None,
    native_xy: np.ndarray | None = None,
    labels: np.ndarray | None = None,
) -> tuple[Path, Path]:
    grid_rc = np.array([[0, 0], [0, 1]], dtype=np.int32) if grid_rc is None else grid_rc
    native_xy = (
        np.array([[200.0, 200.0], [400.0, 200.0]]) if native_xy is None else native_xy
    )
    labels = np.array([0, 1], dtype=np.int16) if labels is None else labels
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
        labels=labels,
        joint_labels=labels,
        grid_rc=grid_rc,
        reference_um_xy=native_xy * 0.5,
        tissue_fraction=np.ones(len(labels), dtype=np.float32),
        grid_shape=np.max(grid_rc, axis=0) + 1,
        patch_size_px=np.int32(224),
        analysis_mpp=np.float64(0.5),
    )
    np.savez_compressed(
        features_dir / "001-section.npz",
        slide_id=np.asarray(source.name),
        native_xy=native_xy,
        grid_rc=grid_rc,
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


def test_qupath_import_does_not_load_heavy_workflow_modules() -> None:
    code = """
import sys
import histopia.qupath

blocked = {
    "histopia.registration._masking",
    "histopia.semantic._atlas",
    "histopia.semantic._pipeline",
}
assert blocked.isdisjoint(sys.modules), blocked.intersection(sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True)
