from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import histopia.qupath._export as qupath_export_module
import histopia.registration._approval as registration_approval_module
from histopia.qupath import export_qupath_bundle
from histopia.registration import approve_registration_run
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
    assert manifest["schema_version"] == 3
    assert manifest["format"] == "histopia-qupath-bundle"
    assert manifest["semantic_clusters"] == 2
    assert manifest["semantic_geometry"] == "regions"
    assert manifest["semantic_geometry_version"] == "regions-v1"
    assert manifest["semantic_approval"]["reviewer"] == "Test Reviewer"
    assert (
        manifest["semantic_approval"]["fingerprint"] == manifest["semantic_fingerprint"]
    )
    assert manifest["semantic_preflight_fingerprint"]
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
    assert first["geometry"]["coordinates"][0][0][0] == [0.0, 0.0]


def test_qupath_bundle_rejects_unavailable_k(tmp_path: Path) -> None:
    registration, semantic = _write_runs(tmp_path)

    with pytest.raises(ValueError, match="K=5 is unavailable"):
        export_qupath_bundle(
            registration,
            tmp_path / "bundle",
            semantic_run=semantic,
            clusters=5,
        )


def test_qupath_bundle_rejects_unapproved_or_different_registration(
    tmp_path: Path,
) -> None:
    registration, semantic = _write_runs(tmp_path)
    review_path = semantic / "semantic_review.json"
    review = json.loads(review_path.read_text())
    review["approved"] = False
    review_path.write_text(json.dumps(review))

    with pytest.raises(ValueError, match="not approved"):
        export_qupath_bundle(
            registration,
            tmp_path / "unapproved",
            semantic_run=semantic,
        )
    assert not (tmp_path / "unapproved").exists()

    review["approved"] = True
    review_path.write_text(json.dumps(review))
    result_path = registration / "registration_result.json"
    registration_payload = json.loads(result_path.read_text())
    registration_payload["changed_after_semantics"] = True
    result_path.write_text(json.dumps(registration_payload))

    with pytest.raises(ValueError, match="different registration result"):
        export_qupath_bundle(
            registration,
            tmp_path / "mismatched",
            semantic_run=semantic,
        )
    assert not (tmp_path / "mismatched").exists()


def test_qupath_bundle_coalesces_adjacent_tiles_and_preserves_audit_mode(
    tmp_path: Path,
) -> None:
    registration, semantic = _write_runs(
        tmp_path,
        grid_rc=np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int32),
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
        [0, 0],
        [448, 0],
        [448, 448],
        [0, 448],
        [0, 0],
    ]


def test_qupath_bundle_does_not_trust_mutable_feature_coordinates(
    tmp_path: Path,
) -> None:
    registration, semantic = _write_runs(tmp_path)
    feature_dir = semantic / "features"
    feature_dir.mkdir()
    np.savez_compressed(
        feature_dir / "stale.npz",
        slide_id=np.asarray("section.ndpi"),
        native_xy=np.array([[900.0, 900.0]]),
        grid_rc=np.array([[99, 99]], dtype=np.int32),
    )

    manifest_path = export_qupath_bundle(
        registration,
        tmp_path / "bundle",
        semantic_run=semantic,
        clusters=2,
    )

    manifest = json.loads(manifest_path.read_text())
    annotation_path = (
        manifest_path.parent / manifest["slides"][0]["semantic_annotations"]
    )
    annotations = json.loads(annotation_path.read_text())
    assert annotations["features"][0]["geometry"]["coordinates"][0][0][0] == [
        0.0,
        0.0,
    ]


def test_qupath_bundle_rejects_label_grid_inconsistent_with_registration(
    tmp_path: Path,
) -> None:
    registration, semantic = _write_runs(tmp_path)
    label_path = semantic / "labels" / "k-2" / "001.npz"
    with np.load(label_path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    arrays["grid_shape"] = np.array([1, 2], dtype=np.int32)
    np.savez_compressed(label_path, **arrays)
    _reseal_semantic_result(semantic)

    with pytest.raises(ValueError, match="differs from registration geometry"):
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


def test_registration_only_bundle_requires_and_records_final_approval(
    tmp_path: Path,
) -> None:
    registration, _ = _write_runs(tmp_path)

    with pytest.raises(ValueError, match="requires a sealed registration approval"):
        export_qupath_bundle(registration, tmp_path / "unapproved")

    _approve_synthetic_registration(registration)
    manifest_path = export_qupath_bundle(registration, tmp_path / "approved")
    manifest = json.loads(manifest_path.read_text())

    assert manifest["schema_version"] == 4
    assert manifest["registration_approval"]["reviewer"] == "Test Reviewer"
    assert (
        manifest["registration_approval"]["registration_result_sha256"]
        == manifest["registration_sha256"]
    )
    assert len(manifest["registration_approval"]["approval_sha256"]) == 64


def test_registration_only_bundle_rejects_approval_changed_during_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registration, _ = _write_runs(tmp_path)
    _approve_synthetic_registration(registration)
    original = registration_approval_module.validate_registration_approval

    def mutate_after_validation(run_dir):
        approval = original(run_dir)
        path = registration / "registration_approval.json"
        payload = json.loads(path.read_text())
        payload["notes"] = "Changed during QuPath export."
        path.write_text(json.dumps(payload))
        return approval

    monkeypatch.setattr(
        registration_approval_module,
        "validate_registration_approval",
        mutate_after_validation,
    )

    with pytest.raises(
        ValueError, match="registration approval changed during validation"
    ):
        export_qupath_bundle(registration, tmp_path / "bundle")


def test_registration_only_bundle_rejects_approval_changed_before_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registration, _ = _write_runs(tmp_path)
    _approve_synthetic_registration(registration)
    original = qupath_export_module._validate_registration_approval

    def mutate_after_validation(run_dir):
        approval, digest = original(run_dir)
        path = registration / "registration_approval.json"
        payload = json.loads(path.read_text())
        payload["notes"] = "Changed before manifest write."
        path.write_text(json.dumps(payload))
        return approval, digest

    monkeypatch.setattr(
        qupath_export_module,
        "_validate_registration_approval",
        mutate_after_validation,
    )

    with pytest.raises(
        ValueError, match="registration approval changed during QuPath export"
    ):
        export_qupath_bundle(registration, tmp_path / "bundle")


def test_schema_four_semantic_bundle_binds_final_registration_approval(
    tmp_path: Path,
) -> None:
    registration, semantic = _write_runs(tmp_path, approval_bound=True)

    manifest_path = export_qupath_bundle(
        registration,
        tmp_path / "bundle",
        semantic_run=semantic,
    )
    manifest = json.loads(manifest_path.read_text())

    assert manifest["schema_version"] == 4
    assert manifest["registration_approval"]["reviewer"] == "Test Reviewer"
    approval_path = registration / "registration_approval.json"
    approval = json.loads(approval_path.read_text())
    approval["notes"] = "Changed after semantic preflight."
    approval_path.write_text(json.dumps(approval))
    with pytest.raises(ValueError, match="registration approval is stale"):
        export_qupath_bundle(
            registration,
            tmp_path / "stale",
            semantic_run=semantic,
        )


def _write_runs(
    root: Path,
    *,
    grid_rc: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    approval_bound: bool = False,
) -> tuple[Path, Path]:
    grid_rc = np.array([[0, 0], [0, 1]], dtype=np.int32) if grid_rc is None else grid_rc
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
    if approval_bound:
        _approve_synthetic_registration(registration)
    semantic = root / "semantic"
    labels_dir = semantic / "labels" / "k-2"
    labels_dir.mkdir(parents=True)
    native_xy = np.column_stack(
        (
            112 + grid_rc[:, 1] * 224,
            112 + grid_rc[:, 0] * 224,
        )
    )
    np.savez_compressed(
        labels_dir / "001.npz",
        labels=labels,
        joint_labels=labels,
        grid_rc=grid_rc,
        reference_um_xy=native_xy * 0.5,
        tissue_fraction=np.ones(len(labels), dtype=np.float32),
        grid_shape=np.array([4, 5], dtype=np.int32),
        patch_size_px=np.int32(224),
        analysis_mpp=np.float64(0.5),
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
    registration_sha256 = hashlib.sha256(
        (registration / "registration_result.json").read_bytes()
    ).hexdigest()
    preflight_core = {
        "schema_version": 3 if approval_bound else 2,
        "registration_result_sha256": registration_sha256,
        "order_review_fingerprint": None,
        "reference_slide": source.name,
        "slides": [
            {
                "slide_name": source.name,
                "source_sha256": "source-sha256",
                "thumbnail_sha256": "thumbnail-sha256",
                "mask_sha256": "mask-sha256",
                "mask_method": "test",
                "mask_review_status": "auto_pass",
                "transform_sha256": "transform-sha256",
                "thumbnail_shape": [100, 120],
                "mpp_xy": [0.5, 0.5],
                "is_reference": True,
            }
        ],
    }
    if approval_bound:
        preflight_core["registration_approval_sha256"] = hashlib.sha256(
            (registration / "registration_approval.json").read_bytes()
        ).hexdigest()
    preflight_fingerprint = hashlib.sha256(
        json.dumps(
            preflight_core,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    preflight = {
        **preflight_core,
        "registration_run": str(registration),
        "slides": [
            {
                **preflight_core["slides"][0],
                "source_path": str(source),
            }
        ],
        "fingerprint": preflight_fingerprint,
        "slide_count": 1,
    }
    (semantic / "preflight.json").write_text(json.dumps(preflight))
    core["feature_provenance"] = {
        "preflight_fingerprint": preflight_fingerprint,
        "expected_slide_ids": [source.name],
    }
    payload = _seal_semantic_result(semantic, core)
    (semantic / "semantic_result.json").write_text(json.dumps(payload))
    (semantic / "semantic_review.json").write_text(
        json.dumps(
            {
                "schema_version": 3,
                "approved": True,
                "fingerprint": payload["fingerprint"],
                "reviewer": "Test Reviewer",
                "reviewed_at": "2026-07-24T18:30:00+00:00",
                "notes": "Reviewed synthetic semantic overlays.",
            }
        )
    )
    return registration, semantic


def _approve_synthetic_registration(registration: Path) -> None:
    result_path = registration / "registration_result.json"
    result = json.loads(result_path.read_text())
    slide = result["slides"][0]
    slide_id = Path(slide["path"]).name
    review = {
        "slide": slide_id,
        "thumbnail_sha256": "mask-fingerprint",
        "status": "auto_pass",
        "method": "test",
        "reviewer": "Test Reviewer",
        "notes": "Reviewed.",
        "override_path": None,
    }
    slide["mask"] = {"accepted": True}
    slide["mask_review"] = dict(review)
    result_path.write_text(json.dumps(result))
    (registration / "mask_review.json").write_text(
        json.dumps({"schema_version": 2, "slides": [review]})
    )
    (registration / "section_order_review.json").write_text(
        json.dumps(
            {
                "schema_version": 3,
                "approved": True,
                "fingerprint": "order-fingerprint",
                "input_fingerprints": {slide_id: "order-input"},
                "slides": [{"order": 1, "slide": slide_id}],
            }
        )
    )
    approve_registration_run(
        registration,
        reviewer="Test Reviewer",
        notes="Registration reviewed.",
        reviewed_at="2026-07-24T18:00:00+00:00",
    )


def _reseal_semantic_result(root: Path) -> None:
    path = root / "semantic_result.json"
    payload = json.loads(path.read_text())
    core = {
        key: value
        for key, value in payload.items()
        if key not in {"artifacts", "fingerprint"}
    }
    sealed = _seal_semantic_result(root, core)
    path.write_text(json.dumps(sealed))
    review = json.loads((root / "semantic_review.json").read_text())
    review["fingerprint"] = sealed["fingerprint"]
    (root / "semantic_review.json").write_text(json.dumps(review))


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
