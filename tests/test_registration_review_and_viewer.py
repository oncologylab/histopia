from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from histopia.registration._masking import TissueMaskResult
from histopia.registration._review import MaskReviewEntry, resolve_reviewed_mask
from histopia.registration._slides import SlideGeometry, discover_slides
from histopia.semantic._result import _seal_semantic_result
from histopia.visualization._viewer import (
    _tissue_review_crop,
    build_mask_review,
    build_section_order_review,
    build_section_viewer,
)


def test_discover_slides_excludes_labels_and_generated_files(tmp_path: Path) -> None:
    for name in (
        "section.ndpi",
        "section.scn",
        "section_final_label.png",
        "section.registered.tiff",
        "notes.txt",
    ):
        (tmp_path / name).touch()

    assert [path.name for path in discover_slides(tmp_path, wsi_only=True)] == [
        "section.ndpi",
        "section.scn",
    ]


def test_reviewed_override_is_shape_checked_and_remeasured(tmp_path: Path) -> None:
    image = np.full((20, 24, 3), 255, dtype=np.uint8)
    geometry = SlideGeometry((200, 240), (0, 0, 240, 200), (20, 24), "test")
    automatic = TissueMaskResult(
        mask=np.ones((20, 24), dtype=bool),
        method="automatic",
        metrics={"foreground_fraction": 1.0},
        accepted=True,
        warnings=[],
    )
    slide_path = tmp_path / "section.ndpi"
    override_dir = tmp_path / "overrides"
    override_dir.mkdir()
    override = np.zeros((20, 24), dtype=np.uint8)
    override[4:16, 6:18] = 255
    Image.fromarray(override).save(override_dir / "section.ndpi.mask.png")

    result, entry = resolve_reviewed_mask(
        slide_path=slide_path,
        image=image,
        geometry=geometry,
        automatic=automatic,
        review_entries={
            slide_path.name: MaskReviewEntry(
                slide=slide_path.name,
                thumbnail_sha256="stale",
                status="override_pass",
            )
        },
        override_dir=override_dir,
        require_approved=False,
    )

    assert result.method == "reviewed_override"
    assert result.metrics["foreground_fraction"] == pytest.approx(0.3)
    assert entry.override_path is not None


def test_viewer_builds_manifest_and_pinned_import_map(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    processed = run_dir / "processed"
    processed.mkdir(parents=True)
    image = np.full((24, 30, 3), 230, dtype=np.uint8)
    image[5:20, 7:24] = (130, 80, 60)
    mask = np.zeros((24, 30), dtype=np.uint8)
    mask[5:20, 7:24] = 255
    Image.fromarray(image).save(processed / "section.thumbnail.png")
    Image.fromarray(mask).save(processed / "section.mask.png")
    (run_dir / "registration_result.json").write_text(
        json.dumps(
            {
                "reference_slide": str(tmp_path / "section.ndpi"),
                "slides": [
                    {
                        "path": str(tmp_path / "section.ndpi"),
                        "is_reference": True,
                        "transform": {"matrix": np.eye(3).tolist()},
                    }
                ],
            }
        )
    )

    index = build_section_viewer({"mouse": run_dir}, tmp_path / "viewer")

    manifest = json.loads((index.parent / "manifest.json").read_text())
    assert len(manifest["mice"][0]["slides"]) == 1
    assert "./vendor/three.module.min.js" in index.read_text()
    assert '<link rel="icon" href="data:">' in index.read_text()
    assert (
        "three/addons/controls/OrbitControls.js"
        in (index.parent / "viewer.js").read_text()
    )
    assert (index.parent / "vendor" / "OrbitControls.js").is_file()
    assert "MIT License" in (index.parent / "vendor" / "LICENSE-three.txt").read_text()
    styles = (index.parent / "styles.css").read_text()
    viewer = (index.parent / "viewer.js").read_text()
    assert "html,body" in styles
    assert "overflow:hidden" in styles
    assert "width:100%!important;height:100%!important" in styles
    assert "renderer.setSize(box.width, box.height, true)" in viewer
    assert "group.children.filter(child => child.visible)" in viewer
    assert "visibleMeshes.forEach(mesh => bounds.expandByObject(mesh))" in viewer
    assert "sphere.radius / 10000" in viewer
    assert "controls.minDistance" in viewer


def test_mask_review_builds_full_thumbnail_audit(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    processed = run_dir / "processed"
    processed.mkdir(parents=True)
    image = np.full((24, 30, 3), 230, dtype=np.uint8)
    mask = np.zeros((24, 30), dtype=np.uint8)
    mask[5:20, 7:24] = 255
    Image.fromarray(image).save(processed / "section.thumbnail.png")
    Image.fromarray(mask).save(processed / "section.mask.png")
    (run_dir / "registration_result.json").write_text(
        json.dumps(
            {
                "reference_slide": str(tmp_path / "section.ndpi"),
                "slides": [
                    {
                        "path": str(tmp_path / "section.ndpi"),
                        "is_reference": True,
                        "mask": {
                            "method": "group_consensus",
                            "metrics": {"foreground_fraction": 0.25},
                            "warnings": [],
                        },
                        "mask_review": {"status": "auto_pass"},
                        "transform": {"matrix": np.eye(3).tolist()},
                    }
                ],
            }
        )
    )

    index = build_mask_review(run_dir, tmp_path / "mask-review")

    manifest = json.loads((index.parent / "manifest.json").read_text())
    assert manifest["approved"] is True
    assert len(manifest["fingerprint"]) == 64
    assert manifest["slides"][0]["method"] == "group_consensus"
    assert (index.parent / manifest["slides"][0]["texture"]).is_file()
    css = (index.parent / "mask-review.css").read_text()
    assert "overflow:hidden" in css
    assert "@media(max-width:600px)" in css


def test_mask_review_builds_at_pre_registration_approval_gate(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    processed = run_dir / "processed"
    processed.mkdir(parents=True)
    image = np.full((24, 30, 3), 230, dtype=np.uint8)
    mask = np.zeros((24, 30), dtype=np.uint8)
    mask[5:20, 7:24] = 255
    Image.fromarray(image).save(processed / "section.thumbnail.png")
    Image.fromarray(mask).save(processed / "section.mask.png")
    (run_dir / "mask_review.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "slides": [
                    {
                        "slide": "section.ndpi",
                        "thumbnail_sha256": "test",
                        "status": "pending",
                        "method": "object_aware_fusion",
                        "reviewer": "",
                        "notes": "",
                        "override_path": None,
                    }
                ],
            }
        )
    )

    index = build_mask_review(run_dir, tmp_path / "mask-review")

    manifest = json.loads((index.parent / "manifest.json").read_text())
    assert manifest["approved"] is False
    assert manifest["slides"][0]["slide"] == "section.ndpi"
    assert manifest["slides"][0]["method"] == "object_aware_fusion"
    assert manifest["slides"][0]["foreground_fraction"] == pytest.approx(255 / 720)
    assert (index.parent / manifest["slides"][0]["texture"]).is_file()


def test_viewer_adds_lazy_semantic_and_blend_modes(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    processed = run_dir / "processed"
    processed.mkdir(parents=True)
    image = np.full((20, 20, 3), 220, dtype=np.uint8)
    mask = np.full((20, 20), 255, dtype=np.uint8)
    Image.fromarray(image).save(processed / "section.thumbnail.png")
    Image.fromarray(mask).save(processed / "section.mask.png")
    geometry = {
        "native_shape": [200, 200],
        "content_bbox_xywh": [0, 0, 200, 200],
        "thumbnail_shape": [20, 20],
        "bounds_source": "test",
        "mpp_xy": [0.5, 0.5],
        "mpp_source": "test",
    }
    (run_dir / "registration_result.json").write_text(
        json.dumps(
            {
                "reference_slide": str(tmp_path / "section.ndpi"),
                "slides": [
                    {
                        "path": str(tmp_path / "section.ndpi"),
                        "is_reference": True,
                        "geometry": geometry,
                        "transform": {"matrix": np.eye(3).tolist()},
                    }
                ],
            }
        )
    )
    semantic = tmp_path / "semantic"
    labels = semantic / "labels" / "k-2"
    labels.mkdir(parents=True)
    np.savez_compressed(
        labels / "001.npz",
        labels=np.array([0, 1], dtype=np.int16),
        reference_um_xy=np.array([[25, 25], [75, 75]], dtype=float),
        patch_size_px=np.int32(100),
        analysis_mpp=np.float64(0.5),
    )
    np.savez_compressed(semantic / "atlas_model.npz", pca_mean=np.zeros(2))
    payload = _seal_semantic_result(
        semantic,
        {
            "schema_version": 3,
            "primary_clusters": 2,
            "selected_k": 2,
            "cluster_counts": [2],
            "model": "atlas_model.npz",
            "palette": ["#d73027", "#1a9850"],
            "slides": [
                {
                    "id": "section.ndpi",
                    "labels": {"2": "labels/k-2/001.npz"},
                }
            ],
            "topology_pairs": [],
        },
    )
    (semantic / "semantic_result.json").write_text(json.dumps(payload))

    index = build_section_viewer(
        {"mouse": run_dir},
        tmp_path / "viewer",
        semantic_runs={"mouse": semantic},
    )

    manifest = json.loads((index.parent / "manifest.json").read_text())
    report = json.loads((index.parent / "build-report.json").read_text())
    slide = manifest["mice"][0]["slides"][0]
    assert report["three_version"] == "0.170.0"
    assert slide["semantic_texture"].endswith("-semantic.webp")
    assert slide["blend_texture"].endswith("-blend.webp")
    assert slide["semantic_textures"]["2"].endswith("-k2-semantic.webp")
    assert manifest["mice"][0]["semantic"]["selected_k"] == 2
    assert manifest["mice"][0]["semantic"]["cluster_count"] == 2
    assert 'id="mode"' in index.read_text()
    viewer = (index.parent / "viewer.js").read_text()
    assert "texture.dispose()" in viewer
    assert "semantic_texture" in viewer

    assert "semantic_textures" in viewer
    assert "new THREE.LineSegments" in viewer
    assert 'id="show-links"' in index.read_text()
    assert 'id="qc"' in index.read_text()
    assert "localStorage" not in viewer
    assert "slide_variance_fraction" in viewer
    assert "clusters" in viewer
    assert 'id="clusters"' in index.read_text()
    assert "const generation = ++loadGeneration" in viewer
    assert "if (generation !== loadGeneration)" in viewer


def test_order_review_builds_fixed_height_fingerprinted_grid(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    processed.mkdir()
    image = np.full((24, 30, 3), 230, dtype=np.uint8)
    image[5:20, 7:24] = (130, 80, 60)
    mask = np.zeros((24, 30), dtype=np.uint8)
    mask[5:20, 7:24] = 255
    Image.fromarray(image).save(processed / "HE.thumbnail.png")
    Image.fromarray(mask).save(processed / "HE.mask.png")
    proposal = tmp_path / "order.json"
    proposal.write_text(
        json.dumps(
            {
                "approved": False,
                "fingerprint": "abc123",
                "objective": 0.0,
                "runner_up_objective": None,
                "confidence_margin": None,
                "physically_calibrated": True,
                "slides": [
                    {
                        "order": 1,
                        "slide": "HE.ndpi",
                        "fixed": True,
                        "distance_from_previous": None,
                        "physical_tissue_area_um2": 2_000_000.0,
                    }
                ],
            }
        )
    )

    index = build_section_order_review(
        proposal,
        processed,
        tmp_path / "order-review",
    )

    manifest = json.loads((index.parent / "manifest.json").read_text())
    assert manifest["fingerprint"] == "abc123"
    assert manifest["slides"][0]["fixed"] is True
    assert "overflow:hidden" in (index.parent / "order-review.css").read_text()


def test_order_review_crop_normalizes_scanner_canvas() -> None:
    image = np.full((100, 160, 3), 240, dtype=np.uint8)
    mask = np.zeros((100, 160), dtype=bool)
    mask[40:60, 70:90] = True

    cropped = _tissue_review_crop(image, mask)

    assert cropped.shape[:2] == (30, 30)
    assert np.count_nonzero(cropped[..., 3] == 255) == 400
