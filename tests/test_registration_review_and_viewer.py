from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from histopia.registration._masking import TissueMaskResult
from histopia.registration._review import MaskReviewEntry, resolve_reviewed_mask
from histopia.registration._slides import SlideGeometry, discover_slides
from histopia.registration._viewer import (
    _tissue_review_crop,
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
    assert "three@0.170.0" in index.read_text()
    assert (
        "three/addons/controls/OrbitControls.js"
        in (index.parent / "viewer.js").read_text()
    )
    styles = (index.parent / "styles.css").read_text()
    viewer = (index.parent / "viewer.js").read_text()
    assert "html,body" in styles
    assert "overflow:hidden" in styles
    assert "width:100%!important;height:100%!important" in styles
    assert "renderer.setSize(box.width, box.height, true)" in viewer
    assert "new THREE.Box3().setFromObject(group)" in viewer
    assert "sphere.radius / 10000" in viewer
    assert "controls.minDistance" in viewer


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
