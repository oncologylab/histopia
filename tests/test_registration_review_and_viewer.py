from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from histopia.registration._masking import TissueMaskResult
from histopia.registration._review import MaskReviewEntry, resolve_reviewed_mask
from histopia.registration._slides import SlideGeometry, discover_slides
from histopia.registration._viewer import build_section_viewer


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
