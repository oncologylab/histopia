from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from histopia.visualization._nonrigid_review import build_non_rigid_review


def test_non_rigid_review_builds_path_free_fixed_viewport_browser(
    tmp_path: Path,
) -> None:
    validation = tmp_path / "validation"
    qc = validation / "qc"
    qc.mkdir(parents=True)
    slides = ("section-<a>.ndpi", "section-b.ndpi")
    colors = (
        (210, 210, 210),
        (180, 100, 80),
        (70, 150, 170),
        (190, 140, 110),
        (120, 190, 100),
        (40, 40, 40),
    )
    for slide in slides:
        _write_contact(qc / f"{Path(slide).stem}.contact.png", colors)
    summary = {
        "schema_version": 2,
        "status": "provisional_validation",
        "mouse_id": "test-mouse",
        "reference_slide": "reference.ndpi",
        "reference_shape": [24, 40],
        "slides": [
            _slide_row(1, slides[0], accepted=True),
            _slide_row(2, slides[1], accepted=False),
        ],
    }
    (validation / "summary.json").write_text(json.dumps(summary))

    index = build_non_rigid_review(validation, tmp_path / "review", workers=2)

    manifest = json.loads((index.parent / "manifest.json").read_text())
    assert manifest["status"] == "provisional_validation"
    assert manifest["slide_count"] == 2
    assert manifest["accepted_count"] == 1
    assert len(tuple((index.parent / "assets").glob("*.webp"))) == 9
    assert all("blend" not in row["assets"] for row in manifest["slides"])
    assert all(
        not value.startswith("/")
        for row in manifest["slides"]
        for value in row["assets"].values()
    )
    assert str(tmp_path) not in (index.parent / "manifest.json").read_text()
    assert "overflow: hidden" in (index.parent / "nonrigid-review.css").read_text()
    script = (index.parent / "nonrigid-review.js").read_text()
    assert "name.textContent = row.slide" in script
    assert "${row.slide}</span>" not in script
    candidate = index.parent / manifest["slides"][0]["assets"]["candidate"]
    with Image.open(candidate) as image:
        assert image.size == (40, 24)
        red, green, blue = image.convert("RGB").getpixel((20, 12))
        assert blue > red
        assert blue > green


def test_non_rigid_review_accepts_standard_registration_run(tmp_path: Path) -> None:
    run = tmp_path / "mouse-1"
    processed = run / "processed"
    qc = run / "qc" / "non_rigid"
    processed.mkdir(parents=True)
    qc.mkdir(parents=True)
    Image.new("RGB", (40, 24), "gray").save(processed / "reference.thumbnail.png")
    _write_contact(
        qc / "section-a.contact.png",
        (
            (210, 210, 210),
            (180, 100, 80),
            (70, 150, 170),
            (190, 140, 110),
            (120, 190, 100),
            (40, 40, 40),
        ),
    )
    transform = {
        key: value
        for key, value in _slide_row(1, "section-a.ndpi", accepted=True).items()
        if key not in {"order", "slide", "label"}
    }
    result = {
        "reference_slide": "/private/raw/reference.ndpi",
        "slides": [
            {
                "path": "/private/raw/reference.ndpi",
                "is_reference": True,
                "non_rigid_transform": None,
            },
            {
                "path": "/private/raw/section-a.ndpi",
                "is_reference": False,
                "non_rigid_transform": transform,
            },
        ],
    }
    (run / "registration_result.json").write_text(json.dumps(result))

    index = build_non_rigid_review(run, tmp_path / "review")

    manifest_text = (index.parent / "manifest.json").read_text()
    manifest = json.loads(manifest_text)
    assert manifest["mouse_id"] == "mouse-1"
    assert manifest["reference_slide"] == "reference.ndpi"
    assert manifest["slide_count"] == 1
    assert manifest["slides"][0]["order"] == 2
    assert manifest["slides"][0]["slide"] == "section-a.ndpi"
    assert "/private/raw" not in manifest_text


def _write_contact(path: Path, colors: tuple[tuple[int, int, int], ...]) -> None:
    pane_width = 40
    pane_height = 24
    gap = 8
    image = Image.new(
        "RGB",
        (6 * pane_width + 5 * gap, 86 + pane_height + 24),
        "white",
    )
    for index, color in enumerate(colors):
        left = index * (pane_width + gap)
        pane = Image.new("RGB", (pane_width, pane_height), color)
        image.paste(pane, (left, 86))
    image.save(path)


def _slide_row(order: int, slide: str, *, accepted: bool) -> dict[str, object]:
    return {
        "order": order,
        "slide": slide,
        "label": Path(slide).stem,
        "accepted": accepted,
        "method": "dis_tissue_supported",
        "initial_similarity": 0.4,
        "final_similarity": 0.6,
        "initial_mask_dice": 0.9,
        "final_mask_dice": 0.89,
        "jacobian_p01": 0.6,
        "jacobian_p99": 1.4,
        "displacement_p95": 8.0,
        "inverse_consistency_p95": 6.0,
        "warnings": [] if accepted else ["candidate rejected"],
        "sparse_feature_validation": {
            "status": "available",
            "detector": "orb",
            "mutual_matches": 20,
            "coherent_matches": 15,
            "initial_median_residual_px": 5.0,
            "final_median_residual_px": 2.0,
            "initial_p95_residual_px": 8.0,
            "final_p95_residual_px": 4.0,
            "improved_fraction": 0.8,
            "warnings": [],
        },
    }
