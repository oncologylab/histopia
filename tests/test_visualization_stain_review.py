from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest
from PIL import Image

from histopia.visualization import build_stain_review
from histopia.visualization._server import create_viewer_server
from histopia.visualization._stain_review import load_stain_review_issues


def test_stain_review_builds_path_free_decision_portal(tmp_path: Path) -> None:
    viewer = _viewer(tmp_path)
    output = tmp_path / "site" / "stain-review"

    index = build_stain_review(
        viewer,
        output,
        mice=["mouse"],
        issues={"mouse": {"2": "Upstream mask includes a glass border."}},
    )

    manifest = json.loads((output / "manifest.json").read_text())
    mouse = manifest["mice"][0]
    assert index == output / "index.html"
    assert manifest["scope"]["decision"] == (
        "continuous_relative_target_optical_density"
    )
    assert manifest["viewer_href"] == "../histopia/index.html"
    assert mouse["summary"] == {
        "quantified_slides": 3,
        "required_slides": 2,
        "blocking_issues": 1,
        "correction_accepted": 2,
        "threshold_accepted": 1,
    }
    assert mouse["slides"][1]["known_issue"] == (
        "Upstream mask includes a glass border."
    )
    assert mouse["slides"][1]["priority"]["blocking"] is True
    assert mouse["families"]["h-dab"] == {
        "selected_method": "fixed",
        "slide_count": 2,
        "correction_accepted": 1,
        "threshold_accepted": 1,
    }
    assert mouse["slides"][0]["assets"]["histology"].startswith(
        "../histopia/assets/mouse/"
    )
    assert str(tmp_path) not in (output / "manifest-data.js").read_text()
    assert "localStorage" in (output / "stain-review.js").read_text()
    assert "overflow: hidden" in (output / "stain-review.css").read_text()


def test_stain_review_rejects_missing_or_unsafe_assets(tmp_path: Path) -> None:
    viewer = _viewer(tmp_path)
    manifest_path = viewer / "manifest.json"
    payload = json.loads(manifest_path.read_text())
    payload["mice"][0]["slides"][0]["texture"] = "../private.webp"
    manifest_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="unsafe review asset"):
        build_stain_review(viewer, tmp_path / "review")


def test_stain_review_issue_file_requires_nested_string_notes(
    tmp_path: Path,
) -> None:
    valid = tmp_path / "valid.json"
    valid.write_text(json.dumps({"mouse": {"2": "  Inspect mask.  "}}))
    assert load_stain_review_issues(valid) == {"mouse": {"2": "Inspect mask."}}

    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps({"mouse": {"2": ""}}))
    with pytest.raises(ValueError, match="keys and notes"):
        load_stain_review_issues(invalid)


@pytest.mark.browser
def test_stain_review_browser_links_evidence_and_persists_draft(
    tmp_path: Path,
) -> None:
    playwright = pytest.importorskip("playwright.sync_api")
    viewer = _viewer(tmp_path)
    output = tmp_path / "site" / "stain-review"
    build_stain_review(
        viewer,
        output,
        issues={"mouse": {"2": "Inspect upstream mask."}},
    )
    server = create_viewer_server(tmp_path / "site", bind="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    errors: list[str] = []
    try:
        with playwright.sync_playwright() as runtime:
            browser = runtime.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1920, "height": 1080})
            page.on(
                "console",
                lambda message: (
                    errors.append(message.text) if message.type == "error" else None
                ),
            )
            page.goto(
                (
                    f"http://127.0.0.1:{server.server_port}/stain-review/"
                    "?mouse=mouse&slide=2"
                ),
                wait_until="networkidle",
            )
            page.wait_for_function(
                """() => [...document.querySelectorAll('[data-image]')]
                  .every(image => image.complete && image.naturalWidth > 0)"""
            )
            assert page.locator("#known-issue").is_visible()
            assert page.locator("[data-image]").count() == 4
            assert page.locator("#family-summary").inner_text() == (
                "sirius-red: fixed vectors | correction 1/1 | binary threshold 0/1"
            )
            page.locator("[data-check='specificity']").check()
            page.locator("[data-decision='hold']").click()
            page.reload(wait_until="networkidle")
            assert page.locator("[data-check='specificity']").is_checked()
            assert page.locator("[data-decision='hold']").get_attribute("class") == (
                "active"
            )
            assert page.evaluate(
                """() => document.documentElement.scrollWidth <= innerWidth
                  && document.documentElement.scrollHeight <= innerHeight"""
            )
            page.set_viewport_size({"width": 390, "height": 844})
            page.locator("#details-toggle").click()
            page.wait_for_timeout(200)
            evidence = page.locator("#evidence").bounding_box()
            assert evidence is not None and evidence["x"] < 390
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert errors == []


def _viewer(tmp_path: Path) -> Path:
    viewer = tmp_path / "site" / "histopia"
    asset_dir = viewer / "assets" / "mouse"
    asset_dir.mkdir(parents=True)
    slides = [
        _slide(
            asset_dir,
            order=1,
            label="DAB",
            family="h-dab",
            correction_accepted=True,
            threshold_accepted=True,
            rank=0.999,
            raw_leakage=0.08,
            corrected_leakage=0.02,
            residual=0.01,
        ),
        _slide(
            asset_dir,
            order=2,
            label="Sirius",
            family="sirius-red",
            correction_accepted=True,
            threshold_accepted=False,
            rank=0.997,
            raw_leakage=0.12,
            corrected_leakage=0.04,
            residual=0.03,
        ),
        _slide(
            asset_dir,
            order=3,
            label="Weak",
            family="h-dab",
            correction_accepted=False,
            threshold_accepted=False,
            rank=0.95,
            raw_leakage=0.03,
            corrected_leakage=0.04,
            residual=0.08,
        ),
    ]
    payload = {
        "mice": [
            {
                "id": "mouse",
                "stain": {
                    "fingerprint": "stain-fingerprint",
                    "display_max_od": 1.25,
                    "measurement": {
                        "quantity": "relative_chromogen_optical_density",
                        "analysis_mpp": 4.0,
                    },
                    "review": {
                        "approved": False,
                        "fingerprint_matches": True,
                    },
                    "families": {
                        "h-dab": {"selected_method": "fixed", "slide_count": 2},
                        "sirius-red": {
                            "selected_method": "fixed",
                            "slide_count": 1,
                        },
                    },
                },
                "slides": slides,
            }
        ]
    }
    (viewer / "manifest.json").write_text(json.dumps(payload))
    (viewer / "index.html").write_text("<!doctype html>")
    return viewer


def _slide(
    asset_dir: Path,
    *,
    order: int,
    label: str,
    family: str,
    correction_accepted: bool,
    threshold_accepted: bool,
    rank: float,
    raw_leakage: float,
    corrected_leakage: float,
    residual: float,
) -> dict[str, object]:
    names = {
        "texture": f"{order:03d}-{label}.webp",
        "raw": f"{order:03d}-{label}-raw.webp",
        "corrected": f"{order:03d}-{label}-corrected.webp",
        "raw_overlay": f"{order:03d}-{label}-raw-overlay.webp",
        "corrected_overlay": f"{order:03d}-{label}-corrected-overlay.webp",
    }
    for name in names.values():
        image = Image.new("RGBA", (80, 64), (245, 245, 242, 255))
        for x in range(12, 68):
            for y in range(8, 56):
                image.putpixel((x, y), (125 + order * 10, 82, 58, 255))
        image.save(asset_dir / name, "WEBP")
    prefix = "assets/mouse/"
    return {
        "id": f"slide-{order}.ndpi",
        "order": order,
        "label": label,
        "texture": prefix + names["texture"],
        "stain": {
            "quantified": True,
            "marker": label,
            "family": family,
            "textures": {
                "raw": prefix + names["raw"],
                "corrected": prefix + names["corrected"],
            },
            "overlay_textures": {
                "raw": prefix + names["raw_overlay"],
                "corrected": prefix + names["corrected_overlay"],
            },
            "quantiles": {
                "0.5": 0.1,
                "0.9": 0.4,
                "0.95": 0.6,
                "0.99": 0.9,
            },
            "qc": {
                "correction_accepted": correction_accepted,
                "threshold_accepted": threshold_accepted,
                "rank_correlation": rank,
                "raw_glass_leakage": raw_leakage,
                "corrected_glass_leakage": corrected_leakage,
                "background_spatial_cv_before": 0.02,
                "background_spatial_cv_after": 0.015,
                "median_reconstruction_residual": residual,
            },
        },
    }
