from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from histopia.visualization import build_methods_review


def test_methods_review_writes_path_free_scientific_contract(tmp_path: Path) -> None:
    index = build_methods_review(
        tmp_path / "methods",
        cohorts=("4312", "mouse<2>"),
        stages=("registration", "stain", "cells", "protein", "topology"),
    )

    manifest = json.loads((index.parent / "manifest.json").read_text())
    assert manifest == {
        "schema_version": 1,
        "method_revision": "histopia-review-methods-v2",
        "cohorts": ["4312", "mouse<2>"],
        "workflow_stages": [
            "registration",
            "stain",
            "cells",
            "protein",
            "topology",
        ],
    }
    html = index.read_text()
    for phrase in (
        "4 µm/px",
        "Hard tissue constraint",
        "target-chromogen-free UNI2-h",
        "Semantic regions are coarse",
        "uniformly assumed",
        "No residual is shown without target truth",
        "If neither passes, it says no approved",
        "leaves exploratory selection explicit",
        "Shared-trunk fits retain a distinct antibody head",
        "s41587-023-02019-9",
        "s41592-025-02770-8",
    ):
        assert phrase in html
    assert "mouse&lt;2&gt;" in html
    assert str(tmp_path) not in html
    css = (index.parent / "methods-review.css").read_text()
    assert "Histopia review theme" in css
    assert "overflow-x:hidden" in css


@pytest.mark.browser
def test_methods_review_is_readable_without_horizontal_overflow(
    tmp_path: Path,
) -> None:
    playwright = pytest.importorskip("playwright.sync_api")
    index = build_methods_review(
        tmp_path / "methods",
        cohorts=("4312", "4630", "6180"),
        stages=(
            "registration",
            "atlas",
            "stain",
            "topology",
            "cells",
            "protein",
            "annotations",
        ),
    )
    errors: list[str] = []
    with playwright.sync_playwright() as runtime:
        browser = getattr(
            runtime, os.environ.get("HISTOPIA_BROWSER", "chromium")
        ).launch(headless=True)
        page = browser.new_page(viewport={"width": 1920, "height": 1080})
        page.on(
            "console",
            lambda message: (
                errors.append(message.text) if message.type == "error" else None
            ),
        )
        page.on("requestfailed", lambda request: errors.append(request.url))
        page.goto(index.as_uri(), wait_until="load")
        assert page.locator(".method-card").count() == 7
        assert (
            page.locator(".build-scope")
            .inner_text()
            .endswith(
                "registration, 3D atlas, stain OD, topology, cell boundaries, "
                "protein prediction, annotations"
            )
        )
        for width, height in ((1920, 1080), (3840, 2160), (390, 844)):
            page.set_viewport_size({"width": width, "height": height})
            page.wait_for_timeout(50)
            layout = page.evaluate(
                """() => ({
                  x: document.documentElement.scrollWidth > innerWidth,
                  mainWidth: document.querySelector('main')
                    .getBoundingClientRect().width,
                  tocWidth: document.querySelector('.methods-toc')
                    .getBoundingClientRect().width,
                })"""
            )
            assert not layout["x"]
            assert layout["mainWidth"] > 0
            assert layout["tocWidth"] == pytest.approx(width, abs=1)
        page.locator('a[href="#protein"]').click()
        assert page.url.endswith("#protein")
        assert page.locator("#protein h2").is_visible()
        browser.close()
    assert errors == []
