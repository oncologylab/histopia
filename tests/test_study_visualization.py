from __future__ import annotations

import json
import mimetypes
from pathlib import Path
from urllib.parse import unquote, urlsplit

import numpy as np
import pytest
from PIL import Image

from histopia.study._figure import PANEL_TITLES, build_study_figure
from histopia.study._manifest import file_sha256
from histopia.study._regions import (
    assign_cells_by_overlap,
    connected_tissue_regions,
    region_expression_summary,
)
from histopia.visualization._study_portal import attach_study_figure
from histopia.visualization._study_regions import build_region_view


def build_example(root):
    regions = connected_tissue_regions(
        np.array([[0, 0, -1, 0], [0, 0, -1, 1]]),
        semantic_fingerprint="semantic",
        approval_fingerprint="approval",
        section_id="m1:s1",
        pixel_size_um=10,
    )
    cells = np.array([[1, 1, 0, 2], [1, 1, 0, 3]])
    assignments = assign_cells_by_overlap(
        cells, regions.labels, regions, cell_fingerprint="cells"
    )
    summary = region_expression_summary(
        regions,
        assignments,
        [1, 2, 3],
        ["YAP"],
        [[1.0], [2.0], [np.nan]],
        np.array([[True], [True], [False]]),
        evidence_kind="measured",
        expression_fingerprint="measured",
    )
    background = root / "background.png"
    Image.new("RGB", (80, 40), "pink").save(background)
    output = root / "regions"
    build_region_view(
        regions,
        output,
        title="Test tissue",
        summary_rows=summary,
        background_png=background,
        study_fingerprint="study",
        summary_scope="Synthetic fixture",
    )
    return regions, output


def test_figure_exports_editable_labels_and_rejects_evidence_changes(tmp_path):
    study = {
        "schema_version": 1,
        "study_id": "synthetic",
        "mice": [],
        "slides": [],
        "evidence": [
            {
                "id": "observed",
                "status": "approved",
                "approval_fingerprint": "approval",
                "scope": "synthetic test",
            }
        ],
    }
    image = tmp_path / "source.png"
    Image.new("RGB", (16, 16), "purple").save(image)
    panels = {
        letter: {
            "status": "pending",
            "caption": "Independent evidence pending.",
            "pending_reason": "Test placeholder",
        }
        for letter in PANEL_TITLES
    }
    panels["A"] = {
        "status": "approved",
        "caption": "Synthetic fixture.",
        "assets": [
            {
                "path": str(image),
                "sha256": file_sha256(image),
                "evidence_id": "observed",
                "label": "Synthetic observed tissue",
                "physical_width_um": 100,
            }
        ],
    }
    out = tmp_path / "figure"
    build_study_figure(study, {"panels": panels}, out)
    svg = (out / "histopia-six-panel.svg").read_text()
    assert "<text" in svg
    assert "PENDING EVIDENCE" in svg
    assert str(tmp_path) not in svg
    provenance = json.loads((out / "figure-provenance.json").read_text())
    assert "path" not in provenance["assets"][0]
    assert provenance["source_evidence"][0]["approval_fingerprint"] == "approval"
    assert (out / "histopia-six-panel.pdf").read_bytes().startswith(b"%PDF")
    for name, sha in provenance["exports"].items():
        assert file_sha256(out / name) == sha
    repeated = tmp_path / "repeated-figure"
    build_study_figure(study, {"panels": panels}, repeated)
    assert all(
        file_sha256(repeated / name) == sha
        for name, sha in provenance["exports"].items()
    )
    portal = tmp_path / "portal"
    portal.mkdir()
    existing_tab = {"id": "registration", "href": "registration/index.html"}
    (portal / "manifest.json").write_text(json.dumps({"tabs": [existing_tab]}))
    (out / "local-inputs.json").write_text('{"private": "not for web export"}')
    attached = attach_study_figure(out, portal)
    assert attached.is_file()
    assert not (attached.parent / "local-inputs.json").exists()
    assert attach_study_figure(out, portal) == attached
    tabs = json.loads((portal / "manifest.json").read_text())["tabs"]
    assert tabs[0] == existing_tab
    assert len(tabs) == 2
    assert tabs[1]["id"] == "study-figure"
    (out / "histopia-six-panel.pdf").write_bytes(b"altered")
    with pytest.raises(ValueError, match="export hash mismatch"):
        attach_study_figure(out, portal)
    Image.new("RGB", (16, 16), "blue").save(image)
    with pytest.raises(ValueError, match="hash mismatch"):
        build_study_figure(study, {"panels": panels}, out)


@pytest.mark.browser
@pytest.mark.parametrize("prefix", ["", "/workspace/proxy/8765"])
def test_single_region_navigation_export_and_proxy_urls(tmp_path, prefix):
    playwright = pytest.importorskip("playwright.sync_api")
    regions, out = build_example(tmp_path)
    origin = "http://histopia.test"
    base = prefix + "/study/regions/"
    errors = []
    escaped = []
    with playwright.sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1400, "height": 900})

        def respond(route):
            path = unquote(urlsplit(route.request.url).path)
            if not path.startswith(base):
                escaped.append(path)
                route.fulfill(status=404)
                return
            file = out / (path[len(base) :] or "index.html")
            if file.is_file():
                route.fulfill(
                    content_type=mimetypes.guess_type(file)[0]
                    or "application/octet-stream",
                    body=file.read_bytes(),
                )
            else:
                route.fulfill(status=404)

        page.route(origin + "/**", respond)
        page.on("pageerror", lambda error: errors.append(str(error)))
        requested = regions.regions[1]["region_id"]
        page.goto(origin + base + "index.html?keep=1&region=" + requested)
        assert page.locator("#outline").get_attribute("data-region-id") == requested
        assert page.locator("#tissue path#outline").count() == 1
        assert page.locator("#outline").get_attribute("fill") == "none"
        page.locator("#previous").click()
        chosen = page.locator("#outline").get_attribute("data-region-id")
        assert chosen != requested
        assert "keep=1" in page.url and "region=" + chosen in page.url
        page.reload()
        assert page.locator("#region").input_value() == chosen
        assert page.locator("#summary tr").get_attribute("data-region-id") == chosen
        page.locator("#background").uncheck()
        assert not page.locator("#tissue-background").is_visible()
        with page.expect_download() as download:
            page.locator("#export").click()
        saved = Path(download.value.path()).read_text()
        assert chosen in saved and "region_fingerprint" in saved
        assert saved.count('id="outline"') == 1
        assert page.locator("#evidence option").evaluate_all(
            "options => options.map(o => o.value)"
        ) == ["measured"]
        page.set_viewport_size({"width": 390, "height": 844})
        assert page.evaluate(
            "document.documentElement.scrollWidth <= window.innerWidth"
        )
        assert not errors and not escaped
        browser.close()


def test_region_export_preserves_rectangular_calibration_and_back_link(tmp_path):
    regions = connected_tissue_regions(
        np.array([[0, 0], [-1, 1]]),
        semantic_fingerprint="semantic",
        approval_fingerprint="review",
        section_id="section",
        pixel_size_um=10,
        pixel_size_um_xy=(10, 20),
    )
    out = tmp_path / "rectangular"
    build_region_view(
        regions,
        out,
        title="Rectangular",
        study_fingerprint="study",
        back_href="../../index.html?source=test&organ=tongue",
        back_label="Tissue review",
    )
    svg = (out / "all-regions.svg").read_text()
    assert 'viewBox="0 0 2 4.0"' in svg and 'transform="scale(1,2.0)"' in svg
    assert json.loads((out / "regions.json").read_text())["pixel_size_um_xy"] == [
        10,
        20,
    ]
    page = (out / "index.html").read_text()
    assert 'href="../../index.html?source=test&amp;organ=tongue">Tissue review' in page


@pytest.mark.browser
def test_region_view_has_an_offline_and_no_javascript_fallback(tmp_path):
    playwright = pytest.importorskip("playwright.sync_api")
    _, out = build_example(tmp_path)
    with playwright.sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True)
        context = browser.new_context(java_script_enabled=False)
        page = context.new_page()
        page.goto((out / "index.html").as_uri())
        assert page.locator("#static-fallback").is_visible()
        assert page.locator('a[href="all-regions.svg"]').is_visible()
        page.locator('a[href="all-regions.svg"]').click()
        assert page.locator("svg path").count() == 3
        context.close()
        page = browser.new_page()
        page.goto((out / "index.html").as_uri())
        assert page.locator("#outline").get_attribute("d")
        browser.close()
