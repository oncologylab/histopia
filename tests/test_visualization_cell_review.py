import hashlib
import io
import json
import os
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from histopia.cells._result import write_cell_result
from histopia.visualization import build_cell_review
from histopia.visualization._server import create_viewer_server
from histopia.visualization._wsi_tiles import WsiTileService


def test_cell_review_builds_fixed_viewport_application(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "cell_result.json").write_text(
        json.dumps(
            {
                "fingerprint": "f" * 64,
                "slides": [
                    {
                        "section": "001",
                        "slide": "sample.ndpi",
                        "cell_count": 42,
                        "is_reference": True,
                    }
                ],
            }
        )
    )

    index = build_cell_review({"mouse": run}, tmp_path / "site")

    assert index.is_file()
    assert "overflow:hidden" in (index.parent / "cell-review.css").read_text().replace(
        " ", ""
    )
    stylesheet = (index.parent / "cell-review.css").read_text().replace(" ", "")
    assert "white-space:nowrap" in stylesheet
    assert "#section{flex:11360px;min-width:150px;}" in stylesheet
    script = (index.parent / "cell-review.js").read_text()
    assert "globalThis.histopiaCellViewer = viewer" in script
    assert "/api/reviews/provisional" in script
    assert "browser-provisional" in script
    assert "OpenSeadragon" in script
    assert "preserveViewport: false" in script
    assert "document.body.dataset.sectionReady = section.id" in script
    assert "document.body.dataset.cohortReady = cohort.id" in script
    assert "active.viewport.goHome(true)" in script
    assert "active.getFullyLoaded()" in script
    assert 'addHandler("fully-loaded-change", check)' in script
    assert 'id="viewer-loading"' in index.read_text()
    assert "item.sections.length" in script
    manifest = json.loads((index.parent / "manifest.json").read_text())
    assert manifest["cohorts"][0]["method"] == "unknown"


def test_cell_review_places_validated_containment_before_direct_baseline(
    tmp_path: Path,
) -> None:
    runs = {}
    for method in ("direct", "containment"):
        run = tmp_path / method
        run.mkdir()
        (run / "cell_result.json").write_text(
            json.dumps(
                {
                    "request": {"model": "cpsam", "method": method},
                    "slides": [
                        {"section": "001", "slide": "sample.ndpi", "cell_count": 1}
                    ],
                }
            )
        )
        runs[method] = run

    build_cell_review(runs, tmp_path / "site")

    manifest = json.loads((tmp_path / "site/manifest.json").read_text())
    assert [item["method"] for item in manifest["cohorts"]] == [
        "containment",
        "direct",
    ]


@pytest.mark.browser
def test_cell_review_accept_is_one_click_and_persists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    playwright = pytest.importorskip("playwright.sync_api")
    registration = tmp_path / "registration"
    registration.mkdir()
    registration_result = registration / "registration_result.json"
    registration_result.write_text("{}\n")
    run = tmp_path / "cell-run"
    (run / "labels").mkdir(parents=True)
    (run / "qc").mkdir()
    (run / "preflight.json").write_text("{}\n")
    (run / "labels/001.cells.tiff").write_bytes(b"labels")
    (run / "labels/002.cells.tiff").write_bytes(b"labels")
    (run / "qc/001.json").write_text("{}\n")
    (run / "qc/002.json").write_text("{}\n")
    write_cell_result(
        run,
        {
            "schema_version": 1,
            "registration_result_sha256": hashlib.sha256(
                registration_result.read_bytes()
            ).hexdigest(),
            "preflight": "preflight.json",
            "slides": [
                {
                    "section": "001",
                    "slide": "sample.ndpi",
                    "cell_count": 42,
                    "labels": "labels/001.cells.tiff",
                    "qc": "qc/001.json",
                },
                {
                    "section": "002",
                    "slide": "sample-2.ndpi",
                    "cell_count": 43,
                    "labels": "labels/002.cells.tiff",
                    "qc": "qc/002.json",
                },
            ],
        },
    )
    root = tmp_path / "site"
    build_cell_review({"mouse": run}, root / "cells")
    config = tmp_path / "review.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "feedback_dir": str(tmp_path / "feedback"),
                "cohorts": {
                    "mouse": {
                        "registration": str(registration),
                        "cells": str(run),
                    }
                },
            }
        )
    )
    raw = np.full((128, 128, 3), 238, dtype=np.uint8)
    raw[20:108, 24:104] = (170, 90, 100)
    raw_buffer = io.BytesIO()
    Image.fromarray(raw).save(raw_buffer, "JPEG")
    cells = np.zeros((128, 128, 4), dtype=np.uint8)
    cells[20:108, (24, 103)] = (255, 36, 24, 235)
    cells[(20, 107), 24:104] = (255, 36, 24, 235)
    cell_buffer = io.BytesIO()
    Image.fromarray(cells).save(cell_buffer, "PNG")
    digest = "a" * 64
    rendered_layers: list[str] = []

    class FakeTiles:
        def metadata(self, cohort: str, section: str) -> dict[str, object]:
            assert cohort == "mouse"
            assert section in {"001", "002"}
            levels = [
                {
                    "width": max(1, (128 + 2 ** (7 - level) - 1) // 2 ** (7 - level)),
                    "height": max(1, (128 + 2 ** (7 - level) - 1) // 2 ** (7 - level)),
                }
                for level in range(8)
            ]
            layer = {
                "digest": digest,
                "tile_size": 128,
                "width": 128,
                "height": 128,
                "levels": levels,
                "microns_per_pixel": 0.5,
            }
            return {
                "cohort": cohort,
                "section": section,
                "label": "sample",
                "layers": {
                    "raw": {**layer, "format": "jpg"},
                    "cells": {**layer, "format": "png"},
                },
            }

        def render_tile(self, *args):
            layer = args[2]
            level = args[4]
            rendered_layers.append(layer)
            size = max(1, (128 + 2 ** (7 - level) - 1) // 2 ** (7 - level))
            if layer == "raw":
                buffer = io.BytesIO()
                Image.fromarray(raw).resize((size, size)).save(buffer, "JPEG")
                return buffer.getvalue(), "image/jpeg", f'"raw-{level}"'
            buffer = io.BytesIO()
            Image.fromarray(cells).resize((size, size)).save(buffer, "PNG")
            return buffer.getvalue(), "image/png", f'"cells-{level}"'

    monkeypatch.setattr(
        WsiTileService,
        "from_runs",
        classmethod(lambda cls, runs, **kwargs: FakeTiles()),
    )
    server = create_viewer_server(
        root,
        bind="127.0.0.1",
        port=0,
        required_routes=("cells",),
        review_config=config,
        public_review_write=True,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    errors: list[str] = []
    try:
        with playwright.sync_playwright() as runtime:
            browser_type = getattr(
                runtime, os.environ.get("HISTOPIA_BROWSER", "chromium")
            )
            browser = browser_type.launch(headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 900})
            page.on(
                "console",
                lambda message: (
                    errors.append(message.text) if message.type == "error" else None
                ),
            )
            page.on("crash", lambda: errors.append("page crashed"))
            page.on("pageerror", lambda error: errors.append(f"page error: {error}"))
            page.goto(
                f"http://127.0.0.1:{server.server_port}/cells/",
                wait_until="domcontentloaded",
            )
            page.wait_for_function(
                "document.body.dataset.sectionReady === '001'",
                polling=100,
            )
            page.evaluate(
                """() => {
                  const select = document.querySelector('#section');
                  select.value = '002';
                  select.dispatchEvent(new Event('change'));
                  select.value = '001';
                  select.dispatchEvent(new Event('change'));
                }"""
            )
            page.wait_for_function(
                "document.body.dataset.sectionReady === '001'",
                polling=100,
            )
            page.wait_for_timeout(300)
            view_state = page.evaluate(
                """() => {
                  const viewer = globalThis.histopiaCellViewer;
                  return {
                    items: viewer.world.getItemCount(),
                    zoomRatio: viewer.viewport.getZoom(true) /
                      viewer.viewport.getHomeZoom(),
                  };
                }"""
            )
            assert view_state["items"] == 2
            assert view_state["zoomRatio"] == pytest.approx(1, abs=0.01)
            for width, height in (
                (1920, 1080),
                (1188, 1000),
                (3840, 2160),
                (390, 844),
            ):
                page.set_viewport_size({"width": width, "height": height})
                page.wait_for_timeout(100)
                overflow = page.evaluate(
                    """() => ({
                      x: document.documentElement.scrollWidth > innerWidth,
                      y: document.documentElement.scrollHeight > innerHeight,
                      viewer: document.querySelector('#viewer')
                        .getBoundingClientRect().toJSON(),
                      title: (() => {
                        const title = document.querySelector('header strong');
                        const box = title.getBoundingClientRect();
                        return {
                          visible: getComputedStyle(title).display !== 'none',
                          height: box.height,
                          lineHeight: parseFloat(getComputedStyle(title).lineHeight),
                        };
                      })(),
                    })"""
                )
                assert not overflow["x"]
                assert not overflow["y"]
                assert overflow["viewer"]["width"] > 0
                assert overflow["viewer"]["height"] > 0
                if overflow["title"]["visible"]:
                    assert overflow["title"]["height"] <= max(
                        24, overflow["title"]["lineHeight"] * 1.1
                    )
            page.wait_for_timeout(500)
            page.evaluate("document.querySelector('#accept').click()")
            result = json.loads((run / "cell_result.json").read_text())
            for _ in range(50):
                review_path = (
                    tmp_path
                    / "feedback/provisional/mouse/cells"
                    / f"{result['fingerprint']}.json"
                )
                if review_path.is_file():
                    break
                time.sleep(0.05)
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert not errors
    assert set(rendered_layers) == {"raw", "cells"}
    formal_review = json.loads((run / "cell_review.json").read_text())
    assert formal_review["sections"]["001"]["accepted"] is False
    provisional = json.loads(review_path.read_text())
    assert provisional["records"][-1]["decision"] == "accept"
    assert provisional["records"][-1]["provisional"] is True
