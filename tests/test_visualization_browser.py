from __future__ import annotations

import io
import json
import threading
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from histopia.visualization._server import create_viewer_server
from histopia.visualization._viewer import (
    _INDEX_HTML,
    _STYLES_CSS,
    _VIEWER_JS,
    THREE_VERSION,
)


@pytest.mark.browser
def test_viewer_fits_desktop_and_ignores_stale_mouse_loads(tmp_path: Path) -> None:
    playwright = pytest.importorskip("playwright.sync_api")
    root = tmp_path / "viewer"
    site = root / "histopia"
    assets = site / "assets"
    assets.mkdir(parents=True)
    mice = [
        _browser_mouse(assets, "first", 2, (170, 40, 40)),
        _browser_mouse(assets, "second", 3, (30, 120, 180)),
    ]
    (site / "manifest.json").write_text(json.dumps({"schema_version": 1, "mice": mice}))
    (site / "index.html").write_text(_INDEX_HTML.replace("__THREE__", THREE_VERSION))
    (site / "viewer.js").write_text(_VIEWER_JS.replace("__THREE__", THREE_VERSION))
    (site / "styles.css").write_text(_STYLES_CSS)

    server = create_viewer_server(root, bind="127.0.0.1", port=0)
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
            page.on("requestfailed", lambda request: errors.append(request.url))
            page.goto(
                f"http://127.0.0.1:{server.server_port}/histopia/",
                wait_until="networkidle",
            )
            page.wait_for_selector("#sections li")
            for width, height in ((1920, 1080), (3840, 2160)):
                page.set_viewport_size({"width": width, "height": height})
                page.wait_for_timeout(100)
                overflow = page.evaluate(
                    """() => ({
                      x: document.documentElement.scrollWidth > innerWidth,
                      y: document.documentElement.scrollHeight > innerHeight,
                      canvas: document.querySelector('canvas').getBoundingClientRect(),
                    })"""
                )
                assert not overflow["x"]
                assert not overflow["y"]
                assert overflow["canvas"]["width"] > 0
                assert overflow["canvas"]["height"] > 0
            page.evaluate(
                """() => {
                  const select = document.querySelector('#mouse');
                  select.value = 'first';
                  select.dispatchEvent(new Event('change'));
                  select.value = 'second';
                  select.dispatchEvent(new Event('change'));
                }"""
            )
            page.wait_for_function(
                """() => document.querySelector('#mouse').value === 'second'
                  && document.querySelectorAll('#sections li').length === 3"""
            )
            screenshot = page.locator("canvas").screenshot()
            pixels = np.asarray(Image.open(io.BytesIO(screenshot)).convert("RGB"))
            assert np.ptp(pixels.reshape(-1, 3), axis=0).max() > 20
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert errors == []


def _browser_mouse(
    assets: Path,
    mouse_id: str,
    slide_count: int,
    color: tuple[int, int, int],
) -> dict[str, object]:
    mouse_assets = assets / mouse_id
    mouse_assets.mkdir()
    slides = []
    for index in range(slide_count):
        image = np.full((160, 240, 4), (*color, 255), dtype=np.uint8)
        image[20:140, 30 + index * 5 : 210 - index * 5, :3] = (
            min(color[0] + 50, 255),
            min(color[1] + 50, 255),
            min(color[2] + 50, 255),
        )
        filename = f"{index + 1:03d}.webp"
        Image.fromarray(image).save(mouse_assets / filename, "WEBP")
        slides.append(
            {
                "id": f"{mouse_id}-{index + 1}.ndpi",
                "label": f"Section {index + 1}",
                "order": index + 1,
                "texture": f"assets/{mouse_id}/{filename}",
                "reference": index == 0,
            }
        )
    return {
        "id": mouse_id,
        "provisional_order": False,
        "width": 240,
        "height": 160,
        "slides": slides,
        "semantic": None,
    }
