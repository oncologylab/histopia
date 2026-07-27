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
    _write_viewer_runtime,
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
        _browser_mouse(assets, "second", 3, (30, 120, 180), semantic=True),
    ]
    (site / "manifest.json").write_text(json.dumps({"schema_version": 1, "mice": mice}))
    (site / "index.html").write_text(_INDEX_HTML)
    (site / "viewer.js").write_text(_VIEWER_JS)
    (site / "styles.css").write_text(_STYLES_CSS)
    _write_viewer_runtime(site)

    server = create_viewer_server(root, bind="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    errors: list[str] = []
    stale_failures: list[str] = []
    stale_console_errors: list[str] = []
    stale_failure_armed = False
    try:
        with playwright.sync_playwright() as runtime:
            browser = runtime.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1920, "height": 1080})
            page.add_init_script(
                """window.__histopiaRafCount = 0;
                const originalRaf = window.requestAnimationFrame;
                window.requestAnimationFrame = function(callback) {
                  return originalRaf.call(window, function(timestamp) {
                    window.__histopiaRafCount += 1;
                    return callback(timestamp);
                  });
                };"""
            )

            def record_console_error(message) -> None:
                nonlocal stale_failure_armed
                if message.type != "error":
                    return
                if (
                    stale_failure_armed
                    and message.text == "Failed to load resource: net::ERR_FAILED"
                ):
                    stale_console_errors.append(message.text)
                    stale_failure_armed = False
                else:
                    errors.append(message.text)

            page.on("console", record_console_error)
            page.on(
                "requestfailed",
                lambda request: (
                    stale_failures.append(request.url)
                    if "/assets/first/001.webp" in request.url
                    else errors.append(request.url)
                ),
            )
            page.on(
                "request",
                lambda request: (
                    errors.append(request.url)
                    if not request.url.startswith(
                        f"http://127.0.0.1:{server.server_port}/"
                    )
                    else None
                ),
            )
            page.goto(
                f"http://127.0.0.1:{server.server_port}/histopia/?mouse=second",
                wait_until="networkidle",
            )
            page.wait_for_function(
                """() => document.querySelector('#mouse').value === 'second'
                  && document.querySelectorAll('#sections li').length === 3
                  && document.querySelector('#viewport').getAttribute('aria-busy')
                    === 'false'"""
            )
            assert page.url.endswith("/histopia/?mouse=second")
            assert page.locator("#order-status").inner_text() == (
                "Registration approval required"
            )
            assert page.locator("#order-status").evaluate(
                "element => element.scrollWidth <= element.clientWidth + 1"
            )
            ready_screenshot = page.locator("canvas").screenshot()
            ready_pixels = np.asarray(
                Image.open(io.BytesIO(ready_screenshot)).convert("RGB")
            )
            assert np.ptp(ready_pixels.reshape(-1, 3), axis=0).max() > 20
            page.locator("#mode button[data-mode='semantic']").click()
            page.wait_for_function(
                """() => document.querySelector('#viewport')
                    .getAttribute('aria-busy') === 'false'
                  && document.querySelector(
                    "#mode button[data-mode='semantic']"
                  ).classList.contains('active')"""
            )
            semantic_screenshot = page.locator("canvas").screenshot()
            semantic_pixels = np.asarray(
                Image.open(io.BytesIO(semantic_screenshot)).convert("RGB")
            )
            assert np.ptp(semantic_pixels.reshape(-1, 3), axis=0).max() > 20
            assert page.evaluate(
                """() => {
                  const canvas = document.querySelector('canvas');
                  const gl = canvas.getContext('webgl2');
                  window.__histopiaLoseContext =
                    gl.getExtension('WEBGL_lose_context');
                  if (!window.__histopiaLoseContext) return false;
                  window.__histopiaLoseContext.loseContext();
                  return true;
                }"""
            )
            page.wait_for_function(
                """() => document.querySelector('#viewport')
                  .getAttribute('aria-busy') === 'true'"""
            )
            page.evaluate("window.__histopiaLoseContext.restoreContext()")
            page.wait_for_function(
                """() => document.querySelector('#viewport')
                  .getAttribute('aria-busy') === 'false'"""
            )
            restored_screenshot = page.locator("canvas").screenshot()
            restored_pixels = np.asarray(
                Image.open(io.BytesIO(restored_screenshot)).convert("RGB")
            )
            assert np.ptp(restored_pixels.reshape(-1, 3), axis=0).max() > 20
            np.testing.assert_allclose(
                restored_pixels[0, 0],
                [244, 245, 243],
                atol=2,
            )
            page.wait_for_timeout(1800)
            page.evaluate("window.__histopiaRafCount = 0")
            page.wait_for_timeout(500)
            assert page.evaluate("window.__histopiaRafCount") <= 2
            idle_screenshot = page.locator("canvas").screenshot()
            idle_pixels = np.asarray(
                Image.open(io.BytesIO(idle_screenshot)).convert("RGB")
            )
            assert np.ptp(idle_pixels.reshape(-1, 3), axis=0).max() > 20
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
            page.set_viewport_size({"width": 390, "height": 844})
            page.evaluate(
                """() => {
                  const sidebar = document.querySelector('aside');
                  sidebar.scrollTop = sidebar.scrollHeight;
                }"""
            )
            assert page.locator("aside").evaluate("element => element.scrollTop") > 0
            stale_failure_armed = True
            page.route(
                "**/assets/first/001.webp",
                lambda route: route.abort("failed"),
            )
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
                  && document.querySelectorAll('#sections li').length === 3
                  && document.querySelector('#viewport').getAttribute('aria-busy')
                    === 'false'"""
            )
            assert page.url.endswith("/histopia/?mouse=second")
            assert page.locator("aside").evaluate("element => element.scrollTop") == 0
            assert page.locator("#slide-focus").inner_text() != "Load failed"
            assert len(stale_failures) == 1
            assert stale_console_errors == ["Failed to load resource: net::ERR_FAILED"]
            page.set_viewport_size({"width": 1920, "height": 1080})
            page.locator("#next-slide").click()
            assert page.locator("#slide-focus").inner_text() == "1 / 3"
            assert page.locator("#sections input:checked").count() == 1
            page.locator("#next-slide").click()
            assert page.locator("#slide-focus").inner_text() == "2 / 3"
            assert page.locator("#sections input:checked").count() == 1
            page.locator("#select-all").click()
            assert page.locator("#slide-focus").inner_text() == "3 selected"
            assert page.locator("#sections input:checked").count() == 3
            page.locator("#deselect-all").click()
            assert page.locator("#slide-focus").inner_text() == "0 selected"
            assert page.locator("#sections input:checked").count() == 0
            page.locator("#sections li").nth(2).locator("span").click()
            assert page.locator("#slide-focus").inner_text() == "3 / 3"
            assert page.locator("#sections input:checked").count() == 1
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
    *,
    semantic: bool = False,
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
        slide = {
            "id": f"{mouse_id}-{index + 1}.ndpi",
            "label": f"Section {index + 1}",
            "order": index + 1,
            "texture": f"assets/{mouse_id}/{filename}",
            "reference": index == 0,
        }
        if semantic:
            semantic_image = image.copy()
            semantic_image[30:130, 45:195, :3] = (215, 50, 65)
            semantic_name = f"{index + 1:03d}-k5-semantic.webp"
            blend_name = f"{index + 1:03d}-blend.webp"
            Image.fromarray(semantic_image).save(
                mouse_assets / semantic_name,
                "WEBP",
            )
            Image.fromarray((image // 2 + semantic_image // 2).astype(np.uint8)).save(
                mouse_assets / blend_name,
                "WEBP",
            )
            slide["semantic_textures"] = {"5": f"assets/{mouse_id}/{semantic_name}"}
            slide["semantic_texture"] = f"assets/{mouse_id}/{semantic_name}"
            slide["blend_texture"] = f"assets/{mouse_id}/{blend_name}"
        slides.append(slide)
    return {
        "id": mouse_id,
        "provisional_order": False,
        "width": 240,
        "height": 160,
        "slides": slides,
        "semantic": (
            {
                "selected_k": 5,
                "cluster_counts": [5],
                "palette": ["#d73027", "#1a9850", "#4575b4", "#fee08b", "#984ea3"],
                "batch_correction": None,
                "k_selection": [],
                "review": {"approved": False, "fingerprint_matches": True},
                "qc": None,
                "links_url": None,
                "link_pair_count": 0,
            }
            if semantic
            else None
        ),
    }
