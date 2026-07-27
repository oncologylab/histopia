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
        _browser_mouse(
            assets,
            "second",
            3,
            (30, 120, 180),
            semantic=True,
            stain=True,
        ),
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
            page.locator("#mode button[data-mode='stain-overlay']").click()
            page.wait_for_function(
                """() => document.querySelector('#viewport')
                    .getAttribute('aria-busy') === 'false'
                  && document.querySelector(
                    "#mode button[data-mode='stain-overlay']"
                  ).classList.contains('active')"""
            )
            assert page.locator("#stain-controls").is_visible()
            canvas_box = page.locator("canvas").bounding_box()
            assert canvas_box is not None
            page.locator("canvas").click(
                position={
                    "x": canvas_box["width"] / 2,
                    "y": canvas_box["height"] / 2,
                }
            )
            page.wait_for_function(
                "() => document.querySelectorAll('#stain-probe .probe-row').length > 0"
            )
            assert "relative OD" in page.locator("#qc").inner_text()
            assert "OD" in page.locator("#stain-probe").inner_text()
            page.locator("#stain-variant").select_option("raw")
            page.wait_for_function(
                """() => document.querySelector('#viewport')
                    .getAttribute('aria-busy') === 'false'"""
            )
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
    stain: bool = False,
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
        if stain:
            stain_image = image.copy()
            stain_image[30:130, 45:195, :3] = (225, 179, 55)
            overlay_image = (
                image.astype(np.uint16) + stain_image.astype(np.uint16)
            ) // 2
            textures = {}
            overlays = {}
            for variant, offset in (("raw", 0), ("corrected", 15)):
                stain_name = f"{index + 1:03d}-stain-{variant}.webp"
                overlay_name = f"{index + 1:03d}-stain-{variant}-overlay.webp"
                variant_image = stain_image.copy()
                variant_image[..., :3] = np.clip(
                    variant_image[..., :3].astype(np.int16) + offset,
                    0,
                    255,
                ).astype(np.uint8)
                Image.fromarray(variant_image).save(
                    mouse_assets / stain_name,
                    "WEBP",
                )
                Image.fromarray(overlay_image.astype(np.uint8)).save(
                    mouse_assets / overlay_name,
                    "WEBP",
                )
                textures[variant] = f"assets/{mouse_id}/{stain_name}"
                overlays[variant] = f"assets/{mouse_id}/{overlay_name}"
            probe_width, probe_height = 40, 30
            raw = np.full(
                (probe_height, probe_width),
                700 + index * 100,
                dtype="<u2",
            )
            corrected = np.full(
                (probe_height, probe_width),
                550 + index * 100,
                dtype="<u2",
            )
            probe_name = f"{index + 1:03d}-stain-probe.bin"
            np.stack([raw, corrected]).tofile(mouse_assets / probe_name)
            slide["stain"] = {
                "quantified": True,
                "marker": f"Marker {index + 1}",
                "family": "h-dab",
                "textures": textures,
                "overlay_textures": overlays,
                "probe": f"assets/{mouse_id}/{probe_name}",
                "probe_width": probe_width,
                "probe_height": probe_height,
                "probe_scale_od": 0.001,
                "probe_nodata": 65535,
                "positive_threshold_od": 0.4,
                "qc": {"correction_accepted": True},
            }
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
        "stain": (
            {
                "display_max_od": 1.0,
                "palette": ["#f6f7f4", "#27807e", "#eebe46", "#b53130"],
                "review": {
                    "approved": False,
                    "fingerprint_matches": True,
                },
                "quantified_slides": slide_count,
            }
            if stain
            else None
        ),
    }
