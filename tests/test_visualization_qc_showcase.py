from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest
from PIL import Image

from histopia.visualization import (
    create_viewer_server,
    export_registration_qc_showcase,
)
from histopia.visualization._viewer import (
    _INDEX_HTML,
    _STYLES_CSS,
    _VIEWER_JS,
    THREE_VERSION,
)


def _write_source(root: Path) -> None:
    for name in ("index.html", "viewer.js", "styles.css"):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(f"{name}\n")
    mice = []
    for mouse_id in ("4435", "4943", "unused"):
        asset = root / "assets" / mouse_id / "section.webp"
        asset.parent.mkdir(parents=True)
        asset.write_bytes(mouse_id.encode())
        mice.append(
            {
                "id": mouse_id,
                "width": 240,
                "height": 160,
                "slides": [
                    {
                        "id": f"{mouse_id}.ndpi",
                        "label": "HE",
                        "order": 1,
                        "texture": f"assets/{mouse_id}/section.webp",
                        "reference": True,
                        "semantic_texture": "private-semantic.webp",
                    }
                ],
                "semantic": {"fingerprint": "private"},
            }
        )
        mask = root / f"{mouse_id}-mask-review"
        mask.mkdir()
        (mask / "index.html").write_text(f"mask {mouse_id}\n")
        (mask / "mask.jpg").write_bytes(b"mask")
        order = root / f"{mouse_id}-order-review"
        order.mkdir()
        (order / "index.html").write_text(f"order {mouse_id}\n")
        (order / "manifest.json").write_text(json.dumps({"approved": True}))
    (root / "manifest.json").write_text(json.dumps({"schema_version": 1, "mice": mice}))


def test_export_registration_qc_showcase_selects_sanitized_mice(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "qc"
    _write_source(source)

    index = export_registration_qc_showcase(source, output, ("4943", "4435"))

    assert index == output / "index.html"
    portal = json.loads((output / "qc-manifest.json").read_text())
    assert [mouse["id"] for mouse in portal["mice"]] == ["4943", "4435"]
    registration = json.loads((output / "registration" / "manifest.json").read_text())
    assert [mouse["id"] for mouse in registration["mice"]] == ["4943", "4435"]
    assert registration["mice"][0]["semantic"] is None
    assert "semantic_texture" not in registration["mice"][0]["slides"][0]
    assert (
        output / "registration" / "assets" / "4943" / "section.webp"
    ).read_bytes() == b"4943"
    assert not (output / "registration" / "assets" / "unused").exists()
    assert (output / "reviews" / "4435" / "mask" / "mask.jpg").exists()
    assert (output / ".nojekyll").exists()


def test_export_registration_qc_showcase_rejects_missing_review(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _write_source(source)
    missing = source / "4435-order-review"
    for path in missing.iterdir():
        path.unlink()
    missing.rmdir()

    with pytest.raises(FileNotFoundError, match="order review"):
        export_registration_qc_showcase(source, tmp_path / "qc", "4435")


def test_export_registration_qc_showcase_supports_provisional_reviews(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "qc"
    _write_source(source)
    for kind in ("mask", "order"):
        review = source / f"provisional-{kind}-review"
        review.mkdir()
        (review / "index.html").write_text(f"{kind} provisional\n")

    export_registration_qc_showcase(source, output, "provisional")

    portal = json.loads((output / "qc-manifest.json").read_text())
    assert portal["mice"] == [
        {
            "id": "provisional",
            "stages": {
                "mask": "reviews/provisional/mask/",
                "order": "reviews/provisional/order/",
            },
        }
    ]
    assert not (output / "registration").exists()
    assert "button.disabled" in (output / "portal.js").read_text()


def test_export_registration_qc_showcase_rejects_nonempty_output(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "qc"
    _write_source(source)
    output.mkdir()
    (output / "keep").write_text("keep")

    with pytest.raises(FileExistsError, match="not empty"):
        export_registration_qc_showcase(source, output, "4435")


@pytest.mark.browser
def test_registration_qc_portal_switches_embedded_mouse(tmp_path: Path) -> None:
    playwright = pytest.importorskip("playwright.sync_api")
    source = tmp_path / "source"
    output = tmp_path / "site" / "histopia" / "qc"
    _write_source(source)
    (source / "index.html").write_text(_INDEX_HTML.replace("__THREE__", THREE_VERSION))
    (source / "viewer.js").write_text(_VIEWER_JS.replace("__THREE__", THREE_VERSION))
    (source / "styles.css").write_text(_STYLES_CSS)
    for mouse_id in ("4435", "4943", "unused"):
        Image.new("RGBA", (24, 16), (80, 120, 160, 255)).save(
            source / "assets" / mouse_id / "section.webp",
            "WEBP",
        )
    export_registration_qc_showcase(source, output, ("4435", "4943"))
    (output.parent / "index.html").write_text("stable viewer")

    server = create_viewer_server(tmp_path / "site", bind="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    errors: list[str] = []
    try:
        with playwright.sync_playwright() as runtime:
            browser = runtime.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 390, "height": 844})
            page.on(
                "console",
                lambda message: (
                    errors.append(message.text) if message.type == "error" else None
                ),
            )
            page.on("requestfailed", lambda request: errors.append(request.url))
            page.goto(
                f"http://127.0.0.1:{server.server_port}/histopia/qc/",
                wait_until="networkidle",
            )
            page.select_option("#mouse", "4943")
            page.locator('button[data-stage="registration"]').click()
            page.wait_for_function(
                """id => document.querySelector('#review')
                  .contentDocument.querySelector('#mouse')?.value === id""",
                arg="4943",
            )
            dimensions = page.evaluate(
                """() => ({
                  x: document.documentElement.scrollWidth > innerWidth,
                  y: document.documentElement.scrollHeight > innerHeight,
                })"""
            )
            assert not dimensions["x"]
            assert not dimensions["y"]
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert errors == []
