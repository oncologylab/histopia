"""Internal input QC must retain organ/scan identity without a 3D bypass."""

import json
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from histopia.study._manifest import file_sha256, fingerprint
from histopia.visualization._internal_review import (
    build_internal_review,
    internal_review_tab,
)
from histopia.visualization._review_portal import _WORKFLOW_HTML, _WORKFLOW_JS


def seal(manifest, rows):
    manifest["records"] = {r["id"]: fingerprint(r) for r in rows}
    manifest["fingerprint"] = fingerprint(
        {k: v for k, v in manifest.items() if k != "fingerprint"}
    )
    return manifest


def inputs(tmp_path):
    image = tmp_path / "field.svg"
    image.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="80" height="60"/>')
    scans, rows = [], []
    for organ, mouse in [("liver", "a"), ("lung", "b"), ("kidney", "c")]:
        scan_id = f"{organ}-{mouse}-scan-1"
        scans.append(
            dict(
                id=scan_id,
                source_id="ours",
                organ=organ,
                subject_id=mouse,
                source_sha256="a" * 64,
            )
        )
        rows.append(
            dict(
                id=f"cells-{scan_id}",
                source_id="ours",
                organ=organ,
                subject_id=mouse,
                title=f"{organ} cells",
                stage="Cells",
                status="ready",
                summary="Provisional cells.",
                fingerprint="b" * 64,
                scientific_approval=False,
                evidence_kind="image",
                acquisition_ids=[scan_id],
                input_scan_id=scan_id,
                input_scan_label="Scan 1 · IHC",
                media=[
                    dict(
                        path=str(image), sha256=file_sha256(image), label="Native cells"
                    )
                ],
            )
        )
    manifest = dict(
        schema_version="internal-input-review-1",
        sources=[
            dict(
                id="ours",
                label="Our data",
                external=False,
                species="mouse",
                description="Internal QC",
            )
        ],
        acquisitions=scans,
    )
    return seal(manifest, rows), rows


def test_internal_export_retains_scope_and_withdraws_stale_assets(tmp_path):
    manifest, rows = inputs(tmp_path)
    out = tmp_path / "out"
    build_internal_review(manifest, rows, out, updated_at="2026-09-18")
    catalog = json.loads((out / "catalog.json").read_text())
    assert catalog["input_review_policy"]["reconstruction_allowed"] is False
    assert "eligibility_policy" not in catalog and "benchmark_policy" not in catalog
    assert catalog["organs"] == ["liver", "lung", "kidney"]
    assert all(not r["scientific_approval"] for r in catalog["datasets"])
    tab = internal_review_tab(catalog)
    assert tab["catalog_contexts"][rows[0]["id"]] == {
        "source_id": "ours",
        "organ": "liver",
        "subject_id": "a",
    }
    assert "Internal tissues" in (out / "index.html").read_text()
    assert str(tmp_path) not in (out / "catalog.json").read_text()
    # A new empty revision withdraws thumbnails/downloads and the HTML fallback.
    seal(manifest, [])
    build_internal_review(manifest, [], out, updated_at="2026-09-19")
    assert not list((out / "assets").iterdir())
    assert "No internal image results" in (out / "index.html").read_text()


@pytest.mark.parametrize(
    "failure",
    [
        "public",
        "identity",
        "missing-scan",
        "duplicate-scan",
        "reconstruction",
        "approval",
        "incomplete",
        "unbound",
        "stale-manifest",
        "missing-result",
        "changed-image",
    ],
)
def test_internal_export_rejects_unbound_or_wrong_scope(tmp_path, failure):
    manifest, rows = inputs(tmp_path)
    if failure == "public":
        manifest["sources"][0]["external"] = True
    elif failure == "identity":
        rows[0]["organ"] = "lung"  # Same mouse alone cannot establish organ identity.
    elif failure == "missing-scan":
        rows[0]["acquisition_ids"] = ["unknown"]
    elif failure == "duplicate-scan":
        rows[0]["acquisition_ids"] *= 2
    elif failure == "reconstruction":
        rows[0]["reconstruction_id"] = "invented"
    elif failure == "approval":
        rows[0]["scientific_approval"] = True
    elif failure == "incomplete":
        rows[0]["status"] = "running"
    seal(manifest, rows)
    if failure == "unbound":
        rows[0]["title"] = "Changed evidence"
    elif failure == "stale-manifest":
        manifest["fingerprint"] = "0" * 64
    elif failure == "missing-result":
        rows.pop()
    elif failure == "changed-image":
        Path(rows[0]["media"][0]["path"]).write_text("changed")
    with pytest.raises(ValueError):
        build_internal_review(manifest, rows, tmp_path / "out", updated_at="2026-09-18")


@pytest.mark.browser
@pytest.mark.parametrize("prefix", ["", "/workspace/proxy/8765"])
def test_internal_navigation_does_not_confuse_organs_or_trap_user(tmp_path, prefix):
    playwright = pytest.importorskip("playwright.sync_api")
    manifest, rows = inputs(tmp_path)
    site = tmp_path / prefix.lstrip("/") / "review"
    output = site / "internal-inputs"
    build_internal_review(manifest, rows, output, updated_at="2026-09-18")
    catalog = json.loads((output / "catalog.json").read_text())
    tabs = [
        internal_review_tab(catalog),
        dict(
            id="cells",
            label="Cells",
            href="cells.html",
            scope=dict(sources=["ours"], organs=["pancreas"], subjects=["a", "b"]),
        ),
    ]
    (site / "index.html").write_text(_WORKFLOW_HTML)
    (site / "workflow-review.js").write_text(_WORKFLOW_JS)
    (site / "workflow-review.css").write_text("")
    (site / "manifest-data.js").write_text(
        "globalThis.HISTOPIA_WORKFLOW_REVIEW=" + json.dumps(dict(tabs=tabs))
    )
    (site / "cells.html").write_text('<p id="pancreas">Pancreas cell image</p>')
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory=str(tmp_path))
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    errors = []
    try:
        with playwright.sync_playwright() as runtime:
            browser = runtime.chromium.launch()
            page = browser.new_page()
            page.on("pageerror", lambda error: errors.append(str(error)))
            root = f"http://127.0.0.1:{server.server_port}{prefix}/review/"
            page.goto(
                root + "index.html?view=cells&source=ours&organ=pancreas&subject=a"
            )
            page.locator('[data-tab="internal-inputs"]').click()
            frame = page.frame_locator("#review")
            playwright.expect(frame.locator("#identity")).to_contain_text("Liver / a")
            assert "organ=liver" in page.url and "subject=a" in page.url
            for organ, mouse in [("lung", "b"), ("kidney", "c"), ("liver", "a")]:
                frame.locator(f'[data-organ="{organ}"]').click()
                playwright.expect(frame.locator("#identity")).to_contain_text(
                    f"{organ.title()} / {mouse}"
                )
                page.wait_for_url(
                    lambda url, organ=organ, mouse=mouse: (
                        f"organ={organ}" in url and f"subject={mouse}" in url
                    )
                )
                assert page.locator("#selection-notice").is_hidden()
                assert frame.locator("#image").evaluate(
                    "(im)=>im.complete&&im.naturalWidth>0"
                )
                assert frame.locator("#status").inner_text() == "Provisional · 2D"
                frame.locator("#section").select_option(f"{organ}-{mouse}-scan-1")
                page.wait_for_url(
                    lambda url, organ=organ, mouse=mouse: (
                        f"section={organ}-{mouse}-scan-1" in url
                    )
                )
                page.reload(wait_until="networkidle")
                playwright.expect(frame.locator("#section")).to_have_value(
                    f"{organ}-{mouse}-scan-1"
                )
            page.locator('[data-tab="cells"]').click()
            playwright.expect(frame.locator("#pancreas")).to_be_visible()
            assert "organ=pancreas" in page.url and "subject=a" in page.url
            page.locator('[data-tab="internal-inputs"]').click()
            playwright.expect(frame.locator("#identity")).to_contain_text("Liver / a")
            assert "section=liver-a-scan-1" in page.url
            page.set_viewport_size(dict(width=390, height=844))
            page.reload(wait_until="networkidle")
            assert frame.locator("#image").evaluate(
                "(im)=>im.complete&&im.naturalWidth>0"
            )
            assert frame.locator("body").evaluate("(el)=>el.scrollWidth<=innerWidth")
            # Unknown direct links stay unavailable, but the user can leave.
            page.goto(
                root + "index.html?view=internal-inputs&source=ours"
                "&organ=liver&dataset=unknown"
            )
            playwright.expect(frame.locator("#empty")).to_contain_text(
                "unavailable in Internal tissues"
            )
            page.locator('[data-tab="cells"]').click()
            playwright.expect(frame.locator("#pancreas")).to_be_visible()
            browser.close()
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
    assert not errors
