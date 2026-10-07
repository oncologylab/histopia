"""Specimen identity must survive standalone links and portal navigation."""

import json
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from histopia.visualization._review_portal import (
    _DECISIONS_HTML,
    _DECISIONS_JS,
    _WORKFLOW_HTML,
    _WORKFLOW_JS,
)
from histopia.visualization._review_selection import REVIEW_SELECTION_JS


@pytest.mark.browser
@pytest.mark.parametrize("prefix", ["", "/workspace/proxy/8765"])
def test_review_keeps_specimen_when_switching_views(
    tmp_path: Path, prefix: str
) -> None:
    playwright = pytest.importorskip("playwright.sync_api")
    site = tmp_path / prefix.lstrip("/") / "review"
    site.mkdir(parents=True)
    (site / "index.html").write_text(_WORKFLOW_HTML)
    (site / "workflow-review.js").write_text(_WORKFLOW_JS)
    (site / "workflow-review.css").write_text("")
    scope = {"sources": ["kpf"], "organs": ["pancreas"], "subjects": ["a", "b"]}
    tabs = [
        {"id": name, "label": name, "href": f"{name}.html", "scope": scope}
        for name in ("cells", "protein")
    ]
    tabs.append(
        {
            "id": "protein-atlas",
            "label": "Cellular protein atlas",
            "href": "protein-atlas.html",
            "scope": {**scope, "subjects": ["b"]},
        }
    )
    (site / "manifest-data.js").write_text(
        "globalThis.HISTOPIA_WORKFLOW_REVIEW=" + json.dumps({"tabs": tabs})
    )
    for name in ("cells", "protein", "protein-atlas"):
        (site / f"{name}.html").write_text(
            '<select id="cohort"><option>a</option><option>b</option></select>'
            '<p id="status"></p><script>'
            + REVIEW_SELECTION_JS
            + """
            const control=document.querySelector('#cohort');
            try {
              control.value=histopiaReviewSelection.requested([{id:'a'},{id:'b'}]);
              histopiaReviewSelection.remember(control.value);
            } catch(error) {
              document.querySelector('#status').textContent=error.message;
            }
            control.onchange=()=>histopiaReviewSelection.remember(control.value);
            </script>"""
        )
    handler = partial(SimpleHTTPRequestHandler, directory=str(tmp_path))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    errors = []
    try:
        with playwright.sync_playwright() as runtime:
            browser = runtime.chromium.launch(headless=True)
            page = browser.new_page()
            page.on("pageerror", lambda error: errors.append(str(error)))
            root = f"http://127.0.0.1:{server.server_port}{prefix}/review/"
            page.goto(root + "index.html?view=cells&subject=b&reconstruction=old")
            embedded = page.frame_locator("#review")
            assert embedded.locator("#cohort").input_value() == "b"
            embedded.locator("#cohort").select_option("a")
            page.wait_for_url(lambda url: "subject=a" in url)
            assert "reconstruction" not in parse_qs(urlsplit(page.url).query)
            page.locator('[data-tab="protein"]').click()
            assert embedded.locator("#cohort").input_value() == "a"
            page.reload(wait_until="networkidle")
            assert embedded.locator("#cohort").input_value() == "a"
            # Missing results must not make a tool disappear from navigation.
            tool = page.locator('[data-tab="protein-atlas"]')
            assert tool.is_visible()
            assert tool.get_attribute("data-available") == "false"
            tool.click()
            playwright.expect(embedded.locator("#cohort")).to_have_value("b")
            assert "subject=b" in page.url
            assert (
                "Showing specimen b" in page.locator("#selection-notice").inner_text()
            )
            assert embedded.locator("#result-subject").count() == 0
            assert embedded.get_by_role("button", name="Open result").count() == 0
            assert prefix + "/review/" in page.url
            # An excluded source cannot populate this chooser with our mice.
            page.goto(root + "index.html?view=protein-atlas&source=hpa&subject=a")
            assert tool.is_visible()
            assert embedded.get_by_text(
                "No result in this view for this source and organ."
            ).is_visible()
            assert embedded.locator("#result-subject").count() == 0
            assert "source=hpa" in page.url
            # Missing specimens open the remembered compatible image immediately.
            page.goto(root + "index.html?view=cells&subject=missing")
            playwright.expect(embedded.locator("#cohort")).to_have_value("a")
            assert "subject=a" in page.url
            page.reload()
            playwright.expect(embedded.locator("#cohort")).to_have_value("a")
            for key in ("subject", "mouse", "cohort"):
                page.goto(root + f"cells.html?{key}=b")
                assert page.locator("#cohort").input_value() == "b"
            page.goto(root + "cells.html?subject=missing")
            assert page.locator("#status").inner_text() == (
                "No result for the requested specimen."
            )
            assert "subject=missing" in page.url
            browser.close()
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
    assert not errors


@pytest.mark.browser
@pytest.mark.parametrize("prefix", ["", "/workspace/proxy/8765"])
def test_external_image_recovers_wrong_view_without_changing_dataset(
    tmp_path: Path, prefix: str
) -> None:
    import mimetypes
    from urllib.parse import urlencode

    from histopia.study._manifest import file_sha256
    from histopia.visualization._external_validation import external_validation_tab
    from histopia.visualization._results_catalog import (
        _write_catalog,
        catalog_navigation_context,
    )
    from histopia.visualization._review_portal import _WORKFLOW_CSS

    playwright = pytest.importorskip("playwright.sync_api")
    site = tmp_path / "review"
    site.mkdir()
    asset = tmp_path / "image.svg"
    asset.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20">'
        '<rect width="20" height="20" fill="brown"/></svg>'
    )
    _write_catalog(
        [
            dict(
                id="hpa", label="HPA", external=True, species="human", description="IHC"
            )
        ],
        [
            dict(
                id="counterstain-64",
                source_id="hpa",
                organ="pancreas",
                subject_id="64",
                title="ACTA2 · 64",
                stage="Stain extraction & UNI2-h",
                status="ready",
                summary="2D IHC",
                fingerprint="a" * 64,
                evidence_kind="image",
                media=[dict(path=str(asset), label="IHC", sha256=file_sha256(asset))],
            )
        ],
        site / "external-validation",
        updated_at="2026-09-18",
        benchmark_policy=dict(
            purpose="2d-protein-benchmark", reconstruction_allowed=False
        ),
    )
    catalog = json.loads((site / "external-validation/catalog.json").read_text())
    _write_catalog(
        [
            dict(
                id="kpf",
                label="Our data",
                external=False,
                species="mouse",
                description="IHC",
            )
        ],
        [
            dict(
                id="own-" + subject,
                source_id="kpf",
                organ="pancreas",
                subject_id=subject,
                title="Our pancreas · " + subject,
                stage="Images",
                status="ready",
                summary="Own images",
                fingerprint="b" * 64,
                evidence_kind="image",
                media=[dict(path=str(asset), label="IHC", sha256=file_sha256(asset))],
            )
            for subject in ["5996", "6180"]
        ],
        site / "data-catalog",
        updated_at="2026-09-18",
    )
    own = json.loads((site / "data-catalog/catalog.json").read_text())
    tissue_tab = dict(
        id="data-catalog",
        label="Tissue review",
        href="data-catalog/index.html",
        export_fingerprint=own["fingerprint"],
        **catalog_navigation_context(own),
    )
    scope = dict(sources=["kpf"], organs=["pancreas"], subjects=["5996", "6180"])
    tabs = [tissue_tab, external_validation_tab(catalog)] + [
        dict(id=key, label=key.title(), href="own.html", scope=scope)
        for key in ("registration", "protein")
    ]
    (site / "index.html").write_text(_WORKFLOW_HTML)
    (site / "workflow-review.js").write_text(_WORKFLOW_JS)
    (site / "workflow-review.css").write_text(_WORKFLOW_CSS)
    (site / "manifest-data.js").write_text(
        "globalThis.HISTOPIA_WORKFLOW_REVIEW=" + json.dumps(dict(tabs=tabs))
    )
    (site / "own.html").write_text('<h1 id="own">Own pancreas result</h1>')
    origin = "http://review.test"
    base = prefix + "/review/"
    selection = dict(
        view="registration",
        source="hpa",
        organ="pancreas",
        subject="64",
        collection="specimens",
        dataset="counterstain-64",
        field="0",
    )
    errors, escaped = [], []
    with playwright.sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1500, "height": 950})
        page.on("pageerror", lambda error: errors.append(str(error)))

        def respond(route):
            path = urlsplit(route.request.url).path
            if not path.startswith(base):
                escaped.append(path)
                route.fulfill(status=404)
                return
            file = site / path[len(base) :]
            route.fulfill(
                status=200 if file.is_file() else 404,
                content_type=mimetypes.guess_type(file)[0]
                or "application/octet-stream",
                body=file.read_bytes() if file.is_file() else b"",
            )

        page.route(origin + "/**", respond)
        root = origin + base + "index.html?"
        page.goto(root + urlencode(selection))
        frame = page.frame_locator("#review")
        playwright.expect(frame.locator("#image")).to_be_visible()
        playwright.expect(frame.locator("#title")).to_have_text("ACTA2 · 64")
        assert frame.locator("#image").evaluate("image => image.naturalWidth") == 20
        assert parse_qs(urlsplit(page.url).query) == {
            key: [value]
            for key, value in {**selection, "view": "external-validation"}.items()
        }
        assert (
            "No verified serial stack" in page.locator("#selection-notice").inner_text()
        )
        for tab in ("registration", "protein"):
            assert page.locator(f'[data-tab="{tab}"]').is_visible()
            page.locator(f'[data-tab="{tab}"]').click()
            playwright.expect(frame.locator("#own")).to_be_visible()
            assert f"view={tab}" in page.url
            assert "source=kpf" in page.url and "subject=5996" in page.url
            assert "counterstain-64" not in page.url
            page.locator('[data-tab="external-validation"]').click()
            playwright.expect(frame.locator("#image")).to_be_visible()
            assert "dataset=counterstain-64" in page.url
        page.go_back()
        playwright.expect(frame.locator("#own")).to_be_visible()
        assert "view=protein" in page.url
        page.go_forward()
        playwright.expect(frame.locator("#image")).to_be_visible()
        assert "view=external-validation" in page.url
        page.locator('[data-tab="data-catalog"]').click()
        playwright.expect(frame.locator("#title")).to_have_text("Our pancreas · 5996")
        frame.locator("#subject").select_option("6180")
        page.wait_for_url(lambda url: "subject=6180" in url)
        own_selection = parse_qs(urlsplit(page.url).query)
        page.locator('[data-tab="external-validation"]').click()
        playwright.expect(frame.locator("#title")).to_have_text("ACTA2 · 64")
        page.locator('[data-tab="data-catalog"]').click()
        playwright.expect(frame.locator("#title")).to_have_text("Our pancreas · 6180")
        assert parse_qs(urlsplit(page.url).query) == own_selection
        # A queued message from the prior catalog cannot overwrite this selection.
        page.evaluate(
            """fingerprint => window.dispatchEvent(new MessageEvent('message', {
          origin:location.origin,
          source:document.querySelector('#review').contentWindow,
          data:{type:'histopia-catalog-selection',catalog_fingerprint:fingerprint,
            source:'hpa',organ:'pancreas',subject:'64',dataset:'counterstain-64'}
        }))""",
            catalog["fingerprint"],
        )
        assert parse_qs(urlsplit(page.url).query) == own_selection
        # New job publications may refresh the embedded catalog while the portal
        # still has its prior manifest. Accept current evidence and remember it.
        refreshed = [
            {
                **row,
                "media": [
                    dict(path=str(asset), label="IHC", sha256=file_sha256(asset))
                ],
            }
            for row in own["datasets"]
        ]
        refreshed.append({**refreshed[-1], "id": "own-new", "fingerprint": "c" * 64})
        _write_catalog(
            own["sources"], refreshed, site / "data-catalog", updated_at="2026-09-19"
        )
        frame.locator("#compute > summary").click()
        frame.locator("#refresh").click()
        frame.locator('[data-dataset="own-new"]').click()
        page.wait_for_url(lambda url: "dataset=own-new" in url)
        page.locator('[data-tab="external-validation"]').click()
        playwright.expect(frame.locator("#title")).to_have_text("ACTA2 · 64")
        page.locator('[data-tab="data-catalog"]').click()
        playwright.expect(frame.locator("#title")).to_have_text("Our pancreas · 6180")
        assert "dataset=own-new" in page.url
        page.set_viewport_size({"width": 390, "height": 844})
        page.reload()
        playwright.expect(frame.locator("#image")).to_be_visible()
        for tab in ("external-validation", "registration", "data-catalog"):
            page.locator(f'[data-tab="{tab}"]').click()
            playwright.expect(page.locator(f'[data-tab="{tab}"]')).to_have_attribute(
                "aria-current", "page"
            )
            playwright.expect(
                frame.locator("#own" if tab == "registration" else "#image")
            ).to_be_visible()
        assert "subject=6180" in page.url
        assert not page.locator("html").evaluate("el => el.scrollWidth > innerWidth")
        assert not frame.locator("html").evaluate("el => el.scrollWidth > innerWidth")
        # A source label, unknown image, or former 3D link is never enough.
        for change in (
            dict(dataset="missing"),
            dict(organ="liver"),
            dict(subject="65"),
            dict(source="tcga"),
            dict(source="hpa-codex"),
            dict(reconstruction="old-hpa-stack"),
            dict(stack_mode="3d"),
        ):
            page.goto(root + urlencode({**selection, **change}))
            playwright.expect(
                frame.get_by_text("No result in this view for this source and organ.")
            ).to_be_visible()
            assert "view=registration" in page.url
            assert frame.locator("#image").count() == 0
            # An explicit sidebar click can always leave an unavailable bookmark.
            page.locator('[data-tab="data-catalog"]').click()
            playwright.expect(frame.locator("#title")).to_have_text(
                "Our pancreas · 6180"
            )
            assert "source=kpf" in page.url
        # Own-data prediction remains accessible with its own specimen identity.
        page.goto(root + "view=protein&source=kpf&organ=pancreas&subject=5996")
        playwright.expect(frame.locator("#own")).to_be_visible()
        assert "view=protein" in page.url
        assert "subject=5996" in page.url
        # Empty catalogs stay empty, but never trap navigation in another source.
        _write_catalog([], [], site / "data-catalog", updated_at="2026-09-18")
        empty = json.loads((site / "data-catalog/catalog.json").read_text())
        tissue_tab.update(
            export_fingerprint=empty["fingerprint"], **catalog_navigation_context(empty)
        )
        (site / "manifest-data.js").write_text(
            "globalThis.HISTOPIA_WORKFLOW_REVIEW=" + json.dumps(dict(tabs=tabs))
        )
        page.reload()
        page.locator('[data-tab="data-catalog"]').click()
        playwright.expect(frame.locator("#empty")).to_be_visible()
        assert "source=hpa" not in page.url
        page.locator('[data-tab="external-validation"]').click()
        playwright.expect(frame.locator("#title")).to_have_text("ACTA2 · 64")
        page.locator('[data-tab="registration"]').click()
        playwright.expect(frame.locator("#own")).to_be_visible()
        browser.close()
    assert not errors
    assert not escaped


@pytest.mark.browser
def test_decisions_resolves_specimen_without_changing_approvals() -> None:
    playwright = pytest.importorskip("playwright.sync_api")
    registry = {
        "stages": ["mask"],
        "cohorts": [
            {"id": name, "stages": {"mask": {"available": True, "approved": True}}}
            for name in ("a", "b")
        ],
    }
    writes = []
    errors = []
    with playwright.sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True)
        page = browser.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))

        def respond(route) -> None:
            path = urlsplit(route.request.url).path
            if route.request.method != "GET":
                writes.append(route.request.url)
            if path.endswith("index.html"):
                route.fulfill(body=_DECISIONS_HTML, content_type="text/html")
            elif path.endswith("review-decisions.js"):
                route.fulfill(body=_DECISIONS_JS, content_type="text/javascript")
            elif path.endswith("/api/reviews/access"):
                route.fulfill(
                    json={"review_configured": True, "authentication_required": False}
                )
            elif path.endswith("/api/reviews"):
                route.fulfill(json=registry)
            else:
                route.fulfill(body="", content_type="text/css")

        page.route("http://review.test/**", respond)
        root = "http://review.test/workspace/proxy/8765/review/decisions/index.html"
        for key in ("subject", "mouse", "cohort"):
            page.goto(root + f"?{key}=b")
            playwright.expect(page.locator("#cohort")).to_have_value("b")
            assert page.locator("#state").inner_text() == "Approved"
            assert page.locator("#approve").is_disabled()
        page.locator("#cohort").select_option("a")
        assert "subject=a" in page.url
        page.reload(wait_until="networkidle")
        assert page.locator("#cohort").input_value() == "a"
        page.goto(root + "?subject=missing")
        playwright.expect(page.locator("#message")).to_have_text(
            "No result for the requested specimen."
        )
        assert page.locator("#approve").is_disabled()
        assert page.locator("#stages button").count() == 0
        browser.close()
    assert not errors
    assert not writes
