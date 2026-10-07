"""The analysis shell changes navigation, never scientific identity or approval."""

import json
import re
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from histopia.study._manifest import fingerprint
from histopia.visualization._analysis_review import (
    STAGES,
    build_analysis_review,
    compile_review_navigation,
)


def catalog(rows, *, external=False, gate="eligibility_policy"):
    result = {
        "sources": [
            {
                "id": "hpa" if external else "own",
                "label": "HPA" if external else "Our mice",
                "external": external,
            }
        ],
        "datasets": rows,
        gate: {"fingerprint": "independent-publication-policy"},
    }
    result["fingerprint"] = fingerprint(result)
    return result


def row(identity, stage, subject="a", *, organ="pancreas", source="own"):
    return dict(
        id=identity,
        stage=stage,
        source_id=source,
        organ=organ,
        subject_id=subject,
        title=identity,
        fingerprint="result-" + identity,
        reconstruction_id="stack-" + subject,
        eligibility_fingerprint="eligible-" + subject,
        media=[{"label": "Evidence", "href": "images/" + identity + ".svg"}],
        downloads=[{"label": "Data", "href": "downloads/" + identity + ".csv"}],
    )


def test_every_published_stage_has_one_canonical_identity():
    rows = [row(str(i), stage) for i, stage in enumerate(STAGES)]
    result = compile_review_navigation({}, {"data-catalog": catalog(rows)})
    assert len(result["entries"]) == len(rows)
    assert result["coverage"]["mapped_records"] == len(rows)
    assert {r["id"]: r["fingerprint"] for r in rows} == {
        r["id"]: r["fingerprint"] for r in result["mappings"]
    }
    legacy = next(
        e for e in result["entries"] if e["stage"] == "Legacy protein diagnostics"
    )
    assert legacy["previous"] and legacy["analysis"] == "protein"


@pytest.mark.parametrize(
    "change", ["unbound", "changed", "unmapped", "unresolved", "external-input"]
)
def test_navigation_fails_closed(change):
    content = catalog([row("x", "Reconstruction")])
    if change == "unbound":
        del content["eligibility_policy"]
    elif change == "unmapped":
        content["datasets"][0]["stage"] = "Undeclared analysis"
    elif change == "unresolved":
        del content["datasets"][0]["reconstruction_id"]
    elif change == "external-input":
        content["input_review_policy"] = content.pop("eligibility_policy")
        content["sources"][0]["external"] = True
    if change != "changed":
        content["fingerprint"] = fingerprint(
            {k: v for k, v in content.items() if k != "fingerprint"}
        )
    else:
        content["datasets"][0]["title"] = "Unbound change"
    with pytest.raises(ValueError):
        compile_review_navigation({}, {"published": content})


def test_benchmark_stays_2d_and_preserves_source_and_performance():
    benchmark = row(
        "hpa-prediction", "Protein prediction · test", source="hpa", organ="brain"
    )
    benchmark.pop("reconstruction_id")
    benchmark.pop("eligibility_fingerprint")
    benchmark["media"].append({"label": "All markers holdout", "href": "holdout.png"})
    result = compile_review_navigation(
        {},
        {
            "external-validation": catalog(
                [benchmark], external=True, gate="benchmark_policy"
            )
        },
    )
    entry = result["entries"][0]
    assert entry["scope_label"] == "2D counterstain benchmark"
    assert entry["analysis"] == "protein"
    assert "reconstruction_id" not in entry
    assert entry["mode_fields"]["performance"] == 1
    assert entry["media"][0]["href"] == "external-validation/images/hpa-prediction.svg"


def test_native_3d_is_gated_and_cell_section_scopes_stay_separate():
    scope = {"sources": ["own"], "organs": ["pancreas"], "subjects": ["a", "b"]}
    workflow = {
        "tabs": [
            {"id": tool, "href": tool + "/index.html", "scope": scope}
            for tool in ("atlas", "topology", "cells", "selected-cells")
        ]
    }
    manifests = {
        "cells": {"cohorts": [{"id": "a", "sections": [{"id": "001"}, {"id": "002"}]}]},
        "selected-cells": {"cohorts": [{"id": "a", "sections": [{"id": "002"}]}]},
    }
    result = compile_review_navigation(
        workflow, {"data-catalog": catalog([row("stack", "Reconstruction")])}, manifests
    )
    entries = {e["id"]: e for e in result["entries"]}
    assert "viewer-topology-b" not in entries
    assert "viewer-atlas-b" not in entries
    assert entries["viewer-atlas-a"]["mode"] == "atlas"
    assert entries["viewer-atlas-a"]["reconstruction_id"] == "stack-a"
    registration = next(w for w in result["workspaces"] if w["id"] == "registration")
    assert ["atlas", "Semantic stack"] in registration["modes"]
    assert len(entries["viewer-cells-a"]["sections"]) == 2
    assert len(entries["viewer-selected-cells-a"]["sections"]) == 1
    assert entries["viewer-cells-a"]["scientific_approval"] is False


def write_site(root: Path, *, semantic_stack=False, display_viewer=None):
    root.mkdir(parents=True)
    scope = {"sources": ["own"], "organs": ["pancreas"], "subjects": ["a", "b"]}
    tabs = [
        {"id": "cells", "href": "cells/index.html", "scope": scope},
        {
            "id": "protein",
            "href": "protein/index.html",
            "scope": {**scope, "subjects": ["b"]},
        },
    ]
    if semantic_stack:
        tabs.append({"id": "atlas", "href": "atlas/index.html", "scope": scope})
    if display_viewer:
        tabs.append(
            {
                "id": display_viewer,
                "href": display_viewer + "/index.html",
                "scope": scope,
            }
        )
    (root / "manifest.json").write_text(json.dumps({"tabs": tabs}))
    rows = [
        row("regions-a", "Tissue-region proteins"),
        row("prediction-b", "Expanded protein prediction", "b"),
    ]
    if semantic_stack:
        rows.append(row("stack-a", "Reconstruction"))
    content = catalog(rows)
    external = catalog(
        [row("public-b", "Protein prediction · test", "donor", source="hpa")],
        external=True,
        gate="benchmark_policy",
    )
    for name, c in [("data-catalog", content), ("external-validation", external)]:
        folder = root / name
        folder.mkdir()
        (folder / "catalog.json").write_text(json.dumps(c))
        (folder / "images").mkdir()
        for e in c["datasets"]:
            (folder / e["media"][0]["href"]).write_text(
                '<svg xmlns="http://www.w3.org/2000/svg" width="200" height="100">'
                '<rect width="200" height="100" fill="purple"/></svg>'
            )
    for name in ("cells", "protein"):
        (root / name).mkdir()
        (
            root / name / "index.html"
        ).write_text("""<!doctype html><html><head></head><body>
<header><label>Mouse<select id="cohort">
<option>a</option><option>b</option></select></label>
<select id="section"><option>001</option><option>002</option></select></header>
<main><canvas width="200" height="100"></canvas><aside>Review form</aside></main>
<script>document.querySelector('#cohort').value=
new URL(location.href).searchParams.get('subject');</script>
</body></html>""")
    if semantic_stack:
        (root / "atlas").mkdir()
        (root / "atlas/index.html").write_text("""<!doctype html><html><head>
<style>main{display:grid;grid-template-columns:300px 1fr}</style></head><body>
<main><aside><h1>Histopia</h1><label>Mouse<select id="mouse">
<option>a</option><option>b</option></select></label>
<div id="mode"><button>Histology</button><button>Semantic</button></div>
<label>Adjacent pair<select id="link-pair"><option>001 to 002</option></select></label>
<label><input id="show-links" type="checkbox" checked>Show topology links</label>
</aside><section id="viewport"><canvas width="200" height="100"></canvas></section>
</main>
<script>document.querySelector('#mouse').value=
new URL(location.href).searchParams.get('mouse');</script></body></html>""")
    if display_viewer:
        from histopia.visualization import _cellular_protein_atlas, _topology_review

        native = (
            _topology_review
            if display_viewer == "topology"
            else _cellular_protein_atlas
        )
        styles = native._CSS if display_viewer == "topology" else native._CSS_V2
        # Exercise the actual native panel markup and responsive layout with
        # tiny controls, without loading a GPU scene or scientific review API.
        page = re.sub(r"<script\b[^>]*>.*?</script>", "", native._HTML, flags=re.S)
        page = re.sub(r"<link\b[^>]*>", "", page)
        page = page.replace("</head>", "<style>" + styles + "</style></head>")
        page = page.replace(
            "</body>",
            """<script>
document.querySelector('#cohort').innerHTML='<option>a</option>';
const region=document.querySelector('#region');
if(region)region.innerHTML='<option value="all">All classes</option>'+
  '<option value="1">Class 1</option><option value="2">Class 2</option>';
const targets=document.querySelector('#target-list');
if(targets)targets.innerHTML='<label><input type="checkbox" value="yap" checked>YAP'+
  '</label><label><input type="checkbox" value="ck19">CK19</label>';
</script></body>""",
        )
        folder = root / display_viewer
        folder.mkdir()
        (folder / "index.html").write_text(page)
    return build_analysis_review(root, root)


def test_builder_emits_portable_fallback_and_coverage(tmp_path):
    index = write_site(tmp_path / "review")
    assert "subject=b&amp;mouse=b&amp;cohort=b" in index.read_text()
    coverage = json.loads((index.parent / "navigation-coverage.json").read_text())
    assert coverage["mapped_records"] == 3
    assert str(tmp_path) not in index.read_text()
    assert (index.parent / "workspace-data.js").is_file()


@pytest.mark.browser
@pytest.mark.parametrize("prefix", ["", "/workspace/proxy/8765"])
def test_analysis_navigation_images_state_and_empty_links(tmp_path, prefix):
    pw = pytest.importorskip("playwright.sync_api")
    site = tmp_path / prefix.lstrip("/") / "review"
    write_site(site)
    handler = partial(SimpleHTTPRequestHandler, directory=str(tmp_path))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    errors = []
    writes = []
    try:
        with pw.sync_playwright() as runtime:
            browser = runtime.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1366, "height": 768})
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.on(
                "request", lambda r: writes.append(r.url) if r.method != "GET" else None
            )
            root = f"http://127.0.0.1:{server.server_port}{prefix}/review/index.html"
            page.goto(
                root + "?view=cells&source=own&organ=pancreas&subject=a&section=002"
            )
            pw.expect(page.locator("#section")).to_have_value("002")
            assert not page.frame_locator("#review").locator("aside").is_visible()
            page.locator("#review-toggle").click()
            assert page.frame_locator("#review").locator("aside").is_visible()
            page.locator('[data-view="protein"]').click()
            pw.expect(page.locator("#image")).to_be_visible()
            pw.expect(page.locator("#image-state")).to_be_hidden()
            assert parse_qs(urlsplit(page.url).query)["subject"] == ["b"]
            assert "Showing" in page.locator("#selection-notice").inner_text()
            page.locator("#source").select_option("hpa")
            pw.expect(page.locator("#scope-label")).to_have_text(
                "2D counterstain benchmark"
            )
            page.reload()
            pw.expect(page.locator("#source")).to_have_value("hpa")
            page.locator('[data-view="cells"]').click()
            pw.expect(page.locator("#source")).to_have_value("own")
            assert "Showing" in page.locator("#selection-notice").inner_text()
            page.go_back()
            pw.expect(page.locator("#source")).to_have_value("hpa")
            page.goto(root + "?view=registration&source=hpa&dataset=public-b")
            pw.expect(page.locator("#unavailable")).to_be_visible()
            assert "dataset=public-b" in page.url
            page.locator('[data-view="spatial"]').click()
            pw.expect(page.locator("#image")).to_be_visible()
            page.set_viewport_size({"width": 390, "height": 844})
            pw.expect(page.locator("#mobile-view")).to_be_visible()
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            page.locator("#details-button").click()
            pw.expect(page.locator("#detail-dialog")).to_be_visible()
            page.locator("#close-dialog").click()
            page.locator("#mobile-view").select_option("cells")
            pw.expect(page.locator("#section")).to_be_visible()
            assert not writes and not errors
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.browser
@pytest.mark.parametrize("prefix", ["", "/workspace/proxy/8765"])
def test_semantic_stack_is_directly_accessible_with_controls(tmp_path, prefix):
    pw = pytest.importorskip("playwright.sync_api")
    site = tmp_path / prefix.lstrip("/") / "review"
    write_site(site, semantic_stack=True)
    handler = partial(SimpleHTTPRequestHandler, directory=str(tmp_path))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    errors, writes = [], []
    try:
        with pw.sync_playwright() as runtime:
            browser = runtime.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1366, "height": 768})
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.on(
                "request", lambda r: writes.append(r.url) if r.method != "GET" else None
            )
            root = f"http://127.0.0.1:{server.server_port}{prefix}/review/index.html"
            scope = "source=own&organ=pancreas&subject=a"
            page.goto(root + "?view=registration&mode=stacks&dataset=stack-a&" + scope)
            page.get_by_role("button", name="Semantic stack", exact=True).click()
            native = page.frame_locator("#review")
            pw.expect(native.locator("#link-pair")).to_be_visible()
            pw.expect(native.locator("#show-links")).to_be_visible()
            pw.expect(native.locator("#mode")).to_be_visible()
            pw.expect(native.locator("#mouse")).to_have_value("a")
            query = parse_qs(urlsplit(page.url).query)
            assert query["dataset"] == ["viewer-atlas-a"]
            assert query["reconstruction"] == ["stack-a"]
            assert query["subject"] == ["a"]
            assert query["mode"] == ["atlas"]
            page.reload()
            pw.expect(native.locator("#link-pair")).to_be_visible()
            page.get_by_role("button", name="3D stacks", exact=True).click()
            pw.expect(page.locator("#image")).to_be_visible()
            assert parse_qs(urlsplit(page.url).query)["dataset"] == ["stack-a"]
            for bookmark in (
                "view=atlas",
                "view=registration&mode=stacks&dataset=viewer-atlas-a",
            ):
                page.goto(root + "?" + bookmark + "&" + scope)
                pw.expect(native.locator("#link-pair")).to_be_visible()
                assert parse_qs(urlsplit(page.url).query)["mode"] == ["atlas"]
            page.set_viewport_size({"width": 390, "height": 844})
            pw.expect(page.locator("#mobile-view")).to_be_visible()
            pw.expect(native.locator("#show-links")).to_be_visible()
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            assert not writes and not errors
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.browser
@pytest.mark.parametrize("prefix", ["", "/workspace/proxy/8765"])
@pytest.mark.parametrize("renderer", ["topology", "protein-atlas"])
def test_native_display_controls_stay_accessible(tmp_path, prefix, renderer):
    pw = pytest.importorskip("playwright.sync_api")
    site = tmp_path / prefix.lstrip("/") / "review"
    write_site(site, display_viewer=renderer)
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory=str(tmp_path))
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    errors, writes = [], []
    try:
        with pw.sync_playwright() as runtime:
            browser = runtime.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.on(
                "request", lambda r: writes.append(r.url) if r.method != "GET" else None
            )
            root = f"http://127.0.0.1:{server.server_port}{prefix}/review/index.html"
            view, mode = (
                ("registration", "volume")
                if renderer == "topology"
                else ("protein", "atlas")
            )
            page.goto(
                root + f"?view={view}&mode={mode}&dataset=viewer-{renderer}-a"
                "&source=own&organ=pancreas&subject=a"
            )
            native = page.frame_locator("#review")
            controls = page.locator("#controls-toggle")
            pw.expect(native.locator("aside")).to_be_visible()
            pw.expect(controls).to_have_attribute("aria-pressed", "true")
            if renderer == "topology":
                pw.expect(native.locator(".review")).to_be_hidden()
                native.locator("#region").select_option("2")
                pw.expect(native.locator("#region")).to_have_value("2")
                native.locator("#region").select_option("all")
            else:
                native.locator('#target-list input[value="ck19"]').check()
                native.locator('#target-list input[value="yap"]').uncheck()
                pw.expect(native.locator("#target-list input:checked")).to_have_value(
                    "ck19"
                )
            controls.click()
            pw.expect(native.locator("aside")).to_be_hidden()
            # The shell polls asynchronously; explicit hiding must persist.
            page.wait_for_timeout(800)
            pw.expect(native.locator("aside")).to_be_hidden()
            controls.click()
            pw.expect(native.locator("aside")).to_be_visible()
            page.set_viewport_size({"width": 390, "height": 844})
            page.reload()
            pw.expect(controls).to_be_visible()
            pw.expect(controls).to_have_attribute("aria-pressed", "false")
            pw.expect(native.locator("aside")).to_be_hidden()
            controls.click()
            pw.expect(native.locator("aside")).to_be_visible()
            controls.click()
            pw.expect(native.locator("aside")).to_be_hidden()
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            assert not errors and not writes
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.browser
def test_empty_publication_has_no_stale_result(tmp_path):
    pw = pytest.importorskip("playwright.sync_api")
    (tmp_path / "manifest.json").write_text('{"tabs": []}')
    index = build_analysis_review(tmp_path, tmp_path)
    with pw.sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(index.as_uri())
        pw.expect(page.locator("#unavailable")).to_be_visible()
        assert page.locator("#alternatives button").count() == 0
        page.locator('[data-view="protein"]').click()
        pw.expect(page.locator("#unavailable")).to_be_visible()
        browser.close()
