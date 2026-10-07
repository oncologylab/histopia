"""Inventory visibility must not imply result availability or eligibility."""

import csv
import io
import json
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import pytest

from histopia.study._curated_metadata import reconcile_curated_metadata
from histopia.study._manifest import file_sha256, fingerprint
from histopia.visualization._organ_metadata import (
    build_organ_metadata,
    organ_metadata_tab,
    validate_inventory,
)
from histopia.visualization._review_portal import (
    _WORKFLOW_CSS,
    _WORKFLOW_HTML,
    _WORKFLOW_JS,
)


def seal(manifest):
    manifest["fingerprint"] = fingerprint(
        {k: v for k, v in manifest.items() if k != "fingerprint"}
    )
    return manifest


def fixture():
    return seal(
        dict(
            schema_version="organ-inventory-1",
            updated_at="2026-09-18T00:00:00Z",
            scope="All audited internal inputs, including unresolved organs.",
            sources=[
                dict(id="ours", label="Our data", external=False),
                dict(id="archive", label="Original archive", external=False),
            ],
            rows=[
                dict(
                    scan_id="sha1-" + str(i) * 40,
                    sha1=str(i) * 40,
                    source_id=source,
                    organ=organ,
                    subject_id="mouse-" + str(i),
                    subject_evidence="filename only",
                    filename=name,
                    stain="Unverified",
                    inclusion="included",
                    copy_count=2 if i == 1 else 1,
                    review_href="../index.html?view=registration&source=ours&organ=liver&subject=mouse-1"
                    if i == 1
                    else None,
                    registration_status="available" if i == 1 else "not reconciled",
                    review_status="existing input review"
                    if i == 1
                    else "metadata only",
                    z_spacing_kind="assumed" if i == 1 else "unknown",
                    private_path="/do/not/export",
                )
                for i, source, organ, name in [
                    (1, "ours", "liver", "scan-a.ndpi"),
                    (2, "archive", "liver", "scan-b.ndpi"),
                    (3, "ours", "spleen", "=not-a-formula.scn"),
                    (4, "archive", "unassigned", "<script>not HTML</script>.scn"),
                ]
            ],
        )
    )


def test_complete_exports_deduplicate_by_identity_preserve_unknown_and_escape(tmp_path):
    manifest = fixture()
    build_organ_metadata(manifest, tmp_path)
    data = json.loads((tmp_path / "inventory.json").read_text())
    assert len(data["rows"]) == 4
    liver = next(o for o in data["organs"] if o["organ"] == "liver")
    assert (liver["scans"], liver["result_scans"], liver["metadata_only"]) == (2, 1, 1)
    rows = list(csv.DictReader((tmp_path / "downloads/liver.csv").open()))
    assert len(rows) == 2 and rows[0]["copy_count"] == "2"
    assert "private_path" not in (tmp_path / "inventory.json").read_text()
    assert "\\u003cscript>" in (tmp_path / "inventory-data.js").read_text()
    assert "'=not-a-formula.scn" in (tmp_path / "downloads/spleen.csv").read_text()
    assert "spleen.csv" in (tmp_path / "index.html").read_text()
    assert organ_metadata_tab(data)["scope"]["organs"] == [
        "liver",
        "spleen",
        "unassigned",
    ]
    # A subsequent empty inventory must remove obsolete exports and navigation.
    manifest["rows"] = []
    build_organ_metadata(seal(manifest), tmp_path)
    assert not (tmp_path / "downloads/liver.csv").exists()
    assert "liver.csv" not in (tmp_path / "index.html").read_text()
    assert (
        list(
            csv.DictReader(
                io.StringIO((tmp_path / "downloads/all-organs.csv").read_text())
            )
        )
        == []
    )


@pytest.mark.parametrize(
    "problem",
    ["fingerprint", "duplicate", "external", "hash", "link", "path", "copies"],
)
def test_inventory_rejects_invalid_bindings(problem):
    manifest = fixture()
    if problem == "duplicate":
        manifest["rows"].append(dict(manifest["rows"][0]))
    elif problem == "external":
        manifest["sources"][0]["external"] = True
    elif problem == "hash":
        manifest["rows"][0]["sha1"] = "a" * 40
    elif problem == "link":
        manifest["rows"][0]["review_href"] = "javascript:alert(1)"
    elif problem == "path":
        manifest["rows"][0]["organ"] = "../downloads"
    elif problem == "copies":
        manifest["rows"][0]["copy_count"] = 0
    seal(manifest)
    if problem == "fingerprint":
        manifest["rows"][0]["filename"] = "modified.ndpi"
    with pytest.raises(ValueError):
        validate_inventory(manifest)


def test_inventory_attachments_are_bound_and_cannot_override_tables(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("filename\na.ndpi\n")
    attachment = dict(
        path=str(source), sha256=file_sha256(source), label="All source files"
    )
    build_organ_metadata(
        fixture(), tmp_path / "out", attachments={"file-index.csv": attachment}
    )
    assert (
        tmp_path / "out/downloads/file-index.csv"
    ).read_bytes() == source.read_bytes()
    source.write_text("changed")
    with pytest.raises(ValueError, match="binding"):
        build_organ_metadata(
            fixture(), tmp_path / "out", attachments={"file-index.csv": attachment}
        )
    with pytest.raises(ValueError, match="colliding"):
        build_organ_metadata(
            fixture(), tmp_path / "out", attachments={"liver.csv": attachment}
        )


def curated_fixture():
    manifest = fixture()
    row = manifest["rows"][0]
    row.update(organ="pancreas", marker="Filename marker", section_order=7)
    row["review_href"] = (
        "../index.html?view=registration&source=ours&organ=pancreas&subject=mouse-1"
    )
    table = [
        dict(
            mouse_id=row["subject_id"],
            **{"Tissue Type": "panc"},
            raw_name="scan-a",
            file_type="NDPI",
            antibody="Curated marker",
            antibody_type="nuclear",
            order=str(i),
            order_text=str(i),
            label="Label",
            note="0.5",
        )
        for i in [1, 2]
    ]
    manifest["rows"], primary = reconcile_curated_metadata(
        manifest["rows"],
        table,
        source_id="ours",
        organ="pancreas",
        filename="curated.csv",
        source_sha256="a" * 64,
    )
    manifest["primary_tables"] = [primary]
    return seal(manifest)


def test_primary_csv_preserves_source_rows_beside_full_inventory(tmp_path):
    manifest = curated_fixture()
    build_organ_metadata(manifest, tmp_path)
    primary = list(csv.DictReader((tmp_path / "downloads/pancreas.csv").open()))
    inventory = list(
        csv.DictReader((tmp_path / "downloads/pancreas-inventory.csv").open())
    )
    assert len(primary) == 2 and len(inventory) == 1
    assert [r["table_order"] for r in primary] == ["1", "2"]
    assert {r["section_order"] for r in primary} == {"7"}
    assert {r["marker"] for r in primary} == {"Curated marker"}
    assert inventory[0]["analysis_marker"] == "Filename marker"
    assert "2 curated rows" in (tmp_path / "index.html").read_text()
    for key, value in [
        ("section_order", 99),
        ("source_id", "archive"),
        ("scan_id", "sha1-" + "b" * 40),
    ]:
        modified = json.loads(json.dumps(manifest))
        modified["primary_tables"][0]["rows"][0][key] = value
        with pytest.raises(ValueError):
            validate_inventory(seal(modified))


def test_curated_browser_retains_table_selection_and_other_sources(tmp_path):
    playwright = pytest.importorskip("playwright.sync_api")
    manifest = curated_fixture()
    manifest["rows"][1]["organ"] = "pancreas"
    site = tmp_path / "proxy/8765/review"
    output = site / "organ-metadata"
    build_organ_metadata(seal(manifest), output)
    metadata = json.loads((output / "inventory.json").read_text())
    tabs = [
        organ_metadata_tab(metadata),
        dict(
            id="registration",
            label="Registration",
            href="registration.html",
            scope=dict(sources=["ours"], organs=["pancreas"], subjects=["mouse-1"]),
        ),
    ]
    for name, content in [
        ("index.html", _WORKFLOW_HTML),
        ("workflow-review.js", _WORKFLOW_JS),
        ("workflow-review.css", _WORKFLOW_CSS),
        (
            "manifest-data.js",
            "globalThis.HISTOPIA_WORKFLOW_REVIEW=" + json.dumps(dict(tabs=tabs)),
        ),
        ("registration.html", '<p id="registered">Existing images</p>'),
    ]:
        (site / name).write_text(content)
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
            page.on("pageerror", lambda e: errors.append(str(e)))
            root = f"http://127.0.0.1:{server.server_port}/proxy/8765/review/"
            page.goto(root + "index.html?view=organ-metadata&organ=pancreas")
            frame = page.frame_locator("#review")
            playwright.expect(frame.locator("#coverage")).to_contain_text(
                "2 table rows · 1 scan file"
            )
            playwright.expect(frame.locator("#order-heading")).to_have_text(
                "Table order"
            )
            assert frame.locator("#rows tr[data-metadata-row]").count() == 2
            assert (
                frame.locator("#rows .stain")
                .first.inner_text()
                .endswith("Curated marker")
            )
            frame.locator("#metadata-table").select_option("inventory")
            page.wait_for_url(lambda url: "inventory_table=inventory" in url)
            playwright.expect(frame.locator("#coverage")).to_contain_text(
                "2 scan files"
            )
            page.reload(wait_until="networkidle")
            playwright.expect(frame.locator("#metadata-table")).to_have_value(
                "inventory"
            )
            assert (
                frame.locator("#organ-csv").get_attribute("href")
                == "downloads/pancreas-inventory.csv"
            )
            frame.locator("#source").select_option("archive")
            playwright.expect(frame.locator("#rows")).to_contain_text("scan-b.ndpi")
            # Explicitly choosing the curated cohort resolves the source too.
            frame.locator("#metadata-table").select_option("curated")
            playwright.expect(frame.locator("#source")).to_have_value("ours")
            playwright.expect(frame.locator("#coverage")).to_contain_text(
                "2 table rows"
            )
            page.set_viewport_size(dict(width=390, height=844))
            assert frame.locator("body").evaluate("el=>el.scrollWidth<=innerWidth")
            with page.expect_download() as download:
                frame.locator("#organ-csv").click()
            downloaded = list(csv.DictReader(open(download.value.path())))
            assert [r["table_order"] for r in downloaded] == ["1", "2"]
            frame.locator("#rows a.open").first.click()
            playwright.expect(
                page.frame_locator("#review").locator("#registered")
            ).to_be_visible()
            page.locator('[data-tab="organ-metadata"]').click()
            playwright.expect(frame.locator("#metadata-table")).to_have_value("curated")
            browser.close()
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
    assert not errors


@pytest.mark.parametrize("prefix", ["", "/proxy/8765"])
def test_inventory_browser_filters_history_downloads_and_return_to_images(
    tmp_path, prefix
):
    playwright = pytest.importorskip("playwright.sync_api")
    site = tmp_path / prefix.lstrip("/") / "review"
    output = site / "organ-metadata"
    build_organ_metadata(fixture(), output)
    metadata = json.loads((output / "inventory.json").read_text())
    tabs = [
        organ_metadata_tab(metadata),
        dict(
            id="registration",
            label="Registration",
            href="registration.html",
            scope=dict(sources=["ours"], organs=["liver"], subjects=["mouse-1"]),
        ),
    ]
    for name, text in [
        ("index.html", _WORKFLOW_HTML),
        ("workflow-review.js", _WORKFLOW_JS),
        ("workflow-review.css", _WORKFLOW_CSS),
        (
            "manifest-data.js",
            "globalThis.HISTOPIA_WORKFLOW_REVIEW=" + json.dumps(dict(tabs=tabs)),
        ),
        (
            "registration.html",
            '<img id="native" alt="Image" src="data:image/svg+xml,%3Csvg '
            "xmlns=%22http://www.w3.org/2000/svg%22 "
            'width=%2210%22 height=%2210%22/%3E">',
        ),
    ]:
        (site / name).write_text(text)
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
            page.on("pageerror", lambda e: errors.append(str(e)))
            root = f"http://127.0.0.1:{server.server_port}{prefix}/review/"
            page.goto(
                root
                + "index.html?view=registration&source=ours&organ=liver&subject=mouse-1"
            )
            page.locator('[data-tab="organ-metadata"]').click()
            frame = page.frame_locator("#review")
            playwright.expect(frame.locator("#coverage")).to_have_text(
                "4 scan files · 1 with review links · 3 metadata only"
            )
            frame.locator('[data-organ="liver"]').click()
            playwright.expect(frame.locator("#coverage")).to_contain_text(
                "2 scan files"
            )
            frame.locator("#source").select_option("archive")
            playwright.expect(frame.locator("#rows")).to_contain_text("scan-b.ndpi")
            page.wait_for_url(lambda url: "source=archive" in url)
            page.reload(wait_until="networkidle")
            playwright.expect(frame.locator("#rows tr[data-scan]")).to_have_count(1)
            frame.locator("#search").fill("unavailable")
            playwright.expect(frame.locator("#coverage")).to_contain_text(
                "0 scan files"
            )
            page.wait_for_url(lambda url: "inventory_q=unavailable" in url)
            page.go_back(wait_until="networkidle")
            playwright.expect(frame.locator("#rows")).to_contain_text("scan-b.ndpi")
            frame.locator("#clear").click()
            frame.locator('[data-organ="spleen"]').click()
            with page.expect_download() as download:
                frame.locator("#organ-csv").click()
            assert download.value.suggested_filename == "spleen.csv"
            page.set_viewport_size(dict(width=390, height=844))
            assert frame.locator("body").evaluate("el=>el.scrollWidth<=innerWidth")
            frame.locator("#clear").click()
            frame.locator("#rows a.open").click()
            playwright.expect(
                page.frame_locator("#review").locator("#native")
            ).to_be_visible()
            assert "view=registration" in page.url and "source=ours" in page.url
            page.locator('[data-tab="organ-metadata"]').click()
            playwright.expect(frame.locator("#coverage")).to_contain_text(
                "4 scan files"
            )
            # Plain HTML fallback retains the complete per-organ downloads.
            fallback = browser.new_page(java_script_enabled=False)
            fallback.goto(root + "organ-metadata/index.html")
            assert fallback.locator(
                'noscript a[href="downloads/unassigned.csv"]'
            ).is_visible()
            browser.close()
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
    assert not errors
