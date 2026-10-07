from __future__ import annotations

import json
import mimetypes
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

import pytest
from PIL import Image

from histopia.study._eligibility import POLICY_ID, assess_reconstruction_eligibility
from histopia.study._manifest import file_sha256, fingerprint
from histopia.study._serial import validate_serial_study
from histopia.visualization._results_catalog import (
    attach_results_catalog,
    build_results_catalog,
)
from histopia.visualization._review_portal import (
    _WORKFLOW_CSS,
    _WORKFLOW_HTML,
    _WORKFLOW_JS,
    themed_review_css,
)


def bind_synthetic_stacks(sources, rows):
    """Synthetic acquisition evidence; source names do not qualify real cohorts."""
    entries = []
    for row in rows:
        source = next(s for s in sources if s["id"] == row["source_id"])
        identity = dict(
            source_id=source["id"],
            species=source["species"],
            subject_id=row["subject_id"],
        )
        physical_ids = [
            row.get("physical_section_id", row["id"] + "-plane1"),
            row["id"] + "-plane2",
            row["id"] + "-plane3",
        ]
        specimen, block = (
            row.get("specimen_id", row["subject_id"]),
            row.get("block_id", "block1"),
        )
        study = validate_serial_study(
            dict(
                schema_version="serial-1",
                study_id=row["id"],
                subjects=[dict(**identity, role="development", exposed=True)],
                sections=[
                    dict(
                        **identity,
                        physical_section_id=key,
                        organ=row["organ"],
                        specimen_id=specimen,
                        block_id=block,
                        section_order=row.get("section_order", 1) + i,
                        z_um=i * 4,
                        z_reference=physical_ids[0],
                        z_evidence="synthetic section log",
                    )
                    for i, key in enumerate(physical_ids)
                ],
                acquisitions=[
                    dict(
                        acquisition_id=row["id"] + f"-scan{i}",
                        physical_section_id=key,
                        modality="IHC",
                        source_binding={"sha256": fingerprint([row["id"], i])},
                    )
                    for i, key in enumerate(physical_ids)
                ],
            )
        )
        assessment = dict(
            reconstruction_id=row["id"] + "-stack",
            study_fingerprint=study["fingerprint"],
            external=source["external"],
            acquisition_ids=[a["acquisition_id"] for a in study["acquisitions"]],
            acquisition_qc={
                a["acquisition_id"]: dict(
                    source_binding_fingerprint=fingerprint(a["source_binding"]),
                    imaging_mode="brightfield",
                    verified_modality="IHC",
                    documentation_sha256=fingerprint("synthetic docs"),
                    image_available=True,
                    image_sha256=a["source_binding"]["sha256"],
                    native_visual_qc="pass",
                    native_visual_qc_sha256=fingerprint("synthetic native QC"),
                )
                for a in study["acquisitions"]
            },
            physical_provenance=dict(
                block_verified=True,
                order_verified=True,
                spacing_status="documented",
                evidence_sha256=fingerprint("synthetic physical evidence"),
            ),
            reconstruction_qc=dict(
                study_fingerprint=study["fingerprint"],
                reviewed_section_ids=physical_ids,
                continuity="pass",
                alignment="pass",
                coverage="pass",
                stack_visual="pass",
                registration_fingerprint=fingerprint("synthetic registration"),
                stack_sha256=fingerprint("synthetic stack"),
                visual_qc_sha256=fingerprint("synthetic stack QC"),
            ),
        )
        decision = assess_reconstruction_eligibility(study, assessment)
        row.update(
            specimen_id=specimen,
            block_id=block,
            reconstruction_id=assessment["reconstruction_id"],
            eligibility_fingerprint=decision["fingerprint"],
            acquisition_ids=decision["acquisition_ids"],
            acquisition_modalities=decision["modalities"],
        )
        entries.append(dict(study=study, assessment=assessment))
    return dict(
        schema_version="reconstruction-registry-1", policy_id=POLICY_ID, entries=entries
    )


def example(root: Path):
    image = root / "private-native-image.png"
    Image.new("RGB", (24, 16), "lavender").save(image)
    sources = [
        {
            "id": "kpf",
            "label": "Our KPF data",
            "external": False,
            "species": "Mouse",
            "description": "Our tissue",
        },
        {
            "id": "hpa",
            "label": "Human Protein Atlas",
            "external": True,
            "species": "Human",
            "description": "Independent source",
        },
        {
            "id": "tcga",
            "label": "TCGA",
            "external": True,
            "species": "Human",
            "description": "Not staged",
        },
    ]
    rows = []
    for source, organ in [("kpf", "pancreas"), ("kpf", "liver"), ("hpa", "pancreas")]:
        key = f"{source}-{organ}-001"
        rows.append(
            {
                "id": key,
                "source_id": source,
                "organ": organ,
                "subject_id": "001",
                "title": key,
                "stage": "Cell fields",
                "status": "ready",
                "summary": "Native candidate, not approved",
                "fingerprint": fingerprint(key),
                "media": [
                    {
                        "path": image,
                        "sha256": file_sha256(image),
                        "label": f"Field {i + 1}",
                    }
                    for i in range(2)
                ],
                "table": [{"cells": 4, "accuracy": None}],
                "facts": {"Scope": "candidate"},
            }
        )
    return sources, rows, bind_synthetic_stacks(sources, rows)


def test_catalog_separates_sources_and_removes_private_paths(tmp_path):
    sources, rows, registry = example(tmp_path)
    out = tmp_path / "review" / "data-catalog"
    build_results_catalog(
        sources, rows, out, updated_at="frozen", reconstruction_registry=registry
    )
    data = json.loads((out / "catalog.json").read_text())
    assert [r["source_id"] for r in data["datasets"]] == ["kpf", "kpf", "hpa"]
    assert all(r["scientific_approval"] is False for r in data["datasets"])
    assert str(tmp_path) not in (out / "catalog.json").read_text()
    assert len(list((out / "assets").iterdir())) == 1
    assert "<noscript>" in (out / "index.html").read_text()
    original = (out / "catalog.json").read_bytes()
    build_results_catalog(
        sources, rows, out, updated_at="frozen", reconstruction_registry=registry
    )
    assert (out / "catalog.json").read_bytes() == original
    # A content-addressed export is checked even when its previous validation
    # is cached; a later file change invalidates that cache entry.
    exported = next((out / "assets").iterdir())
    exported.write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        build_results_catalog(
            sources, rows, out, updated_at="later", reconstruction_registry=registry
        )


@pytest.mark.parametrize(
    "change",
    [
        {"status": "approved"},
        {"links": [{"label": "unsafe", "href": "javascript:alert(1)"}]},
        {"links": [{"label": "proxy escape", "href": "/review/index.html"}]},
    ],
)
def test_catalog_rejects_ambiguous_scope_and_unsafe_links(tmp_path, change):
    sources, rows, registry = example(tmp_path)
    rows[0].update(change)
    with pytest.raises(ValueError):
        build_results_catalog(
            sources,
            rows,
            tmp_path / "out",
            updated_at="now",
            reconstruction_registry=registry,
        )


def test_catalog_preserves_workflow_tabs_and_rejects_bad_evidence(tmp_path):
    sources, rows, registry = example(tmp_path)
    review = tmp_path / "review"
    out = review / "data-catalog"
    build_results_catalog(
        sources, rows, out, updated_at="now", reconstruction_registry=registry
    )
    previous = {"id": "cells", "label": "Cells", "href": "cells/index.html"}
    (review / "manifest.json").write_text(json.dumps({"tabs": [previous]}))
    attach_results_catalog(out, review)
    attach_results_catalog(out, review)
    tabs = json.loads((review / "manifest.json").read_text())["tabs"]
    assert len(tabs) == 2 and tabs[1] == previous
    rows[0]["media"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="result binding"):
        build_results_catalog(
            sources, rows, out, updated_at="now", reconstruction_registry=registry
        )


def test_catalog_accepts_new_organs_and_preserves_physical_identity(tmp_path):
    sources, rows, registry = example(tmp_path)
    original = rows[0]["fingerprint"]
    rows[0].update(
        organ="colorectal",
        specimen_id="CRC1",
        physical_section_id="physical-048",
        section_order=48,
        evidence_kind="stack",
        short_title="Serial tissue",
    )
    out = tmp_path / "review"
    registry = bind_synthetic_stacks(sources, rows)
    build_results_catalog(
        sources, rows, out, updated_at="now", reconstruction_registry=registry
    )
    data = json.loads((out / "catalog.json").read_text())
    assert "colorectal" in data["organs"]
    assert data["datasets"][0]["physical_section_id"] == "physical-048"
    assert data["datasets"][0]["fingerprint"] == original


@pytest.mark.browser
@pytest.mark.parametrize("prefix", ["", "/workspace/proxy/8765"])
def test_catalog_navigation_notes_exports_and_proxy(tmp_path, prefix):
    playwright = pytest.importorskip("playwright.sync_api")
    sources, rows, registry = example(tmp_path)
    review = tmp_path / "review"
    out = review / "data-catalog"
    build_results_catalog(
        sources, rows, out, updated_at="now", reconstruction_registry=registry
    )
    (review / "manifest.json").write_text(
        json.dumps({"tabs": [{"id": "cells", "label": "Cells", "href": "cells.html"}]})
    )
    (review / "cells.html").write_text("<h1>Existing cells</h1>")
    (review / "index.html").write_text(_WORKFLOW_HTML)
    (review / "workflow-review.css").write_text(themed_review_css(_WORKFLOW_CSS))
    (review / "workflow-review.js").write_text(_WORKFLOW_JS)
    attach_results_catalog(out, review)
    origin = "http://histopia.test"
    base = prefix + "/review/"
    errors, escaped = [], []
    with playwright.sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1500, "height": 950})

        def respond(route):
            path = unquote(urlsplit(route.request.url).path)
            if not path.startswith(base):
                escaped.append(path)
                route.fulfill(status=404)
                return
            file = review / (path[len(base) :] or "index.html")
            route.fulfill(
                status=200 if file.is_file() else 404,
                content_type=mimetypes.guess_type(file)[0]
                or "application/octet-stream",
                body=file.read_bytes() if file.is_file() else b"",
            )

        page.route(origin + "/**", respond)
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(
            origin
            + base
            + "index.html?view=data-catalog&source=hpa&organ=pancreas&field=1"
        )
        frame = page.frame_locator("#review")
        assert frame.locator("#title").inner_text() == "hpa-pancreas-001"
        assert frame.locator("#field").input_value() == "1"
        assert frame.locator("#datasets button").count() == 1
        frame.locator("#previous").click()
        page.wait_for_url(
            lambda url: parse_qs(urlsplit(url).query).get("field") == ["0"]
        )
        frame.locator('[data-source="kpf"]').click()
        frame.locator('[data-organ="liver"]').click()
        assert frame.locator("#title").inner_text() == "kpf-liver-001"
        page.wait_for_url(
            lambda url: parse_qs(urlsplit(url).query).get("organ") == ["liver"]
        )
        assert "source=kpf" in page.url and "organ=liver" in page.url
        frame.locator("#native").click()
        assert frame.locator("#image").evaluate("im => im.width") == 24
        frame.locator("#notes summary").click()
        frame.locator("#review-state").select_option("needs-changes")
        frame.locator("#note").fill("Review boundary at field 1")
        with page.expect_download() as download:
            frame.locator("#export-notes").click()
        notes = json.loads(Path(download.value.path()).read_text())
        assert notes["scientific_approval"] is False
        assert notes["notes"][0]["source_id"] == "kpf"
        assert notes["notes"][0]["organ"] == "liver"
        page.reload()
        assert frame.locator("#note").input_value() == "Review boundary at field 1"
        assert frame.locator('[data-source="tcga"]').count() == 0
        page.goto(
            origin
            + base
            + "index.html?view=data-catalog&source=kpf&organ=liver&subject=missing"
        )
        playwright.expect(frame.locator("#image")).to_be_visible()
        playwright.expect(frame.locator("#subject")).to_have_value("001")
        assert "source=kpf" in page.url and "organ=liver" in page.url
        assert "Showing specimen 001" in frame.locator(".selection-notice").inner_text()
        page.goto(origin + base + "index.html?view=data-catalog&source=tcga")
        assert frame.locator("#empty").is_visible()
        page.set_viewport_size({"width": 390, "height": 844})
        assert not frame.locator("html").evaluate("el => el.scrollWidth > innerWidth")
        page.locator('[data-tab="cells"]').click()
        assert (
            page.frame_locator("#review")
            .get_by_role("heading", name="Existing cells")
            .is_visible()
        )
        # A scoped tool must not display a default mouse for public tissue.
        manifest = json.loads((review / "manifest.json").read_text())
        manifest["tabs"][1]["scope"] = {
            "sources": ["kpf"],
            "organs": ["pancreas"],
            "subjects": ["001"],
        }
        (review / "manifest.json").write_text(json.dumps(manifest))
        attach_results_catalog(out, review)
        page.goto(
            origin
            + base
            + "index.html?view=cells&source=hpa&organ=pancreas&subject=001"
        )
        assert (
            page.frame_locator("#review")
            .get_by_text("No result in this view for this source and organ.")
            .is_visible()
        )
        # Tools stay discoverable even when this source has no matching result.
        assert page.locator('[data-tab="cells"]').is_visible()
        page.goto(
            origin
            + base
            + "index.html?view=cells&source=kpf&organ=pancreas&subject=001"
        )
        assert (
            page.frame_locator("#review")
            .get_by_role("heading", name="Existing cells")
            .is_visible()
        )
        assert "mouse=001" in page.locator("#review").get_attribute("src")
        assert "cohort=001" in page.locator("#review").get_attribute("src")
        fallback = browser.new_context(java_script_enabled=False).new_page()
        fallback.route(origin + "/**", respond)
        fallback.goto(origin + base + "data-catalog/index.html")
        assert fallback.get_by_role("heading", name="Human Protein Atlas").is_visible()
        assert fallback.get_by_role("link", name="Field 1").count() == 3
        browser.close()
    assert not escaped and not errors
