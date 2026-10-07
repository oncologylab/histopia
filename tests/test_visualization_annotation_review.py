from __future__ import annotations

import hashlib
import http.client
import io
import json
import os
import threading
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from histopia.annotation import AnnotationStore
from histopia.visualization import _review_portal, build_annotation_review
from histopia.visualization._review_api import ReviewDecisionService
from histopia.visualization._server import create_viewer_server
from histopia.visualization._wsi_tiles import WsiTileService


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _semantic_run(root: Path, slides: tuple[str, ...] = ("a.ndpi", "b.ndpi")) -> Path:
    root.mkdir()
    (root / "semantic_result.json").write_text(
        json.dumps({"slides": [{"id": slide} for slide in slides]})
    )
    return root


def test_annotation_review_builds_fixed_high_resolution_editor(tmp_path: Path) -> None:
    semantic = _semantic_run(tmp_path / "semantic")
    annotation = tmp_path / "annotations"

    index = build_annotation_review(
        {"mouse": annotation},
        {"mouse": semantic},
        tmp_path / "site",
    )

    assert index.is_file()
    manifest = json.loads((index.parent / "manifest.json").read_text())
    assert manifest["cohorts"] == [
        {
            "id": "mouse",
            "sections": [
                {"id": "001", "slide": "a.ndpi"},
                {"id": "002", "slide": "b.ndpi"},
            ],
        }
    ]
    script = (index.parent / "annotation-review.js").read_text()
    assert "OpenSeadragon" in script
    assert "new OpenSeadragon.TileSource" in script
    assert "const minLevel = maxLevel - item.levels.length + 1" in script
    assert "imageLoaderLimit: 2" in script
    assert "preserveViewport: false" in script
    assert "animationTime: 0, blendTime: 0" in script
    assert 'viewer.addHandler("tile-loaded", onTile)' in script
    assert "setLayerLoading(true" in script
    assert 'byId("viewer").histopiaViewer = viewer' in script
    assert "${name}/${item.digest}/dzi/${minLevel}/" in script
    assert "request !== layerRequest" in script
    assert "layerOpening" in script
    assert "viewState = currentViewState()" in script
    assert "document.body.dataset.layerReady" in script
    assert "`${adjacent.id} ${data.label}`" in script
    assert "homeZoomRatio" in script
    assert "imageToViewportCoordinates" in script
    assert "applyConstraints(true)" in script
    assert "Checking native WSI and annotation provenance" in script
    assert "async function initialize()" in script
    assert "option.disabled = true" in script
    assert "No provenance-valid annotation cohorts" in script
    assert "Native WSI tiles unavailable for this cohort" in script
    assert "request !== loadRequest" in script
    assert "/api/annotations/section" in script
    assert "viewerElementToImageCoordinates" in script
    assert "Registered comparison; draw on Raw" in script
    assert "innerHTML" not in script
    assert "overflow:hidden" in (index.parent / "annotation-review.css").read_text()
    assert 'id="loader" role="status"' in index.read_text()
    assert str(tmp_path) not in (index.parent / "manifest-data.js").read_text()


@pytest.mark.browser
def test_annotation_layer_switch_preserves_exact_viewport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    playwright = pytest.importorskip("playwright.sync_api")
    semantic = _semantic_run(tmp_path / "semantic", ("a.ndpi",))
    unavailable_semantic = _semantic_run(
        tmp_path / "unavailable-semantic", ("missing.ndpi",)
    )
    annotations = tmp_path / "annotations"
    site = tmp_path / "site"
    build_annotation_review(
        {"bad": tmp_path / "bad-annotations", "mouse": annotations},
        {"bad": unavailable_semantic, "mouse": semantic},
        site / "annotations",
    )
    registration = tmp_path / "registration"
    registration.mkdir()
    config = tmp_path / "review-config.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cohorts": {
                    "mouse": {
                        "registration": str(registration),
                        "semantic": str(semantic),
                        "annotations": str(annotations),
                    }
                },
            }
        )
    )
    rgb = np.full((256, 512, 3), 242, dtype=np.uint8)
    rgb[24:232, 48:464] = (155, 82, 105)
    encoded = io.BytesIO()
    Image.fromarray(rgb).save(encoded, "JPEG", quality=90)
    raw_tile = encoded.getvalue()
    square = np.full((512, 512, 3), 242, dtype=np.uint8)
    square[48:464, 48:464] = (155, 82, 105)
    encoded = io.BytesIO()
    Image.fromarray(square).save(encoded, "JPEG", quality=90)
    registered_tile = encoded.getvalue()

    class FakeTiles:
        def catalog(self, cohort: str) -> dict[str, object]:
            if cohort == "bad":
                raise ValueError("cohort bad has no provenance-valid annotation review")
            assert cohort == "mouse"
            return {"cohort": cohort, "sections": [{"section": "001"}]}

        def metadata(self, cohort: str, section: str) -> dict[str, object]:
            assert (cohort, section) == ("mouse", "001")
            layers = {}
            for name, digest, width, height in (
                ("raw", "a", 512, 256),
                ("registered", "b", 512, 512),
            ):
                layers[name] = {
                    "digest": digest * 64,
                    "tile_size": 512,
                    "width": width,
                    "height": height,
                    "levels": [{"width": width, "height": height}],
                    "format": "jpg",
                }
            return {
                "cohort": cohort,
                "section": section,
                "label": "a.ndpi",
                "layers": layers,
            }

        def render_tile(self, *args):
            payload = registered_tile if args[2] == "registered" else raw_tile
            return payload, "image/jpeg", '"tile"'

    monkeypatch.setattr(
        WsiTileService,
        "from_runs",
        classmethod(lambda cls, *args, **kwargs: FakeTiles()),
    )
    server = create_viewer_server(
        site,
        bind="127.0.0.1",
        port=0,
        required_routes=("annotations",),
        review_config=config,
        public_review_write=True,
    )
    server.review_service.annotation_catalog = lambda cohort: {
        "ontology": {
            "classes": [{"id": "tissue", "label": "Tissue", "color": "#aa3355"}]
        }
    }
    server.review_service.annotation_section = lambda cohort, section: {
        "type": "FeatureCollection",
        "features": [],
        "histopia": {"revision": 0, "notes": ""},
    }
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with playwright.sync_playwright() as runtime:
            browser = getattr(
                runtime, os.environ.get("HISTOPIA_BROWSER", "chromium")
            ).launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 900})
            page_errors: list[str] = []
            page.on("pageerror", lambda error: page_errors.append(str(error)))
            page.goto(
                f"http://127.0.0.1:{server.server_port}/annotations/",
                wait_until="domcontentloaded",
            )
            page.wait_for_timeout(250)
            assert not page_errors
            state = page.evaluate(
                """() => ({status:document.querySelector('#status').textContent,
                  viewer:Boolean(document.querySelector('#viewer').histopiaViewer),
                  items:document.querySelector('#viewer').histopiaViewer?.world
                    .getItemCount(),
                  cohorts:globalThis.HISTOPIA_ANNOTATION_REVIEW?.cohorts.length})"""
            )
            assert state["viewer"] and state["cohorts"] == 2, state
            assert state["status"] != (
                "Checking native WSI and annotation provenance"
            ), state
            assert page.locator('#cohort option[value="bad"]').is_disabled()
            assert page.locator("#cohort").input_value() == "mouse"
            page.wait_for_function(
                """() => document.querySelector('#viewer').histopiaViewer.world
                  .getItemCount() === 1""",
                polling=100,
            )
            before = page.evaluate(
                """() => {const v=document.querySelector('#viewer').histopiaViewer;
                  const item=v.world.getItemAt(0), size=item.getContentSize();
                  v.viewport.panTo(item.imageToViewportCoordinates(
                    new OpenSeadragon.Point(size.x*.55,size.y*.55)), true);
                  v.viewport.zoomTo(v.viewport.getHomeZoom()*1.3, null, true);
                  v.viewport.applyConstraints(true);
                  const c=item.viewportToImageCoordinates(v.viewport.getCenter(true));
                  return {x:c.x/size.x,y:c.y/size.y,
                    z:v.viewport.getZoom(true)/v.viewport.getHomeZoom()};}"""
            )
            page.locator('[data-layer="registered"]').click(force=True)
            page.locator('[data-layer="raw"]').click(force=True)
            page.locator('[data-layer="registered"]').click(force=True)
            page.wait_for_function(
                """() => document.querySelector('[data-layer="registered"]')
                  .getAttribute('aria-pressed') === 'true'
                  && document.body.dataset.layerReady === 'registered'
                  && document.querySelector('#status').textContent
                    .startsWith('Registered comparison')""",
                polling=100,
            )
            page.wait_for_timeout(250)
            after = page.evaluate(
                """() => {const v=document.querySelector('#viewer').histopiaViewer;
                  const item=v.world.getItemAt(0), size=item.getContentSize();
                  const c=item.viewportToImageCoordinates(v.viewport.getCenter(true));
                  return {x:c.x/size.x,y:c.y/size.y,
                    z:v.viewport.getZoom(true)/v.viewport.getHomeZoom()};}"""
            )
            assert after == pytest.approx(before, abs=1e-10)
            for width, height in ((1920, 1080), (3840, 2160), (390, 844)):
                page.set_viewport_size({"width": width, "height": height})
                page.wait_for_timeout(100)
                layout = page.evaluate(
                    """() => ({
                      x: document.documentElement.scrollWidth > innerWidth,
                      y: document.documentElement.scrollHeight > innerHeight,
                      viewer: document.querySelector('#viewer')
                        .getBoundingClientRect().toJSON(),
                      controls: document.querySelector('aside')
                        .getBoundingClientRect().toJSON(),
                    })"""
                )
                assert not layout["x"]
                assert not layout["y"]
                assert layout["viewer"]["width"] > 0
                assert layout["viewer"]["height"] > 0
                assert layout["controls"]["width"] > 0
                assert layout["controls"]["height"] > 0
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize("include_methods", [False, True])
def test_workflow_review_adds_annotation_tab(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    include_methods: bool,
) -> None:
    registration = tmp_path / "registration"
    semantic = tmp_path / "semantic"
    registration.mkdir()
    semantic.mkdir()
    (registration / "registration_result.json").write_text("{}")

    def page(*args, **kwargs):
        output = Path(args[1])
        output.mkdir(parents=True)
        (output / "index.html").write_text("page")
        return output / "index.html"

    def atlas(*args, **kwargs):
        output = Path(args[1])
        output.mkdir(parents=True)
        (output / "manifest.json").write_text('{"mice":[]}')
        (output / "index.html").write_text("atlas")
        return output / "index.html"

    def annotations(*args, **kwargs):
        output = Path(args[2])
        output.mkdir(parents=True)
        (output / "index.html").write_text("annotations")
        return output / "index.html"

    monkeypatch.setattr(_review_portal, "build_registration_cohort_review", page)
    monkeypatch.setattr(_review_portal, "build_section_viewer", atlas)
    monkeypatch.setattr(
        "histopia.visualization._annotation_review.build_annotation_review",
        annotations,
    )

    output = tmp_path / "review"
    _review_portal.build_workflow_review(
        {"mouse": registration},
        output,
        semantic_runs={"mouse": semantic},
        annotation_runs={"mouse": tmp_path / "annotations"},
        **({"include_methods": True} if include_methods else {}),
    )

    manifest = json.loads((output / "manifest.json").read_text())
    assert [tab["id"] for tab in manifest["tabs"]] == [
        "registration",
        "atlas",
        "annotations",
        *(["methods"] if include_methods else []),
        "decisions",
    ]
    assert (output / "methods" / "index.html").is_file()


def test_workflow_review_rejects_annotation_without_semantic_run(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="requires semantic runs for: mouse"):
        _review_portal.build_workflow_review(
            {"mouse": tmp_path / "registration"},
            tmp_path / "review",
            annotation_runs={"mouse": tmp_path / "annotations"},
        )


def test_review_service_routes_annotations_to_bound_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registration = tmp_path / "registration"
    semantic = tmp_path / "semantic"
    annotations = tmp_path / "annotations"
    registration.mkdir()
    semantic.mkdir()
    store = AnnotationStore.initialize(
        annotations,
        registration_fingerprint=_digest("registration"),
        semantic_fingerprint=_digest("semantic"),
    )
    monkeypatch.setattr(
        AnnotationStore,
        "from_runs",
        classmethod(lambda cls, *args, **kwargs: store),
    )
    config = tmp_path / "registry.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cohorts": {
                    "mouse": {
                        "registration": str(registration),
                        "semantic": str(semantic),
                        "annotations": str(annotations),
                    }
                },
            }
        )
    )
    service = ReviewDecisionService.from_file(config)
    collection = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"class_id": "stroma"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[1, 1], [4, 1], [4, 4], [1, 1]]],
                },
            }
        ],
    }

    catalog = service.annotation_catalog("mouse")
    saved = service.save_annotation_section(
        {
            "cohort": "mouse",
            "section": "001",
            "reviewer": "Reviewer",
            "expected_revision": 0,
            "feature_collection": collection,
        }
    )

    assert catalog["cohort"] == "mouse"
    assert saved["histopia"]["revision"] == 1
    assert service.annotation_section("mouse", "001") == saved


def test_annotation_http_routes_allow_larger_revision_payloads(tmp_path: Path) -> None:
    for route in ("histopia", "review"):
        directory = tmp_path / route
        directory.mkdir()
        (directory / "index.html").write_text(route)
    registration = tmp_path / "registration"
    registration.mkdir()
    config = tmp_path / "registry.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cohorts": {"mouse": {"registration": str(registration)}},
            }
        )
    )
    server = create_viewer_server(
        tmp_path,
        bind="127.0.0.1",
        port=0,
        required_routes=("histopia", "review"),
        review_config=config,
        public_review_write=True,
    )
    service = server.review_service
    service.annotation_catalog = lambda cohort: {"cohort": cohort, "sections": {}}
    service.annotation_section = lambda cohort, section: {
        "cohort": cohort,
        "section": section,
    }
    service.save_annotation_section = lambda request: {
        "revision": request["expected_revision"] + 1
    }
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port)
        connection.request("GET", "/api/annotations?cohort=mouse")
        catalog = connection.getresponse()
        assert catalog.status == 200
        assert json.loads(catalog.read()) == {"cohort": "mouse", "sections": {}}

        connection.request(
            "GET",
            "/api/annotations/section?cohort=mouse&section=001",
        )
        section = connection.getresponse()
        assert section.status == 200
        assert json.loads(section.read())["section"] == "001"

        body = json.dumps(
            {
                "cohort": "mouse",
                "section": "001",
                "reviewer": "Reviewer",
                "expected_revision": 2,
                "feature_collection": {"type": "FeatureCollection", "features": []},
                "notes": "x" * 20_000,
            }
        )
        connection.request(
            "POST",
            "/api/annotations/section",
            body=body,
            headers={"Content-Type": "application/json"},
        )
        saved = connection.getresponse()
        assert saved.status == 200
        assert json.loads(saved.read())["annotations"]["revision"] == 3
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
