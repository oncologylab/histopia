from __future__ import annotations

import io
import json
import threading
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from histopia.topology._feedback import TopologyFeedbackStore
from histopia.topology._model import ReconstructedPlane
from histopia.topology._pipeline import _write_meshes, _write_planes
from histopia.topology._result import write_topology_result
from histopia.visualization import build_topology_review
from histopia.visualization._server import create_viewer_server


def test_topology_feedback_requires_every_current_pair(tmp_path: Path) -> None:
    run = _topology_run(tmp_path / "run")
    store = TopologyFeedbackStore(tmp_path / "feedback")
    review = store.review(cohort="sample", topology_run=run)

    assert len(review["slides"]) == 1
    with pytest.raises(ValueError, match="incomplete"):
        store.require_accepted(cohort="sample", topology_run=run)

    updated = store.save(
        {
            "cohort": "sample",
            "stage": "topology",
            "fingerprint": review["fingerprint"],
            "slide_id": "000-001",
            "decision": "accept",
            "labels": [],
            "comment": "",
            "reviewer": "Reviewer",
            "suggested_intervals": 1,
        },
        topology_run=run,
    )
    store.require_accepted(cohort="sample", topology_run=run)
    assert updated["summary"]["by_decision"] == {"accept": 1}


def test_topology_viewer_builds_section_assets_and_gates_surfaces(
    tmp_path: Path,
) -> None:
    run = _topology_run(tmp_path / "run")
    index = build_topology_review({"sample": run}, tmp_path / "viewer")

    manifest = json.loads((index.parent / "manifest.json").read_text())
    assert manifest["schema_version"] == 2
    cohort = manifest["cohorts"][0]
    assert cohort["id"] == "sample"
    assert cohort["planes"][0]["viewer_asset"].startswith("assets/sample/")
    plane_asset = index.parent / cohort["planes"][0]["viewer_asset"]
    assert plane_asset.read_bytes()[:4] == b"HTP1"
    assert cohort["meshes"][0]["viewer_asset"].startswith("assets/sample/")
    assert (index.parent / cohort["meshes"][0]["viewer_asset"]).is_file()
    assert cohort["surface_qc"] == {
        "status": "failed",
        "reasons": [
            "median_adjacent_agreement_below_0.75",
        ],
        "median_adjacent_agreement": 0.0,
        "component_count": 1,
    }
    assert (index.parent / "vendor" / "three.module.min.js").is_file()
    javascript = (index.parent / "topology-review.js").read_text()
    assert 'mode="sections"' in javascript
    assert "Diagnostic surfaces failed continuity QC" in javascript


@pytest.mark.browser
def test_topology_viewer_defaults_to_section_faithful_rendering(
    tmp_path: Path,
) -> None:
    playwright = pytest.importorskip("playwright.sync_api")
    run = _topology_run(tmp_path / "run")
    build_topology_review({"sample": run}, tmp_path / "viewer")
    server = create_viewer_server(
        tmp_path,
        bind="127.0.0.1",
        port=0,
        required_routes=("viewer",),
    )
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
            page.goto(
                f"http://127.0.0.1:{server.server_port}/viewer/",
                wait_until="networkidle",
            )
            page.wait_for_function(
                "() => document.querySelector('#provenance').textContent"
                ".includes('ready')"
            )
            assert (
                page.locator("[data-mode='sections']").get_attribute("class")
                == "active"
            )
            assert page.locator(".metric").count() == 4
            assert page.locator("#surface-status").get_attribute("class") == "fail"
            screenshot = page.locator("canvas").screenshot()
            pixels = np.asarray(Image.open(io.BytesIO(screenshot)).convert("RGB"))
            assert np.ptp(pixels.reshape(-1, 3), axis=0).max() > 20
            page.locator(".pair").first.click()
            assert page.locator(".pair.selected").count() == 1
            page.locator("#reset").click()
            assert page.locator(".pair.selected").count() == 0
            for width, height in ((1920, 1080), (3840, 2160)):
                page.set_viewport_size({"width": width, "height": height})
                overflow = page.evaluate(
                    """() => ({
                      x: document.documentElement.scrollWidth > innerWidth,
                      y: document.documentElement.scrollHeight > innerHeight,
                      aside: document.querySelector('aside').getBoundingClientRect(),
                    })"""
                )
                assert not overflow["x"]
                assert not overflow["y"]
                assert overflow["aside"]["width"] <= 351
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert not errors


def _topology_run(root: Path) -> Path:
    root.mkdir()
    labels = np.full((8, 9), -1, dtype=np.int16)
    labels[2:6, 2:7] = 0
    membership = np.stack((labels == 0, labels == 1)).astype(np.float32)
    planes = tuple(
        ReconstructedPlane(
            z_um=float(index * 5),
            segment=0,
            source_section=index,
            target_section=index,
            fraction=0,
            observed=True,
            slide_id=f"slide-{index}",
            membership=membership,
            labels=labels,
            support=labels >= 0,
            uncertainty=np.zeros(labels.shape, dtype=np.float32),
        )
        for index in range(2)
    )
    plane_rows = _write_planes(root, planes)
    mesh_rows, class_rows = _write_meshes(
        root,
        planes,
        class_count=2,
        palette=("#ff0000", "#00ff00"),
        origin_um_xy=(0, 0),
        spacing_um=10,
        thickness_um=5,
    )
    (root / "preflight.json").write_text("{}")
    (root / "benchmark.json").write_text(
        json.dumps(
            {
                "summary": {
                    "flow_macro_class_dice": 0.8,
                    "flow_gain_over_zero": 0.1,
                    "gap_interval_accuracy": 0.7,
                }
            }
        )
    )
    write_topology_result(
        root,
        {
            "schema_version": 1,
            "preflight": "preflight.json",
            "benchmark": "benchmark.json",
            "selected_k": 2,
            "palette": ["#ff0000", "#00ff00"],
            "z_source": "manifest_measured",
            "section_thickness_um": 5,
            "reference_grid": {
                "origin_um_xy": [0, 0],
                "shape_rc": [8, 9],
                "spacing_um": 10,
            },
            "observed_section_count": 2,
            "virtual_section_count": 0,
            "segment_count": 1,
            "gap_decisions": [
                {
                    "source_section": 0,
                    "target_section": 1,
                    "intervals": 1,
                    "missing_sections": 0,
                    "status": "manifest",
                    "confidence": 1,
                    "score": 0.1,
                    "reasons": [],
                }
            ],
            "pair_evidence": [{}],
            "planes": plane_rows,
            "meshes": mesh_rows,
            "classes": class_rows,
        },
    )
    return root
