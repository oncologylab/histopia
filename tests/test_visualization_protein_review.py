from __future__ import annotations

import io
import json
import os
import threading
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from histopia.protein import (
    ProteinPredictions,
    approve_protein_result,
    write_protein_result,
)
from histopia.visualization import build_protein_review
from histopia.visualization._server import (
    _WSI_DZI_TILE_RE,
    _WSI_MODEL_DZI_TILE_RE,
    create_viewer_server,
)
from histopia.visualization._wsi_tiles import (
    WsiLayer,
    WsiLevel,
    WsiTileService,
    _protein_heatmap_rgba,
    _protein_layer,
    _ProteinOverview,
    _read_spatial_protein_overview,
    _render_protein_layer_tile,
    _validated_protein_predictions,
)


def _protein_run(root: Path, sections: tuple[str, ...] = ("001",)) -> Path:
    root.mkdir()
    (root / "model.npz").write_bytes(b"portable-model")
    slides = []
    for section in sections:
        predictions = ProteinPredictions(
            target_id="perk",
            model_fingerprint="1" * 64,
            label_ids=np.asarray([1, 2], dtype=np.uint32),
            section_ids=np.asarray([section, section]),
            expression_probability=np.asarray([0.1, 0.9], dtype=np.float32),
            relative_expression=np.asarray([0.2, 0.8], dtype=np.float32),
            predicted_od_reference=np.asarray([0.3, 1.1], dtype=np.float32),
            measured_od=np.asarray([0.4, np.nan], dtype=np.float32),
            uncertainty=np.asarray([0.02, 0.2], dtype=np.float32),
            supported=np.asarray([True, True]),
            provenance={"feature_view": "stain_neutral_v1"},
        )
        predictions.save(root / f"{section}.npz")
        slides.append(
            {
                "section": section,
                "predictions": f"{section}.npz",
                "prediction_fingerprint": predictions.fingerprint,
                "cells": 2,
                "measured_cells": 1,
            }
        )
    write_protein_result(
        root,
        {
            "schema_version": 1,
            "target_id": "perk",
            "assay_domain": "perk-v1",
            "model": "model.npz",
            "model_fingerprint": "1" * 64,
            "registration_result_sha256": "2" * 64,
            "cell_result_fingerprint": "3" * 64,
            "metrics": {"auroc": 0.81, "average_precision": 0.75},
            "slides": slides,
        },
    )
    return root


def _study_protein_run(
    root: Path,
    *,
    prediction_protocol: str = "leave-one-mouse-out",
) -> Path:
    root.mkdir()
    for name in ("bundle.json", "table.npz", "audit.json", "model.npz"):
        (root / name).write_bytes(name.encode())
    slides = []
    for cohort, section in (("a", "001"), ("b", "009")):
        prediction = ProteinPredictions(
            target_id="yap",
            model_fingerprint="1" * 64,
            label_ids=np.asarray([1], dtype=np.uint32),
            section_ids=np.asarray([section]),
            expression_probability=np.asarray([np.nan], dtype=np.float32),
            relative_expression=np.asarray([0.5], dtype=np.float32),
            predicted_od_reference=np.asarray([0.4], dtype=np.float32),
            measured_od=np.asarray([np.nan], dtype=np.float32),
            uncertainty=np.asarray([0.1], dtype=np.float32),
            supported=np.asarray([True]),
            provenance={"feature_view": "native-hdab-neutral-spatial-uni2h-v2"},
        )
        relative = f"{cohort}-{section}.npz"
        prediction.save(root / relative)
        slides.append(
            {
                "cohort": cohort,
                "section": section,
                "predictions": relative,
                "prediction_fingerprint": prediction.fingerprint,
                "cells": 1,
                "measured_cells": 0,
                "evaluation_role": "external-transfer",
            }
        )
    binding = {
        "registration_result_sha256": "2" * 64,
        "cell_result_fingerprint": "3" * 64,
        "stain_result_fingerprint": "4" * 64,
        "semantic_result_fingerprint": "5" * 64,
    }
    write_protein_result(
        root,
        {
            "schema_version": 4,
            "model_id": "model",
            "model_label": "YAP · extra trees",
            "target_id": "yap",
            "assay_domain": "yap-v1",
            "architecture": "extra_trees",
            "model_version": "adaptive-v1",
            "prediction_protocol": prediction_protocol,
            "training_cohorts": ["a", "b"],
            "cohort_bindings": {"a": binding, "b": binding},
            "measurement_view": ("tissue-masked-adaptive-corrected-target-od-4um-v1"),
            "model": "bundle.json",
            "training_table": "table.npz",
            "measurement_audit": "audit.json",
            "model_artifacts": ["model.npz"],
            "model_fingerprint": "1" * 64,
            "slides": slides,
            "metrics": {"spearman": 0.6},
        },
    )
    return root


def test_protein_review_is_path_free_and_uses_three_expression_panes(
    tmp_path: Path,
) -> None:
    run = _protein_run(tmp_path / "run")
    index = build_protein_review({"mouse-a": run}, tmp_path / "review")
    manifest = json.loads((index.parent / "manifest.json").read_text())
    assert manifest["cohorts"][0]["target_id"] == "perk"
    assert (
        manifest["cohorts"][0]["prediction_resolution"] == "legacy_tile_pooled_baseline"
    )
    assert str(tmp_path) not in json.dumps(manifest)
    html = index.read_text()
    assert html.count("data-pane=") == 3
    assert "Cell-resolved protein prediction" in html
    assert "Histology QC" not in html
    assert "Cell boundaries" not in html
    assert "<label>Model" not in html
    script = (index.parent / "protein-review.js").read_text()
    assert "protein_predicted" in script
    assert "protein_dense" in script
    assert "protein_uncertainty" in script
    assert "nearest_target_section" in script
    assert "No YAP ground truth exists" not in script
    assert "not presentation-ready" not in script
    assert "reconcileCohort" in script
    assert "Loading protein maps" in script
    assert "Live tile API is unavailable" not in script
    assert "not presentation-approved" not in script
    assert '"viewport-change","pan","zoom"' not in script
    assert '"canvas-drag","canvas-drag-end"' in script
    assert '"canvas-click","canvas-key","canvas-key-press"' in script
    assert "viewportToImageCoordinates" in script
    assert "animationTime:0" in script
    assert "imageLoaderLimit:2" in script
    assert "setSharedZoomLimit" in script
    assert "primary.viewport.maxZoomLevel=null" in script
    assert "mine!==request" in script
    assert "getIndexOfItem(item)" in script
    assert 'addHandler("tile-loaded"' in script
    assert 'classList.add("loading")' in script
    assert "function fitFocus" in script
    assert "meta?.focus_bbox" in script
    assert "viewport.fitBounds" in script
    assert '<link rel="icon" href="data:,">' in html


def test_protein_review_selects_best_model_internally_and_deduplicates_runs(
    tmp_path: Path,
) -> None:
    run = _protein_run(tmp_path / "run", sections=("001", "002"))

    index = build_protein_review(
        {"mouse-a": {"alpha": run, "duplicate": run}},
        tmp_path / "review",
    )
    manifest = json.loads((index.parent / "manifest.json").read_text())
    cohort = manifest["cohorts"][0]

    assert cohort["default_model_id"] == "alpha"
    assert [model["id"] for model in cohort["models"]] == ["alpha"]
    assert cohort["models"][0]["api_scope"] == "model"
    assert str(tmp_path) not in json.dumps(manifest)
    assert '<select id="model-target">' in index.read_text()
    assert '<select id="architecture" hidden' in index.read_text()
    assert "<label>Architecture" not in index.read_text()
    assert '<select id="model" hidden' in index.read_text()
    assert "<label>Model" not in index.read_text()
    assert 'id="recommended"' not in index.read_text()
    assert '<div class="controls">' in index.read_text()
    assert '<div class="model-state">' not in index.read_text()
    css = (index.parent / "protein-review.css").read_text()
    assert "grid-template-rows:auto minmax(0,1fr)" in css
    assert ".model-state{display:flex" not in css
    assert ".recommended" not in css
    assert cohort["recommended_model_ids"] == {}
    script = (index.parent / "protein-review.js").read_text()
    assert "/protein/${encodeURIComponent(activeModel.id)}" in script
    assert 'name.startsWith("protein_")' in script
    assert "function targetLabel" in script
    assert "function chooseTarget" in script
    assert "function chooseArchitecture" in script
    assert "function bestModelId" in script
    assert "function applyBestModel" in script
    assert "function recommendedModelId" in script
    assert "function applyRecommendedModel" not in script
    assert "No approved model for section" not in script
    assert "Exploratory transfer · not presentation-approved" not in script
    assert "models.filter(row=>row.promoted||row.approved)" in script

    approve_protein_result(run, reviewer="test", accepted=True)
    approved_index = build_protein_review(
        {"mouse-a": {"alpha": run}},
        tmp_path / "approved-review",
    )
    approved = json.loads((approved_index.parent / "manifest.json").read_text())
    assert approved["cohorts"][0]["recommended_model_ids"]["perk"] == {
        "measured": "alpha",
        "unmeasured": "alpha",
    }


def test_protein_review_accepts_server_run_descriptors(tmp_path: Path) -> None:
    run = _protein_run(tmp_path / "run")

    index = build_protein_review(
        {
            "mouse-a": {
                "alpha": {
                    "run": run,
                    "stain": tmp_path / "stain",
                }
            }
        },
        tmp_path / "review",
    )

    model = json.loads((index.parent / "manifest.json").read_text())["cohorts"][0][
        "models"
    ][0]
    assert model["id"] == "alpha"


def test_study_review_filters_shared_result_to_each_mouse(tmp_path: Path) -> None:
    run = _study_protein_run(tmp_path / "study")

    index = build_protein_review(
        {"a": {"model": run}, "b": {"model": run}},
        tmp_path / "review",
    )
    cohorts = {
        row["id"]: row
        for row in json.loads((index.parent / "manifest.json").read_text())["cohorts"]
    }

    assert [row["section"] for row in cohorts["a"]["models"][0]["slides"]] == ["001"]
    assert [row["section"] for row in cohorts["b"]["models"][0]["slides"]] == ["009"]


def test_review_preserves_protocol_but_uses_presentation_labels(tmp_path: Path) -> None:
    run = _study_protein_run(
        tmp_path / "study",
        prediction_protocol="training-visible",
    )

    index = build_protein_review({"a": {"model": run}}, tmp_path / "review")
    model = json.loads((index.parent / "manifest.json").read_text())["cohorts"][0][
        "models"
    ][0]
    script = (index.parent / "protein-review.js").read_text()

    assert model["prediction_protocol"] == "training-visible"
    assert "Diagnostic fit · ground truth visible" not in script
    assert "Within-mouse transfer · target seen elsewhere" not in script
    assert "External transfer · no local ground truth" not in script
    assert "Measured-fit diagnostic" not in script
    assert "Held-out transfer" not in script
    assert "Held-out mouse" not in script
    assert "Cross-mouse reference" not in script
    assert "90th-percentile" in script
    assert "Single-cell expression" in script
    assert "Predicted and observed" in script
    assert "Absolute per-cell difference" in script
    assert "Training-visible upper bound" not in script
    assert 'ncad:"N-Cad"' in script
    assert 'cjun:"cJun"' in script
    assert "?.target_label" in script


@pytest.mark.browser
def test_protein_review_zoom_is_one_way_stable_and_mixed_resolution(
    tmp_path: Path,
) -> None:
    playwright = pytest.importorskip("playwright.sync_api")
    site = tmp_path / "site"
    run = _protein_run(tmp_path / "run", sections=("001", "002"))
    review = site / "protein-review"
    build_protein_review({"mouse-a": {"current": run}}, review)
    manifest_path = review / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    current_model = manifest["cohorts"][0]["models"][0]
    current_model["metrics"] = {
        **current_model["metrics"],
        "spearman": 0.8,
        "folds": [
            {
                "held_out": "mouse-a",
                "spearman": 0.42,
                "spearman_64um": 0.48,
                "mae": 0.31,
                "mean_bias_fraction": 0.12,
            }
        ],
    }
    stale_model = {
        **current_model,
        "id": "retired-model",
        "label": "Retired diagnostic",
        "target_id": "ki67",
        "architecture": "extra_trees",
        "version": "retired-v0",
    }
    manifest["cohorts"][0]["models"].insert(0, stale_model)
    manifest["cohorts"][0]["default_model_id"] = "retired-model"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    (review / "manifest-data.js").write_text(
        "globalThis.HISTOPIA_PROTEIN_REVIEW="
        + json.dumps(manifest, separators=(",", ":"))
        + ";\n"
    )
    server = create_viewer_server(
        site,
        bind="127.0.0.1",
        port=0,
        required_routes=("protein-review",),
    )
    calls: list[tuple[str, str, int, int, int]] = []
    digests = {
        "protein_predicted": "a" * 64,
        "protein_observed_target": "b" * 64,
        "protein_residual": "c" * 64,
        "raw": "d" * 64,
        "protein_nearest_observed_target": "e" * 64,
    }

    def pyramid(width: int, height: int) -> list[dict[str, int]]:
        maximum = int(np.ceil(np.log2(max(width, height))))
        return [
            {
                "width": max(1, int(np.ceil(width / (2 ** (maximum - level))))),
                "height": max(1, int(np.ceil(height / (2 ** (maximum - level))))),
            }
            for level in range(maximum + 1)
        ]

    def layer(
        name: str,
        width: int,
        height: int,
        *,
        analysis_mpp: float | None = None,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "digest": digests[name],
            "tile_size": 256,
            "width": width,
            "height": height,
            "levels": pyramid(width, height),
            "format": "jpg" if name == "raw" else "png",
            "display_max": 1.5,
        }
        if analysis_mpp is not None:
            payload["analysis_mpp"] = analysis_mpp
        return payload

    class FakeTiles:
        def catalog(self, cohort: str) -> dict[str, object]:
            assert cohort == "mouse-a"
            return {
                "schema_version": 1,
                "cohort": cohort,
                "sections": [{"section": "001"}, {"section": "002"}],
                "default_protein_model_id": "current",
                "protein_models": [
                    {
                        "id": "current",
                        "label": current_model["label"],
                        "target_id": current_model["target_id"],
                        "architecture": current_model["architecture"],
                        "training_cohorts": [],
                        "version": current_model["version"],
                        "model_fingerprint": current_model["model_fingerprint"],
                        "result_fingerprint": current_model["fingerprint"],
                        "measurement_view": current_model["measurement_view"],
                        "measurement_statistic": "mean",
                        "metrics": current_model["metrics"],
                        "approved": False,
                        "promoted": False,
                        "sections": ["001", "002"],
                    }
                ],
            }

        def metadata(
            self,
            cohort: str,
            section: str,
            *,
            protein_model: str | None = None,
        ) -> dict[str, object]:
            assert cohort == "mouse-a"
            if protein_model is not None:
                assert protein_model == "current"
            high = (4096, 3072)
            low = (512, 384)
            layers = {"protein_predicted": layer("protein_predicted", *high)}
            if section == "001":
                layers.update(
                    {
                        "protein_observed_target": layer(
                            "protein_observed_target", *low, analysis_mpp=4.0
                        ),
                        "protein_residual": layer("protein_residual", *high),
                    }
                )
                comparison = {
                    "target_id": "perk",
                    "predicted_layer": "protein_predicted",
                    "current_layer": "protein_observed_target",
                    "current_label": "PERK",
                    "current_scope": "calibrated_target_od_4um",
                    "comparison_layer": "protein_residual",
                    "comparison_label": "Absolute per-cell residual",
                    "ground_truth_available": True,
                    "nearest_target_section": None,
                }
            else:
                layers.update(
                    {
                        "raw": layer("raw", *high),
                        "protein_nearest_observed_target": layer(
                            "protein_nearest_observed_target",
                            *low,
                            analysis_mpp=4.0,
                        ),
                    }
                )
                comparison = {
                    "target_id": "perk",
                    "predicted_layer": "protein_predicted",
                    "current_layer": "raw",
                    "current_label": "H&E",
                    "current_scope": "native_histology",
                    "comparison_layer": "protein_nearest_observed_target",
                    "comparison_label": "Nearest observed PERK · section 001",
                    "ground_truth_available": False,
                    "nearest_target_section": "001",
                }
            return {
                "schema_version": 1,
                "cohort": cohort,
                "section": section,
                "slide": f"slide-{section}.ndpi",
                "label": "PERK" if section == "001" else "H&E",
                "reference": section == "001",
                "layers": layers,
                "protein_comparison": comparison,
                "protein_model_id": protein_model,
            }

        def render_tile(
            self,
            cohort: str,
            section: str,
            layer_name: str,
            digest: str,
            level: int,
            x: int,
            y: int,
            *,
            protein_model: str | None = None,
        ) -> tuple[bytes, str, str]:
            if layer_name.startswith("protein_"):
                assert protein_model == "current"
            else:
                assert protein_model is None
            metadata = self.metadata(
                cohort,
                section,
                protein_model=protein_model,
            )
            layer_metadata = dict(metadata["layers"])[layer_name]
            assert digest == layer_metadata["digest"]
            dimensions = layer_metadata["levels"][level]
            tile_size = int(layer_metadata["tile_size"])
            width = min(tile_size, int(dimensions["width"]) - x * tile_size)
            height = min(tile_size, int(dimensions["height"]) - y * tile_size)
            if width <= 0 or height <= 0:
                raise FileNotFoundError("invalid synthetic tile")
            calls.append((section, layer_name, level, x, y))
            output = io.BytesIO()
            if layer_name == "raw":
                Image.new("RGB", (width, height), (172, 116, 94)).save(output, "JPEG")
                media_type = "image/jpeg"
            else:
                colors = {
                    "protein_predicted": (151, 89, 42, 255),
                    "protein_observed_target": (119, 67, 31, 255),
                    "protein_residual": (177, 55, 48, 255),
                    "protein_nearest_observed_target": (109, 59, 27, 255),
                }
                Image.new("RGBA", (width, height), colors[layer_name]).save(
                    output, "PNG"
                )
                media_type = "image/png"
            return (
                output.getvalue(),
                media_type,
                f'"{digest}-{level}-{x}-{y}"',
            )

    server.wsi_tiles = FakeTiles()  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    api_errors: list[str] = []

    state_script = """() => Object.fromEntries(
      ['predicted', 'current', 'comparison'].map(name => {
        const viewer = OpenSeadragon.getViewer(
          document.querySelector(`[data-pane="${name}"]`));
        const item = viewer.world.getItemAt(0);
        const center = item.viewportToImageCoordinates(
          viewer.viewport.getCenter(true));
        return [name, {
          zoom: viewer.viewport.getZoom(true) / viewer.viewport.getHomeZoom(),
          x: center.x / item.source.dimensions.x,
          y: center.y / item.source.dimensions.y,
          max: viewer.viewport.getMaxZoom() / viewer.viewport.getHomeZoom(),
        }];
      }))"""

    def assert_synchronized(states: dict[str, dict[str, float]]) -> None:
        reference = states["predicted"]
        for state in states.values():
            for key in ("zoom", "x", "y", "max"):
                assert abs(state[key] - reference[key]) < 1e-6

    try:
        with playwright.sync_playwright() as runtime:
            browser_type = getattr(
                runtime, os.environ.get("HISTOPIA_BROWSER", "chromium")
            )
            browser = browser_type.launch(headless=True)
            page = browser.new_page(viewport={"width": 1600, "height": 900})
            page.on(
                "response",
                lambda response: (
                    api_errors.append(f"{response.status} {response.url}")
                    if response.status >= 400 and "/api/" in response.url
                    else None
                ),
            )
            page.on(
                "requestfailed",
                lambda request: (
                    api_errors.append(request.url) if "/api/" in request.url else None
                ),
            )
            page.goto(
                f"http://127.0.0.1:{server.server_port}/protein-review/",
                wait_until="domcontentloaded",
            )
            page.wait_for_function(
                """() => document.querySelectorAll('[data-pane] canvas').length === 3 &&
                  !document.querySelector('#message').classList.contains('visible')""",
                timeout=10_000,
            )
            assert page.locator("#model option").count() == 1
            assert page.locator("#model").input_value() == "current"
            assert "retired" not in page.locator("#model").inner_text().lower()
            assert page.locator("#metric-title").inner_text() == (
                "pERK prediction and measurement"
            )
            assert page.locator("#metrics").inner_text() == (
                "Single-cell expression · mean target OD"
            )
            natural_limit = page.evaluate(
                """() => {
                  const viewer = OpenSeadragon.getViewer(
                    document.querySelector('[data-pane="current"]'));
                  const shared = viewer.viewport.maxZoomLevel;
                  viewer.viewport.maxZoomLevel = null;
                  const natural = viewer.viewport.getMaxZoom() /
                    viewer.viewport.getHomeZoom();
                  viewer.viewport.maxZoomLevel = shared;
                  return natural;
                }"""
            )
            page.evaluate(
                """() => {
                  window.__proteinEvents = {predicted: 0, current: 0, comparison: 0};
                  window.__proteinZoom = [];
                  for (const name of Object.keys(window.__proteinEvents)) {
                    const viewer = OpenSeadragon.getViewer(
                      document.querySelector(`[data-pane="${name}"]`));
                    viewer.addHandler('viewport-change', () => {
                      window.__proteinEvents[name] += 1;
                      if (name === 'current') window.__proteinZoom.push(
                        viewer.viewport.getZoom(true) / viewer.viewport.getHomeZoom());
                    });
                  }
                }"""
            )
            current = page.locator('[data-pane="current"]').bounding_box()
            assert current is not None
            page.mouse.move(
                current["x"] + current["width"] * 0.61,
                current["y"] + current["height"] * 0.43,
            )
            for _ in range(10):
                page.mouse.wheel(0, -220)
                page.wait_for_timeout(25)
            page.wait_for_timeout(250)

            states = page.evaluate(state_script)
            assert_synchronized(states)
            assert states["current"]["zoom"] > natural_limit * 1.5
            zooms = page.evaluate("window.__proteinZoom")
            assert zooms
            assert all(
                next_zoom >= zoom - 1e-7
                for zoom, next_zoom in zip(zooms, zooms[1:], strict=False)
            )
            settled_events = page.evaluate("({...window.__proteinEvents})")
            settled_states = page.evaluate(state_script)
            page.wait_for_timeout(500)
            assert page.evaluate("window.__proteinEvents") == settled_events
            assert page.evaluate(state_script) == settled_states

            page.mouse.down()
            page.mouse.move(
                current["x"] + current["width"] * 0.43,
                current["y"] + current["height"] * 0.56,
                steps=8,
            )
            page.mouse.up()
            page.wait_for_timeout(150)
            dragged = page.evaluate(state_script)
            assert_synchronized(dragged)
            assert abs(dragged["current"]["x"] - states["current"]["x"]) > 1e-3

            page.select_option("#section", "002")
            page.wait_for_function(
                """() => document.querySelector('#current-title')
                  .textContent.startsWith('Observed H&E') &&
                  !document.querySelector('#message').classList.contains('visible')""",
                timeout=10_000,
            )
            switched = page.evaluate(state_script)
            assert_synchronized(switched)
            assert abs(switched["predicted"]["zoom"] - 1) < 1e-6

            comparison = page.locator('[data-pane="comparison"]').bounding_box()
            assert comparison is not None
            page.mouse.move(
                comparison["x"] + comparison["width"] * 0.55,
                comparison["y"] + comparison["height"] * 0.48,
            )
            for _ in range(7):
                page.mouse.wheel(0, -220)
                page.wait_for_timeout(25)
            page.wait_for_timeout(200)
            assert_synchronized(page.evaluate(state_script))

            page.set_viewport_size({"width": 390, "height": 844})
            page.wait_for_timeout(250)
            assert_synchronized(page.evaluate(state_script))
            for selector in (
                "#cohort",
                "#model-target",
                "#section",
            ):
                bounds = page.locator(selector).bounding_box()
                assert bounds is not None
                assert bounds["x"] >= 0
                assert bounds["x"] + bounds["width"] <= 390
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert api_errors == []
    assert {name for _, name, _, _, _ in calls} >= {
        "protein_predicted",
        "protein_observed_target",
        "protein_residual",
        "raw",
        "protein_nearest_observed_target",
    }


def test_protein_dzi_route_accepts_immutable_png_layer() -> None:
    path = "/api/wsi/mouse-a/001/protein_predicted/" + "a" * 64 + "/dzi/0/4/2_3.png"
    assert _WSI_DZI_TILE_RE.fullmatch(path)
    dense_path = "/api/wsi/mouse-a/001/protein_dense/" + "b" * 64 + "/dzi/0/4/2_3.png"
    assert _WSI_DZI_TILE_RE.fullmatch(dense_path)
    contrast_path = (
        "/api/wsi/mouse-a/001/protein_measured_contrast/"
        + "c" * 64
        + "/dzi/0/4/2_3.png"
    )
    assert _WSI_DZI_TILE_RE.fullmatch(contrast_path)
    observed_path = (
        "/api/wsi/mouse-a/001/protein_observed_target/" + "d" * 64 + "/dzi/0/4/2_3.png"
    )
    nearest_path = (
        "/api/wsi/mouse-a/002/protein_nearest_observed_target/"
        + "e" * 64
        + "/dzi/0/4/2_3.png"
    )
    assert _WSI_DZI_TILE_RE.fullmatch(observed_path)
    assert _WSI_DZI_TILE_RE.fullmatch(nearest_path)

    model_stain_path = (
        "/api/wsi/mouse-a/002/protein/perk-model/stain_adaptive_v3_map/"
        + "f" * 64
        + "/dzi/0/4/2_3.png"
    )
    assert _WSI_MODEL_DZI_TILE_RE.fullmatch(model_stain_path)


def test_measured_and_predicted_od_share_one_quantitative_scale(tmp_path: Path) -> None:
    run = _protein_run(tmp_path / "run")
    rows, _artifacts, metadata = _validated_protein_predictions(run, "2" * 64, "3" * 64)
    scales = metadata["display_max"]
    assert scales["measured_od"] == scales["predicted_od_reference"]
    assert (
        rows["001"]["section_display_max"]["measured_od"]
        == rows["001"]["section_display_max"]["predicted_od_reference"]
    )


def test_protein_registry_reads_only_the_sealed_index(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run = _protein_run(tmp_path / "run")

    def fail_eager_load(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("prediction arrays must be lazy")

    monkeypatch.setattr(ProteinPredictions, "load", fail_eager_load)
    rows, artifacts, metadata = _validated_protein_predictions(run, "2" * 64, "3" * 64)

    assert rows["001"]["has_measured"] is True
    assert rows["001"]["has_probability"] is False
    assert artifacts["001.npz"]
    assert metadata["target_id"] == "perk"


def test_protein_prediction_artifact_is_verified_when_first_opened(
    tmp_path: Path,
) -> None:
    run = _protein_run(tmp_path / "run")
    rows, artifacts, _metadata = _validated_protein_predictions(run, "2" * 64, "3" * 64)
    layer = WsiLayer(
        name="protein_predicted",
        path=run / "001.npz",
        digest="a" * 64,
        levels=(WsiLevel(1, 1, 0),),
        tile_size=256,
        microns_per_pixel=0.5,
        protein_target="perk",
        protein_model_fingerprint="1" * 64,
        prediction_section="001",
        prediction_artifact_digest=artifacts["001.npz"],
        prediction_fingerprint=rows["001"]["prediction_fingerprint"],
    )
    service = WsiTileService({})
    assert service._load_protein_predictions(layer).target_id == "perk"

    with layer.path.open("ab") as stream:
        stream.write(b"stale")
    with pytest.raises(ValueError, match="missing or stale"):
        WsiTileService({})._load_protein_predictions(layer)


def test_protein_tile_maps_cell_ids_without_seams(monkeypatch, tmp_path: Path) -> None:
    predictions = ProteinPredictions(
        target_id="perk",
        model_fingerprint="1" * 64,
        label_ids=np.asarray([1, 2], dtype=np.uint32),
        section_ids=np.asarray(["001", "001"]),
        expression_probability=np.asarray([0.2, 0.9], dtype=np.float32),
        relative_expression=np.asarray([0.1, 0.9], dtype=np.float32),
        predicted_od_reference=np.asarray([0.2, 0.8], dtype=np.float32),
        measured_od=np.asarray([0.3, 0.7], dtype=np.float32),
        uncertainty=np.asarray([0.1, 0.2], dtype=np.float32),
        supported=np.asarray([True, True]),
        provenance={},
    )
    labels = np.asarray([[0, 1, 1], [2, 2, 1]], dtype=np.uint32)
    monkeypatch.setattr(
        "histopia.visualization._wsi_tiles._read_protein_label_tile",
        lambda *_args: labels,
    )
    layer = WsiLayer(
        name="protein_predicted",
        path=tmp_path / "prediction.npz",
        digest="a" * 64,
        levels=(WsiLevel(3, 2, 0),),
        tile_size=512,
        microns_per_pixel=0.25,
        map_kind="relative_expression",
        label_path=tmp_path / "labels.tiff",
        protein_target="perk",
        protein_model_fingerprint="1" * 64,
        prediction_section="001",
    )
    payload = _render_protein_layer_tile(layer, 0, 0, 0, predictions)
    rgba = np.asarray(Image.open(io.BytesIO(payload)))
    assert rgba.shape == (2, 3, 4)
    assert rgba[0, 0, 3] == 0
    np.testing.assert_array_equal(rgba[0, 1], rgba[0, 2])
    assert rgba[1, 0, 3] == rgba[0, 1, 3] == 255
    assert rgba[1, 0, :3].sum() < rgba[0, 1, :3].sum()


def test_expression_and_diagnostic_layers_have_distinct_palettes() -> None:
    values = np.asarray([[0.8]], dtype=np.float32)
    supported = np.asarray([[True]])
    expression = _protein_heatmap_rgba(values, supported, kind="relative_expression")
    uncertainty = _protein_heatmap_rgba(values, supported, kind="uncertainty")
    residual = _protein_heatmap_rgba(values, supported, kind="residual")
    assert expression[0, 0, 0] > expression[0, 0, 2]
    assert uncertainty[0, 0, 2] > uncertainty[0, 0, 0]
    assert residual[0, 0, 0] > residual[0, 0, 1]


def test_dense_protein_layer_has_explicit_four_micron_geometry(
    tmp_path: Path,
) -> None:
    cells = WsiLayer(
        name="cells",
        path=tmp_path / "labels.tiff",
        digest="a" * 64,
        levels=(WsiLevel(1000, 500, 0),),
        tile_size=256,
        microns_per_pixel=0.5,
        source_shape=(1000, 500),
    )
    dense = _protein_layer(
        "protein_dense",
        tmp_path / "prediction.npz",
        "b" * 64,
        cells.path,
        cells.digest,
        cells,
        value_kind="relative_expression",
        section="001",
        target_id="yap",
        model_fingerprint="c" * 64,
        analysis_mpp=4.0,
    )

    assert dense.analysis_mpp == 4.0
    assert dense.microns_per_pixel == 4.0
    assert (dense.levels[-1].width, dense.levels[-1].height) == (125, 63)


def test_protein_overview_is_visible_without_reading_native_labels(
    monkeypatch, tmp_path: Path
) -> None:
    mask_path = tmp_path / "mask.png"
    Image.fromarray(np.full((8, 8), 255, dtype=np.uint8)).save(mask_path)
    predictions = ProteinPredictions(
        target_id="yap",
        model_fingerprint="1" * 64,
        label_ids=np.asarray([1], dtype=np.uint32),
        section_ids=np.asarray(["001"]),
        expression_probability=np.asarray([0.5], dtype=np.float32),
        relative_expression=np.asarray([0.7], dtype=np.float32),
        predicted_od_reference=np.asarray([0.8], dtype=np.float32),
        measured_od=np.asarray([np.nan], dtype=np.float32),
        uncertainty=np.asarray([0.1], dtype=np.float32),
        supported=np.asarray([True]),
        provenance={},
    )
    layer = WsiLayer(
        name="protein_predicted",
        path=tmp_path / "prediction.npz",
        digest="a" * 64,
        levels=(WsiLevel(64, 32, 0),),
        tile_size=64,
        microns_per_pixel=4.0,
        map_kind="relative_expression",
        label_path=tmp_path / "labels.tiff",
        mask_path=mask_path,
    )
    monkeypatch.setattr(
        "histopia.visualization._wsi_tiles._read_protein_label_tile",
        lambda *_args: (_ for _ in ()).throw(AssertionError("native read")),
    )

    rgba = np.asarray(
        Image.open(io.BytesIO(_render_protein_layer_tile(layer, 0, 0, 0, predictions)))
    )
    assert rgba.shape == (32, 64, 4)
    assert np.all(rgba[..., 3] > 0)


def test_spatial_protein_overview_preserves_variation_without_tile_seams(
    tmp_path: Path,
) -> None:
    layer = WsiLayer(
        name="protein_predicted",
        path=tmp_path / "prediction.npz",
        digest="a" * 64,
        levels=(WsiLevel(4, 2, 0),),
        tile_size=2,
        microns_per_pixel=0.5,
    )
    overview = _ProteinOverview(
        values=np.asarray([[0.1, 0.2, 0.7, 0.9], [0.2, 0.3, 0.8, 1.0]]),
        supported=np.ones((2, 4), dtype=bool),
    )

    left_values, left_support = _read_spatial_protein_overview(layer, 0, 0, 0, overview)
    right_values, right_support = _read_spatial_protein_overview(
        layer, 0, 1, 0, overview
    )

    np.testing.assert_allclose(
        np.column_stack([left_values, right_values]), overview.values
    )
    np.testing.assert_array_equal(
        np.column_stack([left_support, right_support]), overview.supported
    )
    assert float(left_values.mean()) < float(right_values.mean())
