from __future__ import annotations

import hashlib
import http.server
import json
import os
import threading
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from histopia.cells._result import write_cell_result
from histopia.protein import (
    ProteinPredictions,
    approve_protein_result,
    write_protein_result,
)
from histopia.protein._multiscale import (
    FEATURE_SCHEMA_ID,
    SHAPE_FEATURE_NAMES,
    TEXTURE_FEATURE_NAMES,
    CellFeatureSet,
)
from histopia.visualization import build_cellular_protein_atlas
from histopia.visualization._cellular_protein_atlas import (
    _boundary_vertices,
    _display_transforms,
    _extract_boundary_profiles,
    _histogram_percentile,
    _overview_allocations,
)


def _inputs(tmp_path: Path) -> tuple[dict[str, Path], ...]:
    cohort = "mouse-a"
    slide_name = "slide-001.ndpi"
    label_image = np.zeros((64, 80), dtype=np.uint32)
    label_image[5:15, 5:15] = 1
    label_image[20:34, 18:26] = 2
    label_image[38:55, 40:61] = 3
    label_image[44:49, 54:68] = 3
    local_centers = np.asarray(
        [
            [
                np.nonzero(label_image == label)[1].mean(),
                np.nonzero(label_image == label)[0].mean(),
            ]
            for label in (1, 2, 3)
        ],
        dtype=np.float64,
    )
    native_centers = local_centers + np.asarray([10.0, 20.0])
    thumbnail_to_native = [[1, 0, 10], [0, 1, 20], [0, 0, 1]]
    transform = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    registration = tmp_path / "registration"
    registration.mkdir()
    (registration / "registration_result.json").write_text(
        json.dumps(
            {
                "reference_slide": slide_name,
                "slides": [
                    {
                        "path": slide_name,
                        "is_reference": True,
                        "geometry": {
                            "content_bbox_xywh": [10, 20, 80, 64],
                            "thumbnail_to_native": thumbnail_to_native,
                            "mpp_xy": [1.0, 1.0],
                        },
                        "transform": {"matrix": transform},
                    }
                ],
            }
        )
    )
    registration_sha = hashlib.sha256(
        (registration / "registration_result.json").read_bytes()
    ).hexdigest()

    cells = tmp_path / "cells"
    (cells / "labels").mkdir(parents=True)
    (cells / "qc").mkdir()
    (cells / "preflight.json").write_text("{}\n")
    Image.fromarray(label_image).save(cells / "labels/001.tiff")
    (cells / "qc/001.json").write_text("{}\n")
    write_cell_result(
        cells,
        {
            "schema_version": 1,
            "registration_result_sha256": registration_sha,
            "preflight": "preflight.json",
            "slides": [
                {
                    "section": "001",
                    "slide": slide_name,
                    "content_bbox_xywh": [10, 20, 80, 64],
                    "transform_sha256": hashlib.sha256(
                        json.dumps(
                            transform, sort_keys=True, separators=(",", ":")
                        ).encode()
                    ).hexdigest(),
                    "labels": "labels/001.tiff",
                    "qc": "qc/001.json",
                    "cells": 3,
                }
            ],
        },
    )
    cell_result = json.loads((cells / "cell_result.json").read_text())

    geometry_root = tmp_path / "geometry"
    geometry = geometry_root / cohort / "multiscale-v3"
    geometry.mkdir(parents=True)
    phenotype = np.zeros((3, 19), dtype=np.float32)
    phenotype[:, 0] = np.log1p([50.0, 100.0, 200.0])
    phenotype[:, 5] = [0.0, 0.5, 0.9]
    phenotype[:, 7] = np.sin([0.0, 0.4, -0.7])
    phenotype[:, 8] = np.cos([0.0, 0.4, -0.7])
    feature_set = CellFeatureSet(
        slide_id=f"{cohort}-001",
        label_ids=np.asarray([1, 2, 3], dtype=np.uint32),
        native_xy=native_centers,
        reference_um_xyz=np.column_stack(
            [native_centers, np.full(3, 5.0, dtype=np.float64)]
        ),
        morphology=np.zeros((3, 2), dtype=np.float32),
        phenotype=phenotype,
        neighborhood=np.zeros((3, 1), dtype=np.float32),
        position=np.zeros((3, 1), dtype=np.float32),
        supported=np.asarray([True, True, False]),
        feature_names={
            "morphology": ("uni2h_0000", "uni2h_0001"),
            "phenotype": SHAPE_FEATURE_NAMES + TEXTURE_FEATURE_NAMES,
            "neighborhood": ("neighbor",),
            "position": ("position",),
        },
        provenance={
            "feature_schema_id": FEATURE_SCHEMA_ID,
            "registration_result_sha256": registration_sha,
            "cell_result_fingerprint": cell_result["fingerprint"],
            "source_identity": "slide-001",
            "token_fingerprint": "a" * 64,
        },
    )
    feature_set.save(geometry / "001.npz")

    transfer = _protein_run(
        tmp_path / "transfer",
        cohort=cohort,
        registration_sha=registration_sha,
        cell_fingerprint=cell_result["fingerprint"],
        prediction_protocol="leave-one-mouse-out",
    )
    diagnostic = _protein_run(
        tmp_path / "diagnostic",
        cohort=cohort,
        registration_sha=registration_sha,
        cell_fingerprint=cell_result["fingerprint"],
        prediction_protocol="training-visible",
    )
    models = {
        cohort: {
            "yap-transfer": transfer,
            "yap-diagnostic": diagnostic,
        }
    }
    return (
        {cohort: registration},
        {cohort: cells},
        models,
        {cohort: geometry_root},
    )


def _protein_run(
    root: Path,
    *,
    cohort: str,
    registration_sha: str,
    cell_fingerprint: str,
    prediction_protocol: str,
    target_id: str = "yap",
    promoted: bool = True,
) -> Path:
    root.mkdir()
    for name in ("model.json", "table.npz", "audit.json", "weights.npz"):
        (root / name).write_bytes(name.encode())
    prediction = ProteinPredictions(
        target_id=target_id,
        model_fingerprint="1" * 64,
        label_ids=np.asarray([1, 2, 3], dtype=np.uint32),
        section_ids=np.asarray(["001", "001", "001"]),
        expression_probability=np.asarray([0.2, 0.8, np.nan], dtype=np.float32),
        relative_expression=np.asarray([0.1, 0.8, 0.0], dtype=np.float32),
        predicted_od_reference=np.asarray([0.4, 1.0, 0.4], dtype=np.float32),
        measured_od=np.asarray([0.2, 0.7, np.nan], dtype=np.float32),
        uncertainty=np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
        supported=np.asarray([True, True, False]),
        provenance={
            "feature_view": FEATURE_SCHEMA_ID,
            "evaluation_role": (
                "training-visible-fit"
                if prediction_protocol == "training-visible"
                else "leave-one-mouse-out"
            ),
        },
    )
    prediction.save(root / "001.npz")
    protocol_label = (
        "diagnostic" if prediction_protocol == "training-visible" else "transfer"
    )
    model_id = f"{target_id}-{protocol_label}"
    binding = {
        "registration_result_sha256": registration_sha,
        "cell_result_fingerprint": cell_fingerprint,
        "stain_result_fingerprint": "3" * 64,
        "semantic_result_fingerprint": "4" * 64,
    }
    write_protein_result(
        root,
        {
            "schema_version": 4,
            "model_id": model_id,
            "model_label": target_id.upper(),
            "target_id": target_id,
            "target_label": target_id.upper(),
            "architecture": "test-model",
            "model_version": "v1",
            "prediction_protocol": prediction_protocol,
            "training_cohorts": [cohort],
            "cohort_bindings": {cohort: binding},
            "feature_schema_id": FEATURE_SCHEMA_ID,
            "measurement_view": ("tissue-masked-adaptive-corrected-target-od-4um-v1"),
            "measurement_statistic": "mean",
            "assay_domain": "yap-v1",
            "model": "model.json",
            "training_table": "table.npz",
            "measurement_audit": "audit.json",
            "model_artifacts": ["weights.npz"],
            "model_fingerprint": "1" * 64,
            "slides": [
                {
                    "cohort": cohort,
                    "section": "001",
                    "predictions": "001.npz",
                    "prediction_fingerprint": prediction.fingerprint,
                    "cells": 3,
                    "measured_cells": 2,
                    "evaluation_role": (
                        "training-visible-fit"
                        if prediction_protocol == "training-visible"
                        else "leave-one-mouse-out"
                    ),
                }
            ],
            "metrics": {
                "spearman": 0.8,
                "spearman_64um": 0.85,
                "folds": [{"held_out": cohort}],
            },
            "status": (
                "candidate"
                if prediction_protocol == "training-visible" or not promoted
                else "promoted"
            ),
            "candidate_promotion": {
                "accepted": promoted,
                "reasons": [] if promoted else ["OD calibration gate was not accepted"],
            },
            "target_global_display_max_od": 1.0,
        },
    )
    if prediction_protocol == "leave-one-mouse-out" and promoted:
        approve_protein_result(root, reviewer="test", accepted=True)
    return root


def _with_balanced_models(
    tmp_path: Path,
    registration: dict[str, Path],
    cells: dict[str, Path],
    models: dict[str, dict[str, Path]],
) -> None:
    cohort = "mouse-a"
    registration_sha = hashlib.sha256(
        (registration[cohort] / "registration_result.json").read_bytes()
    ).hexdigest()
    cell_result = json.loads((cells[cohort] / "cell_result.json").read_text())
    for target in ("ck19", "ecad", "sma"):
        run = _protein_run(
            tmp_path / f"{target}-transfer",
            cohort=cohort,
            registration_sha=registration_sha,
            cell_fingerprint=cell_result["fingerprint"],
            prediction_protocol="leave-one-mouse-out",
            target_id=target,
            promoted=False,
        )
        models[cohort][f"{target}-transfer"] = run


def test_cellular_protein_atlas_is_static_path_free_and_cell_resolved(
    tmp_path: Path,
) -> None:
    registration, cells, models, geometry = _inputs(tmp_path)

    index = build_cellular_protein_atlas(
        registration,
        cells,
        models,
        geometry,
        tmp_path / "atlas",
        max_overview_cells=2,
    )

    manifest = json.loads((index.parent / "manifest.json").read_text())
    cohort = manifest["cohorts"][0]
    target = cohort["targets"][0]
    section = cohort["sections"][0]
    assert manifest["schema_version"] == 3
    assert manifest["binary_format"] == "HCPA1"
    assert manifest["max_channels"] == 10
    assert manifest["default"]["layer"] == "protein"
    assert manifest["default"]["depth_mode"] == "adaptive"
    assert manifest["default"]["z_scale"] == 2
    assert manifest["default"]["extra_cutoff"] == 0.25
    assert manifest["default"]["targets"] == ["yap"]
    assert len(manifest["cell_identity_palette"]) >= 12
    assert cohort["cell_count"] == 3
    assert cohort["overview_cell_count"] == 2
    assert cohort["default_section"] == "001"
    assert cohort["cell_identity"]["boundary_derived"] is True
    assert cohort["cell_identity"]["detail_boundary_vertices"] == 16
    assert cohort["cell_identity"]["overview_boundary_vertices"] == 8
    assert "cell-size-derived display spacing" in cohort["cell_identity"]["scope"]
    assert (
        cohort["cell_identity"]["morphology_aware_z"]["method"]
        == "boundary-size-local-packing-display-v1"
    )
    depth_model = cohort["cell_identity"]["morphology_aware_z"]
    assert depth_model["median_cell_diameter_um"] > 0
    assert depth_model["physical_section_spacing_um"] == 5.0
    assert depth_model["visual_section_spacing_um"] > 5.0
    assert depth_model["visual_z_scale"] == pytest.approx(
        depth_model["visual_section_spacing_um"] / 5.0
    )
    assert section["median_boundary_diameter_um"] > 0
    assert set(cohort["fallback_previews"]) == {"cells", "protein"}
    assert target["model_id"] == "yap-transfer"
    assert target["status"] == "promoted"
    assert target["display_transform"]["method"].endswith("percentile-v1")
    assert target["display_transform"]["scope"].startswith("display only")
    assert "architecture" not in target
    assert section["targets"]["yap"]["observed"]
    assert str(tmp_path) not in json.dumps(manifest)
    assert not any(
        "/api/" in path.read_text(errors="ignore") for path in index.parent.glob("*.js")
    )
    assert (index.parent / "vendor/three.module.min.js").is_file()
    assert (index.parent / ".nojekyll").is_file()

    geometry_path = index.parent / section["geometry"]["asset"]
    payload = geometry_path.read_bytes()
    assert payload[:5] == b"HCPA1"
    assert payload[5] == 2
    assert int.from_bytes(payload[8:12], "little") == 3
    assert int.from_bytes(payload[56:58], "little") == 16
    assert len(payload) == 160 + 3 * 34
    assert section["geometry"]["representation"].endswith("boundary-v1")
    assert section["geometry"]["quality_control"]["median_mask_vector_iou"] >= 0.82
    assert section["geometry"]["digest"] == hashlib.sha256(payload).hexdigest()

    prediction_path = index.parent / section["targets"]["yap"]["predicted"]["asset"]
    values = prediction_path.read_bytes()
    assert values[:5] == b"HCPA1"
    assert values[5] == 3
    assert list(values[32:]) == [103, 255, 0]
    observed_path = index.parent / section["targets"]["yap"]["observed"]["asset"]
    assert list(observed_path.read_bytes()[32:]) == [52, 179, 0]

    html = index.read_text()
    script = (index.parent / "protein-atlas.js").read_text()
    assert "Cell identity" in html
    assert "Morphology-aware" in html
    assert "Scientific scope" not in html
    assert "Orthogonal" in html
    assert "Section" in html
    assert "Show exploratory targets" not in html
    assert "Pattern focus cutoff" in html
    assert 'id="threshold" type="range" min="0" max="50" value="25"' in html
    assert 'id="envelope-opacity" type="range" min="0" max="40" value="14"' in html
    assert 'data-layer="protein" class="active"' in html
    assert '<button data-z="adaptive" class="active">Morphology-aware</button>' in html
    assert '<button data-z="exploded">Exploded</button>' in html
    assert "Z source:" not in script
    assert "Display spacing from cell size" in script
    assert "Not measured tissue depth" in script
    assert "sharedModel?.visual_section_spacing_um" in script
    assert "sharedModel?.visual_z_scale" in script
    assert any(
        preset["label"].startswith("Mechanotransduction & epithelial–stromal state")
        for preset in manifest["presets"]
    )
    assert all(
        word not in preset["label"].lower()
        for preset in manifest["presets"]
        for word in ("validated", "exploratory", "failed")
    )
    assert "Full cell set near selected section" in html
    assert "new Worker" in script
    assert "new THREE.LineLoop" in script
    assert "accepted boundary vector" in script
    assert 'patternNormalized: expression !== "residual"' in script
    assert 'query.set("cutoff", el("threshold").value)' in script
    assert "cellShape" not in script
    assert "parseLegacyGeometry" in script
    assert "kind === 2" in script
    assert "projectionStates" in script
    assert "deriveCellMorphology" in script
    assert "cellDepth" in script
    assert "cellZPhase" in script
    assert "vertices.push(sector, 1, -0.34)" in script
    assert "vertices.push(sector, 1, 0.34)" in script
    assert "lower, lowerNext, upperNext" in script
    assert "adaptiveSectionSpacingUm" in script
    assert "bodyDepthScale" not in script
    assert "webglcontextlost" in script
    assert "Compatibility mode intentionally stays asset-only" in script
    assert "training-visible" not in json.dumps(manifest)


def test_cellular_protein_atlas_rejects_stale_geometry_binding(
    tmp_path: Path,
) -> None:
    registration, cells, models, geometry = _inputs(tmp_path)
    path = geometry["mouse-a"] / "mouse-a/multiscale-v3/001.npz"
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files if name != "metadata_json"}
        metadata = json.loads(str(data["metadata_json"]))
    metadata["provenance"]["cell_result_fingerprint"] = "f" * 64
    np.savez_compressed(
        path,
        metadata_json=np.asarray(json.dumps(metadata)),
        **arrays,
    )

    with pytest.raises(ValueError, match="cell binding differs"):
        build_cellular_protein_atlas(
            registration,
            cells,
            models,
            geometry,
            tmp_path / "atlas",
        )


def test_cellular_protein_atlas_budget_failure_preserves_existing_output(
    tmp_path: Path,
) -> None:
    registration, cells, models, geometry = _inputs(tmp_path)
    output = tmp_path / "atlas"
    output.mkdir()
    (output / "keep.txt").write_text("stable")

    with pytest.raises(ValueError, match="exceeds max_bytes"):
        build_cellular_protein_atlas(
            registration,
            cells,
            models,
            geometry,
            output,
            max_bytes=128,
        )

    assert (output / "keep.txt").read_text() == "stable"


def test_cellular_protein_atlas_overview_retains_every_section() -> None:
    with pytest.raises(ValueError, match="at least one cell per section"):
        _overview_allocations(np.asarray([3, 4]), 1)


def test_cellular_protein_atlas_extracts_irregular_touching_boundaries(
    tmp_path: Path,
) -> None:
    image = np.zeros((52, 66), dtype=np.uint32)
    image[5:24, 5:13] = 1
    image[14:22, 13:22] = 1
    image[10:27, 22:31] = 2  # Touches cell 1 without merging labels.
    image[32:45, 35:58] = 3
    image[36:41, 47:63] = 3
    path = tmp_path / "irregular-labels.tiff"
    Image.fromarray(image).save(path)
    labels = np.asarray([1, 2, 3], dtype=np.uint32)
    centers = np.asarray(
        [
            [np.nonzero(image == label)[1].mean(), np.nonzero(image == label)[0].mean()]
            for label in labels
        ],
        dtype=np.float64,
    )

    radii, offsets, qc = _extract_boundary_profiles(
        path,
        labels,
        centers,
        bbox=(0, 0, image.shape[1], image.shape[0]),
        matrix=np.eye(3),
        reference_xy=centers,
        stripe_height=7,
    )

    assert radii.shape == (3, 16)
    assert offsets.shape == (3, 16)
    assert np.all(radii > 0)
    assert np.all(offsets <= 15)
    assert _boundary_vertices(radii, offsets, 16).shape == (3, 16, 2)
    assert qc["median_mask_vector_iou"] >= 0.82
    assert qc["p05_mask_vector_iou"] >= 0.60
    assert qc["invalid_or_self_intersecting"] == 0


def test_cellular_protein_atlas_reuses_only_validated_boundary_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registration, cells, models, geometry = _inputs(tmp_path)
    output = tmp_path / "atlas"
    build_cellular_protein_atlas(
        registration, cells, models, geometry, output, workers=2
    )
    first = json.loads((output / "manifest.json").read_text())

    def fail_extraction(*_args, **_kwargs):
        raise AssertionError("validated boundary geometry should have been reused")

    monkeypatch.setattr(
        "histopia.visualization._cellular_protein_atlas._extract_boundary_profiles",
        fail_extraction,
    )
    build_cellular_protein_atlas(registration, cells, models, geometry, output)
    second = json.loads((output / "manifest.json").read_text())
    first_section = first["cohorts"][0]["sections"][0]
    second_section = second["cohorts"][0]["sections"][0]
    assert second_section["geometry"] == first_section["geometry"]
    assert second_section["overview_geometry"] == first_section["overview_geometry"]


def test_cellular_protein_atlas_rejects_mutated_label_artifact(
    tmp_path: Path,
) -> None:
    registration, cells, models, geometry = _inputs(tmp_path)
    (cells["mouse-a"] / "labels/001.tiff").write_bytes(b"stale")

    with pytest.raises(ValueError, match="artifact digest mismatch"):
        build_cellular_protein_atlas(
            registration,
            cells,
            models,
            geometry,
            tmp_path / "atlas",
        )


def test_cellular_protein_atlas_uses_balanced_default_and_display_only_scaling(
    tmp_path: Path,
) -> None:
    registration, cells, models, geometry = _inputs(tmp_path)
    _with_balanced_models(tmp_path, registration, cells, models)

    index = build_cellular_protein_atlas(
        registration,
        cells,
        models,
        geometry,
        tmp_path / "atlas",
        max_overview_cells=3,
    )

    manifest = json.loads((index.parent / "manifest.json").read_text())
    cohort = manifest["cohorts"][0]
    assert manifest["default"]["targets"] == ["yap", "ck19", "ecad", "sma"]
    assert cohort["default_targets"] == ["yap", "ck19", "ecad", "sma"]
    targets = {row["id"]: row for row in cohort["targets"]}
    assert targets["yap"]["color"] == "#f14ed3"
    assert targets["ck19"]["color"] == "#ffd43b"
    assert targets["ecad"]["color"] == "#25c7f5"
    assert targets["sma"]["color"] == "#76e06a"
    assert targets["yap"]["status"] == "promoted"
    assert all(targets[target]["exploratory"] for target in ("ck19", "ecad", "sma"))
    for row in targets.values():
        transform = row["display_transform"]
        assert transform["lower_percentile"] == 65.0
        assert transform["upper_percentile"] == 99.5
        assert transform["gamma"] == 0.72
        assert 0 <= transform["lower_od"] < transform["upper_od"]


def test_cellular_protein_atlas_display_percentiles_are_bounded_and_deterministic() -> (
    None
):
    histogram = np.zeros(256, dtype=np.int64)
    histogram[1] = 10
    histogram[64] = 30
    histogram[192] = 10
    assert _histogram_percentile(histogram, 0) == 1
    assert _histogram_percentile(histogram, 65) == 64
    assert _histogram_percentile(histogram, 99.5) == 192
    transform = _display_transforms({"target": histogram})["target"]
    assert transform["supported_cells"] == 50
    assert 0 <= transform["lower_fraction"] < transform["upper_fraction"] <= 1

    constant = np.zeros(256, dtype=np.int64)
    constant[255] = 5
    constant_transform = _display_transforms({"target": constant})["target"]
    assert constant_transform["lower_fraction"] < constant_transform["upper_fraction"]


def test_cellular_protein_atlas_validates_default_layer_and_targets(
    tmp_path: Path,
) -> None:
    registration, cells, models, geometry = _inputs(tmp_path)
    with pytest.raises(ValueError, match="layer must be cells or protein"):
        build_cellular_protein_atlas(
            registration,
            cells,
            models,
            geometry,
            tmp_path / "atlas",
            default_layer="volume",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="unavailable"):
        build_cellular_protein_atlas(
            registration,
            cells,
            models,
            geometry,
            tmp_path / "atlas",
            default_targets=("yap", "ck19"),
        )


@pytest.mark.browser
def test_cellular_protein_atlas_loads_without_clicks_and_is_responsive(
    tmp_path: Path,
) -> None:
    playwright = pytest.importorskip("playwright.sync_api")
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    registration, cells, models, geometry = _inputs(inputs)
    _with_balanced_models(inputs, registration, cells, models)
    build_cellular_protein_atlas(
        registration,
        cells,
        models,
        geometry,
        tmp_path / "atlas",
        max_overview_cells=3,
    )

    def handler(*args, **kwargs):
        return http.server.SimpleHTTPRequestHandler(
            *args, directory=str(tmp_path), **kwargs
        )

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    errors: list[str] = []
    try:
        with playwright.sync_playwright() as runtime:
            browser = getattr(
                runtime, os.environ.get("HISTOPIA_BROWSER", "chromium")
            ).launch(headless=True)
            page = browser.new_page(viewport={"width": 1920, "height": 1080})
            page.on(
                "console",
                lambda message: (
                    errors.append(message.text) if message.type == "error" else None
                ),
            )
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.on("requestfailed", lambda request: errors.append(request.url))
            page.goto(
                f"http://127.0.0.1:{server.server_port}/atlas/index.html",
                wait_until="networkidle",
            )
            page.wait_for_function("() => document.querySelector('#loading').hidden")
            assert page.locator("#viewport").get_attribute("data-renderer") in {
                "webgl",
                "fallback",
            }
            assert page.locator("#cohort").input_value() == "mouse-a"
            assert page.locator(".app").get_attribute("data-layer") == "protein"
            assert page.locator("#threshold").input_value() == "25"
            assert page.locator("#threshold-value").inner_text() == "25%"
            assert page.locator("#target-list input:checked").count() == 4
            scene_label = page.locator("#scene-badge").inner_text()
            assert (
                "Predicted" in scene_label or "Static prediction preview" in scene_label
            )
            assert "3 cells" in page.locator("#provenance").inner_text()
            if page.locator("#viewport").get_attribute("data-renderer") == "webgl":
                assert page.locator("#viewport canvas").count() == 1
                assert (
                    page.locator("#viewport").get_attribute("data-depth-mode")
                    == "adaptive"
                )
                adaptive_scale = float(
                    page.locator("#viewport").get_attribute("data-z-scale") or 0
                )
                assert 1.1 <= adaptive_scale <= 3.2
                depth_p10 = float(
                    page.locator("#viewport").get_attribute("data-cell-depth-p10-um")
                    or 0
                )
                depth_p90 = float(
                    page.locator("#viewport").get_attribute("data-cell-depth-p90-um")
                    or 0
                )
                phase_rms = float(
                    page.locator("#viewport").get_attribute("data-z-phase-rms") or 0
                )
                assert 0 < depth_p10 < depth_p90
                assert phase_rms > 0.1
                assert (
                    "Morphology-aware 3D" in page.locator("#depth-badge").inner_text()
                )
                assert (
                    "Not measured tissue depth"
                    in page.locator("#z-warning").inner_text()
                )
                page.locator('[data-z="physical"]').click()
                assert (
                    page.locator("#viewport").get_attribute("data-depth-mode")
                    == "physical"
                )
                assert (
                    page.locator("#viewport").get_attribute("data-z-scale") == "1.000"
                )
                assert "Assumed" in page.locator("#depth-badge").inner_text()
                page.locator('[data-z="adaptive"]').click()
                assert page.locator(".app").get_attribute("data-layer") == "protein"
                assert (
                    "YAP / CK19 / E-Cad / SMA"
                    in page.locator("#scene-badge").inner_text()
                )
                page.goto(
                    "http://127.0.0.1:"
                    f"{server.server_port}/atlas/index.html"
                    "?mouse=mouse-a&section=001&markers=yap"
                    "&mode=section&expression=residual&depth=adaptive"
                    "&cutoff=0&exploratory=1",
                    wait_until="networkidle",
                )
                page.wait_for_function(
                    "() => document.querySelector('#loading').hidden"
                )
                assert not page.locator('[data-expression="residual"]').is_disabled()
                assert page.locator('#target-list input[value="ck19"]').is_visible()
                assert page.locator(".exploratory-switch").count() == 0
                assert page.locator("#target-list em").count() == 0
                status_text = page.locator("#target-status").inner_text().lower()
                assert all(
                    word not in status_text
                    for word in ("validated", "exploratory", "failed")
                )
                assert "exploratory" not in page.url
                assert (
                    page.locator("[data-expression].active").get_attribute(
                        "data-expression"
                    )
                    == "residual"
                )
                colored_pixels = """(box) => {
                  const {left=0, top=0, width=1, height=1} = box || {};
                  const source = document.querySelector('#viewport canvas');
                  const copy = document.createElement('canvas');
                  copy.width = 320;
                  copy.height = 180;
                  const context = copy.getContext('2d');
                  context.drawImage(
                    source,
                    source.width * left,
                    source.height * top,
                    source.width * width,
                    source.height * height,
                    0,
                    0,
                    copy.width,
                    copy.height,
                  );
                  const pixels = context.getImageData(
                    0, 0, copy.width, copy.height,
                  ).data;
                  let colored = 0;
                  for (let index = 0; index < pixels.length; index += 4) {
                    if (pixels[index] + pixels[index + 1] + pixels[index + 2] > 50) {
                      colored += 1;
                    }
                  }
                  return colored;
                }"""
                assert page.evaluate(colored_pixels) > 10
                page.locator('[data-mode="orthogonal"]').click()
                page.wait_for_timeout(300)
                assert page.locator("#viewport").get_attribute("class") == "orthogonal"
                assert (
                    page.evaluate(
                        colored_pixels,
                        {"left": 0.68, "top": 0, "width": 0.32, "height": 0.5},
                    )
                    > 5
                )
                assert (
                    page.evaluate(
                        colored_pixels,
                        {"left": 0.68, "top": 0.5, "width": 0.32, "height": 0.5},
                    )
                    > 5
                )
                page.locator('[data-mode="section"]').click()
                page.locator('[data-expression="observed"]').click()
                page.locator('[data-expression="residual"]').click()
                page.wait_for_timeout(300)
            for width, height in ((3840, 2160), (390, 844)):
                page.set_viewport_size({"width": width, "height": height})
                page.wait_for_timeout(100)
                overflow = page.evaluate(
                    """() => ({
                      x: document.documentElement.scrollWidth > innerWidth,
                      y: document.documentElement.scrollHeight > innerHeight,
                    })"""
                )
                assert overflow == {"x": False, "y": False}
            fallback = browser.new_page(viewport={"width": 1280, "height": 720})
            fallback.add_init_script(
                """(() => {
                  const original = HTMLCanvasElement.prototype.getContext;
                  HTMLCanvasElement.prototype.getContext = function(kind, ...args) {
                    if (String(kind).startsWith('webgl')) return null;
                    return original.call(this, kind, ...args);
                  };
                })()"""
            )
            fallback.on(
                "console",
                lambda message: (
                    errors.append(message.text) if message.type == "error" else None
                ),
            )
            fallback.on("pageerror", lambda error: errors.append(str(error)))
            fallback.on(
                "requestfailed",
                lambda request: errors.append(request.url),
            )
            fallback.goto(
                f"http://127.0.0.1:{server.server_port}/atlas/index.html",
                wait_until="networkidle",
            )
            fallback.wait_for_function(
                "() => document.querySelector('#loading').hidden"
            )
            assert (
                fallback.locator("#viewport").get_attribute("data-renderer")
                == "fallback"
            )
            assert fallback.locator("#compatibility").is_visible()
            assert (
                fallback.locator("#compatibility img").evaluate(
                    "image => image.naturalWidth"
                )
                > 0
            )
            # A prepared preview cannot render another section or protein
            # selection. Keep those controls from relabelling a fixed image.
            assert fallback.locator('[data-mode="section"]').is_disabled()
            assert fallback.locator("#preset").is_disabled()
            assert fallback.locator("#section").is_disabled()
            assert fallback.locator("#cohort").is_enabled()
            assert (
                "Static prediction preview"
                in fallback.locator("#scene-badge").inner_text()
            )
            protein_preview = fallback.locator("#compatibility img").get_attribute(
                "src"
            )
            fallback.locator('button[data-layer="cells"]').click()
            assert (
                fallback.locator("#compatibility img").get_attribute("src")
                != protein_preview
            )
            assert (
                "Static cell-instance preview"
                in fallback.locator("#scene-badge").inner_text()
            )
            assert (
                "measured section-center Z"
                not in fallback.locator("#scope").inner_text()
            )
            assert (
                fallback.locator('button[data-layer="cells"]').get_attribute(
                    "aria-pressed"
                )
                == "true"
            )
            assert fallback.locator("#loading").is_hidden()
            fallback.close()
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert errors == []
