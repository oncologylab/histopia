from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from histopia.stain._artifacts import StainMap
from histopia.stain._result import write_stain_result
from histopia.visualization._stain_viewer import (
    _overlay_rgba,
    load_stain_viewer_run,
)
from histopia.visualization._viewer import build_section_viewer


def test_viewer_builds_bound_stain_layers_and_probe_grid(tmp_path: Path) -> None:
    registration, registration_payload = _registration_run(tmp_path)
    stain = _stain_run(tmp_path, registration, registration_payload)
    viewer = tmp_path / "viewer"

    index = build_section_viewer(
        {"mouse": registration},
        viewer,
        stain_runs={"mouse": stain},
    )

    manifest = json.loads((viewer / "manifest.json").read_text())
    mouse = manifest["mice"][0]
    slide = mouse["slides"][0]
    assert mouse["stain"]["quantified_slides"] == 1
    assert mouse["stain"]["qc"]["correction_accepted"] == 1
    assert mouse["stain"]["qc"]["threshold_accepted"] == 1
    assert mouse["stain"]["review"] == {
        "approved": False,
        "fingerprint_matches": True,
    }
    assert slide["stain"]["quantified"] is True
    assert set(slide["stain"]["textures"]) == {"raw", "corrected"}
    assert set(slide["stain"]["overlay_textures"]) == {"raw", "corrected"}
    for relative in (
        *slide["stain"]["textures"].values(),
        *slide["stain"]["overlay_textures"].values(),
    ):
        with Image.open(viewer / relative) as image:
            assert image.size == (40, 32)
            assert np.asarray(image.convert("RGBA"))[..., 3].max() > 0
    probe_path = viewer / slide["stain"]["probe"]
    probe = np.fromfile(probe_path, dtype="<u2")
    expected = 2 * slide["stain"]["probe_width"] * slide["stain"]["probe_height"]
    assert len(probe) == expected
    assert np.any(probe != slide["stain"]["probe_nodata"])
    assert 'data-mode="stain-overlay"' in index.read_text()
    script = (viewer / "viewer.js").read_text()
    assert "probeStatistics" in script
    assert "relative OD; not cross-antibody normalized" in script

    build_section_viewer(
        {"mouse": registration},
        viewer,
        stain_runs={"mouse": stain},
    )
    report = json.loads((viewer / "build-report.json").read_text())
    assert report["mice_reused"] == 1
    assert report["mice_rendered"] == 0


def test_stain_viewer_rejects_a_different_registration_result(
    tmp_path: Path,
) -> None:
    registration, registration_payload = _registration_run(tmp_path)
    stain = _stain_run(tmp_path, registration, registration_payload)
    registration_payload["reference_slide"] = str(tmp_path / "changed.ndpi")
    (registration / "registration_result.json").write_text(
        json.dumps(registration_payload)
    )

    with pytest.raises(
        ValueError,
        match="different registration result",
    ):
        load_stain_viewer_run(registration, registration_payload, stain)


def test_stain_overlay_does_not_tint_unsupported_registered_tissue() -> None:
    registered = np.full((2, 2, 3), 120, dtype=np.uint8)
    registered_mask = np.ones((2, 2), dtype=bool)
    signal_mask = np.zeros((2, 2), dtype=bool)
    signal_mask[0, 0] = True
    heatmap = np.full((2, 2, 4), (220, 30, 20, 255), dtype=np.uint8)
    values = np.ones((2, 2), dtype=np.float32)

    overlay = _overlay_rgba(
        registered,
        registered_mask,
        signal_mask,
        heatmap,
        values,
        1.0,
    )

    assert not np.array_equal(overlay[0, 0, :3], registered[0, 0])
    np.testing.assert_array_equal(overlay[1, 1, :3], registered[1, 1])
    assert np.all(overlay[..., 3] == 255)


def _registration_run(
    tmp_path: Path,
) -> tuple[Path, dict[str, object]]:
    registration = tmp_path / "registration"
    processed = registration / "processed"
    processed.mkdir(parents=True)
    source = tmp_path / "section.ndpi"
    source.touch()
    image = np.full((32, 40, 3), 242, dtype=np.uint8)
    image[5:28, 7:34] = (154, 101, 75)
    mask = np.zeros((32, 40), dtype=np.uint8)
    mask[5:28, 7:34] = 255
    Image.fromarray(image).save(processed / "section.thumbnail.png")
    Image.fromarray(mask).save(processed / "section.mask.png")
    payload: dict[str, object] = {
        "reference_slide": str(source),
        "slides": [
            {
                "path": str(source),
                "is_reference": True,
                "geometry": {
                    "native_shape": [320, 400],
                    "content_bbox_xywh": [0, 0, 400, 320],
                    "thumbnail_shape": [32, 40],
                    "mpp_xy": [0.5, 0.5],
                },
                "transform": {"matrix": np.eye(3).tolist()},
            }
        ],
    }
    (registration / "registration_result.json").write_text(json.dumps(payload))
    return registration, payload


def _stain_run(
    tmp_path: Path,
    registration: Path,
    registration_payload: dict[str, object],
) -> Path:
    stain = tmp_path / "stain"
    maps = stain / "maps"
    models = stain / "models"
    maps.mkdir(parents=True)
    models.mkdir()
    height, width = 16, 20
    tissue = np.zeros((height, width), dtype=bool)
    tissue[2:14, 3:17] = True
    target = np.zeros((height, width), dtype=np.float32)
    target[tissue] = np.linspace(0.02, 0.8, tissue.sum())
    stain_map = StainMap(
        slide_id="section.ndpi",
        raw_target_od=target * 1.15,
        corrected_target_od=target,
        counterstain_od=target * 0.25,
        reconstruction_residual=np.where(tissue, 0.02, 0).astype(np.float32),
        tissue_mask=tissue,
        confidence=np.where(tissue, 0.95, 0).astype(np.float32),
        positive_mask=tissue & (target > 0.4),
        analysis_mpp=4.0,
        content_origin_native_xy=(0, 0),
        source_mpp_xy=(0.5, 0.5),
        provenance={"fixture": "stain-viewer"},
    )
    map_path = maps / "001-section.ndpi.npz"
    stain_map.save(map_path)
    (models / "001-section.ndpi.json").write_text("{}")
    (stain / "preflight.json").write_text("{}")
    (stain / "benchmark.json").write_text("{}")
    registration_sha = hashlib.sha256(
        (registration / "registration_result.json").read_bytes()
    ).hexdigest()
    write_stain_result(
        stain,
        {
            "schema_version": 1,
            "measurement": {
                "quantity": "relative_chromogen_optical_density",
                "analysis_mpp": 4.0,
            },
            "registration_result_sha256": registration_sha,
            "preflight": "preflight.json",
            "benchmark": "benchmark.json",
            "slides": [
                {
                    "id": Path(str(registration_payload["slides"][0]["path"])).name,
                    "order": 1,
                    "marker": "DAB",
                    "family": "h-dab",
                    "quantified": True,
                    "map": map_path.relative_to(stain).as_posix(),
                    "model": "models/001-section.ndpi.json",
                    "quantiles": {"0.99": 0.78},
                    "positive_fraction": 0.5,
                    "qc": {
                        "correction_accepted": True,
                        "threshold_accepted": True,
                        "raw_glass_leakage": 0.04,
                        "corrected_glass_leakage": 0.01,
                        "positive_threshold_od": 0.4,
                        "flags": [],
                    },
                }
            ],
        },
    )
    return stain
