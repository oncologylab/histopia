from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from histopia.registration._slides import SlideGeometry
from histopia.stain import StainMap, StainQuantificationConfig
from histopia.stain._assays import StainFamily
from histopia.stain._model import canonical_vectors
from histopia.stain._od import od_to_rgb
from histopia.stain._pipeline import run_stain_quantification
from histopia.stain._qc import summarize_stain_run
from histopia.stain._result_validation import validate_stain_result


@pytest.mark.integration
def test_tiny_wsi_pipeline_is_sealed_and_reuses_maps(tmp_path: Path) -> None:
    pyvips = pytest.importorskip("pyvips")
    run = tmp_path / "registration"
    processed = run / "processed"
    processed.mkdir(parents=True)
    height, width = 96, 128
    tissue = np.zeros((height, width), dtype=bool)
    tissue[16:82, 20:110] = True
    concentrations = np.zeros((height, width, 2), dtype=np.float32)
    concentrations[..., 0][tissue] = 0.18
    concentrations[35:72, 48:92, 1] = 0.65
    white = np.array([248.0, 247.0, 245.0])
    rgb = np.full((height, width, 3), white, dtype=np.uint8)
    rgb[tissue] = od_to_rgb(
        (concentrations @ canonical_vectors(StainFamily.H_DAB))[tissue],
        white,
    )
    source = tmp_path / "section.tif"
    pyvips.Image.new_from_memory(
        rgb.tobytes(),
        width,
        height,
        3,
        "uchar",
    ).tiffsave(str(source), tile=True)
    Image.fromarray(rgb).save(processed / "section.thumbnail.png")
    Image.fromarray(tissue.astype(np.uint8) * 255).save(processed / "section.mask.png")
    geometry = SlideGeometry(
        native_shape=(height, width),
        content_bbox_xywh=(0, 0, width, height),
        thumbnail_shape=(height, width),
        bounds_source="synthetic",
        mpp_xy=(0.5, 0.5),
        mpp_source="synthetic",
    )
    (run / "registration_result.json").write_text(
        json.dumps(
            {
                "reference_slide": str(source),
                "slides": [
                    {
                        "path": str(source),
                        "is_reference": True,
                        "geometry": geometry.to_json_dict(),
                        "mask": {"accepted": True, "method": "synthetic"},
                        "transform": {"matrix": np.eye(3).tolist()},
                    }
                ],
            }
        )
    )
    config = StainQuantificationConfig(
        registration_run=run,
        output_dir=tmp_path / "stain",
        analysis_mpp=0.5,
        methods=("fixed",),
        sample_pixels=5_000,
        white_sample_pixels=2_000,
        workers=2,
    )

    result_path = run_stain_quantification(config)
    first_fingerprint = validate_stain_result(config.output_dir)["fingerprint"]
    first_mtime = (
        (config.output_dir / "maps" / "001-section.tif.npz").stat().st_mtime_ns
    )
    run_stain_quantification(config)

    payload = validate_stain_result(config.output_dir)
    performance = json.loads((config.output_dir / "stain_performance.json").read_text())
    row = payload["slides"][0]
    stain_map = StainMap.load(config.output_dir / row["map"])
    qc = summarize_stain_run(config.output_dir)
    assert result_path == config.output_dir / "stain_result.json"
    assert payload["fingerprint"] == first_fingerprint
    assert performance["result_fingerprint"] == first_fingerprint
    assert performance["maps_reused"] == 1
    assert performance["maps_written"] == 0
    assert stain_map.corrected_target_od[50, 60] > 0.5
    assert stain_map.corrected_target_od[20, 25] < 0.05
    assert row["quantified"] is True
    assert qc.quantified_slides == 1
    assert (
        config.output_dir / "maps" / "001-section.tif.npz"
    ).stat().st_mtime_ns == first_mtime
