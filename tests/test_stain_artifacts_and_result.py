from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from histopia.stain import (
    AdaptiveStainMap,
    StainFamily,
    StainMap,
    approve_stain_result,
    stain_review_status,
    validate_stain_approval,
)
from histopia.stain._result import write_stain_result
from histopia.stain._result_validation import validate_stain_result


def _map() -> StainMap:
    mask = np.array([[False, True], [True, True]])
    values = np.array([[0.0, 0.1], [0.2, 0.3]], dtype=np.float32)
    return StainMap(
        slide_id="section.ndpi",
        raw_target_od=values,
        corrected_target_od=values,
        counterstain_od=values / 2,
        reconstruction_residual=values / 10,
        tissue_mask=mask,
        confidence=mask.astype(np.float32),
        positive_mask=values > 0.15,
        analysis_mpp=4.0,
        content_origin_native_xy=(10, 20),
        source_mpp_xy=(0.5, 0.5),
        provenance={"registration": "abc"},
    )


def test_stain_map_detects_changed_content(tmp_path: Path) -> None:
    stain_map = _map()
    assert stain_map.content_fingerprint
    path = stain_map.save(tmp_path / "map.npz")
    loaded = StainMap.load(path)

    assert loaded.content_fingerprint
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    arrays["raw_target_od"] = arrays["raw_target_od"].copy()
    arrays["raw_target_od"][1, 1] += 1
    np.savez_compressed(path, **arrays)

    with pytest.raises(ValueError, match="content fingerprint"):
        StainMap.load(path)


def test_adaptive_stain_map_is_bound_to_physical_content(tmp_path: Path) -> None:
    source = _map()
    target = np.where(source.tissue_mask, source.corrected_target_od / 2, 0)
    adaptive = AdaptiveStainMap(
        slide_id=source.slide_id,
        target_od=target,
        tissue_mask=source.tissue_mask,
        analysis_mpp=source.analysis_mpp,
        content_origin_native_xy=source.content_origin_native_xy,
        source_mpp_xy=source.source_mpp_xy,
        source_content_fingerprint=str(source.content_fingerprint),
        method="counterstain-conditioned-v3",
        diagnostics={
            "method": "counterstain-conditioned-v3",
            "accepted": True,
        },
    )
    path = adaptive.save(tmp_path / "adaptive.npz")
    loaded = AdaptiveStainMap.load(path)

    np.testing.assert_array_equal(loaded.target_od, target)
    assert loaded.source_content_fingerprint == source.content_fingerprint
    assert len(str(loaded.content_fingerprint)) == 64
    with pytest.raises(ValueError, match="source fingerprint"):
        replace(adaptive, source_content_fingerprint="stale", content_fingerprint=None)


def test_stain_result_seals_optional_adaptive_map(tmp_path: Path) -> None:
    (tmp_path / "preflight.json").write_text("{}")
    (tmp_path / "benchmark.json").write_text("{}")
    source = _map()
    source.save(tmp_path / "map.npz")
    adaptive = AdaptiveStainMap(
        slide_id=source.slide_id,
        target_od=source.corrected_target_od,
        tissue_mask=source.tissue_mask,
        analysis_mpp=source.analysis_mpp,
        content_origin_native_xy=source.content_origin_native_xy,
        source_mpp_xy=source.source_mpp_xy,
        source_content_fingerprint=str(source.content_fingerprint),
        method="counterstain-conditioned-v3",
        diagnostics={"method": "counterstain-conditioned-v3"},
    )
    adaptive.save(tmp_path / "adaptive.npz")
    (tmp_path / "model.json").write_text("{}")
    write_stain_result(
        tmp_path,
        {
            "schema_version": 1,
            "preflight": "preflight.json",
            "benchmark": "benchmark.json",
            "slides": [
                {
                    "id": source.slide_id,
                    "quantified": True,
                    "map": "map.npz",
                    "model": "model.json",
                    "adaptive_map": "adaptive.npz",
                }
            ],
        },
    )

    result = validate_stain_result(tmp_path)
    assert "adaptive.npz" in result["artifacts"]
    (tmp_path / "adaptive.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="digest mismatch"):
        validate_stain_result(tmp_path)


@pytest.mark.parametrize(
    "field",
    [
        "raw_target_od",
        "corrected_target_od",
        "counterstain_od",
        "reconstruction_residual",
        "confidence",
    ],
)
def test_stain_map_rejects_continuous_values_outside_tissue(field: str) -> None:
    stain_map = _map()
    leaked = np.array(getattr(stain_map, field), copy=True)
    leaked[0, 0] = 0.1

    with pytest.raises(ValueError, match="outside the tissue mask"):
        replace(
            stain_map,
            **{field: leaked},
            fingerprint=None,
            content_fingerprint=None,
        )


def test_stain_map_rejects_positive_pixels_outside_tissue() -> None:
    stain_map = _map()
    leaked = np.array(stain_map.positive_mask, copy=True)
    leaked[0, 0] = True

    with pytest.raises(ValueError, match="positive mask"):
        replace(
            stain_map,
            positive_mask=leaked,
            fingerprint=None,
            content_fingerprint=None,
        )


def test_result_sealing_and_approval_reject_tampering(tmp_path: Path) -> None:
    (tmp_path / "preflight.json").write_text("{}")
    (tmp_path / "benchmark.json").write_text("{}")
    maps = tmp_path / "maps"
    models = tmp_path / "models"
    maps.mkdir()
    models.mkdir()
    _map().save(maps / "001.npz")
    (models / "001.json").write_text('{"schema_version":1}')
    result_path = write_stain_result(
        tmp_path,
        {
            "schema_version": 1,
            "preflight": "preflight.json",
            "benchmark": "benchmark.json",
            "slides": [
                {
                    "id": "section.ndpi",
                    "family": "h-dab",
                    "quantified": True,
                    "map": "maps/001.npz",
                    "model": "models/001.json",
                }
            ],
        },
    )
    payload = validate_stain_result(tmp_path)

    approval = approve_stain_result(
        tmp_path,
        reviewer="reviewer",
        notes="Synthetic maps inspected.",
    )

    assert approval.fingerprint == payload["fingerprint"]
    assert approval.families == (StainFamily.H_DAB,)
    assert validate_stain_approval(tmp_path, family="h-dab") == approval
    assert stain_review_status(tmp_path)["approved_families"] == ["h-dab"]
    models.joinpath("001.json").write_text('{"schema_version":2}')
    with pytest.raises(ValueError, match="digest mismatch"):
        validate_stain_result(tmp_path)
    with pytest.raises(ValueError, match="digest mismatch"):
        stain_review_status(tmp_path, payload)
    assert stain_review_status(
        tmp_path,
        payload,
        verify_artifacts=False,
    )["approved_families"] == ["h-dab"]
    assert json.loads(result_path.read_text())["fingerprint"] == payload["fingerprint"]


def test_stain_approval_is_scoped_by_family(tmp_path: Path) -> None:
    (tmp_path / "preflight.json").write_text("{}")
    (tmp_path / "benchmark.json").write_text("{}")
    maps = tmp_path / "maps"
    models = tmp_path / "models"
    maps.mkdir()
    models.mkdir()
    slides = []
    for index, family in enumerate(("h-dab", "sirius-red"), start=1):
        _map().save(maps / f"{index:03d}.npz")
        (models / f"{index:03d}.json").write_text("{}")
        slides.append(
            {
                "id": f"section-{index}.ndpi",
                "family": family,
                "quantified": True,
                "map": f"maps/{index:03d}.npz",
                "model": f"models/{index:03d}.json",
            }
        )
    write_stain_result(
        tmp_path,
        {
            "schema_version": 1,
            "preflight": "preflight.json",
            "benchmark": "benchmark.json",
            "slides": slides,
        },
    )

    approve_stain_result(
        tmp_path,
        reviewer="reviewer",
        notes="H-DAB maps inspected.",
        families=["h-dab"],
    )

    status = stain_review_status(tmp_path)
    assert status["approved"] is False
    assert status["approved_families"] == ["h-dab"]
    assert status["pending_families"] == ["sirius-red"]
    validate_stain_approval(tmp_path, family="h-dab")
    with pytest.raises(ValueError, match="sirius-red"):
        validate_stain_approval(tmp_path, family="sirius-red")

    review_path = tmp_path / "stain_review.json"
    review = json.loads(review_path.read_text())
    review["families"]["h-dab"]["reviewer"] = ""
    review_path.write_text(json.dumps(review))
    with pytest.raises(ValueError, match="approval is incomplete"):
        stain_review_status(tmp_path)
