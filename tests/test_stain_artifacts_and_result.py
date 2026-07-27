from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from histopia.stain import (
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
