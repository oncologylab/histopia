from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from histopia.semantic._cli import main
from histopia.semantic._preflight import (
    preflight_registration,
    write_preflight,
)


def _write_registration(tmp_path: Path) -> Path:
    run = tmp_path / "registration"
    processed = run / "processed"
    processed.mkdir(parents=True)
    slides = []
    for index, name in enumerate(("HE.ndpi", "CK19.ndpi")):
        source = tmp_path / "raw" / name
        source.parent.mkdir(exist_ok=True)
        source.write_bytes(f"slide-{index}".encode())
        image = np.full((8, 10, 3), 230 - index * 20, dtype=np.uint8)
        mask = np.zeros((8, 10), dtype=np.uint8)
        mask[1:7, 2:9] = 255
        Image.fromarray(image).save(processed / f"{source.stem}.thumbnail.png")
        Image.fromarray(mask).save(processed / f"{source.stem}.mask.png")
        slides.append(
            {
                "path": str(source),
                "is_reference": index == 0,
                "geometry": {
                    "native_shape": [80, 100],
                    "content_bbox_xywh": [0, 0, 100, 80],
                    "thumbnail_shape": [8, 10],
                    "bounds_source": "test",
                    "mpp_xy": [0.5, 0.5],
                    "mpp_source": "test",
                },
                "transform": {"matrix": np.eye(3).tolist()},
                "mask": {"accepted": True, "method": "group_consensus"},
                "mask_review": {"status": "approved", "approved": True},
            }
        )
    (run / "registration_result.json").write_text(
        json.dumps({"reference_slide": slides[0]["path"], "slides": slides})
    )
    (run / "section_order_review.json").write_text(
        json.dumps({"approved": True, "fingerprint": "accepted-order"})
    )
    return run


def test_preflight_records_complete_fingerprinted_registration(tmp_path: Path) -> None:
    run = _write_registration(tmp_path)

    result = preflight_registration(run)
    output = write_preflight(result, tmp_path / "semantic" / "preflight.json")
    payload = json.loads(output.read_text())

    assert result.slide_count == 2
    assert result.reference_slide == "HE.ndpi"
    assert len(result.fingerprint) == 64
    assert [slide.slide_name for slide in result.slides] == ["HE.ndpi", "CK19.ndpi"]
    assert all(len(slide.mask_sha256) == 64 for slide in result.slides)
    assert all(slide.mask_method == "group_consensus" for slide in result.slides)
    assert all(slide.mask_review_status == "approved" for slide in result.slides)
    assert payload["fingerprint"] == result.fingerprint
    assert payload["order_review_fingerprint"] == "accepted-order"


def test_preflight_rejects_missing_mask(tmp_path: Path) -> None:
    run = _write_registration(tmp_path)
    (run / "processed" / "CK19.mask.png").unlink()

    with pytest.raises(FileNotFoundError, match="CK19.ndpi.*mask"):
        preflight_registration(run)


def test_preflight_rejects_nonfinite_transform(tmp_path: Path) -> None:
    run = _write_registration(tmp_path)
    path = run / "registration_result.json"
    payload = json.loads(path.read_text())
    payload["slides"][1]["transform"]["matrix"][0][0] = float("nan")
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="CK19.ndpi.*finite"):
        preflight_registration(run)


def test_preflight_rejects_unapproved_order_when_manifest_exists(
    tmp_path: Path,
) -> None:
    run = _write_registration(tmp_path)
    (run / "section_order_review.json").write_text(
        json.dumps({"approved": False, "fingerprint": "pending"})
    )

    with pytest.raises(ValueError, match="section order is not approved"):
        preflight_registration(run)


def test_run_cli_checks_registration_before_requiring_model_cache(
    tmp_path: Path,
) -> None:
    run = _write_registration(tmp_path)
    (run / "section_order_review.json").write_text(
        json.dumps({"approved": False, "fingerprint": "pending"})
    )
    config = tmp_path / "semantic.json"
    config.write_text(
        json.dumps(
            {
                "registration_run": str(run),
                "output_dir": str(tmp_path / "semantic"),
            }
        )
    )

    with pytest.raises(ValueError, match="section order is not approved"):
        main(["run", "--config", str(config)])


def test_preflight_rejects_unapproved_mask_review(tmp_path: Path) -> None:
    run = _write_registration(tmp_path)
    path = run / "registration_result.json"
    payload = json.loads(path.read_text())
    payload["slides"][1]["mask_review"] = {"status": "pending"}
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="CK19.ndpi.*mask review is not approved"):
        preflight_registration(run)


def test_preflight_rejects_unaccepted_registration_mask(tmp_path: Path) -> None:
    run = _write_registration(tmp_path)
    path = run / "registration_result.json"
    payload = json.loads(path.read_text())
    payload["slides"][1]["mask"]["accepted"] = False
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="CK19.ndpi.*mask is not accepted"):
        preflight_registration(run)


def test_preflight_rejects_mask_shape_mismatch(tmp_path: Path) -> None:
    run = _write_registration(tmp_path)
    Image.fromarray(np.ones((7, 10), dtype=np.uint8) * 255).save(
        run / "processed" / "CK19.mask.png"
    )

    with pytest.raises(ValueError, match="CK19.ndpi.*mask shape"):
        preflight_registration(run)


def test_preflight_cli_writes_output_manifest(tmp_path: Path) -> None:
    run = _write_registration(tmp_path)
    output = tmp_path / "semantic"
    config = tmp_path / "semantic.json"
    config.write_text(
        json.dumps({"registration_run": str(run), "output_dir": str(output)})
    )

    assert main(["preflight", "--config", str(config)]) == 0
    assert json.loads((output / "preflight.json").read_text())["slide_count"] == 2
