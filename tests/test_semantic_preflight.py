from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import histopia.semantic._preflight as preflight_module
from histopia.semantic import PatchFeatures, SemanticAtlasConfig
from histopia.semantic._cli import main
from histopia.semantic._extract import extract_registration_features
from histopia.semantic._performance import (
    load_performance_report,
    write_performance_stage,
)
from histopia.semantic._preflight import (
    preflight_registration,
    write_preflight,
)


def _write_registration(tmp_path: Path) -> Path:
    run = tmp_path / "registration"
    processed = run / "processed"
    processed.mkdir(parents=True)
    slides = []
    reviews = []
    for index, name in enumerate(("HE.ndpi", "CK19.ndpi")):
        source = tmp_path / "raw" / name
        source.parent.mkdir(exist_ok=True)
        source.write_bytes(f"slide-{index}".encode())
        image = np.full((8, 10, 3), 230 - index * 20, dtype=np.uint8)
        mask = np.zeros((8, 10), dtype=np.uint8)
        mask[1:7, 2:9] = 255
        Image.fromarray(image).save(processed / f"{source.stem}.thumbnail.png")
        Image.fromarray(mask).save(processed / f"{source.stem}.mask.png")
        review = {
            "slide": name,
            "thumbnail_sha256": f"mask-fingerprint-{index}",
            "status": "auto_pass",
            "method": "group_consensus",
            "reviewer": "Test Reviewer",
            "notes": "Reviewed.",
            "override_path": None,
        }
        reviews.append(review)
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
                "mask_review": dict(review),
            }
        )
    (run / "registration_result.json").write_text(
        json.dumps({"reference_slide": slides[0]["path"], "slides": slides})
    )
    (run / "mask_review.json").write_text(
        json.dumps({"schema_version": 2, "slides": reviews})
    )
    (run / "section_order_review.json").write_text(
        json.dumps(
            {
                "schema_version": 3,
                "approved": True,
                "fingerprint": "accepted-order",
                "input_fingerprints": {
                    slide["slide"]: f"order-input-{index}"
                    for index, slide in enumerate(reviews)
                },
                "slides": [
                    {"order": index + 1, "slide": slide["slide"]}
                    for index, slide in enumerate(reviews)
                ],
            }
        )
    )
    _seal_registration_approval(run)
    return run


def _seal_registration_approval(run: Path) -> None:
    result = json.loads((run / "registration_result.json").read_text())
    order = json.loads((run / "section_order_review.json").read_text())
    artifacts = {}
    for name in (
        "registration_result.json",
        "mask_review.json",
        "section_order_review.json",
    ):
        artifacts[name] = hashlib.sha256((run / name).read_bytes()).hexdigest()
    (run / "registration_approval.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "reviewer": "Test Reviewer",
                "reviewed_at": "2026-07-24T10:00:00+00:00",
                "notes": "Registration reviewed.",
                "slide_count": len(result["slides"]),
                "order_fingerprint": order["fingerprint"],
                "artifacts": artifacts,
            }
        )
    )


def test_preflight_records_complete_fingerprinted_registration(tmp_path: Path) -> None:
    run = _write_registration(tmp_path)

    result = preflight_registration(run)
    output = write_preflight(result, tmp_path / "semantic" / "preflight.json")
    payload = json.loads(output.read_text())

    assert result.slide_count == 2
    assert result.schema_version == 3
    assert result.reference_slide == "HE.ndpi"
    assert len(result.fingerprint) == 64
    assert [slide.slide_name for slide in result.slides] == ["HE.ndpi", "CK19.ndpi"]
    assert all(len(slide.mask_sha256) == 64 for slide in result.slides)
    assert all(slide.mask_method == "group_consensus" for slide in result.slides)
    assert all(slide.mask_review_status == "auto_pass" for slide in result.slides)
    assert payload["fingerprint"] == result.fingerprint
    assert (
        payload["registration_approval_sha256"]
        == hashlib.sha256((run / "registration_approval.json").read_bytes()).hexdigest()
    )
    assert payload["order_review_fingerprint"] == "accepted-order"


def test_extraction_records_cache_and_compute_performance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _write_registration(tmp_path)
    preflight = preflight_registration(run)
    output = tmp_path / "semantic"
    config = SemanticAtlasConfig(
        registration_run=run,
        output_dir=output,
        batch_size=8,
        patch_workers=2,
        vips_threads=3,
    )

    class Encoder:
        model_fingerprint = "model-fingerprint"
        runtime_provenance = {
            "device": "cuda:0",
            "precision": "bfloat16-autocast",
            "accelerator": {
                "name": "Test GPU",
                "compute_capability": [8, 0],
            },
        }

    class Reader:
        provenance_id = "test-reader"

        def __init__(self, _path: Path) -> None:
            pass

    def extract(**kwargs: object) -> PatchFeatures:
        slide_id = str(kwargs["slide_id"])
        provenance = kwargs["provenance"]
        return PatchFeatures(
            slide_id=slide_id,
            features=np.ones((3, 4), dtype=np.float32),
            grid_rc=np.array([[0, 0], [0, 1], [1, 0]], dtype=np.int32),
            native_xy=np.array([[1, 1], [2, 1], [1, 2]], dtype=np.float64),
            reference_um_xy=np.array([[1, 1], [2, 1], [1, 2]], dtype=np.float64),
            tissue_fraction=np.ones(3, dtype=np.float32),
            grid_shape=(2, 2),
            patch_size_px=config.patch_size_px,
            analysis_mpp=config.analysis_mpp,
            provenance=provenance,
        )

    import histopia.semantic._extract as extraction

    monkeypatch.setattr(extraction, "configure_vips_threads", lambda _value: None)
    monkeypatch.setattr(extraction, "_VipsPatchReader", Reader)
    monkeypatch.setattr(
        extraction,
        "_read_mask",
        lambda _path: np.ones((8, 10), dtype=bool),
    )
    monkeypatch.setattr(extraction, "extract_patch_features", extract)
    write_performance_stage(output, "fit", {"status": "stale"})

    paths = extract_registration_features(config, Encoder(), preflight=preflight)
    first = load_performance_report(output / "semantic_performance.json")

    assert len(paths) == 2
    assert "fit" not in first
    assert first["extraction"]["status"] == "completed"
    assert first["extraction"]["extracted_slides"] == 2
    assert first["extraction"]["cached_slides"] == 0
    assert first["extraction"]["total_patches"] == 6
    assert first["extraction"]["controls"] == {
        "batch_size": 8,
        "patch_workers": 2,
        "vips_threads": 3,
        "device": "cuda:0",
        "precision": "bfloat16-autocast",
        "accelerator": {
            "name": "Test GPU",
            "compute_capability": [8, 0],
        },
    }

    monkeypatch.setattr(
        extraction,
        "extract_patch_features",
        lambda **_kwargs: pytest.fail("valid feature cache should be reused"),
    )
    extract_registration_features(config, Encoder(), preflight=preflight)
    second = load_performance_report(output / "semantic_performance.json")

    assert second["extraction"]["status"] == "completed"
    assert second["extraction"]["cached_slides"] == 2
    assert second["extraction"]["extracted_slides"] == 0
    assert second["extraction"]["total_patches"] == 6

    def fail_extract(**_kwargs: object) -> None:
        raise RuntimeError("test failure")

    monkeypatch.setattr(extraction, "extract_patch_features", fail_extract)
    with pytest.raises(RuntimeError, match="test failure"):
        extract_registration_features(
            config,
            Encoder(),
            preflight=preflight,
            overwrite=True,
        )
    failed = load_performance_report(output / "semantic_performance.json")

    assert failed["extraction"]["status"] == "failed"
    assert failed["extraction"]["failure_type"] == "RuntimeError"
    assert failed["extraction"]["completed_slides"] == 0


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
    _seal_registration_approval(run)

    with pytest.raises(ValueError, match="CK19.ndpi.*finite"):
        preflight_registration(run)


def test_preflight_rejects_unapproved_order_when_manifest_exists(
    tmp_path: Path,
) -> None:
    run = _write_registration(tmp_path)
    (run / "section_order_review.json").write_text(
        json.dumps({"approved": False, "fingerprint": "pending"})
    )
    _seal_registration_approval(run)

    with pytest.raises(ValueError, match="registration approval order fingerprint"):
        preflight_registration(run)


def test_run_cli_checks_registration_before_requiring_model_cache(
    tmp_path: Path,
) -> None:
    run = _write_registration(tmp_path)
    (run / "section_order_review.json").write_text(
        json.dumps({"approved": False, "fingerprint": "pending"})
    )
    _seal_registration_approval(run)
    config = tmp_path / "semantic.json"
    config.write_text(
        json.dumps(
            {
                "registration_run": str(run),
                "output_dir": str(tmp_path / "semantic"),
            }
        )
    )

    with pytest.raises(ValueError, match="registration approval order fingerprint"):
        main(["run", "--config", str(config)])


def test_preflight_rejects_unapproved_mask_review(tmp_path: Path) -> None:
    run = _write_registration(tmp_path)
    path = run / "registration_result.json"
    payload = json.loads(path.read_text())
    payload["slides"][1]["mask_review"] = {"status": "pending"}
    path.write_text(json.dumps(payload))
    _seal_registration_approval(run)

    with pytest.raises(ValueError, match="registration approval.*unapproved mask"):
        preflight_registration(run)


def test_preflight_rejects_unaccepted_registration_mask(tmp_path: Path) -> None:
    run = _write_registration(tmp_path)
    path = run / "registration_result.json"
    payload = json.loads(path.read_text())
    payload["slides"][1]["mask"]["accepted"] = False
    path.write_text(json.dumps(payload))
    _seal_registration_approval(run)

    with pytest.raises(ValueError, match="CK19.ndpi.*mask is not accepted"):
        preflight_registration(run)


def test_preflight_requires_final_registration_approval(tmp_path: Path) -> None:
    run = _write_registration(tmp_path)
    (run / "registration_approval.json").unlink()

    with pytest.raises(ValueError, match="requires a sealed registration approval"):
        preflight_registration(run)


def test_preflight_rejects_approval_changed_during_slide_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _write_registration(tmp_path)
    original = preflight_module._validate_slide
    changed = False

    def mutate_after_validation(*args, **kwargs):
        nonlocal changed
        slide = original(*args, **kwargs)
        if not changed:
            approval_path = run / "registration_approval.json"
            approval = json.loads(approval_path.read_text())
            approval["notes"] = "Changed during semantic preflight."
            approval_path.write_text(json.dumps(approval))
            changed = True
        return slide

    monkeypatch.setattr(preflight_module, "_validate_slide", mutate_after_validation)

    with pytest.raises(
        ValueError, match="registration approval changed during semantic preflight"
    ):
        preflight_registration(run)


def test_preflight_rejects_mask_shape_mismatch(tmp_path: Path) -> None:
    run = _write_registration(tmp_path)
    Image.fromarray(np.ones((7, 10), dtype=np.uint8) * 255).save(
        run / "processed" / "CK19.mask.png"
    )

    with pytest.raises(ValueError, match="CK19.ndpi.*mask shape"):
        preflight_registration(run)


def test_preflight_rejects_content_bounds_outside_native_slide(
    tmp_path: Path,
) -> None:
    run = _write_registration(tmp_path)
    path = run / "registration_result.json"
    payload = json.loads(path.read_text())
    payload["slides"][1]["geometry"]["content_bbox_xywh"] = [90, 0, 20, 80]
    path.write_text(json.dumps(payload))
    _seal_registration_approval(run)

    with pytest.raises(ValueError, match="CK19.ndpi.*invalid slide geometry"):
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
