import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from histopia.registration import RegistrationConfig, _pipeline, register_sections
from histopia.registration._pipeline import (
    _create_tissue_masks,
    _crop_to_mask,
    _load_automatic_mask_snapshot,
)


def test_register_sections_writes_thumbnail_result(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    fixed = np.full((80, 80, 3), 255, dtype=np.uint8)
    fixed[20:55, 22:58] = np.array([238, 223, 204], dtype=np.uint8)
    moving = np.roll(fixed, shift=(3, -4), axis=(0, 1))
    Image.fromarray(fixed).save(input_dir / "[#001] fixed.png")
    Image.fromarray(moving).save(input_dir / "[#002] moving.png")

    result = register_sections(
        RegistrationConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            rigid_method="phase_correlation",
            max_processed_image_dim_px=80,
        )
    )

    result_path = output_dir / "registration_result.json"
    assert result_path.exists()
    payload = json.loads(result_path.read_text())
    assert len(payload["slides"]) == 2
    assert result.reference_slide.name == "[#001] fixed.png"
    assert payload["slides"][1]["aligned_to"].endswith("[#001] fixed.png")
    assert payload["slides"][1]["alignment_metrics"]["dice"] > 0.9
    assert (output_dir / "qc" / "[#001] fixed.mask_overlay.png").exists()
    assert (output_dir / "validation_report.md").exists()


def test_tissue_crop_ignores_tiny_remote_artifact() -> None:
    image = np.full((200, 240, 3), 255, dtype=np.uint8)
    mask = np.zeros((200, 240), dtype=bool)
    mask[70:150, 120:210] = True
    mask[5:10, 5:10] = True

    crop = _crop_to_mask(image, mask, target_dim_px=200, padding_fraction=0)

    assert np.array_equal(crop.offset_xy, np.array([120.0, 70.0]))
    assert crop.image.shape[:2] == (178, 200)


def test_parallel_mask_creation_matches_sequential_results(tmp_path: Path) -> None:
    images = {}
    for index, shift in enumerate((0, 3, 6)):
        image = np.full((90, 110, 3), 255, dtype=np.uint8)
        image[20 + shift : 70 + shift, 25:85] = [175, 95, 120]
        images[tmp_path / f"section-{index}.png"] = image
    sequential = _create_tissue_masks(
        images,
        RegistrationConfig(tmp_path, tmp_path / "sequential", mask_workers=1),
    )
    parallel = _create_tissue_masks(
        images,
        RegistrationConfig(tmp_path, tmp_path / "parallel", mask_workers=2),
    )

    assert sequential.keys() == parallel.keys()
    for path in sequential:
        assert sequential[path].method == parallel[path].method
        assert np.array_equal(sequential[path].mask, parallel[path].mask)
        assert sequential[path].candidate_masks.keys() == (
            parallel[path].candidate_masks.keys()
        )
        for method in sequential[path].candidate_masks:
            assert np.array_equal(
                sequential[path].candidate_masks[method],
                parallel[path].candidate_masks[method],
            )


def test_mask_workers_must_be_positive(tmp_path: Path) -> None:
    with np.testing.assert_raises_regex(ValueError, "mask_workers must be positive"):
        RegistrationConfig(tmp_path, tmp_path / "output", mask_workers=0)


def test_automatic_mask_snapshot_requires_exact_hash_and_slide_set(
    tmp_path: Path,
) -> None:
    slide = tmp_path / "section.ndpi"
    image = np.full((20, 24, 3), 255, dtype=np.uint8)
    mask = np.zeros((20, 24), dtype=np.uint8)
    mask[4:16, 6:18] = 255
    mask_path = tmp_path / "section.mask.png"
    Image.fromarray(mask).save(mask_path)
    manifest = tmp_path / "snapshot.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "slides": [
                    {
                        "slide": slide.name,
                        "mask": mask_path.name,
                        "sha256": hashlib.sha256(mask_path.read_bytes()).hexdigest(),
                    }
                ],
            }
        )
    )

    loaded = _load_automatic_mask_snapshot(
        manifest,
        (slide,),
        {slide: image},
    )

    assert loaded[slide].method == "approved_automatic_snapshot"
    assert np.array_equal(loaded[slide].mask, mask > 127)

    mask_path.write_bytes(b"changed")
    with np.testing.assert_raises_regex(ValueError, "hash mismatch"):
        _load_automatic_mask_snapshot(manifest, (slide,), {slide: image})


def test_anchored_order_reuses_exact_distance_cache(
    tmp_path: Path, monkeypatch
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    base = np.full((80, 80, 3), 255, dtype=np.uint8)
    base[18:62, 22:58] = np.array([185, 100, 120], dtype=np.uint8)
    for index, shift in enumerate((0, 2, 4), start=1):
        Image.fromarray(np.roll(base, shift=(shift, 0), axis=(0, 1))).save(
            input_dir / f"[#{index:03d}] section.png"
        )
    original = _pipeline._section_distance_matrix
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_pipeline, "_section_distance_matrix", counted)
    config = RegistrationConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        rigid_method="phase_correlation",
        section_order_strategy="anchored_similarity",
        ordering_workers=2,
        max_processed_image_dim_px=80,
    )

    register_sections(config)
    first = json.loads((output_dir / "section_order_review.json").read_text())
    register_sections(config)
    second = json.loads((output_dir / "section_order_review.json").read_text())

    assert calls == 1
    assert first["fingerprint"] == second["fingerprint"]
    assert (output_dir / ".cache" / "section-order-distances.npz").is_file()

    sequential_output = tmp_path / "sequential"
    register_sections(
        RegistrationConfig(
            input_dir=input_dir,
            output_dir=sequential_output,
            rigid_method="phase_correlation",
            section_order_strategy="anchored_similarity",
            ordering_workers=1,
            max_processed_image_dim_px=80,
        )
    )
    with np.load(
        output_dir / ".cache" / "section-order-distances.npz",
        allow_pickle=False,
    ) as parallel_cache:
        parallel = parallel_cache["distances"]
    with np.load(
        sequential_output / ".cache" / "section-order-distances.npz",
        allow_pickle=False,
    ) as sequential_cache:
        sequential = sequential_cache["distances"]
    assert np.array_equal(parallel, sequential)
