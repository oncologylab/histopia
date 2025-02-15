import json
from pathlib import Path

import numpy as np
from PIL import Image

from histopia.registration import RegistrationConfig, register_sections
from histopia.registration._pipeline import _crop_to_mask


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
