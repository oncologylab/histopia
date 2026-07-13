import numpy as np

from histopia.registration import BrightfieldMaskConfig, create_tissue_mask
from histopia.registration._masking import _clean_mask, _mask_score


def test_create_tissue_mask_detects_weak_brightfield_tissue() -> None:
    image = np.full((100, 120, 3), 255, dtype=np.uint8)
    image[25:80, 35:95] = np.array([238, 223, 204], dtype=np.uint8)

    result = create_tissue_mask(image)

    assert result.accepted
    assert result.method != "full_fallback"
    assert result.mask[40:70, 50:85].mean() > 0.9
    assert 0.15 < result.metrics["foreground_fraction"] < 0.45


def test_create_tissue_mask_rejects_blank_slide_by_default() -> None:
    image = np.full((64, 64, 3), 255, dtype=np.uint8)

    result = create_tissue_mask(image)

    assert not result.accepted
    assert result.method != "full_fallback"
    assert not result.mask.any()


def test_create_tissue_mask_allows_explicit_legacy_full_fallback() -> None:
    image = np.full((64, 64, 3), 255, dtype=np.uint8)

    result = create_tissue_mask(
        image,
        BrightfieldMaskConfig(allow_full_fallback=True),
    )

    assert result.accepted
    assert result.method == "full_fallback"
    assert result.mask.all()
    assert "all auto_tissue candidates failed QC" in result.warnings


def test_full_mask_mode_is_explicit() -> None:
    image = np.full((16, 12, 3), 255, dtype=np.uint8)

    result = create_tissue_mask(image, BrightfieldMaskConfig(mode="full"))

    assert result.method == "full"
    assert result.mask.shape == (16, 12)
    assert result.mask.all()


def test_clean_mask_removes_scanner_frame_but_keeps_tissue() -> None:
    raw = np.zeros((200, 240), dtype=bool)
    raw[5:15, 5:235] = True
    raw[5:195, 5:15] = True
    raw[90:150, 150:210] = True

    cleaned = _clean_mask(
        raw,
        BrightfieldMaskConfig(close_radius_px=0, open_radius_px=0),
    )

    assert not cleaned[10, 100]
    assert cleaned[120, 180]


def test_mask_score_prefers_balanced_tissue_over_scattered_artifacts() -> None:
    coherent = np.zeros((200, 240), dtype=bool)
    coherent[60:150, 80:180] = True
    scattered = np.zeros_like(coherent)
    for row in range(10, 190, 30):
        for col in range(10, 230, 30):
            scattered[row : row + 6, col : col + 6] = True

    config = BrightfieldMaskConfig()

    assert _mask_score(coherent, config) > _mask_score(scattered, config)
