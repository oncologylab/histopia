import numpy as np
import pytest

from histopia.registration._mask_recovery import recover_supported_mask


def test_pale_tissue_recovered_while_glass_and_neutral_shading_stay_missing():
    pytest.importorskip("cv2")
    rgb = np.full((200, 300, 3), 255, "uint8")
    rgb[30:130, 30:130] = [242, 219, 230]
    rgb[30:130, 170:270] = 242
    base = np.zeros(rgb.shape[:2], bool)
    mask, evidence = recover_supported_mask(
        rgb, base, {"group_pale_tissue": np.ones(base.shape, bool)}
    )
    assert mask[60, 60]
    assert not mask[60, 200]
    assert not mask[170, 170]
    assert evidence["added_pixels"] > 5000
    assert not evidence["trained_detector"]


def test_recovery_excludes_isolated_black_annotation():
    pytest.importorskip("cv2")
    rgb = np.full((200, 300, 3), 255, "uint8")
    rgb[20:60, 30:40] = 5
    rgb[30:40, 20:60] = 5
    rgb[80:180, 100:250] = [240, 205, 230]
    base = np.any(rgb < 65, axis=2)
    mask, evidence = recover_supported_mask(
        rgb, base, {"group_pale_tissue": np.any(rgb < 245, axis=2)}
    )
    assert mask[100, 150]
    assert not mask[35, 35]
    assert evidence["excluded_pixels"] > 0


def test_annotation_rim_exclusion_preserves_coloured_and_distant_tissue(monkeypatch):
    pytest.importorskip("cv2")
    from histopia.registration import _mask_recovery as module

    rgb = np.full((40, 50, 3), 220, dtype="uint8")
    rgb[19, 21] = [230, 200, 220]
    base = np.ones(rgb.shape[:2], bool)
    excluded = np.zeros(base.shape, bool)
    excluded[20, 20] = True
    monkeypatch.setattr(
        module,
        "isolated_annotation_exclusion",
        lambda *args: (excluded.copy(), excluded.copy()),
    )
    default, _ = recover_supported_mask(rgb, base, {})
    revised, evidence = recover_supported_mask(
        rgb, base, {}, exclude_annotation_rims=True
    )
    assert default[20, 21] and not revised[20, 21]
    assert revised[19, 21] and revised[30, 30]
    assert evidence["annotation_rim_pixels"] > 0
