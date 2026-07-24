from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from histopia.registration import _slides
from histopia.registration._masking import create_tissue_mask


def test_exact_slide_selection_preserves_external_ui_order(tmp_path: Path) -> None:
    first = tmp_path / "first.ndpi"
    second = tmp_path / "second.scn"
    first.touch()
    second.touch()

    selected = _slides.validate_slide_selection((second, first), wsi_only=True)

    assert selected == (second.resolve(), first.resolve())


def test_exact_slide_selection_rejects_duplicate_filenames(tmp_path: Path) -> None:
    first = tmp_path / "one" / "section.ndpi"
    second = tmp_path / "two" / "section.ndpi"
    first.parent.mkdir()
    second.parent.mkdir()
    first.touch()
    second.touch()

    with pytest.raises(ValueError, match="filenames must be unique"):
        _slides.validate_slide_selection((first, second), wsi_only=True)


def test_exact_slide_selection_rejects_missing_or_derived_inputs(
    tmp_path: Path,
) -> None:
    derived = tmp_path / "section.thumbnail.png"
    derived.touch()

    with pytest.raises(ValueError, match="derived image"):
        _slides.validate_slide_selection((derived,))
    with pytest.raises(FileNotFoundError, match="not found"):
        _slides.validate_slide_selection((tmp_path / "missing.ndpi",))


@pytest.mark.integration
def test_wsi_thumbnail_composites_grayscale_alpha_onto_white(
    tmp_path: Path,
) -> None:
    pyvips = pytest.importorskip("pyvips")
    pixels = np.array(
        [[[0, 0], [64, 128], [200, 255]], [[10, 255], [100, 255], [250, 255]]],
        dtype=np.uint8,
    )
    path = tmp_path / "grayscale-alpha.tif"
    pyvips.Image.new_from_memory(
        pixels.tobytes(),
        3,
        2,
        2,
        "uchar",
    ).copy(interpretation="b-w").tiffsave(str(path))

    thumbnail, geometry = _slides.load_slide_thumbnail(path, 3)

    assert thumbnail.shape == (2, 3, 3)
    assert thumbnail[0].tolist() == [
        [255, 255, 255],
        [159, 159, 159],
        [200, 200, 200],
    ]
    assert geometry.thumbnail_shape == (2, 3)


@pytest.mark.integration
def test_transparent_grayscale_background_is_not_masked_as_tissue(
    tmp_path: Path,
) -> None:
    pyvips = pytest.importorskip("pyvips")
    pixels = np.zeros((256, 256, 2), dtype=np.uint8)
    pixels[48:208, 64:192, 0] = 80
    pixels[48:208, 64:192, 1] = 255
    path = tmp_path / "transparent-background.tif"
    pyvips.Image.new_from_memory(
        pixels.tobytes(),
        256,
        256,
        2,
        "uchar",
    ).copy(interpretation="b-w").tiffsave(str(path))

    thumbnail, _ = _slides.load_slide_thumbnail(path, 256)
    result = create_tissue_mask(thumbnail)

    assert thumbnail[0, 0].tolist() == [255, 255, 255]
    assert result.accepted
    assert result.mask[100, 100]
    assert not result.mask[0, 0]
    assert result.metrics["component_count"] == 1
    assert result.metrics["border_touch_fraction"] == 0
    assert result.warnings == []


def test_wsi_thumbnail_retries_a_finer_pyramid_level_after_decode_error(
    monkeypatch,
) -> None:
    class DecodeError(Exception):
        pass

    class NativeImage:
        width = 1_600
        height = 1_200

        def get_fields(self) -> list[str]:
            return [
                "openslide.level-count",
                "openslide.level[0].width",
                "openslide.level[0].height",
                "openslide.level[1].width",
                "openslide.level[1].height",
                "openslide.level[2].width",
                "openslide.level[2].height",
            ]

        def get(self, key: str) -> str:
            values = {
                "openslide.level-count": "3",
                "openslide.level[0].width": "1600",
                "openslide.level[0].height": "1200",
                "openslide.level[1].width": "800",
                "openslide.level[1].height": "600",
                "openslide.level[2].width": "400",
                "openslide.level[2].height": "300",
            }
            return values[key]

    native = NativeImage()
    opened_levels: list[int] = []

    class LevelImage:
        def __init__(self, level: int) -> None:
            self.level = level

        def thumbnail_image(self, _width: int, **_kwargs: object) -> LevelImage:
            return self

    def new_from_file(
        _path: str,
        *,
        access: str,
        level: int | None = None,
    ) -> object:
        assert access == "sequential"
        if level is None:
            return native
        opened_levels.append(level)
        return LevelImage(level)

    fake_vips = SimpleNamespace(
        Error=DecodeError,
        Image=SimpleNamespace(
            new_from_file=new_from_file,
            thumbnail=lambda *_args, **_kwargs: SimpleNamespace(level="automatic"),
        ),
    )

    def convert(image: object) -> np.ndarray:
        if image.level in {"automatic", 2}:
            raise DecodeError("corrupt JPEG pyramid level")
        return np.full((192, 256, 3), 128, dtype=np.uint8)

    monkeypatch.setattr(_slides, "_import_pyvips", lambda: fake_vips)
    monkeypatch.setattr(_slides, "_vips_to_rgb", convert)

    thumbnail, geometry = _slides.load_slide_thumbnail(
        Path("corrupt-smallest-level.ndpi"),
        256,
    )

    assert thumbnail.shape == (192, 256, 3)
    assert geometry.thumbnail_shape == (192, 256)
    assert opened_levels == [2, 1]


def test_pyramid_fallback_level_is_sized_for_content_bounds() -> None:
    class NativeImage:
        width = 1_600
        height = 1_200

        def get_fields(self) -> list[str]:
            return [
                "openslide.level-count",
                "openslide.level[0].width",
                "openslide.level[0].height",
                "openslide.level[1].width",
                "openslide.level[1].height",
                "openslide.level[2].width",
                "openslide.level[2].height",
            ]

        def get(self, key: str) -> str:
            values = {
                "openslide.level-count": "3",
                "openslide.level[0].width": "1600",
                "openslide.level[0].height": "1200",
                "openslide.level[1].width": "800",
                "openslide.level[1].height": "600",
                "openslide.level[2].width": "400",
                "openslide.level[2].height": "300",
            }
            return values[key]

    levels = _slides._suitable_pyramid_levels(
        NativeImage(),
        256,
        content_bbox_xywh=(100, 100, 400, 300),
    )

    assert levels == (0,)
