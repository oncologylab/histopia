from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from histopia.registration import _slides


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
