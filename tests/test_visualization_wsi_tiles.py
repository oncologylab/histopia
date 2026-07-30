from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from histopia.visualization import _wsi_tiles
from histopia.visualization._wsi_tiles import (
    WsiLayer,
    WsiLevel,
    WsiSection,
    WsiTileCapacityError,
    WsiTileService,
)


def _service(tmp_path: Path) -> tuple[WsiTileService, str]:
    image = tmp_path / "registered.tiff"
    image.write_bytes(b"registered")
    digest = hashlib.sha256(b"registered").hexdigest()
    layer = WsiLayer(
        name="registered",
        path=image,
        digest=digest,
        levels=(
            WsiLevel(256, 128, 1),
            WsiLevel(1024, 512, 0),
        ),
        tile_size=256,
        microns_per_pixel=0.5,
    )
    section = WsiSection(
        cohort="mouse",
        section="001",
        slide="slide.ndpi",
        label="H&E",
        reference=True,
        layers={"registered": layer},
    )
    return (
        WsiTileService(
            {("mouse", "001"): section},
            max_concurrent_tiles=1,
        ),
        digest,
    )


def test_wsi_metadata_is_path_free(tmp_path: Path) -> None:
    service, digest = _service(tmp_path)

    payload = service.metadata("mouse", "001")
    catalog = service.catalog("mouse")

    assert payload["slide"] == "slide.ndpi"
    assert payload["layers"]["registered"] == {
        "digest": digest,
        "tile_size": 256,
        "width": 1024,
        "height": 512,
        "levels": [
            {"width": 256, "height": 128},
            {"width": 1024, "height": 512},
        ],
        "microns_per_pixel": 0.5,
        "format": "jpg",
    }
    assert str(tmp_path) not in str(payload)
    assert catalog == {
        "schema_version": 1,
        "cohort": "mouse",
        "sections": [
            {
                "section": "001",
                "slide": "slide.ndpi",
                "label": "H&E",
                "reference": True,
                "layers": ["registered"],
            }
        ],
    }
    assert str(tmp_path) not in str(catalog)
    with pytest.raises(FileNotFoundError, match="cohort"):
        service.catalog("unknown")


def test_wsi_tile_validates_digest_and_coordinates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, digest = _service(tmp_path)
    monkeypatch.setattr(
        _wsi_tiles,
        "_render_layer_tile",
        lambda layer, level, x, y: f"{layer.name}:{level}:{x}:{y}".encode(),
    )

    payload, media_type, etag = service.render_tile(
        "mouse", "001", "registered", digest, 1, 3, 1
    )

    assert payload == b"registered:1:3:1"
    assert media_type == "image/jpeg"
    assert etag == f'"{digest}-1-3-1"'
    with pytest.raises(FileNotFoundError, match="stale"):
        service.render_tile("mouse", "001", "registered", "0" * 64, 1, 0, 0)
    with pytest.raises(FileNotFoundError, match="coordinates"):
        service.render_tile("mouse", "001", "registered", digest, 1, 4, 0)


def test_wsi_tile_capacity_is_bounded(tmp_path: Path) -> None:
    service, digest = _service(tmp_path)
    assert service._capacity.acquire(blocking=False)
    try:
        with pytest.raises(WsiTileCapacityError, match="capacity"):
            service.render_tile("mouse", "001", "registered", digest, 0, 0, 0)
    finally:
        service._capacity.release()
