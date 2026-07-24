from pathlib import Path

import numpy as np

from histopia.registration import BrightfieldMaskConfig
from histopia.registration._masking import TissueMaskResult
from histopia.registration._preprocessing_cache import (
    load_or_create_group_masks,
    load_or_create_independent_mask,
    load_or_create_thumbnail,
)
from histopia.registration._slides import SlideGeometry


def test_thumbnail_cache_reuses_valid_entry(tmp_path: Path) -> None:
    source = tmp_path / "slide.ndpi"
    source.write_bytes(b"source")
    calls = 0
    image = np.full((12, 16, 3), 137, dtype=np.uint8)
    geometry = SlideGeometry(
        native_shape=(120, 160),
        content_bbox_xywh=(0, 0, 160, 120),
        thumbnail_shape=(12, 16),
        bounds_source="full_slide",
        mpp_xy=(0.5, 0.5),
        mpp_source="test",
    )

    def loader(path: Path, max_dim_px: int) -> tuple[np.ndarray, SlideGeometry]:
        nonlocal calls
        calls += 1
        assert path == source
        assert max_dim_px == 32
        return image.copy(), geometry

    first = load_or_create_thumbnail(source, 32, tmp_path / "cache", loader)
    second = load_or_create_thumbnail(source, 32, tmp_path / "cache", loader)

    assert calls == 1
    assert np.array_equal(first[0], second[0])
    assert first[1] == second[1]


def test_thumbnail_cache_invalidates_changed_source(tmp_path: Path) -> None:
    source = tmp_path / "slide.ndpi"
    source.write_bytes(b"first")
    calls = 0

    def loader(path: Path, max_dim_px: int) -> tuple[np.ndarray, SlideGeometry]:
        nonlocal calls
        calls += 1
        image = np.full((4, 5, 3), calls, dtype=np.uint8)
        geometry = SlideGeometry(
            native_shape=(4, 5),
            content_bbox_xywh=(0, 0, 5, 4),
            thumbnail_shape=(4, 5),
            bounds_source="full_slide",
        )
        return image, geometry

    load_or_create_thumbnail(source, 32, tmp_path / "cache", loader)
    source.write_bytes(b"second-source")
    loaded, _ = load_or_create_thumbnail(source, 32, tmp_path / "cache", loader)

    assert calls == 2
    assert np.all(loaded == 2)


def test_thumbnail_cache_rejects_shape_valid_corruption(tmp_path: Path) -> None:
    source = tmp_path / "slide.ndpi"
    source.write_bytes(b"source")
    calls = 0

    def loader(path: Path, max_dim_px: int) -> tuple[np.ndarray, SlideGeometry]:
        nonlocal calls
        calls += 1
        return np.full((4, 5, 3), calls, dtype=np.uint8), SlideGeometry(
            native_shape=(4, 5),
            content_bbox_xywh=(0, 0, 5, 4),
            thumbnail_shape=(4, 5),
            bounds_source="full_slide",
        )

    cache = tmp_path / "cache"
    load_or_create_thumbnail(source, 32, cache, loader)
    image_path = next((cache / "thumbnails").glob("*/image.npy"))
    with image_path.open("wb") as stream:
        np.save(stream, np.full((4, 5, 3), 99, dtype=np.uint8))

    loaded, _ = load_or_create_thumbnail(source, 32, cache, loader)

    assert calls == 2
    assert np.all(loaded == 2)


def test_independent_mask_cache_round_trips_candidates(tmp_path: Path) -> None:
    source = tmp_path / "slide.ndpi"
    source.write_bytes(b"source")
    image = np.full((20, 24, 3), 210, dtype=np.uint8)
    config = BrightfieldMaskConfig()
    calls = 0

    def creator(
        value: np.ndarray,
        settings: BrightfieldMaskConfig,
    ) -> TissueMaskResult:
        nonlocal calls
        calls += 1
        mask = np.zeros(value.shape[:2], dtype=bool)
        mask[4:16, 5:19] = True
        return TissueMaskResult(
            mask=mask,
            method="test",
            metrics={"foreground_fraction": float(mask.mean())},
            accepted=True,
            warnings=[],
            candidate_metrics={"candidate": {"foreground_fraction": 0.25}},
            candidate_warnings={"candidate": []},
            candidate_masks={"candidate": mask.copy()},
        )

    first = load_or_create_independent_mask(
        source,
        image,
        config,
        tmp_path / "cache",
        creator,
    )
    second = load_or_create_independent_mask(
        source,
        image,
        config,
        tmp_path / "cache",
        creator,
    )

    assert calls == 1
    assert second.method == first.method
    assert second.metrics == first.metrics
    assert np.array_equal(second.mask, first.mask)
    assert np.array_equal(
        second.candidate_masks["candidate"],
        first.candidate_masks["candidate"],
    )


def test_independent_mask_cache_rejects_shape_valid_corruption(
    tmp_path: Path,
) -> None:
    source = tmp_path / "slide.ndpi"
    source.write_bytes(b"source")
    image = np.zeros((8, 9, 3), dtype=np.uint8)
    calls = 0

    def creator(
        value: np.ndarray,
        settings: BrightfieldMaskConfig,
    ) -> TissueMaskResult:
        nonlocal calls
        calls += 1
        return TissueMaskResult(
            mask=np.full(value.shape[:2], calls % 2, dtype=bool),
            method="test",
            metrics={},
            accepted=True,
            warnings=[],
        )

    cache = tmp_path / "cache"
    load_or_create_independent_mask(
        source, image, BrightfieldMaskConfig(), cache, creator
    )
    mask_path = next((cache / "masks").glob("*/masks.npz"))
    with mask_path.open("wb") as stream:
        np.savez_compressed(stream, mask=np.zeros((8, 9), dtype=bool))

    loaded = load_or_create_independent_mask(
        source, image, BrightfieldMaskConfig(), cache, creator
    )

    assert calls == 2
    assert not loaded.mask.any()


def test_mask_cache_invalidates_configuration_and_image(tmp_path: Path) -> None:
    source = tmp_path / "slide.ndpi"
    source.write_bytes(b"source")
    image = np.zeros((8, 9, 3), dtype=np.uint8)
    calls = 0

    def creator(
        value: np.ndarray,
        settings: BrightfieldMaskConfig,
    ) -> TissueMaskResult:
        nonlocal calls
        calls += 1
        return TissueMaskResult(
            mask=np.full(value.shape[:2], calls % 2, dtype=bool),
            method="test",
            metrics={},
            accepted=True,
            warnings=[],
        )

    cache = tmp_path / "cache"
    load_or_create_independent_mask(
        source, image, BrightfieldMaskConfig(), cache, creator
    )
    changed_image = image.copy()
    changed_image[0, 0] = 1
    load_or_create_independent_mask(
        source, changed_image, BrightfieldMaskConfig(), cache, creator
    )
    load_or_create_independent_mask(
        source,
        changed_image,
        BrightfieldMaskConfig(close_radius_px=5),
        cache,
        creator,
    )

    assert calls == 3


def test_group_mask_cache_reuses_exact_cohort_result(tmp_path: Path) -> None:
    paths = (tmp_path / "first.ndpi", tmp_path / "second.ndpi")
    for path in paths:
        path.write_bytes(path.name.encode())
    images = {
        path: np.full((16, 20, 3), index * 40, dtype=np.uint8)
        for index, path in enumerate(paths, start=1)
    }
    base_masks = {
        path: TissueMaskResult(
            mask=np.eye(16, 20, dtype=bool),
            method="base",
            metrics={},
            accepted=True,
            warnings=[],
            candidate_masks={"candidate": np.eye(16, 20, dtype=bool)},
        )
        for path in paths
    }
    calls = 0

    def creator() -> dict[Path, TissueMaskResult]:
        nonlocal calls
        calls += 1
        return {
            path: TissueMaskResult(
                mask=np.ones((16, 20), dtype=bool),
                method=f"refined-{index}",
                metrics={"foreground_fraction": 1.0},
                accepted=True,
                warnings=[],
                candidate_masks=base_masks[path].candidate_masks,
            )
            for index, path in enumerate(paths)
        }

    cache = tmp_path / "cache"
    first = load_or_create_group_masks(
        base_masks,
        images,
        {path: 1.0 for path in paths},
        cache,
        creator,
    )
    second = load_or_create_group_masks(
        base_masks,
        images,
        {path: 1.0 for path in paths},
        cache,
        creator,
    )

    assert calls == 1
    assert [result.method for result in second.values()] == [
        result.method for result in first.values()
    ]
    assert all(np.all(result.mask) for result in second.values())


def test_group_mask_cache_invalidates_physical_calibration(tmp_path: Path) -> None:
    path = tmp_path / "slide.ndpi"
    path.write_bytes(b"source")
    image = np.zeros((8, 9, 3), dtype=np.uint8)
    result = TissueMaskResult(
        mask=np.ones((8, 9), dtype=bool),
        method="base",
        metrics={},
        accepted=True,
        warnings=[],
    )
    calls = 0

    def creator() -> dict[Path, TissueMaskResult]:
        nonlocal calls
        calls += 1
        return {path: result}

    cache = tmp_path / "cache"
    load_or_create_group_masks(
        {path: result}, {path: image}, {path: 1.0}, cache, creator
    )
    load_or_create_group_masks(
        {path: result}, {path: image}, {path: 2.0}, cache, creator
    )

    assert calls == 2
