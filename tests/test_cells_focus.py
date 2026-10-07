import numpy as np

from histopia.cells import (
    FocusArtifactPolicy,
    detect_focus_artifact_blocks,
    grow_focus_artifact_regions,
    remove_labels_touching_mask,
)


def _block_image(rows: int = 8, columns: int = 10) -> np.ndarray:
    return np.full((rows * 32, columns * 32, 3), 235, dtype=np.uint8)


def test_focus_gate_accepts_compact_dark_low_detail_component() -> None:
    image = _block_image()
    image[64:224, 96:320] = 70
    tissue = np.ones(image.shape[:2], dtype=np.float32)

    mask, components = detect_focus_artifact_blocks(image, tissue, downsample=4.0)

    assert len(components) == 1
    assert components[0].blocks >= 30
    assert components[0].fill_fraction > 0.9
    assert components[0].aspect_ratio == 1.4
    assert np.all(mask[96:192, 128:288])
    assert not np.any(mask[:32])


def test_focus_gate_rejects_elongated_or_high_detail_dark_tissue() -> None:
    elongated = _block_image()
    elongated[64:128, 32:288] = 70
    tissue = np.ones(elongated.shape[:2], dtype=np.float32)
    elongated_mask, elongated_components = detect_focus_artifact_blocks(
        elongated, tissue, downsample=4.0
    )

    textured = _block_image()
    checker = np.indices((160, 224)).sum(axis=0) % 2
    textured[64:224, 96:320] = np.where(
        checker[..., None] == 0,
        np.array((25, 25, 25), dtype=np.uint8),
        np.array((145, 145, 145), dtype=np.uint8),
    )
    textured_mask, textured_components = detect_focus_artifact_blocks(
        textured, tissue, downsample=4.0
    )

    assert not np.any(elongated_mask)
    assert elongated_components == ()
    assert not np.any(textured_mask)
    assert textured_components == ()


def test_focus_region_growth_follows_connected_source_object() -> None:
    rows, columns = np.indices((128, 128))
    dark_object = (rows - 64) ** 2 + (columns - 64) ** 2 <= 34**2
    image = np.full((128, 128, 3), 235, dtype=np.uint8)
    image[dark_object] = 70
    seed = np.zeros((128, 128), dtype=bool)
    seed[56:72, 56:72] = True
    policy = FocusArtifactPolicy(
        growth_gaussian_sigma_px=2.0,
        growth_minimum_seed_pixels=32,
    )

    grown = grow_focus_artifact_regions(image, seed, policy=policy)

    assert grown[64, 64]
    assert grown[64, 88]
    assert not grown[0, 0]
    assert np.count_nonzero(grown) > np.count_nonzero(seed)


def test_focus_filter_removes_whole_touching_instances() -> None:
    labels = np.zeros((8, 10), dtype=np.uint32)
    labels[1:5, 1:4] = 7
    labels[2:6, 6:9] = 11
    mask = np.zeros(labels.shape, dtype=bool)
    mask[1, 1] = True

    filtered, rejected = remove_labels_touching_mask(labels, mask)

    assert rejected.tolist() == [7]
    assert not np.any(filtered == 7)
    assert np.array_equal(filtered == 11, labels == 11)
    assert filtered.dtype == labels.dtype
