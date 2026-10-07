from collections import Counter

import numpy as np
import pytest

from histopia.study import boundary_overlap_counts
from histopia.study._regions import connected_tissue_regions


def test_rectangular_grid_spacing_changes_areas_and_identities_without_resampling():
    kwargs = dict(
        semantic_fingerprint="s",
        approval_fingerprint="a",
        section_id="m:1",
        pixel_size_um=2,
    )
    square = connected_tissue_regions(np.array([[0, 0], [-1, 1]]), **kwargs)
    explicit_square = connected_tissue_regions(
        np.array([[0, 0], [-1, 1]]), **kwargs, pixel_size_um_xy=(2, 2)
    )
    rectangle = connected_tissue_regions(
        np.array([[0, 0], [-1, 1]]), **kwargs, pixel_size_um_xy=(2, 3)
    )
    assert square.fingerprint == explicit_square.fingerprint
    assert square.regions[0]["area_um2"] == 8
    assert rectangle.regions[0]["area_um2"] == 12
    assert rectangle.fingerprint != square.fingerprint
    assert rectangle.pixel_size_um_xy == (2, 3)
    np.testing.assert_array_equal(rectangle.labels, square.labels)
    with pytest.raises(ValueError, match="XY"):
        connected_tissue_regions(np.array([[0]]), **kwargs, pixel_size_um_xy=(3, 2))


def reference(cells, regions):
    counts = Counter()
    for y, x in np.ndindex(cells.shape):
        cell = int(cells[y, x])
        if not cell:
            continue
        for yy, xx in [(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)]:
            if (
                not (0 <= yy < cells.shape[0] and 0 <= xx < cells.shape[1])
                or cells[yy, xx] != cell
            ):
                counts[cell, int(regions[y, x])] += 1
                break
    return counts


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("shape", [(1, 1), (1, 9), (8, 1), (13, 17)])
def test_striped_native_boundaries_match_independent_pixel_reference(device, shape):
    if device == "cuda":
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("CUDA is optional")
    rng = np.random.default_rng(4)
    cells = rng.choice(np.array([0, 2, 7, 2**32 - 1], np.uint32), size=shape)
    # Include a cell interior and a component crossing multiple stripe boundaries.
    if min(shape) > 4:
        cells[1:7, 2:8] = 7
    regions = rng.integers(0, 4, size=shape, dtype=np.int32)
    expected = reference(cells, regions)
    full = boundary_overlap_counts(cells, regions, device=device)
    assert {(int(a), int(b)): int(c) for a, b, c in full} == expected
    merged = Counter()
    for top in range(0, shape[0], 3):
        bottom = min(shape[0], top + 3)
        lo, hi = max(0, top - 1), min(shape[0], bottom + 1)
        rows = boundary_overlap_counts(
            cells[lo:hi],
            regions[lo:hi],
            core_rows=(top - lo, bottom - lo),
            device=device,
        )
        merged.update({(int(a), int(b)): int(c) for a, b, c in rows})
    assert merged == expected


def test_background_is_counted_and_cell_interiors_are_excluded():
    cells = np.full((5, 5), 9, np.int32)
    regions = np.zeros((5, 5), np.int32)
    regions[:, 3:] = 1
    np.testing.assert_array_equal(
        boundary_overlap_counts(cells, regions), [[9, 0, 9], [9, 1, 7]]
    )
    assert boundary_overlap_counts(cells * 0, regions).shape == (0, 3)


@pytest.mark.parametrize("failure", ["negative", "float", "shape", "core", "overflow"])
def test_rejects_invalid_native_mapping(failure):
    cells = np.ones((3, 4), np.int64)
    regions = np.ones((3, 4), np.int32)
    kwargs = {}
    if failure == "negative":
        regions[0, 0] = -1
    elif failure == "float":
        cells = cells.astype(float)
    elif failure == "shape":
        regions = regions[:2]
    elif failure == "core":
        kwargs["core_rows"] = (1, 4)
    else:
        cells[:] = np.iinfo(np.int64).max
    with pytest.raises(ValueError):
        boundary_overlap_counts(cells, regions, **kwargs)
