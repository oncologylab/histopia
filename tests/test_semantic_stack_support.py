from __future__ import annotations

import numpy as np

from histopia.semantic._features import PatchFeatures
from histopia.semantic._result import _fit_config_payload
from histopia.semantic._stack_support import filter_stack_supported_components


def _section(
    slide_id: str,
    grid_rc: list[tuple[int, int]],
    reference_um_xy: list[tuple[float, float]],
) -> PatchFeatures:
    count = len(grid_rc)
    return PatchFeatures(
        slide_id=slide_id,
        features=np.arange(count * 3, dtype=np.float32).reshape(count, 3),
        grid_rc=np.asarray(grid_rc, dtype=np.int32),
        native_xy=np.asarray(reference_um_xy, dtype=np.float64),
        reference_um_xy=np.asarray(reference_um_xy, dtype=np.float64),
        tissue_fraction=np.ones(count, dtype=np.float32),
        grid_shape=(12, 12),
        patch_size_px=10,
        analysis_mpp=1.0,
        provenance={"slide_name": slide_id},
    )


def test_filter_removes_one_section_fragment_and_preserves_recurring_lobule() -> None:
    dominant_grid = [(row, column) for row in range(3) for column in range(4)]
    dominant_xy = [(c * 10.0, r * 10.0) for r, c in dominant_grid]
    lobule_grid = [(8, 8), (8, 9)]
    lobule_xy = [(80.0, 80.0), (90.0, 80.0)]
    fragment_grid = [(5, 8), (5, 9), (6, 8), (6, 9)]
    fragment_xy = [(800.0, 500.0), (810.0, 500.0), (800.0, 510.0), (810.0, 510.0)]
    sections = (
        _section("a.ndpi", dominant_grid + lobule_grid, dominant_xy + lobule_xy),
        _section(
            "b.ndpi",
            dominant_grid + lobule_grid + fragment_grid,
            dominant_xy + lobule_xy + fragment_xy,
        ),
        _section("c.ndpi", dominant_grid + lobule_grid, dominant_xy + lobule_xy),
    )

    filtered, report = filter_stack_supported_components(
        sections,
        min_neighbor_fraction=0.1,
    )

    assert report is not None
    assert report["method"] == "adjacent-component-support-v1"
    assert report["removed_components"] == 1
    assert report["removed_patches"] == 4
    assert [len(section.features) for section in filtered] == [14, 14, 14]
    np.testing.assert_array_equal(filtered[1].grid_rc[-2:], lobule_grid)
    assert filtered[1].content_fingerprint is not None
    assert len(sections[1].features) == 18


def test_disabled_filter_returns_exact_inputs_and_default_fit_identity() -> None:
    sections = (
        _section("a.ndpi", [(0, 0)], [(0.0, 0.0)]),
        _section("b.ndpi", [(0, 0)], [(0.0, 0.0)]),
    )

    filtered, report = filter_stack_supported_components(
        sections,
        min_neighbor_fraction=0.0,
    )

    assert filtered is sections
    assert report is None
    fit_config = _fit_config_payload(
        requested_pca_components=64,
        balanced_patch_cap=4096,
        seed=0,
        max_cross_section_distance_um=112.0,
    )
    assert "stack_component_filter" not in fit_config


def test_nondefault_fit_identity_records_exact_stack_filter() -> None:
    fit_config = _fit_config_payload(
        requested_pca_components=64,
        balanced_patch_cap=4096,
        seed=0,
        max_cross_section_distance_um=112.0,
        min_stack_component_neighbor_fraction=0.1,
    )

    assert fit_config["stack_component_filter"] == {
        "method": "adjacent-component-support-v1",
        "min_neighbor_fraction": 0.1,
        "neighbor_radius_patch_widths": 1.5,
        "preserve_relative_area": 0.5,
    }
