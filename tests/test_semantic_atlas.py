from __future__ import annotations

import numpy as np

from histopia.semantic import PatchFeatures
from histopia.semantic._atlas import balanced_sample_indices, fit_joint_atlas


def _section(slide_id: str, shift: float) -> PatchFeatures:
    rng = np.random.default_rng(4)
    first = rng.normal([1, 0, 0], 0.03, size=(6, 3))
    second = rng.normal([0, 1, 0], 0.03, size=(6, 3))
    features = np.vstack([first, second]).astype(np.float32)
    grid = np.array([(row, col) for row in range(3) for col in range(4)])
    xy = np.column_stack([grid[:, 1] * 112 + shift, grid[:, 0] * 112])
    return PatchFeatures(
        slide_id=slide_id,
        features=features,
        grid_rc=grid,
        native_xy=xy,
        reference_um_xy=xy,
        tissue_fraction=np.ones(12, dtype=np.float32),
        grid_shape=(3, 4),
        patch_size_px=224,
        analysis_mpp=0.5,
    )


def test_balanced_sample_caps_each_slide_deterministically() -> None:
    first = balanced_sample_indices((2, 8, 4), per_slide_cap=3, seed=17)
    second = balanced_sample_indices((2, 8, 4), per_slide_cap=3, seed=17)

    np.testing.assert_array_equal(first, second)
    assert len(first) == 8
    assert np.sum(first < 2) == 2
    assert np.sum((first >= 2) & (first < 10)) == 3
    assert np.sum(first >= 10) == 3


def test_joint_atlas_is_deterministic_and_returns_each_sensitivity() -> None:
    sections = (_section("a", 0), _section("b", 2))

    first = fit_joint_atlas(
        sections,
        cluster_counts=(2, 3),
        pca_components=2,
        balanced_patch_cap=12,
        seed=8,
        regularize=False,
    )
    second = fit_joint_atlas(
        sections,
        cluster_counts=(2, 3),
        pca_components=2,
        balanced_patch_cap=12,
        seed=8,
        regularize=False,
    )

    assert set(first.clusterings) == {2, 3}
    np.testing.assert_array_equal(
        first.clusterings[2].labels,
        second.clusterings[2].labels,
    )
    assert first.section_offsets.tolist() == [0, 12, 24]
    assert first.pca_components == 2
    assert first.clusterings[2].labels.shape == (24,)
