from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from histopia.semantic import PatchFeatures
from histopia.semantic._atlas import AtlasClustering, JointAtlas
from histopia.semantic._result import write_atlas_result


def test_write_atlas_result_is_review_gated_and_keeps_per_slide_grids(
    tmp_path: Path,
) -> None:
    sections = tuple(
        PatchFeatures(
            slide_id=name,
            features=np.ones((2, 3), dtype=np.float32),
            grid_rc=np.array([[0, 0], [0, 1]], dtype=np.int32),
            native_xy=np.array([[1, 1], [2, 1]], dtype=float),
            reference_um_xy=np.array([[10, 10], [20, 10]], dtype=float),
            tissue_fraction=np.ones(2, dtype=np.float32),
            grid_shape=(1, 2),
            patch_size_px=224,
            analysis_mpp=0.5,
        )
        for name in ("a.ndpi", "b.ndpi")
    )
    labels = np.array([0, 1, 1, 0], dtype=np.int32)
    atlas = JointAtlas(
        slide_ids=("a.ndpi", "b.ndpi"),
        section_offsets=np.array([0, 2, 4]),
        pca_components=2,
        pca_mean=np.zeros(3, dtype=np.float32),
        pca_basis=np.zeros((2, 3), dtype=np.float32),
        clusterings={2: AtlasClustering(2, labels, labels, np.zeros((2, 2)), None)},
    )

    result_path = write_atlas_result(atlas, sections, tmp_path, primary_clusters=2)

    payload = json.loads(result_path.read_text())
    review = json.loads((tmp_path / "semantic_review.json").read_text())
    assert payload["primary_clusters"] == 2
    assert payload["selected_k"] == 2
    assert payload["schema_version"] == 3
    assert payload["feature_normalization"] == "patch_l2_v2"
    assert payload["correspondence"]["geometry_score_weight"] == 0.65
    assert payload["topology_pairs"] == []
    assert payload["slides"][0]["labels"]["2"].endswith("001.npz")
    assert not review["approved"]
    assert review["fingerprint"] == payload["fingerprint"]
    with np.load(tmp_path / payload["slides"][1]["labels"]["2"]) as saved:
        np.testing.assert_array_equal(saved["labels"], [1, 0])
        np.testing.assert_array_equal(saved["grid_rc"], [[0, 0], [0, 1]])
