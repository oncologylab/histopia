from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from histopia.semantic import PatchFeatures
from histopia.semantic._atlas import AtlasClustering, JointAtlas
from histopia.semantic._result import validate_semantic_result, write_atlas_result


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
    changed = dict(payload)
    changed["selected_k"] = 3
    with pytest.raises(ValueError, match="fingerprint is stale"):
        validate_semantic_result(tmp_path, changed)


def test_result_fingerprint_rejects_changed_artifact_bytes(tmp_path: Path) -> None:
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
    write_atlas_result(atlas, sections, tmp_path, primary_clusters=2)
    payload = validate_semantic_result(tmp_path)
    label_path = tmp_path / payload["slides"][0]["labels"]["2"]

    np.savez_compressed(label_path, labels=np.array([1, 1], dtype=np.int16))

    with pytest.raises(ValueError, match="artifact digest"):
        validate_semantic_result(tmp_path)


def test_result_records_and_checks_expected_preflight_slide_order(
    tmp_path: Path,
) -> None:
    provenance = {
        "preflight_fingerprint": "preflight-fingerprint",
        "model_fingerprint": "model-fingerprint",
        "analysis_mpp": 0.5,
        "patch_size_px": 224,
        "min_tissue_fraction": 0.5,
    }
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
            provenance={**provenance, "slide_name": name},
        )
        for name in ("a.ndpi", "b.ndpi")
    )
    (tmp_path / "preflight.json").write_text(
        json.dumps(
            {
                "fingerprint": "preflight-fingerprint",
                "slides": [
                    {"slide_name": "a.ndpi"},
                    {"slide_name": "b.ndpi"},
                ],
            }
        )
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

    result = write_atlas_result(atlas, sections, tmp_path, primary_clusters=2)
    payload = json.loads(result.read_text())

    assert payload["feature_provenance"]["expected_slide_ids"] == [
        "a.ndpi",
        "b.ndpi",
    ]


def test_result_rejects_feature_sections_incomplete_against_preflight(
    tmp_path: Path,
) -> None:
    provenance = {
        "preflight_fingerprint": "preflight-fingerprint",
        "model_fingerprint": "model-fingerprint",
        "analysis_mpp": 0.5,
        "patch_size_px": 224,
        "min_tissue_fraction": 0.5,
        "slide_name": "a.ndpi",
    }
    section = PatchFeatures(
        slide_id="a.ndpi",
        features=np.ones((2, 3), dtype=np.float32),
        grid_rc=np.array([[0, 0], [0, 1]], dtype=np.int32),
        native_xy=np.array([[1, 1], [2, 1]], dtype=float),
        reference_um_xy=np.array([[10, 10], [20, 10]], dtype=float),
        tissue_fraction=np.ones(2, dtype=np.float32),
        grid_shape=(1, 2),
        patch_size_px=224,
        analysis_mpp=0.5,
        provenance=provenance,
    )
    (tmp_path / "preflight.json").write_text(
        json.dumps(
            {
                "fingerprint": "preflight-fingerprint",
                "slides": [
                    {"slide_name": "a.ndpi"},
                    {"slide_name": "b.ndpi"},
                ],
            }
        )
    )
    labels = np.array([0, 1], dtype=np.int32)
    atlas = JointAtlas(
        slide_ids=("a.ndpi",),
        section_offsets=np.array([0, 2]),
        pca_components=2,
        pca_mean=np.zeros(3, dtype=np.float32),
        pca_basis=np.zeros((2, 3), dtype=np.float32),
        clusterings={2: AtlasClustering(2, labels, labels, np.zeros((2, 2)), None)},
    )

    with pytest.raises(ValueError, match="preflight slide order"):
        write_atlas_result(atlas, (section,), tmp_path, primary_clusters=2)


@pytest.mark.parametrize("artifact", ["/tmp/model.npz", "../model.npz"])
def test_result_fingerprint_rejects_unsafe_artifact_paths(
    tmp_path: Path,
    artifact: str,
) -> None:
    (tmp_path / "semantic_result.json").write_text(
        json.dumps(
            {
                "schema_version": 3,
                "fingerprint": "invalid",
                "model": artifact,
                "slides": [],
                "topology_pairs": [],
                "artifacts": {},
            }
        )
    )

    with pytest.raises(ValueError, match="relative"):
        validate_semantic_result(tmp_path)
