from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import histopia.semantic._approval as approval_module
from histopia.semantic import PatchFeatures, approve_semantic_result
from histopia.semantic._atlas import AtlasClustering, JointAtlas
from histopia.semantic._performance import write_performance_stage
from histopia.semantic._result import (
    _common_feature_provenance,
    validate_semantic_result,
    write_atlas_result,
)


def test_write_atlas_result_is_review_gated_and_keeps_per_slide_grids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        approval_module,
        "validate_semantic_registration_binding",
        lambda *args, **kwargs: None,
    )
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
    assert payload["fit_config"] == {
        "algorithm": "global-semantic-atlas",
        "algorithm_version": 1,
        "requested_pca_components": 64,
        "balanced_patch_cap": 4096,
        "seed": 0,
        "max_cross_section_distance_um": 112.0,
    }
    assert payload["correspondence"]["geometry_score_weight"] == 0.65
    assert payload["topology_pairs"] == []
    assert payload["slides"][0]["labels"]["2"].endswith("001.npz")
    assert not review["approved"]
    assert review["fingerprint"] == payload["fingerprint"]
    write_performance_stage(tmp_path, "fit", {"elapsed_seconds": 1.25})
    validated = validate_semantic_result(tmp_path)
    assert validated["fingerprint"] == payload["fingerprint"]
    assert "semantic_performance.json" not in validated["artifacts"]
    with np.load(tmp_path / payload["slides"][1]["labels"]["2"]) as saved:
        np.testing.assert_array_equal(saved["labels"], [1, 0])
        np.testing.assert_array_equal(saved["grid_rc"], [[0, 0], [0, 1]])
    approve_semantic_result(
        tmp_path,
        registration_run=tmp_path / "registration",
        reviewer="Test Reviewer",
        notes="Exact deterministic rerun reviewed.",
        reviewed_at="2026-07-25T01:00:00+00:00",
    )
    original_result = result_path.read_bytes()

    write_atlas_result(atlas, sections, tmp_path, primary_clusters=2)

    preserved_review = json.loads((tmp_path / "semantic_review.json").read_text())
    assert result_path.read_bytes() == original_result
    assert preserved_review["approved"] is True
    assert preserved_review["reviewer"] == "Test Reviewer"
    assert preserved_review["reviewed_at"] == "2026-07-25T01:00:00+00:00"
    changed = dict(payload)
    changed["selected_k"] = 3
    with pytest.raises(ValueError, match="fingerprint is stale"):
        validate_semantic_result(tmp_path, changed)

    atlas.clusterings[2].labels[0] = 1
    write_atlas_result(atlas, sections, tmp_path, primary_clusters=2)

    changed_review = json.loads((tmp_path / "semantic_review.json").read_text())
    assert changed_review["approved"] is False
    assert changed_review["fingerprint"] != payload["fingerprint"]


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
        "batch_size": 128,
        "encoder_runtime": {"device": "cuda", "torch": "test"},
        "extraction_method": "test-grid-v2",
        "patch_reader": "test-reader-v1",
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
    assert payload["feature_provenance"]["batch_size"] == 128
    assert payload["feature_provenance"]["encoder_runtime"]["device"] == "cuda"
    assert payload["feature_provenance"]["feature_integrity"] == "legacy-unsealed"
    assert payload["fit_runtime"]["native_threads"] == 4
    assert set(payload["fit_runtime"]) == {
        "numpy",
        "scikit-learn",
        "scipy",
        "threadpoolctl",
        "native_threads",
    }

    sealed_root = tmp_path / "sealed"
    sealed_root.mkdir()
    (sealed_root / "preflight.json").write_text(
        (tmp_path / "preflight.json").read_text()
    )
    sealed_sections = tuple(
        PatchFeatures.load(section.save(sealed_root / "features" / f"{index:03d}.npz"))
        for index, section in enumerate(sections, start=1)
    )
    sealed = _common_feature_provenance(sealed_sections, sealed_root)

    assert sealed is not None
    assert sealed["feature_integrity"] == "content-sha256-v1"
    assert sealed["feature_content_fingerprints"] == [
        section.content_fingerprint for section in sealed_sections
    ]
    with pytest.raises(ValueError, match="content sealing is inconsistent"):
        _common_feature_provenance(
            (sealed_sections[0], sections[1]),
            sealed_root,
        )


def test_result_rejects_partial_execution_provenance(tmp_path: Path) -> None:
    provenance = {
        "preflight_fingerprint": "preflight-fingerprint",
        "model_fingerprint": "model-fingerprint",
        "analysis_mpp": 0.5,
        "patch_size_px": 224,
        "min_tissue_fraction": 0.5,
        "batch_size": 128,
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
                "slides": [{"slide_name": "a.ndpi"}],
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

    with pytest.raises(ValueError, match="execution provenance is incomplete"):
        write_atlas_result(atlas, (section,), tmp_path, primary_clusters=2)


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
    model = tmp_path / "atlas_model.npz"
    model.write_bytes(b"accepted-model")

    with pytest.raises(ValueError, match="preflight slide order"):
        write_atlas_result(atlas, (section,), tmp_path, primary_clusters=2)
    assert model.read_bytes() == b"accepted-model"


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


def test_result_fingerprint_rejects_symlinked_artifact_escape(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "semantic"
    run_dir.mkdir()
    outside = tmp_path / "outside.npz"
    outside.write_bytes(b"outside")
    (run_dir / "model.npz").symlink_to(outside)
    (run_dir / "semantic_result.json").write_text(
        json.dumps(
            {
                "schema_version": 3,
                "fingerprint": "invalid",
                "model": "model.npz",
                "slides": [],
                "topology_pairs": [],
                "artifacts": {},
            }
        )
    )

    with pytest.raises(ValueError, match="relative"):
        validate_semantic_result(run_dir)
