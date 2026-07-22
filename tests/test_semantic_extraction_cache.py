from __future__ import annotations

from pathlib import Path

import numpy as np

from histopia.semantic import PatchFeatures
from histopia.semantic._extract import feature_cache_matches


def _artifact(provenance: dict[str, object]) -> PatchFeatures:
    return PatchFeatures(
        slide_id="section.ndpi",
        features=np.zeros((1, 2), dtype=np.float32),
        grid_rc=np.zeros((1, 2), dtype=np.int32),
        native_xy=np.zeros((1, 2), dtype=np.float64),
        reference_um_xy=np.zeros((1, 2), dtype=np.float64),
        tissue_fraction=np.ones(1, dtype=np.float32),
        grid_shape=(1, 1),
        patch_size_px=224,
        analysis_mpp=0.5,
        provenance=provenance,
    )


def test_feature_cache_requires_exact_provenance(tmp_path: Path) -> None:
    path = _artifact({"preflight": "a", "model": "m"}).save(tmp_path / "feature.npz")

    assert feature_cache_matches(path, {"preflight": "a", "model": "m"})
    assert not feature_cache_matches(path, {"preflight": "b", "model": "m"})
    assert not feature_cache_matches(path, {"preflight": "a", "model": "n"})


def test_feature_cache_rejects_legacy_artifact(tmp_path: Path) -> None:
    path = tmp_path / "legacy.npz"
    artifact = _artifact({"unused": True})
    np.savez_compressed(
        path,
        schema_version=np.int16(1),
        slide_id=np.asarray(artifact.slide_id),
        features=artifact.features,
        grid_rc=artifact.grid_rc,
        native_xy=artifact.native_xy,
        reference_um_xy=artifact.reference_um_xy,
        tissue_fraction=artifact.tissue_fraction,
        grid_shape=np.asarray(artifact.grid_shape),
        patch_size_px=np.int32(artifact.patch_size_px),
        analysis_mpp=np.float64(artifact.analysis_mpp),
    )

    assert not feature_cache_matches(path, {"unused": True})
