from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest

from histopia.semantic import (
    SemanticLinearProbe,
    SemanticProbeConfig,
    fit_semantic_probe,
    load_stamp_features,
)


def _classified_features(seed: int = 8) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = np.repeat(["acinar", "ductal", "stroma"], 30)
    centers = {
        "acinar": np.array([2.0, 0.0, 0.0, 0.0]),
        "ductal": np.array([0.0, 2.0, 0.0, 0.0]),
        "stroma": np.array([0.0, 0.0, 2.0, 0.0]),
    }
    features = np.vstack(
        [centers[label] + rng.normal(0, 0.12, 4) for label in labels]
    ).astype(np.float32)
    return features, labels


def _train_validation_indices() -> tuple[np.ndarray, np.ndarray]:
    train = np.concatenate([np.arange(0, 20), np.arange(30, 50), np.arange(60, 80)])
    validation = np.setdiff1d(np.arange(90), train)
    return train, validation


def test_semantic_probe_is_deterministic_and_calibrates_validation() -> None:
    features, labels = _classified_features()
    config = SemanticProbeConfig(pca_components=3, seed=17)
    train, validation = _train_validation_indices()

    first = fit_semantic_probe(
        features[train],
        labels[train],
        validation_features=features[validation],
        validation_labels=labels[validation],
        config=config,
    )
    second = fit_semantic_probe(
        features[train],
        labels[train],
        validation_features=features[validation],
        validation_labels=labels[validation],
        config=config,
    )

    assert first.model.fingerprint == second.model.fingerprint
    np.testing.assert_array_equal(
        first.model.predict_proba(features),
        second.model.predict_proba(features),
    )
    assert first.diagnostics.validation_nll_calibrated is not None
    assert first.diagnostics.validation_nll_uncalibrated is not None
    assert (
        first.diagnostics.validation_nll_calibrated
        <= first.diagnostics.validation_nll_uncalibrated + 1e-8
    )


def test_semantic_probe_pca_uses_training_rows_only() -> None:
    train, labels = _classified_features()
    validation = train[:12] + 1_000
    validation_labels = labels[:12]

    with_validation = fit_semantic_probe(
        train,
        labels,
        validation_features=validation,
        validation_labels=validation_labels,
        config=SemanticProbeConfig(pca_components=3),
    )
    without_validation = fit_semantic_probe(
        train,
        labels,
        config=SemanticProbeConfig(pca_components=3),
    )

    np.testing.assert_array_equal(
        with_validation.model.feature_mean,
        without_validation.model.feature_mean,
    )
    np.testing.assert_array_equal(
        with_validation.model.pca_components,
        without_validation.model.pca_components,
    )
    np.testing.assert_array_equal(
        with_validation.model.coefficients,
        without_validation.model.coefficients,
    )


def test_semantic_probe_round_trip_and_fingerprint_guard(tmp_path: Path) -> None:
    features, labels = _classified_features()
    fit = fit_semantic_probe(
        features,
        labels,
        config=SemanticProbeConfig(pca_components=3),
    )
    path = fit.model.save(tmp_path / "probe.npz")
    loaded = SemanticLinearProbe.load(path)

    assert loaded.fingerprint == fit.model.fingerprint
    assert loaded.class_names == fit.model.class_names
    np.testing.assert_allclose(
        loaded.predict_proba(features),
        fit.model.predict_proba(features),
        rtol=1e-6,
        atol=1e-7,
    )

    with pytest.raises(ValueError, match="fingerprint"):
        replace(loaded, coefficients=loaded.coefficients + 1)


def test_semantic_probe_rejects_unknown_validation_class() -> None:
    features, labels = _classified_features()
    with pytest.raises(ValueError, match="unknown class"):
        fit_semantic_probe(
            features,
            labels,
            validation_features=features[:3],
            validation_labels=np.array(["unknown"] * 3),
        )


def test_stamp_hdf5_import_requires_explicit_physical_binding(tmp_path: Path) -> None:
    path = tmp_path / "stamp.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "feats",
            data=np.arange(12, dtype=np.float32).reshape(3, 4),
        )
        handle.create_dataset(
            "coords",
            data=np.array([[0, 0], [224, 0], [0, 224]], dtype=np.int32),
        )

    table = load_stamp_features(path)
    features = table.to_patch_features(
        slide_id="sample.svs",
        native_to_reference_um=np.diag([0.5, 0.5, 1.0]),
        native_patch_size_px=224,
    )

    np.testing.assert_array_equal(features.grid_rc, [[0, 0], [0, 1], [1, 0]])
    np.testing.assert_allclose(
        features.reference_um_xy,
        [[56, 56], [168, 56], [56, 168]],
    )
    assert features.grid_shape == (2, 2)
    assert features.provenance is not None
    assert features.provenance["source_format"] == "stamp-hdf5"
    assert len(str(features.provenance["source_sha256"])) == 64


def test_stamp_hdf5_import_rejects_missing_datasets(tmp_path: Path) -> None:
    path = tmp_path / "bad.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("feats", data=np.ones((2, 3)))

    with pytest.raises(ValueError, match="feature and coordinate"):
        load_stamp_features(path)
