from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from histopia.semantic import PatchFeatures, SemanticAtlasConfig
from histopia.semantic import _pipeline as pipeline
from histopia.semantic._extract import feature_artifact_path


def _config(tmp_path: Path) -> SemanticAtlasConfig:
    return SemanticAtlasConfig(
        registration_run=tmp_path / "registration",
        output_dir=tmp_path / "semantic",
        cluster_min=2,
        cluster_max=2,
        pca_components=2,
    )


def _write_preflight(config: SemanticAtlasConfig) -> tuple[dict[str, object], ...]:
    slides: tuple[dict[str, object], ...] = (
        {
            "slide_name": "z-section.ndpi",
            "source_sha256": "source-z",
            "mask_sha256": "mask-z",
            "transform_sha256": "transform-z",
        },
        {
            "slide_name": "a-section.ndpi",
            "source_sha256": "source-a",
            "mask_sha256": "mask-a",
            "transform_sha256": "transform-a",
        },
    )
    config.output_dir.mkdir(parents=True)
    (config.output_dir / "preflight.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "slide_count": len(slides),
                "fingerprint": "preflight-fingerprint",
                "slides": slides,
            }
        )
    )
    return slides


def _feature(
    slide: dict[str, object],
    *,
    slide_id: str | None = None,
    preflight_fingerprint: str = "preflight-fingerprint",
) -> PatchFeatures:
    name = str(slide["slide_name"])
    provenance = {
        "preflight_fingerprint": preflight_fingerprint,
        "slide_name": name,
        "source_sha256": slide["source_sha256"],
        "mask_sha256": slide["mask_sha256"],
        "transform_sha256": slide["transform_sha256"],
        "model_fingerprint": "model-fingerprint",
        "analysis_mpp": 0.5,
        "patch_size_px": 224,
        "min_tissue_fraction": 0.5,
    }
    return PatchFeatures(
        slide_id=slide_id or name,
        features=np.ones((2, 3), dtype=np.float32),
        grid_rc=np.array([[0, 0], [0, 1]], dtype=np.int32),
        native_xy=np.array([[0, 0], [1, 0]], dtype=np.float64),
        reference_um_xy=np.array([[0, 0], [112, 0]], dtype=np.float64),
        tissue_fraction=np.ones(2, dtype=np.float32),
        grid_shape=(1, 2),
        patch_size_px=224,
        analysis_mpp=0.5,
        provenance=provenance,
    )


def _save_expected_features(
    config: SemanticAtlasConfig,
    slides: tuple[dict[str, object], ...],
) -> None:
    feature_dir = config.output_dir / "features"
    for order, slide in enumerate(slides, start=1):
        _feature(slide).save(
            feature_artifact_path(
                feature_dir,
                order,
                str(slide["slide_name"]),
            )
        )


def test_fit_uses_preflight_order_and_ignores_stale_extra_features(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    slides = _write_preflight(config)
    _save_expected_features(config, slides)
    stale = {
        "slide_name": "stale.ndpi",
        "source_sha256": "stale-source",
        "mask_sha256": "stale-mask",
        "transform_sha256": "stale-transform",
    }
    _feature(stale).save(config.output_dir / "features" / "000-stale.npz")
    captured: list[tuple[str, ...]] = []
    thread_limits: list[int] = []
    runtime_events: list[str] = []
    atlas = SimpleNamespace(selected_k=2)
    result = config.output_dir / "semantic_result.json"

    class CapturedLimit:
        def __init__(self, limit: int) -> None:
            thread_limits.append(limit)

        def __enter__(self) -> None:
            runtime_events.append("limit-enter")
            return None

        def __exit__(self, *args: object) -> None:
            return None

    def capture_fit(sections: tuple[PatchFeatures, ...], **_: object) -> object:
        runtime_events.append("fit")
        captured.append(tuple(section.slide_id for section in sections))
        return atlas

    def capture_write(*args: object, **kwargs: object) -> Path:
        assert args[0] is atlas
        assert kwargs["primary_clusters"] == 2
        assert kwargs["fit_threads"] == 4
        result.write_text(json.dumps({"fingerprint": "result-fingerprint"}))
        return result

    monkeypatch.setattr(pipeline, "fit_joint_atlas", capture_fit)
    monkeypatch.setattr(pipeline, "write_atlas_result", capture_write)
    monkeypatch.setattr(
        pipeline,
        "_sklearn_estimators",
        lambda: runtime_events.append("runtime-loaded"),
    )
    monkeypatch.setattr(
        "threadpoolctl.threadpool_limits",
        lambda *, limits: CapturedLimit(limits),
    )

    fitted, output = pipeline.fit_saved_features(config)

    assert fitted is atlas
    assert output == result
    assert captured == [("z-section.ndpi", "a-section.ndpi")]
    assert thread_limits == [4]
    assert runtime_events == ["runtime-loaded", "limit-enter", "fit"]
    performance = json.loads(
        (config.output_dir / "semantic_performance.json").read_text()
    )
    assert performance["fit"]["status"] == "completed"
    assert performance["fit"]["fit_threads"] == 4
    assert performance["fit"]["semantic_result_fingerprint"] == "result-fingerprint"
    assert performance["fit"]["total_patches"] == 4


def test_fit_rejects_wrong_feature_slide_before_global_computation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    slides = _write_preflight(config)
    _save_expected_features(config, slides)
    wrong = _feature(slides[0], slide_id="other.ndpi")
    wrong.save(
        feature_artifact_path(
            config.output_dir / "features",
            1,
            str(slides[0]["slide_name"]),
        )
    )
    real_load = PatchFeatures.load
    loaded: list[str] = []

    def counted_load(path: Path) -> PatchFeatures:
        loaded.append(path.name)
        return real_load(path)

    monkeypatch.setattr(
        pipeline.PatchFeatures,
        "load",
        staticmethod(counted_load),
    )
    monkeypatch.setattr(
        pipeline,
        "fit_joint_atlas",
        lambda *args, **kwargs: pytest.fail("global fit should not start"),
    )

    with pytest.raises(ValueError, match="slide identity differs"):
        pipeline.fit_saved_features(config)
    assert loaded == ["001-z-section.npz"]
    performance = json.loads(
        (config.output_dir / "semantic_performance.json").read_text()
    )
    assert performance["fit"]["status"] == "failed"
    assert performance["fit"]["failure_type"] == "ValueError"


def test_fit_rejects_stale_feature_provenance_before_global_computation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    slides = _write_preflight(config)
    _save_expected_features(config, slides)
    stale = _feature(slides[1], preflight_fingerprint="previous-preflight")
    stale.save(
        feature_artifact_path(
            config.output_dir / "features",
            2,
            str(slides[1]["slide_name"]),
        )
    )
    monkeypatch.setattr(
        pipeline,
        "fit_joint_atlas",
        lambda *args, **kwargs: pytest.fail("global fit should not start"),
    )

    with pytest.raises(ValueError, match="preflight_fingerprint"):
        pipeline.fit_saved_features(config)
