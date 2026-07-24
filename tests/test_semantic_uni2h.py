from __future__ import annotations

from pathlib import Path

import numpy as np

from histopia.semantic._uni2h import (
    Uni2hEncoder,
    _cached_model_revision,
    _LazyUni2hEncoder,
    _preload_wsi_backend,
    _Uni2hRuntime,
)


def test_uni2h_preloads_wsi_backend_before_gpu_stack() -> None:
    imported: list[str] = []

    _preload_wsi_backend(imported.append)

    assert imported == ["pyvips"]


def test_cached_model_revision_reads_pinned_snapshot(tmp_path) -> None:
    model = tmp_path / "models--MahmoodLab--UNI2-h"
    (model / "refs").mkdir(parents=True)
    (model / "snapshots" / "abc123").mkdir(parents=True)
    (model / "refs" / "main").write_text("abc123\n")

    assert _cached_model_revision(tmp_path) == "abc123"


def test_lazy_encoder_loads_weights_only_for_first_encode(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[tuple[Path, str]] = []
    runtime = _Uni2hRuntime(
        cache_dir=tmp_path,
        device="cuda:0",
        model_fingerprint="model",
        provenance={"device": "cuda:0"},
        torch=object(),
    )
    encoder = _LazyUni2hEncoder(
        cache_dir=tmp_path,
        device="cuda:0",
        local_only=True,
        vips_threads=4,
        runtime=runtime,
    )

    class Loaded:
        model_fingerprint = "model"
        runtime_provenance = {"device": "cuda:0"}

        def encode(self, images: np.ndarray) -> np.ndarray:
            return np.ones((len(images), 2), dtype=np.float32)

    def load(
        cls,
        cache_dir: Path,
        *,
        device: str,
        local_only: bool,
        vips_threads: int | None,
    ) -> Loaded:
        assert local_only
        assert vips_threads == 4
        calls.append((cache_dir, device))
        return Loaded()

    monkeypatch.setattr(Uni2hEncoder, "from_cache", classmethod(load))

    assert calls == []
    assert encoder.model_fingerprint == "model"
    result = encoder.encode(np.zeros((3, 4, 4, 3), dtype=np.uint8))
    np.testing.assert_array_equal(result, np.ones((3, 2), dtype=np.float32))
    encoder.encode(np.zeros((1, 4, 4, 3), dtype=np.uint8))
    assert calls == [(tmp_path, "cuda:0")]
