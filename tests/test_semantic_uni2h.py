from __future__ import annotations

import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from histopia.semantic._uni2h import (
    Uni2hEncoder,
    _autocast_dtype,
    _cached_model_revision,
    _inference_precision,
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


class _CudaPrecision:
    def __init__(self, support: dict[int, bool]) -> None:
        self.current = 0
        self.support = support

    @contextmanager
    def device(self, index: int):
        previous = self.current
        self.current = index
        try:
            yield
        finally:
            self.current = previous

    def is_bf16_supported(self, *, including_emulation: bool) -> bool:
        assert not including_emulation
        return self.support[self.current]


def _precision_torch(*, bfloat16: dict[int, bool]):
    cuda = _CudaPrecision(bfloat16)
    return SimpleNamespace(
        bfloat16=object(),
        float16=object(),
        cuda=cuda,
        device=lambda value: SimpleNamespace(index=int(value.partition(":")[2] or 0)),
    )


def test_uni2h_precision_is_selected_for_requested_cuda_device() -> None:
    torch = _precision_torch(bfloat16={0: False, 1: True})

    assert _inference_precision(torch, "cuda:0") == "float16-autocast"
    assert _inference_precision(torch, "cuda:1") == "bfloat16-autocast"
    assert torch.cuda.current == 0
    assert _inference_precision(torch, "cpu") == "float32"
    assert _inference_precision(torch, "mps") == "float32"


def test_uni2h_autocast_dtype_matches_recorded_precision() -> None:
    torch = _precision_torch(bfloat16={0: True})

    assert _autocast_dtype(torch, "bfloat16-autocast") is torch.bfloat16
    assert _autocast_dtype(torch, "float16-autocast") is torch.float16
    assert _autocast_dtype(torch, "float32") is None


def test_uni2h_encoder_uses_recorded_float16_fallback(monkeypatch) -> None:
    calls: dict[str, object] = {}
    float16 = object()

    class Batch:
        def to(self, device: str, *, non_blocking: bool):
            calls["transfer"] = (device, non_blocking)
            return self

    class Output:
        def float(self):
            return self

        def cpu(self):
            return self

        def numpy(self) -> np.ndarray:
            return np.ones((2, 3), dtype=np.float32)

    def autocast(**kwargs):
        calls["autocast"] = kwargs
        return nullcontext()

    torch = SimpleNamespace(
        OutOfMemoryError=RuntimeError,
        autocast=autocast,
        bfloat16=object(),
        float16=float16,
        inference_mode=nullcontext,
        stack=lambda tensors: Batch(),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    encoder = Uni2hEncoder(
        lambda batch: Output(),
        lambda image: image,
        device="cuda:0",
        model_fingerprint="model",
        runtime_provenance={
            "device": "cuda:0",
            "precision": "float16-autocast",
        },
    )

    result = encoder.encode(np.zeros((2, 4, 4, 3), dtype=np.uint8))

    np.testing.assert_array_equal(result, np.ones((2, 3), dtype=np.float32))
    assert calls["transfer"] == ("cuda:0", True)
    assert calls["autocast"] == {
        "device_type": "cuda",
        "dtype": float16,
        "enabled": True,
    }


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
