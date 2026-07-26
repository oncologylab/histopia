from __future__ import annotations

import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np

import histopia.semantic._uni2h as uni2h
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


def test_uni2h_loads_config_and_weights_from_fingerprinted_revision(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model_ids: list[str] = []

    class Model:
        pretrained_cfg = {}

        def eval(self):
            return self

        def to(self, device: str):
            assert device == "cuda:0"
            return self

    timm = ModuleType("timm")
    timm.layers = SimpleNamespace(SwiGLUPacked=object())

    def create_model(model_id: str, *, pretrained: bool, **kwargs):
        assert pretrained
        assert kwargs["cache_dir"] == str(tmp_path)
        model_ids.append(model_id)
        return Model()

    timm.create_model = create_model
    timm_data = ModuleType("timm.data")
    timm_data.resolve_data_config = lambda config, *, model: {}
    timm_transforms = ModuleType("timm.data.transforms_factory")
    timm_transforms.create_transform = lambda **kwargs: object()
    monkeypatch.setitem(sys.modules, "timm", timm)
    monkeypatch.setitem(sys.modules, "timm.data", timm_data)
    monkeypatch.setitem(sys.modules, "timm.data.transforms_factory", timm_transforms)
    monkeypatch.setattr(
        uni2h,
        "_prepare_uni2h_runtime",
        lambda *args, **kwargs: _Uni2hRuntime(
            cache_dir=tmp_path,
            device="cuda:0",
            revision="abc123",
            model_fingerprint="model",
            provenance={
                "device": "cuda:0",
                "precision": "bfloat16-autocast",
            },
            torch=SimpleNamespace(nn=SimpleNamespace(SiLU=object())),
        ),
    )

    encoder = Uni2hEncoder.from_cache(tmp_path)

    assert model_ids == ["hf-hub:MahmoodLab/UNI2-h@abc123"]
    assert encoder.model_fingerprint == "model"


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


def test_uni2h_encoder_batches_standard_224_pixel_transform(monkeypatch) -> None:
    calls: dict[str, object] = {}
    float32 = object()

    class Batch:
        def __init__(self, values: np.ndarray) -> None:
            self.values = values

        def permute(self, *axes: int):
            self.values = self.values.transpose(axes)
            return self

        def to(self, *args, **kwargs):
            if "dtype" in kwargs:
                assert kwargs["dtype"] is float32
                self.values = self.values.astype(np.float32)
            else:
                calls["transfer"] = (args, kwargs)
            return self

        def div_(self, value: int):
            self.values /= value
            return self

    class Output:
        def float(self):
            return self

        def cpu(self):
            return self

        def numpy(self) -> np.ndarray:
            return np.ones((2, 3), dtype=np.float32)

    def transform(batch: Batch) -> Batch:
        calls["transform_shape"] = batch.values.shape
        calls["transform_range"] = (
            float(batch.values.min()),
            float(batch.values.max()),
        )
        return batch

    torch = SimpleNamespace(
        OutOfMemoryError=RuntimeError,
        autocast=lambda **kwargs: nullcontext(),
        bfloat16=object(),
        cuda=SimpleNamespace(),
        float16=object(),
        float32=float32,
        from_numpy=lambda values: Batch(values),
        inference_mode=nullcontext,
        stack=lambda tensors: (_ for _ in ()).throw(
            AssertionError("standard patches must use the batched transform")
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    encoder = Uni2hEncoder(
        lambda batch: Output(),
        transform,
        device="cpu",
        model_fingerprint="model",
        runtime_provenance={"device": "cpu", "precision": "float32"},
    )
    images = np.stack(
        [
            np.zeros((224, 224, 3), dtype=np.uint8),
            np.full((224, 224, 3), 255, dtype=np.uint8),
        ]
    )

    result = encoder.encode(images)

    np.testing.assert_array_equal(result, np.ones((2, 3), dtype=np.float32))
    assert calls["transform_shape"] == (2, 3, 224, 224)
    assert calls["transform_range"] == (0.0, 1.0)
    assert calls["transfer"] == (("cpu",), {"non_blocking": False})


def test_uni2h_encoder_transfers_compact_cuda_batch_before_transform(
    monkeypatch,
) -> None:
    calls: list[tuple[str, object]] = []
    float32 = object()

    class Batch:
        def __init__(self, values: np.ndarray, location: str = "cpu") -> None:
            self.values = values
            self.location = location

        def __len__(self) -> int:
            return len(self.values)

        def permute(self, *axes: int):
            self.values = self.values.transpose(axes)
            return self

        def to(self, *args, **kwargs):
            if args:
                assert args == ("cuda:0",)
                assert kwargs == {"non_blocking": True}
                calls.append(("transfer", (self.values.dtype, self.values.shape)))
                self.location = "cuda:0"
            else:
                assert kwargs == {"dtype": float32}
                calls.append(("convert", self.location))
                self.values = self.values.astype(np.float32)
            return self

        def div_(self, value: int):
            self.values /= value
            return self

    class Output:
        def float(self):
            return self

        def cpu(self):
            return self

        def numpy(self) -> np.ndarray:
            return np.ones((2, 3), dtype=np.float32)

    def transform(batch: Batch) -> Batch:
        calls.append(("transform", batch.location))
        assert batch.values.dtype == np.float32
        return batch

    def model(batch: Batch) -> Output:
        calls.append(("model", batch.location))
        return Output()

    torch = SimpleNamespace(
        OutOfMemoryError=RuntimeError,
        autocast=lambda **kwargs: nullcontext(),
        bfloat16=object(),
        cuda=SimpleNamespace(),
        float16=object(),
        float32=float32,
        from_numpy=lambda values: Batch(values),
        inference_mode=nullcontext,
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    encoder = Uni2hEncoder(
        model,
        transform,
        device="cuda:0",
        model_fingerprint="model",
        runtime_provenance={
            "device": "cuda:0",
            "precision": "bfloat16-autocast",
        },
    )

    result = encoder.encode(np.zeros((2, 224, 224, 3), dtype=np.uint8))

    np.testing.assert_array_equal(result, np.ones((2, 3), dtype=np.float32))
    assert calls == [
        ("transfer", (np.dtype("uint8"), (2, 3, 224, 224))),
        ("convert", "cuda:0"),
        ("transform", "cuda:0"),
        ("model", "cuda:0"),
    ]


def test_uni2h_cuda_compact_batch_retries_after_oom(monkeypatch) -> None:
    transfers: list[list[int]] = []
    attempts: list[list[int]] = []
    cache_clears = 0
    float32 = object()

    class OutOfMemoryError(RuntimeError):
        pass

    class Batch:
        def __init__(self, values: np.ndarray) -> None:
            self.values = values

        def __len__(self) -> int:
            return len(self.values)

        def __getitem__(self, index):
            return Batch(self.values[index])

        def permute(self, *axes: int):
            self.values = self.values.transpose(axes)
            return self

        def to(self, *args, **kwargs):
            if args:
                assert args == ("cuda:0",)
                assert kwargs == {"non_blocking": True}
                transfers.append(self.values[:, 0, 0, 0].tolist())
                return Batch(self.values.copy())
            else:
                assert kwargs == {"dtype": float32}
                return Batch(self.values.astype(np.float32))

        def div_(self, value: int):
            self.values /= value
            return self

    class Output:
        def __init__(self, values: list[int]) -> None:
            self.values = values

        def float(self):
            return self

        def cpu(self):
            return self

        def numpy(self) -> np.ndarray:
            return np.asarray(self.values, dtype=np.float32)[:, None]

    def model(batch: Batch) -> Output:
        values = np.rint(batch.values[:, 0, 0, 0] * 255).astype(int).tolist()
        attempts.append(values)
        if len(values) > 2:
            raise OutOfMemoryError
        return Output(values)

    def empty_cache() -> None:
        nonlocal cache_clears
        cache_clears += 1

    torch = SimpleNamespace(
        OutOfMemoryError=OutOfMemoryError,
        autocast=lambda **kwargs: nullcontext(),
        bfloat16=object(),
        cuda=SimpleNamespace(empty_cache=empty_cache),
        float16=object(),
        float32=float32,
        from_numpy=lambda values: Batch(values),
        inference_mode=nullcontext,
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    encoder = Uni2hEncoder(
        model,
        lambda batch: batch,
        device="cuda:0",
        model_fingerprint="model",
        runtime_provenance={
            "device": "cuda:0",
            "precision": "bfloat16-autocast",
        },
    )
    images = np.stack(
        [np.full((224, 224, 3), value, dtype=np.uint8) for value in range(5)]
    )

    result = encoder.encode(images)

    np.testing.assert_array_equal(result[:, 0], np.arange(5, dtype=np.float32))
    expected_attempts = [[0, 1, 2, 3, 4], [0, 1], [2, 3, 4], [2], [3, 4]]
    assert transfers == expected_attempts
    assert attempts == expected_attempts
    assert cache_clears == 2


def test_uni2h_oom_retry_reuses_transforms_and_preserves_order(monkeypatch) -> None:
    transformed: list[int] = []
    attempts: list[list[int]] = []
    cache_clears = 0

    class OutOfMemoryError(RuntimeError):
        pass

    class Batch:
        def __init__(self, values: list[int]) -> None:
            self.values = values

        def __len__(self) -> int:
            return len(self.values)

        def __getitem__(self, index):
            return Batch(self.values[index])

        def to(self, device: str, *, non_blocking: bool):
            assert (device, non_blocking) == ("cuda:0", True)
            return self

    class Output:
        def __init__(self, values: list[int]) -> None:
            self.values = values

        def float(self):
            return self

        def cpu(self):
            return self

        def numpy(self) -> np.ndarray:
            return np.asarray(self.values, dtype=np.float32)[:, None]

    def model(batch: Batch) -> Output:
        attempts.append(batch.values)
        if len(batch) > 2:
            raise OutOfMemoryError
        return Output(batch.values)

    def transform(image) -> int:
        value = image.getpixel((0, 0))[0]
        transformed.append(value)
        return value

    def empty_cache() -> None:
        nonlocal cache_clears
        cache_clears += 1

    torch = SimpleNamespace(
        OutOfMemoryError=OutOfMemoryError,
        autocast=lambda **kwargs: nullcontext(),
        bfloat16=object(),
        cuda=SimpleNamespace(empty_cache=empty_cache),
        float16=object(),
        inference_mode=nullcontext,
        stack=lambda tensors: Batch(list(tensors)),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    encoder = Uni2hEncoder(
        model,
        transform,
        device="cuda:0",
        model_fingerprint="model",
        runtime_provenance={
            "device": "cuda:0",
            "precision": "bfloat16-autocast",
        },
    )
    images = np.stack([np.full((4, 4, 3), value, dtype=np.uint8) for value in range(5)])

    result = encoder.encode(images)

    np.testing.assert_array_equal(result[:, 0], np.arange(5, dtype=np.float32))
    assert transformed == [0, 1, 2, 3, 4]
    assert attempts == [[0, 1, 2, 3, 4], [0, 1], [2, 3, 4], [2], [3, 4]]
    assert cache_clears == 2


def test_lazy_encoder_loads_weights_only_for_first_encode(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[tuple[Path, str]] = []
    runtime = _Uni2hRuntime(
        cache_dir=tmp_path,
        device="cuda:0",
        revision="abc123",
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
