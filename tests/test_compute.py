from __future__ import annotations

from types import SimpleNamespace

import pytest

from histopia.compute import inspect_compute, resolve_compute_device


class _Cuda:
    def __init__(self, available: bool, count: int = 0) -> None:
        self.available = available
        self.count = count

    def is_available(self) -> bool:
        return self.available

    def device_count(self) -> int:
        return self.count

    def get_device_name(self, index: int) -> str:
        return f"Test GPU {index}"

    def get_device_properties(self, index: int) -> SimpleNamespace:
        return SimpleNamespace(total_memory=(index + 1) * 1024)


def _torch(*, cuda: bool = False, count: int = 0, mps: bool = False):
    return SimpleNamespace(
        __version__="test",
        cuda=_Cuda(cuda, count),
        backends=SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: mps),
        ),
    )


def test_auto_prefers_cuda_and_reports_hardware() -> None:
    torch = _torch(cuda=True, count=2, mps=True)

    device = resolve_compute_device("auto", torch_module=torch)
    report = inspect_compute(torch_module=torch)

    assert device.resolved == "cuda:0"
    assert device.accelerator_name == "Test GPU 0"
    assert [row["name"] for row in report["cuda_devices"]] == [
        "Test GPU 0",
        "Test GPU 1",
    ]
    assert report["selected_device"]["resolved"] == "cuda:0"


def test_auto_falls_back_from_mps_to_cpu() -> None:
    assert (
        resolve_compute_device("auto", torch_module=_torch(mps=True)).resolved == "mps"
    )
    assert resolve_compute_device("auto", torch_module=_torch()).resolved == "cpu"


def test_explicit_unavailable_or_invalid_device_fails() -> None:
    with pytest.raises(RuntimeError, match="CUDA was requested"):
        resolve_compute_device("cuda", torch_module=_torch())
    with pytest.raises(RuntimeError, match="only 1 device"):
        resolve_compute_device("cuda:2", torch_module=_torch(cuda=True, count=1))
    with pytest.raises(ValueError, match="device must be"):
        resolve_compute_device("gpu", torch_module=_torch())
    with pytest.raises(RuntimeError, match="CUDA was requested"):
        inspect_compute("cuda", torch_module=_torch())


def test_inspection_reports_explicit_cpu_selection_on_gpu_machine() -> None:
    report = inspect_compute("cpu", torch_module=_torch(cuda=True, count=1))

    assert report["automatic_device"]["resolved"] == "cuda:0"
    assert report["selected_device"]["resolved"] == "cpu"
