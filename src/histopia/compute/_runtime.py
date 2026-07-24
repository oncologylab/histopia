"""Runtime selection without importing accelerator frameworks at package import."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import import_module
from typing import Any


@dataclass(frozen=True, slots=True)
class ComputeDevice:
    """Resolved execution device and relevant hardware metadata."""

    requested: str
    resolved: str
    backend: str
    accelerator_name: str | None = None

    def to_json_dict(self) -> dict[str, str | None]:
        return asdict(self)


def resolve_compute_device(
    requested: str = "auto",
    *,
    torch_module: Any | None = None,
) -> ComputeDevice:
    """Resolve an explicit or automatic Torch device with availability checks."""

    value = requested.strip().lower()
    if not value:
        raise ValueError("compute device must not be empty")
    if value != "auto" and value != "cpu" and value != "mps":
        if value != "cuda" and not value.startswith("cuda:"):
            raise ValueError("device must be auto, cpu, cuda, cuda:N, or mps")

    torch = torch_module
    if torch is None:
        try:
            torch = import_module("torch")
        except ImportError:
            if value in {"auto", "cpu"}:
                return ComputeDevice(value, "cpu", "cpu")
            raise RuntimeError(
                f"requested device {value!r} requires PyTorch and its accelerator "
                "runtime"
            ) from None

    if value == "auto":
        if torch.cuda.is_available():
            value = "cuda"
        elif _mps_available(torch):
            value = "mps"
        else:
            value = "cpu"

    if value.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available to PyTorch")
        index = _cuda_index(value)
        count = int(torch.cuda.device_count())
        if index >= count:
            raise RuntimeError(
                f"CUDA device {index} was requested, but only {count} device(s) "
                "are available"
            )
        resolved = f"cuda:{index}"
        name = str(torch.cuda.get_device_name(index))
        return ComputeDevice(requested, resolved, "cuda", name)

    if value == "mps":
        if not _mps_available(torch):
            raise RuntimeError("MPS was requested but is not available to PyTorch")
        return ComputeDevice(requested, "mps", "mps", "Apple Metal")
    return ComputeDevice(requested, "cpu", "cpu")


def inspect_compute(
    requested_device: str = "auto",
    *,
    torch_module: Any | None = None,
) -> dict[str, object]:
    """Return capabilities and validate the device intended for a run."""

    torch = torch_module
    if torch is None:
        try:
            torch = import_module("torch")
        except ImportError:
            selected = resolve_compute_device(requested_device, torch_module=None)
            return {
                "torch_available": False,
                "automatic_device": ComputeDevice("auto", "cpu", "cpu").to_json_dict(),
                "selected_device": selected.to_json_dict(),
                "cuda_devices": [],
                "mps_available": False,
            }
    cuda_devices = []
    if torch.cuda.is_available():
        for index in range(int(torch.cuda.device_count())):
            properties = torch.cuda.get_device_properties(index)
            cuda_devices.append(
                {
                    "index": index,
                    "name": str(torch.cuda.get_device_name(index)),
                    "total_memory_bytes": int(properties.total_memory),
                }
            )
    return {
        "torch_available": True,
        "torch_version": str(torch.__version__),
        "automatic_device": resolve_compute_device(
            "auto", torch_module=torch
        ).to_json_dict(),
        "selected_device": resolve_compute_device(
            requested_device, torch_module=torch
        ).to_json_dict(),
        "cuda_devices": cuda_devices,
        "mps_available": _mps_available(torch),
    }


def _mps_available(torch: Any) -> bool:
    backend = getattr(getattr(torch, "backends", None), "mps", None)
    return bool(backend is not None and backend.is_available())


def _cuda_index(value: str) -> int:
    if value == "cuda":
        return 0
    _, _, raw_index = value.partition(":")
    if not raw_index.isdigit():
        raise ValueError("CUDA device must use the form cuda or cuda:N")
    return int(raw_index)
