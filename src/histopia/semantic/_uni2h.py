"""Optional UNI2-h encoder loaded from a local Hugging Face cache."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

from histopia.compute import resolve_compute_device
from histopia.semantic._vips import configure_vips_threads


def _preload_wsi_backend(
    importer: Callable[[str], object] = import_module,
    *,
    vips_threads: int | None = None,
) -> None:
    """Load libvips before Torchvision in mixed native-library environments."""

    configure_vips_threads(vips_threads)
    try:
        importer("pyvips")
    except ImportError as exc:
        raise RuntimeError("UNI2-h WSI extraction requires the 'uni2h' extra") from exc


class Uni2hEncoder:
    """Batch encoder matching the published UNI2-h timm model contract."""

    def __init__(
        self,
        model,
        transform,
        *,
        device: str,
        model_fingerprint: str,
        runtime_provenance: dict[str, object] | None = None,
    ) -> None:
        self.model = model
        self.transform = transform
        self.device = device
        self.model_fingerprint = model_fingerprint
        self.runtime_provenance = runtime_provenance or {"device": device}

    @classmethod
    def from_cache(
        cls,
        cache_dir: Path | str,
        *,
        device: str = "auto",
        local_only: bool = True,
        vips_threads: int | None = None,
    ) -> Uni2hEncoder:
        """Load gated weights from an external cache; never package model data."""

        runtime = _prepare_uni2h_runtime(
            cache_dir,
            device=device,
            local_only=local_only,
            vips_threads=vips_threads,
        )
        try:
            import timm
            from timm.data import resolve_data_config
            from timm.data.transforms_factory import create_transform
        except ImportError as exc:
            raise RuntimeError("UNI2-h extraction requires the 'uni2h' extra") from exc
        torch = runtime.torch
        kwargs = {
            "img_size": 224,
            "patch_size": 14,
            "depth": 24,
            "num_heads": 24,
            "init_values": 1e-5,
            "embed_dim": 1536,
            "mlp_ratio": 2.66667 * 2,
            "num_classes": 0,
            "no_embed_class": True,
            "mlp_layer": timm.layers.SwiGLUPacked,
            "act_layer": torch.nn.SiLU,
            "reg_tokens": 8,
            "dynamic_img_size": True,
            "cache_dir": str(runtime.cache_dir),
        }
        try:
            model = timm.create_model(
                "hf-hub:MahmoodLab/UNI2-h", pretrained=True, **kwargs
            )
        except Exception as exc:
            mode = "local cache" if local_only else "Hugging Face access"
            raise RuntimeError(
                f"UNI2-h weights unavailable from {mode} at {runtime.cache_dir}; "
                "accept the model license and populate this external cache first"
            ) from exc
        model.eval().to(runtime.device)
        transform = create_transform(
            **resolve_data_config(model.pretrained_cfg, model=model)
        )
        return cls(
            model,
            transform,
            device=runtime.device,
            model_fingerprint=runtime.model_fingerprint,
            runtime_provenance=runtime.provenance,
        )

    @classmethod
    def lazy_from_cache(
        cls,
        cache_dir: Path | str,
        *,
        device: str = "auto",
        local_only: bool = True,
        vips_threads: int | None = None,
    ) -> _LazyUni2hEncoder:
        """Resolve exact provenance now and load model weights only when needed."""

        runtime = _prepare_uni2h_runtime(
            cache_dir,
            device=device,
            local_only=local_only,
            vips_threads=vips_threads,
        )
        return _LazyUni2hEncoder(
            cache_dir=runtime.cache_dir,
            device=runtime.device,
            local_only=local_only,
            vips_threads=vips_threads,
            runtime=runtime,
        )

    def encode(self, images: np.ndarray) -> np.ndarray:
        try:
            import torch
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("UNI2-h extraction requires the 'uni2h' extra") from exc
        tensors = [
            self.transform(Image.fromarray(image, mode="RGB")) for image in images
        ]
        uses_cuda = self.device.startswith("cuda")
        batch = torch.stack(tensors).to(self.device, non_blocking=uses_cuda)
        try:
            with (
                torch.inference_mode(),
                torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=uses_cuda,
                ),
            ):
                output = self.model(batch)
        except torch.OutOfMemoryError:
            if len(images) == 1:
                raise
            midpoint = len(images) // 2
            if uses_cuda:
                torch.cuda.empty_cache()
            return np.concatenate(
                [self.encode(images[:midpoint]), self.encode(images[midpoint:])]
            )
        return output.float().cpu().numpy()


@dataclass(frozen=True, slots=True)
class _Uni2hRuntime:
    cache_dir: Path
    device: str
    model_fingerprint: str
    provenance: dict[str, object]
    torch: Any


class _LazyUni2hEncoder:
    """Patch encoder that materializes UNI2-h only on its first cache miss."""

    def __init__(
        self,
        *,
        cache_dir: Path,
        device: str,
        local_only: bool,
        vips_threads: int | None,
        runtime: _Uni2hRuntime,
    ) -> None:
        self.device = device
        self.model_fingerprint = runtime.model_fingerprint
        self.runtime_provenance = runtime.provenance
        self._cache_dir = cache_dir
        self._local_only = local_only
        self._vips_threads = vips_threads
        self._encoder: Uni2hEncoder | None = None

    def encode(self, images: np.ndarray) -> np.ndarray:
        if self._encoder is None:
            self._encoder = Uni2hEncoder.from_cache(
                self._cache_dir,
                device=self.device,
                local_only=self._local_only,
                vips_threads=self._vips_threads,
            )
            if (
                self._encoder.model_fingerprint != self.model_fingerprint
                or self._encoder.runtime_provenance != self.runtime_provenance
            ):
                raise RuntimeError("UNI2-h runtime changed during lazy model loading")
        return self._encoder.encode(images)


def _prepare_uni2h_runtime(
    cache_dir: Path | str,
    *,
    device: str,
    local_only: bool,
    vips_threads: int | None,
) -> _Uni2hRuntime:
    _preload_wsi_backend(vips_threads=vips_threads)
    cache_dir = Path(cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    revision = _cached_model_revision(cache_dir)
    os.environ["HF_HOME"] = str(cache_dir)
    if local_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
    else:
        os.environ.pop("HF_HUB_OFFLINE", None)
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("UNI2-h extraction requires the 'uni2h' extra") from exc
    resolved_device = resolve_compute_device(device, torch_module=torch).resolved
    identity = f"MahmoodLab/UNI2-h@{revision}".encode()
    provenance: dict[str, object] = {
        "device": resolved_device,
        "precision": (
            "bfloat16-autocast" if resolved_device.startswith("cuda") else "float32"
        ),
        "packages": {
            package: _package_version(package)
            for package in (
                "numpy",
                "pillow",
                "pyvips",
                "timm",
                "torch",
                "torchvision",
            )
        },
        "libvips": _libvips_version(),
        "cuda_runtime": str(torch.version.cuda or ""),
    }
    if resolved_device.startswith("cuda"):
        device_index = torch.device(resolved_device).index or 0
        provenance["accelerator"] = {
            "name": torch.cuda.get_device_name(device_index),
            "compute_capability": list(torch.cuda.get_device_capability(device_index)),
        }
    return _Uni2hRuntime(
        cache_dir=cache_dir,
        device=resolved_device,
        model_fingerprint=hashlib.sha256(identity).hexdigest(),
        provenance=provenance,
        torch=torch,
    )


def _cached_model_revision(cache_dir: Path) -> str:
    model_root = cache_dir / "models--MahmoodLab--UNI2-h"
    reference = model_root / "refs" / "main"
    if not reference.is_file():
        raise RuntimeError("UNI2-h cache has no pinned main revision")
    revision = reference.read_text().strip()
    if not revision or not (model_root / "snapshots" / revision).is_dir():
        raise RuntimeError("UNI2-h cache revision has no local snapshot")
    return revision


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "unavailable"


def _libvips_version() -> str:
    try:
        import pyvips
    except ImportError:
        return "unavailable"
    return ".".join(str(pyvips.version(index)) for index in range(3))
