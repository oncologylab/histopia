"""Optional UNI2-h encoder loaded from a local Hugging Face cache."""

from __future__ import annotations

import os
from collections.abc import Callable
from importlib import import_module
from pathlib import Path

import numpy as np


def _preload_wsi_backend(
    importer: Callable[[str], object] = import_module,
) -> None:
    """Load libvips before Torchvision in mixed native-library environments."""

    try:
        importer("pyvips")
    except ImportError as exc:
        raise RuntimeError("UNI2-h WSI extraction requires the 'uni2h' extra") from exc


class Uni2hEncoder:
    """Batch encoder matching the published UNI2-h timm model contract."""

    def __init__(self, model, transform, *, device: str) -> None:
        self.model = model
        self.transform = transform
        self.device = device

    @classmethod
    def from_cache(
        cls,
        cache_dir: Path | str,
        *,
        device: str = "cuda",
        local_only: bool = True,
    ) -> Uni2hEncoder:
        """Load gated weights from an external cache; never package model data."""

        _preload_wsi_backend()
        cache_dir = Path(cache_dir).expanduser().resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(cache_dir)
        if local_only:
            os.environ["HF_HUB_OFFLINE"] = "1"
        else:
            os.environ.pop("HF_HUB_OFFLINE", None)
        try:
            import timm
            import torch
            from timm.data import resolve_data_config
            from timm.data.transforms_factory import create_transform
        except ImportError as exc:
            raise RuntimeError("UNI2-h extraction requires the 'uni2h' extra") from exc
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
            "cache_dir": str(cache_dir),
        }
        try:
            model = timm.create_model(
                "hf-hub:MahmoodLab/UNI2-h", pretrained=True, **kwargs
            )
        except Exception as exc:
            mode = "local cache" if local_only else "Hugging Face access"
            raise RuntimeError(
                f"UNI2-h weights unavailable from {mode} at {cache_dir}; "
                "accept the model license and populate this external cache first"
            ) from exc
        model.eval().to(device)
        transform = create_transform(
            **resolve_data_config(model.pretrained_cfg, model=model)
        )
        return cls(model, transform, device=device)

    def encode(self, images: np.ndarray) -> np.ndarray:
        try:
            import torch
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("UNI2-h extraction requires the 'uni2h' extra") from exc
        tensors = [
            self.transform(Image.fromarray(image, mode="RGB")) for image in images
        ]
        batch = torch.stack(tensors).to(self.device, non_blocking=True)
        try:
            with (
                torch.inference_mode(),
                torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=self.device.startswith("cuda"),
                ),
            ):
                output = self.model(batch)
        except torch.OutOfMemoryError:
            if len(images) == 1:
                raise
            midpoint = len(images) // 2
            torch.cuda.empty_cache()
            return np.concatenate(
                [self.encode(images[:midpoint]), self.encode(images[midpoint:])]
            )
        return output.float().cpu().numpy()
