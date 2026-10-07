"""Narrow Cellpose 4 adapter for Histopia cell-boundary inference."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from histopia.cells._algorithms import (
    clean_mask,
    containment_merge,
    hematoxylin_evidence,
    merge_small_into_large,
    retain_nuclear_supported_instances,
)
from histopia.cells._config import CellSegmentationConfig
from histopia.compute import ComputeDevice, resolve_compute_device


@dataclass(frozen=True, slots=True)
class CellposeRuntime:
    """Loaded model plus exact runtime provenance."""

    model: Any
    device: ComputeDevice
    cellpose_version: str
    weight_path: str
    weight_sha256: str


def load_cellpose_runtime(config: CellSegmentationConfig) -> CellposeRuntime:
    """Load an explicit Cellpose model and record its resolved weight file."""

    try:
        import torch
        from cellpose import models
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "cell segmentation requires Histopia's optional 'cells' dependencies"
        ) from error
    cellpose_version = version("cellpose")
    _require_supported_cellpose_version(cellpose_version, config.model)
    device = resolve_compute_device(config.device, torch_module=torch)
    if config.model_cache is not None:
        config.model_cache.expanduser().resolve().mkdir(parents=True, exist_ok=True)
        models.MODEL_DIR = config.model_cache.expanduser().resolve()
    expected_weight = Path(models.MODEL_DIR).expanduser().resolve() / config.model
    if not expected_weight.is_file() and not config.allow_model_download:
        raise FileNotFoundError(
            f"Cellpose model weight is not cached: {expected_weight}. "
            f"Run 'histopia-cells cache-model --model {config.model}' first."
        )
    model = models.CellposeModel(
        gpu=device.backend == "cuda",
        device=torch.device(device.resolved),
        pretrained_model=config.model,
    )
    weight = _resolve_weight_path(model, models, config.model)
    if not weight.is_file():
        raise FileNotFoundError(f"Cellpose model weight is unavailable: {weight}")
    if weight.name != config.model:
        raise RuntimeError(
            f"Cellpose resolved requested model {config.model!r} to unexpected "
            f"weight {weight.name!r}. Refusing a silent model fallback."
        )
    return CellposeRuntime(
        model=model,
        device=device,
        cellpose_version=cellpose_version,
        weight_path=str(weight),
        weight_sha256=_sha256_file(weight),
    )


def segment_tile(
    image: np.ndarray,
    runtime: CellposeRuntime,
    config: CellSegmentationConfig,
) -> np.ndarray:
    """Run the configured direct, multiscale, Combined, or containment method."""

    return segment_tiles((image,), runtime, config)[0]


def segment_tiles(
    images: tuple[np.ndarray, ...] | list[np.ndarray],
    runtime: CellposeRuntime,
    config: CellSegmentationConfig,
) -> list[np.ndarray]:
    """Run a deterministic tile batch using the validated method profile."""

    rgbs = [np.asarray(image, dtype=np.uint8) for image in images]
    if not rgbs:
        return []
    if any(rgb.ndim != 3 or rgb.shape[2] != 3 for rgb in rgbs):
        raise ValueError("Cellpose input must be an RGB image")
    if config.method == "direct":
        return _direct_masks(
            rgbs,
            runtime.model,
            cellprob=config.first_cellprob,
            diameter=config.first_diameter,
            min_size=config.min_size,
            batch_size=config.inference_batch_size,
            flow_threshold=config.flow_threshold,
        )
    if config.method == "multiscale":
        first = _direct_masks(
            rgbs,
            runtime.model,
            cellprob=config.first_cellprob,
            diameter=config.first_diameter,
            min_size=config.min_size,
            batch_size=config.inference_batch_size,
            flow_threshold=config.flow_threshold,
        )
        second = _direct_masks(
            rgbs,
            runtime.model,
            cellprob=config.second_cellprob,
            diameter=config.second_diameter,
            min_size=config.min_size,
            batch_size=config.inference_batch_size,
            flow_threshold=config.flow_threshold,
        )
        return [
            containment_merge(
                large,
                _retain_nuclear_support(small, rgb, config)
                if config.multiscale_nuclear_support
                else small,
                config.merge_threshold,
            )
            for rgb, large, small in zip(rgbs, first, second, strict=True)
        ]
    first = _combined_masks(
        rgbs,
        runtime.model,
        cellprob=config.first_cellprob,
        diameter=config.first_diameter,
        min_size=config.min_size,
        batch_size=config.inference_batch_size,
        flow_threshold=config.flow_threshold,
    )
    if config.method == "combined":
        return first
    second = _combined_masks(
        rgbs,
        runtime.model,
        cellprob=config.second_cellprob,
        diameter=config.second_diameter,
        min_size=config.min_size,
        batch_size=config.inference_batch_size,
        flow_threshold=config.flow_threshold,
    )
    return [
        containment_merge(
            large,
            _retain_nuclear_support(small, rgb, config)
            if config.multiscale_nuclear_support
            else small,
            config.merge_threshold,
        )
        for rgb, large, small in zip(rgbs, first, second, strict=True)
    ]


def runtime_provenance(runtime: CellposeRuntime) -> dict[str, object]:
    """Return portable runtime metadata without exposing local cache paths."""

    return {
        "cellpose_version": runtime.cellpose_version,
        "weight_name": Path(runtime.weight_path).name,
        "weight_sha256": runtime.weight_sha256,
        "device": runtime.device.to_json_dict(),
    }


def _direct_masks(
    images: list[np.ndarray],
    model: Any,
    *,
    cellprob: float,
    diameter: int,
    min_size: int,
    batch_size: int,
    flow_threshold: float,
) -> list[np.ndarray]:
    masks, _flows, _styles = model.eval(
        images,
        batch_size=int(batch_size),
        diameter=int(diameter),
        flow_threshold=float(flow_threshold),
        cellprob_threshold=float(cellprob),
        min_size=int(min_size),
    )
    return [clean_mask(mask) for mask in _mask_rows(masks, len(images))]


def _combined_masks(
    images: list[np.ndarray],
    model: Any,
    *,
    cellprob: float,
    diameter: int,
    min_size: int,
    batch_size: int,
    flow_threshold: float,
) -> list[np.ndarray]:
    try:
        from cellpose import dynamics
    except ImportError as error:
        raise RuntimeError("Cellpose dynamics module is unavailable") from error
    _masks1, flows1, _styles1 = model.eval(
        images,
        batch_size=int(batch_size),
        diameter=int(diameter),
        flow_threshold=float(flow_threshold),
        cellprob_threshold=0.0,
        compute_masks=False,
    )
    first_flows = _flow_rows(flows1, len(images))
    cellprobs1 = [flow[2] for flow in first_flows]
    _masks2, flows2, _styles2 = model.eval(
        cellprobs1,
        batch_size=int(batch_size),
        diameter=int(diameter),
        flow_threshold=float(flow_threshold),
        cellprob_threshold=0.0,
        compute_masks=False,
    )
    second_flows = _flow_rows(flows2, len(images))
    niter = int(200 / (30.0 / int(diameter)))
    output = []
    for flow1, flow2 in zip(first_flows, second_flows, strict=True):
        one_pass = dynamics.resize_and_compute_masks(
            flow1[1],
            flow1[2],
            niter=niter,
            cellprob_threshold=float(cellprob),
            flow_threshold=float(flow_threshold),
            min_size=int(min_size),
            device=model.device,
        )
        two_pass = dynamics.resize_and_compute_masks(
            flow2[1],
            flow2[2],
            niter=niter,
            cellprob_threshold=float(cellprob),
            flow_threshold=float(flow_threshold),
            min_size=int(min_size),
            device=model.device,
        )
        output.append(clean_mask(merge_small_into_large(one_pass, two_pass)))
    return output


def _retain_nuclear_support(
    labels: np.ndarray,
    rgb: np.ndarray,
    config: CellSegmentationConfig,
) -> np.ndarray:
    return retain_nuclear_supported_instances(
        labels,
        hematoxylin_evidence(
            rgb,
            minimum_optical_density=config.nuclear_minimum_optical_density,
        ),
        minimum_pixels=config.nuclear_minimum_pixels,
        minimum_fraction=config.nuclear_minimum_fraction,
    )


def _mask_rows(value: Any, expected: int) -> list[np.ndarray]:
    if isinstance(value, list) and len(value) == expected:
        return [np.asarray(row) for row in value]
    array = np.asarray(value)
    if expected == 1:
        return [_first_array(value)]
    if array.ndim >= 3 and array.shape[0] == expected:
        return [array[index] for index in range(expected)]
    raise ValueError("Cellpose returned an unexpected mask batch")


def _flow_rows(value: Any, expected: int) -> list[Any]:
    if not isinstance(value, (list, tuple)) or len(value) != expected:
        raise ValueError("Cellpose returned an unexpected flow batch")
    rows = list(value)
    if any(not isinstance(row, (list, tuple)) or len(row) < 3 for row in rows):
        raise ValueError("Cellpose flow output is incompatible with the tested API")
    return rows


def _first_array(value: Any) -> np.ndarray:
    if isinstance(value, list):
        if not value:
            raise ValueError("Cellpose returned no masks")
        return np.asarray(value[0])
    array = np.asarray(value)
    if array.ndim == 3 and array.shape[0] == 1:
        return array[0]
    return array


def _first_flow(value: Any) -> Any:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("Cellpose returned no flow fields")
    candidate = value[0]
    if not isinstance(candidate, (list, tuple)) or len(candidate) < 3:
        raise ValueError("Cellpose flow output is incompatible with the tested API")
    return candidate


def _resolve_weight_path(model: Any, models: Any, name: str) -> Path:
    candidates = getattr(model, "pretrained_model", None)
    if isinstance(candidates, (list, tuple)) and candidates:
        candidates = candidates[0]
    if isinstance(candidates, (str, Path)) and Path(candidates).is_file():
        return Path(candidates).expanduser().resolve()
    return (Path(models.MODEL_DIR).expanduser().resolve() / name).resolve()


def _require_supported_cellpose_version(cellpose_version: str, model: str) -> None:
    """Reject runtimes known to silently substitute legacy Cellpose weights."""

    try:
        major = int(cellpose_version.split(".", maxsplit=1)[0])
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            f"Cannot validate Cellpose version {cellpose_version!r} for model {model!r}"
        ) from error
    if model.startswith("cpsam") and major < 4:
        raise RuntimeError(
            f"Cellpose {cellpose_version} cannot safely load requested model "
            f"{model!r}; Histopia requires Cellpose 4 or newer for CPSAM weights. "
            "Older releases may silently substitute a legacy model."
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
