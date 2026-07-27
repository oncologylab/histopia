"""Registration-aware extraction from source whole-slide images."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from zipfile import BadZipFile

import numpy as np

from histopia._vips_image import normalize_vips_rgb_uchar
from histopia.compute import configure_vips_threads
from histopia.registration._slides import SlideGeometry
from histopia.semantic._config import SemanticAtlasConfig
from histopia.semantic._features import (
    PatchEncoder,
    PatchFeatures,
    extract_patch_features,
)
from histopia.semantic._performance import (
    elapsed_seconds,
    utc_timestamp,
    write_performance_stage,
)
from histopia.semantic._preflight import (
    SemanticPreflight,
    preflight_registration,
    write_preflight,
)

_EXTRACTION_METHOD = "histopia-source-grid-v2"


def extract_registration_features(
    config: SemanticAtlasConfig,
    encoder: PatchEncoder,
    *,
    preflight: SemanticPreflight | None = None,
    overwrite: bool = False,
    progress: Callable[[str], None] | None = None,
) -> tuple[Path, ...]:
    """Extract compact features in accepted registration section order."""

    configure_vips_threads(config.vips_threads)
    registration_path = config.registration_run / "registration_result.json"
    payload = json.loads(registration_path.read_text())
    slides = payload["slides"]
    if preflight is None:
        preflight = preflight_registration(config.registration_run)
    write_preflight(preflight, config.output_dir / "preflight.json")
    preflight_slides = {slide.slide_name: slide for slide in preflight.slides}
    model_fingerprint = getattr(encoder, "model_fingerprint", None)
    if not model_fingerprint:
        raise ValueError("encoder must expose a model_fingerprint")
    runtime_provenance = getattr(
        encoder,
        "runtime_provenance",
        {"device": getattr(encoder, "device", config.device)},
    )
    reference = next(slide for slide in slides if slide["is_reference"])
    reference_geometry = _geometry_from_json(reference["geometry"])
    feature_dir = config.output_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    extraction_started = time.perf_counter()
    performance_rows: list[dict[str, object]] = []
    performance: dict[str, object] = {
        "status": "running",
        "started_at": utc_timestamp(),
        "preflight_fingerprint": preflight.fingerprint,
        "slide_count": len(slides),
        "completed_slides": 0,
        "cached_slides": 0,
        "extracted_slides": 0,
        "total_patches": 0,
        "elapsed_seconds": 0.0,
        "controls": {
            "batch_size": config.batch_size,
            "patch_workers": config.patch_workers,
            "vips_threads": config.vips_threads,
            **_safe_encoder_runtime(runtime_provenance, config.device),
        },
        "slides": performance_rows,
    }

    def checkpoint() -> None:
        performance["completed_slides"] = len(performance_rows)
        performance["cached_slides"] = sum(
            row["status"] == "cached" for row in performance_rows
        )
        performance["extracted_slides"] = sum(
            row["status"] == "extracted" for row in performance_rows
        )
        performance["total_patches"] = sum(
            int(row["patches"]) for row in performance_rows
        )
        performance["elapsed_seconds"] = elapsed_seconds(extraction_started)
        write_performance_stage(config.output_dir, "extraction", performance)

    write_performance_stage(
        config.output_dir,
        "extraction",
        performance,
        clear_stages=("fit",),
    )
    try:
        for order, slide in enumerate(slides, start=1):
            slide_started = time.perf_counter()
            slide_path = Path(slide["path"])
            output = feature_artifact_path(feature_dir, order, slide_path.name)
            output_paths.append(output)
            source = preflight_slides[slide_path.name]
            provenance = {
                "preflight_fingerprint": preflight.fingerprint,
                "slide_name": source.slide_name,
                "source_sha256": source.source_sha256,
                "mask_sha256": source.mask_sha256,
                "transform_sha256": source.transform_sha256,
                "model_fingerprint": str(model_fingerprint),
                "analysis_mpp": config.analysis_mpp,
                "patch_size_px": config.patch_size_px,
                "min_tissue_fraction": config.min_tissue_fraction,
                "batch_size": config.batch_size,
                "encoder_runtime": runtime_provenance,
                "extraction_method": _EXTRACTION_METHOD,
                "patch_reader": _VipsPatchReader.provenance_id,
            }
            cached = (
                _matching_feature_cache(output, provenance)
                if output.exists() and not overwrite
                else None
            )
            if cached is not None:
                elapsed = elapsed_seconds(slide_started)
                performance_rows.append(
                    _performance_slide_row(
                        order,
                        slide_path.name,
                        status="cached",
                        patches=len(cached.features),
                        elapsed_seconds=elapsed,
                    )
                )
                checkpoint()
                if progress is not None:
                    progress(f"[{order}/{len(slides)}] cached {slide_path.name}")
                continue
            if progress is not None:
                progress(f"[{order}/{len(slides)}] extracting {slide_path.name}")
            geometry = _geometry_from_json(slide["geometry"])
            mask = _read_mask(
                config.registration_run / "processed" / f"{slide_path.stem}.mask.png"
            )
            reader = _VipsPatchReader(slide_path)
            artifact = extract_patch_features(
                slide_id=slide_path.name,
                geometry=geometry,
                tissue_mask=mask,
                moving_to_reference_thumbnail=np.asarray(
                    slide["transform"]["matrix"], dtype=float
                ),
                reference_geometry=reference_geometry,
                reader=reader,
                encoder=encoder,
                analysis_mpp=config.analysis_mpp,
                patch_size_px=config.patch_size_px,
                min_tissue_fraction=config.min_tissue_fraction,
                batch_size=config.batch_size,
                patch_workers=config.patch_workers,
                provenance=provenance,
            )
            artifact.save(output)
            elapsed = elapsed_seconds(slide_started)
            performance_rows.append(
                _performance_slide_row(
                    order,
                    slide_path.name,
                    status="extracted",
                    patches=len(artifact.features),
                    elapsed_seconds=elapsed,
                )
            )
            checkpoint()
            if progress is not None:
                progress(
                    f"[{order}/{len(slides)}] completed {slide_path.name}: "
                    f"{len(artifact.features):,} patches in {elapsed:.1f}s"
                )
    except BaseException as exc:
        performance["status"] = (
            "interrupted"
            if isinstance(exc, (KeyboardInterrupt, SystemExit))
            else "failed"
        )
        performance["failure_type"] = type(exc).__name__
        performance["completed_at"] = utc_timestamp()
        checkpoint()
        raise
    performance["status"] = "completed"
    performance["completed_at"] = utc_timestamp()
    checkpoint()
    return tuple(output_paths)


def feature_cache_matches(
    path: Path | str, expected_provenance: dict[str, object]
) -> bool:
    """Return whether an artifact is a valid cache for exact campaign inputs."""

    return _matching_feature_cache(path, expected_provenance) is not None


def _matching_feature_cache(
    path: Path | str, expected_provenance: dict[str, object]
) -> PatchFeatures | None:
    try:
        artifact = PatchFeatures.load(path)
    except (BadZipFile, EOFError, KeyError, OSError, ValueError):
        return None
    matches = (
        artifact.fingerprint is not None
        and artifact.content_fingerprint is not None
        and artifact.provenance == expected_provenance
    )
    return artifact if matches else None


def _safe_encoder_runtime(
    runtime: object,
    fallback_device: str,
) -> dict[str, object]:
    values = runtime if isinstance(runtime, dict) else {}
    summary: dict[str, object] = {
        "device": str(values.get("device", fallback_device)),
    }
    precision = values.get("precision")
    if isinstance(precision, str):
        summary["precision"] = precision
    input_pipeline = values.get("input_pipeline")
    if isinstance(input_pipeline, str):
        summary["input_pipeline"] = input_pipeline
    accelerator = values.get("accelerator")
    if isinstance(accelerator, dict):
        safe_accelerator = {
            key: accelerator[key]
            for key in ("name", "compute_capability")
            if isinstance(accelerator.get(key), (str, list, tuple))
        }
        if safe_accelerator:
            summary["accelerator"] = safe_accelerator
    return summary


def _performance_slide_row(
    order: int,
    slide_id: str,
    *,
    status: str,
    patches: int,
    elapsed_seconds: float,
) -> dict[str, object]:
    return {
        "order": order,
        "slide_id": slide_id,
        "status": status,
        "patches": patches,
        "elapsed_seconds": elapsed_seconds,
        "patches_per_second": round(patches / max(elapsed_seconds, 1e-9), 3),
    }


def _geometry_from_json(data: dict[str, Any]) -> SlideGeometry:
    return SlideGeometry.from_json_dict(data)


class _VipsPatchReader:
    provenance_id = "pyvips-context-row-batch-v2"

    def __init__(self, path: Path) -> None:
        try:
            import pyvips
        except ImportError as exc:
            raise RuntimeError(
                "WSI feature extraction requires the 'wsi' extra"
            ) from exc
        self.image = pyvips.Image.new_from_file(str(path), access="random")

    def __call__(
        self, x: int, y: int, width: int, height: int, output_px: int
    ) -> np.ndarray:
        image = self.image.crop(x, y, width, height)
        image = image.resize(output_px / width, vscale=output_px / height)
        return self._as_rgb(image)

    def read_many(
        self, requests: tuple[tuple[int, int, int, int, int], ...]
    ) -> tuple[np.ndarray, ...]:
        """Decode batch-invariant grid patches from context-padded row strips."""

        if not requests:
            return ()
        groups: dict[
            tuple[int, int, int, int],
            list[tuple[int, tuple[int, int, int, int, int]]],
        ] = {}
        for index, request in enumerate(requests):
            x, y, width, height, output_px = request
            groups.setdefault((y, width, height, output_px), []).append(
                (index, request)
            )
        patches: list[np.ndarray | None] = [None] * len(requests)
        for (y, width, height, output_px), items in groups.items():
            minimum = min(request[0] for _, request in items)
            maximum = max(request[0] + width for _, request in items)
            left = minimum - width if minimum >= width else minimum
            right = maximum + width if maximum + width <= self.image.width else maximum
            strip = self.image.crop(left, y, right - left, height)
            strip = strip.resize(
                output_px / width,
                vscale=output_px / height,
            )
            array = self._as_rgb(strip)
            for index, request in items:
                start = round((request[0] - left) * output_px / width)
                patch = array[:, start : start + output_px]
                if patch.shape != (output_px, output_px, 3):
                    patch = self(*request)
                patches[index] = patch
        if any(patch is None for patch in patches):
            raise RuntimeError("batch patch reader did not fill every request")
        return tuple(patch for patch in patches if patch is not None)

    @staticmethod
    def _as_rgb(image: Any) -> np.ndarray:
        image = normalize_vips_rgb_uchar(image)
        return np.frombuffer(image.write_to_memory(), dtype=np.uint8).reshape(
            image.height, image.width, image.bands
        )


def _read_mask(path: Path) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("semantic extraction requires the 'semantic' extra") from exc
    with Image.open(path) as image:
        return np.asarray(image.convert("L")) > 127


def _safe_stem(stem: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in stem
    )


def feature_artifact_path(
    feature_dir: Path | str,
    order: int,
    slide_name: str,
) -> Path:
    """Return the deterministic artifact path for one ordered section."""

    if order <= 0:
        raise ValueError("feature artifact order must be positive")
    name = Path(slide_name).name
    if not name:
        raise ValueError("feature artifact slide name must not be empty")
    return Path(feature_dir) / f"{order:03d}-{_safe_stem(Path(name).stem)}.npz"
