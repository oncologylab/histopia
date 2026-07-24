"""Registration-aware extraction from source whole-slide images."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from histopia.registration._slides import SlideGeometry
from histopia.semantic._config import SemanticAtlasConfig
from histopia.semantic._features import (
    PatchEncoder,
    PatchFeatures,
    extract_patch_features,
)
from histopia.semantic._preflight import (
    SemanticPreflight,
    preflight_registration,
    write_preflight,
)
from histopia.semantic._vips import configure_vips_threads

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
    for order, slide in enumerate(slides, start=1):
        slide_path = Path(slide["path"])
        output = feature_dir / f"{order:03d}-{_safe_stem(slide_path.stem)}.npz"
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
        if (
            output.exists()
            and not overwrite
            and feature_cache_matches(output, provenance)
        ):
            if progress is not None:
                progress(f"[{order}/{len(slides)}] cached {slide_path.name}")
            continue
        if progress is not None:
            progress(f"[{order}/{len(slides)}] extracting {slide_path.name}")
        started = time.perf_counter()
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
        if progress is not None:
            progress(
                f"[{order}/{len(slides)}] completed {slide_path.name}: "
                f"{len(artifact.features):,} patches in "
                f"{time.perf_counter() - started:.1f}s"
            )
    return tuple(output_paths)


def feature_cache_matches(
    path: Path | str, expected_provenance: dict[str, object]
) -> bool:
    """Return whether an artifact is a valid cache for exact campaign inputs."""

    try:
        artifact = PatchFeatures.load(path)
    except (KeyError, OSError, ValueError):
        return False
    return (
        artifact.fingerprint is not None and artifact.provenance == expected_provenance
    )


def _geometry_from_json(data: dict[str, Any]) -> SlideGeometry:
    mpp = data.get("mpp_xy")
    return SlideGeometry(
        native_shape=tuple(int(x) for x in data["native_shape"]),
        content_bbox_xywh=tuple(int(x) for x in data["content_bbox_xywh"]),
        thumbnail_shape=tuple(int(x) for x in data["thumbnail_shape"]),
        bounds_source=str(data["bounds_source"]),
        mpp_xy=tuple(float(x) for x in mpp) if mpp is not None else None,
        mpp_source=str(data.get("mpp_source", "unavailable")),
    )


class _VipsPatchReader:
    provenance_id = "pyvips-row-batch-v1"

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
        """Decode adjacent grid patches as bounded row strips."""

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
            left = min(request[0] for _, request in items)
            right = max(request[0] + width for _, request in items)
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
        if image.bands > 3:
            image = image[:3]
        if image.bands == 1:
            image = image.bandjoin([image, image])
        if image.format != "uchar":
            image = image.cast("uchar")
        return np.frombuffer(image.write_to_memory(), dtype=np.uint8).reshape(
            image.height, image.width, image.bands
        )[..., :3]


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
