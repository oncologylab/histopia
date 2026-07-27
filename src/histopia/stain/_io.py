"""Bounded analysis-resolution WSI reads for stain quantification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from histopia._vips_image import normalize_vips_rgb_uchar
from histopia.stain._preflight import StainPreflightSlide


@dataclass(frozen=True, slots=True)
class AnalysisSlide:
    """One source-space RGB image and registration-derived tissue mask."""

    rgb: np.ndarray
    tissue_mask: np.ndarray
    analysis_mpp: float
    content_origin_native_xy: tuple[int, int]
    source_mpp_xy: tuple[float, float]


def read_analysis_slide(
    registration_run: Path | str,
    slide: StainPreflightSlide,
    *,
    analysis_mpp: float,
) -> AnalysisSlide:
    """Read only the scanner content bounds at a calibrated physical scale."""

    try:
        import pyvips
    except ImportError as exc:
        raise RuntimeError("stain WSI processing requires the 'stain' extra") from exc
    source = pyvips.Image.new_from_file(slide.source_path, access="random")
    source = normalize_vips_rgb_uchar(source)
    x, y, width, height = slide.content_bbox_xywh
    cropped = source.crop(x, y, width, height)
    scale_x = slide.mpp_xy[0] / analysis_mpp
    scale_y = slide.mpp_xy[1] / analysis_mpp
    resized = cropped.resize(scale_x, vscale=scale_y, kernel="lanczos3")
    rgb = np.frombuffer(resized.write_to_memory(), dtype=np.uint8).reshape(
        resized.height,
        resized.width,
        resized.bands,
    )
    mask_path = (
        Path(registration_run)
        / "processed"
        / f"{Path(slide.source_path).stem}.mask.png"
    )
    tissue = _resize_mask(mask_path, (resized.height, resized.width))
    return AnalysisSlide(
        rgb=rgb,
        tissue_mask=tissue,
        analysis_mpp=analysis_mpp,
        content_origin_native_xy=(x, y),
        source_mpp_xy=slide.mpp_xy,
    )


def _resize_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("stain WSI processing requires the 'stain' extra") from exc
    with Image.open(path) as image:
        resized = image.convert("L").resize(
            (shape[1], shape[0]),
            resample=Image.Resampling.NEAREST,
        )
        return np.asarray(resized) > 127
