"""Bounded analysis-resolution WSI reads for stain quantification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from histopia._vips_image import normalize_vips_rgb_uchar
from histopia._vips_retry import retry_transient_vips_read
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
    x, y, width, height = slide.content_bbox_xywh
    scale_x = slide.mpp_xy[0] / analysis_mpp
    scale_y = slide.mpp_xy[1] / analysis_mpp

    def read_rgb() -> np.ndarray:
        source = pyvips.Image.new_from_file(slide.source_path, access="random")
        source = normalize_vips_rgb_uchar(source)
        cropped = source.crop(x, y, width, height)
        resized = cropped.resize(scale_x, vscale=scale_y, kernel="lanczos3")
        return np.frombuffer(resized.write_to_memory(), dtype=np.uint8).reshape(
            resized.height,
            resized.width,
            resized.bands,
        )

    rgb = retry_transient_vips_read(read_rgb)
    mask_path = (
        Path(registration_run)
        / "processed"
        / f"{Path(slide.source_path).stem}.mask.png"
    )
    tissue = _resize_mask(mask_path, rgb.shape[:2])
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
