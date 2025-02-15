"""Validation helpers for comparing Histopia runs to existing references."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from histopia.registration._io import load_thumbnail, save_rgb, side_by_side
from histopia.registration._manifest import build_kpf_manifest
from histopia.registration._masking import create_tissue_mask
from histopia.registration._rigid import estimate_rigid_transform


def compare_kpf_run(
    mouse_dir: Path | str,
    run_dir: Path | str,
    *,
    max_dim_px: int = 900,
) -> dict[str, Any]:
    """Compare a Histopia run to historical registered KPF references.

    The comparison is intentionally tissue-crop based. Historical registered
    OME-TIFFs and Histopia thumbnail outputs can live in different coordinate
    frames, so this validation measures normalized tissue-shape similarity and
    writes visual panels rather than claiming pixel-perfect equivalence.
    """

    mouse_dir = Path(mouse_dir)
    run_dir = Path(run_dir)
    result_path = run_dir / "registration_result.json"
    result = json.loads(result_path.read_text())
    manifest = build_kpf_manifest(mouse_dir)
    reference_by_raw = {
        pair.raw_path.resolve(): pair.reference_path for pair in manifest.pairs
    }

    output_dir = run_dir / "historical_reference_qc"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for slide in result["slides"]:
        raw_path = Path(slide["path"]).resolve()
        historical_path = reference_by_raw.get(raw_path)
        if historical_path is None:
            continue
        current_path = _current_qc_image(run_dir, raw_path, slide["is_reference"])
        if current_path is None:
            continue

        current = load_thumbnail(current_path, max_dim_px)
        historical = load_thumbnail(historical_path, max_dim_px)
        current_crop = _normalized_tissue_crop(current, max_dim_px)
        historical_crop = _normalized_tissue_crop(historical, max_dim_px)
        direct_score = _mask_dice(current_crop.mask, historical_crop.mask)
        moment_transform = estimate_rigid_transform(
            historical_crop.image,
            current_crop.image,
            fixed_mask=historical_crop.mask,
            moving_mask=current_crop.mask,
            method="mask_moments",
        )
        aligned_current = _warp_for_comparison(
            current_crop.image,
            moment_transform.matrix,
            historical_crop.image.shape[:2],
        )
        aligned_current_mask = _warp_mask_for_comparison(
            current_crop.mask,
            moment_transform.matrix,
            historical_crop.mask.shape,
        )
        aligned_score = _mask_dice(aligned_current_mask, historical_crop.mask)
        panel = side_by_side(
            [current_crop.image, aligned_current, historical_crop.image]
        )
        panel_path = output_dir / f"{raw_path.stem}.current_vs_historical.png"
        save_rgb(panel_path, panel)
        rows.append(
            {
                "slide": raw_path.name,
                "historical_reference": str(historical_path),
                "current_qc_image": str(current_path),
                "panel": str(panel_path),
                "normalized_tissue_dice": direct_score,
                "moment_aligned_tissue_dice": aligned_score,
                "current_foreground_fraction": current_crop.foreground_fraction,
                "historical_foreground_fraction": historical_crop.foreground_fraction,
            }
        )

    rows.sort(key=lambda row: row["moment_aligned_tissue_dice"])
    summary = {
        "mouse_dir": str(mouse_dir),
        "run_dir": str(run_dir),
        "slide_count": len(rows),
        "median_normalized_tissue_dice": _median(
            [row["normalized_tissue_dice"] for row in rows]
        ),
        "median_moment_aligned_tissue_dice": _median(
            [row["moment_aligned_tissue_dice"] for row in rows]
        ),
        "worst_slides": rows[:10],
    }
    (output_dir / "comparison_summary.json").write_text(
        json.dumps({"summary": summary, "slides": rows}, indent=2) + "\n"
    )
    _write_comparison_report(output_dir / "comparison_report.md", summary, rows)
    return summary


class _Crop:
    def __init__(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        foreground_fraction: float,
    ) -> None:
        self.image = image
        self.mask = mask
        self.foreground_fraction = foreground_fraction


def _current_qc_image(
    run_dir: Path,
    raw_path: Path,
    is_reference: bool,
) -> Path | None:
    if is_reference:
        path = run_dir / "processed" / f"{raw_path.stem}.thumbnail.png"
    else:
        path = run_dir / "qc" / "alignment" / f"{raw_path.stem}.warped.png"
    return path if path.exists() else None


def _normalized_tissue_crop(image: np.ndarray, max_dim_px: int) -> _Crop:
    mask_result = create_tissue_mask(image)
    mask = _comparison_mask(mask_result)
    if not mask.any():
        return _Crop(image, mask, 0.0)
    rows, cols = np.nonzero(mask)
    row0, row1 = rows.min(), rows.max() + 1
    col0, col1 = cols.min(), cols.max() + 1
    crop_image = image[row0:row1, col0:col1]
    crop_mask = mask[row0:row1, col0:col1]
    scale = max_dim_px / max(crop_mask.shape)
    crop_image = _resize_image(crop_image, scale)
    crop_mask = _resize_mask(crop_mask, scale)
    return _Crop(crop_image, crop_mask, float(crop_mask.mean()))


def _comparison_mask(mask_result: Any) -> np.ndarray:
    if not mask_result.candidate_masks:
        return mask_result.mask

    scored: list[tuple[float, str, np.ndarray]] = []
    for method, mask in mask_result.candidate_masks.items():
        metrics = mask_result.candidate_metrics[method]
        foreground = metrics["foreground_fraction"]
        border = metrics["max_border_strip_foreground_fraction"]
        largest = metrics["largest_component_fraction"]
        if foreground < 0.01 or foreground > 0.55:
            continue
        if border > 0.20:
            continue
        if largest < 0.15:
            continue
        score = largest - abs(foreground - 0.22) - border
        scored.append((score, method, mask))

    if not scored:
        return mask_result.mask
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][2]


def _resize_image(image: np.ndarray, scale: float) -> np.ndarray:
    from histopia.registration._io import resize_rgb

    return resize_rgb(image, scale)


def _resize_mask(mask: np.ndarray, scale: float) -> np.ndarray:
    from histopia.registration._io import resize_mask

    return resize_mask(mask, scale)


def _mask_dice(first: np.ndarray, second: np.ndarray) -> float:
    height = max(first.shape[0], second.shape[0])
    width = max(first.shape[1], second.shape[1])
    first_padded = _pad_mask(first, height, width)
    second_padded = _pad_mask(second, height, width)
    intersection = np.logical_and(first_padded, second_padded).sum()
    denominator = first_padded.sum() + second_padded.sum()
    if denominator == 0:
        return 0.0
    return float(2 * intersection / denominator)


def _warp_for_comparison(
    image: np.ndarray,
    matrix: np.ndarray,
    output_shape: tuple[int, int],
) -> np.ndarray:
    from histopia.registration._io import warp_rgb_thumbnail

    return warp_rgb_thumbnail(image, matrix, output_shape)


def _warp_mask_for_comparison(
    mask: np.ndarray,
    matrix: np.ndarray,
    output_shape: tuple[int, int],
) -> np.ndarray:
    import cv2

    moving = mask.astype(np.uint8) * 255
    warped = cv2.warpAffine(
        moving,
        matrix[:2, :].astype(np.float32),
        dsize=(output_shape[1], output_shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped > 0


def _pad_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    padded = np.zeros((height, width), dtype=bool)
    padded[: mask.shape[0], : mask.shape[1]] = mask
    return padded


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.median(np.asarray(values, dtype=float)))


def _write_comparison_report(
    path: Path,
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Historical Reference Comparison",
        "",
        f"Run directory: `{summary['run_dir']}`",
        f"Slide count: {summary['slide_count']}",
        (
            "Median normalized tissue Dice: "
            f"{summary['median_normalized_tissue_dice']:.3f}"
            if summary["median_normalized_tissue_dice"] is not None
            else "Median normalized tissue Dice: n/a"
        ),
        (
            "Median moment-aligned tissue Dice: "
            f"{summary['median_moment_aligned_tissue_dice']:.3f}"
            if summary["median_moment_aligned_tissue_dice"] is not None
            else "Median moment-aligned tissue Dice: n/a"
        ),
        "",
        "| Slide | Direct Dice | Moment Dice | Current FG | Historical FG | Panel |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        panel = Path(row["panel"]).name
        lines.append(
            "| "
            f"{row['slide']} | "
            f"{row['normalized_tissue_dice']:.3f} | "
            f"{row['moment_aligned_tissue_dice']:.3f} | "
            f"{row['current_foreground_fraction']:.3f} | "
            f"{row['historical_foreground_fraction']:.3f} | "
            f"`{panel}` |"
        )
    path.write_text("\n".join(lines) + "\n")
