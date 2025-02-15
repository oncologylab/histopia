"""Audit native KPF registration outputs against validated thumbnail warps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from histopia.registration._io import load_thumbnail
from histopia.registration._masking import create_tissue_mask

DEFAULT_MICE = ("4257", "4577", "4630", "5997")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registration-root",
        type=Path,
        default=Path("/tmp/histopia-registration-runs"),
    )
    parser.add_argument(
        "--full-resolution-root",
        type=Path,
        default=Path("/tmp/histopia-full-resolution-runs"),
    )
    parser.add_argument("--mice", nargs="+", default=list(DEFAULT_MICE))
    parser.add_argument("--max-dim-px", type=int, default=1200)
    parser.add_argument("--max-median-mae", type=float, default=5.0)
    parser.add_argument("--min-median-mask-dice", type=float, default=0.95)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def audit_mouse(
    mouse: str,
    registration_root: Path,
    full_resolution_root: Path,
    max_dim_px: int,
) -> dict[str, Any]:
    pyvips = _import_pyvips()
    run_dir = registration_root / mouse / "qc-1200-hybrid"
    output_dir = full_resolution_root / mouse
    payload = json.loads((run_dir / "registration_result.json").read_text())
    warp_summary = json.loads((output_dir / "full_resolution_warps.json").read_text())
    expected_paths = {Path(slide["path"]).stem: slide for slide in payload["slides"]}
    output_paths = {
        path.name.removesuffix(".registered.tiff"): path
        for path in output_dir.glob("*.registered.tiff")
    }
    missing = sorted(set(expected_paths) - set(output_paths))
    unexpected = sorted(set(output_paths) - set(expected_paths))
    temporary_files = sorted(path.name for path in output_dir.glob(".*.tmp"))
    bad_headers: list[dict[str, Any]] = []
    page_counts: list[int] = []
    maes: list[float] = []
    mask_dices: list[float] = []
    reference_path = Path(payload["reference_slide"])
    reference_shape = tuple(warp_summary[0]["reference_shape"])

    for stem, slide in expected_paths.items():
        output_path = output_paths.get(stem)
        if output_path is None:
            continue
        image = pyvips.Image.new_from_file(str(output_path))
        page_count = int(image.get("n-pages")) if image.get_typeof("n-pages") else 1
        page_counts.append(page_count)
        if (image.height, image.width) != reference_shape or image.bands != 3:
            bad_headers.append(
                {
                    "slide": stem,
                    "shape": [image.height, image.width],
                    "bands": image.bands,
                }
            )

        native_thumbnail = load_thumbnail(output_path, max_dim_px)
        if slide["is_reference"]:
            expected_path = (
                run_dir / "processed" / f"{reference_path.stem}.thumbnail.png"
            )
        else:
            expected_path = run_dir / "qc" / "alignment" / f"{stem}.warped.png"
        expected_thumbnail = load_thumbnail(expected_path, max_dim_px)
        native_thumbnail = _resize_exact(
            native_thumbnail,
            expected_thumbnail.shape[:2],
        )
        maes.append(
            float(
                np.abs(
                    native_thumbnail.astype(np.float32)
                    - expected_thumbnail.astype(np.float32)
                ).mean()
            )
        )
        native_mask = create_tissue_mask(native_thumbnail).mask
        expected_mask = create_tissue_mask(expected_thumbnail).mask
        denominator = native_mask.sum() + expected_mask.sum()
        dice = (
            2 * np.logical_and(native_mask, expected_mask).sum() / denominator
            if denominator
            else 0.0
        )
        mask_dices.append(float(dice))

    return {
        "mouse": mouse,
        "expected_files": len(expected_paths),
        "output_files": len(output_paths),
        "warp_records": len(warp_summary),
        "reference_shape": list(reference_shape),
        "pyramid_level_range": [min(page_counts), max(page_counts)],
        "median_thumbnail_mae": float(np.median(maes)),
        "median_mask_dice": float(np.median(mask_dices)),
        "missing": missing,
        "unexpected": unexpected,
        "temporary_files": temporary_files,
        "bad_headers": bad_headers,
    }


def main() -> int:
    args = parse_args()
    reports = [
        audit_mouse(
            mouse,
            args.registration_root,
            args.full_resolution_root,
            args.max_dim_px,
        )
        for mouse in args.mice
    ]
    failures: list[str] = []
    for report in reports:
        mouse = report["mouse"]
        if report["expected_files"] != report["output_files"]:
            failures.append(f"{mouse}: output file count mismatch")
        if report["expected_files"] != report["warp_records"]:
            failures.append(f"{mouse}: warp manifest count mismatch")
        if report["missing"] or report["unexpected"]:
            failures.append(f"{mouse}: slide-name mismatch")
        if report["temporary_files"] or report["bad_headers"]:
            failures.append(f"{mouse}: incomplete or invalid TIFF output")
        if report["median_thumbnail_mae"] > args.max_median_mae:
            failures.append(f"{mouse}: median native-thumbnail MAE is too high")
        if report["median_mask_dice"] < args.min_median_mask_dice:
            failures.append(f"{mouse}: median native-thumbnail mask Dice is too low")
    result = {"reports": reports, "failures": failures}
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    return 1 if failures else 0


def _resize_exact(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if image.shape[:2] == shape:
        return image
    from PIL import Image

    height, width = shape
    return np.asarray(
        Image.fromarray(image).resize((width, height), Image.Resampling.BILINEAR)
    )


def _import_pyvips() -> Any:
    try:
        import pyvips
    except ImportError as exc:
        msg = "pyvips is required for full-resolution validation"
        raise RuntimeError(msg) from exc
    return pyvips


if __name__ == "__main__":
    raise SystemExit(main())
