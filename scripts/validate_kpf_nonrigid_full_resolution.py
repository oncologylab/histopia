"""Audit accepted native KPF non-rigid exports against thumbnail-space warps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from histopia.registration._io import load_thumbnail, warp_rgb_thumbnail
from histopia.registration._masking import create_tissue_mask
from histopia.registration._nonrigid import warp_with_displacement

DEFAULT_MICE = ("4257", "4577", "4630", "5997")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--non-rigid-root",
        type=Path,
        default=Path("/tmp/histopia-nonrigid-runs"),
    )
    parser.add_argument(
        "--full-resolution-root",
        type=Path,
        default=Path("/tmp/histopia-full-resolution-nonrigid"),
    )
    parser.add_argument("--mice", nargs="+", default=list(DEFAULT_MICE))
    parser.add_argument("--max-dim-px", type=int, default=1200)
    parser.add_argument("--max-median-mae", type=float, default=5.0)
    parser.add_argument("--min-median-mask-dice", type=float, default=0.90)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def audit_mouse(
    mouse: str,
    non_rigid_root: Path,
    full_resolution_root: Path,
    max_dim_px: int,
) -> dict[str, Any]:
    pyvips = _import_pyvips()
    run_dir = non_rigid_root / mouse
    output_dir = full_resolution_root / mouse
    payload = json.loads((run_dir / "registration_result.json").read_text())
    accepted = {
        Path(slide["path"]).stem: slide
        for slide in payload["slides"]
        if not slide["is_reference"] and slide["non_rigid_transform"]["accepted"]
    }
    output_paths = {
        path.name.removesuffix(".registered.tiff"): path
        for path in output_dir.glob("*.registered.tiff")
    }
    summary_path = output_dir / "full_resolution_warps.json"
    warp_summary = json.loads(summary_path.read_text())
    records = {
        Path(record["output_path"]).name.removesuffix(".registered.tiff"): record
        for record in warp_summary
    }
    reference_stem = Path(payload["reference_slide"]).stem
    reference_thumbnail = load_thumbnail(
        run_dir / "processed" / f"{reference_stem}.thumbnail.png",
        max_dim_px,
    )
    missing = sorted(set(accepted) - set(output_paths))
    unexpected = sorted(set(output_paths) - set(accepted))
    missing_records = sorted(set(accepted) - set(records))
    bad_provenance = sorted(
        stem
        for stem in accepted.keys() & records.keys()
        if not records[stem].get("non_rigid_applied", False)
    )
    temporary_files = sorted(path.name for path in output_dir.glob(".*.tmp"))
    bad_headers: list[dict[str, Any]] = []
    page_counts: list[int] = []
    maes: list[float] = []
    mask_dices: list[float] = []
    slide_metrics: list[dict[str, Any]] = []

    for stem, slide in accepted.items():
        output_path = output_paths.get(stem)
        record = records.get(stem)
        if output_path is None or record is None:
            continue
        image = pyvips.Image.new_from_file(str(output_path))
        page_count = int(image.get("n-pages")) if image.get_typeof("n-pages") else 1
        page_counts.append(page_count)
        expected_shape = tuple(record["reference_shape"])
        if (image.height, image.width) != expected_shape or image.bands != 3:
            bad_headers.append(
                {
                    "slide": stem,
                    "shape": [image.height, image.width],
                    "bands": image.bands,
                }
            )

        moving_thumbnail = load_thumbnail(
            run_dir / "processed" / f"{stem}.thumbnail.png",
            max_dim_px,
        )
        rigid = warp_rgb_thumbnail(
            moving_thumbnail,
            np.asarray(slide["transform"]["matrix"], dtype=float),
            reference_thumbnail.shape[:2],
        )
        displacement_path = Path(slide["non_rigid_transform"]["displacement_path"])
        if not displacement_path.is_absolute():
            displacement_path = run_dir / displacement_path
        with np.load(displacement_path) as archive:
            displacement = archive["displacement"]
        expected = warp_with_displacement(rigid, displacement)
        native = _resize_exact(
            load_thumbnail(output_path, max_dim_px),
            expected.shape[:2],
        )
        difference = native.astype(np.float32) - expected.astype(np.float32)
        mae = float(np.abs(difference).mean())
        maes.append(mae)
        native_mask = create_tissue_mask(native).mask
        expected_mask = create_tissue_mask(expected).mask
        denominator = native_mask.sum() + expected_mask.sum()
        mask_dice = (
            float(2 * np.logical_and(native_mask, expected_mask).sum() / denominator)
            if denominator
            else 0.0
        )
        mask_dices.append(mask_dice)
        slide_metrics.append(
            {"slide": stem, "thumbnail_mae": mae, "mask_dice": mask_dice}
        )

    return {
        "mouse": mouse,
        "accepted_slides": len(accepted),
        "output_files": len(output_paths),
        "warp_records": len(warp_summary),
        "pyramid_level_range": (
            [min(page_counts), max(page_counts)] if page_counts else []
        ),
        "median_thumbnail_mae": float(np.median(maes)) if maes else 0.0,
        "maximum_thumbnail_mae": float(np.max(maes)) if maes else 0.0,
        "median_mask_dice": float(np.median(mask_dices)) if mask_dices else 0.0,
        "minimum_mask_dice": float(np.min(mask_dices)) if mask_dices else 0.0,
        "missing": missing,
        "unexpected": unexpected,
        "missing_records": missing_records,
        "bad_provenance": bad_provenance,
        "temporary_files": temporary_files,
        "bad_headers": bad_headers,
        "slides": slide_metrics,
    }


def main() -> int:
    args = parse_args()
    reports = [
        audit_mouse(
            mouse,
            args.non_rigid_root,
            args.full_resolution_root,
            args.max_dim_px,
        )
        for mouse in args.mice
    ]
    failures: list[str] = []
    for report in reports:
        mouse = report["mouse"]
        if not (
            report["accepted_slides"]
            == report["output_files"]
            == report["warp_records"]
        ):
            failures.append(f"{mouse}: output or manifest count mismatch")
        if report["missing"] or report["unexpected"] or report["missing_records"]:
            failures.append(f"{mouse}: slide-name mismatch")
        if report["bad_provenance"]:
            failures.append(f"{mouse}: non-rigid provenance is missing")
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
