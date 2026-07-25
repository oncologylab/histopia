"""Audit native KPF registration outputs against validated thumbnail warps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from histopia._atomic import write_json_atomic
from histopia._vips_image import normalize_vips_rgb_uchar
from histopia.registration._io import load_thumbnail, warp_mask_thumbnail

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
    parser.add_argument("--max-slide-mae", type=float, default=15.0)
    parser.add_argument("--max-median-tissue-mae", type=float, default=10.0)
    parser.add_argument("--max-slide-tissue-mae", type=float, default=25.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def audit_mouse(
    mouse: str,
    registration_root: Path,
    full_resolution_root: Path,
    max_dim_px: int,
    max_slide_mae: float,
    max_slide_tissue_mae: float,
) -> dict[str, Any]:
    pyvips = _import_pyvips()
    run_dir = registration_root / mouse
    if not (run_dir / "registration_result.json").exists():
        run_dir = run_dir / "qc-1200-hybrid"
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
    temporary_files = sorted(path.name for path in output_dir.glob(".*.tmp*"))
    bad_headers: list[dict[str, Any]] = []
    page_counts: list[int] = []
    maes: list[float] = []
    tissue_maes: list[float] = []
    slide_metrics: list[dict[str, Any]] = []
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

        native_thumbnail = _load_native_thumbnail(output_path, max_dim_px)
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
        mae = float(
            np.abs(
                native_thumbnail.astype(np.float32)
                - expected_thumbnail.astype(np.float32)
            ).mean()
        )
        maes.append(mae)
        source_mask = _load_mask(run_dir / "processed" / f"{stem}.mask.png")
        expected_mask = warp_mask_thumbnail(
            source_mask,
            np.asarray(slide["transform"]["matrix"], dtype=float),
            expected_thumbnail.shape[:2],
        )
        if not expected_mask.any():
            raise ValueError(f"registered tissue mask is empty: {stem}")
        tissue_mae = float(
            np.abs(
                native_thumbnail.astype(np.float32)
                - expected_thumbnail.astype(np.float32)
            )[expected_mask].mean()
        )
        tissue_maes.append(tissue_mae)
        slide_metrics.append(
            {
                "slide": stem,
                "thumbnail_mae": mae,
                "tissue_mae": tissue_mae,
                "accepted": (
                    mae <= max_slide_mae and tissue_mae <= max_slide_tissue_mae
                ),
            }
        )

    provenance_records = sum(
        isinstance(row.get("provenance"), dict)
        and row["provenance"].get("schema_version") == 1
        for row in warp_summary
        if isinstance(row, dict)
    )
    rejected_slides = [
        row["slide"] for row in slide_metrics if row["accepted"] is not True
    ]

    return {
        "mouse": mouse,
        "expected_files": len(expected_paths),
        "output_files": len(output_paths),
        "warp_records": len(warp_summary),
        "provenance_records": provenance_records,
        "reference_shape": list(reference_shape),
        "pyramid_level_range": (
            [min(page_counts), max(page_counts)] if page_counts else None
        ),
        "median_thumbnail_mae": float(np.median(maes)) if maes else None,
        "maximum_thumbnail_mae": max(maes, default=None),
        "median_tissue_mae": (float(np.median(tissue_maes)) if tissue_maes else None),
        "maximum_tissue_mae": max(tissue_maes, default=None),
        "rejected_slides": rejected_slides,
        "slide_metrics": slide_metrics,
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
            args.max_slide_mae,
            args.max_slide_tissue_mae,
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
        if report["expected_files"] != report["provenance_records"]:
            failures.append(f"{mouse}: warp provenance is incomplete")
        if report["missing"] or report["unexpected"]:
            failures.append(f"{mouse}: slide-name mismatch")
        if report["temporary_files"] or report["bad_headers"]:
            failures.append(f"{mouse}: incomplete or invalid TIFF output")
        if (
            report["median_thumbnail_mae"] is None
            or report["median_thumbnail_mae"] > args.max_median_mae
        ):
            failures.append(f"{mouse}: median native-thumbnail MAE is too high")
        if (
            report["median_tissue_mae"] is None
            or report["median_tissue_mae"] > args.max_median_tissue_mae
        ):
            failures.append(f"{mouse}: median native tissue MAE is too high")
        if report["rejected_slides"]:
            failures.append(f"{mouse}: one or more native slide audits failed")
    result = {"reports": reports, "failures": failures}
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        write_json_atomic(args.output, result)
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


def _load_mask(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("L")) > 0


def _load_native_thumbnail(path: Path, max_dim_px: int) -> np.ndarray:
    pyvips = _import_pyvips()
    image = normalize_vips_rgb_uchar(
        pyvips.Image.thumbnail(
            str(path),
            max_dim_px,
            height=max_dim_px,
            no_rotate=True,
        )
    )
    return np.frombuffer(image.write_to_memory(), dtype=np.uint8).reshape(
        image.height,
        image.width,
        3,
    )


def _import_pyvips() -> Any:
    try:
        import pyvips
    except ImportError as exc:
        msg = "pyvips is required for full-resolution validation"
        raise RuntimeError(msg) from exc
    pyvips.cache_set_max(0)
    pyvips.cache_set_max_files(4)
    pyvips.cache_set_max_mem(64 * 1024 * 1024)
    return pyvips


if __name__ == "__main__":
    raise SystemExit(main())
