"""Command line entry point for Histopia registration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from histopia.registration._config import (
    BrightfieldMaskConfig,
    MaskRefinementConfig,
    NonRigidRefinementConfig,
    RegistrationConfig,
)
from histopia.registration._manifest import build_kpf_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Histopia registration, validation, and WSI export."
    )
    parser.add_argument("--config", type=Path, help="JSON or TOML registration config.")
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Build a KPF manifest for a mouse dir.",
    )
    parser.add_argument(
        "--compare-kpf-run",
        type=Path,
        help="Compare a completed run directory against KPF registered references.",
    )
    parser.add_argument(
        "--mouse-dir",
        type=Path,
        help="KPF mouse directory used with --compare-kpf-run.",
    )
    parser.add_argument(
        "--warp-run",
        type=Path,
        help="Apply a saved registration run to full-resolution source slides.",
    )
    parser.add_argument(
        "--registered-output-dir",
        type=Path,
        help="Output directory used with --warp-run.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing registered TIFFs used with --warp-run.",
    )
    parser.add_argument(
        "--warp-crop-mode",
        choices=("reference", "overlap"),
        default="reference",
        help="Canvas crop used with --warp-run. Default: reference.",
    )
    parser.add_argument(
        "--accepted-non-rigid-only",
        action="store_true",
        help="Export only accepted non-rigid slides used with --warp-run.",
    )
    parser.add_argument(
        "--viewer-run",
        action="append",
        default=[],
        metavar="MOUSE=RUN_DIR",
        help="Add a completed mouse run to a static section viewer.",
    )
    parser.add_argument(
        "--viewer-output-dir",
        type=Path,
        help="Output directory used with --viewer-run.",
    )
    parser.add_argument(
        "--viewer-detail-max-dim-px",
        type=int,
        help="Generate lazy inspection textures up to this WSI dimension.",
    )
    parser.add_argument(
        "--provisional-mouse",
        action="append",
        default=[],
        help="Mark a viewer mouse as having provisional physical order.",
    )
    args = parser.parse_args(argv)

    if args.viewer_run:
        if args.viewer_output_dir is None:
            parser.error("--viewer-output-dir is required with --viewer-run")
        from histopia.registration._viewer import build_section_viewer

        runs: dict[str, Path] = {}
        for item in args.viewer_run:
            if "=" not in item:
                parser.error("--viewer-run must use MOUSE=RUN_DIR")
            mouse, run_dir = item.split("=", 1)
            runs[mouse] = Path(run_dir)
        index_path = build_section_viewer(
            runs,
            args.viewer_output_dir,
            provisional_mice=set(args.provisional_mouse),
            detail_max_dim_px=args.viewer_detail_max_dim_px,
        )
        print(index_path)
        return 0

    if args.warp_run is not None:
        from histopia.registration._wsi import warp_saved_registration

        results = warp_saved_registration(
            args.warp_run,
            args.registered_output_dir,
            overwrite=args.overwrite,
            crop_mode=args.warp_crop_mode,
            accepted_non_rigid_only=args.accepted_non_rigid_only,
        )
        print(json.dumps([result.to_json_dict() for result in results], indent=2))
        return 0

    if args.compare_kpf_run is not None:
        if args.mouse_dir is None:
            parser.error("--mouse-dir is required with --compare-kpf-run")
        from histopia.registration._validation import compare_kpf_run

        summary = compare_kpf_run(args.mouse_dir, args.compare_kpf_run)
        print(json.dumps(summary, indent=2))
        return 0

    if args.manifest is not None:
        manifest = build_kpf_manifest(args.manifest)
        payload = {
            "mouse_dir": str(manifest.mouse_dir),
            "pair_count": len(manifest.pairs),
            "is_complete": manifest.is_complete,
            "missing_raw_keys": list(manifest.missing_raw_keys),
            "missing_reference_keys": list(manifest.missing_reference_keys),
            "ambiguous_keys": list(manifest.ambiguous_keys),
        }
        print(json.dumps(payload, indent=2))
        return 0 if manifest.is_complete else 1

    if args.config is None:
        parser.error("--config, --manifest, --warp-run, or --viewer-run is required")

    from histopia.registration._pipeline import register_sections

    config = _load_config(args.config)
    result = register_sections(config)
    print(json.dumps(result.to_json_dict(), indent=2))
    return 0


def _load_config(path: Path) -> RegistrationConfig:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
    elif path.suffix.lower() in {".toml", ".tml"}:
        if sys.version_info >= (3, 11):
            import tomllib
        else:
            import tomli as tomllib

        data = tomllib.loads(path.read_text())
    else:
        msg = "config must be JSON or TOML"
        raise ValueError(msg)
    return _config_from_mapping(data)


def _config_from_mapping(data: dict[str, Any]) -> RegistrationConfig:
    mask_data = dict(data.pop("mask", {}))
    mask = BrightfieldMaskConfig(**mask_data)
    refinement_data = dict(data.pop("refinement", {}))
    refinement = MaskRefinementConfig(**refinement_data)
    non_rigid_data = dict(data.pop("non_rigid_refinement", {}))
    non_rigid_refinement = NonRigidRefinementConfig(**non_rigid_data)
    registered_reference_dir_value = data.pop("registered_reference_dir", None)
    registered_reference_dir = (
        Path(registered_reference_dir_value)
        if registered_reference_dir_value is not None
        else None
    )
    registered_output_dir_value = data.pop("registered_output_dir", None)
    registered_output_dir = (
        Path(registered_output_dir_value)
        if registered_output_dir_value is not None
        else None
    )
    section_order_value = data.pop("section_order_path", None)
    section_order_review_value = data.pop("section_order_review_path", None)
    section_orientation_value = data.pop("section_orientation_path", None)
    mask_review_value = data.pop("mask_review_path", None)
    mask_override_value = data.pop("mask_override_dir", None)
    automatic_mask_snapshot_value = data.pop("automatic_mask_snapshot_path", None)
    affine_override_value = data.pop("affine_override_path", None)
    return RegistrationConfig(
        input_dir=Path(data.pop("input_dir")),
        output_dir=Path(data.pop("output_dir")),
        reference_slide=data.pop("reference_slide", None),
        reference_policy=data.pop("reference_policy", "best_connected"),
        section_order_path=Path(section_order_value) if section_order_value else None,
        section_order_strategy=data.pop("section_order_strategy", "natural"),
        section_order_review_path=(
            Path(section_order_review_value) if section_order_review_value else None
        ),
        section_orientation_path=(
            Path(section_orientation_value) if section_orientation_value else None
        ),
        require_approved_order=data.pop("require_approved_order", False),
        mask_review_path=Path(mask_review_value) if mask_review_value else None,
        mask_override_dir=Path(mask_override_value) if mask_override_value else None,
        automatic_mask_snapshot_path=(
            Path(automatic_mask_snapshot_value)
            if automatic_mask_snapshot_value
            else None
        ),
        affine_override_path=(
            Path(affine_override_value) if affine_override_value else None
        ),
        require_approved_masks=data.pop("require_approved_masks", False),
        wsi_only=data.pop("wsi_only", False),
        registered_reference_dir=registered_reference_dir,
        max_processed_image_dim_px=data.pop("max_processed_image_dim_px", 1200),
        crop_mode=data.pop("crop_mode", "reference"),
        rigid_method=data.pop("rigid_method", "feature"),
        align_strategy=data.pop("align_strategy", "hybrid"),
        non_rigid=data.pop("non_rigid", False),
        mask=mask,
        refinement=refinement,
        non_rigid_refinement=non_rigid_refinement,
        write_processed_images=data.pop("write_processed_images", True),
        write_warped_images=data.pop("write_warped_images", False),
        registered_output_dir=registered_output_dir,
        wsi_compression=data.pop("wsi_compression", "jpeg"),
        wsi_jpeg_quality=data.pop("wsi_jpeg_quality", 95),
        wsi_tile_size=data.pop("wsi_tile_size", 512),
    )


if __name__ == "__main__":
    sys.exit(main())
