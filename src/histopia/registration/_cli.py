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
    args = parser.parse_args(argv)

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
        parser.error("--config, --manifest, or --warp-run is required")

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
    return RegistrationConfig(
        input_dir=Path(data.pop("input_dir")),
        output_dir=Path(data.pop("output_dir")),
        reference_slide=data.pop("reference_slide", None),
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
