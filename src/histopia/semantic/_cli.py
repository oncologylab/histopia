"""Command line entry point for Histopia semantic atlases."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from histopia.semantic._config import load_semantic_config, override_compute_config


def _named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not name or not raw_path:
        raise argparse.ArgumentTypeError("expected non-empty NAME=PATH")
    return name, Path(raw_path)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _add_compute_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--device",
        help="Override config device: auto, cpu, cuda, cuda:N, or mps.",
    )
    parser.add_argument(
        "--batch-size",
        type=_positive_int,
        help="Override encoder inference batch size.",
    )
    parser.add_argument(
        "--patch-workers",
        type=_positive_int,
        help="Override concurrent WSI patch readers.",
    )
    parser.add_argument(
        "--vips-threads",
        type=_positive_int,
        help="Override the native libvips worker limit.",
    )


def _add_fit_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--fit-threads",
        type=_positive_int,
        help="Override native BLAS/OpenMP threads used for global atlas fitting.",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract UNI2-h features and fit a global serial-section atlas."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    doctor = subparsers.add_parser(
        "doctor",
        help="Report available CPU and accelerator execution backends.",
    )
    doctor.add_argument(
        "--device",
        default="auto",
        help="Validate a run device: auto, cpu, cuda, cuda:N, or mps.",
    )
    for command in ("preflight", "extract", "fit", "run"):
        child = subparsers.add_parser(command)
        child.add_argument("--config", type=Path, required=True)
        if command in {"extract", "run"}:
            _add_compute_arguments(child)
            child.add_argument("--overwrite-features", action="store_true")
            child.add_argument(
                "--allow-model-download",
                action="store_true",
                help=(
                    "Allow authenticated Hugging Face access instead of "
                    "cache-only mode."
                ),
            )
        if command in {"fit", "run"}:
            _add_fit_arguments(child)
    cache = subparsers.add_parser("cache-model")
    cache.add_argument("--cache-dir", type=Path, required=True)
    approve = subparsers.add_parser(
        "approve",
        help="Approve one exact, integrity-checked semantic result.",
    )
    approve.add_argument("--run", type=Path, required=True)
    approve.add_argument("--reviewer", required=True)
    approve.add_argument("--review-notes", required=True)
    cohort = subparsers.add_parser("cohort-qc")
    cohort.add_argument("--run", type=_named_path, action="append", required=True)
    cohort.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "doctor":
        from histopia.compute import inspect_compute

        print(json.dumps(inspect_compute(args.device), indent=2))
        return 0
    if args.command == "cache-model":
        return _cache_model(args.cache_dir)
    if args.command == "approve":
        from histopia.semantic._approval import approve_semantic_result

        approval = approve_semantic_result(
            args.run,
            reviewer=args.reviewer,
            notes=args.review_notes,
        )
        print(
            f"{approval.run_dir / 'semantic_review.json'}: "
            f"fingerprint={approval.fingerprint}"
        )
        return 0
    if args.command == "cohort-qc":
        from histopia.semantic._qc import write_cohort_qc

        output = write_cohort_qc(dict(args.run), args.output)
        print(output)
        return 0

    config = load_semantic_config(args.config)
    if args.command in {"extract", "fit", "run"}:
        config = override_compute_config(
            config,
            device=getattr(args, "device", None),
            batch_size=getattr(args, "batch_size", None),
            patch_workers=getattr(args, "patch_workers", None),
            vips_threads=getattr(args, "vips_threads", None),
            fit_threads=getattr(args, "fit_threads", None),
        )
    if args.command == "preflight":
        from histopia.semantic._preflight import (
            preflight_registration,
            write_preflight,
        )

        preflight = preflight_registration(config.registration_run)
        output = write_preflight(preflight, config.output_dir / "preflight.json")
        print(
            f"{output}: {preflight.slide_count} slides, "
            f"fingerprint={preflight.fingerprint}"
        )
        return 0
    preflight = None
    if args.command in {"extract", "run"}:
        from histopia.semantic._preflight import preflight_registration

        preflight = preflight_registration(config.registration_run)
    if args.command == "fit":
        from histopia.semantic._pipeline import fit_saved_features

        _, result = fit_saved_features(config)
    else:
        if config.model_cache_dir is None:
            parser.error("model_cache_dir is required for UNI2-h extraction")
        from histopia.semantic._extract import extract_registration_features
        from histopia.semantic._pipeline import run_semantic_atlas
        from histopia.semantic._uni2h import Uni2hEncoder

        encoder = Uni2hEncoder.lazy_from_cache(
            config.model_cache_dir,
            device=config.device,
            local_only=not args.allow_model_download,
            vips_threads=config.vips_threads,
        )
        if args.command == "extract":
            paths = extract_registration_features(
                config,
                encoder,
                preflight=preflight,
                overwrite=args.overwrite_features,
                progress=print,
            )
            print(f"Extracted or verified {len(paths)} section artifacts.")
            return 0
        result = run_semantic_atlas(
            config,
            encoder,
            preflight=preflight,
            overwrite_features=args.overwrite_features,
            progress=print,
        )
    print(result)
    print(
        "Scientific review required: inspect semantic overlays, then approve "
        f"{config.output_dir / 'semantic_review.json'} for this fingerprint."
    )
    return 0


def _cache_model(cache_dir: Path) -> int:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("model caching requires the 'uni2h' extra") from exc
    path = snapshot_download("MahmoodLab/UNI2-h", cache_dir=cache_dir)
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
