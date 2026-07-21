"""Command line entry point for Histopia semantic atlases."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from histopia.semantic._config import load_semantic_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract UNI2-h features and fit a global serial-section atlas."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("extract", "fit", "run"):
        child = subparsers.add_parser(command)
        child.add_argument("--config", type=Path, required=True)
        if command in {"extract", "run"}:
            child.add_argument("--overwrite-features", action="store_true")
            child.add_argument(
                "--allow-model-download",
                action="store_true",
                help=(
                    "Allow authenticated Hugging Face access instead of "
                    "cache-only mode."
                ),
            )
    cache = subparsers.add_parser("cache-model")
    cache.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "cache-model":
        return _cache_model(args.cache_dir)

    config = load_semantic_config(args.config)
    if args.command == "fit":
        from histopia.semantic._pipeline import fit_saved_features

        _, result = fit_saved_features(config)
    else:
        if config.model_cache_dir is None:
            parser.error("model_cache_dir is required for UNI2-h extraction")
        from histopia.semantic._extract import extract_registration_features
        from histopia.semantic._pipeline import run_semantic_atlas
        from histopia.semantic._uni2h import Uni2hEncoder

        encoder = Uni2hEncoder.from_cache(
            config.model_cache_dir,
            device=config.device,
            local_only=not args.allow_model_download,
        )
        if args.command == "extract":
            paths = extract_registration_features(
                config,
                encoder,
                overwrite=args.overwrite_features,
            )
            print(f"Extracted or verified {len(paths)} section artifacts.")
            return 0
        result = run_semantic_atlas(
            config,
            encoder,
            overwrite_features=args.overwrite_features,
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
