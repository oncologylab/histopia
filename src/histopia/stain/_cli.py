"""Command-line entry point for quantitative stain workflows."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from histopia._signals import graceful_sigterm
from histopia.stain._config import load_stain_config


def _named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not name or not raw_path:
        raise argparse.ArgumentTypeError("expected non-empty NAME=PATH")
    return name, Path(raw_path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Quantify brightfield chromogen deposition in registered sections."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor")
    for name in ("preflight", "benchmark", "run"):
        child = commands.add_parser(name)
        child.add_argument("--config", type=Path, required=True)
        if name in {"benchmark", "run"}:
            child.add_argument("--overwrite-fits", action="store_true")
        if name == "run":
            child.add_argument("--overwrite-maps", action="store_true")
    approve = commands.add_parser("approve")
    approve.add_argument("--run", type=Path, required=True)
    approve.add_argument("--reviewer", required=True)
    approve.add_argument("--review-notes", required=True)
    cohort = commands.add_parser("cohort-qc")
    cohort.add_argument("--run", type=_named_path, action="append", required=True)
    cohort.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the stain CLI with graceful cancellation."""

    with graceful_sigterm():
        return _main(argv)


def _main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "doctor":
        print(
            json.dumps(
                {
                    "core_device": "cpu",
                    "packages": {
                        package: importlib.util.find_spec(module) is not None
                        for package, module in {
                            "numpy": "numpy",
                            "scipy": "scipy",
                            "scikit-learn": "sklearn",
                            "pillow": "PIL",
                            "pyvips": "pyvips",
                            "tifffile": "tifffile",
                        }.items()
                    },
                },
                indent=2,
            )
        )
        return 0
    if args.command == "approve":
        from histopia.stain._approval import approve_stain_result

        approval = approve_stain_result(
            args.run,
            reviewer=args.reviewer,
            notes=args.review_notes,
        )
        print(
            f"{approval.run_dir / 'stain_review.json'}: "
            f"fingerprint={approval.fingerprint}"
        )
        return 0
    if args.command == "cohort-qc":
        from histopia.stain._qc import write_stain_cohort_qc

        print(write_stain_cohort_qc(dict(args.run), args.output))
        return 0
    config = load_stain_config(args.config)
    if args.command == "preflight":
        from histopia.stain._preflight import (
            preflight_stain_run,
            write_stain_preflight,
        )

        preflight = preflight_stain_run(config)
        path = write_stain_preflight(
            preflight,
            config.output_dir / "preflight.json",
        )
        print(
            f"{path}: {preflight.slide_count} slides, "
            f"fingerprint={preflight.fingerprint}"
        )
        return 0
    if args.command == "benchmark":
        from histopia.stain._pipeline import benchmark_stain_methods

        print(
            benchmark_stain_methods(
                config,
                overwrite=args.overwrite_fits,
                progress=print,
            )
        )
        return 0
    from histopia.stain._pipeline import run_stain_quantification

    result = run_stain_quantification(
        config,
        overwrite_fits=args.overwrite_fits,
        overwrite_maps=args.overwrite_maps,
        progress=print,
    )
    print(result)
    print(
        "Scientific review required: inspect stain maps and QC, then approve "
        f"{config.output_dir / 'stain_review.json'}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
