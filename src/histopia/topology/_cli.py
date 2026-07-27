"""Command-line entry point for semantic topology reconstruction."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from histopia._signals import graceful_sigterm
from histopia.topology._config import load_topology_config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reconstruct selected-K semantic topology across tissue sections."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor")
    for name in ("preflight", "benchmark", "run"):
        child = commands.add_parser(name)
        child.add_argument("--config", type=Path, required=True)
    approve = commands.add_parser("approve")
    approve.add_argument("--run", type=Path, required=True)
    approve.add_argument("--reviewer", required=True)
    approve.add_argument("--review-notes", required=True)
    qc = commands.add_parser("qc")
    qc.add_argument("--run", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the topology CLI with graceful cancellation."""

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
                            "scikit-image": "skimage",
                        }.items()
                    },
                },
                indent=2,
            )
        )
        return 0
    if args.command == "approve":
        from histopia.topology._approval import approve_topology_result

        approval = approve_topology_result(
            args.run,
            reviewer=args.reviewer,
            notes=args.review_notes,
        )
        print(
            f"{approval.run_dir / 'topology_review.json'}: "
            f"fingerprint={approval.fingerprint}"
        )
        return 0
    if args.command == "qc":
        from dataclasses import asdict

        from histopia.topology._qc import summarize_topology_run

        print(json.dumps(asdict(summarize_topology_run(args.run)), indent=2))
        return 0
    config = load_topology_config(args.config)
    if args.command == "preflight":
        from histopia._atomic import write_json_atomic
        from histopia.topology._pipeline import preflight_topology

        payload = preflight_topology(config)
        output = config.output_dir / "preflight.json"
        write_json_atomic(output, payload)
        print(
            f"{output}: {len(payload['slide_ids'])} sections, "
            f"fingerprint={payload['fingerprint']}"
        )
        return 0
    if args.command == "benchmark":
        from histopia.topology._pipeline import benchmark_topology

        path = benchmark_topology(config, progress=print)
        print(path)
        return 0
    from histopia.topology._pipeline import build_topology

    result = build_topology(config, progress=print)
    print(result)
    print(
        "Scientific review required: inspect gap decisions, uncertainty, and "
        f"surfaces, then approve {config.output_dir / 'topology_review.json'}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
