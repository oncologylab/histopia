"""Command line interface for QuPath interchange bundles."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from histopia.qupath._doctor import (
    QUPATH_WORKFLOW_API_VERSION,
    QUPATH_WORKFLOWS,
    inspect_qupath_environment,
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check Histopia or export validated results for QuPath."
    )
    parser.add_argument(
        "--doctor",
        action="store_true",
        help="Validate the selected Python environment for a QuPath workflow.",
    )
    parser.add_argument(
        "--workflow",
        choices=QUPATH_WORKFLOWS,
        default="full",
        help="Workflow checked by --doctor (default: full).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Compute device checked for semantic or full workflows.",
    )
    parser.add_argument(
        "--require-api",
        type=_positive_int,
        default=QUPATH_WORKFLOW_API_VERSION,
        help="Minimum QuPath workflow API required by the caller.",
    )
    parser.add_argument("--registration-run", type=Path)
    parser.add_argument("--semantic-run", type=Path)
    parser.add_argument("--clusters", type=int)
    parser.add_argument(
        "--semantic-geometry",
        choices=("regions", "tiles"),
        default="regions",
        help="Coalesced regions (default) or one rectangle per source patch.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.doctor:
        if any(
            value is not None
            for value in (
                args.registration_run,
                args.semantic_run,
                args.clusters,
                args.output,
            )
        ):
            parser.error("--doctor cannot be combined with export arguments")
        try:
            report = inspect_qupath_environment(
                args.workflow,
                device=args.device,
                required_api=args.require_api,
            )
        except (RuntimeError, ValueError) as error:
            parser.error(str(error))
        print(json.dumps(report, indent=2))
        return 0
    if args.registration_run is None:
        parser.error("--registration-run is required for export")
    if args.output is None:
        parser.error("--output is required for export")
    if args.clusters is not None and args.semantic_run is None:
        parser.error("--clusters requires --semantic-run")
    from histopia.qupath._export import export_qupath_bundle

    result = export_qupath_bundle(
        args.registration_run,
        args.output,
        semantic_run=args.semantic_run,
        clusters=args.clusters,
        semantic_geometry=args.semantic_geometry,
    )
    print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
