"""Command line interface for QuPath interchange bundles."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from histopia.qupath._export import export_qupath_bundle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export Histopia registration and semantic results for QuPath."
    )
    parser.add_argument("--registration-run", type=Path, required=True)
    parser.add_argument("--semantic-run", type=Path)
    parser.add_argument("--clusters", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.clusters is not None and args.semantic_run is None:
        parser.error("--clusters requires --semantic-run")
    result = export_qupath_bundle(
        args.registration_run,
        args.output,
        semantic_run=args.semantic_run,
        clusters=args.clusters,
    )
    print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
