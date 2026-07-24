"""Command line entry point for Histopia viewer generation and serving."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from histopia.visualization._qc_showcase import export_registration_qc_showcase
from histopia.visualization._server import serve_viewer
from histopia.visualization._showcase import export_static_showcase
from histopia.visualization._viewer import build_section_viewer


def _named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not name or not raw_path:
        raise argparse.ArgumentTypeError("expected non-empty NAME=PATH")
    return name, Path(raw_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build and serve Histopia viewers.")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="Serve a generated viewer root.")
    serve.add_argument("root", type=Path)
    serve.add_argument("--bind", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8765)
    build = commands.add_parser("build", help="Build the stable viewer endpoint.")
    build.add_argument("root", type=Path, help="Viewer root containing histopia/.")
    build.add_argument("--run", type=_named_path, action="append", required=True)
    build.add_argument("--semantic-run", type=_named_path, action="append", default=[])
    build.add_argument("--cohort-qc", type=Path)
    mask_review = commands.add_parser(
        "mask-review",
        help="Build a fixed-viewport accepted-mask audit.",
    )
    mask_review.add_argument("registration_run", type=Path)
    mask_review.add_argument("output", type=Path)
    order_review = commands.add_parser(
        "order-review",
        help="Build a fixed-viewport section-order review.",
    )
    order_review.add_argument("proposal", type=Path)
    order_review.add_argument("processed", type=Path)
    order_review.add_argument("output", type=Path)
    order_review.add_argument("--workers", type=int, default=1)
    showcase = commands.add_parser(
        "showcase",
        help="Export selected viewer mice as a static site.",
    )
    showcase.add_argument("source", type=Path, help="Generated Histopia site.")
    showcase.add_argument("output", type=Path, help="New static output directory.")
    showcase.add_argument(
        "--mouse",
        action="append",
        required=True,
        help="Exact viewer mouse ID; repeat to export a cohort.",
    )
    qc_showcase = commands.add_parser(
        "qc-showcase",
        help="Export registration workflow reviews as a static portal.",
    )
    qc_showcase.add_argument("source", type=Path, help="Generated Histopia site.")
    qc_showcase.add_argument("output", type=Path, help="New static QC directory.")
    qc_showcase.add_argument(
        "--mouse",
        action="append",
        required=True,
        help="Exact viewer mouse ID; repeat to export a cohort.",
    )
    args = parser.parse_args(argv)

    if args.command == "mask-review":
        from histopia.visualization._viewer import build_mask_review

        index = build_mask_review(args.registration_run, args.output)
        print(index)
        return 0
    if args.command == "order-review":
        from histopia.visualization._viewer import build_section_order_review

        index = build_section_order_review(
            args.proposal,
            args.processed,
            args.output,
            workers=args.workers,
        )
        print(index)
        return 0
    if args.command == "build":
        index = build_section_viewer(
            dict(args.run),
            args.root / "histopia",
            semantic_runs=dict(args.semantic_run),
            cohort_qc=args.cohort_qc,
        )
        print(index)
        return 0
    if args.command == "showcase":
        index = export_static_showcase(args.source, args.output, args.mouse)
        print(index)
        return 0
    if args.command == "qc-showcase":
        index = export_registration_qc_showcase(args.source, args.output, args.mouse)
        print(index)
        return 0
    if args.command == "serve":
        serve_viewer(args.root, bind=args.bind, port=args.port)
        return 0
    parser.error(f"unsupported command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
