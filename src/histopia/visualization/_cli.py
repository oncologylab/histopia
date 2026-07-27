"""Command line entry point for Histopia viewer generation and serving."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from histopia._signals import graceful_sigterm


def build_section_viewer(*args, **kwargs) -> Path:
    """Lazily dispatch viewer generation without importing optional dependencies."""

    from histopia.visualization._viewer import build_section_viewer as build

    return build(*args, **kwargs)


def export_static_showcase(source: Path, output: Path, mice: list[str]) -> Path:
    """Lazily dispatch static showcase export."""

    from histopia.visualization._showcase import export_static_showcase as export

    return export(source, output, mice)


def export_registration_qc_showcase(
    source: Path,
    output: Path,
    mice: list[str],
) -> Path:
    """Lazily dispatch registration QC showcase export."""

    from histopia.visualization._qc_showcase import (
        export_registration_qc_showcase as export,
    )

    return export(source, output, mice)


def serve_viewer(
    root: Path,
    *,
    bind: str,
    port: int,
    required_routes: tuple[str, ...],
    review_config: Path | None,
) -> None:
    """Lazily dispatch the static viewer server."""

    from histopia.visualization._server import serve_viewer as serve

    serve(
        root,
        bind=bind,
        port=port,
        required_routes=required_routes,
        review_config=review_config,
    )


def audit_workflows(*args, **kwargs):
    """Lazily dispatch workflow integrity auditing."""

    from histopia.visualization._audit import audit_workflows as audit

    return audit(*args, **kwargs)


def _named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not name or not raw_path:
        raise argparse.ArgumentTypeError("expected non-empty NAME=PATH")
    return name, Path(raw_path)


def main(argv: list[str] | None = None) -> int:
    """Run viewer commands with graceful launcher cancellation."""

    with graceful_sigterm():
        return _main(argv)


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build and serve Histopia viewers.")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="Serve a generated viewer root.")
    serve.add_argument("root", type=Path)
    serve.add_argument("--bind", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument(
        "--require-route",
        action="append",
        default=[],
        help="Route directory that must contain index.html; repeat as needed.",
    )
    serve.add_argument(
        "--review-config",
        type=Path,
        help="Local path registry enabling authenticated web approvals.",
    )
    build = commands.add_parser("build", help="Build the stable viewer endpoint.")
    build.add_argument("root", type=Path, help="Viewer root containing histopia/.")
    build.add_argument("--run", type=_named_path, action="append", required=True)
    build.add_argument("--semantic-run", type=_named_path, action="append", default=[])
    build.add_argument("--stain-run", type=_named_path, action="append", default=[])
    build.add_argument("--cohort-qc", type=Path)
    build.add_argument(
        "--include-unapproved",
        action="store_true",
        help="Build a review viewer containing unapproved workflow stages.",
    )
    build.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Bound concurrent WebP encoders; default 1 minimizes memory use.",
    )
    mask_review = commands.add_parser(
        "mask-review",
        help="Build a fixed-viewport accepted-mask audit.",
    )
    mask_review.add_argument("registration_run", type=Path)
    mask_review.add_argument("output", type=Path)
    mask_review.add_argument("--workers", type=int, default=1)
    non_rigid_review = commands.add_parser(
        "non-rigid-review",
        help="Build a fixed-viewport provisional dense-field audit.",
    )
    non_rigid_review.add_argument(
        "source_run",
        type=Path,
        help="Validation bundle or non-rigid registration run.",
    )
    non_rigid_review.add_argument("output", type=Path)
    non_rigid_review.add_argument("--workers", type=int, default=1)
    registration_review = commands.add_parser(
        "registration-review",
        help="Build one local portal for mask and section-order review.",
    )
    registration_review.add_argument("registration_run", type=Path)
    registration_review.add_argument("output", type=Path)
    registration_review.add_argument("--workers", type=int, default=1)
    cohort_review = commands.add_parser(
        "registration-cohort-review",
        help="Build one local portal for multiple registration reviews.",
    )
    cohort_review.add_argument("output", type=Path)
    cohort_review.add_argument(
        "--run",
        type=_named_path,
        action="append",
        required=True,
    )
    cohort_review.add_argument("--workers", type=int, default=1)
    workflow_review = commands.add_parser(
        "review",
        help="Build one stable review hub for prepared workflow stages.",
    )
    workflow_review.add_argument("output", type=Path)
    workflow_review.add_argument(
        "--run",
        type=_named_path,
        action="append",
        required=True,
    )
    workflow_review.add_argument(
        "--semantic-run",
        type=_named_path,
        action="append",
        default=[],
    )
    workflow_review.add_argument(
        "--stain-run",
        type=_named_path,
        action="append",
        default=[],
    )
    workflow_review.add_argument(
        "--topology-run",
        type=_named_path,
        action="append",
        default=[],
    )
    workflow_review.add_argument("--cohort-qc", type=Path)
    workflow_review.add_argument("--workers", type=int, default=1)
    stain_review = commands.add_parser(
        "stain-review",
        help="Build a decision-focused review portal from generated stain assets.",
    )
    stain_review.add_argument(
        "viewer",
        type=Path,
        help="Generated Histopia application containing manifest.json.",
    )
    stain_review.add_argument("output", type=Path)
    stain_review.add_argument(
        "--mouse",
        action="append",
        help="Exact viewer mouse ID; repeat to select a cohort.",
    )
    stain_review.add_argument(
        "--issues",
        type=Path,
        help="Optional JSON notes keyed by mouse then slide ID or order.",
    )
    topology_review = commands.add_parser(
        "topology-review",
        help="Build a fixed-viewport semantic topology surface reviewer.",
    )
    topology_review.add_argument("output", type=Path)
    topology_review.add_argument(
        "--run",
        type=_named_path,
        action="append",
        required=True,
    )
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
    audit = commands.add_parser(
        "audit",
        help="Validate registration, semantic, and viewer workflow integrity.",
    )
    audit.add_argument(
        "--run",
        type=_named_path,
        action="append",
        required=True,
        help="Named registration run as NAME=PATH; repeat for a cohort.",
    )
    audit.add_argument(
        "--semantic-run",
        type=_named_path,
        action="append",
        default=[],
        help="Named semantic run as NAME=PATH; repeat for a cohort.",
    )
    audit.add_argument(
        "--stain-run",
        type=_named_path,
        action="append",
        default=[],
        help="Named stain run as NAME=PATH; repeat for a cohort.",
    )
    audit.add_argument(
        "--viewer-manifest",
        type=Path,
        help="Optional generated viewer manifest.json to verify.",
    )
    audit.add_argument(
        "--output",
        type=Path,
        help="Optional path for the portable JSON audit.",
    )
    feedback_export = commands.add_parser(
        "feedback-export",
        help="Export latest registration feedback as flat learning rows.",
    )
    feedback_export.add_argument("feedback_root", type=Path)
    feedback_export.add_argument("output", type=Path)
    args = parser.parse_args(argv)

    if args.command == "mask-review":
        from histopia.visualization._viewer import build_mask_review

        index = build_mask_review(
            args.registration_run,
            args.output,
            workers=args.workers,
        )
        print(index)
        return 0
    if args.command == "non-rigid-review":
        from histopia.visualization._nonrigid_review import build_non_rigid_review

        index = build_non_rigid_review(
            args.source_run,
            args.output,
            workers=args.workers,
        )
        print(index)
        return 0
    if args.command == "registration-review":
        from histopia.visualization._review_portal import build_registration_review

        index = build_registration_review(
            args.registration_run,
            args.output,
            workers=args.workers,
        )
        print(index)
        return 0
    if args.command == "registration-cohort-review":
        from histopia.visualization._review_portal import (
            build_registration_cohort_review,
        )

        index = build_registration_cohort_review(
            dict(args.run),
            args.output,
            workers=args.workers,
        )
        print(index)
        return 0
    if args.command == "review":
        from histopia.visualization._review_portal import build_workflow_review

        index = build_workflow_review(
            _unique_named_paths(args.run, "registration"),
            args.output,
            semantic_runs=_unique_named_paths(args.semantic_run, "semantic"),
            stain_runs=_unique_named_paths(args.stain_run, "stain"),
            topology_runs=_unique_named_paths(args.topology_run, "topology"),
            cohort_qc=args.cohort_qc,
            workers=args.workers,
        )
        print(index)
        return 0
    if args.command == "topology-review":
        from histopia.visualization._topology_review import build_topology_review

        index = build_topology_review(
            _unique_named_paths(args.run, "topology"),
            args.output,
        )
        print(index)
        return 0
    if args.command == "stain-review":
        from histopia.visualization._stain_review import (
            build_stain_review,
            load_stain_review_issues,
        )

        issues = load_stain_review_issues(args.issues) if args.issues else None
        index = build_stain_review(
            args.viewer,
            args.output,
            mice=args.mouse,
            issues=issues,
        )
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
            stain_runs=dict(args.stain_run),
            cohort_qc=args.cohort_qc,
            workers=args.workers,
            require_approvals=not args.include_unapproved,
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
    if args.command == "audit":
        from histopia.visualization._audit import write_workflow_audit

        report = audit_workflows(
            _unique_named_paths(args.run, "registration"),
            semantic_runs=_unique_named_paths(args.semantic_run, "semantic"),
            stain_runs=_unique_named_paths(args.stain_run, "stain"),
            viewer_manifest=args.viewer_manifest,
        )
        if args.output is not None:
            write_workflow_audit(report, args.output)
        print(json.dumps(report.to_json_dict(), sort_keys=True))
        return report.exit_code
    if args.command == "feedback-export":
        from histopia._atomic import write_json_atomic
        from histopia.visualization._feedback import (
            registration_feedback_rows,
            summarize_registration_feedback,
        )

        payload = {
            "schema_version": 1,
            "summary": summarize_registration_feedback(args.feedback_root),
            "rows": registration_feedback_rows(args.feedback_root),
        }
        write_json_atomic(args.output, payload)
        print(args.output)
        return 0
    if args.command == "serve":
        serve_viewer(
            args.root,
            bind=args.bind,
            port=args.port,
            required_routes=tuple(args.require_route or ["histopia"]),
            review_config=args.review_config,
        )
        return 0
    parser.error(f"unsupported command: {args.command}")


def _unique_named_paths(
    values: list[tuple[str, Path]],
    kind: str,
) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for name, path in values:
        if name in result:
            raise ValueError(f"duplicate {kind} run name: {name}")
        result[name] = path
    return result


if __name__ == "__main__":
    sys.exit(main())
