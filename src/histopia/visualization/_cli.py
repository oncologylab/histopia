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


def export_static_showcase(
    source: Path,
    output: Path,
    mice: list[str],
    **kwargs,
) -> Path:
    """Lazily dispatch static showcase export."""

    from histopia.visualization._showcase import export_static_showcase as export

    return export(source, output, mice, **kwargs)


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
    public_review_write: bool,
) -> None:
    """Lazily dispatch the static viewer server."""

    from histopia.visualization._server import serve_viewer as serve

    serve(
        root,
        bind=bind,
        port=port,
        required_routes=required_routes,
        review_config=review_config,
        public_review_write=public_review_write,
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


def _named_text(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=VALUE")
    name, text = value.split("=", 1)
    if not name or not text:
        raise argparse.ArgumentTypeError("expected non-empty NAME=VALUE")
    return name, text


def _scoped_named_path(value: str) -> tuple[str, str, Path]:
    """Parse a model-scoped path as ``COHORT:MODEL=PATH``."""

    if "=" not in value or ":" not in value.split("=", 1)[0]:
        raise argparse.ArgumentTypeError("expected COHORT:MODEL=PATH")
    scope, raw_path = value.split("=", 1)
    cohort, model = scope.split(":", 1)
    if not cohort or not model or not raw_path:
        raise argparse.ArgumentTypeError("expected non-empty COHORT:MODEL=PATH")
    return cohort, model, Path(raw_path)


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
        help="Local path registry enabling fingerprint-bound web approvals.",
    )
    serve.add_argument(
        "--public-review-write",
        action="store_true",
        help=(
            "Allow same-origin review writes without an access key; "
            "authentication remains required by default."
        ),
    )
    build = commands.add_parser("build", help="Build the stable viewer endpoint.")
    build.add_argument("root", type=Path, help="Viewer root containing histopia/.")
    build.add_argument("--run", type=_named_path, action="append", required=True)
    build.add_argument("--semantic-run", type=_named_path, action="append", default=[])
    build.add_argument("--stain-run", type=_named_path, action="append", default=[])
    build.add_argument(
        "--registered-wsi",
        type=_named_path,
        action="append",
        default=[],
        help="Approved full-resolution export as NAME=PATH; repeat as needed.",
    )
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
    workflow_review.add_argument(
        "--cell-run",
        type=_named_path,
        action="append",
        default=[],
    )
    workflow_review.add_argument(
        "--cell-geometry",
        type=_named_path,
        action="append",
        default=[],
        help=(
            "Reusable multiscale cell-geometry cache as NAME=PATH; repeat for "
            "cohorts included in the static cellular protein atlas."
        ),
    )
    workflow_review.add_argument(
        "--protein-run",
        type=_named_path,
        action="append",
        default=[],
        help="Sealed protein prediction run as NAME=PATH; repeat per cohort.",
    )
    workflow_review.add_argument(
        "--protein-model",
        type=_scoped_named_path,
        action="append",
        default=[],
        help=(
            "Model-scoped protein run as COHORT:MODEL=PATH; repeat to expose "
            "targets, architectures, or training-data variants."
        ),
    )
    workflow_review.add_argument(
        "--annotation-run",
        type=_named_path,
        action="append",
        default=[],
        help="Annotation store as NAME=PATH; repeat for a cohort.",
    )
    workflow_review.add_argument(
        "--registered-wsi",
        type=_named_path,
        action="append",
        default=[],
        help="Approved full-resolution export as NAME=PATH; repeat as needed.",
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
    cell_review = commands.add_parser(
        "cell-review",
        help="Build a native-resolution cell-boundary reviewer.",
    )
    cell_review.add_argument("output", type=Path)
    cell_review.add_argument(
        "--run",
        type=_named_path,
        action="append",
        required=True,
    )
    protein_review = commands.add_parser(
        "protein-review",
        help="Build a three-pane protein-expression prediction reviewer.",
    )
    protein_review.add_argument("output", type=Path)
    protein_review.add_argument(
        "--run", type=_named_path, action="append", required=True
    )
    protein_atlas = commands.add_parser(
        "protein-atlas",
        help="Build a static cell-resolved 3D and orthogonal protein atlas.",
    )
    protein_atlas.add_argument("output", type=Path)
    protein_atlas.add_argument(
        "--run", type=_named_path, action="append", required=True
    )
    protein_atlas.add_argument(
        "--cell-run", type=_named_path, action="append", required=True
    )
    protein_atlas.add_argument(
        "--protein-model",
        type=_scoped_named_path,
        action="append",
        required=True,
    )
    protein_atlas.add_argument(
        "--cell-geometry", type=_named_path, action="append", required=True
    )
    protein_atlas.add_argument(
        "--topology-run", type=_named_path, action="append", default=[]
    )
    protein_atlas.add_argument("--overview-cells", type=int, default=500_000)
    protein_atlas.add_argument("--max-bytes", type=int, default=650 * 1024 * 1024)
    protein_atlas.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Bounded label-artifact validation workers. Default: 1.",
    )
    protein_atlas.add_argument(
        "--default-layer",
        choices=("cells", "protein"),
        default="protein",
        help="Initial atlas layer. Default: protein.",
    )
    protein_atlas.add_argument(
        "--default-target",
        action="append",
        help="Initial protein target; repeat to define a multichannel composite.",
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
    showcase.add_argument(
        "--review-config",
        type=Path,
        help="Local review registry required for native-resolution static tiles.",
    )
    showcase.add_argument(
        "--wsi-section",
        type=_named_text,
        action="append",
        default=[],
        metavar="MOUSE=SECTION",
        help="Embed one approved three-digit WSI section; repeat as needed.",
    )
    showcase.add_argument(
        "--max-bytes",
        type=int,
        default=900 * 1024 * 1024,
        help="Hard size limit when embedding WSI tiles. Default: 900 MiB.",
    )
    showcase.add_argument(
        "--protein-atlas",
        type=Path,
        help="Include a prebuilt static cellular protein atlas.",
    )
    showcase.add_argument(
        "--protein-atlas-max-bytes",
        type=int,
        default=650 * 1024 * 1024,
        help="Independent size limit for the cellular protein atlas.",
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
        help="Validate scientific workflow and viewer integrity.",
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
        "--topology-run",
        type=_named_path,
        action="append",
        default=[],
        help="Named topology run as NAME=PATH; repeat for a cohort.",
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
    completeness = commands.add_parser(
        "completeness",
        help="Report registration, stain, and reference-cell coverage.",
    )
    completeness.add_argument("--review-config", type=Path, required=True)
    completeness.add_argument(
        "--cohort",
        action="append",
        default=[],
        help="Exact configured cohort ID; repeat to restrict the audit.",
    )
    completeness.add_argument("--output", type=Path)
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

        cell_runs = _unique_named_paths(args.cell_run, "cell")
        annotation_runs = _unique_named_paths(args.annotation_run, "annotation")
        options = {
            "semantic_runs": _unique_named_paths(args.semantic_run, "semantic"),
            "stain_runs": _unique_named_paths(args.stain_run, "stain"),
            "topology_runs": _unique_named_paths(args.topology_run, "topology"),
            "registered_wsi": _unique_named_paths(
                args.registered_wsi,
                "registered WSI",
            ),
            "cohort_qc": args.cohort_qc,
            "workers": args.workers,
        }
        if cell_runs:
            options["cell_runs"] = cell_runs
        cell_geometry_runs = _unique_named_paths(args.cell_geometry, "cell geometry")
        if cell_geometry_runs:
            options["cell_geometry_runs"] = cell_geometry_runs
        if annotation_runs:
            options["annotation_runs"] = annotation_runs
        protein_runs = _unique_named_paths(args.protein_run, "protein")
        if protein_runs:
            options["protein_runs"] = protein_runs
        protein_models = _unique_scoped_paths(args.protein_model, "protein model")
        if protein_models:
            options["protein_models"] = protein_models
        index = build_workflow_review(
            _unique_named_paths(args.run, "registration"),
            args.output,
            **options,
        )
        print(index)
        return 0
    if args.command == "protein-review":
        from histopia.visualization._protein_review import build_protein_review

        index = build_protein_review(
            _unique_named_paths(args.run, "protein"), args.output
        )
        print(index)
        return 0
    if args.command == "protein-atlas":
        from histopia.visualization._cellular_protein_atlas import (
            build_cellular_protein_atlas,
        )

        index = build_cellular_protein_atlas(
            _unique_named_paths(args.run, "registration"),
            _unique_named_paths(args.cell_run, "cell"),
            _unique_scoped_paths(args.protein_model, "protein model"),
            _unique_named_paths(args.cell_geometry, "cell geometry"),
            args.output,
            topology_runs=_unique_named_paths(args.topology_run, "topology"),
            max_overview_cells=args.overview_cells,
            max_bytes=args.max_bytes,
            default_layer=args.default_layer,
            default_targets=args.default_target,
            workers=args.workers,
            progress=print,
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
    if args.command == "cell-review":
        from histopia.visualization._cell_review import build_cell_review

        index = build_cell_review(
            _unique_named_paths(args.run, "cell"),
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
            registered_wsi=dict(args.registered_wsi),
            cohort_qc=args.cohort_qc,
            workers=args.workers,
            require_approvals=not args.include_unapproved,
        )
        print(index)
        return 0
    if args.command == "showcase":
        options = {}
        if args.review_config is not None or args.wsi_section:
            sections: dict[str, list[str]] = {}
            for mouse, section in args.wsi_section:
                sections.setdefault(mouse, []).append(section)
            options = {
                "review_config": args.review_config,
                "wsi_sections": sections,
                "max_bytes": args.max_bytes,
            }
        if args.protein_atlas is not None:
            options.update(
                {
                    "protein_atlas": args.protein_atlas,
                    "protein_atlas_max_bytes": args.protein_atlas_max_bytes,
                    "max_bytes": args.max_bytes,
                }
            )
        index = export_static_showcase(
            args.source,
            args.output,
            args.mouse,
            **options,
        )
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
            topology_runs=_unique_named_paths(args.topology_run, "topology"),
            viewer_manifest=args.viewer_manifest,
        )
        if args.output is not None:
            write_workflow_audit(report, args.output)
        print(json.dumps(report.to_json_dict(), sort_keys=True))
        return report.exit_code
    if args.command == "completeness":
        from histopia.visualization._completeness import (
            audit_analysis_completeness,
            write_analysis_completeness,
        )
        from histopia.visualization._review_api import ReviewDecisionService

        configured = ReviewDecisionService.from_file(
            args.review_config
        ).configured_runs()
        unknown = sorted(set(args.cohort) - set(configured))
        if unknown:
            raise ValueError("unknown completeness cohorts: " + ", ".join(unknown))
        selected = (
            {name: configured[name] for name in args.cohort}
            if args.cohort
            else configured
        )
        report = audit_analysis_completeness(selected)
        if args.output is not None:
            write_analysis_completeness(report, args.output)
        print(json.dumps(report.to_json_dict(), sort_keys=True))
        return 0 if report.complete else 2
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
            public_review_write=args.public_review_write,
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


def _unique_scoped_paths(
    values: list[tuple[str, str, Path]],
    kind: str,
) -> dict[str, dict[str, Path]]:
    result: dict[str, dict[str, Path]] = {}
    for cohort, name, path in values:
        scoped = result.setdefault(cohort, {})
        if name in scoped:
            raise ValueError(f"duplicate {kind} run name: {cohort}:{name}")
        scoped[name] = path
    return result


if __name__ == "__main__":
    sys.exit(main())
