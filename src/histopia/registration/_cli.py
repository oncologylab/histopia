"""Command line entry point for Histopia registration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from histopia._signals import graceful_sigterm
from histopia.registration._config import (
    load_registration_config,
    registration_config_from_mapping,
)
from histopia.registration._manifest import build_kpf_manifest


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def main(argv: list[str] | None = None) -> int:
    """Run the registration CLI with graceful launcher cancellation."""

    with graceful_sigterm():
        return _main(argv)


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Histopia registration, validation, and WSI export."
    )
    parser.add_argument("--config", type=Path, help="JSON or TOML registration config.")
    parser.add_argument(
        "--approve-run",
        type=Path,
        help="Seal exact reviewed masks and section order for a completed run.",
    )
    parser.add_argument(
        "--approve-masks",
        type=Path,
        help="Approve exact prepared tissue masks before registration.",
    )
    parser.add_argument(
        "--approve-order",
        type=Path,
        help="Approve the exact prepared section-order proposal.",
    )
    parser.add_argument(
        "--prepare-completed-review",
        type=Path,
        help=(
            "Prepare a fingerprinted order review for a completed legacy run "
            "without granting approval."
        ),
    )
    parser.add_argument(
        "--reviewer",
        help="Reviewer name required with --approve-run.",
    )
    parser.add_argument(
        "--review-notes",
        help="Review notes required with --approve-run.",
    )
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
        "--staged",
        action="store_true",
        help=(
            "Return review-required stages as successful JSON statuses instead "
            "of exceptions."
        ),
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
    parser.add_argument(
        "--warp-slide",
        action="append",
        help=(
            "Export one exact source filename or stem used with --warp-run; "
            "repeat to select multiple slides."
        ),
    )
    parser.add_argument(
        "--vips-threads",
        type=_positive_int,
        help="Bound native libvips workers used with --warp-run.",
    )
    parser.add_argument(
        "--viewer-run",
        action="append",
        default=[],
        metavar="MOUSE=RUN_DIR",
        help="Add a completed mouse run to a static section viewer.",
    )
    parser.add_argument(
        "--viewer-output-dir",
        type=Path,
        help="Output directory used with --viewer-run.",
    )
    parser.add_argument(
        "--viewer-semantic-run",
        action="append",
        default=[],
        metavar="MOUSE=SEMANTIC_DIR",
        help="Add semantic atlas textures to a viewer mouse.",
    )
    parser.add_argument(
        "--viewer-workers",
        type=int,
        default=1,
        help=(
            "Bound concurrent WebP encoders used with --viewer-run; "
            "default 1 minimizes memory use."
        ),
    )
    parser.add_argument(
        "--provisional-mouse",
        action="append",
        default=[],
        help="Mark a viewer mouse as having provisional physical order.",
    )
    args = parser.parse_args(argv)

    approval_actions = tuple(
        path
        for path in (
            args.prepare_completed_review,
            args.approve_masks,
            args.approve_order,
            args.approve_run,
        )
        if path is not None
    )
    if len(approval_actions) > 1:
        parser.error(
            "--prepare-completed-review and approval actions are mutually exclusive"
        )

    if args.prepare_completed_review is not None:
        from histopia.registration._approval import (
            prepare_completed_registration_review,
        )

        path = prepare_completed_registration_review(args.prepare_completed_review)
        payload = json.loads(path.read_text())
        print(
            json.dumps(
                {
                    "status": "review_required",
                    "stage": "order",
                    "run_dir": str(args.prepare_completed_review),
                    "slide_count": len(payload["slides"]),
                    "order_fingerprint": payload["fingerprint"],
                },
                indent=2,
            )
        )
        return 0

    if args.approve_masks is not None:
        if not args.reviewer or not args.review_notes:
            parser.error(
                "--reviewer and --review-notes are required with --approve-masks"
            )
        from histopia.registration._approval import approve_mask_review

        approval = approve_mask_review(
            args.approve_masks,
            reviewer=args.reviewer,
            notes=args.review_notes,
        )
        print(
            json.dumps(
                {
                    "status": "approved",
                    "stage": "masks",
                    "run_dir": str(approval.run_dir),
                    "slide_count": approval.slide_count,
                    "mask_fingerprint": approval.mask_fingerprint,
                    "reviewer": approval.reviewer,
                    "reviewed_at": approval.reviewed_at,
                },
                indent=2,
            )
        )
        return 0

    if args.approve_order is not None:
        if not args.reviewer or not args.review_notes:
            parser.error(
                "--reviewer and --review-notes are required with --approve-order"
            )
        from histopia.registration._approval import approve_section_order

        approval = approve_section_order(
            args.approve_order,
            reviewer=args.reviewer,
            notes=args.review_notes,
        )
        print(
            json.dumps(
                {
                    "status": "approved",
                    "stage": "order",
                    "run_dir": str(approval.run_dir),
                    "slide_count": approval.slide_count,
                    "order_fingerprint": approval.order_fingerprint,
                    "reviewer": approval.reviewer,
                    "reviewed_at": approval.reviewed_at,
                },
                indent=2,
            )
        )
        return 0

    if args.approve_run is not None:
        if not args.reviewer or not args.review_notes:
            parser.error(
                "--reviewer and --review-notes are required with --approve-run"
            )
        from histopia.registration._approval import approve_registration_run

        approval = approve_registration_run(
            args.approve_run,
            reviewer=args.reviewer,
            notes=args.review_notes,
        )
        print(
            json.dumps(
                {
                    "run_dir": str(approval.run_dir),
                    "slide_count": approval.slide_count,
                    "order_fingerprint": approval.order_fingerprint,
                    "reviewer": approval.reviewer,
                    "reviewed_at": approval.reviewed_at,
                    "registration_result_sha256": (approval.registration_result_sha256),
                },
                indent=2,
            )
        )
        return 0

    if args.viewer_run:
        if args.viewer_output_dir is None:
            parser.error("--viewer-output-dir is required with --viewer-run")
        from histopia.visualization import build_section_viewer

        runs: dict[str, Path] = {}
        for item in args.viewer_run:
            if "=" not in item:
                parser.error("--viewer-run must use MOUSE=RUN_DIR")
            mouse, run_dir = item.split("=", 1)
            runs[mouse] = Path(run_dir)
        semantic_runs: dict[str, Path] = {}
        for item in args.viewer_semantic_run:
            if "=" not in item:
                parser.error("--viewer-semantic-run must use MOUSE=SEMANTIC_DIR")
            mouse, semantic_dir = item.split("=", 1)
            if mouse not in runs:
                parser.error("--viewer-semantic-run mouse must also use --viewer-run")
            semantic_runs[mouse] = Path(semantic_dir)
        index_path = build_section_viewer(
            runs,
            args.viewer_output_dir,
            provisional_mice=set(args.provisional_mouse),
            semantic_runs=semantic_runs,
            workers=args.viewer_workers,
        )
        print(index_path)
        return 0

    if args.warp_run is not None:
        from histopia.registration._wsi import warp_saved_registration

        results = warp_saved_registration(
            args.warp_run,
            args.registered_output_dir,
            overwrite=args.overwrite,
            crop_mode=args.warp_crop_mode,
            accepted_non_rigid_only=args.accepted_non_rigid_only,
            slide_names=args.warp_slide,
            vips_threads=args.vips_threads,
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
        parser.error(
            "--config, --approve-masks, --approve-order, --approve-run, --manifest, "
            "--warp-run, or --viewer-run is required"
        )

    from histopia.registration._errors import RegistrationApprovalRequired
    from histopia.registration._pipeline import register_sections

    config = load_registration_config(args.config)
    try:
        result = register_sections(config)
    except RegistrationApprovalRequired as error:
        if not args.staged:
            raise
        print(json.dumps(error.to_json_dict(), indent=2))
        return 0
    print(json.dumps(result.to_json_dict(), indent=2))
    return 0


_load_config = load_registration_config
_config_from_mapping = registration_config_from_mapping


if __name__ == "__main__":
    sys.exit(main())
