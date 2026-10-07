"""Command-line entry point for cell-boundary workflows."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from histopia._signals import graceful_sigterm
from histopia.cells._config import CELLPOSE_MODELS, load_cell_config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect and review cell boundaries in approved native WSIs."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor")
    doctor.add_argument("--device", default="auto")
    cache = commands.add_parser("cache-model")
    cache.add_argument("--model", choices=CELLPOSE_MODELS, default="cpsam")
    cache.add_argument("--cache", type=Path)
    cache.add_argument("--device", default="cpu")
    benchmark = commands.add_parser("benchmark")
    benchmark.add_argument("--images", type=Path, required=True)
    benchmark.add_argument("--annotations", type=Path, required=True)
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--model-cache", type=Path, required=True)
    benchmark.add_argument("--model", action="append", choices=CELLPOSE_MODELS)
    benchmark.add_argument(
        "--method",
        action="append",
        choices=("direct", "multiscale", "combined", "containment"),
    )
    benchmark.add_argument("--device", default="auto")
    preflight = commands.add_parser("preflight")
    preflight.add_argument("--config", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument(
        "--processing-section",
        action="append",
        default=[],
        help=(
            "Process only this section while retaining the complete-run preflight "
            "identity. Repeat for disjoint distributed workers. A worker checkpoint "
            "is written instead of a final cell result."
        ),
    )
    validate = commands.add_parser("validate")
    validate.add_argument("--run", type=Path, required=True)
    review = commands.add_parser("review-section")
    review.add_argument("--run", type=Path, required=True)
    review.add_argument("--section", required=True)
    review.add_argument("--reviewer", default="browser")
    review.add_argument("--notes", default="")
    review.add_argument("--issue", action="append", default=[])
    decision = review.add_mutually_exclusive_group(required=True)
    decision.add_argument("--accept", action="store_true")
    decision.add_argument("--reject", action="store_true")
    approve = commands.add_parser("approve")
    approve.add_argument("--run", type=Path, required=True)
    compose = commands.add_parser("compose")
    compose.add_argument("--base-run", type=Path, required=True)
    compose.add_argument("--output", type=Path, required=True)
    compose.add_argument(
        "--replace",
        action="append",
        required=True,
        metavar="SECTION=RUN",
        help="Use SECTION from a separately validated source run; repeat as needed.",
    )
    chromatic = commands.add_parser("chromatic-refilter")
    chromatic.add_argument("--source-run", type=Path, required=True)
    chromatic.add_argument("--output", type=Path, required=True)
    chromatic.add_argument(
        "--section-profile",
        action="append",
        required=True,
        metavar="SECTION=PROFILE",
        help="Apply one immutable chromatic profile; repeat for multiple sections.",
    )
    foreign = commands.add_parser("foreign-material-refilter")
    foreign.add_argument("--source-run", type=Path, required=True)
    foreign.add_argument("--output", type=Path, required=True)
    foreign.add_argument(
        "--section",
        action="append",
        required=True,
        help="Apply the exact saturated-cyan safeguard; repeat as needed.",
    )
    reviewed = commands.add_parser("reviewed-exclusion-refilter")
    reviewed.add_argument("--source-run", type=Path, required=True)
    reviewed.add_argument("--output", type=Path, required=True)
    reviewed.add_argument(
        "--model",
        choices=CELLPOSE_MODELS,
        default="cpsam",
        help="Require this exact Cellpose model identity in the sealed source.",
    )
    reviewed.add_argument(
        "--spec",
        type=Path,
        required=True,
        help="Path-free, source-bound JSON manifest of reviewed whole-label IDs.",
    )
    export = commands.add_parser("export-qupath")
    export.add_argument("--run", type=Path, required=True)
    identity = export.add_mutually_exclusive_group(required=True)
    identity.add_argument("--section")
    identity.add_argument("--slide")
    export.add_argument("--roi", required=True, help="Native pixel x,y,width,height")
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--max-cells", type=int, default=50_000)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the cell CLI with graceful cancellation."""

    with graceful_sigterm():
        return _main(argv)


def _main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "doctor":
        from histopia.compute import inspect_compute

        packages = {
            name: importlib.util.find_spec(module) is not None
            for name, module in {
                "numpy": "numpy",
                "scipy": "scipy",
                "pillow": "PIL",
                "pyvips": "pyvips",
                "tifffile": "tifffile",
                "cellpose": "cellpose",
                "torch": "torch",
            }.items()
        }
        compute = inspect_compute(args.device) if packages["torch"] else None
        print(json.dumps({"packages": packages, "compute": compute}, indent=2))
        return 0 if all(packages.values()) else 1
    if args.command == "cache-model":
        from histopia.cells._cellpose import load_cellpose_runtime, runtime_provenance
        from histopia.cells._config import CellSegmentationConfig

        runtime = load_cellpose_runtime(
            CellSegmentationConfig(
                registration_run=Path("."),
                output_dir=Path("."),
                model=args.model,
                model_cache=args.cache,
                device=args.device,
                allow_model_download=True,
            )
        )
        print(json.dumps(runtime_provenance(runtime), indent=2))
        return 0
    if args.command == "benchmark":
        from histopia.cells._benchmark import benchmark_cell_methods

        print(
            benchmark_cell_methods(
                args.images,
                args.annotations,
                args.output,
                model_cache=args.model_cache,
                models=tuple(args.model or CELLPOSE_MODELS),
                methods=tuple(args.method or ("direct", "combined", "containment")),
                device=args.device,
                progress=print,
            )
        )
        return 0
    if args.command == "validate":
        from histopia.cells._result import validate_cell_result

        result = validate_cell_result(args.run)
        print(f"{args.run}: fingerprint={result['fingerprint']}")
        return 0
    if args.command == "review-section":
        from histopia.cells._approval import review_cell_section

        status = review_cell_section(
            args.run,
            args.section,
            accepted=args.accept,
            reviewer=args.reviewer,
            notes=args.notes,
            issues=tuple(args.issue),
        )
        print(json.dumps(status, indent=2))
        return 0
    if args.command == "approve":
        from histopia.cells._approval import approve_cell_result

        approval = approve_cell_result(args.run)
        print(f"{approval.run_dir}: fingerprint={approval.fingerprint}")
        return 0
    if args.command == "compose":
        from histopia.cells._composition import compose_cell_result

        replacements: dict[str, Path] = {}
        for value in args.replace:
            section, separator, run = value.partition("=")
            if not separator or not section or not run:
                raise ValueError("--replace must use SECTION=RUN")
            if section in replacements:
                raise ValueError(f"duplicate replacement section: {section}")
            replacements[section] = Path(run)
        print(compose_cell_result(args.base_run, args.output, replacements))
        print(
            "Scientific review required: accept every composed section before approval."
        )
        return 0
    if args.command == "chromatic-refilter":
        from histopia.cells._chromatic_refilter import (
            refilter_chromatic_microclusters,
        )

        profiles: dict[str, str] = {}
        for value in args.section_profile:
            section, separator, profile = value.partition("=")
            if not separator or not section or not profile:
                raise ValueError("--section-profile must use SECTION=PROFILE")
            if section in profiles:
                raise ValueError(f"duplicate chromatic section: {section}")
            profiles[section] = profile
        print(
            refilter_chromatic_microclusters(
                args.source_run,
                args.output,
                profiles,
                progress=print,
            )
        )
        print("Scientific review required before section composition or approval.")
        return 0
    if args.command == "foreign-material-refilter":
        from histopia.cells._foreign_material_refilter import (
            refilter_saturated_cyan_foreign_material,
        )

        print(
            refilter_saturated_cyan_foreign_material(
                args.source_run,
                args.output,
                args.section,
                progress=print,
            )
        )
        print("Scientific review required before section composition or approval.")
        return 0
    if args.command == "reviewed-exclusion-refilter":
        from histopia.cells._reviewed_exclusion_refilter import (
            refilter_reviewed_artifact_exclusions,
        )

        print(
            refilter_reviewed_artifact_exclusions(
                args.source_run,
                args.output,
                args.spec,
                expected_model=args.model,
                progress=print,
            )
        )
        print("Only the exact approved, source-bound whole labels were removed.")
        return 0
    if args.command == "export-qupath":
        from histopia.cells._qupath import export_qupath_cell_roi

        try:
            roi = tuple(int(value) for value in args.roi.split(","))
        except ValueError as error:
            raise ValueError("ROI must use x,y,width,height integers") from error
        if len(roi) != 4:
            raise ValueError("ROI must use x,y,width,height integers")
        section = args.section or _section_for_slide(args.run, args.slide)
        print(
            export_qupath_cell_roi(
                args.run,
                section,
                roi,
                args.output,
                max_cells=args.max_cells,
            )
        )
        return 0
    config = load_cell_config(args.config)
    if args.command == "preflight":
        from histopia.cells._preflight import preflight_cell_run, write_cell_preflight

        preflight = preflight_cell_run(config)
        path = write_cell_preflight(preflight, config.output_dir / "preflight.json")
        print(
            f"{path}: {preflight.slide_count} slides, "
            f"fingerprint={preflight.fingerprint}"
        )
        return 0
    from histopia.cells._pipeline import run_cell_segmentation

    processing_sections = tuple(args.processing_section)
    print(
        run_cell_segmentation(
            config,
            processing_sections=processing_sections or None,
            progress=print,
        )
    )
    if processing_sections:
        print(
            "Worker checkpoint complete. Run the complete configuration without "
            "--processing-section to verify and seal the result."
        )
    else:
        print("Scientific review required: accept every section before run approval.")
    return 0


def _section_for_slide(run: Path, slide: str) -> str:
    payload = json.loads((run / "cell_result.json").read_text())
    matches = [
        str(row["section"])
        for row in payload.get("slides", [])
        if isinstance(row, dict)
        and isinstance(row.get("slide"), str)
        and row["slide"] == slide
    ]
    if len(matches) != 1:
        raise ValueError(
            f"cell result does not contain exactly one slide named {slide!r}"
        )
    return matches[0]


if __name__ == "__main__":
    sys.exit(main())
