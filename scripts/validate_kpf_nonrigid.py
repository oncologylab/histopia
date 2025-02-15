"""Audit saved KPF non-rigid acceptance decisions and flow artifacts."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_MICE = ("4257", "4577", "4630", "5997")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--non-rigid-root",
        type=Path,
        default=Path("/tmp/histopia-nonrigid-runs"),
    )
    parser.add_argument("--mice", nargs="+", default=list(DEFAULT_MICE))
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def audit_mouse(mouse: str, root: Path) -> tuple[dict[str, Any], list[str]]:
    run_dir = root / mouse
    payload = json.loads((run_dir / "registration_result.json").read_text())
    slides = [slide for slide in payload["slides"] if not slide["is_reference"]]
    accepted = [slide for slide in slides if slide["non_rigid_transform"]["accepted"]]
    rejected = [
        slide for slide in slides if not slide["non_rigid_transform"]["accepted"]
    ]
    failures: list[str] = []
    for slide in accepted:
        transform = slide["non_rigid_transform"]
        name = Path(slide["path"]).name
        shape = transform["displacement_shape"]
        maximum_dimension = max(shape[:2])
        if transform["final_similarity"] < transform["initial_similarity"] + 0.01:
            failures.append(f"{mouse} {name}: insufficient similarity gain")
        if transform["final_mask_dice"] < transform["initial_mask_dice"] - 0.01:
            failures.append(f"{mouse} {name}: mask Dice regression")
        if transform["jacobian_p01"] < 0.25 or transform["jacobian_p99"] > 4.0:
            failures.append(f"{mouse} {name}: Jacobian limit violation")
        if transform["displacement_p95"] > maximum_dimension * 0.03:
            failures.append(f"{mouse} {name}: displacement limit violation")
        if transform["inverse_consistency_p95"] > maximum_dimension * 0.02:
            failures.append(f"{mouse} {name}: inverse consistency violation")
        displacement_path = Path(transform["displacement_path"])
        if not displacement_path.is_absolute():
            displacement_path = run_dir / displacement_path
        if not displacement_path.exists():
            failures.append(f"{mouse} {name}: accepted flow file is missing")
    for slide in rejected:
        transform = slide["non_rigid_transform"]
        if transform["displacement_path"] is not None:
            failures.append(
                f"{mouse} {Path(slide['path']).name}: rejected flow was persisted"
            )

    transforms = [slide["non_rigid_transform"] for slide in accepted]
    reasons = collections.Counter(
        warning
        for slide in rejected
        for warning in slide["non_rigid_transform"]["warnings"]
    )
    report = {
        "mouse": mouse,
        "non_reference_slides": len(slides),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "median_similarity_gain": _median(
            item["final_similarity"] - item["initial_similarity"] for item in transforms
        ),
        "median_mask_dice_change": _median(
            item["final_mask_dice"] - item["initial_mask_dice"] for item in transforms
        ),
        "median_inverse_consistency_p95": _median(
            item["inverse_consistency_p95"] for item in transforms
        ),
        "rejection_reasons": dict(reasons),
    }
    return report, failures


def main() -> int:
    args = parse_args()
    reports: list[dict[str, Any]] = []
    failures: list[str] = []
    for mouse in args.mice:
        report, mouse_failures = audit_mouse(mouse, args.non_rigid_root)
        reports.append(report)
        failures.extend(mouse_failures)
    result = {"reports": reports, "failures": failures}
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    return 1 if failures else 0


def _median(values: Any) -> float:
    collected = list(values)
    return float(np.median(collected)) if collected else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
