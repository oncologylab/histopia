"""Run ``python -m histopia.study`` without extending existing workflow CLIs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from histopia.study._manifest import load_study_manifest


def main():
    parser = argparse.ArgumentParser(description="Versioned Histopia figure studies")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate", help="validate a frozen study manifest")
    validate.add_argument("manifest", type=Path)
    figure = sub.add_parser("figure", help="build an evidence-bound six-panel figure")
    figure.add_argument("manifest", type=Path)
    figure.add_argument("spec", type=Path)
    figure.add_argument("output", type=Path)
    args = parser.parse_args()
    study = load_study_manifest(args.manifest)
    if args.command == "validate":
        print(
            json.dumps(
                {
                    "study_id": study["study_id"],
                    "fingerprint": study["fingerprint"],
                    "mice": len(study["mice"]),
                    "slides": len(study["slides"]),
                }
            )
        )
    else:
        from histopia.study._figure import build_study_figure

        print(build_study_figure(study, json.loads(args.spec.read_text()), args.output))


if __name__ == "__main__":
    main()
