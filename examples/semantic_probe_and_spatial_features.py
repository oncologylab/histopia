"""Fit a portable semantic probe and export interpretable spatial features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from histopia.semantic import SemanticProbeConfig, fit_semantic_probe
from histopia.topology import extract_spatial_feature_spectrum


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("training_npz", type=Path)
    parser.add_argument("label_stack_npz", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    with np.load(args.training_npz, allow_pickle=False) as data:
        fit = fit_semantic_probe(
            data["train_features"],
            data["train_labels"],
            validation_features=data["validation_features"],
            validation_labels=data["validation_labels"],
            config=SemanticProbeConfig(pca_components=64),
        )
    with np.load(args.label_stack_npz, allow_pickle=False) as data:
        class_names = tuple(str(value) for value in data["class_names"])
        spectrum = extract_spatial_feature_spectrum(
            data["labels"],
            spacing_um_xy=tuple(float(value) for value in data["spacing_um_xy"]),
            z_positions_um=data["z_positions_um"],
            class_names=class_names,
        )

    args.output.mkdir(parents=True, exist_ok=True)
    fit.model.save(args.output / "semantic_probe.npz")
    (args.output / "spatial_features.json").write_text(
        json.dumps(spectrum.rows(), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
