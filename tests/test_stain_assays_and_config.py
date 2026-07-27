from __future__ import annotations

import json
from pathlib import Path

import pytest

from histopia.stain import StainFamily, StainQuantificationConfig
from histopia.stain._assays import (
    infer_slide_assay,
    load_assay_manifest,
    resolve_slide_assays,
)
from histopia.stain._config import load_stain_config


@pytest.mark.parametrize(
    ("name", "family", "marker"),
    [
        ("[#204] Yi_#5996_panc_Siriusred.ndpi", StainFamily.SIRIUS_RED, "Sirius Red"),
        ("Yi_#4257Panc_Sirusred-[236].scn", StainFamily.SIRIUS_RED, "Sirius Red"),
        ("Yi_#4257Panc_AlcianBlue-[247].scn", StainFamily.ALCIAN_BLUE, "Alcian Blue"),
        ("[#120] Yi_#4257_panc_PAS.ndpi", StainFamily.PAS, "PAS"),
        ("[#467] Yi_#5996_panc_HE.ndpi", StainFamily.CONTEXT_HE, "H&E"),
        ("[#463] Yi_#5996_panc_pERK.ndpi", StainFamily.H_DAB, "pERK"),
    ],
)
def test_kpf_assay_inference(
    name: str,
    family: StainFamily,
    marker: str,
) -> None:
    assay = infer_slide_assay(name, default_family=StainFamily.H_DAB)

    assert assay.family is family
    assert assay.marker == marker


def test_unknown_assay_requires_explicit_default() -> None:
    with pytest.raises(ValueError, match="ambiguous"):
        infer_slide_assay("unstructured-slide.ndpi", default_family=None)


def test_manifest_is_exact_and_rejects_unregistered_slides(tmp_path: Path) -> None:
    path = tmp_path / "assays.json"
    path.write_text(
        json.dumps(
            {
                "slides": {
                    "one.ndpi": {
                        "marker": "pERK",
                        "family": "h-dab",
                        "batch_id": "batch-a",
                    }
                }
            }
        )
    )
    manifest = load_assay_manifest(path)

    assert (
        resolve_slide_assays(
            ("one.ndpi",),
            manifest=manifest,
            default_family=None,
        )[0].batch_id
        == "batch-a"
    )
    with pytest.raises(ValueError, match="outside registration"):
        resolve_slide_assays(
            ("different.ndpi",),
            manifest=manifest,
            default_family=StainFamily.H_DAB,
        )


def test_config_loads_toml_and_validates_guards(tmp_path: Path) -> None:
    path = tmp_path / "stain.toml"
    path.write_text(
        """
registration_run = "registration"
output_dir = "stain"
analysis_mpp = 2.0
methods = ["fixed", "macenko"]
"""
    )

    config = load_stain_config(path)

    assert config.analysis_mpp == 2.0
    assert config.methods == ("fixed", "macenko")
    with pytest.raises(ValueError, match="between zero and one"):
        StainQuantificationConfig(
            registration_run=tmp_path,
            output_dir=tmp_path / "out",
            correction_rank_guard=1.1,
        )
    parallel = StainQuantificationConfig(
        registration_run=tmp_path,
        output_dir=tmp_path / "out",
        workers=2,
    )
    assert parallel.workers == 2
