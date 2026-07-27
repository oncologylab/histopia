from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from histopia.topology._feedback import TopologyFeedbackStore
from histopia.topology._model import ReconstructedPlane
from histopia.topology._pipeline import _write_meshes, _write_planes
from histopia.topology._result import write_topology_result
from histopia.visualization import build_topology_review


def test_topology_feedback_requires_every_current_pair(tmp_path: Path) -> None:
    run = _topology_run(tmp_path / "run")
    store = TopologyFeedbackStore(tmp_path / "feedback")
    review = store.review(cohort="sample", topology_run=run)

    assert len(review["slides"]) == 1
    with pytest.raises(ValueError, match="incomplete"):
        store.require_accepted(cohort="sample", topology_run=run)

    updated = store.save(
        {
            "cohort": "sample",
            "stage": "topology",
            "fingerprint": review["fingerprint"],
            "slide_id": "000-001",
            "decision": "accept",
            "labels": [],
            "comment": "",
            "reviewer": "Reviewer",
            "suggested_intervals": 1,
        },
        topology_run=run,
    )
    store.require_accepted(cohort="sample", topology_run=run)
    assert updated["summary"]["by_decision"] == {"accept": 1}


def test_topology_viewer_copies_only_declared_mesh_assets(tmp_path: Path) -> None:
    run = _topology_run(tmp_path / "run")
    index = build_topology_review({"sample": run}, tmp_path / "viewer")

    manifest = json.loads((index.parent / "manifest.json").read_text())
    cohort = manifest["cohorts"][0]
    assert cohort["id"] == "sample"
    assert cohort["meshes"][0]["viewer_asset"].startswith("assets/sample/")
    assert (index.parent / cohort["meshes"][0]["viewer_asset"]).is_file()
    assert (index.parent / "vendor" / "three.module.min.js").is_file()


def _topology_run(root: Path) -> Path:
    root.mkdir()
    labels = np.full((8, 9), -1, dtype=np.int16)
    labels[2:6, 2:7] = 0
    membership = np.stack((labels == 0, labels == 1)).astype(np.float32)
    planes = tuple(
        ReconstructedPlane(
            z_um=float(index * 5),
            segment=0,
            source_section=index,
            target_section=index,
            fraction=0,
            observed=True,
            slide_id=f"slide-{index}",
            membership=membership,
            labels=labels,
            support=labels >= 0,
            uncertainty=np.zeros(labels.shape, dtype=np.float32),
        )
        for index in range(2)
    )
    plane_rows = _write_planes(root, planes)
    mesh_rows, class_rows = _write_meshes(
        root,
        planes,
        class_count=2,
        palette=("#ff0000", "#00ff00"),
        origin_um_xy=(0, 0),
        spacing_um=10,
        thickness_um=5,
    )
    (root / "preflight.json").write_text("{}")
    (root / "benchmark.json").write_text(
        json.dumps(
            {
                "summary": {
                    "flow_macro_class_dice": 0.8,
                    "flow_gain_over_zero": 0.1,
                    "gap_interval_accuracy": 0.7,
                }
            }
        )
    )
    write_topology_result(
        root,
        {
            "schema_version": 1,
            "preflight": "preflight.json",
            "benchmark": "benchmark.json",
            "selected_k": 2,
            "palette": ["#ff0000", "#00ff00"],
            "z_source": "manifest_measured",
            "section_thickness_um": 5,
            "reference_grid": {
                "origin_um_xy": [0, 0],
                "shape_rc": [8, 9],
                "spacing_um": 10,
            },
            "observed_section_count": 2,
            "virtual_section_count": 0,
            "segment_count": 1,
            "gap_decisions": [
                {
                    "source_section": 0,
                    "target_section": 1,
                    "intervals": 1,
                    "missing_sections": 0,
                    "status": "manifest",
                    "confidence": 1,
                    "score": 0.1,
                    "reasons": [],
                }
            ],
            "pair_evidence": [{}],
            "planes": plane_rows,
            "meshes": mesh_rows,
            "classes": class_rows,
        },
    )
    return root
