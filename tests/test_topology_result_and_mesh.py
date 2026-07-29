from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from histopia.topology import (
    approve_topology_result,
    summarize_topology_run,
    validate_topology_approval,
    validate_topology_result,
)
from histopia.topology._model import ReconstructedPlane
from histopia.topology._pipeline import _write_meshes, _write_planes
from histopia.topology._result import write_topology_result


def test_topology_result_is_sealed_and_approval_is_fingerprint_bound(
    tmp_path: Path,
) -> None:
    planes = (_plane(0, observed=True), _plane(5, observed=True))
    plane_rows = _write_planes(tmp_path, planes)
    mesh_rows, class_rows = _write_meshes(
        tmp_path,
        planes,
        class_count=2,
        palette=("#ff0000", "#00ff00"),
        origin_um_xy=(0, 0),
        spacing_um=10,
        thickness_um=5,
    )
    preflight = tmp_path / "preflight.json"
    preflight.write_text(json.dumps(_approved_preflight()))
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(
        json.dumps(
            {
                "summary": {
                    "flow_gain_over_zero": 0.1,
                    "gap_interval_accuracy": 0.8,
                    "supports_flow_interpolation": True,
                }
            }
        )
    )
    result = write_topology_result(
        tmp_path,
        {
            "schema_version": 1,
            "preflight": "preflight.json",
            "registration_result_sha256": "registration-sha",
            "semantic_fingerprint": "semantic-fingerprint",
            "benchmark": "benchmark.json",
            "selected_k": 2,
            "observed_section_count": 2,
            "virtual_section_count": 0,
            "segment_count": 1,
            "gap_decisions": [],
            "planes": plane_rows,
            "meshes": mesh_rows,
            "classes": class_rows,
        },
    )

    payload = validate_topology_result(tmp_path)
    approval = approve_topology_result(
        tmp_path,
        reviewer="Reviewer",
        notes="Inspected surfaces and transitions.",
        reviewed_at="2026-07-27T20:00:00+00:00",
    )

    assert result == tmp_path / "topology_result.json"
    assert validate_topology_approval(tmp_path) == approval
    assert summarize_topology_run(tmp_path).mesh_faces > 0

    mesh_path = tmp_path / payload["meshes"][0]["artifact"]
    with np.load(mesh_path) as data:
        vertices = data["vertices_um_xyz"]
    np.savez_compressed(mesh_path, vertices_um_xyz=vertices + 1)
    with pytest.raises(ValueError, match="digest mismatch"):
        validate_topology_approval(tmp_path)


def test_stale_topology_review_is_reset(tmp_path: Path) -> None:
    preflight = tmp_path / "preflight.json"
    preflight.write_text(json.dumps(_approved_preflight()))
    (tmp_path / "topology_review.json").write_text(
        json.dumps({"schema_version": 1, "approved": True, "fingerprint": "old"})
    )

    write_topology_result(
        tmp_path,
        {
            "schema_version": 1,
            "preflight": "preflight.json",
            "registration_result_sha256": "registration-sha",
            "semantic_fingerprint": "semantic-fingerprint",
            "selected_k": 2,
            "observed_section_count": 2,
            "virtual_section_count": 0,
            "segment_count": 1,
            "gap_decisions": [],
            "planes": [],
            "meshes": [],
            "classes": [],
        },
    )

    review = json.loads((tmp_path / "topology_review.json").read_text())
    assert review["approved"] is False
    assert review["fingerprint"] != "old"


def test_topology_result_seals_optional_partition_surfaces(tmp_path: Path) -> None:
    (tmp_path / "preflight.json").write_text("{}")
    volume = tmp_path / "volume"
    volume.mkdir()
    np.savez_compressed(volume / "dense-fields.npz", field=np.zeros(1))
    meshes = tmp_path / "meshes"
    meshes.mkdir()
    np.savez_compressed(meshes / "partition.npz", vertices=np.zeros((3, 3)))
    (meshes / "partition.bin").write_bytes(b"HTM1")
    partition = {
        "artifact": "meshes/partition.npz",
        "viewer_asset": "meshes/partition.bin",
    }

    write_topology_result(
        tmp_path,
        {
            "schema_version": 2,
            "preflight": "preflight.json",
            "selected_k": 1,
            "observed_section_count": 2,
            "virtual_section_count": 0,
            "segment_count": 1,
            "gap_decisions": [],
            "planes": [],
            "envelope": None,
            "semantic_regions": [],
            "semantic_partition_regions": [partition],
            "uncertainty": None,
            "classes": [],
            "reconstruction_grid": {"artifact": "volume/dense-fields.npz"},
        },
    )

    result = validate_topology_result(tmp_path)
    assert "meshes/partition.npz" in result["artifacts"]
    assert "meshes/partition.bin" in result["artifacts"]


def test_topology_approval_rejects_unbound_legacy_inputs(tmp_path: Path) -> None:
    (tmp_path / "preflight.json").write_text('{"semantic_approval":null}')
    write_topology_result(
        tmp_path,
        {
            "schema_version": 1,
            "preflight": "preflight.json",
            "selected_k": 2,
            "observed_section_count": 2,
            "virtual_section_count": 0,
            "segment_count": 1,
            "gap_decisions": [],
            "planes": [],
            "meshes": [],
            "classes": [],
        },
    )

    with pytest.raises(ValueError, match="approval-bound"):
        approve_topology_result(
            tmp_path,
            reviewer="Reviewer",
            notes="Reviewed.",
        )


def test_topology_approval_rejects_malformed_input_snapshot(tmp_path: Path) -> None:
    preflight = _approved_preflight()
    preflight["semantic_approval"]["semantic_fingerprint"] = "stale"
    (tmp_path / "preflight.json").write_text(json.dumps(preflight))
    write_topology_result(
        tmp_path,
        {
            "schema_version": 1,
            "preflight": "preflight.json",
            "registration_result_sha256": "registration-sha",
            "semantic_fingerprint": "semantic-fingerprint",
            "selected_k": 2,
            "observed_section_count": 2,
            "virtual_section_count": 0,
            "segment_count": 1,
            "gap_decisions": [],
            "planes": [],
            "meshes": [],
            "classes": [],
        },
    )

    with pytest.raises(ValueError, match="input binding"):
        approve_topology_result(
            tmp_path,
            reviewer="Reviewer",
            notes="Reviewed.",
        )


def _approved_preflight() -> dict[str, object]:
    return {
        "registration_result_sha256": "registration-sha",
        "semantic_fingerprint": "semantic-fingerprint",
        "semantic_approval": {
            "semantic_fingerprint": "semantic-fingerprint",
            "semantic_reviewer": "Semantic Reviewer",
            "registration_result_sha256": "registration-sha",
        },
    }


def _plane(z: float, *, observed: bool) -> ReconstructedPlane:
    labels = np.full((8, 9), -1, dtype=np.int16)
    labels[2:6, 2:7] = 0
    membership = np.zeros((2, 8, 9), dtype=np.float32)
    membership[0] = labels == 0
    return ReconstructedPlane(
        z_um=z,
        segment=0,
        source_section=0,
        target_section=0 if observed else 1,
        fraction=0 if observed else 0.5,
        observed=observed,
        slide_id="slide" if observed else None,
        membership=membership,
        labels=labels,
        support=labels >= 0,
        uncertainty=np.zeros((8, 9), dtype=np.float32),
    )
