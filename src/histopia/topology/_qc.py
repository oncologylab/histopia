"""Compact quality summaries for reconstructed topology runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from histopia.topology._result import validate_topology_result


@dataclass(frozen=True, slots=True)
class TopologyRunQc:
    """Comparable topology reconstruction measurements."""

    fingerprint: str
    selected_k: int
    observed_sections: int
    virtual_sections: int
    segments: int
    inferred_pairs: int
    unresolved_pairs: int
    mesh_faces: int
    estimated_volume_mm3: float
    inferred_volume_fraction: float
    flow_gain_over_zero: float
    gap_interval_accuracy: float
    flags: tuple[str, ...]


def summarize_topology_run(run_dir: Path | str) -> TopologyRunQc:
    """Summarize one fully integrity-checked topology result."""

    payload = validate_topology_result(run_dir)
    decisions = payload.get("gap_decisions", [])
    classes = payload.get("classes", [])
    meshes = (
        payload.get("meshes", [])
        if payload.get("schema_version") == 1
        else [
            payload.get("envelope"),
            *payload.get("semantic_regions", []),
        ]
    )
    benchmark = json.loads((Path(run_dir) / str(payload["benchmark"])).read_text())
    benchmark_summary = benchmark["summary"]
    total_volume = sum(float(row["estimated_volume_mm3"]) for row in classes)
    inferred_volume = sum(float(row.get("inferred_volume_mm3", 0.0)) for row in classes)
    unresolved = sum(row.get("status") == "unresolved" for row in decisions)
    flags: list[str] = []
    if unresolved:
        flags.append("unresolved_transitions")
    if not meshes:
        flags.append("no_surface_meshes")
    if payload.get("schema_version") == 2:
        reconstruction_qc = payload.get("reconstruction_qc", {})
        if (
            not isinstance(reconstruction_qc, dict)
            or reconstruction_qc.get("status") != "passed"
        ):
            flags.append("envelope_reconstruction_qc_failed")
    if payload.get("z_source") == "uniform_assumed_after_failed_gap_calibration":
        flags.append("uniform_z_assumed")
    if not benchmark_summary.get("supports_flow_interpolation"):
        flags.append("flow_interpolation_not_supported")
    return TopologyRunQc(
        fingerprint=str(payload["fingerprint"]),
        selected_k=int(payload["selected_k"]),
        observed_sections=int(payload["observed_section_count"]),
        virtual_sections=int(payload["virtual_section_count"]),
        segments=int(payload["segment_count"]),
        inferred_pairs=sum(row.get("status") == "inferred" for row in decisions),
        unresolved_pairs=unresolved,
        mesh_faces=sum(
            int(row["face_count"]) for row in meshes if isinstance(row, dict)
        ),
        estimated_volume_mm3=total_volume,
        inferred_volume_fraction=(
            inferred_volume / total_volume if total_volume > 0 else 0.0
        ),
        flow_gain_over_zero=float(benchmark_summary["flow_gain_over_zero"]),
        gap_interval_accuracy=float(benchmark_summary["gap_interval_accuracy"]),
        flags=tuple(flags),
    )
