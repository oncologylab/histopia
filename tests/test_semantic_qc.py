from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from histopia.semantic._qc import summarize_semantic_run, write_cohort_qc


def _write_run(root: Path, *, selected_k: int, patch_count: int) -> Path:
    labels = root / "labels" / f"k-{selected_k}"
    topology = root / "topology"
    labels.mkdir(parents=True)
    topology.mkdir()
    np.savez_compressed(
        labels / "001.npz",
        labels=np.arange(patch_count, dtype=np.int16) % selected_k,
        tissue_fraction=np.full(patch_count, 0.8, dtype=np.float32),
    )
    np.savez_compressed(
        topology / "001-002.npz",
        confidence=np.array([0.9, 0.7], dtype=np.float32),
    )
    (root / "semantic_result.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "fingerprint": f"run-k{selected_k}",
                "selected_k": selected_k,
                "cluster_counts": list(range(5, 16)),
                "slides": [
                    {
                        "id": "slide.ndpi",
                        "labels": {str(selected_k): f"labels/k-{selected_k}/001.npz"},
                    }
                ],
                "topology_pairs": [
                    {"accepted_links": 2, "artifact": "topology/001-002.npz"}
                ],
                "batch_correction": {
                    "accepted": False,
                    "unsupported_sections": [1],
                },
            }
        )
    )
    (root / "semantic_review.json").write_text(
        json.dumps({"approved": False, "fingerprint": f"run-k{selected_k}"})
    )
    return root


def test_semantic_qc_summarizes_portable_result_artifacts(tmp_path: Path) -> None:
    run = _write_run(tmp_path / "mouse", selected_k=5, patch_count=10)

    qc = summarize_semantic_run(run)

    assert qc.selected_k == 5
    assert qc.slide_count == 1
    assert qc.patch_count == 10
    assert qc.accepted_topology_links == 2
    assert qc.median_topology_confidence == pytest.approx(0.8)
    assert qc.unsupported_sections == (1,)
    assert not qc.review_approved


def test_cohort_qc_writes_comparable_mouse_rows(tmp_path: Path) -> None:
    first = _write_run(tmp_path / "first", selected_k=5, patch_count=10)
    second = _write_run(tmp_path / "second", selected_k=7, patch_count=20)

    output = write_cohort_qc(
        {"a": first, "b": second}, tmp_path / "cohort" / "cohort_qc.json"
    )
    payload = json.loads(output.read_text())

    assert payload["schema_version"] == 1
    assert [row["mouse_id"] for row in payload["mice"]] == ["a", "b"]
    assert [row["selected_k"] for row in payload["mice"]] == [5, 7]
