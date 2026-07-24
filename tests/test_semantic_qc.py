from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from histopia.semantic._cli import main
from histopia.semantic._qc import summarize_semantic_run, write_cohort_qc
from histopia.semantic._result import _seal_semantic_result


def _write_run(root: Path, *, selected_k: int, patch_count: int) -> Path:
    if patch_count % 2:
        raise ValueError("fixture patch count must be even")
    labels = root / "labels" / f"k-{selected_k}"
    topology = root / "topology"
    labels.mkdir(parents=True)
    topology.mkdir()
    np.savez_compressed(root / "atlas_model.npz", pca_mean=np.zeros(2))
    per_slide = patch_count // 2
    slide_rows = []
    all_labels = np.arange(patch_count, dtype=np.int16) % selected_k
    for index in range(2):
        start = index * per_slide
        stop = start + per_slide
        path = labels / f"{index + 1:03d}.npz"
        np.savez_compressed(
            path,
            labels=all_labels[start:stop],
            joint_labels=all_labels[start:stop],
            tissue_fraction=np.full(per_slide, 0.8, dtype=np.float32),
            grid_rc=np.column_stack(
                [np.zeros(per_slide, dtype=np.int32), np.arange(per_slide)]
            ),
            reference_um_xy=np.column_stack(
                [np.arange(per_slide, dtype=float), np.full(per_slide, index)]
            ),
        )
        slide_rows.append(
            {
                "id": f"slide-{index + 1}.ndpi",
                "labels": {
                    str(selected_k): f"labels/k-{selected_k}/{index + 1:03d}.npz"
                },
            }
        )
    np.savez_compressed(
        topology / "001-002.npz",
        source_indices=np.array([0, 1], dtype=np.int64),
        target_indices=np.array([0, 1], dtype=np.int64),
        source_um_xy=np.array([[0, 0], [1, 0]], dtype=float),
        target_um_xy=np.array([[0, 1], [1, 1]], dtype=float),
        confidence=np.array([0.9, 0.7], dtype=np.float32),
    )
    core = {
        "schema_version": 3,
        "feature_provenance": {
            "preflight_fingerprint": "preflight",
            "expected_slide_ids": ["slide-1.ndpi", "slide-2.ndpi"],
            "model_fingerprint": "model",
            "analysis_mpp": 0.5,
            "patch_size_px": 224,
            "min_tissue_fraction": 0.5,
        },
        "model": "atlas_model.npz",
        "selected_k": selected_k,
        "cluster_counts": [selected_k],
        "slides": slide_rows,
        "topology_pairs": [
            {
                "source_section": 0,
                "target_section": 1,
                "accepted_links": 2,
                "artifact": "topology/001-002.npz",
            }
        ],
        "batch_correction": {
            "accepted": False,
            "unsupported_sections": [1],
            "raw": {
                "slide_variance_fraction": 0.3,
                "slide_prediction_accuracy": 0.8,
                "anchor_coverage": 0.5,
            },
            "corrected": {
                "slide_variance_fraction": 0.1,
                "slide_prediction_accuracy": 0.3,
                "anchor_coverage": 0.5,
            },
        },
        "k_selection": [
            {
                "k": selected_k,
                "stability_ari": 0.75,
                "composite_score": 0.9,
            }
        ],
    }
    payload = _seal_semantic_result(root, core)
    (root / "semantic_result.json").write_text(json.dumps(payload))
    (root / "semantic_review.json").write_text(
        json.dumps({"approved": False, "fingerprint": payload["fingerprint"]})
    )
    return root


def test_semantic_qc_summarizes_portable_result_artifacts(tmp_path: Path) -> None:
    run = _write_run(tmp_path / "mouse", selected_k=5, patch_count=10)

    qc = summarize_semantic_run(run)

    assert qc.selected_k == 5
    assert qc.slide_count == 2
    assert qc.patch_count == 10
    assert qc.accepted_topology_links == 2
    assert qc.median_topology_confidence == pytest.approx(0.8)
    assert qc.unsupported_sections == (1,)
    assert not qc.review_approved
    assert qc.selected_k_stability == 0.75
    assert qc.selected_k_score == 0.9
    assert qc.minimum_cluster_fraction == 0.2
    assert qc.zero_link_pairs == 0
    assert qc.topology_coverage == 0.4
    assert qc.raw_slide_variance_fraction == 0.3
    assert "batch_correction_rejected" in qc.flags
    assert "unsupported_sections" in qc.flags


def test_semantic_qc_rejects_changed_label_artifact(tmp_path: Path) -> None:
    run = _write_run(tmp_path / "mouse", selected_k=5, patch_count=10)
    np.savez_compressed(
        run / "labels/k-5/001.npz",
        labels=np.zeros(5, dtype=np.int16),
    )

    with pytest.raises(ValueError, match="artifact digest"):
        summarize_semantic_run(run)


def test_semantic_qc_rejects_missing_adjacent_pair(tmp_path: Path) -> None:
    run = _write_run(tmp_path / "mouse", selected_k=5, patch_count=10)
    payload = json.loads((run / "semantic_result.json").read_text())
    core = {
        key: value
        for key, value in payload.items()
        if key not in {"artifacts", "fingerprint"}
    }
    core["topology_pairs"] = []
    resealed = _seal_semantic_result(run, core)
    (run / "semantic_result.json").write_text(json.dumps(resealed))

    with pytest.raises(ValueError, match="adjacent"):
        summarize_semantic_run(run)


def test_semantic_qc_rejects_nonreciprocal_duplicate_topology_links(
    tmp_path: Path,
) -> None:
    run = _write_run(tmp_path / "mouse", selected_k=5, patch_count=10)
    topology = run / "topology/001-002.npz"
    np.savez_compressed(
        topology,
        source_indices=np.array([0, 0], dtype=np.int64),
        target_indices=np.array([0, 1], dtype=np.int64),
        source_um_xy=np.array([[0, 0], [0, 0]], dtype=float),
        target_um_xy=np.array([[0, 1], [1, 1]], dtype=float),
        confidence=np.array([0.9, 1.1], dtype=np.float32),
    )
    payload = json.loads((run / "semantic_result.json").read_text())
    core = {
        key: value
        for key, value in payload.items()
        if key not in {"artifacts", "fingerprint"}
    }
    resealed = _seal_semantic_result(run, core)
    (run / "semantic_result.json").write_text(json.dumps(resealed))

    with pytest.raises(ValueError, match="topology artifact is inconsistent"):
        summarize_semantic_run(run)


def test_semantic_qc_rejects_slide_list_that_differs_from_preflight(
    tmp_path: Path,
) -> None:
    run = _write_run(tmp_path / "mouse", selected_k=5, patch_count=10)
    payload = json.loads((run / "semantic_result.json").read_text())
    core = {
        key: value
        for key, value in payload.items()
        if key not in {"artifacts", "fingerprint"}
    }
    core["feature_provenance"]["expected_slide_ids"] = ["slide-1.ndpi"]
    resealed = _seal_semantic_result(run, core)
    (run / "semantic_result.json").write_text(json.dumps(resealed))

    with pytest.raises(ValueError, match="preflight slide order"):
        summarize_semantic_run(run)


def test_semantic_qc_rejects_partial_execution_provenance(tmp_path: Path) -> None:
    run = _write_run(tmp_path / "mouse", selected_k=5, patch_count=10)
    payload = json.loads((run / "semantic_result.json").read_text())
    core = {
        key: value
        for key, value in payload.items()
        if key not in {"artifacts", "fingerprint"}
    }
    core["feature_provenance"]["batch_size"] = 128
    resealed = _seal_semantic_result(run, core)
    (run / "semantic_result.json").write_text(json.dumps(resealed))

    with pytest.raises(ValueError, match="execution provenance is incomplete"):
        summarize_semantic_run(run)


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
    assert "thresholds" in payload
    assert (output.with_suffix(".tsv")).is_file()


def test_cohort_qc_flags_deterministic_relative_patch_outlier(tmp_path: Path) -> None:
    runs = {
        name: _write_run(tmp_path / name, selected_k=5, patch_count=count)
        for name, count in zip(
            ("a", "b", "c", "d", "outlier"),
            (100, 102, 98, 104, 1_000),
            strict=True,
        )
    }

    output = write_cohort_qc(runs, tmp_path / "cohort" / "qc.json")
    rows = {row["mouse_id"]: row for row in json.loads(output.read_text())["mice"]}

    assert "cohort_high_patch_count" in rows["outlier"]["flags"]
    assert all(
        "cohort_high_patch_count" not in rows[name]["flags"]
        for name in ("a", "b", "c", "d")
    )


def test_cohort_qc_flags_outlier_when_cohort_mad_is_zero(tmp_path: Path) -> None:
    runs = {
        name: _write_run(tmp_path / name, selected_k=5, patch_count=count)
        for name, count in zip(
            ("a", "b", "c", "d", "e", "f", "outlier"),
            (100, 100, 100, 100, 100, 100, 1_000),
            strict=True,
        )
    }

    output = write_cohort_qc(runs, tmp_path / "cohort" / "qc.json")
    rows = {row["mouse_id"]: row for row in json.loads(output.read_text())["mice"]}

    assert "cohort_high_patch_count" in rows["outlier"]["flags"]
    assert all(
        "cohort_high_patch_count" not in rows[name]["flags"]
        for name in ("a", "b", "c", "d", "e", "f")
    )


def test_cohort_qc_cli_writes_json_and_tsv(tmp_path: Path) -> None:
    first = _write_run(tmp_path / "first", selected_k=5, patch_count=10)
    output = tmp_path / "cohort" / "qc.json"

    status = main(
        [
            "cohort-qc",
            "--run",
            f"first={first}",
            "--output",
            str(output),
        ]
    )

    assert status == 0
    assert output.is_file()
    assert output.with_suffix(".tsv").is_file()
