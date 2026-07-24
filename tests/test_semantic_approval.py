from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from histopia.semantic import (
    approve_semantic_result,
    validate_semantic_approval,
)
from histopia.semantic._result import _seal_semantic_result


def test_semantic_approval_is_fingerprint_bound_and_atomic(tmp_path: Path) -> None:
    fingerprint = _write_semantic_result(tmp_path)

    with pytest.raises(ValueError, match="not approved"):
        validate_semantic_approval(tmp_path)

    approval = approve_semantic_result(
        tmp_path,
        reviewer=" Test Reviewer ",
        notes=" Reviewed overlays and K sensitivity. ",
        reviewed_at="2026-07-24T18:30:00+00:00",
    )

    assert approval.fingerprint == fingerprint
    assert approval.reviewer == "Test Reviewer"
    assert approval.reviewed_at == "2026-07-24T18:30:00+00:00"
    assert validate_semantic_approval(tmp_path) == approval
    review = json.loads((tmp_path / "semantic_review.json").read_text())
    assert review["approved"] is True
    assert review["notes"] == "Reviewed overlays and K sensitivity."


def test_semantic_approval_rejects_stale_review_and_result_artifacts(
    tmp_path: Path,
) -> None:
    _write_semantic_result(tmp_path)
    review_path = tmp_path / "semantic_review.json"
    review = json.loads(review_path.read_text())
    review.update(
        {
            "approved": True,
            "fingerprint": "stale",
            "reviewer": "Reviewer",
            "notes": "Reviewed",
        }
    )
    review_path.write_text(json.dumps(review))

    with pytest.raises(ValueError, match="fingerprint is stale"):
        validate_semantic_approval(tmp_path)

    review["fingerprint"] = json.loads(
        (tmp_path / "semantic_result.json").read_text()
    )["fingerprint"]
    review_path.write_text(json.dumps(review))
    np.savez_compressed(tmp_path / "atlas_model.npz", changed=np.ones(1))

    with pytest.raises(ValueError, match="artifact digest mismatch"):
        validate_semantic_approval(tmp_path)


def test_semantic_approval_cli_writes_auditable_timestamp(tmp_path: Path) -> None:
    fingerprint = _write_semantic_result(tmp_path)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "histopia.semantic._cli",
            "approve",
            "--run",
            str(tmp_path),
            "--reviewer",
            "CLI Reviewer",
            "--review-notes",
            "Reviewed semantic and blend views.",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    review = json.loads((tmp_path / "semantic_review.json").read_text())
    assert review["fingerprint"] == fingerprint
    assert review["reviewed_at"].endswith("+00:00")
    assert f"fingerprint={fingerprint}" in completed.stdout


def _write_semantic_result(root: Path) -> str:
    np.savez_compressed(root / "atlas_model.npz", pca_mean=np.zeros(2))
    core = {
        "schema_version": 3,
        "primary_clusters": 2,
        "selected_k": 2,
        "cluster_counts": [2],
        "palette": ["#d73027", "#1a9850"],
        "model": "atlas_model.npz",
        "slides": [],
        "topology_pairs": [],
    }
    payload = _seal_semantic_result(root, core)
    (root / "semantic_result.json").write_text(json.dumps(payload))
    (root / "semantic_review.json").write_text(
        json.dumps(
            {
                "schema_version": 3,
                "approved": False,
                "fingerprint": payload["fingerprint"],
                "reviewer": None,
                "notes": "",
            }
        )
    )
    return str(payload["fingerprint"])
