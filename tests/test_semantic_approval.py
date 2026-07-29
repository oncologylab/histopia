from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import histopia.semantic._approval as approval_module
from histopia.semantic import (
    approve_semantic_result,
    validate_semantic_approval,
)
from histopia.semantic._result import _seal_semantic_result


def test_semantic_approval_is_fingerprint_bound_and_atomic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fingerprint = _write_semantic_result(tmp_path)
    registration = tmp_path / "registration"
    checked: list[tuple[Path, Path]] = []

    def validate_binding(registration_run, semantic_run, **kwargs):
        checked.append((Path(registration_run), Path(semantic_run)))
        return SimpleNamespace(approval_bound=True)

    monkeypatch.setattr(
        approval_module,
        "validate_semantic_registration_binding",
        validate_binding,
    )

    with pytest.raises(ValueError, match="not approved"):
        validate_semantic_approval(tmp_path)

    approval = approve_semantic_result(
        tmp_path,
        registration_run=registration,
        reviewer=" Test Reviewer ",
        notes=" Reviewed overlays and K sensitivity. ",
        reviewed_at="2026-07-24T18:30:00+00:00",
    )

    assert approval.fingerprint == fingerprint
    assert approval.reviewer == "Test Reviewer"
    assert approval.reviewed_at == "2026-07-24T18:30:00+00:00"
    assert checked == [(registration, tmp_path)]
    assert validate_semantic_approval(tmp_path) == approval
    review = json.loads((tmp_path / "semantic_review.json").read_text())
    assert review["approved"] is True
    assert review["notes"] == "Reviewed overlays and K sensitivity."


def test_semantic_approval_rejects_legacy_registration_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_semantic_result(tmp_path)
    monkeypatch.setattr(
        approval_module,
        "validate_semantic_registration_binding",
        lambda *args, **kwargs: SimpleNamespace(approval_bound=False),
    )

    with pytest.raises(ValueError, match="final registration approval"):
        approve_semantic_result(
            tmp_path,
            registration_run=tmp_path / "registration",
            reviewer="Reviewer",
            notes="Reviewed.",
        )

    review = json.loads((tmp_path / "semantic_review.json").read_text())
    assert review["approved"] is False


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

    review["fingerprint"] = json.loads((tmp_path / "semantic_result.json").read_text())[
        "fingerprint"
    ]
    review_path.write_text(json.dumps(review))
    np.savez_compressed(tmp_path / "atlas_model.npz", changed=np.ones(1))

    with pytest.raises(ValueError, match="artifact digest mismatch"):
        validate_semantic_approval(tmp_path)


def test_semantic_approval_cli_requires_registration_binding(tmp_path: Path) -> None:
    _write_semantic_result(tmp_path)

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
        check=False,
        capture_output=True,
        text=True,
    )

    review = json.loads((tmp_path / "semantic_review.json").read_text())
    assert completed.returncode == 2
    assert "--registration-run" in completed.stderr
    assert review["approved"] is False


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
