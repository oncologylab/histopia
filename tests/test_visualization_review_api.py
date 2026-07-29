from __future__ import annotations

import json
from pathlib import Path

import pytest

from histopia.visualization._review_api import (
    ReviewDecisionService,
    _topology_status,
)


def _service(tmp_path: Path) -> tuple[ReviewDecisionService, dict[str, Path]]:
    paths = {
        name: tmp_path / name
        for name in ("registration", "semantic", "topology", "stain")
    }
    for path in paths.values():
        path.mkdir()
    config = tmp_path / "registry.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cohorts": {"mouse": {name: str(path) for name, path in paths.items()}},
            }
        )
    )
    return ReviewDecisionService.from_file(config), paths


@pytest.mark.parametrize(
    ("stage", "module", "function"),
    (
        ("mask", "histopia.registration", "approve_mask_review"),
        ("order", "histopia.registration", "approve_section_order"),
        ("registration", "histopia.registration", "approve_registration_run"),
        ("semantic", "histopia.semantic", "approve_semantic_result"),
        ("topology", "histopia.topology", "approve_topology_result"),
        ("stain", "histopia.stain", "approve_stain_result"),
    ),
)
def test_review_service_dispatches_existing_approval_functions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    module: str,
    function: str,
) -> None:
    service, paths = _service(tmp_path)
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def approve(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(f"{module}.{function}", approve)
    monkeypatch.setattr(
        service,
        "_cohort_status",
        lambda cohort, runs: {"id": cohort},
    )
    request: dict[str, object] = {
        "cohort": "mouse",
        "stage": stage,
        "reviewer": "Reviewer",
        "notes": "Evidence checked.",
    }
    if stage == "stain":
        request["families"] = ["h-dab"]

    result = service.approve(request)

    assert result == {"id": "mouse"}
    assert len(calls) == 1
    args, kwargs = calls[0]
    expected_run = (
        paths[stage]
        if stage in {"semantic", "topology", "stain"}
        else paths["registration"]
    )
    assert args == (expected_run,)
    assert kwargs["reviewer"] == "Reviewer"
    assert kwargs["notes"] == "Evidence checked."
    if stage == "semantic":
        assert kwargs["registration_run"] == paths["registration"]
    if stage == "stain":
        assert kwargs["families"] == ["h-dab"]


def test_topology_status_explains_approval_bound_rebuild(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = tmp_path / "topology"
    run.mkdir()
    (run / "topology_result.json").write_text("{}")
    (run / "preflight.json").write_text(
        json.dumps(
            {
                "semantic_fingerprint": "semantic",
                "registration_result_sha256": "registration",
                "semantic_approval": None,
            }
        )
    )
    monkeypatch.setattr(
        "histopia.topology.validate_topology_result",
        lambda _run: {"preflight": "preflight.json"},
    )

    assert _topology_status(run) == {
        "available": True,
        "approved": False,
        "approval_ready": False,
        "issue": "approval_bound_rebuild_required",
    }


def test_topology_status_distinguishes_pending_and_approved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = tmp_path / "topology"
    run.mkdir()
    (run / "topology_result.json").write_text("{}")
    (run / "topology_review.json").write_text(
        json.dumps({"schema_version": 1, "approved": False})
    )
    (run / "preflight.json").write_text(
        json.dumps(
            {
                "semantic_fingerprint": "semantic",
                "registration_result_sha256": "registration",
                "semantic_approval": {
                    "semantic_fingerprint": "semantic",
                    "semantic_reviewer": "Reviewer",
                    "registration_result_sha256": "registration",
                },
            }
        )
    )
    monkeypatch.setattr(
        "histopia.topology.validate_topology_result",
        lambda _run: {"preflight": "preflight.json"},
    )
    monkeypatch.setattr(
        "histopia.topology.validate_topology_approval",
        lambda _run: (_ for _ in ()).throw(ValueError("not approved")),
    )

    assert _topology_status(run) == {
        "available": True,
        "approved": False,
        "approval_ready": True,
        "issue": "topology_approval_required",
    }

    monkeypatch.setattr(
        "histopia.topology.validate_topology_approval",
        lambda _run: object(),
    )
    assert _topology_status(run) == {
        "available": True,
        "approved": True,
        "approval_ready": False,
        "issue": None,
    }


def test_review_service_rejects_unconfigured_or_incomplete_decisions(
    tmp_path: Path,
) -> None:
    service, _ = _service(tmp_path)

    with pytest.raises(ValueError, match="unknown review cohort"):
        service.approve(
            {
                "cohort": "missing",
                "stage": "mask",
                "reviewer": "Reviewer",
                "notes": "Checked.",
            }
        )
    with pytest.raises(ValueError, match="at least one family"):
        service.approve(
            {
                "cohort": "mouse",
                "stage": "stain",
                "reviewer": "Reviewer",
                "notes": "Checked.",
                "families": [],
            }
        )
