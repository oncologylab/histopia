from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from histopia.visualization._review_api import (
    ReviewDecisionService,
    _semantic_status,
    _stain_status,
    _topology_status,
    _validate_topology_for_approval,
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
        monkeypatch.setattr(
            "histopia.visualization._review_api._stain_status",
            lambda *args: {
                "available": True,
                "approved": False,
                "approval_ready": True,
            },
        )
        monkeypatch.setattr(
            "histopia.visualization._review_api._validate_stain_for_approval",
            lambda *args: None,
        )
    if stage == "topology":
        monkeypatch.setattr(
            "histopia.visualization._review_api._validate_topology_for_approval",
            lambda *args: None,
        )

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
) -> None:
    run = tmp_path / "topology"
    run.mkdir()
    preflight = _write_fingerprinted_json(
        run / "preflight.json",
        {
            "semantic_fingerprint": "semantic",
            "registration_result_sha256": "registration",
            "semantic_approval": None,
        },
    )
    _write_fingerprinted_json(
        run / "topology_result.json",
        {
            "preflight": "preflight.json",
            "preflight_fingerprint": preflight["fingerprint"],
            "semantic_fingerprint": "semantic",
            "registration_result_sha256": "registration",
            "artifacts": {},
        },
    )

    assert _topology_status(tmp_path / "registration", None, run) == {
        "available": True,
        "approved": False,
        "approval_ready": False,
        "issue": "approval_bound_rebuild_required",
    }


def test_semantic_status_rejects_legacy_approval_as_unbound(
    tmp_path: Path,
) -> None:
    registration = tmp_path / "registration"
    semantic = tmp_path / "semantic"
    registration.mkdir()
    semantic.mkdir()
    registration_bytes = json.dumps({"slides": [{}, {}]}).encode()
    (registration / "registration_result.json").write_bytes(registration_bytes)
    (semantic / "preflight.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "fingerprint": "preflight",
                "registration_result_sha256": hashlib.sha256(
                    registration_bytes
                ).hexdigest(),
            }
        )
    )
    _write_fingerprinted_json(
        semantic / "semantic_result.json",
        {
            "feature_provenance": {"preflight_fingerprint": "preflight"},
            "artifacts": {},
        },
    )

    assert _semantic_status(registration, semantic) == {
        "available": True,
        "approved": False,
        "approval_ready": False,
        "issue": "semantic_registration_approval_binding_required",
    }


def test_semantic_status_distinguishes_pending_and_approved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registration = tmp_path / "registration"
    semantic = tmp_path / "semantic"
    registration.mkdir()
    semantic.mkdir()
    registration_bytes = json.dumps({"slides": [{}, {}]}).encode()
    (registration / "registration_result.json").write_bytes(registration_bytes)
    approval = registration / "registration_approval.json"
    approval.write_text("{}")
    (semantic / "preflight.json").write_text(
        json.dumps(
            {
                "schema_version": 3,
                "fingerprint": "preflight",
                "registration_result_sha256": hashlib.sha256(
                    registration_bytes
                ).hexdigest(),
                "registration_approval_sha256": hashlib.sha256(
                    approval.read_bytes()
                ).hexdigest(),
            }
        )
    )
    result = _write_fingerprinted_json(
        semantic / "semantic_result.json",
        {
            "feature_provenance": {"preflight_fingerprint": "preflight"},
            "artifacts": {},
        },
    )
    (semantic / "semantic_review.json").write_text(
        json.dumps(
            {
                "schema_version": 3,
                "fingerprint": result["fingerprint"],
                "approved": False,
            }
        )
    )
    monkeypatch.setattr(
        "histopia.visualization._review_api._registration_status",
        lambda _run: {"available": True, "approved": True},
    )

    assert _semantic_status(registration, semantic) == {
        "available": True,
        "approved": False,
        "approval_ready": True,
        "issue": "semantic_approval_required",
    }

    (semantic / "semantic_review.json").write_text(
        json.dumps(
            {
                "schema_version": 3,
                "fingerprint": result["fingerprint"],
                "approved": True,
                "reviewer": "Reviewer",
                "notes": "Reviewed.",
            }
        )
    )
    assert _semantic_status(registration, semantic) == {
        "available": True,
        "approved": True,
        "approval_ready": False,
        "issue": None,
    }


def test_stain_status_validates_registration_binding_and_family_review(
    tmp_path: Path,
) -> None:
    registration = tmp_path / "registration"
    stain = tmp_path / "stain"
    registration.mkdir()
    stain.mkdir()
    registration_bytes = json.dumps({"slides": [{}, {}]}).encode()
    (registration / "registration_result.json").write_bytes(registration_bytes)
    result = _write_fingerprinted_json(
        stain / "stain_result.json",
        {
            "schema_version": 1,
            "slides": [{}, {}],
            "registration_result_sha256": hashlib.sha256(
                registration_bytes
            ).hexdigest(),
            "families": {"h-dab": {}, "sirius-red": {}},
            "artifacts": {},
        },
    )
    (stain / "stain_review.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "fingerprint": result["fingerprint"],
                "families": {
                    "h-dab": {
                        "approved": True,
                        "reviewer": "Reviewer",
                        "reviewed_at": "2026-07-29T12:00:00+00:00",
                        "notes": "Reviewed.",
                    },
                    "sirius-red": {"approved": False},
                },
            }
        )
    )

    assert _stain_status(registration, stain) == {
        "available": True,
        "approved": False,
        "approval_ready": True,
        "issue": "stain_approval_required",
        "families": [
            {"id": "h-dab", "approved": True},
            {"id": "sirius-red", "approved": False},
        ],
    }

    (stain / "stain_review.json").unlink()
    assert _stain_status(registration, stain) == {
        "available": True,
        "approved": False,
        "approval_ready": True,
        "issue": "stain_approval_required",
        "families": [
            {"id": "h-dab", "approved": False},
            {"id": "sirius-red", "approved": False},
        ],
    }

    result["registration_result_sha256"] = "stale"
    core = {key: value for key, value in result.items() if key != "fingerprint"}
    result["fingerprint"] = _json_fingerprint(core)
    (stain / "stain_result.json").write_text(json.dumps(result))
    assert _stain_status(registration, stain) == {
        "available": True,
        "approved": False,
        "approval_ready": False,
        "families": [],
        "invalid": True,
        "issue": "stain_result_binding_or_approval_invalid",
    }


def test_topology_status_distinguishes_pending_and_approved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registration = tmp_path / "registration"
    semantic = tmp_path / "semantic"
    run = tmp_path / "topology"
    registration.mkdir()
    semantic.mkdir()
    run.mkdir()
    registration_bytes = b'{"slides":[{},{}]}'
    semantic_bytes = b'{"fingerprint":"semantic"}'
    (registration / "registration_result.json").write_bytes(registration_bytes)
    (semantic / "semantic_result.json").write_bytes(semantic_bytes)
    (semantic / "semantic_review.json").write_text(json.dumps({"reviewer": "Reviewer"}))
    monkeypatch.setattr(
        "histopia.visualization._review_api._semantic_status",
        lambda *_args: {"available": True, "approved": True},
    )
    preflight = _write_fingerprinted_json(
        run / "preflight.json",
        {
            "semantic_fingerprint": "semantic",
            "semantic_result_sha256": hashlib.sha256(semantic_bytes).hexdigest(),
            "registration_result_sha256": hashlib.sha256(
                registration_bytes
            ).hexdigest(),
            "semantic_approval": {
                "semantic_fingerprint": "semantic",
                "semantic_reviewer": "Reviewer",
                "registration_result_sha256": hashlib.sha256(
                    registration_bytes
                ).hexdigest(),
            },
        },
    )
    result = _write_fingerprinted_json(
        run / "topology_result.json",
        {
            "preflight": "preflight.json",
            "preflight_fingerprint": preflight["fingerprint"],
            "semantic_fingerprint": "semantic",
            "registration_result_sha256": hashlib.sha256(
                registration_bytes
            ).hexdigest(),
            "artifacts": {},
        },
    )
    (run / "topology_review.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "fingerprint": result["fingerprint"],
                "approved": False,
            }
        )
    )

    assert _topology_status(registration, semantic, run) == {
        "available": True,
        "approved": False,
        "approval_ready": True,
        "issue": "topology_approval_required",
    }

    (run / "topology_review.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "fingerprint": result["fingerprint"],
                "approved": True,
                "reviewer": "Reviewer",
                "notes": "Reviewed.",
            }
        )
    )
    assert _topology_status(registration, semantic, run) == {
        "available": True,
        "approved": True,
        "approval_ready": False,
        "issue": None,
    }


def test_topology_status_rejects_stale_current_semantic_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registration = tmp_path / "registration"
    semantic = tmp_path / "semantic"
    topology = tmp_path / "topology"
    for path in (registration, semantic, topology):
        path.mkdir()
    registration_bytes = b'{"slides":[{},{}]}'
    semantic_bytes = b'{"fingerprint":"semantic"}'
    (registration / "registration_result.json").write_bytes(registration_bytes)
    (semantic / "semantic_result.json").write_bytes(semantic_bytes)
    (semantic / "semantic_review.json").write_text(
        json.dumps({"reviewer": "Current reviewer"})
    )
    monkeypatch.setattr(
        "histopia.visualization._review_api._semantic_status",
        lambda *_args: {"available": True, "approved": True},
    )
    preflight = _write_fingerprinted_json(
        topology / "preflight.json",
        {
            "semantic_fingerprint": "semantic",
            "semantic_result_sha256": hashlib.sha256(semantic_bytes).hexdigest(),
            "registration_result_sha256": hashlib.sha256(
                registration_bytes
            ).hexdigest(),
            "semantic_approval": {
                "semantic_fingerprint": "semantic",
                "semantic_reviewer": "Stale reviewer",
                "registration_result_sha256": hashlib.sha256(
                    registration_bytes
                ).hexdigest(),
            },
        },
    )
    _write_fingerprinted_json(
        topology / "topology_result.json",
        {
            "preflight": "preflight.json",
            "preflight_fingerprint": preflight["fingerprint"],
            "semantic_fingerprint": "semantic",
            "registration_result_sha256": hashlib.sha256(
                registration_bytes
            ).hexdigest(),
            "artifacts": {},
        },
    )

    assert _topology_status(registration, semantic, topology) == {
        "available": True,
        "approved": False,
        "approval_ready": False,
        "invalid": True,
        "issue": "topology_result_or_approval_invalid",
    }


@pytest.mark.parametrize(
    ("registration_status", "semantic_status", "topology_status", "issue", "match"),
    (
        ("review_required", "approved", "review_required", None, "registration"),
        ("approved", "review_required", "review_required", None, "semantic"),
        (
            "approved",
            "approved",
            "review_required",
            "topology_approval_bound_rebuild_required",
            "approval_bound_rebuild_required",
        ),
    ),
)
def test_topology_full_approval_gate_rejects_noncurrent_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    registration_status: str,
    semantic_status: str,
    topology_status: str,
    issue: str | None,
    match: str,
) -> None:
    from types import SimpleNamespace

    cohort = SimpleNamespace(
        registration=SimpleNamespace(status=registration_status),
        semantic=SimpleNamespace(status=semantic_status),
        topology=SimpleNamespace(status=topology_status, issue=issue),
    )
    monkeypatch.setattr(
        "histopia.visualization._audit.audit_workflows",
        lambda *args, **kwargs: SimpleNamespace(cohorts=(cohort,)),
    )

    with pytest.raises(ValueError, match=match):
        _validate_topology_for_approval(
            tmp_path / "registration",
            tmp_path / "semantic",
            tmp_path / "topology",
        )


def test_review_service_rejects_unconfigured_or_incomplete_decisions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
    monkeypatch.setattr(
        "histopia.visualization._review_api._stain_status",
        lambda *args: {
            "available": True,
            "approved": False,
            "approval_ready": True,
        },
    )
    monkeypatch.setattr(
        "histopia.visualization._review_api._validate_stain_for_approval",
        lambda *args: None,
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


def _write_fingerprinted_json(
    path: Path,
    core: dict[str, object],
) -> dict[str, object]:
    payload = {**core, "fingerprint": _json_fingerprint(core)}
    path.write_text(json.dumps(payload))
    return payload


def _json_fingerprint(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
