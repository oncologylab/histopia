from __future__ import annotations

import json
from pathlib import Path

import pytest

from histopia.registration import _cli
from histopia.registration._errors import RegistrationApprovalRequired


def test_staged_registration_reports_review_gate_as_success(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    config = tmp_path / "registration.json"
    config.write_text(
        json.dumps(
            {
                "input_dir": str(tmp_path / "input"),
                "output_dir": str(tmp_path / "output"),
            }
        )
    )

    def require_review(_config):
        raise RegistrationApprovalRequired(
            "masks",
            tmp_path / "output" / "mask_review.json",
            pending_slides=("HE.ndpi", "CK19.ndpi"),
        )

    monkeypatch.setattr(
        "histopia.registration._pipeline.register_sections", require_review
    )

    result = _cli.main(["--config", str(config), "--staged"])

    payload = json.loads(capsys.readouterr().out)
    assert result == 0
    assert payload["status"] == "review_required"
    assert payload["stage"] == "masks"
    assert payload["pending_slides"] == ["HE.ndpi", "CK19.ndpi"]


def test_nonstaged_registration_preserves_strict_failure(
    tmp_path: Path, monkeypatch
) -> None:
    config = tmp_path / "registration.json"
    config.write_text(
        json.dumps(
            {
                "input_dir": str(tmp_path / "input"),
                "output_dir": str(tmp_path / "output"),
            }
        )
    )

    def require_review(_config):
        raise RegistrationApprovalRequired(
            "order", tmp_path / "output" / "section_order_review.json"
        )

    monkeypatch.setattr(
        "histopia.registration._pipeline.register_sections", require_review
    )

    with pytest.raises(RegistrationApprovalRequired, match="current section order"):
        _cli.main(["--config", str(config)])
