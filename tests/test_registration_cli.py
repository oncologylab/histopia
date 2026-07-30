from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from histopia.registration import _cli
from histopia.registration._errors import RegistrationApprovalRequired


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX SIGTERM semantics")
def test_sigterm_records_interrupted_registration_stage(tmp_path: Path) -> None:
    output = tmp_path / "output"
    config = tmp_path / "registration.json"
    config.write_text(
        json.dumps(
            {
                "input_dir": str(tmp_path / "input"),
                "output_dir": str(output),
            }
        )
    )
    child_code = """
import sys
import time
import histopia.registration._pipeline as pipeline

def hold(config, performance):
    performance.start_stage("slide_discovery")
    print("READY", flush=True)
    while True:
        time.sleep(1)

pipeline._register_sections = hold
from histopia.registration._cli import main
raise SystemExit(main(["--config", sys.argv[1]]))
"""
    environment = os.environ.copy()
    source = Path(__file__).parents[1] / "src"
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(source), environment.get("PYTHONPATH", "")) if value
    )
    process = subprocess.Popen(
        [sys.executable, "-c", child_code, str(config)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=environment,
    )
    try:
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "READY"
        process.send_signal(signal.SIGTERM)
        remaining_output, _ = process.communicate(timeout=5)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)

    assert process.returncode == 128 + signal.SIGTERM
    assert remaining_output == ""
    performance = json.loads((output / "registration_performance.json").read_text())
    assert performance["status"] == "interrupted"
    assert performance["failure_type"] == "SystemExit"
    assert performance["stages"]["slide_discovery"]["status"] == "interrupted"


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


def test_prepare_completed_review_reports_pending_order(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    order = tmp_path / "section_order_review.json"
    order.write_text(
        json.dumps(
            {
                "fingerprint": "prepared-fingerprint",
                "slides": [{"slide": "HE.ndpi"}],
            }
        )
    )
    monkeypatch.setattr(
        "histopia.registration._approval.prepare_completed_registration_review",
        lambda run: order,
    )

    result = _cli.main(["--prepare-completed-review", str(tmp_path)])

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "review_required"
    assert payload["stage"] == "order"
    assert payload["slide_count"] == 1


def test_registration_viewer_passes_worker_count(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    calls = []

    def capture(
        runs,
        output,
        *,
        provisional_mice,
        semantic_runs,
        workers,
    ):
        calls.append((runs, output, provisional_mice, semantic_runs, workers))
        return output / "index.html"

    monkeypatch.setattr(
        "histopia.visualization.build_section_viewer",
        capture,
    )
    run = tmp_path / "registration"
    semantic = tmp_path / "semantic"
    output = tmp_path / "viewer"

    result = _cli.main(
        [
            "--viewer-run",
            f"mouse={run}",
            "--viewer-semantic-run",
            f"mouse={semantic}",
            "--viewer-output-dir",
            str(output),
            "--viewer-workers",
            "3",
        ]
    )

    assert result == 0
    assert calls == [
        (
            {"mouse": run},
            output,
            set(),
            {"mouse": semantic},
            3,
        )
    ]
    assert capsys.readouterr().out.strip() == str(output / "index.html")


def test_saved_warp_passes_repeated_slide_selectors(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    calls = []

    def capture(run, output, **kwargs):
        calls.append((run, output, kwargs))
        return ()

    monkeypatch.setattr(
        "histopia.registration._wsi.warp_saved_registration",
        capture,
    )
    run = tmp_path / "registration"
    output = tmp_path / "registered"

    result = _cli.main(
        [
            "--warp-run",
            str(run),
            "--registered-output-dir",
            str(output),
            "--warp-slide",
            "section-001.ndpi",
            "--warp-slide",
            "section-003",
            "--vips-threads",
            "6",
        ]
    )

    assert result == 0
    assert calls == [
        (
            run,
            output,
            {
                "overwrite": False,
                "crop_mode": "reference",
                "accepted_non_rigid_only": False,
                "slide_names": ["section-001.ndpi", "section-003"],
                "vips_threads": 6,
            },
        )
    ]
    assert json.loads(capsys.readouterr().out) == []


def test_saved_warp_rejects_nonpositive_vips_threads() -> None:
    with pytest.raises(SystemExit):
        _cli.main(["--warp-run", "registration", "--vips-threads", "0"])
