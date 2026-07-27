from __future__ import annotations

import json
from pathlib import Path

import pytest

from histopia.visualization import _cli


def test_serve_command_dispatches_explicit_network_settings(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list[tuple[Path, str, int, tuple[str, ...], Path | None]] = []

    def capture(
        root: Path,
        *,
        bind: str,
        port: int,
        required_routes: tuple[str, ...],
        review_config: Path | None,
    ) -> None:
        calls.append((root, bind, port, required_routes, review_config))

    monkeypatch.setattr(_cli, "serve_viewer", capture)

    result = _cli.main(
        ["serve", str(tmp_path), "--bind", "127.0.0.1", "--port", "9876"]
    )

    assert result == 0
    assert calls == [(tmp_path, "127.0.0.1", 9876, ("histopia",), None)]


def test_feedback_export_writes_flat_dataset(tmp_path: Path) -> None:
    output = tmp_path / "feedback.json"

    result = _cli.main(["feedback-export", str(tmp_path / "empty"), str(output)])

    assert result == 0
    assert json.loads(output.read_text()) == {
        "schema_version": 1,
        "summary": {
            "schema_version": 1,
            "reviewed_slides": 0,
            "by_stage": {},
            "by_decision": {},
            "by_issue": {},
            "by_cohort": {},
        },
        "rows": [],
    }


def test_build_command_targets_stable_histopia_directory(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list[
        tuple[
            dict[str, Path],
            Path,
            dict[str, Path],
            dict[str, Path],
            Path | None,
            int,
            bool,
        ]
    ] = []

    def capture(
        runs: dict[str, Path],
        output: Path,
        *,
        semantic_runs: dict[str, Path],
        stain_runs: dict[str, Path],
        cohort_qc: Path | None,
        workers: int,
        require_approvals: bool,
    ) -> Path:
        calls.append(
            (
                runs,
                output,
                semantic_runs,
                stain_runs,
                cohort_qc,
                workers,
                require_approvals,
            )
        )
        return output / "index.html"

    monkeypatch.setattr(_cli, "build_section_viewer", capture)
    registration = tmp_path / "registration"
    semantic = tmp_path / "semantic"
    stain = tmp_path / "stain"

    result = _cli.main(
        [
            "build",
            str(tmp_path / "viewer"),
            "--run",
            f"mouse={registration}",
            "--semantic-run",
            f"mouse={semantic}",
            "--stain-run",
            f"mouse={stain}",
            "--cohort-qc",
            str(tmp_path / "cohort.json"),
            "--workers",
            "4",
        ]
    )

    assert result == 0
    assert calls == [
        (
            {"mouse": registration},
            tmp_path / "viewer" / "histopia",
            {"mouse": semantic},
            {"mouse": stain},
            tmp_path / "cohort.json",
            4,
            True,
        )
    ]


def test_mask_review_command_builds_requested_run(tmp_path: Path, monkeypatch) -> None:
    calls: list[tuple[Path, Path, int]] = []

    def capture(run: Path, output: Path, *, workers: int) -> Path:
        calls.append((run, output, workers))
        return output / "index.html"

    monkeypatch.setattr(
        "histopia.visualization._viewer.build_mask_review",
        capture,
    )
    run = tmp_path / "registration"
    output = tmp_path / "review"

    result = _cli.main(["mask-review", str(run), str(output), "--workers", "4"])

    assert result == 0
    assert calls == [(run, output, 4)]


def test_registration_review_command_passes_worker_count(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list[tuple[Path, Path, int]] = []

    def capture(run: Path, output: Path, *, workers: int) -> Path:
        calls.append((run, output, workers))
        return output / "index.html"

    monkeypatch.setattr(
        "histopia.visualization._review_portal.build_registration_review",
        capture,
    )
    run = tmp_path / "registration"
    output = tmp_path / "review"

    result = _cli.main(
        [
            "registration-review",
            str(run),
            str(output),
            "--workers",
            "4",
        ]
    )

    assert result == 0
    assert calls == [(run, output, 4)]


def test_non_rigid_review_command_passes_worker_count(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[tuple[Path, Path, int]] = []

    def capture(run: Path, output: Path, *, workers: int) -> Path:
        calls.append((run, output, workers))
        return output / "index.html"

    monkeypatch.setattr(
        "histopia.visualization._nonrigid_review.build_non_rigid_review",
        capture,
    )
    run = tmp_path / "validation"
    output = tmp_path / "review"

    result = _cli.main(["non-rigid-review", str(run), str(output), "--workers", "3"])

    assert result == 0
    assert calls == [(run, output, 3)]


def test_registration_cohort_review_command_passes_named_runs(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list[tuple[dict[str, Path], Path, int]] = []

    def capture(
        runs: dict[str, Path],
        output: Path,
        *,
        workers: int,
    ) -> Path:
        calls.append((runs, output, workers))
        return output / "index.html"

    monkeypatch.setattr(
        "histopia.visualization._review_portal.build_registration_cohort_review",
        capture,
    )
    output = tmp_path / "review"
    first = tmp_path / "run-4845"
    second = tmp_path / "run-8471"

    result = _cli.main(
        [
            "registration-cohort-review",
            str(output),
            "--run",
            f"4845={first}",
            "--run",
            f"8471={second}",
            "--workers",
            "4",
        ]
    )

    assert result == 0
    assert calls == [({"4845": first, "8471": second}, output, 4)]


def test_stain_review_command_passes_selected_mice_and_issues(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[
        tuple[Path, Path, list[str] | None, dict[str, dict[str, str]] | None]
    ] = []

    def capture(
        viewer: Path,
        output: Path,
        *,
        mice: list[str] | None,
        issues: dict[str, dict[str, str]] | None,
    ) -> Path:
        calls.append((viewer, output, mice, issues))
        return output / "index.html"

    monkeypatch.setattr(
        "histopia.visualization._stain_review.build_stain_review",
        capture,
    )
    viewer = tmp_path / "viewer" / "histopia"
    output = tmp_path / "viewer" / "stain-review"
    issues = tmp_path / "issues.json"
    issues.write_text(json.dumps({"4785": {"41": "Inspect upstream mask."}}))

    result = _cli.main(
        [
            "stain-review",
            str(viewer),
            str(output),
            "--mouse",
            "4785",
            "--mouse",
            "4269",
            "--issues",
            str(issues),
        ]
    )

    assert result == 0
    assert calls == [
        (
            viewer,
            output,
            ["4785", "4269"],
            {"4785": {"41": "Inspect upstream mask."}},
        )
    ]


def test_workflow_review_command_passes_all_named_runs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[tuple[object, ...]] = []

    def capture(
        runs,
        output,
        *,
        semantic_runs,
        stain_runs,
        topology_runs,
        cohort_qc,
        workers,
    ):
        calls.append(
            (
                runs,
                output,
                semantic_runs,
                stain_runs,
                topology_runs,
                cohort_qc,
                workers,
            )
        )
        return output / "index.html"

    monkeypatch.setattr(
        "histopia.visualization._review_portal.build_workflow_review",
        capture,
    )
    registration = tmp_path / "registration"
    semantic = tmp_path / "semantic"
    stain = tmp_path / "stain"
    output = tmp_path / "review"

    result = _cli.main(
        [
            "review",
            str(output),
            "--run",
            f"mouse={registration}",
            "--semantic-run",
            f"mouse={semantic}",
            "--stain-run",
            f"mouse={stain}",
            "--cohort-qc",
            str(tmp_path / "cohort.json"),
            "--workers",
            "4",
        ]
    )

    assert result == 0
    assert calls == [
        (
            {"mouse": registration},
            output,
            {"mouse": semantic},
            {"mouse": stain},
            {},
            tmp_path / "cohort.json",
            4,
        )
    ]


def test_order_review_command_passes_worker_count(tmp_path: Path, monkeypatch) -> None:
    calls: list[tuple[Path, Path, Path, int]] = []

    def capture(
        proposal: Path,
        processed: Path,
        output: Path,
        *,
        workers: int,
    ) -> Path:
        calls.append((proposal, processed, output, workers))
        return output / "index.html"

    monkeypatch.setattr(
        "histopia.visualization._viewer.build_section_order_review",
        capture,
    )
    proposal = tmp_path / "proposal.json"
    processed = tmp_path / "processed"
    output = tmp_path / "review"

    result = _cli.main(
        [
            "order-review",
            str(proposal),
            str(processed),
            str(output),
            "--workers",
            "4",
        ]
    )

    assert result == 0
    assert calls == [(proposal, processed, output, 4)]


def test_showcase_command_exports_selected_static_mice(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list[tuple[Path, Path, list[str]]] = []

    def capture(source: Path, output: Path, mice: list[str]) -> Path:
        calls.append((source, output, mice))
        return output / "index.html"

    monkeypatch.setattr(_cli, "export_static_showcase", capture)
    source = tmp_path / "viewer" / "histopia"
    output = tmp_path / "showcase"

    result = _cli.main(
        [
            "showcase",
            str(source),
            str(output),
            "--mouse",
            "5997",
            "--mouse",
            "4257",
        ]
    )

    assert result == 0
    assert calls == [(source, output, ["5997", "4257"])]


def test_qc_showcase_command_exports_selected_mice(tmp_path: Path, monkeypatch) -> None:
    calls: list[tuple[Path, Path, list[str]]] = []

    def capture(source: Path, output: Path, mice: list[str]) -> Path:
        calls.append((source, output, mice))
        return output / "index.html"

    monkeypatch.setattr(_cli, "export_registration_qc_showcase", capture)
    source = tmp_path / "viewer" / "histopia"
    output = tmp_path / "qc"

    result = _cli.main(
        [
            "qc-showcase",
            str(source),
            str(output),
            "--mouse",
            "4435",
            "--mouse",
            "4943",
        ]
    )

    assert result == 0
    assert calls == [(source, output, ["4435", "4943"])]


def test_audit_command_emits_portable_json_and_integrity_exit_code(
    tmp_path: Path,
    capsys,
) -> None:
    output = tmp_path / "audit.json"

    result = _cli.main(
        [
            "audit",
            "--run",
            f"mouse={tmp_path / 'missing-registration'}",
            "--output",
            str(output),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert result == 1
    assert payload == json.loads(output.read_text())
    assert payload["status"] == "incomplete"
    assert payload["cohorts"][0]["registration"]["issue"] == (
        "registration_result_missing"
    )
    assert str(tmp_path) not in json.dumps(payload)


def test_audit_command_rejects_duplicate_named_runs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duplicate registration run name"):
        _cli.main(
            [
                "audit",
                "--run",
                f"mouse={tmp_path / 'first'}",
                "--run",
                f"mouse={tmp_path / 'second'}",
            ]
        )
