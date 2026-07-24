from __future__ import annotations

from pathlib import Path

from histopia.visualization import _cli


def test_serve_command_dispatches_explicit_network_settings(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list[tuple[Path, str, int]] = []

    def capture(root: Path, *, bind: str, port: int) -> None:
        calls.append((root, bind, port))

    monkeypatch.setattr(_cli, "serve_viewer", capture)

    result = _cli.main(
        ["serve", str(tmp_path), "--bind", "127.0.0.1", "--port", "9876"]
    )

    assert result == 0
    assert calls == [(tmp_path, "127.0.0.1", 9876)]


def test_build_command_targets_stable_histopia_directory(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list[tuple[dict[str, Path], Path, dict[str, Path], Path | None]] = []

    def capture(
        runs: dict[str, Path],
        output: Path,
        *,
        semantic_runs: dict[str, Path],
        cohort_qc: Path | None,
    ) -> Path:
        calls.append((runs, output, semantic_runs, cohort_qc))
        return output / "index.html"

    monkeypatch.setattr(_cli, "build_section_viewer", capture)
    registration = tmp_path / "registration"
    semantic = tmp_path / "semantic"

    result = _cli.main(
        [
            "build",
            str(tmp_path / "viewer"),
            "--run",
            f"mouse={registration}",
            "--semantic-run",
            f"mouse={semantic}",
            "--cohort-qc",
            str(tmp_path / "cohort.json"),
        ]
    )

    assert result == 0
    assert calls == [
        (
            {"mouse": registration},
            tmp_path / "viewer" / "histopia",
            {"mouse": semantic},
            tmp_path / "cohort.json",
        )
    ]


def test_mask_review_command_builds_requested_run(tmp_path: Path, monkeypatch) -> None:
    calls: list[tuple[Path, Path]] = []

    def capture(run: Path, output: Path) -> Path:
        calls.append((run, output))
        return output / "index.html"

    monkeypatch.setattr(
        "histopia.visualization._viewer.build_mask_review",
        capture,
    )
    run = tmp_path / "registration"
    output = tmp_path / "review"

    result = _cli.main(["mask-review", str(run), str(output)])

    assert result == 0
    assert calls == [(run, output)]


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
