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
    calls: list[tuple[dict[str, Path], Path, dict[str, Path]]] = []

    def capture(
        runs: dict[str, Path], output: Path, *, semantic_runs: dict[str, Path]
    ) -> Path:
        calls.append((runs, output, semantic_runs))
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
        ]
    )

    assert result == 0
    assert calls == [
        (
            {"mouse": registration},
            tmp_path / "viewer" / "histopia",
            {"mouse": semantic},
        )
    ]
