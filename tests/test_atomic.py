from __future__ import annotations

import json
from pathlib import Path

import pytest

from histopia import _atomic


def test_atomic_json_replace_preserves_previous_mode(tmp_path: Path) -> None:
    target = tmp_path / "result.json"
    target.write_text('{"old": true}\n')
    target.chmod(0o640)

    returned = _atomic.write_json_atomic(target, {"new": [1, 2]})

    assert returned == target
    assert json.loads(target.read_text()) == {"new": [1, 2]}
    assert target.stat().st_mode & 0o777 == 0o640
    assert not tuple(tmp_path.glob(".result.json.*.tmp"))


def test_atomic_json_if_changed_preserves_identical_file(tmp_path: Path) -> None:
    target = tmp_path / "result.json"
    target.write_text('{\n  "value": true\n}\n')
    original_stat = target.stat()

    returned = _atomic.write_json_atomic_if_changed(target, {"value": True})

    assert returned == target
    assert target.stat().st_mtime_ns == original_stat.st_mtime_ns
    assert target.stat().st_ino == original_stat.st_ino
    assert not tuple(tmp_path.glob(".result.json.*.tmp"))


def test_atomic_json_if_changed_replaces_different_file(tmp_path: Path) -> None:
    target = tmp_path / "result.json"
    target.write_text('{"value": false}\n')

    returned = _atomic.write_json_atomic_if_changed(target, {"value": True})

    assert returned == target
    assert json.loads(target.read_text()) == {"value": True}
    assert not tuple(tmp_path.glob(".result.json.*.tmp"))


def test_atomic_binary_writer_keeps_old_file_after_interruption(
    tmp_path: Path,
) -> None:
    target = tmp_path / "artifact.npz"
    target.write_bytes(b"previous")

    def interrupted(stream) -> None:
        stream.write(b"partial")
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        _atomic.write_binary_atomic(target, interrupted)

    assert target.read_bytes() == b"previous"
    assert not tuple(tmp_path.glob(".artifact.npz.*.tmp"))


def test_atomic_replace_failure_keeps_old_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "result.json"
    target.write_text('{"old": true}\n')

    def fail_replace(_source, _target) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(_atomic.os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        _atomic.write_json_atomic(target, {"new": True})

    assert json.loads(target.read_text()) == {"old": True}
    assert not tuple(tmp_path.glob(".result.json.*.tmp"))
