from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from histopia.registration._qc_cache import (
    RegistrationQcCache,
    qc_artifact_fingerprint,
)


def test_qc_cache_validates_exact_bundle_bytes_atomically(tmp_path: Path) -> None:
    manifest = tmp_path / ".cache" / "qc.json"
    first = tmp_path / "qc" / "first.png"
    second = tmp_path / "qc" / "second.png"
    first.parent.mkdir()
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    fingerprint = qc_artifact_fingerprint(
        "test-v1",
        arrays=(np.arange(12, dtype=np.uint8).reshape(3, 4),),
        metadata={"method": "test"},
    )
    cache = RegistrationQcCache(tmp_path, manifest)

    assert not cache.is_current(fingerprint, (first, second))
    cache.record(fingerprint, (first, second))
    assert cache.is_current(fingerprint, (first, second))

    payload = json.loads(manifest.read_text())
    rows = payload["entries"]["qc/first.png"]["artifacts"]
    assert payload["schema"] == "histopia-registration-qc-artifacts-v1"
    assert [row["path"] for row in rows] == ["qc/first.png", "qc/second.png"]
    assert all(len(row["sha256"]) == 64 for row in rows)
    assert cache.misses == 1
    assert cache.hits == 1
    assert cache.rendered == 1
    assert not tuple(manifest.parent.glob(f".{manifest.name}.*.tmp"))

    second.write_bytes(b"changed")

    assert not cache.is_current(fingerprint, (first, second))


def test_qc_cache_rejects_corrupt_manifest_and_escaping_artifacts(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "qc.json"
    manifest.write_text("{invalid")
    artifact = tmp_path / "artifact.png"
    artifact.write_bytes(b"image")
    cache = RegistrationQcCache(tmp_path, manifest)

    assert not cache.is_current("fingerprint", (artifact,))
    with pytest.raises(ValueError, match="escapes"):
        cache.is_current("fingerprint", (tmp_path.parent / "outside.png",))


def test_qc_cache_rejects_symlink_replacement(tmp_path: Path) -> None:
    target = tmp_path / "target.png"
    artifact = tmp_path / "artifact.png"
    target.write_bytes(b"same")
    artifact.write_bytes(b"same")
    cache = RegistrationQcCache(tmp_path, tmp_path / "qc.json")
    cache.record("fingerprint", (artifact,))
    artifact.unlink()
    artifact.symlink_to(target)

    assert not cache.is_current("fingerprint", (artifact,))


def test_qc_fingerprint_binds_array_and_metadata() -> None:
    array = np.arange(12, dtype=np.uint8).reshape(3, 4)
    baseline = qc_artifact_fingerprint(
        "panel-v1",
        arrays=(array,),
        metadata={"slide": "one"},
    )
    changed = array.copy()
    changed[0, 0] += 1

    assert (
        qc_artifact_fingerprint(
            "panel-v1",
            arrays=(changed,),
            metadata={"slide": "one"},
        )
        != baseline
    )
    assert (
        qc_artifact_fingerprint(
            "panel-v1",
            arrays=(array,),
            metadata={"slide": "two"},
        )
        != baseline
    )
