"""Checksum-validated cache for deterministic registration QC images."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np

from histopia._atomic import write_json_atomic

_SCHEMA = "histopia-registration-qc-artifacts-v1"


class RegistrationQcCache:
    """Validate and checkpoint rendered QC bundles under one run directory."""

    def __init__(self, run_dir: Path | str, manifest_path: Path | str) -> None:
        self.run_dir = Path(run_dir).resolve()
        self.manifest_path = Path(manifest_path)
        self._entries = _load_entries(self.manifest_path)
        self._lock = Lock()
        self.hits = 0
        self.misses = 0
        self.rendered = 0

    def is_current(
        self,
        fingerprint: str,
        artifact_paths: tuple[Path, ...],
    ) -> bool:
        """Return whether an exact bundle exists with matching output bytes."""

        relative_paths = self._relative_paths(artifact_paths)
        key = relative_paths[0]
        with self._lock:
            entry = self._entries.get(key)
        expected_rows = entry.get("artifacts") if isinstance(entry, dict) else None
        current = bool(
            isinstance(entry, dict)
            and entry.get("fingerprint") == fingerprint
            and isinstance(expected_rows, list)
            and [row.get("path") for row in expected_rows if isinstance(row, dict)]
            == relative_paths
            and len(expected_rows) == len(relative_paths)
            and all(
                _artifact_matches(self.run_dir, row)
                for row in expected_rows
                if isinstance(row, dict)
            )
        )
        with self._lock:
            if current:
                self.hits += 1
            else:
                self.misses += 1
        return current

    def record(
        self,
        fingerprint: str,
        artifact_paths: tuple[Path, ...],
    ) -> None:
        """Record exact output digests after one bundle has been rendered."""

        relative_paths = self._relative_paths(artifact_paths)
        rows: list[dict[str, object]] = []
        for relative, path in zip(relative_paths, artifact_paths, strict=True):
            if not path.is_file() or path.is_symlink():
                raise FileNotFoundError(f"registration QC artifact is missing: {path}")
            rows.append(
                {
                    "path": relative,
                    "size": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
        with self._lock:
            self._entries[relative_paths[0]] = {
                "fingerprint": fingerprint,
                "artifacts": rows,
            }
            self.rendered += 1
            try:
                write_json_atomic(
                    self.manifest_path,
                    {
                        "schema": _SCHEMA,
                        "entries": self._entries,
                    },
                    sort_keys=True,
                )
            except OSError:
                # Cache metadata must not determine scientific execution.
                pass

    def _relative_paths(self, paths: tuple[Path, ...]) -> list[str]:
        if not paths:
            raise ValueError("registration QC artifact bundle must not be empty")
        relative: list[str] = []
        for path in paths:
            resolved = path.resolve()
            if not resolved.is_relative_to(self.run_dir):
                raise ValueError("registration QC artifact escapes the run directory")
            relative.append(str(resolved.relative_to(self.run_dir)))
        if len(relative) != len(set(relative)):
            raise ValueError("registration QC artifact bundle contains duplicates")
        return relative


def qc_artifact_fingerprint(
    kind: str,
    *,
    arrays: tuple[np.ndarray, ...],
    metadata: dict[str, object],
) -> str:
    """Fingerprint all deterministic inputs to one rendered QC bundle."""

    if not kind:
        raise ValueError("registration QC artifact kind must not be blank")
    digest = hashlib.sha256(_SCHEMA.encode())
    digest.update(kind.encode())
    digest.update(
        json.dumps(
            metadata,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    )
    for values in arrays:
        array = np.asarray(values)
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _load_entries(path: Path) -> dict[str, dict[str, Any]]:
    try:
        payload = json.loads(path.read_text())
        entries = payload.get("entries")
        if payload.get("schema") != _SCHEMA or not isinstance(entries, dict):
            return {}
        return {
            str(key): value
            for key, value in entries.items()
            if isinstance(key, str) and isinstance(value, dict)
        }
    except (json.JSONDecodeError, OSError, TypeError):
        return {}


def _artifact_matches(run_dir: Path, row: dict[str, object]) -> bool:
    relative = row.get("path")
    expected_size = row.get("size")
    expected_sha256 = row.get("sha256")
    if (
        not isinstance(relative, str)
        or not relative
        or not isinstance(expected_size, int)
        or isinstance(expected_size, bool)
        or expected_size < 0
        or not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
    ):
        return False
    candidate = run_dir / relative
    path = candidate.resolve()
    if (
        not path.is_relative_to(run_dir)
        or not path.is_file()
        or candidate.is_symlink()
        or path.stat().st_size != expected_size
    ):
        return False
    return _sha256_file(path) == expected_sha256


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
