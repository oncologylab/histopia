"""Checksum-validated cache for deterministic registration QC images."""

from __future__ import annotations

import hashlib
import json
import os
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

    def prune_prefixes(
        self,
        relative_prefixes: tuple[str, ...],
    ) -> tuple[int, int]:
        """Remove only manifest-tracked bundles below generated QC prefixes."""

        prefixes = tuple(
            _normalize_relative_prefix(value) for value in relative_prefixes
        )
        if not prefixes:
            return 0, 0
        artifact_count = 0
        byte_count = 0
        with self._lock:
            selected: list[tuple[str, dict[str, Any]]] = []
            for key, entry in self._entries.items():
                rows = entry.get("artifacts")
                relative_paths = (
                    [row.get("path") for row in rows if isinstance(row, dict)]
                    if isinstance(rows, list)
                    else []
                )
                if _matches_prefix(key, prefixes) or any(
                    isinstance(path, str) and _matches_prefix(path, prefixes)
                    for path in relative_paths
                ):
                    selected.append((key, entry))
            for key, entry in selected:
                rows = entry.get("artifacts")
                if isinstance(rows, list):
                    for row in rows:
                        relative = row.get("path") if isinstance(row, dict) else None
                        if not isinstance(relative, str):
                            continue
                        removed_bytes = _unlink_generated_artifact(
                            self.run_dir,
                            relative,
                        )
                        if removed_bytes is not None:
                            artifact_count += 1
                            byte_count += removed_bytes
                self._entries.pop(key, None)
            if selected:
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
                    # Cleanup metadata must not determine scientific execution.
                    pass
        return artifact_count, byte_count

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


def _normalize_relative_prefix(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("registration QC prune prefix must not be blank")
    normalized = Path(value).as_posix().strip("/")
    parts = Path(normalized).parts
    if (
        not normalized
        or normalized == "."
        or ".." in parts
        or Path(value).is_absolute()
    ):
        raise ValueError("registration QC prune prefix must be a safe relative path")
    return f"{normalized}/"


def _matches_prefix(relative: str, prefixes: tuple[str, ...]) -> bool:
    normalized = Path(relative).as_posix().lstrip("/")
    return any(normalized.startswith(prefix) for prefix in prefixes)


def _unlink_generated_artifact(run_dir: Path, relative: str) -> int | None:
    root = run_dir.resolve()
    candidate = Path(os.path.abspath(root / relative))
    if not candidate.is_relative_to(root):
        return None
    try:
        parent = candidate.parent.resolve()
        if not parent.is_relative_to(root):
            return None
        path = parent / candidate.name
        if not path.is_symlink() and not path.is_file():
            return None
        size = path.lstat().st_size
        path.unlink()
        return size
    except OSError:
        return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
