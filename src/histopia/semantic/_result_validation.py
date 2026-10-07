"""Lightweight integrity checks for portable semantic-atlas results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def validate_semantic_result(
    run_dir: Path | str,
    payload: dict[str, object] | None = None,
) -> dict[str, object]:
    """Load and verify a schema-3 result and every referenced artifact."""

    root = Path(run_dir)
    loaded = validate_semantic_result_index(root, payload)
    references = _referenced_artifacts(root, loaded)
    declared = loaded["artifacts"]
    assert isinstance(declared, dict)
    for relative, path in references.items():
        if declared[relative] != _sha256_file(path):
            raise ValueError(f"semantic result artifact digest mismatch: {relative}")
    return loaded


def validate_semantic_result_index(
    run_dir: Path | str,
    payload: dict[str, object] | None = None,
) -> dict[str, object]:
    """Validate semantic metadata without eagerly hashing large label maps."""

    root = Path(run_dir)
    loaded = (
        json.loads((root / "semantic_result.json").read_text())
        if payload is None
        else dict(payload)
    )
    if loaded.get("schema_version") != 3:
        raise ValueError("semantic result must use schema version 3")
    references = _referenced_artifacts(root, loaded)
    declared = loaded.get("artifacts")
    if not isinstance(declared, dict) or set(declared) != set(references):
        raise ValueError("semantic result artifact manifest is incomplete or stale")
    for relative, path in references.items():
        digest = declared[relative]
        if (
            not path.is_file()
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"semantic result artifact is missing: {relative}")
    fingerprint = loaded.get("fingerprint")
    core = {key: value for key, value in loaded.items() if key != "fingerprint"}
    if fingerprint != _fingerprint_core(core):
        raise ValueError("semantic result fingerprint is stale")
    return loaded


def validate_semantic_artifact(
    path: Path | str,
    expected_sha256: str,
) -> Path:
    """Validate one semantic artifact immediately before its first use."""

    artifact = Path(path)
    if (
        len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
        or not artifact.is_file()
        or _sha256_file(artifact) != expected_sha256
    ):
        raise ValueError(f"semantic result artifact digest mismatch: {artifact.name}")
    return artifact


def _seal_semantic_result(
    root: Path,
    core: dict[str, object],
) -> dict[str, object]:
    sealed = dict(core)
    references = _referenced_artifacts(root, sealed)
    sealed["artifacts"] = {
        relative: _sha256_file(path) for relative, path in sorted(references.items())
    }
    return {**sealed, "fingerprint": _fingerprint_core(sealed)}


def _referenced_artifacts(
    root: Path,
    payload: dict[str, object],
) -> dict[str, Path]:
    root_resolved = root.resolve()
    raw_paths: list[object] = [payload.get("model")]
    for slide in payload.get("slides", []):
        raw_paths.extend(slide.get("labels", {}).values())
    raw_paths.extend(pair.get("artifact") for pair in payload.get("topology_pairs", []))
    references: dict[str, Path] = {}
    for raw_path in raw_paths:
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError("semantic artifact paths must be non-empty relative paths")
        relative, resolved = _safe_artifact_path(root_resolved, raw_path)
        if relative in references:
            raise ValueError(
                f"semantic artifact is referenced more than once: {relative}"
            )
        references[relative] = resolved
    return references


def _safe_artifact_path(root_resolved: Path, value: str) -> tuple[str, Path]:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(
            "semantic artifact paths must be relative to the run directory"
        )
    resolved = (root_resolved / relative).resolve()
    if not resolved.is_relative_to(root_resolved):
        raise ValueError(
            "semantic artifact paths must be relative to the run directory"
        )
    return relative.as_posix(), resolved


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint_core(core: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
