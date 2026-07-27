"""Integrity checks for sealed quantitative stain results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def validate_stain_result(
    run_dir: Path | str,
    payload: dict[str, object] | None = None,
) -> dict[str, object]:
    """Validate the schema, fingerprint, and every declared artifact."""

    root = Path(run_dir)
    loaded = (
        json.loads((root / "stain_result.json").read_text())
        if payload is None
        else dict(payload)
    )
    if loaded.get("schema_version") != 1:
        raise ValueError("stain result must use schema version 1")
    references = _referenced_artifacts(root, loaded)
    declared = loaded.get("artifacts")
    if not isinstance(declared, dict) or set(declared) != set(references):
        raise ValueError("stain result artifact manifest is incomplete or stale")
    for relative, path in references.items():
        if not path.is_file():
            raise ValueError(f"stain result artifact is missing: {relative}")
        if declared[relative] != _sha256_file(path):
            raise ValueError(f"stain result artifact digest mismatch: {relative}")
    fingerprint = loaded.get("fingerprint")
    core = {key: value for key, value in loaded.items() if key != "fingerprint"}
    if fingerprint != _fingerprint_core(core):
        raise ValueError("stain result fingerprint is stale")
    return loaded


def _seal_stain_result(
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
    raw_paths: list[object] = [payload.get("preflight"), payload.get("benchmark")]
    for slide in payload.get("slides", []):
        if not isinstance(slide, dict):
            raise ValueError("stain result slide rows must be objects")
        raw_paths.extend((slide.get("model"), slide.get("map")))
    root_resolved = root.resolve()
    references: dict[str, Path] = {}
    for raw_path in raw_paths:
        if raw_path is None:
            continue
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError("stain artifact paths must be non-empty strings")
        relative = Path(raw_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("stain artifact paths must stay inside the run directory")
        resolved = (root_resolved / relative).resolve()
        if not resolved.is_relative_to(root_resolved):
            raise ValueError("stain artifact paths must stay inside the run directory")
        key = relative.as_posix()
        if key in references:
            raise ValueError(f"stain artifact is referenced more than once: {key}")
        references[key] = resolved
    return references


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
