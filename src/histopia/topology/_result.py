"""Sealed result and review-state handling for topology reconstruction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from histopia._atomic import write_json_atomic


def write_topology_result(
    root: Path,
    core: dict[str, object],
) -> Path:
    """Seal declared artifacts and reset stale review state."""

    payload = _seal_topology_result(root, core)
    result_path = root / "topology_result.json"
    write_json_atomic(result_path, payload)
    review_path = root / "topology_review.json"
    review = _current_review(review_path, str(payload["fingerprint"]))
    write_json_atomic(review_path, review)
    return result_path


def validate_topology_result(
    run_dir: Path | str,
    payload: dict[str, object] | None = None,
) -> dict[str, object]:
    """Validate schema, fingerprint, and every topology artifact."""

    root = Path(run_dir)
    loaded = (
        json.loads((root / "topology_result.json").read_text())
        if payload is None
        else dict(payload)
    )
    if loaded.get("schema_version") not in {1, 2}:
        raise ValueError("topology result must use schema version 1 or 2")
    references = _referenced_artifacts(root, loaded)
    declared = loaded.get("artifacts")
    if not isinstance(declared, dict) or set(declared) != set(references):
        raise ValueError("topology result artifact manifest is incomplete or stale")
    for relative, path in references.items():
        if not path.is_file():
            raise ValueError(f"topology result artifact is missing: {relative}")
        if declared[relative] != _sha256_file(path):
            raise ValueError(f"topology result artifact digest mismatch: {relative}")
    fingerprint = loaded.get("fingerprint")
    core = {key: value for key, value in loaded.items() if key != "fingerprint"}
    if fingerprint != _fingerprint_core(core):
        raise ValueError("topology result fingerprint is stale")
    return loaded


def _seal_topology_result(
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
    raw_paths: list[object] = [payload.get("preflight")]
    if payload.get("benchmark") is not None:
        raw_paths.append(payload["benchmark"])
    raw_paths.extend(
        row.get("artifact")
        for row in payload.get("planes", [])
        if isinstance(row, dict)
    )
    mesh_rows = (
        payload.get("meshes", [])
        if payload.get("schema_version") == 1
        else [
            payload.get("envelope"),
            *payload.get("semantic_regions", []),
            *payload.get("semantic_partition_regions", []),
            payload.get("uncertainty"),
        ]
    )
    if payload.get("schema_version") == 2:
        grid = payload.get("reconstruction_grid")
        if not isinstance(grid, dict):
            raise ValueError("topology v2 result has no reconstruction grid")
        raw_paths.append(grid.get("artifact"))
    for row in mesh_rows:
        if row is None:
            continue
        if not isinstance(row, dict):
            raise ValueError("topology mesh rows must be objects")
        raw_paths.extend((row.get("artifact"), row.get("viewer_asset")))
    root_resolved = root.resolve()
    references: dict[str, Path] = {}
    for value in raw_paths:
        if not isinstance(value, str) or not value:
            raise ValueError("topology artifact paths must be non-empty strings")
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("topology artifact paths must stay inside the run")
        resolved = (root_resolved / relative).resolve()
        if not resolved.is_relative_to(root_resolved):
            raise ValueError("topology artifact paths must stay inside the run")
        key = relative.as_posix()
        if key in references:
            raise ValueError(f"topology artifact is referenced more than once: {key}")
        references[key] = resolved
    return references


def _current_review(path: Path, fingerprint: str) -> dict[str, object]:
    default: dict[str, object] = {
        "schema_version": 1,
        "approved": False,
        "fingerprint": fingerprint,
        "reviewer": None,
        "reviewed_at": None,
        "notes": "",
    }
    try:
        payload = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or payload.get("fingerprint") != fingerprint
    ):
        return default
    return payload


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
