"""Sealed cell-boundary result manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from histopia._atomic import write_json_atomic


def write_cell_result(root: Path, core: dict[str, object]) -> Path:
    """Seal declared artifacts and reset review state for changed results."""

    payload = _seal(root, core)
    result_path = write_json_atomic(root / "cell_result.json", payload)
    review_path = root / "cell_review.json"
    write_json_atomic(review_path, _current_review(review_path, payload))
    return result_path


def validate_cell_result(
    run_dir: Path | str,
    payload: dict[str, object] | None = None,
) -> dict[str, object]:
    """Validate schema, fingerprint, and every declared cell artifact."""

    loaded = validate_cell_result_index(run_dir, payload)
    root = Path(run_dir)
    references = _referenced_artifacts(root, loaded)
    declared = loaded["artifacts"]
    assert isinstance(declared, dict)
    for relative, path in references.items():
        if declared[relative] != _sha256_file(path):
            raise ValueError(f"cell result artifact digest mismatch: {relative}")
    return loaded


def validate_cell_result_index(
    run_dir: Path | str,
    payload: dict[str, object] | None = None,
) -> dict[str, object]:
    """Validate a sealed cell index without eagerly hashing large labels."""

    root = Path(run_dir)
    loaded = (
        json.loads((root / "cell_result.json").read_text())
        if payload is None
        else dict(payload)
    )
    if loaded.get("schema_version") != 1:
        raise ValueError("cell result must use schema version 1")
    references = _referenced_artifacts(root, loaded)
    declared = loaded.get("artifacts")
    if not isinstance(declared, dict) or set(declared) != set(references):
        raise ValueError("cell result artifact manifest is incomplete or stale")
    for relative, path in references.items():
        digest = declared[relative]
        if (
            not path.is_file()
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"cell result artifact is missing: {relative}")
    fingerprint = loaded.get("fingerprint")
    core = {key: value for key, value in loaded.items() if key != "fingerprint"}
    if fingerprint != _fingerprint(core):
        raise ValueError("cell result fingerprint is stale")
    return loaded


def validate_cell_artifact(path: Path | str, expected_sha256: str) -> Path:
    """Validate one cell artifact immediately before its first use."""

    artifact = Path(path)
    if (
        len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
        or not artifact.is_file()
        or _sha256_file(artifact) != expected_sha256
    ):
        raise ValueError(f"cell result artifact digest mismatch: {artifact.name}")
    return artifact


def _seal(root: Path, core: dict[str, object]) -> dict[str, object]:
    sealed = dict(core)
    references = _referenced_artifacts(root, sealed)
    sealed["artifacts"] = {
        relative: _sha256_file(path) for relative, path in sorted(references.items())
    }
    return {**sealed, "fingerprint": _fingerprint(sealed)}


def _referenced_artifacts(root: Path, payload: dict[str, object]) -> dict[str, Path]:
    raw_paths: list[object] = [payload.get("preflight")]
    benchmark = payload.get("benchmark")
    if benchmark is not None:
        raw_paths.append(benchmark)
    slides = payload.get("slides")
    if not isinstance(slides, list) or not slides:
        raise ValueError("cell result must contain at least one slide")
    for row in slides:
        if not isinstance(row, dict):
            raise ValueError("cell result slide rows must be objects")
        raw_paths.extend((row.get("labels"), row.get("qc")))
    root_resolved = root.resolve()
    output: dict[str, Path] = {}
    for value in raw_paths:
        if not isinstance(value, str) or not value:
            raise ValueError("cell artifact paths must be non-empty strings")
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("cell artifact paths must stay inside the run")
        resolved = (root_resolved / relative).resolve()
        if not resolved.is_relative_to(root_resolved):
            raise ValueError("cell artifact paths must stay inside the run")
        key = relative.as_posix()
        if key in output:
            raise ValueError(f"cell artifact is referenced more than once: {key}")
        output[key] = resolved
    return output


def _current_review(path: Path, result: dict[str, object]) -> dict[str, object]:
    fingerprint = str(result["fingerprint"])
    sections = [str(row["section"]) for row in result["slides"]]
    defaults = {
        section: {
            "accepted": False,
            "reviewer": None,
            "reviewed_at": None,
            "notes": "",
            "issues": [],
        }
        for section in sections
    }
    try:
        payload = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        payload = None
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or payload.get("fingerprint") != fingerprint
        or not isinstance(payload.get("sections"), dict)
    ):
        return {
            "schema_version": 1,
            "fingerprint": fingerprint,
            "approved": False,
            "sections": defaults,
        }
    rows = payload["sections"]
    return {
        "schema_version": 1,
        "fingerprint": fingerprint,
        "approved": bool(payload.get("approved"))
        and all(
            isinstance(rows.get(section), dict)
            and rows[section].get("accepted") is True
            for section in sections
        ),
        "sections": {
            section: dict(rows[section])
            if isinstance(rows.get(section), dict)
            else defaults[section]
            for section in sections
        },
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
