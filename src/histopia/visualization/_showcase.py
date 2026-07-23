"""Export validated viewer mice as a self-contained static showcase."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

_VIEWER_FILES = ("index.html", "viewer.js", "styles.css")
_WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")


def export_static_showcase(
    source_dir: Path | str,
    output_dir: Path | str,
    mouse_ids: str | Sequence[str],
) -> Path:
    """Export selected viewer mice without retaining unrelated artifacts.

    ``source_dir`` must be an already generated Histopia viewer directory. The
    output contains browser code, the selected manifest entries, their static
    textures, a ``.nojekyll`` marker, and a SHA-256 inventory. Existing
    non-empty output directories are never replaced.
    """

    source = Path(source_dir)
    output = Path(output_dir)
    selected_ids = (mouse_ids,) if isinstance(mouse_ids, str) else tuple(mouse_ids)
    if not selected_ids:
        raise ValueError("showcase requires at least one viewer mouse")
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError("showcase contains a duplicate viewer mouse")
    if source.resolve() == output.resolve():
        raise ValueError("showcase output must differ from the source viewer")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"showcase output directory is not empty: {output}")

    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    mice = manifest.get("mice")
    if not isinstance(mice, list):
        raise ValueError("viewer manifest must contain a mice list")
    mice_by_id = {str(mouse.get("id")): mouse for mouse in mice}
    unknown = [mouse_id for mouse_id in selected_ids if mouse_id not in mice_by_id]
    if unknown:
        raise ValueError(f"unknown viewer mouse: {unknown[0]}")
    selected = [mice_by_id[mouse_id] for mouse_id in selected_ids]
    static_manifest = {
        "schema_version": manifest.get("schema_version", 1),
        "mice": selected,
    }
    _reject_local_paths(static_manifest)
    semantic_results: dict[str, dict[str, bool | str | None]] = {}
    for mouse_id, mouse in zip(selected_ids, selected, strict=True):
        semantic = mouse.get("semantic")
        semantic = semantic if isinstance(semantic, dict) else {}
        review = semantic.get("review")
        review = review if isinstance(review, dict) else {}
        semantic_approved = bool(
            review.get("approved") and review.get("fingerprint_matches")
        )
        if semantic and not semantic_approved:
            raise ValueError(
                f"semantic showcase result is not fingerprint-approved: {mouse_id}"
            )
        semantic_results[mouse_id] = {
            "fingerprint": semantic.get("fingerprint"),
            "approved": semantic_approved,
        }
        assets = source / "assets" / mouse_id
        if not assets.is_dir():
            raise FileNotFoundError(f"viewer assets not found for mouse: {mouse_id}")
    for filename in _VIEWER_FILES:
        if not (source / filename).is_file():
            raise FileNotFoundError(f"viewer file not found: {filename}")

    output.mkdir(parents=True, exist_ok=True)
    for filename in _VIEWER_FILES:
        shutil.copy2(source / filename, output / filename)
    for mouse_id in selected_ids:
        shutil.copytree(source / "assets" / mouse_id, output / "assets" / mouse_id)
    (output / "manifest.json").write_text(json.dumps(static_manifest, indent=2) + "\n")
    (output / ".nojekyll").touch()

    inventory = {
        "schema_version": 2,
        "mouse_ids": list(selected_ids),
        "semantic_results": semantic_results,
        "files": _file_inventory(output),
    }
    (output / "showcase.json").write_text(json.dumps(inventory, indent=2) + "\n")
    return output / "index.html"


def _reject_local_paths(value: Any) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _reject_local_paths(item)
    elif isinstance(value, list):
        for item in value:
            _reject_local_paths(item)
    elif isinstance(value, str):
        if value.startswith(("/", "file://")) or _WINDOWS_ABSOLUTE.match(value):
            raise ValueError("viewer manifest contains a local absolute path")


def _file_inventory(root: Path) -> dict[str, dict[str, int | str]]:
    inventory: dict[str, dict[str, int | str]] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        inventory[relative] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        }
    return inventory
