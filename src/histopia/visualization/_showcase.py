"""Export a validated viewer mouse as a self-contained static showcase."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

_VIEWER_FILES = ("index.html", "viewer.js", "styles.css")
_WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")


def export_static_showcase(
    source_dir: Path | str,
    output_dir: Path | str,
    mouse_id: str,
) -> Path:
    """Export one viewer mouse without retaining unrelated research artifacts.

    ``source_dir`` must be an already generated Histopia viewer directory. The
    output contains browser code, the selected manifest entry, its static
    textures, a ``.nojekyll`` marker, and a SHA-256 inventory. Existing
    non-empty output directories are never replaced.
    """

    source = Path(source_dir)
    output = Path(output_dir)
    if source.resolve() == output.resolve():
        raise ValueError("showcase output must differ from the source viewer")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"showcase output directory is not empty: {output}")

    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    mice = manifest.get("mice")
    if not isinstance(mice, list):
        raise ValueError("viewer manifest must contain a mice list")
    matches = [mouse for mouse in mice if str(mouse.get("id")) == mouse_id]
    if len(matches) != 1:
        raise ValueError(f"unknown viewer mouse: {mouse_id}")
    selected = matches[0]
    static_manifest = {
        "schema_version": manifest.get("schema_version", 1),
        "mice": [selected],
    }
    _reject_local_paths(static_manifest)
    semantic = selected.get("semantic")
    semantic = semantic if isinstance(semantic, dict) else {}
    review = semantic.get("review")
    review = review if isinstance(review, dict) else {}
    semantic_approved = bool(
        review.get("approved") and review.get("fingerprint_matches")
    )
    if semantic and not semantic_approved:
        raise ValueError("semantic showcase result is not fingerprint-approved")

    assets = source / "assets" / mouse_id
    if not assets.is_dir():
        raise FileNotFoundError(f"viewer assets not found for mouse: {mouse_id}")
    for filename in _VIEWER_FILES:
        if not (source / filename).is_file():
            raise FileNotFoundError(f"viewer file not found: {filename}")

    output.mkdir(parents=True, exist_ok=True)
    for filename in _VIEWER_FILES:
        shutil.copy2(source / filename, output / filename)
    shutil.copytree(assets, output / "assets" / mouse_id)
    (output / "manifest.json").write_text(json.dumps(static_manifest, indent=2) + "\n")
    (output / ".nojekyll").touch()

    inventory = {
        "schema_version": 1,
        "mouse_id": mouse_id,
        "semantic_fingerprint": semantic.get("fingerprint"),
        "semantic_approved": semantic_approved,
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
