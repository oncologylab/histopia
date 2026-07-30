"""Export validated viewer mice as a self-contained static showcase."""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

_VIEWER_FILES = (
    "index.html",
    "viewer.js",
    "styles.css",
    "focus-viewer.css",
    "focus-viewer.js",
)
_VIEWER_DIRECTORIES = ("vendor",)
_WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")


def export_static_showcase(
    source_dir: Path | str,
    output_dir: Path | str,
    mouse_ids: str | Sequence[str],
    *,
    review_config: Path | str | None = None,
    wsi_sections: dict[str, Sequence[str]] | None = None,
    max_bytes: int = 900 * 1024 * 1024,
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
    wsi_sections = wsi_sections or {}
    if not selected_ids:
        raise ValueError("showcase requires at least one viewer mouse")
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError("showcase contains a duplicate viewer mouse")
    if source.resolve() == output.resolve():
        raise ValueError("showcase output must differ from the source viewer")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"showcase output directory is not empty: {output}")
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ValueError("showcase max_bytes must be a positive integer")
    unknown_wsi_mice = set(wsi_sections) - set(selected_ids)
    if unknown_wsi_mice:
        raise ValueError(
            f"WSI showcase mouse is not selected: {sorted(unknown_wsi_mice)[0]}"
        )
    if wsi_sections and review_config is None:
        raise ValueError("WSI showcase requires a review configuration")

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
    for mouse in selected:
        mouse_id = str(mouse.get("id"))
        if mouse_id in wsi_sections:
            mouse["native_resolution"] = {
                "sections": list(wsi_sections[mouse_id]),
                "metadata_template": "wsi/{cohort}/{section}/metadata.json",
            }
    _reject_local_paths(static_manifest)
    semantic_results: dict[str, dict[str, bool | str | None]] = {}
    stain_results: dict[str, dict[str, bool | str | None]] = {}
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
        stain = mouse.get("stain")
        stain = stain if isinstance(stain, dict) else {}
        stain_review = stain.get("review")
        stain_review = stain_review if isinstance(stain_review, dict) else {}
        stain_approved = bool(
            stain_review.get("approved") and stain_review.get("fingerprint_matches")
        )
        if stain and not stain_approved:
            raise ValueError(
                f"stain showcase result is not fingerprint-approved: {mouse_id}"
            )
        stain_results[mouse_id] = {
            "fingerprint": stain.get("fingerprint"),
            "approved": stain_approved,
        }
        assets = source / "assets" / mouse_id
        if not assets.is_dir():
            raise FileNotFoundError(f"viewer assets not found for mouse: {mouse_id}")
    for filename in _VIEWER_FILES:
        if not (source / filename).is_file():
            raise FileNotFoundError(f"viewer file not found: {filename}")
    for directory in _VIEWER_DIRECTORIES:
        if not (source / directory).is_dir():
            raise FileNotFoundError(f"viewer directory not found: {directory}")

    output.mkdir(parents=True, exist_ok=True)
    for filename in _VIEWER_FILES:
        shutil.copy2(source / filename, output / filename)
    for directory in _VIEWER_DIRECTORIES:
        shutil.copytree(source / directory, output / directory)
    for mouse_id in selected_ids:
        shutil.copytree(source / "assets" / mouse_id, output / "assets" / mouse_id)
    (output / "manifest.json").write_text(json.dumps(static_manifest, indent=2) + "\n")
    (output / ".nojekyll").touch()
    wsi_inventory = (
        _export_static_wsi(
            output,
            review_config,
            wsi_sections,
            max_bytes=max_bytes,
        )
        if wsi_sections
        else {}
    )

    inventory = {
        "schema_version": 4,
        "mouse_ids": list(selected_ids),
        "semantic_results": semantic_results,
        "stain_results": stain_results,
        "wsi_sections": wsi_inventory,
        "files": _file_inventory(output),
    }
    (output / "showcase.json").write_text(json.dumps(inventory, indent=2) + "\n")
    final_size = _directory_size(output)
    if final_size > max_bytes:
        raise ValueError(f"showcase size {final_size} exceeds max_bytes {max_bytes}")
    return output / "index.html"


def _export_static_wsi(
    output: Path,
    review_config: Path | str | None,
    selections: dict[str, Sequence[str]],
    *,
    max_bytes: int,
) -> dict[str, list[str]]:
    try:
        from PIL import Image
    except ImportError as error:
        from histopia.registration._errors import OptionalDependencyError

        raise OptionalDependencyError("pillow", "wsi") from error

    from histopia.visualization._review_api import ReviewDecisionService
    from histopia.visualization._wsi_tiles import WsiTileService

    assert review_config is not None
    runs = ReviewDecisionService.from_file(review_config).wsi_runs()
    selected_runs = {cohort: runs[cohort] for cohort in selections if cohort in runs}
    missing = set(selections) - set(selected_runs)
    if missing:
        raise ValueError(f"WSI showcase has no registered export: {sorted(missing)[0]}")
    service = WsiTileService.from_runs(selected_runs)
    current_size = _directory_size(output)
    exported: dict[str, list[str]] = {}
    for cohort, section_ids in sorted(selections.items()):
        if not section_ids:
            raise ValueError(f"WSI showcase has no sections for {cohort}")
        if len(set(section_ids)) != len(section_ids):
            raise ValueError(f"WSI showcase has duplicate sections for {cohort}")
        exported[cohort] = []
        for section_id in section_ids:
            section = service.section(cohort, section_id)
            layer = section.layers.get("registered")
            if layer is None:
                raise ValueError(
                    f"WSI showcase section is not registered: {cohort}/{section_id}"
                )
            metadata = service.metadata(cohort, section_id)
            registered_metadata = dict(metadata["layers"]["registered"])
            registered_metadata["tile_url_template"] = (
                f"registered/{layer.digest}/{{level}}/{{x}}/{{y}}.jpg"
            )
            existing_tiles: list[str] = []
            for level, dimensions in enumerate(layer.levels):
                columns = math.ceil(dimensions.width / layer.tile_size)
                rows = math.ceil(dimensions.height / layer.tile_size)
                for x in range(columns):
                    for y in range(rows):
                        tile, _, _ = service.render_tile(
                            cohort,
                            section_id,
                            "registered",
                            layer.digest,
                            level,
                            x,
                            y,
                        )
                        if _is_blank_jpeg(Image, tile):
                            continue
                        relative = (
                            Path("wsi")
                            / cohort
                            / section_id
                            / "registered"
                            / layer.digest
                            / str(level)
                            / str(x)
                            / f"{y}.jpg"
                        )
                        if current_size + len(tile) > max_bytes - 1024 * 1024:
                            raise ValueError("WSI showcase exceeds its byte budget")
                        destination = output / relative
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        destination.write_bytes(tile)
                        current_size += len(tile)
                        existing_tiles.append(f"{level}/{x}/{y}")
            registered_metadata["existing_tiles"] = existing_tiles
            metadata["layers"] = {"registered": registered_metadata}
            metadata_path = output / "wsi" / cohort / section_id / "metadata.json"
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            encoded = json.dumps(metadata, indent=2) + "\n"
            metadata_path.write_text(encoded)
            current_size += len(encoded.encode())
            exported[cohort].append(section_id)
    return exported


def _is_blank_jpeg(image_module, payload: bytes) -> bool:  # type: ignore[no-untyped-def]
    with image_module.open(io.BytesIO(payload)) as image:
        extrema = image.convert("RGB").getextrema()
    return all(low >= 250 for low, _ in extrema)


def _directory_size(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


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
