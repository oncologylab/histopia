from __future__ import annotations

import json
from pathlib import Path

import pytest

from histopia.visualization import export_static_showcase


def _write_viewer(root: Path) -> None:
    (root / "assets" / "5997").mkdir(parents=True)
    (root / "assets" / "4257").mkdir()
    for name in ("index.html", "viewer.js", "styles.css"):
        (root / name).write_text(f"{name}\n")
    (root / "assets" / "5997" / "section.webp").write_bytes(b"5997")
    (root / "assets" / "4257" / "section.webp").write_bytes(b"4257")
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "mice": [
                    {
                        "id": "4257",
                        "slides": [
                            {
                                "id": "other.ndpi",
                                "texture": "assets/4257/section.webp",
                            }
                        ],
                    },
                    {
                        "id": "5997",
                        "slides": [
                            {
                                "id": "demo.ndpi",
                                "texture": "assets/5997/section.webp",
                            }
                        ],
                        "semantic": {
                            "fingerprint": "approved-fingerprint",
                            "review": {
                                "approved": True,
                                "fingerprint_matches": True,
                            },
                        },
                    },
                ],
            }
        )
    )


def test_export_static_showcase_copies_only_selected_mouse(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "showcase"
    _write_viewer(source)

    index = export_static_showcase(source, output, "5997")

    assert index == output / "index.html"
    manifest = json.loads((output / "manifest.json").read_text())
    assert [mouse["id"] for mouse in manifest["mice"]] == ["5997"]
    assert (output / "assets" / "5997" / "section.webp").read_bytes() == b"5997"
    assert not (output / "assets" / "4257").exists()
    assert (output / ".nojekyll").exists()
    inventory = json.loads((output / "showcase.json").read_text())
    assert inventory["mouse_id"] == "5997"
    assert inventory["semantic_fingerprint"] == "approved-fingerprint"
    assert inventory["semantic_approved"] is True
    assert "manifest.json" in inventory["files"]
    assert all(
        len(metadata["sha256"]) == 64 for metadata in inventory["files"].values()
    )


def test_export_static_showcase_rejects_unknown_mouse(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_viewer(source)

    with pytest.raises(ValueError, match="unknown viewer mouse"):
        export_static_showcase(source, tmp_path / "showcase", "missing")


def test_export_static_showcase_refuses_nonempty_output(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "showcase"
    _write_viewer(source)
    output.mkdir()
    (output / "keep.txt").write_text("do not replace")

    with pytest.raises(FileExistsError, match="not empty"):
        export_static_showcase(source, output, "5997")


def test_export_static_showcase_rejects_local_manifest_paths(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _write_viewer(source)
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["mice"][1]["slides"][0]["source"] = "/private/data/demo.ndpi"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="local absolute path"):
        export_static_showcase(source, tmp_path / "showcase", "5997")


def test_export_static_showcase_requires_matching_semantic_approval(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _write_viewer(source)
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["mice"][1]["semantic"]["review"]["approved"] = False
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="not fingerprint-approved"):
        export_static_showcase(source, tmp_path / "showcase", "5997")
