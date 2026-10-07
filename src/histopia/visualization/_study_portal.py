"""Attach verified study exports to an existing workflow review portal."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from histopia._atomic import write_json_atomic, write_text_atomic
from histopia.study._manifest import file_sha256, fingerprint


def attach_study_figure(figure_dir: Path | str, review_dir: Path | str) -> Path:
    """Add a versioned figure tab, retaining every existing workflow tab.

    Only the named figure and region-view exports are copied. Raw inputs,
    models and local operational manifests are never included. Export hashes,
    the region's study binding and selected region identity are checked first.
    This adds a review surface; it does not approve any scientific result.
    """
    source, destination = Path(figure_dir), Path(review_dir)
    portal = json.loads((destination / "manifest.json").read_text())
    if not isinstance(portal.get("tabs"), list):
        raise ValueError("destination is not a workflow review portal")
    provenance = json.loads((source / "figure-provenance.json").read_text())
    expected = {f"histopia-six-panel.{ext}" for ext in ("svg", "pdf", "png")}
    if set(provenance.get("exports", {})) != expected:
        raise ValueError("figure export manifest is incomplete")
    for name, digest in provenance["exports"].items():
        if file_sha256(source / name) != digest:
            raise ValueError("figure export hash mismatch")
    names = sorted(expected) + ["index.html", "captions.md", "figure-provenance.json"]
    region_source = source / "regions"
    if region_source.is_dir():
        regions = json.loads((region_source / "regions.json").read_text())
        if regions.get("study_fingerprint") != provenance.get("study_fingerprint"):
            raise ValueError("region view belongs to a different study")
        selected = provenance.get("selected_region_id")
        if selected and selected not in {r["region_id"] for r in regions["regions"]}:
            raise ValueError("figure selection is missing from the region view")
        names += [
            "regions/" + name
            for name in (
                "index.html",
                "regions.json",
                "regions-data.js",
                "regions.js",
                "regions.css",
                "all-regions.svg",
            )
        ]
    bindings = {name: file_sha256(source / name) for name in names}
    version = fingerprint(bindings)
    target = destination / ("study-figure-" + version[:16])
    for name, digest in bindings.items():
        output = target / name
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            if file_sha256(output) != digest:
                raise ValueError("existing versioned figure has changed")
        else:
            shutil.copyfile(source / name, output)
        if file_sha256(output) != digest:
            raise ValueError("figure copy verification failed")
    tab = {
        "id": "study-figure",
        "label": "Study figure",
        "href": (target.relative_to(destination) / "index.html").as_posix(),
        "study_fingerprint": provenance["study_fingerprint"],
        "export_fingerprint": version,
    }
    portal["tabs"] = [r for r in portal["tabs"] if r["id"] != tab["id"]] + [tab]
    write_json_atomic(destination / "manifest.json", portal)
    write_text_atomic(
        destination / "manifest-data.js",
        "globalThis.HISTOPIA_WORKFLOW_REVIEW="
        + json.dumps(portal, separators=(",", ":"))
        + ";\n",
    )
    return target / "index.html"
