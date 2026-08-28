#!/usr/bin/env python3
"""Export the three-mouse Histopia presentation as a read-only static site.

The live WSI service is intentionally used only while building.  The exported
site contains bounded, tissue-focused image sources and client-side 3D assets,
so it can run on GitHub Pages without an API, credentials, or write controls.
"""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests
from PIL import Image

COHORTS = ("4312", "4630", "6180")
TARGET_LABELS = {
    "yap": "YAP",
    "ecad": "E-Cad",
    "ck19": "CK19",
    "ki67": "Ki67",
    "perk": "pERK",
    "ncad": "N-Cad",
    "cjun": "cJun",
}
TARGET_SECTIONS = {
    "4312": {
        "yap": "003",
        "ecad": "008",
        "ck19": "019",
        "ki67": "016",
        "perk": "017",
        "ncad": "020",
        "cjun": "025",
    },
    "4630": {
        "yap": "012",
        "ecad": "013",
        "ck19": "002",
        "ki67": "004",
        "perk": "009",
        "ncad": "007",
        "cjun": "023",
    },
    "6180": {
        "yap": "003",
        "ecad": "017",
        "ck19": "007",
        "ki67": "008",
        "perk": "006",
        "ncad": "002",
        "cjun": "015",
    },
}
CELL_SECTIONS = {
    "4312": ("001", "003", "008"),
    "4630": ("001", "012", "013"),
    "6180": ("001", "003", "017"),
}


@dataclass(frozen=True)
class LayerRoute:
    cohort: str
    section: str
    name: str
    digest: str
    level: int
    width: int
    height: int
    tile_size: int
    image_format: str
    model_id: str | None = None


class ExportError(RuntimeError):
    """Raised when the static export cannot be made scientifically coherent."""


def _json_get(session: requests.Session, url: str) -> dict[str, Any]:
    response = session.get(url, timeout=60)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ExportError(f"Expected an object from {url}")
    return payload


def _level_route(
    metadata: dict[str, Any],
    name: str,
    *,
    max_dimension: int,
    model_id: str | None = None,
) -> LayerRoute:
    layers = metadata.get("layers") or {}
    layer = layers.get(name)
    if not isinstance(layer, dict):
        raise ExportError(
            f"Missing layer {name!r} for {metadata.get('cohort')}/"
            f"{metadata.get('section')}"
        )
    levels = layer.get("levels") or []
    if not levels:
        raise ExportError(f"Layer {name!r} has no pyramid levels")
    candidates = [
        (index, row)
        for index, row in enumerate(levels)
        if max(int(row["width"]), int(row["height"])) <= max_dimension
    ]
    index, selected = candidates[-1] if candidates else (0, levels[0])
    return LayerRoute(
        cohort=str(metadata["cohort"]),
        section=str(metadata["section"]),
        name=name,
        digest=str(layer["digest"]),
        level=index,
        width=int(selected["width"]),
        height=int(selected["height"]),
        tile_size=int(layer["tile_size"]),
        image_format=str(layer["format"]),
        model_id=model_id,
    )


def _tile_url(api_base: str, route: LayerRoute, x: int, y: int) -> str:
    scope = (
        f"/protein/{quote(route.model_id, safe='')}" if route.model_id else ""
    )
    return (
        f"{api_base}/api/wsi/{quote(route.cohort, safe='')}/"
        f"{quote(route.section, safe='')}{scope}/"
        f"{quote(route.name, safe='')}/{quote(route.digest, safe='')}/"
        f"{route.level}/{x}/{y}.{route.image_format}"
    )


def _fetch_tile(url: str) -> tuple[str, bytes]:
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    return url, response.content


def _stitch_layer(api_base: str, route: LayerRoute, *, workers: int) -> Image.Image:
    columns = math.ceil(route.width / route.tile_size)
    rows = math.ceil(route.height / route.tile_size)
    urls = {
        _tile_url(api_base, route, x, y): (x, y)
        for y in range(rows)
        for x in range(columns)
    }
    canvas = Image.new("RGBA", (route.width, route.height), (0, 0, 0, 0))
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(_fetch_tile, url) for url in urls]
        for future in as_completed(futures):
            url, content = future.result()
            x, y = urls[url]
            with Image.open(BytesIO(content)) as tile:
                canvas.alpha_composite(tile.convert("RGBA"), (x * route.tile_size, y * route.tile_size))
    return canvas


def _crop_focus(image: Image.Image, metadata: dict[str, Any]) -> Image.Image:
    box = metadata.get("focus_bbox") or {}
    values = [box.get(key) for key in ("x", "y", "width", "height")]
    if not all(isinstance(value, (int, float)) for value in values):
        return image
    x, y, width, height = (float(value) for value in values)
    if width <= 0 or height <= 0:
        return image
    padding = 0.025
    left = max(0.0, x - width * padding)
    top = max(0.0, y - height * padding)
    right = min(1.0, x + width * (1 + padding))
    bottom = min(1.0, y + height * (1 + padding))
    return image.crop(
        (
            round(left * image.width),
            round(top * image.height),
            round(right * image.width),
            round(bottom * image.height),
        )
    )


def _fit_to(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    if image.size == size:
        return image
    return image.resize(size, Image.Resampling.LANCZOS)


def _on_white(image: Image.Image) -> Image.Image:
    layer = image.convert("RGBA")
    background = Image.new("RGBA", layer.size, (248, 250, 252, 255))
    background.alpha_composite(layer)
    return background.convert("RGB")


def _overlay(base: Image.Image, layer: Image.Image, opacity: float) -> Image.Image:
    background = base.convert("RGBA")
    foreground = _fit_to(layer.convert("RGBA"), background.size)
    if opacity < 1:
        alpha = foreground.getchannel("A").point(lambda value: round(value * opacity))
        foreground.putalpha(alpha)
    background.alpha_composite(foreground)
    return background.convert("RGB")


def _asset_name(parts: list[str], digests: list[str]) -> str:
    clean = "-".join(re.sub(r"[^a-z0-9]+", "-", part.lower()).strip("-") for part in parts)
    fingerprint = hashlib.sha256("|".join(digests).encode()).hexdigest()[:12]
    return f"{clean}-{fingerprint}.webp"


def _write_webp(image: Image.Image, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(destination, "WEBP", quality=88, method=6)
    return destination.as_posix()


def _copy_static_views(source: Path, output: Path) -> None:
    for name in ("registration", "atlas", "topology", "protein-atlas"):
        source_dir = source / name
        if not source_dir.is_dir():
            raise ExportError(f"Missing static view: {source_dir}")
        shutil.copytree(
            source_dir,
            output / name,
            ignore=shutil.ignore_patterns(".histopia-*", "__pycache__"),
        )
    _scrub_registration(output / "registration")
    _scrub_atlas(output / "atlas")
    _scrub_topology(output / "topology")


def _replace_file(path: Path, replacements: list[tuple[str, str]]) -> None:
    text = path.read_text(encoding="utf-8")
    for old, new in replacements:
        if old not in text:
            raise ExportError(f"Expected export marker missing from {path}: {old[:80]!r}")
        text = text.replace(old, new)
    path.write_text(text, encoding="utf-8")


def _scrub_registration(root: Path) -> None:
    for path in root.rglob("index.html"):
        text = path.read_text(encoding="utf-8")
        text = re.sub(r"\s*<link rel=\"stylesheet\" href=\"registration-feedback\.css\">", "", text)
        text = re.sub(r"\s*<link rel=\"stylesheet\" href=\"focus-viewer\.css\">", "", text)
        text = re.sub(r"\s*<aside id=\"registration-feedback\"></aside>", "", text)
        text = re.sub(r"\s*<script src=\"vendor/openseadragon\.min\.js\"></script>", "", text)
        text = re.sub(r"\s*<script src=\"focus-viewer\.js\"></script>", "", text)
        text = re.sub(r"\s*<script src=\"registration-feedback\.js\"></script>", "", text)
        text = text.replace("Registration Alignment Review", "Registered Section Stack")
        text = text.replace("Section Order Review", "Section Order")
        text = text.replace("Tissue Mask Review", "Tissue Masks")
        text = text.replace("registration cohort review", "registration presentation")
        text = text.replace("registration review", "registration presentation")
        text = text.replace("Review stage", "Registration stage")
        path.write_text(text, encoding="utf-8")
    for path in root.rglob("registration-feedback.*"):
        path.unlink()
    for path in root.rglob("focus-viewer.*"):
        path.unlink()
    for vendor in root.rglob("vendor"):
        if vendor.is_dir():
            shutil.rmtree(vendor)
    for path in root.rglob("registration-review.js"):
        text = path.read_text(encoding="utf-8")
        text = text.replace(
            'const approval=row.approved?"approved":"review required";\n'
            '  const continuity=row.review_recommended?" · area continuity flag":"";\n'
            '  status.textContent=`${row.slide_count} slides · ${approval}${continuity}`;',
            'status.textContent=`${row.slide_count} registered sections`;',
        )
        path.write_text(text, encoding="utf-8")
    cohort_js = root / "cohort-review.js"
    if cohort_js.is_file():
        cohort_js.write_text(
            '''const manifest=globalThis.HISTOPIA_COHORT_REVIEW_MANIFEST;
if(!manifest||!manifest.reviews.length)throw new Error("Missing cohorts");
const select=document.querySelector("#cohort");
const frame=document.querySelector("#review");
const status=document.querySelector("#status");
for(const row of manifest.reviews){
  const option=document.createElement("option");
  option.value=row.id;option.textContent=row.id;select.append(option);
}
function choose(id){
  const row=manifest.reviews.find(item=>item.id===id)||manifest.reviews[0];
  select.value=row.id;frame.src=row.href;
  status.textContent=`${row.slide_count} registered sections`;
  const url=new URL(location.href);url.searchParams.set("cohort",row.id);
  history.replaceState(null,"",url);
}
select.addEventListener("change",()=>choose(select.value));
choose(new URL(location.href).searchParams.get("cohort"));
''',
            encoding="utf-8",
        )


def _scrub_atlas(root: Path) -> None:
    index = root / "index.html"
    text = index.read_text(encoding="utf-8")
    text = re.sub(r"\s*<link rel=\"stylesheet\" href=\"focus-viewer\.css\">", "", text)
    text = re.sub(r"\s*<script src=\"vendor/openseadragon\.min\.js\"></script>", "", text)
    text = re.sub(r"\s*<script src=\"focus-viewer\.js\"></script>", "", text)
    index.write_text(text, encoding="utf-8")
    for name in ("focus-viewer.css", "focus-viewer.js", "vendor/openseadragon.min.js"):
        path = root / name
        if path.exists():
            path.unlink()
    viewer = root / "viewer.js"
    text = viewer.read_text(encoding="utf-8")
    text = text.replace(
        "3D preview unavailable; open a section for native-resolution review.",
        "3D preview unavailable; select a section to focus the stack.",
    )
    text = text.replace("Open native-resolution section", "Focus section in the stack")
    text = text.replace("  const review = current?.semantic?.review;\n", "")
    text = text.replace(
        "  if (review) {\n"
        "    details.push(review.approved\n"
        "      ? 'Approved'\n"
        "      : (review.fingerprint_matches ? 'Approval required' : 'Review fingerprint mismatch'));\n"
        "  }\n",
        "",
    )
    text = text.replace("    const stainReview = current.stain.review;\n", "")
    text = text.replace(
        "    details.push(stainReview?.approved\n"
        "      ? 'Stain QC approved'\n"
        "      : (stainReview?.fingerprint_matches\n"
        "        ? 'Stain QC approval required'\n"
        "        : 'Stain QC fingerprint mismatch'));\n",
        "",
    )
    text = text.replace(
        "? `Approved registration · ${approval.reviewed_at.slice(0, 10)}`\n"
        "    : (mouse.provisional_order ? 'Provisional section order' : 'Registration approval required');",
        "? `${mouse.slides.length} registered sections`\n"
        "    : `${mouse.slides.length} section stack`;",
    )
    viewer.write_text(text, encoding="utf-8")


def _scrub_topology(root: Path) -> None:
    index = root / "index.html"
    text = index.read_text(encoding="utf-8")
    text = re.sub(r"\s*<section class=\"review\">.*?</section>", "", text, flags=re.S)
    index.write_text(text, encoding="utf-8")
    script = root / "topology-review.js"
    lines = script.read_text(encoding="utf-8").splitlines()
    output: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("let authenticationRequired="):
            continue
        if "const flagged=new Set" in line:
            continue
        if "updateReviewTarget();try" in line:
            line = line.replace("updateReviewTarget();try", "try")
        if stripped.startswith("function updateReviewTarget"):
            continue
        if stripped.startswith('const labels=["envelope_shape"'):
            continue
        if stripped.startswith('document.querySelectorAll("[data-decision]"'):
            continue
        if stripped.startswith("function reviewHeaders"):
            continue
        if stripped.startswith("async function configureReviewAccess"):
            continue
        if stripped.startswith("async function postFeedback"):
            continue
        if stripped.startswith("async function save"):
            continue
        if stripped.startswith('el("accept-passing").onclick'):
            continue
        line = line.replace(
            ';el("review-target").onchange=updateReviewTarget',
            "",
        )
        line = line.replace(
            'configureReviewAccess().catch(error=>{el("status").textContent=error.message});',
            "",
        )
        line = line.replace(
            "3D topology preview unavailable; quantitative review controls remain available.",
            "3D topology preview unavailable; quantitative summaries remain available.",
        )
        line = line.replace(
            "Quantitative metrics and review controls remain active",
            "Quantitative summaries remain available",
        )
        if 'el("provenance").textContent=' in line:
            line = (
                line.split('el("provenance").textContent=', 1)[0]
                + 'el("provenance").textContent=`${current.observed_section_count} '
                'observed sections · ${current.numerical_sample_count} samples`+'
                '(depth?` · cell-sized ${Number(depth.visual_section_spacing_um).toFixed(1)} µm`:"");'
            )
        output.append(line)
    script.write_text("\n".join(output) + "\n", encoding="utf-8")


def _catalogs(api_base: str, session: requests.Session) -> dict[str, dict[str, Any]]:
    return {
        cohort: _json_get(session, f"{api_base}/api/wsi/{cohort}")
        for cohort in COHORTS
    }


def _labels(catalog: dict[str, Any]) -> dict[str, str]:
    return {
        str(row["section"]): str(row.get("label") or row["section"])
        for row in catalog.get("sections", [])
    }


def _render(
    api_base: str,
    metadata: dict[str, Any],
    layer: str,
    *,
    max_dimension: int,
    workers: int,
    model_id: str | None = None,
) -> tuple[Image.Image, LayerRoute]:
    route = _level_route(
        metadata,
        layer,
        max_dimension=max_dimension,
        model_id=model_id,
    )
    image = _crop_focus(_stitch_layer(api_base, route, workers=workers), metadata)
    return image, route


def _export_stain(
    api_base: str,
    output: Path,
    catalogs: dict[str, dict[str, Any]],
    session: requests.Session,
    *,
    max_dimension: int,
    workers: int,
) -> dict[str, Any]:
    cohorts: list[dict[str, Any]] = []
    assets = output / "static" / "previews"
    for cohort in COHORTS:
        rows: list[dict[str, Any]] = []
        labels = _labels(catalogs[cohort])
        for target, section in TARGET_SECTIONS[cohort].items():
            print(f"stain {cohort}/{section} {TARGET_LABELS[target]}", flush=True)
            metadata = _json_get(session, f"{api_base}/api/wsi/{cohort}/{section}")
            output_layer = (
                "stain_adaptive_v3"
                if "stain_adaptive_v3" in metadata["layers"]
                else "stain_output"
                if "stain_output" in metadata["layers"]
                else "stain_corrected"
            )
            map_layer = (
                "stain_adaptive_v3_map"
                if "stain_adaptive_v3_map" in metadata["layers"]
                else "stain_output_map"
                if "stain_output_map" in metadata["layers"]
                else output_layer
            )
            raw, raw_route = _render(
                api_base, metadata, "raw", max_dimension=max_dimension, workers=workers
            )
            raw = _on_white(raw)
            raw_od, raw_od_route = _render(
                api_base,
                metadata,
                "stain_raw",
                max_dimension=max_dimension,
                workers=workers,
            )
            corrected, corrected_route = _render(
                api_base,
                metadata,
                output_layer,
                max_dimension=max_dimension,
                workers=workers,
            )
            corrected_map, map_route = _render(
                api_base,
                metadata,
                map_layer,
                max_dimension=max_dimension,
                workers=workers,
            )
            images = {
                "histology": raw,
                "raw": _overlay(raw, raw_od, 0.72),
                "corrected": _overlay(raw, corrected, 0.72),
                "map": _on_white(corrected_map),
            }
            panes = []
            pane_specs = (
                ("histology", "Native histology", "Native-resolution source context"),
                ("raw", "Raw target OD", "Tissue-supported extraction over histology"),
                ("corrected", "Adaptive target OD", "Adaptive corrected signal over histology"),
                ("map", "Adaptive target OD only", "Validated measurement support · 4 µm/px"),
            )
            routes = {
                "histology": raw_route,
                "raw": raw_od_route,
                "corrected": corrected_route,
                "map": map_route,
            }
            for role, title, subtitle in pane_specs:
                route = routes[role]
                name = _asset_name(
                    ["stain", cohort, section, target, role],
                    [route.digest, raw_route.digest],
                )
                _write_webp(images[role], assets / name)
                panes.append(
                    {
                        "title": title,
                        "subtitle": subtitle,
                        "src": f"../static/previews/{name}",
                        "digest": route.digest,
                    }
                )
            rows.append(
                {
                    "section": section,
                    "section_label": labels.get(section, TARGET_LABELS[target]),
                    "target": target,
                    "target_label": TARGET_LABELS[target],
                    "panes": panes,
                }
            )
        cohorts.append({"id": cohort, "examples": rows})
    manifest = {
        "schema_version": 1,
        "view": "stain",
        "title": "Stain extraction",
        "subtitle": "Tissue-supported target optical density · validated at 4 µm/px",
        "cohorts": cohorts,
    }
    _write_comparison_view(output / "stain", manifest)
    return manifest


def _export_cells(
    api_base: str,
    output: Path,
    catalogs: dict[str, dict[str, Any]],
    session: requests.Session,
    *,
    max_dimension: int,
    workers: int,
) -> dict[str, Any]:
    cohorts: list[dict[str, Any]] = []
    assets = output / "static" / "previews"
    for cohort in COHORTS:
        labels = _labels(catalogs[cohort])
        rows: list[dict[str, Any]] = []
        for section in CELL_SECTIONS[cohort]:
            print(f"cells {cohort}/{section}", flush=True)
            metadata = _json_get(session, f"{api_base}/api/wsi/{cohort}/{section}")
            histology_dimension = max(max_dimension, 3_072)
            raw, raw_route = _render(
                api_base,
                metadata,
                "raw",
                max_dimension=histology_dimension,
                workers=workers,
            )
            boundaries, cell_route = _render(
                api_base,
                metadata,
                "cells",
                max_dimension=max(max_dimension, 6_000),
                workers=workers,
            )
            raw = _on_white(raw)
            # Cell contours are deliberately absent from low pyramid levels: a
            # one-pixel outline would otherwise imply detections where no cell
            # is resolvable. Sample the high-resolution contour layer first,
            # then reduce its validated red boundary ink to the bounded demo.
            # Geometry is unchanged.
            composed = _overlay(raw, boundaries, 0.62)
            panes = []
            for role, title, subtitle, image, route in (
                (
                    "histology",
                    "Native histology",
                    "Tissue-focused source context",
                    raw,
                    raw_route,
                ),
                (
                    "boundaries",
                    "Detected cell boundaries",
                    "Cell geometry over native histology",
                    composed,
                    cell_route,
                ),
            ):
                name = _asset_name(
                    ["cells", cohort, section, role],
                    [route.digest, raw_route.digest],
                )
                _write_webp(image, assets / name)
                panes.append(
                    {
                        "title": title,
                        "subtitle": subtitle,
                        "src": f"../static/previews/{name}",
                        "digest": route.digest,
                    }
                )
            rows.append(
                {
                    "section": section,
                    "section_label": labels.get(section, section),
                    "target": labels.get(section, section).lower(),
                    "target_label": labels.get(section, section),
                    "panes": panes,
                }
            )
        cohorts.append({"id": cohort, "examples": rows})
    manifest = {
        "schema_version": 1,
        "view": "cells",
        "title": "Cell boundaries",
        "subtitle": "Detected single-cell geometry on tissue-supported regions",
        "cohorts": cohorts,
    }
    _write_comparison_view(output / "cells", manifest)
    return manifest


def _recommended_model(catalog: dict[str, Any], target: str) -> str:
    recommended = catalog.get("recommended_protein_models") or {}
    target_row = recommended.get(target) or {}
    model_id = target_row.get("measured")
    if not model_id:
        raise ExportError(f"No measured presentation model for {target}")
    return str(model_id)


def _export_protein(
    api_base: str,
    output: Path,
    catalogs: dict[str, dict[str, Any]],
    session: requests.Session,
    *,
    max_dimension: int,
    workers: int,
) -> dict[str, Any]:
    cohorts: list[dict[str, Any]] = []
    assets = output / "static" / "previews"
    for cohort in COHORTS:
        labels = _labels(catalogs[cohort])
        rows: list[dict[str, Any]] = []
        for target, section in TARGET_SECTIONS[cohort].items():
            model_id = _recommended_model(catalogs[cohort], target)
            print(f"protein {cohort}/{section} {TARGET_LABELS[target]}", flush=True)
            encoded = quote(model_id, safe="")
            metadata = _json_get(
                session,
                f"{api_base}/api/wsi/{cohort}/{section}/protein/{encoded}",
            )
            comparison = metadata.get("protein_comparison") or {}
            if not comparison.get("ground_truth_available"):
                print(
                    f"skip protein {cohort}/{section} {TARGET_LABELS[target]}: "
                    "no local measured target map",
                    flush=True,
                )
                continue
            layer_specs = (
                (
                    str(comparison["predicted_layer"]),
                    f"Predicted {TARGET_LABELS[target]}",
                    "Single-cell expression · detected cell geometry",
                ),
                (
                    str(comparison["current_layer"]),
                    f"Observed {TARGET_LABELS[target]}",
                    "Tissue-supported target OD · 4 µm/px",
                ),
                (
                    str(comparison["comparison_layer"]),
                    "Absolute difference",
                    "Predicted versus observed target OD",
                ),
            )
            panes = []
            for role_index, (layer, title, subtitle) in enumerate(layer_specs):
                image, route = _render(
                    api_base,
                    metadata,
                    layer,
                    max_dimension=max_dimension,
                    workers=workers,
                    model_id=model_id,
                )
                image = _on_white(image)
                role = ("predicted", "observed", "difference")[role_index]
                name = _asset_name(
                    ["protein", cohort, section, target, role],
                    [route.digest, model_id],
                )
                _write_webp(image, assets / name)
                panes.append(
                    {
                        "title": title,
                        "subtitle": subtitle,
                        "src": f"../static/previews/{name}",
                        "digest": route.digest,
                    }
                )
            rows.append(
                {
                    "section": section,
                    "section_label": labels.get(section, TARGET_LABELS[target]),
                    "target": target,
                    "target_label": TARGET_LABELS[target],
                    "panes": panes,
                }
            )
        cohorts.append({"id": cohort, "examples": rows})
    manifest = {
        "schema_version": 1,
        "view": "protein",
        "title": "Protein expression",
        "subtitle": "Cell-resolved prediction alongside measured target expression",
        "cohorts": cohorts,
    }
    _write_comparison_view(output / "protein", manifest)
    return manifest


def _write_comparison_view(directory: Path, manifest: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (directory / "manifest-data.js").write_text(
        "globalThis.HISTOPIA_PRESENTATION = "
        + json.dumps(manifest, separators=(",", ":"))
        + ";\n",
        encoding="utf-8",
    )
    (directory / "index.html").write_text(COMPARISON_HTML, encoding="utf-8")


def _write_shell(output: Path) -> None:
    static = output / "static"
    static.mkdir(parents=True, exist_ok=True)
    (output / ".nojekyll").write_text("", encoding="utf-8")
    (output / "index.html").write_text(PORTAL_HTML, encoding="utf-8")
    (static / "presentation.css").write_text(PRESENTATION_CSS, encoding="utf-8")
    (static / "portal.js").write_text(PORTAL_JS, encoding="utf-8")
    (static / "comparison.js").write_text(COMPARISON_JS, encoding="utf-8")


def _copy_openseadragon(source: Path, output: Path) -> None:
    candidates = (
        source / "protein" / "openseadragon.min.js",
        source / "cells" / "openseadragon.min.js",
        source / "stain" / "openseadragon.min.js",
    )
    selected = next((path for path in candidates if path.is_file()), None)
    if selected is None:
        raise ExportError("OpenSeadragon was not found in the live review export")
    shutil.copy2(selected, output / "static" / "openseadragon.min.js")


def _validate_export(output: Path, max_bytes: int) -> dict[str, Any]:
    required = [
        "index.html",
        "static/presentation.css",
        "static/portal.js",
        "registration/index.html",
        "atlas/index.html",
        "stain/index.html",
        "cells/index.html",
        "topology/index.html",
        "protein/index.html",
        "protein-atlas/index.html",
    ]
    for name in required:
        if not (output / name).is_file():
            raise ExportError(f"Missing exported file: {name}")
    files = [path for path in output.rglob("*") if path.is_file()]
    if any(path.is_symlink() for path in output.rglob("*")):
        raise ExportError("The static export must not contain symlinks")
    total = sum(path.stat().st_size for path in files)
    if total > max_bytes:
        raise ExportError(
            f"Static export is {total:,} bytes, above the {max_bytes:,}-byte ceiling"
        )
    forbidden_text = (
        "/api/reviews",
        "placeholder=\"Reviewer\"",
        "placeholder=\"Review comment\"",
        "data-decision=",
    )
    bad: list[str] = []
    path_leaks: list[str] = []
    for path in files:
        if path.suffix.lower() not in {".html", ".js", ".json", ".css"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if any(token in text for token in forbidden_text):
            bad.append(path.relative_to(output).as_posix())
        if any(token in text for token in ("/home/exouser", "/media/volume", "yilab:")):
            path_leaks.append(path.relative_to(output).as_posix())
    if bad:
        raise ExportError(f"Write-oriented review controls remain in: {bad[:8]}")
    if path_leaks:
        raise ExportError(f"Local paths remain in public metadata: {path_leaks[:8]}")
    inventory = {
        "schema_version": 1,
        "title": "Histopia three-mouse presentation demo",
        "cohorts": list(COHORTS),
        "views": [
            "registration",
            "atlas",
            "stain",
            "cells",
            "topology",
            "protein",
            "protein-atlas",
        ],
        "read_only": True,
        "review_writes": False,
        "file_count": len(files),
        "total_bytes": total,
    }
    (output / "demo-manifest.json").write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return inventory


def export_demo(
    source: Path,
    output: Path,
    *,
    api_base: str,
    max_dimension: int,
    workers: int,
    max_bytes: int,
) -> dict[str, Any]:
    if output.exists():
        raise ExportError(f"Output already exists: {output}")
    output.mkdir(parents=True)
    session = requests.Session()
    session.headers["User-Agent"] = "histopia-static-presentation-export/1"
    api_base = api_base.rstrip("/")
    catalogs = _catalogs(api_base, session)
    _write_shell(output)
    _copy_openseadragon(source, output)
    _copy_static_views(source, output)
    _export_stain(
        api_base,
        output,
        catalogs,
        session,
        max_dimension=max_dimension,
        workers=workers,
    )
    _export_cells(
        api_base,
        output,
        catalogs,
        session,
        max_dimension=max_dimension,
        workers=workers,
    )
    _export_protein(
        api_base,
        output,
        catalogs,
        session,
        max_dimension=max_dimension,
        workers=workers,
    )
    return _validate_export(output, max_bytes)


PORTAL_HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="theme-color" content="#07111f"><link rel="icon" href="data:,">
<title>Histopia interactive atlas</title>
<link rel="stylesheet" href="static/presentation.css"></head>
<body class="portal"><header class="portal-header">
  <a class="brand" href="./" data-view="home"><span>H</span><b>Histopia</b><small>Interactive spatial histology</small></a>
  <nav aria-label="Presentation views">
    <button data-view="home">Home</button><button data-view="registration">Registration</button>
    <button data-view="atlas">3D atlas</button><button data-view="stain">Stain</button>
    <button data-view="cells">Cells</button><button data-view="topology">Topology</button>
    <button data-view="protein">Protein</button><button data-view="protein-atlas">Protein atlas</button>
  </nav></header>
<main class="portal-main">
  <section id="home" class="home-screen">
    <div class="hero"><p class="eyebrow">THREE-MOUSE INTERACTIVE DEMO</p>
      <h1>Serial histology becomes a connected cellular atlas.</h1>
      <p>Explore registration, tissue-supported stain measurements, detected cells, spatial topology, and predicted protein landscapes across mice 4312, 4630, and 6180.</p>
      <div class="mouse-chips"><span>4312 · 25 sections</span><span>4630 · 24 sections</span><span>6180 · 19 sections</span></div>
      <button class="primary" data-view="protein-atlas">Open the cellular protein atlas</button>
    </div>
    <div class="feature-grid">
      <article data-view="registration"><i>01</i><h2>Registration</h2><p>Tissue masks, section order, and aligned stack.</p></article>
      <article data-view="atlas"><i>02</i><h2>3D atlas</h2><p>Interactive serial-section histology and semantic regions.</p></article>
      <article data-view="stain"><i>03</i><h2>Stain extraction</h2><p>Tissue-supported target optical density at 4 µm/px.</p></article>
      <article data-view="cells"><i>04</i><h2>Cell boundaries</h2><p>Single-cell geometry over native histology.</p></article>
      <article data-view="topology"><i>05</i><h2>Spatial topology</h2><p>Connected tissue envelopes and semantic volumes.</p></article>
      <article data-view="protein"><i>06</i><h2>Protein expression</h2><p>Predicted and measured single-cell expression.</p></article>
      <article class="wide" data-view="protein-atlas"><i>07</i><h2>Cellular protein atlas</h2><p>Multi-protein cellular landscapes in 3D, orthogonal, and section views.</p></article>
    </div>
  </section>
  <iframe id="view-frame" title="Histopia interactive view" hidden></iframe>
</main><script src="static/portal.js"></script></body></html>
'''


COMPARISON_HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="theme-color" content="#07111f"><link rel="icon" href="data:,">
<title>Histopia presentation</title><link rel="stylesheet" href="../static/presentation.css"></head>
<body class="comparison"><header class="comparison-header">
  <div><strong id="view-title">Histopia</strong><small id="view-subtitle"></small></div>
  <div class="selectors"><label>Mouse<select id="cohort"></select></label>
    <label id="target-label">Marker<select id="target"></select></label>
    <label>Section<select id="section"></select></label>
    <button id="previous" aria-label="Previous example">‹</button><button id="next" aria-label="Next example">›</button>
    <button id="fit">Fit tissue</button></div>
</header><main><section id="summary"><b id="example-title"></b><span id="example-detail"></span></section>
<section id="panes"></section></main><output id="loading">Preparing tissue view…</output>
<script src="manifest-data.js"></script><script src="../static/openseadragon.min.js"></script>
<script src="../static/comparison.js"></script></body></html>
'''


PORTAL_JS = r'''"use strict";
const routes={registration:"registration/index.html",atlas:"atlas/index.html",stain:"stain/index.html",cells:"cells/index.html",topology:"topology/index.html",protein:"protein/index.html","protein-atlas":"protein-atlas/index.html"};
const home=document.querySelector("#home"),frame=document.querySelector("#view-frame");
function select(raw,push=true){const view=routes[raw]?raw:"home";document.body.dataset.view=view;
 document.querySelectorAll("[data-view]").forEach(node=>node.classList.toggle("active",node.dataset.view===view));
 if(view==="home"){home.hidden=false;frame.hidden=true;frame.removeAttribute("src");}else{home.hidden=true;frame.hidden=false;if(frame.getAttribute("src")!==routes[view])frame.src=routes[view];}
 if(push){const url=new URL(location.href);if(view==="home")url.searchParams.delete("view");else url.searchParams.set("view",view);history.pushState({view},"",url);}}
document.querySelectorAll("[data-view]").forEach(node=>node.addEventListener("click",event=>{event.preventDefault();select(node.dataset.view);}));
addEventListener("popstate",()=>select(new URL(location.href).searchParams.get("view"),false));
select(new URL(location.href).searchParams.get("view"),false);
'''


COMPARISON_JS = r'''"use strict";
const manifest=globalThis.HISTOPIA_PRESENTATION;if(!manifest?.cohorts?.length)throw new Error("Missing static presentation data");
const $=selector=>document.querySelector(selector),cohortSelect=$("#cohort"),targetSelect=$("#target"),sectionSelect=$("#section"),panes=$("#panes"),loading=$("#loading");
const viewers=[];let cohort=manifest.cohorts[0],example=cohort.examples[0],syncing=false,syncFrame=0,request=0;
$("#view-title").textContent=manifest.title;$("#view-subtitle").textContent=manifest.subtitle;document.title=`Histopia · ${manifest.title}`;
if(manifest.view==="cells"){$("#target-label").hidden=true;}
manifest.cohorts.forEach(row=>cohortSelect.add(new Option(row.id,row.id)));
function uniqueTargets(){return [...new Map(cohort.examples.map(row=>[row.target,row.target_label])).entries()];}
function setOptions(select,rows,value){select.replaceChildren(...rows.map(([id,label])=>new Option(label,id)));if(rows.some(([id])=>id===value))select.value=value;}
function currentRows(){return manifest.view==="cells"?cohort.examples:cohort.examples.filter(row=>row.target===targetSelect.value);}
function populate(preserveTarget=true){const prior=preserveTarget?targetSelect.value:"";setOptions(targetSelect,uniqueTargets(),prior);populateSections();}
function populateSections(preserve=true){const prior=preserve?sectionSelect.value:"";const rows=currentRows();setOptions(sectionSelect,rows.map(row=>[row.section,`${row.section} · ${row.section_label}`]),prior);example=rows.find(row=>row.section===sectionSelect.value)||rows[0];sectionSelect.value=example.section;load();}
function viewer(element,index){const instance=OpenSeadragon({element,showNavigator:index===0,navigatorPosition:"BOTTOM_RIGHT",showNavigationControl:false,animationTime:0,blendTime:0,immediateRender:true,constrainDuringPan:true,visibilityRatio:.15,minZoomImageRatio:.8,maxZoomPixelRatio:6,imageSmoothingEnabled:true});
 ["canvas-drag","canvas-scroll","canvas-pinch","canvas-double-click"].forEach(name=>instance.addHandler(name,()=>scheduleSynchronize(instance)));viewers.push(instance);return instance;}
function state(source){if(!source.world.getItemCount())return null;const item=source.world.getItemAt(0),center=item.viewportToImageCoordinates(source.viewport.getCenter(true));return{x:center.x/item.source.dimensions.x,y:center.y/item.source.dimensions.y,z:source.viewport.getZoom(true)/source.viewport.getHomeZoom()};}
function scheduleSynchronize(source){cancelAnimationFrame(syncFrame);syncFrame=requestAnimationFrame(()=>{syncFrame=requestAnimationFrame(()=>synchronize(source));});}
function synchronize(source){if(syncing)return;const next=state(source);if(!next)return;syncing=true;try{viewers.forEach(other=>{if(other===source||!other.world.getItemCount())return;const item=other.world.getItemAt(0),point=item.imageToViewportCoordinates(next.x*item.source.dimensions.x,next.y*item.source.dimensions.y);other.viewport.panTo(point,true);other.viewport.zoomTo(next.z*other.viewport.getHomeZoom(),null,true);other.viewport.applyConstraints(true);});}finally{syncing=false;}}
function open(instance,src,mine){return new Promise((resolve,reject)=>{const opened=()=>{cleanup();if(mine===request)resolve();else reject(new Error("superseded"));},failed=event=>{cleanup();reject(event);},cleanup=()=>{instance.removeHandler("open",opened);instance.removeHandler("open-failed",failed);};instance.addHandler("open",opened);instance.addHandler("open-failed",failed);instance.open({type:"image",url:src,buildPyramid:false});});}
async function load(){const mine=++request;loading.hidden=false;loading.textContent="Preparing tissue view…";example=currentRows().find(row=>row.section===sectionSelect.value)||currentRows()[0];
 $("#example-title").textContent=`Mouse ${cohort.id} · section ${example.section} · ${example.target_label}`;$("#example-detail").textContent=manifest.view==="stain"?"Native source coordinates · optical-density measurements at 4 µm/px":manifest.view==="cells"?"Tissue-focused cell geometry":"Shared expression scale within each protein comparison";
 viewers.forEach(instance=>instance.destroy());viewers.length=0;panes.replaceChildren(...example.panes.map((pane,index)=>{const article=document.createElement("article"),heading=document.createElement("h2"),title=document.createElement("span"),subtitle=document.createElement("small"),stage=document.createElement("div");title.textContent=pane.title;subtitle.textContent=pane.subtitle;heading.append(title,subtitle);stage.className="osd-stage";article.append(heading,stage);requestAnimationFrame(()=>viewer(stage,index));return article;}));
 await new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)));try{await Promise.all(viewers.map((instance,index)=>open(instance,example.panes[index].src,mine)));if(mine!==request)return;viewers.forEach(instance=>instance.viewport.goHome(true));synchronize(viewers[0]);loading.hidden=true;document.body.dataset.ready=`${cohort.id}-${example.section}`;}catch(error){if(mine===request){loading.textContent="This presentation image could not be loaded.";document.body.dataset.error=error.message||String(error);}}}
cohortSelect.onchange=()=>{cohort=manifest.cohorts.find(row=>row.id===cohortSelect.value)||manifest.cohorts[0];populate(false);};targetSelect.onchange=()=>populateSections(false);sectionSelect.onchange=load;
$("#previous").onclick=()=>step(-1);$("#next").onclick=()=>step(1);$("#fit").onclick=()=>{viewers.forEach(instance=>instance.viewport.goHome(true));};
function step(delta){const rows=cohort.examples,index=rows.findIndex(row=>row.section===example.section&&row.target===example.target),next=rows[(index+delta+rows.length)%rows.length];targetSelect.value=next.target;populateSections(false);sectionSelect.value=next.section;load();}
populate(false);
'''


PRESENTATION_CSS = r''':root{color-scheme:light;--ink:#102338;--muted:#647487;--line:#dce4e8;--paper:#f4f7f6;--card:#fff;--teal:#0d766e;--cyan:#27a89e;--navy:#071827;font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}*{box-sizing:border-box}[hidden]{display:none!important}html,body{margin:0;width:100%;height:100%;overflow:hidden}button,select{font:inherit}button{cursor:pointer}.portal{background:var(--paper);color:var(--ink);display:grid;grid-template-rows:auto 1fr}.portal-header{min-height:68px;background:#fff;border-bottom:1px solid var(--line);display:flex;align-items:center;padding:0 24px;gap:24px}.brand{display:flex;align-items:center;gap:9px;color:var(--ink);text-decoration:none;white-space:nowrap}.brand>span{display:grid;place-items:center;width:34px;height:34px;border-radius:10px;background:linear-gradient(145deg,var(--teal),#12a89a);color:#fff;font-weight:800}.brand b{font-size:18px}.brand small{color:var(--muted);font-size:11px;margin-left:2px}.portal nav{display:flex;align-items:center;gap:5px;overflow-x:auto;margin-left:auto}.portal nav button{border:0;background:transparent;color:#526476;padding:9px 11px;border-radius:9px;white-space:nowrap;font-weight:650;font-size:13px}.portal nav button:hover,.portal nav button.active{background:#e9f5f2;color:#08645d}.portal-main{min-height:0;position:relative}.home-screen{height:100%;overflow:auto;padding:clamp(26px,5vw,72px)}.hero{max-width:950px;margin:0 auto 44px}.eyebrow{margin:0 0 14px;color:var(--teal);font-size:12px;letter-spacing:.17em;font-weight:800}.hero h1{font-size:clamp(38px,6vw,76px);line-height:1.02;letter-spacing:-.052em;max-width:920px;margin:0;color:#092336}.hero>p:not(.eyebrow){font-size:clamp(16px,2vw,21px);line-height:1.55;max-width:790px;color:#53687a;margin:25px 0}.mouse-chips{display:flex;flex-wrap:wrap;gap:8px;margin:0 0 24px}.mouse-chips span{border:1px solid #bcd9d4;background:#f9fffd;color:#27655f;border-radius:999px;padding:7px 11px;font-size:12px;font-weight:700}.primary{border:0;border-radius:12px;color:#fff;background:var(--teal);padding:13px 18px;font-weight:750;box-shadow:0 8px 20px #0d766e28}.feature-grid{max-width:1180px;margin:auto;display:grid;grid-template-columns:repeat(4,1fr);gap:13px;padding-bottom:42px}.feature-grid article{min-height:160px;background:#fff;border:1px solid var(--line);border-radius:16px;padding:20px;cursor:pointer;transition:transform .16s,border-color .16s,box-shadow .16s}.feature-grid article:hover{transform:translateY(-3px);border-color:#94c9c2;box-shadow:0 12px 30px #15334912}.feature-grid article i{font-style:normal;color:var(--teal);font-size:12px;font-weight:850}.feature-grid h2{font-size:18px;margin:28px 0 7px}.feature-grid p{font-size:13px;line-height:1.5;color:var(--muted);margin:0}.feature-grid .wide{grid-column:span 2;background:linear-gradient(125deg,#082337,#0b4c51);color:#fff;border-color:transparent}.feature-grid .wide i,.feature-grid .wide p{color:#9ee4d9}#view-frame{width:100%;height:100%;border:0;background:#fff}.comparison{background:#edf2f2;color:var(--ink);display:grid;grid-template-rows:auto 1fr;position:relative}.comparison-header{min-height:74px;background:#fff;border-bottom:1px solid var(--line);display:flex;align-items:center;justify-content:space-between;padding:10px 18px;gap:18px}.comparison-header>div:first-child{display:grid;gap:3px}.comparison-header strong{font-size:19px}.comparison-header small{font-size:11px;color:var(--muted)}.selectors{display:flex;align-items:end;justify-content:flex-end;gap:7px;flex-wrap:wrap}.selectors label{display:grid;gap:3px;color:var(--muted);font-size:10px;font-weight:700}.selectors select,.selectors button{height:34px;border:1px solid #cbd7dc;border-radius:8px;background:#fff;color:var(--ink);padding:0 9px}.selectors button:hover{border-color:#6bb7ae;color:var(--teal)}.comparison main{min-height:0;padding:12px;display:grid;grid-template-rows:auto 1fr;gap:10px}.comparison #summary{background:#fff;border:1px solid var(--line);border-radius:10px;padding:9px 12px;display:flex;gap:12px;align-items:baseline;min-width:0}.comparison #summary b{font-size:13px}.comparison #summary span{font-size:11px;color:var(--muted);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.comparison #panes{min-height:0;display:grid;grid-template-columns:repeat(var(--pane-count,3),minmax(0,1fr));gap:10px}.comparison #panes:has(article:nth-child(4)){--pane-count:2;grid-template-rows:repeat(2,minmax(0,1fr))}.comparison #panes:has(article:nth-child(2):last-child){--pane-count:2}.comparison article{min-width:0;min-height:0;background:#fff;border:1px solid var(--line);border-radius:12px;overflow:hidden;display:grid;grid-template-rows:auto 1fr;box-shadow:0 3px 12px #16333c0a}.comparison h2{margin:0;padding:9px 11px;border-bottom:1px solid #e7edef;display:grid;gap:2px}.comparison h2 span{font-size:12px}.comparison h2 small{font-size:10px;color:var(--muted);font-weight:500}.osd-stage{min-height:0;background:#f8fafc}.comparison #loading{position:fixed;inset:50% auto auto 50%;translate:-50% -50%;padding:11px 16px;border-radius:10px;background:#071827e8;color:#fff;font-size:12px;z-index:5}.comparison #loading[hidden]{display:none}@media(max-width:1000px){.portal-header{padding:8px 12px;display:grid;gap:5px}.portal-header .brand small{display:none}.portal nav{width:100%;margin:0}.portal{grid-template-rows:auto 1fr}.feature-grid{grid-template-columns:repeat(2,1fr)}.comparison-header{align-items:flex-start}.comparison-header>div:first-child{display:none}.comparison #panes,.comparison #panes:has(article:nth-child(4)){grid-template-columns:1fr;grid-template-rows:none;overflow:auto}.comparison article{min-height:330px}.comparison main{overflow:auto}.comparison #summary{position:sticky;top:0;z-index:2}.selectors{justify-content:flex-start}}@media(max-width:560px){.home-screen{padding:28px 15px}.hero h1{font-size:42px}.feature-grid{grid-template-columns:1fr}.feature-grid .wide{grid-column:span 1}.comparison-header{padding:8px;display:block;overflow-x:auto}.selectors{flex-wrap:nowrap;min-width:max-content}.comparison main{padding:7px}.comparison article{min-height:300px}.brand{justify-content:center}.portal nav button{font-size:12px;padding:8px 9px}}
'''


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Live review export directory")
    parser.add_argument("output", type=Path, help="New static-site output directory")
    parser.add_argument("--api-base", default="http://127.0.0.1:8765")
    parser.add_argument("--max-dimension", type=int, default=2304)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-bytes", type=int, default=943_718_400)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    inventory = export_demo(
        args.source.resolve(),
        args.output.resolve(),
        api_base=args.api_base,
        max_dimension=args.max_dimension,
        workers=args.workers,
        max_bytes=args.max_bytes,
    )
    print(json.dumps(inventory, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
