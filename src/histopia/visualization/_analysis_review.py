"""Analysis-first navigation over already published scientific evidence.

This is a presentation index, never a reconstruction eligibility or approval
decision. Catalogs pass their existing gates before entering this module.
"""

from __future__ import annotations

import html
import json
import posixpath
import shutil
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from histopia._atomic import write_json_atomic, write_text_atomic
from histopia.study._manifest import fingerprint

WORKSPACES = [
    {
        "id": "registration",
        "label": "Registration & 3D",
        "icon": "layers",
        "modes": [
            ["stacks", "3D stacks"],
            ["atlas", "Semantic stack"],
            ["alignment", "Alignment"],
            ["volume", "Continuous volume"],
        ],
    },
    {
        "id": "stain",
        "label": "Stain measurement",
        "icon": "drop",
        "modes": [["measurements", "Measurements"], ["qc", "Measurement QC"]],
    },
    {
        "id": "cells",
        "label": "Cells & annotation",
        "icon": "cells",
        "modes": [
            ["boundaries", "Boundaries"],
            ["annotations", "Annotations"],
            ["features", "Cell features"],
        ],
    },
    {
        "id": "protein",
        "label": "Protein prediction",
        "icon": "signal",
        "modes": [
            ["predictions", "Predictions"],
            ["transfer", "Section transfer"],
            ["maps", "Cell maps"],
            ["atlas", "Cell atlas"],
            ["performance", "Performance"],
        ],
    },
    {
        "id": "spatial",
        "label": "Spatial analysis",
        "icon": "regions",
        "modes": [
            ["regions", "Tissue regions"],
            ["neighborhoods", "Neighborhoods"],
            ["stability", "Stability"],
        ],
    },
    {
        "id": "data",
        "label": "Data & metadata",
        "icon": "table",
        "utility": True,
        "modes": [["inventory", "All scans"], ["images", "Input images"]],
    },
    {
        "id": "figures",
        "label": "Figures",
        "icon": "figure",
        "utility": True,
        "modes": [["figures", "Study figures"]],
    },
    {
        "id": "reviews",
        "label": "Review history",
        "icon": "check",
        "utility": True,
        "modes": [["reviews", "Recorded decisions"]],
    },
]

# Explicit taxonomy: unknown stages stop publication instead of disappearing.
STAGES = {
    "Reconstruction": ("registration", "stacks"),
    "Registration": ("registration", "alignment"),
    "Native stain measurements": ("stain", "measurements"),
    "Stain extraction & UNI2-h": ("stain", "measurements"),
    "Measured targets · adaptive 64 µm": ("stain", "measurements"),
    "Assembled cell maps": ("cells", "boundaries"),
    "Cell segmentation · field pilots": ("cells", "boundaries"),
    "Cell boundary pilots": ("cells", "boundaries"),
    "Whole-slide tile inference": ("cells", "boundaries"),
    "Cell morphology features": ("cells", "features"),
    "Protein vector transfer": ("protein", "transfer"),
    "Cell protein hypotheses": ("protein", "maps"),
    "Expanded protein prediction": ("protein", "predictions"),
    "Protein prediction · adaptive 64 µm": ("protein", "performance"),
    "Protein prediction · test": ("protein", "predictions"),
    "Legacy protein diagnostics": ("protein", "performance"),
    "Protein neighborhoods": ("spatial", "neighborhoods"),
    "Morphology neighborhoods": ("spatial", "neighborhoods"),
    "Protein-neighborhood stability": ("spatial", "stability"),
    "Tissue-region proteins": ("spatial", "regions"),
    "Figure": ("figures", "figures"),
    "Geometry & coverage": ("data", "images"),
    "Native input QC": ("data", "images"),
    "Repaired tissue support": ("data", "images"),
}

TOOLS = {
    "registration": ("registration", "alignment", "Masks, order & alignment"),
    "atlas": ("registration", "atlas", "Semantic regions & section connections"),
    "topology": ("registration", "volume", "Connected tissue volume"),
    "stain": ("stain", "qc", "Native stain measurement & QC"),
    "cells": ("cells", "boundaries", "Whole-slide boundaries"),
    "selected-cells": ("cells", "boundaries", "Reviewed section subset"),
    "annotations": ("cells", "annotations", "Pathology annotations"),
    "protein": ("protein", "maps", "Interactive cell protein maps"),
    "protein-atlas": ("protein", "atlas", "Cellular protein atlas"),
    "organ-metadata": ("data", "inventory", "All internal scans"),
    "study-figure": ("figures", "figures", "Workflow & current evidence"),
    "decisions": ("reviews", "reviews", "Scientific decisions"),
    "methods": ("data", "inventory", "Methods reference"),
}


def _asset(prefix: str, href: str) -> str:
    if href.startswith(("/", "https://", "http://", "#")):
        return href
    return posixpath.normpath(posixpath.join(prefix, href))


def compile_review_navigation(
    workflow: Mapping,
    catalogs: Mapping[str, Mapping],
    viewer_manifests: Mapping[str, Mapping] | None = None,
) -> dict:
    """Compile canonical destinations, retaining original scientific identities."""
    viewer_manifests = viewer_manifests or {}
    entries, sources, mappings = [], {}, []
    contexts: dict[tuple[str, str, str], str] = {}
    catalog_fingerprints = {}
    for name, catalog in catalogs.items():
        bound = fingerprint({k: v for k, v in catalog.items() if k != "fingerprint"})
        if catalog.get("fingerprint") != bound:
            raise ValueError(f"Changed published catalog: {name}")
        policy = catalog.get("eligibility_policy")
        benchmark = catalog.get("benchmark_policy")
        internal = catalog.get("input_review_policy")
        if not (policy or benchmark or internal):
            raise ValueError("Navigation requires an independently gated catalog")
        catalog_fingerprints[name] = bound
        for source in catalog["sources"]:
            if (
                source["id"] in sources
                and sources[source["id"]]["external"] != source["external"]
            ):
                raise ValueError("Conflicting source domains")
            sources[source["id"]] = dict(source)
        for row in catalog["datasets"]:
            if row["stage"] not in STAGES:
                raise ValueError(f"Unmapped analysis stage: {row['stage']}")
            if policy and not (
                row.get("reconstruction_id") and row.get("eligibility_fingerprint")
            ):
                raise ValueError("Unresolved reconstruction identity")
            if sources[row["source_id"]]["external"] and not (policy or benchmark):
                raise ValueError("External evidence lacks its publication gate")
            analysis, mode = STAGES[row["stage"]]
            entry = dict(row)
            entry.update(
                analysis=analysis,
                mode=mode,
                catalog=name,
                renderer="media",
                external=sources[row["source_id"]]["external"],
                previous=row["stage"] == "Legacy protein diagnostics",
                priority=20,
            )
            for key in ("media", "downloads", "links"):
                entry[key] = [
                    {**asset, "href": _asset(name, asset["href"])}
                    for asset in row.get(key, [])
                ]
            if row.get("analysis_view") or row.get("reconstruction_view"):
                entry.update(
                    renderer="regions" if row.get("analysis_view") else "stack",
                    href=_asset(
                        name, row.get("analysis_view") or row["reconstruction_view"]
                    ),
                )
            if policy:
                contexts[row["source_id"], row["organ"], str(row["subject_id"])] = row[
                    "reconstruction_id"
                ]
            if row["stage"] == "Reconstruction":
                entry["priority"] = 0
            if row["stage"] == "Expanded protein prediction":
                entry["priority"] = 0
            if row["stage"] == "Protein prediction · test":
                entry["related_modes"] = ["performance"]
                entry["mode_fields"] = {
                    "performance": next(
                        (
                            i
                            for i, m in enumerate(row.get("media", []))
                            if "holdout" in m.get("label", "").lower()
                        ),
                        0,
                    )
                }
            if row["stage"] == "Assembled cell maps":
                entry["priority"] = 5
            entry["scope_label"] = (
                "2D counterstain benchmark"
                if benchmark
                else "Provisional · 2D"
                if internal
                else "Observed sections"
                if entry["renderer"] == "stack"
                else "Exploratory result"
            )
            entries.append(entry)
            mappings.append(
                {
                    "catalog": name,
                    "id": row["id"],
                    "fingerprint": row["fingerprint"],
                    "analysis": analysis,
                    "mode": mode,
                    "previous": entry["previous"],
                }
            )
    mapped_tools = []
    for tab in workflow.get("tabs", []):
        tool = tab["id"]
        if tool in {"data-catalog", "external-validation", "internal-inputs"}:
            mapped_tools.append(
                {"old_view": tool, "destination": "record's canonical analysis"}
            )
            continue
        if tool == "serial-stack":
            # Retired fluorescence entry: never resurrect it through compatibility.
            mapped_tools.append({"old_view": tool, "destination": "unavailable"})
            continue
        if tool not in TOOLS:
            raise ValueError(f"Unmapped review tool: {tool}")
        analysis, mode, label = TOOLS[tool]
        mapped_tools.append({"old_view": tool, "analysis": analysis, "mode": mode})
        scope = tab.get("scope", {})
        source = (scope.get("sources") or ["internal"])[0]
        organ = (scope.get("organs") or ["unspecified"])[0]
        metadata = viewer_manifests.get(tool, {})
        cohort_rows = (
            metadata.get("cohorts")
            or metadata.get("reviews")
            or metadata.get("mice")
            or []
        )
        if not isinstance(cohort_rows, list):
            cohort_rows = []
        cohorts = {
            str(row["id"]): row
            for row in cohort_rows
            if isinstance(row, dict) and "id" in row
        }
        subjects = scope.get("subjects") or list(cohorts)
        if tool == "decisions" and not subjects:
            registration = viewer_manifests.get("registration", {})
            subjects = [str(r["id"]) for r in registration.get("reviews", [])]
        if tool in {"organ-metadata", "study-figure", "methods"}:
            subjects = [""]
        if tool == "organ-metadata":
            organ = ""
        for subject in subjects:
            identity = (source, organ, str(subject))
            if (
                tool in {"atlas", "topology", "protein-atlas"}
                and catalogs
                and identity not in contexts
            ):
                continue
            sources.setdefault(
                source,
                {
                    "id": source,
                    "label": "Our data" if source == "internal" else source.upper(),
                    "external": False,
                    "species": "",
                    "description": "",
                },
            )
            cohort = cohorts.get(str(subject), {})
            entry = dict(
                id=f"viewer-{tool}-{subject or 'all'}",
                analysis=analysis,
                mode=mode,
                source_id=source,
                organ=organ,
                subject_id=str(subject),
                title=label,
                short_title=label,
                renderer=tool,
                href=tab["href"],
                external=False,
                previous=False,
                priority=10,
                legacy_view=tool,
                fingerprint=cohort.get("fingerprint") or fingerprint(cohort or tab),
                scope_label="Interactive review",
                scientific_approval=False,
                sections=cohort.get("sections", []),
                media=[],
                downloads=[],
                links=[],
            )
            if identity in contexts:
                entry["reconstruction_id"] = contexts[identity]
            if tool in {"cells", "selected-cells"}:
                entry["priority"] = 0
                entry["scope_label"] = "Provisional boundaries" + (
                    " · selected sections" if tool == "selected-cells" else ""
                )
            if tool == "study-figure":
                entry["priority"] = 0
                entry["subject_id"] = (scope.get("subjects") or [""])[0]
            if tool == "organ-metadata":
                entry["global_context"] = True
            if tool == "protein":
                entry["scope_label"] = "Cell protein predictions"
            if tool in {"topology", "protein-atlas"}:
                spacing = cohort.get("section_thickness_um")
                entry["scope_label"] = (
                    f"Assumed Z · {spacing:g} µm"
                    if spacing
                    else "See view-specific Z model"
                )
            entries.append(entry)
    ids = [e["id"] for e in entries]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate canonical result identity")
    modes = {(w["id"], m[0]) for w in WORKSPACES for m in w["modes"]}
    assert all((e["analysis"], e["mode"]) in modes for e in entries)
    payload = dict(
        schema_version="analysis-review-1",
        workspaces=WORKSPACES,
        sources=list(sources.values()),
        entries=entries,
        catalog_fingerprints=catalog_fingerprints,
        legacy_tools=mapped_tools,
        coverage={
            "catalog_records": len(mappings),
            "mapped_records": len(mappings),
            "tools": len(mapped_tools),
            "by_catalog": dict(Counter(m["catalog"] for m in mappings)),
        },
        mappings=mappings,
    )
    payload["fingerprint"] = fingerprint(payload)
    return payload


def _fallback_href(entry: Mapping) -> str:
    """Keep specimen identity when JavaScript is unavailable."""
    if not entry.get("href"):
        return (entry.get("media") or entry.get("downloads") or [{}])[0].get(
            "href", "#"
        )
    url = urlsplit(entry["href"])
    query = dict(parse_qsl(url.query))
    if not entry.get("global_context"):
        query.update(source=entry["source_id"], organ=entry["organ"])
        if entry["subject_id"]:
            query.update(
                subject=entry["subject_id"],
                mouse=entry["subject_id"],
                cohort=entry["subject_id"],
            )
    return urlunsplit(url._replace(query=urlencode(query)))


def build_analysis_review(
    review_dir: Path | str,
    output_dir: Path | str,
    *,
    catalog_dir: Path | str | None = None,
    workflow: Mapping | None = None,
) -> Path:
    """Write a portable shell; renderer links resolve from the review root."""
    review, output = Path(review_dir), Path(output_dir)
    catalogs_root = Path(catalog_dir) if catalog_dir else review
    workflow = workflow or json.loads((review / "manifest.json").read_text())
    catalogs = {}
    for name in ("data-catalog", "internal-inputs", "external-validation"):
        path = catalogs_root / name / "catalog.json"
        if path.exists():
            catalogs[name] = json.loads(path.read_text())
    metadata = {}
    for tab in workflow.get("tabs", []):
        href = str(tab["href"])
        # Root URLs are anchored at the publication parent, not the filesystem root.
        folder = (
            review.parent / href.lstrip("/") if href.startswith("/") else review / href
        )
        path = folder.parent / "manifest.json"
        if path.is_file():
            metadata[tab["id"]] = json.loads(path.read_text())
    data = compile_review_navigation(workflow, catalogs, metadata)
    inventory = catalogs_root / "organ-metadata" / "inventory.json"
    if inventory.exists():
        data["coverage"]["inventory_scans"] = len(
            json.loads(inventory.read_text())["rows"]
        )
    output.mkdir(parents=True, exist_ok=True)
    assets = Path(__file__).with_name("_analysis_review_assets")
    data["presentation_fingerprint"] = fingerprint(
        {p.name: p.read_text() for p in sorted(assets.iterdir()) if p.is_file()}
    )
    data["fingerprint"] = fingerprint(
        {k: v for k, v in data.items() if k != "fingerprint"}
    )
    for name in ("workspace.css", "workspace.js"):
        shutil.copyfile(assets / name, output / name)
    fallback = "".join(
        '<li><a href="'
        + html.escape(_fallback_href(e), quote=True)
        + '">'
        + html.escape(
            " · ".join(
                [
                    e["source_id"],
                    e["organ"],
                    e["subject_id"],
                    e.get("short_title") or e["title"],
                ]
            )
        )
        + "</a></li>"
        for e in data["entries"]
    )
    page = (
        (assets / "index.html")
        .read_text()
        .replace("<!-- FALLBACK -->", "<ul>" + fallback + "</ul>")
    )
    for asset in ("workspace.css", "workspace.js", "workspace-data.js"):
        page = page.replace(
            '"' + asset + '"', '"' + asset + "?v=" + data["fingerprint"][:20] + '"'
        )
    write_text_atomic(output / "index.html", page)
    write_json_atomic(output / "workspace-index.json", data)
    write_text_atomic(
        output / "workspace-data.js",
        "globalThis.HISTOPIA_ANALYSIS_REVIEW="
        + json.dumps(data, separators=(",", ":"), allow_nan=False).replace(
            "<", "\\u003c"
        )
        + ";\n",
    )
    write_json_atomic(
        output / "navigation-coverage.json",
        {
            "fingerprint": data["fingerprint"],
            **data["coverage"],
            "tools": data["legacy_tools"],
            "records": data["mappings"],
        },
    )
    return output / "index.html"
