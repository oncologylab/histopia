"""Editable workflow-plus-results figure built only from fingerprinted evidence."""

from __future__ import annotations

import html
import json
import math
import textwrap
from pathlib import Path
from urllib.parse import urlencode

from histopia.study._manifest import file_sha256, validate_study_manifest

PANEL_TITLES = {
    "A": "3D Tissue Registration & Reconstruction",
    "B": "Cell Segmentation & Registration",
    "C": "Feature Extraction & Model Training",
    "D": "Whole-tissue Protein Expression Predictions",
    "E": "Whole-tissue Cell Annotation & Interpolation",
    "F": "3D Embedding- and Cell-based Niche Discovery",
}
PANEL_WORKFLOWS = {
    "A": "Serial sections → registered tissue → spatial stack",
    "B": "Native CPSAM instances → registered cell positions",
    "C": "H&E → cell embeddings → protein model → held-out mouse",
    "D": "Direct prediction / section transfer → measured stain QC",
    "E": "Broad class probabilities → lab review → supported interpolation",
    "F": "Embedding regions / cell neighborhoods → spatial controls",
}
STATUS_COLORS = {
    "approved": "#245989",
    "exploratory": "#8a6010",
    "pending": "#727b89",
    "rejected": "#a8343d",
}


def build_study_figure(
    manifest: dict[str, object],
    spec: dict[str, object],
    output_dir: Path | str,
) -> Path:
    """Export SVG (editable text), PDF, presentation PNG, captions and web view.

    Scientific micrographs remain embedded rasters; headings, captions, panels
    and scale bars are vectors. Each image must identify its current file hash,
    upstream evidence and displayed physical width. Pending analyses have
    explicit labelled placeholders. Nothing here promotes an upstream result.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    study = validate_study_manifest(manifest)
    panels = spec.get("panels", {})
    if set(panels) != set(PANEL_TITLES):
        raise ValueError("figure must specify all six panels A–F")
    evidence = {row["id"]: row for row in study.get("evidence", [])}
    assets = []
    for letter, panel in panels.items():
        if panel.get("status") not in STATUS_COLORS or not panel.get("caption"):
            raise ValueError("each panel needs an evidence status and caption")
        if panel["status"] in {"pending", "rejected"} and not panel.get(
            "pending_reason"
        ):
            raise ValueError("pending/rejected panels require a visible reason")
        for asset in panel.get("assets", []):
            source = Path(asset["path"])
            if file_sha256(source) != asset["sha256"]:
                raise ValueError("figure image hash mismatch")
            item = evidence.get(asset["evidence_id"])
            if item is None:
                raise ValueError("figure image lacks study evidence")
            if panel["status"] == "approved" and item["status"] != "approved":
                raise ValueError("unapproved evidence cannot enter an approved panel")
            if not asset.get("label"):
                raise ValueError("figure images require a source/status label")
            physical = asset.get("physical_width_um")
            if physical is not None and (
                not isinstance(physical, (int, float))
                or not math.isfinite(physical)
                or physical <= 0
            ):
                raise ValueError("scale bars require a positive physical image width")
            assets.append(
                {
                    "panel": letter,
                    "sha256": asset["sha256"],
                    "label": asset["label"],
                    "evidence_id": asset["evidence_id"],
                    "physical_width_um": physical,
                    "coordinate_space": asset.get("coordinate_space", "not_applicable"),
                }
            )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "svg.fonttype": "none",
            "svg.hashsalt": study["fingerprint"],
            "pdf.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(18, 11.8), facecolor="white")
    fig.text(0.03, 0.962, "HISTOPIA", fontsize=23, color="#152b43", weight="bold")
    fig.text(
        0.18,
        0.964,
        spec.get("subtitle", "Serial histology · workflow and current evidence"),
        fontsize=15,
        color="#425a73",
    )
    for i, (letter, title) in enumerate(PANEL_TITLES.items()):
        panel = panels[letter]
        x = 0.025 + (i % 3) * 0.327
        y = 0.505 if i < 3 else 0.07
        width, height = 0.308, 0.4
        ax = fig.add_axes([x, y, width, height])
        ax.axis("off")
        ax.add_patch(
            plt.Rectangle(
                (0, 0),
                1,
                1,
                transform=ax.transAxes,
                facecolor="#f8fafc",
                edgecolor="#ced9e5",
                linewidth=0.8,
            )
        )
        ax.text(
            0.025, 0.95, letter, fontsize=22, weight="bold", color="#245989", va="top"
        )
        ax.text(
            0.11,
            0.95,
            textwrap.fill(title, 37),
            fontsize=13,
            weight="bold",
            color="#152b43",
            va="top",
            linespacing=1.2,
        )
        ax.text(
            0.025,
            0.80,
            textwrap.fill(PANEL_WORKFLOWS[letter], 62),
            fontsize=10,
            color="#425a73",
            va="top",
        )
        color = STATUS_COLORS[panel["status"]]
        ax.text(
            0.025,
            0.015,
            panel.get("status_label", panel["status"].capitalize()),
            fontsize=9,
            color=color,
            va="bottom",
            weight="bold",
        )
        images = panel.get("assets", [])
        if images:
            per_width = 0.93 / len(images)
            for j, asset in enumerate(images):
                iax = fig.add_axes(
                    [
                        x + width * (0.035 + j * per_width),
                        y + height * 0.32,
                        width * (per_width - 0.025),
                        height * 0.34,
                    ]
                )
                with Image.open(asset["path"]) as im:
                    iax.imshow(im.convert("RGB"))
                iax.axis("off")
                iax.set_title(
                    textwrap.fill(asset["label"], 28),
                    fontsize=8.5,
                    color="#152b43",
                    pad=4,
                )
                physical = asset.get("physical_width_um")
                if physical:
                    bar = 10 ** math.floor(math.log10(physical / 3))
                    length = bar / physical
                    iax.plot(
                        [0.06, 0.06 + length],
                        [0.075, 0.075],
                        color="black",
                        lw=4,
                        transform=iax.transAxes,
                    )
                    iax.plot(
                        [0.06, 0.06 + length],
                        [0.075, 0.075],
                        color="white",
                        lw=2,
                        transform=iax.transAxes,
                    )
                    iax.text(
                        0.06,
                        0.11,
                        f"{bar:g} µm",
                        color="black",
                        fontsize=8,
                        transform=iax.transAxes,
                        bbox={
                            "facecolor": "white",
                            "alpha": 0.85,
                            "edgecolor": "none",
                            "pad": 1,
                        },
                    )
        else:
            ax.add_patch(
                plt.Rectangle(
                    (0.035, 0.33),
                    0.93,
                    0.38,
                    facecolor="#eef2f7",
                    edgecolor="#b6c3d1",
                    linestyle="--",
                )
            )
            ax.text(
                0.5,
                0.55,
                "PENDING EVIDENCE",
                ha="center",
                fontsize=13,
                color="#586b80",
                weight="bold",
            )
            ax.text(
                0.5,
                0.44,
                textwrap.fill(panel.get("pending_reason", "Analysis is not ready"), 51),
                ha="center",
                va="center",
                fontsize=10,
                color="#425a73",
            )
        ax.text(
            0.035,
            0.27,
            textwrap.fill(panel["caption"], 69),
            fontsize=9.5,
            va="top",
            color="#152b43",
            linespacing=1.4,
        )
    fig.text(
        0.03,
        0.025,
        "Observed images / measured stains: source evidence  ·  "
        "Predicted / transferred: inferred  ·  Missing support stays missing",
        fontsize=10,
        color="#425a73",
    )
    fig.text(0.97, 0.025, study["study_id"], ha="right", fontsize=9, color="#425a73")
    for suffix, kwargs in (("svg", {}), ("pdf", {}), ("png", {"dpi": 180})):
        metadata = {
            "svg": {"Date": None, "Creator": "Histopia"},
            "pdf": {"CreationDate": None, "ModDate": None, "Creator": "Histopia"},
            "png": {"Software": "Histopia"},
        }[suffix]
        fig.savefig(out / f"histopia-six-panel.{suffix}", metadata=metadata, **kwargs)
    plt.close(fig)
    captions = "\n\n".join(
        f"{letter}. {PANEL_TITLES[letter]}\n{panels[letter]['caption']}\n"
        f"Evidence: {panels[letter].get('status_label', panels[letter]['status'])}."
        + (
            f"\nPending: {panels[letter]['pending_reason']}"
            if panels[letter].get("pending_reason")
            else ""
        )
        for letter in PANEL_TITLES
    )
    (out / "captions.md").write_text(captions + "\n")
    provenance = {
        "schema_version": 1,
        "study_id": study["study_id"],
        "study_fingerprint": study["fingerprint"],
        "assets": assets,
        "source_evidence": [
            {
                key: row[key]
                for key in (
                    "id",
                    "status",
                    "approval_fingerprint",
                    "result_sha256",
                    "scope",
                )
                if key in row
            }
            for row in evidence.values()
            if row["id"] in {asset["evidence_id"] for asset in assets}
        ],
        "panels": {
            k: {field: value for field, value in v.items() if field != "assets"}
            for k, v in panels.items()
        },
        "exports": {
            f"histopia-six-panel.{suffix}": file_sha256(
                out / f"histopia-six-panel.{suffix}"
            )
            for suffix in ("svg", "pdf", "png")
        },
        "limitations": spec.get("limitations", []),
        "selected_region_id": spec.get("selected_region_id"),
        "analysis_fingerprints": spec.get("analysis_fingerprints", {}),
        "figure_claim": "workflow_plus_scoped_evidence",
    }
    (out / "figure-provenance.json").write_text(
        json.dumps(provenance, indent=2, allow_nan=False) + "\n"
    )
    _write_figure_page(out, study, panels, spec.get("selected_region_id"))
    return out / "histopia-six-panel.svg"


def _write_figure_page(out, study, panels, selected_region_id):
    rows = "".join(
        f"<tr><td>{letter}</td><td>{html.escape(PANEL_TITLES[letter])}</td>"
        f"<td>{html.escape(panel.get('status_label', panel['status']))}</td></tr>"
        for letter, panel in panels.items()
    )
    region_link = ""
    if (out / "regions" / "index.html").is_file():
        query = (
            "?" + urlencode({"region": selected_region_id})
            if selected_region_id
            else ""
        )
        region_link = (
            f'<a href="regions/index.html{html.escape(query, quote=True)}">'
            "Select a tissue region</a>"
        )
    (out / "index.html").write_text(
        """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Histopia · six-panel study figure</title><style>
body{font:16px/1.5 Arial,sans-serif;color:#152b43;background:#f4f7fb;margin:0}
main{max-width:1500px;margin:auto;padding:1.3rem}h1{font-size:1.5rem}
nav{display:flex;gap:1.2rem;flex-wrap:wrap}a{color:#245989}
img{width:100%;height:auto;margin:1rem 0;background:white}
table{width:100%;border-collapse:collapse;background:white}
td,th{padding:.7rem;text-align:left;border-bottom:1px solid #cbd5e1}
summary{cursor:pointer}p{max-width:80ch}</style></head><body><main>
<h1>Histopia · workflow and current evidence</h1><nav>
<a href="histopia-six-panel.svg" download>Editable SVG</a>
<a href="histopia-six-panel.pdf" download>PDF</a>
<a href="histopia-six-panel.png" download>PNG</a><a href="captions.md">Captions</a>
<a href="figure-provenance.json">Provenance</a>"""
        + region_link
        + """</nav><img src="histopia-six-panel.svg"
alt="Six-panel Histopia workflow with scoped tissue examples and pending evidence">
<details open><summary>Evidence scope</summary>
<p>Observed images and approved processing are shown within their recorded scope.
Protein prediction, cell-type accuracy and biological niches require their own
validation. This figure does not complete the remaining mouse repairs.</p>
<table><thead><tr><th>Panel</th><th>Analysis</th><th>Current evidence</th></tr>
</thead><tbody>"""
        + rows
        + "</tbody></table></details><p>Study: "
        + html.escape(study["study_id"])
        + "</p></main></body></html>"
    )
