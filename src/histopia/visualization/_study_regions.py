"""Portable single-region outline explorer with exact region/summary identities."""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import numpy as np

from histopia.study._regions import TissueRegions


def region_outline_path(labels: np.ndarray, index: int) -> str:
    """SVG unit-grid edges, including holes; no filled semantic overlay."""
    selected = labels == index
    padded = np.pad(selected, 1)
    edges = []
    for dy, dx, offset in (
        (-1, 0, (0, 0, 1, 0)),
        (1, 0, (0, 1, 1, 1)),
        (0, -1, (0, 0, 0, 1)),
        (0, 1, (1, 0, 1, 1)),
    ):
        neighbor = padded[
            1 + dy : 1 + dy + labels.shape[0], 1 + dx : 1 + dx + labels.shape[1]
        ]
        rr, cc = np.nonzero(selected & ~neighbor)
        x1, y1, x2, y2 = offset
        edges.extend(
            f"M{c + x1},{r + y1}L{c + x2},{r + y2}" for r, c in zip(rr, cc, strict=True)
        )
    return "".join(edges)


def build_region_view(
    regions: TissueRegions,
    output_dir: Path | str,
    *,
    title: str,
    summary_rows: list[dict[str, object]] | None = None,
    background_png: Path | str | None = None,
    palette=None,
    study_fingerprint: str,
    summary_scope: str = "No expression summaries available",
    back_href: str = "../index.html",
    back_label: str = "Study figure",
) -> Path:
    """Export self-contained data and SVG; works offline and behind path proxies.

    Background must be resampled into the exact semantic raster frame by the
    caller. Selection persists as ``?region=<immutable ID>``; exports carry the
    same ID and evidence fingerprint. No web service or WebGL is required.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    palette = palette or ["#d73027", "#1a9850", "#4575b4", "#b68c00", "#984ea3"]
    rows = summary_rows or []
    identifiers = {r["region_id"] for r in regions.regions}
    if any(
        row.get("region_id") not in identifiers
        or row.get("region_fingerprint") != regions.fingerprint
        for row in rows
    ):
        raise ValueError("summary rows must bind to these exact regions")
    if not study_fingerprint:
        raise ValueError("region view requires a study fingerprint")
    h, w = regions.labels.shape
    background = None
    if background_png:
        background = (
            "data:image/png;base64,"
            + base64.b64encode(Path(background_png).read_bytes()).decode()
        )
    data = {
        "title": title,
        "width": w,
        "height": h,
        "pixel_size_um": regions.pixel_size_um,
        "pixel_size_um_xy": regions.pixel_size_um_xy or (regions.pixel_size_um,) * 2,
        "origin_um_xy": regions.origin_um_xy,
        "coordinate_units": "um",
        "fingerprint": regions.fingerprint,
        "study_fingerprint": study_fingerprint,
        "background": background,
        "summary_scope": summary_scope,
        "summaries": rows,
        "regions": [
            {
                **row,
                "path": region_outline_path(regions.labels, row["index"]),
                "color": palette[row["semantic_class"] % len(palette)],
            }
            for row in regions.regions
        ],
    }
    encoded = json.dumps(data, ensure_ascii=True, allow_nan=False)
    (out / "regions.json").write_text(encoded + "\n")
    (out / "regions-data.js").write_text(
        "window.HISTOPIA_REGIONS=" + encoded.replace("<", "\\u003c") + ";\n"
    )
    assets = Path(__file__).with_name("_study_regions_assets")
    for name in ("regions.js", "regions.css"):
        (out / name).write_text((assets / name).read_text())
    from histopia.visualization._results_catalog import _link

    page = (assets / "index.html").read_text().replace("__TITLE__", html.escape(title))
    page = page.replace(
        'href="../index.html">Study figure',
        f'href="{html.escape(_link(back_href), quote=True)}">{html.escape(back_label)}',
    )
    (out / "index.html").write_text(page)
    # A static outline fallback is useful even with all browser scripting disabled.
    paths = "".join(
        f'<path d="{row["path"]}" stroke="{row["color"]}"/>' for row in data["regions"]
    )
    sx, sy = data["pixel_size_um_xy"]
    ratio = sy / sx
    physical_height = h * ratio
    metadata = html.escape(
        json.dumps(
            dict(
                region_fingerprint=regions.fingerprint,
                pixel_size_um_xy=[sx, sy],
                origin_um_xy=regions.origin_um_xy,
            )
        )
    )
    (out / "all-regions.svg").write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {physical_height}">'
        f"<metadata>{metadata}</metadata>"
        f'<rect width="{w}" height="{physical_height}" fill="white"/>'
        f'<g transform="scale(1,{ratio})" fill="none" stroke-width="0.25">'
        f"{paths}</g></svg>"
    )
    return out / "index.html"
