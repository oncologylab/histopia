"""High-resolution OpenSeadragon review for cell-boundary results."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from histopia.visualization._cell_scope import CellSectionScope
from histopia.visualization._review_selection import REVIEW_SELECTION_JS
from histopia.visualization._review_theme import themed_review_css


def build_cell_review(
    cell_runs: dict[str, Path | str],
    output_dir: Path | str,
    *,
    cell_section_scopes: dict[str, CellSectionScope] | None = None,
) -> Path:
    """Build one fixed-viewport reviewer for selected cell runs."""

    if not cell_runs:
        raise ValueError("cell review requires at least one run")
    cell_section_scopes = cell_section_scopes or {}
    if set(cell_section_scopes) - set(cell_runs):
        raise ValueError("cell scope has no matching cell run")
    cohorts = []
    for cohort, raw_run in sorted(cell_runs.items()):
        run = Path(raw_run)
        result = json.loads((run / "cell_result.json").read_text())
        slides = result.get("slides")
        if not isinstance(slides, list) or not slides:
            raise ValueError(f"cell run contains no sections: {cohort}")
        scope = cell_section_scopes.get(cohort)
        if scope is not None:
            slides = scope.select(result)
        section_rows = []
        for row in slides:
            if not isinstance(row, dict):
                continue
            median_area_um2 = None
            qc_value = row.get("qc")
            mpp = row.get("mpp_xy")
            if isinstance(qc_value, str) and isinstance(mpp, list) and len(mpp) == 2:
                qc_path = run / qc_value
                if qc_path.is_file():
                    qc = json.loads(qc_path.read_text())
                    quantiles = qc.get("area_px_quantiles")
                    if isinstance(quantiles, list) and len(quantiles) == 3:
                        median_area_um2 = round(
                            float(quantiles[1]) * float(mpp[0]) * float(mpp[1]),
                            2,
                        )
            section_rows.append(
                {
                    "id": str(row["section"]),
                    "slide": str(row["slide"]),
                    "cell_count": int(row.get("cell_count", 0)),
                    "reference": bool(row.get("is_reference")),
                    "median_area_um2": median_area_um2,
                }
            )
        cohorts.append(
            {
                "id": cohort,
                "fingerprint": result.get("fingerprint"),
                "model": str(result.get("request", {}).get("model", "unknown")),
                "method": str(result.get("request", {}).get("method", "unknown")),
                "sections": section_rows,
                "selected_section_scope": scope is not None,
            }
        )
    method_order = {"containment": 0, "combined": 1, "multiscale": 2, "direct": 3}
    cohorts.sort(
        key=lambda item: (
            len(item["sections"]) <= 1,
            method_order.get(str(item["method"]), 4),
            item["id"],
        )
    )
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    vendor = Path(__file__).with_name("_vendor")
    shutil.copy2(vendor / "openseadragon.min.js", root / "openseadragon.min.js")
    manifest = {"schema_version": 1, "cohorts": cohorts}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (root / "manifest-data.js").write_text(
        "globalThis.HISTOPIA_CELL_REVIEW="
        + json.dumps(manifest, separators=(",", ":"))
        + ";\n"
    )
    (root / "index.html").write_text(_HTML)
    (root / "cell-review.css").write_text(themed_review_css(_CSS))
    (root / "cell-review.js").write_text(_JS)
    return root / "index.html"


_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Histopia cell boundary review</title>
  <link rel="stylesheet" href="cell-review.css">
</head>
<body>
  <header>
    <strong>Cell boundary review</strong>
    <select id="cohort" aria-label="Cohort"></select>
    <button id="previous" type="button" title="Previous section">&#8249;</button>
    <select id="section" aria-label="Section"></select>
    <button id="next" type="button" title="Next section">&#8250;</button>
    <button id="fit" type="button" title="Fit tissue">&#8962;</button>
    <label>Boundary
      <input id="opacity" type="range" min="0" max="1" step="0.05" value="0.5">
    </label>
  </header>
  <main>
    <div id="viewer-shell">
      <div id="viewer" aria-label="Whole-slide cell boundary viewer"></div>
      <output id="viewer-loading" role="status">
        Loading native histology and cell boundaries…
      </output>
    </div>
    <aside>
      <div><b id="title"></b><span id="detail"></span></div>
      <fieldset>
        <legend>Provisional boundary assessment</legend>
        <label><input type="checkbox" value="missing_cells"> Missing cells</label>
        <label><input type="checkbox" value="false_cells"> False cells or debris</label>
        <label><input type="checkbox" value="merged_cells"> Merged cells</label>
        <label><input type="checkbox" value="split_cells"> Split cells</label>
        <label><input type="checkbox" value="boundary_too_tight">
          Boundary too tight</label>
        <label><input type="checkbox" value="boundary_too_broad">
          Boundary too broad</label>
        <label><input type="checkbox" value="tile_seam"> Tile seam</label>
      </fieldset>
      <textarea id="notes" rows="4" placeholder="Optional review note"></textarea>
      <div class="actions">
        <button id="reject" type="button">Needs work</button>
        <button id="accept" type="button">Accept</button>
      </div>
      <p id="status" role="status"></p>
    </aside>
  </main>
  <script src="manifest-data.js"></script>
  <script src="openseadragon.min.js"></script>
  <script src="cell-review.js"></script>
</body>
</html>
"""

_CSS = """* { box-sizing: border-box; }
html, body {
  width: 100%; height: 100%; margin: 0; overflow: hidden;
  font: 14px/1.35 system-ui, sans-serif; color: #17202a; background: #eef1f3;
}
header {
  height: 52px; display: flex; align-items: center; gap: 8px; padding: 8px 12px;
  background: #fff; border-bottom: 1px solid #cbd2d8;
}
header strong {
  flex: 0 0 auto; margin-right: 8px; white-space: nowrap; font-size: 16px;
}
#cohort { flex: 0 1 225px; min-width: 150px; max-width: 240px; }
#section { flex: 1 1 360px; min-width: 150px; }
header > button { flex: 0 0 auto; }
button, select, textarea, input { font: inherit; }
button, select {
  height: 34px; border: 1px solid #aeb7bf; background: #fff;
  border-radius: 4px; padding: 0 10px;
}
button { cursor: pointer; }
button:hover { background: #f2f4f5; }
header label {
  display: flex; flex: 0 1 190px; align-items: center; gap: 6px;
  min-width: 140px; margin-left: auto; white-space: nowrap;
}
header label input { min-width: 72px; width: 100%; }
main {
  height: calc(100% - 52px); display: grid;
  grid-template-columns: minmax(0, 1fr) 292px;
}
#viewer-shell { position: relative; min-width: 0; min-height: 0; }
#viewer { width: 100%; height: 100%; background: #171a1c; }
#viewer-loading {
  position: absolute; inset: 50% auto auto 50%; transform: translate(-50%, -50%);
  z-index: 20; padding: 8px 12px; border: 1px solid #cbd2d8;
  border-radius: 5px; background: rgba(255, 255, 255, 0.95);
  color: #4c5963; white-space: nowrap; pointer-events: none;
}
#viewer-loading[hidden] { display: none; }
aside {
  padding: 14px; background: #fff; border-left: 1px solid #cbd2d8;
  overflow: auto;
}
#title, #detail { display: block; }
#title { font-size: 15px; overflow-wrap: anywhere; }
#detail { color: #5d6871; margin: 4px 0 14px; }
fieldset {
  border: 1px solid #cbd2d8; border-radius: 4px;
  padding: 10px; margin: 0 0 12px;
}
legend { font-weight: 600; }
fieldset label { display: block; padding: 4px 0; }
textarea {
  width: 100%; resize: vertical; border: 1px solid #aeb7bf;
  border-radius: 4px; padding: 8px;
}
.actions { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 10px; }
.actions button { font-weight: 600; }
.actions #accept { background: #176b45; color: #fff; border-color: #176b45; }
.actions #reject { color: #9b2c2c; border-color: #c98989; }
#status { min-height: 38px; color: #4c5963; }
.accepted #status { color: #176b45; font-weight: 600; }
@media(max-width: 850px) {
  main { grid-template-columns: minmax(0, 1fr) 240px; }
  header strong { display: none; }
  #cohort { flex-basis: 190px; }
}
@media(max-width: 620px) {
  body { display: grid; grid-template-rows: auto minmax(0, 1fr); }
  header { height: auto; display: grid; gap: 6px;
    grid-template-columns: 34px 34px 34px minmax(0, 1fr); padding: 7px 8px; }
  #cohort, #section { grid-column: 1 / -1; width: 100%;
    min-width: 0; max-width: none; }
  #cohort { grid-row: 1; }
  #section { grid-row: 2; }
  header > button { grid-row: 3; padding: 0; }
  header label { grid-column: 4; grid-row: 3; width: 100%; min-width: 0;
    margin-left: 0; }
  header label input { flex: 1; min-width: 0; }
  main { height: auto; min-height: 0; overflow: auto; grid-template-columns: 1fr;
    grid-template-rows: minmax(240px, 1fr) minmax(180px, 42%); }
  #viewer-loading { max-width: calc(100% - 24px); white-space: normal;
    text-align: center; }
  aside { border-top: 1px solid #cbd2d8; border-left: 0; padding: 10px; }
  fieldset { margin-bottom: 7px; }
}
"""

_JS = (
    REVIEW_SELECTION_JS
    + r"""(() => {
  "use strict";
  const manifest = globalThis.HISTOPIA_CELL_REVIEW;
  const byId = id => document.getElementById(id);
  const cohortSelect = byId("cohort");
  const sectionSelect = byId("section");
  let cohort = manifest.cohorts[0];
  let sectionIndex = 0;
  let reviewState = {};
  let reviewFingerprint = null;
  let viewer;
  let loadRequest = 0;
  let cancelPendingTileWait = () => {};

  function setLoading(value) {
    document.body.classList.toggle("loading", value);
    byId("viewer-loading").hidden = !value;
    for (const control of [cohortSelect, sectionSelect, byId("previous"),
      byId("next"), byId("fit"), byId("opacity"), byId("accept"),
      byId("reject")]) control.disabled = value;
  }

  function tileSource(metadata, name) {
    const layer = metadata.layers[name];
    const maxLevel = Math.ceil(Math.log2(Math.max(layer.width, layer.height)));
    const minLevel = maxLevel - layer.levels.length + 1;
    const prefix = histopiaUrl(`/api/wsi/${encodeURIComponent(metadata.cohort)}/`) +
      `${metadata.section}/${name}/${layer.digest}/dzi/${minLevel}/`;
    const source = new OpenSeadragon.TileSource({
      width: layer.width,
      height: layer.height,
      tileSize: layer.tile_size,
      tileOverlap: 0,
      minLevel,
      maxLevel,
    });
    source.getTileUrl = (level, x, y) =>
      `${prefix}${level}/${x}_${y}.${layer.format}`;
    return source;
  }

  function ensureViewer() {
    if (viewer) return viewer;
    viewer = OpenSeadragon({
      element: byId("viewer"),
      drawer: navigator.webdriver ? ["html"] : ["canvas"],
      showNavigator: true,
      navigatorPosition: "BOTTOM_RIGHT",
      showNavigationControl: false,
      animationTime: 0.25,
      blendTime: 0.05,
      immediateRender: true,
      imageLoaderLimit: 2,
      tileRetryMax: 3,
      tileRetryDelay: 1000,
      maxZoomPixelRatio: 3,
      preserveViewport: false,
    });
    globalThis.histopiaCellViewer = viewer;
    return viewer;
  }

  async function load() {
    const request = ++loadRequest;
    cancelPendingTileWait();
    cancelPendingTileWait = () => {};
    const section = cohort.sections[sectionIndex];
    sectionSelect.value = section.id;
    delete document.body.dataset.sectionReady;
    delete document.body.dataset.cohortReady;
    setLoading(true);
    byId("status").textContent = "Loading native section";
    const response = await fetch(
      histopiaUrl(`/api/wsi/${encodeURIComponent(cohort.id)}/${section.id}`));
    if (!response.ok) throw new Error(`WSI metadata failed (${response.status})`);
    const metadata = await response.json();
    if (request !== loadRequest) return;
    if (!metadata.layers.raw || !metadata.layers.cells)
      throw new Error("Native histology and cell labels are both required");
    byId("title").textContent = `${section.id} ${metadata.label}`;
    const area = section.median_area_um2 == null ? "" :
      ` | ${section.median_area_um2.toLocaleString()} µm² median`;
    byId("detail").textContent = `${cohort.model} ${cohort.method} | ` +
      `${section.cell_count.toLocaleString()} cells${area} | ` +
      `${metadata.layers.raw.width.toLocaleString()} × ` +
      `${metadata.layers.raw.height.toLocaleString()} px`;
    showDecision(section.id);
    const active = ensureViewer();
    active.close();
    active.addOnceHandler("open", () => {
      if (request !== loadRequest) return;
      active.viewport.goHome(true);
      active.addTiledImage({
        tileSource: tileSource(metadata, "cells"),
        opacity: Number(byId("opacity").value),
        success: event => {
          if (request !== loadRequest) return;
          const item = event.item;
          let timer = 0;
          let settled = false;
          const cleanup = () => {
            active.removeHandler("tile-loaded", check);
            active.removeHandler("fully-loaded-change", check);
            if (timer) clearTimeout(timer);
            if (cancelPendingTileWait === cleanup)
              cancelPendingTileWait = () => {};
          };
          const finish = () => {
            if (settled || request !== loadRequest) return;
            settled = true;
            cleanup();
            document.body.dataset.sectionReady = section.id;
            document.body.dataset.cohortReady = cohort.id;
            setLoading(false);
          };
          const check = tileEvent => {
            if (request !== loadRequest) { cleanup(); return; }
            if (tileEvent?.tiledImage && tileEvent.tiledImage !== item) return;
            if (active.getFullyLoaded())
              requestAnimationFrame(() => requestAnimationFrame(finish));
          };
          cancelPendingTileWait = cleanup;
          active.addHandler("tile-loaded", check);
          active.addHandler("fully-loaded-change", check);
          timer = setTimeout(() => {
            if (request !== loadRequest) { cleanup(); return; }
            cleanup();
            setLoading(false);
            showError(new Error("Cell boundary tiles did not finish loading"));
          }, 45000);
          requestAnimationFrame(() => requestAnimationFrame(check));
        },
        error: () => {
          if (request !== loadRequest) return;
          setLoading(false);
          showError(new Error("Cell boundary tiles failed to open"));
        },
      });
    });
    active.addOnceHandler("open-failed", () => {
      if (request !== loadRequest) return;
      setLoading(false);
      showError(new Error("Native histology tiles failed to open"));
    });
    active.open(tileSource(metadata, "raw"));
  }

  function showDecision(section) {
    const row = reviewState[section] || {};
    document.body.classList.toggle("accepted", row.decision === "accept");
    byId("notes").value = row.comment || "";
    const issues = new Set(row.labels || []);
    document.querySelectorAll("fieldset input").forEach(input => {
      input.checked = issues.has(input.value);
    });
    byId("status").textContent = row.decision === "accept"
      ? `Provisional accept by ${row.reviewer || "reviewer"}`
      : row.reviewed_at
        ? `Provisional ${row.decision}: changes requested`
        : "Not provisionally reviewed";
  }

  async function refreshReview(request) {
    const activeCohort = cohort;
    const response = await fetch(
      histopiaUrl(`/api/reviews/provisional?cohort=${encodeURIComponent(cohort.id)}`) +
      "&stage=cells",
      {cache: "no-store"},
    );
    if (!response.ok)
      return request === loadRequest && cohort === activeCohort;
    const payload = await response.json();
    if (request !== loadRequest || cohort !== activeCohort) return false;
    if (payload.fingerprint !== activeCohort.fingerprint)
      throw new Error("Cell artifact changed; reload the reviewer");
    reviewFingerprint = payload.fingerprint;
    reviewState = payload.feedback || {};
    showDecision(activeCohort.sections[sectionIndex].id);
    return true;
  }

  async function decide(accepted) {
    const section = cohort.sections[sectionIndex].id;
    const issues = [...document.querySelectorAll("fieldset input:checked")]
      .map(input => input.value);
    byId("status").textContent = "Saving review";
    if (!reviewFingerprint) throw new Error("Review artifact is not ready");
    const response = await fetch(histopiaUrl("/api/reviews/provisional"), {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        cohort: cohort.id,
        stage: "cells",
        fingerprint: reviewFingerprint,
        slide_id: section,
        decision: accepted ? "accept" : "reject",
        reviewer: "browser-provisional",
        comment: byId("notes").value,
        labels: issues,
        checks: {},
      }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Review could not be saved");
    reviewState = payload.provisional_review.feedback || reviewState;
    showDecision(section);
  }

  function setCohort(id) {
    const request = ++loadRequest;
    cohort = manifest.cohorts.find(item => item.id === id) || manifest.cohorts[0];
    cohortSelect.value = cohort.id;
    histopiaReviewSelection.remember(cohort.id);
    sectionIndex = 0;
    reviewFingerprint = null;
    reviewState = {};
    delete document.body.dataset.sectionReady;
    delete document.body.dataset.cohortReady;
    setLoading(true);
    sectionSelect.replaceChildren(...cohort.sections.map(item => {
      const option = document.createElement("option");
      option.value = item.id;
      option.textContent = `${item.id} ${item.slide}`;
      return option;
    }));
    refreshReview(request).then(refreshed => {
      if (!refreshed || request !== loadRequest) return;
      load().catch(error => {setLoading(false); showError(error);});
    }).catch(error => {setLoading(false); showError(error);});
  }

  function step(offset) {
    sectionIndex = (sectionIndex + offset + cohort.sections.length) %
      cohort.sections.length;
    load().catch(error => {setLoading(false); showError(error);});
  }

  function showError(error) {
    byId("status").textContent = error instanceof Error ? error.message : String(error);
  }

  cohortSelect.replaceChildren(...manifest.cohorts.map(item => {
    const option = document.createElement("option"); option.value = item.id;
    const sectionLabel = item.selected_section_scope
      ? `${item.sections.length} selected sections`
      : item.sections.length === 1 ? "1 section" : `${item.sections.length} sections`;
    option.textContent = `${item.id} · ${item.method} · ${sectionLabel}`;
    return option;
  }));
  cohortSelect.addEventListener("change", () => setCohort(cohortSelect.value));
  sectionSelect.addEventListener("change", () => {
    sectionIndex = cohort.sections.findIndex(item => item.id === sectionSelect.value);
    load().catch(error => {setLoading(false); showError(error);});
  });
  byId("previous").addEventListener("click", () => step(-1));
  byId("next").addEventListener("click", () => step(1));
  byId("fit").addEventListener("click", () => viewer?.viewport.goHome(true));
  byId("opacity").addEventListener("input", event =>
    viewer?.world.getItemAt(1)?.setOpacity(Number(event.target.value)));
  byId("accept").addEventListener("click", () => decide(true).catch(showError));
  byId("reject").addEventListener("click", () => decide(false).catch(showError));
  try {setCohort(histopiaReviewSelection.requested(manifest.cohorts));}
  catch (error) {setLoading(false); showError(error);}
})();

// Keep same-origin data requests inside a code-server port proxy.
function histopiaUrl(path) {
  const match = location.pathname.match(/^.*?\/proxy\/[0-9]+(?=\/|$)/);
  return (match ? match[0] : "") + path;
}
"""
)
