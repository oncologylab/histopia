"""High-resolution pathology annotation review over native WSI."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from histopia.visualization._review_selection import REVIEW_SELECTION_JS
from histopia.visualization._review_theme import themed_review_css

_COHORT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def build_annotation_review(
    annotation_runs: dict[str, Path | str],
    semantic_runs: dict[str, Path | str],
    output_dir: Path | str,
) -> Path:
    """Build a fixed-viewport editor for configured annotation cohorts."""

    if not annotation_runs:
        raise ValueError("annotation review requires at least one run")
    missing = set(annotation_runs) - set(semantic_runs)
    if missing:
        raise ValueError(
            "annotation review requires semantic runs for: "
            + ", ".join(sorted(missing))
        )
    cohorts: list[dict[str, object]] = []
    for cohort in sorted(annotation_runs):
        if not _COHORT_RE.fullmatch(cohort):
            raise ValueError(f"invalid annotation cohort name: {cohort!r}")
        semantic = json.loads(
            (Path(semantic_runs[cohort]) / "semantic_result.json").read_text()
        )
        slides = semantic.get("slides")
        if not isinstance(slides, list) or not slides:
            raise ValueError(f"semantic run contains no sections: {cohort}")
        sections = []
        for index, row in enumerate(slides):
            if not isinstance(row, dict) or not isinstance(row.get("id"), str):
                raise ValueError(f"semantic section metadata is invalid: {cohort}")
            sections.append(
                {
                    "id": f"{index + 1:03d}",
                    "slide": row["id"],
                }
            )
        cohorts.append({"id": cohort, "sections": sections})

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    vendor = Path(__file__).with_name("_vendor")
    shutil.copy2(vendor / "openseadragon.min.js", root / "openseadragon.min.js")
    manifest = {"schema_version": 1, "cohorts": cohorts}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (root / "manifest-data.js").write_text(
        "globalThis.HISTOPIA_ANNOTATION_REVIEW="
        + json.dumps(manifest, separators=(",", ":"))
        + ";\n"
    )
    (root / "index.html").write_text(_HTML)
    (root / "annotation-review.css").write_text(themed_review_css(_CSS))
    (root / "annotation-review.js").write_text(_JS)
    return root / "index.html"


_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Histopia pathology annotation</title>
  <link rel="stylesheet" href="annotation-review.css">
</head>
<body>
  <header>
    <strong>Pathology annotation</strong>
    <select id="cohort" aria-label="Cohort"></select>
    <button id="previous" type="button" title="Previous section">&#8249;</button>
    <select id="section" aria-label="Section"></select>
    <button id="next" type="button" title="Next section">&#8250;</button>
    <div class="segments" aria-label="Image layer">
      <button type="button" data-layer="raw" aria-pressed="true">Raw</button>
      <button type="button" data-layer="registered" aria-pressed="false">
        Registered
      </button>
    </div>
    <button id="fit" type="button" title="Fit tissue">&#8962;</button>
  </header>
  <main>
    <section id="stage">
      <div id="viewer" aria-label="Whole-slide annotation viewer"></div>
      <div id="loader" role="status"><span>Loading native WSI…</span></div>
      <svg id="overlay" aria-label="Pathology annotation overlay"></svg>
    </section>
    <section class="context">
      <div class="context-title">Adjacent section</div>
      <div id="neighbor" aria-label="Adjacent whole-slide context"></div>
      <div id="neighbor-label"></div>
    </section>
    <aside>
      <b id="title"></b>
      <span id="detail"></span>
      <fieldset id="palette"><legend>Class</legend></fieldset>
      <label class="confidence">Confidence
        <input id="confidence" type="range" min="0" max="1" step="0.05" value="1">
        <output id="confidence-value">1.00</output>
      </label>
      <label>Reviewer<input id="reviewer" type="text" value="browser"></label>
      <label>Notes<textarea id="notes" rows="2"></textarea></label>
      <div class="tools">
        <button id="draw" type="button" title="Draw freehand polygon">&#9998;</button>
        <button id="delete" type="button" title="Delete selected polygon">
          &#128465;
        </button>
        <button id="undo" type="button" title="Undo">&#8630;</button>
        <button id="redo" type="button" title="Redo">&#8631;</button>
      </div>
      <button id="save" class="primary" type="button">Save revision</button>
      <p id="status" role="status"></p>
      <ol id="annotations"></ol>
    </aside>
  </main>
  <script src="manifest-data.js"></script>
  <script src="openseadragon.min.js"></script>
  <script src="annotation-review.js"></script>
</body>
</html>
"""

_CSS = """*{box-sizing:border-box}
html,body{width:100%;height:100%;margin:0;overflow:hidden;font:14px/1.35
system-ui,sans-serif;color:#17202a;background:#eef1f3}
header{height:52px;display:flex;align-items:center;gap:8px;padding:8px 12px;
background:#fff;border-bottom:1px solid #cbd2d8}header strong{font-size:16px;
margin-right:8px}button,select,textarea,input{font:inherit}button,select,input[type=text]
{height:34px;border:1px solid #aeb7bf;background:#fff;border-radius:4px;
padding:0 10px}button{cursor:pointer}button:hover{background:#f2f4f5}
.segments{display:flex}.segments button{border-radius:0}.segments button:first-child
{border-radius:4px 0 0 4px}.segments button:last-child{border-radius:0 4px 4px 0;
margin-left:-1px}.segments button[aria-pressed=true]{background:#176b45;color:#fff;
border-color:#176b45}main{height:calc(100% - 52px);display:grid;
grid-template-columns:minmax(0,1fr) 230px 292px;min-height:0}#stage{position:relative;
min-width:0;min-height:0;background:#171a1c}#viewer,#overlay{position:absolute;inset:0;
width:100%;height:100%}#overlay{z-index:20;pointer-events:none;touch-action:none}
#loader{position:absolute;inset:0;z-index:15;display:grid;place-items:center;
background:#171a1c;color:#fff;pointer-events:none}#loader[hidden]{display:none}
#loader span{padding:8px 12px;border:1px solid #596168;border-radius:4px;
background:#202529;font-size:12px;font-weight:600}
#overlay.drawing{pointer-events:all;cursor:crosshair}.shape{fill-opacity:.22;
stroke-width:2;vector-effect:non-scaling-stroke;pointer-events:visiblePainted;cursor:pointer}
.shape.selected{fill-opacity:.38;stroke:#fff;stroke-width:3}.draft{fill:none;stroke:#fff;
stroke-width:2;stroke-dasharray:5 4;vector-effect:non-scaling-stroke}
.context{display:grid;grid-template-rows:32px minmax(0,1fr) 44px;background:#202529;
color:#fff;border-left:1px solid #596168;min-height:0}.context-title,#neighbor-label
{padding:8px 10px;font-size:12px;overflow:hidden;text-overflow:ellipsis;
white-space:nowrap}#neighbor{min-height:0}.context #neighbor-label{white-space:normal;
color:#d5d9dc}
aside{padding:12px;background:#fff;border-left:1px solid #cbd2d8;overflow:auto;
min-height:0}#title,#detail{display:block}#title{font-size:15px;overflow-wrap:anywhere}
#detail{color:#5d6871;margin:4px 0 10px}fieldset{border:1px solid #cbd2d8;
border-radius:4px;padding:8px;margin:0 0 10px}legend{font-weight:600}.class-choice
{display:flex;align-items:center;gap:7px;padding:3px 0}.swatch{width:14px;height:14px;
border:1px solid #667;border-radius:2px;flex:none}.confidence,aside>label{display:grid;
grid-template-columns:82px minmax(0,1fr) 38px;align-items:center;gap:6px;margin:8px 0}
aside>label:not(.confidence){grid-template-columns:72px minmax(0,1fr)}textarea
{width:100%;resize:vertical;border:1px solid #aeb7bf;border-radius:4px;padding:6px}
.tools{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-top:10px}
.tools button{font-size:18px;padding:0}.tools button.active{background:#176b45;
color:#fff}
.primary{width:100%;margin-top:8px;background:#176b45;color:#fff;border-color:#176b45;
font-weight:600}#status{min-height:36px;color:#4c5963;font-size:12px}#annotations
{padding-left:24px;margin:0;font-size:12px}#annotations li{padding:4px;cursor:pointer;
border-bottom:1px solid #e2e6e8}#annotations li.selected{background:#e9f5ef}
@media(max-width:1000px){main{grid-template-columns:minmax(0,1fr) 210px 250px}
header strong{display:none}}
@media(max-width:700px){
html,body{overflow:hidden}header{height:92px;align-content:center;flex-wrap:wrap;
padding:7px 8px}header select{min-width:92px;flex:1}.segments{order:2;flex:1}
.segments button{flex:1}main{height:calc(100% - 92px);grid-template-columns:1fr;
grid-template-rows:minmax(220px,1fr) minmax(180px,42%)}#stage{grid-row:1}
.context{display:none}aside{grid-row:2;border-left:0;border-top:1px solid #cbd2d8;
padding:9px}.tools{margin-top:6px}fieldset{margin-bottom:6px}}
"""

_JS = (
    REVIEW_SELECTION_JS
    + r"""(() => {
  "use strict";
  const manifest = globalThis.HISTOPIA_ANNOTATION_REVIEW;
  const byId = id => document.getElementById(id);
  const cohortSelect = byId("cohort");
  const sectionSelect = byId("section");
  const overlay = byId("overlay");
  let cohort = manifest.cohorts[0];
  let sectionIndex = 0;
  let metadata = null;
  let catalog = null;
  let collection = null;
  let layer = "raw";
  let selected = -1;
  let drawing = false;
  let draft = [];
  let undo = [];
  let redo = [];
  let cohortRequest = 0;
  let loadRequest = 0;
  let layerRequest = 0;
  let viewState = null;
  let layerOpening = false;
  let layerWaitCleanup = null;
  let renderPending = false;

  const viewer = OpenSeadragon({element: byId("viewer"), showNavigator: true,
    navigatorPosition: "BOTTOM_RIGHT", showNavigationControl: false,
    drawer: navigator.webdriver ? ["html"] : ["canvas"],
    animationTime: 0, blendTime: 0, immediateRender: true,
    imageLoaderLimit: 2, maxZoomPixelRatio: 4, preserveViewport: false,
    constrainDuringPan: true, visibilityRatio: .2});
  const neighbor = OpenSeadragon({element: byId("neighbor"), showNavigator: false,
    showNavigationControl: false,
    drawer: navigator.webdriver ? ["html"] : ["canvas"],
    animationTime: 0, blendTime: 0,
    imageLoaderLimit: 1, immediateRender: true, mouseNavEnabled: false});
  byId("viewer").histopiaViewer = viewer;
  byId("neighbor").histopiaViewer = neighbor;

  function tileSource(meta, name) {
    const item = meta.layers[name];
    const maxLevel = Math.ceil(Math.log2(Math.max(item.width, item.height)));
    const minLevel = maxLevel - item.levels.length + 1;
    const prefix = histopiaUrl(`/api/wsi/${encodeURIComponent(meta.cohort)}/` +
      `${meta.section}/`) +
      `${name}/${item.digest}/dzi/${minLevel}/`;
    const source = new OpenSeadragon.TileSource({width:item.width,
      height:item.height, tileSize:item.tile_size, tileOverlap:0,
      minLevel:0, maxLevel:item.levels.length - 1});
    source.getTileUrl = (level, x, y) => {
      const dziLevel = minLevel + level;
      return `${prefix}${dziLevel}/${x}_${y}.${item.format}`;
    };
    return source;
  }

  function snapshot() { return JSON.stringify(collection.features); }
  function restore(value) {
    collection.features = JSON.parse(value); selected = -1; render();
  }
  function checkpoint() { undo.push(snapshot()); redo.length = 0; }

  function imagePoint(event) {
    const rect = overlay.getBoundingClientRect();
    return viewer.viewport.viewerElementToImageCoordinates(
      new OpenSeadragon.Point(event.clientX - rect.left, event.clientY - rect.top));
  }
  function screenPoint(point) {
    return viewer.viewport.imageToViewerElementCoordinates(
      new OpenSeadragon.Point(point[0], point[1]));
  }
  function polygonPath(geometry) {
    const polygons = geometry.type === "Polygon" ? [geometry.coordinates] :
      geometry.coordinates;
    return polygons.flatMap(polygon => polygon.map(ring => ring.map((point, index) => {
      const screen = screenPoint(point);
      return `${index ? "L" : "M"}${screen.x.toFixed(1)},${screen.y.toFixed(1)}`;
    }).join(" ") + " Z")).join(" ");
  }

  function render() {
    overlay.replaceChildren();
    if (!collection || layer !== "raw") return;
    collection.features.forEach((feature, index) => {
      const row = catalog.ontology.classes.find(
        item => item.id === feature.properties.class_id);
      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path.setAttribute("d", polygonPath(feature.geometry));
      path.setAttribute("fill", row?.color || "#ffffff");
      path.setAttribute("stroke", row?.color || "#ffffff");
      path.setAttribute("class", `shape${selected === index ? " selected" : ""}`);
      path.addEventListener("pointerdown", event => {
        if (drawing) return; event.stopPropagation(); selected = index; render();
      });
      overlay.append(path);
    });
    if (draft.length > 1) {
      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path.setAttribute("class", "draft");
      path.setAttribute("d", draft.map((point, index) => {
        const screen = screenPoint(point);
        return `${index ? "L" : "M"}${screen.x.toFixed(1)},${screen.y.toFixed(1)}`;
      }).join(" "));
      overlay.append(path);
    }
    renderList();
  }

  function scheduleRender() {
    if (renderPending) return;
    renderPending = true;
    requestAnimationFrame(() => {
      renderPending = false;
      render();
    });
  }

  function renderList() {
    byId("annotations").replaceChildren(...collection.features.map((feature, index) => {
      const item = document.createElement("li");
      const row = catalog.ontology.classes.find(
        value => value.id === feature.properties.class_id);
      item.textContent = `${row?.label || feature.properties.class_id} · `+
        `${Number(feature.properties.confidence).toFixed(2)}`;
      item.classList.toggle("selected", index === selected);
      item.addEventListener("click", () => {selected = index; render();});
      return item;
    }));
  }

  async function fetchJson(url) {
    const response = await fetch(url); const payload = await response.json();
    if (!response.ok)
      throw new Error(payload.error || `Request failed (${response.status})`);
    return payload;
  }

  function setLayerLoading(value, text = "Loading native WSI…") {
    const loader = byId("loader");
    loader.hidden = !value;
    loader.querySelector("span").textContent = text;
  }

  async function load() {
    const request = ++loadRequest;
    const activeCohort = cohort;
    const activeIndex = sectionIndex;
    const section = cohort.sections[sectionIndex];
    sectionSelect.value = section.id; selected = -1; undo = []; redo = [];
    byId("status").textContent = "Loading section";
    ++layerRequest;
    viewState = null; layerOpening = false;
    if (layerWaitCleanup) layerWaitCleanup();
    setLayerLoading(true, "Loading section…");
    delete document.body.dataset.layerReady;
    viewer.close(); neighbor.close(); overlay.replaceChildren();
    const [nextMetadata, nextCatalog, nextCollection] = await Promise.all([
      fetchJson(histopiaUrl(`/api/wsi/${encodeURIComponent(cohort.id)}/${section.id}`)),
      fetchJson(histopiaUrl(`/api/annotations?cohort=${encodeURIComponent(cohort.id)}`)),
      fetchJson(histopiaUrl(`/api/annotations/section?cohort=${encodeURIComponent(cohort.id)}`)+
        `&section=${section.id}`),
    ]);
    if (request !== loadRequest || cohort !== activeCohort ||
        sectionIndex !== activeIndex) return;
    [metadata, catalog, collection] = [nextMetadata, nextCatalog, nextCollection];
    if (!metadata.layers.raw) throw new Error("Native histology is required");
    if (!metadata.layers[layer]) layer = "raw";
    byId("title").textContent = `${section.id} ${metadata.label}`;
    byId("detail").textContent = `Revision ${collection.histopia.revision} · `+
      `${collection.features.length} annotations`;
    byId("notes").value = collection.histopia.notes || "";
    buildPalette(); updateLayerButtons(); openLayer({fit: true});
    loadNeighbor(request).catch(showError); renderList();
    byId("status").textContent = layer === "raw" ? "Ready" :
      "Registered comparison; draw on Raw";
  }

  function currentViewState() {
    if (!viewer.world.getItemCount()) return null;
    const item = viewer.world.getItemAt(0);
    const size = item.getContentSize();
    const center = item.viewportToImageCoordinates(viewer.viewport.getCenter(true));
    return center && size ? {
      centerFraction: new OpenSeadragon.Point(
        center.x / size.x,
        center.y / size.y,
      ),
      homeZoomRatio: viewer.viewport.getZoom(true) /
        viewer.viewport.getHomeZoom(),
    } : null;
  }

  function rememberViewState() {
    if (layerOpening) return viewState;
    viewState = currentViewState() || viewState;
    return viewState;
  }

  function openLayer({fit = false} = {}) {
    const request = ++layerRequest;
    const targetLayer = layer;
    const previous = fit ? null : rememberViewState();
    if (layerWaitCleanup) layerWaitCleanup();
    layerOpening = true;
    setLayerLoading(true, `Loading ${targetLayer} tiles…`);
    delete document.body.dataset.layerReady;
    let settled = false;
    let timer = 0;
    function cleanup() {
      viewer.removeHandler("tile-loaded", onTile);
      viewer.removeHandler("fully-loaded-change", onFullyLoaded);
      if (timer) clearTimeout(timer);
      if (layerWaitCleanup === cleanup) layerWaitCleanup = null;
    }
    function finish() {
      if (settled) return;
      settled = true;
      cleanup();
      requestAnimationFrame(() => requestAnimationFrame(() => {
        if (request !== layerRequest) return;
        layerOpening = false;
        viewState = currentViewState();
        document.body.dataset.layerReady = targetLayer;
        setLayerLoading(false);
        render();
      }));
    }
    function onTile(event) {
      if (request !== layerRequest ||
          viewer.world.getIndexOfItem(event.tiledImage) < 0) return;
      finish();
    }
    function onFullyLoaded(event) {
      if (request !== layerRequest) return;
      if (event?.fullyLoaded === true || viewer.getFullyLoaded()) finish();
    }
    layerWaitCleanup = cleanup;
    viewer.addHandler("tile-loaded", onTile);
    viewer.addHandler("fully-loaded-change", onFullyLoaded);
    viewer.addOnceHandler("open", () => {
      if (request !== layerRequest) return;
      if (previous) {
        const nextItem = viewer.world.getItemAt(0);
        const nextSize = nextItem.getContentSize();
        const nextImageCenter = new OpenSeadragon.Point(
          previous.centerFraction.x * nextSize.x,
          previous.centerFraction.y * nextSize.y,
        );
        viewer.viewport.panTo(
          nextItem.imageToViewportCoordinates(nextImageCenter), true,
        );
        viewer.viewport.zoomTo(
          viewer.viewport.getHomeZoom() * previous.homeZoomRatio, null, true,
        );
        viewer.viewport.applyConstraints(true);
      } else {
        viewer.viewport.goHome(true);
      }
      if (viewer.getFullyLoaded()) finish();
    });
    viewer.addOnceHandler("open-failed", () => {
      if (request !== layerRequest) return;
      cleanup();
      layerOpening = false;
      setLayerLoading(false);
      byId("status").textContent = `Could not open ${targetLayer} tiles`;
    });
    timer = setTimeout(() => {
      if (request !== layerRequest) return;
      cleanup();
      layerOpening = false;
      setLayerLoading(false);
      byId("status").textContent = `${targetLayer} tile response is delayed`;
    }, 30000);
    viewer.open(tileSource(metadata, targetLayer));
  }

  async function loadNeighbor(request) {
    const activeCohort = cohort;
    const activeIndex = sectionIndex;
    const index = Math.min(sectionIndex + 1, cohort.sections.length - 1);
    const adjacent = cohort.sections[index];
    byId("neighbor-label").textContent = index === sectionIndex ?
      "No following section" : `Loading ${adjacent.id}`;
    if (index === sectionIndex) {neighbor.close(); return;}
    try {
      const data = await fetchJson(
        histopiaUrl(`/api/wsi/${encodeURIComponent(cohort.id)}/${adjacent.id}`));
      if (request !== loadRequest || cohort !== activeCohort ||
          sectionIndex !== activeIndex) return;
      byId("neighbor-label").textContent = `${adjacent.id} ${data.label}`;
      neighbor.addOnceHandler("open", () => neighbor.viewport.goHome(true));
      neighbor.open(tileSource(data, data.layers.registered ? "registered" : "raw"));
    } catch (error) {
      if (request === loadRequest) byId("neighbor-label").textContent = String(error);
    }
  }

  function buildPalette() {
    const controls = catalog.ontology.classes.map((row, index) => {
      const label = document.createElement("label"); label.className = "class-choice";
      const input = document.createElement("input"); input.type = "radio";
      input.name = "class"; input.value = row.id; input.checked = index === 0;
      const swatch = document.createElement("span"); swatch.className = "swatch";
      swatch.style.background = row.color;
      const text = document.createElement("span"); text.textContent = row.label;
      label.append(input, swatch, text);
      return label;
    });
    byId("palette").querySelectorAll(".class-choice").forEach(item => item.remove());
    byId("palette").append(...controls);
  }

  function setDrawing(value) {
    drawing = value && layer === "raw"; draft = [];
    overlay.classList.toggle("drawing", drawing);
    byId("draw").classList.toggle("active", drawing);
    viewer.setMouseNavEnabled(!drawing); render();
  }

  overlay.addEventListener("pointerdown", event => {
    if (!drawing) return; event.preventDefault();
    overlay.setPointerCapture(event.pointerId);
    const point = imagePoint(event); draft = [[point.x, point.y]]; render();
  });
  overlay.addEventListener("pointermove", event => {
    if (!drawing || !overlay.hasPointerCapture(event.pointerId)) return;
    const point = imagePoint(event); const previous = draft.at(-1);
    if (!previous || Math.hypot(point.x - previous[0], point.y - previous[1]) > 8)
      draft.push([Math.max(0, point.x), Math.max(0, point.y)]);
    render();
  });
  overlay.addEventListener("pointerup", event => {
    if (!drawing || draft.length < 3) {draft = []; render(); return;}
    checkpoint(); draft.push([...draft[0]]);
    const classId = document.querySelector("input[name=class]:checked").value;
    collection.features.push({type:"Feature", properties:{class_id:classId,
      confidence:Number(byId("confidence").value), provenance:"manual", accepted:true},
      geometry:{type:"Polygon", coordinates:[draft]}});
    selected = collection.features.length - 1; draft = []; setDrawing(false); render();
    overlay.releasePointerCapture(event.pointerId);
  });

  async function save() {
    const section = cohort.sections[sectionIndex].id;
    byId("status").textContent = "Saving revision";
    const response = await fetch(histopiaUrl("/api/annotations/section"), {
      method:"POST",
      headers:{"Content-Type":"application/json"}, body:JSON.stringify({
        cohort:cohort.id, section, reviewer:byId("reviewer").value,
        notes:byId("notes").value, expected_revision:collection.histopia.revision,
        feature_collection:{type:"FeatureCollection", features:collection.features}})});
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Annotation save failed");
    collection = payload.annotations; undo = []; redo = [];
    byId("detail").textContent = `Revision ${collection.histopia.revision} · `+
      `${collection.features.length} annotations`;
    byId("status").textContent = "Revision saved"; render();
  }

  async function setCohort(id) {
    const request = ++cohortRequest;
    ++loadRequest;
    ++layerRequest;
    const selectedCohort = manifest.cohorts.find(item => item.id === id) ||
      manifest.cohorts[0];
    cohortSelect.value = selectedCohort.id;
    histopiaReviewSelection.remember(selectedCohort.id);
    byId("status").textContent = "Checking native WSI and annotation provenance";
    const [wsiCatalog] = await Promise.all([
      fetchJson(histopiaUrl(`/api/wsi/${encodeURIComponent(selectedCohort.id)}`)),
      fetchJson(histopiaUrl(`/api/annotations?cohort=${encodeURIComponent(selectedCohort.id)}`)),
    ]);
    if (request !== cohortRequest) return;
    viewer.close(); neighbor.close(); overlay.replaceChildren();
    const available = new Set(wsiCatalog.sections.map(item => item.section));
    cohort = {...selectedCohort, sections:selectedCohort.sections.filter(
      item => available.has(item.id))};
    sectionIndex = 0;
    sectionSelect.replaceChildren(...cohort.sections.map(item => {
      const option = document.createElement("option"); option.value = item.id;
      option.textContent = `${item.id} ${item.slide}`; return option;
    }));
    const hasSections = cohort.sections.length > 0;
    sectionSelect.disabled = !hasSections;
    byId("previous").disabled = !hasSections;
    byId("next").disabled = !hasSections;
    byId("fit").disabled = !hasSections;
    byId("save").disabled = !hasSections;
    if (!hasSections) {
      if (layerWaitCleanup) layerWaitCleanup();
      setLayerLoading(false);
      metadata = catalog = collection = null;
      byId("title").textContent = selectedCohort.id;
      byId("detail").textContent = "No native WSI sections are configured";
      byId("neighbor-label").textContent = "No adjacent section";
      byId("status").textContent = "Native WSI tiles unavailable for this cohort";
      document.querySelectorAll("[data-layer]").forEach(button => {
        button.disabled = true;
      });
      return;
    }
    await load();
  }
  function step(offset) {if (!cohort.sections.length) return;
    sectionIndex = (sectionIndex + offset + cohort.sections.length) %
      cohort.sections.length; load().catch(showError);}
  function showError(error) {setLayerLoading(false);
    byId("status").textContent = error instanceof Error ?
      error.message : String(error);}
  async function initialize() {
    const query = new URLSearchParams(location.search);
    if (["subject", "mouse", "cohort"].some(key => query.has(key))) {
      try {await setCohort(histopiaReviewSelection.requested(manifest.cohorts));}
      catch (error) {showError(error);}
      return;
    }
    let lastError = null;
    for (const candidate of manifest.cohorts) {
      try {
        await setCohort(candidate.id);
        return;
      } catch (error) {
        lastError = error;
        const option = [...cohortSelect.options].find(
          item => item.value === candidate.id);
        if (option) {
          option.disabled = true;
          option.title = error instanceof Error ? error.message : String(error);
        }
      }
    }
    showError(lastError || new Error("No provenance-valid annotation cohorts"));
  }
  function updateLayerButtons() {document.querySelectorAll("[data-layer]").forEach(
    button => {button.disabled = !metadata.layers[button.dataset.layer];
      button.setAttribute("aria-pressed", String(button.dataset.layer === layer));});}

  viewer.addHandler("animation", scheduleRender);
  viewer.addHandler("resize", scheduleRender);
  viewer.addHandler("update-viewport", scheduleRender);
  cohortSelect.replaceChildren(...manifest.cohorts.map(item => {
    const option = document.createElement("option"); option.value = item.id;
    option.textContent = item.id; return option;}));
  cohortSelect.addEventListener("change", () =>
    setCohort(cohortSelect.value).catch(showError));
  sectionSelect.addEventListener("change", () => {sectionIndex = cohort.sections
    .findIndex(item => item.id === sectionSelect.value); load().catch(showError);});
  byId("previous").addEventListener("click", () => step(-1));
  byId("next").addEventListener("click", () => step(1));
  byId("fit").addEventListener("click", () => viewer.viewport.goHome(true));
  byId("draw").addEventListener("click", () => setDrawing(!drawing));
  byId("delete").addEventListener("click", () => {if (selected < 0) return;
    checkpoint(); collection.features.splice(selected, 1); selected = -1; render();});
  byId("undo").addEventListener("click", () => {if (!undo.length) return;
    redo.push(snapshot()); restore(undo.pop());});
  byId("redo").addEventListener("click", () => {if (!redo.length) return;
    undo.push(snapshot()); restore(redo.pop());});
  byId("save").addEventListener("click", () => save().catch(showError));
  byId("confidence").addEventListener("input", event =>
    byId("confidence-value").value = Number(event.target.value).toFixed(2));
  document.querySelectorAll("[data-layer]").forEach(button => button.addEventListener(
    "click", () => {if (!metadata?.layers[button.dataset.layer]) return;
      layer = button.dataset.layer; setDrawing(false); updateLayerButtons();
      openLayer({fit: false});
      byId("status").textContent = layer === "raw" ? "Ready" :
        "Registered comparison; draw on Raw";}));
  initialize();
})();

// Keep same-origin data requests inside a code-server port proxy.
function histopiaUrl(path) {
  const match = location.pathname.match(/^.*?\/proxy\/[0-9]+(?=\/|$)/);
  return (match ? match[0] : "") + path;
}
"""
)
