"use strict";

const manifest = globalThis.HISTOPIA_STAIN_REVIEW;
if (!manifest || !Array.isArray(manifest.mice) || !manifest.mice.length) {
  throw new Error("Missing Histopia stain review manifest");
}

const elements = {
  mouse: document.querySelector("#mouse"),
  outcome: document.querySelector("#outcome"),
  viewer: document.querySelector("#viewer"),
  queue: document.querySelector("#queue"),
  progress: document.querySelector("#progress"),
  title: document.querySelector("#slide-title"),
  meta: document.querySelector("#slide-meta"),
  chromogen: document.querySelector("#chromogen"),
  scaleMaximum: document.querySelector("#scale-maximum"),
  familySummary: document.querySelector("#family-summary"),
  metrics: document.querySelector("#metrics"),
  reasons: document.querySelector("#reasons"),
  issue: document.querySelector("#known-issue"),
  badge: document.querySelector("#priority-badge"),
  threshold: document.querySelector("#threshold-status"),
  notes: document.querySelector("#notes"),
  outputOverlayLabel: document.querySelector("#output-overlay-label"),
  outputMapLabel: document.querySelector("#output-map-label"),
};
const checks = [...document.querySelectorAll("[data-check]")];
const decisionButtons = [...document.querySelectorAll("[data-decision]")];
const filterButtons = [...document.querySelectorAll("[data-filter]")];
const images = [...document.querySelectorAll("[data-image]")];
const stages = [...document.querySelectorAll(".image-stage")];
const viewerNodes = [...document.querySelectorAll("[data-viewer]")];
let currentMouse = null;
let currentSlide = null;
let currentFilter = "priority";
let zoom = {scale: 1, x: 0, y: 0};
let drag = null;
let noteTimer = null;
let liveRequest = 0;
let synchronizing = false;
const viewers = new Map();
let openSeadragonLoad = null;
const provisionalFingerprints = new Map();
const chromogens = {
  "h-dab": {label: "brown DAB", color: "#8a5b36", stops: ["#faf8f2", "#d3ae70", "#8b5b2d", "#4c2b15"]},
  "sirius-red": {label: "red collagen", color: "#bf3030", stops: ["#fff8f6", "#f6b1a2", "#d3463b", "#7d151a"]},
  "pas": {label: "magenta PAS", color: "#b12f75", stops: ["#fff7fc", "#eea7d2", "#bd4192", "#671753"]},
  "alcian-blue": {label: "blue mucin", color: "#2872a5", stops: ["#f7fbff", "#9ecae1", "#3182bd", "#0e3d6b"]},
};

function storageKey(mouse) {
  return `histopia-stain-review-v1:${mouse.id}:${mouse.fingerprint}`;
}

function loadDraft(mouse) {
  try {
    const value = JSON.parse(localStorage.getItem(storageKey(mouse)) || "{}");
    if (value && typeof value === "object" && Object.keys(value).length) return value;
    // Preserve owner-entered concerns across immutable result fingerprints.
    // Old accepts are deliberately not carried forward because the displayed
    // quantitative output changed and must be reviewed again.
    const prefix = `histopia-stain-review-v1:${mouse.id}:`;
    const concerns = {};
    for (let index = 0; index < localStorage.length; index += 1) {
      const key = localStorage.key(index);
      if (!key?.startsWith(prefix) || key === storageKey(mouse)) continue;
      const prior = JSON.parse(localStorage.getItem(key) || "{}");
      Object.entries(prior).forEach(([slideId, draft]) => {
        if (!["hold", "reject"].includes(draft?.decision)) return;
        const current = concerns[slideId];
        if (!current || draft.decision === "reject") concerns[slideId] = draft;
      });
    }
    if (Object.keys(concerns).length) {
      localStorage.setItem(storageKey(mouse), JSON.stringify(concerns));
    }
    return concerns;
  } catch {
    return {};
  }
}

function saveDraft() {
  localStorage.setItem(storageKey(currentMouse), JSON.stringify(currentMouse.draft));
}

function slideDraft(slide) {
  if (!currentMouse.draft[slide.id]) {
    currentMouse.draft[slide.id] = {decision: "", checks: {}, note: ""};
  }
  return currentMouse.draft[slide.id];
}

function setMouse(mouseId, preferredSlide = null) {
  currentMouse = manifest.mice.find((mouse) => mouse.id === mouseId) || manifest.mice[0];
  currentMouse.draft = loadDraft(currentMouse);
  elements.mouse.value = currentMouse.id;
  elements.viewer.href = `${manifest.viewer_href}?mouse=${encodeURIComponent(currentMouse.id)}`;
  const fromUrl = new URLSearchParams(location.search).get("slide");
  currentSlide = currentMouse.slides.find((slide) => String(slide.order) === String(preferredSlide || fromUrl))
    || orderedSlides()[0]
    || currentMouse.slides[0];
  updateUrl();
  render();
  const requestedMouse = currentMouse;
  loadProvisional(requestedMouse).then((changed) => {
    if (changed && currentMouse === requestedMouse) render();
  }).catch(() => {});
}

async function loadProvisional(mouse) {
  const response = await fetch(
    histopiaUrl(`/api/reviews/provisional?cohort=${encodeURIComponent(mouse.id)}&stage=stain`),
    {cache: "no-store"},
  );
  if (!response.ok) return false;
  const payload = await response.json();
  if (payload.fingerprint !== mouse.fingerprint) return false;
  provisionalFingerprints.set(mouse.id, payload.fingerprint);
  Object.values(payload.feedback || {}).forEach((record) => {
    const slide = mouse.slides.find((row) => row.id === record.slide_id);
    if (!slide) return;
    mouse.draft[slide.id] = {
      decision: record.decision || "",
      checks: record.checks || {},
      note: record.comment || "",
    };
  });
  localStorage.setItem(storageKey(mouse), JSON.stringify(mouse.draft));
  return Object.keys(payload.feedback || {}).length > 0;
}

async function persistProvisional(slide) {
  const draft = slideDraft(slide);
  if (!draft.decision || !provisionalFingerprints.has(currentMouse.id)) return;
  const response = await fetch(histopiaUrl("/api/reviews/provisional"), {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
      cohort: currentMouse.id,
      stage: "stain",
      fingerprint: provisionalFingerprints.get(currentMouse.id),
      slide_id: slide.id,
      decision: draft.decision,
      labels: [],
      checks: draft.checks,
      comment: draft.note,
      reviewer: "browser-provisional",
    }),
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.error || "Provisional review could not be saved");
  }
}

function orderedSlides() {
  return [...currentMouse.slides].sort((left, right) => {
    if (currentFilter === "all") return left.order - right.order;
    return Number(right.priority.blocking) - Number(left.priority.blocking)
      || right.priority.score - left.priority.score
      || left.order - right.order;
  });
}

function visibleSlides() {
  const rows = orderedSlides();
  if (currentFilter === "priority") return rows.filter((slide) => slide.priority.required);
  if (currentFilter === "unreviewed") return rows.filter((slide) => !slideDraft(slide).decision);
  if (currentFilter === "concern") {
    return rows.filter((slide) => ["hold", "reject"].includes(slideDraft(slide).decision));
  }
  return rows;
}

function render() {
  renderQueue();
  renderSlide();
  renderOutcome();
}

function renderQueue() {
  elements.queue.replaceChildren();
  const rows = visibleSlides();
  rows.forEach((slide) => {
    const draft = slideDraft(slide);
    const item = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.className = slide.id === currentSlide.id ? "active" : "";
    button.innerHTML = `
      <span class="order">${String(slide.order).padStart(2, "0")}</span>
      <span class="label">${escapeHtml(slide.label)}
        <small>${escapeHtml(slide.family)}${slide.priority.required ? " | priority" : ""}</small>
      </span>
      <span class="decision-dot ${draft.decision}" aria-label="${draft.decision || "unreviewed"}"></span>`;
    button.addEventListener("click", () => {
      currentSlide = slide;
      updateUrl();
      render();
    });
    item.append(button);
    elements.queue.append(item);
  });
  if (!rows.length) {
    const item = document.createElement("li");
    item.textContent = "No slides in this view";
    item.style.padding = "14px";
    elements.queue.append(item);
  }
  const required = currentMouse.slides.filter((slide) => slide.priority.required);
  const reviewed = required.filter((slide) => slideDraft(slide).decision).length;
  elements.progress.textContent =
    `${reviewed}/${required.length} priority reviewed | ${currentMouse.slides.length} quantified`;
}

function renderSlide() {
  const slide = currentSlide;
  const draft = slideDraft(slide);
  elements.title.textContent = `${String(slide.order).padStart(2, "0")} ${slide.label}`;
  const method = currentMouse.families[slide.family]?.selected_method || "unknown";
  elements.meta.textContent = `${slide.family} | ${method} vectors | ${slide.id}`;
  const chromogen = chromogens[slide.family] || {label: slide.family, color: "#777"};
  elements.chromogen.textContent = chromogen.label;
  elements.chromogen.style.setProperty("--chromogen", chromogen.color);
  document.querySelector("#color-scale i").style.background =
    `linear-gradient(90deg, ${(chromogen.stops || chromogens["h-dab"].stops).join(",")})`;
  elements.scaleMaximum.textContent = currentMouse.display_max_od.toFixed(2);
  const family = currentMouse.families[slide.family];
  elements.familySummary.textContent = family
    ? `${slide.family}: ${family.selected_method} vectors | correction ${family.correction_accepted}/${family.slide_count} | binary threshold ${family.threshold_accepted}/${family.slide_count}`
    : "";
  const adaptiveAccepted = slide.qc.adaptive_background?.accepted === true;
  const adaptiveMethod = slide.qc.adaptive_background?.method || "";
  const physicalLabel = slide.qc.correction_accepted
    ? "corrected target OD"
    : "raw target OD";
  const outputLabel = adaptiveAccepted
    ? (adaptiveMethod === "counterstain-conditioned-v3"
      ? "Counterstain-conditioned target OD (validated 4 µm/px)"
      : `Adaptive ${physicalLabel} (validated 4 µm/px)`)
    : `${physicalLabel[0].toUpperCase()}${physicalLabel.slice(1)} fallback`;
  elements.outputOverlayLabel.textContent = `${outputLabel} + histology`;
  elements.outputMapLabel.textContent = `${outputLabel} only`;
  images.forEach((image) => {
    const kind = image.dataset.image;
    image.classList.add("loading");
    image.classList.remove("error");
    image.alt = `${slide.label}: ${kind.replaceAll("_", " ")}`;
    image.onload = () => {
      image.classList.remove("loading", "error");
      applyZoom();
    };
    image.onerror = () => {
      image.classList.remove("loading");
      image.classList.add("error");
    };
    image.src = slide.assets[kind];
  });
  loadNativePanes(slide, ++liveRequest);
  resetZoom();
  renderMetrics(slide);
  elements.reasons.replaceChildren();
  slide.priority.reasons.forEach((reason) => {
    const item = document.createElement("li");
    item.textContent = reason;
    elements.reasons.append(item);
  });
  elements.issue.textContent = slide.known_issue || "";
  elements.issue.classList.toggle("visible", Boolean(slide.known_issue));
  elements.badge.textContent = slide.priority.required ? "Priority review" : "Routine";
  elements.badge.className = slide.priority.required ? "required" : "";
  checks.forEach((input) => {
    input.checked = Boolean(draft.checks[input.dataset.check]);
  });
  decisionButtons.forEach((button) => {
    button.classList.toggle("active", button.dataset.decision === draft.decision);
  });
  elements.notes.value = draft.note || "";
  elements.threshold.textContent = slide.qc.threshold_accepted
    ? "Threshold fit passed its stability checks."
    : "Threshold fit was unstable; no binary call should be used.";
}

function metricRow(label, value, state = "") {
  const row = document.createElement("div");
  const term = document.createElement("dt");
  const description = document.createElement("dd");
  term.textContent = label;
  description.textContent = value;
  description.className = state;
  row.append(term, description);
  return row;
}

function renderMetrics(slide) {
  const qc = slide.qc;
  const leakageChange = qc.raw_glass_leakage > 0
    ? 100 * (qc.raw_glass_leakage - qc.corrected_glass_leakage) / qc.raw_glass_leakage
    : 0;
  const counterstainChange = qc.raw_counterstain_leakage > 0
    ? 100 * (qc.raw_counterstain_leakage - qc.corrected_counterstain_leakage)
      / qc.raw_counterstain_leakage
    : 0;
  const cvChange = qc.background_cv_before > 0
    ? 100 * (qc.background_cv_before - qc.background_cv_after) / qc.background_cv_before
    : 0;
  elements.metrics.replaceChildren(
    metricRow(
      "Correction gate",
      qc.correction_accepted ? "Accepted correction" : "Rejected -> raw fallback",
      qc.correction_accepted ? "pass" : "warn",
    ),
    metricRow(
      "Adaptive background",
      qc.adaptive_background
        ? (qc.adaptive_background.accepted
          ? (qc.adaptive_background.method === "counterstain-conditioned-v3"
            ? `Accepted v3; nuisance ${qc.adaptive_background.minimum_nuisance_od.toFixed(3)}–${qc.adaptive_background.maximum_nuisance_od.toFixed(3)} OD`
            : `Accepted; floor ${qc.adaptive_background.floor_od.toFixed(3)} OD`)
          : "Rejected -> physical OD fallback")
        : "Off",
      qc.adaptive_background?.accepted ? "pass" : "warn",
    ),
    metricRow(
      "Rank preservation",
      qc.rank_correlation.toFixed(4),
      qc.rank_correlation >= 0.98 ? "pass" : "fail",
    ),
    metricRow(
      "Candidate leakage",
      `${qc.raw_glass_leakage.toFixed(3)} -> ${qc.corrected_glass_leakage.toFixed(3)}`,
      qc.corrected_glass_leakage <= qc.raw_glass_leakage ? "pass" : "fail",
    ),
    metricRow(
      "Candidate reduction",
      `${signed(leakageChange)}%`,
      leakageChange >= 0 ? "pass" : "fail",
    ),
    metricRow(
      "Counterstain-only leakage",
      `${qc.raw_counterstain_leakage.toFixed(3)} -> ${qc.corrected_counterstain_leakage.toFixed(3)}`,
      counterstainChange >= -5 ? "pass" : "fail",
    ),
    metricRow(
      "Background CV",
      `${qc.background_cv_before.toFixed(3)} -> ${qc.background_cv_after.toFixed(3)}`,
      cvChange >= 0 ? "pass" : "warn",
    ),
    metricRow("Median residual", qc.reconstruction_residual.toFixed(4)),
    metricRow("Median / q95 OD", `${quantile(slide, "0.5")} / ${quantile(slide, "0.95")}`),
  );
}

function renderOutcome() {
  const required = currentMouse.slides.filter((slide) => slide.priority.required);
  const decisions = required.map((slide) => slideDraft(slide).decision);
  const blockers = currentMouse.summary.blocking_issues;
  let text;
  if (blockers) {
    text = `${currentMouse.id}: blocked by ${blockers} known upstream issue${blockers === 1 ? "" : "s"}`;
  } else if (decisions.includes("reject")) {
    text = `${currentMouse.id}: draft reject`;
  } else if (decisions.includes("hold")) {
    text = `${currentMouse.id}: draft hold`;
  } else if (decisions.every((decision) => decision === "accept")) {
    text = `${currentMouse.id}: priority set supports continuous OD approval`;
  } else {
    const remaining = decisions.filter((decision) => !decision).length;
    text = `${currentMouse.id}: ${remaining} priority slide${remaining === 1 ? "" : "s"} unresolved`;
  }
  elements.outcome.textContent = text;
}

function setDecision(decision) {
  const draft = slideDraft(currentSlide);
  draft.decision = draft.decision === decision ? "" : decision;
  saveDraft();
  persistProvisional(currentSlide).catch((error) => {
    document.body.dataset.reviewError = String(error);
  });
  render();
}

function selectAdjacent(direction, priorityOnly = false) {
  let rows = priorityOnly
    ? orderedSlides().filter((slide) => slide.priority.required && !slideDraft(slide).decision)
    : visibleSlides();
  if (!rows.length) rows = orderedSlides();
  const index = rows.findIndex((slide) => slide.id === currentSlide.id);
  const nextIndex = index < 0 ? 0 : (index + direction + rows.length) % rows.length;
  currentSlide = rows[nextIndex];
  updateUrl();
  render();
}

function updateUrl() {
  const url = new URL(location.href);
  url.searchParams.set("mouse", currentMouse.id);
  url.searchParams.set("slide", currentSlide.order);
  history.replaceState(null, "", url);
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

function loadOpenSeadragon() {
  if (typeof globalThis.OpenSeadragon === "function") return Promise.resolve();
  if (openSeadragonLoad) return openSeadragonLoad;
  openSeadragonLoad = new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = "openseadragon.min.js";
    script.onload = resolve;
    script.onerror = reject;
    document.head.append(script);
  });
  return openSeadragonLoad;
}

function ensureViewer(name) {
  if (typeof globalThis.OpenSeadragon !== "function") {
    throw new Error("OpenSeadragon is unavailable");
  }
  if (viewers.has(name)) return viewers.get(name);
  const node = viewerNodes.find((row) => row.dataset.viewer === name);
  const viewer = OpenSeadragon({
    element: node,
    drawer: navigator.webdriver ? ["html"] : ["canvas"],
    showNavigator: name === "histology",
    navigatorPosition: "BOTTOM_RIGHT",
    showNavigationControl: false,
    animationTime: 0.2,
    blendTime: 0.05,
    imageLoaderLimit: 1,
    imageSmoothingEnabled: name !== "corrected",
    immediateRender: true,
    // The map-only pane is intentionally allowed to magnify its validated
    // 4 µm/px pixels so its physical viewport can stay locked to native WSI
    // panes. Disabling smoothing above makes that coarser measurement support
    // visible instead of implying native-resolution quantification.
    maxZoomPixelRatio: name === "corrected" ? 96 : 4,
    preserveViewport: false,
    visibilityRatio: 0.2,
  });
  viewer.addHandler("tile-load-failed", () => {
    if (viewer.histopiaRequest === liveRequest) useFallback(liveRequest);
  });
  viewers.set(name, viewer);
  node.histopiaViewer = viewer;
  return viewer;
}

function enableViewportSync() {
  viewers.forEach((viewer) => {
    if (viewer.histopiaSyncEnabled) return;
    const synchronize = () => requestAnimationFrame(() => syncViewports(viewer));
    viewer.addHandler("canvas-drag", synchronize);
    viewer.addHandler("canvas-scroll", synchronize);
    viewer.addHandler("canvas-pinch", synchronize);
    viewer.addHandler("canvas-double-click", synchronize);
    viewer.histopiaSyncEnabled = true;
  });
}

function syncViewports(source) {
  if (synchronizing || !document.body.classList.contains("live-tiles")) return;
  synchronizing = true;
  try {
    // Canvas input updates the source viewport's target before its animation
    // reaches the current value. Copying the current value leaves every
    // follower one animation frame behind and produces visible bounce during
    // repeated wheel or pinch gestures.
    const center = source.viewport.getCenter(false);
    const zoomLevel = source.viewport.getZoom(false);
    viewers.forEach((viewer) => {
      if (viewer === source || !viewer.world.getItemCount()) return;
      viewer.viewport.panTo(center, true);
      viewer.viewport.zoomTo(zoomLevel, null, true);
    });
  } finally {
    synchronizing = false;
  }
}

function addLayer(viewer, metadata, name, opacity, request) {
  return new Promise((resolve, reject) => viewer.addTiledImage({
    tileSource: tileSource(metadata, name),
    opacity,
    success: (event) => request === liveRequest ? resolve(event) : reject(
      new Error("Superseded stain pane request")),
    error: reject,
  }));
}

function openBase(viewer, metadata, name, request) {
  return new Promise((resolve, reject) => {
    const opened = (event) => {
      cleanup();
      if (request === liveRequest) resolve(event);
      else reject(new Error("Superseded stain pane request"));
    };
    const failed = (event) => { cleanup(); reject(event); };
    const cleanup = () => {
      viewer.removeHandler("open", opened);
      viewer.removeHandler("open-failed", failed);
    };
    viewer.addHandler("open", opened);
    viewer.addHandler("open-failed", failed);
    viewer.open(tileSource(metadata, name));
  });
}

function waitForTiles(viewer, request) {
  return new Promise((resolve, reject) => {
    let timer;
    const ready = new Set();
    const check = (event) => {
      if (request !== liveRequest) {
        cleanup();
        reject(new Error("Superseded stain pane request"));
        return;
      }
      const count = viewer.world.getItemCount();
      const eventItem = event?.tiledImage || event?.item;
      if (eventItem) ready.add(eventItem);
      for (let index = 0; index < count; index += 1) {
        const item = viewer.world.getItemAt(index);
        if (item.getFullyLoaded()) ready.add(item);
      }
      if (count && ready.size >= count) {
        cleanup();
        resolve();
      }
    };
    const failed = () => {
      cleanup();
      reject(new Error("Stain pane tile failed"));
    };
    const cleanup = () => {
      clearTimeout(timer);
      viewer.removeHandler("fully-loaded-change", check);
      viewer.removeHandler("tile-loaded", check);
      viewer.removeHandler("tile-load-failed", failed);
    };
    viewer.addHandler("fully-loaded-change", check);
    viewer.addHandler("tile-loaded", check);
    viewer.addHandler("tile-load-failed", failed);
    timer = setTimeout(() => {
      cleanup();
      resolve();
    }, 60_000);
    check();
  });
}

async function loadPane(name, metadata, layers, request) {
  const viewer = ensureViewer(name);
  viewer.histopiaRequest = request;
  const [[base], ...overlays] = layers;
  await openBase(viewer, metadata, base, request);
  for (const [layer, opacity] of overlays) {
    await addLayer(viewer, metadata, layer, opacity, request);
  }
  viewer.viewport.goHome(true);
}

async function loadNativePanes(slide, request) {
  useFallback(request);
  try {
    const catalogResponse = await fetch(
      histopiaUrl(`/api/wsi/${encodeURIComponent(currentMouse.id)}`),
      {cache: "no-store"},
    );
    if (!catalogResponse.ok) throw new Error("WSI catalog is unavailable");
    const catalog = await catalogResponse.json();
    if (!catalog.sections?.some((row) => row.section === slide.section)) {
      throw new Error("Native stain tiles are not configured for this slide");
    }
    const response = await fetch(
      histopiaUrl(`/api/wsi/${encodeURIComponent(currentMouse.id)}/${slide.section}`),
      {cache: "no-store"},
    );
    if (!response.ok) throw new Error(`WSI metadata failed (${response.status})`);
    const metadata = await response.json();
    if (request !== liveRequest) return;
    if (metadata.slide !== slide.id) throw new Error("WSI slide identity differs");
    for (const layer of ["raw", "stain_raw", "stain_corrected"]) {
      if (!metadata.layers?.[layer]) throw new Error(`WSI layer is missing: ${layer}`);
    }
    await loadOpenSeadragon();
    if (request !== liveRequest) return;
    const raw = metadata.layers.raw;
    const outputLayer = metadata.layers.stain_adaptive_v3
      ? "stain_adaptive_v3" : (metadata.layers.stain_output
        ? "stain_output" : "stain_corrected");
    const outputMapLayer = metadata.layers.stain_adaptive_v3_map
      ? "stain_adaptive_v3_map" : (metadata.layers.stain_output_map
        ? "stain_output_map" : outputLayer);
    stages.forEach((stage) => {
      stage.classList.add("live", "pane-loading");
      stage.classList.remove("fallback", "pane-ready", "pane-error");
    });
    await Promise.all([
      loadPane("histology", metadata, [["raw", 1]], request),
      loadPane(
        "raw_overlay",
        metadata,
        [["raw", 1], ["stain_raw", 0.72]],
        request,
      ),
      loadPane(
        "corrected_overlay",
        metadata,
        [["raw", 1], [outputLayer, 0.72]],
        request,
      ),
      loadPane("corrected", metadata, [[outputMapLayer, 1]], request),
    ]);
    if (request !== liveRequest) return;
    document.body.classList.add("live-tiles");
    delete document.body.dataset.tileError;
    enableViewportSync();
    const primary = viewers.get("histology");
    primary.viewport.goHome(true);
    syncViewports(primary);
    stages.forEach((stage) => {
      const viewer = viewers.get(stage.dataset.pane);
      waitForTiles(viewer, request).then(() => {
        if (request !== liveRequest) return;
        stage.classList.remove("pane-loading");
        stage.classList.add("pane-ready");
      }).catch(() => {
        if (request === liveRequest) stage.classList.add("pane-error");
      });
    });
    const output = metadata.layers[outputLayer];
    elements.meta.textContent += ` | native ${raw.width.toLocaleString()} × ` +
      `${raw.height.toLocaleString()} px; OD ${metadata.layers.stain_raw.analysis_mpp} µm/px` +
      `; output ${output.selected_source || (slide.qc.correction_accepted ? "corrected" : "raw")}`;
  } catch (error) {
    if (request === liveRequest) {
      document.body.dataset.tileError = String(error);
      stages.forEach((stage) => stage.classList.add("pane-error"));
      useFallback(request);
    }
  }
}

function useFallback(request = liveRequest) {
  if (request !== liveRequest) return;
  document.body.classList.remove("live-tiles");
  stages.forEach((stage) => {
    stage.classList.remove("live", "pane-loading", "pane-ready");
    stage.classList.add("fallback");
  });
}

function setZoom(scale, x = zoom.x, y = zoom.y) {
  zoom = {scale: Math.min(8, Math.max(1, scale)), x, y};
  if (zoom.scale === 1) zoom = {scale: 1, x: 0, y: 0};
  applyZoom();
}

function resetZoom() {
  zoom = {scale: 1, x: 0, y: 0};
  applyZoom();
}

function applyZoom() {
  if (document.body.classList.contains("live-tiles")) return;
  images.forEach((image) => {
    image.style.transform = `translate(${zoom.x}px, ${zoom.y}px) scale(${zoom.scale})`;
  });
}

function signed(value) {
  return `${value >= 0 ? "+" : ""}${value.toFixed(1)}`;
}

function quantile(slide, key) {
  return Number.isFinite(slide.quantiles[key]) ? slide.quantiles[key].toFixed(3) : "n/a";
}

function escapeHtml(value) {
  const span = document.createElement("span");
  span.textContent = value;
  return span.innerHTML;
}

elements.mouse.replaceChildren(...manifest.mice.map((mouse) => {
  const option = document.createElement("option");
  option.value = mouse.id;
  option.textContent = mouse.id;
  return option;
}));
elements.mouse.addEventListener("change", () => setMouse(elements.mouse.value));
filterButtons.forEach((button) => button.addEventListener("click", () => {
  currentFilter = button.dataset.filter;
  filterButtons.forEach((row) => row.setAttribute("aria-pressed", String(row === button)));
  if (!visibleSlides().some((slide) => slide.id === currentSlide.id)) {
    currentSlide = visibleSlides()[0] || currentMouse.slides[0];
  }
  render();
}));
decisionButtons.forEach((button) => {
  button.addEventListener("click", () => setDecision(button.dataset.decision));
});
checks.forEach((input) => input.addEventListener("change", () => {
  slideDraft(currentSlide).checks[input.dataset.check] = input.checked;
  saveDraft();
  persistProvisional(currentSlide).catch(() => {});
}));
elements.notes.addEventListener("input", () => {
  clearTimeout(noteTimer);
  const slide = currentSlide;
  const value = elements.notes.value;
  noteTimer = setTimeout(() => {
    slideDraft(slide).note = value.trim();
    saveDraft();
    persistProvisional(slide).catch(() => {});
  }, 180);
});
document.querySelector("#previous").addEventListener("click", () => selectAdjacent(-1));
document.querySelector("#next").addEventListener("click", () => selectAdjacent(1, true));
document.querySelectorAll("[data-zoom]").forEach((button) => {
  button.addEventListener("click", () => {
    const direction = Number(button.dataset.zoom);
    const primary = viewers.get("histology");
    if (document.body.classList.contains("live-tiles") && primary) {
      if (!direction) primary.viewport.goHome(true);
      else primary.viewport.zoomBy(
        direction > 0 ? 1.35 : 1 / 1.35,
        null,
        true,
      );
      primary.viewport.applyConstraints(true);
      syncViewports(primary);
    } else if (!direction) resetZoom();
    else setZoom(zoom.scale * (direction > 0 ? 1.35 : 1 / 1.35));
  });
});
stages.forEach((stage) => {
  stage.addEventListener("wheel", (event) => {
    if (document.body.classList.contains("live-tiles")) return;
    event.preventDefault();
    setZoom(zoom.scale * (event.deltaY < 0 ? 1.18 : 1 / 1.18));
  }, {passive: false});
  stage.addEventListener("dblclick", resetZoom);
  stage.addEventListener("pointerdown", (event) => {
    if (document.body.classList.contains("live-tiles")) return;
    if (zoom.scale <= 1) return;
    drag = {pointer: event.pointerId, x: event.clientX, y: event.clientY, ox: zoom.x, oy: zoom.y};
    stage.setPointerCapture(event.pointerId);
    stages.forEach((row) => row.classList.add("dragging"));
  });
  stage.addEventListener("pointermove", (event) => {
    if (!drag || drag.pointer !== event.pointerId) return;
    setZoom(zoom.scale, drag.ox + event.clientX - drag.x, drag.oy + event.clientY - drag.y);
  });
  stage.addEventListener("pointerup", () => {
    drag = null;
    stages.forEach((row) => row.classList.remove("dragging"));
  });
});
document.querySelector("#details-toggle").addEventListener("click", (event) => {
  const open = document.body.classList.toggle("details-open");
  event.currentTarget.setAttribute("aria-pressed", String(open));
});
document.querySelector("#export").addEventListener("click", () => {
  const payload = {
    schema_version: 1,
    mouse_id: currentMouse.id,
    stain_fingerprint: currentMouse.fingerprint,
    decision_scope: manifest.scope.decision,
    slides: currentMouse.slides.map((slide) => ({
      slide_id: slide.id,
      order: slide.order,
      label: slide.label,
      required: slide.priority.required,
      ...slideDraft(slide),
    })),
  };
  const blob = new Blob([`${JSON.stringify(payload, null, 2)}\n`], {type: "application/json"});
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `histopia-stain-review-${currentMouse.id}.json`;
  link.click();
  URL.revokeObjectURL(link.href);
});
document.addEventListener("keydown", (event) => {
  if (event.target.matches("textarea,input,select")) return;
  if (event.key === "ArrowLeft") selectAdjacent(-1);
  if (event.key === "ArrowRight") selectAdjacent(1);
});

const params = new URLSearchParams(location.search);
setMouse(params.get("mouse") || manifest.mice[0].id, params.get("slide"));

// Keep same-origin data requests inside a code-server port proxy.
function histopiaUrl(path) {
  const match = location.pathname.match(/^.*?\/proxy\/[0-9]+(?=\/|$)/);
  return (match ? match[0] : "") + path;
}
