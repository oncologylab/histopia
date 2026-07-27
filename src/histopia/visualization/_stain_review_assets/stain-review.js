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
let currentMouse = null;
let currentSlide = null;
let currentFilter = "priority";
let zoom = {scale: 1, x: 0, y: 0};
let drag = null;
let noteTimer = null;
const chromogens = {
  "h-dab": {label: "brown DAB", color: "#8a5b36"},
  "sirius-red": {label: "red collagen", color: "#bf3030"},
  "pas": {label: "magenta PAS", color: "#b12f75"},
  "alcian-blue": {label: "blue mucin", color: "#2872a5"},
};

function storageKey(mouse) {
  return `histopia-stain-review-v1:${mouse.id}:${mouse.fingerprint}`;
}

function loadDraft(mouse) {
  try {
    const value = JSON.parse(localStorage.getItem(storageKey(mouse)) || "{}");
    return value && typeof value === "object" ? value : {};
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
  elements.scaleMaximum.textContent = currentMouse.display_max_od.toFixed(2);
  const family = currentMouse.families[slide.family];
  elements.familySummary.textContent = family
    ? `${slide.family}: ${family.selected_method} vectors | correction ${family.correction_accepted}/${family.slide_count} | binary threshold ${family.threshold_accepted}/${family.slide_count}`
    : "";
  const outputLabel = slide.qc.correction_accepted
    ? "Accepted corrected target OD"
    : "Raw target OD fallback";
  elements.outputOverlayLabel.textContent = `${outputLabel} + histology`;
  elements.outputMapLabel.textContent = `${outputLabel} only`;
  images.forEach((image) => {
    const kind = image.dataset.image;
    image.className = "loading";
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
}));
elements.notes.addEventListener("input", () => {
  clearTimeout(noteTimer);
  const slide = currentSlide;
  const value = elements.notes.value;
  noteTimer = setTimeout(() => {
    slideDraft(slide).note = value.trim();
    saveDraft();
  }, 180);
});
document.querySelector("#previous").addEventListener("click", () => selectAdjacent(-1));
document.querySelector("#next").addEventListener("click", () => selectAdjacent(1, true));
document.querySelectorAll("[data-zoom]").forEach((button) => {
  button.addEventListener("click", () => {
    const direction = Number(button.dataset.zoom);
    if (!direction) resetZoom();
    else setZoom(zoom.scale * (direction > 0 ? 1.35 : 1 / 1.35));
  });
});
stages.forEach((stage) => {
  stage.addEventListener("wheel", (event) => {
    event.preventDefault();
    setZoom(zoom.scale * (event.deltaY < 0 ? 1.18 : 1 / 1.18));
  }, {passive: false});
  stage.addEventListener("dblclick", resetZoom);
  stage.addEventListener("pointerdown", (event) => {
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
