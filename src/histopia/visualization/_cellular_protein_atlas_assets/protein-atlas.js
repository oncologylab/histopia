import * as THREE from "./vendor/three.module.min.js";
import { OrbitControls } from "./vendor/OrbitControls.js";

const manifest = globalThis.HISTOPIA_CELLULAR_PROTEIN_ATLAS;
const el = (id) => document.getElementById(id);
const app = document.querySelector(".app");
const viewport = el("viewport");
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x02070e);

if (!manifest || ![1, 2, 3].includes(manifest.schema_version)) {
  throw new Error("Unsupported cellular protein atlas manifest");
}

let cohort = null;
let layer = manifest.default?.layer || "protein";
let mode = manifest.default?.mode || "3d";
let expression = "predicted";
let depthMode = manifest.default?.depth_mode || "adaptive";
let zScale = 2;
let adaptiveZScale = 2;
let adaptiveCellDiameterUm = 10;
let adaptiveSectionSpacingUm = 10;
let adaptiveJitterUm = 2.8;
let adaptiveDepthP10Um = 7;
let adaptiveDepthP90Um = 14;
let adaptivePhaseRms = 0.5;
let generation = 0;
let colorGeneration = 0;
let contextRecovery = 0;
let framePending = false;
let renderer = null;
let controls = null;
let perspective = null;
let root = null;
let envelope = null;
let locator = null;
let selection = null;
let selectionGeneration = 0;
let pointerStart = null;
let orthoDrag = null;
let worker = null;
let activeTargets = [];
let selectedSection = "";

const EXPLODED_Z_SCALE = 25;
const MORPHOLOGY_SAMPLE_LIMIT = 8192;

const sectionMeshes = new Map();
const geometryCache = new Map();
const colorCache = new Map();
const valueChunkCache = new Map();
const detailSections = new Set();
const compatibilityControls = new Map();
const clipping = new THREE.Plane(new THREE.Vector3(1, 0, 0), 1e12);
const projectionStates = {
  top: { zoom: 1, u: 0, v: 0 },
  front: { zoom: 1, u: 0, v: 0 },
  side: { zoom: 1, u: 0, v: 0 },
  section: { zoom: 1, u: 0, v: 0 },
};

function webglAvailable() {
  const probe = document.createElement("canvas");
  let context = null;
  try {
    context = probe.getContext("webgl2") || probe.getContext("webgl");
    if (!context) return false;
    const info = context.getExtension("WEBGL_debug_renderer_info");
    const device = info ? context.getParameter(info.UNMASKED_RENDERER_WEBGL) : "";
    // The full atlas can stall CPU graphics; retain its prepared preview there.
    return !/swiftshader|llvmpipe|softpipe|software rasterizer/i.test(device);
  } catch (_error) {
    return false;
  } finally {
    context?.getExtension("WEBGL_lose_context")?.loseContext();
  }
}

function fallbackAsset() {
  return (
    cohort?.fallback_previews?.[layer]?.asset ||
    cohort?.fallback_preview?.asset ||
    ""
  );
}

function updateFallbackPreview() {
  const image = el("compatibility").querySelector("img");
  image.src = fallbackAsset();
  image.alt =
    layer === "cells"
      ? "Static categorical section-cell atlas preview"
      : "Static default-protein orthogonal atlas preview";
}

function setCompatibilityControls(locked) {
  const selectors = "aside select:not(#cohort), aside input, aside button, " +
    "[data-mode], [data-view], [data-z]";
  for (const control of document.querySelectorAll(selectors)) {
    if (locked) {
      if (!compatibilityControls.has(control))
        compatibilityControls.set(control, {disabled: control.disabled, title: control.title});
      control.disabled = true;
      control.title = "Requires interactive WebGL";
    } else if (compatibilityControls.has(control)) {
      Object.assign(control, compatibilityControls.get(control));
      compatibilityControls.delete(control);
    }
  }
}

function showCompatibility(
  message = "Interactive WebGL is unavailable in this browser.",
) {
  if (renderer) {
    renderer.dispose?.();
    renderer.domElement?.remove();
  }
  renderer = null;
  viewport.dataset.renderer = "fallback";
  // The prepared image contains the default targets only. Its legend and
  // controls must never suggest a different marker or section was rendered.
  activeTargets = [...(cohort.default_targets || [cohort.default_target])];
  expression = "predicted";
  el("composite").value = "dominant";
  populateTargets();
  updateFallbackPreview();
  updateProvenance();
  setCompatibilityControls(true);
  el("compatibility").hidden = false;
  el("compatibility").querySelector("strong").textContent = "Static atlas preview";
  el("compatibility").querySelector("span").textContent = message;
  setLoading("");
}

function createRenderer() {
  if (!webglAvailable()) {
    showCompatibility();
    return false;
  }
  try {
    renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: false,
      powerPreference: "high-performance",
      preserveDrawingBuffer: true,
    });
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.localClippingEnabled = true;
    renderer.autoClear = false;
    viewport.prepend(renderer.domElement);
    viewport.dataset.renderer = "webgl";
    setCompatibilityControls(false);
    el("compatibility").hidden = true;
    bindCanvasInteractions(renderer.domElement);
    renderer.domElement.addEventListener("webglcontextlost", (event) => {
      event.preventDefault();
      if (contextRecovery++ < 1) {
        const state = cameraState();
        setTimeout(() => recoverContext(state), 80);
      } else {
        showCompatibility(
          "The WebGL context could not be restored. A fingerprinted preview is shown instead.",
        );
      }
    });
    return true;
  } catch (_error) {
    showCompatibility();
    return false;
  }
}

function recoverContext(state) {
  renderer?.dispose();
  renderer?.domElement?.remove();
  renderer = null;
  if (!createRenderer()) return;
  bindControls();
  resize();
  restoreCameraState(state);
  requestRender(3);
}

function bindControls() {
  controls?.dispose?.();
  if (!renderer) return;
  controls = new OrbitControls(perspective, renderer.domElement);
  controls.enableDamping = false;
  controls.screenSpacePanning = true;
  controls.enabled = mode === "3d";
  controls.addEventListener("change", () => requestRender(2));
}

function bindCanvasInteractions(canvas) {
  canvas.addEventListener("pointerdown", (event) => {
    pointerStart = [event.clientX, event.clientY];
    if (mode === "3d") return;
    const pointer = projectionForPointer(event);
    const state = projectionStates[pointer.stateKey];
    orthoDrag = {
      id: event.pointerId,
      x: event.clientX,
      y: event.clientY,
      u: state.u,
      v: state.v,
      state,
      camera: pointer.camera,
      width: pointer.width,
      height: pointer.height,
    };
    canvas.setPointerCapture?.(event.pointerId);
  });
  canvas.addEventListener("pointermove", (event) => {
    if (!orthoDrag || orthoDrag.id !== event.pointerId) return;
    const dx = event.clientX - orthoDrag.x;
    const dy = event.clientY - orthoDrag.y;
    const horizontal = orthoDrag.camera.right - orthoDrag.camera.left;
    const vertical = orthoDrag.camera.top - orthoDrag.camera.bottom;
    orthoDrag.state.u = orthoDrag.u - (dx / orthoDrag.width) * horizontal;
    orthoDrag.state.v = orthoDrag.v + (dy / orthoDrag.height) * vertical;
    requestRender(1);
  });
  const finishPointer = (event) => {
    const moved = pointerStart
      ? Math.hypot(event.clientX - pointerStart[0], event.clientY - pointerStart[1])
      : Infinity;
    if (orthoDrag?.id === event.pointerId) {
      canvas.releasePointerCapture?.(event.pointerId);
      orthoDrag = null;
    }
    if (moved < 5) selectCell(event);
    pointerStart = null;
  };
  canvas.addEventListener("pointerup", finishPointer);
  canvas.addEventListener("pointercancel", () => {
    pointerStart = null;
    orthoDrag = null;
  });
  canvas.addEventListener(
    "wheel",
    (event) => {
      if (mode === "3d") return;
      event.preventDefault();
      const pointer = projectionForPointer(event);
      const state = projectionStates[pointer.stateKey];
      state.zoom = THREE.MathUtils.clamp(
        state.zoom * Math.exp(-event.deltaY * 0.0014),
        0.45,
        40,
      );
      requestRender(2);
    },
    { passive: false },
  );
  canvas.addEventListener("dblclick", (event) => {
    if (mode === "3d") return;
    event.preventDefault();
    const pointer = projectionForPointer(event);
    resetProjectionState(pointer.stateKey);
    requestRender(2);
  });
}

function requestRender(frames = 1) {
  if (!renderer) return;
  requestRender.frames = Math.max(requestRender.frames || 0, frames);
  if (framePending) return;
  framePending = true;
  requestAnimationFrame(drawFrame);
}

function drawFrame() {
  framePending = false;
  if (!renderer) return;
  renderer.setScissorTest(false);
  renderer.setViewport(0, 0, viewport.clientWidth, viewport.clientHeight);
  renderer.clear(true, true, true);
  if (mode === "orthogonal") {
    renderOrthogonal();
  } else if (mode === "section") {
    const width = Math.max(viewport.clientWidth, 1);
    const height = Math.max(viewport.clientHeight, 1);
    renderer.render(
      scene,
      orthographicCamera("top", width / height, true),
    );
  } else {
    renderer.render(scene, perspective);
  }
  if ((requestRender.frames || 0) > 0) {
    requestRender.frames -= 1;
    requestRender(0);
  }
}

function orthogonalPanels(width, height) {
  const split = width < 780 ? 0.66 : 0.68;
  const leftWidth = Math.max(1, Math.floor(width * split));
  const rightWidth = Math.max(1, width - leftWidth);
  const half = Math.max(1, Math.floor(height / 2));
  return [
    { view: "top", stateKey: "top", left: 0, top: 0, width: leftWidth, height },
    {
      view: "front",
      stateKey: "front",
      left: leftWidth,
      top: 0,
      width: rightWidth,
      height: half,
    },
    {
      view: "side",
      stateKey: "side",
      left: leftWidth,
      top: half,
      width: rightWidth,
      height: height - half,
    },
  ];
}

function renderOrthogonal() {
  const width = Math.max(viewport.clientWidth, 1);
  const height = Math.max(viewport.clientHeight, 1);
  const panels = orthogonalPanels(width, height);
  renderer.setScissorTest(true);
  for (const panel of panels) {
    setCellProjection(panel.view === "front" ? 1 : panel.view === "side" ? 2 : 0);
    const camera = orthographicCamera(
      panel.view,
      panel.width / Math.max(panel.height, 1),
      false,
    );
    panel.camera = camera;
    const bottom = height - panel.top - panel.height;
    renderer.setViewport(panel.left, bottom, panel.width, panel.height);
    renderer.setScissor(panel.left + 1, bottom + 1, panel.width - 2, panel.height - 2);
    renderer.render(scene, camera);
  }
  setCellProjection(0);
  renderer.setScissorTest(false);
  renderer.setViewport(0, 0, width, height);
  updateScaleBars(panels);
}

function setCellProjection(value) {
  for (const mesh of sectionMeshes.values()) {
    mesh.material.uniforms.projectionMode.value = value;
  }
}

function worldBounds() {
  const bounds = cohort.bounds_um;
  const x = (bounds.x[0] + bounds.x[1]) / 2;
  const y = (bounds.y[0] + bounds.y[1]) / 2;
  const z = (bounds.z[0] + bounds.z[1]) / 2;
  const depthMargin = depthMode === "adaptive"
    ? adaptiveCellDiameterUm * 1.7 + adaptiveJitterUm * 2
    : Math.max(Number(cohort.section_thickness_um) || 5, 1);
  return {
    center: new THREE.Vector3(0, 0, 0),
    size: new THREE.Vector3(
      bounds.x[1] - bounds.x[0],
      (bounds.z[1] - bounds.z[0]) * zScale + depthMargin,
      bounds.y[1] - bounds.y[0],
    ),
    physicalCenter: { x, y, z },
  };
}

function sectionWorldBounds() {
  const row = cohort.sections.find((value) => value.id === selectedSection);
  const global = worldBounds().physicalCenter;
  const bounds = row.bounds_um;
  const x = (bounds.x[0] + bounds.x[1]) / 2 - global.x;
  const y = (bounds.y[0] + bounds.y[1]) / 2 - global.y;
  const z = (row.z_um - global.z) * zScale;
  const sectionDepth = depthMode === "adaptive"
    ? adaptiveCellDiameterUm * 1.7 + adaptiveJitterUm * 2
    : Math.max(Number(cohort.section_thickness_um) || 5, 1);
  return {
    center: new THREE.Vector3(x, z, -y),
    size: new THREE.Vector3(
      bounds.x[1] - bounds.x[0],
      sectionDepth,
      bounds.y[1] - bounds.y[0],
    ),
    physicalCenter: global,
  };
}

function orthographicCamera(view, aspect, sectionOnly) {
  const bounds = sectionOnly ? sectionWorldBounds() : worldBounds();
  const stateKey = sectionOnly ? "section" : view;
  const state = projectionStates[stateKey];
  const horizontal = view === "side" ? bounds.size.z : bounds.size.x;
  const vertical = view === "top" ? bounds.size.z : bounds.size.y;
  const half =
    (Math.max(vertical, horizontal / Math.max(aspect, 0.001), 1) * 0.54) /
    state.zoom;
  const camera = new THREE.OrthographicCamera(
    -half * aspect,
    half * aspect,
    half,
    -half,
    -1e7,
    1e7,
  );
  const center = bounds.center.clone();
  if (view === "top") {
    center.x += state.u;
    center.z -= state.v;
    camera.position.set(center.x, center.y + 1e6, center.z + 0.001);
    camera.up.set(0, 0, -1);
  } else if (view === "front") {
    center.x += state.u;
    center.y += state.v;
    camera.position.set(center.x, center.y, center.z + 1e6);
    camera.up.set(0, 1, 0);
  } else {
    center.z -= state.u;
    center.y += state.v;
    camera.position.set(center.x + 1e6, center.y, center.z);
    camera.up.set(0, 1, 0);
  }
  camera.lookAt(center);
  camera.updateProjectionMatrix();
  return camera;
}

function resetProjectionState(key) {
  const state = projectionStates[key];
  state.zoom = 1;
  state.u = 0;
  state.v = 0;
}

function resetProjectionStates() {
  Object.keys(projectionStates).forEach(resetProjectionState);
}

function niceScale(value) {
  const power = 10 ** Math.floor(Math.log10(Math.max(value, 1e-9)));
  const normalized = value / power;
  const step = normalized >= 5 ? 5 : normalized >= 2 ? 2 : 1;
  return step * power;
}

function updateScaleBars(panels) {
  for (const panel of panels) {
    const row = document.querySelector(`[data-panel="${panel.view}"]`);
    const bar = row?.querySelector("i");
    const label = row?.querySelector("em");
    if (!bar || !label || !panel.camera) continue;
    const span = panel.camera.right - panel.camera.left;
    const scale = niceScale(span * 0.18);
    bar.style.width = `${Math.max(34, (scale / span) * panel.width)}px`;
    label.textContent =
      scale >= 1000
        ? `${(scale / 1000).toLocaleString(undefined, { maximumFractionDigits: 1 })} mm`
        : `${Math.round(scale)} µm`;
  }
}

function resize() {
  if (!renderer) return;
  const rect = viewport.getBoundingClientRect();
  const width = Math.max(rect.width, 1);
  const height = Math.max(rect.height, 1);
  const pixelRatio = Math.min(
    devicePixelRatio,
    2,
    Math.sqrt(12_000_000 / Math.max(width * height, 1)),
  );
  renderer.setPixelRatio(Math.max(pixelRatio, 1));
  renderer.setSize(width, height, false);
  perspective.aspect = width / height;
  perspective.updateProjectionMatrix();
  requestRender(2);
}

function cameraState() {
  return perspective && controls
    ? { position: perspective.position.toArray(), target: controls.target.toArray() }
    : null;
}

function restoreCameraState(state) {
  if (!state || !controls) return fit("home");
  perspective.position.fromArray(state.position);
  controls.target.fromArray(state.target);
  controls.update();
}

function fit(view = "home") {
  if (!cohort || !perspective || !controls) return;
  if (mode !== "3d") {
    if (mode === "section") resetProjectionState("section");
    else resetProjectionStates();
    requestRender(2);
    return;
  }
  const bounds = worldBounds();
  const radius = Math.max(bounds.size.length() / 2, 1);
  const vertical = THREE.MathUtils.degToRad(perspective.fov);
  const horizontal = 2 * Math.atan(Math.tan(vertical / 2) * perspective.aspect);
  const distance =
    (0.84 * radius) / Math.sin(Math.max(Math.min(vertical, horizontal) / 2, 0.05));
  let direction;
  if (view === "top") direction = new THREE.Vector3(0, 1, 0.001);
  else if (view === "front") direction = new THREE.Vector3(0, 0.04, 1);
  else if (view === "side") direction = new THREE.Vector3(1, 0.04, 0);
  else {
    direction = (
      depthMode === "adaptive"
        ? new THREE.Vector3(0.86, 0.34, 1)
        : new THREE.Vector3(0.86, 0.7, 1)
    ).normalize();
  }
  controls.target.set(0, 0, 0);
  perspective.position.copy(direction.multiplyScalar(distance));
  perspective.near = Math.max(radius / 5000, 0.01);
  perspective.far = distance + radius * 10;
  perspective.updateProjectionMatrix();
  controls.update();
  recenterProjectedEnvelope();
  requestRender(3);
}

function recenterProjectedEnvelope() {
  if (!envelope || !controls) return;
  root.updateMatrixWorld(true);
  perspective.updateMatrixWorld(true);
  const position = envelope.geometry.attributes.position;
  if (!position) return;
  const world = new THREE.Vector3();
  const projected = new THREE.Vector3();
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (let index = 0; index < position.count; index += 1) {
    world.fromBufferAttribute(position, index).applyMatrix4(envelope.matrixWorld);
    projected.copy(world).project(perspective);
    minX = Math.min(minX, projected.x);
    maxX = Math.max(maxX, projected.x);
    minY = Math.min(minY, projected.y);
    maxY = Math.max(maxY, projected.y);
  }
  if (!Number.isFinite(minX)) return;
  const ndcX = (minX + maxX) / 2;
  const ndcY = (minY + maxY) / 2;
  if (Math.max(Math.abs(ndcX), Math.abs(ndcY)) < 0.01) return;
  const distance = perspective.position.distanceTo(controls.target);
  const vertical = Math.tan(THREE.MathUtils.degToRad(perspective.fov) / 2);
  const horizontal = vertical * perspective.aspect;
  const right = new THREE.Vector3().setFromMatrixColumn(perspective.matrixWorld, 0);
  const up = new THREE.Vector3().setFromMatrixColumn(perspective.matrixWorld, 1);
  const shift = right
    .multiplyScalar(ndcX * horizontal * distance)
    .add(up.multiplyScalar(ndcY * vertical * distance));
  perspective.position.add(shift);
  controls.target.add(shift);
  controls.update();
}

function parseGeometry(buffer) {
  const view = new DataView(buffer);
  const magic = String.fromCharCode(...new Uint8Array(buffer, 0, 5));
  const kind = view.getUint8(5);
  if (magic !== "HCPA1" || ![1, 2].includes(kind) || view.getUint8(6) !== 1) {
    throw new Error("Invalid HCPA1 cell geometry");
  }
  return kind === 2 ? parseBoundaryGeometry(buffer, view) : parseLegacyGeometry(buffer, view);
}

function splitProfileAttributes(values, count, sectors) {
  const groups = [];
  for (let group = 0; group < sectors / 4; group += 1) {
    const output = new Uint8Array(count * 4);
    for (let index = 0; index < count; index += 1) {
      const source = index * sectors + group * 4;
      output.set(values.subarray(source, source + 4), index * 4);
    }
    groups.push(output);
  }
  return groups;
}

function sampledQuantile(values, quantile) {
  if (!values.length) return 1;
  const stride = Math.max(1, Math.ceil(values.length / MORPHOLOGY_SAMPLE_LIMIT));
  const sample = [];
  for (let index = 0; index < values.length; index += stride) {
    const value = Number(values[index]);
    if (Number.isFinite(value) && value > 0) sample.push(value);
  }
  sample.sort((left, right) => left - right);
  if (!sample.length) return 1;
  const position = Math.max(0, Math.min(sample.length - 1, (sample.length - 1) * quantile));
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  const fraction = position - lower;
  return sample[lower] * (1 - fraction) + sample[upper] * fraction;
}

function morphologyPhase(label, x, y, z) {
  let hash = (label >>> 0) ^ Math.imul(Math.round(x * 2), 73856093);
  hash ^= Math.imul(Math.round(y * 2), 19349663);
  hash ^= Math.imul(Math.round(z * 2), 83492791);
  hash = Math.imul(hash ^ (hash >>> 16), 0x45d9f3b);
  hash = Math.imul(hash ^ (hash >>> 16), 0x45d9f3b);
  hash ^= hash >>> 16;
  return (hash >>> 0) / 2147483647.5 - 1;
}

function deriveCellMorphology(geometry) {
  const count = geometry.count;
  const sectors = geometry.sectors;
  const diameters = new Float32Array(count);
  const compactness = new Float32Array(count);
  const rawPhases = new Float32Array(count);
  const linear = geometry.nativeLinear;
  const areaRadiusScale = Math.sqrt(
    Math.max(Math.abs(linear[0] * linear[3] - linear[1] * linear[2]), 1e-8),
  );
  for (let index = 0; index < count; index += 1) {
    let squared = 0;
    for (let sector = 0; sector < sectors; sector += 1) {
      const fraction = geometry.profiles[index * sectors + sector] / 255;
      squared += fraction * fraction;
    }
    const radialCompactness = Math.sqrt(squared / sectors);
    compactness[index] = radialCompactness;
    diameters[index] = Math.max(
      0.5,
      2 * geometry.maximumRadii[index] * radialCompactness * areaRadiusScale,
    );
    rawPhases[index] = morphologyPhase(
      geometry.labels[index],
      geometry.positions[index * 3],
      geometry.positions[index * 3 + 1],
      geometry.positions[index * 3 + 2],
    );
  }

  const medianDiameter = sampledQuantile(diameters, 0.5);
  const gridSize = Math.max(medianDiameter * 1.8, 2);
  const bins = new Map();
  for (let index = 0; index < count; index += 1) {
    const column = Math.floor(geometry.positions[index * 3] / gridSize);
    const row = Math.floor(geometry.positions[index * 3 + 1] / gridSize);
    const key = `${column}:${row}`;
    const values = bins.get(key) || [0, 0, 0, 0];
    values[0] += 1;
    values[1] += diameters[index];
    values[2] += compactness[index];
    values[3] += rawPhases[index];
    bins.set(key, values);
  }

  const cellDepths = new Float32Array(count);
  const cellZPhases = new Float32Array(count);
  for (let index = 0; index < count; index += 1) {
    const column = Math.floor(geometry.positions[index * 3] / gridSize);
    const row = Math.floor(geometry.positions[index * 3 + 1] / gridSize);
    let neighbors = 0;
    let diameterSum = 0;
    let compactnessSum = 0;
    let phaseSum = 0;
    for (let dx = -1; dx <= 1; dx += 1) {
      for (let dy = -1; dy <= 1; dy += 1) {
        const values = bins.get(`${column + dx}:${row + dy}`);
        if (!values) continue;
        neighbors += values[0];
        diameterSum += values[1];
        compactnessSum += values[2];
        phaseSum += values[3];
      }
    }
    const localDiameter = diameterSum / Math.max(neighbors, 1);
    const localCompactness = compactnessSum / Math.max(neighbors, 1);
    const localPhase = phaseSum / Math.max(neighbors, 1);
    const relativeDiameter = diameters[index] / Math.max(localDiameter, 0.5);
    const edgeSlice = THREE.MathUtils.clamp((1.12 - relativeDiameter) / 0.55, 0, 1);
    const shapeContrast = Math.abs(compactness[index] - localCompactness);
    const balancedPhase = THREE.MathUtils.clamp(
      rawPhases[index] - 0.6 * localPhase,
      -1,
      1,
    );
    cellZPhases[index] = THREE.MathUtils.clamp(
      balancedPhase * (0.55 + 0.45 * edgeSlice) * (1 + Math.min(shapeContrast, 0.25)),
      -1,
      1,
    );
    const localScale = THREE.MathUtils.clamp(localDiameter / medianDiameter - 1, -0.5, 0.5);
    const depthFactor = 0.72 + 0.38 * compactness[index] + 0.08 * localScale;
    cellDepths[index] = THREE.MathUtils.clamp(
      diameters[index] * depthFactor,
      medianDiameter * 0.55,
      medianDiameter * 1.65,
    );
  }
  geometry.cellDepths = cellDepths;
  geometry.cellZPhases = cellZPhases;
  geometry.medianDiameterUm = medianDiameter;
  geometry.diameterP10Um = sampledQuantile(diameters, 0.1);
  geometry.diameterP90Um = sampledQuantile(diameters, 0.9);
  geometry.depthP10Um = sampledQuantile(cellDepths, 0.1);
  geometry.depthP90Um = sampledQuantile(cellDepths, 0.9);
  geometry.zPhaseRms = Math.sqrt(
    cellZPhases.reduce((sum, value) => sum + value * value, 0) / Math.max(count, 1),
  );
  return geometry;
}

function parseBoundaryGeometry(buffer, view) {
  const count = view.getUint32(8, true);
  const header = view.getUint32(12, true);
  if (header < 160) throw new Error("Invalid HCPA1 boundary header");
  const x0 = view.getFloat64(16, true);
  const y0 = view.getFloat64(24, true);
  const xs = view.getFloat64(32, true);
  const ys = view.getFloat64(40, true);
  const radiusMaximum = view.getFloat32(48, true);
  const z = view.getFloat32(52, true);
  const sectors = view.getUint16(56, true);
  if (![8, 16].includes(sectors) || !(radiusMaximum > 0)) {
    throw new Error("Invalid HCPA1 boundary profile");
  }
  const transform = Array.from(
    { length: 9 },
    (_value, index) => view.getFloat64(64 + index * 8, true),
  );
  if (Math.abs(transform[6]) > 1e-10 || Math.abs(transform[7]) > 1e-10 || Math.abs(transform[8]) < 1e-12) {
    throw new Error("HCPA1 boundary transform must be affine");
  }
  let offset = header;
  const labels = new Uint32Array(buffer, offset, count);
  offset += count * 4;
  const qx = new Uint16Array(buffer, offset, count);
  offset += count * 2;
  const qy = new Uint16Array(buffer, offset, count);
  offset += count * 2;
  const qmax = new Uint16Array(buffer, offset, count);
  offset += count * 2;
  const profiles = new Uint8Array(buffer, offset, count * sectors);
  offset += count * sectors;
  const packedAngles = new Uint8Array(buffer, offset, count * sectors / 2);
  if (offset + packedAngles.byteLength !== buffer.byteLength) {
    throw new Error("Invalid HCPA1 boundary payload length");
  }
  const angleFractions = new Uint8Array(count * sectors);
  const positions = new Float32Array(count * 3);
  const maximumRadii = new Float32Array(count);
  const center = worldBounds().physicalCenter;
  for (let index = 0; index < count; index += 1) {
    const nativeX = x0 + (qx[index] / 65535) * xs;
    const nativeY = y0 + (qy[index] / 65535) * ys;
    const denominator = transform[6] * nativeX + transform[7] * nativeY + transform[8];
    const referenceX = (transform[0] * nativeX + transform[1] * nativeY + transform[2]) / denominator;
    const referenceY = (transform[3] * nativeX + transform[4] * nativeY + transform[5]) / denominator;
    positions[3 * index] = referenceX - center.x;
    positions[3 * index + 1] = referenceY - center.y;
    positions[3 * index + 2] = z - center.z;
    maximumRadii[index] = (qmax[index] / 65535) * radiusMaximum;
    const packedStart = index * sectors / 2;
    const angleStart = index * sectors;
    for (let sector = 0; sector < sectors; sector += 1) {
      const packed = packedAngles[packedStart + Math.floor(sector / 2)];
      const nibble = sector % 2 ? packed >>> 4 : packed & 15;
      angleFractions[angleStart + sector] = Math.round(((nibble + 0.5) / 16) * 255);
    }
  }
  const scale = transform[8];
  return deriveCellMorphology({
    kind: 2,
    boundaryDerived: true,
    count,
    labels,
    positions,
    sectors,
    profiles,
    angleFractions,
    profileGroups: splitProfileAttributes(profiles, count, sectors),
    angleGroups: splitProfileAttributes(angleFractions, count, sectors),
    maximumRadii,
    nativeLinear: [
      transform[0] / scale,
      transform[1] / scale,
      transform[3] / scale,
      transform[4] / scale,
    ],
  });
}

function parseLegacyGeometry(buffer, view) {
  const count = view.getUint32(8, true);
  const header = view.getUint32(12, true);
  const x0 = view.getFloat64(16, true);
  const y0 = view.getFloat64(24, true);
  const xs = view.getFloat64(32, true);
  const ys = view.getFloat64(40, true);
  const area0 = view.getFloat32(48, true);
  const areaSpan = view.getFloat32(52, true);
  const z = view.getFloat32(56, true);
  const sectors = view.getUint8(7) & 1 ? 8 : 16;
  let offset = header;
  const labels = new Uint32Array(buffer, offset, count);
  offset += count * 4;
  const qx = new Uint16Array(buffer, offset, count);
  offset += count * 2;
  const qy = new Uint16Array(buffer, offset, count);
  offset += count * 2;
  const qarea = new Uint16Array(buffer, offset, count);
  offset += count * 2;
  const qeccentricity = new Uint8Array(buffer, offset, count);
  offset += count;
  const qorientation = new Uint8Array(buffer, offset, count);
  const positions = new Float32Array(count * 3);
  const maximumRadii = new Float32Array(count);
  const profiles = new Uint8Array(count * sectors);
  const angleFractions = new Uint8Array(count * sectors);
  const center = worldBounds().physicalCenter;
  for (let index = 0; index < count; index += 1) {
    positions[3 * index] = x0 + (qx[index] / 65535) * xs - center.x;
    positions[3 * index + 1] = y0 + (qy[index] / 65535) * ys - center.y;
    positions[3 * index + 2] = z - center.z;
    const area = Math.max(Math.expm1(area0 + (qarea[index] / 65535) * areaSpan), 1);
    const eccentricity = qeccentricity[index] / 255;
    const minorRatio = Math.sqrt(Math.max(1 - eccentricity * eccentricity, 0.04));
    const equivalentRadius = Math.sqrt(area / Math.PI);
    const major = equivalentRadius / Math.sqrt(minorRatio);
    const minor = equivalentRadius * Math.sqrt(minorRatio);
    const orientation = (qorientation[index] / 255) * Math.PI * 2 - Math.PI;
    maximumRadii[index] = major;
    for (let sector = 0; sector < sectors; sector += 1) {
      const theta = ((sector + 0.5) / sectors) * Math.PI * 2;
      const relative = theta - orientation;
      const radius = (major * minor) / Math.sqrt(
        (minor * Math.cos(relative)) ** 2 + (major * Math.sin(relative)) ** 2,
      );
      profiles[index * sectors + sector] = Math.max(1, Math.round((radius / major) * 255));
      angleFractions[index * sectors + sector] = 128;
    }
  }
  return deriveCellMorphology({
    kind: 1,
    boundaryDerived: false,
    count,
    labels,
    positions,
    sectors,
    profiles,
    angleFractions,
    profileGroups: splitProfileAttributes(profiles, count, sectors),
    angleGroups: splitProfileAttributes(angleFractions, count, sectors),
    maximumRadii,
    nativeLinear: [1, 0, 0, 1],
  });
}

function profileAccessor(prefix, sectors) {
  const components = ["x", "y", "z", "w"];
  return Array.from({ length: sectors }, (_value, index) => {
    const group = Math.floor(index / 4);
    return `if(slot<${(index + 0.5).toFixed(1)})return ${prefix}${group}.${components[index % 4]};`;
  }).join("");
}

function lensGeometry(sectors) {
  const vertices = [];
  const lowerRing = 0;
  for (let sector = 0; sector < sectors; sector += 1) {
    vertices.push(sector, 1, -0.34);
  }
  const upperRing = sectors;
  for (let sector = 0; sector < sectors; sector += 1) {
    vertices.push(sector, 1, 0.34);
  }
  const top = sectors * 2;
  const bottom = top + 1;
  vertices.push(0, 0, 1, 0, 0, -1);
  const indices = [];
  for (let sector = 0; sector < sectors; sector += 1) {
    const next = (sector + 1) % sectors;
    const lower = lowerRing + sector;
    const lowerNext = lowerRing + next;
    const upper = upperRing + sector;
    const upperNext = upperRing + next;
    indices.push(
      bottom, lowerNext, lower,
      lower, lowerNext, upperNext,
      lower, upperNext, upper,
      top, upper, upperNext,
    );
  }
  return {
    vertices: new Float32Array(vertices),
    indices: new Uint16Array(indices),
  };
}

function cellMaterial(geometry, section) {
  const groups = geometry.sectors / 4;
  const declarations = Array.from(
    { length: groups },
    (_value, index) => `attribute vec4 cellRadius${index};attribute vec4 cellAngle${index};`,
  ).join("");
  return new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: true,
    depthTest: true,
    side: THREE.DoubleSide,
    blending: THREE.NormalBlending,
    clippingPlanes: [clipping],
    uniforms: {
      glyphScale: { value: +el("glyph-size").value / 100 },
      zScale: { value: zScale },
      alphaScale: { value: 1 },
      identityMode: { value: layer === "cells" ? 1 : 0 },
      projectionMode: { value: 0 },
      nativeLinear: { value: new THREE.Vector4(...geometry.nativeLinear) },
      sectionThickness: { value: Math.max(+cohort.section_thickness_um || 5, 0.1) },
      adaptiveDepth: { value: depthMode === "adaptive" ? 1 : 0 },
      zJitterAmplitude: { value: adaptiveJitterUm },
    },
    vertexShader: `
      attribute vec3 cellPosition;
      attribute float cellMaximumRadius;
      attribute float cellDepth;
      attribute float cellZPhase;
      attribute vec4 cellColor;
      ${declarations}
      varying vec4 vColor;
      varying vec3 vNormal;
      uniform float glyphScale;
      uniform float zScale;
      uniform float projectionMode;
      uniform vec4 nativeLinear;
      uniform float sectionThickness;
      uniform float adaptiveDepth;
      uniform float zJitterAmplitude;
      float profileRadius(float slot){${profileAccessor("cellRadius", geometry.sectors)}return 1.0;}
      float profileAngle(float slot){${profileAccessor("cellAngle", geometry.sectors)}return .5;}
      void main(){
        vColor=cellColor;
        float slot=position.x;
        float theta=(slot+profileAngle(slot))*6.28318530718/${geometry.sectors.toFixed(1)};
        float radius=profileRadius(slot)*cellMaximumRadius*glyphScale*position.y;
        vec2 nativeOffset=vec2(cos(theta),sin(theta))*radius;
        vec2 referenceOffset=vec2(
          nativeLinear.x*nativeOffset.x+nativeLinear.y*nativeOffset.y,
          nativeLinear.z*nativeOffset.x+nativeLinear.w*nativeOffset.y
        );
        float displayDepth=mix(sectionThickness,cellDepth,adaptiveDepth);
        float displayZ=cellPosition.z*zScale+cellZPhase*zJitterAmplitude*adaptiveDepth;
        vec3 center=vec3(cellPosition.x,displayZ,-cellPosition.y);
        vec3 offset=vec3(referenceOffset.x,position.z*displayDepth*.5,-referenceOffset.y);
        vec3 radial=normalize(vec3(referenceOffset.x,0.0,-referenceOffset.y)+vec3(1e-8));
        vNormal=normalize(mix(vec3(0.0,position.z,0.0),radial,position.y));
        vec4 mvPosition=modelViewMatrix*vec4(center+offset,1.0);
        gl_Position=projectionMatrix*mvPosition;
      }`,
    fragmentShader: `
      varying vec4 vColor;
      varying vec3 vNormal;
      uniform float alphaScale;
      uniform float identityMode;
      void main(){
        if(vColor.a<=0.0)discard;
        vec3 normal=normalize(vNormal);
        float light=.48+.52*max(dot(normal,normalize(vec3(-.42,.67,.74))),0.0);
        float identityShade=mix(light,1.0,0.18);
        float proteinShade=.84+.16*light;
        vec3 color=vColor.rgb*mix(proteinShade,identityShade,identityMode);
        gl_FragColor=vec4(color,vColor.a*alphaScale);
      }`,
  });
}

function makeMesh(geometry, section, lod) {
  const prepared = new THREE.InstancedBufferGeometry();
  const lens = lensGeometry(geometry.sectors);
  prepared.setAttribute("position", new THREE.BufferAttribute(lens.vertices, 3));
  prepared.setIndex(new THREE.BufferAttribute(lens.indices, 1));
  prepared.setAttribute(
    "cellPosition",
    new THREE.InstancedBufferAttribute(geometry.positions, 3),
  );
  prepared.setAttribute(
    "cellMaximumRadius",
    new THREE.InstancedBufferAttribute(geometry.maximumRadii, 1),
  );
  prepared.setAttribute(
    "cellDepth",
    new THREE.InstancedBufferAttribute(geometry.cellDepths, 1),
  );
  prepared.setAttribute(
    "cellZPhase",
    new THREE.InstancedBufferAttribute(geometry.cellZPhases, 1),
  );
  geometry.profileGroups.forEach((values, index) => {
    prepared.setAttribute(
      `cellRadius${index}`,
      new THREE.InstancedBufferAttribute(values, 4, true),
    );
    prepared.setAttribute(
      `cellAngle${index}`,
      new THREE.InstancedBufferAttribute(geometry.angleGroups[index], 4, true),
    );
  });
  prepared.setAttribute(
    "cellColor",
    new THREE.InstancedBufferAttribute(new Uint8Array(geometry.count * 4), 4, true),
  );
  prepared.instanceCount = geometry.count;
  const mesh = new THREE.Mesh(prepared, cellMaterial(geometry, section));
  mesh.frustumCulled = false;
  mesh.renderOrder = cohort.sections.findIndex((row) => row.id === section.id);
  mesh.userData = { section: section.id, lod, geometry };
  root.add(mesh);
  return mesh;
}

async function fetchGeometry(section, lod, loadId) {
  const key = `${cohort.id}:${section.id}:${lod}`;
  if (geometryCache.has(key)) return geometryCache.get(key);
  const entry = lod === "detail" ? section.geometry : section.overview_geometry;
  const response = await fetch(entry.asset, { cache: "force-cache" });
  if (!response.ok) throw new Error(`Could not load cell geometry ${section.id}`);
  const parsed = parseGeometry(await response.arrayBuffer());
  if (loadId !== generation) return null;
  geometryCache.set(key, parsed);
  return parsed;
}

async function mapLimit(items, limit, task) {
  let next = 0;
  const runners = Array.from(
    { length: Math.min(limit, items.length) },
    async () => {
      while (next < items.length) {
        const index = next++;
        await task(items[index], index);
      }
    },
  );
  await Promise.all(runners);
}

function clearScene() {
  clearSelection();
  for (const mesh of sectionMeshes.values()) {
    root.remove(mesh);
    mesh.geometry.dispose();
    mesh.material.dispose();
  }
  sectionMeshes.clear();
  detailSections.clear();
  geometryCache.clear();
  colorCache.clear();
  valueChunkCache.clear();
  worker?.terminate();
  worker = null;
  if (envelope) {
    root.remove(envelope);
    envelope.geometry.dispose();
    envelope.material.dispose();
    envelope = null;
  }
  removeLocator();
}

function physicalSectionSpacing() {
  const positions = cohort.sections
    .map((section) => Number(section.z_um))
    .filter(Number.isFinite)
    .sort((left, right) => left - right);
  const gaps = [];
  for (let index = 1; index < positions.length; index += 1) {
    const gap = positions[index] - positions[index - 1];
    if (gap > 0) gaps.push(gap);
  }
  return gaps.length
    ? sampledQuantile(gaps, 0.5)
    : Math.max(Number(cohort.section_thickness_um) || 5, 0.1);
}

function depthScaleForMode() {
  if (depthMode === "physical") return 1;
  if (depthMode === "exploded") return EXPLODED_Z_SCALE;
  return adaptiveZScale;
}

function resetAdaptiveDepth() {
  const physicalGap = physicalSectionSpacing();
  adaptiveCellDiameterUm = Number(
    cohort.cell_identity?.morphology_aware_z?.median_cell_diameter_um,
  ) || Math.max(physicalGap * 2, 8);
  adaptiveSectionSpacingUm = THREE.MathUtils.clamp(
    adaptiveCellDiameterUm * 1.05,
    physicalGap * 1.1,
    physicalGap * 3.2,
  );
  adaptiveZScale = adaptiveSectionSpacingUm / physicalGap;
  adaptiveJitterUm = Math.min(
    adaptiveSectionSpacingUm * 0.3,
    adaptiveCellDiameterUm * 0.34,
  );
  adaptiveDepthP10Um = adaptiveCellDiameterUm * 0.7;
  adaptiveDepthP90Um = adaptiveCellDiameterUm * 1.4;
  adaptivePhaseRms = 0.5;
  zScale = depthScaleForMode();
}

function weightedMedian(rows) {
  const ordered = rows
    .filter((row) => Number.isFinite(row.value) && row.value > 0 && row.weight > 0)
    .sort((left, right) => left.value - right.value);
  const total = ordered.reduce((sum, row) => sum + row.weight, 0);
  let cumulative = 0;
  for (const row of ordered) {
    cumulative += row.weight;
    if (cumulative >= total / 2) return row.value;
  }
  return ordered.at(-1)?.value || adaptiveCellDiameterUm;
}

function applyDepthModel() {
  zScale = depthScaleForMode();
  for (const mesh of sectionMeshes.values()) {
    mesh.material.uniforms.zScale.value = zScale;
    mesh.material.uniforms.adaptiveDepth.value = depthMode === "adaptive" ? 1 : 0;
    mesh.material.uniforms.zJitterAmplitude.value = adaptiveJitterUm;
  }
  if (envelope) envelope.scale.y = zScale;
  viewport.dataset.depthMode = depthMode;
  viewport.dataset.zScale = zScale.toFixed(3);
  viewport.dataset.cellDiameterUm = adaptiveCellDiameterUm.toFixed(2);
  viewport.dataset.sectionSpacingUm = adaptiveSectionSpacingUm.toFixed(2);
  viewport.dataset.cellDepthP10Um = adaptiveDepthP10Um.toFixed(2);
  viewport.dataset.cellDepthP90Um = adaptiveDepthP90Um.toFixed(2);
  viewport.dataset.zPhaseRms = adaptivePhaseRms.toFixed(3);
  document.querySelectorAll("[data-z]").forEach((button) => {
    button.classList.toggle("active", button.dataset.z === depthMode);
    button.setAttribute("aria-pressed", String(button.dataset.z === depthMode));
  });
  updateLocator();
  updateProvenance();
  requestRender(3);
}

function configureAdaptiveDepth() {
  const rows = [];
  for (const mesh of sectionMeshes.values()) {
    if (mesh.userData.lod !== "overview") continue;
    rows.push({
      value: mesh.userData.geometry.medianDiameterUm,
      weight: mesh.userData.geometry.count,
      geometry: mesh.userData.geometry,
    });
  }
  const sharedModel = cohort.cell_identity?.morphology_aware_z;
  adaptiveCellDiameterUm = Number(sharedModel?.median_cell_diameter_um)
    || weightedMedian(rows);
  adaptiveDepthP10Um = weightedMedian(
    rows.map((row) => ({
      value: row.geometry.depthP10Um,
      weight: row.weight,
    })),
  );
  adaptiveDepthP90Um = weightedMedian(
    rows.map((row) => ({
      value: row.geometry.depthP90Um,
      weight: row.weight,
    })),
  );
  adaptivePhaseRms = weightedMedian(
    rows.map((row) => ({
      value: row.geometry.zPhaseRms,
      weight: row.weight,
    })),
  );
  const physicalGap = physicalSectionSpacing();
  adaptiveSectionSpacingUm = Number(sharedModel?.visual_section_spacing_um)
    || THREE.MathUtils.clamp(
      adaptiveCellDiameterUm * 1.05,
      physicalGap * 1.1,
      physicalGap * 3.2,
    );
  adaptiveZScale = Number(sharedModel?.visual_z_scale)
    || adaptiveSectionSpacingUm / physicalGap;
  adaptiveJitterUm = Math.min(
    adaptiveSectionSpacingUm * 0.3,
    adaptiveCellDiameterUm * 0.34,
  );
  applyDepthModel();
}

async function loadCohort(id, useDefaults = false) {
  const loadId = ++generation;
  colorGeneration += 1;
  clearScene();
  cohort = manifest.cohorts.find((row) => row.id === id) || manifest.cohorts[0];
  resetAdaptiveDepth();
  const validTargets = new Set(cohort.targets.map((row) => row.id));
  activeTargets = activeTargets.filter((target) => validTargets.has(target));
  if (useDefaults || !activeTargets.length) {
    activeTargets = [...(cohort.default_targets || [cohort.default_target])];
  }
  if (
    useDefaults ||
    !cohort.sections.some((section) => section.id === selectedSection)
  ) {
    selectedSection =
      cohort.default_section || cohort.sections[Math.floor(cohort.sections.length / 2)].id;
  }
  el("cohort").value = cohort.id;
  populateSections();
  populateTargets();
  updateProvenance();
  updateFallbackPreview();
  if (!renderer) {
    showCompatibility();
    updateUrl();
    return;
  }
  setLoading("Loading fingerprinted section-cell overview…");
  await loadEnvelope(loadId);
  await mapLimit(cohort.sections, 4, async (section) => {
    const geometry = await fetchGeometry(section, "overview", loadId);
    if (!geometry || loadId !== generation) return;
    sectionMeshes.set(
      `overview:${section.id}`,
      makeMesh(geometry, section, "overview"),
    );
  });
  if (loadId !== generation) return;
  configureAdaptiveDepth();
  await refreshColors(loadId);
  updateMaterialMode();
  updateVisibility();
  if (mode === "section" || el("detail").checked) {
    await loadDetails();
    if (loadId !== generation) return;
  }
  resetProjectionStates();
  fit("home");
  setLoading("");
  updateUrl();
}

async function loadEnvelope(loadId) {
  const row = cohort.topology_envelope;
  if (!row) return;
  const response = await fetch(row.asset, { cache: "force-cache" });
  if (!response.ok) throw new Error("Could not load tissue envelope");
  const buffer = await response.arrayBuffer();
  if (loadId !== generation) return;
  const view = new DataView(buffer);
  if (String.fromCharCode(...new Uint8Array(buffer, 0, 4)) !== "HTM1") {
    throw new Error("Invalid topology envelope");
  }
  const vertices = view.getUint32(4, true);
  const faces = view.getUint32(8, true);
  const raw = new Float32Array(buffer, 12, vertices * 3);
  const positions = new Float32Array(raw.length);
  const center = worldBounds().physicalCenter;
  for (let index = 0; index < vertices; index += 1) {
    positions[3 * index] = raw[3 * index] - center.x;
    positions[3 * index + 1] = raw[3 * index + 2] - center.z;
    positions[3 * index + 2] = -raw[3 * index + 1] + center.y;
  }
  const indices = new Uint32Array(buffer, 12 + vertices * 12, faces * 3);
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setIndex(new THREE.BufferAttribute(indices, 1));
  geometry.computeVertexNormals();
  const material = new THREE.MeshPhysicalMaterial({
    color: 0x7896a6,
    roughness: 0.9,
    transparent: true,
    opacity: +el("envelope-opacity").value / 100,
    side: THREE.FrontSide,
    depthWrite: false,
    clippingPlanes: [clipping],
  });
  envelope = new THREE.Mesh(geometry, material);
  envelope.scale.y = zScale;
  envelope.renderOrder = -1;
  root.add(envelope);
}

function populateSections() {
  el("section").replaceChildren(
    ...cohort.sections.map((row) => {
      const option = document.createElement("option");
      option.value = row.id;
      option.textContent = `${row.id} · ${row.cell_count.toLocaleString()} cells`;
      return option;
    }),
  );
  el("section").value = selectedSection;
}

function populateTargets() {
  const targetIds = new Set(cohort.targets.map((row) => row.id));
  activeTargets = activeTargets.filter((id) => targetIds.has(id));
  if (!activeTargets.length) {
    activeTargets = [...(cohort.default_targets || [cohort.default_target])];
  }
  const list = el("target-list");
  list.replaceChildren(
    ...cohort.targets.map((row) => {
      const label = document.createElement("label");
      label.className = "target";
      const input = document.createElement("input");
      input.type = "checkbox";
      input.value = row.id;
      input.checked = activeTargets.includes(row.id);
      input.addEventListener("change", () => toggleTarget(row.id, input.checked));
      const dot = document.createElement("i");
      dot.style.color = dot.style.background = row.color;
      const text = document.createElement("span");
      text.textContent = row.label;
      label.append(input, dot, text);
      return label;
    }),
  );
  const availablePresets = manifest.presets.filter((preset) =>
    preset.targets.some((target) => targetIds.has(target)),
  );
  el("preset").replaceChildren(
    ...availablePresets.map((row) => {
      const option = document.createElement("option");
      option.value = row.id;
      option.textContent = row.label;
      return option;
    }),
  );
  const match = availablePresets.find((preset) => {
    const targets = preset.targets.filter((target) => targetIds.has(target));
    return (
      targets.length === activeTargets.length &&
      targets.every((target, index) => target === activeTargets[index])
    );
  });
  el("preset").value = match?.id || "";
  updateTargetStatus();
  updateExpressionButtons();
}

function toggleTarget(target, checked) {
  if (checked) {
    if (activeTargets.length >= manifest.max_channels) {
      const input = el("target-list").querySelector(
        `input[value="${CSS.escape(target)}"]`,
      );
      if (input) input.checked = false;
      flashStatus(`Select at most ${manifest.max_channels} proteins.`);
      return;
    }
    activeTargets.push(target);
  } else {
    activeTargets = activeTargets.filter((value) => value !== target);
    if (!activeTargets.length) {
      activeTargets = [...(cohort.default_targets || [cohort.default_target])];
      populateTargets();
    }
  }
  updateTargetStatus();
  updateExpressionButtons();
  updateProvenance();
  updateUrl();
  if (layer === "protein") refreshColors(generation).catch(showError);
}

function updateTargetStatus(message = "") {
  const rows = activeTargets
    .map((id) => cohort.targets.find((row) => row.id === id))
    .filter(Boolean);
  el("channel-count").textContent = `${rows.length} / ${manifest.max_channels}`;
  const status = el("target-status");
  if (message) {
    const text = document.createElement("span");
    text.className = "status-message";
    text.textContent = message;
    status.replaceChildren(text);
    return;
  }
  status.replaceChildren(
    ...rows.map((row) => {
      const chip = document.createElement("span");
      chip.className = "status-chip";
      const dot = document.createElement("i");
      dot.style.background = row.color;
      chip.append(dot, row.label);
      return chip;
    }),
  );
}

function flashStatus(message) {
  updateTargetStatus(message);
  setTimeout(() => updateTargetStatus(), 2600);
}

function updateExpressionButtons() {
  const single = activeTargets.length === 1;
  const section = cohort.sections.find((row) => row.id === selectedSection);
  const entry = section?.targets?.[activeTargets[0]];
  const observed = Boolean(entry?.observed);
  if (expression !== "predicted" && (!single || !observed)) expression = "predicted";
  document.querySelectorAll("[data-expression]").forEach((button) => {
    button.disabled = button.dataset.expression !== "predicted" && (!single || !observed);
    button.classList.toggle("active", button.dataset.expression === expression);
  });
}

function updateProvenance() {
  if (!cohort) return;
  el("provenance").textContent = `${cohort.section_count} sections · ${cohort.cell_count.toLocaleString()} cells · ${cohort.overview_cell_count.toLocaleString()} deterministic overview cells`;
  const sectionThickness = Number(cohort.section_thickness_um || 5);
  const assumedZ = /assum/i.test(cohort.z_source || "assumed");
  const spacingLabel = assumedZ ? "Assumed" : "Recorded";
  const staticPreview = viewport.dataset.renderer === "fallback";
  const spacingControl = document.querySelector('[data-z="physical"]');
  if (spacingControl) spacingControl.textContent = assumedZ ? "Assumed spacing" : "Section spacing";
  if (staticPreview) {
    const names = (cohort.default_targets || [cohort.default_target])
      .map(id => cohort.targets.find(row => row.id === id)?.label || id).join(" / ");
    el("scene-badge").textContent = layer === "cells"
      ? "Static cell-instance preview" : `Static prediction preview · ${names}`;
    el("depth-badge").textContent = `${spacingLabel} Z spacing · ${sectionThickness} µm`;
    el("scope").textContent = layer === "cells"
      ? `Section-local cell instances · ${spacingLabel.toLowerCase()} Z`
      : `Prepared prediction preview · ${spacingLabel.toLowerCase()} Z · unvalidated protein maps`;
    el("cell-count").textContent = `${cohort.cell_count.toLocaleString()} candidate cells`;
    el("z-warning").textContent = `${spacingLabel} ${sectionThickness} µm spacing in this atlas. Static projections; no interpolation.`;
    return;
  }
  const adaptive = depthMode === "adaptive";
  el("z-warning").textContent = adaptive
    ? `Display spacing from cell size: ${adaptiveSectionSpacingUm.toFixed(1)} µm. Not measured tissue depth.`
    : depthMode === "physical"
      ? `${spacingLabel} ${sectionThickness.toLocaleString()} µm section spacing; no Z interpolation or enlargement.`
      : `Exploded section inspection at ${EXPLODED_Z_SCALE}× Z; measurements are unchanged.`;
  el("depth-badge").textContent = adaptive
    ? "Morphology-aware 3D · cell-sized spacing"
    : depthMode === "physical"
      ? `${spacingLabel} section spacing · 1×`
      : `Exploded sections · ${EXPLODED_Z_SCALE}× Z`;
  if (layer === "cells") {
    el("scene-badge").textContent = `Cell identity · ${cohort.overview_cell_count.toLocaleString()} overview cells`;
    el("scope").textContent = adaptive
      ? "Accepted boundary-derived XY footprints · morphology-aware display Z"
      : `Accepted boundary-derived XY footprints · ${spacingLabel.toLowerCase()} section-center Z`;
    el("cell-count").textContent = `${cohort.cell_count.toLocaleString()} cells · 16-vertex detail / 8-vertex overview`;
  } else {
    const proteinNames = activeTargets
      .map((id) => cohort.targets.find((row) => row.id === id)?.label || id)
      .join(" / ");
    const expressionLabel = {
      predicted: "Predicted",
      observed: "Observed OD",
      residual: "Residual",
    }[expression];
    const compositeLabel =
      activeTargets.length > 1
        ? ` · ${el("composite").value === "dominant" ? "dominant" : "co-expression"}`
        : "";
    el("scene-badge").textContent = `${expressionLabel} · ${proteinNames}${compositeLabel}`;
    el("scope").textContent = "Predicted per cell · atlas-global within-antibody display · observed OD at 4 µm/px";
    el("cell-count").textContent = `${cohort.cell_count.toLocaleString()} cells · ${activeTargets.length} protein channels`;
  }
}

function identityColors(geometry, section, lod) {
  const key = `identity:${cohort.id}:${section.id}:${lod}`;
  if (colorCache.has(key)) return colorCache.get(key);
  const palette = manifest.cell_identity_palette || [
    "#ff5ca8", "#2dd4ff", "#ffd43b", "#73e36b", "#a78bfa", "#ff875f",
  ];
  const rgb = palette.map(hexRgb255);
  const sectionIndex = cohort.sections.findIndex((row) => row.id === section.id) + 1;
  const colors = new Uint8Array(geometry.count * 4);
  for (let index = 0; index < geometry.count; index += 1) {
    const color = rgb[hashCell(geometry.labels[index], sectionIndex) % rgb.length];
    colors[4 * index] = color[0];
    colors[4 * index + 1] = color[1];
    colors[4 * index + 2] = color[2];
    colors[4 * index + 3] = lod === "detail" ? 250 : 235;
  }
  colorCache.set(key, colors);
  return colors;
}

function hashCell(label, sectionIndex) {
  let value = (label ^ Math.imul(sectionIndex, 0x9e3779b9)) >>> 0;
  value ^= value >>> 16;
  value = Math.imul(value, 0x7feb352d) >>> 0;
  value ^= value >>> 15;
  value = Math.imul(value, 0x846ca68b) >>> 0;
  value ^= value >>> 16;
  return value >>> 0;
}

function hexRgb255(hex) {
  return [
    parseInt(hex.slice(1, 3), 16),
    parseInt(hex.slice(3, 5), 16),
    parseInt(hex.slice(5, 7), 16),
  ];
}

function applyIdentityColors() {
  for (const [key, mesh] of sectionMeshes) {
    const section = cohort.sections.find((row) => row.id === mesh.userData.section);
    const lod = key.startsWith("detail:") ? "detail" : "overview";
    const colors = identityColors(mesh.userData.geometry, section, lod);
    const attribute = mesh.geometry.getAttribute("cellColor");
    attribute.array.set(colors);
    attribute.needsUpdate = true;
  }
}

function workerRequest(sections, lod, loadId) {
  if (!worker) worker = new Worker("protein-atlas-worker.js");
  const targets = activeTargets.map((id) => cohort.targets.find((row) => row.id === id));
  return new Promise((resolve, reject) => {
    const request = `${loadId}:${lod}:${colorGeneration}:${Math.random()}`;
    const listener = (event) => {
      if (event.data.request !== request) return;
      worker.removeEventListener("message", listener);
      if (event.data.error) reject(new Error(event.data.error));
      else resolve(event.data.rows);
    };
    worker.addEventListener("message", listener);
    worker.postMessage({
      request,
      expression,
      composite: el("composite").value,
      threshold: +el("threshold").value / 100,
      patternNormalized: expression !== "residual",
      lod,
      sections: sections.map((section) => ({
        id: section.id,
        count: lod === "detail" ? section.cell_count : section.overview_cell_count,
        channels: targets.map((target) => {
          const entry = section.targets[target.id];
          const observed = entry[`${lod === "detail" ? "" : "overview_"}observed`];
          return {
            id: target.id,
            color: target.color,
            transform: target.display_transform || {
              lower_fraction: 0,
              upper_fraction: 1,
              gamma: 1,
            },
            predicted: new URL(
              (lod === "detail" ? entry.predicted : entry.overview_predicted).asset,
              location.href,
            ).href,
            observed: observed ? new URL(observed.asset, location.href).href : null,
          };
        }),
      })),
    });
  });
}

async function refreshColors(loadId = generation) {
  if (!cohort || !renderer) return;
  const colorId = ++colorGeneration;
  if (layer === "cells") {
    applyIdentityColors();
    updateMaterialMode();
    setLoading("");
    updateProvenance();
    requestRender(3);
    return;
  }
  setLoading("Compositing cell-resolved protein channels…");
  const overviewRows = await workerRequest(cohort.sections, "overview", loadId);
  if (loadId !== generation || colorId !== colorGeneration || layer !== "protein") return;
  applyColorRows(overviewRows, "overview");
  if (detailSections.size) {
    const rows = cohort.sections.filter((row) => detailSections.has(row.id));
    const detailRows = await workerRequest(rows, "detail", loadId);
    if (loadId !== generation || colorId !== colorGeneration || layer !== "protein") return;
    applyColorRows(detailRows, "detail");
  }
  updateMaterialMode();
  setLoading("");
  updateProvenance();
  requestRender(3);
}

function applyColorRows(rows, lod) {
  for (const row of rows) {
    const mesh = sectionMeshes.get(`${lod}:${row.id}`);
    if (!mesh) continue;
    const attribute = mesh.geometry.getAttribute("cellColor");
    attribute.array.set(new Uint8Array(row.colors));
    attribute.needsUpdate = true;
  }
}

function updateMaterialMode() {
  for (const mesh of sectionMeshes.values()) {
    mesh.material.uniforms.identityMode.value = layer === "cells" ? 1 : 0;
    mesh.material.uniforms.alphaScale.value = layer === "cells" ? 1 : 0.96;
    mesh.material.depthWrite = mode !== "orthogonal";
    mesh.material.blending = THREE.NormalBlending;
    mesh.material.needsUpdate = true;
  }
}

async function loadDetails() {
  // Compatibility mode intentionally stays asset-only. Loading hundreds of
  // thousands of cell vectors after WebGL is unavailable makes otherwise
  // functional layer/mode controls appear frozen and cannot improve the
  // fingerprinted fallback preview.
  if (!cohort || !renderer) return;
  const loadId = generation;
  const wanted = new Set();
  if (el("detail").checked || mode === "section") {
    const index = cohort.sections.findIndex((row) => row.id === selectedSection);
    const radius = mode === "section" ? 0 : 1;
    for (let offset = -radius; offset <= radius; offset += 1) {
      const row = cohort.sections[index + offset];
      if (row) wanted.add(row.id);
    }
  }
  for (const id of [...detailSections]) {
    if (wanted.has(id)) continue;
    const mesh = sectionMeshes.get(`detail:${id}`);
    if (mesh) {
      root.remove(mesh);
      mesh.geometry.dispose();
      mesh.material.dispose();
      sectionMeshes.delete(`detail:${id}`);
    }
    detailSections.delete(id);
  }
  await mapLimit(
    cohort.sections.filter((row) => wanted.has(row.id) && !detailSections.has(row.id)),
    2,
    async (section) => {
      const geometry = await fetchGeometry(section, "detail", loadId);
      if (!geometry || loadId !== generation) return;
      sectionMeshes.set(
        `detail:${section.id}`,
        makeMesh(geometry, section, "detail"),
      );
      detailSections.add(section.id);
    },
  );
  if (loadId !== generation) return;
  await refreshColors(loadId);
  updateVisibility();
}

function updateVisibility() {
  if (!cohort) return;
  const sectionIndex = cohort.sections.findIndex((row) => row.id === selectedSection);
  const range = Math.round(
    (+el("section-range").value / 100) * (cohort.sections.length - 1),
  );
  const minimum = Math.max(0, sectionIndex - range);
  const maximum = Math.min(cohort.sections.length - 1, sectionIndex + range);
  for (const [key, mesh] of sectionMeshes) {
    const index = cohort.sections.findIndex((row) => row.id === mesh.userData.section);
    const inRange = index >= minimum && index <= maximum;
    const detail = key.startsWith("detail:");
    const hasDetail = detailSections.has(mesh.userData.section);
    const sectionDetailReady = detailSections.has(selectedSection);
    mesh.visible =
      inRange &&
      (mode !== "section"
        ? !hasDetail || detail
        : mesh.userData.section === selectedSection &&
          (sectionDetailReady ? detail : !detail));
  }
  if (envelope) envelope.visible = mode !== "section";
  if (range < cohort.sections.length - 1) updateLocator();
  else removeLocator();
  el("range-value").textContent =
    range >= cohort.sections.length - 1 ? "All" : `±${range}`;
  requestRender(2);
}

function removeLocator() {
  if (!locator) return;
  root?.remove(locator);
  locator.geometry.dispose();
  locator.material.dispose();
  locator = null;
}

function updateLocator() {
  removeLocator();
  if (!cohort || mode === "section") return;
  const row = cohort.sections.find((value) => value.id === selectedSection);
  const bounds = cohort.bounds_um;
  const center = worldBounds().physicalCenter;
  const y = (row.z_um - center.z) * zScale;
  const points = [
    new THREE.Vector3(bounds.x[0] - center.x, y, -bounds.y[0] + center.y),
    new THREE.Vector3(bounds.x[1] - center.x, y, -bounds.y[0] + center.y),
    new THREE.Vector3(bounds.x[1] - center.x, y, -bounds.y[1] + center.y),
    new THREE.Vector3(bounds.x[0] - center.x, y, -bounds.y[1] + center.y),
    new THREE.Vector3(bounds.x[0] - center.x, y, -bounds.y[0] + center.y),
  ];
  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  const material = new THREE.LineBasicMaterial({
    color: 0x72e0b2,
    transparent: true,
    opacity: 0.5,
    depthTest: false,
  });
  locator = new THREE.Line(geometry, material);
  locator.renderOrder = 5;
  root.add(locator);
}

function clearSelection() {
  selectionGeneration += 1;
  if (selection) {
    root?.remove(selection);
    selection.geometry.dispose();
    selection.material.dispose();
    selection = null;
  }
  el("cell-detail").hidden = true;
  requestRender(2);
}

function panelForPoint(x, y, width, height) {
  if (mode !== "orthogonal") {
    return {
      view: "top",
      stateKey: "section",
      left: 0,
      top: 0,
      width,
      height,
    };
  }
  return (
    orthogonalPanels(width, height).find(
      (panel) =>
        x >= panel.left &&
        x < panel.left + panel.width &&
        y >= panel.top &&
        y < panel.top + panel.height,
    ) || orthogonalPanels(width, height)[0]
  );
}

function projectionForPointer(event) {
  const rect = renderer.domElement.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  if (mode === "3d") {
    return {
      camera: perspective,
      stateKey: "top",
      left: 0,
      top: 0,
      width: rect.width,
      height: rect.height,
      x,
      y,
    };
  }
  const panel = panelForPoint(x, y, rect.width, rect.height);
  const camera = orthographicCamera(
    panel.view,
    panel.width / Math.max(panel.height, 1),
    mode === "section",
  );
  return { ...panel, camera, x, y };
}

async function selectCell(event) {
  if (!renderer || !cohort) return;
  const mesh =
    sectionMeshes.get(`detail:${selectedSection}`) ||
    sectionMeshes.get(`overview:${selectedSection}`);
  if (!mesh?.visible) {
    clearSelection();
    return;
  }
  const pointer = projectionForPointer(event);
  const camera = pointer.camera;
  camera.updateMatrixWorld(true);
  const geometry = mesh.userData.geometry;
  const colors = mesh.geometry.getAttribute("cellColor").array;
  const point = new THREE.Vector3();
  let selected = -1;
  let best = Infinity;
  const glyphScale = +el("glyph-size").value / 100;
  const referenceScale = Math.max(
    Math.hypot(geometry.nativeLinear[0], geometry.nativeLinear[2]),
    Math.hypot(geometry.nativeLinear[1], geometry.nativeLinear[3]),
  );
  const orthographicPixelsPerUm =
    camera.isOrthographicCamera
      ? pointer.width / Math.max(camera.right - camera.left, 1e-9)
      : 0;
  for (let index = 0; index < geometry.count; index += 1) {
    if (!colors[4 * index + 3]) continue;
    point.set(
      geometry.positions[3 * index],
      geometry.positions[3 * index + 2] * zScale +
        (depthMode === "adaptive" ? geometry.cellZPhases[index] * adaptiveJitterUm : 0),
      -geometry.positions[3 * index + 1],
    );
    if (el("cutaway").checked && point.x + clipping.constant < 0) continue;
    point.project(camera);
    if (point.z < -1 || point.z > 1) continue;
    const sx = pointer.left + ((point.x + 1) * pointer.width) / 2;
    const sy = pointer.top + ((1 - point.y) * pointer.height) / 2;
    const distance = (sx - pointer.x) ** 2 + (sy - pointer.y) ** 2;
    const radius = camera.isOrthographicCamera
      ? Math.max(
          18,
          geometry.maximumRadii[index] * referenceScale * glyphScale *
            orthographicPixelsPerUm * 1.15,
        )
      : 18;
    const score = distance / (radius * radius);
    if (score <= 1 && score < best) {
      best = score;
      selected = index;
    }
  }
  clearSelection();
  if (selected < 0) return;
  const token = ++selectionGeneration;
  const points = cellBoundaryPoints(geometry, selected, glyphScale);
  const outlineGeometry = new THREE.BufferGeometry().setFromPoints(points);
  const material = new THREE.LineBasicMaterial({
    color: 0xffffff,
    transparent: true,
    opacity: 0.96,
    depthTest: false,
  });
  selection = new THREE.LineLoop(outlineGeometry, material);
  const positionIndex = selected * 3;
  selection.position.set(
    geometry.positions[positionIndex],
    geometry.positions[positionIndex + 2] * zScale +
      (depthMode === "adaptive" ? geometry.cellZPhases[selected] * adaptiveJitterUm : 0),
    -geometry.positions[positionIndex + 1],
  );
  selection.renderOrder = 20;
  root.add(selection);
  const center = worldBounds().physicalCenter;
  const area = polygonArea(cellBoundaryPoints(geometry, selected, 1));
  const detail = el("cell-detail");
  const title = document.createElement("strong");
  title.textContent = `Cell ${geometry.labels[selected].toLocaleString()} · section ${selectedSection}`;
  const meta = document.createElement("span");
  const source = geometry.boundaryDerived ? "accepted boundary vector" : "legacy morphology approximation";
  meta.textContent = `${mesh.userData.lod === "detail" ? "Full cell set" : "Deterministic overview"} · ${source} · ${geometry.sectors} vertices · footprint ${area.toFixed(1)} µm² · XYZ ${(geometry.positions[positionIndex] + center.x).toFixed(1)}, ${(geometry.positions[positionIndex + 1] + center.y).toFixed(1)}, ${(geometry.positions[positionIndex + 2] + center.z).toFixed(1)} µm`;
  const values = document.createElement("div");
  values.className = "detail-values";
  detail.replaceChildren(title, meta, values);
  detail.hidden = false;
  requestRender(3);
  try {
    const rows = await inspectProteinValues(
      cohort.sections.find((row) => row.id === selectedSection),
      mesh.userData.lod,
      selected,
    );
    if (token !== selectionGeneration) return;
    values.replaceChildren(
      ...rows.flatMap((row) => {
        const label = document.createElement("span");
        const dot = document.createElement("i");
        dot.style.background = row.color;
        label.append(dot, row.label);
        const value = document.createElement("b");
        value.textContent = row.value;
        return [label, value];
      }),
    );
  } catch (_error) {
    if (token !== selectionGeneration) return;
    const unavailable = document.createElement("span");
    unavailable.textContent = "Protein values unavailable for this selected glyph.";
    values.replaceChildren(unavailable);
  }
}

function cellBoundaryPoints(geometry, index, scale = 1) {
  const points = [];
  const start = index * geometry.sectors;
  const maximum = geometry.maximumRadii[index] * scale;
  const linear = geometry.nativeLinear;
  for (let sector = 0; sector < geometry.sectors; sector += 1) {
    const radius = (geometry.profiles[start + sector] / 255) * maximum;
    const fraction = geometry.angleFractions[start + sector] / 255;
    const angle = ((sector + fraction) / geometry.sectors) * Math.PI * 2;
    const nativeX = Math.cos(angle) * radius;
    const nativeY = Math.sin(angle) * radius;
    const referenceX = linear[0] * nativeX + linear[1] * nativeY;
    const referenceY = linear[2] * nativeX + linear[3] * nativeY;
    points.push(new THREE.Vector3(referenceX, 0, -referenceY));
  }
  return points;
}

function polygonArea(points) {
  let twiceArea = 0;
  for (let index = 0; index < points.length; index += 1) {
    const next = points[(index + 1) % points.length];
    twiceArea += points[index].x * -next.z - next.x * -points[index].z;
  }
  return Math.abs(twiceArea) / 2;
}

async function inspectProteinValues(section, lod, index) {
  return Promise.all(
    activeTargets.map(async (targetId) => {
      const target = cohort.targets.find((row) => row.id === targetId);
      const entry = section.targets[targetId];
      const asset = lod === "detail" ? entry.predicted : entry.overview_predicted;
      const count = lod === "detail" ? section.cell_count : section.overview_cell_count;
      const chunk = await valueChunk(asset.asset, count);
      const encoded = chunk[index];
      const od = encoded ? ((encoded - 1) / 254) * target.display_max_od : null;
      return {
        label: target.label,
        color: target.color,
        value: od === null ? "unsupported" : `${od.toFixed(3)} OD`,
      };
    }),
  );
}

async function valueChunk(asset, count) {
  const url = new URL(asset, location.href).href;
  if (valueChunkCache.has(url)) return valueChunkCache.get(url);
  const response = await fetch(url, { cache: "force-cache" });
  if (!response.ok) throw new Error("Could not load selected-cell protein values");
  const buffer = await response.arrayBuffer();
  const view = new DataView(buffer);
  if (String.fromCharCode(...new Uint8Array(buffer, 0, 5)) !== "HCPA1") {
    throw new Error("Invalid selected-cell protein values");
  }
  const cells = view.getUint32(8, true);
  const header = view.getUint32(12, true);
  if (cells !== count) throw new Error("Protein values and cell geometry differ");
  const values = new Uint8Array(buffer, header, cells);
  valueChunkCache.set(url, values);
  if (valueChunkCache.size > 24) {
    valueChunkCache.delete(valueChunkCache.keys().next().value);
  }
  return values;
}

function updateCutaway() {
  const enabled = el("cutaway").checked;
  el("cut-row").hidden = !enabled;
  const bounds = cohort.bounds_um;
  const span = bounds.x[1] - bounds.x[0];
  clipping.constant = enabled
    ? -(+el("cut-position").value / 100) * (span / 2)
    : 1e12;
  for (const object of root.children) {
    if (object.material) object.material.needsUpdate = true;
  }
  requestRender(2);
}

function setLayer(value) {
  if (!['cells', 'protein'].includes(value) || value === layer) return;
  clearSelection();
  layer = value;
  app.dataset.layer = layer;
  document.querySelectorAll("[data-layer]").forEach((button) => {
    button.classList.toggle("active", button.dataset.layer === layer);
    button.setAttribute("aria-pressed", String(button.dataset.layer === layer));
  });
  updateFallbackPreview();
  updateProvenance();
  updateUrl();
  refreshColors(generation).catch(showError);
}

function setMode(value) {
  if (!['3d', 'orthogonal', 'section'].includes(value) || value === mode) return;
  clearSelection();
  mode = value;
  viewport.classList.toggle("orthogonal", mode === "orthogonal");
  document.querySelectorAll("[data-mode],[data-sidebar-mode]").forEach((button) => {
    const value = button.dataset.mode || button.dataset.sidebarMode;
    button.classList.toggle("active", value === mode);
    button.setAttribute("aria-pressed", String(value === mode));
  });
  if (controls) controls.enabled = mode === "3d";
  updateMaterialMode();
  loadDetails().catch(showError);
  updateVisibility();
  updateUrl();
  requestRender(3);
}

function setZ(value) {
  clearSelection();
  depthMode = ["physical", "adaptive", "exploded"].includes(value)
    ? value
    : "adaptive";
  applyDepthModel();
  resetProjectionStates();
  fit("home");
  updateUrl();
}

function setLoading(message) {
  el("loading").hidden = !message;
  el("loading").textContent = message;
}

function showError(error) {
  setLoading(error.message || String(error));
}

function updateUrl() {
  if (!cohort) return;
  const query = new URLSearchParams(location.search);
  query.set("mouse", cohort.id);
  query.set("layer", layer);
  query.set("markers", activeTargets.join(","));
  query.set("mode", mode);
  query.set("section", selectedSection);
  query.set("expression", expression);
  query.set("depth", depthMode);
  query.delete("z");
  query.set("composite", el("composite").value);
  query.set("cutoff", el("threshold").value);
  query.delete("exploratory");
  history.replaceState(null, "", `${location.pathname}?${query}${location.hash}`);
}

function readInitialState() {
  const query = new URLSearchParams(location.search);
  const selected =
    manifest.cohorts.find((row) => row.id === query.get("mouse")) ||
    manifest.cohorts[0];
  const defaults = selected.default_targets || [selected.default_target];
  const targets = (query.get("markers") || defaults.join(","))
    .split(",")
    .filter((id) => selected.targets.some((row) => row.id === id))
    .slice(0, manifest.max_channels);
  const selectedLayer = ["cells", "protein"].includes(query.get("layer"))
    ? query.get("layer")
    : ["observed", "residual"].includes(query.get("expression"))
      ? "protein"
      : manifest.default?.layer || "protein";
  const selectedMode = ["3d", "orthogonal", "section"].includes(query.get("mode"))
    ? query.get("mode")
    : manifest.default?.mode || "3d";
  const selectedExpression = ["predicted", "observed", "residual"].includes(
    query.get("expression"),
  )
    ? query.get("expression")
    : "predicted";
  const requestedDepth = query.get("depth");
  const legacyZ = Number(query.get("z"));
  const selectedDepth = ["physical", "adaptive", "exploded"].includes(requestedDepth)
    ? requestedDepth
    : legacyZ === 1
      ? "physical"
      : legacyZ === 25
        ? "exploded"
        : "adaptive";
  const composite = ["overlap", "additive"].includes(query.get("composite"))
    ? "overlap"
    : "dominant";
  const manifestCutoff = Math.round(
    Math.max(0, Math.min(0.5, Number(manifest.default?.extra_cutoff ?? 0.25))) * 100,
  );
  const requestedCutoff = Number(query.get("cutoff"));
  const cutoff =
    query.has("cutoff") &&
    Number.isFinite(requestedCutoff) &&
    requestedCutoff >= 0 &&
    requestedCutoff <= 50
      ? requestedCutoff
      : manifestCutoff;
  const section = selected.sections.some((row) => row.id === query.get("section"))
    ? query.get("section")
    : selected.default_section || selected.sections[Math.floor(selected.sections.length / 2)].id;
  return {
    cohort: selected,
    layer: selectedLayer,
    mode: selectedMode,
    expression: selectedExpression,
    depthMode: selectedDepth,
    composite,
    cutoff,
    section,
    targets: targets.length ? targets : defaults,
  };
}

function bind() {
  manifest.cohorts.forEach((row) => {
    const option = document.createElement("option");
    option.value = row.id;
    option.textContent = row.id;
    el("cohort").append(option);
  });
  el("cohort").onchange = () => {
    loadCohort(el("cohort").value, true).catch(showError);
  };
  el("section").onchange = () => {
    clearSelection();
    selectedSection = el("section").value;
    resetProjectionState("section");
    updateExpressionButtons();
    loadDetails().catch(showError);
    updateVisibility();
    updateUrl();
  };
  document.querySelectorAll("[data-layer]").forEach((button) => {
    button.onclick = () => setLayer(button.dataset.layer);
  });
  document.querySelectorAll("[data-mode]").forEach((button) => {
    button.onclick = () => setMode(button.dataset.mode);
  });
  document.querySelectorAll("[data-sidebar-mode]").forEach((button) => {
    button.onclick = () => setMode(button.dataset.sidebarMode);
  });
  document.querySelectorAll("[data-view]").forEach((button) => {
    button.onclick = () => fit(button.dataset.view);
  });
  document.querySelectorAll("[data-z]").forEach((button) => {
    button.onclick = () => setZ(button.dataset.z);
  });
  document.querySelectorAll("[data-expression]").forEach((button) => {
    button.onclick = () => {
      expression = button.dataset.expression;
      updateExpressionButtons();
      refreshColors(generation).catch(showError);
      updateUrl();
    };
  });
  el("preset").onchange = () => {
    const preset = manifest.presets.find((row) => row.id === el("preset").value);
    if (!preset) return;
    activeTargets = preset.targets
      .filter((id) => cohort.targets.some((row) => row.id === id))
      .slice(0, manifest.max_channels);
    populateTargets();
    updateProvenance();
    if (layer === "protein") refreshColors(generation).catch(showError);
    updateUrl();
  };
  el("composite").onchange = () => {
    updateProvenance();
    if (layer === "protein") refreshColors(generation).catch(showError);
    updateUrl();
  };
  el("threshold").oninput = () => {
    el("threshold-value").textContent = `${el("threshold").value}%`;
    if (layer === "protein") refreshColors(generation).catch(showError);
    updateUrl();
  };
  el("glyph-size").oninput = () => {
    clearSelection();
    const value = +el("glyph-size").value / 100;
    el("glyph-value").textContent = `${value.toFixed(2)}×`;
    for (const mesh of sectionMeshes.values()) {
      mesh.material.uniforms.glyphScale.value = value;
    }
    requestRender(2);
  };
  el("envelope-opacity").oninput = () => {
    el("envelope-value").textContent = `${el("envelope-opacity").value}%`;
    if (envelope) envelope.material.opacity = +el("envelope-opacity").value / 100;
    requestRender(2);
  };
  el("section-range").oninput = updateVisibility;
  el("detail").onchange = () => {
    clearSelection();
    loadDetails().catch(showError);
  };
  el("cutaway").onchange = () => {
    clearSelection();
    updateCutaway();
  };
  el("cut-position").oninput = () => {
    clearSelection();
    updateCutaway();
  };
  el("fullscreen").onclick = () =>
    document.fullscreenElement ? document.exitFullscreen() : app.requestFullscreen();
  el("panel-toggle").onclick = () => {
    const open = app.classList.toggle("panel-open");
    el("panel-toggle").setAttribute("aria-expanded", String(open));
    el("panel-toggle").textContent = open ? "Close" : "Controls";
  };
  addEventListener("resize", resize);
}

function initialize() {
  const initial = readInitialState();
  cohort = initial.cohort;
  layer = initial.layer;
  mode = initial.mode;
  expression = initial.expression;
  depthMode = initial.depthMode;
  resetAdaptiveDepth();
  activeTargets = [...initial.targets];
  selectedSection = initial.section;
  el("composite").value = initial.composite;
  el("threshold").value = String(initial.cutoff);
  el("threshold-value").textContent = `${initial.cutoff}%`;
  app.dataset.layer = layer;
  viewport.classList.toggle("orthogonal", mode === "orthogonal");
  perspective = new THREE.PerspectiveCamera(36, 1, 0.1, 1e8);
  root = new THREE.Group();
  scene.add(root);
  bind();
  document.querySelectorAll("[data-layer]").forEach((button) => {
    button.classList.toggle("active", button.dataset.layer === layer);
    button.setAttribute("aria-pressed", String(button.dataset.layer === layer));
  });
  document.querySelectorAll("[data-mode],[data-sidebar-mode]").forEach((button) => {
    const value = button.dataset.mode || button.dataset.sidebarMode;
    button.classList.toggle("active", value === mode);
    button.setAttribute("aria-pressed", String(value === mode));
  });
  document.querySelectorAll("[data-z]").forEach((button) => {
    button.classList.toggle("active", button.dataset.z === depthMode);
    button.setAttribute("aria-pressed", String(button.dataset.z === depthMode));
  });
  if (createRenderer()) {
    bindControls();
    resize();
  }
  loadCohort(initial.cohort.id, false).catch(showError);
}

initialize();
