(function () {
  "use strict";

  let host;
  let canvas;
  let title;
  let detail;
  let scale;
  let layers;
  let opacity;
  let previous;
  let next;
  let viewer;
  let state = null;
  let generation = 0;

  function ensureHost(layout) {
    if (host) {
      host.classList.toggle("atlas", layout === "atlas");
      return;
    }
    host = document.createElement("section");
    host.className = "histopia-focus-host";
    host.hidden = true;
    host.innerHTML = `
      <div class="histopia-focus-toolbar">
        <button type="button" data-action="previous" title="Previous section"
          aria-label="Previous section">&#8249;</button>
        <button type="button" data-action="next" title="Next section"
          aria-label="Next section">&#8250;</button>
        <strong></strong>
        <div class="histopia-focus-layers" aria-label="Image layer"></div>
        <label class="histopia-focus-opacity">Overlay
          <input type="range" min="0" max="1" step="0.05" value="0.45">
        </label>
        <button type="button" data-action="fit" title="Fit tissue"
          aria-label="Fit tissue">&#8962;</button>
        <button type="button" data-action="close" title="Close detail"
          aria-label="Close detail">&times;</button>
      </div>
      <div class="histopia-focus-canvas"></div>
      <div class="histopia-focus-status"><span></span><span></span></div>`;
    document.body.append(host);
    canvas = host.querySelector(".histopia-focus-canvas");
    title = host.querySelector("strong");
    detail = host.querySelector(".histopia-focus-status span:first-child");
    scale = host.querySelector(".histopia-focus-status span:last-child");
    layers = host.querySelector(".histopia-focus-layers");
    opacity = host.querySelector("input");
    previous = host.querySelector('[data-action="previous"]');
    next = host.querySelector('[data-action="next"]');
    host.querySelector('[data-action="close"]').addEventListener("click", close);
    host.querySelector('[data-action="fit"]').addEventListener(
      "click",
      () => viewer?.viewport.goHome(true),
    );
    previous.addEventListener("click", () => step(-1));
    next.addEventListener("click", () => step(1));
    opacity.addEventListener("input", () => {
      const item = viewer?.world.getItemAt(1);
      if (item) item.setOpacity(Number(opacity.value));
    });
    document.addEventListener("keydown", event => {
      if (host.hidden) return;
      if (event.key === "Escape") close();
      if (event.key === "ArrowLeft") step(-1);
      if (event.key === "ArrowRight") step(1);
    });
    host.classList.toggle("atlas", layout === "atlas");
  }

  function customSource(metadata, layerName) {
    const layer = metadata.layers[layerName];
    const levels = layer.levels;
    return {
      width: layer.width,
      height: layer.height,
      tileSize: layer.tile_size,
      minLevel: 0,
      maxLevel: levels.length - 1,
      getLevelScale(level) {
        return levels[level].width / layer.width;
      },
      getNumTiles(level) {
        return new OpenSeadragon.Point(
          Math.ceil(levels[level].width / layer.tile_size),
          Math.ceil(levels[level].height / layer.tile_size),
        );
      },
      getTileUrl(level, x, y) {
        if (layer.tile_url_template) {
          const relative = layer.tile_url_template
            .replace("{level}", level)
            .replace("{x}", x)
            .replace("{y}", y);
          return new URL(relative, metadata.metadata_url).href;
        }
        return `/api/wsi/${encodeURIComponent(metadata.cohort)}/` +
          `${metadata.section}/${layerName}/${layer.digest}/` +
          `${level}/${x}/${y}.${layer.format}`;
      },
      tileExists(level, x, y) {
        const count = this.getNumTiles(level);
        const valid = level >= 0 && level < levels.length &&
          x >= 0 && y >= 0 && x < count.x && y < count.y;
        if (!valid) return false;
        return !layer.existing_tiles ||
          layer.existing_tiles.includes(`${level}/${x}/${y}`);
      },
    };
  }

  function ensureViewer() {
    if (viewer) return viewer;
    if (!globalThis.OpenSeadragon)
      throw new Error("Native-resolution viewer runtime is unavailable");
    viewer = OpenSeadragon({
      element: canvas,
      showNavigationControl: false,
      showNavigator: true,
      navigatorPosition: "BOTTOM_RIGHT",
      animationTime: 0.35,
      blendTime: 0.1,
      immediateRender: false,
      maxZoomPixelRatio: 2,
      minZoomImageRatio: 0.8,
      preserveViewport: false,
      visibilityRatio: 0.5,
      gestureSettingsMouse: {clickToZoom: false, dblClickToZoom: true},
    });
    return viewer;
  }

  function showError(message) {
    let error = canvas.querySelector(".histopia-focus-error");
    if (!error) {
      error = document.createElement("div");
      error.className = "histopia-focus-error";
      canvas.append(error);
    }
    error.textContent = message;
  }

  function clearError() {
    canvas.querySelector(".histopia-focus-error")?.remove();
  }

  function setLayerButtons(available, selected) {
    layers.replaceChildren();
    for (const name of available) {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = name === "registered" ? "Registered" :
        name === "raw" ? "Source" : "Mask";
      button.classList.toggle("active", name === selected);
      button.addEventListener("click", () => openLayer(name));
      layers.append(button);
    }
  }

  async function loadMetadata(cohort, section, metadataTemplate) {
    let response;
    if (metadataTemplate) {
      const relative = metadataTemplate
        .replace("{cohort}", encodeURIComponent(cohort))
        .replace("{section}", section);
      response = await fetch(
        new URL(relative, document.baseURI),
        {headers: {"Accept": "application/json"}},
      );
    } else {
      response = await fetch(
        `/api/wsi/${encodeURIComponent(cohort)}/${section}`,
        {headers: {"Accept": "application/json"}},
      );
    }
    if (!metadataTemplate && response.status === 404) {
      const fallback = new URL(
        `wsi/${encodeURIComponent(cohort)}/${section}/metadata.json`,
        document.baseURI,
      );
      response = await fetch(fallback, {headers: {"Accept": "application/json"}});
    }
    if (!response.ok) throw new Error(
      response.status === 404
        ? "Native-resolution data are not available for this section."
        : `Native-resolution request failed (${response.status}).`,
    );
    const payload = await response.json();
    payload.metadata_url = response.url;
    return payload;
  }

  async function openLayer(requested) {
    if (!state?.metadata) return;
    const metadata = state.metadata;
    const available = Object.keys(metadata.layers);
    let base = requested;
    let overlayName = null;
    let baseMetadata = metadata;
    let comparisonSource = null;
    if (requested === "mask") {
      base = "raw";
      overlayName = metadata.layers.mask ? "mask" : null;
    }
    if (!metadata.layers[base]) base = available.includes("registered")
      ? "registered" : available[0];
    state.layer = requested;
    if (requested === "registered" && state.referenceSection &&
        state.referenceSection !== metadata.section) {
      const reference = await loadMetadata(
        metadata.cohort,
        state.referenceSection,
        state.metadataTemplate,
      );
      if (reference.layers.registered) {
        baseMetadata = reference;
        base = "registered";
        comparisonSource = customSource(metadata, "registered");
      }
    }
    setLayerButtons(
      ["raw", "registered", "mask"].filter(name => metadata.layers[name]),
      requested,
    );
    const currentGeneration = ++generation;
    const activeViewer = ensureViewer();
    clearError();
    activeViewer.open(customSource(baseMetadata, base));
    activeViewer.addOnceHandler("open", () => {
      if (currentGeneration !== generation) return;
      if (comparisonSource) {
        activeViewer.addTiledImage({
          tileSource: comparisonSource,
          opacity: Number(opacity.value),
        });
      } else if (overlayName) {
        activeViewer.addTiledImage({
          tileSource: customSource(metadata, overlayName),
          opacity: Number(opacity.value),
        });
      } else if (state.overlayUrl) {
        activeViewer.addTiledImage({
          tileSource: {type: "image", url: state.overlayUrl},
          opacity: Number(opacity.value),
          width: 1,
        });
      }
    });
    const layer = metadata.layers[base];
    const mpp = layer.microns_per_pixel;
    detail.textContent =
      `${layer.width.toLocaleString()} × ${layer.height.toLocaleString()} px`;
    scale.textContent = mpp == null
      ? (state.overlayResolution || "Native scanner resolution")
      : `${mpp.toFixed(mpp < 1 ? 3 : 2)} µm/px` +
        (state.overlayResolution ? ` | ${state.overlayResolution}` : "");
  }

  async function open(options) {
    ensureHost(options.layout || "review");
    state = {...options, layer: options.layer || "registered"};
    host.hidden = false;
    host.setAttribute("aria-busy", "true");
    title.textContent = options.title || `Section ${options.section}`;
    previous.disabled = !options.onStep;
    next.disabled = !options.onStep;
    detail.textContent = "Loading native resolution";
    scale.textContent = "";
    try {
      state.metadata = await loadMetadata(
        options.cohort,
        options.section,
        options.metadataTemplate,
      );
      title.textContent =
        `${String(options.section).padStart(3, "0")} ${state.metadata.label}`;
      await openLayer(state.layer);
    } catch (error) {
      if (state?.onUnavailable) {
        const callback = state.onUnavailable;
        close();
        callback();
        return;
      }
      showError(error instanceof Error ? error.message : String(error));
    } finally {
      host.removeAttribute("aria-busy");
    }
  }

  function close() {
    if (!host || host.hidden) return;
    generation += 1;
    viewer?.close();
    host.hidden = true;
    state?.onClose?.();
    state = null;
  }

  function step(offset) {
    if (!state?.onStep) return;
    const replacement = state.onStep(offset);
    if (replacement) open({...state, ...replacement});
  }

  function cohortFromLocation(fallback) {
    const parts = location.pathname.split("/").filter(Boolean);
    const stage = parts.findIndex(part =>
      ["mask", "order", "alignment"].includes(part));
    return stage > 0 ? parts[stage - 1] : fallback;
  }

  function attachGrid({container, data, mode}) {
    if (location.protocol === "file:") return;
    const cards = [...container.querySelectorAll("article")];
    const fallback = data.feedback?.cohort || "";
    const cohort = cohortFromLocation(fallback);
    const referenceIndex = data.slides.findIndex(slide => slide.reference);
    fetch(`/api/wsi/${encodeURIComponent(cohort)}`)
      .then(response => response.ok ? response.json() : {sections: []})
      .then(catalog => {
        const available = new Set(
          (catalog.sections || []).map(item => item.section),
        );
        const availableCards = cards.filter((card, index) => {
          const order = data.slides[index].order || index + 1;
          return available.has(String(order).padStart(3, "0"));
        });
        cards.forEach((card, index) => {
          const section = String(
            data.slides[index].order || index + 1,
          ).padStart(3, "0");
          if (!available.has(section)) return;
          card.classList.add("histopia-wsi-available");
          card.tabIndex = 0;
          const activate = () => open({
            cohort,
            section,
            title: data.slides[index].label,
            layer: mode === "alignment" ? "registered" : "raw",
            layout: "review",
            onStep(offset) {
              const current = availableCards.indexOf(card);
              const target = (
                current + offset + availableCards.length
              ) % availableCards.length;
              availableCards[target].click();
              return null;
            },
            referenceSection: referenceIndex >= 0
              ? String(referenceIndex + 1).padStart(3, "0") : null,
          });
          card.addEventListener("click", activate);
          card.addEventListener("keydown", event => {
            if (event.key === "Enter" || event.key === " ") {
              event.preventDefault();
              activate();
            }
          });
        });
      })
      .catch(() => {});
  }

  globalThis.HistopiaFocusViewer = {attachGrid, close, open};
}());
