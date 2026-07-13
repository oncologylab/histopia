# ruff: noqa: E501
"""Generate a static Three.js viewer for registered section stacks."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from histopia.registration._errors import OptionalDependencyError
from histopia.registration._io import warp_mask_thumbnail, warp_rgb_thumbnail

THREE_VERSION = "0.170.0"


def build_section_viewer(
    runs: dict[str, Path | str],
    output_dir: Path | str,
    *,
    provisional_mice: set[str] | None = None,
) -> Path:
    """Build a browser viewer from completed registration run directories."""

    try:
        from PIL import Image
    except ImportError as exc:
        raise OptionalDependencyError("pillow", "wsi") from exc

    output_dir = Path(output_dir)
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    provisional_mice = provisional_mice or set()
    mouse_payloads: list[dict[str, object]] = []

    for mouse_id, run_value in sorted(runs.items()):
        run_dir = Path(run_value)
        payload = json.loads((run_dir / "registration_result.json").read_text())
        reference_path = Path(payload["reference_slide"])
        reference_image = _read_rgb(
            run_dir / "processed" / f"{reference_path.stem}.thumbnail.png"
        )
        mouse_assets = assets_dir / _safe_name(mouse_id)
        mouse_assets.mkdir(parents=True, exist_ok=True)
        slides: list[dict[str, object]] = []
        for order, slide in enumerate(payload["slides"], start=1):
            source_path = Path(slide["path"])
            source = _read_rgb(
                run_dir / "processed" / f"{source_path.stem}.thumbnail.png"
            )
            mask = _read_mask(run_dir / "processed" / f"{source_path.stem}.mask.png")
            matrix = np.asarray(slide["transform"]["matrix"], dtype=float)
            registered = warp_rgb_thumbnail(source, matrix, reference_image.shape[:2])
            registered_mask = warp_mask_thumbnail(
                mask,
                matrix,
                reference_image.shape[:2],
            )
            rgba = np.dstack([registered, (registered_mask * 255).astype(np.uint8)])
            filename = f"{order:03d}-{_safe_name(source_path.stem)}.webp"
            Image.fromarray(rgba).save(
                mouse_assets / filename,
                "WEBP",
                lossless=False,
                quality=88,
                method=6,
            )
            slides.append(
                {
                    "id": source_path.name,
                    "label": _marker_label(source_path.stem),
                    "order": order,
                    "texture": f"assets/{_safe_name(mouse_id)}/{filename}",
                    "reference": bool(slide["is_reference"]),
                }
            )
        mouse_payloads.append(
            {
                "id": mouse_id,
                "provisional_order": mouse_id in provisional_mice,
                "width": int(reference_image.shape[1]),
                "height": int(reference_image.shape[0]),
                "slides": slides,
            }
        )

    (output_dir / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "mice": mouse_payloads}, indent=2) + "\n"
    )
    (output_dir / "index.html").write_text(
        _INDEX_HTML.replace("__THREE__", THREE_VERSION)
    )
    (output_dir / "viewer.js").write_text(
        _VIEWER_JS.replace("__THREE__", THREE_VERSION)
    )
    (output_dir / "styles.css").write_text(_STYLES_CSS)
    return output_dir / "index.html"


def _read_rgb(path: Path) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:
        raise OptionalDependencyError("pillow", "wsi") from exc
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def _read_mask(path: Path) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:
        raise OptionalDependencyError("pillow", "wsi") from exc
    with Image.open(path) as image:
        return np.asarray(image.convert("L")) > 127


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "section"


def _marker_label(stem: str) -> str:
    match = re.search(r"panc[_-](.+?)(?:-\[|$)", stem, flags=re.IGNORECASE)
    return match.group(1) if match else stem


_INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Histopia Section Stack</title>
  <link rel="stylesheet" href="styles.css">
  <script type="importmap">
    {"imports": {
      "three": "https://unpkg.com/three@__THREE__/build/three.module.js",
      "three/addons/": "https://unpkg.com/three@__THREE__/examples/jsm/"
    }}
  </script>
</head>
<body>
  <main>
    <aside>
      <h1>Histopia</h1>
      <label>Mouse<select id="mouse"></select></label>
      <p id="order-status"></p>
      <label>Spacing<input id="spacing" type="range" min="2" max="80" value="24"></label>
      <label>Opacity<input id="opacity" type="range" min="0.05" max="1" step="0.05" value="0.72"></label>
      <div class="commands">
        <button id="reset" title="Reset camera">Reset view</button>
        <button id="export" title="Export section order">Export order</button>
      </div>
      <ol id="sections"></ol>
    </aside>
    <section id="viewport" aria-label="Interactive registered section stack"></section>
  </main>
  <script type="module" src="viewer.js"></script>
</body>
</html>
"""

_VIEWER_JS = """import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const manifest = await (await fetch('manifest.json')).json();
const viewport = document.querySelector('#viewport');
const renderer = new THREE.WebGLRenderer({antialias: true, alpha: false});
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setClearColor(0xf4f5f3);
viewport.append(renderer.domElement);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 10000);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
const group = new THREE.Group();
scene.add(group);
const loader = new THREE.TextureLoader();
let current;

function resize() {
  const box = viewport.getBoundingClientRect();
  renderer.setSize(box.width, box.height, false);
  camera.aspect = box.width / box.height;
  camera.updateProjectionMatrix();
}
function resetCamera() {
  camera.position.set(0, -520, 390);
  controls.target.set(0, 0, 0);
  controls.update();
}
function orderedSlides() {
  return [...document.querySelectorAll('#sections li')].map(li =>
    current.slides.find(slide => slide.id === li.dataset.id));
}
function layout() {
  const spacing = Number(document.querySelector('#spacing').value);
  const opacity = Number(document.querySelector('#opacity').value);
  orderedSlides().forEach((slide, index, all) => {
    slide.mesh.position.z = (index - (all.length - 1) / 2) * spacing;
    slide.mesh.material.opacity = opacity;
  });
}
function buildList() {
  const list = document.querySelector('#sections');
  list.replaceChildren();
  current.slides.forEach(slide => {
    const item = document.createElement('li');
    item.dataset.id = slide.id;
    item.draggable = true;
    const toggle = document.createElement('input');
    toggle.type = 'checkbox'; toggle.checked = true;
    toggle.addEventListener('change', () => slide.mesh.visible = toggle.checked);
    const text = document.createElement('span');
    text.textContent = `${slide.label}${slide.reference ? ' (reference)' : ''}`;
    item.append(toggle, text);
    item.addEventListener('dragstart', event => event.dataTransfer.setData('text/plain', slide.id));
    item.addEventListener('dragover', event => event.preventDefault());
    item.addEventListener('drop', event => {
      event.preventDefault();
      const dragged = list.querySelector(`[data-id="${CSS.escape(event.dataTransfer.getData('text/plain'))}"]`);
      if (dragged && dragged !== item) list.insertBefore(dragged, item);
      localStorage.setItem(`histopia-order-${current.id}`, JSON.stringify(orderedSlides().map(s => s.id)));
      layout();
    });
    list.append(item);
  });
  const saved = JSON.parse(localStorage.getItem(`histopia-order-${current.id}`) || '[]');
  saved.forEach(id => { const item = list.querySelector(`[data-id="${CSS.escape(id)}"]`); if (item) list.append(item); });
}
async function loadMouse(mouse) {
  group.clear(); current = mouse;
  document.querySelector('#order-status').textContent = mouse.provisional_order ? 'Provisional section order' : 'Confirmed section order';
  const scale = 320 / Math.max(mouse.width, mouse.height);
  await Promise.all(mouse.slides.map(async slide => {
    const texture = await loader.loadAsync(slide.texture);
    texture.colorSpace = THREE.SRGBColorSpace;
    const material = new THREE.MeshBasicMaterial({map: texture, transparent: true, side: THREE.DoubleSide, depthWrite: false});
    slide.mesh = new THREE.Mesh(new THREE.PlaneGeometry(mouse.width * scale, mouse.height * scale), material);
    group.add(slide.mesh);
  }));
  buildList(); layout(); resetCamera();
}
const select = document.querySelector('#mouse');
manifest.mice.forEach(mouse => select.add(new Option(mouse.id, mouse.id)));
select.addEventListener('change', () => loadMouse(manifest.mice.find(mouse => mouse.id === select.value)));
document.querySelector('#spacing').addEventListener('input', layout);
document.querySelector('#opacity').addEventListener('input', layout);
document.querySelector('#reset').addEventListener('click', resetCamera);
document.querySelector('#export').addEventListener('click', () => {
  const blob = new Blob([JSON.stringify({mouse: current.id, slides: orderedSlides().map((s, i) => ({slide: s.id, order: i + 1}))}, null, 2)], {type: 'application/json'});
  const link = document.createElement('a'); link.href = URL.createObjectURL(blob); link.download = `${current.id}-section-order.json`; link.click(); URL.revokeObjectURL(link.href);
});
new ResizeObserver(resize).observe(viewport); resize(); resetCamera();
function animate() { requestAnimationFrame(animate); controls.update(); renderer.render(scene, camera); }
await loadMouse(manifest.mice[0]); animate();
"""

_STYLES_CSS = """*{box-sizing:border-box}body{margin:0;font-family:Arial,sans-serif;color:#202426;background:#f4f5f3}main{display:grid;grid-template-columns:300px 1fr;height:100vh}aside{padding:18px;border-right:1px solid #c9ceca;background:#fff;overflow:auto}h1{font-size:22px;margin:0 0 18px}label{display:grid;gap:6px;font-size:13px;margin:14px 0}select,input{width:100%}.commands{display:flex;gap:8px;margin:16px 0}button{border:1px solid #88918b;background:#fff;padding:7px 10px;border-radius:4px;cursor:pointer}#order-status{font-size:12px;color:#8a4f12}ol{padding:0;list-style:none}li{display:grid;grid-template-columns:20px 1fr;align-items:center;min-height:32px;border-bottom:1px solid #eceeec;font-size:12px;cursor:grab}li input{width:14px}#viewport{min-width:0;min-height:0}canvas{display:block}@media(max-width:720px){main{grid-template-columns:1fr;grid-template-rows:250px 1fr}aside{border-right:0;border-bottom:1px solid #c9ceca}#sections{display:none}}"""
