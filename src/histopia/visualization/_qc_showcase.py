"""Export registration workflow reviews as a static QC portal."""

from __future__ import annotations

import json
import shutil
from collections.abc import Sequence
from pathlib import Path, PurePosixPath

from histopia.visualization._showcase import (
    _VIEWER_DIRECTORIES,
    _VIEWER_FILES,
    _file_inventory,
    _reject_local_paths,
)

_INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Histopia Registration QC</title>
  <link rel="icon" href="data:">
  <link rel="stylesheet" href="styles.css">
</head>
<body>
  <header>
    <strong>Histopia Registration QC</strong>
    <label>Mouse <select id="mouse"></select></label>
    <nav aria-label="Review stage">
      <button data-stage="mask" class="active">Mask &amp; rotation</button>
      <button data-stage="order">Section order</button>
      <button data-stage="registration">3D registration</button>
    </nav>
    <a href="../">Semantic atlas</a>
  </header>
  <main><iframe id="review" title="Registration quality-control review"></iframe></main>
  <script type="module" src="portal.js"></script>
</body>
</html>
"""

_STYLES_CSS = """*{box-sizing:border-box}
html,body{width:100%;height:100%;margin:0;overflow:hidden}
body{background:#f4f5f3;color:#1d2326;font:13px Arial,sans-serif}
header{height:52px;display:flex;align-items:center;gap:18px;padding:7px 14px;
  border-bottom:1px solid #aeb5b7;background:#fff}
strong{font-size:17px;white-space:nowrap}
label{display:flex;align-items:center;gap:7px}
select{width:112px;height:32px;border:1px solid #899295;background:#fff}
nav{display:flex;height:32px}
button{height:32px;padding:0 14px;border:1px solid #899295;border-right:0;
  background:#fff;color:#1d2326;cursor:pointer}
button:last-child{border-right:1px solid #899295}
button.active{background:#1d292c;color:#fff}
button:disabled{background:#eef0ef;color:#8a9294;cursor:not-allowed}
a{margin-left:auto;color:#006d77;text-decoration:none;font-weight:600}
main{width:100%;height:calc(100vh - 52px)}
iframe{display:block;width:100%;height:100%;border:0;background:#fff}
@media(max-width:760px){
  header{height:126px;display:grid;grid-template-columns:1fr auto;
    grid-template-rows:26px 32px 32px;gap:5px 10px}
  strong{font-size:15px}a{grid-column:2;grid-row:1;margin-left:0}
  label{grid-column:1/3;grid-row:2}select{flex:1;width:auto}
  nav{grid-column:1/3;grid-row:3;width:100%}
  nav button{flex:1;padding:0 5px}
  main{height:calc(100vh - 126px)}
}
"""

_PORTAL_JS = """const data = await (await fetch('qc-manifest.json')).json();
const mouse = document.querySelector('#mouse');
const frame = document.querySelector('#review');
const buttons = [...document.querySelectorAll('[data-stage]')];
let stage = 'mask';

data.mice.forEach(item => mouse.add(new Option(item.id, item.id)));

function current() {
  return data.mice.find(item => item.id === mouse.value);
}

function selectRegistrationMouse(attempt = 0) {
  if (stage !== 'registration') return;
  try {
    const select = frame.contentDocument?.querySelector('#mouse');
    const ready = select && [...select.options].some(
      option => option.value === mouse.value);
    if (!ready) {
      if (attempt < 20) setTimeout(() => selectRegistrationMouse(attempt + 1), 100);
      return;
    }
    select.value = mouse.value;
    select.dispatchEvent(new Event('change'));
  } catch (_) {
    // The exported portal is same-origin; ignore transient iframe load state.
  }
}

function render() {
  const item = current();
  if (!item.stages[stage]) stage = 'mask';
  buttons.forEach(button => {
    button.disabled = !item.stages[button.dataset.stage];
  });
  buttons.forEach(button =>
    button.classList.toggle('active', button.dataset.stage === stage));
  frame.src = item.stages[stage];
}

mouse.addEventListener('change', () => {
  render();
  if (stage === 'registration') selectRegistrationMouse();
});
buttons.forEach(button => button.addEventListener('click', () => {
  if (button.disabled) return;
  stage = button.dataset.stage;
  render();
}));
frame.addEventListener('load', () => selectRegistrationMouse());
render();
"""


def export_registration_qc_showcase(
    source_dir: Path | str,
    output_dir: Path | str,
    mouse_ids: str | Sequence[str],
) -> Path:
    """Export selected mask, orientation, order, and registration reviews."""

    source = Path(source_dir)
    output = Path(output_dir)
    selected_ids = (mouse_ids,) if isinstance(mouse_ids, str) else tuple(mouse_ids)
    if not selected_ids:
        raise ValueError("QC showcase requires at least one viewer mouse")
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError("QC showcase contains a duplicate viewer mouse")
    if source.resolve() == output.resolve():
        raise ValueError("QC showcase output must differ from the source viewer")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"QC showcase output directory is not empty: {output}")

    manifest = json.loads((source / "manifest.json").read_text())
    mice = manifest.get("mice")
    if not isinstance(mice, list):
        raise ValueError("viewer manifest must contain a mice list")
    mice_by_id = {str(mouse.get("id")): mouse for mouse in mice}
    for mouse_id in selected_ids:
        for kind in ("mask", "order"):
            review = source / f"{mouse_id}-{kind}-review"
            if not review.is_dir():
                raise FileNotFoundError(
                    f"{kind} review not found for mouse: {mouse_id}"
                )

    output.mkdir(parents=True, exist_ok=True)
    registration: Path | None = None
    registration_mice = []
    portal_mice = []
    for mouse_id in selected_ids:
        review_root = output / "reviews" / mouse_id
        ignore_cache = shutil.ignore_patterns(".histopia-*")
        shutil.copytree(
            source / f"{mouse_id}-mask-review",
            review_root / "mask",
            ignore=ignore_cache,
        )
        shutil.copytree(
            source / f"{mouse_id}-order-review",
            review_root / "order",
            ignore=ignore_cache,
        )
        stages = {
            "mask": f"reviews/{mouse_id}/mask/",
            "order": f"reviews/{mouse_id}/order/",
        }
        mouse = mice_by_id.get(mouse_id)
        if mouse is not None:
            if registration is None:
                registration = output / "registration"
                registration.mkdir()
                for filename in _VIEWER_FILES:
                    source_file = source / filename
                    if not source_file.is_file():
                        raise FileNotFoundError(f"viewer file not found: {filename}")
                    shutil.copy2(source_file, registration / filename)
                for directory in _VIEWER_DIRECTORIES:
                    source_directory = source / directory
                    if not source_directory.is_dir():
                        raise FileNotFoundError(
                            f"viewer directory not found: {directory}"
                        )
                    shutil.copytree(source_directory, registration / directory)
            slides = []
            for slide in mouse.get("slides", []):
                texture = _relative_asset(str(slide["texture"]), mouse_id)
                source_texture = source / texture
                if not source_texture.is_file():
                    raise FileNotFoundError(
                        f"registration texture not found: {texture}"
                    )
                destination = registration / texture
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_texture, destination)
                slides.append(
                    {
                        key: value
                        for key, value in slide.items()
                        if key
                        not in {
                            "semantic_texture",
                            "semantic_textures",
                            "blend_texture",
                        }
                    }
                )
            registration_mice.append(
                {
                    key: value
                    for key, value in mouse.items()
                    if key not in {"slides", "semantic"}
                }
                | {"slides": slides, "semantic": None}
            )
            stages["registration"] = "registration/"
        portal_mice.append(
            {
                "id": mouse_id,
                "stages": stages,
            }
        )

    portal_manifest = {"schema_version": 1, "mice": portal_mice}
    if registration is not None:
        registration_manifest = {
            "schema_version": manifest.get("schema_version", 1),
            "mice": registration_mice,
        }
        _reject_local_paths(registration_manifest)
        (registration / "manifest.json").write_text(
            json.dumps(registration_manifest, indent=2) + "\n"
        )
    _reject_local_paths(portal_manifest)
    (output / "qc-manifest.json").write_text(
        json.dumps(portal_manifest, indent=2) + "\n"
    )
    (output / "index.html").write_text(_INDEX_HTML)
    (output / "styles.css").write_text(_STYLES_CSS)
    (output / "portal.js").write_text(_PORTAL_JS)
    (output / ".nojekyll").touch()
    _reject_unsafe_text(output)

    inventory = {
        "schema_version": 1,
        "mouse_ids": list(selected_ids),
        "files": _file_inventory(output),
    }
    (output / "qc-showcase.json").write_text(json.dumps(inventory, indent=2) + "\n")
    return output / "index.html"


def _relative_asset(value: str, mouse_id: str) -> Path:
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("viewer manifest contains an unsafe registration texture")
    expected = ("assets", mouse_id)
    if relative.parts[:2] != expected:
        raise ValueError("registration texture is outside the selected mouse assets")
    return Path(*relative.parts)


def _reject_unsafe_text(root: Path) -> None:
    for path in root.rglob("*"):
        if path.suffix.lower() not in {".css", ".html", ".js", ".json", ".txt"}:
            continue
        text = path.read_text()
        if "/home/" in text or "/media/" in text or "file://" in text:
            raise ValueError(f"QC review contains a local absolute path: {path.name}")
