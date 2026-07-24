"""Combined static registration-review portal."""

from __future__ import annotations

import json
import re
from pathlib import Path

from histopia.visualization._viewer import (
    build_alignment_review,
    build_mask_review,
    build_section_order_review,
)


def build_registration_review(
    registration_run: Path | str,
    output_dir: Path | str,
    *,
    workers: int = 1,
) -> Path:
    """Build one local entry point for every prepared registration-review stage."""

    registration_run = Path(registration_run)
    output_dir = Path(output_dir)
    mask_index = build_mask_review(registration_run, output_dir / "mask")
    mask = json.loads((mask_index.parent / "manifest.json").read_text())
    manifest = {
        "schema_version": 1,
        "mask": {
            "approved": bool(mask.get("approved")),
            "fingerprint": str(mask.get("fingerprint", "")),
            "slide_count": len(mask.get("slides", [])),
            "href": "mask/index.html",
        },
    }
    order_proposal = registration_run / "section_order_review.json"
    if order_proposal.is_file():
        order_index = build_section_order_review(
            order_proposal,
            registration_run / "processed",
            output_dir / "order",
            workers=workers,
        )
        order = json.loads((order_index.parent / "manifest.json").read_text())
        manifest["order"] = {
            "approved": bool(order.get("approved")),
            "fingerprint": str(order.get("fingerprint", "")),
            "slide_count": len(order.get("slides", [])),
            "href": "order/index.html",
        }
    registration_result = registration_run / "registration_result.json"
    if registration_result.is_file():
        alignment_index = build_alignment_review(
            registration_run,
            output_dir / "alignment",
            workers=workers,
        )
        alignment = json.loads((alignment_index.parent / "manifest.json").read_text())
        manifest["alignment"] = {
            "approved": bool(alignment.get("approved")),
            "fingerprint": str(alignment.get("fingerprint", "")),
            "slide_count": len(alignment.get("slides", [])),
            "href": "alignment/index.html",
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    encoded = json.dumps(manifest, separators=(",", ":"))
    (output_dir / "manifest-data.js").write_text(
        f"globalThis.HISTOPIA_REVIEW_MANIFEST={encoded};\n"
    )
    (output_dir / "index.html").write_text(_PORTAL_HTML)
    (output_dir / "registration-review.css").write_text(_PORTAL_CSS)
    (output_dir / "registration-review.js").write_text(_PORTAL_JS)
    return output_dir / "index.html"


def build_registration_cohort_review(
    runs: dict[str, Path | str],
    output_dir: Path | str,
    *,
    workers: int = 1,
) -> Path:
    """Build one fixed-viewport entry point for multiple registration reviews."""

    if not runs:
        raise ValueError("registration cohort review requires at least one run")
    output_dir = Path(output_dir)
    reviews: list[dict[str, object]] = []
    for name, run in runs.items():
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            raise ValueError(f"invalid registration review name: {name!r}")
        index = build_registration_review(
            run,
            output_dir / name,
            workers=workers,
        )
        manifest = json.loads((index.parent / "manifest.json").read_text())
        reviews.append(
            {
                "id": name,
                "href": f"{name}/index.html",
                "stages": [
                    stage
                    for stage in ("mask", "order", "alignment")
                    if stage in manifest
                ],
            }
        )
    manifest = {"schema_version": 1, "reviews": reviews}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    encoded = json.dumps(manifest, separators=(",", ":"))
    (output_dir / "manifest-data.js").write_text(
        f"globalThis.HISTOPIA_COHORT_REVIEW_MANIFEST={encoded};\n"
    )
    (output_dir / "index.html").write_text(_COHORT_PORTAL_HTML)
    (output_dir / "cohort-review.css").write_text(_COHORT_PORTAL_CSS)
    (output_dir / "cohort-review.js").write_text(_COHORT_PORTAL_JS)
    return output_dir / "index.html"


_PORTAL_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Histopia registration review</title>
  <link rel="stylesheet" href="registration-review.css">
</head>
<body>
  <header>
    <strong>Histopia registration review</strong>
    <nav aria-label="Review stage">
      <button type="button" data-stage="mask" aria-pressed="true">Tissue masks</button>
      <button type="button" data-stage="order" aria-pressed="false">
        Section order
      </button>
      <button type="button" data-stage="alignment" aria-pressed="false">
        Registered stack
      </button>
    </nav>
    <span id="status" role="status"></span>
  </header>
  <main>
    <iframe id="review" title="Histopia registration review"></iframe>
  </main>
  <script src="manifest-data.js"></script>
  <script src="registration-review.js"></script>
</body>
</html>
"""

_PORTAL_CSS = """
:root{font-family:Inter,system-ui,sans-serif;color:#17202a;background:#f4f6f7}
*{box-sizing:border-box}
html,body{width:100%;height:100%;margin:0;overflow:hidden}
body{display:grid;grid-template-rows:48px minmax(0,1fr)}
header{display:flex;align-items:center;gap:18px;padding:0 16px;background:#fff;
border-bottom:1px solid #ccd1d1;min-width:0}
header strong{white-space:nowrap}
nav{display:flex;align-self:stretch}
button{border:0;border-bottom:3px solid transparent;background:transparent;
padding:0 14px;color:#566573;font:inherit;cursor:pointer}
button[aria-pressed="true"]{border-bottom-color:#117864;color:#0b5345;font-weight:600}
#status{margin-left:auto;color:#566573;font-size:13px;white-space:nowrap}
main,iframe{width:100%;height:100%;min-width:0;min-height:0;border:0}
@media(max-width:700px){
  body{grid-template-rows:76px minmax(0,1fr)}
  header{gap:6px;padding:4px 8px;flex-wrap:wrap}
  header strong{width:100%;font-size:13px}
  nav{height:34px}
  button{padding:0 9px;font-size:12px}
  #status{font-size:11px}
}
"""

_PORTAL_JS = """
const manifest=globalThis.HISTOPIA_REVIEW_MANIFEST;
if(!manifest)throw new Error("Missing embedded Histopia review manifest");
const frame=document.querySelector("#review");
const status=document.querySelector("#status");
const buttons=[...document.querySelectorAll("[data-stage]")];
const stages=["mask","order","alignment"].filter(stage=>manifest[stage]);
buttons.forEach(button=>button.hidden=!manifest[button.dataset.stage]);
function select(stage){
  const selected=stages.includes(stage)?stage:stages[0];
  const row=manifest[selected];
  buttons.forEach(button=>button.setAttribute(
    "aria-pressed",String(button.dataset.stage===selected)));
  frame.src=row.href;
  const approval=row.approved?"approved":"review required";
  status.textContent=`${row.slide_count} slides · ${approval}`;
  const url=new URL(location.href);
  url.searchParams.set("stage",selected);
  history.replaceState(null,"",url);
}
buttons.forEach(button=>button.addEventListener("click",()=>select(button.dataset.stage)));
select(new URL(location.href).searchParams.get("stage")||"mask");
"""

_COHORT_PORTAL_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Histopia registration cohort review</title>
  <link rel="stylesheet" href="cohort-review.css">
</head>
<body>
  <header>
    <strong>Histopia registration review</strong>
    <label for="cohort">Cohort</label>
    <select id="cohort"></select>
    <span id="status" role="status"></span>
  </header>
  <main>
    <iframe id="review" title="Histopia cohort registration review"></iframe>
  </main>
  <script src="manifest-data.js"></script>
  <script src="cohort-review.js"></script>
</body>
</html>
"""

_COHORT_PORTAL_CSS = """
:root{font-family:Inter,system-ui,sans-serif;color:#17202a;background:#f4f6f7}
*{box-sizing:border-box}
html,body{width:100%;height:100%;margin:0;overflow:hidden}
body{display:grid;grid-template-rows:48px minmax(0,1fr)}
header{display:flex;align-items:center;gap:10px;padding:0 16px;background:#fff;
border-bottom:1px solid #ccd1d1;min-width:0}
header strong{margin-right:12px;white-space:nowrap}
label,#status{font-size:13px;color:#566573}
select{min-width:110px;padding:5px 8px;border:1px solid #aeb6bf;background:#fff}
#status{margin-left:auto;white-space:nowrap}
main,iframe{width:100%;height:100%;min-width:0;min-height:0;border:0}
@media(max-width:600px){
  header{padding:0 8px;gap:6px}
  header strong{font-size:12px;margin-right:2px}
  label{display:none}
  select{min-width:80px}
  #status{font-size:10px}
}
"""

_COHORT_PORTAL_JS = """
const manifest=globalThis.HISTOPIA_COHORT_REVIEW_MANIFEST;
if(!manifest||!manifest.reviews.length)throw new Error("Missing cohort reviews");
const select=document.querySelector("#cohort");
const frame=document.querySelector("#review");
const status=document.querySelector("#status");
for(const row of manifest.reviews){
  const option=document.createElement("option");
  option.value=row.id;
  option.textContent=row.id;
  select.append(option);
}
function choose(id){
  const row=manifest.reviews.find(item=>item.id===id)||manifest.reviews[0];
  select.value=row.id;
  frame.src=row.href;
  status.textContent=row.stages.join(" · ");
  const url=new URL(location.href);
  url.searchParams.set("cohort",row.id);
  history.replaceState(null,"",url);
}
select.addEventListener("change",()=>choose(select.value));
choose(new URL(location.href).searchParams.get("cohort"));
"""
