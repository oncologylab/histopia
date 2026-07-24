"""Combined static registration-review portal."""

from __future__ import annotations

import json
from pathlib import Path

from histopia.visualization._viewer import (
    build_mask_review,
    build_section_order_review,
)


def build_registration_review(
    registration_run: Path | str,
    output_dir: Path | str,
    *,
    workers: int = 1,
) -> Path:
    """Build one local entry point for mask and section-order review."""

    registration_run = Path(registration_run)
    output_dir = Path(output_dir)
    mask_index = build_mask_review(registration_run, output_dir / "mask")
    order_index = build_section_order_review(
        registration_run / "section_order_review.json",
        registration_run / "processed",
        output_dir / "order",
        workers=workers,
    )
    mask = json.loads((mask_index.parent / "manifest.json").read_text())
    order = json.loads((order_index.parent / "manifest.json").read_text())
    manifest = {
        "schema_version": 1,
        "mask": {
            "approved": bool(mask.get("approved")),
            "fingerprint": str(mask.get("fingerprint", "")),
            "slide_count": len(mask.get("slides", [])),
            "href": "mask/index.html",
        },
        "order": {
            "approved": bool(order.get("approved")),
            "fingerprint": str(order.get("fingerprint", "")),
            "slide_count": len(order.get("slides", [])),
            "href": "order/index.html",
        },
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
function select(stage){
  const selected=manifest[stage]?"mask"===stage?"mask":"order":"mask";
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
