"""Combined static registration-review portal."""

from __future__ import annotations

import json
import re
from pathlib import Path

from histopia.visualization._registration_state import (
    current_registration_review_stages,
    registration_artifact_slide_names,
)
from histopia.visualization._viewer import (
    build_alignment_review,
    build_mask_review,
    build_section_order_review,
    build_section_viewer,
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
    available_stages = current_registration_review_stages(registration_run)
    if "mask" not in available_stages:
        raise ValueError("latest registration execution has not prepared mask review")
    mask_index = build_mask_review(
        registration_run,
        output_dir / "mask",
        workers=workers,
    )
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
    mask_names = registration_artifact_slide_names(
        registration_run / "mask_review.json",
        field="slide",
    )
    order_is_current = mask_names is None
    if "order" in available_stages:
        order_proposal = registration_run / "section_order_review.json"
        order_names = registration_artifact_slide_names(order_proposal, field="slide")
        order_is_current = mask_names is None or order_names == mask_names
        if order_names is not None and order_is_current:
            order_index = build_section_order_review(
                order_proposal,
                registration_run / "processed",
                output_dir / "order",
                workers=workers,
            )
            order = json.loads((order_index.parent / "manifest.json").read_text())
            area_continuity = order.get("physical_area_continuity")
            area_review_recommended = bool(
                isinstance(area_continuity, dict)
                and area_continuity.get("review_recommended") is True
            )
            manifest["order"] = {
                "approved": bool(order.get("approved")),
                "fingerprint": str(order.get("fingerprint", "")),
                "slide_count": len(order.get("slides", [])),
                "review_recommended": area_review_recommended,
                "href": "order/index.html",
            }
    if "alignment" in available_stages:
        registration_result = registration_run / "registration_result.json"
        result_names = registration_artifact_slide_names(
            registration_result,
            field="path",
        )
        result_is_current = mask_names is None or result_names == mask_names
        if result_names is not None and result_is_current and order_is_current:
            alignment_index = build_alignment_review(
                registration_run,
                output_dir / "alignment",
                workers=workers,
            )
            alignment = json.loads(
                (alignment_index.parent / "manifest.json").read_text()
            )
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
        stages: list[str] = []
        stage_summary: dict[str, dict[str, object]] = {}
        slide_counts: set[int] = set()
        for stage in ("mask", "order", "alignment"):
            row = manifest.get(stage)
            if row is None:
                continue
            if not isinstance(row, dict):
                raise ValueError(f"{name} {stage} review summary must be an object")
            approved = row.get("approved")
            slide_count = row.get("slide_count")
            if not isinstance(approved, bool):
                raise ValueError(f"{name} {stage} review approval must be a boolean")
            if (
                isinstance(slide_count, bool)
                or not isinstance(slide_count, int)
                or slide_count < 1
            ):
                raise ValueError(
                    f"{name} {stage} review slide count must be a positive integer"
                )
            stages.append(stage)
            slide_counts.add(slide_count)
            stage_summary[stage] = {
                "approved": approved,
                "slide_count": slide_count,
            }
            review_recommended = row.get("review_recommended")
            if review_recommended is not None:
                if not isinstance(review_recommended, bool):
                    raise ValueError(
                        f"{name} {stage} review recommendation must be a boolean"
                    )
                stage_summary[stage]["review_recommended"] = review_recommended
        if not stages:
            raise ValueError(f"{name} registration review has no prepared stages")
        if len(slide_counts) != 1:
            raise ValueError(f"{name} registration review stage slide counts differ")
        reviews.append(
            {
                "id": name,
                "href": f"{name}/index.html",
                "slide_count": slide_counts.pop(),
                "stages": stages,
                "stage_summary": stage_summary,
            }
        )
    reviews.sort(
        key=lambda review: (
            all(
                bool(summary.get("approved"))
                for summary in review["stage_summary"].values()
            ),
            str(review["id"]),
        )
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


def build_workflow_review(
    registration_runs: dict[str, Path | str],
    output_dir: Path | str,
    *,
    semantic_runs: dict[str, Path | str] | None = None,
    stain_runs: dict[str, Path | str] | None = None,
    topology_runs: dict[str, Path | str] | None = None,
    cohort_qc: Path | str | None = None,
    workers: int = 1,
) -> Path:
    """Build one stable review hub for all prepared workflow stages."""

    semantic_runs = semantic_runs or {}
    stain_runs = stain_runs or {}
    topology_runs = topology_runs or {}
    unknown = (set(semantic_runs) | set(stain_runs) | set(topology_runs)) - set(
        registration_runs
    )
    if unknown:
        raise ValueError(
            "review inputs have no matching registration: " + ", ".join(sorted(unknown))
        )
    output_dir = Path(output_dir)
    tabs: list[dict[str, str]] = []
    registration_index = build_registration_cohort_review(
        registration_runs,
        output_dir / "registration",
        workers=workers,
    )
    tabs.append(
        {
            "id": "registration",
            "label": "Registration",
            "href": registration_index.relative_to(output_dir).as_posix(),
        }
    )
    completed = {
        name: run
        for name, run in registration_runs.items()
        if (Path(run) / "registration_result.json").is_file()
    }
    if completed:
        atlas_index = build_section_viewer(
            completed,
            output_dir / "atlas",
            semantic_runs={
                name: semantic_runs[name] for name in completed if name in semantic_runs
            },
            stain_runs={
                name: stain_runs[name] for name in completed if name in stain_runs
            },
            cohort_qc=cohort_qc,
            workers=workers,
            require_approvals=False,
        )
        tabs.append(
            {
                "id": "atlas",
                "label": "3D atlas",
                "href": atlas_index.relative_to(output_dir).as_posix(),
            }
        )
        stain_mice = sorted(set(completed) & set(stain_runs))
        if stain_mice:
            from histopia.visualization._stain_review import build_stain_review

            stain_index = build_stain_review(
                atlas_index.parent,
                output_dir / "stain",
                mice=stain_mice,
            )
            tabs.append(
                {
                    "id": "stain",
                    "label": "Stain",
                    "href": stain_index.relative_to(output_dir).as_posix(),
                }
            )
        topology_mice = sorted(set(completed) & set(topology_runs))
        if topology_mice:
            from histopia.visualization._topology_review import build_topology_review

            topology_index = build_topology_review(
                {name: topology_runs[name] for name in topology_mice},
                output_dir / "topology",
            )
            tabs.append(
                {
                    "id": "topology",
                    "label": "Topology",
                    "href": topology_index.relative_to(output_dir).as_posix(),
                }
            )
    decisions_index = _write_decisions_page(output_dir / "decisions")
    tabs.append(
        {
            "id": "decisions",
            "label": "Decisions",
            "href": decisions_index.relative_to(output_dir).as_posix(),
        }
    )
    manifest = {"schema_version": 1, "tabs": tabs}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    encoded = json.dumps(manifest, separators=(",", ":"))
    (output_dir / "manifest-data.js").write_text(
        f"globalThis.HISTOPIA_WORKFLOW_REVIEW={encoded};\n"
    )
    (output_dir / "index.html").write_text(_WORKFLOW_HTML)
    (output_dir / "workflow-review.css").write_text(_WORKFLOW_CSS)
    (output_dir / "workflow-review.js").write_text(_WORKFLOW_JS)
    return output_dir / "index.html"


def _write_decisions_page(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "index.html").write_text(_DECISIONS_HTML)
    (output_dir / "review-decisions.css").write_text(_DECISIONS_CSS)
    (output_dir / "review-decisions.js").write_text(_DECISIONS_JS)
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
iframe{display:block}
@media(max-width:700px){
  body{grid-template-rows:82px minmax(0,1fr)}
  header{display:grid;grid-template-rows:20px 30px 20px;gap:2px;padding:4px 8px}
  header strong{grid-row:1;width:100%;font-size:13px}
  nav{grid-row:2;height:30px}
  button{padding:0 9px;font-size:12px}
  #status{grid-row:3;margin-left:0;min-width:0;overflow:hidden;
    text-overflow:ellipsis;font-size:11px}
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
  const continuity=row.review_recommended?" · area continuity flag":"";
  status.textContent=`${row.slide_count} slides · ${approval}${continuity}`;
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
#status{min-width:0;margin-left:auto;overflow:hidden;text-overflow:ellipsis;
white-space:nowrap;text-align:right}
main,iframe{width:100%;height:100%;min-width:0;min-height:0;border:0}
iframe{display:block}
@media(max-width:600px){
  body{grid-template-rows:70px minmax(0,1fr)}
  header{display:grid;grid-template-columns:minmax(0,1fr) 92px;
    grid-template-rows:30px 24px;padding:4px 8px;gap:4px 8px}
  header strong{grid-column:1;grid-row:1;font-size:12px;margin-right:0}
  label{display:none}
  select{grid-column:2;grid-row:1;width:100%;min-width:0}
  #status{grid-column:1/3;grid-row:2;margin-left:0;font-size:10px;text-align:left}
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
  const names={mask:"masks",order:"order",alignment:"registration"};
  const parts=row.stages.map(stage=>{
    const summary=row.stage_summary?.[stage];
    if(!summary)return names[stage]||stage;
    const continuity=summary.review_recommended?" (continuity flag)":"";
    return `${names[stage]||stage} ${
      summary.approved?"approved":"review required"}${continuity}`;
  });
  const count=Number.isInteger(row.slide_count)?`${row.slide_count} slides · `:"";
  status.textContent=count+parts.join(" · ");
  status.title=status.textContent;
  const url=new URL(location.href);
  url.searchParams.set("cohort",row.id);
  history.replaceState(null,"",url);
}
select.addEventListener("change",()=>choose(select.value));
choose(new URL(location.href).searchParams.get("cohort"));
"""

_WORKFLOW_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Histopia scientific review</title>
  <link rel="stylesheet" href="workflow-review.css">
</head>
<body>
  <header>
    <strong>Histopia scientific review</strong>
    <nav aria-label="Workflow stage"></nav>
  </header>
  <main><iframe id="review" title="Histopia scientific review"></iframe></main>
  <script src="manifest-data.js"></script>
  <script src="workflow-review.js"></script>
</body>
</html>
"""

_WORKFLOW_CSS = """
:root{font-family:Inter,system-ui,sans-serif;color:#17202a;background:#f4f6f7}
*{box-sizing:border-box}
html,body{width:100%;height:100%;margin:0;overflow:hidden}
body{display:grid;grid-template-rows:48px minmax(0,1fr)}
header{display:flex;align-items:center;gap:20px;padding:0 16px;background:#fff;
border-bottom:1px solid #ccd1d1;min-width:0}
header strong{white-space:nowrap}
nav{display:flex;align-self:stretch;min-width:0}
button{border:0;border-bottom:3px solid transparent;background:transparent;
padding:0 14px;color:#566573;font:inherit;cursor:pointer;white-space:nowrap}
button[aria-pressed="true"]{border-bottom-color:#117864;color:#0b5345;font-weight:600}
main,iframe{width:100%;height:100%;min-width:0;min-height:0;border:0}
iframe{display:block}
@media(max-width:560px){
  body{grid-template-rows:72px minmax(0,1fr)}
  header{display:grid;grid-template-rows:22px 42px;gap:0;padding:4px 8px}
  header strong{font-size:12px}
  nav{overflow-x:auto}
  button{padding:0 10px;font-size:12px}
}
"""

_WORKFLOW_JS = """
const manifest=globalThis.HISTOPIA_WORKFLOW_REVIEW;
if(!manifest||!manifest.tabs.length)throw new Error("Missing workflow review tabs");
const nav=document.querySelector("nav");
const frame=document.querySelector("#review");
for(const tab of manifest.tabs){
  const button=document.createElement("button");
  button.type="button";
  button.dataset.tab=tab.id;
  button.textContent=tab.label;
  button.addEventListener("click",()=>select(tab.id));
  nav.append(button);
}
function select(id){
  const tab=manifest.tabs.find(item=>item.id===id)||manifest.tabs[0];
  for(const button of nav.querySelectorAll("button")){
    button.setAttribute("aria-pressed",String(button.dataset.tab===tab.id));
  }
  frame.src=tab.href;
  const url=new URL(location.href);
  url.searchParams.set("view",tab.id);
  history.replaceState(null,"",url);
}
select(new URL(location.href).searchParams.get("view"));
"""

_DECISIONS_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Histopia review decisions</title>
  <link rel="stylesheet" href="review-decisions.css">
</head>
<body>
  <header>
    <strong>Review decisions</strong>
    <label for="access-key">Access key</label>
    <input id="access-key" type="password" autocomplete="current-password">
    <button id="connect" type="button">Connect</button>
    <span id="connection" role="status">Locked</span>
  </header>
  <main>
    <aside>
      <label for="cohort">Cohort</label>
      <select id="cohort" disabled></select>
      <nav id="stages" aria-label="Approval stage"></nav>
    </aside>
    <form id="decision">
      <div class="heading">
        <h1 id="title">Select a review</h1>
        <span id="state"></span>
      </div>
      <fieldset id="families" hidden>
        <legend>Stain families</legend>
      </fieldset>
      <label for="reviewer">Reviewer</label>
      <input id="reviewer" autocomplete="name" required>
      <label for="notes">Decision notes</label>
      <textarea id="notes" required></textarea>
      <div class="actions">
        <span id="message" role="status"></span>
        <button id="approve" class="primary" type="submit" disabled>
          Approve exact result
        </button>
      </div>
    </form>
  </main>
  <script src="review-decisions.js"></script>
</body>
</html>
"""

_DECISIONS_CSS = """
:root{font-family:Inter,system-ui,sans-serif;color:#17202a;background:#f4f6f7}
*{box-sizing:border-box}
html,body{width:100%;height:100%;margin:0;overflow:hidden}
body{display:grid;grid-template-rows:52px minmax(0,1fr)}
header{display:flex;align-items:center;gap:10px;padding:0 16px;background:#fff;
border-bottom:1px solid #ccd1d1}
header strong{margin-right:auto}
header label{font-size:12px;color:#566573}
input,textarea,select,button{font:inherit}
input,textarea,select{border:1px solid #aeb6bf;background:#fff;padding:7px 9px}
#access-key{width:min(260px,30vw)}
button{border:1px solid #aeb6bf;background:#fff;padding:7px 12px;cursor:pointer}
button:disabled{cursor:not-allowed;opacity:.5}
#connection{width:72px;font-size:12px;color:#7b241c}
main{display:grid;grid-template-columns:230px minmax(0,1fr);min-height:0}
aside{padding:16px;border-right:1px solid #ccd1d1;background:#fff;overflow:auto}
aside label{display:block;margin-bottom:5px;font-size:12px;color:#566573}
select{width:100%;margin-bottom:18px}
nav{display:grid;gap:6px}
nav button{text-align:left;border:0;border-left:3px solid transparent}
nav button[aria-pressed="true"]{border-left-color:#117864;background:#e8f6f3;
color:#0b5345;font-weight:600}
nav button.approved::after{content:"Approved";float:right;font-size:10px;color:#117864}
form{display:grid;grid-template-columns:130px minmax(0,680px);
grid-template-rows:auto auto auto minmax(100px,1fr) auto;align-content:start;
gap:12px 16px;padding:24px 32px;overflow:auto}
.heading{grid-column:1/3;display:flex;align-items:center;gap:14px}
h1{font-size:22px;margin:0;letter-spacing:0}
#state{font-size:12px;color:#566573}
form>label{font-size:13px;color:#566573;padding-top:8px}
textarea{min-height:120px;resize:vertical}
fieldset{grid-column:1/3;border:1px solid #ccd1d1;padding:12px}
fieldset label{display:inline-flex;align-items:center;gap:6px;margin:0 18px 4px 0}
.actions{grid-column:1/3;display:flex;align-items:center;justify-content:flex-end;
gap:14px}
#message{margin-right:auto;font-size:12px;color:#7b241c}
.primary{border-color:#0e6655;background:#117864;color:#fff}
@media(max-width:700px){
  body{grid-template-rows:92px minmax(0,1fr)}
  header{display:grid;grid-template-columns:minmax(0,1fr) auto;
    grid-template-rows:28px 40px;padding:8px}
  header strong{grid-column:1/3}
  header label{display:none}
  #access-key{width:100%}
  #connection{display:none}
  main{grid-template-columns:130px minmax(0,1fr)}
  aside{padding:10px}
  form{grid-template-columns:1fr;padding:16px;gap:8px}
  .heading,fieldset,.actions{grid-column:1}
  form>label{padding-top:4px}
  h1{font-size:17px}
}
"""

_DECISIONS_JS = """
const keyInput=document.querySelector("#access-key");
const connect=document.querySelector("#connect");
const connection=document.querySelector("#connection");
const cohortSelect=document.querySelector("#cohort");
const stages=document.querySelector("#stages");
const form=document.querySelector("#decision");
const title=document.querySelector("#title");
const state=document.querySelector("#state");
const families=document.querySelector("#families");
const reviewer=document.querySelector("#reviewer");
const notes=document.querySelector("#notes");
const message=document.querySelector("#message");
const approve=document.querySelector("#approve");
const labels={mask:"Tissue masks",order:"Section order",
  registration:"Registered stack",semantic:"Semantic atlas",
  topology:"Semantic topology",stain:"Stain"};
let registry=null;
let selectedStage=null;
keyInput.value=sessionStorage.getItem("histopiaReviewKey")||"";
reviewer.value=sessionStorage.getItem("histopiaReviewer")||"";
function headers(){
  return {"Authorization":`Bearer ${keyInput.value}`,"Content-Type":"application/json"};
}
async function load(){
  message.textContent="";
  const response=await fetch("/api/reviews",{headers:headers(),cache:"no-store"});
  if(!response.ok)throw new Error((await response.json()).error||"Unable to connect");
  registry=await response.json();
  sessionStorage.setItem("histopiaReviewKey",keyInput.value);
  connection.textContent="Connected";
  connection.style.color="#117864";
  cohortSelect.disabled=false;
  const previous=cohortSelect.value;
  cohortSelect.replaceChildren();
  for(const row of registry.cohorts){
    const option=document.createElement("option");
    option.value=row.id;
    option.textContent=row.id;
    cohortSelect.append(option);
  }
  const requested=new URL(location.href).searchParams.get("cohort");
  cohortSelect.value=registry.cohorts.some(row=>row.id===(requested||previous))
    ?(requested||previous):registry.cohorts[0].id;
  renderStages();
}
function current(){
  return registry?.cohorts.find(row=>row.id===cohortSelect.value);
}
function renderStages(){
  const row=current();
  stages.replaceChildren();
  if(!row)return;
  const available=registry.stages.filter(id=>row.stages[id]?.available);
  if(!available.includes(selectedStage))selectedStage=available[0]||null;
  for(const id of available){
    const button=document.createElement("button");
    button.type="button";
    button.textContent=labels[id]||id;
    button.classList.toggle("approved",Boolean(row.stages[id].approved));
    button.setAttribute("aria-pressed",String(id===selectedStage));
    button.addEventListener("click",()=>{selectedStage=id;renderStages();});
    stages.append(button);
  }
  renderDecision();
  const url=new URL(location.href);
  url.searchParams.set("cohort",row.id);
  history.replaceState(null,"",url);
}
function renderDecision(){
  const stage=current()?.stages[selectedStage];
  title.textContent=selectedStage?labels[selectedStage]:"No prepared review";
  state.textContent=stage?.approved?"Approved":
    stage?.invalid?"Invalid artifacts":
    stage?.approval_ready===false?"Upstream rebuild required":"Review required";
  state.title=stage?.issue||"";
  approve.disabled=!stage||stage.approved||stage.invalid||
    stage.approval_ready===false;
  families.hidden=selectedStage!=="stain";
  families.querySelectorAll("label").forEach(element=>element.remove());
  if(selectedStage==="stain"){
    for(const family of stage.families||[]){
      const label=document.createElement("label");
      const input=document.createElement("input");
      input.type="checkbox";
      input.value=family.id;
      input.checked=!family.approved;
      input.disabled=family.approved;
      label.append(input,document.createTextNode(family.id));
      families.append(label);
    }
  }
}
connect.addEventListener("click",()=>load().catch(error=>{
  connection.textContent="Locked";
  connection.style.color="#7b241c";
  message.textContent=error.message;
}));
keyInput.addEventListener("keydown",event=>{
  if(event.key==="Enter"){event.preventDefault();connect.click();}
});
cohortSelect.addEventListener("change",renderStages);
form.addEventListener("submit",async event=>{
  event.preventDefault();
  message.textContent="";
  sessionStorage.setItem("histopiaReviewer",reviewer.value);
  const payload={cohort:cohortSelect.value,stage:selectedStage,
    reviewer:reviewer.value,notes:notes.value};
  if(selectedStage==="stain"){
    payload.families=[...families.querySelectorAll("input:checked")]
      .map(input=>input.value);
  }
  if(!confirm(`Approve ${labels[selectedStage]} for ${payload.cohort}?`))return;
  approve.disabled=true;
  try{
    const response=await fetch("/api/reviews/approve",
      {method:"POST",headers:headers(),body:JSON.stringify(payload)});
    const result=await response.json();
    if(!response.ok)throw new Error(result.error||"Approval failed");
    notes.value="";
    await load();
    message.style.color="#117864";
    message.textContent="Approval recorded";
  }catch(error){
    message.style.color="#7b241c";
    message.textContent=error.message;
    renderDecision();
  }
});
if(keyInput.value)load().catch(()=>{});
"""
