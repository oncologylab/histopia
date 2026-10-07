"""Combined static registration-review portal."""

from __future__ import annotations

import json
import re
from pathlib import Path

from histopia.visualization._cell_scope import CellSectionScope
from histopia.visualization._registration_state import (
    current_registration_review_stages,
    registration_artifact_slide_names,
)
from histopia.visualization._review_selection import REVIEW_SELECTION_JS
from histopia.visualization._review_theme import themed_review_css
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
    (output_dir / "registration-review.css").write_text(themed_review_css(_PORTAL_CSS))
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
    (output_dir / "cohort-review.css").write_text(themed_review_css(_COHORT_PORTAL_CSS))
    (output_dir / "cohort-review.js").write_text(_COHORT_PORTAL_JS)
    return output_dir / "index.html"


def build_workflow_review(
    registration_runs: dict[str, Path | str],
    output_dir: Path | str,
    *,
    semantic_runs: dict[str, Path | str] | None = None,
    stain_runs: dict[str, Path | str] | None = None,
    topology_runs: dict[str, Path | str] | None = None,
    cell_runs: dict[str, Path | str] | None = None,
    cell_section_scopes: dict[str, CellSectionScope] | None = None,
    cell_geometry_runs: dict[str, Path | str] | None = None,
    protein_runs: dict[str, Path | str] | None = None,
    protein_models: dict[
        str,
        dict[str, Path | str | dict[str, Path | str]],
    ]
    | None = None,
    annotation_runs: dict[str, Path | str] | None = None,
    registered_wsi: dict[str, Path | str] | None = None,
    cohort_qc: Path | str | None = None,
    workers: int = 1,
    include_methods: bool = False,
    study_figure_dir: Path | str | None = None,
) -> Path:
    """Build one stable review hub for all prepared workflow stages."""

    semantic_runs = semantic_runs or {}
    stain_runs = stain_runs or {}
    topology_runs = topology_runs or {}
    cell_runs = cell_runs or {}
    cell_section_scopes = cell_section_scopes or {}
    if set(cell_section_scopes) - set(cell_runs):
        raise ValueError("cell scope has no matching cell run")
    cell_geometry_runs = cell_geometry_runs or {}
    protein_runs = protein_runs or {}
    protein_models = protein_models or {}
    annotation_runs = annotation_runs or {}
    registered_wsi = registered_wsi or {}
    overlap = set(protein_runs) & set(protein_models)
    if overlap:
        raise ValueError(
            "review cohorts cannot configure legacy and model-scoped protein runs: "
            + ", ".join(sorted(overlap))
        )
    unknown = (
        set(semantic_runs)
        | set(stain_runs)
        | set(topology_runs)
        | set(cell_runs)
        | set(cell_geometry_runs)
        | set(protein_runs)
        | set(protein_models)
        | set(annotation_runs)
        | set(registered_wsi)
    ) - set(registration_runs)
    if unknown:
        raise ValueError(
            "review inputs have no matching registration: " + ", ".join(sorted(unknown))
        )
    missing_annotation_semantic = set(annotation_runs) - set(semantic_runs)
    if missing_annotation_semantic:
        raise ValueError(
            "annotation review requires semantic runs for: "
            + ", ".join(sorted(missing_annotation_semantic))
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
            registered_wsi={
                name: registered_wsi[name]
                for name in completed
                if name in registered_wsi
            },
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
        topology_index: Path | None = None
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
        cell_mice = sorted(set(completed) & set(cell_runs))
        if cell_mice:
            from histopia.visualization._cell_review import build_cell_review

            selected_scopes = {
                name: cell_section_scopes[name]
                for name in cell_mice
                if name in cell_section_scopes
            }
            cell_options = (
                {"cell_section_scopes": selected_scopes} if selected_scopes else {}
            )
            cell_index = build_cell_review(
                {name: cell_runs[name] for name in cell_mice},
                output_dir / "cells",
                **cell_options,
            )
            tabs.append(
                {
                    "id": "cells",
                    "label": "Cells",
                    "href": cell_index.relative_to(output_dir).as_posix(),
                }
            )
        protein_inputs: dict[str, Path | str | dict[str, Path | str]] = {
            **{name: protein_runs[name] for name in completed if name in protein_runs},
            **{
                name: protein_models[name]
                for name in completed
                if name in protein_models
            },
        }
        protein_mice = sorted(protein_inputs)
        if protein_mice:
            from histopia.visualization._protein_review import build_protein_review

            protein_index = build_protein_review(
                {name: protein_inputs[name] for name in protein_mice},
                output_dir / "protein",
            )
            tabs.append(
                {
                    "id": "protein",
                    "label": "Protein prediction",
                    "href": protein_index.relative_to(output_dir).as_posix(),
                }
            )
        protein_atlas_mice = sorted(
            set(completed)
            & set(cell_runs)
            & set(cell_geometry_runs)
            & set(protein_models)
        )
        if protein_atlas_mice:
            from histopia.visualization._cellular_protein_atlas import (
                build_cellular_protein_atlas,
            )

            protein_atlas_index = build_cellular_protein_atlas(
                {name: completed[name] for name in protein_atlas_mice},
                {name: cell_runs[name] for name in protein_atlas_mice},
                {name: protein_models[name] for name in protein_atlas_mice},
                {name: cell_geometry_runs[name] for name in protein_atlas_mice},
                output_dir / "protein-atlas",
                topology_runs={
                    name: topology_runs[name]
                    for name in protein_atlas_mice
                    if name in topology_runs
                },
                workers=workers,
            )
            topology_manifest = (
                topology_index.parent / "manifest.json"
                if topology_index is not None
                else None
            )
            cellular_manifest = protein_atlas_index.parent / "manifest.json"
            if (
                topology_manifest is not None
                and topology_manifest.is_file()
                and cellular_manifest.is_file()
            ):
                from histopia.visualization._topology_review import (
                    bind_cellular_depth_models,
                )

                bind_cellular_depth_models(
                    topology_index.parent,
                    cellular_manifest,
                )
            tabs.append(
                {
                    "id": "protein-atlas",
                    "label": "Cellular protein atlas",
                    "href": protein_atlas_index.relative_to(output_dir).as_posix(),
                }
            )
        annotation_mice = sorted(
            set(completed) & set(semantic_runs) & set(annotation_runs)
        )
        if annotation_mice:
            from histopia.visualization._annotation_review import (
                build_annotation_review,
            )

            annotation_index = build_annotation_review(
                {name: annotation_runs[name] for name in annotation_mice},
                {name: semantic_runs[name] for name in annotation_mice},
                output_dir / "annotations",
            )
            tabs.append(
                {
                    "id": "annotations",
                    "label": "Annotations",
                    "href": annotation_index.relative_to(output_dir).as_posix(),
                }
            )
    from histopia.visualization._methods_review import build_methods_review

    methods_index = build_methods_review(
        output_dir / "methods",
        cohorts=registration_runs,
        stages=(tab["id"] for tab in tabs),
    )
    if include_methods:
        tabs.append(
            {
                "id": "methods",
                "label": "Methods",
                "href": methods_index.relative_to(output_dir).as_posix(),
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
    (output_dir / "workflow-review.css").write_text(themed_review_css(_WORKFLOW_CSS))
    (output_dir / "workflow-review.js").write_text(_WORKFLOW_JS)
    if study_figure_dir is not None:
        from histopia.visualization._study_portal import attach_study_figure

        attach_study_figure(study_figure_dir, output_dir)
    from histopia.visualization._analysis_review import build_analysis_review

    return build_analysis_review(output_dir, output_dir)


def _write_decisions_page(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "index.html").write_text(_DECISIONS_HTML)
    (output_dir / "review-decisions.css").write_text(themed_review_css(_DECISIONS_CSS))
    (output_dir / "review-decisions.js").write_text(_DECISIONS_JS)
    return output_dir / "index.html"


_PORTAL_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <link rel="icon" href="data:,">
  <title>Registration stages · Histopia</title>
  <link rel="stylesheet" href="registration-review.css">
</head>
<body>
  <header class="stage-bar">
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
    <iframe id="review" title="Selected registration stage"></iframe>
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
body{display:grid;grid-template-rows:46px minmax(0,1fr)}
.stage-bar{display:flex;align-items:center;gap:16px;padding:5px 12px;
background:#f8fafc;border-bottom:1px solid #dbe3ec;min-width:0}
nav{display:flex;align-items:center;gap:5px;min-width:0}
.stage-bar nav button{position:relative;min-height:34px;padding:0 13px;
border:1px solid transparent;
border-radius:9px;background:transparent;color:#526174;font:700 13px inherit;
cursor:pointer;white-space:nowrap}
.stage-bar nav button:hover{color:#172033;background:#fff;border-color:#d5dee8}
.stage-bar nav button[aria-pressed="true"]{color:#173b67;background:#fff;
border-color:#b9cff0;
box-shadow:0 3px 10px rgba(15,23,42,.08)}
.stage-bar nav button[aria-pressed="true"]::after{content:"";position:absolute;
left:13px;
right:13px;bottom:-6px;height:3px;border-radius:4px 4px 0 0;background:#2563eb}
#status{margin-left:auto;padding:4px 9px;border:1px solid #b9dfce;border-radius:999px;
color:#17613e;background:#edf8f2;font-size:11px;font-weight:800;white-space:nowrap}
#status[data-state="review"]{border-color:#ead09c;color:#89550a;background:#fff8e8}
main,iframe{width:100%;height:100%;min-width:0;min-height:0;border:0}
iframe{display:block}
@media(max-width:700px){
  body{grid-template-rows:44px minmax(0,1fr)}
  .stage-bar{gap:6px;padding:4px 7px}
  nav{gap:2px;overflow-x:auto;scrollbar-width:none}
  nav::-webkit-scrollbar{display:none}
  .stage-bar nav button{min-height:32px;padding:0 9px;font-size:11px}
  .stage-bar nav button[aria-pressed="true"]::after{left:9px;right:9px;bottom:-6px}
  #status{margin-left:auto;flex:0 0 auto;padding:3px 7px;font-size:9px}
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
  const approval=row.approved?"Approved":"Review required";
  const continuity=row.review_recommended?" · continuity check":"";
  status.textContent=`${approval}${continuity}`;
  status.dataset.state=row.approved?"approved":"review";
  status.title=`${row.slide_count} sections · ${approval.toLowerCase()}${continuity}`;
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
  <link rel="icon" href="data:,">
  <title>Serial-section registration · Histopia</title>
  <link rel="stylesheet" href="cohort-review.css">
</head>
<body>
  <header class="registration-context">
    <div class="registration-heading">
      <span>Registration</span>
      <strong>Serial-section alignment</strong>
    </div>
    <label class="cohort-picker" for="cohort">
      <span>Mouse</span>
      <select id="cohort"></select>
    </label>
    <span id="status" role="status"></span>
  </header>
  <main>
    <iframe id="review" title="Selected mouse registration"></iframe>
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
body{display:grid;grid-template-rows:54px minmax(0,1fr)}
.registration-context{display:flex;align-items:center;gap:16px;padding:6px 14px;
background:#fff;border-bottom:1px solid #dbe3ec;min-width:0}
.registration-heading{display:flex;flex-direction:column;min-width:190px}
.registration-heading>span{color:#2563eb;font-size:9px;font-weight:900;
letter-spacing:.1em;line-height:1;text-transform:uppercase}
.registration-heading strong{margin-top:4px;color:#182235;font-size:15px;
line-height:1;white-space:nowrap}
.cohort-picker{display:flex;align-items:center;gap:7px;color:#5b6777;
font-size:11px;font-weight:800}
select{min-width:104px;height:34px;padding:5px 30px 5px 9px;border:1px solid #c8d3df;
border-radius:8px;background:#fff;color:#172033;font-weight:700}
#status{min-width:0;margin-left:auto;overflow:hidden;text-overflow:ellipsis;
white-space:nowrap;text-align:right;color:#526174;font-size:11px;font-weight:700}
#status[data-state="complete"]{color:#17613e}
#status[data-state="attention"]{color:#89550a}
main,iframe{width:100%;height:100%;min-width:0;min-height:0;border:0}
iframe{display:block}
@media(max-width:600px){
  body{grid-template-rows:66px minmax(0,1fr)}
  .registration-context{display:grid;grid-template-columns:minmax(0,1fr) 108px;
    grid-template-rows:31px 21px;padding:5px 8px;gap:3px 8px}
  .registration-heading{grid-column:1;grid-row:1;min-width:0}
  .registration-heading>span{font-size:8px}
  .registration-heading strong{font-size:12px;overflow:hidden;text-overflow:ellipsis}
  .cohort-picker{grid-column:2;grid-row:1}
  .cohort-picker>span{display:none}
  select{width:100%;min-width:0;height:31px}
  #status{grid-column:1/3;grid-row:2;margin-left:0;font-size:9px;text-align:left}
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
  const approved=row.stages.filter(stage=>row.stage_summary?.[stage]?.approved).length;
  const total=row.stages.length;
  const count=Number.isInteger(row.slide_count)?`${row.slide_count} sections · `:"";
  const continuity=row.stages.some(
    stage=>row.stage_summary?.[stage]?.review_recommended);
  status.textContent=count+(approved===total
    ? "Registration complete"
    : `${approved} of ${total} outputs approved${continuity?" · continuity check":""}`);
  status.dataset.state=approved===total?"complete":"attention";
  status.title=parts.join(" · ");
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
  <link rel="icon" href="data:,">
  <title>Histopia scientific review</title>
  <link rel="stylesheet" href="workflow-review.css">
</head>
<body>
  <header class="workflow-nav">
    <div class="workflow-brand">
      <span class="brand-mark" aria-hidden="true">H</span>
      <span class="brand-copy">
        <strong>Histopia</strong>
        <span>Spatial histology</span>
      </span>
    </div>
    <nav aria-label="Workflow stage"></nav>
    <p class="workflow-note">
      <span class="workflow-note-mark" aria-hidden="true"></span>
      <span>Interactive workspace<small>Native-resolution imagery</small></span>
    </p>
  </header>
  <main><iframe id="review" title="Histopia scientific review"></iframe></main>
  <script src="manifest-data.js"></script>
  <script src="workflow-review.js"></script>
</body>
</html>
"""

_WORKFLOW_CSS = """
:root{font-family:Arial,system-ui,sans-serif;color:#111827;background:#f3f6fa}
*{box-sizing:border-box}
html,body{width:100%;height:100%;margin:0;overflow:hidden}
body{display:grid;grid-template-columns:252px minmax(0,1fr);min-width:0}
.workflow-nav{display:flex;flex-direction:column;align-items:stretch;gap:0;
padding:18px 16px 14px;background:#0f172a;color:#e5edf6;
border:0;border-right:1px solid #1e293b;min-width:0;overflow:auto}
.workflow-brand{display:flex;align-items:center;gap:11px;min-width:0}
.brand-mark{display:grid;place-items:center;width:38px;height:38px;flex:0 0 38px;
border:1px solid #5eead4;border-radius:11px;color:#071a2d;
background:linear-gradient(145deg,#99f6e4,#60a5fa);font-size:18px;font-weight:900;
box-shadow:0 8px 22px rgba(37,99,235,.28)}
.brand-copy{display:flex;flex-direction:column;min-width:0}
.workflow-nav .brand-copy strong{color:#fff;font-size:20px;line-height:1.05;
white-space:nowrap}
.brand-copy>span{margin-top:4px;color:#9fb0c6;font-size:11px;line-height:1.1;
white-space:nowrap}
.workflow-nav nav{display:flex;flex-direction:column;align-items:stretch;
gap:5px;margin-top:24px;min-width:0}
.workflow-nav .nav-group{margin:14px 7px 5px;color:#8295ad;font-size:9px;
font-weight:800;letter-spacing:.08em;text-transform:uppercase}
.workflow-nav .nav-group:first-child{margin-top:0}
.workflow-nav button{position:relative;display:grid;
grid-template-columns:30px minmax(0,1fr);align-items:center;gap:9px;width:100%;
min-height:48px;padding:7px 12px 7px 8px;border:1px solid transparent;
border-radius:11px;background:transparent;color:#dce6f2;font-family:inherit;
font-size:14px;font-weight:700;line-height:1.15;cursor:pointer;text-align:left;
box-shadow:none;transition:background .16s ease,border-color .16s ease,
transform .16s ease,box-shadow .16s ease}
.workflow-nav button:hover{color:#fff;background:#1a2840;border-color:#31435d;
transform:translateX(2px)}
.nav-icon{display:grid;place-items:center;width:30px;height:30px;border-radius:9px;
color:#9fb2ca;background:#19263b;border:1px solid #2c3d56}
.nav-icon svg{width:17px;height:17px;fill:none;stroke:currentColor;
stroke-width:1.8;stroke-linecap:round;stroke-linejoin:round}
.nav-copy{display:flex;flex-direction:column;min-width:0}
.nav-copy b{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
font-size:13px;line-height:1.15}
.nav-copy small{margin-top:3px;overflow:hidden;text-overflow:ellipsis;
color:#8fa2ba;font-size:9px;font-weight:600;line-height:1.15;white-space:nowrap}
.workflow-nav nav button[aria-current="page"]{color:#0f172a;
background:linear-gradient(135deg,#fff 0%,#edf6ff 100%);border-color:#bfdbfe;
box-shadow:0 10px 24px rgba(2,8,23,.34),inset 0 0 0 1px rgba(255,255,255,.8);
transform:translateX(5px)}
.workflow-nav nav button[aria-current="page"]::after{content:"";position:absolute;
right:11px;top:10px;width:7px;height:7px;border:2px solid #fff;border-radius:50%;
background:#2563eb;box-shadow:0 0 0 2px #bfdbfe}
.workflow-nav nav button[aria-current="page"] .nav-icon{color:#fff;
border-color:#2563eb;background:linear-gradient(145deg,#2563eb,#0f766e);
box-shadow:0 5px 12px rgba(37,99,235,.25)}
.workflow-nav nav button[aria-current="page"] .nav-copy small{color:#52657b}
.workflow-note{display:flex;align-items:center;gap:8px;margin:auto 6px 0;
padding-top:18px;color:#a8b7ca;font-size:10px;line-height:1.25}
.workflow-note-mark{width:8px;height:8px;flex:0 0 8px;border-radius:50%;
background:#5eead4;box-shadow:0 0 0 4px rgba(94,234,212,.10)}
.workflow-note span:last-child{display:flex;flex-direction:column}
.workflow-note small{margin-top:2px;color:#7488a2;font-size:9px}
main,iframe{width:100%;height:100%;min-width:0;min-height:0;border:0}
iframe{display:block}
@media(max-width:1100px){
  body{grid-template-columns:1fr;grid-template-rows:68px minmax(0,1fr)}
  .workflow-nav{display:grid;grid-template-columns:154px minmax(0,1fr);
    grid-template-rows:1fr;gap:0 12px;padding:8px 10px;
    border-right:0;border-bottom:1px solid #1e293b;overflow:hidden}
  .workflow-brand{grid-column:1;grid-row:1;align-self:center}
  .brand-mark{width:34px;height:34px;flex-basis:34px;border-radius:9px}
  .workflow-nav .brand-copy strong{font-size:16px}
  .workflow-nav nav{grid-column:2;grid-row:1;flex-direction:row;align-self:stretch;
    gap:4px;margin:0;overflow-x:auto;scrollbar-width:thin}
  .workflow-nav .nav-group,.workflow-note{display:none}
  .workflow-nav button{display:flex;width:auto;min-height:46px;padding:6px 11px;
    align-self:center;gap:7px;transform:none}
  .nav-icon{width:27px;height:27px;flex:0 0 27px}
  .nav-copy small{display:none}
  .workflow-nav nav button[aria-current="page"]{transform:none;
    box-shadow:0 6px 14px rgba(2,8,23,.28)}
}
@media(max-width:560px){
  body{grid-template-rows:98px minmax(0,1fr)}
  .workflow-nav{display:grid;grid-template-columns:1fr;grid-template-rows:30px 54px;
    gap:4px;padding:6px 8px}
  .workflow-brand{grid-column:1;grid-row:1}
  .brand-mark{width:25px;height:25px;flex-basis:25px;border-radius:7px;font-size:13px}
  .workflow-nav .brand-copy strong{font-size:14px}
  .brand-copy>span{display:none}
  .workflow-nav nav{grid-column:1;grid-row:2}
  .workflow-nav button{min-height:44px;padding:6px 10px;font-size:12px}
  .nav-icon{display:none}.nav-copy b{font-size:12px}
}
"""

_WORKFLOW_JS = """
const manifest=globalThis.HISTOPIA_WORKFLOW_REVIEW;
if(!manifest||!manifest.tabs.length)throw new Error("Missing workflow review tabs");
const nav=document.querySelector("nav");
const frame=document.querySelector("#review");
const tabDetails={
  "data-catalog":{
    description:"Organs · external sources",
    icon:'<path d="M3 4h7v7H3zM14 4h7v7h-7zM3 15h7v6H3zM14 15h7v6h-7z"/>',
  },
  "external-validation":{
    description:"Public IHC · prediction",
    icon:'<path d="M3 4h18v16H3zM3 12h18M12 4v16"/>',
  },
  "internal-inputs":{
    description:"Liver · lung · kidney",
    icon:'<path d="M3 4h18v16H3zM3 12h18M12 4v16"/>',
  },
  "organ-metadata":{
    description:"All internal scans · coverage",
    icon:'<path d="M3 4h18v16H3zM3 9h18M3 14h18M9 4v16"/>',
  },
  registration:{
    description:"Masks · order · alignment",
    icon:'<path d="M4 5h16v4H4zM6 11h12v4H6zM8 17h8v3H8z"/>',
  },
  stain:{
    description:"Adaptive target OD",
    icon:'<path d="M12 3s5 6 5 10a5 5 0 0 1-10 0c0-4 5-10 5-10Z"/>'+
      '<path d="M9.5 14.5c.7 1 1.6 1.5 2.8 1.5"/>',
  },
  cells:{
    description:"Boundaries · slide QC",
    icon:'<circle cx="8" cy="9" r="3"/>'+
      '<circle cx="15.5" cy="8" r="2.5"/>'+
      '<circle cx="13" cy="15" r="3.5"/>',
  },
  "selected-cells":{
    description:"Approved sections",
    icon:'<circle cx="8" cy="9" r="3"/>'+
      '<circle cx="15.5" cy="8" r="2.5"/>'+
      '<path d="m10 17 3 3 6-7"/>',
  },
  annotations:{
    description:"Regions · spatial labels",
    icon:'<path d="M7 4H4v5M17 4h3v5M7 20H4v-5M17 20h3v-5' +
      'M8 15l7-7 2 2-7 7-3 1z"/>',
  },
  "study-figure":{
    description:"Workflow · tissue regions",
    icon:'<path d="M3 4h18v16H3zM9 4v16M15 4v16M3 12h18"/>',
  },
  atlas:{
    description:"Registered tissue stack",
    icon:'<path d="m4 8 8-4 8 4-8 4-8-4zM4 12l8 4 8-4M4 16l8 4 8-4"/>',
  },
  topology:{
    description:"Continuity · physical depth",
    icon:'<path d="M6 6l6 5 6-4M12 11l-4 7M12 11l5 7"/>'+
      '<circle cx="6" cy="6" r="2"/><circle cx="18" cy="7" r="2"/>'+
      '<circle cx="8" cy="18" r="2"/><circle cx="17" cy="18" r="2"/>',
  },
  protein:{
    description:"Observed · inferred signal",
    icon:'<path d="M4 15c3-8 5 3 8-5s5 4 8-2"/>'+
      '<circle cx="4" cy="15" r="1.5"/><circle cx="20" cy="8" r="1.5"/>',
  },
  "protein-atlas":{
    description:"Single-cell 2D · 3D",
    icon:'<circle cx="7" cy="7" r="2.5"/>'+
      '<circle cx="16.5" cy="6.5" r="2"/>'+
      '<circle cx="11.5" cy="14" r="3"/><circle cx="18" cy="17" r="2"/>',
  },
  methods:{
    description:"Methods · measurement scope",
    icon:'<path d="M7 4h10M9 4v5l-4 8a2 2 0 0 0 2 3h10' +
      'a2 2 0 0 0 2-3l-4-8V4M8 15h8"/>',
  },
  decisions:{
    description:"Approvals · notes",
    icon:'<path d="M5 4h14v16H5zM8 9l2 2 5-5M8 15h8"/>',
  },
};
const groups=[
  ["Data sources",["data-catalog","external-validation"]],
  ["Review inputs",["internal-inputs","organ-metadata","registration","stain","cells",
    "selected-cells","annotations"]],
  ["Spatial analysis",["serial-stack","atlas","topology","protein","protein-atlas"]],
  ["Reference",["methods","study-figure"]],
  ["Governance",["decisions"]],
];
const rendered=new Set();
function appendTab(tab){
  const button=document.createElement("button");
  button.type="button";
  button.dataset.tab=tab.id;
  const detail=tabDetails[tab.id]||{
    description:"Open workspace",
    icon:'<path d="M5 5h14v14H5z"/>',
  };
  const icon=document.createElement("span");
  icon.className="nav-icon";
  icon.setAttribute("aria-hidden","true");
  icon.innerHTML=`<svg viewBox="0 0 24 24">${detail.icon}</svg>`;
  const copy=document.createElement("span");
  copy.className="nav-copy";
  const label=document.createElement("b");
  label.textContent=tab.label;
  const description=document.createElement("small");
  description.textContent=detail.description;
  copy.append(label,description);
  button.append(icon,copy);
  button.addEventListener("click",()=>select(tab.id,true));
  nav.append(button);
  rendered.add(tab.id);
}
for(const [label,ids] of groups){
  const tabs=ids.map(id=>manifest.tabs.find(item=>item.id===id)).filter(Boolean);
  if(!tabs.length)continue;
  const heading=document.createElement("div");
  heading.className="nav-group";
  heading.textContent=label;
  nav.append(heading);
  tabs.forEach(appendTab);
}
manifest.tabs.filter(tab=>!rendered.has(tab.id)).forEach(appendTab);
function revealActiveTab(){
  nav.querySelector('button[aria-pressed="true"]')?.scrollIntoView(
    {block:"nearest",inline:"nearest"});
}
const contextKeys=["region","evidence","source","organ","subject","reconstruction",
  "stack_section","stack_mode","section","stage","collection","dataset","field",
  "inventory_q","inventory_status","inventory_table"];
const identityKeys=["source","organ","subject"];
function catalogContexts(tab){return tab.catalog_contexts||tab.image_contexts;}
function rememberPublishedCatalog(tab,catalog){
  const keys=["source_id","organ","subject_id","reconstruction_id"];
  const contexts=Object.fromEntries(catalog.datasets.map(row=>[row.id,
    Object.fromEntries(keys.filter(key=>row[key]!=null).map(
      key=>[key,String(row[key])]))]));
  if(tab.id==="external-validation")tab.image_contexts=contexts;
  else tab.catalog_contexts=contexts;
  tab.catalog_sources=catalog.sources.map(source=>source.id);
  if(!tab.catalog_sources.includes(tab.default_source))
    tab.default_source=tab.catalog_sources[0]||"";
  tab.export_fingerprint=catalog.fingerprint;
}
function matchesIdentity(record,url){
  return [["source","source_id"],["organ","organ"],["subject","subject_id"]].every(
    ([key,field])=>!url.searchParams.get(key)||
      String(record[field])===url.searchParams.get(key));
}
function supportsContext(tab,url){
  const records=catalogContexts(tab);
  if(records)return Object.values(records).some(row=>matchesIdentity(row,url));
  const scope=tab.scope;
  if(!scope)return true;
  const source=url.searchParams.get("source"),organ=url.searchParams.get("organ");
  const subject=url.searchParams.get("subject");
  return (!source||!scope.sources||scope.sources.includes(source))&&
    (!organ||!scope.organs||scope.organs.includes(organ))&&
    (!subject||!scope.subjects||scope.subjects.includes(subject));
}
function externalImageView(tab,url){
  if(supportsContext(tab,url)||tab.id==="decisions"||
      ["reconstruction","stack_section","stack_mode"].some(
        key=>url.searchParams.has(key)))return;
  const external=manifest.tabs.find(item=>item.id==="external-validation");
  const record=external?.image_contexts?.[url.searchParams.get("dataset")];
  if(!record||record.source_id!==url.searchParams.get("source")||
      record.organ!==url.searchParams.get("organ")||
      record.subject_id!==url.searchParams.get("subject"))return;
  return external;
}
function updateContextTabs(){
  const url=new URL(location.href);
  for(const tab of manifest.tabs){
    const button=nav.querySelector(`[data-tab="${tab.id}"]`);
    if(!button)continue;
    const available=supportsContext(tab,url);
    button.dataset.available=String(available);
    button.title=available?tab.label:
      `Open ${tab.label} with available data`;
    // Tools remain discoverable even when this specimen has no result.
    button.style.display="";
  }
}
const switchNotice=document.createElement("div");
switchNotice.id="selection-notice";switchNotice.setAttribute("role","status");
switchNotice.style.cssText="position:absolute;bottom:14px;left:14px;z-index:10;"+
  "max-width:calc(100% - 28px);padding:9px 12px;border:1px solid #ccdce5;"+
  "border-radius:8px;background:#eff8ff;color:#24364b;font:13px system-ui;"+
  "box-shadow:0 2px 10px #0001";
switchNotice.hidden=true;frame.parentElement.style.position="relative";
frame.parentElement.append(switchNotice);
let noticeTimer;
function showSelectionNotice(message){
  clearTimeout(noticeTimer);switchNotice.textContent=message;switchNotice.hidden=false;
  noticeTimer=setTimeout(()=>{switchNotice.hidden=true;},8000);
}
const rememberedContexts=new Map();
function readContext(key){
  try{
    const value=JSON.parse(sessionStorage.getItem("histopia-context-v2:"+key));
    if(value&&typeof value.context==="object")return value;
  }catch{}
  return rememberedContexts.get(key);
}
function contextValues(url,keys=contextKeys){
  return Object.fromEntries(keys.filter(key=>url.searchParams.has(key)).map(
    key=>[key,url.searchParams.get(key)]));
}
function contextUrl(values){
  const url=new URL(location.href);
  for(const key of [...contextKeys,"mouse","cohort"])url.searchParams.delete(key);
  for(const key of contextKeys)if(typeof values?.[key]==="string")
    url.searchParams.set(key,values[key]);
  return url;
}
function validRememberedContext(tab,url){
  if(!supportsContext(tab,url))return false;
  const records=catalogContexts(tab);
  if(!records)return true;
  const dataset=url.searchParams.get("dataset");
  if(dataset&&(!records[dataset]||!matchesIdentity(records[dataset],url)))return false;
  const reconstruction=url.searchParams.get("reconstruction");
  return !reconstruction||Object.values(records).some(row=>
    row.reconstruction_id===reconstruction&&matchesIdentity(row,url));
}
function rememberContext(tab,url){
  if(!tab||!validRememberedContext(tab,url))return;
  const value={context:contextValues(url),updated:Date.now()};
  const keys=["view:"+tab.id];
  if(value.context.source)keys.push("source:"+value.context.source);
  for(const key of keys){
    rememberedContexts.set(key,value);
    try{sessionStorage.setItem("histopia-context-v2:"+key,JSON.stringify(value));}catch{}
  }
}
function navigationContext(tab,url){
  if(tab.id==="organ-metadata"){
    // The inventory opens the full collection, independently of the last image.
    const saved=readContext("view:"+tab.id)?.context;
    for(const key of [...contextKeys,"mouse","cohort"])url.searchParams.delete(key);
    for(const key of ["source","organ","inventory_q","inventory_status",
      "inventory_table"])
      if(saved?.[key])url.searchParams.set(key,saved[key]);
    return false;
  }
  const identity=contextValues(url,identityKeys);
  const domain=contextUrl(identity);domain.searchParams.delete("subject");
  const saved=readContext("view:"+tab.id)?.context;
  let next=identity;
  if(!supportsContext(tab,domain)){
    const sources=tab.catalog_sources||tab.scope?.sources||[];
    const remembered=sources.map(source=>readContext("source:"+source)).filter(Boolean)
      .sort((a,b)=>b.updated-a.updated).map(value=>value.context);
    next=[...remembered,saved].find(value=>value&&
      supportsContext(tab,contextUrl(value)));
    if(!next){
      const records=Object.values(catalogContexts(tab)||{});
      const record=records.find(row=>row.organ===identity.organ)||records[0];
      next=record?{source:record.source_id,organ:record.organ,subject:record.subject_id}:
        {source:tab.default_source||tab.scope?.sources?.[0],
          organ:tab.scope?.organs?.includes(identity.organ)?identity.organ:
            tab.scope?.organs?.[0]};
    }
  }
  const nextIdentity=contextValues(contextUrl(next),identityKeys);
  const sameSaved=saved&&identityKeys.every(key=>saved[key]===nextIdentity[key]);
  const restored=contextUrl(sameSaved&&validRememberedContext(tab,contextUrl(saved))
    ?saved:nextIdentity);
  for(const key of [...contextKeys,"mouse","cohort"])url.searchParams.delete(key);
  for(const [key,value] of Object.entries(contextValues(restored)))
    url.searchParams.set(key,value);
  return identity.source!==url.searchParams.get("source")||
    identity.organ!==url.searchParams.get("organ");
}
function selectionKey(tab,url){
  return "histopia-view-selection:"+[tab.id,url.searchParams.get("source")||"",
    url.searchParams.get("organ")||""].join(":");
}
function rememberSelection(tab,url){
  rememberContext(tab,url);
  if(!tab.scope?.subjects||!supportsContext(tab,url))return;
  const subject=url.searchParams.get("subject");
  if(subject)try{sessionStorage.setItem(selectionKey(tab,url),subject);}catch{}
}
function resolveAvailableSelection(tab,url){
  const context=new URL(url);context.searchParams.delete("subject");
  if(!supportsContext(tab,context)||!tab.scope?.subjects?.length)return;
  if(!url.searchParams.has("source")&&tab.scope.sources?.length===1)
    url.searchParams.set("source",tab.scope.sources[0]);
  if(!url.searchParams.has("organ")&&tab.scope.organs?.length===1)
    url.searchParams.set("organ",tab.scope.organs[0]);
  const previous=url.searchParams.get("subject");
  if(previous&&tab.scope.subjects.includes(previous)){rememberSelection(tab,url);return;}
  let remembered;try{remembered=sessionStorage.getItem(selectionKey(tab,url));}catch{}
  const selected=[remembered,tab.default_subject,...tab.scope.subjects].find(
    subject=>tab.scope.subjects.includes(subject));
  for(const key of ["region","evidence","reconstruction","dataset","field","section",
    "stage","stack_section","stack_mode","mouse","cohort"])url.searchParams.delete(key);
  url.searchParams.set("subject",selected);rememberSelection(tab,url);
  if(previous){
    showSelectionNotice(`Showing specimen ${selected}; `+
      `this view has no result for ${previous}.`);
  }
}
function showAvailableResults(tab,url){
  const doc=document.implementation.createHTMLDocument(tab.label);
  doc.documentElement.lang="en";
  const make=(tag,text)=>{
    const node=doc.createElement(tag);if(text)node.textContent=text;return node;
  };
  const viewport=make("meta");viewport.name="viewport";
  viewport.content="width=device-width,initial-scale=1";doc.head.append(viewport);
  const style=make("style",`
    *{box-sizing:border-box}body{margin:0;background:#f3f6fa;color:#24364b;
    font:14px/1.5 system-ui,sans-serif}main{max-width:560px;margin:32px auto;
    padding:24px;background:white;border:1px solid #dbe3ec;border-radius:10px}
    h1{font-size:21px;margin:0 0 8px}p{color:#64748b;margin:0 0 22px}
    label{display:block;font-size:12px;margin-bottom:6px}
    .controls{display:flex;gap:8px}select,button{font:inherit;padding:9px 12px;
    border:1px solid #cbd5e1;border-radius:6px}select{flex:1;min-width:0}
    button{background:#215e73;color:white;cursor:pointer}
    button:disabled{opacity:.45;cursor:default}a{color:#216b91}
    @media(max-width:600px){main{margin:16px 12px;padding:20px}}
  `);doc.head.append(style);
  const main=make("main");main.append(make("h1",tab.label));
  main.append(make("p","No result in this view for this source and organ."));
  const external=url.searchParams.get("source")!=="kpf"&&
    manifest.tabs.some(t=>t.id==="external-validation");
  const back=make("a",external?"External validation":"Tissue review");
  const target=new URL(url);
  target.searchParams.set("view",external?"external-validation":"data-catalog");
  back.href=target.href;back.target="_top";main.append(back);
  doc.body.append(main);frame.removeAttribute("src");
  frame.srcdoc="<!doctype html>"+doc.documentElement.outerHTML;
}
function select(id,navigation=false){
  clearTimeout(noticeTimer);switchNotice.hidden=true;
  if(id&&!manifest.tabs.some(item=>item.id===id)){
    frame.removeAttribute("src");
    frame.srcdoc='<main style="font:15px system-ui;padding:28px;color:#334155">'+
      'This view is unavailable. Choose an eligible stack in Tissue review.</main>';
    updateContextTabs();return;
  }
  const requested=manifest.tabs.find(item=>item.id===id)||manifest.tabs[0];
  const url=new URL(location.href);
  let changedDomain=false;
  if(navigation){
    rememberContext(manifest.tabs.find(tab=>tab.id===url.searchParams.get("view")),url);
    changedDomain=navigationContext(requested,url);
  }
  const external=!navigation&&externalImageView(requested,url);
  const tab=external||requested;
  if(external){
    showSelectionNotice(["registration","atlas","topology"].includes(requested.id)
      ? "No verified serial stack for this image. Showing its 2D IHC result."
      : `${requested.label} is unavailable for this image. Showing its 2D IHC result.`);
  }
  for(const button of nav.querySelectorAll("button")){
    const active=button.dataset.tab===tab.id;
    button.setAttribute("aria-pressed",String(active));
    if(active)button.setAttribute("aria-current","page");
    else button.removeAttribute("aria-current");
  }
  revealActiveTab();
  resolveAvailableSelection(tab,url);
  if(changedDomain&&url.searchParams.get("source")){
    const source=url.searchParams.get("source");
    showSelectionNotice("Showing "+[source==="kpf"?"our data":source.toUpperCase(),
      url.searchParams.get("organ"),url.searchParams.get("subject")].filter(Boolean)
      .join(" · ")+".");
  }
  const target=new URL(tab.href.startsWith("/") ?
    histopiaUrl(tab.href) : tab.href,location.href);
  if(tab.scope&&!supportsContext(tab,url)){
    // Bookmarks remain bound to their requested source and organ.
    showAvailableResults(tab,url);
  }else{
    frame.removeAttribute("srcdoc");
    for(const key of contextKeys){
      if(url.searchParams.has(key))target.searchParams.set(key,url.searchParams.get(key));
    }
    if(tab.scope?.sources?.includes("kpf")&&url.searchParams.get("subject")){
      target.searchParams.set("mouse",url.searchParams.get("subject"));
      target.searchParams.set("cohort",url.searchParams.get("subject"));
    }
    if(!frame.hasAttribute("src"))frame.src=target.href;
    else frame.contentWindow.location.replace(target.href);
  }
  url.searchParams.set("view",tab.id);
  if(navigation&&url.href!==location.href)history.pushState(null,"",url);
  else history.replaceState(null,"",url);
  rememberSelection(tab,url);
  updateContextTabs();
}
let revealRequest=0;
addEventListener("resize",()=>{
  cancelAnimationFrame(revealRequest);
  revealRequest=requestAnimationFrame(revealActiveTab);
});
select(new URL(location.href).searchParams.get("view"));
addEventListener("popstate",()=>
  select(new URL(location.href).searchParams.get("view")));

function syncReviewSelection(subject){
  const url=new URL(location.href);
  const tab=manifest.tabs.find(item=>item.id===url.searchParams.get("view"));
  if(typeof subject!=="string"||!tab?.scope?.subjects?.includes(subject))return;
  if(url.searchParams.get("subject")!==subject){
    for(const key of ["reconstruction","dataset","field","section","stage",
      "stack_section","stack_mode"])url.searchParams.delete(key);
  }
  url.searchParams.set("subject",subject);
  for(const key of ["mouse","cohort"])url.searchParams.delete(key);
  if(tab.scope.sources?.length===1)url.searchParams.set("source",tab.scope.sources[0]);
  if(tab.scope.organs?.length===1)url.searchParams.set("organ",tab.scope.organs[0]);
  history.replaceState(null,"",url);updateContextTabs();
  rememberSelection(tab,url);
}
frame.addEventListener("load",()=>{
  // Existing viewers own their data loading; keep the surrounding navigation
  // in step with explicit changes made in their specimen controls.
  const doc=frame.contentDocument;
  for(const control of doc?.querySelectorAll("select#mouse,select#cohort")||[])
    control.addEventListener("change",()=>syncReviewSelection(control.value));
  const tab=manifest.tabs.find(item=>item.id===
    new URL(location.href).searchParams.get("view"));
  if(tab?.id==="study-figure"&&doc?.querySelector("#region")&&
      doc.querySelector("#outline"))
    doc.querySelector("header a")?.setAttribute("href",
      new URL(tab.href,location.href).href);
});

addEventListener("message",event=>{
  if(event.source!==frame.contentWindow||event.origin!==location.origin)return;
  if(event.data?.type==="histopia-organ-metadata"){
    const url=new URL(location.href);
    const tab=manifest.tabs.find(item=>item.id===url.searchParams.get("view"));
    const published=frame.contentWindow.HISTOPIA_ORGAN_INVENTORY;
    const child=new URL(frame.contentWindow.location.href);
    if(tab?.id!=="organ-metadata"||!published||
        event.data.fingerprint!==published.fingerprint||
        event.data.search!==child.search)return;
    tab.export_fingerprint=published.fingerprint;
    for(const key of [...contextKeys,"mouse","cohort"])url.searchParams.delete(key);
    for(const key of ["source","organ","inventory_q","inventory_status",
      "inventory_table"])
      if(child.searchParams.has(key))url.searchParams.set(key,child.searchParams.get(key));
    if(event.data.push&&url.href!==location.href)history.pushState(null,"",url);
    else history.replaceState(null,"",url);
    rememberSelection(tab,url);updateContextTabs();return;
  }
  if(event.data?.type==="histopia-review-selection"){
    syncReviewSelection(event.data.subject);return;
  }
  if(event.data?.type==="histopia-serial-selection"){
    const url=new URL(location.href);
    for(const key of ["source","organ","subject","section"])
      if(typeof event.data[key]==="string")url.searchParams.set(key,event.data[key]);
    history.replaceState(null,"",url);updateContextTabs();return;
  }
  if(event.data?.type!=="histopia-catalog-selection"||
      !["data-catalog","external-validation","internal-inputs"].includes(
        new URL(location.href).searchParams.get("view")))return;
  const tab=manifest.tabs.find(item=>item.id===
    new URL(location.href).searchParams.get("view"));
  if(tab?.export_fingerprint){
    const published=frame.contentWindow.HISTOPIA_RESULTS_CATALOG;
    if(!published||event.data.catalog_fingerprint!==published.fingerprint||
        (tab.id==="external-validation")!==Boolean(published.benchmark_policy)||
        (tab.id==="internal-inputs")!==Boolean(published.input_review_policy))return;
    const childUrl=new URL(frame.contentWindow.location.href);
    if(contextKeys.some(key=>(event.data[key]==null?null:String(event.data[key]))!==
        childUrl.searchParams.get(key)))return;
    // An open portal can receive a newer, gated publication after a job finishes.
    if(tab.export_fingerprint!==published.fingerprint)
      rememberPublishedCatalog(tab,published);
  }
  const url=new URL(location.href);
  for(const key of contextKeys){
    const value=event.data[key];
    if(value==null)url.searchParams.delete(key);
    else if(typeof value==="string"||typeof value==="number")
      url.searchParams.set(key,String(value));
  }
  const previous=new URL(location.href);
  if(identityKeys.some(key=>previous.searchParams.get(key)!==url.searchParams.get(key))){
    clearTimeout(noticeTimer);switchNotice.hidden=true;
  }
  history.replaceState(null,"",url);
  rememberSelection(tab,url);
  updateContextTabs();
});

// Keep same-origin data requests inside a code-server port proxy.
function histopiaUrl(path) {
  const match = location.pathname.match(new RegExp("^.*?/proxy/[0-9]+(?=/|$)"));
  return (match ? match[0] : "") + path;
}
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
    <div id="review-access">
      <label for="access-key">Access key</label>
      <input id="access-key" type="password" autocomplete="current-password">
      <button id="connect" type="button">Connect</button>
    </div>
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
#review-access{display:flex;align-items:center;gap:10px}
#review-access[hidden]{display:none}
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

_DECISIONS_JS = (
    REVIEW_SELECTION_JS
    + """
const keyInput=document.querySelector("#access-key");
const connect=document.querySelector("#connect");
const connection=document.querySelector("#connection");
const accessControls=document.querySelector("#review-access");
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
  topology:"Semantic topology",stain:"Stain",cells:"Cell segmentation"};
let registry=null;
let selectedStage=null;
let authenticationRequired=true;
keyInput.value=sessionStorage.getItem("histopiaReviewKey")||"";
reviewer.value=sessionStorage.getItem("histopiaReviewer")||"";
function headers(){
  const values={"Content-Type":"application/json"};
  if(authenticationRequired)values.Authorization=`Bearer ${keyInput.value}`;
  return values;
}
async function configureAccess(){
  const response=await fetch(histopiaUrl("/api/reviews/access"),{cache:"no-store"});
  const payload=await response.json();
  if(!response.ok)throw new Error(payload.error||"Review service unavailable");
  if(!payload.review_configured)throw new Error("Review service unavailable");
  authenticationRequired=Boolean(payload.authentication_required);
  accessControls.hidden=!authenticationRequired;
  if(!authenticationRequired){
    connection.textContent="Ready";
    connection.style.color="#117864";
    await load();
  }else if(keyInput.value){
    await load();
  }
}
async function load(){
  message.textContent="";
  const response=await fetch(histopiaUrl("/api/reviews"),{
    headers:headers(),cache:"no-store"});
  if(!response.ok)throw new Error((await response.json()).error||"Unable to connect");
  registry=await response.json();
  if(authenticationRequired){
    sessionStorage.setItem("histopiaReviewKey",keyInput.value);
  }
  connection.textContent="Connected";
  connection.style.color="#117864";
  cohortSelect.disabled=false;
  approve.disabled=true;
  stages.replaceChildren();
  const requested=histopiaReviewSelection.requested(registry.cohorts);
  cohortSelect.replaceChildren();
  for(const row of registry.cohorts){
    const option=document.createElement("option");
    option.value=row.id;
    option.textContent=row.id;
    cohortSelect.append(option);
  }
  cohortSelect.value=requested;
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
  histopiaReviewSelection.remember(row.id);
}
function renderDecision(){
  const stage=current()?.stages[selectedStage];
  title.textContent=selectedStage?labels[selectedStage]:"No prepared review";
  state.textContent=stage?.approved?"Approved":
    stage?.invalid?"Invalid artifacts":
    stage?.approval_ready===false?"Upstream rebuild required":"Review required";
  state.title=stage?.issue||"";
  let recorded=document.querySelector("#recorded-decision");
  if(!recorded){recorded=document.createElement("section");recorded.id="recorded-decision";
    form.querySelector(".heading").after(recorded);}
  recorded.replaceChildren();
  const scope=document.createElement("p");scope.textContent=stage?.approved
    ? "Recorded pipeline approval · current prepared result"
    : stage?.issue||"No approval recorded for this prepared result.";
  recorded.append(scope);
  for(const family of stage?.families||[]){const line=document.createElement("p");
    line.textContent=family.id+" · "+(family.approved?"Approved":"Review required");
    recorded.append(line);}
  for(const section of stage?.sections||[]){
    const item=document.createElement("details");
    const summary=document.createElement("summary");
    summary.textContent="Section "+section.id+" · "+
      (section.accepted?"Accepted":"Review required")+
      (section.reviewer?" · "+section.reviewer:"");
    const note=document.createElement("p");
    note.textContent=[section.reviewed_at,section.notes].filter(Boolean).join(" · ");
    item.append(summary,note);recorded.append(item);}
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
    const response=await fetch(histopiaUrl("/api/reviews/approve"),
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
configureAccess().catch(error=>{
  connection.textContent="Unavailable";
  connection.style.color="#7b241c";
  message.textContent=error.message;
});

// Keep same-origin data requests inside a code-server port proxy.
function histopiaUrl(path) {
  const match = location.pathname.match(new RegExp("^.*?/proxy/[0-9]+(?=/|$)"));
  return (match ? match[0] : "") + path;
}
"""
)
