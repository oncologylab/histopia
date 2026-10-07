# ruff: noqa: E501
"""Interactive expression-only review for cell-resolved protein transfer."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

from histopia.visualization._review_selection import REVIEW_SELECTION_JS
from histopia.visualization._review_theme import themed_review_css

ProteinRunDescriptor = Path | str | dict[str, Path | str]
ProteinCohortRuns = Path | str | dict[str, ProteinRunDescriptor]


def build_protein_review(
    runs: dict[str, ProteinCohortRuns],
    output_dir: Path | str,
) -> Path:
    """Build a path-free review page for configured sealed protein runs."""

    if not runs:
        raise ValueError("protein review requires at least one run")
    from histopia.protein._manifest import (
        validate_protein_artifact,
        validate_protein_result_index,
    )
    from histopia.protein._result import ProteinPredictions

    cohorts: list[dict[str, object]] = []
    validated_results: dict[Path, dict[str, object]] = {}
    validated_predictions: set[tuple[Path, str]] = set()
    for cohort, raw_runs in sorted(runs.items()):
        scoped = isinstance(raw_runs, dict)
        configured = raw_runs if scoped else {"legacy": raw_runs}
        models: list[dict[str, object]] = []
        seen: set[str] = set()
        for configured_id, raw_root in sorted(configured.items()):
            root = _configured_run_root(configured_id, raw_root)
            result = validated_results.get(root)
            if result is None:
                result = validate_protein_result_index(root)
                validated_results[root] = result
            declared_id = result.get("model_id")
            if scoped and declared_id is not None and declared_id != configured_id:
                raise ValueError(
                    f"configured protein model {configured_id!r} differs from "
                    "its sealed model ID"
                )
            fingerprint = str(result["fingerprint"])
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            approval_path = root / "protein_approval.json"
            try:
                approval = json.loads(approval_path.read_text())
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                approval = {}
            current_approval = (
                approval.get("fingerprint") == fingerprint
                and approval.get("accepted") is True
            )
            artifacts = result.get("artifacts")
            if not isinstance(artifacts, dict):
                raise ValueError(
                    f"protein model {configured_id!r} has no artifact manifest"
                )
            slides = []
            feature_views: set[str] = set()
            for row in result["slides"]:
                if result.get("schema_version") == 4 and row.get("cohort") != cohort:
                    continue
                prediction_relative = str(row["predictions"])
                prediction_key = (root, prediction_relative)
                if prediction_key not in validated_predictions:
                    digest = artifacts.get(prediction_relative)
                    if not isinstance(digest, str):
                        raise ValueError(
                            f"protein model {configured_id!r} has no sealed "
                            f"prediction for {prediction_relative}"
                        )
                    validate_protein_artifact(
                        root / prediction_relative,
                        digest,
                    )
                    validated_predictions.add(prediction_key)
                prediction = ProteinPredictions.load(root / prediction_relative)
                feature_views.add(
                    str(prediction.provenance.get("feature_view", "unknown"))
                )
                slides.append(
                    {
                        "section": row["section"],
                        "cells": row.get("cells", 0),
                        "measured_cells": row.get("measured_cells", 0),
                        "evaluation_role": row.get(
                            "evaluation_role",
                            "legacy-unknown"
                            if row.get("measured_cells", 0)
                            else "unmeasured-transfer",
                        ),
                    }
                )
            if not slides:
                raise ValueError(
                    f"protein model {configured_id!r} has no slides for {cohort}"
                )
            metrics = result.get("metrics", {})
            architecture = result.get("architecture")
            if not isinstance(architecture, str) and isinstance(metrics, dict):
                architecture = metrics.get("deployed_candidate")
            if not isinstance(architecture, str):
                architecture = "legacy"
            model_id = str(declared_id or configured_id)
            target_id = str(result["target_id"])
            raw_target_label = result.get("target_label")
            target_label = (
                raw_target_label.strip()
                if isinstance(raw_target_label, str) and raw_target_label.strip()
                else _target_display_name(target_id)
            )
            declared_label = result.get("model_label")
            fallback_label = f"{target_label} · {architecture.replace('_', ' ')}"
            models.append(
                {
                    "id": model_id,
                    "label": _normalized_model_label(
                        str(declared_label or fallback_label),
                        target_id,
                        target_label,
                    ),
                    "target_id": target_id,
                    "target_label": target_label,
                    "assay_domain": result["assay_domain"],
                    "architecture": architecture,
                    "training_cohorts": result.get("training_cohorts", []),
                    "version": result.get("model_version")
                    or result.get("schema_version"),
                    "prediction_protocol": result.get(
                        "prediction_protocol", "leave-one-mouse-out"
                    ),
                    "fingerprint": fingerprint,
                    "model_fingerprint": result["model_fingerprint"],
                    "approved": current_approval,
                    "promoted": bool(
                        result.get("status") == "promoted" and current_approval
                    ),
                    "metrics": metrics,
                    "slides": slides,
                    "measurement_view": result.get("measurement_view"),
                    "measurement_statistic": result.get(
                        "measurement_statistic", "mean"
                    ),
                    "feature_schema_id": result.get("feature_schema_id"),
                    "prediction_resolution": (
                        "cell_resolved_multiscale_v3"
                        if feature_views
                        and all(
                            view == "native-hdab-neutral-cell-multiscale-v3"
                            for view in feature_views
                        )
                        else (
                            "cell_resolved_spatial_tokens_v2"
                            if feature_views
                            and all(
                                view == "native-hdab-neutral-spatial-uni2h-v2"
                                for view in feature_views
                            )
                            else "legacy_tile_pooled_baseline"
                        )
                    ),
                    "api_scope": "model" if scoped else "legacy",
                }
            )
        if not models:
            raise ValueError(f"protein review cohort {cohort} has no unique models")
        models.sort(key=_model_sort_key)
        default = models[0]
        recommended = _recommended_model_ids(models)
        cohorts.append(
            {
                "id": cohort,
                "models": models,
                "default_model_id": default["id"],
                "recommended_model_ids": recommended,
                **{
                    key: default[key]
                    for key in (
                        "target_id",
                        "assay_domain",
                        "fingerprint",
                        "model_fingerprint",
                        "approved",
                        "metrics",
                        "slides",
                        "prediction_resolution",
                    )
                },
            }
        )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": 3, "cohorts": cohorts}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "manifest-data.js").write_text(
        "globalThis.HISTOPIA_PROTEIN_REVIEW="
        + json.dumps(manifest, separators=(",", ":"))
        + ";\n"
    )
    vendor = Path(__file__).parent / "_vendor"
    shutil.copyfile(vendor / "openseadragon.min.js", output / "openseadragon.min.js")
    shutil.copyfile(vendor / "LICENSE-openseadragon.txt", output / "LICENSE.txt")
    (output / "index.html").write_text(_HTML)
    (output / "protein-review.css").write_text(themed_review_css(_CSS))
    (output / "protein-review.js").write_text(_JS)
    return output / "index.html"


def _configured_run_root(
    configured_id: str,
    value: ProteinRunDescriptor,
) -> Path:
    """Resolve either a bare run or the server's additive run descriptor."""

    raw_root: Path | str | None
    if isinstance(value, dict):
        raw_root = value.get("run")
        if raw_root is None:
            raise ValueError(
                f"configured protein model {configured_id!r} has no run path"
            )
    else:
        raw_root = value
    if not isinstance(raw_root, (str, Path)):
        raise TypeError(
            f"configured protein model {configured_id!r} run must be a path"
        )
    return Path(raw_root).expanduser().resolve()


def _target_display_name(target_id: str) -> str:
    return {
        "yap": "YAP",
        "ecad": "E-Cad",
        "ck19": "CK19",
        "ki67": "Ki67",
        "perk": "pERK",
        "ncad": "N-Cad",
        "cjun": "cJun",
    }.get(target_id.lower(), target_id.upper())


def _normalized_model_label(
    label: str,
    target_id: str,
    target_label: str | None = None,
) -> str:
    """Keep sealed model detail while normalizing familiar marker spelling."""

    display = target_label or _target_display_name(target_id)
    raw_prefix = target_id.upper()
    return display + label[len(raw_prefix) :] if label.startswith(raw_prefix) else label


def _model_sort_key(row: dict[str, object]) -> tuple[object, ...]:
    """Prefer reviewed models with the strongest held-out spatial agreement."""

    metrics = row.get("metrics")
    values = metrics if isinstance(metrics, dict) else {}

    def descending(name: str) -> float:
        value = values.get(name)
        return (
            -float(value)
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            else math.inf
        )

    return (
        not bool(row.get("promoted")),
        not bool(row.get("approved")),
        descending("spearman_64um"),
        descending("spearman"),
        str(row.get("target_id", "")),
        str(row.get("label", "")),
        str(row.get("id", "")),
    )


def _recommended_model_ids(
    models: list[dict[str, object]],
) -> dict[str, dict[str, str]]:
    """Select only approved role-aware defaults while retaining all models."""

    output: dict[str, dict[str, str]] = {}
    for target in sorted({str(row["target_id"]) for row in models}):
        rows = [
            row
            for row in models
            if row["target_id"] == target and bool(row.get("approved"))
        ]
        measured = [
            row for row in rows if row.get("prediction_protocol") == "training-visible"
        ]
        transfer = [
            row for row in rows if row.get("prediction_protocol") != "training-visible"
        ]
        selected: dict[str, str] = {}
        if measured or transfer:
            selected["measured"] = str(
                min(measured or transfer, key=_model_sort_key)["id"]
            )
        if transfer:
            selected["unmeasured"] = str(min(transfer, key=_model_sort_key)["id"])
        if selected:
            output[target] = selected
    return output


_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="icon" href="data:,">
<title>Histopia protein expression</title>
<link rel="stylesheet" href="protein-review.css"></head>
<body><header><div class="controls"><strong>Protein expression</strong>
<label>Mouse <select id="cohort"></select></label>
<label>Protein <select id="model-target"></select></label>
<label>Section <select id="section"></select></label>
<select id="architecture" hidden aria-hidden="true" tabindex="-1"></select>
<select id="model" hidden aria-hidden="true" tabindex="-1"></select></div></header>
<main><section id="summary"><div><b id="metric-title">Cell-resolved protein prediction</b><span id="metrics">One expression value per segmented cell</span></div>
<p id="scope-note">Predicted expression and measured stain references are shown on aligned sections.</p>
<div id="legend"><span id="legend-low">0 OD</span><i></i><span id="legend-high">OD</span></div></section>
<section id="panes">
<article><h2><span id="predicted-title">Predicted YAP</span><small id="predicted-scope"></small></h2><div data-pane="predicted"></div></article>
<article><h2><span id="current-title">Actual section stain</span><small id="current-scope"></small></h2><div data-pane="current"></div></article>
<article><h2><span id="comparison-title">Comparison</span><small id="comparison-scope"></small></h2><div data-pane="comparison"></div></article>
</section></main><div id="message" role="status"></div>
<script src="manifest-data.js"></script><script src="openseadragon.min.js"></script>
<script src="protein-review.js"></script></body></html>"""

_CSS = """
:root{font-family:Inter,system-ui,sans-serif;color:#17202a;background:#f5f7f8}
*{box-sizing:border-box}html,body{width:100%;height:100%;margin:0;overflow:hidden}
body{display:grid;grid-template-rows:auto minmax(0,1fr)}
header{display:grid;gap:5px;padding:8px 16px;background:#fff;
border-bottom:1px solid #d5dbdb;min-width:0}.controls{display:flex;align-items:center;
gap:8px 14px;min-width:0}.controls strong{color:#145a4a;white-space:nowrap}
label{display:flex;align-items:center;gap:6px;font-size:12px;color:#566573}
select{min-width:90px;max-width:240px;padding:6px;border:1px solid #abb2b9;background:#fff}
main{display:grid;grid-template-rows:auto minmax(0,1fr);min-height:0;padding:10px;gap:8px}
#summary{display:flex;align-items:center;justify-content:space-between;gap:16px;padding:8px 12px;background:#fff;
border:1px solid #dfe6e9;border-radius:6px;font-size:12px}#summary div{display:grid;gap:3px}
#summary p{margin:0;color:#566573;text-align:center;max-width:720px}#legend{display:flex!important;align-items:center;gap:5px!important;white-space:nowrap}
#legend i{display:block;width:86px;height:9px;border-radius:5px;background:linear-gradient(90deg,#faf6eb,#be8548,#5c3014)}#panes{display:grid;
grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;min-height:0}
article{display:grid;grid-template-rows:54px minmax(0,1fr);min-width:0;min-height:0;
background:#fff;border:1px solid #dfe6e9;border-radius:6px;overflow:hidden}
h2{margin:0;padding:7px 10px;font-size:12px;font-weight:650;border-bottom:1px solid #edf1f2;display:grid;align-content:center;gap:3px}
h2 small{font-size:10px;line-height:1.25;font-weight:450;color:#66757f;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
[data-pane]{position:relative;min-width:0;min-height:0;background:#fff}
[data-pane].loading::after{content:"Loading expression tiles…";position:absolute;left:50%;top:50%;
transform:translate(-50%,-50%);z-index:10;padding:7px 10px;border:1px solid #dfe6e9;border-radius:4px;
background:rgba(255,255,255,.94);color:#566573;font-size:11px;white-space:nowrap}
#message{position:fixed;left:50%;bottom:16px;
transform:translateX(-50%);padding:7px 12px;background:#17202a;color:#fff;border-radius:4px;
font-size:12px;display:none;z-index:20}#message.visible{display:block}
@media(max-width:1500px){header{padding-inline:12px}.controls{flex-wrap:wrap;gap:5px 10px}}
@media(max-width:1100px){.controls label{flex:1 1 auto}}
@media(max-width:720px){header{gap:4px;padding:5px 8px}.controls{display:grid;
grid-template-columns:repeat(2,minmax(0,1fr));gap:4px 8px}.controls strong{grid-column:1/-1}.controls label{display:grid;
grid-template-columns:auto minmax(0,1fr);min-width:0;overflow:hidden}.controls select{width:100%;min-width:0;
max-width:100%}main{padding:5px}
#summary{display:block}#summary p{display:none}#legend{margin-top:5px}#panes{grid-template-columns:1fr;grid-template-rows:repeat(3,minmax(190px,1fr));overflow:auto}
article{min-height:190px}}
"""

_JS = (
    REVIEW_SELECTION_JS
    + r""""use strict";
const manifest=globalThis.HISTOPIA_PROTEIN_REVIEW;
if(!manifest?.cohorts?.length)throw new Error("Missing protein review manifest");
const cohortSelect=document.querySelector("#cohort"),targetSelect=document.querySelector("#model-target"),architectureSelect=document.querySelector("#architecture"),modelSelect=document.querySelector("#model"),sectionSelect=document.querySelector("#section");
const metrics=document.querySelector("#metrics"),message=document.querySelector("#message");
const viewers=new Map();let cohort=null;let activeModel=null;let syncFrame=0;let syncSource=null;let resizeFrame=0;let request=0;let catalogRequest=0;
const interactionEvents=["canvas-drag","canvas-drag-end","canvas-scroll","canvas-pinch","canvas-double-click","canvas-click","canvas-key","canvas-key-press"];
function targetLabel(value){const key=String(value).toLowerCase();const declared=cohort?.models?.find(row=>String(row.target_id).toLowerCase()===key)?.target_label;return declared||({yap:"YAP",ecad:"E-Cad",ck19:"CK19",ki67:"Ki67",perk:"pERK",ncad:"N-Cad",cjun:"cJun"})[key]||String(value).toUpperCase();}
function stainLabel(value){const key=String(value).trim().toLowerCase();return({he:"H&E",h_and_e:"H&E",ecad:"E-Cad",ncad:"N-Cad",perk:"pERK",cjun:"cJun"})[key]||String(value);}
function architectureLabel(value){return({equal_ensemble:"ensemble",extra_trees:"ExtraTrees",cross_attention:"attention",hurdle_mlp:"hurdle MLP",multi_tower:"multiscale towers",dual_bank_attention:"dual-bank attention",graph_transformer:"graph transformer",shared_multitask:"shared multi-target"})[value]||String(value).replaceAll("_"," ");}
function statisticLabel(value){return({mean:"mean",q90:"90th-percentile"})[value]||String(value).replaceAll("_"," ");}
function show(text){message.textContent=text;message.classList.toggle("visible",Boolean(text));}
function markLoading(value){document.querySelectorAll("[data-pane]").forEach(element=>element.classList.toggle("loading",value));}
function requestError(message,statusCode=0){const error=new Error(message);error.status=statusCode;return error;}
async function fetchJson(endpoint,label){let response;
 try{response=await fetch(endpoint,{cache:"no-store"});}catch(error){throw requestError(`${label} could not reach the live tile API.`);}
 if(!response.ok)throw requestError(`${label} failed (HTTP ${response.status}).`,response.status);
 try{return await response.json();}catch(error){throw requestError(`${label} returned invalid metadata.`);}}
function reconcileCohort(raw,catalog){const liveRows=Array.isArray(catalog.protein_models)?catalog.protein_models:[];
 const liveById=new Map(liveRows.map(row=>[row.id,row]));
 let models=raw.models.flatMap(row=>{if(row.api_scope!=="model")return [row];const current=liveById.get(row.id);if(!current)return [];
  const declared=String(row.fingerprint||"");const published=String(current.result_fingerprint||"");if(declared&&published&&declared!==published)return [];
  const sections=new Set(Array.isArray(current.sections)?current.sections:[]);const slides=(row.slides||[]).filter(slide=>sections.has(slide.section));if(!slides.length)return [];
  return[{...row,...current,fingerprint:published||declared,slides,api_scope:"model"}];});
 if(!models.length)throw new Error(`No protein maps are currently published for mouse ${raw.id}.`);
 const reviewed=models.filter(row=>row.promoted||row.approved);if(reviewed.length)models=reviewed;
 const publishedDefault=String(catalog.default_protein_model_id||"");
 const preferred=models.find(row=>row.id===publishedDefault)||models.find(row=>row.id===raw.default_model_id)||models.find(row=>row.promoted)||models.find(row=>row.approved)||models[0];
 const recommended=catalog.recommended_protein_models||raw.recommended_model_ids||{};
 return{...raw,models,default_model_id:preferred.id,recommended_model_ids:recommended};}
function tileSource(meta,name){const layer=meta.layers[name];if(!layer)return null;
 const min=Math.max(0,layer.levels.length-1-Math.ceil(Math.log2(Math.max(layer.width,layer.height))));
 // Protein outputs and their exact stain provenance are model-scoped. Native
 // histology, masks, and cell labels stay section-scoped for browser reuse.
 const modelScope=activeModel?.api_scope==="model"&&meta.protein_model_id&&(name.startsWith("protein_")||name.startsWith("stain_"));
 const prefix=modelScope?histopiaUrl(`/api/wsi/${encodeURIComponent(meta.cohort)}/${meta.section}/protein/${encodeURIComponent(meta.protein_model_id)}/${name}/${layer.digest}/dzi/${min}/`):histopiaUrl(`/api/wsi/${encodeURIComponent(meta.cohort)}/${meta.section}/${name}/${layer.digest}/dzi/${min}/`);
 return new OpenSeadragon.TileSource({width:layer.width,height:layer.height,tileSize:layer.tile_size,
  minLevel:min,maxLevel:min+layer.levels.length-1,getTileUrl(level,x,y){return `${prefix}${level}/${x}_${y}.${layer.format}`;}});}
function viewer(name){if(viewers.has(name))return viewers.get(name);const instance=OpenSeadragon({
 element:document.querySelector(`[data-pane="${name}"]`),showNavigator:false,showNavigationControl:false,
 animationTime:0,blendTime:0,immediateRender:true,imageLoaderLimit:2,constrainDuringPan:true,
 visibilityRatio:.2,minZoomImageRatio:.7});
 interactionEvents.forEach(event=>instance.addHandler(event,()=>scheduleSync(instance)));
 viewers.set(name,instance);return instance;}
function scheduleSync(source){if(!source.world.getItemCount())return;syncSource=source;if(syncFrame)return;
 syncFrame=requestAnimationFrame(()=>{syncFrame=0;const next=syncSource;syncSource=null;sync(next);});}
function viewportState(instance){if(!instance?.world.getItemCount())return null;const item=instance.world.getItemAt(0);
 const centerImage=item.viewportToImageCoordinates(instance.viewport.getCenter(true));
 const relativeZoom=instance.viewport.getZoom(true)/instance.viewport.getHomeZoom();
 if(!Number.isFinite(centerImage.x)||!Number.isFinite(centerImage.y)||!Number.isFinite(relativeZoom))return null;
 return{fx:centerImage.x/item.source.dimensions.x,fy:centerImage.y/item.source.dimensions.y,relativeZoom};}
function sameViewport(instance,state){const current=viewportState(instance);return current&&
 Math.abs(current.fx-state.fx)<1e-9&&Math.abs(current.fy-state.fy)<1e-9&&
 Math.abs(current.relativeZoom-state.relativeZoom)<1e-9;}
function sync(source){const state=viewportState(source);if(!state)return;
 const {fx,fy,relativeZoom}=state;
 viewers.forEach(other=>{if(other===source||!other.world.getItemCount()||sameViewport(other,state))return;
  const otherItem=other.world.getItemAt(0);
  const point=otherItem.imageToViewportCoordinates(fx*otherItem.source.dimensions.x,fy*otherItem.source.dimensions.y);
  other.viewport.panTo(point,true);other.viewport.zoomTo(relativeZoom*other.viewport.getHomeZoom(),null,true);
  other.viewport.applyConstraints(true);});}
function setSharedZoomLimit(){const primary=viewers.get("predicted");if(!primary?.world.getItemCount())return;
 primary.viewport.maxZoomLevel=null;
 const relativeLimit=primary.viewport.getMaxZoom()/primary.viewport.getHomeZoom();
 if(!Number.isFinite(relativeLimit)||relativeLimit<=0)return;
 viewers.forEach(instance=>{if(instance.world.getItemCount())instance.viewport.maxZoomLevel=
  relativeLimit*instance.viewport.getHomeZoom();});}
function cancelSync(){if(syncFrame)cancelAnimationFrame(syncFrame);syncFrame=0;syncSource=null;}
function synchronizeAfterResize(){if(resizeFrame)return;resizeFrame=requestAnimationFrame(()=>requestAnimationFrame(()=>{
 resizeFrame=0;const primary=viewers.get("predicted");if(!primary?.world.getItemCount())return;
 setSharedZoomLimit();primary.viewport.applyConstraints(true);scheduleSync(primary);}));}
function add(instance,meta,name,mine){const source=tileSource(meta,name);if(!source)return Promise.resolve(false);
 return new Promise((resolve,reject)=>{let item=null,settled=false,timer=0;
  function cleanup(){instance.removeHandler("tile-loaded",onTile);if(timer)clearTimeout(timer);}
  function finish(value){if(settled)return;settled=true;cleanup();requestAnimationFrame(()=>{instance.forceRedraw();resolve(value);});}
  function onTile(event){if(!item||event.tiledImage!==item)return;finish(mine===request);}
  instance.addHandler("tile-loaded",onTile);
  instance.addTiledImage({tileSource:source,opacity:1,
   success:event=>{item=event.item;if(mine!==request){if(item&&instance.world.getIndexOfItem(item)>=0)instance.world.removeItem(item);finish(false);return;}
    timer=setTimeout(()=>finish(true),10000);},
   error:event=>{cleanup();if(mine!==request){finish(false);return;}reject(new Error(`Tile layer ${name} could not be opened.`));}});});}
async function loadPane(name,meta,layer,mine){const element=document.querySelector(`[data-pane="${name}"]`);element.classList.add("loading");
 const instance=viewer(name);instance.viewport.maxZoomLevel=null;instance.world.removeAll();
 const loaded=await add(instance,meta,layer,mine);if(mine!==request)return instance;
 if(!loaded)throw new Error(`Missing comparison layer: ${layer}`);
 instance.viewport.goHome(true);return instance;}
function waitForVisibleTiles(instance,mine){return new Promise(resolve=>{let settled=false;
 const timer=setTimeout(finish,45000);function cleanup(){clearTimeout(timer);instance.removeHandler("fully-loaded-change",check);}
 function finish(){if(settled)return;settled=true;cleanup();resolve();}
 function check(event){if(mine!==request||event?.fullyLoaded===true||instance.getFullyLoaded())finish();}
 instance.addHandler("fully-loaded-change",check);requestAnimationFrame(()=>requestAnimationFrame(()=>check()));});}
function fallbackComparison(meta){const measured=Boolean(meta.layers.protein_measured);return{
 mode:measured?"measured-target":"legacy-diagnostic",target_id:activeModel.target_id,predicted_layer:"protein_predicted",
 current_layer:measured?"protein_measured":"protein_dense",current_label:measured?activeModel.target_id:"Prediction raster",
 current_scope:measured?"legacy_cell_aggregate":"legacy_diagnostic",comparison_layer:measured?"protein_residual":"protein_uncertainty",
 comparison_label:measured?"Absolute per-cell residual":"Prediction uncertainty",ground_truth_available:measured,nearest_target_section:null};}
function setText(id,text){document.querySelector(id).textContent=text;}
function restoreViewport(instance,state){if(!state||!instance?.world.getItemCount())return false;const item=instance.world.getItemAt(0);
 const point=item.imageToViewportCoordinates(state.fx*item.source.dimensions.x,state.fy*item.source.dimensions.y);
 instance.viewport.panTo(point,true);instance.viewport.zoomTo(state.relativeZoom*instance.viewport.getHomeZoom(),null,true);
 instance.viewport.applyConstraints(true);return true;}
function fitFocus(instance,meta){const box=meta?.focus_bbox;if(!box||!instance?.world.getItemCount())return false;
 const values=[box.x,box.y,box.width,box.height].map(Number);if(values.some(value=>!Number.isFinite(value))||values[2]<=0||values[3]<=0)return false;
 const [x,y,width,height]=values;const item=instance.world.getItemAt(0),dimensions=item.source.dimensions;
 const start=item.imageToViewportCoordinates(x*dimensions.x,y*dimensions.y);
 const end=item.imageToViewportCoordinates((x+width)*dimensions.x,(y+height)*dimensions.y);
 instance.viewport.fitBounds(new OpenSeadragon.Rect(start.x,start.y,end.x-start.x,end.y-start.y),true);
 instance.viewport.applyConstraints(true);return true;}
async function load(preserveViewport=false){const mine=++request;const saved=preserveViewport?viewportState(viewers.get("predicted")):null;cancelSync();markLoading(true);show("Loading expression maps…");
 const section=sectionSelect.value;const base=histopiaUrl(`/api/wsi/${encodeURIComponent(cohort.id)}/${section}`);
 const endpoint=activeModel.api_scope==="model"?`${base}/protein/${encodeURIComponent(activeModel.id)}`:base;
 let next;try{next=await fetchJson(endpoint,`Protein model ${activeModel.id}`);}catch(error){if(mine!==request)return;
  markLoading(false);if(error.status===404&&activeModel.api_scope==="model"){show("The selected model is no longer published. Refreshing the validated model registry…");chooseCohort();return;}throw error;}if(mine!==request)return;
 const comparison=next.protein_comparison||fallbackComparison(next);const measured=Boolean(comparison.ground_truth_available);
 const predictedLayer=next.layers[comparison.predicted_layer];
 document.querySelector("#legend-high").textContent=`${Number(predictedLayer?.display_max||1).toFixed(2)} OD`;
 const legacy=activeModel.prediction_resolution==="legacy_tile_pooled_baseline";
 const targetUpper=targetLabel(comparison.target_id);const statistic=statisticLabel(activeModel.measurement_statistic||"mean");
 const noLocalTruth=comparison.mode==="external-no-local-truth";
 setText("#metric-title",measured?`${targetUpper} prediction and measurement`:`${targetUpper} cross-section prediction`);
 metrics.textContent=`Single-cell expression · ${statistic} target OD`;
 document.querySelector("#scope-note").textContent=measured?`Predicted and observed ${targetUpper} are shown on one shared OD scale.`:(comparison.nearest_target_section?`Predicted ${targetUpper} is shown with this section's measured stain and the nearest observed ${targetUpper} section.`:`Predicted ${targetUpper} is shown with this section's measured stain and spatial confidence map.`);
 setText("#predicted-title",`Predicted ${targetUpper}`);
 setText("#predicted-scope",legacy?"Cell-resolved expression map":"Single-cell expression · detected cell geometry");
 if(comparison.current_scope==="counterstain_conditioned_target_od_4um"){
  setText("#current-title",`Observed ${targetUpper} · 4 µm/px`);
  setText("#current-scope",`Counterstain-conditioned target OD · tissue mask`);
 }else if(comparison.current_scope==="harmonized_adaptive_target_od_4um"){
  setText("#current-title",`Observed ${targetUpper} · 4 µm/px`);
  setText("#current-scope",`Harmonized adaptive target OD · tissue mask`);
 }else if(comparison.current_scope==="adaptive_target_od_4um"){
  setText("#current-title",`Observed ${targetUpper} · 4 µm/px`);
  setText("#current-scope",`Adaptive target OD · tissue mask`);
 }else if(comparison.current_scope==="calibrated_target_od_4um"){
  setText("#current-title",`Observed ${targetUpper} · 4 µm/px`);
  setText("#current-scope",`Calibrated target OD · tissue mask`);
 }else if(comparison.current_scope==="current_marker_counterstain_conditioned_od_4um"||comparison.current_scope==="current_marker_adaptive_od_4um"||comparison.current_scope==="current_marker_od_4um"){
  setText("#current-title",`Observed ${stainLabel(comparison.current_label)} · 4 µm/px`);
  setText("#current-scope",`This section's measured antibody · corrected target OD · tissue mask`);
 }else if(comparison.current_scope==="no_quantified_stain"){
  setText("#current-title",`No quantitative stain map · ${stainLabel(comparison.current_label)}`);
  setText("#current-scope",`Tissue-supported section context`);
 }else{
  setText("#current-title",`Observed ${stainLabel(comparison.current_label)}`);
  setText("#current-scope",`This section's measured stain · native spatial context`);
 }
 setText("#comparison-title",measured?"Absolute per-cell difference":(noLocalTruth?comparison.comparison_label:`Nearest observed ${targetUpper} · section ${comparison.nearest_target_section}`));
 setText("#comparison-scope",measured?`Predicted versus observed ${targetUpper} OD`:(noLocalTruth?"Spatial confidence across model estimates":"Registered spatial reference"));
 await Promise.all([loadPane("predicted",next,comparison.predicted_layer,mine),
  loadPane("current",next,comparison.current_layer,mine),loadPane("comparison",next,comparison.comparison_layer,mine)]).catch(error=>{if(mine===request)markLoading(false);throw error;});
 if(mine!==request)return;
 const primary=viewers.get("predicted");setSharedZoomLimit();if(!restoreViewport(primary,saved)&&!fitFocus(primary,next))viewers.forEach(instance=>instance.viewport.goHome(true));sync(primary);
 await Promise.all([...viewers.values()].map(instance=>waitForVisibleTiles(instance,mine)));if(mine!==request)return;markLoading(false);show("");}
function sectionRole(target,section){const row=cohort.models.filter(model=>model.target_id===target).flatMap(model=>model.slides||[]).find(slide=>slide.section===section);return row?.measured_cells>0?"measured":"unmeasured";}
function recommendedModelId(target=targetSelect.value,section=sectionSelect.value){const role=sectionRole(target,section);return cohort.recommended_model_ids?.[target]?.[role]||"";}
function bestModelId(target=targetSelect.value,section=sectionSelect.value){const approved=recommendedModelId(target,section);if(approved&&cohort.models.some(row=>row.id===approved))return approved;
 const rows=cohort.models.filter(row=>row.target_id===target);const measured=sectionRole(target,section)==="measured";
 const roleRows=rows.filter(row=>measured?row.prediction_protocol==="training-visible":row.prediction_protocol!=="training-visible");
 const ordered=[...roleRows,...rows.filter(row=>!roleRows.includes(row))];
 return (ordered.find(row=>row.promoted)||ordered.find(row=>row.approved)||ordered[0])?.id||"";}
function applyBestModel(preserveViewport=true){const id=bestModelId();const selected=cohort.models.find(row=>row.id===id);if(!selected){load(preserveViewport).catch(error=>show(error.message));return;}
 if(activeModel?.id===selected.id){load(preserveViewport).catch(error=>show(error.message));return;}
 targetSelect.value=selected.target_id;architectureSelect.value=selected.architecture;chooseModelGroup(preserveViewport);}
function updateModel(preserveViewport=true){activeModel=cohort.models.find(row=>row.id===modelSelect.value)||cohort.models[0];
 setText("#metric-title",`Cell-resolved ${targetLabel(activeModel.target_id)} prediction`);metrics.textContent=`One ${statisticLabel(activeModel.measurement_statistic||"mean")} expression value per segmented cell`;
 const priorSection=sectionSelect.value;sectionSelect.replaceChildren(...activeModel.slides.map(row=>{const option=document.createElement("option");option.value=row.section;
  option.textContent=`${row.section} · ${row.cells} cells${row.measured_cells?` · ${targetLabel(activeModel.target_id)} measured`:""}`;return option;}));
 if(activeModel.slides.some(row=>row.section===priorSection))sectionSelect.value=priorSection;
 setText("#legend-low",`${targetLabel(activeModel.target_id)} 0 OD`);
 load(preserveViewport).catch(error=>show(error.message));}
function chooseModelGroup(preserveViewport=true){const candidates=cohort.models.filter(row=>row.target_id===targetSelect.value&&row.architecture===architectureSelect.value);
 const prior=modelSelect.value;modelSelect.replaceChildren(...candidates.map(row=>{const option=document.createElement("option");option.value=row.id;option.textContent=targetLabel(row.target_id);return option;}));
 const automatic=bestModelId();const preferred=candidates.find(row=>row.id===automatic)||candidates.find(row=>row.id===prior)||candidates.find(row=>row.id===cohort.default_model_id)||candidates[0];if(!preferred)return;modelSelect.value=preferred.id;updateModel(preserveViewport);}
function chooseArchitecture(preserveViewport=true){const rows=cohort.models.filter(row=>row.target_id===targetSelect.value);const values=[...new Set(rows.map(row=>row.architecture))];
 const prior=architectureSelect.value;architectureSelect.replaceChildren(...values.map(value=>{const option=document.createElement("option");option.value=value;option.textContent=architectureLabel(value);return option;}));
 const automatic=bestModelId();const preferred=rows.find(row=>row.id===automatic)||rows.find(row=>row.id===cohort.default_model_id);architectureSelect.value=preferred?.architecture||(values.includes(prior)?prior:values[0]);chooseModelGroup(preserveViewport);}
function chooseTarget(preserveViewport=true){const values=[...new Set(cohort.models.map(row=>row.target_id))];
 const prior=targetSelect.value;const preferred=cohort.models.find(row=>row.id===cohort.default_model_id);targetSelect.replaceChildren(...values.map(value=>{const option=document.createElement("option");option.value=value;option.textContent=targetLabel(value);return option;}));
 targetSelect.value=values.includes(prior)?prior:(preferred?.target_id||values[0]);chooseArchitecture(preserveViewport);}
async function chooseCohort(){const mine=++catalogRequest;++request;cancelSync();markLoading(true);show("Loading protein maps…");
 const selected=manifest.cohorts.find(row=>row.id===cohortSelect.value)||manifest.cohorts[0];
 histopiaReviewSelection.remember(selected.id);
 const models=selected.models?.length?selected.models:[{...selected,id:"legacy",label:`${targetLabel(selected.target_id)} · legacy`,architecture:"legacy",api_scope:"legacy"}];
 const configured={...selected,models};try{if(models.some(row=>row.api_scope==="model")){const catalog=await fetchJson(histopiaUrl(`/api/wsi/${encodeURIComponent(selected.id)}`),"Protein model registry");
   if(mine!==catalogRequest)return;cohort=reconcileCohort(configured,catalog);}else{cohort=configured;}
  if(mine!==catalogRequest)return;chooseTarget(false);}catch(error){if(mine===catalogRequest){markLoading(false);show(error.message);}}}
manifest.cohorts.forEach(row=>{const option=document.createElement("option");option.value=row.id;option.textContent=row.id;cohortSelect.append(option);});
cohortSelect.addEventListener("change",()=>chooseCohort());targetSelect.addEventListener("change",()=>chooseArchitecture(true));sectionSelect.addEventListener("change",()=>applyBestModel(false));
addEventListener("resize",synchronizeAfterResize);
try{cohortSelect.value=histopiaReviewSelection.requested(manifest.cohorts);chooseCohort();}
catch(error){markLoading(false);show(error.message);}

// Keep same-origin data requests inside a code-server port proxy.
function histopiaUrl(path) {
  const match = location.pathname.match(/^.*?\/proxy\/[0-9]+(?=\/|$)/);
  return (match ? match[0] : "") + path;
}
"""
)
