"""Connected-volume topology viewer with fingerprint-bound review controls."""

# ruff: noqa: E501

from __future__ import annotations

import json
import shutil
from pathlib import Path
from statistics import median

from histopia.topology import validate_topology_result


def build_topology_review(
    runs: dict[str, Path | str],
    output_dir: Path | str,
) -> Path:
    """Build the canonical connected-volume topology application."""

    if not runs:
        raise ValueError("topology review requires at least one run")
    output = Path(output_dir)
    assets = output / "assets"
    vendor = output / "vendor"
    assets.mkdir(parents=True, exist_ok=True)
    vendor.mkdir(parents=True, exist_ok=True)
    source_vendor = Path(__file__).with_name("_vendor")
    for name in ("three.module.min.js", "OrbitControls.js"):
        shutil.copyfile(source_vendor / name, vendor / name)

    cohorts = []
    for cohort, value in sorted(runs.items()):
        root = Path(value)
        payload = validate_topology_result(root)
        benchmark = json.loads((root / str(payload["benchmark"])).read_text())
        cohort_assets = assets / cohort
        cohort_assets.mkdir(parents=True, exist_ok=True)
        legacy = payload["schema_version"] == 1
        if legacy:
            legacy_meshes = payload.get("meshes", [])
            envelope_source = legacy_meshes[0] if legacy_meshes else None
            region_sources = legacy_meshes
            partition_sources = []
            uncertainty_source = None
            reconstruction_qc = _legacy_surface_qc(payload, benchmark["summary"])
            numerical_samples = len(payload.get("planes", []))
        else:
            envelope_source = payload["envelope"]
            region_sources = payload["semantic_regions"]
            partition_sources = payload.get("semantic_partition_regions", [])
            uncertainty_source = payload.get("uncertainty")
            reconstruction_qc = payload["reconstruction_qc"]
            numerical_samples = payload["reconstruction_grid"]["numerical_sample_count"]

        envelope = _copy_mesh(
            root,
            output,
            cohort_assets,
            envelope_source,
            "envelope.bin",
        )
        regions = [
            _copy_mesh(
                root,
                output,
                cohort_assets,
                row,
                f"region-{index:02d}.bin",
            )
            for index, row in enumerate(region_sources)
        ]
        partitions = [
            _copy_mesh(
                root,
                output,
                cohort_assets,
                row,
                f"partition-{index:02d}.bin",
            )
            for index, row in enumerate(partition_sources)
        ]
        uncertainty = _copy_mesh(
            root,
            output,
            cohort_assets,
            uncertainty_source,
            "uncertainty.bin",
        )
        meaningful_regions = [
            (index, row)
            for index, row in enumerate(regions)
            if row is not None
            and float(row.get("viewer_core_volume_fraction_of_tissue", 0.0)) >= 0.015
        ]
        total_scientific_volume = sum(
            float(row.get("estimated_volume_mm3", 0.0)) for row in payload["classes"]
        )
        default_region_index = min(
            meaningful_regions or list(enumerate(regions)),
            key=lambda item: _region_review_score(
                item[1],
                total_scientific_volume=total_scientific_volume,
            ),
        )[0]
        observed = [
            {
                "slide_id": row.get("slide_id"),
                "source_section": row.get("source_section"),
                "z_um": row.get("z_um"),
            }
            for row in payload.get("planes", [])
            if row.get("observed")
        ]
        transition_qc = [
            {
                "id": (
                    f"{int(decision['source_section']):03d}-"
                    f"{int(decision['target_section']):03d}"
                ),
                "flags": _transition_outlier_flags(
                    (
                        payload["pair_evidence"][index]
                        if index < len(payload.get("pair_evidence", []))
                        else {}
                    ),
                    decision,
                ),
            }
            for index, decision in enumerate(payload["gap_decisions"])
        ]
        for row in transition_qc:
            row["outlier"] = bool(row["flags"])
        cohorts.append(
            {
                "id": cohort,
                "fingerprint": payload["fingerprint"],
                "schema_version": payload["schema_version"],
                "legacy_diagnostic": legacy,
                "selected_k": payload["selected_k"],
                "palette": payload["palette"],
                "z_source": payload["z_source"],
                "section_thickness_um": payload["section_thickness_um"],
                "source_patch_width_um": payload.get("source_patch_width_um"),
                "observed_section_count": payload["observed_section_count"],
                "virtual_section_count": payload["virtual_section_count"],
                "numerical_sample_count": numerical_samples,
                "gap_decisions": payload["gap_decisions"],
                "pair_evidence": payload["pair_evidence"],
                "transition_qc": transition_qc,
                "observed_sections": observed,
                "envelope": envelope,
                "semantic_regions": regions,
                "semantic_partition_regions": partitions,
                "default_region_index": default_region_index,
                "uncertainty": uncertainty,
                "classes": payload["classes"],
                "benchmark": benchmark["summary"],
                "reconstruction_qc": reconstruction_qc,
            }
        )
    manifest = {"schema_version": 4, "cohorts": cohorts}
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    encoded = json.dumps(manifest, separators=(",", ":"))
    (output / "manifest-data.js").write_text(
        f"globalThis.HISTOPIA_TOPOLOGY_REVIEW={encoded};\n"
    )
    (output / "index.html").write_text(_HTML)
    (output / "topology-review.css").write_text(_CSS)
    (output / "topology-review.js").write_text(_JS)
    return output / "index.html"


def _region_review_score(
    row: dict[str, object] | None,
    *,
    total_scientific_volume: float,
) -> tuple[float, int]:
    """Rank a moderate, compact semantic core for the initial review view."""

    if row is None:
        return (float("inf"), 1_000_000)
    fraction = float(row.get("viewer_core_volume_fraction_of_tissue", 0.0))
    core_volume = fraction * total_scientific_volume
    surface_area = float(row.get("surface_area_mm2", 0.0))
    compactness = (
        surface_area / core_volume ** (2.0 / 3.0)
        if core_volume > 0 and surface_area > 0
        else 1_000_000.0
    )
    components = int(row.get("component_count_after_filter", 1_000_000))
    if fraction < 0.05:
        size_penalty = 1_000.0 * (0.05 - fraction)
    elif fraction > 0.22:
        size_penalty = 100.0 * (fraction - 0.22)
    else:
        size_penalty = 10.0 * abs(fraction - 0.10)
    score = compactness + 7.0 * max(components - 1, 0) + size_penalty
    return (score, int(row.get("class_index", 1_000_000)))


def _transition_outlier_flags(
    evidence: object,
    decision: object,
) -> list[str]:
    """Return deterministic review flags for one reconstructed transition."""

    if not isinstance(evidence, dict):
        evidence = {}
    if not isinstance(decision, dict):
        decision = {}
    flags: list[str] = []
    status = decision.get("status")
    if status in {"inferred", "unresolved"}:
        flags.append(f"{status}_z_spacing")
    lower_gates = {
        "support_dice": 0.85,
        "matched_label_agreement": 0.50,
        "correspondence_coverage": 0.50,
        "median_confidence": 0.55,
    }
    upper_gates = {
        "semantic_js": 0.20,
        "displacement_strain": 0.75,
    }
    for name, threshold in lower_gates.items():
        value = evidence.get(name)
        if isinstance(value, (int, float)) and float(value) < threshold:
            flags.append(f"low_{name}")
    for name, threshold in upper_gates.items():
        value = evidence.get(name)
        if isinstance(value, (int, float)) and float(value) > threshold:
            flags.append(f"high_{name}")
    return flags


def _copy_mesh(
    root: Path,
    output: Path,
    destination_dir: Path,
    row: object,
    filename: str,
) -> dict[str, object] | None:
    if row is None:
        return None
    if not isinstance(row, dict):
        raise ValueError("topology mesh row must be an object")
    source = root / str(row["viewer_asset"])
    destination = destination_dir / filename
    shutil.copyfile(source, destination)
    return {
        **row,
        "viewer_asset": destination.relative_to(output).as_posix(),
    }


def _legacy_surface_qc(
    payload: dict[str, object],
    benchmark: dict[str, object],
) -> dict[str, object]:
    evidence = payload.get("pair_evidence", [])
    agreements = sorted(
        float(row["matched_label_agreement"])
        for row in evidence
        if isinstance(row, dict) and row.get("matched_label_agreement") is not None
    )
    median_agreement = float(median(agreements)) if agreements else 0.0
    components = sum(
        int(row.get("component_count", 0))
        for row in payload.get("classes", [])
        if isinstance(row, dict)
    )
    return {
        "status": "legacy",
        "selected_method": "sparse_semantic_extrusion",
        "median_tissue_dice": float(benchmark.get("flow_macro_class_dice", 0.0)),
        "median_boundary_f1": median_agreement,
        "component_count": components,
    }


_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Histopia connected topology</title>
  <link rel="icon" href="data:,">
  <link rel="stylesheet" href="topology-review.css">
</head>
<body>
  <div class="app">
    <main class="workspace">
      <header>
        <strong>Histopia connected topology</strong>
        <button id="panel-toggle" type="button" aria-expanded="false">
          Controls
        </button>
        <div class="segments" aria-label="Z-axis display scale">
          <button data-z="1">Physical</button>
          <button data-z="12" class="active">Review 12x</button>
          <button data-z="25">Strong 25x</button>
        </div>
        <div class="view-controls">
          <button data-view="home" title="Home view">Home</button>
          <button data-view="top" title="Top view">Top</button>
          <button data-view="front" title="Front view">Front</button>
          <button data-view="side" title="Side view">Side</button>
        </div>
      </header>
      <section id="viewport" aria-label="Interactive connected tissue volume">
        <output id="loading">Loading tissue envelope...</output>
      </section>
    </main>
    <aside>
      <section class="dataset">
        <label>Dataset <select id="cohort" aria-label="Cohort"></select></label>
        <output id="provenance"></output>
      </section>
      <section>
        <div id="metrics" class="metrics"></div>
        <output id="qc-status"></output>
      </section>
      <section class="controls">
        <div class="segments representation" aria-label="Semantic representation">
          <button data-surface="core" class="active">Core</button>
          <button data-surface="full">Full</button>
        </div>
        <label>Semantic class <select id="region"></select></label>
        <label>Region opacity <input id="region-opacity" type="range" min="10" max="100" value="82"></label>
        <label>Envelope opacity <input id="envelope-opacity" type="range" min="5" max="70" value="18"></label>
        <label class="check"><input id="show-region" type="checkbox" checked> Semantic region</label>
        <label class="check"><input id="show-envelope" type="checkbox" checked> Tissue envelope</label>
        <label class="check"><input id="show-uncertainty" type="checkbox"> Reconstruction uncertainty</label>
        <label class="check"><input id="show-locator" type="checkbox"> Observed section locator</label>
        <label id="section-row">Section <select id="section"></select></label>
        <label class="check"><input id="cutaway" type="checkbox"> Cutaway</label>
        <label id="cut-row">Cut position <input id="cut-position" type="range" min="-100" max="100" value="0"></label>
      </section>
      <section class="review">
        <h2>Scientific review</h2>
        <select id="review-target" aria-label="Review target"></select>
        <output id="transition-summary"></output>
        <div class="credentials">
          <input id="token" type="password" placeholder="Access key" autocomplete="off">
          <input id="reviewer" placeholder="Reviewer">
        </div>
        <div id="issues"></div>
        <label id="interval-row">Suggested intervals <input id="intervals" type="number" min="1" max="4"></label>
        <textarea id="comment" placeholder="Review comment"></textarea>
        <div class="decision">
          <button data-decision="accept">Accept</button>
          <button data-decision="hold">Hold</button>
          <button data-decision="reject">Reject</button>
        </div>
        <button id="accept-passing" type="button">
          Accept passing transitions
        </button>
        <output id="status"></output>
      </section>
    </aside>
  </div>
  <script src="manifest-data.js"></script>
  <script type="importmap">{"imports":{"three":"./vendor/three.module.min.js"}}</script>
  <script type="module" src="topology-review.js"></script>
</body>
</html>
"""


_CSS = """
:root{color-scheme:dark;font-family:Inter,system-ui,sans-serif;background:#090d12;color:#e6edf3}
*{box-sizing:border-box;letter-spacing:0}body{margin:0;height:100dvh;overflow:hidden}.app{height:100dvh;display:grid;grid-template-columns:minmax(0,1fr) clamp(400px,25vw,720px)}.workspace{min-width:0;min-height:0;display:grid;grid-template-rows:48px minmax(0,1fr)}header{display:flex;align-items:center;gap:8px;padding:0 12px;border-bottom:1px solid #30363d;background:#151a21}header strong{font-size:14px;white-space:nowrap;margin-right:auto}#panel-toggle{display:none}.view-controls{display:flex;gap:5px}button,select,input,textarea{font:inherit;color:inherit;background:#20262e;border:1px solid #48515c;border-radius:4px}button,select,input{height:30px;padding:0 8px}button{cursor:pointer}button:hover{border-color:#8c959f}.segments{display:flex}.segments button{border-radius:0;font-size:11px}.segments button:first-child{border-radius:4px 0 0 4px}.segments button:last-child{border-radius:0 4px 4px 0}.segments button.active{background:#236aa5;border-color:#58a6ff}#viewport{position:relative;min-width:0;min-height:0;overflow:hidden;background:#070a0e}#viewport canvas{display:block}#loading{position:absolute;left:16px;bottom:14px;padding:6px 9px;background:#151a21cc;border:1px solid #30363d;border-radius:4px;font-size:11px;color:#b9c3ce;z-index:2}aside{min-width:0;min-height:0;overflow:auto;border-left:1px solid #30363d;background:#0f141a}aside>section{padding:10px 12px;border-bottom:1px solid #30363d}.dataset{min-height:48px;display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.7fr);align-items:center;gap:9px;background:#151a21}.dataset label{display:flex;align-items:center;gap:7px;font-size:11px}.dataset select{flex:1;min-width:0}.dataset output{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#9da7b3;font-size:10px}.metrics{display:grid;grid-template-columns:1fr 1fr;gap:5px}.metric{background:#181e25;padding:6px;border-radius:4px;font-size:10px;color:#9da7b3}.metric b{display:block;color:#e6edf3;font-size:14px}.metric.pass b,#qc-status.pass{color:#56d364}.metric.fail b,#qc-status.fail{color:#ff7b72}#qc-status{display:block;margin-top:7px;font-size:11px}.controls{display:grid;gap:7px;font-size:11px}.controls label{display:flex;align-items:center;gap:7px}.controls label>select,.controls label>input[type=range]{flex:1;min-width:0}.controls .check{justify-content:flex-start}.representation{width:100%}.representation button{flex:1}.check input{height:auto}.review{display:grid;gap:7px}.review h2{margin:0;font-size:11px;text-transform:uppercase;color:#9da7b3}.review select,.review textarea{width:100%}.credentials{display:grid;grid-template-columns:1fr 1fr;gap:5px}.credentials.public{grid-template-columns:1fr}.credentials input{min-width:0;width:100%}.review textarea{height:58px;padding:6px;resize:none}.review>label{font-size:11px;color:#9da7b3}.review>label input{width:58px;margin-left:5px}.decision{display:grid;grid-template-columns:repeat(3,1fr);gap:5px}.decision button:first-child{border-color:#3fb950}.decision button:last-child{border-color:#f85149}#issues{display:flex;flex-wrap:wrap;gap:4px;font-size:10px}#issues label{padding:3px 5px;background:#181e25;border-radius:3px;color:#b9c3ce}#issues input{height:auto}#status{min-height:14px;font-size:11px;color:#9da7b3}@media(max-width:1280px){.app{grid-template-columns:minmax(0,1fr) 360px}.view-controls button:not([data-view=home]){display:none}.dataset{grid-template-columns:1fr}.dataset output{display:none}}@media(max-height:760px){aside>section{padding:7px 10px}.controls{gap:4px}.review{gap:4px}.review textarea{height:38px}}@media(max-width:700px){.app{display:block;position:relative}.workspace{height:100dvh}header{padding:0 8px}header strong{overflow:hidden;text-overflow:ellipsis}.segments,.view-controls{display:none}#panel-toggle{display:block;flex:0 0 auto}aside{display:none;position:absolute;inset:48px 0 0;z-index:10;border:0;border-top:1px solid #30363d}.app.panel-open aside{display:block}}
#transition-summary{font-size:10px;color:#9da7b3}#accept-passing{border-color:#3fb950}
"""


_JS = r"""
import * as THREE from "./vendor/three.module.min.js";
import {OrbitControls} from "./vendor/OrbitControls.js";
const data=globalThis.HISTOPIA_TOPOLOGY_REVIEW,el=id=>document.getElementById(id);
let authenticationRequired=true;
const scene=new THREE.Scene();scene.background=new THREE.Color(0x070a0e);
const camera=new THREE.PerspectiveCamera(38,1,.1,1e8);
const renderer=new THREE.WebGLRenderer({antialias:true,powerPreference:"high-performance"});
renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.localClippingEnabled=true;el("viewport").append(renderer.domElement);
let contextLost=false;renderer.domElement.style.visibility="hidden";renderer.domElement.addEventListener("webglcontextlost",()=>{contextLost=true;renderer.domElement.style.visibility="hidden";el("loading").hidden=false;el("loading").textContent="Restoring 3D renderer..."});renderer.domElement.addEventListener("webglcontextrestored",()=>{contextLost=false;fit("home")});
const controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;controls.screenSpacePanning=true;
const app=document.querySelector(".app"),panelToggle=el("panel-toggle");
panelToggle.onclick=()=>{const open=app.classList.toggle("panel-open");panelToggle.setAttribute("aria-expanded",String(open));panelToggle.textContent=open?"Close":"Controls"};
scene.add(new THREE.HemisphereLight(0xffffff,0x23303c,2.5));const key=new THREE.DirectionalLight(0xffffff,2.4);key.position.set(1,2,2);scene.add(key);const fill=new THREE.DirectionalLight(0x9fc5e8,1.1);fill.position.set(-2,.5,-1);scene.add(fill);
const root=new THREE.Group();scene.add(root);const clipping=new THREE.Plane(new THREE.Vector3(1,0,0),1e9);
let current=null,generation=0,physicalCenter=new THREE.Vector3(),physicalSize=new THREE.Vector3(),envelope=null,uncertainty=null,locator=null,activeRegions=[],surfaceMode="core",zScale=12;
const regionCache=new Map();
function resize(){const rect=el("viewport").getBoundingClientRect();renderer.setSize(rect.width,rect.height,false);camera.aspect=Math.max(rect.width,1)/Math.max(rect.height,1);camera.updateProjectionMatrix()}addEventListener("resize",()=>{resize();fit("home")});
function disposeObject(object){if(!object)return;root.remove(object);object.geometry?.dispose();object.material?.dispose()}
function clear(){for(const item of [...root.children])disposeObject(item);regionCache.clear();envelope=uncertainty=locator=null;activeRegions=[]}
async function binaryMesh(row,role,loadId){
 const response=await fetch(row.viewer_asset);if(!response.ok)throw Error(`Could not load ${role}`);const buffer=await response.arrayBuffer();if(loadId!==generation)return null;
 const view=new DataView(buffer);if(String.fromCharCode(...new Uint8Array(buffer,0,4))!=="HTM1")throw Error(`Invalid ${role} mesh`);
 const vertices=view.getUint32(4,true),faces=view.getUint32(8,true),raw=new Float32Array(buffer,12,vertices*3),positions=new Float32Array(raw.length);
 if(role==="envelope"){const box=new THREE.Box3();for(let i=0;i<vertices;i++)box.expandByPoint(new THREE.Vector3(raw[3*i],raw[3*i+2],-raw[3*i+1]));box.getCenter(physicalCenter);box.getSize(physicalSize)}
 for(let i=0;i<vertices;i++){positions[3*i]=raw[3*i]-physicalCenter.x;positions[3*i+1]=raw[3*i+2]-physicalCenter.y;positions[3*i+2]=-raw[3*i+1]-physicalCenter.z}
 const indices=new Uint32Array(buffer,12+vertices*12,faces*3),geometry=new THREE.BufferGeometry();geometry.setAttribute("position",new THREE.BufferAttribute(positions,3));geometry.setIndex(new THREE.BufferAttribute(indices,1));geometry.computeVertexNormals();
 let material;if(role==="envelope")material=new THREE.MeshPhysicalMaterial({color:0xd7dde5,roughness:.82,metalness:0,transparent:true,opacity:+el("envelope-opacity").value/100,side:THREE.FrontSide,depthWrite:false,clippingPlanes:[clipping]});
 else if(role==="uncertainty")material=new THREE.MeshStandardMaterial({color:0xffb454,roughness:.7,transparent:true,opacity:.48,side:THREE.FrontSide,depthWrite:false,clippingPlanes:[clipping]});
 else material=new THREE.MeshStandardMaterial({color:row.color,roughness:.7,metalness:.01,transparent:true,opacity:+el("region-opacity").value/100,side:THREE.FrontSide,clippingPlanes:[clipping]});
 const mesh=new THREE.Mesh(geometry,material);mesh.renderOrder=role==="envelope"?2:role==="uncertainty"?3:1;mesh.userData={role,row};root.add(mesh);return mesh
}
async function selectCohort(id){
 const loadId=++generation;current=data.cohorts.find(row=>row.id===id);clear();el("loading").hidden=false;el("loading").textContent="Loading tissue envelope...";
 el("provenance").textContent=`${current.observed_section_count} observed | ${current.numerical_sample_count} numerical samples | z: ${current.z_source}`;
 const qc=current.reconstruction_qc,chosen=qc.candidates?.find(row=>row.method===qc.selected_method)||qc;
 el("metrics").innerHTML=`<div class="metric"><b>${current.observed_section_count}</b>observed sections</div><div class="metric"><b>${current.numerical_sample_count}</b>numerical samples</div><div class="metric"><b>${pct(chosen.median_tissue_dice||0)}</b>held-out tissue Dice</div><div class="metric ${qc.status==="passed"?"pass":"fail"}"><b>${qc.status}</b>envelope QC</div>`;
 el("qc-status").className=qc.status==="passed"?"pass":"fail";el("qc-status").textContent=current.legacy_diagnostic?"Legacy sparse diagnostic surface":`${qc.selected_method} | boundary F1 ${pct(chosen.median_boundary_f1||0)}`;
 updateRegionOptions();
 el("section").innerHTML=current.observed_sections.map((row,index)=>`<option value="${index}">${String(index+1).padStart(2,"0")} ${escapeHtml(row.slide_id||"Section")}</option>`).join("");
 const flagged=new Set(current.transition_qc.filter(row=>row.outlier).map(row=>row.id));el("review-target").innerHTML=`<option value="volume">Connected volume</option>`+current.gap_decisions.map((row,index)=>{const id=`${String(row.source_section).padStart(3,"0")}-${String(row.target_section).padStart(3,"0")}`;return `<option value="${index}">${flagged.has(id)?"Review · ":""}Transition ${row.source_section+1} to ${row.target_section+1}</option>`}).join("");el("transition-summary").textContent=`${flagged.size} flagged · ${current.gap_decisions.length-flagged.size} passing transitions`;
 updateReviewTarget();try{envelope=await binaryMesh(current.envelope,"envelope",loadId);if(!envelope)return;root.scale.y=zScale;await selectRegion(el("region").value,loadId);if(loadId!==generation)return;updateCutaway();updateLocator();fit("home");await settleRenderer(loadId)}catch(error){el("loading").textContent=error.message}
}
async function settleRenderer(loadId){let stableFrames=0;while(stableFrames<4){await new Promise(resolve=>setTimeout(resolve,120));if(loadId!==generation)return;if(contextLost){stableFrames=0;continue}renderer.render(scene,camera);stableFrames++}renderer.domElement.style.visibility="visible";el("loading").hidden=true}
function surfaceRows(){return (surfaceMode==="full"?current.semantic_partition_regions:current.semantic_regions).filter(Boolean)}
function updateRegionOptions(){const rows=surfaceRows(),metric=surfaceMode==="full"?"viewer_partition_volume_fraction_of_tissue":"viewer_core_volume_fraction_of_tissue";el("region").innerHTML=`<option value="all">All classes</option>`+rows.map((row,index)=>`<option value="${index}">Class ${row.class_index??index} · ${pct(row[metric]||0)}</option>`).join("");el("region").value=surfaceMode==="full"?"all":String(Math.min(current.default_region_index,Math.max(rows.length-1,0)))}
async function selectRegion(value,loadId=generation){for(const mesh of activeRegions)mesh.visible=false;activeRegions=[];const rows=surfaceRows(),indices=value==="all"?rows.map((_,index)=>index):[+value];for(const index of indices){const row=rows[index];if(!row)continue;const key=`${surfaceMode}:${index}`;let mesh=regionCache.get(key);if(!mesh){el("loading").hidden=false;el("loading").textContent=`Loading ${surfaceMode} class ${row.class_index??index}...`;mesh=await binaryMesh(row,"region",loadId);if(!mesh)return;regionCache.set(key,mesh)}mesh.visible=el("show-region").checked;activeRegions.push(mesh)}}
async function toggleUncertainty(){if(!current.uncertainty){el("show-uncertainty").checked=false;return}if(!uncertainty)uncertainty=await binaryMesh(current.uncertainty,"uncertainty",generation);if(uncertainty)uncertainty.visible=el("show-uncertainty").checked}
function updateLocator(){if(locator){root.remove(locator);locator.geometry.dispose();locator.material.dispose();locator=null}el("section-row").style.display=el("show-locator").checked?"flex":"none";if(!el("show-locator").checked||!current.observed_sections.length)return;const row=current.observed_sections[+el("section").value||0],x=physicalSize.x/2,z=physicalSize.z/2,y=row.z_um-physicalCenter.y,points=[-x,y,-z,x,y,-z,x,y,z,-x,y,z,-x,y,-z],geometry=new THREE.BufferGeometry().setFromPoints(Array.from({length:5},(_,i)=>new THREE.Vector3(points[3*i],points[3*i+1],points[3*i+2]))),material=new THREE.LineBasicMaterial({color:0xffffff,transparent:true,opacity:.85,depthTest:false});locator=new THREE.Line(geometry,material);locator.renderOrder=5;root.add(locator)}
function updateCutaway(){const enabled=el("cutaway").checked;el("cut-row").style.display=enabled?"flex":"none";clipping.constant=enabled?-(+el("cut-position").value/100)*(physicalSize.x/2):1e9;for(const mesh of root.children)if(mesh.material)mesh.material.needsUpdate=true}
function projectedGeometryBounds(object){root.updateMatrixWorld(true);camera.updateMatrixWorld(true);const bounds={minX:Infinity,maxX:-Infinity,minY:Infinity,maxY:-Infinity,minXDepth:1,maxXDepth:1,minYDepth:1,maxYDepth:1},world=new THREE.Vector3(),projected=new THREE.Vector3(),cameraPoint=new THREE.Vector3(),position=object?.geometry?.attributes.position;if(!position)return bounds;for(let index=0;index<position.count;index++){world.fromBufferAttribute(position,index).applyMatrix4(object.matrixWorld);cameraPoint.copy(world).applyMatrix4(camera.matrixWorldInverse);const depth=Math.max(-cameraPoint.z,.001);projected.copy(world).project(camera);if(projected.x<bounds.minX){bounds.minX=projected.x;bounds.minXDepth=depth}if(projected.x>bounds.maxX){bounds.maxX=projected.x;bounds.maxXDepth=depth}if(projected.y<bounds.minY){bounds.minY=projected.y;bounds.minYDepth=depth}if(projected.y>bounds.maxY){bounds.maxY=projected.y;bounds.maxYDepth=depth}}return bounds}
function recenterProjectedGeometry(){for(let iteration=0;iteration<2;iteration++){const bounds=projectedGeometryBounds(envelope);if(!Number.isFinite(bounds.minX))return;const ndcX=(bounds.minX+bounds.maxX)/2,ndcY=(bounds.minY+bounds.maxY)/2;if(Math.max(Math.abs(ndcX),Math.abs(ndcY))<.001)return;const vertical=Math.tan(THREE.MathUtils.degToRad(camera.fov)/2),horizontal=vertical*camera.aspect,xDepth=2/(1/bounds.minXDepth+1/bounds.maxXDepth),yDepth=2/(1/bounds.minYDepth+1/bounds.maxYDepth),right=new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld,0),up=new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld,1),shift=right.multiplyScalar(ndcX*horizontal*xDepth).add(up.multiplyScalar(ndcY*vertical*yDepth));camera.position.add(shift);controls.target.add(shift);controls.update()}}
function fit(view="home"){if(!envelope)return;root.updateMatrixWorld(true);const box=new THREE.Box3().setFromObject(envelope,true);if(box.isEmpty())return;const sphere=box.getBoundingSphere(new THREE.Sphere()),radius=Math.max(sphere.radius,1),vertical=THREE.MathUtils.degToRad(camera.fov),horizontal=2*Math.atan(Math.tan(vertical/2)*camera.aspect),distance=.82*radius/Math.sin(Math.min(vertical,horizontal)/2);let direction;if(view==="top")direction=new THREE.Vector3(0,1,.001);else if(view==="front")direction=new THREE.Vector3(0,.05,1);else if(view==="side")direction=new THREE.Vector3(1,.05,0);else direction=new THREE.Vector3(.85,.65,1).normalize();controls.target.copy(sphere.center);camera.position.copy(sphere.center).addScaledVector(direction,distance);camera.near=Math.max(radius/1000,.01);camera.far=distance+radius*8;camera.updateProjectionMatrix();controls.update();recenterProjectedGeometry()}
function updateReviewTarget(){const value=el("review-target").value,isVolume=value==="volume";el("interval-row").style.display=isVolume?"none":"block";if(!isVolume)el("intervals").value=current.gap_decisions[+value].intervals}
function pct(value){return `${(100*value).toFixed(1)}%`}function escapeHtml(value){const node=document.createElement("span");node.textContent=value;return node.innerHTML}
el("cohort").innerHTML=data.cohorts.map(row=>`<option>${row.id}</option>`).join("");el("cohort").onchange=()=>selectCohort(el("cohort").value);
document.querySelectorAll("[data-z]").forEach(button=>button.onclick=()=>{zScale=+button.dataset.z;root.scale.y=zScale;document.querySelectorAll("[data-z]").forEach(item=>item.classList.toggle("active",item===button));updateLocator();fit("home")});
document.querySelectorAll("[data-view]").forEach(button=>button.onclick=()=>fit(button.dataset.view));
document.querySelectorAll("[data-surface]").forEach(button=>button.onclick=async()=>{surfaceMode=button.dataset.surface;document.querySelectorAll("[data-surface]").forEach(item=>item.classList.toggle("active",item===button));updateRegionOptions();const loadId=generation;await selectRegion(el("region").value,loadId);fit("home");await settleRenderer(loadId)});
el("region").onchange=async()=>{const loadId=generation;await selectRegion(el("region").value,loadId);fit("home");await settleRenderer(loadId)};el("region-opacity").oninput=()=>{for(const mesh of activeRegions)mesh.material.opacity=+el("region-opacity").value/100};el("envelope-opacity").oninput=()=>{if(envelope)envelope.material.opacity=+el("envelope-opacity").value/100};el("show-region").onchange=()=>{for(const mesh of activeRegions)mesh.visible=el("show-region").checked};el("show-envelope").onchange=()=>{if(envelope)envelope.visible=el("show-envelope").checked};el("show-uncertainty").onchange=toggleUncertainty;el("show-locator").onchange=updateLocator;el("section").onchange=updateLocator;el("cutaway").onchange=updateCutaway;el("cut-position").oninput=updateCutaway;el("review-target").onchange=updateReviewTarget;
const labels=["envelope_shape","missing_component","spurious_component","semantic_discontinuity","excess_uncertainty","wrong_z_spacing","framing","performance","other"];el("issues").innerHTML=labels.map(value=>`<label><input type="checkbox" value="${value}">${value.replaceAll("_"," ")}</label>`).join("");
document.querySelectorAll("[data-decision]").forEach(button=>button.onclick=()=>save(button.dataset.decision));
function reviewHeaders(){const headers={"Content-Type":"application/json"};if(authenticationRequired)headers.Authorization=`Bearer ${el("token").value}`;return headers}
async function configureReviewAccess(){if(location.protocol==="file:")return;const response=await fetch("/api/reviews/access",{cache:"no-store"}),payload=await response.json();if(!response.ok)throw Error(payload.error||"Review service unavailable");if(!payload.review_configured)return;authenticationRequired=Boolean(payload.authentication_required);el("token").hidden=!authenticationRequired;el("token").parentElement.classList.toggle("public",!authenticationRequired)}
async function postFeedback(body){const response=await fetch("/api/reviews/feedback",{method:"POST",headers:reviewHeaders(),body:JSON.stringify(body)}),payload=await response.json();if(!response.ok)throw Error(payload.error||"Review save failed");return payload}
async function save(decision){const target=el("review-target").value,isVolume=target==="volume",row=isVolume?null:current.gap_decisions[+target],body={cohort:current.id,stage:"topology",fingerprint:current.fingerprint,slide_id:isVolume?"volume":`${String(row.source_section).padStart(3,"0")}-${String(row.target_section).padStart(3,"0")}`,decision,labels:[...el("issues").querySelectorAll("input:checked")].map(input=>input.value),comment:el("comment").value,reviewer:el("reviewer").value};if(!isVolume)body.suggested_intervals=+el("intervals").value;if(decision==="accept")body.labels=[];el("status").textContent="Saving...";try{await postFeedback(body);el("status").textContent=`Saved ${decision}`}catch(error){el("status").textContent=error.message}}
el("accept-passing").onclick=async()=>{const reviewer=el("reviewer").value.trim(),passing=new Set(current.transition_qc.filter(row=>!row.outlier).map(row=>row.id));if(!reviewer){el("status").textContent="Reviewer is required";return}if(!confirm(`Accept ${passing.size} quantitatively passing transitions for ${current.id}?`))return;el("status").textContent=`Saving 0 of ${passing.size}...`;let saved=0;try{for(const row of current.gap_decisions){const id=`${String(row.source_section).padStart(3,"0")}-${String(row.target_section).padStart(3,"0")}`;if(!passing.has(id))continue;await postFeedback({cohort:current.id,stage:"topology",fingerprint:current.fingerprint,slide_id:id,decision:"accept",labels:[],comment:"Quantitative transition gates passed; accepted with cohort volume review.",reviewer,suggested_intervals:row.intervals});saved++;el("status").textContent=`Saving ${saved} of ${passing.size}...`}el("status").textContent=`Accepted ${saved} passing transitions`}catch(error){el("status").textContent=`Saved ${saved}; ${error.message}`}}
function animate(){requestAnimationFrame(animate);controls.update();renderer.render(scene,camera)}resize();el("section-row").style.display="none";el("cut-row").style.display="none";configureReviewAccess().catch(error=>{el("status").textContent=error.message});selectCohort(data.cohorts[0].id);animate();
"""
