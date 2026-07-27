"""Static, fixed-viewport review application for topology surfaces."""

# ruff: noqa: E501

from __future__ import annotations

import json
import shutil
import struct
from pathlib import Path
from statistics import median

from histopia.topology import validate_topology_result


def build_topology_review(
    runs: dict[str, Path | str],
    output_dir: Path | str,
) -> Path:
    """Build a multi-cohort topology surface and transition reviewer."""

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
        mesh_rows = []
        plane_rows = []
        cohort_assets = assets / cohort
        cohort_assets.mkdir(parents=True, exist_ok=True)
        for index, row in enumerate(payload["planes"]):
            source = root / str(row["artifact"])
            destination = cohort_assets / f"plane-{index:03d}.bin"
            shape_rc = _write_plane_binary(source, destination)
            plane_rows.append(
                {
                    **row,
                    "viewer_asset": destination.relative_to(output).as_posix(),
                    "shape_rc": list(shape_rc),
                }
            )
        for index, row in enumerate(payload["meshes"]):
            source = root / str(row["viewer_asset"])
            destination = cohort_assets / f"mesh-{index:03d}.bin"
            shutil.copyfile(source, destination)
            mesh_rows.append(
                {
                    **row,
                    "viewer_asset": destination.relative_to(output).as_posix(),
                }
            )
        cohorts.append(
            {
                "id": cohort,
                "fingerprint": payload["fingerprint"],
                "selected_k": payload["selected_k"],
                "palette": payload["palette"],
                "z_source": payload["z_source"],
                "section_thickness_um": payload["section_thickness_um"],
                "reference_grid": payload["reference_grid"],
                "observed_section_count": payload["observed_section_count"],
                "virtual_section_count": payload["virtual_section_count"],
                "segment_count": payload["segment_count"],
                "gap_decisions": payload["gap_decisions"],
                "pair_evidence": payload["pair_evidence"],
                "planes": plane_rows,
                "meshes": mesh_rows,
                "classes": payload["classes"],
                "benchmark": benchmark["summary"],
                "surface_qc": _surface_qc(payload, benchmark["summary"]),
            }
        )
    manifest = {"schema_version": 2, "cohorts": cohorts}
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


def _write_plane_binary(source: Path, destination: Path) -> tuple[int, int]:
    """Write compact labels and uncertainty for a browser section texture."""

    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "topology review generation requires the 'topology' extra"
        ) from exc
    with np.load(source, allow_pickle=False) as archive:
        labels = np.asarray(archive["labels"])
        uncertainty = np.asarray(archive["uncertainty"], dtype=np.float32)
    if labels.ndim != 2 or uncertainty.shape != labels.shape:
        raise ValueError(f"invalid topology plane arrays: {source}")
    if np.any((labels < -1) | (labels > 254)):
        raise ValueError(f"topology plane labels exceed browser format: {source}")
    encoded_labels = np.where(labels < 0, 255, labels).astype(np.uint8)
    encoded_uncertainty = np.rint(np.clip(uncertainty, 0.0, 1.0) * 255.0).astype(
        np.uint8
    )
    rows, columns = labels.shape
    destination.write_bytes(
        struct.pack("<4sII", b"HTP1", columns, rows)
        + encoded_labels.tobytes(order="C")
        + encoded_uncertainty.tobytes(order="C")
    )
    return int(rows), int(columns)


def _surface_qc(
    payload: dict[str, object],
    benchmark: dict[str, object],
) -> dict[str, object]:
    """Summarize whether sparse class surfaces are suitable for display."""

    evidence = payload.get("pair_evidence", [])
    agreements = sorted(
        float(row["matched_label_agreement"])
        for row in evidence
        if isinstance(row, dict) and row.get("matched_label_agreement") is not None
    )
    median_agreement = float(median(agreements)) if agreements else 0.0
    flow_dice = float(benchmark.get("flow_macro_class_dice", 0.0))
    components = sum(
        int(row.get("component_count", 0))
        for row in payload.get("classes", [])
        if isinstance(row, dict)
    )
    reasons = []
    if flow_dice < 0.75:
        reasons.append("heldout_semantic_dice_below_0.75")
    if median_agreement < 0.75:
        reasons.append("median_adjacent_agreement_below_0.75")
    if components > 500:
        reasons.append("excess_surface_fragmentation")
    return {
        "status": "passed" if not reasons else "failed",
        "reasons": reasons,
        "median_adjacent_agreement": median_agreement,
        "component_count": components,
    }


_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Histopia topology review</title>
  <link rel="icon" href="data:,">
  <link rel="stylesheet" href="topology-review.css">
</head>
<body>
  <header>
    <strong>Histopia topology</strong>
    <select id="cohort" aria-label="Cohort"></select>
    <span id="provenance"></span>
    <div class="mode" role="group" aria-label="Rendering mode">
      <button class="active" data-mode="sections">Sections</button>
      <button data-mode="surfaces">Diagnostic surface</button>
    </div>
    <button id="fit" title="Fit reconstruction to view">Fit</button>
    <button id="reset" title="Show every section">All</button>
  </header>
  <main>
    <section id="viewport" aria-label="Interactive semantic topology"></section>
    <aside>
      <section class="controls">
        <div id="metrics"></div>
        <output id="surface-status"></output>
        <label>Display spacing <input id="z-scale" type="range" min="1" max="100" value="25"></label>
        <label>Section opacity <input id="opacity" type="range" min="20" max="100" value="82"></label>
        <label><input id="observed" type="checkbox" checked> Observed sections</label>
        <label><input id="virtual" type="checkbox" checked> Inferred sections</label>
        <div id="classes"></div>
      </section>
      <section class="pairs">
        <h2>Section transitions</h2>
        <div id="pair-list"></div>
      </section>
      <section class="review">
        <h2 id="pair-title">Select a transition</h2>
        <input id="token" type="password" placeholder="Review access key" autocomplete="off">
        <input id="reviewer" placeholder="Reviewer">
        <div id="issues"></div>
        <label>Suggested intervals <input id="intervals" type="number" min="1" max="4"></label>
        <textarea id="comment" placeholder="Review comment"></textarea>
        <div class="decision">
          <button data-decision="accept">Accept</button>
          <button data-decision="hold">Hold</button>
          <button data-decision="reject">Reject</button>
        </div>
        <output id="status"></output>
      </section>
    </aside>
  </main>
  <script src="manifest-data.js"></script>
  <script type="importmap">
    {"imports":{"three":"./vendor/three.module.min.js"}}
  </script>
  <script type="module" src="topology-review.js"></script>
</body>
</html>
"""

_CSS = """
:root{color-scheme:dark;font-family:Inter,system-ui,sans-serif;background:#0d1117;color:#e6edf3}
*{box-sizing:border-box}body{margin:0;height:100dvh;overflow:hidden}header{height:48px;display:flex;align-items:center;gap:10px;padding:0 14px;border-bottom:1px solid #30363d;background:#161b22}header strong{font-size:15px}header span{color:#9da7b3;font-size:12px;flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}button,select,input,textarea{font:inherit;color:inherit;background:#21262d;border:1px solid #484f58;border-radius:4px}button,select,input{height:30px;padding:0 9px}button{cursor:pointer}button:hover{border-color:#8c959f}.mode{display:flex}.mode button{border-radius:0}.mode button:first-child{border-radius:4px 0 0 4px}.mode button:last-child{border-radius:0 4px 4px 0}.mode button.active{background:#1f6feb;border-color:#58a6ff}main{height:calc(100dvh - 48px);display:grid;grid-template-columns:minmax(0,1fr) 350px}#viewport{min-width:0;position:relative;background:#080b10}#viewport canvas{display:block;background:#080b10}aside{min-width:0;overflow:hidden;border-left:1px solid #30363d;display:grid;grid-template-rows:auto minmax(100px,1fr) auto;min-height:0;background:#0d1117}aside>section{min-width:0;padding:10px 12px;border-bottom:1px solid #30363d}.controls{font-size:12px;display:grid;gap:7px}.controls label{display:flex;align-items:center;gap:7px}.controls input[type=range]{padding:0;flex:1}.controls input[type=checkbox],#issues input{height:auto}.metric-grid{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:5px}.metric{min-width:0;background:#161b22;padding:6px;border-radius:4px}.metric b{display:block;font-size:14px}.metric.fail b,#surface-status.fail{color:#ff7b72}.metric.pass b,#surface-status.pass{color:#56d364}#surface-status{font-size:11px;min-height:14px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.class-row{display:flex;align-items:center;gap:7px;margin-top:5px}.swatch{width:11px;height:11px;border-radius:2px}.pairs{min-width:0;min-height:0;display:flex;flex-direction:column}.pairs h2,.review h2{font-size:12px;text-transform:uppercase;color:#9da7b3;margin:0 0 7px}#pair-list{overflow:auto;min-width:0;min-height:0}.pair{min-width:0;width:100%;height:auto;text-align:left;padding:7px;margin-bottom:4px;background:#161b22}.pair.selected{border-color:#58a6ff;background:#17243a}.pair span{display:flex;justify-content:space-between;font-size:11px}.pair small{display:block;max-width:100%;color:#9da7b3;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.review{min-width:0;display:grid;gap:6px}.review input,.review textarea{min-width:0;width:100%}.review textarea{height:54px;padding:6px;resize:none}.review label{font-size:11px;color:#9da7b3}.review label input{width:58px;margin-left:5px}.decision{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:5px}.decision button:first-child{border-color:#3fb950}.decision button:last-child{border-color:#f85149}#issues{display:flex;flex-wrap:wrap;gap:4px;font-size:10px}#issues label{padding:3px 5px;background:#161b22;border-radius:3px}#status{font-size:11px;min-height:15px;color:#9da7b3}@media(max-width:1050px){header span{display:none}main{grid-template-columns:minmax(0,1fr) 310px}.mode button{font-size:11px;padding:0 6px}}
"""

_JS = r"""
import * as THREE from "./vendor/three.module.min.js";
import {OrbitControls} from "./vendor/OrbitControls.js";
const data=globalThis.HISTOPIA_TOPOLOGY_REVIEW;
const el=id=>document.getElementById(id);
const scene=new THREE.Scene();scene.background=new THREE.Color(0x080b10);
const camera=new THREE.PerspectiveCamera(42,1,0.1,1e7);
const renderer=new THREE.WebGLRenderer({antialias:true,powerPreference:"high-performance"});
renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.outputColorSpace=THREE.SRGBColorSpace;el("viewport").append(renderer.domElement);
const controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;
scene.add(new THREE.HemisphereLight(0xffffff,0x263238,2.2));
const light=new THREE.DirectionalLight(0xffffff,2);light.position.set(1,2,2);scene.add(light);
const sections=new THREE.Group(),surfaces=new THREE.Group();scene.add(sections);scene.add(surfaces);
let current=null,selectedPair=null,loadGeneration=0,mode="sections",surfacesLoaded=false,paletteRgb=[];
let enabledClasses=new Set();
function resize(){const r=el("viewport").getBoundingClientRect();renderer.setSize(r.width,r.height,false);camera.aspect=r.width/r.height;camera.updateProjectionMatrix()}addEventListener("resize",resize);
function dispose(root){while(root.children.length){const item=root.children[0];root.remove(item);item.material?.map?.dispose();item.geometry?.dispose();item.material?.dispose()}}
function clear(){dispose(sections);dispose(surfaces);surfacesLoaded=false}
function colorBytes(value){const color=new THREE.Color(value);return [Math.round(255*color.r),Math.round(255*color.g),Math.round(255*color.b)]}
function sectionTexture(item){
 const pixels=new Uint8Array(item.labels.length*4),opacity=+el("opacity").value/100;
 for(let i=0;i<item.labels.length;i++){const label=item.labels[i],offset=i*4;if(label===255||!enabledClasses.has(label)){pixels[offset+3]=0;continue}const rgb=paletteRgb[label];pixels[offset]=rgb[0];pixels[offset+1]=rgb[1];pixels[offset+2]=rgb[2];const confidence=1-item.uncertainty[i]/255;pixels[offset+3]=Math.round(255*opacity*(item.row.observed?1:.55+.45*confidence))}
 const texture=new THREE.DataTexture(pixels,item.width,item.height,THREE.RGBAFormat);texture.colorSpace=THREE.SRGBColorSpace;texture.magFilter=THREE.NearestFilter;texture.minFilter=THREE.NearestFilter;texture.flipY=true;texture.needsUpdate=true;return texture
}
async function loadSection(row,generation){
 const buffer=await fetch(row.viewer_asset).then(r=>{if(!r.ok)throw Error("Section load failed");return r.arrayBuffer()});
 if(String.fromCharCode(...new Uint8Array(buffer,0,4))!=="HTP1")throw Error("Invalid section texture");
 const view=new DataView(buffer),width=view.getUint32(4,true),height=view.getUint32(8,true),count=width*height;
 const item={row,width,height,labels:new Uint8Array(buffer,12,count),uncertainty:new Uint8Array(buffer,12+count,count)};
 const grid=current.reference_grid,w=grid.shape_rc[1]*grid.spacing_um,h=grid.shape_rc[0]*grid.spacing_um;
 const geometry=new THREE.PlaneGeometry(w,h),material=new THREE.MeshBasicMaterial({map:sectionTexture(item),transparent:true,side:THREE.DoubleSide,depthWrite:false,alphaTest:.02});
 const plane=new THREE.Mesh(geometry,material);plane.rotation.x=-Math.PI/2;plane.position.set(grid.origin_um_xy[0]+w/2,0,-grid.origin_um_xy[1]-h/2);plane.userData=item;
 if(generation!==loadGeneration){geometry.dispose();material.map.dispose();material.dispose();return}sections.add(plane)
}
async function loadMesh(row,generation){
 const buffer=await fetch(row.viewer_asset).then(r=>{if(!r.ok)throw Error("Mesh load failed");return r.arrayBuffer()});
 const view=new DataView(buffer);if(String.fromCharCode(...new Uint8Array(buffer,0,4))!=="HTM1")throw Error("Invalid mesh");
 const nv=view.getUint32(4,true),nf=view.getUint32(8,true),raw=new Float32Array(buffer,12,nv*3),pos=new Float32Array(raw.length);
 for(let i=0;i<nv;i++){pos[3*i]=raw[3*i];pos[3*i+1]=raw[3*i+2];pos[3*i+2]=-raw[3*i+1]}
 const index=new Uint32Array(buffer,12+nv*12,nf*3),geometry=new THREE.BufferGeometry();
 geometry.setAttribute("position",new THREE.BufferAttribute(pos,3));geometry.setIndex(new THREE.BufferAttribute(index,1));geometry.computeVertexNormals();
 const material=new THREE.MeshStandardMaterial({color:row.color,roughness:.72,metalness:.02,transparent:true,opacity:.72,side:THREE.DoubleSide});
 const mesh=new THREE.Mesh(geometry,material);mesh.userData={classIndex:row.class_index};if(generation!==loadGeneration){geometry.dispose();material.dispose();return}surfaces.add(mesh)
}
async function ensureSurfaces(){
 if(surfacesLoaded)return;const generation=loadGeneration;el("surface-status").textContent="Loading diagnostic surfaces...";
 await Promise.all(current.meshes.map(row=>loadMesh(row,generation)));if(generation!==loadGeneration)return;surfacesLoaded=true;applySpacing();applyClasses();showSurfaceStatus()
}
function fit(){
 const root=mode==="sections"?sections:surfaces,box=new THREE.Box3().setFromObject(root);if(box.isEmpty())return;const center=box.getCenter(new THREE.Vector3()),size=box.getSize(new THREE.Vector3()),radius=Math.max(size.x,size.y,size.z);controls.target.copy(center);camera.position.set(center.x+radius*.82,center.y+radius*.72,center.z+radius*1.18);camera.near=Math.max(radius/10000,.1);camera.far=radius*20;camera.updateProjectionMatrix();controls.update();renderer.render(scene,camera)
}
async function selectCohort(id){
 const generation=++loadGeneration;current=data.cohorts.find(c=>c.id===id);selectedPair=null;clear();el("provenance").textContent=`${current.observed_section_count} observed | ${current.virtual_section_count} inferred | z: ${current.z_source} | loading`;
 el("metrics").innerHTML=`<div class="metric-grid"><div class="metric"><b>${pct(current.benchmark.flow_macro_class_dice)}</b>held-out Dice</div><div class="metric"><b>${pct(current.surface_qc.median_adjacent_agreement)}</b>adjacent agreement</div><div class="metric"><b>${signed(current.benchmark.flow_gain_over_zero)}</b>gain vs zero</div><div class="metric ${current.surface_qc.status==="passed"?"pass":"fail"}"><b>${current.surface_qc.status}</b>surface QC</div></div>`;
 showSurfaceStatus();enabledClasses=new Set(current.classes.map(row=>row.class_index));paletteRgb=current.palette.map(colorBytes);
 el("classes").innerHTML=current.classes.map(r=>`<label class="class-row"><input type="checkbox" checked data-class="${r.class_index}"><span class="swatch" style="background:${r.color}"></span>Class ${r.class_index} | ${r.estimated_volume_mm3.toFixed(3)} mm3</label>`).join("");
 el("classes").querySelectorAll("input").forEach(x=>x.onchange=()=>{x.checked?enabledClasses.add(+x.dataset.class):enabledClasses.delete(+x.dataset.class);applyClasses()});
 el("pair-list").innerHTML=current.gap_decisions.map((r,i)=>`<button class="pair" data-i="${i}"><span><b>${String(r.source_section+1).padStart(2,"0")} to ${String(r.target_section+1).padStart(2,"0")}</b><em>${r.status}</em></span><small>${sectionName(r.source_section)} → ${sectionName(r.target_section)}</small><small>${r.intervals} interval${r.intervals===1?"":"s"} | confidence ${pct(r.confidence)}</small></button>`).join("");
 el("pair-list").querySelectorAll(".pair").forEach(x=>x.onclick=()=>choosePair(+x.dataset.i));
 await Promise.all(current.planes.map(row=>loadSection(row,generation)));if(generation!==loadGeneration)return;applySpacing();applyVisibility();setMode("sections");fit();await new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)));if(generation!==loadGeneration)return;el("provenance").textContent=el("provenance").textContent.replace("loading","ready")
}
function sectionName(index){return current.planes.find(row=>row.observed&&row.source_section===index)?.slide_id||`Section ${index+1}`}
function choosePair(i){selectedPair=i;el("pair-list").querySelectorAll(".pair").forEach((x,j)=>x.classList.toggle("selected",i===j));const r=current.gap_decisions[i];el("pair-title").textContent=`Transition ${r.source_section+1} to ${r.target_section+1}`;el("intervals").value=r.intervals;el("status").textContent=(r.reasons||[]).join(", ")||"Ready for review";applyVisibility();if(mode==="sections")fit()}
function applySpacing(){if(!current)return;const scale=+el("z-scale").value,z0=Math.min(...current.planes.map(row=>row.z_um));for(const plane of sections.children)plane.position.y=(plane.userData.row.z_um-z0)*scale;surfaces.scale.y=scale}
function applyClasses(){for(const plane of sections.children){plane.material.map.dispose();plane.material.map=sectionTexture(plane.userData);plane.material.needsUpdate=true}for(const mesh of surfaces.children)mesh.visible=enabledClasses.has(mesh.userData.classIndex)}
function applyVisibility(){if(!current)return;const observed=el("observed").checked,inferred=el("virtual").checked,pair=selectedPair===null?null:current.gap_decisions[selectedPair];for(const plane of sections.children){const row=plane.userData.row,kind=row.observed?observed:inferred,inPair=!pair||(row.observed?(row.source_section===pair.source_section||row.source_section===pair.target_section):(row.source_section===pair.source_section&&row.target_section===pair.target_section));plane.visible=kind&&inPair}}
function showSurfaceStatus(){if(!current)return;const output=el("surface-status"),failed=current.surface_qc.status!=="passed";output.className=failed?"fail":"pass";output.textContent=failed?`Diagnostic surfaces failed continuity QC (${current.surface_qc.component_count.toLocaleString()} components)`:"Surface continuity QC passed"}
async function setMode(next){mode=next;document.querySelectorAll("[data-mode]").forEach(button=>button.classList.toggle("active",button.dataset.mode===mode));sections.visible=mode==="sections";surfaces.visible=mode==="surfaces";if(mode==="surfaces")await ensureSurfaces();fit()}
function pct(x){return `${(100*x).toFixed(1)}%`}function signed(x){return `${x>=0?"+":""}${(100*x).toFixed(1)} pp`}
el("z-scale").oninput=applySpacing;el("opacity").oninput=applyClasses;el("observed").onchange=applyVisibility;el("virtual").onchange=applyVisibility;el("fit").onclick=fit;el("reset").onclick=()=>{selectedPair=null;el("pair-list").querySelectorAll(".pair").forEach(x=>x.classList.remove("selected"));el("pair-title").textContent="Select a transition";applyVisibility();fit()};document.querySelectorAll("[data-mode]").forEach(button=>button.onclick=()=>setMode(button.dataset.mode));
const labels=["wrong_gap_count","unsupported_interpolation","surface_fragmentation","class_discontinuity","excess_uncertainty","wrong_z_spacing","other"];el("issues").innerHTML=labels.map(x=>`<label><input type="checkbox" value="${x}">${x.replaceAll("_"," ")}</label>`).join("");
document.querySelectorAll("[data-decision]").forEach(b=>b.onclick=()=>save(b.dataset.decision));
async function save(decision){if(selectedPair===null){el("status").textContent="Select a transition first";return}const r=current.gap_decisions[selectedPair],body={cohort:current.id,stage:"topology",fingerprint:current.fingerprint,slide_id:`${String(r.source_section).padStart(3,"0")}-${String(r.target_section).padStart(3,"0")}`,decision,labels:[...el("issues").querySelectorAll("input:checked")].map(x=>x.value),comment:el("comment").value,reviewer:el("reviewer").value,suggested_intervals:+el("intervals").value};if(decision==="accept")body.labels=[];el("status").textContent="Saving...";try{const response=await fetch("/api/reviews/feedback",{method:"POST",headers:{"Content-Type":"application/json","Authorization":`Bearer ${el("token").value}`},body:JSON.stringify(body)}),payload=await response.json();if(!response.ok)throw Error(payload.error||"Review save failed");el("status").textContent=`Saved ${decision}`}catch(error){el("status").textContent=error.message}}
function animate(){requestAnimationFrame(animate);controls.update();renderer.render(scene,camera)}resize();const select=el("cohort");select.innerHTML=data.cohorts.map(c=>`<option>${c.id}</option>`).join("");select.onchange=()=>selectCohort(select.value);selectCohort(data.cohorts[0].id);animate();
"""
