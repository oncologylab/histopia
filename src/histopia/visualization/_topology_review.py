"""Static, fixed-viewport review application for topology surfaces."""

# ruff: noqa: E501

from __future__ import annotations

import json
import shutil
from pathlib import Path

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
        cohort_assets = assets / cohort
        cohort_assets.mkdir(parents=True, exist_ok=True)
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
                "planes": payload["planes"],
                "meshes": mesh_rows,
                "classes": payload["classes"],
                "benchmark": benchmark["summary"],
            }
        )
    manifest = {"schema_version": 1, "cohorts": cohorts}
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
    <button id="fit" title="Fit reconstruction to view">Fit</button>
    <button id="reset" title="Reset camera">Reset</button>
  </header>
  <main>
    <section id="viewport" aria-label="Interactive semantic topology"></section>
    <aside>
      <section class="controls">
        <div id="metrics"></div>
        <label>Z exaggeration <input id="z-scale" type="range" min="1" max="250" value="100"></label>
        <label><input id="observed" type="checkbox" checked> Observed planes</label>
        <label><input id="virtual" type="checkbox" checked> Inferred planes</label>
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
*{box-sizing:border-box}body{margin:0;height:100dvh;overflow:hidden}header{height:48px;display:flex;align-items:center;gap:12px;padding:0 14px;border-bottom:1px solid #30363d;background:#161b22}header strong{font-size:15px}header span{color:#9da7b3;font-size:12px;flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}button,select,input,textarea{font:inherit;color:inherit;background:#21262d;border:1px solid #484f58;border-radius:4px}button,select,input{height:30px;padding:0 9px}button{cursor:pointer}button:hover{border-color:#8c959f}main{height:calc(100dvh - 48px);display:grid;grid-template-columns:minmax(0,1fr) 340px}#viewport{min-width:0;position:relative;background:#080b10}aside{border-left:1px solid #30363d;display:grid;grid-template-rows:auto minmax(100px,1fr) auto;min-height:0;background:#0d1117}aside>section{padding:10px 12px;border-bottom:1px solid #30363d}.controls{font-size:12px;display:grid;gap:7px}.controls label{display:flex;align-items:center;gap:7px}.controls input[type=range]{padding:0;flex:1}.controls input[type=checkbox],#issues input{height:auto}.metric-grid{display:grid;grid-template-columns:1fr 1fr;gap:5px}.metric{background:#161b22;padding:6px;border-radius:4px}.metric b{display:block;font-size:14px}.class-row{display:flex;align-items:center;gap:7px;margin-top:5px}.swatch{width:11px;height:11px;border-radius:2px}.pairs{min-height:0;display:flex;flex-direction:column}.pairs h2,.review h2{font-size:12px;text-transform:uppercase;color:#9da7b3;margin:0 0 7px}#pair-list{overflow:auto;min-height:0}.pair{width:100%;height:auto;text-align:left;padding:7px;margin-bottom:4px;background:#161b22}.pair.selected{border-color:#58a6ff}.pair span{display:flex;justify-content:space-between;font-size:11px}.pair small{color:#9da7b3}.review{display:grid;gap:6px}.review input,.review textarea{width:100%}.review textarea{height:54px;padding:6px;resize:none}.review label{font-size:11px;color:#9da7b3}.review label input{width:58px;margin-left:5px}.decision{display:grid;grid-template-columns:repeat(3,1fr);gap:5px}.decision button:first-child{border-color:#3fb950}.decision button:last-child{border-color:#f85149}#issues{display:flex;flex-wrap:wrap;gap:4px;font-size:10px}#issues label{padding:3px 5px;background:#161b22;border-radius:3px}#status{font-size:11px;min-height:15px;color:#9da7b3}@media(max-width:900px){main{grid-template-columns:minmax(0,1fr) 290px}aside{font-size:11px}}
"""

_JS = r"""
import * as THREE from "./vendor/three.module.min.js";
import {OrbitControls} from "./vendor/OrbitControls.js";
const data=globalThis.HISTOPIA_TOPOLOGY_REVIEW;
const el=id=>document.getElementById(id);
const scene=new THREE.Scene();scene.background=new THREE.Color(0x080b10);
const camera=new THREE.PerspectiveCamera(42,1,0.1,1e7);
const renderer=new THREE.WebGLRenderer({antialias:true,powerPreference:"high-performance"});
renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.localClippingEnabled=true;el("viewport").append(renderer.domElement);
const controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;
scene.add(new THREE.HemisphereLight(0xffffff,0x263238,2.2));
const light=new THREE.DirectionalLight(0xffffff,2);light.position.set(1,2,2);scene.add(light);
let group=new THREE.Group(),planes=new THREE.Group(),current=null,selectedPair=null,loadGeneration=0;
scene.add(group);scene.add(planes);
function resize(){const r=el("viewport").getBoundingClientRect();renderer.setSize(r.width,r.height,false);camera.aspect=r.width/r.height;camera.updateProjectionMatrix()}addEventListener("resize",resize);
async function loadMesh(row,generation){
 const buffer=await fetch(row.viewer_asset).then(r=>{if(!r.ok)throw Error("Mesh load failed");return r.arrayBuffer()});
 const view=new DataView(buffer);if(String.fromCharCode(...new Uint8Array(buffer,0,4))!=="HTM1")throw Error("Invalid mesh");
 const nv=view.getUint32(4,true),nf=view.getUint32(8,true),raw=new Float32Array(buffer,12,nv*3),pos=new Float32Array(raw.length);
 for(let i=0;i<nv;i++){pos[3*i]=raw[3*i];pos[3*i+1]=raw[3*i+2];pos[3*i+2]=-raw[3*i+1]}
 const index=new Uint32Array(buffer,12+nv*12,nf*3),geometry=new THREE.BufferGeometry();
 geometry.setAttribute("position",new THREE.BufferAttribute(pos,3));geometry.setIndex(new THREE.BufferAttribute(index,1));geometry.computeVertexNormals();
 const material=new THREE.MeshStandardMaterial({color:row.color,roughness:.72,metalness:.02,transparent:true,opacity:.78,side:THREE.DoubleSide});
 const mesh=new THREE.Mesh(geometry,material);mesh.userData={classIndex:row.class_index};if(generation!==loadGeneration){geometry.dispose();material.dispose();return}group.add(mesh);
}
function clear(){for(const root of [group,planes]){while(root.children.length){const o=root.children.pop();o.geometry?.dispose();o.material?.dispose()}}}
function planeObjects(c){
 const grid=c.reference_grid,ox=grid.origin_um_xy[0],oy=grid.origin_um_xy[1],w=grid.shape_rc[1]*grid.spacing_um,h=grid.shape_rc[0]*grid.spacing_um;
 for(const row of c.planes){const g=new THREE.PlaneGeometry(w,h),m=new THREE.MeshBasicMaterial({color:row.observed?0x58a6ff:0xd29922,wireframe:true,transparent:true,opacity:row.observed?.10:.24,side:THREE.DoubleSide});const p=new THREE.Mesh(g,m);p.rotation.x=-Math.PI/2;p.position.set(ox+w/2,row.z_um,-oy-h/2);p.userData={observed:row.observed};planes.add(p)}
}
function fit(){
 const box=new THREE.Box3().setFromObject(group);if(box.isEmpty())return;const center=box.getCenter(new THREE.Vector3()),size=box.getSize(new THREE.Vector3()),radius=Math.max(size.x,size.y,size.z);controls.target.copy(center);camera.position.set(center.x+radius*.9,center.y+radius*.75,center.z+radius*1.15);camera.near=Math.max(radius/10000,.1);camera.far=radius*20;camera.updateProjectionMatrix();controls.update()
}
async function selectCohort(id){
 const generation=++loadGeneration;current=data.cohorts.find(c=>c.id===id);selectedPair=null;clear();el("provenance").textContent=`${current.observed_section_count} observed | ${current.virtual_section_count} inferred | z: ${current.z_source} | loading`;
 el("metrics").innerHTML=`<div class="metric-grid"><div class="metric"><b>${pct(current.benchmark.flow_macro_class_dice)}</b>flow Dice</div><div class="metric"><b>${signed(current.benchmark.flow_gain_over_zero)}</b>gain vs zero</div><div class="metric"><b>${pct(current.benchmark.gap_interval_accuracy)}</b>gap accuracy</div><div class="metric"><b>${current.segment_count}</b>segments</div></div>`;
 el("classes").innerHTML=current.classes.map(r=>`<label class="class-row"><input type="checkbox" checked data-class="${r.class_index}"><span class="swatch" style="background:${r.color}"></span>Class ${r.class_index} | ${r.estimated_volume_mm3.toFixed(3)} mm3</label>`).join("");
 el("classes").querySelectorAll("input").forEach(x=>x.onchange=()=>group.children.filter(m=>m.userData.classIndex===+x.dataset.class).forEach(m=>m.visible=x.checked));
 el("pair-list").innerHTML=current.gap_decisions.map((r,i)=>`<button class="pair" data-i="${i}"><span><b>${String(r.source_section+1).padStart(2,"0")} to ${String(r.target_section+1).padStart(2,"0")}</b><em>${r.status}</em></span><small>${r.intervals} interval${r.intervals===1?"":"s"} | confidence ${pct(r.confidence)}</small></button>`).join("");
 el("pair-list").querySelectorAll(".pair").forEach(x=>x.onclick=()=>choosePair(+x.dataset.i));
 planeObjects(current);await Promise.all(current.meshes.map(row=>loadMesh(row,generation)));if(generation!==loadGeneration)return;applyScale();fit();el("provenance").textContent=el("provenance").textContent.replace("loading","ready")
}
function choosePair(i){selectedPair=i;el("pair-list").querySelectorAll(".pair").forEach((x,j)=>x.classList.toggle("selected",i===j));const r=current.gap_decisions[i];el("pair-title").textContent=`Transition ${r.source_section+1} to ${r.target_section+1}`;el("intervals").value=r.intervals;el("status").textContent=(r.reasons||[]).join(", ")||"Ready for review"}
function applyScale(){const z=+el("z-scale").value;group.scale.y=z;planes.scale.y=z}
function pct(x){return `${(100*x).toFixed(1)}%`}function signed(x){return `${x>=0?"+":""}${(100*x).toFixed(1)} pp`}
el("z-scale").oninput=applyScale;el("observed").onchange=e=>planes.children.filter(p=>p.userData.observed).forEach(p=>p.visible=e.target.checked);el("virtual").onchange=e=>planes.children.filter(p=>!p.userData.observed).forEach(p=>p.visible=e.target.checked);el("fit").onclick=fit;el("reset").onclick=fit;
const labels=["wrong_gap_count","unsupported_interpolation","surface_fragmentation","class_discontinuity","excess_uncertainty","wrong_z_spacing","other"];el("issues").innerHTML=labels.map(x=>`<label><input type="checkbox" value="${x}">${x.replaceAll("_"," ")}</label>`).join("");
document.querySelectorAll("[data-decision]").forEach(b=>b.onclick=()=>save(b.dataset.decision));
async function save(decision){if(selectedPair===null){el("status").textContent="Select a transition first";return}const r=current.gap_decisions[selectedPair],body={cohort:current.id,stage:"topology",fingerprint:current.fingerprint,slide_id:`${String(r.source_section).padStart(3,"0")}-${String(r.target_section).padStart(3,"0")}`,decision,labels:[...el("issues").querySelectorAll("input:checked")].map(x=>x.value),comment:el("comment").value,reviewer:el("reviewer").value,suggested_intervals:+el("intervals").value};if(decision==="accept")body.labels=[];el("status").textContent="Saving...";try{const response=await fetch("/api/reviews/feedback",{method:"POST",headers:{"Content-Type":"application/json","Authorization":`Bearer ${el("token").value}`},body:JSON.stringify(body)}),payload=await response.json();if(!response.ok)throw Error(payload.error||"Review save failed");el("status").textContent=`Saved ${decision}`}catch(error){el("status").textContent=error.message}}
function animate(){requestAnimationFrame(animate);controls.update();renderer.render(scene,camera)}resize();const select=el("cohort");select.innerHTML=data.cohorts.map(c=>`<option>${c.id}</option>`).join("");select.onchange=()=>selectCohort(select.value);selectCohort(data.cohorts[0].id);animate();
"""
