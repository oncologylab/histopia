"use strict";
const data=globalThis.HISTOPIA_SERIAL_STACK,$=id=>document.getElementById(id),params=new URL(location.href).searchParams;
let selected=Math.max(0,data.planes.findIndex(p=>p.id===params.get("section"))),three=null,show3d=false;
let requestedSection=null,imageRevision=0,imageReady=false,imageFailed=false,wants3d=params.get('mode')!=='section';
if(params.get("embedded")==="1")document.body.classList.add("embedded");
$("title").textContent=data.title;$("context").textContent=`${data.source_id} / ${data.organ} / ${data.specimen_id}`;$("fingerprint").textContent=data.fingerprint;$("registration-status").textContent=data.registration_status==="previously_reviewed"?"Previously reviewed alignment":"Registration candidate";
for(const p of data.planes){$("section").add(new Option(`Section ${p.order} · ${p.modality}`,p.id));const tr=document.createElement("tr");for(const value of [p.id,p.modality,p.z_um??"Unknown",(p.registration.mask_dice??p.registration.metrics?.mask_dice)==null?(p.registration.reference?"Reference":"Not reported"):Number(p.registration.mask_dice??p.registration.metrics?.mask_dice).toFixed(3),p.registration.passes_geometry_screen===true?"Passed coarse screen":p.registration.reference?"Reference":data.registration_status==="previously_reviewed"?"Previously reviewed":"Needs landmarks"]){const td=document.createElement("td");td.textContent=value;tr.append(td);}$("metrics").append(tr);}
$("observation").textContent=`${data.planes.length} observed planes · No interpolation`;
const z=data.planes.map(p=>p.z_um);$("depth").textContent=data.physical_stack_available?`${data.spacing_status==="assumed"?"Assumed Z":data.spacing_status==="documented"?"Physical Z":"Z"} ${Math.min(...z)}–${Math.max(...z)} µm${data.spacing_status==="assumed"?" · spacing unmeasured":""}`:'Physical depth unknown';
function sync(){const url=new URL(location.href);url.searchParams.set("section",data.planes[selected].id);url.searchParams.set("source",data.source_id);url.searchParams.set("organ",data.organ);url.searchParams.set("subject",data.subject_id);url.searchParams.set("mode",show3d?"3d":"section");history.replaceState(null,"",url);if(parent!==window)parent.postMessage({type:"histopia-serial-selection",source:data.source_id,organ:data.organ,subject:data.subject_id,section:data.planes[selected].id,mode:show3d?"3d":"section"},location.origin);}
function showImageState(){
 $("section-image").hidden=show3d||!imageReady;
 $("image-status").hidden=show3d||imageReady;
 $("image-status").textContent=imageFailed?"Section image unavailable":"Loading section…";
 $("viewport").setAttribute("aria-busy",String(!show3d&&!imageReady&&!imageFailed));
}
function showSection(){
 const p=data.planes[selected];$("section").value=p.id;
 if(requestedSection!==p.id){
  requestedSection=p.id;imageReady=false;imageFailed=false;
  const revision=++imageRevision,nextImage=new Image();nextImage.id="section-image";
  nextImage.alt=`Observed ${p.modality} · ${p.id} · Z ${p.z_um??"unknown"} µm`;
  nextImage.onload=()=>{if(revision!==imageRevision)return;$("section-image").replaceWith(nextImage);imageReady=true;showImageState();};
  nextImage.onerror=()=>{if(revision!==imageRevision)return;imageFailed=true;showImageState();};
  nextImage.src=p.image.href;
 }
 showImageState();$("mode-message").textContent=show3d?"Drag to rotate · Scroll to zoom":`Section ${selected+1} / ${data.planes.length}`;sync();
}
function set3d(active){wants3d=Boolean(active);show3d=Boolean(wants3d&&three);$("view-3d").setAttribute("aria-pressed",String(show3d));if(three)three.renderer.domElement.hidden=!show3d;$("viewport").dataset.renderer=show3d?"webgl":"section";$("z-control").hidden=!show3d;showSection();if(show3d)three.render();}
function step(amount){selected=(selected+amount+data.planes.length)%data.planes.length;set3d(false);}
$("section").onchange=e=>{selected=data.planes.findIndex(p=>p.id===e.target.value);set3d(false);};$("previous").onclick=()=>step(-1);$("next").onclick=()=>step(1);$("view-3d").onclick=()=>set3d(!show3d);$("reset").onclick=()=>three?.reset();
$("z-scale").oninput=e=>{$("z-value").value=e.target.value;three?.layout(Number(e.target.value));};
$("z-control").hidden=true;$("view-3d").disabled=true;showSection();
async function initialize3d(){
 if(!data.physical_stack_available){$("mode-message").textContent="3D requires documented physical depth";return;}
 try{
  const THREE=await import('three'),{OrbitControls}=await import('three/addons/controls/OrbitControls.js');
  const renderer=new THREE.WebGLRenderer({antialias:true,alpha:true});renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.setClearColor(0xe4eaf0,1);renderer.domElement.hidden=true;$("viewport").append(renderer.domElement);
  const scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera(35,1,.01,20000),controls=new OrbitControls(camera,renderer.domElement),group=new THREE.Group();scene.add(group);
  const plane=data.planes[0],width=plane.frame_shape_yx[1]*plane.mpp_xy[0],height=plane.frame_shape_yx[0]*plane.mpp_xy[1],scale=360/Math.max(width,height),center=(Math.min(...z)+Math.max(...z))/2,loader=new THREE.TextureLoader();
  for(const p of data.planes){const tex=await loader.loadAsync(p.image.href);tex.colorSpace=THREE.SRGBColorSpace;const mesh=new THREE.Mesh(new THREE.PlaneGeometry(width*scale,height*scale),new THREE.MeshBasicMaterial({map:tex,transparent:true,side:THREE.DoubleSide,alphaTest:.1,depthWrite:true}));mesh.userData.plane=p;group.add(mesh);}
  function render(){if(show3d)renderer.render(scene,camera);}
  function layout(factor){for(const mesh of group.children)mesh.position.z=(mesh.userData.plane.z_um-center)*scale*factor;reset();}
  function reset(){
   const bounds=new THREE.Box3().setFromObject(group),sphere=bounds.getBoundingSphere(new THREE.Sphere());
   const vertical=THREE.MathUtils.degToRad(camera.fov),horizontal=2*Math.atan(Math.tan(vertical/2)*camera.aspect),distance=1.12*sphere.radius/Math.sin(Math.min(vertical,horizontal)/2);
   camera.position.copy(sphere.center).add(new THREE.Vector3(0,-.78,.63).normalize().multiplyScalar(distance));
   controls.target.copy(sphere.center);controls.update();render();
  }
  function resize(){const r=$("viewport").getBoundingClientRect();renderer.setSize(r.width,r.height,false);camera.aspect=r.width/r.height;camera.updateProjectionMatrix();reset();}
  controls.addEventListener('change',render);new ResizeObserver(resize).observe($("viewport"));renderer.domElement.addEventListener('webglcontextlost',e=>{e.preventDefault();set3d(false);$("view-3d").disabled=true;$("mode-message").textContent="3D unavailable · Observed sections remain available";});
  three={renderer,render,layout,reset};layout(150);reset();resize();$("view-3d").disabled=false;set3d(wants3d);
 }catch(error){$("mode-message").textContent="3D unavailable · Observed sections remain available";$("viewport").dataset.renderer="fallback";$("view-3d").disabled=true;}
}
initialize3d();
