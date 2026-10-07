"use strict";
(() => {
const data=globalThis.HISTOPIA_ANALYSIS_REVIEW;
if(!data||data.schema_version!=="analysis-review-1")throw new Error("Missing analysis navigation index");
const $=id=>document.getElementById(id), entries=data.entries, byId=new Map(entries.map(e=>[e.id,e]));
const frame=$("review"), dialog=$("detail-dialog");
const displayPanels=new Set(["atlas","topology","protein-atlas"]);
const pretty=s=>String(s||"").replace(/[-_]/g," ").replace(/^./,c=>c.toUpperCase());
const sourceName=id=>{const s=data.sources.find(s=>s.id===id);return s?(s.external?"Public data · ":"Our data · ")+s.label.replace(/ · Our data$/,""):id;};
const compare=(a,b)=>(a.priority??20)-(b.priority??20)||Number(a.external)-Number(b.external)||a.organ.localeCompare(b.organ)||a.subject_id.localeCompare(b.subject_id,undefined,{numeric:true})||(a.section_order??0)-(b.section_order??0)||(a.short_title||a.title).localeCompare(b.short_title||b.title,undefined,{numeric:true})||a.id.localeCompare(b.id);
const all=entries.slice().sort(compare);
const aliases={"data-catalog":"registration","internal-inputs":"data","external-validation":"protein","organ-metadata":"data",atlas:"registration",topology:"registration","selected-cells":"cells",annotations:"cells","protein-atlas":"protein","study-figure":"figures",decisions:"reviews",methods:"data"};
const catalogViews=new Set(["data-catalog","internal-inputs","external-validation"]);
const legacyModes={atlas:"atlas",topology:"volume","selected-cells":"boundaries",annotations:"annotations","protein-atlas":"atlas","organ-metadata":"inventory","study-figure":"figures",decisions:"reviews",methods:"inventory"};
const icons={layers:'<path d="m3 8 9-5 9 5-9 5-9-5zm0 5 9 5 9-5M3 18l9 5 9-5"/>',drop:'<path d="M12 3s6 7 6 12a6 6 0 0 1-12 0c0-5 6-12 6-12Z"/>',cells:'<circle cx="7" cy="8" r="4"/><circle cx="17" cy="7" r="3"/><circle cx="14" cy="17" r="5"/>',signal:'<path d="M2 15c4-15 5 10 10-2S18 8 22 5"/>',regions:'<circle cx="6" cy="6" r="3"/><circle cx="18" cy="9" r="3"/><circle cx="10" cy="19" r="3"/><path d="m8 8 7 1m-8 0 2 7m7-4-4 5"/>',table:'<path d="M3 4h18v16H3zM3 10h18M9 4v16"/>',figure:'<path d="M3 4h18v16H3zM3 12h18M11 4v16"/>',check:'<path d="M5 3h14v18H5zM8 9l2 2 6-5M8 16h8"/>'};
let current=null,state={},reviewOpen=false,showPrevious=false,sectionControl=null,renderGeneration=0,lastAnalysis={},unavailable=false;
let cleanupFrame=()=>{};
function el(tag,text,cls){const e=document.createElement(tag);if(text!==undefined)e.textContent=String(text);if(cls)e.className=cls;return e;}
function path(href){const u=new URL(href,location.href);if(href.startsWith("/")){const p=location.pathname.match(/^.*?\/proxy\/\d+(?=\/|$)/)?.[0]||"";u.pathname=p+href;}if(!["http:","https:","file:"].includes(u.protocol))throw new Error("Unsupported evidence URL");return u.href;}
function remember(){if(!current)return;try{sessionStorage.setItem("histopia-analysis-v1:"+state.view,JSON.stringify(state));}catch{}if(!data.workspaces.find(w=>w.id===state.view)?.utility)lastAnalysis={...state};}
function recalled(view){try{return JSON.parse(sessionStorage.getItem("histopia-analysis-v1:"+view)||"null");}catch{return null;}}
function notice(message){$("selection-notice").querySelector("span").textContent=message;$("selection-notice").hidden=!message;}
$("selection-notice").querySelector("button").onclick=()=>notice("");
function contextText(e){return [sourceName(e.source_id),pretty(e.organ),e.subject_id].filter(Boolean).join(" / ");}
function options(select,rows,value){select.replaceChildren(...rows.map(([v,t])=>new Option(t,v)));select.value=value;}
function supports(e,mode){return !mode||e.mode===mode||e.related_modes?.includes(mode);}
function candidates(view=state.view,mode=state.mode){return all.filter(e=>e.analysis===view&&supports(e,mode)&&(showPrevious||!e.previous));}
function writeUrl(push=false){const u=new URL(location.href);u.search="";for(const [key,value]of Object.entries(state))if(value!==undefined&&value!==null&&value!=="")u.searchParams.set(key,String(value));if(push&&u.href!==location.href)history.pushState(null,"",u);else history.replaceState(null,"",u);}
function selectedMode(view,requested,source,organ){const rows=candidates(view,null);if(rows.some(e=>supports(e,requested)&&(!source||e.source_id===source)&&(!organ||e.organ===organ)))return requested;return data.workspaces.find(w=>w.id===view)?.modes.find(([m])=>rows.some(e=>supports(e,m)&&(!source||e.source_id===source)&&(!organ||e.organ===organ)))?.[0]||data.workspaces.find(w=>w.id===view)?.modes.find(([m])=>rows.some(e=>supports(e,m)))?.[0]||requested;}
function resolve(next){
 next.mode=selectedMode(next.view,next.mode,next.source,next.organ);
 const rows=candidates(next.view,next.mode),saved=recalled(next.view);
 const exact=rows.find(e=>e.id===next.dataset&&e.source_id===next.source&&e.organ===next.organ&&e.subject_id===next.subject);
 if(exact)return exact;
 const same=rows.filter(e=>(!next.source||e.source_id===next.source)&&(!next.organ||e.organ===next.organ));
 if(next.view==="data"&&next.mode==="inventory")return rows.find(e=>e.global_context)||rows[0];
 if(next.view==="figures")return rows.find(e=>e.renderer==="study-figure")||rows[0];
 return same.find(e=>e.subject_id===next.subject)||same.find(e=>e.id===saved?.dataset)||same[0]||rows.find(e=>e.id===saved?.dataset)||rows[0];
}
function use(entry,next,push=false,explain=false){
 const previous=current;const same=current?.id===entry?.id;
 current=entry;unavailable=!entry;
 if(!entry){empty(next,"No published result is available for this analysis.");return;}
 state={view:entry.analysis,mode:supports(entry,next.mode)&&next.mode?next.mode:entry.mode,source:entry.source_id,organ:entry.organ,subject:entry.subject_id,dataset:entry.id};
 if(entry.global_context){delete state.source;delete state.organ;for(const k of ["source","organ"])if(next[k])state[k]=next[k];}
 for(const key of ["section","slide","field","region","evidence","stage","stack_section","stack_mode","model","target","inventory_q","inventory_status","inventory_table"])
  if(next[key]!=null&&(same||next.dataset===entry.id))state[key]=next[key];
 if(entry.mode_fields?.[state.mode]!=null&&next.field==null)state.field=String(entry.mode_fields[state.mode]);
 if(entry.reconstruction_id)state.reconstruction=entry.reconstruction_id;
 if(entry.previous)showPrevious=true;
 reviewOpen=false;$("review-toggle").setAttribute("aria-pressed","false");
 const moved=!entry.global_context&&((next.subject&&next.subject!==entry.subject_id)||(next.source&&next.source!==entry.source_id)||(next.organ&&next.organ!==entry.organ));
 notice((explain&&previous&&moved)||(!previous&&moved)?`Showing ${contextText(entry)}. The previous selection has no result in this view.`:"");
 writeUrl(push);remember();renderChrome();renderEvidence();
}
function navigate(view,mode){remember();const saved=recalled(view);const workspace=data.workspaces.find(w=>w.id===view);const previous=data.workspaces.find(w=>w.id===state.view)?.utility?lastAnalysis:state;
 const next={...previous,view,mode:mode||saved?.mode||workspace?.modes[0][0]};for(const k of ["dataset","section","slide","field","region","evidence","model","target","stage","stack_section","stack_mode"])delete next[k];
 use(resolve(next),next,true,true);
}
let utilityAdded=false;
for(const w of data.workspaces){
 if(w.utility&&!utilityAdded){$("navigation").append(el("span","Reference","utilities"));utilityAdded=true;}
 const b=el("button");b.dataset.view=w.id;b.innerHTML='<svg viewBox="0 0 24 26" aria-hidden="true">'+icons[w.icon]+"</svg>";b.append(el("span",w.label));b.onclick=()=>navigate(w.id);$("navigation").append(b);
 $("mobile-view").add(new Option(w.label,w.id));
}
$("mobile-view").onchange=e=>navigate(e.target.value);
function renderChrome(){
 const w=data.workspaces.find(w=>w.id===state.view);$("workspace-title").textContent=w?.label||"Scientific review";document.title="Histopia · "+$("workspace-title").textContent;
 for(const b of $("navigation").querySelectorAll("button"))b.setAttribute("aria-current",b.dataset.view===state.view?"page":"false");$("mobile-view").value=state.view;
 $("context-line").textContent=current?.global_context?"All internal scans · curated metadata and file inventory":current?contextText(current):"";
 $("modes").replaceChildren();for(const [id,label]of w?.modes||[]){const rows=candidates(state.view,id);if(!rows.length)continue;const b=el("button",label);b.dataset.mode=id;b.setAttribute("aria-current",id===state.mode?"page":"false");b.disabled=!!current&&!current.global_context&&state.view!=="data"&&!rows.some(e=>e.source_id===state.source&&e.organ===state.organ);if(b.disabled)b.title="No published result for this source and organ";b.onclick=()=>navigate(state.view,id);$("modes").append(b);}
 $("context-bar").hidden=["inventory","figures"].includes(state.mode);$("result-heading").hidden=state.mode==="inventory";
 $("controls-toggle").hidden=!displayPanels.has(current?.renderer);$("controls-toggle").setAttribute("aria-pressed","false");
 if(!current)return;
 const rows=candidates(state.view,state.mode),sourceIds=new Set(candidates(state.view,null).map(e=>e.source_id));
 $("source").replaceChildren();for(const external of [false,true]){const group=el("optgroup");group.label=external?"Public data":"Our data";for(const s of data.sources.filter(s=>s.external===external&&sourceIds.has(s.id)))group.append(new Option(sourceName(s.id),s.id));if(group.children.length)$("source").append(group);}$("source").value=state.source;
 const organs=[...new Set(candidates(state.view,null).filter(e=>e.source_id===state.source).map(e=>e.organ))].sort();options($("organ"),organs.map(o=>[o,pretty(o)]),state.organ);
 const same=rows.filter(e=>e.source_id===state.source&&e.organ===state.organ),subjects=[...new Set(same.map(e=>e.subject_id))].sort((a,b)=>a.localeCompare(b,undefined,{numeric:true}));
 $("subject-label").textContent=current.external?"Donor / specimen":"Mouse / specimen";options($("subject"),subjects.map(s=>[s,s||"All"]),state.subject);
 const results=same.filter(e=>e.subject_id===state.subject);options($("result"),results.map(e=>[e.id,(e.previous?"Previous · ":"")+(e.short_title||e.title)]),current.id);
 $("history-button").hidden=!entries.some(e=>e.analysis===state.view&&e.source_id===state.source&&e.organ===state.organ&&e.previous);$("history-button").setAttribute("aria-pressed",String(showPrevious));
 $("result-title").textContent=current.short_title||current.title;$("scope-label").textContent=current.scope_label||"Published evidence";
 $("section-control").hidden=true;sectionControl=null;
 $("review-toggle").hidden=["inventory","figures","methods"].includes(state.mode)||["atlas","stack","protein","protein-atlas"].includes(current.renderer);
}
for(const key of ["source","organ","subject"])$(key).onchange=e=>{const next={...state,[key]:e.target.value};delete next.dataset;delete next.section;delete next.field;delete next.region;
 if(key==="source"){delete next.organ;delete next.subject;}else if(key==="organ")delete next.subject;
 use(resolve(next),next,true,false);
};
$("result").onchange=e=>{const entry=byId.get(e.target.value);use(entry,{dataset:entry.id},true);};
$("history-button").onclick=()=>{showPrevious=!showPrevious;const next={...state};if(current.previous&&!showPrevious)delete next.dataset;use(resolve(next),next,false);};
function empty(next,message){cleanupFrame();renderGeneration++;frame.hidden=true;frame.removeAttribute("src");$("media-view").hidden=true;current=null;state=next;unavailable=true;notice("");renderChrome();$("review-toggle").hidden=true;$("unavailable").hidden=false;$("unavailable-reason").textContent=message;$("alternatives").replaceChildren();
 for(const e of candidates(next.view,null).slice(0,4)){const b=el("button");if(e.media?.[0]){const im=el("img");im.src=path(e.media[0].href);im.alt="";b.append(im);}b.append(el("span",[pretty(e.organ),e.subject_id,e.short_title||e.title].join(" · ")));b.onclick=()=>use(e,{dataset:e.id},true);$("alternatives").append(b);}$("result-heading").hidden=true;$("context-bar").hidden=true;
}
function renderEvidence(){
 cleanupFrame();cleanupFrame=()=>{};renderGeneration++;$("unavailable").hidden=true;$("image-viewport").classList.remove("native");$("native-size").setAttribute("aria-pressed","false");
 if(current.renderer==="media"){
  frame.hidden=true;frame.removeAttribute("src");$("media-view").hidden=false;
  if(!current.media.length){$("image").removeAttribute("src");$("image-state").hidden=false;$("image-state").textContent="This result provides tables and downloads in Details and Export.";$("field").replaceChildren();return;}
  options($("field"),current.media.map((m,i)=>[String(i),(m.label||"Image")+(current.media.filter(x=>x.label===m.label).length>1?` · ${i+1}`:"")]),String(Math.min(Number(state.field)||0,current.media.length-1)));
  showField(Number($("field").value));
 }else{
  $("media-view").hidden=true;frame.hidden=false;
  const target=new URL(path(current.href));for(const [k,v]of Object.entries(state))if(!["view","mode"].includes(k))target.searchParams.set(k,v);
  if(current.subject_id){target.searchParams.set("mouse",current.subject_id);target.searchParams.set("cohort",current.subject_id);}
  target.searchParams.set("embedded","1");
  if(current.renderer==="stack"){
   if(state.stack_section)target.searchParams.set("section",state.stack_section);
   target.searchParams.set("mode",state.stack_mode||"3d");
  }
  frame.src=target.href;
 }
}
function showField(i){if(!current?.media?.length)return;const item=current.media[Math.max(0,Math.min(i,current.media.length-1))];state.field=String(current.media.indexOf(item));$("field").value=state.field;$("image-state").hidden=false;$("image-state").textContent="Loading image…";$("image").src=path(item.href);$("image").alt=(current.short_title||current.title)+" · "+(item.label||"Published evidence");$("open-image").href=path(item.href);$("image-caption").textContent=item.caption||item.label||"";$("previous-field").disabled=state.field==="0";$("next-field").disabled=Number(state.field)===current.media.length-1;writeUrl();remember();}
$("image").onload=()=>{$("image-state").hidden=true;};$("image").onerror=()=>{$("image-state").textContent="Image unavailable. The recorded result and downloads remain in Details.";$("image-state").hidden=false;};
$("field").onchange=e=>showField(Number(e.target.value));$("previous-field").onclick=()=>showField(Number(state.field)-1);$("next-field").onclick=()=>showField(Number(state.field)+1);
$("native-size").onclick=()=>{const native=$("image-viewport").classList.toggle("native");$("native-size").setAttribute("aria-pressed",String(native));$("native-size").textContent=native?"Fit image":"Native size";};
function selectSection(value){if(!sectionControl)return;sectionControl.value=value;sectionControl.dispatchEvent(new sectionControl.ownerDocument.defaultView.Event("change",{bubbles:true}));state.section=value;writeUrl();remember();syncSection();}
$("section").onchange=e=>selectSection(e.target.value);
for(const [id,step]of [["previous-section",-1],["next-section",1]])$(id).onclick=()=>{const s=$("section");const i=s.selectedIndex+step;if(i>=0&&i<s.options.length)selectSection(s.options[i].value);};
function shortSection(o){const text=o.textContent||"";if(/\.ndpi|\.scn|collection_|Yi_/i.test(text))return "Section "+o.value;return text.length>65?"Section "+o.value:text;}
function syncSection(){if(!sectionControl?.options.length)return;const native=sectionControl;const signature=[...native.options].map(o=>o.value+":"+o.textContent).join("|");if($("section").dataset.signature!==signature){options($("section"),[...native.options].map(o=>[o.value,shortSection(o)]),native.value);$("section").dataset.signature=signature;}$("section").value=native.value;$("section-control").hidden=false;$("previous-section").disabled=native.selectedIndex<=0;$("next-section").disabled=native.selectedIndex>=native.options.length-1;}

// Adapter styling changes presentation only; original viewers own every write.
const embedCSS=`
html.workspace-embed body{background:#f5f8fa!important}
.workspace-embed .shell-owned{display:none!important}
.workspace-embed body>header>strong,.workspace-embed body>header>h1{display:none!important}
.workspace-embed body>header{padding:6px 10px!important}
.workspace-embed[data-renderer=cells]:not(.review-open) main,.workspace-embed[data-renderer=selected-cells]:not(.review-open) main{grid-template-columns:minmax(0,1fr)!important}
.workspace-embed[data-renderer=cells]:not(.review-open) main>aside,.workspace-embed[data-renderer=selected-cells]:not(.review-open) main>aside{display:none!important}
.workspace-embed[data-renderer=annotations]:not(.review-open) main{grid-template-columns:minmax(0,1fr) 180px!important}
.workspace-embed[data-renderer=annotations]:not(.review-open) main>aside{display:none!important}
.workspace-embed[data-renderer=topology]:not(.review-open):not(.controls-open) body{grid-template-columns:minmax(0,1fr)!important}
.workspace-embed[data-renderer=topology]:not(.review-open):not(.controls-open) body>aside{display:none!important}
.workspace-embed[data-renderer=topology]:not(.review-open) .review{display:none!important}
.workspace-embed[data-renderer=stain] #scope,.workspace-embed[data-renderer=stain] #viewer,.workspace-embed[data-renderer=stain] #details-toggle{display:none!important}
.workspace-embed[data-renderer=stain] body{grid-template-rows:42px minmax(0,1fr)!important}
.workspace-embed[data-renderer=stain]:not(.review-open):not(.controls-open) #evidence{display:none!important}
.workspace-embed[data-renderer=stain]:not(.review-open):not(.controls-open) main{grid-template-columns:180px minmax(0,1fr)!important}
.workspace-embed[data-renderer=stain] #slide-meta{display:none!important}
.workspace-embed[data-renderer=topology]:not(.review-open):not(.controls-open) .app>aside{display:none!important}
.workspace-embed[data-renderer=topology]:not(.review-open):not(.controls-open) .app{grid-template-columns:minmax(0,1fr)!important}
.workspace-embed[data-renderer=atlas]:not(.controls-open) .app>aside{display:none!important}
.workspace-embed[data-renderer=atlas]:not(.controls-open) .app{grid-template-columns:minmax(0,1fr)!important}
.workspace-embed[data-renderer=atlas]:not(.controls-open) body>main>aside{display:none!important}
.workspace-embed[data-renderer=atlas]:not(.controls-open) body>main{grid-template-columns:minmax(0,1fr)!important;grid-template-rows:minmax(0,1fr)!important}
.workspace-embed[data-renderer=atlas] body>main>aside>h1{display:none!important}
.workspace-embed[data-renderer=atlas] label.check{display:flex;align-items:center;gap:8px}
.workspace-embed[data-renderer=atlas] label.check input{width:auto}
.workspace-embed[data-renderer=registration]:not(.review-open) aside{display:none!important}
.workspace-embed[data-renderer=registration]:not(.review-open) body:has(>#registration-feedback){grid-template-columns:minmax(0,1fr)!important}
.workspace-embed[data-renderer=registration]:not(.review-open) body:has(>#registration-feedback)>header{grid-column:1!important}
.workspace-embed[data-renderer=registration] .registration-context{display:none!important}
.workspace-embed[data-renderer=registration] body:has(>.registration-context){grid-template-rows:minmax(0,1fr)!important}
.workspace-embed[data-renderer=decisions]:not(.review-open) form>label,.workspace-embed[data-renderer=decisions]:not(.review-open) form>input,.workspace-embed[data-renderer=decisions]:not(.review-open) form>textarea,.workspace-embed[data-renderer=decisions]:not(.review-open) form>.actions{display:none!important}
.workspace-embed[data-renderer=decisions]:not(.review-open) #families{display:none!important}
.workspace-embed[data-renderer=decisions] #recorded-decision{grid-column:1/-1;font-size:12px;color:#52677a;overflow:auto;max-height:60vh}
.workspace-embed[data-renderer=decisions]:not(.review-open) form{grid-template-rows:auto minmax(0,1fr)!important}
.workspace-embed[data-renderer=decisions] #recorded-decision details{padding:8px 0;border-bottom:1px solid #dce5eb}
.workspace-embed[data-renderer=protein-atlas]:not(.controls-open) .app>aside{display:none!important}
.workspace-embed[data-renderer=protein-atlas]:not(.controls-open) .app{grid-template-columns:minmax(0,1fr)!important}
.workspace-embed[data-renderer=protein-atlas].controls-open .app>aside,.workspace-embed[data-renderer=topology].controls-open .app>aside,.workspace-embed[data-renderer=topology].review-open .app>aside{display:block!important}
.workspace-embed[data-renderer=protein-atlas] #panel-toggle,.workspace-embed[data-renderer=topology] #panel-toggle{display:none!important}
.workspace-embed[data-renderer=topology] .dataset{grid-template-columns:minmax(0,1fr)!important}
.workspace-embed[data-renderer=topology] .dataset output{display:block!important;white-space:normal}
.workspace-embed[data-renderer=topology] .controls .segments,.workspace-embed[data-renderer=topology] .workspace>header .segments{display:flex!important}
.workspace-embed[data-renderer=protein-atlas] .brand,.workspace-embed[data-renderer=topology] .workspace>header>strong{display:none!important}
.workspace-embed[data-renderer=protein] .controls>strong{display:none!important}
.workspace-embed[data-renderer=study-figure] body>h2{display:none!important}
.workspace-embed[data-renderer=regions] body>p:last-of-type{display:none!important}
.workspace-embed[data-renderer=organ-metadata] header h1{display:none!important}
.workspace-embed #workspace-legend{position:absolute;left:12px;bottom:12px;z-index:50;display:flex;gap:10px;flex-wrap:wrap;max-width:calc(100% - 24px);padding:7px 10px;border-radius:6px;background:#ffffffed;color:#28404e;font-size:10px}
.workspace-embed #workspace-legend:empty,.workspace-embed #workspace-legend[hidden],.workspace-embed.controls-open #workspace-legend{display:none!important}
.workspace-embed #workspace-legend span{display:flex;align-items:center;gap:5px}.workspace-embed #workspace-legend i{display:inline-block;width:8px;height:8px;border-radius:50%}
.workspace-embed #workspace-legend .status-chip{background:transparent;border-color:#cad5df;color:#28404e}
@media(max-width:760px){
 .workspace-embed[data-renderer=cells]:not(.review-open) main,.workspace-embed[data-renderer=selected-cells]:not(.review-open) main{grid-template-rows:minmax(0,1fr)!important}
 .workspace-embed[data-renderer=cells] body>header,.workspace-embed[data-renderer=selected-cells] body>header{display:flex!important}
 .workspace-embed[data-renderer=cells] body>header>button,.workspace-embed[data-renderer=selected-cells] body>header>button{min-width:30px!important}
 .workspace-embed[data-renderer=annotations]:not(.review-open) main{grid-template-columns:minmax(0,1fr)!important}
 .workspace-embed[data-renderer=annotations] .context{display:none!important}
 .workspace-embed[data-renderer=stain]:not(.review-open):not(.controls-open) main{grid-template-columns:100px minmax(0,1fr)!important}
 .workspace-embed[data-renderer=stain].review-open #evidence,.workspace-embed[data-renderer=stain].controls-open #evidence{transform:translateX(0)!important;top:42px!important}
 .workspace-embed[data-renderer=registration]:not(.review-open) .workspace{grid-template-columns:minmax(0,1fr)!important}
}
`;
function setDisplayControls(doc,open){
 doc.documentElement.classList.toggle("controls-open",open);
 if(["topology","protein-atlas"].includes(doc.documentElement.dataset.renderer))doc.querySelector(".app")?.classList.toggle("panel-open",open);
 const panel=doc.getElementById("panel-toggle");if(panel){panel.setAttribute("aria-expanded",String(open));panel.textContent=open?"Close":"Controls";}
 if(doc===frame.contentDocument)$("controls-toggle").setAttribute("aria-pressed",String(open));
 doc.defaultView?.requestAnimationFrame(()=>doc.defaultView?.dispatchEvent(new doc.defaultView.Event("resize")));
}
function toggleDisplayControls(){
 const open=!frame.contentDocument?.documentElement.classList.contains("controls-open");
 if(!open){reviewOpen=false;$("review-toggle").setAttribute("aria-pressed","false");}
 eachDoc(doc=>{if(!open)doc.documentElement.classList.remove("review-open");setDisplayControls(doc,open);});
}
$("controls-toggle").onclick=toggleDisplayControls;
function decorate(doc,renderer){
 if(!doc?.body)return;doc.documentElement.classList.add("workspace-embed");doc.documentElement.dataset.renderer=renderer;doc.documentElement.classList.toggle("review-open",reviewOpen);
 if(renderer==="stain"){const slide=new URL(doc.location.href).searchParams.get("slide");if(slide&&state.slide!==slide){state.slide=slide;writeUrl();remember();}}
 if(!doc.getElementById("analysis-embed-style")){
  const style=doc.createElement("style");style.id="analysis-embed-style";style.textContent=embedCSS;doc.head.append(style);
  // Display controls are essential; review forms alone are hidden by default.
  // Mobile drawers start closed so the image remains visible. The shell's
  // Controls button is always available, and explicit choices survive polling.
  if(displayPanels.has(renderer))setDisplayControls(doc,renderer==="atlas"||doc.defaultView.innerWidth>(renderer==="topology"?700:780));
 }
 for(const c of doc.querySelectorAll("select#cohort,select#mouse")){c.classList.add("shell-owned");c.closest("label")?.classList.add("shell-owned");}
 for(const label of doc.querySelectorAll('label[for="cohort"],label[for="mouse"]'))label.classList.add("shell-owned");
 const panel=doc.getElementById("panel-toggle");if(panel&&!panel.dataset.workspaceBound){panel.dataset.workspaceBound="1";panel.addEventListener("click",()=>setDisplayControls(doc,doc.querySelector(".app")?.classList.contains("panel-open")));}
 if(["atlas","protein-atlas"].includes(renderer)){
  const source=doc.getElementById(renderer==="atlas"?"legend":"target-status"),viewport=doc.getElementById("viewport");
  if(source&&viewport){let legend=doc.getElementById("workspace-legend");if(!legend){legend=doc.createElement("div");legend.id="workspace-legend";viewport.append(legend);}
   if(legend.dataset.signature!==source.innerHTML){legend.replaceChildren(...[...source.childNodes].map(n=>n.cloneNode(true)));legend.dataset.signature=source.innerHTML;}
   legend.hidden=renderer==="protein-atlas"&&doc.querySelector(".app")?.dataset.layer!=="protein";
  }
 }
 const target=doc.getElementById("model-target");
 if(renderer==="protein"&&target?.options.length&&!target.dataset.workspaceRestored){
  target.dataset.workspaceRestored="1";
  if(state.target&&[...target.options].some(o=>o.value===state.target)&&target.value!==state.target){target.value=state.target;target.dispatchEvent(new doc.defaultView.Event("change",{bubbles:true}));}
 }
 if(!["stain","topology","regions","stack"].includes(renderer)){
  const s=doc.querySelector("select#section");if(s?.options.length){sectionControl=s;
   if(!s.dataset.workspaceRestored){s.dataset.workspaceRestored="1";if(state.section&&[...s.options].some(o=>o.value===state.section)&&s.value!==state.section){s.value=state.section;s.dispatchEvent(new doc.defaultView.Event("change",{bubbles:true}));}}
   s.classList.add("shell-owned");s.closest("label")?.classList.add("shell-owned");for(const id of ["previous","next"])doc.getElementById(id)?.classList.add("shell-owned");syncSection();}
 }
 if(!doc.documentElement.dataset.selectionBound){doc.documentElement.dataset.selectionBound="1";const generation=renderGeneration;
  doc.addEventListener("change",()=>queueMicrotask(()=>{if(generation!==renderGeneration||doc.defaultView?.closed)return;for(const [id,key]of [["section","section"],["model-target","target"],["model","model"]]){const c=doc.getElementById(id);if(c?.value&&!["stack","regions"].includes(renderer))state[key]=c.value;}writeUrl();remember();}));
  doc.addEventListener("click",event=>{const stage=event.target.closest("[data-stage]")?.dataset.stage;if(generation===renderGeneration&&renderer==="registration"&&stage){state.stage=stage;writeUrl();remember();}});
 }
 if(renderer==="registration"&&!doc.documentElement.dataset.stageRestored&&doc.querySelector("[data-stage]")){
  doc.documentElement.dataset.stageRestored="1";if(state.stage){const button=[...doc.querySelectorAll("[data-stage]")].find(b=>b.dataset.stage===state.stage);if(button&&!button.hidden)button.click();}
 }
 for(const nested of doc.querySelectorAll("iframe")){const apply=()=>{try{decorate(nested.contentDocument,renderer);}catch{}};if(!nested.dataset.analysisBound){nested.dataset.analysisBound="1";nested.addEventListener("load",apply);}apply();}
}
frame.addEventListener("load",()=>{
 if(!current||frame.hidden)return;const generation=renderGeneration,renderer=current.renderer;
 const update=()=>{if(generation!==renderGeneration)return;try{decorate(frame.contentDocument,renderer);}catch{}};update();
 // Options arrive asynchronously; poll a bounded DOM state without owning data I/O.
 cleanupFrame();const timer=setInterval(update,600);cleanupFrame=()=>clearInterval(timer);
});
function eachDoc(fn){try{const visit=doc=>{if(!doc?.body)return;fn(doc);for(const f of doc.querySelectorAll("iframe"))visit(f.contentDocument);};visit(frame.contentDocument);}catch{}}
$("review-toggle").onclick=()=>{
 if(!current)return;if(current.renderer==="media"||current.renderer==="regions"||current.renderer==="stack"){openPanel("Review notes");draftForm();return;}
 reviewOpen=!reviewOpen;$("review-toggle").setAttribute("aria-pressed",String(reviewOpen));eachDoc(d=>d.documentElement.classList.toggle("review-open",reviewOpen));frame.contentWindow?.dispatchEvent(new Event("resize"));
};
function openPanel(title){$("dialog-title").textContent=title;$("dialog-content").replaceChildren();if(!dialog.open)dialog.showModal();}
$("close-dialog").onclick=()=>dialog.close();dialog.addEventListener("click",e=>{if(e.target===dialog&&e.clientX<dialog.getBoundingClientRect().left)dialog.close();});
function factList(facts){const dl=el("dl");for(const [k,v]of Object.entries(facts||{})){dl.append(el("dt",pretty(k)),el("dd",typeof v==="object"?JSON.stringify(v):v));}return dl;}
function link(label,href){const a=el("a",label,"download");a.href=path(href);a.target="_blank";a.rel="noopener";return a;}
$("details-button").onclick=()=>{
 if(!current)return;openPanel("Details");const box=$("dialog-content");box.append(el("h3",current.short_title||current.title),el("p",contextText(current)),el("p",current.scope_label||"Published evidence"));
 if(current.renderer!=="media"&&!frame.hidden){const b=el("button","Show / hide display controls");b.onclick=()=>{toggleDisplayControls();dialog.close();};box.append(b);}
 if(current.summary)box.append(el("p",current.summary));box.append(factList(current.facts));
 if(current.table?.length){const details=el("details"),summary=el("summary","Recorded metrics"),table=el("table");details.append(summary,table);const columns=Object.keys(current.table[0]);const head=el("tr");for(const col of columns)head.append(el("th",col));table.append(head);for(const row of current.table){const tr=el("tr");for(const col of columns)tr.append(el("td",typeof row[col]==="object"?JSON.stringify(row[col]):row[col]));table.append(tr);}box.append(details);}
 box.append(el("h3","Provenance"),factList({result:current.id,source:current.source_id,organ:current.organ,specimen:current.subject_id,...(current.reconstruction_id?{reconstruction:current.reconstruction_id}:{})}),el("code",current.fingerprint));
 if(current.eligibility_fingerprint)box.append(el("code","Eligibility · "+current.eligibility_fingerprint));
 const source=data.sources.find(s=>s.id===current.source_id);if(source?.description)box.append(el("p",source.description));
};
$("export-button").onclick=()=>{
 if(!current)return;openPanel("Export");const box=$("dialog-content");for(const item of [...(current.downloads||[]),...(current.links||[])])box.append(link(item.label||"Download",item.href));
 if(current.media?.length)for(const item of current.media)box.append(link(item.label||"Image",item.href));
 if(current.href){const u=new URL(path(current.href));for(const [k,v]of Object.entries(state))if(!["view","mode","dataset"].includes(k))u.searchParams.set(k,v);if(current.subject_id)for(const key of ["subject","mouse","cohort"])u.searchParams.set(key,current.subject_id);if(current.renderer==="stack"){u.searchParams.set("mode",state.stack_mode||"3d");if(state.stack_section)u.searchParams.set("section",state.stack_section);}box.append(link("Open standalone viewer",u.href));}
 const b=el("button","Copy result link");b.onclick=async()=>{try{await navigator.clipboard.writeText(location.href);b.textContent="Link copied";}catch{const input=el("input");input.value=location.href;box.append(input);input.select();}};box.append(b);
};
function draftForm(){const row=current,key="histopia-review-draft:"+row.id+":"+row.fingerprint,box=$("dialog-content");let saved={};try{saved=JSON.parse(localStorage.getItem(key)||"{}");}catch{}
 box.append(el("p",contextText(row)),el("p","Browser-local draft · scientific approvals are recorded separately.","draft-status"));const label=el("label","Assessment"),select=el("select");options(select,[["unreviewed","Not reviewed"],["needs-changes","Needs changes"],["lab-review","Ready for lab review"]],saved.assessment||"unreviewed");const noteLabel=el("label","Notes"),note=el("textarea");note.value=saved.note||"";note.placeholder="Field, finding, uncertainty";const status=el("p","","draft-status");box.append(label,select,noteLabel,note,status);let draft;
 const save=()=>{draft={dataset_id:row.id,source_id:row.source_id,organ:row.organ,fingerprint:row.fingerprint,assessment:select.value,note:note.value,updated_at:new Date().toISOString(),scientific_approval:false};try{localStorage.setItem(key,JSON.stringify(draft));status.textContent="Draft saved in this browser.";}catch{status.textContent="Export the draft to retain it.";}};select.onchange=save;note.oninput=save;
 const download=el("button","Export draft");download.onclick=()=>{save();const u=URL.createObjectURL(new Blob([JSON.stringify(draft,null,2)],{type:"application/json"}));const a=el("a");a.href=u;a.download="histopia-review-note.json";a.click();setTimeout(()=>URL.revokeObjectURL(u),1000);};box.append(download);
}
$("fullscreen").onclick=async()=>{try{if(document.fullscreenElement)await document.exitFullscreen();else await $("workspace").requestFullscreen();}catch{document.body.classList.toggle("presentation");}};
addEventListener("message",event=>{
 if(event.source!==frame.contentWindow||event.origin!==location.origin||!current)return;
 const m=event.data;
 if(m?.type==="histopia-serial-selection"&&current.renderer==="stack"){
  if(m.source!==current.source_id||m.organ!==current.organ||m.subject!==current.subject_id)return;
  if(typeof m.section==="string")state.stack_section=m.section;if(["3d","section"].includes(m.mode))state.stack_mode=m.mode;
 }else if(m?.type==="histopia-region-selection"&&current.renderer==="regions"){
  if(typeof m.region==="string")state.region=m.region;if(typeof m.evidence==="string")state.evidence=m.evidence;
 }else if(m?.type==="histopia-organ-metadata"&&current.renderer==="organ-metadata"){
  const published=frame.contentWindow.HISTOPIA_ORGAN_INVENTORY;if(!published||m.fingerprint!==published.fingerprint||m.search!==frame.contentWindow.location.search)return;
  const q=new URLSearchParams(m.search);for(const key of ["source","organ","inventory_q","inventory_status","inventory_table"]){if(q.has(key))state[key]=q.get(key);else delete state[key];}
 }else return;
 writeUrl();remember();
});
function start(){
 const q=Object.fromEntries(new URL(location.href).searchParams),requested=q.view||"registration";
 const view=aliases[requested]||requested;let next={...q,view};
 if(!data.workspaces.some(w=>w.id===view)){empty(next,"This view is unavailable. Choose an analysis to see its published results.");return;}
 if(q.dataset){
  const row=byId.get(q.dataset);
  if(!row||(!row.global_context&&["source","organ","subject","reconstruction"].some(k=>q[k]&&q[k]!==String(row[{source:"source_id",organ:"organ",subject:"subject_id",reconstruction:"reconstruction_id"}[k]])))||
     (["registration","atlas","topology","serial-stack"].includes(requested)&&row.analysis!=="registration")){
   empty(next,"This bookmarked result is unavailable in the requested scope. Its identity has not been replaced.");return;
  }
  use(row,{...q,dataset:row.id});return;
 }
 if(q.reconstruction&&!entries.some(e=>e.reconstruction_id===q.reconstruction&&(!q.source||q.source===e.source_id)&&(!q.organ||q.organ===e.organ))){empty(next,"This reconstruction is outside the published serial-tissue scope.");return;}
 if(q.source&&!entries.some(e=>e.analysis===view&&e.source_id===q.source)&&!catalogViews.has(requested)){empty(next,"This source has no published result in the requested analysis.");return;}
 if(catalogViews.has(requested)&&!next.source){const first=all.find(e=>e.catalog===requested);if(first){next.source=first.source_id;next.organ=next.organ||first.organ;}}
 if(legacyModes[requested])next.mode=legacyModes[requested];
 else if(requested==="registration"&&q.subject&&!q.mode)next.mode="alignment";
 else if(requested==="stain"&&q.subject&&!q.mode&&entries.some(e=>e.legacy_view==="stain"&&e.subject_id===q.subject))next.mode="qc";
 else if(requested==="protein"&&q.subject&&!q.mode&&entries.some(e=>e.legacy_view==="protein"&&e.subject_id===q.subject))next.mode="maps";
 next.mode=next.mode||data.workspaces.find(w=>w.id===view).modes[0][0];
 if(legacyModes[requested]){const native=entries.find(e=>e.legacy_view===requested&&(e.global_context||((!q.subject||q.subject===e.subject_id)&&(!q.source||q.source===e.source_id)&&(!q.organ||q.organ===e.organ))));if(native){use(native,{...next,dataset:native.id});return;}}
 const resolved=resolve(next);
 const matches=resolved&&(!q.subject||q.subject===resolved.subject_id)&&(!q.source||q.source===resolved.source_id)&&(!q.organ||q.organ===resolved.organ);
 use(resolved,matches?{...next,dataset:resolved.id}:next);
}
addEventListener("popstate",start);start();
globalThis.histopiaAnalysisReview={getState:()=>({...state,unavailable}),coverage:data.coverage};
})();
