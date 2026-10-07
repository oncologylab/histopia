"use strict";
const data=globalThis.HISTOPIA_RESULTS_CATALOG;
if(!data)throw new Error("Missing results catalog");
const $=id=>document.getElementById(id);
const labels={ready:"Ready to review",running:"Running",pending:"Pending",failed:"Needs attention"};
const query=new URL(location.href).searchParams;
const strict=!!data.eligibility_policy;
const benchmark=!!data.benchmark_policy;
const internal=!!data.input_review_policy;
if(internal)labels.ready="Provisional · 2D";
const scanId=d=>internal?d.input_scan_id:d.physical_section_id;
const title=v=>String(v).replace(/[-_]/g," ").replace(/^./,c=>c.toUpperCase());
const specimen=d=>d.specimen_id||d.subject_id||"Cohort";
const kind=d=>d.evidence_kind||(/inventory|cohort|development mice|not staged/i.test(d.subject_id)?"inventory":d.media.length?"image":"inventory");
const score=d=>(d.featured?100:0)+(kind(d)==="stack"?100:kind(d)==="image"?40:0)+(d.status==="ready"?10:0)+(/observed section|external IHC reference|external IHC segmentation/i.test(d.stage)?25:/section|tissue/i.test(d.stage)?8:0)-(/feature|geometry|accuracy|summary|cohort/i.test(d.stage)?5:0);
const ordered=rows=>rows.slice().sort((a,b)=>score(b)-score(a)||specimen(a).localeCompare(specimen(b),undefined,{numeric:true})||(a.section_order??0)-(b.section_order??0)||a.title.localeCompare(b.title));
let source=query.get("source")||data.sources[0]?.id||"";
let organ=query.get("organ")||sourceOrgans()[0]||data.organs[0]||"";
let selected=query.get("dataset"),field=Math.max(0,Number.parseInt(query.get("field"),10)||0),stage=query.get("stage")||"";
let subject=query.get("subject")||"",section=query.get("section")||"",collection=query.get("collection")||"specimens",current=null;
let reconstruction=query.get("reconstruction")||"",stackSection=query.get("stack_section")||"",stackMode=query.get("stack_mode")||"";
let region=query.get("region")||"",evidence=query.get("evidence")||"";
const selectionNotice=text("p","","selection-notice");selectionNotice.hidden=true;selectionNotice.setAttribute("role","status");$("app").prepend(selectionNotice);
let selectionTimer;
let unavailableRequest=!!((source&&!data.sources.some(s=>s.id===source))||(selected&&!data.datasets.some(d=>d.id===selected&&d.source_id===source))||(reconstruction&&!data.datasets.some(d=>d.reconstruction_id===reconstruction&&d.source_id===source&&d.organ===organ)));
const initial=data.datasets.find(d=>d.id===selected&&d.source_id===source&&d.organ===organ);
if(initial){subject=specimen(initial);collection=["summary","inventory"].includes(kind(initial))?"summaries":"specimens";}
function text(tag,value,className){const el=document.createElement(tag);el.textContent=String(value??"—");if(className)el.className=className;return el;}
function link(label,href){const a=text("a",label);a.href=href;if(href.startsWith("https:")){a.target="_blank";a.rel="noopener noreferrer";}return a;}
function sourceOrgans(){const present=new Set(data.datasets.filter(d=>d.source_id===source).map(d=>d.organ));return data.organs.filter(o=>present.has(o));}
function sync(){const values={region:current?.analysis_view?region||null:null,evidence:current?.analysis_view?evidence||null:null,source,organ,subject:subject||null,reconstruction:reconstruction||null,stack_section:stackSection||null,stack_mode:stackMode||null,section:section||null,stage:stage||null,collection,dataset:selected,field:String(field)};const u=new URL(location.href);for(const [k,v]of Object.entries(values)){if(v!=null)u.searchParams.set(k,v);else u.searchParams.delete(k);}history.replaceState(null,"",u);if(parent!==window)parent.postMessage({type:"histopia-catalog-selection",catalog_fingerprint:data.fingerprint,...values},location.origin);}
function baseRows(){return data.datasets.filter(d=>d.source_id===source&&d.organ===organ);}
function collectionRows(){return strict?baseRows():baseRows().filter(d=>(["summary","inventory"].includes(kind(d)))===(collection==="summaries"));}
function reset(){region="";evidence="";selected=null;subject="";section="";field=0;stage="";reconstruction="";stackSection="";stackMode="";unavailableRequest=false;}
function selectSource(id){source=id;reset();collection="specimens";if(!sourceOrgans().includes(organ))organ=sourceOrgans()[0]||organ;render();}
for(const s of data.sources){const label=s.label.replace(" · Our data","");const b=text("button",label);b.type="button";b.dataset.source=s.id;b.title=s.external?"External source":"Our data";b.onclick=()=>selectSource(s.id);$("sources").append(b);$("source-select").add(new Option((s.external?"External · ":"Our data · ")+label,s.id));}
$("source-select").onchange=e=>selectSource(e.target.value);
function render(){
 $("source-select").value=source;
 document.querySelectorAll("[data-source]").forEach(b=>b.setAttribute("aria-pressed",String(b.dataset.source===source)));
 $("organs").replaceChildren();for(const o of sourceOrgans()){const b=text("button",title(o));b.type="button";b.dataset.organ=o;b.setAttribute("aria-pressed",String(o===organ));b.onclick=()=>{organ=o;reset();collection="specimens";render();};$("organs").append(b);}
 let available=ordered(collectionRows());
 if(strict||benchmark||internal){collection="specimens";$("collection").closest("label").hidden=true;}
 if(unavailableRequest)available=[];
 // Sources without staged images open their explicit inventory, never another source.
 if(!strict&&!unavailableRequest&&!available.length&&collection==="specimens"&&baseRows().length){collection="summaries";available=ordered(collectionRows());}
 $("collection").value=collection;
 const subjects=[...new Set(available.map(specimen))],memoryKey=`histopia-catalog-subject:${strict?"serial":internal?"internal":"external"}:${source}:${organ}`;
 if(available.length&&!subjects.includes(subject)&&!unavailableRequest){
  const previous=subject;let remembered;try{remembered=sessionStorage.getItem(memoryKey);}catch{}
  const explicit=available.find(d=>d.id===selected)||available.find(d=>reconstruction&&d.reconstruction_id===reconstruction);
  subject=explicit?specimen(explicit):subjects.includes(remembered)?remembered:specimen(available[0]);
  if(previous){section="";stage="";reconstruction="";region="";evidence="";}
  if(previous){selectionNotice.textContent=`Showing specimen ${subject}; no result for ${previous}.`;selectionNotice.hidden=false;clearTimeout(selectionTimer);selectionTimer=setTimeout(()=>{selectionNotice.hidden=true;},8000);}
 }
 if(subjects.includes(subject))try{sessionStorage.setItem(memoryKey,subject);}catch{}
 $("subject").replaceChildren(...subjects.map(s=>new Option(s,s)));$("subject").value=subject;
 const specimenRows=available.filter(d=>specimen(d)===subject),stacks=[...new Set(specimenRows.map(d=>d.reconstruction_id).filter(Boolean))];
 if(strict&&!reconstruction)reconstruction=specimenRows.find(d=>d.id===selected)?.reconstruction_id||stacks[0]||"";
 $("reconstruction-label").hidden=!strict||stacks.length<2;$("reconstruction").replaceChildren(...stacks.map(s=>new Option(specimenRows.find(d=>d.reconstruction_id===s)?.block_id||s,s)));$("reconstruction").value=reconstruction;
 const scoped=specimenRows.filter(d=>!strict||d.reconstruction_id===reconstruction),sections=[...new Set(scoped.map(scanId).filter(Boolean))].sort((a,b)=>internal?a.localeCompare(b,undefined,{numeric:true}):(scoped.find(d=>scanId(d)===a)?.section_order??0)-(scoped.find(d=>scanId(d)===b)?.section_order??0));
 $("section").replaceChildren(new Option(internal?"All scans":"All sections",""),...sections.map(s=>new Option(internal?scoped.find(d=>scanId(d)===s)?.input_scan_label||s:s.replace(/^WD-76845-/,"Section "),s)));$("section").value=section;$("section-label").hidden=!sections.length;
 const inSection=scoped.filter(d=>!section||scanId(d)===section);const stages=[...new Set(inSection.map(d=>d.stage))];
 $("stage").replaceChildren(new Option("All analyses",""),...stages.map(s=>new Option(s,s)));$("stage").value=stage;
 const matches=inSection.filter(d=>!stage||d.stage===stage);current=matches.find(d=>d.id===selected)||matches[0]||null;selected=current?.id||(unavailableRequest?selected:null);
 $("empty").textContent=unavailableRequest?(internal?"This dataset is unavailable in Internal tissues.":benchmark?"This dataset is unavailable in External validation.":"This dataset is outside the serial IHC review scope. Choose an available tissue stack."):strict?"No eligible serial IHC stacks are available for this selection.":"No results are available for this source and organ yet.";
 $("list-title").textContent=collection==="summaries"?"Summaries":"Evidence";$("count").textContent=`· ${matches.length}`;$("datasets").replaceChildren();$("empty").hidden=!!matches.length;$("detail").hidden=!current;
 for(const d of matches){const b=document.createElement("button");b.type="button";b.className="dataset";b.dataset.dataset=d.id;b.setAttribute("aria-pressed",String(d.id===selected));if(d.media.length){const im=document.createElement("img");im.src=d.media[0].href;im.alt="";im.loading="lazy";b.append(im);}const copy=document.createElement("span");copy.append(text("b",d.short_title||d.title),text("small",labels[d.status]));b.append(copy);b.onclick=()=>{selected=d.id;field=0;region="";evidence="";stackSection="";stackMode="";render();};$("datasets").append(b);}
 if(current)detail();sync();
}
function facts(target,values){$(target).replaceChildren();for(const [k,v]of values){const group=document.createElement("div");group.append(text("dt",k),text("dd",v));$(target).append(group);}}
function shortSentence(value,max=170){const sentence=String(value||"").split(/(?<=[.!?])\s+/)[0];if(sentence.length<=max)return sentence;return sentence.slice(0,max-1).replace(/\s+\S*$/,"")+"…";}
function detail(){
 const d=current,s=data.sources.find(s=>s.id===source);$("identity").textContent=`${s.external?"External":"Our data"} / ${title(organ)} / ${specimen(d)}`;$("title").textContent=d.short_title||d.title;$("status").textContent=labels[d.status];$("status").className="badge "+d.status;
 $("summary").textContent=shortSentence(d.summary_short||d.summary);$("full-summary").textContent=d.summary;$("source-description").textContent=s.description;$("fingerprint").textContent=d.fingerprint;
 facts("facts",Object.entries(d.facts||{}).filter(([k,v])=>!/fingerprint|sha256|image identity|source image|path|archive|file/i.test(k)&&String(v).length<90).slice(0,4));facts("all-facts",Object.entries(d.facts||{}));
 $("links").replaceChildren();for(const item of [...d.links,...d.downloads]){const a=link(item.label,item.href);if(item.top)a.target="_top";$("links").append(a);}
 const viewer=d.analysis_view?d.analysis_view+"?embedded=1"+(region?"&region="+encodeURIComponent(region):"")+(evidence?"&evidence="+encodeURIComponent(evidence):""):d.reconstruction_view?d.reconstruction_view+(d.reconstruction_view.includes("?")?"&":"?")+"embedded=1"+(stackSection?"&section="+encodeURIComponent(stackSection):"")+(stackMode?"&mode="+encodeURIComponent(stackMode):""):null;$("stack-preview").hidden=!viewer;
 $("stack-preview").title=d.analysis_view?"Tissue region expression summaries":"Serial tissue reconstruction";
 if(viewer){if($("stack-preview").getAttribute("src")!==viewer)$("stack-preview").src=viewer;}else $("stack-preview").removeAttribute("src");
 $("evidence").hidden=!d.media.length||!!viewer;field=Math.min(field,Math.max(0,d.media.length-1));$("field").replaceChildren(...d.media.map((m,i)=>new Option(m.label||`Evidence ${i+1}`,String(i))));showField();
 const table=d.table||[];$("metrics").hidden=!table.length;$("table").replaceChildren();if(table.length){const keys=[...new Set(table.flatMap(r=>Object.keys(r)))];const tr=document.createElement("tr");keys.forEach(k=>tr.append(text("th",k)));const thead=document.createElement("thead");thead.append(tr);const tbody=document.createElement("tbody");for(const row of table){const tr=document.createElement("tr");for(const key of keys){const v=row[key];tr.append(text("td",typeof v==="number"&&!Number.isInteger(v)?v.toPrecision(4):v));}tbody.append(tr);}$("table").append(thead,tbody);}
 let saved={};try{saved=JSON.parse(localStorage.getItem(noteKey())||"{}");}catch{}$("review-state").value=saved.assessment||"unreviewed";$("note").value=saved.note||"";$("note-state").textContent="Browser-local draft; upstream approvals stay intact.";
}
function showField(){if(!current?.media.length)return;const m=current.media[field];$("field").value=String(field);$("image").src=m.href;$("image").alt=`${current.title} · ${m.label||"Evidence"}`;$("open-image").href=m.href;$("caption").textContent=m.caption||"";$("caption-short").textContent=m.label||"";$("previous").disabled=field===0;$("next").disabled=field===current.media.length-1;$("image-viewport").scrollTo(0,0);}
$("subject").onchange=e=>{subject=e.target.value;selected=null;section="";stage="";field=0;reconstruction="";stackSection="";stackMode="";unavailableRequest=false;render();};
$("reconstruction").onchange=e=>{reconstruction=e.target.value;selected=null;section="";stage="";field=0;unavailableRequest=false;render();};
$("section").onchange=e=>{section=e.target.value;selected=null;stage="";field=0;render();};
$("stage").onchange=e=>{stage=e.target.value;selected=null;field=0;render();};
$("collection").onchange=e=>{collection=e.target.value;reset();render();};
$("field").onchange=e=>{field=Number(e.target.value);showField();sync();};
$("previous").onclick=()=>{field--;showField();sync();};$("next").onclick=()=>{field++;showField();sync();};
$("native").onclick=()=>{const active=$("image-viewport").classList.toggle("native");$("native").setAttribute("aria-pressed",String(active));$("native").textContent=active?"Fit image":"Native size";};
function noteKey(){return "histopia-review-draft:"+current.id+":"+current.fingerprint;}
function saveNote(){if(!current)return;try{localStorage.setItem(noteKey(),JSON.stringify({dataset_id:current.id,source_id:source,organ,fingerprint:current.fingerprint,assessment:$("review-state").value,note:$("note").value,updated_at:new Date().toISOString(),scientific_approval:false}));$("note-state").textContent="Draft saved in this browser. Export it to share for review.";}catch{$("note-state").textContent="Browser storage is unavailable. Export your draft to retain it.";}}
$("review-state").onchange=saveNote;$("note").oninput=saveNote;
$("export-notes").onclick=()=>{saveNote();const notes=[];for(const d of data.datasets){try{const value=localStorage.getItem("histopia-review-draft:"+d.id+":"+d.fingerprint);if(value)notes.push(JSON.parse(value));}catch{}}const blob=new Blob([JSON.stringify({catalog_fingerprint:data.fingerprint,notes,scientific_approval:false},null,2)],{type:"application/json"});const url=URL.createObjectURL(blob),a=document.createElement("a");a.href=url;a.download="histopia-review-notes.json";a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};

const stamp=new Date(data.updated_at);$("updated").textContent="Updated "+(Number.isFinite(stamp.getTime())?stamp.toLocaleString([],{month:"short",day:"numeric",hour:"2-digit",minute:"2-digit"}):data.updated_at);
function renderCompute(nodes,updatedAt){
 const active=nodes.filter(n=>n.status==="running"&&n.active_job).length,totals=nodes.find(n=>n.queue_totals)?.queue_totals;
 const attention=totals?(totals.failed||0)+(totals.blocked||0):0;
 $("compute-count").textContent=`· ${active} nodes working`+(totals?` · ${totals.pending||0} queued · ${totals.complete||0} complete`:"")+(attention?` · ${attention} need attention`:"");
 $("compute-rows").replaceChildren();
 for(const n of nodes){const box=document.createElement("div");box.className="node";box.append(text("b",n.node),text("small",`${n.status} · ${n.work||n.active_job||"No active job"}`));if(n.progress)box.append(text("small",n.progress));if(n.updated_at)box.append(text("small",n.updated_at));$("compute-rows").append(box);}
 if(updatedAt)$("compute").title="Compute checked "+updatedAt;
}
renderCompute(data.compute,data.updated_at);
if(internal&&!data.compute.length)$("compute").hidden=true;
if(data.compute.some(n=>n.queue_totals)){
 async function refreshCompute(){try{const response=await fetch("compute-status.json",{cache:"no-store"});if(response.ok){const update=await response.json();renderCompute(update.compute,update.updated_at);}}catch{}}
 refreshCompute();setInterval(refreshCompute,30000);
}
$("refresh").onclick=()=>location.reload();if(matchMedia("(max-width:850px)").matches)$("browse").open=false;$("app").hidden=false;render();

addEventListener("message",event=>{
 if(event.origin===location.origin&&event.source===$("stack-preview").contentWindow&&event.data?.type==="histopia-region-selection"&&current?.analysis_view){
  if(typeof event.data.region==="string")region=event.data.region;
  if(typeof event.data.evidence==="string")evidence=event.data.evidence;
  sync();return;
 }
 if(event.origin!==location.origin||event.source!==$("stack-preview").contentWindow||event.data?.type!=="histopia-serial-selection"||!current?.reconstruction_view)return;
 if(event.data.source!==current.source_id||event.data.organ!==current.organ||event.data.subject!==current.subject_id)return;
 if(typeof event.data.section==="string")stackSection=event.data.section;
 if(["section","3d"].includes(event.data.mode))stackMode=event.data.mode;
 sync();
});
