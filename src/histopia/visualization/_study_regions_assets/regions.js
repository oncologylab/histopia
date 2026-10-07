"use strict";
(() => {
  const data = window.HISTOPIA_REGIONS;
  const $ = id => document.getElementById(id);
  if (!data) { $("selection").textContent = "Region data could not be loaded. Open all-regions.svg for the static outlines."; return; }
  const regions = [...data.regions].sort((a,b) => b.area_um2-a.area_um2 || a.region_id.localeCompare(b.region_id));
  const params = new URLSearchParams(location.search);
  let index = Math.max(0, regions.findIndex(r => r.region_id === params.get("region")));
  const kinds=new Set(data.summaries.map(r=>r.evidence_kind));
  for(const option of [...$("evidence").options])if(!kinds.has(option.value))option.remove();
  if(kinds.has(params.get("evidence")))$("evidence").value=params.get("evidence");
  $("evidence").disabled=!kinds.size;
  const spacing=data.pixel_size_um_xy||[data.pixel_size_um,data.pixel_size_um];
  const ratio=spacing[1]/spacing[0],displayHeight=data.height*ratio;
  $("outline").setAttribute("transform",`scale(1,${ratio})`);
  if(params.get("embedded")==="1")document.querySelector("header").hidden=true;
  const ns = "http://www.w3.org/2000/svg";
  $("tissue").setAttribute("viewBox", `0 0 ${data.width} ${displayHeight}`);
  $("tissue-background").setAttribute("width", data.width);
  $("tissue-background").setAttribute("height", displayHeight);
  if (data.background) $("tissue-background").setAttribute("href", data.background);
  $("background").disabled = !data.background;
  $("scope").textContent = data.summary_scope;
  regions.forEach((r,i) => { const option=document.createElement("option");option.value=r.region_id;option.textContent=`${i+1} · class ${r.semantic_class} · ${(r.area_um2/1e6).toFixed(3)} mm²`;$("region").append(option); });
  const number = n => n == null ? "—" : Number(n).toPrecision(3);
  function render() {
    if (!regions.length) { $("selection").textContent="No supported tissue regions."; ["previous","next","export","region"].forEach(id => $(id).disabled=true); return; }
    const r=regions[index];$("region").value=r.region_id;
    $("outline").setAttribute("d",r.path);$("outline").setAttribute("stroke",r.color);$("outline").dataset.regionId=r.region_id;
    $("selection").textContent=`${index+1} / ${regions.length} · class ${r.semantic_class}`;$("selection").title=r.region_id;
    $("annotation").textContent=r.annotation_status==="pending_lab_review"?"Cell labels pending review":`Annotation: ${r.annotation}`;
    $("previous").disabled=index===0;$("next").disabled=index===regions.length-1;
    const url=new URL(location.href);url.searchParams.set("region",r.region_id);url.searchParams.set("evidence",$("evidence").value);
    try { history.replaceState(null,"",url); } catch (_) { /* file: browser fallback */ }
    if(parent!==window)parent.postMessage({type:"histopia-region-selection",region:r.region_id,evidence:$("evidence").value},location.origin);
    $("summary").replaceChildren();
    const rows=data.summaries.filter(s=>s.region_id===r.region_id && s.evidence_kind===$("evidence").value);
    rows.forEach(row => { const tr=document.createElement("tr");tr.dataset.regionId=r.region_id;[row.protein_id,row.cell_count,row.supported_coverage==null?"—":`${(100*row.supported_coverage).toFixed(1)}%`,number(row.mean),number(row.median),number(row.dispersion_sd)].forEach(value=>{const td=document.createElement("td");td.textContent=value;tr.append(td);});tr.addEventListener("click",()=>$("outline").focus());$("summary").append(tr); });
    $("empty").textContent=rows.length?"":"No supported values for this region and evidence source.";
    $("tissue-background").style.display=$("background").checked?"":"none";
  }
  const physicalWidth=data.width*data.pixel_size_um;
  const length=10**Math.floor(Math.log10(physicalWidth/4));
  const line=document.createElementNS(ns,"path");line.setAttribute("d",`M${data.width*.05},${displayHeight*.93}h${length/data.pixel_size_um}`);line.setAttribute("stroke","#152b43");line.setAttribute("stroke-width",data.width*.003);$("scale").append(line);
  const text=document.createElementNS(ns,"text");text.setAttribute("x",data.width*.05);text.setAttribute("y",displayHeight*.9);text.setAttribute("font-size",data.width*.025);text.textContent=`${length} µm`;$("scale").append(text);
  $("region").addEventListener("change",()=>{index=regions.findIndex(r=>r.region_id===$("region").value);render();});
  $("previous").addEventListener("click",()=>{index=Math.max(0,index-1);render();});
  $("next").addEventListener("click",()=>{index=Math.min(regions.length-1,index+1);render();});
  $("background").addEventListener("change",render);$("evidence").addEventListener("change",render);
  $("export").addEventListener("click",()=>{const svg=$("tissue").cloneNode(true);svg.setAttribute("xmlns",ns);const metadata=document.createElementNS(ns,"metadata");metadata.textContent=JSON.stringify({region_id:regions[index].region_id,region_fingerprint:data.fingerprint,study_fingerprint:data.study_fingerprint,coordinate_units:"um",pixel_size_um:data.pixel_size_um,pixel_size_um_xy:spacing,evidence_kind:$("evidence").value});svg.prepend(metadata);const url=URL.createObjectURL(new Blob([new XMLSerializer().serializeToString(svg)],{type:"image/svg+xml"}));const link=document.createElement("a");link.href=url;link.download=regions[index].region_id+".svg";link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
  render();
})();
