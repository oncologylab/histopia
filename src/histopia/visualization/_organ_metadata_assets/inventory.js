"use strict";
const inventory=globalThis.HISTOPIA_ORGAN_INVENTORY;
const $=id=>document.getElementById(id);
const sources=new Map(inventory.sources.map(s=>[s.id,s.label]));
const primary=new Map((inventory.primary_tables||[]).map(t=>[t.organ,t]));
let organ="",page=0;
const PAGE_SIZE=40;
const title=s=>s==="unassigned"?"Unassigned":s[0].toUpperCase()+s.slice(1);
const number=n=>n.toLocaleString("en-US");
const node=(tag,text,cls)=>{const el=document.createElement(tag);if(text!=null)el.textContent=text;if(cls)el.className=cls;return el;};
$('updated').textContent=inventory.updated_at.slice(0,10);
$('scope').textContent=inventory.scope;
for(const [id,label] of sources){const option=node('option',label);option.value=id;$('source').append(option);}
const organItems=[{organ:'',scans:inventory.rows.length},...inventory.organs];
for(const item of organItems){
  const button=node('button',item.organ?title(item.organ):'All organs');button.type='button';button.dataset.organ=item.organ;
  const curatedTable=primary.get(item.organ);
  button.append(node('strong',number(curatedTable?curatedTable.source_rows:item.scans)),node('small',curatedTable?'curated rows':'scan files'));
  button.onclick=()=>{organ=item.organ;page=0;$('metadata-table').value=primary.has(organ)?'curated':'inventory';if(primary.has(organ)&&$('source').value&&$('source').value!==primary.get(organ).source_id)$('metadata-table').value='inventory';render(true);};$('organs').append(button);
}
for(const item of inventory.attachments){
  const a=node('a',item.label);a.href=item.href;a.download='';$('extra-downloads').append(a);
  if(item.href.endsWith('.xlsx')){$('workbook').hidden=false;$('workbook').href=item.href;}
}
const all=node('a','All scans · CSV');all.href='downloads/all-organs.csv';all.download='';$('extra-downloads').prepend(all);
const fs=inventory.file_summary;
$('file-summary').textContent=fs.total_files?`${number(fs.total_files)} source file entries retained in the file index, including conversions, aliases and unresolved image files.`:'';
function readUrl(){
  const params=new URL(location.href).searchParams;
  organ=params.get('organ')||'';
  // Unknown requested contexts stay visible as an empty selection, never fall back.
  $('source').value=params.get('source')||'';
  if(params.get('source')&&!sources.has(params.get('source'))){const o=node('option',params.get('source')+' (unavailable)');o.value=params.get('source');$('source').append(o);$('source').value=o.value;}
  $('availability').value=params.get('inventory_status')||'';
  $('search').value=params.get('inventory_q')||params.get('subject')||'';
  $('metadata-table').value=params.get('inventory_table')||(primary.has(organ)&&(!$('source').value||$('source').value===primary.get(organ).source_id)?'curated':'inventory');
}
function sync(push){
  const url=new URL(location.href);
  for(const key of ['source','organ','subject','dataset','field','collection','section','reconstruction','mouse','cohort','inventory_q','inventory_status','inventory_table','stage','region','evidence','stack_section','stack_mode'])url.searchParams.delete(key);
  for(const [key,value] of [['organ',organ],['source',$('source').value],['inventory_q',$('search').value],['inventory_status',$('availability').value]])if(value)url.searchParams.set(key,value);
  if(primary.has(organ))url.searchParams.set('inventory_table',$('metadata-table').value);
  if(push&&window.parent===window&&url.href!==location.href)history.pushState(null,'',url);else history.replaceState(null,'',url);
  window.parent.postMessage({type:'histopia-organ-metadata',fingerprint:inventory.fingerprint,search:url.search,push:!!push},location.origin);
}
function detail(row){
  const details=node('details',null,'record');details.append(node('summary','Metadata'));
  const dl=node('dl');
  for(const [key,label] of [['scan_id','Scan identity'],['metadata_source','Metadata source'],['metadata_rows','Source rows'],['table_label','Slide label'],['table_order','Table order'],['table_order_text','Table order text'],['antibody_type','Antibody type'],['table_note','Table note'],['analysis_marker','Existing result marker'],['metadata_conflicts','Source conflicts'],['subject_evidence','Mouse evidence'],['organ_evidence','Organ evidence'],['stain_evidence','Stain evidence'],['copy_count','Exact file copies'],['local_status','Local input'],['mouse_role','Study role'],['section_order','Registered order'],['order_evidence','Order provenance'],['z_um','Z position (µm)'],['z_spacing_kind','Spacing type'],['z_evidence','Spacing evidence'],['reconstruction_id','Reconstruction'],['legacy_derivative_count','Legacy file links'],['next_step','Next step']]){
    dl.append(node('dt',label),node('dd',row[key]??'Unknown'));
  }
  details.append(dl);return details;
}
function render(push=false){
  const source=$('source').value,availability=$('availability').value,q=$('search').value.trim().toLowerCase();
  const table=primary.get(organ),curated=!!table&&$('metadata-table').value==='curated';
  $('table-choice').hidden=!table;
  if(table)$('metadata-table').querySelector('[value="curated"]').textContent=table.label;
  $('table-source').hidden=!curated;
  if(curated)$('table-source').textContent=`${table.filename} · ${table.subjects} mice`;
  $('order-heading').textContent=curated?'Table order':'Section';
  const rows=(curated?table.rows:inventory.rows).filter(r=>(!organ||r.organ===organ)&&(!source||r.source_id===source)&&
    (!availability||(availability==='results'?!!r.review_href:!r.review_href))&&
    (!q||[r.subject_id,r.filename,r.marker,r.stain,r.scan_id].join(' ').toLowerCase().includes(q)));
  page=Math.min(page,Math.max(0,Math.ceil(rows.length/PAGE_SIZE)-1));
  const ready=rows.filter(r=>r.review_href).length;
  const unique=new Set(rows.map(r=>r.scan_id).filter(Boolean)).size,missing=rows.length-ready;
  $('coverage').textContent=curated?`${number(rows.length)} table row${rows.length===1?'':'s'} · ${number(unique)} scan file${unique===1?'':'s'} · ${number(missing)} row${missing===1?'':'s'} without review links`:`${number(rows.length)} scan files · ${number(ready)} with review links · ${number(missing)} metadata only`;
  $('organ-csv').href='downloads/'+(organItems.some(o=>o.organ===organ)&&organ?organ:'all-organs')+(table&&!curated?'-inventory':'')+'.csv';
  $('organ-csv').textContent=(curated?'Curated '+title(organ):organ?title(organ):'All scans')+' · CSV';
  $('organ-csv').title='Complete organ table, including rows hidden by other filters';
  for(const b of $('organs').children)b.setAttribute('aria-pressed',String(b.dataset.organ===organ));
  $('rows').replaceChildren();
  for(const r of rows.slice(page*PAGE_SIZE,(page+1)*PAGE_SIZE)){
    const tr=node('tr');tr.dataset.scan=r.scan_id||'';if(curated)tr.dataset.metadataRow=r.metadata_row;
    const mouse=node('td',r.subject_id||'Unassigned');
    if((r.subject_evidence||'').includes('filename'))mouse.append(node('small','Filename identity'));
    const scan=node('td',null,'scan-name');scan.append(node('span',r.filename),node('span',[r.stain,r.marker].filter(Boolean).join(' · '),'stain'),detail(r));
    if(curated&&r.metadata_duplicate_scan)scan.append(node('small','Repeated scan in source table'));
    const section=node('td',(curated?r.table_order:r.section_order)??'Unknown');
    section.dataset.label=curated?'Table order':'Section';
    if(curated&&r.section_order!=null)section.append(node('small','Registered '+r.section_order));
    if(r.z_spacing_kind==='assumed')section.append(node('small','Assumed Z'));
    const stages=[['registration_status','Registration'],['cell_status','Cells'],['stain_status','Stain'],['feature_status','UNI-2h'],['semantic_status','Regions']].filter(([k])=>['available','provisional','archived'].includes(r[k])).map(([,v])=>v);
    const results=node('td',stages.length?stages.join(' · '):'Metadata only','stages'+(stages.length?'':' muted'));
    if(r.review_status==='provisional')results.append(node('small','Provisional QC'));
    const review=node('td');
    if(r.review_href){const label=r.review_status==='existing serial review'?'Open stack':r.review_status==='existing input review'?'Open specimen':'Open images';const a=node('a',label,'open');a.href=r.review_href;a.target='_top';review.append(a);}else review.append(node('span','Pending','muted'));
    tr.append(mouse,node('td',sources.get(r.source_id)),scan,section,results,review);$('rows').append(tr);
  }
  if(!rows.length){const td=node('td','No scans match these filters.','empty');td.colSpan=6;const tr=node('tr');tr.append(td);$('rows').append(tr);}
  $('previous').disabled=page===0;$('next').disabled=(page+1)*PAGE_SIZE>=rows.length;
  $('page').textContent=rows.length?`${page*PAGE_SIZE+1}–${Math.min((page+1)*PAGE_SIZE,rows.length)} of ${number(rows.length)}`:'0 scans';
  sync(push);
}
$('availability').onchange=()=>{page=0;render(true);};
$('metadata-table').onchange=()=>{page=0;const t=primary.get(organ);if(t&&$('metadata-table').value==='curated'&&$('source').value&&$('source').value!==t.source_id)$('source').value=t.source_id;render(true);};
$('source').onchange=()=>{page=0;const t=primary.get(organ);if(t&&$('source').value&&$('source').value!==t.source_id)$('metadata-table').value='inventory';render(true);};
let searchTimer;$('search').oninput=()=>{clearTimeout(searchTimer);searchTimer=setTimeout(()=>{page=0;render(true);},150);};
$('clear').onclick=()=>{organ='';$('source').value='';$('availability').value='';$('search').value='';page=0;render(true);};
$('previous').onclick=()=>{page--;render();};$('next').onclick=()=>{page++;render();};
addEventListener('popstate',()=>{readUrl();page=0;render();});
readUrl();render();
