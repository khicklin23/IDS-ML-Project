import {human,count,escapeHTML as esc} from './data.mjs?v=sona-release-clarity-4';
import {primaryColumns,packetColumns,unpackRows,filterRaw} from './raw-data.mjs?v=sona-release-clarity-4';
const $=id=>document.getElementById(id);
let rows=[],manifest,page=0,sort='event_time',descending=false;
let session=new URLSearchParams(location.search).get('session')??'';
if(session)$('show-provenance').checked=true;
function render() {
  const size=Number($('raw-page-size').value)||100;
  const selected=filterRaw(rows,{search:$('raw-search').value,family:$('raw-family').value,outcome:$('raw-outcome').value,session,sort,descending});
  const pages=Math.max(1,Math.ceil(selected.length/size));page=Math.max(0,Math.min(page,pages-1));
  const columns=[...primaryColumns,...($('show-packets').checked?packetColumns:[]),...($('show-provenance').checked?manifest.columns.filter(c=>!primaryColumns.includes(c)&&!packetColumns.includes(c)):[])].filter(c=>manifest.columns.includes(c));
  $('raw-head').innerHTML='<tr><th scope="col">Row</th>'+columns.map(column=>`<th scope="col" aria-sort="${column===sort?descending?'descending':'ascending':'none'}"><button data-column="${esc(column)}">${esc(human(column))}${column===sort?descending?' ↓':' ↑':''}</button></th>`).join('')+'</tr>';
  $('raw-body').innerHTML=selected.slice(page*size,(page+1)*size).map((row,index)=>'<tr><td class="muted">'+count(page*size+index+1)+'</td>'+columns.map(column=>`<td class="${['src_ip','dst_ip','session_id','flow_id'].includes(column)?'mono':''}">${esc(typeof row[column]==='number'&&!Number.isInteger(row[column])?row[column].toFixed(6):row[column])}</td>`).join('')+'</tr>').join('')||`<tr><td class="empty" colspan="${columns.length+1}">No records match these filters.</td></tr>`;
  $('raw-summary').textContent=`${selected.length?count(page*size+1)+'–'+count(Math.min((page+1)*size,selected.length)):'0'} of ${count(selected.length)} matching / ${count(rows.length)} total${session?' · session '+session:''}`;
  $('raw-page').value=page+1;$('raw-page').max=pages;$('raw-pages').textContent=`of ${count(pages)}`;$('raw-prev').disabled=page===0;$('raw-next').disabled=page===pages-1;
  $('raw-head').querySelectorAll('[data-column]').forEach(button=>button.addEventListener('click',()=>{const column=button.dataset.column;descending=column===sort?!descending:false;sort=column;page=0;render();}));
}
for(const id of ['raw-family','raw-outcome','raw-page-size','show-packets','show-provenance'])$(id).addEventListener('change',()=>{page=0;render();});
let searchTimer;$('raw-search').addEventListener('input',()=>{clearTimeout(searchTimer);searchTimer=setTimeout(()=>{page=0;render();},150);});
$('raw-prev').addEventListener('click',()=>{page--;render();});$('raw-next').addEventListener('click',()=>{page++;render();});
$('raw-page').addEventListener('change',()=>{page=Math.floor(Number($('raw-page').value)||1)-1;render();});
$('raw-reset').addEventListener('click',()=>{$('raw-search').value='';$('raw-family').value='';$('raw-outcome').value='';session='';history.replaceState(null,'','raw.html');page=0;render();});
async function read(url){const response=await fetch(url, {cache:'no-store'});if(!response.ok)throw new Error(`Cannot load ${url} (HTTP ${response.status}).`);return response.json();}
try {
  const [index,report]=await Promise.all([read('./raw-data/manifest.json'),read('./report.json')]);manifest=index;
  if(index.run_id!==report.run_id||index.row_count!==report.dataset.manifest.row_count||index.dataset_generated_at!==(report.dataset.manifest.generated_at ?? report.generated_at))throw new Error('Raw data and report refer to different runs. Rebuild the matching static dataset.');
  if(!Array.isArray(index.columns)||!Array.isArray(index.chunks))throw new Error('Invalid raw data manifest.');
  for(const name of index.chunks)if(!/^flows-\d{4}\.json$/.test(name))throw new Error('Invalid chunk filename.');
  let loaded=0;
  // Four concurrent reads bound request pressure while preserving chunk order.
  for(let start=0;start<index.chunks.length;start+=4) {
    const batch=await Promise.all(index.chunks.slice(start,start+4).map(async name=>{const data=unpackRows(index.columns,await read('./raw-data/'+name));loaded+=data.length;$('raw-status').textContent=`Loading raw data: ${count(loaded)} / ${count(index.row_count)} records…`;return data;}));
    rows.push(...batch.flat());
  }
  if(rows.length!==index.row_count)throw new Error('Incomplete raw dataset; record count does not match manifest.');
  for(const [id,column]of [['raw-family','scenario'],['raw-outcome','outcome']])$(id).innerHTML=`<option value="">All ${column==='scenario'?'families':'outcomes'}</option>`+[...new Set(rows.map(row=>row[column]))].sort().map(value=>`<option value="${esc(value)}">${esc(human(value))}</option>`).join('');
  $('raw-meta').textContent=`${index.run_id} · ${count(rows.length)} flows · ${index.columns.length} original columns${session?' · session '+session:''}`;
  $('raw-version').textContent=`Schema ${index.schema_version}`;
  $('raw-status').hidden=true;$('raw-browser').hidden=false;render();
}catch(error){$('raw-status').textContent=error.message;$('raw-status').className='status error';}
