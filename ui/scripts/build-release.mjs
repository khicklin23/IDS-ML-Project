import {readFile, readdir, writeFile, unlink} from 'node:fs/promises';
import {resolve, dirname} from 'node:path';
import {fileURLToPath} from 'node:url';
import {Script} from 'node:vm';
import {validateReport} from '../data.mjs';
import {validateVersions} from '../versions-data.mjs';
import {unpackRows} from '../raw-data.mjs';
import {validateResearch} from '../research-data.mjs';

export function nextRelease(names) {
  const numbers=names.map(name=>/^Release_1\.(\d+)\.html$/.exec(name)).filter(Boolean).map(m=>Number(m[1]));
  return `Release_1.${numbers.length ? Math.max(...numbers)+1 : 0}.html`;
}
const serialize = value => JSON.stringify(value).replaceAll('<','\\u003c').replaceAll('\u2028','\\u2028').replaceAll('\u2029','\\u2029');
const classic = text => text.replace(/^import .*?;\s*$/gm,'').replace(/^export /gm,'');
export async function packageRelease(ui) {
  const read = path => readFile(resolve(ui,path),'utf8');
  const report=validateReport(JSON.parse(await read('report.json'))), versions=validateVersions(JSON.parse(await read('versions.json')));
  const manifest=JSON.parse(await read('raw-data/manifest.json'));
  if(manifest.run_id!==report.run_id||manifest.row_count!==report.dataset.manifest.row_count||manifest.dataset_generated_at!==(report.dataset.manifest.generated_at??report.generated_at))throw new Error('Report and raw data do not match. Refresh them together before building.');
  const research=validateResearch(JSON.parse(await read('research.json')));
  const resources={'report.json':report,'versions.json':versions,'research.json':research,'raw-data/manifest.json':manifest};
  let total=0;const ids=new Set();
  for(const name of manifest.chunks){
    if(!/^flows-\d{4}\.json$/.test(name))throw new Error('Invalid raw chunk name.');
    const path='raw-data/'+name,rows=JSON.parse(await read(path));resources[path]=rows;
    for(const row of unpackRows(manifest.columns,rows)){if(ids.has(row.flow_id))throw new Error('Duplicate raw flow.');ids.add(row.flow_id);total++;}
  }
  if(total!==manifest.row_count)throw new Error('Incomplete raw dataset.');
  const common=classic(await read('data.mjs'))+'\nconst esc=escapeHTML;\n'+classic(await read('raw-data.mjs'))+'\n'+classic(await read('versions-data.mjs'))+'\n'+classic(await read('research-data.mjs'));
  const css=await read('styles.css'),pages={};
  for(const [page,entry] of [['index.html','app.js'],['raw.html','raw.js'],['model.html','model.js'],['progression.html','versions.js']]){
    let source=classic(await read(entry));
    if(entry==='raw.js') source=source.replace('location.search','window.__releaseSearch').replace("history.replaceState(null,'','raw.html')","parent.releaseReset()");
    const code=`(async()=>{\n${common}\n${source}\n})().catch(error=>{console.error(error);});`;
    new Script(code,{filename:entry});
    const bridge=`window.__releaseSearch=parent.releaseSearch;window.fetch=async(path)=>{const key=String(path).replace(/^\\.\\//,'').split('?')[0];const value=parent.releaseResources[key];if(value===undefined)throw new Error('Missing embedded data: '+key);return {ok:true,status:200,json:async()=>value};};document.addEventListener('click',event=>{const link=event.target.closest('a');if(!link)return;const href=link.getAttribute('href');if(!href||href.startsWith('#')||/^(https?:|mailto:)/.test(href))return;event.preventDefault();parent.releaseNavigate(href);});`;
    let html=await read(page);
    html=html.replace(/<link[^>]+rel="stylesheet"[^>]*>/g,()=>`<style>${css}</style>`).replace(/<script[^>]*src="[^"]+"[^>]*><\/script>/g,'');
    html=html.replace('</body>',()=>`<script>${bridge}\n${code.replace(/<\/script/gi,'<\\/script')}</script></body>`);
    pages[page]=html;
  }
  const payload=serialize({pages,resources});
  const runtime=`const bundle=JSON.parse(document.getElementById('release-data').textContent);window.releaseResources=bundle.resources;const frame=document.getElementById('workspace');window.releaseSearch='';window.releaseNavigate=function(target){if(!bundle.pages[target.split('?')[0]])return;location.hash=target;};window.releaseReset=function(){history.replaceState(null,'','#raw.html');};function render(){const target=location.hash.slice(1)||'index.html',page=target.split('?')[0];window.releaseSearch=target.includes('?')?'?'+target.split('?').slice(1).join('?'):'';frame.srcdoc=bundle.pages[page]||bundle.pages['index.html'];frame.title=page==='index.html'?'Sona overview':page==='raw.html'?'Sona raw data':page==='model.html'?'Sona model explained':'Sona model progression';}window.addEventListener('hashchange',render);render();`;
  new Script(runtime);
  const html=`<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Sona</title><style>html,body{margin:0;height:100%;background:#f4f5f7}iframe{display:block;border:0;width:100%;height:100dvh}</style></head><body><iframe id="workspace" title="Sona overview"></iframe><noscript>This interactive release requires JavaScript.</noscript><script type="application/json" id="release-data">${payload}</script><script>${runtime}</script></body></html>`;
  return {html,pages,resources};
}
export async function buildRelease(ui, output) {
  const {html}=await packageRelease(ui);
  const previous=await readdir(output);
  const filename=nextRelease(previous);
  await writeFile(resolve(output,filename),html,{flag:'wx'});
  await writeFile(resolve(output,'CURRENT_RELEASE.txt'),filename+'\n');
  for(const name of previous) if(/^Release_1\.\d+\.html$/.test(name)) await unlink(resolve(output,name));
  return {filename,bytes:Buffer.byteLength(html)};
}
if(process.argv[1]&&resolve(process.argv[1])===fileURLToPath(import.meta.url)) {
  const ui=resolve(dirname(fileURLToPath(import.meta.url)),'..');
  const result=await buildRelease(ui,resolve(ui,'..'));
  console.log(`Built ${result.filename} (${(result.bytes/1024/1024).toFixed(1)} MiB). Standalone: all four pages, code, styles and data are embedded.`);
}
