import test from 'node:test';
import assert from 'node:assert/strict';
import {Script,createContext} from 'node:vm';
import {fileURLToPath} from 'node:url';
import {packageRelease,DOWNLOAD_FILENAME,buildRelease} from '../scripts/build-release.mjs';
const bundle=await packageRelease(fileURLToPath(new URL('..',import.meta.url)));
test('download filename is a clear stable entry point',()=>{assert.equal(DOWNLOAD_FILENAME,'Download_Project_Here.html');});
test('standalone release embeds all pages and exactly the current raw dataset',()=>{
 assert.deepEqual(Object.keys(bundle.pages),['index.html','raw.html','model.html','progression.html','neurons.html']);
 const manifest=bundle.resources['raw-data/manifest.json'];
 assert.equal(manifest.chunks.reduce((n,name)=>n+bundle.resources['raw-data/'+name].length,0),87505);
 assert.equal(bundle.resources['report.json'].detector.model_version,'v4');
 assert.equal(bundle.resources['research.json'].public_model_replaced,false);
 assert.equal((bundle.html.match(/<\/script>/g)||[]).length,2);
 for(const page of Object.values(bundle.pages))assert.doesNotMatch(page,/<script[^>]*src=|<link[^>]*stylesheet/);
});
class Element {
 constructor(){this.value='';this.hidden=false;this.checked=false;this.innerHTML='';this.textContent='';this.style={};this.dataset={};this.parentElement=this;this.classList={add(){}};}
 addEventListener(){}setAttribute(){}querySelector(){return new Element();}querySelectorAll(){return [];}focus(){}scrollIntoView(){}
}
test('all packaged pages render using embedded data without network requests',async()=>{
 for(const [name,page] of Object.entries(bundle.pages)){
  const elements=new Map(),get=id=>{if(!elements.has(id))elements.set(id,new Element());return elements.get(id);};
  get('sort').value='time';get('raw-page-size').value='100';get('weight-bytes').value='.8';get('weight-duration').value='-.4';get('weight-outcome').value='1.2';
  const errors=[];
  const context=createContext({document:{getElementById:get,querySelectorAll:()=>[],addEventListener(){}},parent:{releaseResources:bundle.resources,releaseSearch:name==='raw.html'?'?session=run-7%3A%3Aport_scan-00000':'',releaseNavigate(){}},URLSearchParams,console:{error:e=>errors.push(e)},setTimeout,clearTimeout});
  context.window=context;
  const code=page.match(/<script>([\s\S]*?)<\/script>/)[1];
  await new Script(code).runInContext(context);
  assert.deepEqual(errors,[],name);
  for(const element of elements.values())assert.doesNotMatch(element.innerHTML+element.textContent,/\u2014/,name);
  assert.doesNotMatch(page,/\u2014|&mdash;|&#8212;/,name);
  if(name==='index.html'){assert.equal(get('dashboard').hidden,false);assert.match(get('metrics').innerHTML,/87,505/);}
  if(name==='raw.html'){assert.equal(get('raw-status').hidden,true);assert.match(get('raw-summary').textContent,/session run-7::port_scan-00000/);}
  if(name==='neurons.html'){assert.match(get('neuron-network').innerHTML,/First hidden layer/);assert.match(get('neuron-calculation').innerHTML,/Failed connections/);}
  if(name==='model.html')assert.match(get('model-network').innerHTML,/Sona neural network/);
  if(name==='progression.html'){assert.equal(get('versions-content').hidden,false);assert.match(get('version-detail').innerHTML,/<h3>Sona/);assert.match(get('version-comparison').innerHTML,/99.4%/);assert.doesNotMatch(page,/research-content|All four models: extended tests/);}
 }
});

test('new build refreshes the named download and preserves older snapshots',async()=>{const {mkdtemp,writeFile,readFile,readdir,rm}=await import('node:fs/promises');const {tmpdir}=await import('node:os');const {join}=await import('node:path');const output=await mkdtemp(join(tmpdir(),'sona-release-'));try{await writeFile(join(output,'Release_1.4.html'),'old');await writeFile(join(output,'notes.txt'),'keep');const result=await buildRelease(fileURLToPath(new URL('..',import.meta.url)),output);assert.equal(result.filename,'Download_Project_Here.html');assert.equal(await readFile(join(output,'CURRENT_RELEASE.txt'),'utf8'),'Download_Project_Here.html\n');assert.deepEqual((await readdir(output)).sort(),['CURRENT_RELEASE.txt','Download_Project_Here.html','Release_1.4.html','notes.txt']);}finally{await rm(output,{recursive:true,force:true});}});
