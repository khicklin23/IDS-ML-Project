import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {validateResearch} from '../research-data.mjs';
const research=JSON.parse(readFileSync(new URL('../research.json',import.meta.url)));
const versions=JSON.parse(readFileSync(new URL('../versions.json',import.meta.url)));
class Element {constructor(){this.value='';this.innerHTML='';this.textContent='';this.hidden=false;this.handlers={};this.dataset={};}addEventListener(n,f){this.handlers[n]=f;}querySelectorAll(){return [];}}
const elements=new Map();const get=id=>{if(!elements.has(id))elements.set(id,new Element());return elements.get(id);};
globalThis.document={getElementById:get};
globalThis.fetch=async path=>({ok:true,json:async()=>structuredClone(String(path).includes('research.json')?research:versions)});
await import('../versions.js');
test('matched experimental comparison stays available without extra research panels',()=>{
 assert.equal(get('versions-content').hidden,false);
 assert.equal(get('research-status').hidden,true);
 assert.match(get('version-comparison').innerHTML,/99.0%/);
 assert.match(get('version-comparison').innerHTML,/50.0%/);
 assert.match(get('version-detail').innerHTML,/separately trained Sona counterpart/);
 const html=readFileSync(new URL('../progression.html',import.meta.url),'utf8');
 assert.doesNotMatch(html,/research-heading|research-content|Latest timing comparison/);
 const source=readFileSync(new URL('../versions.js',import.meta.url),'utf8');
 assert.doesNotMatch(source,/Earlier timing study|All four models: extended tests|research-content/);
});
test('research validation rejects impossible scores, counts, or release relabeling',()=>{
 assert.equal(validateResearch(research),research);
 for(const mutate of [d=>d.public_model_replaced=true,d=>d.studies[0].rows[0].sona=2,d=>d.studies[0].rows[0].sona_min=1,d=>d.studies.at(-1).benign_controls[0].false_flags=10000,d=>delete d.studies[0].seeds]){
  const bad=structuredClone(research);mutate(bad);assert.throws(()=>validateResearch(bad));
 }
});

test('comparison scores validate every model and reject duplicate columns',()=>{
 const data=structuredClone(research),study=data.studies[0];
 delete study.comparison;
 delete data.four_model_comparison;
 study.models=[{id:'example',label:'Comparison'}];
 for(const row of study.rows)row.scores={example:{mean:.8,min:.7,max:.9}};
 assert.equal(validateResearch(data),data);
 study.rows[0].scores.example.max=.6;
 assert.throws(()=>validateResearch(data));
 study.rows[0].scores.example={mean:.8,min:.7,max:.9};
 study.models.push({id:'example',label:'Duplicate'});
 assert.throws(()=>validateResearch(data));
});

test('original progression leads with matched scores and preserved historical evidence',()=>{
 const html=readFileSync(new URL('../progression.html',import.meta.url),'utf8');
 assert.ok(html.includes('id="version-heading"'));
 assert.match(html,/goal was to recognize more kinds of behavior, not to run faster/);
 assert.match(get('version-comparison').innerHTML,/50.0%/);
 assert.match(get('version-comparison').innerHTML,/99.0%/);
 if(research.four_model_comparison){
  assert.doesNotMatch(get('version-comparison').innerHTML,/Not tested/);
  assert.match(get('version-comparison').innerHTML,/Slowed-down/);
  assert.match(get('version-findings').textContent,/Adding memory/);
  assert.match(get('version-findings').textContent,/91.7%/);
  assert.match(get('version-findings').textContent,/99.4%/);
  assert.match(get('version-findings').textContent,/Beta 3 is slightly ahead on timing alone/);
  for(const key of ['v1','v2','v3','v4']){
   const scores=research.four_model_comparison.results[key];
   const value=((scores.normal.mean+scores.stealth.mean+scores.timing.mean)/3*100).toFixed(1)+'%';
   assert.ok(get('version-comparison').innerHTML.includes(value));
  }
 }else{
  assert.match(get('version-comparison').innerHTML,/75.0%/);
  assert.match(get('version-comparison').innerHTML,/99.5%/);
 }
 assert.match(get('version-comparison').innerHTML,/100.0%/);
 get('version-select').value='v2';get('version-select').handlers.change();
 assert.match(get('version-comparison').innerHTML,/99.0%/);
 assert.match(get('version-detail').innerHTML,/More varied training/);
});

test('two-test score stays tied to matched timing evidence',()=>{
 const bad=structuredClone(research);bad.studies[0].comparison.models.sona.timing=.9;
 assert.throws(()=>validateResearch(bad));
 const altered=structuredClone(research);altered.studies[0].comparison.models.sona.ordinary=.8;
 if(research.four_model_comparison)assert.throws(()=>validateResearch(altered));
 else assert.equal(validateResearch(altered),altered);
 // Rendering derives the score rather than loading a separately editable total.
 const original=research.studies[0].comparison.models.sona;
 assert.equal(Math.round((original.ordinary+original.timing)/2*1000)/10,99.5);
});

test('four-model comparisons reject missing rows, mismatched sample counts and altered reused results',()=>{
 if(!research.four_model_comparison)return;
 for(const mutate of [d=>delete d.four_model_comparison.results.v1,
  d=>d.four_model_comparison.results.v3.timing.sessions++,
  d=>d.four_model_comparison.results.v2.timing.mean=.6,
  d=>d.four_model_comparison.results.v1.normal.mean=2]){
  const bad=structuredClone(research);mutate(bad);assert.throws(()=>validateResearch(bad));
 }
});

test('comparison headings expose distinct hover and keyboard explanations',()=>{
 const html=get('version-comparison').innerHTML;
 for(const key of ['ordinary','slowed','timing','combined']){
  assert.ok(html.includes(`aria-describedby="comparison-help-${key}"`));
  assert.ok(html.includes(`id="comparison-help-${key}"`));
 }
 assert.equal((html.match(/role="tooltip"/g)||[]).length,4);
 assert.match(html,/carried out more slowly or spread over time/);
 assert.match(html,/recognition of the sequence, not execution speed/);
 assert.match(html,/divided by three/);
 assert.match(html,/regardless of its number of sessions/);
});

test('production progression omits all requested expandable result sections',()=>{
 const html=readFileSync(new URL('../progression.html',import.meta.url),'utf8');
 assert.doesNotMatch(html,/About the matched tests|Original traffic test scores|About these results/);
 const source=readFileSync(new URL('../versions.js',import.meta.url),'utf8');
 assert.doesNotMatch(source,/saved-version-comparison|timing-comparison-note|version-provenance/);
});
