// Rendering/event smoke harness. This deliberately does not claim browser or
// visual coverage: it supplies only the DOM surface the application uses.
import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';

class Element {
  constructor(id){this.id=id;this.value='';this.hidden=false;this.disabled=false;this.open=false;this.dataset={};this.handlers=new Map();this.attrs={};this.innerHTML='';this.textContent='';this.files=[];}
  addEventListener(name,fn){if(!this.handlers.has(name))this.handlers.set(name,[]);this.handlers.get(name).push(fn);}
  async fire(name,event={}){for(const fn of this.handlers.get(name)??[])await fn({target:this,...event});}
  setAttribute(name,value){this.attrs[name]=value;}
  querySelector(){return this.option??=new Element('option');}
  querySelectorAll(){this.buttons=[...this.innerHTML.matchAll(/data-row="(\d+)"/g)].map(match=>{const button=new Element('record');button.dataset.row=match[1];return button;});return this.buttons;}
  showModal(){this.open=true;}close(){this.open=false;}scrollIntoView(){}focus(){}click(){return this.fire('click');}
  getBoundingClientRect(){return {left:0,right:100,top:0,bottom:100};}
}
const elements=new Map();
const get=id=>{if(!elements.has(id))elements.set(id,new Element(id));return elements.get(id);};
const report=JSON.parse(await readFile(new URL('../report.json',import.meta.url),'utf8'));
let downloadLink;
globalThis.document={getElementById:get,createElement:()=>downloadLink=new Element('download')};
get('sort').value='time';
get('tab-alerts').parentElement=new Element('tablist');
globalThis.fetch=async()=>({ok:true,json:async()=>structuredClone(report)});
await import('../app.js');

test('real report renders evaluation without a training panel',()=>{
  assert.equal(get('dashboard').hidden,false);
  assert.ok(get('metrics').innerHTML.includes(report.dataset.manifest.row_count.toLocaleString('en-US')));
  assert.match(get('comparison-note').textContent,/false positives/);
  assert.match(get('evidence-note').textContent,/One v4 session prediction/);
  assert.match(get('detector-note').textContent,/not independent predictions/);
  assert.ok(get('disposition-summary').innerHTML.includes(report.binary_model.test_metrics.confusion_matrix.tn.toLocaleString('en-US')));
  assert.match(get('disposition-summary').innerHTML,/correctly left unflagged/);
  assert.ok(get('disposition-summary').innerHTML.includes(`${report.binary_model.test_metrics.confusion_matrix.fp.toLocaleString('en-US')} benign flows`));
  assert.match(get('class-performance').innerHTML,/Normal traffic recognition/);
  assert.match(get('composition').innerHTML,/Normal background/);
});
test('benign false-alarm selection persists through search rerenders',async()=>{
  get('category-filter').value='benign';await get('category-filter').fire('change');
  assert.match(get('table-body').innerHTML,/Benign · false alarm/);
  get('search').value='';await get('search').fire('input');
  assert.equal(get('category-filter').value,'benign');
  await get('reset-filters').fire('click');
});
test('search, reset and flow tab update the inline workspace',async()=>{
  get('search').value='no-such-record';await get('search').fire('input');
  assert.match(get('table-body').innerHTML,/No embedded records/);
  await get('reset-filters').fire('click');assert.equal(get('page-number').textContent,`1 / ${Math.max(1,Math.ceil(report.alerts.sample.length/20))}`);
  await get('tab-flows').fire('click');
  assert.equal(get('severity-label').hidden,true);
  assert.equal(get('page-number').textContent,`1 / ${Math.ceil(report.dataset.sample_flows.length/20)}`);
  await get('next').fire('click');assert.equal(get('page-number').textContent,`2 / ${Math.ceil(report.dataset.sample_flows.length/20)}`);
  assert.match(get('evidence-note').textContent,/earliest flows/);
});
test('inline alert/session navigation and recorded incidents work',async()=>{
  await get('table-body').buttons[0].fire('click');
  assert.equal(get('detail-panel').hidden,false);
  assert.match(get('detail-body').innerHTML,/Model predictions and port fields are not included/);
  await get('close-detail').fire('click');assert.equal(get('detail-panel').hidden,true);
  await get('tab-alerts').parentElement.fire('keydown',{key:'Home',preventDefault(){}});
  await get('table-body').buttons[0].fire('click');
  assert.match(get('detail-body').innerHTML,/raw.html\?session=/);
  assert.match(get('detail-body').innerHTML,/GROUND TRUTH/);
  await get('tab-incidents').fire('click');
  assert.equal(get('detail-panel').hidden,true);
  assert.match(get('evidence-note').textContent,/recorded host/);
  await get('table-body').buttons[0].fire('click');
  assert.match(get('detail-body').innerHTML,/Related alerts/);
  await get('detail-body').buttons[0].fire('click');
  assert.equal(get('detail-title').textContent,'Alert investigation');
  await get('download-report').fire('click');assert.equal(downloadLink.download,`${report.run_id}-report.json`);
});
test('invalid import preserves the current report, then empty samples render honestly',async()=>{
  const oldMetrics=get('metrics').innerHTML;
  get('report-upload').files=[{name:'bad.json',size:20,text:async()=>'{broken'}];
  await get('report-upload').fire('change');
  assert.equal(get('metrics').innerHTML,oldMetrics);
  assert.match(get('status').textContent,/previous report remains/);
  const empty=structuredClone(report);empty.alerts.sample=[];empty.dataset.sample_flows=[];
  empty.run_id='<script>alert(1)</script>';
  empty.multiclass_model.test_metrics.per_class.beaconing={precision:0,recall:0,f1:0,support:0};
  get('report-upload').files=[{name:'empty.json',size:50,text:async()=>JSON.stringify(empty)}];
  await get('report-upload').fire('change');
  assert.match(get('table-body').innerHTML,/No recorded incidents/);
  assert.match(get('class-note').textContent,/No held-out examples/);
  assert.match(get('run-meta').innerHTML,/&lt;script&gt;/);
  assert.doesNotMatch(get('run-meta').innerHTML,/<script>/);
  assert.equal(get('status').hidden,true);
});
