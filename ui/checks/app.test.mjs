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
  assert.match(get('comparison-note').textContent,/held-out flows/);
  assert.match(get('evidence-note').textContent,/One Sona session prediction/);
  assert.doesNotMatch(get('detector-note').textContent,/One model reads/);
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
});
test('public overview omits report-management controls and run metadata',async()=>{const html=await readFile(new URL('../index.html',import.meta.url),'utf8');assert.doesNotMatch(html,/id="(?:report-upload|download-report|run-meta)"/);assert.match(html,/class="sona-title">Sona/);});

test('classification and dataset panels display the current report values',()=>{
 for(const [name,n] of Object.entries(report.dataset.manifest.scenario_counts))assert.ok(get('composition').innerHTML.includes(n.toLocaleString('en-US')));
 for(const n of Object.values(report.dataset.split_sizes))assert.ok(get('dataset-details').innerHTML.includes(n.toLocaleString('en-US')));
 assert.ok(get('dataset-details').innerHTML.includes(report.dataset.manifest.row_count.toLocaleString('en-US')));
 for(const row of report.multiclass_model.test_metrics.confusion_matrix)for(const n of row)assert.ok(get('confusion').innerHTML.includes(n.toLocaleString('en-US')));
});

test('Sona-only performance retains every percentage and explains metrics',()=>{
 const html=get('comparison').innerHTML;
 assert.doesNotMatch(html,/Fixed rule|Rule baseline/);
 for(const key of ['f1','recall','precision','false_positive_rate']){
  assert.ok(html.includes((report.binary_model.test_metrics[key]*100).toFixed(1)+'%'));
  assert.ok(html.includes(`aria-describedby="detection-help-${key}"`));
 }
 assert.equal((html.match(/role="tooltip"/g)||[]).length,4);
 assert.match(html,/how many did Sona catch/);
 assert.match(html,/normal flows as the denominator/);
 assert.match(html,/Lower is better/);
});
