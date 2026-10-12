import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
const report=JSON.parse(readFileSync(new URL('../report.json',import.meta.url)));
class Element {constructor(){this.value='';this.innerHTML='';this.textContent='';this.style={};this.events={};this.attributes={};this.classList={add(){}};} addEventListener(n,f){this.events[n]=f;} setAttribute(n,v){this.attributes[n]=v;}}
const elements=new Map();const get=id=>{if(!elements.has(id))elements.set(id,new Element());return elements.get(id);};
const buttons=['input','hidden1','hidden2','output'].map(layer=>{const e=new Element();e.dataset={layer};return e;});
globalThis.document={getElementById:get,querySelectorAll:()=>buttons};globalThis.fetch=async()=>({ok:true,json:async()=>report});await import('../model.js');
test('saved architecture and dataset counts',()=>{assert.match(get('model-network').innerHTML,/32 memory values/);assert.match(get('model-network').innerHTML,/6 family outputs/);assert.ok(get('training-examples').innerHTML.includes(report.dataset.manifest.scenario_counts.benign.toLocaleString('en-US')));assert.ok(get('training-source').textContent.includes(report.dataset.manifest.row_count.toLocaleString('en-US')));});
test('shared network stage selection',()=>{const initial=get('model-network').innerHTML;buttons[3].events.click();assert.notEqual(get('model-network').innerHTML,initial);assert.match(get('model-network').innerHTML,/<circle/);assert.match(get('model-network').innerHTML,/memory-arrow/);assert.match(get('layer-explanation').innerHTML,/Softmax/);assert.equal(buttons[3].attributes['aria-pressed'],'true');});
test('four actual pages in navigation',()=>{for(const file of ['index.html','raw.html','model.html','progression.html']){const html=readFileSync(new URL(`../${file}`,import.meta.url),'utf8');const nav=html.match(/<nav[^>]*>([\s\S]*?)<\/nav>/)[1];assert.deepEqual([...nav.matchAll(/href="([^"]+)"/g)].map(m=>m[1]),['index.html','model.html','progression.html','raw.html']);assert.equal([...nav.matchAll(/aria-current="page"/g)].length,1);}});

test('requested configuration sections are removed while the neural diagram remains',()=>{
 const html=readFileSync(new URL('../model.html',import.meta.url),'utf8');
 assert.doesNotMatch(html,/Training settings in this saved run|Architecture details|saved-training-settings|architecture-details|Connections learn; class weights balance|weight-bytes|neuron-result/);
 assert.match(html,/id="model-network"/);
});

test('network controls share the diagram scroll area and attack cards have layout identities',()=>{
 const html=readFileSync(new URL('../model.html',import.meta.url),'utf8');
 assert.match(html,/<div class="network-scroll"><div class="network-diagram">[\s\S]*?<div class="layer-controls"/);
 assert.match(get('training-examples').innerHTML,/data-scenario="port_scan"/);
 assert.match(get('training-examples').innerHTML,/data-scenario="service_flood"/);
});
