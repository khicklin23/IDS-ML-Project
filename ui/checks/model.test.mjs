import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
const report=JSON.parse(readFileSync(new URL('../report.json',import.meta.url)));
class Element {constructor(){this.value='';this.innerHTML='';this.textContent='';this.style={};this.events={};this.attributes={};this.classList={add(){}};} addEventListener(n,f){this.events[n]=f;} setAttribute(n,v){this.attributes[n]=v;}}
const elements=new Map();const get=id=>{if(!elements.has(id))elements.set(id,new Element());return elements.get(id);};
get('model-choice').value='binary_model';
for(const [n,v] of [['bytes','.8'],['duration','-.4'],['outcome','1.2']])get(`weight-${n}`).value=v;
const buttons=['input','hidden1','hidden2','output'].map(layer=>{const e=new Element();e.dataset={layer};return e;});
globalThis.document={getElementById:get,querySelectorAll:()=>buttons};globalThis.fetch=async()=>({ok:true,json:async()=>report});await import('../model.js');
test('saved architecture and dataset counts',()=>{assert.match(get('model-network').innerHTML,/32 neurons/);assert.match(get('architecture-details').textContent,/\(1 output\)/);assert.match(get('training-examples').innerHTML,/14,453/);assert.match(get('training-source').textContent,/21,079/);});
test('family network and layer selection',()=>{get('model-choice').value='multiclass_model';get('model-choice').events.change();buttons[3].events.click();assert.match(get('architecture-details').textContent,/\(6 outputs\)/);assert.match(get('layer-explanation').innerHTML,/Softmax/);assert.equal(buttons[3].attributes['aria-pressed'],'true');});
test('teaching neuron applies weighted sum and ReLU',()=>{assert.match(get('neuron-result').innerHTML,/1.50/);get('weight-bytes').value='-2';get('weight-duration').value='2';get('weight-outcome').value='-2';get('weight-bytes').events.input();assert.match(get('neuron-result').innerHTML,/-4.90/);assert.match(get('neuron-result').innerHTML,/After ReLU<\/span><strong>0.00/);});
test('three actual pages in navigation',()=>{for(const file of ['index.html','raw.html','model.html']){const html=readFileSync(new URL(`../${file}`,import.meta.url),'utf8');const nav=html.match(/<nav[^>]*>([\s\S]*?)<\/nav>/)[1];assert.deepEqual([...nav.matchAll(/href="([^"]+)"/g)].map(m=>m[1]),['index.html','raw.html','model.html']);assert.equal([...nav.matchAll(/aria-current="page"/g)].length,1);}});
