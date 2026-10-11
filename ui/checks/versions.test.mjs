import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {validateVersions,selectedSummary} from '../versions-data.mjs';
const data=JSON.parse(readFileSync(new URL('../versions.json',import.meta.url)));
class Element {constructor(){this.value='';this.innerHTML='';this.textContent='';this.hidden=false;this.handlers={};this.dataset={};}addEventListener(n,f){this.handlers[n]=f;}querySelectorAll(){return this.buttons=[...this.innerHTML.matchAll(/data-version="([^"]+)"/g)].map(m=>{const e=new Element();e.dataset.version=m[1];return e;});}}
const elements=new Map();const get=id=>{if(!elements.has(id))elements.set(id,new Element());return elements.get(id);};
globalThis.document={getElementById:get};globalThis.fetch=async()=>({ok:true,json:async()=>structuredClone(data)});
await import('../versions.js');
test('defaults to v4 with recorded session results',()=>{assert.equal(get('version-select').value,'v4');assert.equal(get('versions-content').hidden,false);assert.match(get('version-detail').innerHTML,/97 \/ 97/);assert.match(get('version-comparison').innerHTML,/100.0%/);assert.match(get('version-detail').innerHTML,/session-memory design as v3/);});
test('selector and progression buttons change selected version',()=>{get('version-select').value='v3';get('version-select').handlers.change();assert.match(get('version-detail').innerHTML,/67 \/ 97/);assert.match(get('version-comparison').innerHTML,/59.9%/);get('version-progression').buttons[0].handlers.click();assert.equal(get('version-select').value,'v1');assert.match(get('version-detail').innerHTML,/62 \/ 97/);assert.match(get('version-comparison').innerHTML,/V1 <span class="sr-only">· Selected/);});
test('comparison data rejects incompatible or inconsistent results',()=>{assert.equal(validateVersions(data),data);for(const mutate of [d=>d.versions_schema=9,d=>delete d.versions.v4,d=>d.versions.v1.stealth_session_metrics.macro_f1=2,d=>d.versions.v2.normal_session_metrics.n=99,d=>d.versions.v3.stealth_session_metrics.confusion_matrix[0][0]=-1]){const bad=structuredClone(data);mutate(bad);assert.throws(()=>validateVersions(bad));}assert.equal(selectedSummary(data,'v4').correct,97);});

test('Overview order and separate progression page',()=>{const overview=readFileSync(new URL('../index.html',import.meta.url),'utf8');const order=['performance','detection-records','classification','evidence'].map(id=>overview.indexOf(`id="${id}"`));assert.ok(order.every(n=>n>=0));assert.deepEqual(order,[...order].sort((a,b)=>a-b));assert.doesNotMatch(overview,/src="versions.js(?:\?[^"]*)?"|id="version-detail"/);const page=readFileSync(new URL('../progression.html',import.meta.url),'utf8');assert.match(page,/src="versions.js\?v=/);assert.match(page,/value="v4" selected/);});
