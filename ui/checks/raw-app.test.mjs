import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
class Element {
  constructor(){this.value='';this.checked=false;this.handlers={};this.hidden=false;this.innerHTML='';this.textContent='';this.dataset={};}
  addEventListener(name,fn){this.handlers[name]=fn;}
  async fire(name){await this.handlers[name]?.();}
  querySelectorAll(){this.buttons=[...this.innerHTML.matchAll(/data-column="([^"]+)"/g)].map(match=>{const button=new Element();button.dataset.column=match[1];return button;});return this.buttons;}
}
const elements=new Map(),get=id=>{if(!elements.has(id))elements.set(id,new Element());return elements.get(id);};
globalThis.document={getElementById:get};globalThis.location={search:'?session=port_scan-00000'};globalThis.history={replaceState(){}};
get('raw-page-size').value='100';
globalThis.fetch=async path=>({ok:true,json:async()=>JSON.parse(await readFile(new URL('../'+path.replace(/^\.\//,''),import.meta.url),'utf8'))});
await import('../raw.js');
test('raw page loads every chunk and follows an alert session link',()=>{
  assert.equal(get('raw-status').hidden,true);
  assert.match(get('raw-summary').textContent,/21,079 total/);
  assert.match(get('raw-summary').textContent,/session port_scan-00000/);
  assert.equal(get('show-provenance').checked,true);
  assert.match(get('raw-head').innerHTML,/Session Id/);
});
test('reset, hundred-row pagination, and page jump work',async()=>{
  await get('raw-reset').fire('click');
  assert.match(get('raw-summary').textContent,/1–100/);
  await get('raw-next').fire('click');assert.match(get('raw-summary').textContent,/101–200/);
  get('raw-page').value='211';await get('raw-page').fire('change');
  assert.match(get('raw-summary').textContent,/21,001–21,079/);
  assert.equal(get('raw-next').disabled,true);
});
