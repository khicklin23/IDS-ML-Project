import test from 'node:test';
import assert from 'node:assert/strict';
import {installMobileSupport} from '../mobile.mjs';
function fixture(){
  const events={},viewportEvents={};
  const doc={addEventListener:(name,fn)=>events[name]=fn,querySelectorAll:()=>[]};
  const win={innerWidth:320,innerHeight:568,addEventListener:(name,fn)=>viewportEvents[name]=fn};
  const button=()=>{
    const classes=new Set(),attributes={},tip={style:{},offsetHeight:140};
    const b={classes,attributes,tip,classList:{add:x=>classes.add(x),remove:x=>classes.delete(x)},setAttribute:(key,value)=>attributes[key]=value,querySelector:()=>tip,getBoundingClientRect:()=>({left:290,bottom:550}),closest:()=>b,matches:()=>false,contains:target=>target===b};return b;
  };
  installMobileSupport(doc,win);
  return {events,viewportEvents,button,win};
}
test('touch help toggles and stays inside a small viewport',()=>{
  const {events,button}=fixture(),b=button();
  events.pointerover({target:b,pointerType:'touch'});
  assert.equal(b.classes.has('help-open'),false);
  events.click({target:b});
  assert.equal(b.attributes['aria-expanded'],'true');
  assert.equal(b.tip.style.left,'24px');assert.equal(b.tip.style.top,'412px');
  events.click({target:b});assert.equal(b.attributes['aria-expanded'],'false');
});
test('only one explanation opens; outside taps, Escape and scrolling dismiss it',()=>{
  const {events,button,viewportEvents}=fixture(),a=button(),b=button();
  events.click({target:a});events.click({target:b});
  assert.equal(a.attributes['aria-expanded'],'false');
  events.click({target:{closest:()=>null}});assert.equal(b.attributes['aria-expanded'],'false');
  events.click({target:a});events.keydown({key:'Escape'});assert.equal(a.attributes['aria-expanded'],'false');
  events.click({target:a});events.scroll();assert.equal(a.attributes['aria-expanded'],'false');
  events.click({target:a});viewportEvents.resize();assert.equal(a.attributes['aria-expanded'],'false');
});
test('keyboard help positions against the visible viewport when zoomed',()=>{
  const {events,win,button}=fixture();win.visualViewport={width:240,height:400,offsetLeft:20,offsetTop:30};
  const b=button();b.matches=()=>true;events.focusin({target:b});
  assert.equal(b.tip.style.width,'208px');assert.equal(b.tip.style.left,'36px');assert.equal(b.tip.style.top,'274px');
});
