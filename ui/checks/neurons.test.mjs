import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {forward, initialNetwork, lossAndGradient, trainStep, softmax, toyExamples} from '../neuron-math.mjs';
const close=(a,b,tolerance=1e-8)=>assert.ok(Math.abs(a-b)<tolerance,`${a} differs from ${b}`);
test('forward pass is deterministic, preserves parameters and computes each separate layer',()=>{
  const model=initialNetwork(),snapshot=structuredClone(model),inputs=[0.7,0.4],trace=forward(model,inputs);
  assert.deepEqual(trace,forward(model,inputs));assert.deepEqual(model,snapshot);
  assert.deepEqual(trace.activations.map(a=>a.length),[2,3,2,2]);
  close(trace.sums[0][0],0.5);close(trace.activations[1][0],Math.tanh(0.5));
  close(trace.sums[0][1],0.05);close(trace.activations[3].reduce((a,b)=>a+b,0),1);
  const changed=structuredClone(model);changed[0].weights[0][0]+=1;
  assert.notDeepEqual(forward(changed,inputs).activations[3],trace.activations[3]);
  assert.notDeepEqual(forward(model,[0.1,0.1]).activations[3],trace.activations[3]);
});
test('softmax is stable for large logits and invariant to a shared offset',()=>{
  assert.deepEqual(softmax([10000,10000]),[0.5,0.5]);
  softmax([1,2]).forEach((p,i)=>close(p,softmax([10001,10002])[i]));
});
test('backpropagation matches independent numerical derivatives for all 23 parameters',()=>{
  const model=initialNetwork(),{gradient}=lossAndGradient(model),epsilon=1e-5;
  let checked=0;
  for(let k=0;k<model.length;k++)for(let j=0;j<model[k].bias.length;j++) {
    for(let i=-1;i<model[k].weights[j].length;i++) {
      const plus=structuredClone(model),minus=structuredClone(model);
      if(i===-1){plus[k].bias[j]+=epsilon;minus[k].bias[j]-=epsilon;}
      else{plus[k].weights[j][i]+=epsilon;minus[k].weights[j][i]-=epsilon;}
      const numeric=(lossAndGradient(plus).loss-lossAndGradient(minus).loss)/(2*epsilon);
      close(numeric,i===-1?gradient[k].bias[j]:gradient[k].weights[j][i],1e-7);checked++;
    }
  }
  assert.equal(checked,23);
});
test('training improves the toy objective and uses the announced gradient update',()=>{
  const model=initialNetwork(),step=trainStep(model),before=lossAndGradient(model).loss;
  close(step.network[0].weights[0][0],model[0].weights[0][0]-0.15*step.gradient[0].weights[0][0]);
  assert.ok(step.after<before);let trained=model;
  for(let i=0;i<500;i++)trained=trainStep(trained).network;
  assert.ok(lossAndGradient(trained).loss<before/3);
  for(const example of toyExamples){const p=forward(trained,example.x).activations[3];assert.equal(p[1]>p[0]?1:0,example.target);}
});

class LabElement {
  constructor(){this.innerHTML='';this.textContent='';this.value='';this.disabled=false;this.events={};this.attributes={};this.dataset={};}
  addEventListener(name,fn){this.events[name]=fn;}
  setAttribute(key,value){this.attributes[key]=value;}
  querySelector(){return null;}
}
const labElements=new Map(),labGet=id=>{if(!labElements.has(id))labElements.set(id,new LabElement());return labElements.get(id);};
const presets=['low','failures','both'].map(preset=>{const button=new LabElement();button.dataset.preset=preset;return button;});
globalThis.document={getElementById:labGet,querySelectorAll:selector=>selector==='[data-preset]'?presets:[]};
await import('../neurons.js');
test('walkthrough reveals separate neurons, editable parameters and final class probabilities',()=>{
  assert.match(labGet('neuron-network').innerHTML,/First hidden layer/);
  assert.equal((labGet('neuron-network').innerHTML.match(/role="button"/g)||[]).length,9);
  assert.equal(labGet('previous-neuron').disabled,true);
  labGet('next-neuron').events.click();
  assert.equal(labGet('neuron-heading').textContent,'Inside Neuron 1');
  assert.match(labGet('neuron-calculation').innerHTML,/Response sent to the next layer<\/span><strong>0.46/);
  assert.match(labGet('neuron-controls').innerHTML,/data-parameter="weight"/);
  labGet('neuron-controls').events.input({target:{value:'1.8',dataset:{parameter:'weight',index:'0'}}});
  assert.match(labGet('neuron-calculation').innerHTML,/Response sent to the next layer<\/span><strong>0.83/);
  for(let i=0;i<7;i++)labGet('next-neuron').events.click();
  assert.equal(labGet('neuron-heading').textContent,'From scores to a decision');
  assert.match(labGet('neuron-calculation').innerHTML,/Compare the final choices/);
  labGet('input-failures').events.input({target:{value:'0.1'}});
  assert.equal(labGet('input-failures-value').textContent,'10.0%');
});
test('simplified explanations remove training controls and equations while reset restores weights',()=>{
  const page=readFileSync(new URL('../neurons.html',import.meta.url),'utf8');
  assert.doesNotMatch(page,/id="(?:learning-rate|train-once|train-many|training-loss)"/);
  assert.match(page,/Backpropagation works backward/);
  assert.match(page,/small change to each connection’s weight/);
  assert.doesNotMatch(labGet('neuron-controls').innerHTML,/Bias|data-parameter="bias"/);
  labGet('tokens-mode').events.click();assert.match(labGet('encoding-copy').textContent,/illustrative, not a real tokenizer/);
  labGet('llm-destination').events.click();assert.match(labGet('destination-copy').innerHTML,/Sampling/);
  labGet('reset-network').events.click();
  assert.equal(labGet('step-counter').textContent,'1 / 10');
  labGet('next-neuron').events.click();assert.match(labGet('neuron-calculation').innerHTML,/Response sent to the next layer<\/span><strong>0.46/);
  assert.doesNotMatch(labGet('neuron-calculation').innerHTML,/tanh|z =|x₁|x₂/);
});
