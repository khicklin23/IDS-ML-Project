import {installMobileSupport} from './mobile.mjs?v=sona-mobile-compact-1';
import {escapeHTML} from './data.mjs?v=sona-prod-polish-1';
import {initialNetwork, forward} from './neuron-math.mjs?v=neuron-simple-2';
installMobileSupport();
const labElement=id=>document.getElementById(id);
const labNumber=value=>Math.abs(value)<0.00005?'0.0000':value.toFixed(4);
const labPercent=value=>`${(value*100).toFixed(1)}%`;
let labNetwork=initialNetwork(),labInputs=[0.7,0.4],labStep=0;
const labNeurons=[
  {step:0,id:'x1',label:'Failed connections',layer:-1,index:0,x:80,y:155},
  {step:0,id:'x2',label:'Transfer volume',layer:-1,index:1,x:80,y:285},
  {step:1,id:'h1a',label:'Neuron 1',layer:0,index:0,x:310,y:105},
  {step:2,id:'h1b',label:'Neuron 2',layer:0,index:1,x:310,y:220},
  {step:3,id:'h1c',label:'Neuron 3',layer:0,index:2,x:310,y:335},
  {step:4,id:'h2a',label:'Neuron 4',layer:1,index:0,x:540,y:155},
  {step:5,id:'h2b',label:'Neuron 5',layer:1,index:1,x:540,y:285},
  {step:6,id:'o1',label:'Normal',layer:2,index:0,x:780,y:155},
  {step:7,id:'o2',label:'Review',layer:2,index:1,x:780,y:285}
];
const labStepNames=['Inputs','Neuron 1','Neuron 2','Neuron 3','Neuron 4','Neuron 5','Normal score','Review score','Decision','Learning'];
const labSelected=()=>labNeurons.find(n=>n.step===labStep&&n.layer>=0);
function labValue(neuron,trace) {
  return neuron.layer===-1?labInputs[neuron.index]:neuron.layer===2?trace.sums[2][neuron.index]:trace.activations[neuron.layer+1][neuron.index];
}
function labRenderMap(trace) {
  let svg='<title>Editable teaching network: select a neuron to inspect its calculation</title>';
  [[80,'Input values'],[310,'First hidden layer'],[540,'Second hidden layer'],[780,'Final choices']].forEach(([x,label])=>svg+=`<text x="${x}" y="35" text-anchor="middle" class="lab-column-title">${label}</text>`);
  for(const neuron of labNeurons.filter(n=>n.layer>=0)) {
    const previous=labNeurons.filter(n=>n.layer===neuron.layer-1);
    previous.forEach((source,i)=>{
      const weight=labNetwork[neuron.layer].weights[neuron.index][i],active=neuron.step===labStep;
      svg+=`<path d="M${source.x+32},${source.y} C${source.x+100},${source.y} ${neuron.x-100},${neuron.y} ${neuron.x-32},${neuron.y}" fill="none" stroke="${weight>=0?'#2b7d9d':'#b46b46'}" stroke-width="${1+Math.min(Math.abs(weight),4)*1.7}" opacity="${active?0.95:neuron.step<=labStep?0.38:0.13}"><title>${source.label} to ${neuron.label}: weight ${labNumber(weight)}</title></path>`;
      if(active) svg+=`<text x="${source.x+105}" y="${source.y+(neuron.y-source.y)*0.32-7}" class="lab-edge-label">${weight.toFixed(2)}</text>`;
    });
  }
  for(const neuron of labNeurons) {
    const revealed=neuron.step<=labStep,selected=neuron.step===labStep;
    const value=labValue(neuron,trace);
    svg+=`<g data-step="${neuron.step}" class="lab-neuron ${selected?'active':''} ${revealed?'revealed':'pending'}" tabindex="0" role="button" aria-pressed="${selected}" aria-label="${neuron.label}, ${neuron.layer===-1?'input':neuron.layer===2?'output neuron':'hidden neuron'}${revealed?`, value ${labNumber(value)}`:', not revealed yet'}"><title>Select ${neuron.label}</title><circle cx="${neuron.x}" cy="${neuron.y}" r="34"/><text x="${neuron.x}" y="${neuron.y+5}" text-anchor="middle">${revealed?value.toFixed(2):'?'}</text><text x="${neuron.x}" y="${neuron.y+59}" text-anchor="middle" class="lab-node-name">${neuron.label}</text></g>`;
  }
  svg+='<text x="450" y="420" text-anchor="middle" class="lab-column-title">Measurements → first responses → combined responses → final choices</text>';
  labElement('neuron-network').innerHTML=svg;
}
function labControls(neuron) {
  const parameters=labNetwork[neuron.layer],previous=labNeurons.filter(n=>n.layer===neuron.layer-1);
  return `<p class="control-caption">Change how much each connection influences this neuron</p>${parameters.weights[neuron.index].map((weight,index)=>`<label class="neuron-slider"><span>From ${previous[index].label}<output id="parameter-weight-${index}" for="slider-weight-${index}">${weight.toFixed(2)}</output></span><input id="slider-weight-${index}" type="range" min="-3" max="3" step="0.01" value="${weight}" data-parameter="weight" data-index="${index}"></label>`).join('')}`;
}
function labRenderCalculation(trace) {
  const neuron=labSelected();
  if(neuron) {
    const previous=labNeurons.filter(n=>n.layer===neuron.layer-1);
    labElement('neuron-calculation').innerHTML=`<p class="calculation-label">Signals reaching this neuron</p>${previous.map((source,index)=>`<div class="calculation-row"><span>${source.label}</span><strong>${trace.activations[neuron.layer][index].toFixed(2)}</strong></div>`).join('')}<div class="calculation-result"><span>${neuron.layer===2?'Score for this choice':'Response sent to the next layer'}</span><strong>${labValue(neuron,trace).toFixed(2)}</strong></div>`;
  } else if(labStep===8) {
    const probabilities=trace.activations[3];
    labElement('neuron-calculation').innerHTML=`<p class="calculation-label">Compare the final choices</p>${['Normal','Review'].map((label,index)=>`<div class="probability-row"><span>${label}</span><div><i style="width:${probabilities[index]*100}%"></i></div><strong>${labPercent(probabilities[index])}</strong></div>`).join('')}<div class="calculation-result"><span>The network’s choice</span><strong>${probabilities[1]>probabilities[0]?'Review':'Normal'}</strong></div>`;
  } else if(labStep===9) {
    labElement('neuron-calculation').innerHTML='<p class="lab-copy">Compare the prediction with the correct answer, trace how the connections affected the error, and make small weight adjustments.</p><a class="version-learn-link" href="#training-heading">How learning works ↓</a>';
  } else {
    labElement('neuron-calculation').innerHTML=`<div class="calculation-row"><span>Failed connections</span><strong>${labPercent(labInputs[0])}</strong></div><div class="calculation-row"><span>Transfer volume</span><strong>${Math.round(labInputs[1]*100)} / 100</strong></div><p class="lab-copy">The network receives these measurements as numbers. Each neuron will give them a different level of influence.</p>`;
  }
}
function labRenderInspector() {
  const neuron=labSelected();
  labElement('neuron-controls').innerHTML=neuron?labControls(neuron):'';
  labElement('neuron-eyebrow').textContent=neuron?`LAYER ${neuron.layer+1} · ${neuron.layer===2?'OUTPUT':'HIDDEN'}`:`STEP ${labStep+1} OF ${labStepNames.length}`;
  labElement('neuron-heading').textContent=neuron?`Inside ${neuron.label}`:labStep===8?'From scores to a decision':labStep===9?'Weights are learned, not guessed forever':'Start with the input values';
  labElement('neuron-copy').textContent=neuron?(neuron.layer===2?'This neuron combines the previous layer’s responses into a score for one possible answer. The two answers are compared at the next step.':'This neuron receives signals from the layer before it. Each connection has a weight: a setting that controls how strongly that signal influences its response.'):labStep===8?'The final scores become percentages using softmax, which makes them add up to 100%. The higher percentage supplies this network’s answer.':labStep===9?'During training, the network compares its prediction with a known answer and adjusts the connections that influence the error.':'Move a slider to change the traffic measurements. Then follow how the same information produces different responses in different neurons.';
  labElement('neuron-takeaway').textContent=neuron?(neuron.layer===2?'A higher score makes this answer more likely relative to the other answer.':'A positive weight passes a signal in the same direction; a negative weight reverses its contribution. The resulting activity, called an activation, becomes input to the next layer.'):labStep===8?'This is a teaching example. Its percentages describe the network’s preference, not a guarantee that it is correct.':labStep===9?'Learning changes connection weights. Making a prediction uses the current weights.':'Try “Both high,” then “Only failures.” Changing the data changes the response, even when the weights stay the same.';
  labRenderCalculation(forward(labNetwork,labInputs));
}
function labRenderLive() {
  const trace=forward(labNetwork,labInputs);
  labRenderMap(trace);labRenderCalculation(trace);
  labElement('input-failures-value').textContent=labPercent(labInputs[0]);
  labElement('input-volume-value').textContent=`${Math.round(labInputs[1]*100)} / 100`;
}
function labChooseStep(step) {
  labStep=Math.max(0,Math.min(labStepNames.length-1,step));
  document.querySelectorAll('#journey-stages [data-step]').forEach(button=>button.setAttribute('aria-pressed',String(Number(button.dataset.step)===labStep)));
  labElement('previous-neuron').disabled=labStep===0;
  labElement('next-neuron').disabled=labStep===labStepNames.length-1;
  labElement('next-neuron').textContent=labStep<7?'Next neuron →':labStep===7?'See the decision →':'How it learns →';
  labElement('step-counter').textContent=`${labStep+1} / ${labStepNames.length}`;
  labRenderInspector();labRenderLive();
}
function labCompactFlow(html) {
  const names={'Connection record':'Record','Measurements & categories':'Features','Scaled numbers & indicator columns':'Numbers','Input values':'Inputs','Text: “Networks learn”':'Text','Tokens: “Net” · “works” · “ learn”':'Tokens','IDs: 17 · 42 · 9':'IDs','Embedding vectors':'Vectors','Text tokens':'Tokens','Embeddings + position':'Vectors','Attention + processing layers':'Context mixing','Vocabulary probabilities':'Choices','Next token ↻':'Next token','Ordered flow features':'Flows','GRU memory updates':'Memory','Final session state':'Summary','Six class probabilities':'Classes','Family + alert decision':'Decision'};
  return html.replace(/<span([^>]*)>([^<]+)<\/span>/g,(_,attributes,label)=>`<span${attributes} title="${escapeHTML(label)}"><span class="flow-full-label">${label}</span><span class="flow-short-label">${escapeHTML(names[label]??label)}</span></span>`);
}
function labEncoding(mode) {
  const tokens=mode==='tokens';
  labElement('features-mode').setAttribute('aria-pressed',String(!tokens));labElement('tokens-mode').setAttribute('aria-pressed',String(tokens));
  labElement('encoding-visual').innerHTML=tokens?'<span class="encoding-chip">Text: “Networks learn”</span><b>→</b><span class="encoding-chip">Tokens: “Net” · “works” · “ learn”</span><b>→</b><span class="encoding-chip">IDs: 17 · 42 · 9</span><b>→</b><span class="encoding-chip">Embedding vectors</span>':'<span class="encoding-chip">Connection record</span><b>→</b><span class="encoding-chip">Measurements & categories</span><b>→</b><span class="encoding-chip">Scaled numbers & indicator columns</span><b>→</b><span class="encoding-chip">Input values</span>';
  labElement('encoding-visual').innerHTML=labCompactFlow(labElement('encoding-visual').innerHTML);
  labElement('encoding-copy').textContent=tokens?'A token is a piece of text: often a word part, sometimes punctuation or a whole word. A tokenizer maps pieces to IDs; an embedding maps each ID to a learned vector of numbers. These splits and IDs are illustrative, not a real tokenizer. Token IDs are lookup keys, not quantities such as “more suspicious.” The numeric network below remains the traffic teaching demo.':'A feature is a measurement supplied to a model. Traffic features can include bytes, timing gaps, and connection outcomes. Numeric features are scaled; categories can become indicator columns. Real Sona uses many features; this teaching network uses just two scaled numbers so every calculation fits on screen.';
}
function labDestination(mode) {
  const llm=mode==='llm';
  labElement('sona-destination').setAttribute('aria-pressed',String(!llm));labElement('llm-destination').setAttribute('aria-pressed',String(llm));
  labElement('destination-visual').innerHTML=llm?'<span>Text tokens</span><b>→</b><span>Embeddings + position</span><b>→</b><span>Attention + processing layers</span><b>→</b><span>Vocabulary probabilities</span><b>→</b><span>Next token ↻</span>':'<span>Ordered flow features</span><b>→</b><span>GRU memory updates</span><b>→</b><span>Final session state</span><b>→</b><span>Six class probabilities</span><b>→</b><span>Family + alert decision</span>';
  labElement('destination-visual').innerHTML=labCompactFlow(labElement('destination-visual').innerHTML);
  labElement('destination-copy').innerHTML=llm?'<article><h3>The network relates tokens</h3><p>A large language model (LLM) processes token vectors. Its context is the available token history, including your prompt and earlier conversation that fits within its context window. A Transformer uses attention to combine information from those positions, plus feedforward layers. It is not simply this tiny network enlarged.</p></article><article><h3>The output is a vocabulary</h3><p>An autoregressive language model produces scores for possible next tokens. Softmax gives a distribution. Choosing a token, appending it, and running again builds a response.</p></article><article><h3>Where variation comes from</h3><p>With fixed context and parameters, an ideal deterministic forward pass gives the same scores. Sampling can choose different tokens from those scores. Training changes weights; ordinary generation does not.</p></article>':'<article><h3>Sona adds sequence memory</h3><p>The Betas use feedforward layers like this demo. Sona uses a gated recurrent unit (GRU): learned gates combine each connection with a 32-value hidden state. The memory is shared across steps within a session, not stored between sessions.</p></article><article><h3>The output is a traffic category</h3><p>The final state feeds six class scores, including benign. Softmax converts them to a distribution. A non-benign winning category flags the session; its flows inherit that decision.</p></article><article><h3>Activity is distributed</h3><p>A single neuron is not assigned one concept or one attack. Useful evidence is represented across units and connections. This walkthrough explains calculations, not an interpretation of Sona’s actual learned units.</p><a href="model.html">Explore the saved Sona architecture →</a></article>';
}
labElement('journey-stages').innerHTML=labStepNames.map((label,step)=>`<button data-step="${step}" aria-pressed="${step===0}"><small>${step+1}</small>${escapeHTML(label)}</button>`).join('');
labElement('journey-stages').addEventListener('click',event=>{const button=event.target.closest('[data-step]');if(button)labChooseStep(Number(button.dataset.step));});
labElement('neuron-network').addEventListener('click',event=>{const node=event.target.closest('[data-step]');if(node)labChooseStep(Number(node.dataset.step));});
labElement('neuron-network').addEventListener('keydown',event=>{
  const node=event.target.closest('[data-step]');if(node&&(event.key==='Enter'||event.key===' ')){event.preventDefault();labChooseStep(Number(node.dataset.step));labElement('neuron-network').querySelector(`[data-step="${labStep}"]`)?.focus();}
});
labElement('next-neuron').addEventListener('click',()=>labChooseStep(labStep+1));
labElement('previous-neuron').addEventListener('click',()=>labChooseStep(labStep-1));
['input-failures','input-volume'].forEach((id,index)=>labElement(id).addEventListener('input',event=>{labInputs[index]=Number(event.target.value);labRenderLive();}));
document.querySelectorAll('[data-preset]').forEach(button=>button.addEventListener('click',()=>{
  labInputs=button.dataset.preset==='low'?[0.1,0.1]:button.dataset.preset==='failures'?[0.9,0.1]:[0.9,0.9];
  labElement('input-failures').value=labInputs[0];labElement('input-volume').value=labInputs[1];labRenderLive();
}));
labElement('neuron-controls').addEventListener('input',event=>{
  const control=event.target,neuron=labSelected();if(!neuron||!control.dataset.parameter)return;
  const value=Number(control.value),index=Number(control.dataset.index),parameters=labNetwork[neuron.layer];
  if(control.dataset.parameter==='weight')parameters.weights[neuron.index][index]=value;else parameters.bias[neuron.index]=value;
  labElement(`parameter-${control.dataset.parameter}-${index}`).textContent=value.toFixed(2);
  labRenderLive();
});
labElement('reset-network').addEventListener('click',()=>{
  labNetwork=initialNetwork();labInputs=[0.7,0.4];
  labElement('input-failures').value=0.7;labElement('input-volume').value=0.4;labChooseStep(0);
});
labElement('features-mode').addEventListener('click',()=>labEncoding('features'));labElement('tokens-mode').addEventListener('click',()=>labEncoding('tokens'));
labElement('sona-destination').addEventListener('click',()=>labDestination('sona'));labElement('llm-destination').addEventListener('click',()=>labDestination('llm'));
labEncoding('features');labDestination('sona');labChooseStep(0);
