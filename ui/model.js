import {installMobileSupport} from './mobile.mjs?v=sona-mobile-compact-1';
installMobileSupport();
import {validateReport, escapeHTML as esc, human, count} from './data.mjs?v=sona-prod-polish-1';
const $ = id => document.getElementById(id);
let report, layer = 'input';
const buttons = [...document.querySelectorAll('[data-layer]')];
function renderNetwork() {
  const family = true;
  const config = report.multiclass_model.train_config;
  if (report.detector?.kind === 'session_sequence') {
    const dimension=report.detector.hidden_dim, outputs=report.multiclass_model.class_names.length;
    const names=['input','hidden1','hidden2','output'], xs=[105,355,615,865];
    const points=[6,8,8,outputs].map((n,i)=>Array.from({length:n},(_,j)=>[xs[i],72+j*160/Math.max(1,n-1)]));
    const active=names.indexOf(layer);
    let svg=`<title>Sona neural network with ${dimension} memory values and ${outputs} family outputs</title><defs><marker id="memory-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8" fill="#497ea6"/></marker></defs>`;
    points.slice(0,-1).forEach((nodes,i)=>nodes.forEach(([x,y])=>points[i+1].forEach(([nx,ny])=>{
      const selected=i===active||i+1===active;
      svg+=`<line x1="${x}" y1="${y}" x2="${nx}" y2="${ny}" stroke="${selected?'#a5bfd4':'#e4ebf1'}" stroke-width="${selected?'1.15':'.8'}"/>`;
    })));
    svg+=`<path d="M367 69C485 6 225 6 343 69" fill="none" stroke="${layer==='hidden1'?'#286397':'#769bb8'}" stroke-width="2" marker-end="url(#memory-arrow)"/><text x="355" y="22" text-anchor="middle" font-size="11" fill="#526d83">Memory carried to the next flow</text>`;
    const labels=['Ordered session inputs',`${dimension} memory values`,'Final session summary',`${outputs} family outputs`];
    const captions=['Related connections in time order','Updated across the session','Memory after the final connection','Normal or an attack family'];
    points.forEach((nodes,i)=>{
      nodes.forEach(([x,y])=>{svg+=`<circle cx="${x}" cy="${y}" r="${i===active?11:9}" fill="${i===active?'#286397':'#f0f5fa'}" stroke="${i===active?'#1e527b':'#7fa1bc'}" stroke-width="1.5"/>`;});
      svg+=`<text x="${xs[i]}" y="269" text-anchor="middle" font-size="14" fill="#263342">${labels[i]}</text><text x="${xs[i]}" y="290" text-anchor="middle" font-size="11" fill="#617080">${captions[i]}</text>`;
    });
    $('model-network').innerHTML=svg;
    const content={input:['Read related flows in order','Sona receives an ordered session of related connections, including traffic measurements and timing gaps. It reads each connection alongside the memory built from earlier ones, so the same event can have a different meaning in a different sequence. Labels and IP addresses stay out of the inputs.'],hidden1:['Update a small memory',`GRU means gated recurrent unit: a neural component that updates its memory for each connection. Learned update gates mix old and new information; reset gates control how much previous context shapes the new state. This saved model keeps ${dimension} memory values. These values work together to represent patterns; individual units are not assigned to specific attack types. Training learns their weights, and memory starts fresh for each session.`],hidden2:['Summarize the session','The final memory summarizes the sequence the model read. It is not a second fully-connected hidden layer. Similar individual flows can lead to different session summaries.'],output:['Choose a family, then decide whether to flag',`The same network has ${outputs} family outputs. Softmax converts their raw scores into values that add up to one. The highest score chooses the family. Any family other than benign flags the session; suspiciousness is 1 minus the benign probability.`]};
    const [title,text]=content[layer];$('layer-explanation').innerHTML=`<h3>${title}</h3><p>${text}</p>`;
    buttons.forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.layer===layer)));
    return;
  }
  const width = config.hidden_dim, second = Math.floor(width / 2);
  const outputs = family ? report.multiclass_model.class_names.length : 1;
  const sizes = [5,7,5,outputs], xs = [110,360,610,880];
  const labels = ['Encoded features', `${width} neurons`, `${second} neurons`, `${outputs} output${outputs === 1 ? '' : 's'}`];
  const layers = ['input','hidden1','hidden2','output'];
  const activeIdx = layers.indexOf(layer);
  const points = sizes.map((n,i) => Array.from({length:n},(_,j) => [xs[i],n === 1 ? 140 : 68 + j * 164 / (n-1)]));
  let svg = '<title>Two hidden layers connecting flow features to model outputs</title>';
  // Connections touching the selected layer light up, so picking a layer
  // below shows not just that layer but where its signal comes from / goes.
  points.slice(0,-1).forEach((nodes,i) => {
    const isNear = i === activeIdx || i + 1 === activeIdx;
    nodes.forEach(([x,y]) => points[i+1].forEach(([nx,ny]) => {
      svg += `<line x1="${x}" y1="${y}" x2="${nx}" y2="${ny}" stroke="${isNear ? '#9bbdda' : '#e6ebef'}" stroke-width="${isNear ? 1.3 : 1}"/>`;
    }));
  });
  points.forEach((nodes,i) => {
    const active = layers[i] === layer;
    nodes.forEach(([x,y]) => {
      svg += `<circle cx="${x}" cy="${y}" r="${active ? 13 : 11}" fill="${active ? '#286397' : '#edf3f8'}" stroke="${active ? '#1b4a73' : '#9fb7cb'}" stroke-width="${active ? 2 : 1.3}"/>`;
    });
    svg += `<text x="${xs[i]}" y="264" text-anchor="middle" fill="#263342" font-size="15" font-weight="${active ? 600 : 400}">${labels[i]}</text><text x="${xs[i]}" y="288" text-anchor="middle" fill="#617080" font-size="12">${['Input','Hidden layer 1 · ReLU','Hidden layer 2 · ReLU',family ? 'Softmax · class choice' : 'Sigmoid · suspicious score'][i]}</text>`;
  });
  $('model-network').innerHTML = svg;
  const explanations = {
    input: ['Start with measurements', 'Each flow becomes a row of numbers. Scaling puts numerical measurements on comparable ranges (so a byte count in the thousands doesn’t drown out a duration measured in single digits); categorical fields (like protocol or service, which aren’t numbers) become indicator columns, with one yes/no switch per possible value. The scenario label is the answer used during training, not an input.'],
    hidden1: ['Combine several clues', `The first layer has ${width} neurons (small processing units, each one a tiny calculator). Each neuron forms a weighted combination of the inputs plus a small adjustment (called a bias), then applies ReLU, a simple rule that keeps positive results and turns negative ones into zero. A neuron can respond to several clues together; it is not assigned a rule such as “detect port scans.”`],
    hidden2: ['Combine the learned responses', `The second layer has ${second} neurons. It combines the first layer’s responses into another representation (its own reorganized version of the same information). These nonlinear combinations (ones that aren’t just a straight-line scaling) let similar byte counts produce different predictions when duration, service, or connection outcome differ.`],
    output: family ? ['Choose a behavior family', `This separately trained network has ${outputs} outputs, one per attack family, plus benign. Softmax (a formula that turns raw scores into percentages that add up to 100%) turns them into a distribution; the highest value names the predicted class. These are learned similarities, not proof of an attack’s intent.`] : ['Make a flagging decision', 'This separately trained network has one output. Sigmoid (a curve that squeezes any number into a range between zero and one) turns it into a suspiciousness score. Comparing that score with the configured threshold decides whether to flag the flow.']
  };
  const [title, text] = explanations[layer];
  $('layer-explanation').innerHTML = `<h3>${title}</h3><p>${text}</p>`;
  buttons.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.layer === layer)));
}
buttons.forEach(button => button.addEventListener('click', () => { layer = button.dataset.layer; if (report) renderNetwork(); }));
const descriptions = {
  benign: 'Ordinary browsing, DNS and other normal activity. These examples teach the models what should be left alone.',
  port_scan: 'Connections probing a range of ports, often with short or unsuccessful flows.',
  auth_bruteforce: 'Repeated authentication attempts, including many failed connections.',
  beaconing: 'Small repeated communications. The session model can read an ordered group of connections; this sketch is not a measured traffic trace.',
  service_flood: 'A burst of connections directed at a service, with varied packet counts and outcomes.',
  outbound_transfer: 'Unusually large outbound transfers, with more bytes sent than received.'
};
const sketches = {benign:[8,15,9,25,14,20,7,18,13,22],port_scan:[24,24,24,24,24,24,24,24,24,24],auth_bruteforce:[8,8,30,8,8,30,8,8,30,8],beaconing:[3,26,3,3,26,3,3,26,3,3],service_flood:[3,8,18,35,36,34,37,34,18,5],outbound_transfer:[5,7,10,20,30,40,38,40,39,35]};
try {
  const response = await fetch('./report.json', {cache:'no-store'});
  if (!response.ok) throw new Error(`Report request failed (${response.status}).`);
  report = validateReport(await response.json());
  renderNetwork();
  $('model-status').hidden = true;
  $('training-source').textContent = `${count(report.dataset.manifest.row_count)} generated flows · ground truth`;
  const entries = Object.entries(report.dataset.manifest.scenario_counts).sort(([a],[b]) => a === 'benign' ? -1 : b === 'benign' ? 1 : a.localeCompare(b));
  $('training-examples').innerHTML = entries.map(([name,n]) => {
    const raw = sketches[name] || [12,25,18,30,15];
    const max = Math.max(...raw);
    // Normalize each sketch to its own tallest bar so every card reads as
    // equally "full" -- these are illustrative shapes, not a scale that's
    // meant to compare across cards, so a shared axis would mislead.
    const bars = raw.map((h,i) => `<rect x="${i*18}" y="${44 - (h / max) * 40}" width="10" height="${(h / max) * 40}" rx="2" fill="currentColor"/>`).join('');
    return `<article data-scenario="${esc(name)}" class="scenario-example ${name === 'benign' ? 'normal-example' : ''}"><div><span class="example-kind">${name === 'benign' ? 'NORMAL TRAFFIC' : 'SIMULATED ATTACK'}</span><h3>${esc(human(name))}</h3><p>${esc(descriptions[name] || 'Generated examples for this behavior family.')}</p></div><svg viewBox="0 0 180 48" aria-hidden="true">${bars}</svg><strong>${count(n)} <small>flows</small></strong></article>`;
  }).join('');
  const threshold = report.binary_model.test_metrics.threshold;
  const thresholdPct = Number.isFinite(threshold) ? Math.max(0, Math.min(1, threshold)) * 100 : 0;
  $('threshold-marker').style.left = `${thresholdPct}%`;
  $('threshold-normal-zone').style.width = `${thresholdPct}%`;
  $('threshold-description').textContent = report.detector?.kind === 'session_sequence' ? 'No fixed score cutoff is used for this report. The session is flagged when the highest-scoring family is not benign. Suspiciousness is 1 minus the benign probability; this score is not severity or a guarantee.' : `Saved decision threshold: ${threshold}. Scores below this are treated as normal; scores at or above it are flagged. This scale shows the cutoff, not an individual flow’s prediction. The score is not a guarantee or an attack severity.`;
  if (report.detector?.kind === 'session_sequence') $('threshold-visual').hidden=true;
  $('class-outputs').innerHTML = report.multiclass_model.class_names.map(name => `<span class="${name === 'benign' ? 'normal-output' : ''}">${esc(human(name))}</span>`).join('');
  $('explanation-version').textContent = `Report ${report.report_version} · Dataset schema ${report.schema_version}`;
} catch (error) {
  $('model-status').textContent = `Saved configuration unavailable: ${error.message} General explanations remain available; run-specific diagrams and counts could not be loaded.`;
  $('model-status').classList.add('error');
}
