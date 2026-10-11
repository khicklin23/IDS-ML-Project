import {validateReport, escapeHTML as esc, human, count} from './data.mjs';
const $ = id => document.getElementById(id);
let report, layer = 'input';
const buttons = [...document.querySelectorAll('[data-layer]')];
function renderNetwork() {
  const family = $('model-choice').value === 'multiclass_model';
  const config = report[$('model-choice').value].train_config;
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
    input: ['Start with measurements', 'Each flow becomes a row of numbers. Scaling puts numerical measurements on comparable ranges (so a byte count in the thousands doesn’t drown out a duration measured in single digits); categorical fields (like protocol or service, which aren’t numbers) become indicator columns — one yes/no switch per possible value. The scenario label is the answer used during training, not an input.'],
    hidden1: ['Combine several clues', `The first layer has ${width} neurons (small processing units, each one a tiny calculator). Each neuron forms a weighted combination of the inputs plus a small adjustment (called a bias), then applies ReLU — a simple rule that keeps positive results and turns negative ones into zero. A neuron can respond to several clues together; it is not assigned a rule such as “detect port scans.”`],
    hidden2: ['Combine the learned responses', `The second layer has ${second} neurons. It combines the first layer’s responses into another representation (its own reorganized version of the same information). These nonlinear combinations (ones that aren’t just a straight-line scaling) let similar byte counts produce different predictions when duration, service, or connection outcome differ.`],
    output: family ? ['Choose a behavior family', `This separately trained network has ${outputs} outputs — one per attack family, plus benign. Softmax (a formula that turns raw scores into percentages that add up to 100%) turns them into a distribution; the highest value names the predicted class. These are learned similarities, not proof of an attack’s intent.`] : ['Make a flagging decision', 'This separately trained network has one output. Sigmoid (a curve that squeezes any number into a range between zero and one) turns it into a suspiciousness score. Comparing that score with the configured threshold decides whether to flag the flow.']
  };
  const [title, text] = explanations[layer];
  $('layer-explanation').innerHTML = `<h3>${title}</h3><p>${text}</p>`;
  buttons.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.layer === layer)));
  $('architecture-details').textContent = `Encoded feature vector → fully-connected layer (${width} neurons) → ReLU → Dropout (randomly ignores some neurons during training only, so the network can’t just memorize examples — here, ${Math.round(config.dropout * 100)}% of the time) → fully-connected layer (${second} neurons) → ReLU → fully-connected layer (${outputs} output${outputs === 1 ? '' : 's'}). Training works with the raw output numbers (called logits); turning those into a 0–1 score with ${family ? 'softmax' : 'sigmoid'} only happens when making an actual prediction. The input width depends on the saved preprocessing and is not included in this report.`;
}
function neuron() {
  const weights = ['bytes','duration','outcome'].map(name => { const value = Number($(`weight-${name}`).value); $(`value-${name}`).textContent = value.toFixed(1); return value; });
  const sum = .8 * weights[0] - .4 * weights[1] + weights[2] - .5;
  $('neuron-result').innerHTML = `<span>Weighted sum + bias (−0.5)</span><strong>${sum.toFixed(2)}</strong><span>After ReLU</span><strong>${Math.max(0,sum).toFixed(2)}</strong>`;
}
buttons.forEach(button => button.addEventListener('click', () => { layer = button.dataset.layer; if (report) renderNetwork(); }));
$('model-choice').addEventListener('change', () => { if (report) renderNetwork(); });
['bytes','duration','outcome'].forEach(name => $(`weight-${name}`).addEventListener('input', neuron));
neuron();
const descriptions = {
  benign: 'Ordinary browsing, DNS and other normal activity. These examples teach the models what should be left alone.',
  port_scan: 'Connections probing a range of ports, often with short or unsuccessful flows.',
  auth_bruteforce: 'Repeated authentication attempts, including many failed connections.',
  beaconing: 'Small repeated communications. This flow-level model sees each record, not the entire timing sequence shown here.',
  service_flood: 'A burst of connections directed at a service, with varied packet counts and outcomes.',
  outbound_transfer: 'Unusually large outbound transfers, with more bytes sent than received.'
};
const sketches = {benign:[8,15,9,25,14,20,7,18,13,22],port_scan:[24,24,24,24,24,24,24,24,24,24],auth_bruteforce:[8,8,30,8,8,30,8,8,30,8],beaconing:[3,26,3,3,26,3,3,26,3,3],service_flood:[3,8,18,35,36,34,37,34,18,5],outbound_transfer:[5,7,10,20,30,40,38,40,39,35]};
try {
  const response = await fetch('./report.json');
  if (!response.ok) throw new Error(`Report request failed (${response.status}).`);
  report = validateReport(await response.json());
  renderNetwork();
  $('model-status').textContent = `Configuration and counts from saved run ${report.run_id}. Illustrations explain the architecture; they do not display recorded neuron activity.`;
  $('training-source').textContent = `${count(report.dataset.manifest.row_count)} generated flows · ground truth`;
  const entries = Object.entries(report.dataset.manifest.scenario_counts).sort(([a],[b]) => a === 'benign' ? -1 : b === 'benign' ? 1 : a.localeCompare(b));
  $('training-examples').innerHTML = entries.map(([name,n]) => {
    const raw = sketches[name] || [12,25,18,30,15];
    const max = Math.max(...raw);
    // Normalize each sketch to its own tallest bar so every card reads as
    // equally "full" -- these are illustrative shapes, not a scale that's
    // meant to compare across cards, so a shared axis would mislead.
    const bars = raw.map((h,i) => `<rect x="${i*18}" y="${44 - (h / max) * 40}" width="10" height="${(h / max) * 40}" rx="2" fill="currentColor"/>`).join('');
    return `<article class="scenario-example ${name === 'benign' ? 'normal-example' : ''}"><div><span class="example-kind">${name === 'benign' ? 'NORMAL TRAFFIC' : 'SIMULATED ATTACK'}</span><h3>${esc(human(name))}</h3><p>${esc(descriptions[name] || 'Generated examples for this behavior family.')}</p></div><svg viewBox="0 0 180 48" aria-hidden="true">${bars}</svg><strong>${count(n)} <small>flows</small></strong></article>`;
  }).join('');
  $('saved-training-settings').innerHTML = ['binary_model','multiclass_model'].map(key => `<div class="detail-item"><span>${key === 'binary_model' ? 'Binary detector' : 'Family classifier'}</span><strong>${Object.entries(report[key].train_config).map(([k,v]) => `${esc(human(k))}: ${esc(v)}`).join(' · ')}</strong></div>`).join('') + `<div class="detail-item"><span>Separate data splits</span><strong>${Object.entries(report.dataset.split_sizes).map(([k,v]) => `${esc(human(k))}: ${count(v)}`).join(' · ')}</strong></div>`;
  const threshold = report.binary_model.test_metrics.threshold;
  const thresholdPct = Math.max(0, Math.min(1, threshold)) * 100;
  $('threshold-marker').style.left = `${thresholdPct}%`;
  $('threshold-normal-zone').style.width = `${thresholdPct}%`;
  $('threshold-description').textContent = `Saved decision threshold: ${threshold}. Scores below this are treated as normal; scores at or above it are flagged. This scale shows the cutoff, not an individual flow’s prediction. The score is not a guarantee or an attack severity.`;
  $('class-outputs').innerHTML = report.multiclass_model.class_names.map(name => `<span class="${name === 'benign' ? 'normal-output' : ''}">${esc(human(name))}</span>`).join('');
  $('explanation-version').textContent = `Report ${report.report_version} · Dataset schema ${report.schema_version}`;
} catch (error) {
  $('model-status').textContent = `Saved configuration unavailable: ${error.message} General explanations remain available; run-specific diagrams and counts could not be loaded.`;
  $('model-status').classList.add('error');
}
