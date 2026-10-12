import {versionNotes, validateVersions, selectedSummary} from './versions-data.mjs?v=sona-prod-polish-1';
import {escapeHTML as esc, human, count, percent, modelName} from './data.mjs?v=sona-prod-polish-1';
import {validateResearch} from './research-data.mjs?v=sona-prod-polish-1';
const $=id=>document.getElementById(id);
let data, researchData, researchFailed=false;
const candidateNotes={
  v1:'Beta 1 judges each connection on its own. This counterpart practices on standard attacks and the shared timing examples, without slowed-down ordinary attacks.',
  v2:'Beta 2 keeps Beta 1’s individual-connection design and adds slowed-down attack examples. That broadens its coverage, but voting on separate guesses still loses the order of events. Identical measurements in a different order receive the same vote.',
  v3:'Beta 3 adds a small memory as it reads related connections in order. It has the same narrower ordinary-attack training as Beta 1 and the same timing practice as every model here. This tests what memory adds before combining it with broader training.',
  v4:'Sona combines the session-memory design explored in Beta 3 with the broader attack examples that helped Beta 2. The aim is coverage of individual signals and patterns over time. This comparison uses a separately trained Sona counterpart; Overview still uses its existing model.'
};

const comparisonHelp={
  ordinary:'Familiar attack examples and benign activity at their usual pace. This checks basic traffic classification. A high score here alone does not show whether a model can recognize a pattern across connections.',
  slowed:'The same attack families carried out more slowly or spread over time, so individual connections can look less unusual. This checks whether training covers changes in attack behavior. Sona combines that broader practice with session memory.',
  timing:'Harmless and attack sessions contain identical individual measurements, arranged in different orders. This checks recognition of the sequence, not execution speed. Beta 2 loses the order when it votes on individual guesses; Sona’s memory can learn it.',
  combined:'The ordinary, slowed-down and timing accuracy scores are added and divided by three. Each test type contributes one-third, regardless of its number of sessions. This shows basic coverage, resistance to slower attacks and sequence recognition together. It summarizes these synthetic tests, not every IDS capability.'
};
function comparisonHeading(label,key,subtitle=''){
  return `<th><span class="comparison-label">${esc(label)} <button type="button" class="metric-info comparison-info" data-metric-help="${key}" aria-label="About ${esc(label)}" aria-describedby="comparison-help-${key}">i<span id="comparison-help-${key}" class="metric-tooltip" role="tooltip">${esc(comparisonHelp[key])}</span></button></span>${subtitle?`<br><span>${esc(subtitle)}</span>`:''}</th>`;
}
function attachComparisonHelp(){
  $('version-comparison').querySelectorAll('[data-metric-help]').forEach(button=>{
    const position=()=>{
      button.classList.remove('help-dismissed');
      const box=button.getBoundingClientRect(),tooltip=button.querySelector('[role="tooltip"]');
      const width=Math.min(260,window.innerWidth-32);
      tooltip.style.left=`${Math.max(16,Math.min(box.left+box.width/2-width/2,window.innerWidth-width-16))}px`;
      tooltip.style.top=`${Math.max(16,Math.min(box.bottom,window.innerHeight-tooltip.offsetHeight-16))}px`;
    };
    button.addEventListener('pointerenter',position);
    button.addEventListener('focus',position);
    button.addEventListener('click',position);
    button.addEventListener('keydown',event=>{if(event.key==='Escape')button.classList.add('help-dismissed');});
  });
}

const versionKeys=Object.keys(versionNotes);
const latestVersion=versionKeys[versionKeys.length-1];
$('version-select').value=latestVersion;
function render() {
  const selected=$('version-select').value,notes=versionNotes[selected],summary=selectedSummary(data,selected);
  const four=researchData?.four_model_comparison;
  $('version-progression').innerHTML=Object.entries(versionNotes).map(([key,n])=>`<button data-version="${key}" aria-pressed="${key===selected}"><span>${modelName(key)}${key===latestVersion?' · Final':''}</span><strong>${n.title}</strong><small>${n.view}</small></button>`).join('');
  $('version-progression').querySelectorAll('[data-version]').forEach(button=>button.addEventListener('click',()=>{$('version-select').value=button.dataset.version;render();}));
  const beta2=data.versions.v2;
  const sona=data.versions.v4;
  const same=sona.normal_session_metrics.macro_f1===beta2.normal_session_metrics.macro_f1 && sona.stealth_session_metrics.macro_f1===beta2.stealth_session_metrics.macro_f1;
  const comparisonText=same?'Beta 2 and Sona tie on the original traffic tests. Broader training helps both recognize familiar attack variations. Sona also reads related connections in order, which lets its timing-trained candidate recognize patterns that individual connections cannot reveal.':'The original traffic tests and the timing study measure different strengths. Broader training helps recognize familiar attack variations; session memory adds a view of how related connections unfold.';
  const misses=data.class_names.filter(name=>summary.stealth.per_class[name].support>0&&summary.stealth.per_class[name].recall<1);
  $('version-detail').innerHTML=`<p class="eyebrow">${modelName(selected)} · ${selected===latestVersion?'FINAL MODEL':'BETA ITERATION'}</p><h3>${selected===latestVersion?modelName(selected):notes.title}</h3><p class="version-description">${four?candidateNotes[selected]:notes.text}${!four&&selected===latestVersion?` <span class="sona-comparison-context">${comparisonText}</span>`:''}</p><dl class="version-facts"><div><dt>Looks at</dt><dd>${notes.view}</dd></div><div><dt>Trained on</dt><dd>${four?(selected==='v1'||selected==='v3'?'Standard attacks + shared timing practice':'Standard and slowed-down attacks + shared timing practice'):notes.training}</dd></div><div><dt>${four?'Slowed-down accuracy':'Correct guesses<br>Original slowed-down test'}</dt><dd><strong>${four?percent(four.results[selected].stealth.mean):`${summary.correct} / ${summary.stealth.n}`}</strong>${four?' across three runs':' sessions'}</dd></div></dl><p class="version-help">${four?`Retrained ${modelName(selected)} counterpart. The comparison uses shared timing practice and the same held-out sessions for every model.`:misses.length?`Missed family guesses include ${misses.map(human).map(esc).join(', ')}.`:'All family guesses were correct in this saved test.'}</p><a class="version-learn-link" href="model.html">How the model learns →</a>`;
  const study=researchData?.studies.find(study=>!study.archived&&study.models?.some(model=>model.id==='relative'));
  const comparison=study?.comparison;
  if(four){
    $('version-comparison').innerHTML=`<table class="version-table"><caption class="sr-only">All four retrained candidates on identical ordinary, slowed-down and timing sessions</caption><thead><tr><th>Model</th>${comparisonHeading('Ordinary','ordinary')}${comparisonHeading('Slowed-down','slowed')}${comparisonHeading('Timing','timing')}${comparisonHeading('Three-test','combined','Equal weight')}</tr></thead><tbody>${versionKeys.map(key=>{const scores=four.results[key];return `<tr class="${key===selected?'selected-version':''}"><th scope="row">${modelName(key)}${key===selected?' <span class="sr-only">· Selected</span>':''}</th><td>${percent(scores.normal.mean)}</td><td>${percent(scores.stealth.mean)}</td><td>${percent(scores.timing.mean)}</td><td><strong>${percent((scores.normal.mean+scores.stealth.mean+scores.timing.mean)/3)}</strong></td></tr>`;}).join('')}</tbody></table>`;
  } else {
  $('version-comparison').innerHTML=`<table class="version-table"><caption class="sr-only">Ordinary and timing accuracy of the same retrained models, plus an equally weighted two-test score</caption><thead><tr><th>Model</th><th>Ordinary<br><span>Accuracy</span></th><th>Timing<br><span>Accuracy</span></th><th>Two-test<br><span>Equal weight</span></th></tr></thead><tbody>${versionKeys.map(key=>{const score=key==='v2'?comparison?.models.beta2:key==='v4'?comparison?.models.sona:null;const empty=key==='v1'||key==='v3'?'Not tested':researchFailed?'Unavailable':'Loading…';return `<tr class="${key===selected?'selected-version':''}"><th scope="row">${modelName(key)}${key===selected?' <span class="sr-only">· Selected</span>':''}</th>${score?`<td>${percent(score.ordinary)}</td><td>${percent(score.timing)}</td><td><strong>${percent((score.ordinary+score.timing)/2)}</strong></td>`:`<td colspan="3">${empty}</td>`}</tr>`;}).join('')}</tbody></table>`;
  }
  attachComparisonHelp();
  if(four){
    const results=four.results;
    const totals=versionKeys.map(key=>({key,score:(results[key].normal.mean+results[key].stealth.mean+results[key].timing.mean)/3}));
    const best=Math.max(...totals.map(item=>item.score));
    const leaders=totals.filter(item=>Math.abs(item.score-best)<1e-12).map(item=>modelName(item.key)).join(' and ');
    $('version-findings').textContent=`Adding memory: Beta 3 scores ${percent(results.v3.timing.mean)} on timing, versus Beta 2’s ${percent(results.v2.timing.mean)}. Adding broader attack practice: Sona scores ${percent(results.v4.stealth.mean)} on slowed-down attacks, versus Beta 3’s ${percent(results.v3.stealth.mean)}. ${leaders} has the strongest three-test average, ${percent(best)}. Beta 3 is slightly ahead on timing alone; Sona’s advantage is the combination of coverage and memory. Beta 1 and Beta 2 tie in this follow-up; their earlier progression comes from the original traffic tests.`;
  } else {
  $('version-findings').textContent='Beta 2 improved by practicing on more varied attacks. Beta 3 tested session memory with narrower training. Sona combines both ideas. The timing study tests the added value of memory; ordinary traffic scores alone do not show it. Subtle patterns and extreme timing noise remain harder cases.';
  }
}
$('version-select').addEventListener('change',()=>{if(data&&versionNotes[$('version-select').value])render();});
try {const response=await fetch('./versions.json', {cache:'no-store'});if(!response.ok)throw new Error(`HTTP ${response.status}`);data=validateVersions(await response.json());render();$('versions-content').hidden=false;$('versions-status').hidden=true;}
catch(error){$('versions-status').textContent=`Version comparison unavailable: ${error.message} Detection records remain available on Overview.`;$('versions-status').className='status error';}

try {
  const response=await fetch('./research.json',{cache:'no-store'});
  if(!response.ok)throw new Error(`HTTP ${response.status}`);
  const research=validateResearch(await response.json());
  researchData=research;
  if(data)render();

  $('research-status').hidden=true;
} catch(error){researchFailed=true;if(data)render();$('research-status').textContent=`Experimental results unavailable: ${error.message}`;$('research-status').className='status error';}
