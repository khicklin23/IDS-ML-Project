import {installMobileSupport} from './mobile.mjs?v=sona-mobile-compact-1';
installMobileSupport();
import {versionNotes, validateVersions, selectedSummary} from './versions-data.mjs?v=sona-prod-polish-1';
import {escapeHTML as esc, human, count, percent, modelName} from './data.mjs?v=sona-prod-polish-1';
import {validateResearch} from './research-data.mjs?v=sona-prod-polish-1';
const $=id=>document.getElementById(id);
let data, researchData, researchFailed=false;
const candidateNotes={
  v1:'Beta 1 learns the clues within one connection. It can recognize familiar traffic, but cannot distinguish two sessions that contain the same connections in different orders.',
  v2:'Beta 2 adds practice on lower-rate attacks. It gets better coverage without changing how it reads traffic: separate connection guesses are combined by voting, so their order is lost.',
  v3:'Beta 3 tests a new ability: remembering related connections as it reads them. It learns ordered patterns, but has narrower attack training. Memory adds context; it does not replace varied examples.',
  v4:'Sona combines Beta 2’s broader attack practice with Beta 3’s memory. It can use clues within connections and patterns across them, giving it the highest average in this synthetic comparison. Practical detection gains still need testing.'
};

const comparisonHelp={
  ordinary:'Familiar attacks and normal activity at their usual pace. This checks whether the model recognizes the basic traffic categories, not whether it understands event order.',
  slowed:'Attacks use fewer connections, longer gaps, or lower throughput. Connection details such as failed outcomes and packet sizes can still reveal them. A model without memory can do well here.',
  timing:'Tests whether a model can tell apart sessions containing the same connection measurements in different sequences. This isolates the value of memory: recognizing relationships across events that individual-connection voting misses, a foundation for learning more complex behavior.',
  combined:'Add the usual-traffic, lower-rate and event-order accuracy scores, then divide by three. Each category contributes one-third, regardless of its number of sessions. This measures balance across these synthetic tests, not every IDS capability.'
};
function comparisonHeading(label,key,subtitle=''){
  return `<th><span class="comparison-label">${esc(label)} <button type="button" class="metric-info comparison-info" data-metric-help="${key}" aria-label="About ${esc(label)}" aria-describedby="comparison-help-${key}">i<span id="comparison-help-${key}" class="metric-tooltip" role="tooltip">${esc(comparisonHelp[key])}</span></button></span>${subtitle?`<br><span>${esc(subtitle)}</span>`:''}</th>`;
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
  $('version-detail').innerHTML=`<p class="eyebrow">${modelName(selected)} · ${selected===latestVersion?'FINAL MODEL':'BETA ITERATION'}</p><h3>${selected===latestVersion?modelName(selected):notes.title}</h3><p class="version-description">${four?candidateNotes[selected]:notes.text}${!four&&selected===latestVersion?` <span class="sona-comparison-context">${comparisonText}</span>`:''}</p><dl class="version-facts"><div><dt>Looks at</dt><dd>${notes.view}</dd></div><div><dt>Trained on</dt><dd>${four?(selected==='v1'||selected==='v3'?'Standard attacks + shared timing practice':'Standard and slowed-down attacks + shared timing practice'):notes.training}</dd></div><div><dt>${four?'Three-test mean':'Correct guesses<br>Original slowed-down test'}</dt><dd><strong>${four?percent((four.results[selected].normal.mean+four.results[selected].stealth.mean+four.results[selected].timing.mean)/3):`${summary.correct} / ${summary.stealth.n}`}</strong>${four?' across three runs':' sessions'}</dd></div></dl><p class="version-help">${four?`Retrained ${modelName(selected)} counterpart with shared timing practice. Overview uses a separately trained Sona checkpoint.`:misses.length?`Missed family guesses include ${misses.map(human).map(esc).join(', ')}.`:'All family guesses were correct in this saved test.'}</p><a class="version-learn-link" href="model.html">How the model learns →</a>`;
  const study=researchData?.studies.find(study=>!study.archived&&study.models?.some(model=>model.id==='relative'));
  const comparison=study?.comparison;
  if(four){
    const tiedBetas=['normal','stealth','timing'].every(test=>four.results.v1[test].mean===four.results.v2[test].mean);
    const tableKeys=tiedBetas?['v2','v3','v4']:versionKeys;
    $('version-comparison').innerHTML=`<table class="version-table"><caption class="sr-only">All four retrained candidates on identical ordinary, slowed-down and timing sessions</caption><thead><tr><th>Model</th>${comparisonHeading('Usual traffic','ordinary')}${comparisonHeading('Lower-rate','slowed')}${comparisonHeading('Sequence timing','timing')}${comparisonHeading('All three','combined','Equal weight')}</tr></thead><tbody>${tableKeys.map(key=>{const scores=four.results[key];const rowSelected=key===selected||(tiedBetas&&key==='v2'&&selected==='v1');return `<tr class="${rowSelected?'selected-version':''}"><th scope="row">${tiedBetas&&key==='v2'?'Beta 1 &amp; 2':modelName(key)}${rowSelected?' <span class="sr-only">· Selected</span>':''}</th><td>${percent(scores.normal.mean)}</td><td>${percent(scores.stealth.mean)}</td><td>${percent(scores.timing.mean)}</td><td><strong>${percent((scores.normal.mean+scores.stealth.mean+scores.timing.mean)/3)}</strong></td></tr>`;}).join('')}</tbody></table>`;
  } else {
  $('version-comparison').innerHTML=`<table class="version-table"><caption class="sr-only">Ordinary and timing accuracy of the same retrained models, plus an equally weighted two-test score</caption><thead><tr><th>Model</th><th>Ordinary<br><span>Accuracy</span></th><th>Timing<br><span>Accuracy</span></th><th>Two-test<br><span>Equal weight</span></th></tr></thead><tbody>${versionKeys.map(key=>{const score=key==='v2'?comparison?.models.beta2:key==='v4'?comparison?.models.sona:null;const empty=key==='v1'||key==='v3'?'Not tested':researchFailed?'Unavailable':'Loading…';return `<tr class="${key===selected?'selected-version':''}"><th scope="row">${modelName(key)}${key===selected?' <span class="sr-only">· Selected</span>':''}</th>${score?`<td>${percent(score.ordinary)}</td><td>${percent(score.timing)}</td><td><strong>${percent((score.ordinary+score.timing)/2)}</strong></td>`:`<td colspan="3">${empty}</td>`}</tr>`;}).join('')}</tbody></table>`;
  }

  if(four){
    const results=four.results;
    const totals=versionKeys.map(key=>({key,score:(results[key].normal.mean+results[key].stealth.mean+results[key].timing.mean)/3}));
    const best=Math.max(...totals.map(item=>item.score));
    const leaders=totals.filter(item=>Math.abs(item.score-best)<1e-12).map(item=>modelName(item.key)).join(' and ');
    $('memory-value').textContent='Beta 1 and 2 classify individual connections. Beta 3 introduces sequence memory; Sona combines it with broader attack training so the model can learn how events relate across a session.';
    $('version-findings').textContent=`Memory makes a new kind of pattern visible: Beta 3 reaches ${percent(results.v3.timing.mean)} on event order. Broader training completes the design: Sona raises lower-rate accuracy from Beta 3’s ${percent(results.v3.stealth.mean)} to ${percent(results.v4.stealth.mean)}. ${leaders} leads the three-test mean at ${percent(best)}. Beta 1 and Beta 2 still score slightly higher on lower-rate traffic, and Beta 3 is slightly ahead on event order. Sona’s strength is the balance, not winning every column.`;
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
