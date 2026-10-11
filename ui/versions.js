import {versionNotes, validateVersions, selectedSummary} from './versions-data.mjs?v=v4-20261011-3';
import {escapeHTML as esc, human, count, percent} from './data.mjs?v=v4-20261011-3';
const $=id=>document.getElementById(id);
let data;
const versionKeys=Object.keys(versionNotes);
const latestVersion=versionKeys[versionKeys.length-1];
$('version-select').value=latestVersion;
function render() {
  const selected=$('version-select').value,notes=versionNotes[selected],summary=selectedSummary(data,selected);
  $('version-progression').innerHTML=Object.entries(versionNotes).map(([key,n])=>`<button data-version="${key}" aria-pressed="${key===selected}"><span>${key.toUpperCase()}${key===latestVersion?' · Latest':''}</span><strong>${n.title}</strong><small>${n.view}</small></button>`).join('');
  $('version-progression').querySelectorAll('[data-version]').forEach(button=>button.addEventListener('click',()=>{$('version-select').value=button.dataset.version;render();}));
  const misses=data.class_names.filter(name=>summary.stealth.per_class[name].support>0&&summary.stealth.per_class[name].recall<1);
  $('version-detail').innerHTML=`<p class="eyebrow">${selected.toUpperCase()} · ${selected===latestVersion?'LATEST VERSION':'EARLIER VERSION'}</p><h3>${notes.title}</h3><p class="version-description">${notes.text}</p><dl class="version-facts"><div><dt>Looks at</dt><dd>${notes.view}</dd></div><div><dt>Trained on</dt><dd>${notes.training}</dd></div><div><dt>Correct guesses<br>Slowed-down test</dt><dd><strong>${summary.correct} / ${summary.stealth.n}</strong> sessions</dd></div></dl><p class="version-help">${misses.length?`Missed family guesses include ${misses.map(human).map(esc).join(', ')}.`:'All family guesses were correct in this saved test.'}</p><a class="version-learn-link" href="model.html">How the model learns →</a>`;
  $('version-comparison').innerHTML=`<table class="version-table"><caption class="sr-only">Balanced class scores for each version on the same test sessions</caption><thead><tr><th>Version</th><th>Standard<br><span>Balanced score</span></th><th>Slowed-down<br><span>Balanced score</span></th><th>Correct<br><span>Slowed-down test</span></th></tr></thead><tbody>${versionKeys.map(key=>{const s=selectedSummary(data,key);return `<tr class="${key===selected?'selected-version':''}"><th scope="row">${key.toUpperCase()}${key===selected?' <span class="sr-only">· Selected</span>':''}</th><td>${percent(s.normal.macro_f1)}</td><td><div class="version-score"><span class="version-score-track"><i style="width:${s.stealth.macro_f1*100}%"></i></span><strong>${percent(s.stealth.macro_f1)}</strong></div></td><td>${s.correct} / ${s.stealth.n}</td></tr>`;}).join('')}</tbody></table>`;
  // Computed from whichever versions are actually in the data, not
  // hardcoded to specific keys -- a fixed "v2 wins" sentence would have
  // gone stale the moment a new version matched or beat it.
  const baseline=data.versions.v1.stealth_session_metrics.macro_f1;
  const latestScore=data.versions[latestVersion].stealth_session_metrics.macro_f1;
  const bestScore=Math.max(...versionKeys.map(key=>data.versions[key].stealth_session_metrics.macro_f1));
  const bestKeys=versionKeys.filter(key=>data.versions[key].stealth_session_metrics.macro_f1===bestScore);
  const delta=(latestScore-baseline)*100;
  const leaders=bestKeys.map(k=>k.toUpperCase()).join(' and ');
  const latestIsBest=bestKeys.includes(latestVersion);
  $('version-findings').textContent=`In this run, ${leaders} ${bestKeys.length>1?'tie for':'has'} the strongest result on slowed-down traffic. ${latestVersion.toUpperCase()}’s balanced class score is ${Math.abs(delta).toFixed(1)} percentage points ${delta>=0?'above':'below'} v1’s. ${latestIsBest?`The latest version (${latestVersion.toUpperCase()}) matches the best result seen so far.`:'The latest version is not the best on every test.'}`;
  $('version-provenance').textContent=`Saved ${data.generated_at.slice(0,10)} · ${count(data.dataset.session_counts.normal_test)} standard test sessions and ${count(data.dataset.session_counts.stealth_test)} slowed-down test sessions. Standard traffic includes benign activity and standard attacks. For the per-flow versions (v1/v2), the most common flow prediction supplies the session guess; the session-memory versions (v3/v4) predict the session directly. These are small synthetic test sets: scores can change with different examples or training runs. Correct family guesses are not binary alerts. Source: bundled versions.json.`;
}
$('version-select').addEventListener('change',()=>{if(data&&versionNotes[$('version-select').value])render();});
try {const response=await fetch('./versions.json', {cache:'no-store'});if(!response.ok)throw new Error(`HTTP ${response.status}`);data=validateVersions(await response.json());render();$('versions-content').hidden=false;$('versions-status').hidden=true;}
catch(error){$('versions-status').textContent=`Version comparison unavailable: ${error.message} Detection records remain available on Overview.`;$('versions-status').className='status error';}
