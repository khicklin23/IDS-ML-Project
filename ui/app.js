import {installMobileSupport} from './mobile.mjs?v=sona-mobile-1';
installMobileSupport();
import {colors, human, count, percent, escapeHTML as esc, elapsed, bytes, validateReport, selectEvidence, incidentGroups, modelName} from './data.mjs?v=sona-filter-labels-1';

const $ = id => document.getElementById(id);
let report, sourceName = 'Bundled showcase report', tab = 'alerts', page = 0;
const PAGE_SIZE = 20;
const isSession = () => report?.detector?.kind === "session_sequence";
const width = value => Math.max(0, Math.min(100, value * 100));
const badge = (text, type) => `<span class="badge ${type}">${text}</span>`;
const item = (label, value) => `<div class="detail-item"><span>${esc(label)}</span><strong>${esc(value)}</strong></div>`;
const bar = (value, color) => `<div class="bar-track"><div class="bar-fill" style="width:${width(value)}%;background:${color}"></div></div>`;
function status(message, error = false) {
  $('status').textContent = message;
  $('status').className = error ? 'status error' : 'status';
  $('status').hidden = !message;
}
function renderOverview() {
  const manifest = report.dataset.manifest, bin = report.binary_model.test_metrics, mc = report.multiclass_model;
  const metrics = [
    ['Synthetic flows', count(manifest.row_count), `${count(manifest.host_count)} generated hosts`, 'GROUND TRUTH','truth'],
    ['Detection F1', percent(bin.f1), `${count(bin.n)} held-out flows · ${isSession()?'session family decision':`threshold ${bin.threshold ?? 'unavailable'}`}`, 'MODEL INFERENCE','inference'],
    ['Classification macro F1', percent(mc.test_metrics.macro_f1), `${mc.class_names.filter(name=>name!=='benign').length} attack families + normal · equally weighted`, 'MODEL INFERENCE','inference'],
    ['Model-generated alerts', count(report.alerts.total), Object.entries(report.alerts.by_severity).map(([key,value])=>`${count(value)} ${key}`).join(' · ') || 'No alerts in this run', 'MODEL INFERENCE','inference'],
  ];
  const metricHelp = {
    'Synthetic flows':'Computer-generated connection records, not captured real-world traffic. This count covers the full dataset, including training and test data.',
    'Detection F1':'A score that balances how many flagged examples were truly suspicious with how many suspicious examples were caught. Higher is better. Related flows share a session decision.',
    'Classification macro F1':'A balanced score for choosing the correct traffic family. Each family, including normal traffic, gets equal weight so a common class cannot hide a weak result.',
    'Model-generated alerts':'Alert records created when the model chose an attack family for a session. An alert is not proof of an attack; it can be a false alarm. This is not the number of flagged flows.'
  };
  $('metrics').innerHTML = metrics.map(([title,value,description,label,type])=>`<article class="metric"><div class="metric-label"><span>${title}</span><button type="button" class="metric-info" aria-label="About ${title}" aria-describedby="help-${type}-${metrics.findIndex(m=>m[0]===title)}">i<span class="metric-tooltip" role="tooltip" id="help-${type}-${metrics.findIndex(m=>m[0]===title)}">${esc(metricHelp[title])}</span></button>${badge(label,type)}</div><div class="metric-value">${value}</div><div class="metric-desc">${esc(description)}</div></article>`).join('');
  const cm=bin.confusion_matrix;
  $('disposition-summary').innerHTML=`<article class="disposition benign-disposition"><h3>Normal traffic</h3><div class="disposition-count">${count(cm.tn)} <span>benign flows correctly left unflagged</span></div><p>The detector classified these as normal. No alert was triggered for these flows.</p><div class="disposition-exception">${count(cm.fp)} benign flows were flagged instead · false positives</div></article><article class="disposition attack-disposition"><h3>Attack traffic</h3><div class="disposition-count">${count(cm.tp)} <span>suspicious flows correctly flagged</span></div><p>The detector classified these generated attack flows as suspicious.</p><div class="disposition-exception">${count(cm.fn)} suspicious flows were left unflagged · missed detections</div></article>`;
  $('disposition-note').textContent=`${isSession()?'Session decisions counted across flows':'Binary model outcomes'} across ${count(bin.n)} held-out flows, checked against generated ground truth. ${isSession()?'Related flows share one session prediction. These counts are not separate model decisions.':'These are flow counts, not session-level alert counts.'} Individual unflagged predictions are not included in the report sample.`;
  $('test-count').textContent = `${count(bin.n)} held-out flows · offline evaluation`;
  $('footer-version').textContent = `REPORT ${report.report_version} / SCHEMA ${report.schema_version}`;
}
function renderPerformance() {
  const bin = report.binary_model.test_metrics;
  const help={
    f1:'One score that balances catching attacks with avoiding false alarms. It combines precision and recall. Higher is better; it is not the percentage of all guesses that were correct.',
    recall:'Of the attack flows in the test, how many did Sona catch? A higher percentage means fewer attacks were missed.',
    precision:'Of the flows Sona flagged, how many really belonged to a generated attack? A higher percentage means fewer flagged flows were false alarms.',
    false_positive_rate:'Of the normal flows in the test, how many did Sona flag by mistake? These are false positives, or false alarms. Lower is better. This uses normal flows as the denominator, unlike precision.'
  };
  $('comparison').innerHTML = [['f1','F1','Higher is better'],['recall','Recall','Higher is better'],['precision','Precision','Higher is better'],['false_positive_rate','False positives','Lower is better']].map(([key,label,direction])=>`<article class="detection-metric"><div class="metric-label"><span>${label}</span><button type="button" class="metric-info detection-info" aria-label="About ${label}" aria-describedby="detection-help-${key}">i<span class="metric-tooltip" role="tooltip" id="detection-help-${key}">${esc(help[key])}</span></button></div><strong class="detection-value">${percent(bin[key])}</strong>${bar(bin[key],colors[2])}<span class="metric-desc">${direction}</span></article>`).join('');
  $('comparison-note').textContent = `${count(bin.n)} held-out flows, checked against generated ground truth. Related flows share a session prediction. Hover or focus an info icon for a plain-language explanation.`;
  const mc = report.multiclass_model, perClass = mc.test_metrics.per_class;
  const classRow=(name)=>{
    const i=mc.class_names.indexOf(name);
    const row=perClass[name];
    return `<div class="class-row"><div class="class-name"><i class="class-dot" style="background:${colors[i%colors.length]}"></i>${esc(human(name))}</div><div>${bar(row.f1,colors[i%colors.length])}<div class="class-scores"><span>P ${percent(row.precision)}</span><span>R ${percent(row.recall)}</span></div></div><div class="class-value">${row.support ? percent(row.f1) : 'N/A'}</div></div>`;
  };
  $('class-performance').innerHTML='<div class="class-labels"><span>Attack family</span><span>F1 score</span><span>F1</span></div>'+mc.class_names.filter(name=>name!=='benign').map(classRow).join('')+(mc.class_names.includes('benign')?`<div class="normal-class-section"><h4>Normal traffic recognition</h4>${classRow('benign')}<p class="footnote">Benign is the normal class, not an attack family. ${isSession()?'This same family output decides whether a session is flagged; a non-benign choice creates an alert.':'This separate multiclass output does not decide whether the binary detector raises an alert.'}</p></div>`:'');
  const supported = mc.class_names.filter(name=>perClass[name].support>0), missing = mc.class_names.filter(name=>!perClass[name].support);
  const weakest = supported.reduce((a,b)=>perClass[a].f1 <= perClass[b].f1 ? a : b, supported[0]);
  $('class-note').textContent = missing.length ? `No held-out examples for ${missing.map(human).join(', ')}. Metrics for those classes do not establish coverage.` : weakest ? `Lowest F1: ${human(weakest)} (${percent(perClass[weakest].f1)}). Precision ${percent(perClass[weakest].precision)} · recall ${percent(perClass[weakest].recall)} · ${count(perClass[weakest].support)} examples.` : 'No held-out class support available.';
  const names = mc.class_names, matrix = mc.test_metrics.confusion_matrix;
  $('confusion').innerHTML = `<table class="matrix"><caption class="sr-only">Ground-truth rows, predicted columns; held-out classification counts</caption><thead><tr><th scope="col">Actual ↓<br>Predicted →</th>${names.map(name=>`<th scope="col">${esc(human(name))}</th>`).join('')}</tr></thead><tbody>${matrix.map((row,i)=>{
    const total = row.reduce((sum,n)=>sum+n,0);
    return `<tr><th scope="row">${esc(human(names[i]))}</th>${row.map((value,j)=>`<td tabindex="0" class="${i===j?'diagonal':''}" style="background:rgba(${i===j?'46,113,185':'184,120,76'},${.08 + .65*(total?value/total:0)})" title="${esc(`Actual ${human(names[i])} → predicted ${human(names[j])}: ${count(value)} (${percent(total?value/total:0)} of row)`)}">${count(value)}</td>`).join('')}</tr>`;
  }).join('')}</tbody></table>`;
  const manifest=report.dataset.manifest, splits=report.dataset.split_sizes;
  const compositionRow=([name,n])=>`<div class="composition-row"><span>${esc(human(name))}</span>${bar(manifest.row_count ? n/manifest.row_count : 0,colors[Math.max(0,names.indexOf(name))%colors.length])}<span>${count(n)}</span></div>`;
  $('composition').innerHTML=(manifest.scenario_counts.benign!=null?`<div class="normal-composition"><h4>Normal background</h4>${compositionRow(['benign',manifest.scenario_counts.benign])}</div>`:'')+'<h4>Generated attack scenarios</h4>'+Object.entries(manifest.scenario_counts).filter(([name])=>name!=='benign').map(compositionRow).join('');
  $('dataset-details').innerHTML = item('Total flows',count(manifest.row_count))+item('Training flows',count(splits.train))+item('Validation flows',count(splits.val))+item('Test flows',count(splits.test))+item('Background realism opt-in',manifest.config.realistic_background?'Enabled':'Disabled');
}
function renderEvidence() {
  const isFlow=tab==='flows',mode=isFlow?'flows':'alerts';
  $('category-label').textContent=isFlow?'Ground-truth category':'Alert category';
  if($('category-filter').dataset.mode!==mode){
    $('category-filter').innerHTML=isFlow?'<option value="">All traffic</option><option value="attack">Attack scenarios</option><option value="benign">Benign background</option>':'<option value="">All alerts</option><option value="attack">Attack-related alerts</option><option value="benign">Benign false alarms</option>';
    $('category-filter').dataset.mode=mode;
  }
  if(tab==='incidents'){renderIncidents();return;}
  const isAlert = tab==='alerts', options={tab,search:$('search').value,family:$('family-filter').value,category:$('category-filter').value,severity:$('severity-filter').value,sort:$('sort').value};
  const rows=selectEvidence(report,options), pages=Math.max(1,Math.ceil(rows.length/PAGE_SIZE));
  page=Math.min(page,pages-1);
  const sample = isAlert ? report.alerts.sample : report.dataset.sample_flows, total=isAlert?report.alerts.total:report.dataset.manifest.row_count;
  $('sample-count').textContent=`${count(sample.length)} embedded ${isAlert?'alerts':'flows'} / ${count(total)} total`;
  $('severity-label').hidden=!isAlert;
  $('sort').querySelector('[value="score"]').disabled=!isAlert;
  $('evidence-content').setAttribute('aria-labelledby',`tab-${tab}`);
  const headers=isAlert?['Severity','Source → destination','Predicted type','Actual type','Suspiciousness score','Flows','Review']:['Sim. time','Source → destination','Service / protocol','Bytes sent / received','Actual type','Outcome','Review'];
  $('table-head').innerHTML=`<tr>${headers.map(h=>`<th scope="col">${h}</th>`).join('')}</tr>`;
  $('table-body').innerHTML=rows.slice(page*PAGE_SIZE,(page+1)*PAGE_SIZE).map((row,index)=>{
    const pair=`<span class="mono">${esc(row.src_ip)} <span class="muted">→</span> ${esc(row.dst_ip)}</span>`;
    const cells=isAlert ? [
      `<span class="severity ${['critical','high','medium','low'].includes(row.severity)?row.severity:''}">${esc(human(row.severity))}</span>`,pair,
      `${esc(human(row.predicted_scenario))}<br><span class="muted mono">${percent(row.predicted_scenario_confidence)} confidence</span>`,
      `<span class="${row.scenario_ground_truth==='benign'?'normal-tag':'muted'}">${row.scenario_ground_truth==='benign'?'Benign · false alarm':esc(human(row.scenario_ground_truth))}</span>`,percent(row.max_score),count(row.flow_count),
    ] : [elapsed(row.event_time),pair,`${esc(String(row.service??'').toUpperCase())} <span class="muted">/ ${esc(String(row.protocol??'').toUpperCase())}</span>`,`${bytes(row.bytes_sent)} <span class="muted">/ ${bytes(row.bytes_recv)}</span>`,`<span class="muted">${esc(human(row.scenario))}</span>`,esc(human(row.outcome))];
    return `<tr>${cells.map(cell=>`<td>${cell}</td>`).join('')}<td><button class="record-button" data-row="${index+page*PAGE_SIZE}" aria-label="Open ${esc(row.alert_id??row.flow_id)}">Open</button></td></tr>`;
  }).join('') || '<tr><td colspan="7" class="empty">No embedded records match these filters. Try another family or reset the search.</td></tr>';
  $('table-summary').textContent=`${rows.length?`${page*PAGE_SIZE+1}–${Math.min((page+1)*PAGE_SIZE,rows.length)}`:'0'} of ${count(rows.length)} matching sample records`;
  $('page-number').textContent=`${page+1} / ${pages}`;
  $('previous').disabled=page===0;
  $('next').disabled=page>=pages-1;
  $('evidence-note').textContent=isAlert?(isSession()?'One Sona session prediction supplies both the family and alert decision. Suspiciousness score is 1 minus the model’s normal-traffic probability. All related flows inherit that decision. Severity reflects score, not verified impact. Filters affect only the embedded alert sample.':'Alerts come from the binary model; predicted family comes from the separate multiclass model. These can disagree. Severity reflects binary confidence, not verified impact. Ground truth is generator provenance. Filters affect only the embedded alert sample.'):'These are the earliest flows embedded in the report, not a representative sample or the full dataset. Family and normal/suspicious labels are ground truth; this sample has no per-flow model predictions or ports. Filters affect only these records.';
  $('table-body').querySelectorAll('[data-row]').forEach(button=>button.addEventListener('click',()=>openDetail(rows[Number(button.dataset.row)],isAlert)));
}
function renderIncidents() {
  const query=$('search').value.toLowerCase().trim(),family=$('family-filter').value,severity=$('severity-filter').value,category=$('category-filter').value;
  let groups=incidentGroups(report.alerts.sample).filter(group=>(!category||(category==='benign'?group.alerts.every(a=>a.scenario_ground_truth==='benign'):group.alerts.some(a=>a.scenario_ground_truth!=='benign')))&&(!severity||group.severity===severity)&&(!family||group.alerts.some(a=>a.scenario_ground_truth===family))&&(!query||[group.id,...group.alerts.flatMap(a=>[a.src_ip,a.dst_ip,a.predicted_scenario,a.scenario_ground_truth])].some(value=>String(value??'').toLowerCase().includes(query))));
  if($('sort').value==='volume')groups.sort((a,b)=>b.flows-a.flows);
  const pages=Math.max(1,Math.ceil(groups.length/PAGE_SIZE));page=Math.min(page,pages-1);
  $('severity-label').hidden=false;$('sort').querySelector('[value="score"]').disabled=true;
  $('evidence-content').setAttribute('aria-labelledby','tab-incidents');
  $('sample-count').textContent=`${count(incidentGroups(report.alerts.sample).length)} recorded incidents in embedded alerts`;
  $('table-head').innerHTML='<tr>'+['Severity','Incident','Source addresses','Alert count','Flow count','Simulation start','Review'].map(label=>`<th scope="col">${label}</th>`).join('')+'</tr>';
  $('table-body').innerHTML=groups.slice(page*PAGE_SIZE,(page+1)*PAGE_SIZE).map((group,i)=>`<tr><td><span class="severity ${group.severity}">${esc(human(group.severity))}</span></td><td class="mono">${esc(group.id)}</td><td class="mono">${esc([...new Set(group.alerts.map(a=>a.src_ip))].join(', '))}</td><td>${group.alerts.length}</td><td>${count(group.flows)}</td><td class="mono">${elapsed(group.start)}</td><td><button class="record-button" data-row="${page*PAGE_SIZE+i}">Open</button></td></tr>`).join('')||'<tr><td colspan="7" class="empty">No recorded incidents match these filters.</td></tr>';
  $('table-summary').textContent=`${count(groups.length)} matching incidents from embedded alerts`;
  $('page-number').textContent=`${page+1} / ${pages}`;$('previous').disabled=page===0;$('next').disabled=page===pages-1;
  $('evidence-note').textContent='These are the backend’s recorded host/time-window incident groups, not learned attack-chain attribution. Counts cover embedded alerts only. Open an incident to review its alerts and their raw sessions.';
  $('table-body').querySelectorAll('[data-row]').forEach(button=>button.addEventListener('click',()=>openIncident(groups[Number(button.dataset.row)])));
}
function openIncident(group) {
  $('detail-title').textContent='Incident investigation';
  $('detail-body').innerHTML=`<p class="mono muted">${esc(group.id)}</p><div class="detail-grid">${item('Recorded alerts',group.alerts.length)+item('Associated flows',count(group.flows))+item('Simulation start',elapsed(group.start))+item('Simulation end',elapsed(group.end))}</div><h3>Related alerts</h3>${group.alerts.map((alert,index)=>`<div class="incident-alert"><strong>${esc(human(alert.predicted_scenario))}</strong><p class="mono">${esc(alert.src_ip)} → ${esc(alert.dst_ip)}</p><p>Binary score ${percent(alert.max_score)} · ground truth ${esc(human(alert.scenario_ground_truth))}</p><button class="record-button" data-row="${index}">Open alert</button></div>`).join('')}<p class="footnote">Grouping is a backend heuristic. Predicted families are model outputs; ground truth is generator provenance.</p>`;
  $('detail-body').querySelectorAll('[data-row]').forEach(button=>button.addEventListener('click',()=>openDetail(group.alerts[Number(button.dataset.row)],true)));
  $('detail-panel').hidden=false;$('detail-panel').scrollIntoView({block:'nearest',behavior:'smooth'});$('detail-title').focus();
}
function closeDetail() { $('detail-panel').hidden=true; }
function openDetail(row,isAlert) {
  $('detail-title').textContent=isAlert ? 'Alert investigation' : 'Flow record';
  const base = item('Source IP',row.src_ip)+item('Destination IP',row.dst_ip)+item('Simulation start',elapsed(row.start_time??row.event_time));
  let html = `<p class="mono muted">${esc(row.alert_id??row.flow_id)}</p><div class="detail-grid">${base}${isAlert?item('Simulation end',elapsed(row.end_time))+item('Maximum suspiciousness score',percent(row.max_score))+item('Mean suspiciousness score',percent(row.mean_score))+item('Flow count',count(row.flow_count))+item('Severity',human(row.severity))+item('Incident',row.incident_id??'Not grouped')+item('Recorded status',human(row.status)):item('Duration',`${Number(row.duration_s).toFixed(3)} seconds`)+item('Service / protocol',`${row.service} / ${row.protocol}`)+item('Outcome',human(row.outcome))+item('Packets sent / received',`${count(row.packets_sent)} / ${count(row.packets_recv)}`)+item('Bytes sent / received',`${bytes(row.bytes_sent)} / ${bytes(row.bytes_recv)}`)}</div>`;
  if(isAlert) html+=`<div class="detail-box">${badge('MODEL INFERENCE','inference')}<p>Predicted family: <strong>${esc(human(row.predicted_scenario))}</strong> · ${percent(row.predicted_scenario_confidence)} confidence</p><p class="muted">${isSession()?'The session model selected a non-benign family, which triggered this alert. The suspiciousness score is 1 minus the model’s normal-traffic probability; it is not a separate binary model.':'The binary detector triggered this alert. Its maximum score and the separate family prediction summarize different model outputs.'}</p></div>`;
  if(isAlert && row.scenario_ground_truth==='benign')html+='<p class="callout">The detector flagged a generated benign session. This is a false-positive example, even if its binary confidence is high.</p>';
  html+=`<div class="detail-box">${badge('GROUND TRUTH','truth')}<p>Generated scenario: <strong>${esc(human(isAlert?row.scenario_ground_truth:row.scenario))}</strong>${!isAlert?` · label: ${esc(row.label)}`:''}</p></div>`;
  if(isAlert) html+=`<p class="footnote">Recorded notes: ${esc(row.notes??'None')}. This static snapshot is read-only; it does not acknowledge or modify alerts.</p>`;
  else html+='<p class="footnote">Model predictions and port fields are not included in this flow sample. No inference is fabricated from the generated label.</p>';
  if(isAlert && sourceName==='Bundled showcase report') html+=`<a class="button primary" href="raw.html?session=${encodeURIComponent(row.session_id)}">View this session’s raw flows</a>`;
  $('detail-body').innerHTML=html;
  $('detail-panel').hidden=false;
  $('detail-panel').scrollIntoView({block:'nearest',behavior:'smooth'});
  $('detail-title').focus();
}
function resetFilters() {
  $('search').value='';$('family-filter').value='';$('category-filter').value='';$('severity-filter').value='';$('sort').value='time';page=0;
}
function renderAll() {
  resetFilters();
  $('family-filter').innerHTML='<option value="">All types</option>'+(report.multiclass_model.class_names.includes('benign')?'<optgroup label="Normal traffic"><option value="benign">Normal traffic</option></optgroup>':'')+'<optgroup label="Attack types">'+report.multiclass_model.class_names.filter(name=>name!=='benign').map(name=>`<option value="${esc(name)}">${esc(human(name))}</option>`).join('')+'</optgroup>';
  renderOverview();renderPerformance();renderEvidence();
  $('dataset-link').textContent=sourceName==='Bundled showcase report'?'Browse full dataset':'Browse bundled dataset';
  $('raw-link').textContent=sourceName==='Bundled showcase report'?'Open raw data':'Bundled raw data';
  $('dashboard').hidden=false;status('');
}
function loadReport(value,name) {
  const checked=validateReport(value);
  report=checked;sourceName=name;page=0;
  closeDetail();
  renderAll();
}
for(const name of ['alerts','incidents','flows']) $('tab-'+name).addEventListener('click',()=>{
  tab=name;resetFilters();closeDetail();
  for(const other of ['alerts','incidents','flows']) { $('tab-'+other).setAttribute('aria-selected',String(other===name));$('tab-'+other).tabIndex=other===name?0:-1; }
  renderEvidence();
});
$('tab-alerts').parentElement.addEventListener('keydown',event=>{
  if(['ArrowLeft','ArrowRight','Home','End'].includes(event.key)) {event.preventDefault();const names=['alerts','incidents','flows'];const target=event.key==='Home'?'alerts':event.key==='End'?'flows':names[(names.indexOf(tab)+(event.key==='ArrowRight'?1:2))%3];$('tab-'+target).click();$('tab-'+target).focus();}
});
for(const id of ['search','family-filter','category-filter','severity-filter','sort']) $(id).addEventListener(id==='search'?'input':'change',()=>{page=0;if(report)renderEvidence();});
$('reset-filters').addEventListener('click',()=>{resetFilters();renderEvidence();});
$('previous').addEventListener('click',()=>{page--;renderEvidence();});
$('next').addEventListener('click',()=>{page++;renderEvidence();});
$('close-detail').addEventListener('click',closeDetail);
try {
  const response=await fetch('./report.json',{cache:'no-cache'});
  if(!response.ok)throw new Error(`Results could not be loaded (HTTP ${response.status}).`);
  loadReport(await response.json(),'Bundled showcase report');
}catch(error){status(`${error.message} Please refresh the page or check that the release contains its data.`,true);}
