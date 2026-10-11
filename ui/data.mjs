export const colors = ['#718096', '#7c65a3', '#3471ac', '#448778', '#a77b35', '#a74b58'];
export const human = value => String(value ?? 'Unavailable').replaceAll('_', ' ').replace(/\b\w/g, c => c.toUpperCase());
export const count = value => Number.isFinite(value) ? value.toLocaleString('en-US') : '—';
export const percent = value => Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : '—';
export const escapeHTML = value => String(value ?? '—').replace(/[&<>"']/g, c => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[c]));
export function elapsed(seconds) {
  if (!Number.isFinite(seconds)) return '—';
  const hours = Math.floor(seconds / 3600), minutes = Math.floor(seconds % 3600 / 60), secs = Math.floor(seconds % 60);
  return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
}
export function bytes(value) {
  if (!Number.isFinite(value)) return '—';
  if (value < 1024) return `${count(value)} B`;
  const power = Math.min(3, Math.floor(Math.log(value) / Math.log(1024)));
  return `${(value / 1024 ** power).toFixed(1)} ${['B','KiB','MiB','GiB'][power]}`;
}
export function validateReport(report) {
  const fail = message => { throw new Error(`Invalid report: ${message}`); };
  if (!report || typeof report !== 'object' || Array.isArray(report)) fail('expected a JSON object.');
  if (report.report_version !== '1.0.0') fail(`unsupported report_version ${String(report.report_version)}; expected 1.0.0.`);
  const required = ['run_id', 'schema_version', 'generated_at', 'dataset', 'binary_model', 'multiclass_model', 'alerts', 'artifact_paths'];
  for (const key of required) if (report[key] == null) fail(`missing ${key}.`);
  const finite = (value, key, min = 0) => { if (!Number.isFinite(value) || value < min) fail(`${key} must be a finite number >= ${min}.`); };
  const rate = (value, key) => { finite(value, key); if (value > 1) fail(`${key} must be <= 1.`); };
  const list = (value, key) => { if (!Array.isArray(value)) fail(`${key} must be an array.`); };
  const manifest = report.dataset.manifest;
  if (!manifest?.config || !manifest.scenario_counts || !report.dataset.split_sizes) fail('missing dataset manifest/config/splits.');
  for (const key of ['row_count','host_count']) finite(manifest[key], `manifest.${key}`);
  for (const key of ['train','val','test']) finite(report.dataset.split_sizes[key], `split_sizes.${key}`);
  for (const [key,value] of Object.entries(manifest.scenario_counts)) finite(value, `scenario_counts.${key}`);
  list(report.dataset.sample_flows, 'dataset.sample_flows');
  list(report.alerts.sample, 'alerts.sample');
  for (const row of report.dataset.sample_flows) {
    if (!row || typeof row !== 'object') fail('sample flow must be an object.');
    for (const key of ['event_time','duration_s','packets_sent','packets_recv','bytes_sent','bytes_recv']) finite(row[key], `sample_flow.${key}`);
  }
  for (const row of report.alerts.sample) {
    if (!row || typeof row !== 'object') fail('sample alert must be an object.');
    for (const key of ['start_time','end_time','flow_count']) finite(row[key], `sample_alert.${key}`);
    for (const key of ['max_score','mean_score']) rate(row[key], `sample_alert.${key}`);
    if (row.predicted_scenario_confidence != null) rate(row.predicted_scenario_confidence, 'sample_alert.predicted_scenario_confidence');
  }
  finite(report.alerts.total, 'alerts.total');
  if (!report.alerts.by_severity) fail('missing alerts.by_severity.');
  for (const model of ['binary_model', 'multiclass_model']) {
    if (!report[model].train_config || !report[model].test_metrics) fail(`missing ${model} training config or metrics.`);
    list(report[model].history, `${model}.history`);
    for (const row of report[model].history) {
      if (row.epoch != null) for (const key of ['epoch','train_loss','val_loss']) finite(row[key], `${model}.history.${key}`);
      else { finite(row.best_epoch, `${model}.history.best_epoch`); finite(row.best_val_loss, `${model}.history.best_val_loss`); }
    }
  }
  for (const [name,metrics] of [['binary',report.binary_model.test_metrics],['rule',report.binary_model.rule_baseline_metrics]]) {
    if (!metrics) fail(`missing ${name} metrics.`);
    finite(metrics.n, `${name}.n`);
    for (const key of ['accuracy','precision','recall','f1','false_positive_rate']) rate(metrics[key], `${name}.${key}`);
    for (const key of ['tp','tn','fp','fn']) finite(metrics.confusion_matrix?.[key], `${name}.confusion_matrix.${key}`);
  }
  const mc = report.multiclass_model, metrics = mc.test_metrics;
  list(mc.class_names, 'multiclass_model.class_names');
  if (!mc.class_names.length || new Set(mc.class_names).size !== mc.class_names.length) fail('class_names must be nonempty and unique.');
  rate(metrics.macro_f1, 'multiclass.macro_f1');
  rate(metrics.accuracy, 'multiclass.accuracy');
  finite(metrics.n, 'multiclass.n');
  list(metrics.confusion_matrix, 'multiclass.confusion_matrix');
  if (metrics.confusion_matrix.length !== mc.class_names.length) fail('confusion matrix dimensions do not match classes.');
  metrics.confusion_matrix.forEach(row => {
    list(row, 'confusion matrix row');
    if (row.length !== mc.class_names.length) fail('confusion matrix dimensions do not match classes.');
    row.forEach(value => finite(value, 'confusion matrix cell'));
  });
  for (const name of mc.class_names) {
    const row = metrics.per_class?.[name];
    if (!row) fail(`missing per_class.${name}.`);
    for (const key of ['precision','recall','f1']) rate(row[key], `per_class.${name}.${key}`);
    finite(row.support, `per_class.${name}.support`);
  }
  return report;
}
export function selectEvidence(report, {tab='alerts', search='', family='', category='', severity='', sort='time'} = {}) {
  const isAlert = tab === 'alerts', query = search.toLowerCase().trim();
  return [...(isAlert ? report.alerts.sample : report.dataset.sample_flows)].filter(row => {
    const truth = isAlert ? row.scenario_ground_truth : row.scenario;
    return (!category || (category==='benign' ? truth==='benign' : truth!=='benign')) && (!family || truth === family) && (!isAlert || !severity || row.severity === severity) &&
      (!query || [row.src_ip,row.dst_ip,row.flow_id,row.alert_id,truth,row.predicted_scenario,row.incident_id].some(value => String(value ?? '').toLowerCase().includes(query)));
  }).sort((a,b) => {
    if (sort === 'score' && isAlert) return (b.max_score ?? 0) - (a.max_score ?? 0);
    if (sort === 'volume') return isAlert ? (b.flow_count ?? 0) - (a.flow_count ?? 0) : (b.bytes_sent + b.bytes_recv) - (a.bytes_sent + a.bytes_recv);
    return (isAlert ? a.start_time - b.start_time : a.event_time - b.event_time);
  });
}

export function incidentGroups(alerts) {
  const groups = new Map();
  const ranks = {critical: 0, high: 1, medium: 2, low: 3};
  for (const alert of alerts) {
    if (!alert.incident_id) continue; // no fabricated grouping
    if (!groups.has(alert.incident_id)) groups.set(alert.incident_id, {id:alert.incident_id,alerts:[],start:Infinity,end:0,flows:0,severity:'low'});
    const group = groups.get(alert.incident_id);
    group.alerts.push(alert);group.start=Math.min(group.start,alert.start_time);group.end=Math.max(group.end,alert.end_time);group.flows+=alert.flow_count;
    if ((ranks[alert.severity]??4)<(ranks[group.severity]??4)) group.severity=alert.severity;
  }
  return [...groups.values()].sort((a,b)=>a.start-b.start);
}
