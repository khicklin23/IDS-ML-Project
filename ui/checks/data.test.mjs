import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {validateReport, selectEvidence, escapeHTML, percent, elapsed, bytes} from '../data.mjs';

const report = JSON.parse(await readFile(new URL('../report.json',import.meta.url),'utf8'));

test('actual showcase contract, including final best-checkpoint summary, is accepted',()=>{
  assert.equal(validateReport(report),report);
  assert.ok(report.binary_model.history.at(-1).best_epoch>=0);
});
test('unsupported versions and malformed metrics are rejected rather than fabricated',()=>{
  assert.throws(()=>validateReport({...report,report_version:'2.0.0'}),/unsupported/);
  const broken=structuredClone(report);broken.binary_model.test_metrics.recall=2;
  assert.throws(()=>validateReport(broken),/recall/);
  broken.binary_model.test_metrics.recall=NaN;
  assert.throws(()=>validateReport(broken),/recall/);
});
test('confusion matrix dimensions are checked',()=>{
  const broken=structuredClone(report);broken.multiclass_model.test_metrics.confusion_matrix[0].pop();
  assert.throws(()=>validateReport(broken),/dimensions/);
});
test('malformed sample rows fail before application state is replaced',()=>{
  const broken=structuredClone(report);broken.alerts.sample=[null];
  assert.throws(()=>validateReport(broken),/sample alert/);
  broken.alerts.sample=[];broken.dataset.sample_flows[0].bytes_sent=Infinity;
  assert.throws(()=>validateReport(broken),/bytes_sent/);
});
test('empty samples and zero-support classes are valid report states',()=>{
  const empty=structuredClone(report);empty.alerts.sample=[];empty.dataset.sample_flows=[];
  empty.multiclass_model.test_metrics.per_class.beaconing={precision:0,recall:0,f1:0,support:0};
  validateReport(empty);
  assert.deepEqual(selectEvidence(empty),[]);
});
test('ground-truth filtering does not silently filter model predictions',()=>{
  const sample={...report,alerts:{...report.alerts,sample:[{...report.alerts.sample[0],scenario_ground_truth:'benign',predicted_scenario:'beaconing'}]}};
  assert.equal(selectEvidence(sample,{family:'benign'}).length,1);
  assert.equal(selectEvidence(sample,{family:'beaconing'}).length,0);
  assert.equal(selectEvidence(sample,{search:'BEACONING'}).length,1);
  assert.equal(selectEvidence(sample,{category:'benign'}).length,1);
  assert.equal(selectEvidence(sample,{category:'attack'}).length,0);
});
test('IP search, severity, score sorting and flow volume work on actual samples',()=>{
  const source=report.alerts.sample[0].src_ip;
  assert.ok(selectEvidence(report,{search:source}).every(row=>row.src_ip===source||row.dst_ip===source));
  assert.ok(selectEvidence(report,{severity:'critical'}).every(row=>row.severity==='critical'));
  const byScore=selectEvidence(report,{sort:'score'});
  assert.ok(byScore.every((row,i)=>!i||byScore[i-1].max_score>=row.max_score));
  const flows=selectEvidence(report,{tab:'flows',sort:'volume'});
  assert.ok(flows.every((row,i)=>!i||flows[i-1].bytes_sent+flows[i-1].bytes_recv>=row.bytes_sent+row.bytes_recv));
  assert.equal(report.alerts.sample[0].max_score,byScore.find(row=>row.alert_id===report.alerts.sample[0].alert_id).max_score);
});
test('untrusted report text is escaped and number formatting is explicit',()=>{
  assert.equal(escapeHTML('<script>"&\''),'&lt;script&gt;&quot;&amp;&#39;');
  assert.equal(percent(undefined),'—');assert.equal(elapsed(3661),'01:01:01');assert.equal(bytes(2048),'2.0 KiB');
});
