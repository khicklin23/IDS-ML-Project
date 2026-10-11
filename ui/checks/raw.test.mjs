import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {unpackRows,filterRaw} from '../raw-data.mjs';
import {incidentGroups} from '../data.mjs';
const manifest=JSON.parse(await readFile(new URL('../raw-data/manifest.json',import.meta.url),'utf8'));
const report=JSON.parse(await readFile(new URL('../report.json',import.meta.url),'utf8'));
const chunks=await Promise.all(manifest.chunks.map(async name=>unpackRows(manifest.columns,JSON.parse(await readFile(new URL('../raw-data/'+name,import.meta.url),'utf8')))));
const rows=chunks.flat();
test('full static export matches the report and has unique flow IDs',()=>{
  assert.equal(rows.length,report.dataset.manifest.row_count);
  assert.equal(manifest.dataset_generated_at,report.dataset.manifest.generated_at);
  assert.equal(new Set(rows.map(row=>row.flow_id)).size,rows.length);
  assert.ok(rows.every(row=>row.run_id===report.run_id));
  assert.ok(rows.every(row=>'src_port'in row&&'dst_port'in row));
});
test('raw filters find records beyond the original hundred-row sample',()=>{
  const last=rows.at(-1);
  assert.equal(filterRaw(rows,{search:last.flow_id})[0].flow_id,last.flow_id);
  assert.ok(!report.dataset.sample_flows.some(row=>row.flow_id===last.flow_id));
  const matches=filterRaw(rows,{session:last.session_id});
  assert.ok(matches.length>0&&matches.every(row=>row.session_id===last.session_id));
  const sorted=filterRaw(rows,{sort:'bytes_sent',descending:true});
  assert.ok(sorted.every((row,i)=>!i||sorted[i-1].bytes_sent>=row.bytes_sent));
});
test('incident groups preserve backend IDs and omit unassigned alerts',()=>{
  const groups=incidentGroups(report.alerts.sample);
  assert.equal(groups.reduce((n,g)=>n+g.alerts.length,0),report.alerts.sample.filter(a=>a.incident_id).length);
  assert.ok(groups.every(g=>g.alerts.every(a=>a.incident_id===g.id)));
  assert.deepEqual(incidentGroups([{...report.alerts.sample[0],incident_id:null}]),[]);
});
test('malformed row chunks are rejected',()=>assert.throws(()=>unpackRows(['a','b'],[[1]]),/Malformed/));
