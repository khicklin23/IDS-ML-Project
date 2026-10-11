// Experimental summaries are independent of the public release report.
export function validateResearch(data) {
  const integer=value=>Number.isSafeInteger(value)&&value>=0;
  const fraction=value=>Number.isFinite(value)&&value>=0&&value<=1;
  if(data?.research_version!==1||data.public_model_replaced!==false||!Array.isArray(data.studies)||!data.studies.length)throw new Error('Unsupported research summary.');
  if(data.four_model_comparison!==undefined){
    const four=data.four_model_comparison;
    if(four.schema!==1||four.source!=='four_model_timing_v1'||typeof four.scope!=='string'||!Array.isArray(four.seeds)||four.seeds.length!==3||!four.seeds.every(integer))throw new Error('Invalid four-model provenance.');
    const tests=Object.keys(four.results?.v1??{});
    if(!['normal','stealth','timing'].every(test=>tests.includes(test)))throw new Error('Missing four-model test categories.');
    if(!['v1','v2','v3','v4'].every(version=>Object.keys(four.results?.[version]??{}).sort().join(',')===tests.slice().sort().join(',')))throw new Error('Models must share every test category.');
    for(const test of tests){
      let sessions;
      for(const version of ['v1','v2','v3','v4']){
        const score=four.results?.[version]?.[test];
        if(!score||!fraction(score.mean)||!fraction(score.min)||!fraction(score.max)||score.min>score.mean+1e-12||score.mean>score.max+1e-12||!integer(score.sessions)||score.sessions===0)throw new Error('Invalid four-model result.');
        if(sessions!==undefined&&sessions!==score.sessions)throw new Error('Four models must use identical test counts.');
        sessions=score.sessions;
      }
    }
    const study=data.studies.find(study=>study.comparison);
    if(!study||four.seeds.join(',')!==study.seeds.join(','))throw new Error('Four-model studies do not match.');
    for(const [version,name] of [['v2','beta2'],['v4','sona']]){
      if(four.results[version].normal.mean!==study.comparison.models[name].ordinary||four.results[version].timing.mean!==study.comparison.models[name].timing||four.results[version].normal.sessions!==study.comparison.ordinary_sessions||four.results[version].timing.sessions!==study.comparison.timing_sessions)throw new Error('Four-model results disagree with saved candidate evidence.');
    }
  }
  for(const study of data.studies){
    if(typeof study.title!=='string'||typeof study.explanation!=='string'||typeof study.limit!=='string'||typeof study.provenance!=='string'||!integer(study.generated_flows)||!Array.isArray(study.seeds)||!study.seeds.length||!study.seeds.every(integer))throw new Error('Invalid study provenance.');
    if(!Array.isArray(study.rows)||!study.rows.length)throw new Error('Missing experiment results.');
    for(const row of study.rows){
      if(typeof row.label!=='string'||!fraction(row.beta2)||!fraction(row.sona)||!fraction(row.sona_min)||!fraction(row.sona_max)||row.sona_min>row.sona+1e-12||row.sona>row.sona_max+1e-12||!integer(row.sessions)||row.sessions===0)throw new Error('Invalid experiment score.');
    }
    if(study.comparison!==undefined){
      const comparison=study.comparison;
      if(comparison.mode!=='relative'||!integer(comparison.ordinary_sessions)||comparison.ordinary_sessions===0||!integer(comparison.timing_sessions)||comparison.timing_sessions===0)throw new Error('Invalid two-test provenance.');
      const timing=study.rows.find(row=>row.label==='Same flows, different order');
      for(const [name,id] of [['beta2','beta2'],['sona','relative']]){
        const score=comparison.models?.[name];
        if(!score||!fraction(score.ordinary)||!fraction(score.timing)||score.timing!==timing?.scores?.[id]?.mean||comparison.timing_sessions!==timing.sessions)throw new Error('Invalid two-test score.');
      }
    }
    if(study.models!==undefined){
      if(!Array.isArray(study.models)||!study.models.length||new Set(study.models.map(m=>m.id)).size!==study.models.length||!study.models.every(m=>typeof m.id==='string'&&typeof m.label==='string'))throw new Error('Invalid comparison models.');
      for(const row of study.rows)for(const model of study.models){
        const score=row.scores?.[model.id];
        if(score===null)continue;
        if(!score||!fraction(score.mean)||!fraction(score.min)||!fraction(score.max)||score.min>score.mean+1e-12||score.mean>score.max+1e-12)throw new Error('Invalid comparison range.');
      }
    }
    for(const control of study.benign_controls??[]){
      if(typeof control.label!=='string'||!integer(control.sessions)||!control.sessions||!integer(control.false_flags)||control.false_flags>control.sessions)throw new Error('Invalid harmless-traffic result.');
    }
  }
  return data;
}
