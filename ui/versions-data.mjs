export const versionNotes = {
  v1: {title:'Starting point', view:'One connection at a time', training:'Normal traffic + standard attacks', text:'The original network learns from individual flows. It handles the standard examples well, but slower attacks can resemble ordinary traffic.', diagram:'Single flow → learned combinations → family guess'},
  v2: {title:'More varied training', view:'One connection at a time', training:'Normal traffic + standard and slowed-down attacks', text:'Beta 2 keeps the original network but adds slowed-down attacks to training. That practice explains its strong result: it has seen this kind of variation before. It still judges one connection at a time.', diagram:'Single flow → same network, broader training → family guess'},
  v3: {title:'Connections over time', view:'Related connections in order', training:'Normal traffic + standard attacks; no slowed-down attacks', text:'Beta 3 was a separate experiment, not a replacement that had already beaten Beta 2. It tested whether reading related connections in order could help without training on slowed-down attacks. A GRU is a network that keeps a small memory as it reads. Beta 3 got that memory, but deliberately did not get Beta 2’s broader examples. Its lower slowed-down score shows that memory alone was not enough. Sona combines both ideas.', diagram:'Flow 1 → memory → Flow 2 → updated memory → session guess'},
  v4: {title:'Memory + varied training', view:'Related connections in order', training:'Normal traffic + standard and slowed-down attacks', text:'Sona combines the session-memory design explored in Beta 3 with the broader training examples that helped Beta 2. It is the final model used for the detection results on Overview. Its final-model role reflects that combined design; the test results below show how it compares with Beta 2.', diagram:'Flow 1 → memory → Flow 2 → updated memory → session guess (broader training)'}
};
export function validateVersions(data) {
  const fail=()=>{throw new Error('The saved version comparison has an unsupported or incomplete format.');};
  if(typeof data?.versions_schema!=='string'||!Array.isArray(data.class_names)||!data.class_names.length||!data.dataset?.session_counts||!data.generated_at)fail();
  for(const version of Object.keys(versionNotes))for(const kind of ['normal','stealth']) {
    const metrics=data.versions?.[version]?.[`${kind}_session_metrics`];
    if(!metrics||!Number.isInteger(metrics.n)||metrics.n<=0||metrics.n!==data.dataset.session_counts[`${kind}_test`])fail();
    for(const key of ['accuracy','macro_f1'])if(!Number.isFinite(metrics[key])||metrics[key]<0||metrics[key]>1)fail();
    if(metrics.confusion_matrix?.length!==data.class_names.length)fail();
    let sum=0;
    for(const row of metrics.confusion_matrix){if(!Array.isArray(row)||row.length!==data.class_names.length)fail();for(const n of row){if(!Number.isInteger(n)||n<0)fail();sum+=n;}}
    if(sum!==metrics.n)fail();
    for(const name of data.class_names){const row=metrics.per_class?.[name];if(!row||!Number.isInteger(row.support)||row.support<0||!Number.isFinite(row.recall)||row.recall<0||row.recall>1)fail();}
  }
  return data;
}
export function selectedSummary(data,version) {
  const record=data.versions[version];
  return {normal:record.normal_session_metrics,stealth:record.stealth_session_metrics,correct:record.stealth_session_metrics.confusion_matrix.reduce((n,row,i)=>n+row[i],0)};
}
