export const primaryColumns=['event_time','src_ip','src_port','dst_ip','dst_port','protocol','service','duration_s','bytes_sent','bytes_recv','outcome','scenario','label'];
export const packetColumns=['packets_sent','packets_recv'];
export function unpackRows(columns,rows) {
  if(!Array.isArray(rows)||rows.some(row=>!Array.isArray(row)||row.length!==columns.length))throw new Error('Malformed raw dataset chunk.');
  return rows.map(row=>Object.fromEntries(columns.map((column,index)=>[column,row[index]])));
}
export function filterRaw(rows,{search='',family='',outcome='',session='',sort='event_time',descending=false}={}) {
  const query=search.toLowerCase().trim();
  return rows.filter(row=>(!session||row.session_id===session)&&(!family||row.scenario===family)&&(!outcome||row.outcome===outcome)&&(!query||Object.values(row).some(value=>String(value??'').toLowerCase().includes(query)))).sort((a,b)=>{
    const first=a[sort],second=b[sort];const difference=typeof first==='number'&&typeof second==='number'?first-second:String(first??'').localeCompare(String(second??''),undefined,{numeric:true});
    return descending?-difference:difference;
  });
}
