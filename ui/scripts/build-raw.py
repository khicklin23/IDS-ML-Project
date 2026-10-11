"""Export existing backend data for static browsing; never modify backend files."""
import json
import sys
from pathlib import Path

import pandas as pd

ui = Path(__file__).resolve().parents[1]
report = json.loads((ui / 'report.json').read_text())
source = Path(sys.argv[1]) if len(sys.argv) > 1 else ui.parent / report['artifact_paths']['dataset']
manifest = json.loads((source / 'manifest.json').read_text())
if manifest != report['dataset']['manifest']:
    raise ValueError('Dataset manifest differs from bundled report; sync the matching report first.')
df = pd.read_parquet(source / 'flows.parquet')
if len(df) != manifest['row_count'] or set(df.run_id) != {report['run_id']}:
    raise ValueError('Dataset does not match the report run/count.')
out = ui / 'raw-data'
out.mkdir(exist_ok=True)
chunks = []
for start in range(0, len(df), 1000):
    name = f'flows-{start // 1000:04d}.json'
    # Column order is preserved in manifest; compact row arrays avoid repeating keys.
    (out / name).write_text(df.iloc[start:start + 1000].to_json(orient='values', double_precision=15), encoding='utf-8')
    chunks.append(name)
(out / 'manifest.json').write_text(json.dumps({
    'run_id': report['run_id'], 'schema_version': report['schema_version'],
    'dataset_generated_at': manifest['generated_at'], 'row_count': len(df),
    'columns': list(df.columns), 'chunks': chunks,
}, indent=2))
print(f'Exported {len(df):,} raw flows in {len(chunks)} static chunks.')
