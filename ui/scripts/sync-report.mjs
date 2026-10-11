import {readFile, writeFile} from 'node:fs/promises';
import {resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {validateReport} from '../data.mjs';

const source = process.argv[2] ? resolve(process.argv[2]) : fileURLToPath(new URL('../../data/generated/showcase/report.json', import.meta.url));
const destination = new URL('../report.json', import.meta.url);
const text = await readFile(source, 'utf8');
const report = validateReport(JSON.parse(text));
await writeFile(destination, text);
console.log(`Copied ${report.run_id}: ${report.dataset.manifest.row_count.toLocaleString()} flows to ui/report.json.`);
