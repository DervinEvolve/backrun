// Count triangles and bytes per GLB in public/models (meshopt-compressed files are decoded first).
// Usage: node scripts/count.mjs [--json]
import fs from 'node:fs';
import path from 'node:path';
import { NodeIO } from '@gltf-transform/core';
import { ALL_EXTENSIONS } from '@gltf-transform/extensions';
import { MeshoptDecoder } from 'meshoptimizer';

const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..', 'public/models');
await MeshoptDecoder.ready;
const io = new NodeIO().registerExtensions(ALL_EXTENSIONS).registerDependencies({ 'meshopt.decoder': MeshoptDecoder });
const files = [];
const walk = (d) => { for (const f of fs.readdirSync(d)) { const p = path.join(d, f); if (fs.statSync(p).isDirectory()) walk(p); else if (f.endsWith('.glb')) files.push(p); } };
walk(root);
const rows = [];
for (const f of files.sort()) {
  const doc = await io.read(f);
  let tris = 0;
  for (const m of doc.getRoot().listMeshes()) for (const p of m.listPrimitives()) {
    if (p.getMode() !== 4) continue; // triangles only
    tris += (p.getIndices()?.getCount() ?? p.getAttribute('POSITION').getCount()) / 3;
  }
  rows.push({ file: path.relative(root, f), tris, bytes: fs.statSync(f).size });
}
const total = rows.reduce((a, r) => ({ tris: a.tris + r.tris, bytes: a.bytes + r.bytes }), { tris: 0, bytes: 0 });
if (process.argv.includes('--json')) console.log(JSON.stringify({ rows, total }, null, 1));
else {
  for (const r of rows) console.log(`${r.file.padEnd(32)} ${String(r.tris).padStart(7)} tris ${String(r.bytes).padStart(9)} bytes`);
  console.log(`${'total'.padEnd(32)} ${String(total.tris).padStart(7)} tris ${String(total.bytes).padStart(9)} bytes`);
}
