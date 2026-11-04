// Compress the raw Blender GLBs (blender/out) into public/models with meshopt,
// and write public/models/manifest.json with triangle counts and byte sizes.
import fs from 'node:fs';
import path from 'node:path';
import { NodeIO } from '@gltf-transform/core';
import { ALL_EXTENSIONS } from '@gltf-transform/extensions';
import { meshopt, reorder, prune, dedup } from '@gltf-transform/functions';
import { MeshoptEncoder, MeshoptDecoder } from 'meshoptimizer';

const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const src = path.join(root, 'blender/out');
const dst = path.join(root, 'public/models');
await MeshoptEncoder.ready;
await MeshoptDecoder.ready;
const io = new NodeIO().registerExtensions(ALL_EXTENSIONS).registerDependencies({
  'meshopt.encoder': MeshoptEncoder, 'meshopt.decoder': MeshoptDecoder,
});

function tris(doc) {
  let t = 0;
  for (const m of doc.getRoot().listMeshes())
    for (const p of m.listPrimitives()) t += (p.getIndices()?.getCount() ?? p.getAttribute('POSITION').getCount()) / 3;
  return t;
}

const manifest = { hand_built: {}, simready: {} };
for (const f of fs.readdirSync(src).filter((f) => f.endsWith('.glb')).sort()) {
  const doc = await io.read(path.join(src, f));
  await doc.transform(dedup(), prune({ keepLeaves: true, keepAttributes: true }), reorder({ encoder: MeshoptEncoder }),
    meshopt({ encoder: MeshoptEncoder, level: 'medium' }));
  const out = path.join(dst, f);
  await io.write(out, doc);
  manifest.hand_built[f] = { tris: tris(doc), bytes: fs.statSync(out).size, raw_bytes: fs.statSync(path.join(src, f)).size };
}
const sr = path.join(dst, 'simready');
for (const f of fs.readdirSync(sr).filter((f) => f.endsWith('.glb')).sort()) {
  const doc = await io.read(path.join(sr, f));
  const b = [[1e9, 1e9, 1e9], [-1e9, -1e9, -1e9]];
  for (const n of doc.getRoot().listNodes()) {
    const m = n.getMesh(); if (!m) continue;
    const w = n.getWorldMatrix();
    for (const p of m.listPrimitives()) {
      const a = p.getAttribute('POSITION'); const v = [];
      for (let i = 0; i < a.getCount(); i++) {
        a.getElement(i, v);
        const x = w[0] * v[0] + w[4] * v[1] + w[8] * v[2] + w[12], y = w[1] * v[0] + w[5] * v[1] + w[9] * v[2] + w[13], z = w[2] * v[0] + w[6] * v[1] + w[10] * v[2] + w[14];
        [x, y, z].forEach((c, k) => { b[0][k] = Math.min(b[0][k], c); b[1][k] = Math.max(b[1][k], c); });
      }
    }
  }
  manifest.simready[f] = { tris: tris(doc), bytes: fs.statSync(path.join(sr, f)).size,
    size_m: b[1].map((v, i) => +(v - b[0][i]).toFixed(3)), min: b[0].map((v) => +v.toFixed(3)),
    materials: doc.getRoot().listMaterials().map((m) => m.getName()) };
}
let T = 0, B = 0;
for (const g of Object.values(manifest)) for (const v of Object.values(g)) { T += v.tris; B += v.bytes; }
manifest.total_file = { tris: T, bytes: B };
fs.writeFileSync(path.join(dst, 'manifest.json'), JSON.stringify(manifest, null, 1));
console.log(JSON.stringify(manifest, null, 1));
