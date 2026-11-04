// Pack a recorded Backrun run folder for the studio gallery.
// Usage: node scripts/pack-run.mjs <run folder> <name>
//
// Copies ledger.jsonl, run.json, strokes.jsonl, target.png, final.png and frames/ into public/runs/<name>/ as they
// were recorded, and writes three derived files the app reads:
//   index.json   run settings without any score, the op count, the frame list (each frame's simulator step and the
//                op it was written during, rebuilt from the recorder's frame rule), phases and timings
//   ops.json     the ledger with every score field removed, so the face-down score sheet holds in the app
//   scores.json  every score (per op, per rejected candidate, final), loaded only at the reveal
import fs from 'node:fs';
import path from 'node:path';

const [src, name] = process.argv.slice(2);
if (!src || !name || !/^[a-z0-9_-]+$/i.test(name)) {
  console.error('usage: node scripts/pack-run.mjs <run folder> <name>   (name: letters, digits, _ or -)');
  process.exit(1);
}
const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const out = path.join(root, 'public', 'runs', name);
const need = ['ledger.jsonl', 'run.json', 'strokes.jsonl', 'target.png', 'final.png'];
for (const f of need) if (!fs.existsSync(path.join(src, f))) throw new Error(`missing ${f} in ${src}`);

fs.rmSync(out, { recursive: true, force: true });
fs.mkdirSync(path.join(out, 'frames'), { recursive: true });
for (const f of need) fs.copyFileSync(path.join(src, f), path.join(out, f));
const frameFiles = fs.readdirSync(path.join(src, 'frames')).filter((f) => /^frame_\d+\.png$/.test(f)).sort();
for (const f of frameFiles) fs.copyFileSync(path.join(src, 'frames', f), path.join(out, 'frames', f));

const run = JSON.parse(fs.readFileSync(path.join(src, 'run.json'), 'utf8'));
const ledger = fs.readFileSync(path.join(src, 'ledger.jsonl'), 'utf8').split('\n').filter(Boolean).map((l) => JSON.parse(l));
const every = run.frames_every ?? 20;

// Rebuild which simulator step each frame shows. cli_paint saves a blank frame before the first op; the recorder then
// saves one whenever its step count reaches the next multiple of frames_every. A stroke is one step; a settle dry runs
// in chunks of frames_every; a phase or final dry runs blocks of 25 steps, each in chunks of frames_every. finish()
// saves the last frame.
const frames = [{ step: 0, op: -1 }];
let steps = 0, next = 0;
const frameAt = (k, op) => {
  steps += k;
  if (steps >= next) {
    frames.push({ step: steps, op });
    while (next <= steps) next += every;
  }
};
if (frameFiles.length) next = 0; // the blank frame does not move the schedule; the first stroke frame lands at step 1
for (const l of ledger) {
  if (l.op.op === 'stroke') frameAt(1, l.index);
  else if (l.op.op === 'dry') {
    const chunks = (n) => { while (n > 0) { const k = Math.min(n, every); frameAt(k, l.index); n -= k; } };
    if (l.dry_kind === 'phase' || l.dry_kind === 'final') {
      let n = l.op.steps;
      while (n > 0) { const b = Math.min(25, n); chunks(b); n -= b; }
    } else chunks(l.op.steps);
  }
  if (steps !== l.sim_steps) console.warn(`step count drift at op ${l.index}: rebuilt ${steps}, ledger ${l.sim_steps}`);
}
frames.push({ step: steps, op: ledger.length - 1, final: true });
if (frames.length !== frameFiles.length) {
  console.warn(`rebuilt ${frames.length} frames, folder has ${frameFiles.length}; frame steps are spread evenly instead`);
  frames.length = 0;
  frameFiles.forEach((_, i) => frames.push({ step: Math.round(i * steps / Math.max(1, frameFiles.length - 1)), op: null }));
}
frames.forEach((f, i) => { f.file = `frames/${frameFiles[i]}`; });

// split scores from everything else
const SCORE_KEYS = ['score', 'predicted_score', 'candidate_score', 'predicted_before_dry'];
const scores = { final: run.final_score ?? run.painter_stats?.final_score ?? null, ops: [], rejected: [] };
const ops = ledger.map((l) => {
  const o = { ...l };
  scores.ops.push(Object.fromEntries(SCORE_KEYS.filter((k) => k in l).map((k) => [k, l[k]])));
  for (const k of SCORE_KEYS) delete o[k];
  if (Array.isArray(o.rejected)) {
    scores.rejected.push({ index: l.index, scores: o.rejected.map((r) => r.score ?? null) });
    o.rejected = o.rejected.map(({ score, ...r }) => r);
  }
  return o;
});

const { final_score, ...runNoScore } = run;
const stats = { ...(run.painter_stats ?? {}) };
delete stats.final_score;
runNoScore.painter_stats = stats;

const phases = [];
for (const l of ledger) {
  const p = l.phase ?? 'paint';
  if (!phases.length || phases.at(-1).name !== p) phases.push({ name: p, index: l.phase_index ?? phases.length, first_op: l.index, last_op: l.index });
  else phases.at(-1).last_op = l.index;
}
const pigments = new Set();
for (const l of ledger) for (const k of Object.keys(l.op.pigments ?? {})) pigments.add(k);

const index = {
  name,
  source: path.basename(path.resolve(src)),
  packed_at: new Date().toISOString(),
  run: runNoScore,
  op_count: ledger.length,
  stroke_count: ledger.filter((l) => l.op.op === 'stroke').length,
  sim_steps: steps,
  frames_every: every,
  frames,
  phases,
  pigments_used: [...pigments],
  timings: run.timings ?? {},
  files: { ops: 'ops.json', scores: 'scores.json', target: 'target.png', final: 'final.png', ledger: 'ledger.jsonl', strokes: 'strokes.jsonl', run: 'run.json' },
};
fs.writeFileSync(path.join(out, 'index.json'), JSON.stringify(index, null, 1));
fs.writeFileSync(path.join(out, 'ops.json'), JSON.stringify(ops));
fs.writeFileSync(path.join(out, 'scores.json'), JSON.stringify(scores));

// gallery list
const galleryPath = path.join(root, 'public', 'runs', 'gallery.json');
const gallery = fs.existsSync(galleryPath) ? JSON.parse(fs.readFileSync(galleryPath, 'utf8')) : { runs: [] };
gallery.runs = gallery.runs.filter((r) => r.name !== name);
gallery.runs.push({ name, painter: run.painter ?? null, target: run.target_path ?? null, strokes: index.stroke_count });
fs.writeFileSync(galleryPath, JSON.stringify(gallery, null, 1));

const bytes = fs.readdirSync(out, { recursive: true }).map((f) => path.join(out, f)).filter((f) => fs.statSync(f).isFile())
  .reduce((s, f) => s + fs.statSync(f).size, 0);
console.log(`packed ${src} -> public/runs/${name}: ${ledger.length} ops, ${index.stroke_count} strokes, ${frames.length} frames, ${(bytes / 1e6).toFixed(2)} MB`);
