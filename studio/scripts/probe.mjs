// Quick checks against the running dev server: page errors, contact measurements over the run, stills at given times.
// Usage: node scripts/probe.mjs [--times 5,20] [--view stroke] [--contacts] [--out review/pass2/probe]
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
const { chromium } = require('playwright');
const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const arg = (k, d) => { const i = process.argv.indexOf(k); return i > 0 ? process.argv[i + 1] : d; };
const out = path.resolve(root, arg('--out', 'review/pass2/probe'));
fs.mkdirSync(out, { recursive: true });
const w = Number(arg('--w', 1920)), h = Number(arg('--h', 1080));
const browser = await chromium.launch({ args: ['--enable-gpu', '--ignore-gpu-blocklist', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
const page = await browser.newPage({ viewport: { width: w, height: h }, deviceScaleFactor: 1 });
const errors = [];
page.on('pageerror', (e) => errors.push(String(e.stack || e)));
page.on('console', (m) => { if (m.type() === 'error' || m.type() === 'warning') errors.push(m.text()); });
await page.goto(`http://127.0.0.1:5426/?still&paused${arg('--q', '') ? '&' + arg('--q') : ''}`);
await page.waitForSelector('body[data-ready="1"]', { timeout: 180000 });
if (process.argv.includes('--contacts')) {
  const res = await page.evaluate(() => {
    const a = window.__app, r = a.replay, out = { sheet: [], well: [], wrongWell: [], ik: [] };
    for (let t = 0; t < r.duration; t += 0.05) {
      const c = a.contactAt(t);
      if (c.ikError_mm > 1) out.ik.push(c);
      if (c.contact === 'sheet') out.sheet.push(c);
      if (c.contact?.startsWith('well:')) { out.well.push(c); }
    }
    const max = (arr, k) => arr.reduce((m, c) => Math.max(m, Math.abs(c[k] ?? 0)), 0);
    // first well dip of each stroke must be its main pigment
    for (const s of r.strokeSegs) {
      const first = s.stops.find((d) => d.at === 'well');
      const names = ['quinacridone_rose', 'cadmium_yellow', 'french_ultramarine', 'cerulean_blue', 'burnt_umber', 'hookers_green', 'indian_red', 'paynes_grey'];
      if (names[first.well] !== s.main) out.wrongWell.push(s.i);
    }
    return { sheetSamples: out.sheet.length, maxTipToSheet_mm: max(out.sheet, 'tipToSheet_mm'), maxTipToPath_mm: max(out.sheet, 'tipToPath_mm'),
      wellSamples: out.well.length, maxTipToCake_mm: max(out.well, 'tipToCake_mm'), ikOver1mm: out.ik.length, ikWorst: out.ik.sort((a, b) => b.ikError_mm - a.ikError_mm).slice(0, 5), wrongWell: out.wrongWell, duration: r.duration };
  });
  console.log(JSON.stringify(res, null, 1));
}
const times = arg('--times', '').split(',').filter(Boolean).map(Number);
const views = arg('--view', 'stroke').split(',');
for (const v of views) for (const t of times) {
  await page.evaluate(([v, t]) => { window.__app.setView(v, true); window.__app.setTime(t); window.__app.renderOnce(); window.__app.renderOnce(); }, [v, t]);
  const f = path.join(out, `${v}-${String(t).replace('.', '_')}.png`);
  await page.screenshot({ path: f });
  console.log(f);
}
if (process.argv.includes('--stats')) console.log(JSON.stringify(await page.evaluate(() => window.__app.stats())));
if (errors.length) console.log('page errors:\n' + [...new Set(errors)].slice(0, 20).join('\n'));
await browser.close();
