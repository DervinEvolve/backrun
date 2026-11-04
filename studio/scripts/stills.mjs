// Render every stage camera and the overview at 1920 x 1080 (deviceScaleFactor 1) into review/pass1/,
// plus a contact sheet. Needs the dev server on port 5426 (npm run dev).
// Usage: node scripts/stills.mjs [--out review/pass1] [--views overview,stroke] [--assets arm,palette]
import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { chromium } = require('playwright');
const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const arg = (k, d) => { const i = process.argv.indexOf(k); return i > 0 ? process.argv[i + 1] : d; };
const out = path.resolve(root, arg('--out', 'review/pass1'));
const base = arg('--url', 'http://127.0.0.1:5426/');
fs.mkdirSync(out, { recursive: true });

const browser = await chromium.launch({ args: ['--enable-gpu', '--ignore-gpu-blocklist', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
const page = await browser.newPage({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });
const errors = [];
page.on('pageerror', (e) => errors.push(String(e)));
page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });

const shots = [];
const assets = arg('--assets', '');
if (assets) {
  for (const a of assets.split(',')) {
    for (const yaw of [35, 215]) {
      await page.goto(`${base}?still&asset=${a}&yaw=${yaw}&figure`);
      await page.waitForSelector('body[data-ready="1"]', { timeout: 120000 });
      await page.evaluate(() => window.__app.renderOnce());
      const f = path.join(out, `asset-${a.replace('/', '_')}-${yaw}.png`);
      await page.screenshot({ path: f });
      shots.push(f);
      console.log(f, JSON.stringify(await page.evaluate(() => window.__app.stats())));
    }
  }
} else {
  await page.goto(`${base}?still`);
  await page.waitForSelector('body[data-ready="1"]', { timeout: 180000 });
  const views = arg('--views', '') ? arg('--views').split(',') : await page.evaluate(() => window.__app.views);
  for (const [i, v] of views.entries()) {
    await page.evaluate((id) => { window.__app.setView(id); window.__app.renderOnce(); }, v);
    await page.waitForTimeout(150);
    await page.evaluate(() => window.__app.renderOnce());
    const f = path.join(out, `${String(i).padStart(2, '0')}-${v}.png`);
    await page.screenshot({ path: f });
    shots.push(f);
    console.log(f);
  }
  const stats = await page.evaluate(() => window.__app.stats());
  fs.writeFileSync(path.join(out, 'stats.json'), JSON.stringify(stats, null, 1));
  console.log(JSON.stringify(stats));
}
if (errors.length) console.log('page errors:\n' + errors.join('\n'));
await browser.close();

// contact sheet: 3 columns of 640 x 360 tiles with the file name burnt in
if (shots.length > 1) {
  const sheet = path.join(out, assets ? 'contact-assets.png' : 'contact.png');
  const inputs = shots.flatMap((f) => ['-i', f]);
  const n = shots.length, cols = 3, rows = Math.ceil(n / cols);
  const tiles = shots.map((f, i) => `[${i}:v]scale=640:360,drawtext=text='${path.basename(f, '.png')}':x=8:y=8:fontsize=20:fontcolor=white:box=1:boxcolor=black@0.5[t${i}]`).join(';');
  const layout = shots.map((_, i) => `${(i % cols) * 640}_${Math.floor(i / cols) * 360}`).join('|');
  const filter = `${tiles};${shots.map((_, i) => `[t${i}]`).join('')}xstack=inputs=${n}:layout=${layout}:fill=black[o]`;
  execFileSync('ffmpeg', ['-y', '-loglevel', 'error', ...inputs, '-filter_complex', filter, '-map', '[o]', '-frames:v', '1', sheet]);
  console.log(sheet, `${cols}x${rows}`);
}
