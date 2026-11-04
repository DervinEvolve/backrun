// Record the 30 s demo of the studio: drives the running app (npm run dev, port 5426) with real mouse and keyboard
// events on a virtual clock, captures one frame per 1/30 s, and encodes review/pass2/demo.mp4 plus a small GIF.
//
// Usage: node scripts/record-demo.mjs [--scale 1|0.5] [--out review/pass2] [--from 0 --to 900] [--keep]
//   --scale 0.5 renders at 960 x 540 physical pixels with the 1920 x 1080 layout (fast preview).
// Progress goes to <out>/record.log. Frames go to <out>/frames/ and are removed after encoding unless --keep.
//
// The init script replaces requestAnimationFrame, performance.now and Date.now, so the app (replay clock, camera
// flights, damping) advances exactly 1/30 s per captured frame, whatever the render takes.
import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { chromium } = require('playwright');
const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const arg = (k, d) => { const i = process.argv.indexOf(k); return i > 0 ? process.argv[i + 1] : d; };
const out = path.resolve(root, arg('--out', 'review/pass2'));
const scale = Number(arg('--scale', 1));
const FPS = 30, N = 30 * FPS;
const from = Number(arg('--from', 0)), to = Number(arg('--to', N));
const framesDir = path.join(out, 'frames');
fs.mkdirSync(framesDir, { recursive: true });
const logf = path.join(out, 'record.log');
const log = (s) => { fs.appendFileSync(logf, `${new Date().toISOString()} ${s}\n`); };
fs.writeFileSync(logf, '');

const T0 = 27.0; // replay time at the first frame: the fifth wash is about to be laid

const browser = await chromium.launch({ args: ['--enable-gpu', '--ignore-gpu-blocklist', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
const ctx = await browser.newContext({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: scale });
await ctx.addInitScript(() => {
  let now = 0;
  const q = [];
  performance.now = () => now;
  const base = 1793760000000;
  Date.now = () => base + now;
  window.requestAnimationFrame = (cb) => { q.push(cb); return q.length; };
  window.cancelAnimationFrame = () => {};
  window.__tick = (ms) => {
    now += ms;
    const cbs = q.splice(0);
    for (const cb of cbs) cb(now);
    for (const a of document.getAnimations()) { try { a.currentTime = now; } catch { /* finished */ } }
  };
  // a visible cursor and a ring where it clicks
  addEventListener('DOMContentLoaded', () => {
    const c = document.createElement('div');
    c.id = '__cursor';
    c.innerHTML = '<svg width="28" height="28" viewBox="0 0 28 28"><path d="M4 2l17 12-7.5 1.2L18 24l-3.6 1.6-4-8.6L4 22z" fill="#fff" stroke="#1d1712" stroke-width="1.6" stroke-linejoin="round"/></svg>';
    Object.assign(c.style, { position: 'fixed', left: '0', top: '0', zIndex: 99, pointerEvents: 'none', transform: 'translate(-4px,-2px)' });
    const ring = document.createElement('div');
    Object.assign(ring.style, { position: 'fixed', width: '34px', height: '34px', marginLeft: '-17px', marginTop: '-17px', borderRadius: '50%',
      border: '3px solid #FFB74D', zIndex: 98, pointerEvents: 'none', opacity: '0' });
    document.body.append(ring, c);
    let ringAt = -1;
    addEventListener('mousemove', (e) => { c.style.left = `${e.clientX}px`; c.style.top = `${e.clientY}px`; }, true);
    addEventListener('mousedown', (e) => { ring.style.left = `${e.clientX}px`; ring.style.top = `${e.clientY}px`; ringAt = now; }, true);
    const tickRing = () => {
      const a = ringAt < 0 ? 1 : (now - ringAt) / 450;
      ring.style.opacity = a < 1 ? String(1 - a) : '0';
      ring.style.transform = `scale(${0.6 + 0.8 * Math.min(1, a)})`;
      window.requestAnimationFrame(tickRing);
    };
    window.requestAnimationFrame(tickRing);
  });
});
const page = await ctx.newPage();
const errors = [];
page.on('pageerror', (e) => errors.push(String(e)));
await page.goto(`http://127.0.0.1:5426/?view=stroke&t=${T0}`);
await page.waitForSelector('body[data-ready="1"]', { timeout: 300000 });
log('app ready');

// ------------------------------------------------------------------ the beats (film seconds)
const box = async (sel) => { const b = await page.locator(sel).first().boundingBox(); return { x: b.x + b.width / 2, y: b.y + b.height / 2 }; };
const speedBtn = (s) => `#speeds .btn[data-speed="${s}"]`;
const scrubAt = async (t) => {
  const b = await page.locator('#scrub').boundingBox();
  const D = await page.evaluate(() => window.__app.replay.duration);
  return { x: b.x + (t / D) * b.width, y: b.y + b.height / 2 };
};
// each action: { at, kind, ... }; move actions glide the cursor from where it is to the target over `dur` seconds
const plan = [
  { at: 0.3, kind: 'move', to: async () => ({ x: 760, y: 470 }), dur: 0.2 },
  { at: 0.5, kind: 'down' },
  { at: 0.5, kind: 'drag', dx: 260, dy: -18, dur: 4.0 },
  { at: 4.5, kind: 'up' },
  { at: 4.9, kind: 'move', to: async () => page.evaluate(() => window.__app.screenOf('brush')), dur: 0.6, track: true },
  { at: 5.6, kind: 'click' },
  { at: 6.3, kind: 'move', to: () => box('#card-follow'), dur: 0.6 },
  { at: 7.0, kind: 'click' },
  { at: 10.9, kind: 'move', to: () => box('#legend-rows .lrow:nth-child(8)'), dur: 0.7 },
  { at: 11.6, kind: 'click' },
  { at: 13.6, kind: 'key', key: 'Escape' },
  { at: 13.7, kind: 'key', key: '4' },
  { at: 17.4, kind: 'move', to: () => box(speedBtn(5)), dur: 0.6 },
  { at: 18.0, kind: 'click' },
  { at: 18.3, kind: 'key', key: '7' },
  { at: 18.6, kind: 'move', to: () => box(speedBtn(1)), dur: 0.5 },
  { at: 19.2, kind: 'click' },
  { at: 21.7, kind: 'move', to: () => box(speedBtn(50)), dur: 0.5 },
  { at: 22.3, kind: 'click' },
  { at: 22.6, kind: 'key', key: '5' },
  { at: 23.3, kind: 'move', to: () => box(speedBtn(1)), dur: 0.45 },
  { at: 23.8, kind: 'click' },
  { at: 24.0, kind: 'key', key: '8' },
  { at: 24.0, kind: 'move', to: () => scrubAt(268.4), dur: 0.45 },
  { at: 24.5, kind: 'click' },
  { at: 24.7, kind: 'move', to: () => box('#guess'), dur: 0.5 },
  { at: 25.2, kind: 'click' },
  { at: 25.4, kind: 'type', text: '6.2' },
  { at: 25.9, kind: 'move', to: () => box('#guess-keep'), dur: 0.45 },
  { at: 26.4, kind: 'click' },
  { at: 26.8, kind: 'move', to: async () => ({ x: 1250, y: 640 }), dur: 1.2 },
];

let cur = { x: 960, y: 540 }, glide = null, drag = null, typing = null;
const guardLog = [];
const ease = (u) => (u < 0.5 ? 2 * u * u : 1 - Math.pow(-2 * u + 2, 2) / 2);
const done = new Set();
const t0 = Date.now();
await page.mouse.move(cur.x, cur.y);
for (let f = 0; f < to; f++) {
  const ft = f / FPS;
  for (const [i, a] of plan.entries()) {
    if (done.has(i) || ft + 1e-6 < a.at) continue;
    done.add(i);
    if (a.kind === 'move') glide = { from: { ...cur }, to: await a.to(), a: ft, dur: a.dur, track: a.track ? a.to : null };
    else if (a.kind === 'down') await page.mouse.down();
    else if (a.kind === 'up') await page.mouse.up();
    else if (a.kind === 'drag') drag = { from: { ...cur }, dx: a.dx, dy: a.dy, a: ft, dur: a.dur };
    else if (a.kind === 'click') { await page.mouse.down(); await page.mouse.up(); }
    else if (a.kind === 'key') await page.keyboard.press(a.key);
    else if (a.kind === 'type') typing = { text: a.text, a: ft, k: 0 };
    log(`f${f} ${a.kind} ${a.key ?? ''}`);
  }
  if (glide) {
    if (glide.track) glide.to = await glide.track();
    const u = Math.min(1, (ft - glide.a) / glide.dur);
    cur = { x: glide.from.x + (glide.to.x - glide.from.x) * ease(u), y: glide.from.y + (glide.to.y - glide.from.y) * ease(u) };
    await page.mouse.move(cur.x, cur.y);
    if (u >= 1) glide = null;
  }
  if (drag) {
    const u = Math.min(1, (ft - drag.a) / drag.dur);
    cur = { x: drag.from.x + drag.dx * ease(u), y: drag.from.y + drag.dy * ease(u) };
    await page.mouse.move(cur.x, cur.y);
    if (u >= 1) drag = null;
  }
  if (typing && typing.k < typing.text.length && ft >= typing.a + typing.k * 0.12) {
    await page.keyboard.type(typing.text[typing.k]); typing.k++;
  }
  await page.evaluate((ms) => window.__tick(ms), 1000 / FPS);
  guardLog.push({ time: ft, report: await page.evaluate(() => window.__app.guard()) });
  if (f >= from) {
    await page.screenshot({ path: path.join(framesDir, `${String(f).padStart(5, '0')}.jpg`), type: 'jpeg', quality: 92 });
  }
  if (f % 30 === 0) {
    const st = await page.evaluate(() => ({ t: window.__app.getTime(), ...window.__app.camera() }));
    log(`frame ${f} film ${ft.toFixed(2)} s replay ${st.t.toFixed(2)} s elapsed ${((Date.now() - t0) / 1000).toFixed(0)} s cam ${JSON.stringify(st.position)}`);
  }
}
if (errors.length) log(`page errors: ${errors.join(' | ')}`);
// frame guard: the brush must stay in frame and uncovered except where the beat is about something else
const { findFaults } = await import('../src/frame-guard.js');
const allowed = [
  { name: 'brush', from: 11.6, to: 13.7 }, // well close-up: the brush arrives from off frame
  { name: 'brush', from: 18.3, to: 30 },   // the panel's ride to the rack, then the reveal
  { name: 'sheet', from: 11.6, to: 13.9 }, // the well close-up
  { name: 'sheet', from: 18.3, to: 22.6 }, // carried off the board, framed by the dry camera
  { name: 'sheet', from: 24.0, to: 30 },
  { name: 'scorecard', from: 0, to: 24.3 },
];
const faults = findFaults(guardLog, { minHeightPixels: 40, minVisibleShare: 0.6, graceSeconds: 0.3, allowed });
fs.writeFileSync(path.join(out, 'guard.json'), JSON.stringify({ faults, frames: guardLog.filter((_, i) => i % 15 === 0) }, null, 1));
log(`frame guard faults: ${JSON.stringify(faults)}`);
await browser.close();

if (from === 0 && to === N) {
  const mp4 = path.join(out, 'demo.mp4'), gif = path.join(out, 'demo-preview.gif');
  const nice = (args) => execFileSync('nice', ['-n', '10', 'ffmpeg', '-y', '-loglevel', 'error', ...args]);
  nice(['-framerate', String(FPS), '-i', path.join(framesDir, '%05d.jpg'), '-vf', 'scale=1920:1080:flags=lanczos', '-c:v', 'libx264',
    '-preset', 'slow', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', mp4]);
  nice(['-i', mp4, '-vf', 'fps=10,scale=480:-2:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer:bayer_scale=4', gif]);
  log(`encoded ${mp4} and ${gif}`);
  if (!process.argv.includes('--keep')) fs.rmSync(framesDir, { recursive: true, force: true });
}
log('done');
