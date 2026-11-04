// Browser measurements against the running dev server: frames per second during playback on the overview and on the
// Stroke camera, triangles and draw calls, and the page's JS heap.
// Usage: node scripts/measure.mjs [--w 1280 --h 720] [--seconds 6] [--gpu]
// Without --gpu it runs Chromium's software rasteriser (SwiftShader), which is what a headless build box has; the fps
// it reports is a floor, not what a GPU gives.
import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
const { chromium } = require('playwright');
const arg = (k, d) => { const i = process.argv.indexOf(k); return i > 0 ? process.argv[i + 1] : d; };
const w = Number(arg('--w', 1280)), h = Number(arg('--h', 720)), secs = Number(arg('--seconds', 6));
const args = process.argv.includes('--gpu') ? ['--enable-gpu', '--ignore-gpu-blocklist']
  : ['--enable-gpu', '--ignore-gpu-blocklist', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'];
const browser = await chromium.launch({ args: [...args, '--enable-precise-memory-info'] });
const page = await browser.newPage({ viewport: { width: w, height: h }, deviceScaleFactor: 1 });
const errors = [];
page.on('pageerror', (e) => errors.push(String(e)));
const cdp = await page.context().newCDPSession(page);
await cdp.send('Performance.enable');
const renderer = async () => page.evaluate(() => {
  const gl = document.createElement('canvas').getContext('webgl2');
  const ext = gl?.getExtension('WEBGL_debug_renderer_info');
  return ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : 'unknown';
});
// count real animation frames over a window of wall time
const fpsOver = (s) => page.evaluate((s) => new Promise((res) => {
  let n = 0; const t0 = performance.now(); const times = [];
  let last = t0;
  const f = (t) => { n++; times.push(t - last); last = t; if (t - t0 < s * 1000) requestAnimationFrame(f); else {
    times.sort((a, b) => a - b);
    res({ fps: +(n / ((t - t0) / 1000)).toFixed(1), frames: n, p95_ms: +times[Math.floor(times.length * 0.95)].toFixed(1) });
  } };
  requestAnimationFrame(f);
}), s);
const heap = async () => {
  const m = Object.fromEntries((await cdp.send('Performance.getMetrics')).metrics.map((x) => [x.name, x.value]));
  return { usedMB: +(m.JSHeapUsedSize / 2 ** 20).toFixed(1), totalMB: +(m.JSHeapTotalSize / 2 ** 20).toFixed(1) };
};
const out = { viewport: `${w} x ${h}`, renderer: null, views: {} };
await page.goto('http://127.0.0.1:5426/?t=40&view=overview');
await page.waitForSelector('body[data-ready="1"]', { timeout: 300000 });
out.renderer = await renderer();
for (const view of ['overview', 'stroke']) {
  await page.evaluate((v) => { const a = window.__app; a.setView(v, true); a.setTime(40); a.setSpeed(1); a.play(); }, view);
  await page.waitForTimeout(1500); // let shaders compile and the AO settle
  const f = await fpsOver(secs);
  const st = await page.evaluate(() => window.__app.stats());
  out.views[view] = { ...f, playing: await page.evaluate(() => window.__app.clock().playing), triangles: st.triangles,
    trianglesRenderedWithShadows: st.renderedTrianglesWithShadows, drawCallsIncludingShadows: st.drawCallsIncludingShadows,
    meshes: st.meshes, textures: st.textures };
}
// dashboard at the end of the replay, for comparison with the ledger
out.dashboardAtEnd = await page.evaluate(async () => {
  const a = window.__app; a.pause(); a.setTime(a.replay.duration - 0.01);
  await new Promise((r) => setTimeout(r, 1500));
  const q = (id) => document.getElementById(id)?.textContent.trim();
  return { strokes: q('k-strokes'), strokesOf: q('k-skipped'), wet: q('k-wet'), steps: q('k-steps'), stepsOf: q('k-stepsof'), wall: q('k-wall'),
    wallOf: q('k-wallof'), evals: q('k-evals'), evalsOf: q('k-evalsof'), total: q('s-total'), parts: q('s-parts'),
    legend: [...document.querySelectorAll('#legend-rows .lrow')].map((r) => r.textContent.replace(/\s+/g, ' ').trim()) };
});
await page.evaluate(() => window.gc?.());
out.heap = await heap();
if (errors.length) out.errors = [...new Set(errors)].slice(0, 10);
console.log(JSON.stringify(out, null, 1));
await browser.close();
