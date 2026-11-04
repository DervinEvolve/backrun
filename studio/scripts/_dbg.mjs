import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
const { chromium } = require('playwright');
const browser = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
const page = await browser.newPage({ viewport: { width: 800, height: 450 } });
page.on('pageerror', (e) => console.log('ERR', String(e)));
await page.goto('http://127.0.0.1:5426/?still&paused&ao=0');
await page.waitForSelector('body[data-ready="1"]', { timeout: 180000 });
console.log(await page.evaluate(() => {
  const a = window.__app; const out = [];
  for (const t of [40, 52, 53, 55, 57, 61]) { a.setTime(t); a.renderOnce(); const p = a.subjects().panel; out.push([t, p.position.toArray().map(v => +v.toFixed(3)), p.quaternion.toArray().map(v => +v.toFixed(3)), p.visible, p.parent?.type]); }
  return JSON.stringify(out);
}));
await browser.close();
