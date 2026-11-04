// The sheet's colour at clock time t: the latest recorded frame, blended toward the next frame under a soft mask that
// follows the brush along each stroke's footprint (frames are sparser than strokes), plus the rejected candidates as
// flickering ghost strokes. Drawn into one canvas that is the sheet material's map.
import * as THREE from 'three';
import { pointAt } from './replay.js';

const S = 2; // canvas pixels per simulator pixel

export class SheetPainter {
  constructor(replay, images) {
    this.r = replay;
    this.images = images;
    const w = replay.W * S, h = replay.H * S;
    const mk = () => { const c = document.createElement('canvas'); c.width = w; c.height = h; return c; };
    this.canvas = mk(); this.tmp = mk(); this.mask = mk();
    this.g = this.canvas.getContext('2d');
    this.gt = this.tmp.getContext('2d');
    this.gm = this.mask.getContext('2d');
    this.grain = grainCanvas(w, h);
    this.texture = new THREE.CanvasTexture(this.canvas);
    this.texture.colorSpace = THREE.SRGBColorSpace;
    this.texture.flipY = false;
    this.texture.anisotropy = 8;
    this.key = '';
    this.showGhosts = true;
  }

  draw(t) {
    const r = this.r;
    const sh = r.sheetAt(t);
    const ghosts = this.showGhosts ? r.ghostsAt(t) : [];
    const key = `${sh.a}|${sh.b}|${sh.global.toFixed(3)}|${sh.reveal.map((x) => `${x.seg.i}:${x.p.toFixed(3)}`).join(',')}|${ghosts.map((x) => x.alpha.toFixed(2)).join(',')}`;
    if (key === this.key) return false;
    this.key = key;
    const { g, gt, gm } = this;
    const w = this.canvas.width, h = this.canvas.height;
    g.imageSmoothingEnabled = true; g.imageSmoothingQuality = 'high';
    g.globalCompositeOperation = 'source-over'; g.globalAlpha = 1;
    g.drawImage(this.images[sh.a], 0, 0, w, h);
    if (sh.b !== sh.a && (sh.global > 0 || sh.reveal.length)) {
      gm.clearRect(0, 0, w, h);
      if (sh.global > 0) { gm.globalAlpha = sh.global; gm.fillStyle = '#fff'; gm.fillRect(0, 0, w, h); gm.globalAlpha = 1; }
      for (const { seg, p } of sh.reveal) footprint(gm, seg, p);
      gt.globalCompositeOperation = 'source-over';
      gt.clearRect(0, 0, w, h);
      gt.imageSmoothingEnabled = true; gt.imageSmoothingQuality = 'high';
      gt.drawImage(this.images[sh.b], 0, 0, w, h);
      gt.globalCompositeOperation = 'destination-in';
      gt.drawImage(this.mask, 0, 0);
      gt.globalCompositeOperation = 'source-over';
      g.drawImage(this.tmp, 0, 0);
    }
    for (const { r: c, alpha } of ghosts) ghost(g, c, alpha);
    // paper tooth: a fixed grain multiplied over everything
    g.globalCompositeOperation = 'multiply';
    g.drawImage(this.grain, 0, 0);
    g.globalCompositeOperation = 'source-over';
    this.texture.needsUpdate = true;
    return true;
  }
}

function tracePath(g, rs, p) {
  const n = Math.max(1, Math.round(p * (rs.pts.length - 1)));
  g.beginPath();
  g.moveTo(rs.pts[0][0] * S, rs.pts[0][1] * S);
  for (let k = 1; k <= n; k++) g.lineTo(rs.pts[k][0] * S, rs.pts[k][1] * S);
  if (p < 1) { const e = pointAt(rs, p); g.lineTo(e[0] * S, e[1] * S); }
}

function footprint(gm, seg, p) {
  if (p <= 0) return;
  const wpx = seg.o.op.width * S;
  gm.save();
  gm.filter = `blur(${Math.max(1.5, wpx * 0.12).toFixed(1)}px)`;
  gm.strokeStyle = '#fff'; gm.lineCap = 'round'; gm.lineJoin = 'round';
  gm.lineWidth = wpx * 1.12;
  tracePath(gm, seg.rs, p);
  gm.stroke();
  gm.restore();
}

function ghost(g, c, alpha) {
  const wpx = c.width * S;
  g.save();
  g.lineCap = 'round'; g.lineJoin = 'round';
  g.globalAlpha = 0.42 * alpha;
  g.strokeStyle = c.colour;
  g.lineWidth = wpx;
  tracePath(g, c.rs, 1);
  g.stroke();
  // a thin dashed centre line marks it as a candidate, not paint
  g.globalAlpha = 0.9 * alpha;
  g.setLineDash([7, 6]);
  g.lineWidth = Math.max(1.5, Math.min(4, wpx * 0.08));
  g.strokeStyle = c.colour;
  tracePath(g, c.rs, 1);
  g.stroke();
  g.restore();
}

function grainCanvas(w, h) {
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  const g = c.getContext('2d');
  const img = g.createImageData(w, h);
  let seed = 12345;
  const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
  // cold-press tooth: low-frequency cells plus fine noise, kept near white so it only darkens a little
  const cells = new Float32Array(Math.ceil(w / 6) * Math.ceil(h / 6)).map(rnd);
  const cw = Math.ceil(w / 6);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const v = 0.6 * cells[Math.floor(y / 6) * cw + Math.floor(x / 6)] + 0.4 * rnd();
      const k = 255 - Math.round(v * 16);
      const i = (y * w + x) * 4;
      img.data[i] = k; img.data[i + 1] = k; img.data[i + 2] = Math.min(255, k + 1); img.data[i + 3] = 255;
    }
  }
  g.putImageData(img, 0, 0);
  g.filter = 'blur(0.6px)';
  g.drawImage(c, 0, 0);
  return c;
}
