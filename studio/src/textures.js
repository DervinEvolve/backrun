// Procedural surface textures for the studio, generated at load time so they cost no download.
// Every texture is seeded, tiles seamlessly, and is near white so the material colour
// (taken from the glTF, in the warm palette) carries the hue.
import * as THREE from 'three';

function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// periodic value noise on a lattice of period p (cells per tile)
function makeNoise(seed, p) {
  const r = rng(seed);
  const g = new Float32Array(p * p).map(() => r());
  const s = (t) => t * t * (3 - 2 * t);
  return (x, y) => {
    const xi = Math.floor(x), yi = Math.floor(y);
    const xf = s(x - xi), yf = s(y - yi);
    const i0 = ((xi % p) + p) % p, j0 = ((yi % p) + p) % p, i1 = (i0 + 1) % p, j1 = (j0 + 1) % p;
    const a = g[j0 * p + i0], b = g[j0 * p + i1], c = g[j1 * p + i0], d = g[j1 * p + i1];
    return a + (b - a) * xf + (c - a) * yf + (a - b - c + d) * xf * yf;
  };
}

function fbm(noise, x, y, oct = 4) {
  let v = 0, amp = 0.5, f = 1;
  for (let i = 0; i < oct; i++) { v += amp * noise(x * f, y * f); amp *= 0.5; f *= 2; }
  return v / (1 - Math.pow(0.5, oct));
}

function toTexture(size, rgb, srgb = true) {
  const tex = new THREE.DataTexture(rgb, size, size, THREE.RGBAFormat);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.colorSpace = srgb ? THREE.SRGBColorSpace : THREE.NoColorSpace;
  tex.generateMipmaps = true;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.anisotropy = 8;
  tex.needsUpdate = true;
  return tex;
}

function normalFromHeight(h, size, strength) {
  const out = new Uint8Array(size * size * 4);
  const at = (x, y) => h[((y + size) % size) * size + ((x + size) % size)];
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const dx = (at(x + 1, y) - at(x - 1, y)) * strength;
      const dy = (at(x, y + 1) - at(x, y - 1)) * strength;
      const l = Math.hypot(dx, dy, 1);
      const i = (y * size + x) * 4;
      out[i] = (-dx / l * 0.5 + 0.5) * 255;
      out[i + 1] = (dy / l * 0.5 + 0.5) * 255;
      out[i + 2] = (1 / l * 0.5 + 0.5) * 255;
      out[i + 3] = 255;
    }
  }
  return toTexture(size, out, false);
}

function pack(size, fn) {
  const col = new Uint8Array(size * size * 4);
  const h = new Float32Array(size * size);
  const rough = new Uint8Array(size * size * 4);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const [r, g, b, hh, ro] = fn(x / size, y / size, x, y);
      const i = y * size + x;
      col[i * 4] = Math.min(255, r * 255); col[i * 4 + 1] = Math.min(255, g * 255); col[i * 4 + 2] = Math.min(255, b * 255); col[i * 4 + 3] = 255;
      h[i] = hh;
      const rv = Math.min(255, (ro ?? 1) * 255);
      rough[i * 4] = rv; rough[i * 4 + 1] = rv; rough[i * 4 + 2] = rv; rough[i * 4 + 3] = 255;
    }
  }
  return { col, h, rough };
}

// Wood grain along U. Tile covers `tile` metres.
export function woodTexture(seed = 1, size = 512, stain = false) {
  const n = makeNoise(seed, 8), n2 = makeNoise(seed + 7, 64), n3 = makeNoise(seed + 13, 6);
  const { col, h, rough } = pack(size, (u, v) => {
    // long straight grain along U: the warp changes slowly along the board and quickly across it
    const warp = fbm(n, u * 2, v * 8, 3) * 2.2;
    const ring = Math.pow(0.5 + 0.5 * Math.sin((v * 90 + warp) * Math.PI), 3);
    const fine = n2(u * 4, v * 64);
    let t = 0.86 + 0.08 * (1 - ring) + 0.06 * fine;
    let rr = 1.0;
    if (stain) {
      const s = fbm(n3, u * 6, v * 6, 3);
      if (s > 0.62) t *= 1 - (s - 0.62) * 0.9;
      rr = 0.85 + 0.15 * fine;
    }
    return [t * 1.0, t * 0.97, t * 0.93, ring * 0.4 + fine * 0.6, rr];
  });
  return { map: toTexture(size, col), normalMap: normalFromHeight(h, size, 0.35), roughnessMap: toTexture(size, rough, false) };
}

// Oiled oak floor planks running along U: 150 mm boards, tile 1.8 m.
export function floorTexture(seed = 3, size = 1024) {
  const tile = 1.8, board = 0.15, rows = Math.round(tile / board);
  const r = rng(seed);
  const rowsDef = [];
  for (let i = 0; i < rows; i++) {
    const cuts = []; let x = r() * 0.6;
    while (x < tile) { cuts.push(x); x += 0.7 + r() * 0.9; }
    rowsDef.push({ cuts, tones: cuts.map(() => 0.88 + r() * 0.2) });
  }
  const n = makeNoise(seed, 16), n2 = makeNoise(seed + 5, 128), n3 = makeNoise(seed + 9, 8);
  const { col, h, rough } = pack(size, (u, v) => {
    const xm = u * tile, ym = v * tile;
    const row = Math.min(rows - 1, Math.floor(ym / board));
    const def = rowsDef[row];
    let k = def.cuts.length - 1;
    for (let i = 0; i < def.cuts.length; i++) if (xm >= def.cuts[i]) k = i;
    if (xm < def.cuts[0]) k = def.cuts.length - 1;
    const tone = def.tones[k];
    const yIn = ym - row * board;
    let dEdge = Math.min(yIn, board - yIn);
    for (const c of def.cuts) dEdge = Math.min(dEdge, Math.abs(xm - c) * 1.0);
    const seam = dEdge < 0.0018 ? 0.55 : 1.0;
    const grain = 0.86 + 0.1 * Math.sin((yIn * 160 + fbm(n, u * 16, v * 16, 3) * 5 + k * 3) * Math.PI) * 0.5 + 0.08 * n2(u * 16, v * 128);
    const wear = 0.94 + 0.06 * fbm(n3, u * 8, v * 8, 3);
    const t = tone * grain * wear * seam;
    return [t, t * 0.96, t * 0.9, seam < 1 ? 0 : 0.6 + 0.4 * grain, seam < 1 ? 1 : 0.55 + 0.25 * (1 - wear)];
  });
  return { map: toTexture(size, col), normalMap: normalFromHeight(h, size, 2.0), roughnessMap: toTexture(size, rough, false), tile };
}

// Limewashed brick, stretcher bond: 215 x 65 brick, 10 mm joints, tile 0.9 m.
export function brickTexture(seed = 11, size = 1024) {
  const tile = 0.9, course = 0.075, unit = 0.225, joint = 0.01;
  const r = rng(seed);
  const courses = Math.round(tile / course), per = Math.round(tile / unit);
  const tones = Array.from({ length: courses * per }, () => r());
  const n = makeNoise(seed, 12), n2 = makeNoise(seed + 3, 96), n3 = makeNoise(seed + 21, 24);
  const { col, h, rough } = pack(size, (u, v) => {
    const xm = u * tile, ym = v * tile;
    const c = Math.floor(ym / course);
    const off = (c % 2) * unit / 2;
    const xs = (xm + off) % tile;
    const b = Math.floor(xs / unit);
    const xin = xs - b * unit, yin = ym - c * course;
    const dj = Math.min(xin, unit - xin, yin, course - yin);
    const mortar = dj < joint / 2;
    const wash = fbm(n, u * 12, v * 12, 4);
    const speck = n2(u * 96, v * 96);
    if (mortar) {
      const t = 0.86 + 0.06 * speck;
      return [t, t * 0.98, t * 0.94, 0.0, 1];
    }
    const tone = tones[(c % courses) * per + (b % per)];
    // limewash is thin in patches, letting the warm brick read through
    const thin = Math.max(0, (fbm(n3, u * 24, v * 24, 3) - 0.6) * 1.6) * (0.4 + tone * 0.5);
    const base = [0.98 - 0.04 * tone + 0.02 * speck, 0.96 - 0.04 * tone + 0.02 * speck, 0.92 - 0.05 * tone + 0.02 * speck];
    const brick = [0.74, 0.6, 0.5];
    const m = Math.min(0.45, thin);
    const edge = Math.min(1, dj / 0.004);
    const hh = 0.5 + 0.5 * edge + 0.15 * (wash - 0.5) + 0.08 * speck;
    return [base[0] * (1 - m) + brick[0] * m, base[1] * (1 - m) + brick[1] * m, base[2] * (1 - m) + brick[2] * m, hh, 0.95];
  });
  return { map: toTexture(size, col), normalMap: normalFromHeight(h, size, 3.0), roughnessMap: toTexture(size, rough, false), tile };
}

// Limewash plaster: soft mottling and trowel marks, tile 2 m.
export function plasterTexture(seed = 5, size = 512) {
  const n = makeNoise(seed, 8), n2 = makeNoise(seed + 1, 128), n3 = makeNoise(seed + 2, 16);
  const { col, h } = pack(size, (u, v) => {
    const m = fbm(n, u * 8, v * 8, 4);
    const s = n2(u * 128, v * 128);
    const tr = fbm(n3, u * 16 + v * 4, v * 16, 3);
    const t = 0.93 + 0.06 * m + 0.015 * s;
    return [t, t * 0.99, t * 0.965, tr * 0.6 + s * 0.15, 1];
  });
  return { map: toTexture(size, col), normalMap: normalFromHeight(h, size, 0.8), tile: 2.0 };
}

export function corkTexture(seed = 9, size = 512) {
  const n = makeNoise(seed, 128), n2 = makeNoise(seed + 4, 32);
  const { col, h } = pack(size, (u, v) => {
    const s = n(u * 128, v * 128), m = n2(u * 32, v * 32);
    const t = 0.78 + 0.25 * s * m + 0.1 * m;
    return [t, t * 0.95, t * 0.88, s, 1];
  });
  return { map: toTexture(size, col), normalMap: normalFromHeight(h, size, 1.5), tile: 0.4 };
}

// cold-press paper tooth, used as a normal map on the sheet and the cards
export function paperNormal(seed = 17, size = 512) {
  const n = makeNoise(seed, 64), n2 = makeNoise(seed + 2, 16);
  const h = new Float32Array(size * size);
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
    h[y * size + x] = 0.7 * n(x / size * 64, y / size * 64) + 0.3 * n2(x / size * 16, y / size * 16);
  }
  return normalFromHeight(h, size, 1.6);
}

// A canvas texture with a few lines of text on paper (placeholder cards and slips).
export function cardTexture(lines, opts = {}) {
  const w = opts.w ?? 512, hgt = opts.h ?? 352;
  const c = document.createElement('canvas');
  c.width = w; c.height = hgt;
  const g = c.getContext('2d');
  g.fillStyle = opts.bg ?? '#F4F0E6';
  g.fillRect(0, 0, w, hgt);
  if (opts.draw) opts.draw(g, w, hgt);
  g.fillStyle = opts.ink ?? '#3A3128';
  let y = opts.top ?? hgt * 0.22;
  for (const [text, size, font] of lines) {
    g.font = `${size}px ${font ?? 'Georgia, serif'}`;
    g.fillText(text, opts.left ?? w * 0.08, y);
    y += size * 1.45;
  }
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.flipY = false;
  t.anisotropy = 8;
  return t;
}
