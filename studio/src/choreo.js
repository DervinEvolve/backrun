// Brush choreography (staging): where the brush tip is at clock time t and which way the brush points. Built once
// from the replay's stops; every pose is a pure function of t. Stops come from the ledger (which wells, whether the
// main pigment changed, the stroke path); the moves between them are staging.
import * as THREE from 'three';
import { pointAt, T } from './replay.js';

const clamp01 = (x) => Math.min(1, Math.max(0, x));
const ease = (x) => { x = clamp01(x); return x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2; };
const UP = new THREE.Vector3(0, 1, 0);

// A: anchors, functions of t returning world points and directions:
//   sheet(x, y) -> { p, n } ; cakeTop(i, t) -> Vector3 ; slant(j) ; rinse(t) -> { surface, centre } ; rim() ; pot() ; park()
//   lean(p, amount) -> unit vector from grip to tip
export function buildChoreo(replay, A) {
  const keys = [];
  const push = (k) => keys.push(k);
  // park above the palette at the start
  push({ kind: 'park', a: 0, b: T.intro * 0.35, inn: 0, out: 0, pose: () => A.park() });

  for (const s of replay.segs) {
    if (s.kind === 'stroke') {
      for (const st of s.stops) {
        if (st.at === 'rinse') {
          push({ kind: 'rinse', a: st.a, b: st.b, inn: 0, out: 0, above: 0.075, seg: s, pose: (t) => {
            const r = A.rinse(t);
            const u = (t - st.a) / (st.b - st.a);
            const depth = 0.012 * Math.sin(Math.PI * clamp01(u * 1.25 - 0.12)) ** 0.5;
            const ang = (t - st.a) * Math.PI * 4.2;
            const rad = 0.011 * Math.sin(Math.PI * clamp01(u));
            const p = new THREE.Vector3(r.centre.x + Math.cos(ang) * rad, r.surface - depth + 0.012 * (1 - Math.sin(Math.PI * clamp01(u))), r.centre.z + Math.sin(ang) * rad);
            return { p, want: A.lean(p, 0.12), contact: depth > 0.004 ? 'rinse' : null };
          } });
        } else if (st.at === 'pot') {
          push({ kind: 'pot', a: st.a, b: st.b, inn: 0, out: 0, above: 0.04, seg: s, pose: (t) => {
            const p = A.pot().clone();
            const u = (t - st.a) / (st.b - st.a);
            p.y += 0.02 * (1 - Math.sin(Math.PI * clamp01(u)));
            return { p, want: A.lean(p, 0.05), contact: null };
          } });
        } else if (st.at === 'blot') {
          push({ kind: 'blot', a: st.a, b: st.b, inn: 0, out: 0, above: 0.02, seg: s, pose: (t) => {
            const u = (t - st.a) / (st.b - st.a);
            const p = A.rim().clone();
            p.y += 0.004 + 0.012 * (1 - Math.sin(Math.PI * clamp01(u)));
            return { p, want: A.lean(p, 0.35), contact: u > 0.25 && u < 0.75 ? 'rim' : null };
          } });
        } else if (st.at === 'well') {
          push({ kind: 'well', well: st.well, a: st.a, b: st.b, inn: 0, out: 0, above: 0.035, seg: s, pose: (t) => {
            const u = (t - st.a) / (st.b - st.a);
            const top = A.cakeTop(st.well, t);
            const press = Math.sin(Math.PI * clamp01(u));
            const ang = (t - st.a) * Math.PI * 5;
            const p = new THREE.Vector3(top.x + Math.cos(ang) * 0.003 * press, top.y + 0.0004 + 0.016 * (1 - press) ** 2, top.z + Math.sin(ang) * 0.003 * press);
            return { p, want: A.lean(p, 0.3), contact: press > 0.9 ? `well:${st.well}` : null };
          } });
        } else if (st.at === 'slant') {
          push({ kind: 'slant', a: st.a, b: st.b, inn: 0, out: 0, above: 0.03, seg: s, pose: (t) => {
            const u = (t - st.a) / (st.b - st.a);
            const c = A.slant(st.slant);
            const press = Math.sin(Math.PI * clamp01(u));
            const p = new THREE.Vector3(c.x + 0.012 * Math.sin(u * Math.PI * 3), c.y + 0.0004 + 0.014 * (1 - press) ** 2, c.z);
            return { p, want: A.lean(p, 0.35), contact: press > 0.9 ? 'slant' : null };
          } });
        } else if (st.at === 'sheet') {
          const rs = s.rs;
          push({ kind: 'sheet', a: st.a, b: st.b, inn: 0.16, out: T.lift, above: 0.03, seg: s, pose: (t) => {
            const u = (t - st.a) / (st.b - st.a);
            const lift = u < 0 ? (-u * (st.b - st.a)) / 0.16 : u > 1 ? ((u - 1) * (st.b - st.a)) / T.lift : 0;
            // even speed with a short ease at both ends
            const pu = clamp01(u);
            const pp = pu < 0.08 ? pu * pu / 0.16 : pu > 0.92 ? 1 - (1 - pu) * (1 - pu) / 0.16 : pu;
            const [x, y] = pointAt(rs, pp);
            const sp = A.sheet(x, y);
            const p = sp.p.clone().addScaledVector(sp.n, 0.03 * ease(clamp01(lift)));
            return { p, want: A.lean(p, s.brush === 'wash' ? 0.22 : 0.42, sp.n), contact: lift <= 0 ? 'sheet' : null, onSheet: { x, y } };
          } });
          // rest above the stroke's end until the next stop
          const [ex, ey] = pointAt(rs, 1);
          push({ kind: 'rest', a: st.b + T.lift, b: null, inn: 0, out: 0, seg: s, pose: (t) => {
            const sp = A.sheet(ex, ey);
            const p = sp.p.clone().addScaledVector(sp.n, 0.03);
            p.x += 0.0015 * Math.sin(t * 1.3); p.z += 0.0012 * Math.sin(t * 0.9 + 1);
            return { p, want: A.lean(p, s.brush === 'wash' ? 0.22 : 0.42, sp.n), contact: null };
          } });
        }
      }
    } else if (s.kind === 'phase' && s.brushChange) {
      const bc = s.brushChange;
      push({ kind: 'pot', a: bc.a, b: bc.b, inn: 0, out: 0, above: 0.04, seg: s, pose: (t) => {
        const p = A.pot().clone();
        const u = (t - bc.a) / (bc.b - bc.a);
        p.y += 0.02 * (1 - Math.sin(Math.PI * clamp01(u)));
        return { p, want: A.lean(p, 0.05), contact: null };
      } });
    }
  }
  push({ kind: 'park', a: replay.endOps + 0.8, b: replay.duration + 1, inn: 0, out: 0, pose: () => A.park() });
  // open-ended rests end where the next key's approach starts
  for (let k = 0; k < keys.length; k++) {
    if (keys[k].b === null) keys[k].b = Math.max(keys[k].a, keys[k + 1].a - (keys[k + 1].inn ?? 0) - 0.5);
  }

  const starts = keys.map((k) => k.a - k.inn);
  function find(t) {
    let lo = 0, hi = keys.length - 1;
    while (lo < hi) { const m = (lo + hi + 1) >> 1; if (starts[m] <= t) lo = m; else hi = m - 1; }
    return lo;
  }
  const tmp = new THREE.Vector3();
  function pose(t) {
    const k = find(t), K = keys[k];
    if (t <= K.b + K.out || k === keys.length - 1) {
      const r = K.pose(Math.min(t, K.b + K.out));
      return { ...r, key: K };
    }
    // travel from K's last pose to the next key's first pose, on an arc
    const N = keys[k + 1];
    const t0 = K.b + K.out, t1 = N.a - N.inn;
    const u = ease((t - t0) / Math.max(1e-3, t1 - t0));
    const P0 = K.pose(t0), P1 = N.pose(t1);
    const d = P0.p.distanceTo(P1.p);
    const lift = Math.min(0.11, 0.035 + 0.16 * d) + Math.max(K.above ?? 0, N.above ?? 0) * 0.6;
    const p = tmp.lerpVectors(P0.p, P1.p, u).clone();
    // rise early and fall late so the tip clears jar rims and the board lip
    const arc = Math.sin(Math.PI * Math.min(1, Math.max(0, u * 1.0))) ** 0.6;
    p.addScaledVector(UP, lift * arc);
    const want = P0.want.clone().lerp(P1.want, u).normalize();
    return { p, want, contact: null, key: null, travel: true };
  }
  return { keys, pose };
}
