// Replay: turns a packed run (index.json + ops.json) into a timeline, and answers "what is true at clock time t"
// for any t, in any order. Nothing here keeps state between calls; every value is a function of t and the run.
//
// Recorded fact: the order of ops, each stroke's path, width, water, pigments and pigment_added, the rejected
// candidates, wet_pixels, sim_steps, the recorded wall time t, phases and the frames.
// Staging (derived here): when the brush dips, rinses, travels and lifts, how long each op lasts on screen, and the
// panel's carry to the drying rack.

export const PIGMENTS = [
  ['quinacridone_rose', 'Quinacridone rose', '#C2416E'],
  ['cadmium_yellow', 'Cadmium yellow', '#E8B820'],
  ['french_ultramarine', 'French ultramarine', '#34449A'],
  ['cerulean_blue', 'Cerulean blue', '#2F86BC'],
  ['burnt_umber', 'Burnt umber', '#5E4130'],
  ['hookers_green', "Hooker's green", '#2F6435'],
  ['indian_red', 'Indian red', '#9A4034'],
  ['paynes_grey', "Payne's grey", '#465062'],
];
export const PIGMENT_INDEX = Object.fromEntries(PIGMENTS.map(([k], i) => [k, i]));
export const PHASE_NAMES = { washes: 'Washes', medium: 'Medium', detail: 'Detail', paint: 'Painting' };

// on-screen timing, seconds at 1x
export const T = {
  intro: 1.6,          // the brush leaves its rest over the sheet before the first op
  rinse: 1.05,         // swirl in the rinse jar
  blot: 0.35,          // touch the rinse jar rim
  dip: 0.55,           // press into a well
  slant: 0.45,         // mix on a slant
  hop: 0.42,           // travel between two stops on the palette and jars
  toSheet: 0.62,       // travel from the palette to the stroke's start
  layMin: 0.9, layMax: 2.1, layMmPerS: 190,
  lift: 0.3,           // lift off the sheet at the stroke's end
  settlePerStep: 0.14, settleMin: 0.28,
  phaseOut: 2.4, phaseBack: 2.4, phasePerStep: 0.012, phaseHoldMin: 2.4,
  finalPerStep: 0.03, finalMin: 2.2,
  pot: 1.4,            // swap brushes at the pot when a phase starts without a dry between (the learned painter)
  outro: 4.0,          // after the last op: the score sheet turns over at outro start + revealDelay
  revealDelay: 1.0, revealTurn: 1.3,
};

const clamp01 = (x) => Math.min(1, Math.max(0, x));
const smooth = (x) => { x = clamp01(x); return x * x * (3 - 2 * x); };

// Catmull-Rom through the points, resampled evenly by arc length (pixel units)
export function resamplePath(points, spacing = 1.5) {
  const P = points.length === 1 ? [points[0], points[0]] : points;
  const dense = [];
  const get = (i) => P[Math.max(0, Math.min(P.length - 1, i))];
  for (let i = 0; i < P.length - 1; i++) {
    const p0 = get(i - 1), p1 = get(i), p2 = get(i + 1), p3 = get(i + 2);
    const n = 24;
    for (let k = 0; k < n; k++) {
      const s = k / n, s2 = s * s, s3 = s2 * s;
      dense.push([0, 1].map((c) => 0.5 * (2 * p1[c] + (-p0[c] + p2[c]) * s + (2 * p0[c] - 5 * p1[c] + 4 * p2[c] - p3[c]) * s2
        + (-p0[c] + 3 * p1[c] - 3 * p2[c] + p3[c]) * s3)));
    }
  }
  dense.push([...P[P.length - 1]]);
  const cum = [0];
  for (let i = 1; i < dense.length; i++) cum.push(cum[i - 1] + Math.hypot(dense[i][0] - dense[i - 1][0], dense[i][1] - dense[i - 1][1]));
  const total = cum[cum.length - 1];
  const n = Math.max(2, Math.ceil(total / spacing) + 1);
  const out = [];
  let j = 0;
  for (let k = 0; k < n; k++) {
    const d = (k / (n - 1)) * total;
    while (j < cum.length - 2 && cum[j + 1] < d) j++;
    const seg = cum[j + 1] - cum[j] || 1;
    const a = (d - cum[j]) / seg;
    out.push([dense[j][0] + (dense[j + 1][0] - dense[j][0]) * a, dense[j][1] + (dense[j + 1][1] - dense[j][1]) * a]);
  }
  return { pts: out, length: total };
}

export function pointAt(rs, p) {
  const f = clamp01(p) * (rs.pts.length - 1);
  const i = Math.min(rs.pts.length - 2, Math.floor(f)), a = f - i;
  const A = rs.pts[i], B = rs.pts[i + 1];
  return [A[0] + (B[0] - A[0]) * a, A[1] + (B[1] - A[1]) * a];
}

export function mixColour(pigments) {
  let r = 0, g = 0, b = 0, w = 0;
  for (const [k, c] of Object.entries(pigments ?? {})) {
    const i = PIGMENT_INDEX[k];
    if (i === undefined) continue;
    const hex = PIGMENTS[i][2];
    r += parseInt(hex.slice(1, 3), 16) * c; g += parseInt(hex.slice(3, 5), 16) * c; b += parseInt(hex.slice(5, 7), 16) * c; w += c;
  }
  if (!w) return '#777777';
  const h = (v) => Math.round(v / w).toString(16).padStart(2, '0');
  return `#${h(r)}${h(g)}${h(b)}`;
}

const mainPigment = (pigs) => Object.entries(pigs ?? {}).sort((a, b) => b[1] - a[1])[0]?.[0] ?? null;

export class Replay {
  constructor(index, ops, base) {
    this.index = index;
    this.ops = ops;
    this.base = base;
    this.W = index.run.canvas.width;
    this.H = index.run.canvas.height;
    this.pxMm = 305 / this.W; // the sheet is 305 mm wide
    this.frames = index.frames;
    this.phaseCount = ops[0]?.phase_count ?? index.phases.length;
    this.build();
  }

  build() {
    const segs = [];
    let t = T.intro;
    let prevMain = null, prevT = 0, prevSteps = 0, prevBrush = null;
    let strokeNo = 0, rinses = 0;
    const cumPig = new Array(PIGMENTS.length).fill(0);
    let cumWater = 0;
    const brushFor = (phase) => (phase === 'washes' ? 'wash' : phase === 'medium' ? 'round12' : 'round4');
    for (const o of this.ops) {
      const kind = o.op.op === 'stroke' ? 'stroke' : (o.dry_kind ?? 'settle');
      const recDt = Math.max(0, o.t - prevT);
      const s = { i: o.index, o, kind, start: t, stepsBefore: prevSteps, stepsAfter: o.sim_steps, tBefore: prevT, tAfter: o.t,
        phase: o.phase ?? 'paint', phaseIndex: o.phase_index ?? 0, brush: brushFor(o.phase) };
      if (kind === 'stroke') {
        strokeNo++;
        const pigs = Object.entries(o.op.pigments ?? {}).sort((a, b) => b[1] - a[1]);
        const main = pigs[0]?.[0] ?? null;
        const rinse = prevMain !== null && main !== prevMain;
        if (rinse) rinses++;
        const rs = resamplePath(o.op.path);
        const lay = Math.min(T.layMax, Math.max(T.layMin, (rs.length * this.pxMm) / T.layMmPerS));
        const stops = [];
        let c = t + T.toSheet; // travel from the rest above the last stroke
        // a new phase with no phase dry before it: the brush is swapped at the pot before loading
        if (prevBrush !== null && prevBrush !== s.brush) {
          stops.push({ at: 'pot', a: c, b: c + T.pot });
          s.brushChange = { from: prevBrush, to: s.brush, a: c, b: c + T.pot };
          c += T.pot + T.hop;
        }
        if (rinse) {
          stops.push({ at: 'rinse', a: c, b: c + T.rinse }); c += T.rinse + T.hop * 0.6;
          stops.push({ at: 'blot', a: c, b: c + T.blot }); c += T.blot + T.hop;
        }
        pigs.forEach(([k], j) => {
          stops.push({ at: 'well', well: PIGMENT_INDEX[k], a: c, b: c + T.dip });
          c += T.dip + T.hop * 0.7;
        });
        if (pigs.length > 1) {
          stops.push({ at: 'slant', slant: PIGMENT_INDEX[main] < 4 ? 0 : 1, a: c, b: c + T.slant });
          c += T.slant + T.hop * 0.4;
        }
        // the search ran while the brush loaded; the rest of the recorded search time is a hover above the sheet
        const need = c - t + T.toSheet + lay + T.lift;
        const hover = Math.max(0, recDt - need);
        const searchEnd = c + hover; // ghosts gone just before the travel down to the sheet ends
        c += hover;
        const layA = c + T.toSheet;
        stops.push({ at: 'sheet', a: layA, b: layA + lay });
        s.dur = layA + lay + T.lift - t;
        Object.assign(s, { pigs, main, rinse, rs, lay: [layA, layA + lay], search: [t, searchEnd + T.toSheet * 0.55], stops,
          strokeNo, rinsesAfter: rinses, colour: mixColour(o.op.pigments) });
        s.rejected = (o.rejected ?? []).map((r) => ({ ...r, rs: resamplePath(r.path), colour: mixColour(r.pigments) }));
        // cumulative amounts at the end of this stroke
        for (const [k, v] of Object.entries(o.pigment_added ?? {})) cumPig[PIGMENT_INDEX[k]] += v ?? 0;
        cumWater += o.op.water ?? 0;
        prevMain = main;
        prevBrush = s.brush;
      } else if (kind === 'phase') {
        prevBrush = null; // the phase dry's own brush change covers the swap
        const hold = Math.max(T.phaseHoldMin, o.op.steps * T.phasePerStep);
        s.dur = T.phaseOut + hold + T.phaseBack;
        s.carry = [t, t + T.phaseOut, t + T.phaseOut + hold, t + s.dur];
      } else if (kind === 'final') {
        s.dur = Math.max(T.finalMin, o.op.steps * T.finalPerStep);
      } else {
        s.dur = Math.max(T.settleMin, (o.op.steps ?? 1) * T.settlePerStep);
      }
      s.end = s.start + s.dur;
      s.cumPig = cumPig.slice();
      s.cumWater = cumWater;
      s.rinses = rinses;
      s.strokesDone = strokeNo;
      segs.push(s);
      t = s.end;
      prevT = o.t; prevSteps = o.sim_steps;
    }
    this.segs = segs;
    this.strokeSegs = segs.filter((s) => s.kind === 'stroke');
    this.endOps = t;
    this.revealAt = t + T.revealDelay;
    this.duration = t + T.outro;
    this.totals = { pig: cumPig.slice(), water: cumWater, rinses, strokes: strokeNo };
    // the next stroke for each segment, for the brush's hover between strokes
    let next = null;
    for (let k = segs.length - 1; k >= 0; k--) { segs[k].nextStroke = next; if (segs[k].kind === 'stroke') next = segs[k]; }
    let prev = null;
    for (const s of segs) { s.prevStroke = prev; if (s.kind === 'stroke') prev = s; }
    // brush changes happen inside the phase dries
    for (const s of segs) {
      if (s.kind !== 'phase') continue;
      const nb = s.nextStroke?.brush;
      if (nb && nb !== s.brush) s.brushChange = { from: s.brush, to: nb, a: s.start + 0.7, b: s.end - 0.9 };
    }
  }

  segAt(t) {
    const S = this.segs;
    if (t < S[0].start) return null;
    let lo = 0, hi = S.length - 1;
    while (lo < hi) { const m = (lo + hi + 1) >> 1; if (S[m].start <= t) lo = m; else hi = m - 1; }
    return S[lo];
  }

  // simulator step shown at t, fractional during dries
  stepAt(t) {
    const s = this.segAt(t);
    if (!s) return 0;
    if (t >= s.end) return s.stepsAfter;
    if (s.kind === 'stroke') return t >= s.lay[1] ? s.stepsAfter : s.stepsBefore;
    let p;
    if (s.kind === 'phase') p = clamp01((t - s.carry[1]) / (s.carry[2] - s.carry[1]));
    else p = clamp01((t - s.start) / s.dur);
    return s.stepsBefore + (s.stepsAfter - s.stepsBefore) * p;
  }

  // the two frames to blend, the global blend, and the strokes whose footprint reveals frame B
  sheetAt(t) {
    const step = this.stepAt(t);
    const F = this.frames;
    let a = 0;
    for (let k = 0; k < F.length; k++) if (F[k].step <= step + 1e-6) a = k;
    if (t >= this.endOps) a = F.length - 1;
    const b = Math.min(F.length - 1, a + 1);
    let global = 0;
    const s = this.segAt(t);
    if (s && s.kind !== 'stroke' && b !== a && t < this.endOps) {
      global = clamp01((step - F[a].step) / Math.max(1e-6, F[b].step - F[a].step));
    }
    // strokes after frame a's op, up to now, reveal frame b
    const fromOp = F[a].op ?? -1;
    const reveal = [];
    if (b !== a) {
      for (const st of this.strokeSegs) {
        if (st.i <= fromOp) continue;
        if (st.lay[0] > t) break;
        if (st.i > (F[b].op ?? Infinity)) break;
        reveal.push({ seg: st, p: clamp01((t - st.lay[0]) / (st.lay[1] - st.lay[0])) });
      }
    }
    return { a, b, global, reveal };
  }

  ghostsAt(t) {
    const s = this.segAt(t);
    const st = s?.kind === 'stroke' ? s : null;
    if (!st || t < st.search[0] || t > st.search[1]) return [];
    const [A, B] = st.search, D = B - A, n = st.rejected.length;
    const out = [];
    st.rejected.forEach((r, k) => {
      // appear in the first quarter, best first; flicker through the search; vanish worst first in the last
      // third, the runner-up last, just before the kept stroke lands
      const on = A + D * 0.25 * (k / Math.max(1, n));
      const off = A + D * (0.68 + 0.3 * (n - 1 - k) / Math.max(1, n - 1));
      if (t < on || t > off) return;
      const life = clamp01((t - on) / 0.12) * clamp01((off - t) / 0.12);
      const flick = 0.7 + 0.3 * Math.sin((t - on) * 23 + k * 1.7);
      out.push({ r, alpha: life * flick, rank: k + 1 });
    });
    return out;
  }

  // cumulative values at t, each ramped over the moment it really happens on screen
  amountsAt(t) {
    const s = this.segAt(t);
    const pig = new Array(PIGMENTS.length).fill(0), well = new Array(PIGMENTS.length).fill(0);
    let water = 0, rinses = 0, strokes = 0, strokesStarted = 0;
    if (s) {
      const prev = this.segs[s.i - 1];
      const base = prev ? prev.cumPig : pig;
      for (let k = 0; k < pig.length; k++) { pig[k] = base[k]; well[k] = base[k]; }
      water = prev ? prev.cumWater : 0;
      rinses = prev ? prev.rinses : 0;
      strokes = prev ? prev.strokesDone : 0;
      strokesStarted = strokes;
      if (s.kind === 'stroke') {
        const layP = clamp01((t - s.lay[0]) / (s.lay[1] - s.lay[0]));
        for (const [k, v] of Object.entries(s.o.pigment_added ?? {})) {
          const i = PIGMENT_INDEX[k];
          pig[i] += (v ?? 0) * layP;
          const dip = s.stops.find((d) => d.at === 'well' && d.well === i);
          well[i] += (v ?? 0) * (dip ? smooth((t - dip.a) / (dip.b - dip.a)) : layP);
        }
        const first = s.stops[0];
        water += (s.o.op.water ?? 0) * smooth((t - first.a) / Math.max(0.01, s.lay[0] - first.a));
        const r = s.stops.find((d) => d.at === 'rinse');
        if (r) rinses += smooth((t - r.a) / (r.b - r.a));
        if (layP >= 1) strokes += 1;
        if (t >= s.lay[0]) strokesStarted += 1;
      }
    }
    return { pig, well, water, rinses, strokes, strokesStarted };
  }

  // wet area and recorded wall time, interpolated across the op
  recordAt(t) {
    const s = this.segAt(t);
    if (!s) return { wet: 0, wall: 0, steps: 0 };
    const prev = this.segs[s.i - 1];
    const w0 = prev?.o.wet_pixels ?? 0, w1 = s.o.wet_pixels ?? w0;
    let p = clamp01((t - s.start) / s.dur);
    if (s.kind === 'stroke') p = clamp01((t - s.lay[0]) / (s.lay[1] - s.lay[0]));
    return { wet: w0 + (w1 - w0) * p, wall: s.tBefore + (s.tAfter - s.tBefore) * clamp01((t - s.start) / s.dur), steps: this.stepAt(t) };
  }

  // the panel's place: 0 on the board, 1 in the rack, with the carry phase
  panelAt(t) {
    const s = this.segAt(t);
    if (!s || s.kind !== 'phase' || t >= s.end) return { out: 0, stage: 'board' };
    const [a, b, c, d] = s.carry;
    if (t < b) return { out: (t - a) / (b - a), stage: 'out' };
    if (t < c) return { out: 1, stage: 'rack', dry: (t - b) / (c - b) };
    return { out: 1 - (t - c) / (d - c), stage: 'back' };
  }

  // phase and stroke counters for the painting card
  phaseAt(t) {
    const s = this.segAt(t) ?? this.segs[0];
    return { name: s.phase, label: PHASE_NAMES[s.phase] ?? s.phase, index: s.phaseIndex, count: this.phaseCount };
  }

  // times when stage events happen, for the tour
  events() {
    const st = this.strokeSegs;
    const firstRinse = st.find((s) => s.rinse);
    const twoPig = st.find((s) => s.pigs.length > 1);
    const phase = this.segs.find((s) => s.kind === 'phase');
    return {
      prompt: T.intro * 0.5,
      calibration: T.intro * 0.5,
      load: twoPig ? twoPig.stops.find((d) => d.at === 'well').a : st[0].start,
      search: st[1] ? st[1].search[1] - 0.6 : st[0].search[1],
      stroke: st[2] ? st[2].lay[0] : st[0].lay[0],
      rinse: firstRinse ? firstRinse.stops[0].a : st[0].start,
      dry: phase ? phase.carry[1] : this.endOps - 1,
      reveal: this.revealAt,
    };
  }
}

export async function loadRun(base) {
  const index = await (await fetch(`${base}/index.json`)).json();
  const ops = await (await fetch(`${base}/ops.json`)).json();
  return new Replay(index, ops, base);
}
