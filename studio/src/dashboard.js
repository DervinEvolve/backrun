// The dashboard: painting card, KPI strip, pigment legend, score sheet, timeline strip and stage bar.
// Every number is read from the replay at the current clock time.
import { PIGMENTS, PHASE_NAMES } from './replay.js';

const $ = (id) => document.getElementById(id);
export const SPEEDS = [0.25, 1, 2, 5, 10, 50, 150];

// status icons: a different shape per state (circle working, square stable, triangle blocked)
const ICON = {
  working: '<svg viewBox="0 0 14 14"><circle cx="7" cy="7" r="5.5" fill="#FFB74D"/></svg>',
  stable: '<svg viewBox="0 0 14 14"><rect x="2" y="2" width="10" height="10" rx="1.5" fill="#64B5F6"/></svg>',
  healthy: '<svg viewBox="0 0 14 14"><path d="M2 7.5l3.2 3.2L12 3.8" stroke="#7CB342" stroke-width="2.4" fill="none" stroke-linecap="round"/></svg>',
  blocked: '<svg viewBox="0 0 14 14"><path d="M7 1.5l6 11H1z" fill="#E57373"/></svg>',
};

const fmtAmount = (v) => (v >= 10000 ? `${(v / 1000).toFixed(1)}k` : v >= 1000 ? `${(v / 1000).toFixed(2)}k` : v.toFixed(0));
const fmtInt = (v) => Math.round(v).toLocaleString('en-US');

export function createDashboard(replay, stages, h) {
  const r = replay;
  const idx = r.index;
  const run = idx.run;
  const name = run.target_path?.startsWith('synthetic:') ? run.target_path.split(':')[1] : idx.name;
  $('p-name').textContent = name.charAt(0).toUpperCase() + name.slice(1);
  $('p-sub').textContent = `Painter ${run.painter ?? 'unknown'} · seed ${run.seed} · ${run.canvas.width} x ${run.canvas.height} px`;
  $('p-target').src = `${r.base}/target.png`;
  $('k-stepsof').textContent = `of ${fmtInt(idx.sim_steps)}`;
  $('k-wallof').textContent = `of ${(run.timings?.paint_s ?? 0).toFixed(1)} s painting`;
  const skipped = run.painter_stats?.slots_skipped ?? 0;
  $('k-skipped').textContent = `of ${r.totals.strokes}${skipped ? `, ${skipped} slot skipped in the run` : ''}`;

  // legend
  const maxAmt = Math.max(...r.totals.pig, 1);
  const strokeCounts = PIGMENTS.map(([k]) => r.strokeSegs.map((s) => (k in (s.o.op.pigments ?? {}) ? s.lay[1] : null)).filter((x) => x !== null));
  const rows = PIGMENTS.map(([k, label, col], i) => {
    const b = document.createElement('button');
    b.className = 'lrow';
    b.innerHTML = `<span class="sw" style="background:${col}"></span><span class="n">${label}</span><span class="c">0</span>`
      + `<span class="bar"><i style="background:${col};width:0%"></i></span><span class="a">0</span>`;
    b.title = `${label}: strokes using it and pigment added. Click to look at its well.`;
    b.addEventListener('click', () => h.onLegend(i));
    $('legend-rows').appendChild(b);
    return { b, c: b.querySelector('.c'), bar: b.querySelector('.bar i'), a: b.querySelector('.a') };
  });

  // speeds and stages
  const speedBtns = SPEEDS.map((s) => {
    const b = document.createElement('button');
    b.className = 'btn'; b.textContent = `${s}x`; b.dataset.speed = s;
    b.addEventListener('click', () => h.onSpeed(s));
    $('speeds').appendChild(b);
    return b;
  });
  const stageBtns = [{ key: '0', id: 'overview', name: 'Overview' }, ...stages].map((s) => {
    const b = document.createElement('button');
    b.className = 'btn'; b.textContent = s.key === '0' ? '0' : `${s.key} ${s.name.split(' ')[0]}`;
    b.title = `${s.name} (key ${s.key})`; b.dataset.view = s.id;
    b.addEventListener('click', () => h.onStage(s.id));
    $('stages').appendChild(b);
    return b;
  });

  // scrubber: phase bands, a tick per stroke, a mark for the rack dries
  const D = r.duration;
  const pct = (t) => `${((t / D) * 100).toFixed(3)}%`;
  const bandCol = ['rgba(255,183,77,0.16)', 'rgba(124,179,66,0.16)', 'rgba(100,181,246,0.16)'];
  for (const p of idx.phases) {
    const a = r.segs[p.first_op].start, b = r.segs[p.last_op].end;
    const d = document.createElement('div');
    d.style.left = pct(a); d.style.width = pct(b - a); d.style.background = bandCol[p.index % 3];
    d.textContent = PHASE_NAMES[p.name] ?? p.name;
    $('bands').appendChild(d);
  }
  for (const s of r.strokeSegs) { const i = document.createElement('i'); i.style.left = pct(s.lay[0]); $('ticks').appendChild(i); }
  for (const s of r.segs.filter((x) => x.kind === 'phase')) { const i = document.createElement('i'); i.className = 'rack'; i.style.left = pct(s.carry[1]); $('ticks').appendChild(i); }
  const scrub = $('scrub');
  const scrubTo = (e) => { const b = scrub.getBoundingClientRect(); h.onScrub(Math.min(1, Math.max(0, (e.clientX - b.left) / b.width)) * D); };
  scrub.addEventListener('pointerdown', (e) => { scrub.setPointerCapture(e.pointerId); scrubTo(e); });
  scrub.addEventListener('pointermove', (e) => { if (e.buttons) scrubTo(e); });

  $('play').addEventListener('click', h.onPlay);
  $('step').addEventListener('click', h.onStep);
  $('tour').addEventListener('click', h.onTour);
  $('follow').addEventListener('click', h.onFollow);
  $('ghosts').addEventListener('click', h.onGhosts);
  $('help').addEventListener('click', () => { const a = $('about'); a.hidden = !a.hidden; });

  // the guess lives only in this browser
  const gkey = `backrun.guess.${idx.name}`;
  const readGuess = () => { try { const v = localStorage.getItem(gkey); return v === null ? null : Number(v); } catch { return null; } };
  const showKept = () => { const g = readGuess(); $('guess-kept').textContent = g === null ? 'No guess kept yet.' : `Kept: ${g.toFixed(2)}`; if (g !== null && !$('guess').value) $('guess').value = g; };
  $('guess-keep').addEventListener('click', () => {
    const v = Number($('guess').value);
    if (!Number.isFinite(v) || $('guess').value === '') return;
    try { localStorage.setItem(gkey, String(v)); } catch { /* private mode: the guess lasts this page only */ }
    showKept();
  });
  showKept();

  let scoresShown = false;
  function showScores(scores) {
    if (scoresShown || !scores?.final) return;
    scoresShown = true;
    const f = scores.final;
    $('score-down').hidden = true; $('score-up').hidden = false;
    $('s-total').textContent = `total ${f.total.toFixed(2)}`;
    $('s-parts').textContent = `lab_l2 ${f.lab_l2.toFixed(1)} · ssim ${f.ssim.toFixed(3)} · ${f.strokes} strokes`;
    const tot = scores.ops.map((o) => o.score?.total).filter((v) => typeof v === 'number');
    const mx = Math.max(...tot), mn = Math.min(...tot);
    const pts = tot.map((v, i) => `${((i / (tot.length - 1)) * 200).toFixed(1)},${(4 + (1 - (v - mn) / (mx - mn || 1)) * 40).toFixed(1)}`).join(' ');
    $('s-chart').innerHTML = `<polyline points="${pts}" fill="none" stroke="#F4F0E6" stroke-width="1.4" vector-effect="non-scaling-stroke"/>`
      + `<text x="2" y="10" fill="rgba(244,240,230,0.66)" font-size="8">${mx.toFixed(1)}</text>`;
    const g = readGuess();
    $('s-guess').textContent = g === null ? 'No guess was kept.' : `Your guess ${g.toFixed(2)}: off by ${Math.abs(g - f.total).toFixed(2)} (${g > f.total ? 'the painter did better' : 'the painter did worse'} than you guessed)`;
  }
  function hideScores() {
    if (!scoresShown) return;
    scoresShown = false;
    $('score-down').hidden = false; $('score-up').hidden = true;
  }

  const sheetCm2 = 30.5 * 22.9, cells = run.canvas.width * run.canvas.height;
  function update(t, ui) {
    const am = r.amountsAt(t), rec = r.recordAt(t), ph = r.phaseAt(t), seg = r.segAt(t);
    const onPaper = seg?.kind === 'stroke' && t >= seg.lay[0] && t < seg.lay[1];
    const inRack = seg?.kind === 'phase';
    const ended = t >= r.endOps;
    const st = ended ? 'healthy' : onPaper ? 'working' : inRack || seg?.kind === 'final' ? 'stable' : 'working';
    const what = ended ? 'finished' : onPaper ? 'brush on paper' : inRack ? 'drying in the rack' : seg?.kind === 'final' ? 'final dry'
      : seg?.kind === 'stroke' ? (t < seg.search[1] ? 'searching and loading' : 'moving to the sheet') : 'settling';
    $('p-phase').innerHTML = `${ICON[st]}<span>${ph.label}, phase ${ph.index + 1} of ${ph.count} · ${what}</span>`;
    const cur = seg?.kind === 'stroke' ? seg : seg?.nextStroke ?? null;
    $('p-stroke').textContent = ended ? `${r.totals.strokes} strokes laid` : `Stroke ${cur ? cur.strokeNo : r.totals.strokes} of ${r.totals.strokes}`;
    $('p-slot').textContent = seg?.kind === 'stroke' ? `${seg.o.candidates_considered ?? '?'} candidates tried, ${seg.rejected.length} best rejected shown` : `op ${seg ? seg.i + 1 : 0} of ${idx.op_count}`;
    $('k-strokes').textContent = String(am.strokes);
    $('k-wet').textContent = `${(rec.wet / cells * sheetCm2).toFixed(0)} cm²`;
    $('k-wetpct').textContent = `${((rec.wet / cells) * 100).toFixed(1)} % of the sheet`;
    $('k-steps').textContent = fmtInt(rec.steps);
    $('k-wall').textContent = `${rec.wall.toFixed(1)} s`;
    let evals = 0;
    for (const s of r.strokeSegs) { if (s.lay[0] <= t) evals += s.o.candidates_considered ?? 0; else break; }
    $('k-evals').textContent = fmtInt(evals);
    PIGMENTS.forEach((_, i) => {
      const R = rows[i];
      R.c.textContent = String(strokeCounts[i].filter((x) => x <= t).length);
      R.bar.style.width = `${((am.pig[i] / maxAmt) * 100).toFixed(1)}%`;
      R.a.textContent = fmtAmount(am.pig[i]);
      R.b.classList.toggle('active', !!seg?.pigs?.some(([k]) => k === PIGMENTS[i][0]) && !ended);
    });
    const D = r.duration;
    $('head').style.left = `${((t / D) * 100).toFixed(3)}%`;
    const mm = (x) => `${Math.floor(x / 60)}:${String(Math.floor(x % 60)).padStart(2, '0')}`;
    $('clock').textContent = `${mm(t)} / ${mm(D)} replay`;
    $('play').innerHTML = ui.playing ? '&#10074;&#10074;' : '&#9654;';
    $('play').setAttribute('aria-label', ui.playing ? 'Pause (Space)' : 'Play (Space)');
    speedBtns.forEach((b) => b.classList.toggle('on', Number(b.dataset.speed) === ui.speed));
    stageBtns.forEach((b) => b.classList.toggle('on', b.dataset.view === ui.view));
    $('tour').classList.toggle('on', ui.touring);
    $('follow').setAttribute('aria-pressed', String(ui.follow));
    $('ghosts').setAttribute('aria-pressed', String(ui.ghosts));
    if (ui.scores && t >= r.revealAt + 0.6) showScores(ui.scores); else if (t < r.revealAt + 0.6) hideScores();
  }
  return { update };
}
