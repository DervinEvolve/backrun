"""The v0 search painter: greedy, coarse to fine, no learning.

For each stroke slot the painter proposes a batch of candidate strokes,
simulates each on a copy of the current painting, scores it against the
target, and keeps the best one only if it lowers the total score. If no
candidate helps, the slot is skipped.

Candidates are built like this:

* **Position.** The centre is drawn at random with probability proportional
  to the squared, blurred colour error between the current painting and the
  target, so strokes go where the painting is most wrong. Error where the
  target is lighter than the painting is down-weighted, because watercolour
  can only darken the paper.
* **Direction.** Most candidates run along the target's isophotes (across
  its brightness gradient), the rest at a random angle.
* **Colour.** The painter looks at the target and the current painting
  under the stroke's path and picks the one or two pigments, with
  concentrations from a small grid, whose glaze over the current colours
  best matches the target in CIELAB.

Painting runs in three phases: wide wet washes, medium strokes, then fine
detail, with a ``dry`` op between phases so each layer sets before the next.

To predict what a pigment mix will look like, the painter first paints a
calibration sheet: one stripe per pigment at two concentrations on a fresh
canvas from ``sim_factory``, dried and rendered. The measured colours give a
per-pigment optical density, and mixes are predicted with a Beer-Lambert
glaze model (densities add, light is multiplied). If calibration is turned
off, the painter falls back to each pigment's representative colour, read
from ``backrun.pigments.PIGMENTS`` if possible and from a built-in table
otherwise.

The painter uses the scorer only as a black box (``backrun.score.score`` by
default, or any function with the same signature). Its own colour maths
below is a separate, deliberately independent implementation.

Everything is deterministic for a given seed and simulator.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw
from scipy import ndimage


class Sim(Protocol):
    """The simulator interface the painter relies on."""

    def apply(self, op: dict[str, Any]) -> Any: ...
    def step(self, n: int = 1) -> Any: ...
    def render(self) -> NDArray[np.uint8]: ...
    def copy(self) -> "Sim": ...


ScoreFn = Callable[..., Mapping[str, Any]]

#: Approximate thin-wash colours used when the pigment module gives nothing
#: usable. Design choices, not measurements.
FALLBACK_PIGMENTS: dict[str, tuple[int, int, int]] = {
    "quinacridone_rose": (226, 80, 130),
    "cadmium_yellow": (250, 205, 40),
    "french_ultramarine": (60, 80, 180),
    "cerulean_blue": (60, 150, 205),
    "burnt_umber": (115, 80, 55),
    "hookers_green": (50, 110, 60),
    "indian_red": (165, 65, 55),
    "paynes_grey": (70, 80, 95),
}

#: Attribute names tried, in order, to find a pigment's representative colour.
_COLOUR_ATTRS = ("rgb_thin", "thin_rgb", "on_white", "rgb", "color", "colour", "masstone")

#: Concentration grid searched when matching colours.
CONCENTRATIONS: tuple[float, ...] = (0.08, 0.15, 0.25, 0.4, 0.6, 0.85)

#: Pixels sampled under a stroke when matching its colour.
_MATCH_SAMPLES = 32

#: Probabilities of using the best, second and third colour match, for variety.
_MIX_PICK = np.array([0.6, 0.25, 0.15])

#: The two concentrations painted on the calibration sheet.
CALIBRATION_CONCS: tuple[float, float] = (0.25, 0.8)


@dataclass(frozen=True)
class Phase:
    """Stroke settings for one phase of the painting.

    Widths and lengths are fractions: widths of the canvas's shorter side,
    lengths of its longer side.
    """

    name: str
    share: float
    width: tuple[float, float]
    length: tuple[float, float]
    water: tuple[float, float]
    points: tuple[int, int]
    settle_steps: int


PHASES: tuple[Phase, ...] = (
    Phase("washes", 0.2, (0.18, 0.30), (0.5, 1.0), (0.75, 0.95), (3, 4), 2),
    Phase("medium", 0.4, (0.06, 0.12), (0.15, 0.4), (0.5, 0.75), (3, 4), 3),
    Phase("detail", 0.4, (0.015, 0.04), (0.04, 0.15), (0.3, 0.55), (2, 3), 3),
)


# ---------------------------------------------------------------------------
# Colour maths (independent of the scorer on purpose).
# ---------------------------------------------------------------------------

_M = np.array(
    [[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]
)
_WHITE = np.array([0.9505, 1.0, 1.089])


def rgb01_to_lab(rgb: NDArray) -> NDArray[np.float64]:
    """Convert sRGB in 0..1, shape (..., 3), to CIELAB (D65).

    Float32 input stays float32 (used for the fast mix search); anything else
    is computed in float64.
    """
    arr = np.asarray(rgb)
    c = np.clip(arr if arr.dtype == np.float32 else arr.astype(np.float64), 0.0, 1.0)
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    t = (lin @ _M.T.astype(c.dtype)) / _WHITE.astype(c.dtype)
    f = np.where(t > 0.008856, np.cbrt(t), 7.787 * t + 16.0 / 116.0)
    return np.stack(
        [116.0 * f[..., 1] - 16.0, 500.0 * (f[..., 0] - f[..., 1]), 200.0 * (f[..., 1] - f[..., 2])],
        axis=-1,
    )


def _colour_to_01(value: Any) -> NDArray[np.float64] | None:
    """Interpret a colour as RGB in 0..1; accepts 0..1 floats or 0..255 values."""
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if arr.size < 3 or not np.all(np.isfinite(arr[:3])):
        return None
    arr = arr[:3]
    if arr.max() > 1.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def discover_pigments() -> dict[str, NDArray[np.float64]]:
    """Return {pigment name: representative RGB in 0..1}.

    Reads ``backrun.pigments.PIGMENTS`` if it can be imported. It may be a
    dict keyed by name or a sequence of objects with a ``name`` attribute;
    the colour is taken from the first attribute or key in a short list of
    likely names. Pigments without a usable colour borrow the built-in
    table's value, and if nothing at all is found the built-in table is used.
    """
    found: dict[str, NDArray[np.float64]] = {}
    try:
        from backrun import pigments as _pig  # imported lazily: optional

        table = getattr(_pig, "PIGMENTS", None)
    except Exception:
        table = None
    items: list[tuple[str, Any]] = []
    if isinstance(table, Mapping):
        items = [(str(k), v) for k, v in table.items()]
    elif isinstance(table, Sequence):
        for obj in table:
            name = getattr(obj, "name", None)
            if name is None and isinstance(obj, Mapping):
                name = obj.get("name")
            if name is not None:
                items.append((str(name), obj))
    for name, obj in items:
        colour = None
        for attr in _COLOUR_ATTRS:
            raw = obj.get(attr) if isinstance(obj, Mapping) else getattr(obj, attr, None)
            if raw is not None:
                colour = _colour_to_01(raw)
                if colour is not None:
                    break
        if colour is None and name in FALLBACK_PIGMENTS:
            colour = np.asarray(FALLBACK_PIGMENTS[name], dtype=np.float64) / 255.0
        if colour is not None:
            found[name] = colour
    if not found:
        found = {k: np.asarray(v, dtype=np.float64) / 255.0 for k, v in FALLBACK_PIGMENTS.items()}
    return found


class MixModel:
    """Predicts the colour of a pigment glaze and finds the best mix.

    Each pigment has an optical density per RGB channel as a function of
    concentration, stored at the grid ``CONCENTRATIONS``. A glaze of one or
    two pigments over a colour ``u`` is predicted as ``u * exp(-(d1 + d2))``.
    """

    def __init__(self, names: Sequence[str], density: NDArray[np.float64]) -> None:
        self.names = list(names)
        self.density = np.asarray(density, dtype=np.float64)  # (P, C, 3)
        p, c, _ = self.density.shape
        combos: list[dict[str, float]] = []
        dens: list[NDArray[np.float64]] = []
        for i in range(p):
            for ci in range(c):
                combos.append({self.names[i]: CONCENTRATIONS[ci]})
                dens.append(self.density[i, ci])
        for i, j in combinations(range(p), 2):
            for ci in range(c):
                for cj in range(c):
                    combos.append({self.names[i]: CONCENTRATIONS[ci], self.names[j]: CONCENTRATIONS[cj]})
                    dens.append(self.density[i, ci] + self.density[j, cj])
        self.combos = combos
        self.transmit = np.exp(-np.stack(dens))  # (M, 3)
        self.transmit32 = self.transmit.astype(np.float32)

    @classmethod
    def from_colours(cls, colours: Mapping[str, NDArray], paper: NDArray, gain: float = 1.5) -> "MixModel":
        """Build a model from representative colours, assuming density is linear in concentration."""
        names = list(colours)
        paper01 = np.clip(np.asarray(paper, dtype=np.float64), 1e-3, 1.0)
        unit = np.stack(
            [-np.log(np.clip(np.asarray(colours[n]) / paper01, 1e-3, 1.0)) for n in names]
        )  # (P, 3) density at concentration 1 / gain
        conc = np.asarray(CONCENTRATIONS)[None, :, None]
        return cls(names, unit[:, None, :] * conc * gain)

    @classmethod
    def from_calibration(
        cls, names: Sequence[str], paper: NDArray, measured: NDArray
    ) -> "MixModel":
        """Build a model from measured colours at ``CALIBRATION_CONCS``.

        ``measured`` has shape (P, 2, 3), RGB in 0..1. Density is
        interpolated piecewise-linearly through zero and extended past the
        last point with the last slope.
        """
        paper01 = np.clip(np.asarray(paper, dtype=np.float64), 1e-3, 1.0)
        d = -np.log(np.clip(np.asarray(measured) / paper01, 1e-3, 1.0))  # (P, 2, 3)
        d = np.maximum.accumulate(np.maximum(d, 0.0), axis=1)
        xs = np.array([0.0, *CALIBRATION_CONCS])
        out = np.empty((len(names), len(CONCENTRATIONS), 3))
        for pi in range(len(names)):
            for ch in range(3):
                ys = np.array([0.0, d[pi, 0, ch], d[pi, 1, ch]])
                slope = (ys[2] - ys[1]) / (xs[2] - xs[1])
                for ci, c in enumerate(CONCENTRATIONS):
                    out[pi, ci, ch] = np.interp(c, xs, ys) if c <= xs[-1] else ys[2] + slope * (c - xs[2])
        return cls(names, out)

    def ranked(self, under: NDArray, target_lab: NDArray, top: int = 3) -> list[int]:
        """Indices of the ``top`` mixes whose glaze over ``under`` best matches.

        ``under`` is (N, 3) RGB in 0..1 sampled under the stroke, and
        ``target_lab`` the target's Lab at the same pixels.
        """
        pred = under[None, :, :].astype(np.float32) * self.transmit32[:, None, :]  # (M, N, 3)
        err = rgb01_to_lab(pred) - target_lab[None]
        cost = np.mean(np.sum(err * err, axis=-1), axis=1)
        order = np.argsort(cost, kind="stable")
        return [int(i) for i in order[:top]]


# ---------------------------------------------------------------------------
# Geometry.
# ---------------------------------------------------------------------------


def catmull_rom(points: NDArray, per_segment: int = 12) -> NDArray[np.float64]:
    """Sample a uniform Catmull-Rom spline through ``points`` (N, 2)."""
    p = np.asarray(points, dtype=np.float64)
    if len(p) < 2:
        return p.copy()
    ext = np.vstack([p[0], p, p[-1]])
    t = np.linspace(0.0, 1.0, per_segment, endpoint=False)[:, None]
    t2, t3 = t * t, t * t * t
    out = []
    for i in range(len(p) - 1):
        p0, p1, p2, p3 = ext[i], ext[i + 1], ext[i + 2], ext[i + 3]
        out.append(
            0.5
            * (
                2 * p1
                + (-p0 + p2) * t
                + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                + (-p0 + 3 * p1 - 3 * p2 + p3) * t3
            )
        )
    out.append(p[-1:])
    return np.vstack(out)


def stroke_mask(path: NDArray, width: float, w: int, h: int) -> NDArray[np.bool_]:
    """Boolean mask of the pixels a stroke of ``width`` along ``path`` covers."""
    im = Image.new("L", (w, h), 0)
    pts = [tuple(map(float, q)) for q in catmull_rom(path)]
    ImageDraw.Draw(im).line(pts, fill=255, width=max(1, int(round(width))), joint="curve")
    return np.asarray(im) > 127


# ---------------------------------------------------------------------------
# The painter.
# ---------------------------------------------------------------------------


def _calibrate(
    sim_factory: Callable[[], Sim], names: Sequence[str], w: int, h: int, dry_steps: int
) -> tuple[NDArray[np.float64], MixModel] | None:
    """Paint one stripe per pigment at two concentrations and measure them.

    Returns (paper colour, mix model), or ``None`` if the sheet is too small
    to measure.
    """
    sim = sim_factory()
    paper_img = sim.render().astype(np.float64) / 255.0
    paper = np.median(paper_img.reshape(-1, 3), axis=0)
    n = len(names)
    col_w = w / n
    sw = max(2.0, min(col_w * 0.55, h * 0.2))
    if col_w < 4 or h < 16:
        return None
    rows = [(0.06 * h, 0.44 * h), (0.56 * h, 0.94 * h)]
    for i, name in enumerate(names):
        x = (i + 0.5) * col_w
        for (y0, y1), conc in zip(rows, CALIBRATION_CONCS):
            sim.apply(
                {
                    "op": "stroke",
                    "path": [[round(x, 2), round(y0, 2)], [round(x, 2), round(y1, 2)]],
                    "width": round(sw, 2),
                    "water": 0.7,
                    "pigments": {name: conc},
                }
            )
    _dry_until_set(sim, dry_steps, None, 0)
    img = sim.render().astype(np.float64) / 255.0
    half = max(1, int(sw * 0.2))
    measured = np.empty((n, 2, 3))
    for i in range(n):
        x = int((i + 0.5) * col_w)
        for r, (y0, y1) in enumerate(rows):
            ya, yb = int(y0 + 0.3 * (y1 - y0)), int(y1 - 0.3 * (y1 - y0))
            patch = img[ya : max(ya + 1, yb), max(0, x - half) : x + half + 1]
            measured[i, r] = np.median(patch.reshape(-1, 3), axis=0)
    return paper, MixModel.from_calibration(names, paper, measured)


def _error_map(current: NDArray[np.uint8], target_lab: NDArray, sigma: float) -> NDArray[np.float64]:
    """Blurred per-pixel colour error, down-weighted where only lightening would help."""
    cur_lab = rgb01_to_lab(current.astype(np.float64) / 255.0)
    diff = cur_lab - target_lab
    err = np.sqrt(np.sum(diff * diff, axis=-1))
    lighter = target_lab[..., 0] > cur_lab[..., 0] + 3.0
    err = np.where(lighter, 0.3 * err, err)
    return ndimage.gaussian_filter(err, sigma=max(0.5, sigma))


def _round_op(op: dict[str, Any]) -> dict[str, Any]:
    """Round floats so the op list is tidy and replays bit-for-bit."""
    out = dict(op)
    if "path" in out:
        out["path"] = [[round(float(x), 2), round(float(y), 2)] for x, y in out["path"]]
    for key in ("width", "water"):
        if key in out:
            out[key] = round(float(out[key]), 3)
    if "pigments" in out:
        out["pigments"] = {k: round(float(v), 3) for k, v in out["pigments"].items()}
    return out


def _is_dry(sim: Sim) -> bool | None:
    """True if the sheet has no wet cells, False if it does, None if unknown."""
    try:
        wet = sim.state["wet_mask"]  # type: ignore[attr-defined]
    except Exception:
        return None
    try:
        return not bool(np.any(wet))
    except Exception:
        return None


def _dry_until_set(sim: Sim, max_steps: int, recorder: Any, chunk: int) -> int:
    """Step until the sheet is dry or ``max_steps`` is reached; return steps run.

    Uses the simulator's ``state["wet_mask"]`` when available. Without it,
    exactly ``max_steps`` steps are run.
    """
    done = 0
    block = max(1, min(25, max_steps))
    while done < max_steps:
        if _is_dry(sim) is True:
            break
        k = min(block, max_steps - done)
        _run_steps(sim, k, recorder, chunk or k)
        done += k
    return done


def _run_steps(sim: Sim, steps: int, recorder: Any, chunk: int) -> None:
    """Advance the simulator, in chunks when recording so frames are captured."""
    if steps <= 0:
        return
    if recorder is None:
        sim.step(steps)
        return
    left = steps
    while left > 0:
        k = min(left, chunk)
        sim.step(k)
        recorder.advance(sim, k)
        left -= k


def paint(
    target: NDArray[np.uint8],
    sim_factory: Callable[[], Sim],
    n_strokes: int = 60,
    candidates_per_stroke: int = 24,
    seed: int = 7,
    recorder: Any = None,
    *,
    score_fn: ScoreFn | None = None,
    phase_dry_steps: int = 600,
    final_dry_steps: int | None = None,
    calibrate: bool = True,
    pigments: Mapping[str, Any] | None = None,
    phases: Sequence[Phase] = PHASES,
    stats: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Paint ``target`` greedily and return the list of ops that reproduces it.

    ``target`` is uint8 (H, W, 3) already at canvas size. ``sim_factory`` is a
    zero-argument callable returning a fresh simulator of the same size and
    seed every time. ``n_strokes`` is the number of stroke slots; a slot
    whose best candidate does not lower the total score is skipped, so the
    painting can use fewer strokes. ``candidates_per_stroke`` candidates are
    tried per slot. The result starts with no canvas header; it holds the
    stroke and dry ops in order, and replaying them on ``sim_factory()``
    gives the final painting.

    Optional keyword arguments: ``score_fn`` (defaults to
    ``backrun.score.score``), ``phase_dry_steps`` (the most steps of
    the ``dry`` between phases; drying stops early once the simulator
    reports no wet cells, and the op records the steps actually run),
    ``final_dry_steps`` (the same for a last ``dry``; defaults to
    ``phase_dry_steps``, 0 to skip), ``calibrate`` (paint a calibration sheet
    first), ``pigments`` (name to representative colour, overriding
    discovery), ``phases`` and ``stats`` (a dict filled with counts and
    timings).

    If ``recorder`` is given it must behave like ``backrun.record.Recorder``
    and already be started; every op is logged with the score after it.
    """
    if score_fn is None:
        from backrun.score import score as score_fn  # black box
    t_start = time.perf_counter()
    tgt = np.asarray(target, dtype=np.uint8)
    h, w = tgt.shape[:2]
    rng = np.random.default_rng(seed)
    final_dry = phase_dry_steps if final_dry_steps is None else final_dry_steps
    frame_chunk = getattr(recorder, "frames_every", 20) if recorder is not None else 0

    if pigments is None:
        colours = discover_pigments()
    else:
        colours = {}
        for k, v in pigments.items():
            c = _colour_to_01(v)
            colours[k] = c if c is not None else np.full(3, 0.5)
    names = list(colours)

    sim = sim_factory()
    blank = sim.render()
    if blank.shape[:2] != (h, w):
        raise ValueError(f"simulator renders {blank.shape[:2]}, target is {(h, w)}")

    t0 = time.perf_counter()
    model: MixModel | None = None
    calibrated = False
    if calibrate:
        cal = _calibrate(sim_factory, names, w, h, max(phase_dry_steps, 1))
        if cal is not None:
            model = cal[1]
            calibrated = True
    if model is None:
        paper = np.median(blank.reshape(-1, 3), axis=0).astype(np.float64) / 255.0
        model = MixModel.from_colours(colours, paper)
    t_cal = time.perf_counter() - t0

    target_lab = rgb01_to_lab(tgt.astype(np.float64) / 255.0)
    target_luma = tgt.astype(np.float64) @ np.array([0.299, 0.587, 0.114])
    short, long_ = min(w, h), max(w, h)

    ops: list[dict[str, Any]] = []
    accepted = 0
    current = blank
    current_score = dict(score_fn(current, tgt, strokes=0))
    evaluations = 0
    skipped = 0
    phase_times: dict[str, float] = {}

    def commit(op: dict[str, Any], steps: int = 0) -> None:
        nonlocal current, current_score
        ops.append(op)
        if op["op"] == "stroke":
            sim.apply(op)
            if recorder is not None:
                recorder.advance(sim, 1)
        else:
            _run_steps(sim, steps, recorder, frame_chunk)
        current = sim.render()
        current_score = dict(score_fn(current, tgt, strokes=accepted))
        if recorder is not None:
            recorder.log_op(op, current_score)

    shares = np.array([p.share for p in phases], dtype=np.float64)
    counts = np.floor(shares / shares.sum() * n_strokes).astype(int)
    counts[-1] += n_strokes - counts.sum()

    for pi, (phase, count) in enumerate(zip(phases, counts)):
        tp = time.perf_counter()
        mid_w = short * float(np.mean(phase.width))
        gy, gx = np.gradient(ndimage.gaussian_filter(target_luma, sigma=max(1.0, mid_w / 2)))
        for _ in range(int(count)):
            err = _error_map(current, target_lab, mid_w / 2)
            stride = 2 if min(w, h) >= 32 else 1
            sub = err[::stride, ::stride]
            prob = sub.ravel() ** 2
            if prob.sum() <= 0:
                prob = np.ones_like(prob)
            prob = prob / prob.sum()
            best: tuple[float, dict[str, Any], Sim] | None = None
            for _c in range(candidates_per_stroke):
                idx = int(rng.choice(prob.size, p=prob))
                cy = (idx // sub.shape[1]) * stride + rng.uniform(0, stride)
                cx = (idx % sub.shape[1]) * stride + rng.uniform(0, stride)
                iy, ix = min(h - 1, int(cy)), min(w - 1, int(cx))
                if rng.random() < 0.7 and (gx[iy, ix] ** 2 + gy[iy, ix] ** 2) > 1e-6:
                    ang = np.arctan2(gy[iy, ix], gx[iy, ix]) + np.pi / 2 + rng.normal(0, 0.25)
                else:
                    ang = rng.uniform(0, np.pi)
                width = max(2.0, short * rng.uniform(*phase.width))
                length = max(width, long_ * rng.uniform(*phase.length))
                npts = int(rng.integers(phase.points[0], phase.points[1] + 1))
                ts = np.linspace(-0.5, 0.5, npts)
                d = np.array([np.cos(ang), np.sin(ang)])
                nrm = np.array([-d[1], d[0]])
                bend = rng.uniform(-0.15, 0.15) * length
                pts = (
                    np.array([cx, cy])[None]
                    + ts[:, None] * length * d[None]
                    + (np.cos(np.pi * ts) * bend)[:, None] * nrm[None]
                )
                pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
                pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)

                mask = stroke_mask(pts, width, w, h)
                ys, xs = np.nonzero(mask)
                if ys.size == 0:
                    continue
                if ys.size > _MATCH_SAMPLES:
                    pick = rng.choice(ys.size, size=_MATCH_SAMPLES, replace=False)
                    ys, xs = ys[pick], xs[pick]
                under = current[ys, xs].astype(np.float64) / 255.0
                ranked = model.ranked(under, target_lab[ys, xs], top=3)
                weights = _MIX_PICK[: len(ranked)] / _MIX_PICK[: len(ranked)].sum()
                choice = ranked[int(rng.choice(len(ranked), p=weights))]
                op = _round_op(
                    {
                        "op": "stroke",
                        "path": pts.tolist(),
                        "width": width,
                        "water": rng.uniform(*phase.water),
                        "pigments": model.combos[choice],
                    }
                )
                cand = sim.copy()
                cand.apply(op)
                if phase.settle_steps:
                    cand.step(phase.settle_steps)
                s = score_fn(cand.render(), tgt, strokes=accepted + 1)
                evaluations += 1
                if best is None or s["total"] < best[0]:
                    best = (float(s["total"]), op, cand)
            if best is None or best[0] >= current_score["total"]:
                skipped += 1
                continue
            accepted += 1
            commit(best[1])
            if phase.settle_steps:
                commit({"op": "dry", "steps": int(phase.settle_steps)}, phase.settle_steps)
        dry_steps = phase_dry_steps if pi < len(phases) - 1 else final_dry
        if dry_steps > 0:
            ran = _dry_until_set(sim, int(dry_steps), recorder, frame_chunk)
            if ran > 0:
                commit({"op": "dry", "steps": ran}, 0)
        phase_times[phase.name] = time.perf_counter() - tp

    if stats is not None:
        stats.update(
            {
                "strokes_accepted": accepted,
                "slots_skipped": skipped,
                "evaluations": evaluations,
                "calibration_s": t_cal,
                "phase_s": phase_times,
                "total_s": time.perf_counter() - t_start,
                "final_score": current_score,
                "calibrated": calibrated,
            }
        )
    if recorder is not None:
        recorder.time("calibration_s", t_cal)
        for k, v in phase_times.items():
            recorder.time(f"phase_{k}_s", v)
    return ops
