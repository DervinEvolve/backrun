"""The gate: reject stroke files that are invalid or that cheat the judge.

``gate(text_or_ops, canvas)`` runs static checks first and stops there if any
fail. If the file is valid it is replayed from the canvas header, dried for
``SETTLE_STEPS`` more steps and rendered, and the render is checked for
coverage, lightness and variety. Every threshold below is a design choice,
picked by looking at renders, not derived from anything.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np

from backrun.pigments import PIGMENT_NAMES
from backrun.sim import Watercolor
from scribe.colour import hue_chroma, painted_mask, srgb_to_lab

# --- static limits (design choices) ---
MIN_STROKES = 6  # fewer than this is not a painting
MAX_STROKES = 200  # more than this is waste; long files were the main cost in the original recipe
MIN_PATH_POINTS = 2  # the simulator's own limits
MAX_PATH_POINTS = 8
MIN_WIDTH = 1.0  # pixels
MAX_WIDTH_FRACTION = 0.4  # of the shorter canvas side; stops one stroke covering the sheet
MAX_PIGMENTS_PER_STROKE = 3  # real palettes rarely mix more than three on one brush
MAX_TOTAL_DRY_STEPS = 20_000  # bounds render time at a few seconds
MAX_REASONS = 12  # keep the list short enough to hand back to the model

# --- rendered checks (design choices) ---
SETTLE_STEPS = 2_000  # extra drying after the file so the judge sees a dry sheet
PAINTED_DELTA_E = 3.0  # Delta E above which a pixel counts as painted
MIN_COVERAGE = 0.05  # below this the sheet is nearly blank
MAX_COVERAGE = 0.98  # above this nothing of the paper shows: the giant-wash cheat
MIN_MEAN_LIGHTNESS = 35.0  # CIELAB L*; below this the sheet is a dark slab
CHROMA_FLOOR = 6.0  # painted pixels below this chroma have no reliable hue
HUE_BINS = 12  # 30 degree hue bins
HUE_BIN_SHARE = 0.05  # a hue counts if this share of painted pixels falls in its bin
LIGHTNESS_BIN = 10.0  # L* width of a lightness level
LIGHTNESS_BIN_SHARE = 0.08  # a level counts if this share of painted pixels falls in it

_FENCE = re.compile(r"^\s*```")


@dataclass
class GateResult:
    """Outcome of the gate.

    ``ok`` is True only if every check passed. ``reasons`` holds short plain
    strings, one per failed check. ``ops`` is the cleaned op list (only known
    fields, canvas lines dropped). ``image`` is the uint8 render when the file
    got as far as rendering, else None. ``stats`` holds the measured numbers.
    """

    ok: bool
    reasons: list[str] = field(default_factory=list)
    ops: list[dict[str, Any]] = field(default_factory=list)
    image: np.ndarray | None = None
    stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly summary without the image."""
        return {"ok": self.ok, "reasons": list(self.reasons), "stats": dict(self.stats),
                "rendered": self.image is not None}


def _looks_like_op(line: str) -> bool:
    s = line.strip()
    return s.startswith("{")


def extract_jsonl(text: str) -> list[str]:
    """Pull the stroke lines out of a model reply.

    If the reply has a fenced code block, the lines inside the first block
    that holds any ``{...}`` line are used, plus any ``{...}`` lines that
    directly follow its closing fence (a stray trailing line). An unclosed
    fence runs to the end of the reply. Without a fence, every line that
    starts with ``{`` is used. Trailing commas are removed. Lines are
    returned as strings; parsing happens in the gate.
    """
    lines = text.splitlines()
    blocks: list[tuple[int, int]] = []  # (first line inside, index of closing fence or len)
    i = 0
    while i < len(lines):
        if _FENCE.match(lines[i]):
            j = i + 1
            while j < len(lines) and not _FENCE.match(lines[j]):
                j += 1
            blocks.append((i + 1, j))
            i = j + 1
        else:
            i += 1
    chosen: list[str] = []
    for start, end in blocks:
        inside = [ln for ln in lines[start:end] if _looks_like_op(ln)]
        if inside:
            chosen = inside
            k = end + 1
            while k < len(lines) and (_looks_like_op(lines[k]) or not lines[k].strip()):
                if lines[k].strip():
                    chosen.append(lines[k])
                k += 1
            break
    if not blocks or not chosen:
        chosen = [ln for ln in lines if _looks_like_op(ln)]
    return [ln.strip().rstrip(",").strip() for ln in chosen]


def _num(v: Any) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    f = float(v)
    return f if math.isfinite(f) else None


def _check_stroke(op: dict[str, Any], n: int, canvas: dict[str, Any], reasons: list[str]) -> dict[str, Any] | None:
    w, h = int(canvas["width"]), int(canvas["height"])
    bad = len(reasons)
    path = op.get("path")
    pts: list[list[float]] = []
    if not isinstance(path, list) or not MIN_PATH_POINTS <= len(path) <= MAX_PATH_POINTS:
        reasons.append(f"line {n}: path needs {MIN_PATH_POINTS} to {MAX_PATH_POINTS} points")
    else:
        for p in path:
            if not isinstance(p, list) or len(p) != 2 or _num(p[0]) is None or _num(p[1]) is None:
                reasons.append(f"line {n}: each path point must be [x, y] numbers")
                break
            x, y = float(p[0]), float(p[1])
            if not (0.0 <= x <= w and 0.0 <= y <= h):
                reasons.append(f"line {n}: point [{x:g}, {y:g}] is outside the {w} x {h} paper")
                break
            pts.append([x, y])
    max_w = MAX_WIDTH_FRACTION * min(w, h)
    width = _num(op.get("width"))
    if width is None or not MIN_WIDTH <= width <= max_w:
        reasons.append(f"line {n}: width must be a number from {MIN_WIDTH:g} to {max_w:g}")
    water = _num(op.get("water"))
    if water is None or not 0.0 <= water <= 1.0:
        reasons.append(f"line {n}: water must be a number from 0 to 1")
    pigments = op.get("pigments", {})
    clean_pig: dict[str, float] = {}
    if not isinstance(pigments, dict):
        reasons.append(f"line {n}: pigments must map names to concentrations")
    else:
        if len(pigments) > MAX_PIGMENTS_PER_STROKE:
            reasons.append(f"line {n}: at most {MAX_PIGMENTS_PER_STROKE} pigments per stroke")
        for name, conc in pigments.items():
            if name not in PIGMENT_NAMES:
                reasons.append(f"line {n}: unknown pigment {name!r}")
                continue
            c = _num(conc)
            if c is None or not 0.0 <= c <= 1.0:
                reasons.append(f"line {n}: concentration of {name} must be 0 to 1")
                continue
            clean_pig[name] = c
    if len(reasons) > bad:
        return None
    return {"op": "stroke", "path": pts, "width": width, "water": water, "pigments": clean_pig}


def static_check(items: list[Any], canvas: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    """Run the static checks on extracted lines or op dicts.

    Returns (reasons, cleaned ops, stats). Canvas lines are dropped because
    the gate supplies the header.
    """
    reasons: list[str] = []
    ops: list[dict[str, Any]] = []
    strokes = drys = dry_steps = 0
    for n, item in enumerate(items, start=1):
        if isinstance(item, str):
            try:
                op = json.loads(item)
            except json.JSONDecodeError:
                reasons.append(f"line {n}: not valid JSON")
                continue
        else:
            op = item
        if not isinstance(op, dict):
            reasons.append(f"line {n}: not a JSON object")
            continue
        kind = op.get("op")
        if kind == "canvas":
            continue
        if kind == "stroke":
            strokes += 1
            clean = _check_stroke(op, n, canvas, reasons)
            if clean is not None:
                ops.append(clean)
        elif kind == "dry":
            steps = op.get("steps")
            if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
                reasons.append(f"line {n}: dry steps must be a positive whole number")
                continue
            drys += 1
            dry_steps += steps
            ops.append({"op": "dry", "steps": steps})
        else:
            reasons.append(f"line {n}: unknown op {kind!r}; only stroke and dry")
    if not items:
        reasons.append("no stroke lines found")
    if not MIN_STROKES <= strokes <= MAX_STROKES:
        reasons.append(f"{strokes} strokes; need {MIN_STROKES} to {MAX_STROKES}")
    if drys == 0:
        reasons.append("no dry op; add at least one")
    if dry_steps > MAX_TOTAL_DRY_STEPS:
        reasons.append(f"{dry_steps} dry steps in total; at most {MAX_TOTAL_DRY_STEPS}")
    stats = {"strokes": strokes, "dry_ops": drys, "dry_steps": dry_steps}
    if len(reasons) > MAX_REASONS:
        extra = len(reasons) - MAX_REASONS + 1
        reasons = reasons[:MAX_REASONS - 1] + [f"and {extra} more problems"]
    return reasons, ops, stats


@lru_cache(maxsize=16)
def _blank(width: int, height: int, seed: int, paper: str) -> np.ndarray:
    return Watercolor(width=width, height=height, seed=seed, paper=paper).render()


def render_ops(ops: list[dict[str, Any]], canvas: dict[str, Any], settle: int = SETTLE_STEPS) -> np.ndarray:
    """Replay cleaned ops from the canvas header, dry ``settle`` more steps, render."""
    sim = Watercolor.from_header(canvas)
    for op in ops:
        sim.apply(op)
    sim.step(settle)
    return sim.render()


def _separated(hist: np.ndarray, share: float, circular: bool) -> bool:
    """True if two bins at least two apart each hold ``share`` or more."""
    idx = np.flatnonzero(hist >= share)
    n = len(hist)
    for a in idx:
        for b in idx:
            d = abs(int(a) - int(b))
            if circular:
                d = min(d, n - d)
            if d >= 2:
                return True
    return False


def rendered_check(image: np.ndarray, canvas: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    """Coverage, lightness floor and variety checks on a render. Returns (reasons, stats)."""
    blank = _blank(int(canvas["width"]), int(canvas["height"]), int(canvas.get("seed", 0)),
                   str(canvas.get("paper", "cold_press")))
    mask = painted_mask(image, blank, PAINTED_DELTA_E)
    lab = srgb_to_lab(image)
    coverage = float(mask.mean())
    mean_l = float(lab[..., 0].mean())
    painted = lab[mask]
    total = max(len(painted), 1)
    hue, chroma = hue_chroma(painted)
    hue_hist = np.histogram(hue[chroma > CHROMA_FLOOR], bins=HUE_BINS, range=(0.0, 360.0))[0] / total
    l_edges = np.arange(0.0, 100.0 + LIGHTNESS_BIN, LIGHTNESS_BIN)
    l_hist = np.histogram(painted[:, 0], bins=l_edges)[0] / total
    two_hues = _separated(hue_hist, HUE_BIN_SHARE, circular=True)
    two_levels = _separated(l_hist, LIGHTNESS_BIN_SHARE, circular=False)
    reasons: list[str] = []
    if coverage < MIN_COVERAGE:
        reasons.append(f"paint covers {coverage:.0%} of the paper; need at least {MIN_COVERAGE:.0%}")
    if coverage > MAX_COVERAGE:
        reasons.append(f"paint covers {coverage:.0%} of the paper; leave some paper showing")
    if mean_l < MIN_MEAN_LIGHTNESS:
        reasons.append(f"the painting is too dark overall (mean lightness {mean_l:.0f}, floor {MIN_MEAN_LIGHTNESS:g})")
    if not (two_hues or two_levels):
        reasons.append("only one flat colour; use at least two hues or two clearly different values")
    stats = {"coverage": round(coverage, 4), "mean_lightness": round(mean_l, 2),
             "two_hues": bool(two_hues), "two_levels": bool(two_levels),
             "hue_hist": [round(float(v), 4) for v in hue_hist],
             "lightness_hist": [round(float(v), 4) for v in l_hist]}
    return reasons, stats


def gate(text_or_ops: str | list[Any], canvas: dict[str, Any]) -> GateResult:
    """Check a model reply (or a list of op dicts or lines) and render it if valid."""
    items = extract_jsonl(text_or_ops) if isinstance(text_or_ops, str) else list(text_or_ops)
    reasons, ops, stats = static_check(items, canvas)
    if reasons:
        return GateResult(ok=False, reasons=reasons, ops=ops, stats=stats)
    try:
        image = render_ops(ops, canvas)
    except (ValueError, KeyError) as exc:  # the simulator refused an op the static checks missed
        return GateResult(ok=False, reasons=[f"simulator refused the file: {exc}"], ops=ops, stats=stats)
    r_reasons, r_stats = rendered_check(image, canvas)
    stats.update(r_stats)
    return GateResult(ok=not r_reasons, reasons=r_reasons, ops=ops, image=image, stats=stats)


def to_jsonl(ops: list[dict[str, Any]], canvas: dict[str, Any]) -> str:
    """A complete stroke file: the canvas header then the ops, one per line."""
    return "\n".join(json.dumps(o) for o in [canvas] + list(ops)) + "\n"
