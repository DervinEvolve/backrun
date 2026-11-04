"""The sealed-set report: how a painter does on targets it never trained on.

For a named painter, every sealed target is painted with each of three
seeds and a fixed stroke budget. Each painting is then dried for real until
no cell is wet and scored with ``backrun.score.score``. The report gives
the mean and the sample standard deviation over all paintings of:

``total``, ``lab_l2``, ``ssim``
    The scorer's numbers on the fully dried sheet.
``strokes``
    Strokes actually laid (the search painter skips slots that do not help).
``sim_steps``
    Simulator steps run by the painter, on every simulator it made: the
    sheet itself, every candidate copy and the calibration sheet. This is
    the compute measure. ``strokes_simulated`` counts stroke applications
    the same way (a search painter simulates many per stroke it keeps).
``sheet_steps``
    Steps run on the painting itself, including the final drying.
``wall_s``
    Wall time of the painter's own call, seconds. The final drying and
    scoring are measurement and are not included.
``predicted_total`` and ``dry_gap``
    The ``total`` the painter expected while painting, and the dried
    ``total`` minus that. Positive means the painter was optimistic. For the
    search painter the prediction is the dry-preview score just before its
    final dry (``predicted_before_dry`` in its ledger); for a policy it is
    the dry-preview score after its last stroke.

``start_std_x_px``, ``start_std_y_px`` and ``distinct_strokes``
    The collapse detector. Per painting, the standard deviation in pixels
    of the strokes' start points along each axis, and the number of
    distinct strokes once paths are rounded to 1 px and width and the
    eight concentrations to 0.05 (width in action units, 0..1). A painter
    that has collapsed onto one stroke shows a spread near 0 and one
    distinct stroke whatever its budget.

For the search painter the report also gives the same gap per phase, from
each phase's dry in the ledger.

Command line::

    python -m train.evaluate search runs/eval/search [--limit 5] [--width 128 --height 96]
    python -m train.evaluate policy:runs/ckpt/stage_a.pt runs/eval/stage_a

Painters: ``search`` (``backrun.painter_search.paint``, ``--candidates`` per
slot), ``policy:PATH`` (a checkpoint from ``train.policy.save_checkpoint``),
and two baselines that say where a learned painter stands between doing
nothing and the search painter: ``blank`` (no strokes, one 2000-step dry)
and ``mean:LEDGER_ROOT`` (the dataset-mean action of that ledger root's
strokes, laid in every slot).

Reports from several painters go side by side with::

    python -m train.evaluate compare OUT.md REPORT_DIR [REPORT_DIR ...]

which also places each one between the ``blank`` and ``search`` reports
when both are among them.
The sealed set is read from ``runs/data/sealed`` unless ``--sealed`` says
otherwise, and its manifest is checked before anything is painted.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from backrun.score import score
from backrun.targets import letterbox
from train import data

#: Seeds every target is painted with.
DEFAULT_SEEDS: tuple[int, ...] = (7, 8, 9)
#: Stroke budget, as in docs/TRAINING.md.
DEFAULT_STROKES = 40
#: Candidates per slot for the search painter.
DEFAULT_CANDIDATES = 16
#: The most steps of drying after a painting before it is scored.
MAX_FINAL_DRY = 10_000

#: Metrics summarised in the report, in table order, with their labels.
METRICS: tuple[tuple[str, str], ...] = (
    ("total", "total (dried, lower is better)"),
    ("lab_l2", "lab_l2 (dried, lower is better)"),
    ("ssim", "ssim (dried, higher is better)"),
    ("strokes", "strokes used"),
    ("sim_steps", "simulator steps per painting, all simulators"),
    ("strokes_simulated", "strokes simulated per painting, all simulators"),
    ("sheet_steps", "simulator steps on the sheet itself"),
    ("wall_s", "wall time per painting, s"),
    ("predicted_total", "total the painter predicted while painting"),
    ("dry_gap", "dried total minus predicted total"),
    ("start_std_x_px", "spread of stroke start points, x, px"),
    ("start_std_y_px", "spread of stroke start points, y, px"),
    ("distinct_strokes", "distinct strokes per painting"),
)
#: The collapse detector's rounding: paths to this many pixels, width and
#: concentrations (both 0..1) to this step.
COLLAPSE_PX = 1.0
COLLAPSE_STEP = 0.05


# ---------------------------------------------------------------------------
# The collapse detector.
# ---------------------------------------------------------------------------


def collapse_stats(ops: Sequence[dict[str, Any]], width: int, height: int) -> dict[str, float]:
    """Spread of stroke start points and the number of distinct strokes in one painting.

    ``start_std_x_px`` and ``start_std_y_px``: population standard
    deviation of the first path point of every stroke, in pixels.
    ``distinct_strokes``: strokes that still differ once every path point
    is rounded to ``COLLAPSE_PX`` and width (as a fraction of
    ``train.policy.WIDTH_RANGE``) and each concentration to
    ``COLLAPSE_STEP``. A painting with no strokes gets zeros.
    """
    from backrun.pigments import PIGMENT_NAMES
    from train.policy import WIDTH_RANGE

    strokes = [op for op in ops if op.get("op") == "stroke"]
    if not strokes:
        return {"start_std_x_px": 0.0, "start_std_y_px": 0.0, "distinct_strokes": 0}
    starts = np.array([op["path"][0] for op in strokes], dtype=np.float64)
    short = min(int(width), int(height))
    lo, hi = WIDTH_RANGE
    keys = set()
    for op in strokes:
        path = tuple(tuple(int(round(float(v) / COLLAPSE_PX)) for v in pt) for pt in op["path"])
        w = int(round(((float(op["width"]) / short - lo) / (hi - lo)) / COLLAPSE_STEP))
        conc = tuple(int(round(float(op.get("pigments", {}).get(n, 0.0)) / COLLAPSE_STEP)) for n in PIGMENT_NAMES)
        keys.add((path, w, conc))
    return {
        "start_std_x_px": float(starts[:, 0].std()),
        "start_std_y_px": float(starts[:, 1].std()),
        "distinct_strokes": len(keys),
    }


def collapse_stats_of_run(run_dir: str | Path) -> dict[str, float]:
    """``collapse_stats`` of a painting folder's ``strokes.jsonl`` (for reports written before the detector)."""
    lines = [json.loads(x) for x in (Path(run_dir) / "strokes.jsonl").read_text().splitlines() if x.strip()]
    header = lines[0]
    return collapse_stats(lines[1:], int(header["width"]), int(header["height"]))


# ---------------------------------------------------------------------------
# Counting simulator steps.
# ---------------------------------------------------------------------------


def counting_factory(width: int, height: int, seed: int, paper: str = "cold_press") -> tuple[Callable[[], Any], dict[str, int]]:
    """A simulator factory whose sheets, and all their copies, count their work.

    Returns (factory, counter). ``counter["steps"]`` grows by every
    simulator step and ``counter["strokes"]`` by every stroke applied, on
    any sheet the factory made or any copy of one.
    """
    from backrun.sim import Watercolor

    counter = {"steps": 0, "strokes": 0}

    class CountingWatercolor(Watercolor):
        """A Watercolor that adds its steps and strokes to a shared counter."""

        def step(self, n: int = 1) -> None:
            counter["steps"] += int(n)
            super().step(n)

        def apply(self, op: dict[str, Any]) -> None:
            if op.get("op") == "stroke":
                counter["strokes"] += 1
            super().apply(op)

    def factory() -> Watercolor:
        return CountingWatercolor(width=width, height=height, seed=seed, paper=paper)

    return factory, counter


def dry_until_set(sim: Any, max_steps: int = MAX_FINAL_DRY, block: int = 25) -> int:
    """Step ``sim`` until no cell is wet or ``max_steps`` is reached; return steps run."""
    done = 0
    while done < max_steps and bool(np.any(sim.state["wet_mask"])):
        k = min(block, max_steps - done)
        sim.step(k)
        done += k
    return done


# ---------------------------------------------------------------------------
# Painters.
# ---------------------------------------------------------------------------


@dataclass
class PaintResult:
    """What a painter hands back: its ops, its own prediction, and extras."""

    ops: list[dict[str, Any]]
    predicted: dict[str, Any]
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchPainter:
    """The v0 search painter, recorded with ``backrun.record.Recorder``."""

    candidates: int = DEFAULT_CANDIDATES
    phase_dry_steps: int = 2000
    name: str = "search"

    def paint(
        self, target: NDArray[np.uint8], factory: Callable[[], Any], n_strokes: int, seed: int, run_dir: Path
    ) -> PaintResult:
        """Paint with ``backrun.painter_search.paint`` and read its prediction from the ledger."""
        from backrun.painter_search import paint
        from backrun.record import Recorder

        rec = Recorder(run_dir, frames_every=1_000_000)
        h, w = target.shape[:2]
        rec.start(canvas={"width": w, "height": h, "seed": seed, "paper": "cold_press"}, seed=seed,
                  target_image=target, painter=self.name, strokes_requested=n_strokes,
                  candidates_per_stroke=self.candidates)
        stats: dict[str, Any] = {}
        ops = paint(target, factory, n_strokes, self.candidates, seed, recorder=rec, stats=stats,
                    phase_dry_steps=self.phase_dry_steps)
        rec.finish(None, painter_stats=stats)
        lines = [json.loads(x) for x in rec.ledger_path.read_text().splitlines() if x.strip()]
        predicted: dict[str, Any] | None = None
        phase_gaps: dict[str, float] = {}
        for line in lines:
            if "predicted_before_dry" in line:
                gap = float(line["score"]["total"]) - float(line["predicted_before_dry"]["total"])
                phase_gaps[line.get("phase", "?")] = round(gap, 4)
                if line.get("dry_kind") == "final":
                    predicted = line["predicted_before_dry"]
        if predicted is None:
            predicted = lines[-1]["predicted_score"] if lines else dict(score(factory().render(), target, 0))
        return PaintResult(ops, dict(predicted), {"phase_gaps": phase_gaps, "evaluations": stats.get("evaluations")})


@dataclass
class PolicyPainter:
    """A learned policy: one observation and one stroke per slot, no search.

    Give either ``checkpoint`` (loaded lazily, once per process) or
    ``policy_fn`` (any callable from observation to action, for tests).
    """

    checkpoint: str | None = None
    policy_fn: Callable[[NDArray[np.float32]], Any] | None = None
    name: str = "policy"
    #: ``argmax`` takes the most likely stroke each time; ``sample`` draws one
    #: from the policy's distribution, seeded per painting so runs repeat.
    mode: str = "argmax"
    temperature: float = 1.0

    def _policy(self, seed: int = 0) -> Callable[[NDArray[np.float32]], Any]:
        if self.policy_fn is not None:
            return self.policy_fn
        from train import policy as pol

        if self.checkpoint is None:
            raise ValueError("PolicyPainter needs a checkpoint or a policy_fn")
        if getattr(self, "_model", None) is None:
            self._model = pol.load_checkpoint(self.checkpoint)
        model = self._model
        if self.mode == "sample":
            import torch

            gen = torch.Generator().manual_seed(int(seed))
            return lambda o: pol.act(model, o, mode="sample", temperature=self.temperature, generator=gen)
        return lambda o: pol.act(model, o)

    def __getstate__(self) -> dict[str, Any]:
        """Pickle the checkpoint path, not a loaded model."""
        state = dict(self.__dict__)
        state.pop("_model", None)
        if state.get("checkpoint") is not None:
            state["policy_fn"] = None
        return state

    def paint(
        self, target: NDArray[np.uint8], factory: Callable[[], Any], n_strokes: int, seed: int, run_dir: Path
    ) -> PaintResult:
        """Roll the policy out, then read its dry preview as its prediction."""
        from train import policy as pol

        sim = factory()
        ops = pol.rollout(self._policy(seed), target, sim, n_strokes)
        strokes = sum(op["op"] == "stroke" for op in ops)
        predicted = dict(score(sim.render(predict_dry=True), target, strokes=strokes))
        return PaintResult(ops, predicted)


@dataclass
class BlankPainter:
    """A baseline that lays no strokes: the sheet is only dried, ``dry_steps`` steps."""

    dry_steps: int = 2000
    name: str = "blank"

    def paint(
        self, target: NDArray[np.uint8], factory: Callable[[], Any], n_strokes: int, seed: int, run_dir: Path
    ) -> PaintResult:
        """One dry op and nothing else; the prediction is the dry preview of the blank sheet."""
        sim = factory()
        ops = [{"op": "dry", "steps": int(self.dry_steps)}] if self.dry_steps > 0 else []
        for op in ops:
            sim.apply(op)
        return PaintResult(ops, dict(score(sim.render(predict_dry=True), target, strokes=0)))


def mean_action(ledger_roots: Sequence[str | Path]) -> NDArray[np.float32]:
    """The dataset-mean action of every accepted stroke under the ledger roots.

    Reads each finished run's ``strokes.jsonl`` (no replay), encodes every
    stroke with the wait that follows it, as Stage A does
    (``train.policy.encode_stroke``), and averages the 17 numbers.
    """
    from train.imitation import find_runs
    from train.policy import encode_stroke

    actions = []
    for run in find_runs(ledger_roots):
        lines = [json.loads(x) for x in (run / "strokes.jsonl").read_text().splitlines() if x.strip()]
        size = (int(lines[0]["width"]), int(lines[0]["height"]))
        ops = lines[1:]
        for i, op in enumerate(ops):
            if op["op"] != "stroke":
                continue
            wait = 0
            for later in ops[i + 1 :]:
                if later["op"] != "dry":
                    break
                wait += int(later["steps"])
            actions.append(encode_stroke(op, wait, size))
    if not actions:
        raise ValueError(f"no strokes found under {list(map(str, ledger_roots))}")
    return np.mean(np.stack(actions).astype(np.float64), axis=0).astype(np.float32)


@dataclass
class MeanPainter:
    """A baseline that lays the same stroke, a fixed action, in every slot.

    With ``mean_action`` it is the stroke a regression policy would collapse
    to if it learned nothing from the observation.
    """

    action: NDArray[np.float32]
    name: str = "mean"

    def paint(
        self, target: NDArray[np.uint8], factory: Callable[[], Any], n_strokes: int, seed: int, run_dir: Path
    ) -> PaintResult:
        """Lay ``action`` ``n_strokes`` times (with its wait), then read the dry preview."""
        from train import policy as pol

        sim = factory()
        action = np.asarray(self.action, dtype=np.float32)
        ops = pol.rollout(lambda _obs: action, target, sim, n_strokes)
        strokes = sum(op["op"] == "stroke" for op in ops)
        return PaintResult(ops, dict(score(sim.render(predict_dry=True), target, strokes=strokes)))


Painter = Any  # SearchPainter | PolicyPainter | BlankPainter | MeanPainter


def make_painter(spec: str, candidates: int = DEFAULT_CANDIDATES) -> Painter:
    """Build a painter from ``search``, ``policy:PATH``, ``blank`` or ``mean:LEDGER_ROOT``."""
    if spec == "search":
        return SearchPainter(candidates=candidates)
    if spec == "blank":
        return BlankPainter()
    if spec.startswith("mean:"):
        root = Path(spec.split(":", 1)[1])
        if not root.is_dir():
            raise FileNotFoundError(f"ledger root not found: {root}")
        return MeanPainter(mean_action([root]), name=f"mean:{root.name}")
    if spec.startswith("policy:"):
        path, mode, temperature = parse_policy_spec(spec)
        if not Path(path).is_file():
            raise FileNotFoundError(f"checkpoint not found: {path}")
        suffix = "" if mode == "argmax" else f"@sample:{temperature:g}"
        return PolicyPainter(checkpoint=path, name=f"policy:{Path(path).name}{suffix}", mode=mode,
                             temperature=temperature)
    raise ValueError(
        f"unknown painter {spec!r}; use 'search', 'policy:PATH', 'policy:PATH@sample[:T]', 'blank' or 'mean:LEDGER_ROOT'"
    )


def parse_policy_spec(spec: str) -> tuple[str, str, float]:
    """Split ``policy:PATH``, ``policy:PATH@sample`` or ``policy:PATH@sample:T`` into (path, mode, temperature).

    ``@sample`` draws each stroke from the policy's distribution instead of
    taking the most likely one; ``T`` is the softmax temperature (default 1).
    """
    rest = spec.split(":", 1)[1]
    if "@" not in rest:
        return rest, "argmax", 1.0
    path, _, tail = rest.rpartition("@")
    parts = tail.split(":")
    if parts[0] != "sample" or len(parts) > 2:
        raise ValueError(f"bad policy spec {spec!r}; the part after '@' must be 'sample' or 'sample:T'")
    temperature = float(parts[1]) if len(parts) == 2 else 1.0
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return path, "sample", temperature


# ---------------------------------------------------------------------------
# Running the evaluation.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Job:
    """One painting of the evaluation."""

    painter: Any
    target_name: str
    target: NDArray[np.uint8]
    seed: int
    n_strokes: int
    run_dir: str
    max_final_dry: int = MAX_FINAL_DRY


def paint_one(job: Job) -> dict[str, Any]:
    """Paint one sealed target with one seed and measure it."""
    from backrun.sim import Watercolor

    h, w = job.target.shape[:2]
    run_dir = Path(job.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    factory, counter = counting_factory(w, h, job.seed)
    t0 = time.perf_counter()
    result = job.painter.paint(job.target, factory, job.n_strokes, job.seed, run_dir)
    wall = time.perf_counter() - t0
    painter_steps, painter_strokes = counter["steps"], counter["strokes"]

    # Measurement, outside the timing: replay the ops on a plain sheet, dry it
    # for real and score it. The replay also checks the ops reproduce the work.
    sheet = Watercolor(width=w, height=h, seed=job.seed)
    for op in result.ops:
        sheet.apply(op)
    extra_dry = dry_until_set(sheet, job.max_final_dry)
    ops = list(result.ops) + ([{"op": "dry", "steps": extra_dry}] if extra_dry else [])
    header = {"op": "canvas", "width": w, "height": h, "seed": job.seed, "paper": "cold_press"}
    (run_dir / "strokes.jsonl").write_text("".join(json.dumps(x) + "\n" for x in [header, *ops]))
    strokes = sum(op["op"] == "stroke" for op in ops)
    final = dict(score(sheet.render(), job.target, strokes=strokes))
    from PIL import Image

    Image.fromarray(sheet.render()).save(run_dir / "final.png")
    sheet_steps = sum(int(op["steps"]) for op in ops if op["op"] == "dry")
    row = {
        "target": job.target_name,
        "seed": job.seed,
        "total": float(final["total"]),
        "lab_l2": float(final["lab_l2"]),
        "ssim": float(final["ssim"]),
        "strokes": int(strokes),
        "sim_steps": int(painter_steps),
        "strokes_simulated": int(painter_strokes),
        "sheet_steps": int(sheet_steps),
        "final_dry_steps": int(extra_dry),
        "dried": not bool(np.any(sheet.state["wet_mask"])),
        "wall_s": round(wall, 4),
        "predicted_total": float(result.predicted["total"]),
        "dry_gap": float(final["total"]) - float(result.predicted["total"]),
        "run_dir": str(run_dir),
        **collapse_stats(result.ops, w, h),
    }
    if result.extra:
        row.update(result.extra)
    return row


def summarise(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Mean and sample standard deviation of each metric over the paintings."""
    out: dict[str, dict[str, float]] = {}
    for key, _ in METRICS:
        vals = np.array([float(r[key]) for r in rows if key in r], dtype=np.float64)
        out[key] = {
            "mean": float(vals.mean()) if vals.size else math.nan,
            "std": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
            "n": int(vals.size),
        }
    phases: dict[str, list[float]] = {}
    for r in rows:
        for name, gap in (r.get("phase_gaps") or {}).items():
            phases.setdefault(name, []).append(float(gap))
    for name, gaps in phases.items():
        g = np.array(gaps)
        out[f"phase_gap_{name}"] = {"mean": float(g.mean()), "std": float(g.std(ddof=1)) if g.size > 1 else 0.0,
                                    "n": int(g.size)}
    return out


def _fmt(x: float) -> str:
    """Format a number for the table with sensible precision."""
    if not math.isfinite(x):
        return "n/a"
    ax = abs(x)
    if ax >= 1000:
        return f"{x:,.0f}"
    if ax >= 100:
        return f"{x:.1f}"
    if ax >= 1:
        return f"{x:.3f}"
    return f"{x:.4f}"


def markdown_report(report: dict[str, Any]) -> str:
    """The report as a Markdown page with one summary table."""
    s = report["settings"]
    summary = report["summary"]
    lines = [
        f"# Sealed-set evaluation: {report['painter']}",
        "",
        f"{report['n_targets']} sealed targets x {len(s['seeds'])} seeds (seeds {', '.join(map(str, s['seeds']))}) "
        f"= {report['n_paintings']} paintings. Canvas {s['width']} x {s['height']}, {s['strokes']} stroke slots"
        + (f", {s['candidates']} candidates per slot" if report["painter"] == "search" else "")
        + ". Scores are measured on the fully dried sheet.",
        "",
        f"Sealed manifest sha256 `{report['sealed_manifest_sha256']}`. Mean and sample standard deviation "
        "over paintings.",
        "",
        "| metric | mean | std |",
        "|---|---|---|",
    ]
    for key, label in METRICS:
        m = summary[key]
        lines.append(f"| {label} | {_fmt(m['mean'])} | {_fmt(m['std'])} |")
    phase_keys = [k for k in summary if k.startswith("phase_gap_")]
    if phase_keys:
        lines += ["", "Gap between the score after each phase's real dry and the painter's prediction just before it "
                  "(positive means optimistic):", "", "| phase | mean gap | std |", "|---|---|---|"]
        for k in phase_keys:
            lines.append(f"| {k[len('phase_gap_'):]} | {_fmt(summary[k]['mean'])} | {_fmt(summary[k]['std'])} |")
    lines += ["", f"Total wall time {report['wall_s']:.1f} s with {s['workers']} worker(s)."]
    return "\n".join(lines) + "\n"


def load_report(report_dir: str | Path) -> dict[str, Any]:
    """Read ``report.json`` from an evaluation folder, filling in the collapse detector if it predates it."""
    report = json.loads((Path(report_dir) / "report.json").read_text())
    rows = report["paintings"]
    if rows and "distinct_strokes" not in rows[0]:
        for r in rows:
            run = Path(r["run_dir"])
            if not run.is_absolute() and not run.exists():
                run = Path(report_dir) / "paintings" / run.name
            r.update(collapse_stats_of_run(run))
        report["summary"] = summarise(rows)
    report["report_dir"] = str(report_dir)
    return report


def compare_markdown(reports: Sequence[dict[str, Any]]) -> str:
    """Several reports side by side, one row per painter.

    When a ``blank`` and a ``search`` report are both present, each row
    also gets ``gap closed``: how far its mean ``total`` has moved from the
    blank sheet's toward the search painter's (0 is no better than blank,
    1 is as good as search, negative is worse than blank). Rows whose
    painter names repeat are told apart by their report folder's name.
    """
    by_name = {r["painter"]: r for r in reports}
    blank = by_name.get("blank", {}).get("summary", {}).get("total", {}).get("mean")
    search = by_name.get("search", {}).get("summary", {}).get("total", {}).get("mean")
    lines = [
        "| painter | paintings | total | std | lab_l2 | ssim | strokes | start spread x, y (px) | distinct strokes"
        + (" | gap closed |" if blank is not None and search is not None else " |"),
        "|---" * (9 + (blank is not None and search is not None)) + "|",
    ]
    names = [r["painter"] for r in reports]
    for r in reports:
        m = r["summary"]
        label = r["painter"]
        if names.count(label) > 1 and r.get("report_dir"):
            label += f" ({Path(r['report_dir']).name})"
        g = lambda k: m.get(k, {}).get("mean", math.nan)  # noqa: E731
        row = (f"| {label} | {r['n_paintings']} | {_fmt(g('total'))} | {_fmt(m['total']['std'])} | "
               f"{_fmt(g('lab_l2'))} | {_fmt(g('ssim'))} | {_fmt(g('strokes'))} | "
               f"{_fmt(g('start_std_x_px'))}, {_fmt(g('start_std_y_px'))} | {_fmt(g('distinct_strokes'))} |")
        if blank is not None and search is not None:
            closed = (blank - g("total")) / (blank - search) if blank != search else math.nan
            row += f" {_fmt(closed)} |"
        lines.append(row)
    return "\n".join(lines) + "\n"


def evaluate(
    painter: Painter,
    out_dir: str | Path,
    *,
    sealed_dir: str | Path | None = None,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    n_strokes: int = DEFAULT_STROKES,
    width: int = 256,
    height: int = 192,
    limit: int | None = None,
    workers: int = 1,
    max_final_dry: int = MAX_FINAL_DRY,
) -> dict[str, Any]:
    """Paint the sealed set and write ``report.json`` and ``report.md`` to ``out_dir``.

    ``limit`` keeps only the first N sealed targets (in manifest order);
    ``width`` and ``height`` letterbox the targets to a smaller canvas for
    quick runs. Returns the report dict.
    """
    t0 = time.perf_counter()
    sealed = Path(sealed_dir) if sealed_dir is not None else data.default_data_root() / "sealed"
    targets = data.load_sealed_set(sealed)
    if limit is not None:
        targets = targets[: max(0, int(limit))]
    out = Path(out_dir)
    if data.sealed_folder_of(out) is not None:
        raise data.SealedSetError(f"refusing to write evaluation output inside the sealed folder: {out}")
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for name, img in targets:
        tgt = img if img.shape[:2] == (height, width) else letterbox(img, width, height)
        for seed in seeds:
            jobs.append(Job(painter, name, tgt, int(seed), n_strokes, str(out / "paintings" / f"{name}_s{seed}"),
                            max_final_dry))
    if workers <= 1 or len(jobs) <= 1:
        rows = [paint_one(j) for j in jobs]
    else:
        with mp.get_context("spawn").Pool(processes=min(workers, len(jobs))) as pool:
            rows = list(pool.imap_unordered(paint_one, jobs, chunksize=1))
    rows.sort(key=lambda r: (r["target"], r["seed"]))
    report = {
        "painter": painter.name,
        "settings": {
            "seeds": list(map(int, seeds)), "strokes": n_strokes, "width": width, "height": height,
            "candidates": getattr(painter, "candidates", None), "limit": limit, "workers": workers,
            "max_final_dry": max_final_dry, "sealed_dir": str(sealed),
        },
        "sealed_manifest_sha256": data.file_sha256(sealed / data.MANIFEST_NAME),
        "n_targets": len(targets),
        "n_paintings": len(rows),
        "summary": summarise(rows),
        "paintings": rows,
        "wall_s": round(time.perf_counter() - t0, 2),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "report.md").write_text(markdown_report(report))
    return report


def build_parser() -> argparse.ArgumentParser:
    """Command-line options."""
    ap = argparse.ArgumentParser(prog="python -m train.evaluate", description="Paint the sealed set and report.")
    ap.add_argument("painter", help="'search', 'policy:PATH', 'blank' or 'mean:LEDGER_ROOT'")
    ap.add_argument("out_dir", help="folder for report.json, report.md and the paintings")
    ap.add_argument("--sealed", default=None, help="sealed set folder (default runs/data/sealed)")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    ap.add_argument("--strokes", type=int, default=DEFAULT_STROKES, help="stroke budget per painting")
    ap.add_argument("--candidates", type=int, default=DEFAULT_CANDIDATES, help="search painter candidates per slot")
    ap.add_argument("--width", type=int, default=256, help="canvas width (smaller for quick runs)")
    ap.add_argument("--height", type=int, default=192, help="canvas height")
    ap.add_argument("--limit", type=int, default=None, help="only the first N sealed targets")
    ap.add_argument("--workers", type=int, default=1)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line and print the Markdown table. Returns an exit code.

    ``compare OUT.md REPORT_DIR ...`` writes and prints ``compare_markdown``
    of existing reports instead of painting.
    """
    args = list(argv) if argv is not None else None
    import sys

    raw = args if args is not None else sys.argv[1:]
    if raw and raw[0] == "compare":
        if len(raw) < 3:
            raise SystemExit("usage: python -m train.evaluate compare OUT.md REPORT_DIR [REPORT_DIR ...]")
        md = compare_markdown([load_report(d) for d in raw[2:]])
        Path(raw[1]).write_text(md)
        print(md)
        return 0
    a = build_parser().parse_args(args)
    if a.workers > 1:
        # One thread per worker process, so paintings do not fight over cores.
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(var, "1")
    painter = make_painter(a.painter, a.candidates)
    report = evaluate(painter, a.out_dir, sealed_dir=a.sealed, seeds=a.seeds, n_strokes=a.strokes, width=a.width,
                      height=a.height, limit=a.limit, workers=a.workers)
    print(markdown_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
