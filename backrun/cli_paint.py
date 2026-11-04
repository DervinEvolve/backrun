"""Paint a target with the search painter and record the run.

Usage::

    python -m backrun.cli_paint TARGET OUT_DIR [--width 256] [--height 192]
        [--strokes 60] [--candidates 24] [--seed 7] [--frames-every 20]

``TARGET`` is an image file, or ``synthetic:KIND`` / ``synthetic:KIND:SEED``
for one of the built-in test targets (``horizon``, ``discs``, ``sky``).

``OUT_DIR`` receives ``strokes.jsonl`` (canvas header plus every op),
``final.png``, ``target.png``, ``ledger.jsonl``, ``run.json`` and
``frames/``. Turn the frames into a GIF with ``scripts/make_gif.sh``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from backrun.painter_search import paint
from backrun.record import Recorder
from backrun.score import score
from backrun.targets import SYNTHETIC_KINDS, load_target, synthetic_target


def resolve_target(spec: str, width: int, height: int) -> np.ndarray:
    """Load a target from a file path or a ``synthetic:KIND[:SEED]`` spec."""
    if spec.startswith("synthetic:"):
        parts = spec.split(":")
        kind = parts[1] if len(parts) > 1 else ""
        if kind not in SYNTHETIC_KINDS:
            raise SystemExit(f"unknown synthetic target {kind!r}; choose from {', '.join(SYNTHETIC_KINDS)}")
        tseed = int(parts[2]) if len(parts) > 2 else 1
        return synthetic_target(kind, width, height, tseed)
    path = Path(spec)
    if not path.is_file():
        raise SystemExit(f"target not found: {spec}")
    return load_target(path, width, height)


def build_parser() -> argparse.ArgumentParser:
    """Command-line options."""
    ap = argparse.ArgumentParser(prog="python -m backrun.cli_paint", description=__doc__.split("\n\n")[0])
    ap.add_argument("target", help="image file, or synthetic:KIND[:SEED]")
    ap.add_argument("out_dir", help="run folder to write")
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--height", type=int, default=192)
    ap.add_argument("--strokes", type=int, default=60, help="stroke slots to fill")
    ap.add_argument("--candidates", type=int, default=24, help="candidates tried per slot")
    ap.add_argument("--seed", type=int, default=7, help="seed for the paper and the painter")
    ap.add_argument("--frames-every", type=int, default=20, help="simulator steps between frames")
    ap.add_argument("--paper", default="cold_press")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    """Run the painter and write the run folder. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    from backrun.sim import Watercolor  # imported here so --help works without it

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    target = resolve_target(args.target, args.width, args.height)
    canvas = {"op": "canvas", "width": args.width, "height": args.height, "seed": args.seed, "paper": args.paper}

    def factory() -> Watercolor:
        return Watercolor(width=args.width, height=args.height, seed=args.seed, paper=args.paper)

    rec = Recorder(out, frames_every=args.frames_every)
    rec.start(
        canvas={k: v for k, v in canvas.items() if k != "op"},
        seed=args.seed,
        target_path=args.target,
        target_image=target,
        strokes_requested=args.strokes,
        candidates_per_stroke=args.candidates,
        painter="search_v0",
    )
    rec.save_frame(factory().render())
    stats: dict = {}
    t0 = time.perf_counter()
    ops = paint(target, factory, args.strokes, args.candidates, args.seed, recorder=rec, stats=stats)
    elapsed = time.perf_counter() - t0

    with (out / "strokes.jsonl").open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(canvas) + "\n")
        for op in ops:
            fh.write(json.dumps(op) + "\n")

    replay = factory()
    for op in ops:
        replay.apply(op)
    final = replay.render()
    final_score = score(final, target, strokes=stats.get("strokes_accepted", 0))
    rec.time("paint_s", elapsed)
    rec.finish(final, final_score=final_score, painter_stats=stats)
    print(
        f"{stats.get('strokes_accepted', 0)} strokes accepted, {stats.get('slots_skipped', 0)} slots skipped, "
        f"{stats.get('evaluations', 0)} candidates simulated in {elapsed:.1f} s; "
        f"total {final_score['total']:.3f} (lab_l2 {final_score['lab_l2']:.1f}, ssim {final_score['ssim']:.3f})"
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
