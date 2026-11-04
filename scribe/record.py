"""Replay a scribe stroke file as a run folder the studio can show.

    python -m scribe.record CANDIDATE_DIR OUT_DIR [--frames-every 20]

``CANDIDATE_DIR`` is one candidate of a pool or batch: its ``strokes.jsonl``
is replayed through ``backrun.record.Recorder`` exactly as the simulator
painted it, with a frame every few steps, so ``studio/scripts/pack-run.mjs``
can pack it like a search or learned-painter run. There is no target image
for a scribe painting, so ``target.png`` is a prompt card: the subject
written on blank paper of the same size. ``run.json`` names the painter
``scribe``, the model that wrote the file, the prompt, the rating if the
pool has one, and the judge's numbers if the candidate's ``meta.json``
carries them.
"""

from __future__ import annotations

import argparse
import json
import textwrap
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from backrun.record import Recorder
from backrun.sim import Watercolor
from scribe.gate import _blank

FINAL_BLOCK = 25
MAX_FINAL_DRY = 2000


def prompt_card(width: int, height: int, seed: int, paper: str, subject: str, model: str = "") -> np.ndarray:
    """Blank paper with the subject written on it, as the run's stand-in for a target image."""
    from PIL import Image, ImageDraw

    img = Image.fromarray(_blank(width, height, seed, paper).copy())
    d = ImageDraw.Draw(img)
    lines = textwrap.wrap(subject, width=max(12, width // 9)) or [subject]
    y = height // 2 - 7 * len(lines)
    for line in lines:
        d.text((max(8, (width - 6 * len(line)) // 2), y), line, fill=(60, 50, 40))
        y += 14
    if model:
        d.text((8, height - 14), model[:40], fill=(130, 120, 110))
    return np.asarray(img)


def rating_for(candidate: Path) -> str | None:
    """The candidate's tier if its parent folder is a rated pool."""
    ratings = candidate.parent / "ratings.jsonl"
    if not ratings.is_file():
        return None
    tier = None
    for line in ratings.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            if rec.get("id") == candidate.name:
                tier = rec.get("tier")
    return tier


def record(candidate: str | Path, out_dir: str | Path, frames_every: int = 20) -> dict[str, Any]:
    """Replay the candidate's stroke file into ``out_dir``; return the run summary."""
    candidate = Path(candidate)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    header, ops = Watercolor.read_jsonl(candidate / "strokes.jsonl")
    meta = json.loads((candidate / "meta.json").read_text()) if (candidate / "meta.json").is_file() else {}
    width, height = int(header["width"]), int(header["height"])
    seed, paper = int(header.get("seed", 0)), str(header.get("paper", "cold_press"))
    subject = meta.get("subject", candidate.name)
    model = meta.get("model", "")
    card = prompt_card(width, height, seed, paper, subject, model)
    n_strokes = sum(op["op"] == "stroke" for op in ops)
    rec = Recorder(out, frames_every=frames_every)
    rec.start(canvas={"width": width, "height": height, "paper": paper}, seed=seed,
              target_path=f"prompt:{meta.get('prompt_id', candidate.name)}", target_image=card,
              strokes_requested=n_strokes, candidates_per_stroke=1, painter="scribe", model=model,
              subject=subject, prompt=meta.get("prompt"), candidate=candidate.name, rating=rating_for(candidate),
              judge=meta.get("judge"), gate_stats=(meta.get("gate") or {}).get("stats"))
    sim = Watercolor(width=width, height=height, seed=seed, paper=paper)
    rec.save_frame(sim.render())
    names = list(getattr(sim, "pigment_names", ()) or ())

    def run_steps(n: int, kind: str) -> None:
        while n > 0:
            k = min(n, frames_every)
            sim.step(k)
            rec.advance(sim, k)
            n -= k

    def wet() -> int:
        return int(np.count_nonzero(sim.state["wet_mask"]))

    t0 = time.perf_counter()
    strokes = 0
    for op in ops:
        if op["op"] == "stroke":
            sim.apply(op)
            rec.advance(sim, 1)
            strokes += 1
            rec.log_op(op, None, phase="scribe", phase_index=0, phase_count=1, pigments_used=dict(op["pigments"]),
                       candidates_considered=1, rejected=[], wet_pixels=wet())
        elif op["op"] == "dry":
            run_steps(int(op["steps"]), "settle")
            rec.log_op(op, None, phase="scribe", phase_index=0, phase_count=1, dry_kind="settle", wet_pixels=wet())
    ran = 0
    while ran < MAX_FINAL_DRY and wet():
        k = min(FINAL_BLOCK, MAX_FINAL_DRY - ran)
        run_steps(k, "final")
        ran += k
    if ran:
        op = {"op": "dry", "steps": ran}
        ops = list(ops) + [op]
        rec.log_op(op, None, phase="scribe", phase_index=0, phase_count=1, dry_kind="final", wet_pixels=wet())
    elapsed = time.perf_counter() - t0
    with (out / "strokes.jsonl").open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(header) + "\n")
        for op in ops:
            fh.write(json.dumps(op) + "\n")
    rec.time("paint_s", elapsed)
    rec.finish(sim.render(), painter_stats={"strokes_accepted": strokes, "slots_skipped": 0, "evaluations": 0,
                                            "looks": 0, "final_dry_steps": ran, "total_s": elapsed})
    return json.loads((out / "run.json").read_text())


def main(argv: Sequence[str] | None = None) -> int:
    """Command line: see the module docstring."""
    ap = argparse.ArgumentParser(prog="python -m scribe.record", description=__doc__.split("\n\n")[0])
    ap.add_argument("candidate", help="a pool or batch candidate folder with strokes.jsonl and meta.json")
    ap.add_argument("out_dir")
    ap.add_argument("--frames-every", type=int, default=20)
    args = ap.parse_args(argv)
    run = record(args.candidate, args.out_dir, args.frames_every)
    print(json.dumps({k: run.get(k) for k in ("painter", "model", "subject", "rating", "ops", "frames", "sim_steps")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
