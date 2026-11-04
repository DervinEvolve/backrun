"""Record a learned-policy painting as a run folder, the way ``backrun.cli_paint`` records the search painter.

Usage::

    python -m train.record_policy CHECKPOINT TARGET OUT_DIR [--strokes 40] [--seed 7]
        [--frames-every 20] [--width 256 --height 192] [--decode argmax|sample] [--temperature 1.0]

``TARGET`` is what ``cli_paint`` accepts (an image file, ``synthetic:KIND[:SEED]``) or
``sealed:NAME`` for one of the sealed evaluation targets (``sealed:sealed_003``, or just
``sealed:3``). Recording a painting of a sealed target is evaluation, not training.

The rollout is ``train.policy.rollout``: one observation and one stroke per slot, the
policy's own wait as a ``dry`` op after the stroke, phases from ``phase_schedule``. After
the last slot the sheet is dried until no cell is wet (at most ``MAX_FINAL_DRY`` steps),
as ``train.evaluate`` does before it scores, so the final score is on the dried sheet.

``OUT_DIR`` gets the files ``cli_paint`` writes: ``strokes.jsonl``, ``ledger.jsonl``,
``run.json``, ``target.png``, ``final.png`` and ``frames/``. Every stroke line carries
``candidates_considered`` 1 and an empty ``rejected`` list, since the policy has no
candidates; ``search_s`` is the time it took to observe and act. The policy's waits are
``dry`` lines with ``dry_kind`` ``settle``; the last dry is ``final``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from backrun.cli_paint import resolve_target as resolve_paint_target
from backrun.painter_search import _pigment_totals, _wet_pixels
from backrun.record import Recorder
from backrun.score import score
from backrun.targets import letterbox

#: Most steps of the dry after the last slot (the same cap as ``train.evaluate``).
MAX_FINAL_DRY = 10_000
#: Steps per block of the final dry, as in ``train.evaluate.dry_until_set``.
FINAL_BLOCK = 25


def resolve_target(spec: str, width: int, height: int) -> NDArray[np.uint8]:
    """Load ``sealed:NAME`` from the sealed set, anything else as ``cli_paint`` does."""
    if not spec.startswith("sealed:"):
        return resolve_paint_target(spec, width, height)
    from train.data import load_sealed_set

    want = spec.split(":", 1)[1]
    if want.isdigit():
        want = f"sealed_{int(want):03d}"
    for name, img in load_sealed_set():
        if name == want:
            return img if img.shape[:2] == (height, width) else letterbox(img, width, height)
    raise SystemExit(f"no sealed target named {want!r}")


def sha256_of(path: str | Path) -> str:
    """Hex sha256 of a file."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _round_score(s: dict[str, Any]) -> dict[str, Any]:
    return {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in s.items()}


def record(
    policy: Callable[[NDArray[np.float32]], Any],
    target: NDArray[np.uint8],
    sim_factory: Callable[[], Any],
    n_strokes: int,
    recorder: Recorder,
    max_final_dry: int = MAX_FINAL_DRY,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Roll ``policy`` out on a fresh sheet through ``recorder``; return the ops and run stats.

    ``recorder`` must already be started. The ops are those ``train.policy.rollout`` applies,
    followed by the final dry.
    """
    from train import policy as pol

    sim = sim_factory()
    size = (int(target.shape[1]), int(target.shape[0]))
    every = recorder.frames_every
    phase_names = list(pol.PHASE_NAMES)
    sim_names = list(getattr(sim, "pigment_names", ()) or ())
    ops: list[dict[str, Any]] = []
    strokes = 0
    look_s = 0.0
    stats_phase: dict[str, float] = {}

    def run_steps(n: int) -> None:
        while n > 0:
            k = min(n, every)
            sim.step(k)
            recorder.advance(sim, k)
            n -= k

    def log(op: dict[str, Any], phase: str, **extra: Any) -> dict[str, Any]:
        real = dict(score(sim.render(), target, strokes=strokes))
        wet = bool(np.any(sim.state["wet_mask"]))
        predicted = dict(score(sim.render(predict_dry=True), target, strokes=strokes)) if wet else real
        recorder.log_op(op, real, phase=phase, phase_index=phase_names.index(phase), phase_count=len(phase_names),
                        predicted_score=_round_score(predicted), wet_pixels=_wet_pixels(sim), **extra)
        return predicted

    phase = phase_names[0]
    predicted: dict[str, Any] = {}
    for phase in pol.phase_schedule(n_strokes):
        tp = time.perf_counter()
        action = policy(pol.observe(sim, target, phase))
        slot_ops = pol.policy_output_to_ops(action, size)
        looked = time.perf_counter() - tp
        look_s += looked
        for op in slot_ops:
            ops.append(op)
            if op["op"] == "stroke":
                before = _pigment_totals(sim)
                sim.apply(op)
                recorder.advance(sim, 1)
                strokes += 1
                after = _pigment_totals(sim)
                added = None
                if before is not None and after is not None and len(sim_names) == after.size:
                    added = {n: round(float(after[sim_names.index(n)] - before[sim_names.index(n)]), 3)
                             for n in op["pigments"] if n in sim_names}
                predicted = log(op, phase, pigments_used=dict(op["pigments"]), pigment_added=added,
                                candidates_considered=1, search_s=round(looked, 4), rejected=[])
            else:
                run_steps(int(op["steps"]))
                predicted = log(op, phase, dry_kind="settle")
        stats_phase[phase] = stats_phase.get(phase, 0.0) + time.perf_counter() - tp

    # the final dry, in the blocks train.evaluate uses, each split into frame chunks
    expected = _round_score(predicted) if predicted else None
    ran = 0
    while ran < max_final_dry and bool(np.any(sim.state["wet_mask"])):
        k = min(FINAL_BLOCK, max_final_dry - ran)
        run_steps(k)
        ran += k
    if ran:
        op = {"op": "dry", "steps": ran}
        ops.append(op)
        log(op, phase, dry_kind="final", predicted_before_dry=expected)
    stats = {"strokes_accepted": strokes, "slots_skipped": 0, "evaluations": strokes, "looks": n_strokes,
             "look_s": look_s, "phase_s": stats_phase, "final_dry_steps": ran}
    return ops, stats


def build_parser() -> argparse.ArgumentParser:
    """Command-line options."""
    ap = argparse.ArgumentParser(prog="python -m train.record_policy", description=__doc__.split("\n\n")[0])
    ap.add_argument("checkpoint", help="policy checkpoint (train.policy.save_checkpoint format)")
    ap.add_argument("target", help="image file, synthetic:KIND[:SEED] or sealed:NAME")
    ap.add_argument("out_dir", help="run folder to write")
    ap.add_argument("--strokes", type=int, default=40, help="stroke slots to fill")
    ap.add_argument("--seed", type=int, default=7, help="seed for the paper and, with --decode sample, the draws")
    ap.add_argument("--frames-every", type=int, default=20, help="simulator steps between frames")
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--height", type=int, default=192)
    ap.add_argument("--decode", choices=("argmax", "sample"), default="argmax")
    ap.add_argument("--temperature", type=float, default=1.0, help="sampling temperature, with --decode sample")
    ap.add_argument("--paper", default="cold_press")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    """Roll the policy out and write the run folder. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    import torch

    from backrun.sim import Watercolor
    from train import policy as pol

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    target = resolve_target(args.target, args.width, args.height)
    blob = torch.load(Path(args.checkpoint), map_location="cpu", weights_only=True)
    pol.check_checkpoint_format(blob, args.checkpoint)
    model = pol.load_checkpoint(args.checkpoint)
    if args.decode == "sample":
        gen = torch.Generator().manual_seed(int(args.seed))
        policy = lambda o: pol.act(model, o, mode="sample", temperature=args.temperature, generator=gen)  # noqa: E731
    else:
        policy = lambda o: pol.act(model, o)  # noqa: E731
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
        candidates_per_stroke=1,
        painter="policy",
        checkpoint=str(args.checkpoint),
        checkpoint_sha256=sha256_of(args.checkpoint),
        policy_format=blob.get("format"),
        policy_parameters=int(sum(p.numel() for p in model.parameters())),
        decode=args.decode,
        temperature=args.temperature if args.decode == "sample" else None,
    )
    rec.save_frame(factory().render())
    t0 = time.perf_counter()
    ops, stats = record(policy, target, factory, args.strokes, rec)
    elapsed = time.perf_counter() - t0

    with (out / "strokes.jsonl").open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(canvas) + "\n")
        for op in ops:
            fh.write(json.dumps(op) + "\n")

    replay = factory()
    for op in ops:
        replay.apply(op)
    final = replay.render()
    final_score = score(final, target, strokes=stats["strokes_accepted"])
    stats["total_s"] = elapsed
    stats["final_score"] = final_score
    rec.time("look_s", stats["look_s"])
    for k, v in stats["phase_s"].items():
        rec.time(f"phase_{k}_s", v)
    rec.time("paint_s", elapsed)
    rec.finish(final, final_score=final_score, painter_stats=stats)
    print(
        f"{stats['strokes_accepted']} strokes from {stats['looks']} looks, final dry {stats['final_dry_steps']} steps, "
        f"{elapsed:.1f} s; total {final_score['total']:.3f} (lab_l2 {final_score['lab_l2']:.1f}, "
        f"ssim {final_score['ssim']:.3f})"
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
