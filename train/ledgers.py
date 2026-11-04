"""Run the search painter on many targets and keep one run folder per painting.

These ledgers are Stage A's data: every accepted stroke, the state before
it and the score after it. The painting itself is done by
``backrun.painter_search.paint`` and recorded by ``backrun.record.Recorder``;
this module only fans the work out and keeps track of it.

Layout under the output root::

    index.jsonl            one line per finished painting (resumable)
    <target name>/         a run folder, as backrun.cli_paint writes it:
        strokes.jsonl      canvas header plus every op
        ledger.jsonl       every op with its scores (see backrun.record)
        run.json, target.png, final.png, frames/

A painting whose name is already in ``index.jsonl`` with status ``done`` and
whose ``run.json`` says ``finished`` is skipped, so an interrupted job picks
up where it stopped. Each painting's seed is derived from the base seed and
the target's name, so the result does not depend on worker count, order or
``--limit``.

Command line::

    python -m train.ledgers TARGETS_DIR OUT_ROOT [--strokes 40] [--candidates 16]
        [--workers N] [--limit N] [--width 256 --height 192] [--seed 7]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from backrun.painter_search import paint
from backrun.record import Recorder
from backrun.score import score
from backrun.targets import letterbox
from train import data

data.declare_training_module(__name__)

INDEX_NAME = "index.jsonl"


@dataclass(frozen=True)
class LedgerJob:
    """Settings for one painting."""

    name: str
    target_path: str
    run_dir: str
    width: int = 256
    height: int = 192
    strokes: int = 40
    candidates: int = 16
    seed: int = 7
    paper: str = "cold_press"
    frames_every: int = 1_000_000
    phase_dry_steps: int = 2000


def painting_seed(base_seed: int, name: str) -> int:
    """A seed for one painting, from the base seed and the target's name."""
    digest = hashlib.sha256(f"{base_seed}:{name}".encode()).digest()
    return int.from_bytes(digest[:4], "little") & 0x7FFFFFFF


def paint_one(job: LedgerJob) -> dict[str, Any]:
    """Paint one target into its run folder and return its index row.

    Mirrors ``backrun.cli_paint.main``: the search painter paints with a
    started ``Recorder``, the ops are written to ``strokes.jsonl`` behind a
    canvas header, and the final image is the replay of that file.
    """
    from backrun.sim import Watercolor

    t0 = time.perf_counter()
    image = data.load_training_image(job.target_path)
    target = image if image.shape[:2] == (job.height, job.width) else letterbox(image, job.width, job.height)
    out = Path(job.run_dir)
    out.mkdir(parents=True, exist_ok=True)
    canvas = {"op": "canvas", "width": job.width, "height": job.height, "seed": job.seed, "paper": job.paper}

    def factory() -> Watercolor:
        return Watercolor(width=job.width, height=job.height, seed=job.seed, paper=job.paper)

    rec = Recorder(out, frames_every=job.frames_every)
    rec.start(
        canvas={k: v for k, v in canvas.items() if k != "op"},
        seed=job.seed,
        target_path=job.target_path,
        target_image=target,
        strokes_requested=job.strokes,
        candidates_per_stroke=job.candidates,
        painter="search_v0",
        target_name=job.name,
    )
    rec.save_frame(factory().render())
    stats: dict[str, Any] = {}
    ops = paint(
        target, factory, job.strokes, job.candidates, job.seed, recorder=rec, stats=stats,
        phase_dry_steps=job.phase_dry_steps,
    )
    with (out / "strokes.jsonl").open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(canvas) + "\n")
        for op in ops:
            fh.write(json.dumps(op) + "\n")
    replay = factory()
    for op in ops:
        replay.apply(op)
    final = replay.render()
    final_score = score(final, target, strokes=stats.get("strokes_accepted", 0))
    elapsed = time.perf_counter() - t0
    rec.time("paint_s", elapsed)
    rec.finish(final, final_score=final_score, painter_stats=stats)
    return {
        "name": job.name,
        "status": "done",
        "run_dir": str(out),
        "seed": job.seed,
        "strokes": int(stats.get("strokes_accepted", 0)),
        "total": round(float(final_score["total"]), 4),
        "seconds": round(elapsed, 3),
    }


def _paint_safe(job: LedgerJob) -> dict[str, Any]:
    """``paint_one`` that reports a failure as a row instead of raising."""
    try:
        return paint_one(job)
    except data.SealedSetError:
        raise
    except Exception as exc:  # one bad target must not end a long job
        return {"name": job.name, "status": "failed", "run_dir": job.run_dir, "error": f"{type(exc).__name__}: {exc}"}


def read_index(out_root: str | Path) -> dict[str, dict[str, Any]]:
    """The index rows by name; a later row for the same name wins."""
    path = Path(out_root) / INDEX_NAME
    rows: dict[str, dict[str, Any]] = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            if line.strip():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a line cut short by an interrupted write
                rows[row["name"]] = row
    return rows


def is_done(row: dict[str, Any] | None) -> bool:
    """True if an index row marks a finished painting whose folder is intact."""
    if not row or row.get("status") != "done":
        return False
    run_json = Path(row["run_dir"]) / "run.json"
    try:
        return json.loads(run_json.read_text()).get("status") == "finished"
    except (OSError, json.JSONDecodeError):
        return False


def plan_jobs(
    targets_dir: str | Path,
    out_root: str | Path,
    *,
    width: int = 256,
    height: int = 192,
    strokes: int = 40,
    candidates: int = 16,
    seed: int = 7,
    limit: int | None = None,
    phase_dry_steps: int = 2000,
    frames_every: int = 1_000_000,
) -> tuple[list[LedgerJob], int]:
    """The jobs still to run, in index order, and how many were already done.

    ``limit`` caps the number of targets considered (done ones included), so
    ``--limit 2`` always means the first two targets of the index.
    """
    targets_dir = data.assert_not_sealed(targets_dir)
    out_root = Path(out_root)
    if data.sealed_folder_of(out_root) is not None:
        raise data.SealedSetError(f"refusing to write ledgers inside a sealed folder: {out_root}")
    rows = data.read_index(targets_dir)
    if limit is not None:
        rows = rows[: max(0, int(limit))]
    done = read_index(out_root)
    jobs: list[LedgerJob] = []
    skipped = 0
    for row in rows:
        name = Path(row["file"]).stem
        if is_done(done.get(name)):
            skipped += 1
            continue
        jobs.append(
            LedgerJob(
                name=name,
                target_path=str(Path(targets_dir) / row["file"]),
                run_dir=str(out_root / name),
                width=width,
                height=height,
                strokes=strokes,
                candidates=candidates,
                seed=painting_seed(seed, name),
                phase_dry_steps=phase_dry_steps,
                frames_every=frames_every,
            )
        )
    return jobs, skipped


def run_jobs(jobs: Sequence[LedgerJob], out_root: str | Path, workers: int = 1) -> list[dict[str, Any]]:
    """Run ``jobs``, appending each result to the index as it arrives.

    With ``workers`` above 1 the paintings run in a process pool (spawned
    processes, so the same code works on Linux, macOS and the Modal image).
    Only this process writes the index, one whole line at a time.
    """
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    index = out_root / INDEX_NAME
    results: list[dict[str, Any]] = []

    def record(row: dict[str, Any]) -> None:
        with index.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        results.append(row)

    if workers <= 1 or len(jobs) <= 1:
        for job in jobs:
            record(_paint_safe(job))
        return results
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=min(workers, len(jobs))) as pool:
        for row in pool.imap_unordered(_paint_safe, jobs, chunksize=1):
            record(row)
    return results


def generate(
    targets_dir: str | Path,
    out_root: str | Path,
    *,
    workers: int = 1,
    **settings: Any,
) -> dict[str, Any]:
    """Plan and run every outstanding painting; return a short summary."""
    t0 = time.perf_counter()
    jobs, skipped = plan_jobs(targets_dir, out_root, **settings)
    rows = run_jobs(jobs, out_root, workers=workers)
    return {
        "ran": len(rows),
        "done": sum(r["status"] == "done" for r in rows),
        "failed": [r for r in rows if r["status"] != "done"],
        "skipped_already_done": skipped,
        "seconds": round(time.perf_counter() - t0, 2),
    }


def build_parser() -> argparse.ArgumentParser:
    """Command-line options."""
    ap = argparse.ArgumentParser(prog="python -m train.ledgers", description="Generate search-painter ledgers.")
    ap.add_argument("targets_dir", help="training targets folder with index.jsonl (see train.data)")
    ap.add_argument("out_root", help="folder that gets one run folder per painting and index.jsonl")
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--height", type=int, default=192)
    ap.add_argument("--strokes", type=int, default=40)
    ap.add_argument("--candidates", type=int, default=16)
    ap.add_argument("--seed", type=int, default=7, help="base seed; each painting derives its own")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1)))
    ap.add_argument("--limit", type=int, default=None, help="only the first N targets (smoke tests)")
    ap.add_argument("--phase-dry-steps", type=int, default=2000)
    ap.add_argument("--frames-every", type=int, default=1_000_000, help="simulator steps between frames")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line. Returns a process exit code."""
    a = build_parser().parse_args(argv)
    summary = generate(
        a.targets_dir, a.out_root, workers=a.workers, width=a.width, height=a.height, strokes=a.strokes,
        candidates=a.candidates, seed=a.seed, limit=a.limit, phase_dry_steps=a.phase_dry_steps,
        frames_every=a.frames_every,
    )
    print(json.dumps(summary, indent=2))
    return 0 if not summary["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
