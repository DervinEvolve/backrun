"""Record a painting run to a folder so it can be inspected and replayed.

A run folder holds:

``frames/``
    PNG snapshots, ``frame_00000.png`` and up, one every ``frames_every``
    simulator steps. An applied stroke counts as one step, so stretches of
    strokes with no drying still produce frames.
``ledger.jsonl``
    One JSON line per op, in order: the op itself and the score after it.
    Every line has ``index``, ``t`` (seconds since ``start``), ``sim_steps``
    (simulator steps counted so far), ``op`` and ``score`` (the score of the
    simulator's real render after the op). Whatever extra keyword arguments
    the caller passes to ``log_op`` are added as further keys.

    The search painter (``backrun.painter_search.paint``) adds these:

    On every line:

    ``phase``, ``phase_index``, ``phase_count``
        Name of the painting phase the op belongs to (``washes``,
        ``medium``, ``detail``), its position from 0, and how many phases
        the run has.
    ``predicted_score``
        The score of the simulator's dry preview after the op: what the
        painter expects the sheet to score once it has dried. Equal to
        ``score`` when the sheet is dry.
    ``wet_pixels``
        Number of wet cells after the op, or ``null`` if the simulator does
        not report a wet mask.

    On ``stroke`` lines (accepted strokes; skipped slots write no line):

    ``pigments_used``
        Pigment name to the concentration loaded on the brush.
    ``pigment_added``
        Pigment name to the amount actually added to the sheet (the change
        in the simulator's total pigment, in its units), or ``null`` if the
        simulator cannot report it.
    ``candidates_considered``
        How many candidate strokes were simulated and scored for this slot.
    ``search_s``
        Wall time spent simulating and scoring those candidates, seconds.
    ``candidate_score``
        The score that won the slot, as the painter judged it (dry preview
        by default).
    ``rejected``
        The best of the other candidates, at most 8, best first. Each is
        ``{"path", "width", "water", "pigments", "score"}`` with the same
        meanings as in a stroke op and the score as the painter judged it.

    On ``dry`` lines:

    ``dry_kind``
        ``settle`` for the few steps after each stroke, ``phase`` for the
        dry between phases, ``final`` for the last one.

    On the ``dry`` line that ends each phase, and the final one:

    ``predicted_before_dry``
        The predicted score just before the real dry, to compare with
        ``score``, which is re-measured on the dried sheet.
``run.json``
    Canvas size, seed, target path, timings and any extra run settings.

``scripts/make_gif.sh`` turns the frames folder into a GIF.

Recorder fields
---------------

The recorder itself writes only ``index``, ``t``, ``sim_steps``, ``op`` and
``score`` on each ledger line; every other key comes from the caller's
``log_op`` keyword arguments, so the ledger is a superset that older readers
can ignore. The search painter's extra keys, listed above, make a run
explainable without replaying it: ``predicted_score`` against ``score``
shows how far the dry preview was from the real sheet, ``pigment_added`` and
``wet_pixels`` show how much paint and water each stroke put down,
``candidates_considered`` and ``search_s`` show the search effort, and
``rejected`` (the best 8 losing candidates per accepted stroke) shows what
the painter turned down and by how much.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image


def _jsonable(value: Any) -> Any:
    """Convert NumPy scalars and arrays into plain Python for JSON."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


class Recorder:
    """Writes frames, a per-op ledger and a run summary into ``run_dir``.

    Typical use::

        rec = Recorder("runs/horizon", frames_every=20)
        rec.start(canvas={"width": 256, "height": 192}, seed=7, target_path="x.png")
        rec.log_op(op, score_dict)        # once per op, after applying it
        rec.advance(sim, steps)           # after running simulator steps
        rec.finish(final_image)

    The ledger file is truncated by ``start`` so a run folder always
    describes a single run.
    """

    def __init__(self, run_dir: str | Path, frames_every: int = 20) -> None:
        if frames_every <= 0:
            raise ValueError("frames_every must be positive")
        self.run_dir = Path(run_dir)
        self.frames_dir = self.run_dir / "frames"
        self.ledger_path = self.run_dir / "ledger.jsonl"
        self.run_path = self.run_dir / "run.json"
        self.frames_every = int(frames_every)
        self.steps = 0
        self.frame_count = 0
        self.op_count = 0
        self.info: dict[str, Any] = {}
        self.timings: dict[str, float] = {}
        self._t0 = time.perf_counter()
        self._started_wall = 0.0
        self._next_frame_at = 0

    def start(
        self,
        canvas: dict[str, Any],
        seed: int,
        target_path: str | Path | None = None,
        target_image: NDArray[np.uint8] | None = None,
        **extra: Any,
    ) -> None:
        """Create the run folder, clear any old ledger and frames, note settings."""
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        for old in self.frames_dir.glob("frame_*.png"):
            old.unlink()
        self.ledger_path.write_text("")
        self.steps = 0
        self.frame_count = 0
        self.op_count = 0
        self._next_frame_at = 0
        self._t0 = time.perf_counter()
        self._started_wall = time.time()
        self.info = {
            "canvas": dict(canvas),
            "seed": int(seed),
            "target_path": None if target_path is None else str(target_path),
            **extra,
        }
        if target_image is not None:
            Image.fromarray(np.asarray(target_image, dtype=np.uint8)).save(self.run_dir / "target.png")
        self._write_run_json(status="running")

    def save_frame(self, image: NDArray[np.uint8]) -> Path:
        """Write one frame unconditionally and return its path."""
        path = self.frames_dir / f"frame_{self.frame_count:05d}.png"
        Image.fromarray(np.asarray(image, dtype=np.uint8)).save(path)
        self.frame_count += 1
        return path

    def advance(self, sim: Any, steps: int = 1) -> None:
        """Count ``steps`` simulator steps and save a frame if one is due.

        ``sim`` is anything with a ``render()`` method returning uint8
        (H, W, 3). It is only rendered when a frame is actually written.
        """
        self.steps += int(steps)
        if self.steps >= self._next_frame_at:
            self.save_frame(sim.render())
            while self._next_frame_at <= self.steps:
                self._next_frame_at += self.frames_every

    def log_op(self, op: dict[str, Any], score: dict[str, Any] | None, **extra: Any) -> None:
        """Append one ledger line: the op, the score after it, and timing."""
        line = {
            "index": self.op_count,
            "t": round(time.perf_counter() - self._t0, 4),
            "sim_steps": self.steps,
            "op": _jsonable(op),
            "score": _jsonable(score),
            **_jsonable(extra),
        }
        with self.ledger_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(line) + "\n")
        self.op_count += 1

    def time(self, name: str, seconds: float) -> None:
        """Add ``seconds`` to the named timing bucket for ``run.json``."""
        self.timings[name] = self.timings.get(name, 0.0) + float(seconds)

    def finish(self, final_image: NDArray[np.uint8] | None = None, **extra: Any) -> None:
        """Save a last frame and the final image, and write ``run.json``."""
        if final_image is not None:
            self.save_frame(final_image)
            Image.fromarray(np.asarray(final_image, dtype=np.uint8)).save(self.run_dir / "final.png")
        self.info.update(extra)
        self._write_run_json(status="finished")

    def _write_run_json(self, status: str) -> None:
        """Write the run summary with current timings."""
        data = {
            **self.info,
            "status": status,
            "frames_every": self.frames_every,
            "frames": self.frame_count,
            "ops": self.op_count,
            "sim_steps": self.steps,
            "timings": {
                "started_unix": round(self._started_wall, 3),
                "elapsed_s": round(time.perf_counter() - self._t0, 3),
                **{k: round(v, 3) for k, v in self.timings.items()},
            },
        }
        self.run_path.write_text(json.dumps(_jsonable(data), indent=2) + "\n")
