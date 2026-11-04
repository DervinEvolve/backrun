"""Command line: ``python -m backrun.cli render strokes.jsonl out.png [--frames DIR --every N]``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from backrun.render import save_png
from backrun.sim import Watercolor


def replay(strokes: Path, out: Path, frames: Path | None = None, every: int = 50) -> Watercolor:
    """Replay a stroke file, write the final PNG and optionally a frame every ``every`` steps.

    Frames are named ``frame_000000.png`` by step count. A frame is also
    written after every stroke so wet strokes show up the moment they land.
    """
    header, ops = Watercolor.read_jsonl(strokes)
    sim = Watercolor.from_header(header)
    if frames is not None:
        frames.mkdir(parents=True, exist_ok=True)
        if every < 1:
            raise ValueError("--every must be at least 1")

    def dump() -> None:
        if frames is not None:
            save_png(sim.render(), str(frames / f"frame_{sim.steps_taken:06d}_{sim.strokes_applied:03d}.png"))

    dump()
    for op in ops:
        if op.get("op") == "dry" and frames is not None:
            steps = op.get("steps")
            if not isinstance(steps, int) or steps < 0:
                raise ValueError("dry op needs a non-negative integer 'steps'")
            for _ in range(steps):
                sim.step()
                if sim.steps_taken % every == 0:
                    dump()
        else:
            sim.apply(op)
            if op.get("op") == "stroke":
                dump()
    save_png(sim.render(), str(out))
    return sim


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``backrun`` command."""
    parser = argparse.ArgumentParser(prog="backrun", description="Backrun watercolor simulator")
    sub = parser.add_subparsers(dest="command", required=True)
    render = sub.add_parser("render", help="replay a stroke file to a PNG")
    render.add_argument("strokes", type=Path, help="JSON Lines stroke file with a canvas header")
    render.add_argument("out", type=Path, help="output PNG path")
    render.add_argument("--frames", type=Path, default=None, help="directory for intermediate frames")
    render.add_argument("--every", type=int, default=50, help="steps between frames (default 50)")
    args = parser.parse_args(argv)
    if args.command == "render":
        sim = replay(args.strokes, args.out, args.frames, args.every)
        print(f"{args.out}: {sim.width}x{sim.height}, {sim.strokes_applied} strokes, "
              f"{sim.steps_taken} steps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
