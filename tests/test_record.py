"""Tests for the run recorder and the GIF script."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from backrun.record import Recorder
from fake_sim import FakeWatercolor

ROOT = Path(__file__).resolve().parents[1]


def _run(tmp_path: Path) -> Path:
    rec = Recorder(tmp_path / "run", frames_every=5)
    target = np.full((12, 16, 3), 200, dtype=np.uint8)
    rec.start(canvas={"width": 16, "height": 12}, seed=3, target_path="t.png", target_image=target)
    sim = FakeWatercolor(16, 12, 3)
    rec.save_frame(sim.render())
    stroke = {"op": "stroke", "path": [[1, 1], [10, 8]], "width": 3, "water": 0.5, "pigments": {"cadmium_yellow": 0.5}}
    sim.apply(stroke)
    rec.advance(sim, 1)
    rec.log_op(stroke, {"total": np.float64(1.5), "strokes": np.int64(1)})
    sim.step(12)
    rec.advance(sim, 12)
    rec.log_op({"op": "dry", "steps": 12}, {"total": 1.25, "strokes": 1})
    rec.time("paint_s", 0.5)
    rec.finish(sim.render(), note="ok")
    return tmp_path / "run"


def test_recorder_writes_everything(tmp_path: Path) -> None:
    run = _run(tmp_path)
    assert (run / "target.png").is_file() and (run / "final.png").is_file()
    lines = [json.loads(x) for x in (run / "ledger.jsonl").read_text().splitlines()]
    assert [x["op"]["op"] for x in lines] == ["stroke", "dry"]
    assert lines[0]["score"]["total"] == 1.5 and lines[1]["index"] == 1
    assert lines[1]["sim_steps"] == 13
    info = json.loads((run / "run.json").read_text())
    assert info["canvas"] == {"width": 16, "height": 12}
    assert info["seed"] == 3 and info["target_path"] == "t.png"
    assert info["status"] == "finished" and info["note"] == "ok"
    assert info["timings"]["paint_s"] == 0.5 and "elapsed_s" in info["timings"]
    frames = sorted((run / "frames").glob("frame_*.png"))
    # initial frame, frame at step 1 (first due at 0), frame at step 13 (>= 5), final frame
    assert [f.name for f in frames] == [f"frame_{i:05d}.png" for i in range(len(frames))]
    assert len(frames) == 4


def test_frames_every_k_steps(tmp_path: Path) -> None:
    rec = Recorder(tmp_path / "r", frames_every=10)
    rec.start(canvas={"width": 8, "height": 8}, seed=0)
    sim = FakeWatercolor(8, 8, 0)
    for _ in range(45):
        rec.advance(sim, 1)
    # Frames at steps 1 (first), 10, 20, 30, 40.
    assert rec.frame_count == 5


def test_start_clears_previous_run(tmp_path: Path) -> None:
    _run(tmp_path)
    rec = Recorder(tmp_path / "run", frames_every=5)
    rec.start(canvas={"width": 16, "height": 12}, seed=4)
    assert (tmp_path / "run" / "ledger.jsonl").read_text() == ""
    assert not list((tmp_path / "run" / "frames").glob("*.png"))


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_make_gif(tmp_path: Path) -> None:
    run = _run(tmp_path)
    out = tmp_path / "run.gif"
    subprocess.run(["bash", str(ROOT / "scripts" / "make_gif.sh"), str(run / "frames"), str(out), "4"], check=True)
    assert out.is_file() and out.read_bytes()[:6] in (b"GIF89a", b"GIF87a")
