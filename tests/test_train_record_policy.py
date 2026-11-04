"""The policy recorder writes a run folder like cli_paint's, and its strokes replay to final.png."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from backrun.sim import Watercolor  # noqa: E402
from train import policy, record_policy  # noqa: E402

TINY = policy.PolicyConfig(widths=(8, 8, 8, 8), blocks=(1, 1, 1, 1), hidden=16, groups=4)


def test_record_policy_writes_a_replayable_run(tmp_path: Path) -> None:
    ckpt = policy.save_checkpoint(tmp_path / "tiny.pt", policy.make_policy(TINY, seed=0))
    out = tmp_path / "run"
    code = record_policy.main([str(ckpt), "synthetic:horizon:1", str(out), "--strokes", "3", "--width", "32",
                               "--height", "24", "--frames-every", "5"])
    assert code == 0
    for f in ("ledger.jsonl", "run.json", "strokes.jsonl", "target.png", "final.png"):
        assert (out / f).is_file(), f
    frames = sorted((out / "frames").glob("frame_*.png"))
    assert frames

    lines = (out / "strokes.jsonl").read_text().splitlines()
    header, ops = json.loads(lines[0]), [json.loads(x) for x in lines[1:]]
    ledger = [json.loads(x) for x in (out / "ledger.jsonl").read_text().splitlines()]
    assert len(ledger) == len(ops)
    assert [line["op"] for line in ledger] == ops
    strokes = [line for line in ledger if line["op"]["op"] == "stroke"]
    assert len(strokes) == 3
    for line in ledger:
        assert "score" in line and "predicted_score" in line and line["phase"] in policy.PHASE_NAMES
        assert "wet_pixels" in line
    for line in strokes:
        assert line["candidates_considered"] == 1 and line["rejected"] == []
        assert line["pigments_used"] == line["op"]["pigments"]
        assert "pigment_added" in line
    assert ledger[-1]["dry_kind"] == "final" and ledger[-1]["wet_pixels"] == 0

    run = json.loads((out / "run.json").read_text())
    assert run["painter"] == "policy" and run["decode"] == "argmax"
    assert run["policy_format"] == policy.CHECKPOINT_FORMAT
    assert run["checkpoint"] == str(ckpt) and len(run["checkpoint_sha256"]) == 64
    assert run["ops"] == len(ops) and run["frames"] == len(frames)
    assert run["sim_steps"] == sum(1 if o["op"] == "stroke" else o["steps"] for o in ops)
    assert run["final_score"]["total"] == pytest.approx(ledger[-1]["score"]["total"])

    sim = Watercolor(width=header["width"], height=header["height"], seed=header["seed"], paper=header["paper"])
    for op in ops:
        sim.apply(op)
    replayed = tmp_path / "replayed.png"
    Image.fromarray(sim.render()).save(replayed)
    assert replayed.read_bytes() == (out / "final.png").read_bytes()
    assert np.array_equal(np.asarray(Image.open(frames[-1])), sim.render())
