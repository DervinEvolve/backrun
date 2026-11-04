"""Tests for train.ledgers: search-painter ledgers at scale, resumable."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from backrun.sim import Watercolor
from train import data, ledgers

SMALL = dict(width=24, height=18, strokes=4, candidates=3, phase_dry_steps=300)


@pytest.fixture(scope="module")
def targets(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Three training targets (one per synthetic kind) with no sealed set around."""
    root = tmp_path_factory.mktemp("targets")
    data.write_training_set(root, seeds_per_kind=1, sealed_dir=root / "no_sealed_here")
    return root


def _check_run(run_dir: Path) -> None:
    """A run folder is complete and its stroke file replays to its final image."""
    run = json.loads((run_dir / "run.json").read_text())
    assert run["status"] == "finished"
    assert run["canvas"] == {"width": 24, "height": 18, "seed": run["seed"], "paper": "cold_press"}
    lines = [json.loads(x) for x in (run_dir / "ledger.jsonl").read_text().splitlines()]
    assert lines and all({"index", "op", "score", "predicted_score", "phase"} <= set(x) for x in lines)
    replayed = Watercolor.from_jsonl(run_dir / "strokes.jsonl").render()
    assert np.array_equal(replayed, np.asarray(Image.open(run_dir / "final.png")))
    header, ops = Watercolor.read_jsonl(run_dir / "strokes.jsonl")
    assert [x["op"] for x in lines] == ops


def test_two_paintings_in_parallel_then_resume(targets: Path, tmp_path: Path) -> None:
    out = tmp_path / "ledgers"
    summary = ledgers.generate(targets, out, workers=2, limit=2, seed=7, **SMALL)
    assert summary["done"] == 2 and summary["failed"] == [] and summary["skipped_already_done"] == 0
    index = ledgers.read_index(out)
    assert len(index) == 2
    for row in index.values():
        assert row["status"] == "done"
        _check_run(Path(row["run_dir"]))
    # Second run: nothing to do.
    again = ledgers.generate(targets, out, workers=2, limit=2, seed=7, **SMALL)
    assert again["ran"] == 0 and again["skipped_already_done"] == 2
    # A painting whose folder was cut short is redone; the other is kept.
    victim = sorted(index.values(), key=lambda r: r["name"])[0]
    (Path(victim["run_dir"]) / "run.json").write_text('{"status": "running"}')
    third = ledgers.generate(targets, out, workers=1, limit=2, seed=7, **SMALL)
    assert third["ran"] == 1 and third["skipped_already_done"] == 1
    _check_run(Path(victim["run_dir"]))


def test_results_do_not_depend_on_workers(targets: Path, tmp_path: Path) -> None:
    a = ledgers.generate(targets, tmp_path / "a", workers=1, limit=2, seed=7, **SMALL)
    assert a["done"] == 2
    b_jobs, _ = ledgers.plan_jobs(targets, tmp_path / "b", limit=2, seed=7, **SMALL)
    ledgers.run_jobs(b_jobs[::-1], tmp_path / "b", workers=1)
    for name in ledgers.read_index(tmp_path / "a"):
        sa = (tmp_path / "a" / name / "strokes.jsonl").read_bytes()
        sb = (tmp_path / "b" / name / "strokes.jsonl").read_bytes()
        assert sa == sb


def test_painting_seed_is_stable() -> None:
    assert ledgers.painting_seed(7, "syn_sky_0000001") == ledgers.painting_seed(7, "syn_sky_0000001")
    assert ledgers.painting_seed(7, "a") != ledgers.painting_seed(8, "a")


def test_refuses_sealed_targets(tmp_path: Path) -> None:
    sealed = data.write_sealed_set(tmp_path / "sealed", n=3)
    with pytest.raises(data.SealedSetError):
        ledgers.plan_jobs(sealed, tmp_path / "out", limit=1, **SMALL)
    with pytest.raises(data.SealedSetError):
        ledgers.plan_jobs(tmp_path, sealed / "ledgers", limit=1, **SMALL)
