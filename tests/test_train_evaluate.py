"""Tests for train.evaluate: the sealed-set report."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from backrun.score import score
from backrun.sim import Watercolor
from train import data, evaluate, policy

SMALL = dict(width=24, height=18, n_strokes=4, limit=2, seeds=(7, 8))


@pytest.fixture(scope="module")
def sealed(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A small sealed set of four targets."""
    return data.write_sealed_set(tmp_path_factory.mktemp("data") / "sealed", n=4)


def _check_report(report: dict, out: Path, painter: str) -> None:
    assert report["painter"] == painter
    assert report["n_targets"] == 2 and report["n_paintings"] == 4
    on_disk = json.loads((out / "report.json").read_text())
    assert on_disk["summary"] == report["summary"]
    rows = report["paintings"]
    assert [(r["target"], r["seed"]) for r in rows] == [
        ("sealed_000", 7), ("sealed_000", 8), ("sealed_001", 7), ("sealed_001", 8)]
    for key, _ in evaluate.METRICS:
        vals = np.array([r[key] for r in rows], dtype=float)
        assert report["summary"][key]["mean"] == pytest.approx(vals.mean())
        assert report["summary"][key]["std"] == pytest.approx(vals.std(ddof=1))
    md = (out / "report.md").read_text()
    assert "| metric | mean | std |" in md
    for _, label in evaluate.METRICS:
        assert label in md
    for r in rows:
        assert r["dried"]
        assert r["dry_gap"] == pytest.approx(r["total"] - r["predicted_total"])
        # The saved stroke file replays to the dried sheet that was scored.
        sheet = Watercolor.from_jsonl(Path(r["run_dir"]) / "strokes.jsonl")
        assert not sheet.state["wet_mask"].any()
        target = dict(data.load_sealed_set(report["settings"]["sealed_dir"]))[r["target"]]
        from backrun.targets import letterbox

        s = score(sheet.render(), letterbox(target, 24, 18), strokes=r["strokes"])
        assert s["total"] == pytest.approx(r["total"])


def test_search_painter_report(sealed: Path, tmp_path: Path) -> None:
    painter = evaluate.SearchPainter(candidates=3, phase_dry_steps=300)
    report = evaluate.evaluate(painter, tmp_path / "a", sealed_dir=sealed, **SMALL)
    _check_report(report, tmp_path / "a", "search")
    rows = report["paintings"]
    # The search painter simulates more strokes than it lays.
    assert all(r["strokes_simulated"] > r["strokes"] for r in rows)
    assert all(r["sim_steps"] >= r["sheet_steps"] - r["final_dry_steps"] for r in rows)
    assert all(set(r["phase_gaps"]) <= set(policy.PHASE_NAMES) for r in rows)
    assert any(k.startswith("phase_gap_") for k in report["summary"])
    # Deterministic apart from timings, and the same with two workers.
    again = evaluate.evaluate(painter, tmp_path / "b", sealed_dir=sealed, workers=2, **SMALL)
    strip = lambda rs: [{k: v for k, v in r.items() if k not in ("wall_s", "run_dir")} for r in rs]  # noqa: E731
    assert strip(again["paintings"]) == strip(rows)


def _fixed_policy(obs: np.ndarray) -> np.ndarray:
    """A stand-in policy: a fixed stroke nudged by the error map."""
    a = np.linspace(0.15, 0.85, policy.ACTION_DIM)
    a[policy.WAIT] = 0.3
    return np.clip(a + 0.1 * float(obs[7].mean()), 0, 1)


def test_policy_painter_report(sealed: Path, tmp_path: Path) -> None:
    painter = evaluate.PolicyPainter(policy_fn=_fixed_policy, name="fixed")
    report = evaluate.evaluate(painter, tmp_path / "p", sealed_dir=sealed, **SMALL)
    _check_report(report, tmp_path / "p", "fixed")
    for r in report["paintings"]:
        assert r["strokes"] == 4 and r["strokes_simulated"] == 4  # one simulation per stroke


def test_policy_checkpoint_painter(sealed: Path, tmp_path: Path) -> None:
    pytest.importorskip("torch")
    ckpt = policy.save_checkpoint(tmp_path / "p.pt", policy.make_policy(seed=0))
    painter = evaluate.make_painter(f"policy:{ckpt}")
    report = evaluate.evaluate(painter, tmp_path / "c", sealed_dir=sealed, **SMALL)
    _check_report(report, tmp_path / "c", "policy:p.pt")


def test_bad_inputs(sealed: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        evaluate.make_painter("nonsense")
    with pytest.raises(FileNotFoundError):
        evaluate.make_painter("policy:/no/such/file.pt")
    with pytest.raises(data.SealedSetError):
        evaluate.evaluate(evaluate.SearchPainter(candidates=2), sealed / "out", sealed_dir=sealed, **SMALL)


def test_cli_small_run(sealed: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = evaluate.main(["search", str(tmp_path / "cli"), "--sealed", str(sealed), "--limit", "1", "--seeds", "7",
                          "--width", "24", "--height", "18", "--strokes", "3", "--candidates", "2"])
    assert code == 0
    assert "| total (dried, lower is better) |" in capsys.readouterr().out
    assert (tmp_path / "cli" / "report.md").is_file()
