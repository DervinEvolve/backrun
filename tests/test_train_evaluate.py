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


def test_collapse_stats() -> None:
    stroke = {"op": "stroke", "path": [[10.0, 5.0], [12.0, 6.0], [14.0, 7.0]], "width": 3.0, "water": 0.5,
              "pigments": {"cadmium_yellow": 0.2}}
    same = [stroke, {"op": "dry", "steps": 3}, dict(stroke, path=[[10.2, 5.1], [12.0, 6.0], [14.0, 7.0]])]
    st = evaluate.collapse_stats(same, 24, 18)
    assert st["distinct_strokes"] == 1 and st["start_std_x_px"] == pytest.approx(0.1)
    moved = same + [dict(stroke, path=[[20.0, 15.0], [12.0, 6.0], [14.0, 7.0]]),
                    dict(stroke, pigments={"cadmium_yellow": 0.3})]
    st = evaluate.collapse_stats(moved, 24, 18)
    assert st["distinct_strokes"] == 3
    xs = np.array([10.0, 10.2, 20.0, 10.0])
    assert st["start_std_x_px"] == pytest.approx(xs.std())
    assert evaluate.collapse_stats([{"op": "dry", "steps": 5}], 24, 18)["distinct_strokes"] == 0


def _fake_ledger_root(root: Path) -> Path:
    """Two finished run folders with hand-written stroke files (no replay is needed for the mean)."""
    strokes = [
        {"op": "stroke", "path": [[2.0, 3.0], [8.0, 3.0], [12.0, 9.0]], "width": 2.0, "water": 0.4,
         "pigments": {"french_ultramarine": 0.3}},
        {"op": "stroke", "path": [[20.0, 15.0], [18.0, 10.0]], "width": 4.0, "water": 0.8,
         "pigments": {"cadmium_yellow": 0.1}},
    ]
    for i, op in enumerate(strokes):
        run = root / f"run{i}"
        run.mkdir(parents=True)
        (run / "run.json").write_text(json.dumps({"status": "finished"}))
        header = {"op": "canvas", "width": 24, "height": 18, "seed": 1, "paper": "cold_press"}
        (run / "strokes.jsonl").write_text("".join(json.dumps(x) + "\n" for x in [header, op,
                                                                                {"op": "dry", "steps": 9 * (i + 1)}]))
    return root


def test_baseline_painters(sealed: Path, tmp_path: Path) -> None:
    blank = evaluate.make_painter("blank")
    report = evaluate.evaluate(blank, tmp_path / "blank", sealed_dir=sealed, **SMALL)
    _check_report(report, tmp_path / "blank", "blank")
    for r in report["paintings"]:
        assert r["strokes"] == 0 and r["distinct_strokes"] == 0 and r["sheet_steps"] >= 2000
    root = _fake_ledger_root(tmp_path / "ledgers")
    mean = evaluate.mean_action([root])
    a = policy.encode_stroke(json.loads((root / "run0" / "strokes.jsonl").read_text().splitlines()[1]), 9, (24, 18))
    b = policy.encode_stroke(json.loads((root / "run1" / "strokes.jsonl").read_text().splitlines()[1]), 18, (24, 18))
    assert np.allclose(mean, (a + b) / 2, atol=1e-6)
    painter = evaluate.make_painter(f"mean:{root}")
    assert painter.name == "mean:ledgers"
    report = evaluate.evaluate(painter, tmp_path / "mean", sealed_dir=sealed, **SMALL)
    _check_report(report, tmp_path / "mean", "mean:ledgers")
    for r in report["paintings"]:
        assert r["strokes"] == 4 and r["distinct_strokes"] == 1
        assert r["start_std_x_px"] == 0 and r["start_std_y_px"] == 0
    with pytest.raises(FileNotFoundError):
        evaluate.make_painter(f"mean:{tmp_path / 'nowhere'}")
    # Side by side, with the share of the blank-to-search gap closed.
    search = dict(report, painter="search", summary={k: dict(v) for k, v in report["summary"].items()})
    search["summary"]["total"]["mean"] = report["summary"]["total"]["mean"] - 10
    blank_report = evaluate.load_report(tmp_path / "blank")
    md = evaluate.compare_markdown([blank_report, report, search])
    assert "gap closed" in md and md.count("\n") == 5
    code = evaluate.main(["compare", str(tmp_path / "cmp.md"), str(tmp_path / "blank"), str(tmp_path / "mean")])
    assert code == 0 and "distinct strokes" in (tmp_path / "cmp.md").read_text()
    twice = evaluate.compare_markdown([blank_report, dict(blank_report, report_dir=str(tmp_path / "other"))])
    assert "| blank (blank) |" in twice and "| blank (other) |" in twice


def test_load_report_fills_in_the_collapse_detector(sealed: Path, tmp_path: Path) -> None:
    painter = evaluate.PolicyPainter(policy_fn=_fixed_policy, name="fixed")
    report = evaluate.evaluate(painter, tmp_path / "old", sealed_dir=sealed, **SMALL)
    old = json.loads((tmp_path / "old" / "report.json").read_text())
    for r in old["paintings"]:
        for k in ("start_std_x_px", "start_std_y_px", "distinct_strokes"):
            r.pop(k)
    (tmp_path / "old" / "report.json").write_text(json.dumps(old))
    again = evaluate.load_report(tmp_path / "old")
    assert again["summary"]["distinct_strokes"] == report["summary"]["distinct_strokes"]
