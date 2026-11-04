"""Tests for the search painter, mostly against the fake simulator."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from backrun import painter_search
from backrun.painter_search import (
    FALLBACK_PIGMENTS,
    MixModel,
    catmull_rom,
    discover_pigments,
    paint,
    rgb01_to_lab,
)
from backrun.record import Recorder
from backrun.score import score
from backrun.targets import synthetic_target
from fake_sim import FAKE_PIGMENTS, FakeWatercolor

W, H = 48, 36


def _factory():
    return FakeWatercolor(W, H, 1)


def _replay(ops):
    sim = _factory()
    for op in ops:
        sim.apply(op)
    return sim.render()


def test_paint_improves_and_is_deterministic() -> None:
    target = synthetic_target("discs", W, H, 1)
    stats: dict = {}
    ops = paint(target, _factory, 12, 6, seed=5, pigments=FAKE_PIGMENTS, stats=stats)
    again = paint(target, _factory, 12, 6, seed=5, pigments=FAKE_PIGMENTS)
    other = paint(target, _factory, 12, 6, seed=6, pigments=FAKE_PIGMENTS)
    assert ops == again
    assert ops != other
    blank = score(_factory().render(), target)["total"]
    final = score(_replay(ops), target, strokes=stats["strokes_accepted"])
    assert final["total"] < blank
    assert final["total"] == pytest.approx(stats["final_score"]["total"])
    assert stats["strokes_accepted"] + stats["slots_skipped"] == 12
    assert stats["evaluations"] <= 12 * 6


def test_ops_are_valid_and_json_ready() -> None:
    target = synthetic_target("horizon", W, H, 1)
    ops = paint(target, _factory, 9, 4, seed=1, pigments=FAKE_PIGMENTS)
    json.dumps(ops)
    strokes = [o for o in ops if o["op"] == "stroke"]
    drys = [o for o in ops if o["op"] == "dry"]
    assert strokes and drys
    for o in strokes:
        assert 2 <= len(o["path"]) <= 8
        assert o["width"] > 0 and 0 <= o["water"] <= 1
        assert 1 <= len(o["pigments"]) <= 2
        assert all(name in FAKE_PIGMENTS and 0 < c <= 1 for name, c in o["pigments"].items())
        assert all(0 <= x <= W - 1 and 0 <= y <= H - 1 for x, y in o["path"])
    for o in drys:
        assert isinstance(o["steps"], int) and o["steps"] > 0
    assert ops[-1]["op"] == "dry"  # final dry


def test_phases_get_finer() -> None:
    target = synthetic_target("sky", 96, 72, 2)
    ops = paint(
        target, lambda: FakeWatercolor(96, 72, 0), 10, 4, seed=2, pigments=FAKE_PIGMENTS, final_dry_steps=0
    )
    widths = [o["width"] for o in ops if o["op"] == "stroke"]
    assert widths[0] > widths[-1]


def test_every_slot_skipped_when_nothing_helps() -> None:
    blank = _factory().render()
    stats: dict = {}
    ops = paint(blank, _factory, 5, 3, seed=0, pigments=FAKE_PIGMENTS, stats=stats, final_dry_steps=0)
    assert stats["strokes_accepted"] == 0
    assert all(o["op"] == "dry" for o in ops)


def test_without_calibration_uses_colour_table() -> None:
    target = synthetic_target("discs", W, H, 3)
    stats: dict = {}
    ops = paint(target, _factory, 6, 4, seed=3, pigments=FAKE_PIGMENTS, calibrate=False, stats=stats)
    assert not stats["calibrated"]
    assert any(o["op"] == "stroke" for o in ops)


def test_recorder_integration(tmp_path: Path) -> None:
    target = synthetic_target("horizon", W, H, 1)
    rec = Recorder(tmp_path, frames_every=1)
    rec.start(canvas={"width": W, "height": H}, seed=1, target_image=target)
    ops = paint(target, _factory, 6, 3, seed=1, recorder=rec, pigments=FAKE_PIGMENTS)
    rec.finish(_replay(ops))
    lines = [json.loads(x) for x in (tmp_path / "ledger.jsonl").read_text().splitlines()]
    assert [x["op"] for x in lines] == ops
    assert all("total" in x["score"] for x in lines)
    assert list((tmp_path / "frames").glob("*.png"))


def test_mix_model_finds_a_single_pigment() -> None:
    paper = np.array([250, 248, 242]) / 255.0
    colours = {k: np.asarray(v) / 255.0 for k, v in FAKE_PIGMENTS.items()}
    model = MixModel.from_colours(colours, paper)
    k = model.combos.index({"cadmium_yellow": 0.4})
    goal = paper * model.transmit[k]
    under = np.tile(paper, (10, 1))
    best = model.combos[model.ranked(under, rgb01_to_lab(np.tile(goal, (10, 1))), top=1)[0]]
    assert best == {"cadmium_yellow": 0.4}


def test_discover_pigments_falls_back() -> None:
    found = discover_pigments()
    assert found and all(v.shape == (3,) and 0 <= v.min() and v.max() <= 1 for v in found.values())
    assert set(FALLBACK_PIGMENTS) == {
        "quinacridone_rose", "cadmium_yellow", "french_ultramarine", "cerulean_blue",
        "burnt_umber", "hookers_green", "indian_red", "paynes_grey",
    }


def test_catmull_rom_passes_through_points() -> None:
    pts = np.array([[0.0, 0.0], [10.0, 5.0], [20.0, 0.0]])
    curve = catmull_rom(pts, per_segment=8)
    for p in pts:
        assert np.min(np.linalg.norm(curve - p, axis=1)) < 1e-9


def test_painter_does_not_read_scorer_internals() -> None:
    tree = ast.parse(Path(painter_search.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "backrun.score":
            assert [a.name for a in node.names] == ["score"]


def test_real_simulator_end_to_end() -> None:
    sim_mod = pytest.importorskip("backrun.sim")
    w, h = 64, 48
    target = synthetic_target("horizon", w, h, 1)

    def factory():
        return sim_mod.Watercolor(width=w, height=h, seed=7)

    stats: dict = {}
    ops = paint(target, factory, 4, 3, seed=7, stats=stats, phase_dry_steps=80)
    sim = factory()
    for op in ops:
        sim.apply(op)
    final = sim.render()
    assert final.shape == (h, w, 3) and final.dtype == np.uint8
    assert score(final, target, stats["strokes_accepted"])["total"] == pytest.approx(stats["final_score"]["total"])
    assert stats["final_score"]["total"] < score(factory().render(), target)["total"]
