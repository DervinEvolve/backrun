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


# ---------------------------------------------------------------------------
# Dry preview, light to dark, acceptance and recorder detail.
# ---------------------------------------------------------------------------


class PreviewFake(FakeWatercolor):
    """Fake simulator whose ``render`` takes ``predict_dry`` and counts its use."""

    previews = 0

    def render(self, predict_dry: bool = False) -> np.ndarray:  # type: ignore[override]
        if predict_dry:
            PreviewFake.previews += 1
        return super().render()


def _ledger(path: Path) -> list[dict]:
    return [json.loads(x) for x in (path / "ledger.jsonl").read_text().splitlines()]


def test_dry_preview_is_used_when_available() -> None:
    target = synthetic_target("discs", W, H, 1)
    PreviewFake.previews = 0
    paint(target, lambda: PreviewFake(W, H, 1), 4, 3, seed=1, pigments=FAKE_PIGMENTS, calibrate=False)
    assert PreviewFake.previews >= 4 * 3
    PreviewFake.previews = 0
    ops = paint(target, lambda: PreviewFake(W, H, 1), 4, 3, seed=1, pigments=FAKE_PIGMENTS, calibrate=False,
                preview="wet")
    # With the wet view candidates are not previewed; only each committed op
    # is, for the ledger's predicted score.
    assert PreviewFake.previews == len(ops)


def test_preview_and_accept_are_checked() -> None:
    target = synthetic_target("discs", W, H, 1)
    with pytest.raises(ValueError):
        paint(target, _factory, 2, 2, pigments=FAKE_PIGMENTS, preview="damp")
    with pytest.raises(ValueError):
        paint(target, _factory, 2, 2, pigments=FAKE_PIGMENTS, accept="maybe")


def test_phase_caps_concentration() -> None:
    target = synthetic_target("horizon", W, H, 1)
    phases = (
        painter_search.Phase("washes", 0.5, (0.2, 0.3), (0.5, 1.0), (0.8, 0.9), (3, 4), 0, max_conc=0.12),
        painter_search.Phase("detail", 0.5, (0.05, 0.1), (0.1, 0.3), (0.4, 0.5), (2, 3), 0, max_conc=1.0),
    )
    stats: dict = {}
    rec_ops = paint(target, _factory, 10, 4, seed=4, pigments=FAKE_PIGMENTS, phases=phases, stats=stats,
                    calibrate=False)
    first_dry = next(i for i, o in enumerate(rec_ops) if o["op"] == "dry")
    washes = [o for o in rec_ops[:first_dry] if o["op"] == "stroke"]
    assert washes
    for o in washes:
        assert sum(o["pigments"].values()) <= 0.12 + 1e-9


def test_mix_model_allowed_limits_search() -> None:
    paper = np.array([250, 248, 242]) / 255.0
    colours = {k: np.asarray(v) / 255.0 for k, v in FAKE_PIGMENTS.items()}
    model = MixModel.from_colours(colours, paper)
    assert model.total_conc.shape == (len(model.combos),)
    mask = model.allowed(0.1)
    assert mask.any() and not mask.all()
    dark = rgb01_to_lab(np.tile([0.1, 0.1, 0.2], (6, 1)))
    under = np.tile(paper, (6, 1))
    for i in model.ranked(under, dark, top=3, allowed=mask):
        assert sum(model.combos[i].values()) <= 0.1 + 1e-9
    # An empty mask falls back to every mix.
    assert model.ranked(under, dark, top=1, allowed=np.zeros_like(mask)) == model.ranked(under, dark, top=1)


def test_calibration_takes_any_number_of_concentrations() -> None:
    paper = np.array([0.96, 0.94, 0.9])
    concs = (0.05, 0.2, 0.5)
    measured = np.stack([paper * np.exp(-np.array([1.0, 0.5, 0.2]) * c) for c in concs])[None]
    model = MixModel.from_calibration(["cerulean_blue"], paper, measured, concs)
    k = model.combos.index({"cerulean_blue": 0.4})
    assert np.allclose(-np.log(model.transmit[k]), np.array([1.0, 0.5, 0.2]) * 0.4, atol=1e-9)
    with pytest.raises(ValueError):
        MixModel.from_calibration(["cerulean_blue"], paper, measured[:, :2], concs)


def test_recorder_gets_stroke_detail(tmp_path: Path) -> None:
    target = synthetic_target("discs", W, H, 2)
    rec = Recorder(tmp_path, frames_every=5)
    rec.start(canvas={"width": W, "height": H}, seed=2)
    ops = paint(target, _factory, 8, 12, seed=2, recorder=rec, pigments=FAKE_PIGMENTS)
    lines = _ledger(tmp_path)
    assert [x["op"] for x in lines] == ops
    strokes = [x for x in lines if x["op"]["op"] == "stroke"]
    assert strokes
    for x in lines:
        # Old fields unchanged, new ones present on every line.
        assert {"index", "t", "sim_steps", "op", "score"} <= set(x)
        assert {"phase", "predicted_score", "wet_pixels"} <= set(x)
    for x in strokes:
        assert x["pigments_used"] == x["op"]["pigments"]
        assert 1 <= x["candidates_considered"] <= 12
        rej = x["rejected"]
        assert len(rej) <= painter_search.REJECTED_KEPT
        assert len(rej) <= x["candidates_considered"] - 1
        totals = [r["score"]["total"] for r in rej]
        assert totals == sorted(totals)
        assert all(t >= x["candidate_score"]["total"] for t in totals)
        for r in rej:
            assert {"path", "width", "water", "pigments", "score"} <= set(r)
            assert 2 <= len(r["path"]) <= 8 and r["width"] > 0
    assert any(len(x["rejected"]) == painter_search.REJECTED_KEPT for x in strokes)
    json.dumps(lines)


def test_strokes_accepted_only_if_predicted_score_improves(tmp_path: Path) -> None:
    target = synthetic_target("horizon", W, H, 3)
    rec = Recorder(tmp_path, frames_every=5)
    rec.start(canvas={"width": W, "height": H}, seed=3)
    paint(target, _factory, 10, 5, seed=3, recorder=rec, pigments=FAKE_PIGMENTS)
    lines = _ledger(tmp_path)
    prev = score(_factory().render(), target)["total"]
    for x in lines:
        if x["op"]["op"] == "stroke":
            assert x["candidate_score"]["total"] < prev + 1e-9
        prev = x["predicted_score"]["total"]


def test_real_simulator_ledger_detail(tmp_path: Path) -> None:
    sim_mod = pytest.importorskip("backrun.sim")
    w, h = 64, 48
    target = synthetic_target("horizon", w, h, 1)

    def factory():
        return sim_mod.Watercolor(width=w, height=h, seed=7)

    rec = Recorder(tmp_path, frames_every=50)
    rec.start(canvas={"width": w, "height": h}, seed=7)
    stats: dict = {}
    ops = paint(target, factory, 6, 3, seed=7, recorder=rec, stats=stats)
    assert stats["preview"] == "dry"
    lines = _ledger(tmp_path)
    strokes = [x for x in lines if x["op"]["op"] == "stroke"]
    assert strokes
    for x in strokes:
        assert isinstance(x["wet_pixels"], int) and x["wet_pixels"] > 0
        assert set(x["pigment_added"]) == set(x["op"]["pigments"])
        assert all(v > 0 for v in x["pigment_added"].values())
    phase_drys = [x for x in lines if "predicted_before_dry" in x]
    assert phase_drys
    # After the last real dry the sheet is dry and the prediction is the truth.
    last = lines[-1]
    assert last["op"]["op"] == "dry" and last["wet_pixels"] == 0
    assert last["predicted_score"]["total"] == pytest.approx(last["score"]["total"], abs=1e-3)
    # Same seed, same ops.
    assert paint(target, factory, 6, 3, seed=7) == ops


def test_ledger_marks_phases_and_dry_kinds(tmp_path: Path) -> None:
    target = synthetic_target("sky", W, H, 1)
    rec = Recorder(tmp_path, frames_every=5)
    rec.start(canvas={"width": W, "height": H}, seed=1)
    paint(target, _factory, 9, 4, seed=1, recorder=rec, pigments=FAKE_PIGMENTS)
    lines = _ledger(tmp_path)
    assert all(x["phase_count"] == len(painter_search.PHASES) for x in lines)
    idx = [x["phase_index"] for x in lines]
    assert idx == sorted(idx)
    assert all(x["phase"] == painter_search.PHASES[x["phase_index"]].name for x in lines)
    kinds = [x["dry_kind"] for x in lines if x["op"]["op"] == "dry"]
    assert set(kinds) <= {"settle", "phase", "final"} and kinds[-1] == "final"
    assert all("dry_kind" not in x for x in lines if x["op"]["op"] == "stroke")
    assert all(x["search_s"] >= 0 for x in lines if x["op"]["op"] == "stroke")


def test_light_quantile_keeps_a_stroke_as_pale_as_its_palest_part() -> None:
    paper = np.array([250, 248, 242]) / 255.0
    colours = {k: np.asarray(v) / 255.0 for k, v in FAKE_PIGMENTS.items()}
    model = MixModel.from_colours(colours, paper)
    pale = paper * model.transmit[model.combos.index({"french_ultramarine": 0.07})]
    dark = paper * model.transmit[model.combos.index({"paynes_grey": 0.85})]
    target = rgb01_to_lab(np.vstack([np.tile(pale, (5, 1)), np.tile(dark, (5, 1))]))
    under = np.tile(paper, (10, 1))

    def mean_l(i: int) -> float:
        return float(rgb01_to_lab(paper * model.transmit[i])[0])

    free = model.ranked(under, target, top=1)[0]
    capped = model.ranked(under, target, top=1, light_quantile=0.2)[0]
    assert mean_l(capped) >= float(rgb01_to_lab(pale)[0]) - 2.0
    assert mean_l(free) < mean_l(capped)
    # Overshoot weighting also pushes the choice lighter.
    assert mean_l(model.ranked(under, target, top=1, overshoot=6.0)[0]) > mean_l(free)


def test_accept_view_can_differ_from_preview(tmp_path: Path) -> None:
    """With ``preview="dry", accept="wet"`` a stroke must lower the real (wet) score."""
    sim_mod = pytest.importorskip("backrun.sim")
    w, h = 64, 48
    target = synthetic_target("horizon", w, h, 1)

    def factory():
        return sim_mod.Watercolor(width=w, height=h, seed=7)

    rec = Recorder(tmp_path, frames_every=50)
    rec.start(canvas={"width": w, "height": h}, seed=7)
    stats: dict = {}
    paint(target, factory, 6, 3, seed=7, recorder=rec, stats=stats, preview="dry", accept="wet")
    assert stats["accept"] == "wet"
    lines = _ledger(tmp_path)
    prev = score(factory().render(), target)["total"]
    for i, x in enumerate(lines):
        if x["op"]["op"] == "stroke":
            after = lines[i + 1] if i + 1 < len(lines) and lines[i + 1].get("dry_kind") == "settle" else x
            assert after["score"]["total"] < prev + 1e-9
        prev = x["score"]["total"]
