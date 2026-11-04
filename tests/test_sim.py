"""Tests for the simulator: determinism, speed, conservation and the examples."""

import json
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from scipy.ndimage import binary_erosion

from backrun.cli import main as cli_main
from backrun.paper import make_paper
from backrun.pigments import PIGMENT_NAMES
from backrun.sim import Watercolor

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def wash(sim: Watercolor, pigments: dict[str, float], water: float = 0.9) -> None:
    """Cover the whole sheet with overlapping horizontal strokes."""
    for y in range(0, sim.height + 24, 24):
        sim.apply({"op": "stroke", "path": [[-10.0, float(y)], [sim.width + 10.0, float(y)]],
                   "width": 40.0, "water": water, "pigments": pigments})


def test_paper_is_seeded_and_in_range():
    a = make_paper(64, 48, seed=5)
    assert a.dtype == np.float32 and a.shape == (48, 64)
    assert a.min() >= 0.0 and a.max() <= 1.0
    assert np.array_equal(a, make_paper(64, 48, seed=5))
    assert not np.array_equal(a, make_paper(64, 48, seed=6))


def test_state_arrays_follow_the_design():
    sim = Watercolor(width=40, height=30, seed=1)
    st = sim.state
    assert set(st) == {"paper_height", "water", "wet_mask", "pigment_fluid",
                       "pigment_deposited", "flow_u", "flow_v"}
    for name in ("paper_height", "water", "flow_u", "flow_v"):
        assert st[name].shape == (30, 40) and st[name].dtype == np.float32
    assert st["wet_mask"].dtype == bool
    assert st["pigment_fluid"].shape == (len(PIGMENT_NAMES), 30, 40)
    assert st["pigment_deposited"].dtype == np.float32
    with pytest.raises(ValueError):
        st["water"][0, 0] = 1.0  # read-only views


def test_replay_is_byte_identical(tmp_path):
    path = EXAMPLES / "backrun.jsonl"
    first = Watercolor.from_jsonl(path).render()
    second = Watercolor.from_jsonl(path).render()
    assert first.dtype == np.uint8 and first.shape == (192, 256, 3)
    assert first.tobytes() == second.tobytes()
    # The same holds for the PNG written by the command line.
    out_a, out_b = tmp_path / "a.png", tmp_path / "b.png"
    cli_main(["render", str(path), str(out_a)])
    cli_main(["render", str(path), str(out_b)])
    assert out_a.read_bytes() == out_b.read_bytes()


def test_seed_changes_the_result(tmp_path):
    header, ops = Watercolor.read_jsonl(EXAMPLES / "flat_wash.jsonl")
    lines = [json.dumps({**header, "seed": header["seed"] + 1})] + [json.dumps(o) for o in ops]
    other = tmp_path / "other.jsonl"
    other.write_text("\n".join(lines) + "\n")
    a = Watercolor.from_jsonl(EXAMPLES / "flat_wash.jsonl").render()
    b = Watercolor.from_jsonl(other).render()
    assert a.tobytes() != b.tobytes()


def test_copy_is_independent_and_replays_the_same():
    sim = Watercolor(width=64, height=48, seed=2)
    sim.apply({"op": "stroke", "path": [[5.0, 20.0], [60.0, 25.0]], "width": 14.0,
               "water": 0.9, "pigments": {"burnt_umber": 0.5}})
    twin = sim.copy()
    stroke = {"op": "stroke", "path": [[30.0, 5.0], [32.0, 44.0]], "width": 8.0,
              "water": 0.8, "pigments": {"indian_red": 0.6}}
    sim.apply(stroke)
    sim.step(20)
    assert not np.array_equal(sim.water, twin.water)
    twin.apply(stroke)
    twin.step(20)
    assert sim.render().tobytes() == twin.render().tobytes()


def test_step_speed():
    """One step on a fully wet 256 x 192 sheet with three pigments."""
    sim = Watercolor(width=256, height=192, seed=1)
    wash(sim, {"quinacridone_rose": 0.3, "french_ultramarine": 0.3, "burnt_umber": 0.2})
    assert sim.wet_mask.all()
    sim.step(5)
    times = []
    for _ in range(60):
        start = time.perf_counter()
        sim.step()
        times.append(time.perf_counter() - start)
    median_ms = 1e3 * float(np.median(times))
    print(f"\nstep() at 256x192, fully wet, 3 pigments: median {median_ms:.2f} ms, "
          f"min {1e3 * min(times):.2f} ms")
    assert median_ms < 25.0


def test_pigment_is_conserved():
    """Fluid plus deposited pigment stays constant while the physics runs."""
    sim = Watercolor(width=128, height=96, seed=4)
    sim.apply({"op": "stroke", "path": [[10.0, 30.0], [64.0, 50.0], [118.0, 40.0]], "width": 30.0,
               "water": 0.9, "pigments": {"french_ultramarine": 0.5, "cadmium_yellow": 0.3}})
    sim.apply({"op": "stroke", "path": [[60.0, 10.0], [64.0, 86.0]], "width": 16.0,
               "water": 1.0, "pigments": {"quinacridone_rose": 0.7}})
    before = sim.total_pigment()
    assert np.any(sim.pigment_fluid > 0)
    for _ in range(40):
        sim.step(10)
        np.testing.assert_allclose(sim.total_pigment(), before, rtol=1e-5, atol=1e-6)
    assert np.all(sim.pigment_fluid >= 0) and np.all(sim.pigment_deposited >= 0)
    assert np.all(sim.water >= 0)


def test_sheet_dries_completely():
    sim = Watercolor.from_jsonl(EXAMPLES / "flat_wash.jsonl")
    assert not sim.wet_mask.any()
    assert float(sim.pigment_fluid.sum()) == 0.0
    assert float(sim.pigment_deposited.sum()) > 0.0


def _darkness(image: np.ndarray) -> np.ndarray:
    return 1.0 - image.astype(np.float64).mean(axis=-1) / 255.0


def test_flat_wash_has_darker_rim():
    """Edge darkening: the outer ring of the wash is darker than its interior."""
    sim = Watercolor.from_jsonl(EXAMPLES / "flat_wash.jsonl")
    footprint = sim.pigment_deposited.sum(axis=0) > 0.02
    rim = footprint & ~binary_erosion(footprint, iterations=3)
    interior = binary_erosion(footprint, iterations=12)
    assert rim.sum() > 100 and interior.sum() > 1000
    for image in (sim.render(), np.asarray(Image.open(EXAMPLES / "flat_wash.png").convert("RGB"))):
        dark = _darkness(image)
        print(f"\nflat wash darkness: rim {dark[rim].mean():.3f}, interior {dark[interior].mean():.3f}")
        assert dark[rim].mean() > dark[interior].mean() * 1.15


def test_examples_exist_at_the_right_size():
    for name in ("flat_wash", "wet_in_wet", "backrun"):
        header, ops = Watercolor.read_jsonl(EXAMPLES / f"{name}.jsonl")
        assert header["op"] == "canvas" and ops
        image = Image.open(EXAMPLES / f"{name}.png")
        assert image.size == (256, 192)


def test_wet_in_wet_is_softer_than_wet_on_dry():
    """A stroke laid into a wet wash spreads further than the same stroke on dry paper."""
    def spread(prewet: bool) -> float:
        sim = Watercolor(width=128, height=96, seed=9)
        if prewet:
            wash(sim, {}, water=1.0)
        sim.apply({"op": "stroke", "path": [[64.0, 20.0], [64.0, 76.0]], "width": 10.0,
                   "water": 0.9, "pigments": {"quinacridone_rose": 0.8}})
        sim.step(800)
        rose = sim.pigment_deposited[PIGMENT_NAMES.index("quinacridone_rose")]
        return float((rose > 0.02).sum())

    assert spread(True) > 1.2 * spread(False)


def test_bad_ops_are_rejected():
    sim = Watercolor(width=32, height=32, seed=0)
    bad = [
        {"op": "paint"},
        {"op": "canvas", "width": 32, "height": 32},
        {"op": "dry", "steps": -1},
        {"op": "stroke", "path": [[1.0, 1.0]], "width": 4.0, "water": 0.5, "pigments": {}},
        {"op": "stroke", "path": [[1.0, 1.0]] * 9, "width": 4.0, "water": 0.5, "pigments": {}},
        {"op": "stroke", "path": [[1.0, 1.0], [5.0, 5.0]], "width": 4.0, "water": 1.5, "pigments": {}},
        {"op": "stroke", "path": [[1.0, 1.0], [5.0, 5.0]], "width": 4.0, "water": 0.5,
         "pigments": {"unobtainium": 0.5}},
    ]
    for op in bad:
        with pytest.raises((ValueError, KeyError)):
            sim.apply(op)


def test_cli_writes_frames(tmp_path):
    frames = tmp_path / "frames"
    out = tmp_path / "out.png"
    cli_main(["render", str(EXAMPLES / "wet_in_wet.jsonl"), str(out),
              "--frames", str(frames), "--every", "500"])
    assert out.exists()
    names = sorted(p.name for p in frames.glob("*.png"))
    assert len(names) >= 3
    assert Image.open(frames / names[-1]).size == (256, 192)


def test_granulating_pigment_collects_in_valleys():
    """French ultramarine (strongly granulating) ends up thicker in paper valleys."""
    sim = Watercolor.from_jsonl(EXAMPLES / "flat_wash.jsonl")
    blue = sim.pigment_deposited[PIGMENT_NAMES.index("french_ultramarine")]
    interior = binary_erosion(blue > 0.02, iterations=12)
    paper = sim.paper_height[interior]
    amount = blue[interior]
    low, high = np.percentile(paper, [25, 75])
    print(f"\nultramarine in valleys {amount[paper < low].mean():.3f}, "
          f"on peaks {amount[paper > high].mean():.3f}")
    assert amount[paper < low].mean() > 1.2 * amount[paper > high].mean()
