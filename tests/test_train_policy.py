"""Tests for train.policy: observation, stroke format, ledger examples, network."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from backrun.sim import Watercolor
from backrun.targets import synthetic_target
from train import data, ledgers, policy
from train.policy import ACTION_DIM, OBS_CHANNELS, OBS_HEIGHT, OBS_WIDTH


def _replays_unchanged(ops: list[dict], width: int, height: int, seed: int, tmp_path: Path) -> None:
    """Applying ops directly and replaying them from a JSONL file give the same bytes."""
    live = Watercolor(width=width, height=height, seed=seed)
    for op in ops:
        live.apply(op)
    header = {"op": "canvas", "width": width, "height": height, "seed": seed, "paper": "cold_press"}
    path = tmp_path / "policy.jsonl"
    path.write_text("".join(json.dumps(x) + "\n" for x in [header, *ops]))
    _, read_back = Watercolor.read_jsonl(path)
    assert read_back == ops
    assert np.array_equal(Watercolor.from_jsonl(path).render(), live.render())


def test_observation_planes() -> None:
    target = synthetic_target("horizon", 256, 192, 3)
    sim = Watercolor(width=256, height=192, seed=1)
    sim.apply({"op": "stroke", "path": [[20, 20], [120, 60], [200, 40]], "width": 20.0, "water": 0.9,
               "pigments": {"french_ultramarine": 0.3}})
    obs = policy.observe(sim, target, "medium")
    assert obs.shape == (OBS_CHANNELS, OBS_HEIGHT, OBS_WIDTH) and obs.dtype == np.float32
    # Target planes: a 2 x 2 mean of the target.
    expect = target.astype(np.float32).reshape(96, 2, 128, 2, 3).mean(axis=(1, 3)) / 255.0
    assert np.allclose(np.moveaxis(obs[0:3], 0, -1), expect, atol=1e-5)
    # Dry render planes come from the dry preview, not the wet render.
    dry = sim.render(predict_dry=True).astype(np.float32).reshape(96, 2, 128, 2, 3).mean(axis=(1, 3)) / 255.0
    assert np.allclose(np.moveaxis(obs[3:6], 0, -1), dry, atol=1e-5)
    assert obs[6].max() == 1.0 and obs[6].min() == 0.0  # wet where the stroke is
    assert obs[7].min() >= 0.0 and obs[7].max() > 0.05
    assert np.all(obs[8] == 0) and np.all(obs[9] == 1) and np.all(obs[10] == 0)
    # Error map is zero when the sheet equals the target; odd canvas sizes resize.
    same = policy.build_observation(target[:18, :24], target[:18, :24], np.zeros((18, 24)), 0)
    assert same.shape == (OBS_CHANNELS, OBS_HEIGHT, OBS_WIDTH)
    assert np.all(same[7] == 0) and np.all(same[8] == 1)
    with pytest.raises(ValueError):
        policy.build_observation(target, target, np.zeros((5, 5)), 0)


def test_encode_then_decode_round_trips() -> None:
    op = {"op": "stroke", "path": [[3.0, 4.5], [10.25, 8.0], [20.0, 2.0]], "width": 3.5, "water": 0.62,
          "pigments": {"cadmium_yellow": 0.25, "paynes_grey": 0.04}}
    a = policy.encode_stroke(op, 37, (24, 18))
    assert a.shape == (ACTION_DIM,) and a.min() >= 0 and a.max() <= 1
    stroke, dry = policy.policy_output_to_ops(a, (24, 18))
    assert dry == {"op": "dry", "steps": 37}
    assert np.allclose(stroke["path"], op["path"], atol=0.011)
    assert stroke["width"] == pytest.approx(3.5, abs=2e-3)
    assert stroke["water"] == pytest.approx(0.62, abs=1e-3)
    assert stroke["pigments"] == pytest.approx(op["pigments"], abs=1e-3)
    # No wait, no dry op; two-point paths get their midpoint.
    two = dict(op, path=[[0.0, 0.0], [10.0, 6.0]])
    ops = policy.policy_output_to_ops(policy.encode_stroke(two, 0, (24, 18)), (24, 18))
    assert len(ops) == 1 and np.allclose(ops[0]["path"][1], [5.0, 3.0], atol=0.011)
    for steps in (0, 1, 2, 7, 150, 2000):
        assert policy.wait_from_unit(policy.wait_to_unit(steps)) == steps


def test_sparsity_penalty() -> None:
    assert policy.concentration_sparsity(np.array([0.5, 0.2, 0, 0, 0, 0, 0, 0])) == 0.0
    assert policy.concentration_sparsity(np.array([[0.5, 0.2, 0.1, 0.05, 0, 0, 0, 0]])) == pytest.approx(0.15)


def test_policy_output_replays_through_simulator(tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    ops: list[dict] = []
    for _ in range(4):
        out = rng.uniform(0, 1, ACTION_DIM)
        out[policy.WAIT] = rng.uniform(0, 0.6)
        ops += policy.policy_output_to_ops(out, (40, 30))
    assert any(op["op"] == "dry" for op in ops)
    _replays_unchanged(ops, 40, 30, 3, tmp_path)


def test_rollout_is_deterministic() -> None:
    target = synthetic_target("sky", 32, 24, 2)
    fixed = np.linspace(0.1, 0.9, ACTION_DIM)

    def paint() -> tuple[list[dict], np.ndarray]:
        sim = Watercolor(width=32, height=24, seed=4)
        ops = policy.rollout(lambda o: fixed * (0.5 + 0.5 * o[7].mean()), target, sim, 5)
        return ops, sim.render()

    (ops_a, img_a), (ops_b, img_b) = paint(), paint()
    assert sum(op["op"] == "stroke" for op in ops_a) == 5
    assert ops_a == ops_b and np.array_equal(img_a, img_b)


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A real 24 x 18 search-painter run with four stroke slots."""
    root = tmp_path_factory.mktemp("ledger")
    data.write_training_set(root / "targets", seeds_per_kind=1, sealed_dir=root / "none")
    summary = ledgers.generate(root / "targets", root / "runs", limit=1, width=24, height=18, strokes=4,
                               candidates=3, phase_dry_steps=300)
    assert summary["done"] == 1
    return Path(next(iter(ledgers.read_index(root / "runs").values()))["run_dir"])


def test_ledger_to_examples(tiny_run: Path) -> None:
    examples = policy.ledger_to_examples(tiny_run)
    lines = [json.loads(x) for x in (tiny_run / "ledger.jsonl").read_text().splitlines()]
    strokes = [x for x in lines if x["op"]["op"] == "stroke"]
    assert len(examples) == len(strokes) > 0
    header, ops = Watercolor.read_jsonl(tiny_run / "strokes.jsonl")
    target = data.load_training_image(tiny_run / "target.png")
    first = policy.observe(Watercolor.from_header(header), target, examples[0].phase)
    assert np.array_equal(examples[0].observation, first)
    for ex, line in zip(examples, strokes):
        assert ex.op == line["op"] and ex.phase == line["phase"]
        assert ex.observation.shape == (OBS_CHANNELS, OBS_HEIGHT, OBS_WIDTH)
        assert ex.action.shape == (ACTION_DIM,) and 0 <= ex.action.min() and ex.action.max() <= 1
        after = ops[ex.ledger_index + 1 :]
        expected_wait = 0
        for op in after:
            if op["op"] != "dry":
                break
            expected_wait += op["steps"]
        assert ex.wait == expected_wait
    # The last stroke's wait includes the final dry.
    assert examples[-1].wait >= examples[0].wait


def test_ledger_to_examples_checks_the_replay(tiny_run: Path, tmp_path: Path) -> None:
    import shutil

    broken = tmp_path / "broken"
    shutil.copytree(tiny_run, broken)
    lines = (broken / "strokes.jsonl").read_text().splitlines()
    first = json.loads(lines[1])
    first["water"] = round(first["water"] * 0.5, 3)
    lines[1] = json.dumps(first)
    (broken / "strokes.jsonl").write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError):
        policy.ledger_to_examples(broken)


# ---------------------------------------------------------------------------
# The network (needs PyTorch).
# ---------------------------------------------------------------------------


def test_parameter_count_is_about_five_million() -> None:
    assert 4_500_000 <= policy.parameter_count() <= 5_500_000


def test_network_matches_parameter_count_and_shapes() -> None:
    torch = pytest.importorskip("torch")
    model = policy.make_policy(seed=0)
    assert sum(p.numel() for p in model.parameters()) == policy.parameter_count()
    out = model(torch.zeros(2, OBS_CHANNELS, OBS_HEIGHT, OBS_WIDTH))
    assert out.shape == (2, ACTION_DIM) and float(out.min()) >= 0 and float(out.max()) <= 1
    conc = out[:, policy.CONC]
    assert float(policy.concentration_sparsity(conc)) >= 0


def test_network_output_replays_through_simulator(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    target = synthetic_target("discs", 48, 36, 1)
    model = policy.make_policy(seed=3).eval()
    sim = Watercolor(width=48, height=36, seed=2)
    ops = policy.rollout(lambda o: policy.act(model, o), target, sim, 3)
    _replays_unchanged(ops, 48, 36, 2, tmp_path)
    # Same seed, same weights, same ops.
    again = policy.rollout(lambda o: policy.act(policy.make_policy(seed=3).eval(), o), target,
                           Watercolor(width=48, height=36, seed=2), 3)
    assert again == ops


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    model = policy.make_policy(seed=1).eval()
    path = policy.save_checkpoint(tmp_path / "p.pt", model, note="test")
    loaded = policy.load_checkpoint(path)
    x = torch.rand(1, OBS_CHANNELS, OBS_HEIGHT, OBS_WIDTH, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        assert torch.equal(model(x), loaded(x))
