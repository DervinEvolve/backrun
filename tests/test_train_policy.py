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


SMALL = policy.PolicyConfig(widths=(8, 16, 16, 32), blocks=(1, 1, 1, 1), hidden=64, groups=4)


def test_network_matches_parameter_count_and_shapes() -> None:
    torch = pytest.importorskip("torch")
    for config in (policy.PolicyConfig(), SMALL):
        model = policy.make_policy(config, seed=0)
        assert sum(p.numel() for p in model.parameters()) == policy.parameter_count(config)
    model = policy.make_policy(seed=0)
    with torch.no_grad():
        out = model(torch.zeros(2, OBS_CHANNELS, OBS_HEIGHT, OBS_WIDTH))
    assert out.cell_logits.shape == (2, policy.N_CELLS) == (2, 24 * 32)
    assert out.offsets.shape == (2, policy.N_CELLS, 2)
    assert float(out.offsets.min()) >= 0 and float(out.offsets.max()) <= 1
    assert out.bin_logits.shape == (2, policy.N_BINNED, 32) and policy.N_BINNED == 15
    assert torch.equal(out.cell, out.cell_logits.argmax(dim=1))
    for mode in ("argmax", "sample"):
        a = model.decode(out, mode=mode, generator=torch.Generator().manual_seed(0))
        assert a.shape == (2, ACTION_DIM) and float(a.min()) >= 0 and float(a.max()) <= 1
    with pytest.raises(ValueError):
        model.decode(out, mode="mean")


def _random_actions(n: int, seed: int) -> np.ndarray:
    """Random actions whose further path points stay within the relative range."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(0, 1, (n, ACTION_DIM))
    a[:, 2:6] = np.clip(np.tile(a[:, 0:2], 2) + rng.uniform(-0.45, 0.45, (n, 4)), 0, 1)
    a[:, policy.CONC] *= rng.uniform(0, 1, (n, 8)) < 0.3  # mostly unloaded pigments
    return a


def test_action_bins_round_trip() -> None:
    config = policy.PolicyConfig()
    half = 0.5 / config.bins
    a = _random_actions(500, 0)
    ab = policy.action_to_bins(a, config)
    assert ab.cell.shape == (500,) and ab.offset.shape == (500, 2) and ab.bins.shape == (500, 15)
    assert ab.cell.min() >= 0 and ab.cell.max() < policy.N_CELLS
    assert ab.bins.min() >= 0 and ab.bins.max() < config.bins
    back = policy.bins_to_action(ab.cell, ab.offset, ab.bins, config)
    assert back.shape == a.shape
    assert np.allclose(back[:, 0:2], a[:, 0:2], atol=1e-6)  # the start point is exact
    err = np.abs(back - a)
    conc = a[:, policy.CONC]
    small = conc < 1.0 / config.bins  # the lowest concentration bin decodes to no pigment
    assert np.all(back[:, policy.CONC][small] == 0)
    assert np.all(err[:, policy.CONC][~small] <= half + 1e-6)
    keep = [i for i in range(ACTION_DIM) if i not in range(policy.CONC.start, policy.CONC.stop)]
    assert np.all(err[:, keep] <= half + 1e-6)
    # The same arithmetic on tensors gives the same numbers.
    torch = pytest.importorskip("torch")
    tb = policy.bins_to_action(torch.from_numpy(ab.cell), torch.from_numpy(ab.offset).double(),
                               torch.from_numpy(ab.bins), config)
    assert np.allclose(tb.numpy(), back, atol=1e-6)


def _one_hot_output(ab: policy.ActionBins, config: policy.PolicyConfig) -> policy.PolicyOutput:
    """A hand-built output that puts all its weight on the given discrete action."""
    torch = pytest.importorskip("torch")
    b = len(ab.cell)
    cell_logits = torch.full((b, policy.N_CELLS), -50.0)
    cell_logits[torch.arange(b), torch.from_numpy(ab.cell)] = 50.0
    offsets = torch.full((b, policy.N_CELLS, 2), 0.5)
    offsets[torch.arange(b), torch.from_numpy(ab.cell)] = torch.from_numpy(ab.offset)
    bins = torch.full((b, policy.N_BINNED, config.bins), -50.0)
    bins.scatter_(2, torch.from_numpy(ab.bins)[..., None], 50.0)
    return policy.PolicyOutput(cell_logits, offsets, bins, torch.from_numpy(ab.cell))


def test_one_hot_search_strokes_decode_and_replay(tiny_run: Path, tmp_path: Path) -> None:
    pytest.importorskip("torch")
    config = SMALL
    model = policy.make_policy(config).eval()
    examples = policy.ledger_to_examples(tiny_run)
    size = (24, 18)
    actions = np.stack([e.action for e in examples]).astype(np.float64)
    ab = policy.action_to_bins(actions, config)
    out = _one_hot_output(ab, config)
    decoded = model.decode(out).numpy()
    expect = policy.bins_to_action(ab.cell, ab.offset, ab.bins, config)
    assert np.allclose(decoded, expect, atol=1e-6)
    assert np.allclose(model.log_prob(out, ab).numpy(), 0.0, atol=1e-6)
    assert np.all(model.entropy(out).numpy() < 1e-6)
    ops: list[dict] = []
    px_half = 0.5 / config.bins * (size[0] - 1)  # half a relative-point bin in pixels, x axis
    for ex, a in zip(examples, decoded):
        stroke_ops = policy.policy_output_to_ops(a, size)
        stroke = stroke_ops[0]
        orig = policy.policy_output_to_ops(ex.action, size)[0]
        assert np.allclose(stroke["path"][0], orig["path"][0], atol=0.011)
        rel = np.abs(np.asarray(orig["path"][1:]) - np.asarray(orig["path"][0]))
        inside = np.all(rel < 0.5 * np.array([size[0] - 1, size[1] - 1]), axis=1)
        err = np.abs(np.asarray(stroke["path"][1:]) - np.asarray(orig["path"][1:]))
        assert np.all(err[inside] <= px_half + 0.011)
        lo, hi = policy.WIDTH_RANGE
        assert abs(stroke["width"] - orig["width"]) <= max(1.0, 0.5 / config.bins * min(size) * (hi - lo)) + 1e-3
        for name, c in orig["pigments"].items():
            assert abs(stroke["pigments"].get(name, 0.0) - c) <= 1.0 / config.bins + 1e-3
        ops += stroke_ops
    _replays_unchanged(ops, size[0], size[1], 3, tmp_path)


def test_log_prob_and_entropy_match_sampling() -> None:
    torch = pytest.importorskip("torch")
    model = policy.make_policy(SMALL, seed=2).eval()
    obs = torch.rand(3, OBS_CHANNELS, OBS_HEIGHT, OBS_WIDTH, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        out = model(obs)
        a1 = model.decode(out, mode="sample", generator=torch.Generator().manual_seed(5))
        a2 = model.decode(out, mode="sample", generator=torch.Generator().manual_seed(5))
        assert torch.equal(a1, a2)
        ab = policy.action_to_bins(a1.numpy(), SMALL)
        lp = model.log_prob(out, ab)
        assert lp.shape == (3,) and torch.all(lp < 0)
        # At the argmax cell, the argmax bins are the most likely ones there.
        bins = out.bin_logits.argmax(dim=-1)
        best = model.log_prob(out, policy.ActionBins(out.cell, None, bins))
        other = model.log_prob(out, policy.ActionBins(out.cell, None, (bins + 1) % SMALL.bins))
        assert torch.all(best > other)
        ent = model.entropy(out)
        assert ent.shape == (3,) and torch.all(ent > 0)
        assert float(ent.max()) <= np.log(policy.N_CELLS) + 15 * np.log(SMALL.bins) + 1e-3


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
    model = policy.make_policy(SMALL, seed=1).eval()
    path = policy.save_checkpoint(tmp_path / "p.pt", model, note="test")
    assert torch.load(path, weights_only=True)["format"] == "discrete-v2"
    loaded = policy.load_checkpoint(path)
    x = torch.rand(1, OBS_CHANNELS, OBS_HEIGHT, OBS_WIDTH, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        a, b = model(x), loaded(x)
        assert torch.equal(a.cell_logits, b.cell_logits) and torch.equal(a.bin_logits, b.bin_logits)
        assert torch.equal(model.decode(a), loaded.decode(b))


def test_old_regression_checkpoint_is_refused(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    old = tmp_path / "old.pt"
    torch.save({"config": {"in_channels": 11, "widths": [8, 16, 16, 32], "blocks": [1, 1, 1, 1], "hidden": 64,
                           "groups": 4, "action_dim": 17}, "state_dict": {}, "extra": {}}, old)
    with pytest.raises(ValueError, match="earlier regression head.*retrained"):
        policy.load_checkpoint(old)
