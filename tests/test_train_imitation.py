"""Tests for train.imitation: the example cache, the split and Stage A training."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from train import data, evaluate, imitation, ledgers, policy

#: Half of one uint8 step, plus float slack.
QUANT = 0.5 / 255 + 1e-6


@pytest.fixture(scope="module")
def ledger_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Six real 24 x 18 search-painter runs with six stroke slots each."""
    root = tmp_path_factory.mktemp("imitation")
    data.write_training_set(root / "targets", seeds_per_kind=2, sealed_dir=root / "none")
    summary = ledgers.generate(root / "targets", root / "runs", width=24, height=18, strokes=6, candidates=3,
                               phase_dry_steps=300, workers=2)
    assert summary["done"] == 6
    return root / "runs"


def test_cache_round_trip(ledger_root: Path, tmp_path: Path) -> None:
    runs = imitation.find_runs([ledger_root])
    assert len(runs) == 6
    summary = imitation.build_cache(runs[:2], tmp_path / "cache", workers=2)
    assert summary["built"] == 2 and summary["failed"] == []
    for run, shard in zip(runs[:2], summary["shards"]):
        examples = policy.ledger_to_examples(run)
        with np.load(shard) as z:
            assert z["target"].dtype == np.uint8 and z["planes"].dtype == np.uint8
            decoded = imitation.decode_observations(z["target"], z["planes"], z["phase"])
            assert np.array_equal(z["action"], np.stack([e.action for e in examples]))
            assert json.loads(str(z["meta"]))["n"] == len(examples)
        ref = np.stack([e.observation for e in examples])
        ref[:, 7] = np.clip(ref[:, 7], 0, 1)  # the error plane is stored clipped to 0..1
        assert decoded.shape == ref.shape and decoded.dtype == np.float32
        assert np.max(np.abs(decoded - ref)) <= QUANT
        assert np.array_equal(decoded[:, 8:], ref[:, 8:])  # phase one-hot is exact
    # A second build reuses the shards; a changed ledger is rebuilt.
    again = imitation.build_cache(runs[:2], tmp_path / "cache")
    assert again["reused"] == 2 and again["built"] == 0
    loaded = imitation.load_examples(again["shards"])
    assert len(loaded.names) == 2 and len(loaded) == loaded.action.shape[0]
    idx = np.arange(len(loaded))
    assert loaded.observations(idx).shape == (len(loaded), policy.OBS_CHANNELS, policy.OBS_HEIGHT,
                                              policy.OBS_WIDTH)


def test_split_is_by_painting_and_deterministic() -> None:
    names = [f"p{i}" for i in range(20)]
    tr, va = imitation.split_paintings(names, 0.25, seed=3)
    assert len(va) == 5 and sorted(tr + va) == list(range(20))
    assert imitation.split_paintings(names, 0.25, seed=3) == (tr, va)
    assert imitation.split_paintings(names, 0.25, seed=4) != (tr, va)
    # Order of the names does not change which paintings are held out.
    rev = names[::-1]
    _, va_rev = imitation.split_paintings(rev, 0.25, seed=3)
    assert sorted(rev[i] for i in va_rev) == sorted(names[i] for i in va)
    assert imitation.split_paintings(["a", "b"], 0.01, 0)[1] != []
    assert imitation.split_paintings(["a"], 0.5, 0) == ([0], [])


def test_subset_keeps_whole_paintings(ledger_root: Path, tmp_path: Path) -> None:
    summary = imitation.build_cache(imitation.find_runs([ledger_root], limit=3), tmp_path / "c")
    ex = imitation.load_examples(summary["shards"])
    sub = ex.subset([0, 2])
    assert sub.names == [ex.names[0], ex.names[2]]
    assert len(sub) == int(np.sum(ex.painting != 1))
    assert np.array_equal(sub.planes, ex.planes[ex.painting != 1])


def test_cache_refuses_sealed(tmp_path: Path) -> None:
    sealed = data.write_sealed_set(tmp_path / "sealed", n=2)
    with pytest.raises(data.SealedSetError):
        imitation.find_runs([sealed])
    with pytest.raises(data.SealedSetError):
        imitation.cache_run(tmp_path, sealed / "cache")


def test_lr_schedule() -> None:
    assert imitation.lr_factor(0, 100, 10) == pytest.approx(0.1)
    assert imitation.lr_factor(9, 100, 10) == pytest.approx(1.0)
    assert imitation.lr_factor(10, 100, 10) == pytest.approx(1.0)
    assert imitation.lr_factor(100, 100, 10) == pytest.approx(0.0)
    assert 0.4 < imitation.lr_factor(55, 100, 10) < 0.6


# ---------------------------------------------------------------------------
# Training (needs PyTorch).
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def trained(ledger_root: Path, tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict]:
    pytest.importorskip("torch")
    out = tmp_path_factory.mktemp("ckpt")
    settings = imitation.TrainSettings(epochs=2, batch=4, lr=3e-3, warmup_fraction=0.0, val_fraction=0.34, seed=1,
                                       device="cpu", config="small", threads=2)
    summary = imitation.train_imitation([ledger_root], out / "run", out / "cache", settings, log=lambda *_: None)
    return out / "run", summary


def test_two_epochs_lower_the_training_loss(trained: tuple[Path, dict]) -> None:
    out, summary = trained
    assert summary["epochs_run"] == 2
    assert summary["paintings"] == {"train": 4, "val": 2}
    assert summary["train_loss_end"] < summary["train_loss_start"]
    rows = [json.loads(x) for x in (out / "train.jsonl").read_text().splitlines()]
    assert [r["epoch"] for r in rows] == [1, 2]
    for r in rows:
        assert {"cell", "offset", "bins", "sparsity", "total", "cell_top1", "cell_top5", "bins_within1"} <= set(
            r["train"])
        assert {f"bin_{n}" for n in policy.BIN_NAMES} <= set(r["train"])
        assert set(r["val"]) >= {"total", "cell", "bins", "cell_top5"}
        assert 0 <= r["val"]["cell_top1"] <= r["val"]["cell_top5"] <= 1
        assert r["lr"] >= 0 and r["examples_per_s"] > 0 and r["wall_s"] > 0
    assert (out / "last.pt").is_file() and (out / "best.pt").is_file()
    assert "Training loss went from" in imitation.describe(summary)


def test_best_checkpoint_loads_and_paints(trained: tuple[Path, dict], tmp_path: Path) -> None:
    out, _ = trained
    model = policy.load_checkpoint(out / "best.pt")
    assert model.config == imitation.CONFIGS["small"]
    sealed = data.write_sealed_set(tmp_path / "data" / "sealed", n=2)
    painter = evaluate.make_painter(f"policy:{out / 'best.pt'}")
    report = evaluate.evaluate(painter, tmp_path / "eval", sealed_dir=sealed, seeds=(7,), n_strokes=4, width=64,
                               height=48, limit=2)
    assert report["n_paintings"] == 2 and report["painter"] == "policy:best.pt"
    assert all(r["strokes"] == 4 and r["dried"] for r in report["paintings"])


def test_resume_continues_from_last(trained: tuple[Path, dict], ledger_root: Path, tmp_path: Path) -> None:
    import shutil

    out, _ = trained
    copy = tmp_path / "resume"
    shutil.copytree(out, copy)
    settings = imitation.TrainSettings(epochs=3, batch=4, lr=3e-3, warmup_fraction=0.0, val_fraction=0.34, seed=1,
                                       device="cpu", config="small", threads=2)
    summary = imitation.train_imitation([ledger_root], copy, out.parent / "cache", settings, log=lambda *_: None)
    assert summary["epochs_run"] == 1
    rows = [json.loads(x) for x in (copy / "train.jsonl").read_text().splitlines()]
    assert [r["epoch"] for r in rows] == [1, 2, 3]
    assert summary["cache"]["reused"] == 6 and summary["cache"]["built"] == 0
