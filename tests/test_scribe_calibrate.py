"""Tests for the judge calibration against a rated pool, with scripted judges."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scribe.calibrate import auc, choose_pairs, choose_sample, run, summarise
from scribe.client import FakeClient, png_data_url
from scribe.gate import _blank
from scribe.judge import PairwiseJudge
from scribe.pool import Pool

SEED = 3


def _add(root: Path, cid: str, model: str, prompt_id: str, shade: int) -> None:
    folder = root / cid
    folder.mkdir(parents=True)
    img = _blank(64, 48, SEED, "cold_press").copy()
    img[10:40, 10:50] = (shade, shade // 2, 60)
    Image.fromarray(img).save(folder / "final.png")
    (folder / "strokes.jsonl").write_text(json.dumps({"op": "canvas", "seed": SEED}) + "\n" + cid + "\n")
    (folder / "meta.json").write_text(json.dumps({"id": cid, "model": model, "prompt_id": prompt_id,
                                                  "subject": f"subject {prompt_id}", "seed": SEED, "status": "ok"}))
    with (root / "index.jsonl").open("a") as fh:
        fh.write(json.dumps({"id": cid, "prompt_id": prompt_id, "model": model, "status": "ok", "dir": cid}) + "\n")


@pytest.fixture()
def pool(tmp_path: Path) -> Pool:
    """Three subjects, two models, four keeps and four nopes each; keeps are the brighter images."""
    root = tmp_path / "pool"
    root.mkdir()
    (root / "prompts.jsonl").write_text("".join(json.dumps({"id": p, "subject": f"subject {p}", "prompt": p}) + "\n"
                                                for p in ("p0", "p1", "p2")))
    shade = 40
    for p in ("p0", "p1", "p2"):
        for i in range(8):
            shade += 7
            _add(root, f"{p}__m{i % 2}__{i:02d}", f"m{i % 2}", p, 200 + shade % 50 if i < 4 else 20 + shade % 50)
    pool = Pool(root)
    # the first 6 ratings are "the owner's", the rest the reviewer's
    order = sorted(c["id"] for c in pool.candidates())
    for n, cid in enumerate(order):
        pool.rate(cid, "okay" if int(cid[-2:]) < 4 else "nope", at=float(n))
    return pool


def _keep_urls(pool: Pool) -> set[str]:
    rated = pool.ratings()
    return {png_data_url(pool.folder(c)/"final.png") for c, t in rated.items() if t == "okay"}


def _images(messages) -> list[str]:
    return [p["image_url"]["url"] for p in messages[0]["content"] if p["type"] == "image_url"]


def test_sample_keeps_every_owner_painting_and_balances_the_rest(pool: Pool) -> None:
    rng = np.random.default_rng(0)
    sample = choose_sample(pool, rng, n=14, owner_lines=6)
    ids = [c["id"] for c in sample]
    owners = {r["id"] for r in pool._lines()[:6]}
    assert owners <= set(ids)
    rest = [pool.ratings()[i] for i in ids if i not in owners]
    assert len(rest) == 8 and rest.count("okay") == 4 and rest.count("nope") == 4
    assert len(set(ids)) == len(ids)


def test_pairs_are_same_subject_keep_against_nope(pool: Pool) -> None:
    pairs = choose_pairs(pool, np.random.default_rng(0), 10)
    rated = pool.ratings()
    assert len(pairs) == 10 and len(set(pairs)) == 10
    for keep, nope in pairs:
        assert rated[keep] == "okay" and rated[nope] == "nope"
        assert keep.split("__")[0] == nope.split("__")[0]


def test_auc_counts_ties_as_half() -> None:
    assert auc([1.0, 1.0], [0.0, 0.0]) == 1.0
    assert auc([0.5], [0.5]) == 0.5
    assert auc([0.0], [1.0]) == 0.0
    assert np.isnan(auc([], [1.0]))


def test_perfect_judge_scores_auc_one_with_same_subject_references(pool: Pool, tmp_path: Path) -> None:
    keeps = _keep_urls(pool)

    def prefers_keeps(model, messages):
        first, second = _images(messages)
        return "A" if (first in keeps) >= (second in keeps) else "B"  # a keep beats a nope; ties go to A

    judge = PairwiseJudge(FakeClient(prefers_keeps), "fake-judge")
    out = tmp_path / "calib"
    summary = run(pool, judge, out, n=24, pairs_n=12, refs=2, owner_lines=6, seed=1, concurrency=2)
    rows = [json.loads(x) for x in (out / "rows.jsonl").read_text().splitlines()]
    assert len(rows) == 24 and summary["same_subject_share"] == 1.0
    assert all(r["id"] not in r["references"] for r in rows)
    # a keep judged against keeps ties (A wins both orders = one win, one loss); a nope loses both
    assert summary["all"]["mean_win_share"] == {"keep": 0.5, "nope": 0.0}
    assert summary["all"]["auc"] == 1.0
    assert summary["direct_pairs"]["keep_picked"] == 1.0 and summary["direct_pairs"]["order_flip_rate"] == 0.0
    assert set(summary["by_rater"]) == {"owner", "reviewer"}
    assert summary["by_rater"]["owner"]["n"] == 6
    report = (out / "report.md").read_text()
    assert "| all | 24 |" in report and "fake-judge" in report
    assert summary["pool_manifest"] == pool.manifest()["sha256"]


def test_position_biased_judge_shows_as_flips_and_coin_flip_auc(pool: Pool, tmp_path: Path) -> None:
    judge = PairwiseJudge(FakeClient(lambda m, msgs: "A"), "always-a")
    summary = run(pool, judge, tmp_path / "c", n=12, pairs_n=6, owner_lines=6, seed=0, concurrency=1)
    assert summary["all"]["auc"] == 0.5
    assert summary["all"]["order_flip_rate"] == 1.0 and summary["all"]["decided"] == 0
    assert summary["direct_pairs"]["keep_picked"] == 0.5 and summary["direct_pairs"]["order_flip_rate"] == 1.0


def test_summarise_excludes_unscored_rows() -> None:
    rows = [{"id": "a", "prompt_id": "p", "model": "m", "tier": "okay", "rater": "owner", "score": 1.0,
             "flips": 0, "flip_pairs": 2},
            {"id": "b", "prompt_id": "p", "model": "m", "tier": "nope", "rater": "owner", "score": None,
             "flips": 0, "flip_pairs": 0},
            {"id": "c", "prompt_id": "p", "model": "m", "tier": "nope", "rater": "owner", "score": 0.25,
             "flips": 1, "flip_pairs": 2}]
    s = summarise(rows, [], seed=0)
    assert s["all"]["scored"] == 2 and s["all"]["auc"] == 1.0 and s["all"]["order_flip_rate"] == 0.25
