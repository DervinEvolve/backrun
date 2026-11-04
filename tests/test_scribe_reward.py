"""Tests for the pairwise judge, the reward composition and the reward report."""

from __future__ import annotations

import io

import numpy as np
import pytest

from scribe.client import ClientError, FakeClient, png_data_url
from scribe.gate import GateResult
from scribe.judge import PairwiseJudge, parse_ab
from scribe.reward import Reward, RewardWeights, reward_report

CAND = np.full((6, 8, 3), 200, dtype=np.uint8)
REF1 = np.full((6, 8, 3), 90, dtype=np.uint8)
REF2 = np.full((6, 8, 3), 30, dtype=np.uint8)
CAND_URL = png_data_url(CAND)


def _images(messages) -> list[str]:
    return [p["image_url"]["url"] for p in messages[0]["content"] if p["type"] == "image_url"]


def prefers_candidate(model, messages):
    first, _ = _images(messages)
    return "A" if first == CAND_URL else "B"


@pytest.mark.parametrize("text,want", [("A", "A"), ("B.", "B"), ("**B**", "B"), ("Answer: A", "A"),
                                       ("The better one is B.", "B"), ("A is better than B", "A"),
                                       ("a lovely painting", None), ("A or B", None), ("Both", None),
                                       ("", None)])
def test_parse_ab(text, want) -> None:
    assert parse_ab(text) == want


def test_judge_asks_both_orders_for_each_reference() -> None:
    fake = FakeClient(prefers_candidate)
    res = PairwiseJudge(fake, "vision").score(CAND, [REF1, REF2], "a lake")
    assert res.score == 1.0 and res.valid == 4 and res.failures == 0
    assert len(fake.calls) == 4
    orders = [_images(c[1]).index(CAND_URL) for c in fake.calls]
    assert orders == [0, 1, 0, 1]
    assert "better watercolour of a lake" in fake.calls[0][1][0]["content"][0]["text"]
    assert all(c[2]["temperature"] == 0.0 for c in fake.calls)


def test_position_bias_cancels_to_one_half() -> None:
    res = PairwiseJudge(FakeClient(lambda m, msgs: "A"), "vision").score(CAND, [REF1, REF2], "x")
    assert res.score == 0.5


def test_failures_are_left_out_until_more_than_half_fail() -> None:
    half = PairwiseJudge(FakeClient(["A", "maybe", ClientError("503"), "A"]), "v").score(CAND, [REF1, REF2], "x")
    assert half.failures == 2 and half.valid == 2 and half.score == 0.5
    most = PairwiseJudge(FakeClient(["A", "maybe", ClientError("503"), "hmm"]), "v").score(CAND, [REF1, REF2], "x")
    assert most.score is None


def _gated(strokes: int = 20, ok: bool = True) -> GateResult:
    return GateResult(ok=ok, image=CAND if ok else None, stats={"strokes": strokes})


def _reward(script, aesthetic=None) -> Reward:
    return Reward(PairwiseJudge(FakeClient(script), "vision"), lambda rng, pid: [REF1, REF2], aesthetic=aesthetic)


def test_composition_without_aesthetic_moves_its_weight() -> None:
    script = ["A", "B", "A", "A"]  # candidate wins first, second, third; loses fourth -> 3/4
    res = _reward(script).score_gated(_gated(20), "x", 0)
    assert res.terms == {"gate": 1.0, "economy": 1.0, "aesthetic": 0.0, "pairwise": 0.75}
    assert res.weights["pairwise"] == pytest.approx(0.9) and res.weights["aesthetic"] == 0.0
    assert res.total == pytest.approx(0.05 + 0.05 + 0.9 * 0.75)
    assert any("moved to pairwise" in n for n in res.notes)


def test_composition_with_aesthetic_and_economy_miss() -> None:
    res = _reward(prefers_candidate, aesthetic=lambda img, s: 0.5).score_gated(_gated(120), "x", 0)
    assert res.terms["economy"] == 0.0
    assert res.total == pytest.approx(0.05 + 0.0 + 0.30 * 0.5 + 0.60 * 1.0)
    assert sum(RewardWeights().__dict__.values()) == pytest.approx(1.0)


def test_gate_failure_scores_zero_without_judge_calls() -> None:
    fake = FakeClient(prefers_candidate)
    r = Reward(PairwiseJudge(fake, "v"), lambda rng, pid: [REF1])
    res = r.score_gated(_gated(ok=False), "x", 0)
    assert res.total == 0.0 and fake.calls == []


def test_infrastructure_failure_makes_reward_none() -> None:
    assert _reward(lambda m, msgs: ClientError("down")).score_gated(_gated(), "x", 0).total is None
    res = _reward(prefers_candidate, aesthetic=lambda img, s: None).score_gated(_gated(), "x", 0)
    assert res.total is None and any("exclude" in n for n in res.notes)


def test_report_flags_saturated_term_and_duplicate_pair() -> None:
    rng = np.random.default_rng(0)
    judge = rng.random(64)
    recs = [{"gate": 1.0, "economy": float(rng.random() < 0.5), "pairwise": float(j),
             "aesthetic": float(np.clip(j + rng.normal(0, 0.03), 0, 1))} for j in judge]
    recs.append({"gate": 1.0, "economy": 1.0, "pairwise": None, "aesthetic": 0.4})
    buf = io.StringIO()
    rep = reward_report(recs, out=buf)
    text = buf.getvalue()
    assert rep["used"] == 64 and rep["excluded"] == 1
    assert rep["stats"]["gate"]["sd"] == 0.0 and rep["stats"]["gate"]["saturated"] == 1.0
    assert any(f.startswith("gate barely varies") for f in rep["flags"])
    assert any("pairwise and aesthetic are correlated" in f for f in rep["flags"])
    assert not any(f.startswith("economy") for f in rep["flags"])
    assert "WARNING: gate barely varies" in text and "correlation" in text


def test_report_on_a_healthy_batch_has_no_flags() -> None:
    rng = np.random.default_rng(1)
    recs = [{"a": float(rng.random()), "b": float(rng.random())} for _ in range(50)]
    assert reward_report(recs, out=io.StringIO())["flags"] == []


def test_report_command_filters_by_step(tmp_path, capsys) -> None:
    import json

    from scribe.reward import main

    log = tmp_path / "terms.jsonl"
    lines = [{"step": s, "kind": "rollout", "terms": {"gate": 1.0, "pairwise": p}}
             for s in (1, 2) for p in (0.25, 0.5, 0.75)]
    lines.append({"step": 2, "kind": "summary", "stats": {}})
    log.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    assert main(["report", str(log), "--step", "2"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("3 rollouts used") and "gate barely varies" in out


def test_score_batch_on_a_generated_folder(tmp_path) -> None:
    import json

    from scribe.generate import run
    from scribe.reward import score_batch
    from scribe_fixtures import as_reply, valid_painting

    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(json.dumps({"id": "sea", "subject": "a calm sea", "prompt": "Paint a sea."}) + "\n")
    replies = iter([as_reply(valid_painting()), "no file", "still no file"])
    run(FakeClient(lambda m, msgs: next(replies)), prompts, tmp_path / "batch", ["base"], per=2)
    reward = Reward(PairwiseJudge(FakeClient(prefers_candidate_any), "j"), lambda rng, pid: [REF1, REF2])
    lines = score_batch(str(tmp_path / "batch"), reward, str(tmp_path / "terms.jsonl"))
    # the fixture has 9 strokes, under the economy floor of 10: 0.05 gate + 0.9 pairwise
    assert sorted(x["total"] for x in lines) == [0.0, pytest.approx(0.95)]
    assert len((tmp_path / "terms.jsonl").read_text().splitlines()) == 2


def prefers_candidate_any(model, messages):
    """Prefers whichever image is not one of the flat references."""
    first, _ = _images(messages)
    return "B" if first in (png_data_url(REF1), png_data_url(REF2)) else "A"
