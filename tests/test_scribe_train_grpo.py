"""The GRPO trainer script imports without TRL; its TRL-free parts are checked here."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scribe import train_grpo as T
from scribe.client import FakeClient
from scribe.gate import GateResult
from scribe.judge import PairwiseJudge
from scribe.reward import Reward, reward_report


def test_defaults_match_the_reference_configuration() -> None:
    args = T.parse_args(["--pool", "p", "--judge-model", "j"])
    cfg = T.grpo_config_kwargs(args)
    assert args.model == "Qwen/Qwen3-4B-Instruct-2507"
    assert cfg["learning_rate"] == 5e-5 and cfg["lr_scheduler_type"] == "constant_with_warmup"
    assert cfg["warmup_steps"] == 5 and cfg["scale_rewards"] == "none"
    assert cfg["num_generations"] == 8 and cfg["max_completion_length"] == 4096
    assert cfg["top_p"] == 0.95 and cfg["top_k"] == 20
    assert cfg["bf16"] and cfg["gradient_checkpointing"]
    assert T.lora_kwargs(args)["target_modules"] == "all-linear"


def test_dataset_rows_carry_reward_columns() -> None:
    rows = T.dataset_rows([{"id": "lake", "subject": "a lake", "prompt": "Paint a lake.", "seed": 4}])
    assert rows[0]["prompt"][0]["role"] == "system" and "seed 4" in rows[0]["prompt"][0]["content"]
    assert rows[0]["subject"] == "a lake" and rows[0]["seed"] == 4


def test_completion_text_handles_both_formats() -> None:
    assert T.completion_text("abc") == "abc"
    assert T.completion_text([{"role": "assistant", "content": "x"}]) == "x"


class _StubReward(Reward):
    """Skips the gate render so the wrapper can be checked quickly."""

    def __call__(self, text, subject, canvas, rng, prompt_id=None):
        g = GateResult(ok=text != "bad", image=np.zeros((4, 4, 3), np.uint8), stats={"strokes": 20})
        return self.score_gated(g, subject, rng, prompt_id)


def test_reward_wrapper_returns_none_for_failures_and_logs(tmp_path: Path) -> None:
    def judge(model, messages):
        return "down" if "fails" in messages[0]["content"][0]["text"] else "A"

    reward = _StubReward(PairwiseJudge(FakeClient(judge), "j"), lambda rng, pid: [np.ones((4, 4, 3), np.uint8)])
    fn = T.ScribeReward(reward, tmp_path / "terms.jsonl", workers=2)
    out = fn(prompts=[None] * 3, completions=[[{"role": "assistant", "content": c}] for c in ("ok", "bad", "ok")],
             subject=["a lake", "a lake", "it fails"], prompt_id=["l", "l", "f"], seed=[1, 1, 2])
    assert out[0] is not None and out[1] == 0.0 and out[2] is None
    lines = [json.loads(x) for x in (tmp_path / "terms.jsonl").read_text().splitlines()]
    assert [x["kind"] for x in lines] == ["rollout"] * 3 + ["summary"]
    assert lines[-1]["excluded"] == 1 and lines[2]["excluded"] is True
    rep = reward_report([x for x in lines if x["kind"] == "rollout"], out=open(tmp_path / "r.txt", "w"))
    assert rep["used"] == 2 and rep["excluded"] == 1


def test_trl_check_reads_source() -> None:
    class Old:
        def f(self):
            return 0

    assert T.check_trl_excludes_none(Old) is False


def test_main_without_trl_fails_cleanly(capsys) -> None:
    try:
        import trl  # noqa: F401
    except ImportError:
        assert T.main(["--pool", "p", "--judge-model", "j"]) == 1
        assert "needs trl" in capsys.readouterr().err
