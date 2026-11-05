"""Tests for the scribe Modal app's plain-Python parts; nothing here needs Modal or a GPU."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scribe.client import FakeClient, png_data_url
from scribe.modal_app import chat_prompts, judge_argv, judge_extra, render_report, score_and_report, write_batch
from scribe.pool import Pool
from scribe_fixtures import as_reply, valid_painting
from test_scribe_calibrate import _add

PROMPTS = [{"id": "p0", "subject": "subject p0", "prompt": "Paint p0."},
           {"id": "p1", "subject": "subject p1", "prompt": "Paint p1."}]


def _pool(root: Path) -> Pool:
    root.mkdir()
    root.joinpath("prompts.jsonl").write_text("".join(json.dumps(p) + "\n" for p in PROMPTS))
    for p in ("p0", "p1"):
        for i in range(3):
            _add(root, f"{p}__m0__{i:02d}", "m0", p, 200 + 10 * i)
    pool = Pool(root)
    for c in pool.candidates():
        pool.rate(c["id"], "okay")
    return pool


def test_module_imports_without_modal(tmp_path: Path) -> None:
    code = ("import sys; sys.modules['modal'] = None\n"
            "import scribe.modal_app as m\n"
            "print(m.app, m.STAGES, m.sample.__name__, m.train.__name__)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(Path.cwd()))
    assert out.returncode == 0, out.stderr
    assert "sample train" in out.stdout


def test_judge_flags_round_trip() -> None:
    assert judge_argv("j", False, "low", 400) == ["--judge-model", "j", "--judge-max-tokens", "400",
                                                   "--judge-reasoning-effort", "low"]
    assert judge_argv("j", True, None, 8)[-1] == "--judge-no-reasoning"
    assert judge_extra(False, None) is None and judge_extra(True, None) == {"reasoning": {"enabled": False}}
    assert judge_extra(True, "minimal") == {"reasoning": {"effort": "minimal"}}


def test_chat_prompts_carry_the_system_prompt() -> None:
    convs = chat_prompts(PROMPTS)
    assert len(convs) == 2 and convs[0][0]["role"] == "system" and "Paint p0." == convs[0][1]["content"]


def test_write_score_and_report_end_to_end(tmp_path: Path) -> None:
    pool = _pool(tmp_path / "pool")
    batch = tmp_path / "batch"
    batch.mkdir()
    good = as_reply(valid_painting())
    write_batch(batch, PROMPTS, "base", [[(good, "stop"), ("nothing", "length")], [(good, "stop"), (good, "stop")]])
    refs = {png_data_url(pool.folder(c["id"]) / "final.png") for c in pool.candidates()}

    def prefers_candidate(model, messages):
        imgs = [p["image_url"]["url"] for p in messages[0]["content"] if p["type"] == "image_url"]
        return "B" if imgs[0] in refs else "A"

    stats = score_and_report(batch, str(pool.root), "fake-judge", None, 8, refs=2, mix=0.0, workers=2,
                             title="Sample: test", model="base", client=FakeClient(prefers_candidate))
    assert stats["completions"] == 4 and stats["gate_pass"] == 3 and stats["truncated"] == 1
    assert stats["groups"] == 2 and stats["pairwise_mean"] == 0.75  # the rejected completion counts as 0
    assert (batch / "terms.jsonl").is_file() and (batch / "stats.json").is_file()
    report = (batch / "report.md").read_text()
    assert "# Sample: test" in report and "| gate pass rate | 0.750 |" in report and "fake-judge" in report
    # a second batch's report lines up against the first
    other = tmp_path / "other"
    other.mkdir()
    write_batch(other, PROMPTS, "tuned", [[(good, "stop")], [(good, "stop")]])
    stats2 = score_and_report(other, str(pool.root), "fake-judge", None, 8, refs=2, mix=0.0, workers=1,
                              title="Sample: tuned", model="tuned", client=FakeClient(prefers_candidate),
                              compare_to=batch)
    assert stats2["gate_pass_rate"] == 1.0
    assert "| batch |" in (other / "report.md").read_text()
    assert render_report("x", {"verdict": "v", "top_rejections": []}, "", None).startswith("# x")


def test_evaluate_merges_into_the_trained_base_and_compares_as_asked() -> None:
    from scribe.modal_app import evaluate_plan

    # trained from a warm-up's merged model: the adapter goes back onto those weights, not the Hub base
    plan = evaluate_plan("grpo_v1", "Qwen/Qwen3-4B-Instruct-2507", "/checkpoints/scribe/warm_v1/merged", "final",
                         "warm_v1_6144_control")
    assert plan == {"merge_base": "/checkpoints/scribe/warm_v1/merged", "batch": "grpo_v1_final",
                    "compare_to": "warm_v1_6144_control"}
    # trained from the Hub base: merge into it, default batch name, comparison still forwarded
    plan = evaluate_plan("grpo_v2", "Qwen/Qwen3-4B-Instruct-2507", "", "checkpoint-20", "base", name="custom")
    assert plan["merge_base"] == "Qwen/Qwen3-4B-Instruct-2507" and plan["batch"] == "custom"
    assert plan["compare_to"] == "base"


def test_entrypoint_evaluate_branch_uses_the_plan(monkeypatch) -> None:
    """Without Modal, stand in for the two remote functions and check what the evaluate branch hands them."""
    import scribe.modal_app as m

    calls: dict[str, tuple] = {}

    class _Fn:
        def __init__(self, key):
            self.key = key

        def remote(self, *args, **kwargs):
            calls[self.key] = (args, kwargs)
            return "/checkpoints/scribe/grpo_v1/final_merged" if self.key == "merge" else "report"

    monkeypatch.setattr(m, "merge_adapter", _Fn("merge"))
    monkeypatch.setattr(m, "sample", _Fn("sample"))
    m.main(stage="evaluate", run_name="grpo_v1", model_path="/checkpoints/scribe/warm_v1/merged",
           compare_to="warm_v1_6144_control", max_tokens=6144)
    assert calls["merge"][0] == ("grpo_v1", "/checkpoints/scribe/warm_v1/merged", "final")
    args, kwargs = calls["sample"]
    assert args[0] == "grpo_v1_final" and args[3] == 6144
    assert kwargs == {"model_path": "/checkpoints/scribe/grpo_v1/final_merged", "compare_to": "warm_v1_6144_control",
                      "ref_batch": None}


def test_score_against_another_batch_with_a_tag_keeps_both_results(tmp_path: Path) -> None:
    pool = _pool(tmp_path / "pool")
    good = as_reply(valid_painting())
    batch, other = tmp_path / "batch", tmp_path / "other"
    batch.mkdir(), other.mkdir()
    write_batch(batch, PROMPTS, "ckpt", [[(good, "stop"), (good, "stop")], [(good, "stop"), ("bad", "stop")]])
    write_batch(other, PROMPTS, "warm", [[(good, "stop"), (good, "stop")], [(good, "stop"), (good, "stop")]])
    judge = FakeClient(lambda m, msgs: "A")  # always the first image: wins as A, loses as B, so every score is 0.5
    first = score_and_report(batch, str(pool.root), "j", None, 8, refs=2, mix=0.0, workers=1, title="pool",
                             model="ckpt", client=judge)
    second = score_and_report(batch, str(pool.root), "j", None, 8, refs=2, mix=0.0, workers=1, title="vs warm",
                              model="ckpt", client=judge, ref_batch=other, tag="vs_warm")
    assert first["references"]["source"] == "pool" and second["references"] == {
        "source": "batch", "root": str(other), "valid_paintings": 4, "subjects": 2}
    assert second["pairwise_mean"] == 0.5 * 3 / 4 and second["tag"] == "vs_warm"
    for name in ("terms.jsonl", "stats.json", "report.md", "terms_vs_warm.jsonl", "stats_vs_warm.json", "report_vs_warm.md"):
        assert (batch / name).is_file(), name
    assert "valid paintings of batch `other`" in (batch / "report_vs_warm.md").read_text()
    import pytest

    with pytest.raises(ValueError, match="rated nope"):  # the fixture pool has no nope paintings to draw from
        score_and_report(batch, str(pool.root), "j", None, 8, refs=2, mix=0.0, workers=1, title="nope rung",
                         model="ckpt", client=judge, tiers=("nope",), tag="vs_nope")
    pool.rate([c["id"] for c in pool.candidates()][0], "nope")
    pool.rate([c["id"] for c in pool.candidates()][1], "nope")
    nope = score_and_report(batch, str(pool.root), "j", None, 8, refs=2, mix=0.0, workers=1, title="nope rung",
                            model="ckpt", client=judge, tiers=("nope",), tag="vs_nope")
    assert nope["references"]["tiers"] == ["nope"] and (batch / "report_vs_nope.md").is_file()


def test_entrypoint_train_and_evaluate_pass_the_reference_batch(monkeypatch) -> None:
    import scribe.modal_app as m

    calls: dict[str, tuple] = {}

    class _Fn:
        def __init__(self, key):
            self.key = key

        def remote(self, *args, **kwargs):
            calls[self.key] = (args, kwargs)
            return "/checkpoints/scribe/r/final_merged" if self.key == "merge" else "ok"

    monkeypatch.setattr(m, "train", _Fn("train"))
    monkeypatch.setattr(m, "merge_adapter", _Fn("merge"))
    monkeypatch.setattr(m, "sample", _Fn("sample"))
    m.main(stage="train", run_name="r", model_path="/checkpoints/scribe/warm_v1/merged", ref_batch="warm_v1_6144_control")
    argv = calls["train"][0][1]
    assert argv[argv.index("--ref-batch") + 1] == "/data/scribe/samples/warm_v1_6144_control"
    m.main(stage="evaluate", run_name="r", model_path="/checkpoints/scribe/warm_v1/merged",
           ref_batch="warm_v1_6144_control", compare_to="warm_v1_6144_control")
    assert calls["sample"][1]["ref_batch"] == "warm_v1_6144_control"
