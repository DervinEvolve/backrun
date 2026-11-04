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
