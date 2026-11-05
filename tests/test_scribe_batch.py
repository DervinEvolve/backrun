"""Tests for the batch writer and its go or no-go numbers."""

from __future__ import annotations

import json
from pathlib import Path

from scribe.batch import batch_stats, verdict, write_candidate
from scribe.pool import Pool
from scribe_fixtures import as_reply, valid_painting

PROMPT = {"id": "sea", "subject": "a calm sea", "prompt": "Paint a calm sea."}


def test_write_candidate_stores_the_pool_layout(tmp_path: Path) -> None:
    ok = write_candidate(tmp_path, PROMPT, "base", 0, as_reply(valid_painting()), finish_reason="stop")
    bad = write_candidate(tmp_path, PROMPT, "base", 1, "no file here", finish_reason="length")
    assert ok["status"] == "ok" and bad["status"] == "rejected"
    folder = tmp_path / ok["id"]
    assert (folder / "final.png").is_file() and (folder / "strokes.jsonl").is_file() and (folder / "reply.txt").is_file()
    assert not (tmp_path / bad["id"] / "final.png").exists()
    meta = json.loads((folder / "meta.json").read_text())
    assert meta["prompt_id"] == "sea" and meta["gate"]["ok"] and meta["finish_reason"] == "stop"
    assert [c["id"] for c in Pool(tmp_path).candidates()] == [ok["id"]]


def test_batch_stats_and_verdict(tmp_path: Path) -> None:
    for k in range(4):
        write_candidate(tmp_path, PROMPT, "base", k, as_reply(valid_painting()) if k < 3 else "nothing",
                        finish_reason="length" if k == 3 else "stop")
    s = batch_stats(tmp_path)
    assert s["completions"] == 4 and s["gate_pass"] == 3 and s["truncated"] == 1
    assert "score the batch" in verdict(s)
    terms = [{"prompt_id": "sea", "total": v, "terms": {"pairwise": v}} for v in (0.2, 0.8, 0.5)] + \
            [{"prompt_id": "sea", "total": 0.0, "terms": {"pairwise": 0.0}}]
    s = batch_stats(tmp_path, terms)
    assert s["groups"] == 1 and s["groups_with_spread"] == 1 and s["excluded"] == 0
    assert "something to push on" in verdict(s)
    flat = batch_stats(tmp_path, [{"prompt_id": "sea", "total": 0.5, "terms": {"pairwise": 0.5}}] * 4)
    assert "flat" in verdict(flat)
    assert "warm the model up" in verdict({"gate_pass_rate": 0.1})


def test_batch_references_draw_same_subject_valid_paintings(tmp_path: Path) -> None:
    import numpy as np

    from scribe.batch import BatchReferences

    good = as_reply(valid_painting())
    for pid in ("sea", "sky"):
        for k in range(3):
            write_candidate(tmp_path, {"id": pid, "subject": pid, "prompt": f"Paint {pid}."}, "m", k, good)
    write_candidate(tmp_path, {"id": "sea", "subject": "sea", "prompt": "Paint sea."}, "m", 9, "nothing valid")
    refs = BatchReferences(tmp_path, k=2)
    assert refs.describe() == {"source": "batch", "root": str(tmp_path), "valid_paintings": 6, "subjects": 2}
    a = refs(np.random.default_rng(1), "sea")
    assert len(a) == 2 and len(set(a)) == 2 and all("/sea__" in p and p.endswith("final.png") for p in a)
    assert a == refs(np.random.default_rng(1), "sea")  # same seed, same draw
    # an unknown subject falls back to the whole batch; the rejected file is never a reference
    b = refs(np.random.default_rng(2), "moon")
    assert len(b) == 2 and all("__09" not in p for p in b)
    import pytest

    small = tmp_path / "small"
    write_candidate(small, {"id": "sea", "subject": "sea", "prompt": "Paint sea."}, "m", 0, good)
    with pytest.raises(ValueError):
        BatchReferences(small, k=2)  # one valid painting cannot supply two references
