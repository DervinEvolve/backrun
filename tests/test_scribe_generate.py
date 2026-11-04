"""End-to-end test of candidate generation with the fake client."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from scribe.client import ClientError, FakeClient
from scribe.generate import candidate_id, read_index, run
from scribe_fixtures import as_reply, valid_painting


def _prompts(tmp_path: Path) -> Path:
    photo = np.zeros((60, 80, 3), dtype=np.uint8)
    photo[:30] = (140, 180, 220)
    photo[30:] = (90, 140, 60)
    Image.fromarray(photo).save(tmp_path / "field.png")
    p = tmp_path / "prompts.jsonl"
    p.write_text(
        json.dumps({"id": "field", "subject": "a field under a pale sky", "prompt": "Paint a field.",
                    "reference": "field.png"}) + "\n"
        + json.dumps({"id": "sea", "subject": "a calm sea", "prompt": "Paint a calm sea."}) + "\n")
    return p


class Script:
    """Model a answers well; model b first sends junk, then a valid file; c critiques."""

    def __init__(self) -> None:
        self.b_calls = 0

    def __call__(self, model, messages):
        if model == "c":
            assert isinstance(messages[0]["content"], list)  # text plus two images
            assert sum(p["type"] == "image_url" for p in messages[0]["content"]) == 2
            return "1. Lower the horizon.\n2. Darken the field.\n3. Leave more sky bare."
        if model == "b":
            self.b_calls += 1
            if self.b_calls == 1:
                return "I cannot paint, but here is a poem."
        return as_reply(valid_painting())


def test_generate_end_to_end_and_resume(tmp_path: Path) -> None:
    prompts = _prompts(tmp_path)
    pool = tmp_path / "pool"
    fake = FakeClient(Script())
    counts = run(fake, prompts, pool, ["a", "b"], per=1, critic="c", rounds=1)
    assert counts == {"ok": 4, "rejected": 0, "failed": 0, "skipped": 0}
    index = read_index(pool)
    assert sorted(x["id"] for x in index) == sorted(
        candidate_id(p, m, 0) for p in ("field", "sea") for m in ("a", "b"))
    assert (pool / "prompts.jsonl").is_file()

    field_a = pool / candidate_id("field", "a", 0)
    for name in ("strokes.jsonl", "final.png", "preview.png", "meta.json", "round_0.png", "round_1.png"):
        assert (field_a / name).is_file(), name
    assert Image.open(field_a / "final.png").size == (256, 192)
    assert Image.open(field_a / "preview.png").size[0] == 512
    meta = json.loads((field_a / "meta.json").read_text())
    assert meta["round"] == 1 and meta["critic"] == "c" and meta["gate"]["ok"]
    assert meta["rounds"][1]["critique"].startswith("1.")
    assert len(meta["usage"]) == 3  # generate, critic, revise
    header = json.loads((field_a / "strokes.jsonl").read_text().splitlines()[0])
    assert header["op"] == "canvas" and header["seed"] == meta["seed"]

    sea_b = json.loads((pool / candidate_id("sea", "b", 0) / "meta.json").read_text())
    assert sea_b["round"] == 0 and not (pool / candidate_id("sea", "b", 0) / "round_1.png").exists()
    # model b's very first reply was junk, so its first candidate needed the retry
    field_b = json.loads((pool / candidate_id("field", "b", 0) / "meta.json").read_text())
    assert field_b["rounds"][0]["gate"]["stats"].get("retried") is True

    before = len(fake.calls)
    again = run(fake, prompts, pool, ["a", "b"], per=1, critic="c", rounds=1)
    assert again["skipped"] == 4 and len(fake.calls) == before


def test_api_failure_is_not_indexed(tmp_path: Path) -> None:
    prompts = _prompts(tmp_path)
    pool = tmp_path / "pool"
    fake = FakeClient(lambda model, messages: ClientError("503 upstream"))
    counts = run(fake, prompts, pool, ["a"], per=1)
    assert counts["failed"] == 2 and read_index(pool) == []
    assert not (pool / candidate_id("sea", "a", 0)).exists()


def test_rejected_candidate_is_indexed_without_final(tmp_path: Path) -> None:
    prompts = _prompts(tmp_path)
    pool = tmp_path / "pool"
    fake = FakeClient(lambda model, messages: "no strokes, sorry")
    counts = run(fake, prompts, pool, ["a"], per=1)
    assert counts["rejected"] == 2
    folder = pool / candidate_id("sea", "a", 0)
    assert not (folder / "final.png").exists()
    assert json.loads((folder / "meta.json").read_text())["status"] == "rejected"
