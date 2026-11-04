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
            images = sum(p["type"] == "image_url" for p in messages[0]["content"])
            text = next(p["text"] for p in messages[0]["content"] if p["type"] == "text")
            # the field prompt has a reference photo (render + photo); the sea prompt has none (render only)
            assert images == (2 if "field" in text else 1)
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
    # without a reference photo the critic still runs, judging against the subject words alone
    assert sea_b["round"] == 1 and (pool / candidate_id("sea", "b", 0) / "round_1.png").exists()
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


def test_concurrent_candidates_write_one_index_line_each(tmp_path: Path) -> None:
    prompts = _prompts(tmp_path)
    pool = tmp_path / "pool"
    fake = FakeClient(lambda model, messages: as_reply(valid_painting()))
    counts = run(fake, prompts, pool, ["a", "b"], per=3, concurrency=4)
    assert counts == {"ok": 12, "rejected": 0, "failed": 0, "skipped": 0}
    ids = [line["id"] for line in read_index(pool)]
    assert len(ids) == 12 and len(set(ids)) == 12
    assert all((pool / cid / "final.png").is_file() for cid in ids)


def test_model_that_rejects_extra_fields_gets_plain_calls(tmp_path: Path) -> None:
    from scribe.generate import Generator

    calls: list[dict] = []

    class Picky:
        """Refuses any call carrying extra fields, like an endpoint whose reasoning cannot be turned off."""

        def chat(self, model, messages, temperature=0.8, max_tokens=4096, tag=None, **extra):
            calls.append(extra)
            if extra:
                raise ClientError(f"{model}: HTTP 400: reasoning is mandatory for this endpoint")
            return FakeClient([as_reply(valid_painting())]).chat(model, messages, tag=tag)

    gen = Generator(Picky(), tmp_path / "pool", extra={"reasoning": {"enabled": False}})
    prompt = {"id": "sea", "subject": "a calm sea", "prompt": "Paint a calm sea."}
    line = gen.candidate(prompt, "picky/model", 0)
    assert line["status"] == "ok"
    assert calls[0] == {"reasoning": {"enabled": False}} and calls[1] == {}
    # the second candidate skips the refused field from the start
    gen.candidate(prompt, "picky/model", 1)
    assert calls[2] == {}
