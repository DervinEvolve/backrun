"""Tests for replaying a scribe stroke file as a studio run folder."""

from __future__ import annotations

import json
from pathlib import Path

from scribe.batch import write_candidate
from scribe.record import main, prompt_card, record
from scribe_fixtures import as_reply, valid_painting

PROMPT = {"id": "sea", "subject": "a calm sea at noon", "prompt": "Paint a calm sea."}


def test_record_replays_the_file_with_frames_and_a_prompt_card(tmp_path: Path) -> None:
    line = write_candidate(tmp_path / "batch", PROMPT, "base", 0, as_reply(valid_painting()), finish_reason="stop")
    (tmp_path / "batch" / "ratings.jsonl").write_text(json.dumps({"id": line["id"], "tier": "okay", "at": 0}) + "\n")
    run = record(tmp_path / "batch" / line["id"], tmp_path / "run", frames_every=10)
    out = tmp_path / "run"
    assert run["painter"] == "scribe" and run["model"] == "base" and run["rating"] == "okay"
    assert run["subject"] == PROMPT["subject"] and run["target_path"] == "prompt:sea"
    assert run["status"] == "finished" and run["frames"] == len(list((out / "frames").glob("frame_*.png")))
    ledger = [json.loads(x) for x in (out / "ledger.jsonl").read_text().splitlines()]
    assert ledger and all(l["phase"] == "scribe" for l in ledger)
    assert sum(l["op"]["op"] == "stroke" for l in ledger) == run["painter_stats"]["strokes_accepted"] == 9
    assert ledger[-1]["op"]["op"] == "dry" and ledger[-1]["dry_kind"] in ("settle", "final")
    assert (ledger[-1]["dry_kind"] == "final") == (run["painter_stats"]["final_dry_steps"] > 0)
    assert ledger[-1]["sim_steps"] == run["sim_steps"]
    for name in ("target.png", "final.png", "strokes.jsonl", "run.json"):
        assert (out / name).is_file()
    # the replay is the file: the same strokes come back, plus the final dry
    lines = (out / "strokes.jsonl").read_text().splitlines()
    assert json.loads(lines[0])["op"] == "canvas" and len(lines) == len(ledger) + 1
    card = prompt_card(256, 192, 1, "cold_press", "a very long subject " * 6, "model")
    assert card.shape == (192, 256, 3)


def test_record_command_line(tmp_path: Path, capsys) -> None:
    line = write_candidate(tmp_path / "batch", PROMPT, "base", 0, as_reply(valid_painting()))
    assert main([str(tmp_path / "batch" / line["id"]), str(tmp_path / "run")]) == 0
    assert '"painter": "scribe"' in capsys.readouterr().out
