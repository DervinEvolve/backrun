"""Tests for the warm-up's data and settings; the trainer itself needs a GPU and TRL."""

from __future__ import annotations

import json
from pathlib import Path

from scribe.gate import extract_jsonl
from scribe.sft import completion_text, parse_args, sft_config_kwargs, sft_rows
from test_scribe_modal_app import _pool


def test_rows_pair_the_trainer_prompt_with_the_painted_file(tmp_path: Path) -> None:
    pool = _pool(tmp_path / "pool")
    rows = sft_rows(pool)
    assert len(rows) == 6 and all(r["tier"] == "okay" for r in rows)
    row = rows[0]
    assert row["prompt"][0]["role"] == "system" and "seed 3" in row["prompt"][0]["content"]
    assert row["prompt"][1] == {"role": "user", "content": "Paint p0."}
    text = row["completion"][0]["content"]
    assert text.startswith("```jsonl\n") and text.endswith("\n```") and '"canvas"' not in text
    # the gate reads the completion back as the same ops the file holds
    ops = [ln for ln in (pool.folder(row["id"]) / "strokes.jsonl").read_text().splitlines()
           if ln.startswith("{") and '"canvas"' not in ln]
    assert extract_jsonl(text) == ops


def test_completion_text_drops_the_header_only(tmp_path: Path) -> None:
    f = tmp_path / "strokes.jsonl"
    f.write_text('{"op": "canvas", "seed": 1}\n{"op": "stroke", "path": [[1, 1], [2, 2]]}\n\n{"op": "dry", "steps": 5}\n')
    assert completion_text(f) == '```jsonl\n{"op": "stroke", "path": [[1, 1], [2, 2]]}\n{"op": "dry", "steps": 5}\n```'


def test_settings_follow_the_flags() -> None:
    args = parse_args(["--pool", "p", "--epochs", "2", "--learning-rate", "5e-5"])
    cfg = sft_config_kwargs(args)
    assert cfg["num_train_epochs"] == 2 and cfg["learning_rate"] == 5e-5 and cfg["completion_only_loss"] is True
    assert cfg["max_length"] == 6144 and args.model.startswith("Qwen/")
