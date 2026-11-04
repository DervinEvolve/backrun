"""Warm the scribe up on the keeps before GRPO: supervised fine-tuning with LoRA.

The base model knows the stroke language but has never seen a finished file:
on the first sample it wrote four or five strokes where six is the minimum,
misspelt pigment names, and put points off the paper, so only 15% passed the
gate and GRPO would have seen groups of zeros. One pass over the rated keeps
(prompt in, fenced stroke file out) puts it in the range where the gate
passes and the judge can tell rollouts apart.

    python -m scribe.sft --pool shared/pool_v1/keeps --output-dir runs/warm --epochs 3

Writes ``adapter/`` (the LoRA weights), ``merged/`` (the base model with the
adapter folded in, for vLLM and as the GRPO starting point), ``train.jsonl``
(the loss per step) and ``run_config.json`` (arguments and the pool
manifest). Needs a GPU, TRL and PEFT; ``scribe/modal_app.py --stage warmup``
runs it on Modal and then samples the merged model.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from scribe.format import system_prompt
from scribe.pool import Pool

DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"


def completion_text(strokes_path: str | Path) -> str:
    """The stroke file as the assistant should write it: the op lines in one fence, no canvas header."""
    lines = []
    for line in Path(strokes_path).read_text().splitlines():
        if not line.strip():
            continue
        try:
            if json.loads(line).get("op") == "canvas":
                continue
        except (json.JSONDecodeError, AttributeError):
            pass  # not an op line; the gate decides what to do with it
        lines.append(line.strip())
    return "```jsonl\n" + "\n".join(lines) + "\n```"


def sft_rows(pool: Pool, tiers: tuple[str, ...] = ("love", "okay")) -> list[dict[str, Any]]:
    """Prompt and completion rows for TRL from the pool's rated keeps.

    The prompt is the trainer's: the system prompt for the candidate's own
    paper seed and the user prompt that produced it, read from the
    candidate's ``meta.json`` or else from the pool's prompts file. The
    completion is the stroke file the simulator painted.
    """
    rated = pool.ratings()
    prompts = {str(p["id"]): p for p in pool.prompts()}
    rows = []
    for c in sorted(pool.candidates(), key=lambda c: c["id"]):
        if rated.get(c["id"]) not in tiers:
            continue
        meta = pool.meta(c["id"])
        text = meta.get("prompt") or prompts.get(str(meta.get("prompt_id", c.get("prompt_id"))), {}).get("prompt")
        if not text:
            continue
        seed = int(meta.get("seed", 0))
        rows.append({"prompt": [{"role": "system", "content": system_prompt(seed=seed)},
                                {"role": "user", "content": text}],
                     "completion": [{"role": "assistant", "content": completion_text(pool.folder(c["id"]) / "strokes.jsonl")}],
                     "id": c["id"], "tier": rated[c["id"]]})
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line options."""
    ap = argparse.ArgumentParser(description="Supervised warm-up on the rated keeps.")
    ap.add_argument("--pool", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--output-dir", default="runs/warm")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--per-device-batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-length", type=int, default=6144, help="prompt plus completion, tokens")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-merge", action="store_true", help="keep only the adapter")
    return ap.parse_args(argv)


def sft_config_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Keyword arguments for ``trl.SFTConfig``, kept separate so they can be checked without TRL."""
    return {"output_dir": args.output_dir, "num_train_epochs": args.epochs, "learning_rate": args.learning_rate,
            "lr_scheduler_type": "cosine", "warmup_steps": 5, "bf16": True, "gradient_checkpointing": True,
            "per_device_train_batch_size": args.per_device_batch, "gradient_accumulation_steps": args.grad_accum,
            "max_length": args.max_length, "completion_only_loss": True, "logging_steps": 1, "save_strategy": "no",
            "report_to": "none", "seed": args.seed}


def main(argv: list[str] | None = None) -> int:
    """Train, save the adapter, merge it, log the losses. Imports TRL and PEFT only here."""
    args = parse_args(argv)
    try:
        from datasets import Dataset
        from peft import LoraConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        print(f"the warm-up needs trl, peft and datasets: {exc}", file=sys.stderr)
        return 1
    pool = Pool(args.pool)
    rows = sft_rows(pool)
    if len(rows) < 10:
        print(f"only {len(rows)} rated keeps with prompts in {args.pool}; need more to warm up on", file=sys.stderr)
        return 1
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = sft_config_kwargs(args)
    lora = {"r": args.lora_r, "lora_alpha": args.lora_alpha, "lora_dropout": 0.05, "target_modules": "all-linear",
            "task_type": "CAUSAL_LM"}
    (out / "run_config.json").write_text(json.dumps(
        {"args": vars(args), "sft": cfg, "lora": lora, "examples": len(rows), "pool_manifest": pool.manifest(),
         "started_at": time.time()}, indent=1))
    dataset = Dataset.from_list([{"prompt": r["prompt"], "completion": r["completion"]} for r in rows])
    trainer = SFTTrainer(model=args.model, train_dataset=dataset, args=SFTConfig(**cfg), peft_config=LoraConfig(**lora))
    t0 = time.time()
    trainer.train()
    with (out / "train.jsonl").open("w") as fh:
        for rec in trainer.state.log_history:
            fh.write(json.dumps(rec) + "\n")
    trainer.save_model(str(out / "adapter"))
    trainer.processing_class.save_pretrained(str(out / "adapter"))
    summary: dict[str, Any] = {"examples": len(rows), "steps": trainer.state.global_step, "wall_s": round(time.time() - t0, 1),
                               "losses": [rec["loss"] for rec in trainer.state.log_history if "loss" in rec]}
    if not args.no_merge:
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(str(out / "merged"))
        trainer.processing_class.save_pretrained(str(out / "merged"))
        summary["merged"] = str(out / "merged")
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "losses"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
