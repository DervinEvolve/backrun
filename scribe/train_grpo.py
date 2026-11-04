"""GRPO training for the scribe line with TRL's GRPOTrainer.

STATUS: written and parsed, never run. This machine has no GPU and TRL is not
installed; the tests only import this module and exercise the parts that do
not need TRL (argument defaults, dataset rows, the reward function wrapper and
its log). Treat the first real run as a debugging run.

What it does: a language model writes Backrun stroke files from the pool's
prompts; each completion goes through the gate and the pairwise judge
(``scribe.reward``); TRL's GRPOTrainer updates LoRA adapters on the model.

Defaults follow Sergio Paniego's open reproduction of Narreddi and Franz's
recipe (huggingface.co/blog/train-to-paint-with-code): LoRA on all linear
layers, bf16, gradient checkpointing, learning rate 5e-5 with
``constant_with_warmup`` and 5 warmup steps, ``scale_rewards="none"``, 8
generations per prompt, top-p 0.95, top-k 20. Max completion length is 4096
tokens because stroke files are short.

Base model: ``Qwen/Qwen3.5-4B``. The id was checked on the Hugging Face Hub
API on 2026-11-04 (it exists, ungated, tagged image-text-to-text with
architecture Qwen3_5ForConditionalGeneration). Whether GRPOTrainer loads it
for text-only prompts without extra arguments has not been tried.

Infrastructure failures: the reward function returns None for a rollout
whose judge or aesthetic call failed. In TRL's main branch source (read on
2026-11-04), a completion for which every reward function returns None is
marked NaN, left out of the group baseline, and given zero advantage, so it
is excluded from the group. Older TRL releases summed it to 0. On
startup ``check_trl_excludes_none`` looks for that handling in the installed
GRPOTrainer and refuses to train without it; the alternative on an old TRL is
to drop such rollouts in a custom ``_generate_and_score_completions``.

Logs, in ``--output-dir``:

``run_config.json``
    Arguments, the GRPO and LoRA settings, and the pool manifest hash.
``terms.jsonl``
    One line per rollout (``kind: rollout``, with ``step`` and ``terms``) and
    one per reward call (``kind: summary``, per-term mean and sd), so
    ``python -m scribe.reward report OUT/terms.jsonl --step N`` works on any
    step.
``usage.jsonl``
    Judge API usage, one line per call.

Commands, UNVERIFIED (neither has been run; flags are from memory of the
``hf`` CLI and Modal and may need adjusting):

    # Hugging Face Jobs. The repo and pool must be reachable from the job,
    # for example as a private dataset repo downloaded at the start.
    hf jobs uv run --flavor a100-large --timeout 8h \\
        --secrets HF_TOKEN --secrets SCRIBE_API_KEY \\
        --env SCRIBE_API_BASE=https://openrouter.ai/api/v1 \\
        --with trl --with peft --with datasets --with numpy --with scipy --with pillow \\
        scribe/train_grpo.py --pool pool --judge-model JUDGE_MODEL --output-dir out

    # Modal, through a small wrapper app (not written) that mounts the repo
    # and pool, requests one A100 and calls scribe.train_grpo.main(argv):
    modal run scribe/modal_grpo.py --pool pool --judge-model JUDGE_MODEL
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from scribe.format import canvas_header, system_prompt
from scribe.generate import prompt_seed
from scribe.pool import Pool
from scribe.reward import Reward

DEFAULT_MODEL = "Qwen/Qwen3.5-4B"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command line arguments with the defaults described in the module docstring."""
    ap = argparse.ArgumentParser(description="GRPO training of a stroke-file writer (written and parsed; never run).")
    ap.add_argument("--pool", required=True, help="pool root with prompts.jsonl and ratings.jsonl")
    ap.add_argument("--prompts", default=None, help="prompts JSONL; defaults to the pool's prompts.jsonl")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="base model id on the Hugging Face Hub")
    ap.add_argument("--judge-model", required=True, help="vision model for the pairwise judge")
    ap.add_argument("--output-dir", default="runs/scribe-grpo")
    ap.add_argument("--learning-rate", type=float, default=5e-5)
    ap.add_argument("--warmup-steps", type=int, default=5)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--max-completion-length", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--per-device-batch", type=int, default=8, help="completions per device per step")
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--save-steps", type=int, default=25)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--refs", type=int, default=2, help="references per rollout")
    ap.add_argument("--mix", type=float, default=0.5, help="share of references drawn from love")
    ap.add_argument("--min-love", type=int, default=50, help="refuse to train on a pool with fewer love ratings")
    ap.add_argument("--reward-workers", type=int, default=8, help="rollouts scored in parallel")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report-to", default="none", help="trackio, wandb, tensorboard or none")
    return ap.parse_args(argv)


def grpo_config_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Keyword arguments for ``trl.GRPOConfig``, kept separate so they can be checked without TRL."""
    return {
        "output_dir": args.output_dir,
        "learning_rate": args.learning_rate,
        "lr_scheduler_type": "constant_with_warmup",
        "warmup_steps": args.warmup_steps,
        "bf16": True,
        "gradient_checkpointing": True,
        "scale_rewards": "none",
        "num_generations": args.num_generations,
        "max_completion_length": args.max_completion_length,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "per_device_train_batch_size": args.per_device_batch,
        "gradient_accumulation_steps": args.grad_accum,
        "max_steps": args.max_steps,
        "save_steps": args.save_steps,
        "logging_steps": 1,
        "log_completions": True,
        "seed": args.seed,
        "report_to": args.report_to,
    }


def lora_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Keyword arguments for ``peft.LoraConfig``."""
    return {"r": args.lora_r, "lora_alpha": args.lora_alpha, "lora_dropout": 0.05,
            "target_modules": "all-linear", "task_type": "CAUSAL_LM"}


def dataset_rows(prompts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Conversational prompt rows for GRPOTrainer, with the columns the reward needs."""
    rows = []
    for p in prompts:
        seed = prompt_seed(p)
        rows.append({"prompt": [{"role": "system", "content": system_prompt(seed=seed)},
                                {"role": "user", "content": p["prompt"]}],
                     "subject": p["subject"], "prompt_id": str(p["id"]), "seed": seed})
    return rows


def completion_text(completion: Any) -> str:
    """The text of one completion, conversational (list of messages) or plain."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        return "".join(m.get("content", "") for m in completion if isinstance(m, dict)
                       and m.get("role", "assistant") == "assistant" and isinstance(m.get("content"), str))
    return str(completion)


class ScribeReward:
    """A TRL reward function: ``(prompts, completions, **columns) -> list[float | None]``.

    Scores each completion with ``Reward``, in a thread pool since the judge
    calls wait on the network, and appends every rollout's terms plus a
    per-call summary to ``log_path``.
    """

    __name__ = "scribe_reward"  # TRL names the logged reward after this

    def __init__(self, reward: Reward, log_path: str | Path, seed: int = 0, workers: int = 8) -> None:
        self.reward = reward
        self.log_path = Path(log_path)
        self.seed = seed
        self.workers = max(1, workers)
        self.calls = 0

    def __call__(self, prompts: list[Any], completions: list[Any], subject: list[str], prompt_id: list[str],
                 seed: list[int], trainer_state: Any = None, **_: Any) -> list[float | None]:
        step = int(getattr(trainer_state, "global_step", self.calls))
        call = self.calls
        self.calls += 1

        def one(i: int) -> dict[str, Any]:
            rng = np.random.default_rng([self.seed, call, i])
            res = self.reward(completion_text(completions[i]), subject[i], canvas_header(seed=int(seed[i])),
                              rng, prompt_id=prompt_id[i])
            return {"total": res.total, "terms": res.terms, "notes": res.notes}

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            results = list(pool.map(one, range(len(completions))))
        self._log(step, prompt_id, results)
        return [r["total"] for r in results]

    def _log(self, step: int, prompt_id: list[str], results: list[dict[str, Any]]) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [{"step": step, "kind": "rollout", "prompt_id": pid, "total": r["total"], "terms": r["terms"],
                  "excluded": r["total"] is None, "notes": r["notes"]} for pid, r in zip(prompt_id, results)]
        kept = [r for r in results if r["total"] is not None]
        summary: dict[str, Any] = {"step": step, "kind": "summary", "n": len(results),
                                   "excluded": len(results) - len(kept), "at": time.time()}
        if kept:
            for key in kept[0]["terms"]:
                vals = np.array([float(r["terms"][key]) for r in kept])
                summary.setdefault("mean", {})[key] = float(vals.mean())
                summary.setdefault("sd", {})[key] = float(vals.std())
            totals = np.array([r["total"] for r in kept], dtype=float)
            summary["total_mean"], summary["total_sd"] = float(totals.mean()), float(totals.std())
        with self.log_path.open("a") as fh:
            for line in lines + [summary]:
                fh.write(json.dumps(line) + "\n")


def check_trl_excludes_none(trainer_cls: type) -> bool:
    """True if the installed GRPOTrainer leaves all-None rows out of the group baseline."""
    try:
        src = inspect.getsource(trainer_cls)
    except (OSError, TypeError):
        return False
    return "unscorable_mask" in src


def main(argv: list[str] | None = None) -> int:
    """Train. Imports TRL, PEFT and datasets only here, so the module imports without them."""
    args = parse_args(argv)
    if not args.model:
        print("--model is required: give a base model id from the Hugging Face Hub", file=sys.stderr)
        return 1
    try:
        from datasets import Dataset
        from peft import LoraConfig
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        print(f"training needs trl, peft and datasets installed ({exc})", file=sys.stderr)
        return 1
    if not check_trl_excludes_none(GRPOTrainer):
        print("this TRL version scores all-None rollouts as 0 instead of excluding them; "
              "upgrade TRL or drop failed rollouts in a custom step", file=sys.stderr)
        return 1

    from scribe.client import ChatClient
    from scribe.judge import PairwiseJudge
    from scribe.reward import aesthetic_from_env

    pool = Pool(args.pool)
    manifest = pool.manifest()
    if manifest["tiers"]["love"] < args.min_love:
        print(f"pool has {manifest['tiers']['love']} love ratings; --min-love is {args.min_love}", file=sys.stderr)
        return 1
    if args.prompts:
        from scribe.generate import read_prompts

        prompts = read_prompts(args.prompts)
    else:
        prompts = pool.prompts()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = grpo_config_kwargs(args)
    lora = lora_kwargs(args)
    (out / "run_config.json").write_text(json.dumps(
        {"args": vars(args), "grpo": cfg, "lora": lora, "pool_manifest": manifest,
         "prompts": len(prompts), "started_at": time.time()}, indent=1))

    client = ChatClient(ledger=out / "usage.jsonl")
    judge = PairwiseJudge(client, args.judge_model)

    def references(rng: np.random.Generator, prompt_id: str | None) -> list[str]:
        return [r["path"] for r in pool.sample_references(rng, k=args.refs, mix=args.mix)]

    reward = Reward(judge, references, aesthetic=aesthetic_from_env())
    reward_fn = ScribeReward(reward, out / "terms.jsonl", seed=args.seed, workers=args.reward_workers)
    trainer = GRPOTrainer(model=args.model, reward_funcs=[reward_fn], args=GRPOConfig(**cfg),
                          train_dataset=Dataset.from_list(dataset_rows(prompts)),
                          peft_config=LoraConfig(**lora))
    trainer.train()
    trainer.save_model(str(out / "final"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
