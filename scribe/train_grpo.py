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

Base model: ``Qwen/Qwen3-4B-Instruct-2507``, a text-only instruct model that
does not think before answering, so the whole reply budget goes on the
stroke file. ``Qwen/Qwen3.5-4B`` was the earlier default; it is tagged
image-text-to-text (Qwen3_5ForConditionalGeneration) and whether GRPOTrainer
loads it for text-only prompts has not been tried, so it stays an option.

The judge takes the settings the calibration found necessary: ``--judge-reasoning-effort low --judge-max-tokens 400``
for Claude, Grok and Qwen3.5 endpoints, ``--judge-no-reasoning`` for the rest.
``--vllm colocate`` runs generation with vLLM on the same GPU, which is the
difference between minutes and seconds per step.

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

Commands: ``scribe/modal_app.py`` wraps this module for Modal (sample the
base model first, then train, then evaluate); its docstring has the three
commands. Running this module directly needs a GPU, TRL, PEFT, datasets and
vLLM installed, the pool on disk and ``SCRIBE_API_KEY``::

    python -m scribe.train_grpo --pool shared/pool_v1/keeps --judge-model anthropic/claude-opus-5.5 \\
        --judge-reasoning-effort low --judge-max-tokens 400 --vllm colocate --max-steps 30 --output-dir runs/grpo
"""

from __future__ import annotations

import argparse
import hashlib
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

DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"


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
    ap.add_argument("--ref-batch", default=None,
                    help="a batch folder whose valid paintings are the references instead of the pool's keeps")
    ap.add_argument("--mix", type=float, default=0.1, help="share of references drawn from love")
    ap.add_argument("--min-love", type=int, default=0, help="refuse to train on a pool with fewer love ratings")
    ap.add_argument("--reward-workers", type=int, default=8, help="rollouts scored in parallel")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report-to", default="none", help="trackio, wandb, tensorboard or none")
    ap.add_argument("--vllm", default="none", choices=["none", "colocate"],
                    help="colocate runs generation with vLLM on the training GPU")
    ap.add_argument("--vllm-gpu-memory", type=float, default=0.3, help="share of GPU memory vLLM may take")
    ap.add_argument("--judge-no-reasoning", action="store_true", help="ask the judge not to think first")
    ap.add_argument("--judge-reasoning-effort", default=None, help="minimal, low, medium or high")
    ap.add_argument("--judge-max-tokens", type=int, default=8, help="reply budget per comparison")
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
        **({"use_vllm": True, "vllm_mode": "colocate", "vllm_gpu_memory_utilization": args.vllm_gpu_memory}
           if args.vllm == "colocate" else {}),
    }


def lora_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Keyword arguments for ``peft.LoraConfig``."""
    return {"r": args.lora_r, "lora_alpha": args.lora_alpha, "lora_dropout": 0.05,
            "target_modules": "all-linear", "task_type": "CAUSAL_LM"}


def judge_extra(args: argparse.Namespace) -> dict[str, Any] | None:
    """Extra request fields for the judge from the ``--judge-*`` flags."""
    if args.judge_reasoning_effort:
        return {"reasoning": {"effort": args.judge_reasoning_effort}}
    if args.judge_no_reasoning:
        return {"reasoning": {"enabled": False}}
    return None


def make_references(args: argparse.Namespace, pool: Pool) -> tuple[Any, dict[str, Any]]:
    """The reference draw for the reward, and a description of it for ``run_config.json``.

    With ``--ref-batch`` the references are another batch's gate-passing
    paintings, same subject when it has enough: a rung the model can
    sometimes beat, so a group's rewards differ by the paintings. Without it
    they are the pool's keeps, which a fresh model loses to almost always.
    """
    if args.ref_batch:
        from scribe.batch import BatchReferences

        refs = BatchReferences(args.ref_batch, k=args.refs)
        return refs, refs.describe()

    def references(rng: np.random.Generator, prompt_id: str | None) -> list[str]:
        return [r["path"] for r in pool.sample_references(rng, k=args.refs, mix=args.mix, prompt_id=prompt_id)]

    return references, {"source": "pool", "tiers": ["love", "okay"], "mix": args.mix}


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


def _prompt_seed(prompt_id: str) -> int:
    """A stable integer for a prompt id, for seeding the reference draw."""
    return int(hashlib.sha256(str(prompt_id).encode()).hexdigest()[:8], 16)


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
            # every rollout of one prompt in one call faces the same references,
            # so the group's rewards differ by the paintings, not by the draw
            rng = np.random.default_rng([self.seed, call, _prompt_seed(prompt_id[i])])
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
    references, ref_info = make_references(args, pool)
    (out / "run_config.json").write_text(json.dumps(
        {"args": vars(args), "grpo": cfg, "lora": lora, "pool_manifest": manifest, "references": ref_info,
         "prompts": len(prompts), "started_at": time.time()}, indent=1))

    client = ChatClient(ledger=out / "usage.jsonl")
    judge = PairwiseJudge(client, args.judge_model, max_tokens=args.judge_max_tokens, extra=judge_extra(args))
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
