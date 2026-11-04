"""The scribe line on Modal: sample the base model, train with GRPO, evaluate a checkpoint.

From a machine that can reach Modal (``modal setup`` done) and holds this
repo with ``shared/pool_v1/keeps`` in it::

    modal secret create openrouter SCRIBE_API_KEY=...   # the API base is set in the image
    modal run scribe/modal_app.py --stage sample --judge-model anthropic/claude-opus-5.5 --judge-reasoning-effort low --judge-max-tokens 400
    modal run scribe/modal_app.py --stage train --run-name grpo_v1 --judge-model anthropic/claude-opus-5.5 --judge-reasoning-effort low --judge-max-tokens 400 --max-steps 30
    modal run scribe/modal_app.py --stage evaluate --run-name grpo_v1 --judge-model anthropic/claude-opus-5.5 --judge-reasoning-effort low --judge-max-tokens 400

``sample`` is the go or no-go before any training: the base model writes
``--per`` stroke files for every prompt, the gate and the judge score them,
and the report gives the gate pass rate, how many replies were cut off at the
token limit, and how many prompt groups have any reward spread. GRPO only
learns from groups whose rollouts score differently, so a batch of flat
groups means warm the model up first rather than train.

``train`` runs ``scribe.train_grpo`` with vLLM colocated on the GPU and the
reference pack as the pool. ``evaluate`` merges a run's LoRA adapter into the
base model, samples the same way ``sample`` does, and reports the same
numbers next to the base sample's, so the two reports read side by side.

The image installs vLLM (which pins its own PyTorch), TRL, PEFT and the
simulator's dependencies, then copies ``backrun/``, ``scribe/`` and the
reference pack in. The slim image has no CUDA compiler, so
``VLLM_USE_FLASHINFER_SAMPLER=0`` keeps vLLM on its PyTorch sampler
instead of building a FlashInfer kernel on the fly (the first run failed
there). STATUS: the image builds and the model loads on Modal; sampling
and training are still being run for the first time.

Two volumes, shared with the painter's app:

``backrun-data`` at ``/data``
    ``scribe/samples/<name>/`` for sampled batches and their reports.
``backrun-checkpoints`` at ``/checkpoints``
    ``scribe/<run>/`` for trainer output: ``run_config.json``, ``terms.jsonl``,
    ``usage.jsonl``, checkpoints and ``final/`` (the adapter).

The module imports without Modal installed, so tests can check it anywhere.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

try:
    import modal
except ModuleNotFoundError:  # pragma: no cover - exercised in a subprocess test
    modal = None  # type: ignore[assignment]

REPO = Path(__file__).resolve().parent.parent
REMOTE_REPO = "/root/backrun"
DATA = "/data"
CHECKPOINTS = "/checkpoints"
POOL = f"{REMOTE_REPO}/pool"
DEFAULT_GPU = "A100-40GB"
DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
STAGES = ("sample", "train", "evaluate")


class _Absent:
    """Stands in for Modal when it is not installed; see ``train/modal_app.py``."""

    def __getattr__(self, name: str) -> "_Absent":
        return self

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return self


_modal: Any = modal if modal is not None else _Absent()

app = _modal.App("backrun-scribe")

_ignore = ["**/__pycache__", "**/*.pyc"]
image = (
    _modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm>=0.11", "trl>=0.24", "peft>=0.17", "datasets>=3", "accelerate>=1",
                 "numpy>=1.26", "scipy>=1.11", "pillow>=10")
    # the slim image has no CUDA compiler, so vLLM must not ask FlashInfer to build its sampling kernel on the fly
    .env({"PYTHONUNBUFFERED": "1", "HF_HOME": "/checkpoints/hf", "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
          "VLLM_USE_FLASHINFER_SAMPLER": "0", "SCRIBE_API_BASE": "https://openrouter.ai/api/v1"})
    .add_local_file(REPO / "pyproject.toml", f"{REMOTE_REPO}/pyproject.toml", copy=True)
    .add_local_file(REPO / "README.md", f"{REMOTE_REPO}/README.md", copy=True)
    .add_local_dir(REPO / "backrun", f"{REMOTE_REPO}/backrun", copy=True, ignore=_ignore)
    .add_local_dir(REPO / "scribe", f"{REMOTE_REPO}/scribe", copy=True, ignore=_ignore)
    .add_local_dir(REPO / "shared" / "pool_v1" / "keeps", POOL, copy=True)
    .run_commands(f"pip install --no-deps -e '{REMOTE_REPO}'")
    .workdir(REMOTE_REPO)
)

data_volume = _modal.Volume.from_name("backrun-data", create_if_missing=True)
checkpoint_volume = _modal.Volume.from_name("backrun-checkpoints", create_if_missing=True)
VOLUMES = {DATA: data_volume, CHECKPOINTS: checkpoint_volume}
SECRETS = [_modal.Secret.from_name("openrouter")]


def _commit(volume: Any) -> None:
    if modal is not None and hasattr(volume, "commit"):
        try:
            volume.commit()
        except Exception as exc:  # outside a Modal container there is nothing to commit
            print(f"volume commit skipped: {exc}")


# ---------------------------------------------------------------------------
# Shared pieces, plain Python so they can be tested without a GPU.
# ---------------------------------------------------------------------------


def chat_prompts(prompts: list[dict[str, Any]]) -> list[list[dict[str, str]]]:
    """System plus user messages for each prompt, as the trainer builds them."""
    from scribe.train_grpo import dataset_rows

    return [row["prompt"] for row in dataset_rows(prompts)]


def judge_argv(judge_model: str, no_reasoning: bool, effort: str | None, max_tokens: int) -> list[str]:
    """The ``--judge-*`` arguments for ``scribe.train_grpo`` from the entrypoint flags."""
    argv = ["--judge-model", judge_model, "--judge-max-tokens", str(max_tokens)]
    if effort:
        argv += ["--judge-reasoning-effort", effort]
    elif no_reasoning:
        argv.append("--judge-no-reasoning")
    return argv


def judge_extra(no_reasoning: bool, effort: str | None) -> dict[str, Any] | None:
    if effort:
        return {"reasoning": {"effort": effort}}
    if no_reasoning:
        return {"reasoning": {"enabled": False}}
    return None


def generate_with_vllm(model_path: str, conversations: list[list[dict[str, str]]], per: int, max_tokens: int,
                       temperature: float, top_p: float, top_k: int, seed: int,
                       gpu_memory: float = 0.85) -> list[list[tuple[str, str]]]:
    """``per`` completions per conversation with vLLM: lists of (text, finish_reason)."""
    from vllm import LLM, SamplingParams

    llm = LLM(model=model_path, dtype="bfloat16", gpu_memory_utilization=gpu_memory, seed=seed,
              max_model_len=max_tokens + 1024)
    params = SamplingParams(n=per, temperature=temperature, top_p=top_p, top_k=top_k, max_tokens=max_tokens,
                            seed=seed)
    outs = llm.chat(conversations, params, use_tqdm=True)
    return [[(o.text, o.finish_reason) for o in out.outputs] for out in outs]


def score_and_report(batch_root: Path, pool_root: str, judge_model: str, extra: dict[str, Any] | None,
                     judge_max_tokens: int, refs: int, mix: float, workers: int, title: str, model: str,
                     compare_to: Path | None = None, client: Any = None) -> dict[str, Any]:
    """Score a batch against the pool and write ``terms.jsonl``, ``stats.json`` and ``report.md``.

    ``client`` defaults to a ``ChatClient`` from the environment; tests pass a
    scripted one.
    """
    import io

    import numpy as np

    from scribe.batch import batch_stats, verdict
    from scribe.judge import PairwiseJudge
    from scribe.pool import Pool
    from scribe.reward import Reward, reward_report, score_batch

    pool = Pool(pool_root)
    if client is None:
        from scribe.client import ChatClient

        client = ChatClient(ledger=batch_root / "usage.jsonl")
    judge = PairwiseJudge(client, judge_model, max_tokens=judge_max_tokens, extra=extra)

    def references(rng: np.random.Generator, prompt_id: str | None) -> list[str]:
        return [r["path"] for r in pool.sample_references(rng, k=refs, mix=mix, prompt_id=prompt_id)]

    t0 = time.time()
    terms_path = batch_root / "terms.jsonl"
    if terms_path.exists():
        terms_path.unlink()
    lines = score_batch(str(batch_root), Reward(judge, references), str(terms_path), workers=workers)
    usage_path = batch_root / "usage.jsonl"
    usage = [json.loads(x) for x in usage_path.read_text().splitlines() if x.strip()] if usage_path.is_file() else []
    stats = batch_stats(batch_root, lines)
    stats.update({"model": model, "judge_model": judge_model, "judge_calls": len(usage),
                  "judge_cost_usd": round(sum(u.get("cost") or 0 for u in usage), 4),
                  "scoring_s": round(time.time() - t0, 1), "pool_manifest": pool.manifest()["sha256"]})
    stats["verdict"] = verdict(stats)
    buf = io.StringIO()
    stats["reward_report"] = reward_report(lines, out=buf)
    (batch_root / "stats.json").write_text(json.dumps(stats, indent=1) + "\n")
    (batch_root / "report.md").write_text(render_report(title, stats, buf.getvalue(), compare_to))
    return stats


def render_report(title: str, stats: dict[str, Any], reward_text: str, compare_to: Path | None) -> str:
    """The batch numbers as Markdown, next to another batch's when ``compare_to`` has a ``stats.json``."""
    def fmt(x: Any) -> str:
        return "-" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))

    keys = [("completions", "completions"), ("gate_pass_rate", "gate pass rate"), ("truncation_rate", "cut off at the token limit"),
            ("pairwise_mean", "mean pairwise win share"), ("reward_mean", "mean reward"),
            ("groups_with_spread", "prompt groups with reward spread"), ("groups", "prompt groups"),
            ("mean_group_sd", "mean within-group reward sd"), ("excluded", "rollouts excluded (judge failures)"),
            ("judge_calls", "judge calls"), ("judge_cost_usd", "judge cost, $")]
    other = None
    if compare_to is not None and (compare_to / "stats.json").is_file():
        other = json.loads((compare_to / "stats.json").read_text())
    head = "| metric | this batch |" + (f" {compare_to.name} |" if other else "") + "\n|---|---|" + ("---|" if other else "")
    rows = []
    for k, label in keys:
        rows.append(f"| {label} | {fmt(stats.get(k))} |" + (f" {fmt(other.get(k))} |" if other else ""))
    rej = "\n".join(f"- {n}: {r}" for r, n in stats.get("top_rejections", [])) or "- none"
    return (f"# {title}\n\nModel `{stats.get('model', '?')}`, judge `{stats.get('judge_model')}`, pool manifest "
            f"`{str(stats.get('pool_manifest', ''))[:12]}`.\n\n{head}\n" + "\n".join(rows)
            + f"\n\n**{stats.get('verdict', '')}**\n\nTop gate rejections:\n{rej}\n\n## Reward terms\n\n```\n{reward_text}```\n")


def write_batch(batch_root: Path, prompts: list[dict[str, Any]], model_name: str,
                completions: list[list[tuple[str, str]]]) -> None:
    """Store every completion with the batch writer, in prompt order."""
    from scribe.batch import write_candidate

    if (batch_root / "index.jsonl").exists():
        (batch_root / "index.jsonl").unlink()
    for prompt, outs in zip(prompts, completions):
        for k, (text, finish) in enumerate(outs):
            write_candidate(batch_root, prompt, model_name, k, text, finish_reason=finish)


# ---------------------------------------------------------------------------
# Stages.
# ---------------------------------------------------------------------------


@app.function(image=image, volumes=VOLUMES, secrets=SECRETS, gpu=DEFAULT_GPU, cpu=8, memory=32768,
              timeout=3 * 3600)
def sample(name: str, model: str, per: int, max_tokens: int, temperature: float, top_p: float, top_k: int,
           seed: int, judge_model: str, judge_no_reasoning: bool, judge_effort: str | None,
           judge_max_tokens: int, refs: int, mix: float, limit: int, workers: int,
           model_path: str | None = None, compare_to: str | None = None) -> str:
    """Sample a model on the pool's prompts, score the batch, and return the report."""
    from scribe.pool import Pool

    prompts = Pool(POOL).prompts()
    if limit:
        prompts = prompts[:limit]
    batch_root = Path(DATA) / "scribe" / "samples" / name
    if batch_root.exists():
        import shutil

        shutil.rmtree(batch_root)
    batch_root.mkdir(parents=True)
    t0 = time.time()
    completions = generate_with_vllm(model_path or model, chat_prompts(prompts), per, max_tokens, temperature,
                                     top_p, top_k, seed)
    gen_s = time.time() - t0
    write_batch(batch_root, prompts, model.split("/")[-1], completions)
    stats = score_and_report(batch_root, POOL, judge_model, judge_extra(judge_no_reasoning, judge_effort),
                             judge_max_tokens, refs, mix, workers, f"Sample: {name}", model_path or model,
                             compare_to=Path(DATA) / "scribe" / "samples" / compare_to if compare_to else None)
    stats.update({"generation_s": round(gen_s, 1), "per": per, "max_tokens": max_tokens})
    (batch_root / "stats.json").write_text(json.dumps(stats, indent=1) + "\n")
    _commit(data_volume)
    return (batch_root / "report.md").read_text()


@app.function(image=image, volumes=VOLUMES, secrets=SECRETS, gpu=DEFAULT_GPU, cpu=8, memory=65536,
              timeout=12 * 3600)
def train(run_name: str, argv: list[str]) -> str:
    """Run the GRPO trainer with the reference pack as its pool; return the last summary lines."""
    from scribe.train_grpo import main as train_main

    out = Path(CHECKPOINTS) / "scribe" / run_name
    out.mkdir(parents=True, exist_ok=True)
    code = train_main(["--pool", POOL, "--output-dir", str(out), "--vllm", "colocate", *argv])
    _commit(checkpoint_volume)
    if code != 0:
        return f"trainer exited with {code}; see {out}"
    terms = out / "terms.jsonl"
    summaries = [json.loads(x) for x in terms.read_text().splitlines() if '"summary"' in x] if terms.is_file() else []
    tail = summaries[-5:]
    return json.dumps({"run": run_name, "steps_logged": len(summaries),
                       "last": [{k: s.get(k) for k in ("step", "n", "excluded", "total_mean", "total_sd")} for s in tail]},
                      indent=1)


@app.function(image=image, volumes=VOLUMES, secrets=SECRETS, gpu=DEFAULT_GPU, cpu=8, memory=32768,
              timeout=3 * 3600)
def merge_adapter(run_name: str, model: str, checkpoint: str = "final") -> str:
    """Merge a run's LoRA adapter into the base model and save it on the checkpoint volume."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter = Path(CHECKPOINTS) / "scribe" / run_name / checkpoint
    merged = Path(CHECKPOINTS) / "scribe" / run_name / f"{checkpoint}_merged"
    if merged.exists():
        return str(merged)
    base = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16)
    model_with = PeftModel.from_pretrained(base, str(adapter))
    model_with.merge_and_unload().save_pretrained(str(merged))
    AutoTokenizer.from_pretrained(model).save_pretrained(str(merged))
    _commit(checkpoint_volume)
    return str(merged)


@app.local_entrypoint()
def main(
    stage: str = "sample",
    run_name: str = "grpo_v1",
    name: str = "",
    model: str = DEFAULT_MODEL,
    per: int = 8,
    max_tokens: int = 4096,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = 20,
    seed: int = 0,
    judge_model: str = "anthropic/claude-opus-5.5",
    judge_no_reasoning: bool = False,
    judge_reasoning_effort: str = "low",
    judge_max_tokens: int = 400,
    refs: int = 2,
    mix: float = 0.1,
    limit: int = 0,
    workers: int = 8,
    max_steps: int = 30,
    grad_accum: int = 4,
    save_steps: int = 10,
    gpu: str = DEFAULT_GPU,
    checkpoint: str = "final",
    extra: str = "",
) -> None:
    """Run one stage. ``--extra`` passes further arguments to the trainer, space separated."""
    if stage not in STAGES:
        raise SystemExit(f"--stage must be one of {', '.join(STAGES)}")
    effort = judge_reasoning_effort or None
    pick = lambda fn: fn if gpu == DEFAULT_GPU else fn.with_options(gpu=gpu)  # noqa: E731
    if stage == "sample":
        print(pick(sample).remote(name or "base", model, per, max_tokens, temperature, top_p, top_k, seed,
                                  judge_model, judge_no_reasoning, effort, judge_max_tokens, refs, mix, limit,
                                  workers))
    elif stage == "train":
        argv = ["--model", model, "--max-steps", str(max_steps), "--grad-accum", str(grad_accum),
                "--save-steps", str(save_steps), "--num-generations", str(per), "--max-completion-length",
                str(max_tokens), "--temperature", str(temperature), "--top-p", str(top_p), "--top-k", str(top_k),
                "--refs", str(refs), "--mix", str(mix), "--seed", str(seed), "--reward-workers", str(workers),
                *judge_argv(judge_model, judge_no_reasoning, effort, judge_max_tokens), *extra.split()]
        print(pick(train).remote(run_name, argv))
    else:
        merged = pick(merge_adapter).remote(run_name, model, checkpoint)
        print(pick(sample).remote(name or f"{run_name}_{checkpoint}", model, per, max_tokens, temperature, top_p,
                                  top_k, seed, judge_model, judge_no_reasoning, effort, judge_max_tokens, refs,
                                  mix, limit, workers, model_path=merged, compare_to="base"))
