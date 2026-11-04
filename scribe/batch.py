"""Store model completions as a batch folder with the pool's layout.

A batch is what the sampling stage and the trainer's evaluation write: one
folder per completion with ``meta.json``, the reply, and when the gate passes
``strokes.jsonl`` and ``final.png``, plus an ``index.jsonl`` line each. The
layout matches what ``scribe.generate`` writes, so ``scribe.reward score``
and ``Pool`` read either.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from scribe.format import canvas_header
from scribe.gate import gate, to_jsonl
from scribe.generate import candidate_id, prompt_seed


def write_candidate(root: str | Path, prompt: dict[str, Any], model: str, k: int, text: str,
                    finish_reason: str | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Gate one completion for ``prompt`` and store it under ``root``; return its index line."""
    root = Path(root)
    seed = prompt_seed(prompt)
    canvas = canvas_header(seed=seed)
    cid = candidate_id(str(prompt["id"]), model, k)
    folder = root / cid
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "reply.txt").write_text(text)
    res = gate(text, canvas)
    if res.ok and res.image is not None:
        from PIL import Image

        (folder / "strokes.jsonl").write_text(to_jsonl(res.ops, canvas))
        Image.fromarray(res.image).save(folder / "final.png")
    meta = {"id": cid, "prompt_id": str(prompt["id"]), "subject": prompt["subject"], "prompt": prompt["prompt"],
            "model": model, "k": k, "seed": seed, "status": "ok" if res.ok else "rejected",
            "finish_reason": finish_reason, "gate": res.to_dict(), **(extra or {})}
    (folder / "meta.json").write_text(json.dumps(meta, indent=1))
    line = {"id": cid, "prompt_id": str(prompt["id"]), "model": model, "status": meta["status"], "dir": cid}
    with (root / "index.jsonl").open("a") as fh:
        fh.write(json.dumps(line) + "\n")
    return line


def batch_stats(root: str | Path, terms: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Pass rate, truncation and per-prompt reward spread for a batch.

    ``terms`` are the lines ``scribe.reward.score_batch`` returned for the
    same batch; without them only the gate numbers are reported. The spread
    that matters for GRPO is within one prompt's group: a group whose
    rewards are all equal gives the trainer nothing.
    """
    root = Path(root)
    lines = [json.loads(x) for x in (root / "index.jsonl").read_text().splitlines() if x.strip()]
    metas = {ln["id"]: json.loads((root / ln["dir"] / "meta.json").read_text()) for ln in lines}
    n = len(lines)
    ok = sum(ln["status"] == "ok" for ln in lines)
    truncated = sum(m.get("finish_reason") == "length" for m in metas.values())
    reasons: dict[str, int] = defaultdict(int)
    for m in metas.values():
        for r in (m.get("gate") or {}).get("reasons", []):
            reasons[r.split(":")[-1].strip()[:60]] += 1
    out: dict[str, Any] = {"completions": n, "gate_pass": ok, "gate_pass_rate": (ok / n) if n else None,
                           "truncated": truncated, "truncation_rate": (truncated / n) if n else None,
                           "top_rejections": sorted(reasons.items(), key=lambda kv: -kv[1])[:6]}
    for ln in lines:  # the tail of one cut-off reply, so the report shows what the model does when it runs on
        m = metas[ln["id"]]
        if m.get("finish_reason") == "length" and (root / ln["dir"] / "reply.txt").is_file():
            text = (root / ln["dir"] / "reply.txt").read_text()
            out["truncated_example"] = {"id": ln["id"], "chars": len(text), "lines": text.count("\n") + 1,
                                        "tail": text[-240:]}
            break
    if terms:
        by_prompt: dict[str, list[float]] = defaultdict(list)
        for t in terms:
            if t.get("total") is not None:
                by_prompt[str(t.get("prompt_id"))].append(float(t["total"]))
        sds = [float(np.std(v)) for v in by_prompt.values() if len(v) > 1]
        totals = [x for v in by_prompt.values() for x in v]
        pw = [float(t["terms"]["pairwise"]) for t in terms if t.get("terms", {}).get("pairwise") is not None]
        out.update({"groups": len(by_prompt), "groups_with_spread": sum(s > 0.02 for s in sds),
                    "mean_group_sd": float(np.mean(sds)) if sds else None,
                    "reward_mean": float(np.mean(totals)) if totals else None,
                    "pairwise_mean": float(np.mean(pw)) if pw else None,
                    "excluded": sum(t.get("total") is None for t in terms)})
    return out


def verdict(stats: dict[str, Any]) -> str:
    """One plain sentence on whether GRPO has something to push on."""
    rate = stats.get("gate_pass_rate")
    if rate is None:
        return "No completions."
    if rate < 0.2:
        return (f"Only {rate:.0%} of completions pass the gate: most groups will score all zero, so warm the "
                f"model up on the keeps (SFT) before GRPO.")
    if "groups" in stats and stats["groups"]:
        share = stats["groups_with_spread"] / stats["groups"]
        if share < 0.5:
            return (f"{rate:.0%} pass the gate but only {share:.0%} of prompt groups have reward spread: "
                    f"GRPO would see mostly flat groups; add references or warm up first.")
        return (f"{rate:.0%} pass the gate and {share:.0%} of prompt groups have reward spread "
                f"(mean within-group sd {stats['mean_group_sd']:.3f}): GRPO has something to push on.")
    return f"{rate:.0%} pass the gate; score the batch to see the reward spread."
