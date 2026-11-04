"""Generate candidate paintings for the rating pool.

For every prompt, every generator model and ``per`` candidates each:

1. ask the generator with the system prompt from ``scribe.format``;
2. extract the stroke lines and run the gate;
3. if the gate fails, retry once with the gate's reasons in the conversation;
4. if a critic model is set, show the critic the render (and the reference photo
   when the prompt has one) and ask for three improvements; without a photo the
   critic judges against the subject words alone. Then show the
   critic the render and the photo, ask for three short improvements, and ask
   the generator to revise, for up to ``rounds`` rounds. Every round's render
   and stroke file are kept.

Each candidate gets a folder ``POOL_ROOT/<id>/`` with ``strokes.jsonl`` and
``final.png`` (256 x 192), ``preview.png`` (512 px wide, for rating), every
round as ``round_N.jsonl`` / ``round_N.png``, and ``meta.json``. One line per
candidate goes into ``POOL_ROOT/index.jsonl``. A run is resumable: ids already
in the index are skipped. A candidate whose API call failed is not indexed, so
the next run tries it again.

    python -m scribe.generate PROMPTS POOL_ROOT --models a,b --per 2 --critic c --rounds 1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from scribe.client import ChatClient, ClientError, user_message
from scribe.format import canvas_header, system_prompt
from scribe.gate import GateResult, gate, to_jsonl

PREVIEW_WIDTH = 512

CRITIC_PROMPT = (
    "The first image is a watercolour painted from a text prompt: {subject}. The second image is a "
    "reference photo of the same subject. Give exactly three short, concrete improvements the painter "
    "could make to the painting, as a numbered list. Talk about placement, values, colour and water, "
    "in plain words. No praise, no preamble."
)

#: The critic's prompt when the subject has no reference photo: it judges against the words alone.
CRITIC_PROMPT_NO_REF = (
    "This is a watercolour painted from a text prompt: {subject}. Give exactly three short, concrete "
    "improvements the painter could make so it reads better as that subject, as a numbered list. Talk "
    "about placement, values, colour and water, in plain words. No praise, no preamble."
)


def read_prompts(path: str | Path) -> list[dict[str, Any]]:
    """Read a prompts JSONL file: ``{"id", "subject", "prompt", "reference"?}`` per line."""
    out = []
    for n, line in enumerate(Path(path).read_text().splitlines(), start=1):
        if not line.strip():
            continue
        rec = json.loads(line)
        for key in ("id", "subject", "prompt"):
            if key not in rec:
                raise ValueError(f"{path}:{n}: prompt is missing {key!r}")
        out.append(rec)
    return out


def slug(text: str) -> str:
    """A filesystem-safe short form of a model name or id."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")[:60] or "x"


def candidate_id(prompt_id: str, model: str, k: int) -> str:
    """The stable id of the k-th candidate for a prompt and model."""
    return f"{slug(prompt_id)}__{slug(model)}__{k:02d}"


def prompt_seed(prompt: dict[str, Any]) -> int:
    """The paper seed for a prompt: its own ``seed`` field, else a stable hash of its id."""
    if "seed" in prompt:
        return int(prompt["seed"])
    return int(hashlib.sha256(str(prompt["id"]).encode()).hexdigest()[:6], 16) % 10_000


def read_index(pool_root: str | Path) -> list[dict[str, Any]]:
    """All lines of the pool's ``index.jsonl``, or an empty list."""
    p = Path(pool_root) / "index.jsonl"
    if not p.is_file():
        return []
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def save_preview(image: np.ndarray, path: Path, width: int = PREVIEW_WIDTH) -> None:
    """Save a larger copy of a render for the rating page."""
    from PIL import Image

    im = Image.fromarray(np.ascontiguousarray(image))
    h = round(im.height * width / im.width)
    im.resize((width, h), Image.Resampling.LANCZOS).save(path)


def _save_png(image: np.ndarray, path: Path) -> None:
    from PIL import Image

    Image.fromarray(np.ascontiguousarray(image)).save(path)


def _fix_message(res: GateResult) -> str:
    lines = "\n".join(f"- {r}" for r in res.reasons)
    return f"The file was rejected:\n{lines}\nWrite the whole file again with these fixed, in one fenced block."


class Generator:
    """Runs the generate, gate, retry, critique and revise loop for one pool."""

    def __init__(self, client: Any, pool_root: str | Path, temperature: float = 0.9,
                 max_tokens: int = 4096, critic_client: Any | None = None) -> None:
        self.client = client
        self.critic_client = critic_client or client
        self.root = Path(pool_root)
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _ask_and_gate(self, model: str, messages: list[dict[str, Any]], canvas: dict[str, Any],
                      usage: list[dict[str, Any]], tag: str) -> tuple[str, GateResult]:
        reply = self.client.chat(model, messages, temperature=self.temperature,
                                 max_tokens=self.max_tokens, tag=tag)
        usage.append(asdict(reply.usage))
        res = gate(reply.text, canvas)
        if res.ok:
            messages.append({"role": "assistant", "content": reply.text})
            return reply.text, res
        messages += [{"role": "assistant", "content": reply.text}, user_message(_fix_message(res))]
        reply = self.client.chat(model, messages, temperature=self.temperature,
                                 max_tokens=self.max_tokens, tag=tag + ":retry")
        usage.append(asdict(reply.usage))
        messages.append({"role": "assistant", "content": reply.text})
        res = gate(reply.text, canvas)
        res.stats["retried"] = True
        return reply.text, res

    def candidate(self, prompt: dict[str, Any], model: str, k: int, critic: str | None = None,
                  rounds: int = 0, reference_dir: Path | None = None) -> dict[str, Any]:
        """Make one candidate and write its folder; returns its index line.

        Raises ``ClientError`` if an API call fails, in which case nothing is
        indexed and the partial folder is removed.
        """
        cid = candidate_id(prompt["id"], model, k)
        folder = self.root / cid
        if folder.exists():
            shutil.rmtree(folder)  # a partial folder from an interrupted run
        folder.mkdir(parents=True)
        t0 = time.time()
        canvas = canvas_header(seed=prompt_seed(prompt))
        usage: list[dict[str, Any]] = []
        messages = [{"role": "system", "content": system_prompt(seed=canvas["seed"])},
                    user_message(prompt["prompt"])]
        round_log: list[dict[str, Any]] = []
        final: tuple[int, GateResult] | None = None
        try:
            _, res = self._ask_and_gate(model, messages, canvas, usage, f"{cid}:r0")
            self._keep_round(folder, 0, res, canvas, round_log)
            if res.ok:
                final = (0, res)
            ref = prompt.get("reference")
            ref_path = (reference_dir / ref) if (ref and reference_dir) else (Path(ref) if ref else None)
            has_ref = ref_path is not None and ref_path.is_file()
            if critic:
                for r in range(1, rounds + 1):
                    if final is None:
                        break
                    if has_ref:
                        ask = user_message(CRITIC_PROMPT.format(subject=prompt["subject"]),
                                           images=[folder / f"round_{final[0]}.png", ref_path])
                        intro = "A critic compared your painting with a reference photo and suggests:"
                    else:
                        ask = user_message(CRITIC_PROMPT_NO_REF.format(subject=prompt["subject"]),
                                           images=[folder / f"round_{final[0]}.png"])
                        intro = "A critic looked at your painting and suggests:"
                    critique = self.critic_client.chat(critic, [ask], temperature=0.3, max_tokens=400,
                                                       tag=f"{cid}:critic{r}")
                    usage.append(asdict(critique.usage))
                    messages.append(user_message(
                        f"{intro}\n{critique.text.strip()}\nRevise the whole file with these in mind, in one fenced block."))
                    _, res = self._ask_and_gate(model, messages, canvas, usage, f"{cid}:r{r}")
                    self._keep_round(folder, r, res, canvas, round_log, critique=critique.text.strip())
                    if res.ok:
                        final = (r, res)
        except ClientError:
            shutil.rmtree(folder, ignore_errors=True)
            raise
        status = "ok" if final is not None else "rejected"
        if final is not None:
            r, res = final
            (folder / "strokes.jsonl").write_text(to_jsonl(res.ops, canvas))
            _save_png(res.image, folder / "final.png")
            save_preview(res.image, folder / "preview.png")
        meta = {"id": cid, "prompt_id": prompt["id"], "subject": prompt["subject"], "prompt": prompt["prompt"],
                "model": model, "k": k, "critic": critic, "seed": canvas["seed"], "status": status,
                "round": final[0] if final else None, "rounds": round_log,
                "gate": final[1].to_dict() if final else round_log[-1]["gate"],
                "usage": usage, "started_at": t0, "seconds": round(time.time() - t0, 2)}
        (folder / "meta.json").write_text(json.dumps(meta, indent=1))
        line = {"id": cid, "prompt_id": prompt["id"], "model": model, "status": status,
                "round": meta["round"], "dir": cid}
        with (self.root / "index.jsonl").open("a") as fh:
            fh.write(json.dumps(line) + "\n")
        return line

    @staticmethod
    def _keep_round(folder: Path, r: int, res: GateResult, canvas: dict[str, Any],
                    log: list[dict[str, Any]], critique: str | None = None) -> None:
        (folder / f"round_{r}.jsonl").write_text(to_jsonl(res.ops, canvas))
        if res.image is not None:
            _save_png(res.image, folder / f"round_{r}.png")
        entry = {"round": r, "gate": res.to_dict()}
        if critique is not None:
            entry["critique"] = critique
        log.append(entry)


def run(client: Any, prompts_path: str | Path, pool_root: str | Path, models: list[str], per: int,
        critic: str | None = None, rounds: int = 0, temperature: float = 0.9, max_tokens: int = 4096,
        log=sys.stderr) -> dict[str, int]:
    """Generate every missing candidate. Returns counts of ok, rejected, failed and skipped."""
    root = Path(pool_root)
    root.mkdir(parents=True, exist_ok=True)
    prompts_path = Path(prompts_path)
    prompts = read_prompts(prompts_path)
    pool_prompts = root / "prompts.jsonl"
    if not pool_prompts.exists():
        shutil.copyfile(prompts_path, pool_prompts)
    done = {line["id"] for line in read_index(root)}
    gen = Generator(client, root, temperature=temperature, max_tokens=max_tokens)
    counts = {"ok": 0, "rejected": 0, "failed": 0, "skipped": 0}
    for prompt in prompts:
        for model in models:
            for k in range(per):
                cid = candidate_id(prompt["id"], model, k)
                if cid in done:
                    counts["skipped"] += 1
                    continue
                try:
                    line = gen.candidate(prompt, model, k, critic=critic, rounds=rounds,
                                         reference_dir=prompts_path.parent)
                except ClientError as exc:
                    counts["failed"] += 1
                    print(f"{cid}: API failure, will retry on the next run ({exc})", file=log)
                    continue
                counts[line["status"]] += 1
                print(f"{cid}: {line['status']}", file=log)
    return counts


def main(argv: list[str] | None = None) -> int:
    """Command line entry point."""
    ap = argparse.ArgumentParser(description="Generate candidate paintings for the rating pool.")
    ap.add_argument("prompts", help="prompts JSONL: id, subject, prompt, optional reference photo path")
    ap.add_argument("pool_root")
    ap.add_argument("--models", required=True, help="comma-separated generator model names")
    ap.add_argument("--per", type=int, default=2, help="candidates per prompt per model")
    ap.add_argument("--critic", default=None, help="vision model that critiques against the reference")
    ap.add_argument("--rounds", type=int, default=0, help="critic rounds")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--max-tokens", type=int, default=4096)
    args = ap.parse_args(argv)
    root = Path(args.pool_root)
    root.mkdir(parents=True, exist_ok=True)
    try:
        client = ChatClient(ledger=root / "usage.jsonl")
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    counts = run(client, args.prompts, root, models, args.per, critic=args.critic, rounds=args.rounds,
                 temperature=args.temperature, max_tokens=args.max_tokens)
    print(json.dumps(counts))
    return 0 if counts["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
