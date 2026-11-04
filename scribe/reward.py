"""The reward: gate, economy, an optional aesthetic score and the pairwise judge.

Weights live in ``RewardWeights``. Each term is a number from 0 to 1 and the
reward is their weighted sum. A file that fails the gate scores 0 and costs no
judge calls. If any term is None because something outside the model failed
(the judge, the aesthetic scorer), the whole reward is None, and a trainer
must leave that rollout out of its group.

Before training, run ``reward_report`` on a batch: a term that barely varies
gives no gradient, and two terms that move together are one signal counted
twice. Both happened in the recipe this line follows.
"""

from __future__ import annotations

import json
import math
import os
import sys
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import numpy as np

from scribe.client import png_data_url
from scribe.gate import GateResult, gate
from scribe.judge import PairwiseJudge

ECONOMY_MIN = 10  # design choice: fewer strokes is rarely a finished painting
ECONOMY_MAX = 80  # design choice: very long files were the waste in the original recipe
NO_GRADIENT_SD = 0.02  # flag a term whose standard deviation is below this
DUPLICATE_R = 0.8  # flag a pair of terms correlated above this
SATURATION_EPS = 1e-6


@dataclass(frozen=True)
class RewardWeights:
    """Weights of the reward terms. They should sum to 1."""

    gate: float = 0.05
    economy: float = 0.05
    aesthetic: float = 0.30
    pairwise: float = 0.60


@dataclass
class RewardResult:
    """One rollout's reward. ``total`` is None when the rollout must be excluded."""

    total: float | None
    terms: dict[str, float | None]
    weights: dict[str, float]
    notes: list[str] = field(default_factory=list)
    gate: dict[str, Any] = field(default_factory=dict)
    judge: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly form."""
        return asdict(self)


class HTTPAestheticScorer:
    """An aesthetic scorer behind HTTP, such as an HPSv3 server.

    POSTs ``{"image": <PNG data URL>, "prompt": subject}`` to ``url`` and
    reads ``{"score": number}``. The raw score is mapped linearly from
    ``[lo, hi]`` onto 0 to 1 and clipped; set ``lo`` and ``hi`` from a batch
    of real scores before use. Returns None on any failure.
    """

    def __init__(self, url: str, lo: float = 0.0, hi: float = 10.0, timeout: float = 60.0) -> None:
        self.url, self.lo, self.hi, self.timeout = url, lo, hi, timeout
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler())

    def __call__(self, image: np.ndarray, subject: str) -> float | None:
        """Score one render, 0 to 1, or None if the scorer failed."""
        body = json.dumps({"image": png_data_url(image), "prompt": subject}).encode()
        req = urllib.request.Request(self.url, data=body, headers={"Content-Type": "application/json"},
                                     method="POST")
        try:
            with self._opener.open(req, timeout=self.timeout) as resp:
                raw = float(json.loads(resp.read())["score"])
        except (urllib.error.URLError, TimeoutError, ConnectionError, KeyError, ValueError, TypeError):
            return None
        if not math.isfinite(raw):
            return None
        return float(np.clip((raw - self.lo) / (self.hi - self.lo), 0.0, 1.0))


def aesthetic_from_env() -> HTTPAestheticScorer | None:
    """The scorer at ``SCRIBE_AESTHETIC_URL``, or None (the default: off)."""
    url = os.environ.get("SCRIBE_AESTHETIC_URL", "").strip()
    if not url:
        return None
    lo = float(os.environ.get("SCRIBE_AESTHETIC_LO", "0"))
    hi = float(os.environ.get("SCRIBE_AESTHETIC_HI", "10"))
    return HTTPAestheticScorer(url, lo, hi)


class Reward:
    """Score a model reply: gate it, then economy, aesthetic and the pairwise judge.

    ``references(rng, prompt_id)`` returns the reference images (paths or
    arrays) for one rollout, for example a bound ``Pool.sample_references``
    mapped to paths. ``aesthetic`` is a callable ``(image, subject) -> float
    or None``, or None to switch the term off.
    """

    def __init__(self, judge: PairwiseJudge, references: Callable[[np.random.Generator, str | None], list[Any]],
                 weights: RewardWeights = RewardWeights(),
                 aesthetic: Callable[[np.ndarray, str], float | None] | None = None) -> None:
        self.judge = judge
        self.references = references
        self.weights = weights
        self.aesthetic = aesthetic

    def effective_weights(self) -> tuple[dict[str, float], list[str]]:
        """Weights after redistributing the aesthetic weight when its scorer is off."""
        w = asdict(self.weights)
        notes = []
        if self.aesthetic is None and w["aesthetic"]:
            notes.append(f"aesthetic scorer off; its weight {w['aesthetic']:g} moved to pairwise")
            w["pairwise"] += w["aesthetic"]
            w["aesthetic"] = 0.0
        return w, notes

    def __call__(self, text: str, subject: str, canvas: dict[str, Any], rng: np.random.Generator | int,
                 prompt_id: str | None = None) -> RewardResult:
        """Gate the reply and score it."""
        return self.score_gated(gate(text, canvas), subject, rng, prompt_id)

    def score_gated(self, g: GateResult, subject: str, rng: np.random.Generator | int,
                    prompt_id: str | None = None) -> RewardResult:
        """Score a rollout whose gate result is already known."""
        if isinstance(rng, (int, np.integer)):
            rng = np.random.default_rng(int(rng))
        w, notes = self.effective_weights()
        terms: dict[str, float | None] = {"gate": 0.0, "economy": 0.0, "aesthetic": 0.0, "pairwise": 0.0}
        if not g.ok or g.image is None:
            return RewardResult(0.0, terms, w, notes + ["gate failed"], g.to_dict())
        terms["gate"] = 1.0
        n = int(g.stats.get("strokes", 0))
        terms["economy"] = 1.0 if ECONOMY_MIN <= n <= ECONOMY_MAX else 0.0
        if self.aesthetic is not None:
            terms["aesthetic"] = self.aesthetic(g.image, subject)
        refs = self.references(rng, prompt_id)
        jr = self.judge.score(g.image, refs, subject)
        terms["pairwise"] = jr.score
        judge_info = {"wins": jr.wins, "valid": jr.valid, "failures": jr.failures}
        missing = [k for k, v in terms.items() if v is None]
        if missing:
            notes.append("infrastructure failure in " + ", ".join(missing) + "; exclude this rollout")
            return RewardResult(None, terms, w, notes, g.to_dict(), judge_info)
        total = sum(w[k] * float(v) for k, v in terms.items())
        return RewardResult(total, terms, w, notes, g.to_dict(), judge_info)


def _as_terms(rec: Any) -> dict[str, float | None]:
    if isinstance(rec, RewardResult):
        return dict(rec.terms)
    if isinstance(rec, dict) and "terms" in rec:
        return dict(rec["terms"])
    return dict(rec)


def reward_report(records: list[Any], out: Any = None, terms: list[str] | None = None) -> dict[str, Any]:
    """Measure each reward term on a batch before training on it.

    ``records`` holds per-rollout term values: dicts of term to number, or
    ``RewardResult`` objects. Rollouts with any None term are excluded and
    counted. Prints the mean, standard deviation and share of saturated values
    (at 0 or 1) per term, the Pearson correlation matrix, and plain warnings
    for any term with standard deviation under ``NO_GRADIENT_SD`` and any pair
    correlated above ``DUPLICATE_R``. Returns the same numbers as a dict.
    """
    out = sys.stdout if out is None else out
    rows = [_as_terms(r) for r in records]
    if terms is None:
        terms = []
        for r in rows:
            for k in r:
                if k != "total" and k not in terms:
                    terms.append(k)
    kept = [r for r in rows if all(r.get(t) is not None for t in terms)]
    excluded = len(rows) - len(kept)
    x = np.array([[float(r[t]) for t in terms] for r in kept], dtype=np.float64).reshape(len(kept), len(terms))
    stats: dict[str, dict[str, float]] = {}
    flags: list[str] = []
    for j, t in enumerate(terms):
        col = x[:, j]
        mean = float(col.mean()) if len(col) else float("nan")
        sd = float(col.std()) if len(col) else float("nan")
        sat = float(np.mean((col <= SATURATION_EPS) | (col >= 1 - SATURATION_EPS))) if len(col) else float("nan")
        stats[t] = {"mean": mean, "sd": sd, "saturated": sat}
        if len(col) and sd < NO_GRADIENT_SD:
            flags.append(f"{t} barely varies (sd {sd:.3f}): it gives the trainer no gradient on this batch")
    corr = np.full((len(terms), len(terms)), np.nan)
    sds = x.std(axis=0) if len(kept) else np.zeros(len(terms))
    for a in range(len(terms)):
        for b in range(len(terms)):
            if len(kept) > 1 and sds[a] > 0 and sds[b] > 0:
                corr[a, b] = float(np.corrcoef(x[:, a], x[:, b])[0, 1])
    for a in range(len(terms)):
        for b in range(a + 1, len(terms)):
            if np.isfinite(corr[a, b]) and corr[a, b] > DUPLICATE_R:
                flags.append(f"{terms[a]} and {terms[b]} are correlated at {corr[a, b]:.2f}: "
                             "the same signal counted twice")
    w = max([len(t) for t in terms] + [8])
    print(f"{len(kept)} rollouts used, {excluded} excluded for infrastructure failures", file=out)
    print(f"{'term':<{w}}  {'mean':>6}  {'sd':>6}  {'saturated':>9}", file=out)
    for t in terms:
        s = stats[t]
        print(f"{t:<{w}}  {s['mean']:6.3f}  {s['sd']:6.3f}  {s['saturated']:9.0%}", file=out)
    print("correlation", file=out)
    print(" " * w + "  " + "  ".join(f"{t[:7]:>7}" for t in terms), file=out)
    for a, t in enumerate(terms):
        cells = "  ".join("    n/a" if not np.isfinite(v) else f"{v:7.2f}" for v in corr[a])
        print(f"{t:<{w}}  {cells}", file=out)
    if flags:
        for f in flags:
            print("WARNING: " + f, file=out)
    else:
        print("No term is flat and no pair moves together above the threshold.", file=out)
    return {"used": len(kept), "excluded": excluded, "terms": terms, "stats": stats,
            "correlation": [[None if not np.isfinite(v) else float(v) for v in row] for row in corr],
            "flags": flags}


def score_batch(batch_root: str, reward: Reward, out_path: str, seed: int = 0, limit: int | None = None,
                step: int = 0) -> list[dict[str, Any]]:
    """Score every indexed candidate in a generated batch folder and append term lines to ``out_path``.

    The batch is a folder written by ``scribe.generate`` (for example the
    base model's replies to the training prompts). Candidates the gate
    rejected score 0 without judge calls. Lines have the same shape as the
    trainer's ``terms.jsonl``, so ``reward_report`` reads either.
    """
    from pathlib import Path

    from backrun.sim import Watercolor
    from scribe.generate import read_index

    root = Path(batch_root)
    lines = []
    for i, rec in enumerate(read_index(root)):
        if limit is not None and i >= limit:
            break
        folder = root / rec.get("dir", rec["id"])
        meta = json.loads((folder / "meta.json").read_text())
        rng = np.random.default_rng([seed, i])
        if rec.get("status") == "ok":
            header, ops = Watercolor.read_jsonl(folder / "strokes.jsonl")
            res = reward.score_gated(gate(ops, header), meta.get("subject", ""), rng, meta.get("prompt_id"))
        else:
            res = reward.score_gated(GateResult(ok=False), meta.get("subject", ""), rng)
        lines.append({"step": step, "kind": "rollout", "id": rec["id"], "prompt_id": meta.get("prompt_id"),
                      "total": res.total, "terms": res.terms, "excluded": res.total is None, "notes": res.notes})
    with open(out_path, "a") as fh:
        for line in lines:
            fh.write(json.dumps(line) + "\n")
    return lines


def main(argv: list[str] | None = None) -> int:
    """Two commands.

    ``python -m scribe.reward score BATCH_ROOT --pool POOL_ROOT --judge-model M --out terms.jsonl``
    scores a generated batch against the rated pool and then prints the
    report. ``python -m scribe.reward report FILE.jsonl [--step N]`` reports
    on logged term values. Each line of FILE is a dict of term values, or a dict with a ``terms``
    key. Lines with ``"kind": "summary"`` are skipped. With ``--step``, only
    lines whose ``step`` equals N are used (the trainer's log writes one line
    per rollout with its step).
    """
    import argparse

    ap = argparse.ArgumentParser(description="Reward term report on a batch of logged rollouts.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    rp = sub.add_parser("report")
    rp.add_argument("file")
    rp.add_argument("--step", type=int, default=None)
    sp = sub.add_parser("score")
    sp.add_argument("batch_root")
    sp.add_argument("--pool", required=True)
    sp.add_argument("--judge-model", required=True)
    sp.add_argument("--out", required=True)
    sp.add_argument("--refs", type=int, default=2)
    sp.add_argument("--mix", type=float, default=0.1, help="share of references drawn from love")
    sp.add_argument("--limit", type=int, default=None)
    sp.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if args.cmd == "score":
        from scribe.client import ChatClient
        from scribe.pool import Pool

        pool = Pool(args.pool)
        try:
            client = ChatClient(ledger=args.out + ".usage.jsonl")
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 1
        refs, mix = args.refs, args.mix

        def references(rng: np.random.Generator, prompt_id: str | None) -> list[str]:
            return [r["path"] for r in pool.sample_references(rng, k=refs, mix=mix, prompt_id=prompt_id)]

        reward = Reward(PairwiseJudge(client, args.judge_model), references, aesthetic=aesthetic_from_env())
        lines = score_batch(args.batch_root, reward, args.out, seed=args.seed, limit=args.limit)
        reward_report(lines)
        print(f"pool manifest {pool.manifest()['sha256'][:12]}")
        return 0
    recs = []
    with open(args.file) as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                if rec.get("kind") == "summary":
                    continue
                if args.step is None or rec.get("step") == args.step:
                    recs.append(rec)
    if not recs:
        print("no matching lines", file=sys.stderr)
        return 1
    reward_report(recs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
