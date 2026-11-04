"""Check the pairwise judge against the rated pool before spending GPU time.

GRPO pushes the model toward whatever the judge rewards, so the judge has to
agree with the ratings before a run is worth launching. This module scores a
sample of rated pool paintings the way the reward does, with references of
the same subject and never the painting itself, and asks two questions.

1. Does the judge's win share separate the keeps (love and okay) from the
   nopes? Reported as the AUC, the chance a random keep outscores a random
   nope, with a bootstrap interval. 0.5 is coin-flipping, 1.0 is perfect.
   Plain agreement (win share above one half for a keep, below for a nope)
   is reported too, but an okay painting judged against okay references
   should win about half the time even for a perfect judge, so agreement is
   capped by the design and the AUC is the number to read.
2. In direct pairs of one keep and one nope of the same subject, how often
   does the judge pick the keep? Both orders are asked, so the order-flip
   rate, how often swapping A and B changes the answer, is reported as well.

Both are split by who rated the painting (the owner's lines at the top of the
ratings file against the rest) and by generator, because a judge can "agree"
by recognising a model's style rather than quality.

    python -m scribe.calibrate POOL --judge-model M --out runs/calib
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from scribe.judge import PairwiseJudge
from scribe.pool import TIERS, Pool

KEEP = ("love", "okay")


def owner_ids(pool: Pool, owner_lines: int) -> set[str]:
    """Ids rated in the first ``owner_lines`` lines of the ratings file."""
    return {rec["id"] for rec in pool._lines()[:owner_lines] if not rec.get("undo")}


def choose_sample(pool: Pool, rng: np.random.Generator, n: int, owner_lines: int) -> list[dict[str, Any]]:
    """All owner-rated candidates plus a half-keep, half-nope draw from the rest, up to ``n`` in all."""
    rated = pool.ratings()
    owners = owner_ids(pool, owner_lines)
    cands = {c["id"]: c for c in pool.candidates() if c["id"] in rated}
    chosen = [cands[i] for i in sorted(owners) if i in cands]
    rest_keep = sorted(i for i in cands if i not in owners and rated[i] in KEEP)
    rest_nope = sorted(i for i in cands if i not in owners and rated[i] == "nope")
    room = max(0, n - len(chosen))
    take_keep = min(len(rest_keep), room // 2)
    take_nope = min(len(rest_nope), room - take_keep)
    for group, k in ((rest_keep, take_keep), (rest_nope, take_nope)):
        picks = rng.choice(len(group), size=k, replace=False) if k else []
        chosen.extend(cands[group[int(i)]] for i in sorted(picks))
    return chosen


def choose_pairs(pool: Pool, rng: np.random.Generator, n: int) -> list[tuple[str, str]]:
    """Up to ``n`` (keep, nope) pairs of the same subject, spread across subjects."""
    rated = pool.ratings()
    by_prompt: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"keep": [], "nope": []})
    for c in sorted(pool.candidates(), key=lambda c: c["id"]):
        t = rated.get(c["id"])
        if t in KEEP:
            by_prompt[c["prompt_id"]]["keep"].append(c["id"])
        elif t == "nope":
            by_prompt[c["prompt_id"]]["nope"].append(c["id"])
    prompts = [p for p in sorted(by_prompt) if by_prompt[p]["keep"] and by_prompt[p]["nope"]]
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    guard = 0
    while len(pairs) < n and prompts and guard < 20 * n:
        guard += 1
        p = prompts[int(rng.integers(len(prompts)))]
        g = by_prompt[p]
        pair = (g["keep"][int(rng.integers(len(g["keep"])))], g["nope"][int(rng.integers(len(g["nope"])))])
        if pair not in seen:
            seen.add(pair)
            pairs.append(pair)
    return pairs


def auc(pos: list[float], neg: list[float]) -> float:
    """Chance that a random positive outscores a random negative; ties count half."""
    if not pos or not neg:
        return float("nan")
    p = np.asarray(pos)[:, None]
    q = np.asarray(neg)[None, :]
    return float(((p > q).sum() + 0.5 * (p == q).sum()) / (len(pos) * len(neg)))


def auc_interval(pos: list[float], neg: list[float], rng: np.random.Generator, draws: int = 1000) -> list[float]:
    """A 95% bootstrap interval for ``auc``, resampling positives and negatives separately."""
    if len(pos) < 2 or len(neg) < 2:
        return [float("nan"), float("nan")]
    vals = []
    pos_a, neg_a = np.asarray(pos), np.asarray(neg)
    for _ in range(draws):
        vals.append(auc(list(pos_a[rng.integers(len(pos_a), size=len(pos_a))]),
                        list(neg_a[rng.integers(len(neg_a), size=len(neg_a))])))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def _split_stats(rows: list[dict[str, Any]], rng: np.random.Generator) -> dict[str, Any]:
    scored = [r for r in rows if r["score"] is not None]
    pos = [r["score"] for r in scored if r["tier"] in KEEP]
    neg = [r["score"] for r in scored if r["tier"] == "nope"]
    decided = [r for r in scored if r["score"] != 0.5]
    agree = sum((r["score"] > 0.5) == (r["tier"] in KEEP) for r in decided)
    flips = sum(r["flips"] for r in scored)
    flip_pairs = sum(r["flip_pairs"] for r in scored)
    return {"n": len(rows), "scored": len(scored), "keeps": len(pos), "nopes": len(neg),
            "auc": auc(pos, neg), "auc_95": auc_interval(pos, neg, rng),
            "mean_win_share": {"keep": float(np.mean(pos)) if pos else None,
                               "nope": float(np.mean(neg)) if neg else None},
            "agreement": (agree / len(decided)) if decided else None, "decided": len(decided),
            "order_flip_rate": (flips / flip_pairs) if flip_pairs else None}


def summarise(rows: list[dict[str, Any]], pairs: list[dict[str, Any]], seed: int = 0) -> dict[str, Any]:
    """Pool-level, per-rater and per-generator numbers from scored rows and direct pairs."""
    rng = np.random.default_rng(seed)
    out: dict[str, Any] = {"all": _split_stats(rows, rng)}
    out["by_rater"] = {who: _split_stats([r for r in rows if r["rater"] == who], rng)
                       for who in sorted({r["rater"] for r in rows})}
    out["by_model"] = {m: _split_stats([r for r in rows if r["model"] == m], rng)
                       for m in sorted({r["model"] for r in rows})}
    valid = [p for p in pairs if p["keep_first"] is not None and p["nope_first"] is not None]
    keep_wins = sum(p["keep_first"] + p["nope_first"] for p in valid)  # each is 1 when the keep won
    flips = sum(p["keep_first"] != p["nope_first"] for p in valid)
    out["direct_pairs"] = {"asked": len(pairs), "valid": len(valid),
                           "keep_picked": (keep_wins / (2 * len(valid))) if valid else None,
                           "order_flip_rate": (flips / len(valid)) if valid else None,
                           "by_rater": {}}
    for who in sorted({p["rater"] for p in valid}):
        sub = [p for p in valid if p["rater"] == who]
        out["direct_pairs"]["by_rater"][who] = {
            "valid": len(sub), "keep_picked": sum(p["keep_first"] + p["nope_first"] for p in sub) / (2 * len(sub))}
    return out


def render_report(summary: dict[str, Any], judge_model: str, refs: int) -> str:
    """The summary as Markdown."""
    def fmt(x: Any) -> str:
        return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.3f}"

    def line(name: str, s: dict[str, Any]) -> str:
        return (f"| {name} | {s['scored']} | {s['keeps']} / {s['nopes']} | {fmt(s['auc'])} "
                f"[{fmt(s['auc_95'][0])}, {fmt(s['auc_95'][1])}] | {fmt(s['mean_win_share']['keep'])} / "
                f"{fmt(s['mean_win_share']['nope'])} | {fmt(s['agreement'])} | {fmt(s['order_flip_rate'])} |")

    head = ("| set | scored | keeps / nopes | AUC [95%] | mean win share keep / nope | agreement | order flips |\n"
            "|---|---|---|---|---|---|---|")
    rows = [line("all", summary["all"])]
    rows += [line(f"rated by {who}", s) for who, s in summary["by_rater"].items()]
    rows += [line(m, s) for m, s in summary["by_model"].items()]
    d = summary["direct_pairs"]
    pair_lines = [f"- {d['valid']} valid pairs of {d['asked']} asked; the judge picked the keep "
                  f"{fmt(d['keep_picked'])} of the time; order flipped the answer in {fmt(d['order_flip_rate'])}."]
    pair_lines += [f"- rated by {who}: {s['valid']} pairs, keep picked {fmt(s['keep_picked'])}"
                   for who, s in d["by_rater"].items()]
    return (f"# Judge calibration\n\nJudge `{judge_model}`, {refs} same-subject references per painting, both "
            f"orders, never the painting itself. AUC is the chance a random keep outscores a random nope "
            f"(0.5 is a coin flip). Agreement is win share above one half for a keep and below for a nope, "
            f"ties left out; it is capped by the design, so read the AUC.\n\n" + head + "\n" + "\n".join(rows)
            + "\n\n## Direct same-subject pairs, one keep against one nope\n\n" + "\n".join(pair_lines) + "\n")


def run(pool: Pool, judge: PairwiseJudge, out: Path, n: int = 300, pairs_n: int = 100, refs: int = 2,
        owner_lines: int = 108, owner_name: str = "owner", other_name: str = "reviewer", seed: int = 0,
        concurrency: int = 8, log: Any = None) -> dict[str, Any]:
    """Score the sample and the direct pairs, write rows and the report under ``out``, return the summary."""
    from PIL import Image

    log = sys.stderr if log is None else log
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng([seed, 1])
    rated = pool.ratings()
    owners = owner_ids(pool, owner_lines)
    sample = choose_sample(pool, rng, n, owner_lines)
    pairs = choose_pairs(pool, np.random.default_rng([seed, 2]), pairs_n)
    subjects = {str(p["id"]): p["subject"] for p in pool.prompts()}

    def image(cid: str) -> np.ndarray:
        return np.asarray(Image.open(pool.folder(cid) / "final.png").convert("RGB"))

    def rater(cid: str) -> str:
        return owner_name if cid in owners else other_name

    def score_one(c: dict[str, Any]) -> dict[str, Any]:
        cid = c["id"]
        draw = np.random.default_rng([seed, 3, int.from_bytes(cid.encode()[:4], "big")])
        chosen = pool.sample_references(draw, k=refs, mix=0.0, prompt_id=c["prompt_id"], exclude=cid)
        res = judge.score(image(cid), [image(r["id"]) for r in chosen], subjects.get(c["prompt_id"], c["prompt_id"]))
        flips = flip_pairs = 0
        for i in range(refs):
            a, b = [d["won"] for d in res.details if d["reference"] == i]
            if a is not None and b is not None:
                flip_pairs += 1
                flips += int(a != b)
        return {"id": cid, "prompt_id": c["prompt_id"], "model": c["model"], "tier": rated[cid], "rater": rater(cid),
                "references": [r["id"] for r in chosen], "same_subject": all(r["prompt_id"] == c["prompt_id"]
                                                                              for r in chosen),
                "score": res.score, "wins": res.wins, "valid": res.valid, "failures": res.failures,
                "flips": flips, "flip_pairs": flip_pairs}

    def pair_one(pair: tuple[str, str]) -> dict[str, Any]:
        keep, nope = pair
        subject = subjects.get(pool.meta(keep)["prompt_id"], "")
        a = judge.compare(image(keep), image(nope), subject, tag=f"pair:{keep}:keep-A")
        b = judge.compare(image(nope), image(keep), subject, tag=f"pair:{keep}:keep-B")
        return {"keep": keep, "nope": nope, "rater": rater(keep) if rater(keep) == rater(nope) else "mixed",
                "keep_first": None if a is None else int(a == "A"),
                "nope_first": None if b is None else int(b == "B")}

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
        rows = list(ex.map(score_one, sample))
        print(f"scored {len(rows)} paintings in {time.time() - t0:.0f} s", file=log)
        pair_rows = list(ex.map(pair_one, pairs))
    print(f"asked {len(pair_rows)} direct pairs; {time.time() - t0:.0f} s in all", file=log)
    with (out / "rows.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    with (out / "pairs.jsonl").open("w") as fh:
        for r in pair_rows:
            fh.write(json.dumps(r) + "\n")
    summary = summarise(rows, pair_rows, seed)
    summary["judge_model"] = judge.model
    summary["refs"] = refs
    summary["same_subject_share"] = float(np.mean([r["same_subject"] for r in rows])) if rows else None
    summary["pool_manifest"] = pool.manifest()["sha256"]
    summary["tiers_in_pool"] = {t: sum(v == t for v in rated.values()) for t in TIERS}
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    (out / "report.md").write_text(render_report(summary, judge.model, refs))
    return summary


def main(argv: list[str] | None = None) -> int:
    """Command line: see the module docstring."""
    import argparse

    ap = argparse.ArgumentParser(description="Check the pairwise judge against the rated pool.")
    ap.add_argument("pool")
    ap.add_argument("--judge-model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=300, help="paintings to score, owner-rated ones always included")
    ap.add_argument("--pairs", type=int, default=100, help="direct keep-against-nope pairs")
    ap.add_argument("--refs", type=int, default=2)
    ap.add_argument("--owner-lines", type=int, default=108, help="leading ratings lines made by the owner")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args(argv)
    from scribe.client import ChatClient

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        client = ChatClient(ledger=out / "usage.jsonl")
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    summary = run(Pool(args.pool), PairwiseJudge(client, args.judge_model), out, n=args.n, pairs_n=args.pairs,
                  refs=args.refs, owner_lines=args.owner_lines, seed=args.seed, concurrency=args.concurrency)
    print((out / "report.md").read_text())
    s = summary["all"]
    print(f"AUC {s['auc']:.3f} over {s['scored']} paintings; report in {out / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
