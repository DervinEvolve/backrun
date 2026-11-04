"""The rated pool: candidates, the owner's ratings, coverage and reference sampling.

A pool root holds what ``scribe.generate`` wrote (``index.jsonl``,
``prompts.jsonl`` and one folder per candidate) plus ``ratings.jsonl``, which
the rating page appends to. Each ratings line is
``{"id", "tier": "love" | "okay" | "nope" | null, "at"}``; the last line for
an id wins and a null tier means unrated again. Undo lines also carry
``"undo": true`` so a run of undos walks back through the history.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from scribe.colour import hue_chroma, painted_mask, srgb_to_lab
from scribe.gate import CHROMA_FLOOR, HUE_BINS, LIGHTNESS_BIN, PAINTED_DELTA_E, _blank

TIERS = ("love", "okay", "nope")

# Rough names for 30-degree bins of CIELAB hue angle. Approximate on purpose:
# Lab hue does not line up with everyday colour names exactly.
HUE_NAMES = ("pink-red", "red", "orange", "yellow", "yellow-green", "green", "teal", "cyan",
             "sky blue", "blue", "violet", "magenta")

# A hue or lightness bin counts as present in a painting when at least this
# share of the painting's painted pixels falls in it (design choice).
PRESENT_SHARE = 0.05
# A bin is thin when it is present in fewer than this share of the rated
# paintings, with a floor of THIN_MIN_IMAGES paintings (design choices).
THIN_SHARE = 0.05
THIN_MIN_IMAGES = 3


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def _order_key(cid: str) -> str:
    """A stable shuffled order, so one prompt's candidates are not rated in a row."""
    return hashlib.sha256(cid.encode()).hexdigest()


class Pool:
    """A pool root on disk. Only candidates that passed the gate can be rated."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.ratings_path = self.root / "ratings.jsonl"

    # ---------------------------------------------------------- candidates
    def candidates(self) -> list[dict[str, Any]]:
        """Index lines of candidates with a final render, in rating order (last line per id wins)."""
        p = self.root / "index.jsonl"
        if not p.is_file():
            return []
        by_id: dict[str, dict[str, Any]] = {}
        for line in p.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                by_id[rec["id"]] = rec
        ok = [r for r in by_id.values() if r.get("status") == "ok"]
        return sorted(ok, key=lambda r: _order_key(r["id"]))

    def folder(self, cid: str) -> Path:
        """The folder of a candidate id that is in the index; KeyError otherwise."""
        for rec in self.candidates():
            if rec["id"] == cid:
                return self.root / rec.get("dir", cid)
        raise KeyError(cid)

    def meta(self, cid: str) -> dict[str, Any]:
        """A candidate's meta.json."""
        return json.loads((self.folder(cid) / "meta.json").read_text())

    def prompts(self) -> list[dict[str, Any]]:
        """The pool's copy of the prompts file."""
        p = self.root / "prompts.jsonl"
        if not p.is_file():
            return []
        return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]

    # ------------------------------------------------------------- ratings
    def _lines(self) -> list[dict[str, Any]]:
        if not self.ratings_path.is_file():
            return []
        return [json.loads(x) for x in self.ratings_path.read_text().splitlines() if x.strip()]

    def ratings(self) -> dict[str, str]:
        """Current tier per id (last line wins; unrated ids are left out)."""
        cur: dict[str, str | None] = {}
        for rec in self._lines():
            cur[rec["id"]] = rec.get("tier")
        return {k: v for k, v in cur.items() if v in TIERS}

    def _append(self, rec: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with self.ratings_path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")

    def rate(self, cid: str, tier: str, at: float | None = None) -> None:
        """Record a rating for a candidate in the index."""
        if tier not in TIERS:
            raise ValueError(f"tier must be one of {TIERS}")
        self.folder(cid)  # raises KeyError for unknown ids
        self._append({"id": cid, "tier": tier, "at": time.time() if at is None else at})

    def undo(self, at: float | None = None) -> str | None:
        """Undo the most recent rating not yet undone. Returns its id, or None if nothing to undo."""
        stack: list[tuple[str, str | None]] = []
        cur: dict[str, str | None] = {}
        for rec in self._lines():
            if rec.get("undo"):
                if stack:
                    stack.pop()
            else:
                stack.append((rec["id"], cur.get(rec["id"])))
            cur[rec["id"]] = rec.get("tier")
        if not stack:
            return None
        cid, previous = stack[-1]
        self._append({"id": cid, "tier": previous, "at": time.time() if at is None else at, "undo": True})
        return cid

    def next_unrated(self) -> dict[str, Any] | None:
        """The first candidate in rating order with no tier, or None when all are rated."""
        rated = self.ratings()
        for rec in self.candidates():
            if rec["id"] not in rated:
                return rec
        return None

    # ------------------------------------------------------------- reports
    def summary(self) -> dict[str, Any]:
        """Counts per tier, unrated and total, and the same per generator model."""
        rated = self.ratings()
        cands = self.candidates()
        tiers = Counter(rated.get(c["id"], "unrated") for c in cands)
        per_model: dict[str, Counter] = {}
        for c in cands:
            per_model.setdefault(c["model"], Counter())[rated.get(c["id"], "unrated")] += 1
        keys = TIERS + ("unrated",)
        return {"total": len(cands), "rated": len(cands) - tiers["unrated"],
                "tiers": {k: tiers[k] for k in keys},
                "models": {m: {k: c[k] for k in keys} for m, c in sorted(per_model.items())}}

    def _histograms(self, cid: str) -> tuple[np.ndarray, np.ndarray]:
        from PIL import Image

        meta = self.meta(cid)
        img = np.asarray(Image.open(self.folder(cid) / "final.png").convert("RGB"))
        h, w = img.shape[:2]
        mask = painted_mask(img, _blank(w, h, int(meta.get("seed", 0)), "cold_press"), PAINTED_DELTA_E)
        painted = srgb_to_lab(img)[mask]
        n = max(len(painted), 1)
        hue, chroma = hue_chroma(painted)
        hue_hist = np.histogram(hue[chroma > CHROMA_FLOOR], bins=HUE_BINS, range=(0.0, 360.0))[0] / n
        l_hist = np.histogram(painted[:, 0], bins=np.arange(0.0, 100.0 + LIGHTNESS_BIN, LIGHTNESS_BIN))[0] / n
        return hue_hist, l_hist

    def colour_coverage(self, tiers: tuple[str, ...] = ("love", "okay")) -> dict[str, Any]:
        """Hue and lightness coverage over the rated paintings in ``tiers``.

        Returns the mean share of painted pixels per 30-degree hue bin and per
        10-unit lightness bin, the number of paintings in which each bin is
        present, the thin bins, and a plain-English line naming them so that
        supplement prompts can be aimed at them.
        """
        rated = self.ratings()
        ids = [c["id"] for c in self.candidates() if rated.get(c["id"]) in tiers]
        hue_share = np.zeros(HUE_BINS)
        l_share = np.zeros(10)
        hue_present = np.zeros(HUE_BINS, dtype=int)
        l_present = np.zeros(10, dtype=int)
        for cid in ids:
            hh, lh = self._histograms(cid)
            hue_share += hh
            l_share += lh
            hue_present += hh >= PRESENT_SHARE
            l_present += lh >= PRESENT_SHARE
        n = len(ids)
        if n:
            hue_share /= n
            l_share /= n
        floor = max(THIN_MIN_IMAGES, int(np.ceil(THIN_SHARE * n)))
        hue_bins = [{"range": f"{30 * i}-{30 * (i + 1)}", "name": HUE_NAMES[i],
                     "share": round(float(hue_share[i]), 4), "paintings": int(hue_present[i])}
                    for i in range(HUE_BINS)]
        l_bins = [{"range": f"{10 * i}-{10 * (i + 1)}", "share": round(float(l_share[i]), 4),
                   "paintings": int(l_present[i])} for i in range(10)]
        thin_hues = [b for b in hue_bins if b["paintings"] < floor]
        thin_l = [b for b in l_bins if b["paintings"] < floor]
        if n == 0:
            line = f"No paintings rated {' or '.join(tiers)} yet, so there is no coverage to report."
        else:
            parts = []
            if thin_hues:
                parts.append("hues " + ", ".join(f"{b['name']} ({b['range']} deg)" for b in thin_hues))
            if thin_l:
                parts.append("lightness " + ", ".join(f"L* {b['range']}" for b in thin_l))
            if parts:
                line = (f"Thin in {n} rated paintings (present in fewer than {floor}): "
                        + "; ".join(parts) + ". Supplement prompts could aim at these.")
            else:
                line = f"Every hue and lightness bin appears in at least {floor} of {n} rated paintings."
        return {"paintings": n, "thin_floor": floor, "hue": hue_bins, "lightness": l_bins,
                "thin_hues": [b["name"] for b in thin_hues], "thin_lightness": [b["range"] for b in thin_l],
                "line": line}

    # ---------------------------------------------------------- references
    def sample_references(self, rng: np.random.Generator | int, k: int = 2,
                          tiers: tuple[str, ...] = ("love", "okay"), mix: float = 0.5,
                          prompt_id: str | None = None, exclude: str | None = None) -> list[dict[str, Any]]:
        """Draw ``k`` distinct rated references.

        Each draw comes from the first tier with probability ``mix`` and from
        the remaining tiers otherwise, falling back to whichever has paintings
        left. With ``prompt_id``, references are drawn from that prompt's rated
        paintings when it has at least ``k``, else from the whole pool. The
        candidate named by ``exclude`` is never drawn, so a pool painting is
        not judged against itself. The same seed and the same ratings give
        the same draw.
        """
        if isinstance(rng, (int, np.integer)):
            rng = np.random.default_rng(int(rng))
        rated = self.ratings()
        cands = sorted((c for c in self.candidates() if rated.get(c["id"]) in tiers and c["id"] != exclude),
                       key=lambda c: c["id"])
        if prompt_id is not None:
            same = [c for c in cands if c["prompt_id"] == prompt_id]
            if len(same) >= k:
                cands = same
        first = [c for c in cands if rated[c["id"]] == tiers[0]]
        rest = [c for c in cands if rated[c["id"]] != tiers[0]]
        if len(first) + len(rest) < k:
            raise ValueError(f"pool has {len(first) + len(rest)} rated references in {tiers}; need {k}")
        out = []
        for _ in range(k):
            use_first = (rng.random() < mix and first) or not rest
            group = first if use_first else rest
            c = group.pop(int(rng.integers(len(group))))
            out.append({"id": c["id"], "tier": rated[c["id"]], "prompt_id": c["prompt_id"],
                        "path": str(self.root / c.get("dir", c["id"]) / "final.png")})
        return out

    def manifest(self) -> dict[str, Any]:
        """A hash of the rated pool: every rated id, its tier and the hashes of its files.

        A training run records this so it is clear exactly which pool it used.
        """
        rated = self.ratings()
        rows = []
        for c in sorted(self.candidates(), key=lambda c: c["id"]):
            tier = rated.get(c["id"])
            if tier is None:
                continue
            folder = self.root / c.get("dir", c["id"])
            rows.append(f"{c['id']}\t{tier}\t{_hash_file(folder / 'strokes.jsonl')}\t{_hash_file(folder / 'final.png')}")
        digest = hashlib.sha256("\n".join(rows).encode()).hexdigest()
        counts = Counter(rated[c.split("\t")[0]] for c in rows)
        return {"sha256": digest, "rated": len(rows), "tiers": {t: counts[t] for t in TIERS},
                "root": str(self.root)}
