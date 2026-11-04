"""The pairwise judge: is the candidate a better watercolour than a reference?

Absolute 0 to 10 scores from a vision judge bunch up near the bottom, so the
judge is only ever asked to pick one of two. Every reference is shown twice,
once with the candidate as A and once as B, which cancels a judge's habit of
preferring one position. With two references that is four comparisons per
rollout, and the score is the share the candidate won.

A reply that is not A or B, or a call that fails, is a comparison failure.
Failures are left out of the share. If more than half of the comparisons
fail, the judge returns None: an infrastructure failure, which a trainer must
leave out of its group. Scoring it zero would punish the model for an outage.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from scribe.client import ClientError, user_message

CRITERION = (
    "A good watercolour has translucent washes, soft wet edges, visible blooms and backruns where "
    "water met damp paint, paper showing through in places, and a composition that reads as the "
    "subject at a glance."
)

MAX_FAIL_SHARE = 0.5  # more failures than this and the judgement is void (design choice)

# The letter itself must be a capital A or B; the words around it may be any case.
_LEAD = re.compile(r"^[\s*_`\"'(\[]*(?i:image\s+|option\s+)?([AB])(?![A-Za-z])")
_ANSWER = re.compile(r"(?i:\b(?:answer|choice|better one|better image|better)\s*(?:is)?\s*[:\-]?)\s*"
                     r"[*_`\"'(\[]*(?i:image\s+)?([AB])(?![A-Za-z])")


def judge_prompt(subject: str) -> str:
    """The question put to the vision model for one pair."""
    return (f"Image A is shown first and image B second. Which of these is the better watercolour of "
            f"{subject}? {CRITERION} Reply with the letter only, A or B, and nothing else.")


def parse_ab(text: str) -> str | None:
    """Read 'A' or 'B' from a judge reply; None if the reply names neither or is ambiguous.

    A reply that starts with the letter counts when the letter stands alone on
    its first line, or when the other letter never appears later (other than
    in "better than B"). Otherwise a single "answer: X" or "the better one is
    X" phrase counts.
    """
    t = (text or "").strip()
    m = _LEAD.match(t)
    if m:
        letter = m.group(1)
        other = "B" if letter == "A" else "A"
        rest = t[m.end():]
        if re.match(r"[\s*_`\"'.)\]]*(\n|$)", rest):  # the letter alone on the first line
            return letter
        if not re.search(rf"(?<![A-Za-z]){other}(?![A-Za-z])", rest):
            return letter
        if re.search(rf"(?i:than)\s+(?i:image\s+)?{other}(?![A-Za-z])", rest):
            return letter
        return None
    found = set(_ANSWER.findall(t))
    if len(found) == 1:
        return found.pop()
    return None


@dataclass
class JudgeResult:
    """Outcome of one rollout's comparisons. ``score`` is None when too many failed."""

    score: float | None
    wins: int
    valid: int
    failures: int
    details: list[dict[str, Any]] = field(default_factory=list)


class PairwiseJudge:
    """Ask a vision model to compare a candidate with references in both orders."""

    def __init__(self, client: Any, model: str, temperature: float = 0.0, max_tokens: int = 8,
                 max_fail_share: float = MAX_FAIL_SHARE, extra: dict[str, Any] | None = None) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_fail_share = max_fail_share
        #: Extra request fields, such as ``{"reasoning": {"enabled": False}}`` for models that
        #: would otherwise spend the whole reply budget thinking and answer with nothing.
        self.extra = dict(extra or {})
        self._plain = False  # set once the model rejects the extra fields with a 400

    def compare(self, first: Any, second: Any, subject: str, tag: str | None = None) -> str | None:
        """One comparison: returns 'A', 'B' or None on a failed or unreadable reply."""
        msg = user_message(judge_prompt(subject), images=[first, second])
        extra = {} if self._plain else self.extra
        try:
            reply = self.client.chat(self.model, [msg], temperature=self.temperature,
                                     max_tokens=self.max_tokens, tag=tag, **extra)
        except ClientError as exc:
            if extra and "HTTP 400" in str(exc):
                self._plain = True
                return self.compare(first, second, subject, tag)
            return None
        return parse_ab(reply.text)

    def score(self, candidate: Any, references: list[Any], subject: str) -> JudgeResult:
        """Compare the candidate with each reference, candidate first and then second."""
        details = []
        wins = valid = 0
        for i, ref in enumerate(references):
            for cand_first in (True, False):
                a, b = (candidate, ref) if cand_first else (ref, candidate)
                ans = self.compare(a, b, subject, tag=f"judge:{i}:{'cand-A' if cand_first else 'cand-B'}")
                won = None if ans is None else (ans == "A") == cand_first
                details.append({"reference": i, "candidate_first": cand_first, "answer": ans, "won": won})
                if won is not None:
                    valid += 1
                    wins += int(won)
        total = len(details)
        failures = total - valid
        if total == 0 or failures > self.max_fail_share * total:
            return JudgeResult(None, wins, valid, failures, details)
        return JudgeResult(wins / valid, wins, valid, failures, details)
