"""The stroke language as the model is told about it.

The description is kept short and opinionated on purpose: a long reference
invites a model to invent fields. It names only what the gate accepts.
"""

from __future__ import annotations

from typing import Any

from backrun.pigments import PIGMENT_NAMES

CANVAS_WIDTH = 256
CANVAS_HEIGHT = 192
PAPER = "cold_press"


def canvas_header(seed: int = 0, width: int = CANVAS_WIDTH, height: int = CANVAS_HEIGHT,
                  paper: str = PAPER) -> dict[str, Any]:
    """The canvas header the gate supplies; the model never writes it."""
    return {"op": "canvas", "width": int(width), "height": int(height), "seed": int(seed), "paper": paper}


def system_prompt(seed: int = 0, width: int = CANVAS_WIDTH, height: int = CANVAS_HEIGHT) -> str:
    """The canonical description of the stroke language for the system prompt."""
    max_w = round(0.4 * min(width, height), 1)
    names = ", ".join(PIGMENT_NAMES)
    return f"""You paint watercolours by writing Backrun stroke files. A simulator paints them.
The paper is {width} x {height} pixels, cold press, seed {seed}. x runs 0..{width} left to right, y runs 0..{height} top to bottom.
Write one JSON object per line. Two ops exist, nothing else:
{{"op": "stroke", "path": [[x, y], ...], "width": w, "water": a, "pigments": {{"name": c}}}}
{{"op": "dry", "steps": n}}
Stroke fields:
- path: 2 to 8 points inside the paper, drawn as a smooth curve.
- width: brush width in pixels, 1 to {max_w}.
- water: 0 to 1. High water spreads and blooms; low water gives a dry, broken edge.
- pigments: up to three of these names, each with a concentration 0 to 1:
  {names}.
  An empty pigments object lays clean water.
dry: let the paper dry for n steps. About 1500 steps dries a wet wash. Strokes with no dry between them mix wet in wet.
How to paint:
- Big pale washes first, with plenty of water and low concentration (0.1 to 0.4).
- Then dry, so later strokes keep their edges.
- Then middle values, dry again, then small dark details last.
- Leave some paper bare; watercolour is lighter than you think.
- Use 10 to 80 strokes. Between 6 and 200 are accepted, with at least one dry.
Output only the JSON Lines inside one fenced code block. No comments, no prose inside the block."""
