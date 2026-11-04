"""A tiny stand-in for ``backrun.sim.Watercolor`` used by the tests.

It implements the same API (``apply``, ``step``, ``render``, ``copy``,
``state``) but has no physics: a stroke multiplies flat pigment colour into
the paper along a straight polyline through the control points. That is
enough to test the scorer, recorder and search painter quickly and without
depending on the real simulator.
"""

from __future__ import annotations

import copy as _copy
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

#: Approximate thin-wash colours for the eight standard pigments.
FAKE_PIGMENTS: dict[str, tuple[int, int, int]] = {
    "quinacridone_rose": (226, 80, 130),
    "cadmium_yellow": (250, 205, 40),
    "french_ultramarine": (60, 80, 180),
    "cerulean_blue": (60, 150, 205),
    "burnt_umber": (115, 80, 55),
    "hookers_green": (50, 110, 60),
    "indian_red": (165, 65, 55),
    "paynes_grey": (70, 80, 95),
}

PAPER = np.array([250.0, 248.0, 242.0])


class FakeWatercolor:
    """Flat-colour stamping canvas with the ``Watercolor`` interface."""

    def __init__(self, width: int = 64, height: int = 48, seed: int = 0) -> None:
        self.width = int(width)
        self.height = int(height)
        self.seed = int(seed)
        self.canvas = np.broadcast_to(PAPER, (self.height, self.width, 3)).astype(np.float64)
        self.steps_run = 0
        self.ops_applied = 0

    @property
    def state(self) -> dict[str, Any]:
        """Read-only view of the internal arrays."""
        return {"canvas": self.canvas}

    def apply(self, op: dict[str, Any]) -> None:
        """Apply one ``stroke`` or ``dry`` op."""
        self.ops_applied += 1
        kind = op.get("op")
        if kind == "dry":
            self.step(int(op.get("steps", 1)))
            return
        if kind != "stroke":
            raise ValueError(f"unknown op {kind!r}")
        mask_im = Image.new("L", (self.width, self.height), 0)
        pts = [tuple(map(float, p)) for p in op["path"]]
        ImageDraw.Draw(mask_im).line(pts, fill=255, width=max(1, int(round(op["width"]))), joint="curve")
        mask = np.asarray(mask_im, dtype=np.float64)[..., None] / 255.0
        trans = np.ones(3)
        for name, conc in op.get("pigments", {}).items():
            rgb = np.asarray(FAKE_PIGMENTS.get(name, (128, 128, 128)), dtype=np.float64)
            trans = trans * (rgb / PAPER).clip(0.01, 1.0) ** (2.0 * float(conc))
        self.canvas = self.canvas * (1.0 - mask + mask * trans)

    def step(self, n: int = 1) -> None:
        """Advance time; the fake has no physics so this only counts."""
        self.steps_run += int(n)

    def render(self) -> np.ndarray:
        """Return the canvas as uint8 (H, W, 3)."""
        return np.clip(np.round(self.canvas), 0, 255).astype(np.uint8)

    def copy(self) -> "FakeWatercolor":
        """Deep copy."""
        return _copy.deepcopy(self)
