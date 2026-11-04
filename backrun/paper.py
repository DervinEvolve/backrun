"""Paper texture: a height field made of fractal noise plus fibre grain.

Height runs from 0 (deepest valley) to 1 (highest peak). The same seed and
paper kind always give the same field. The numbers in ``PAPER_KINDS`` are
design choices picked by eye, not measurements of real paper.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass(frozen=True)
class PaperKind:
    """Settings for one kind of paper (design choices, not measurements)."""

    base_sigma: float  # blur of the coarsest noise octave, in pixels
    octaves: int  # how many noise octaves to sum, each half the size of the last
    persistence: float  # amplitude of each octave relative to the previous one
    fibre_weight: float  # how much the fibre grain contributes, 0 to 1
    fibre_density: float  # fibres per pixel
    contrast: float  # final squeeze toward 0.5; smaller means smoother paper


PAPER_KINDS: dict[str, PaperKind] = {
    "cold_press": PaperKind(base_sigma=4.0, octaves=4, persistence=0.55,
                            fibre_weight=0.10, fibre_density=0.004, contrast=1.0),
    "hot_press": PaperKind(base_sigma=3.0, octaves=3, persistence=0.45,
                           fibre_weight=0.15, fibre_density=0.003, contrast=0.45),
    "rough": PaperKind(base_sigma=6.0, octaves=5, persistence=0.65,
                       fibre_weight=0.08, fibre_density=0.004, contrast=1.0),
}


def _normalise(field: np.ndarray) -> np.ndarray:
    """Shift and scale ``field`` to zero mean and unit standard deviation."""
    std = float(field.std())
    return (field - field.mean()) / (std if std > 0 else 1.0)


def fractal_noise(width: int, height: int, rng: np.random.Generator,
                  base_sigma: float, octaves: int, persistence: float) -> np.ndarray:
    """Sum of blurred white-noise octaves, coarse to fine, unit variance."""
    total = np.zeros((height, width), dtype=np.float64)
    amplitude = 1.0
    sigma = base_sigma
    for _ in range(octaves):
        layer = gaussian_filter(rng.standard_normal((height, width)), sigma, mode="wrap")
        total += amplitude * _normalise(layer)
        amplitude *= persistence
        sigma = max(sigma / 2.0, 0.5)
    return _normalise(total)


def fibre_grain(width: int, height: int, rng: np.random.Generator,
                density: float) -> np.ndarray:
    """Short random line segments, like paper fibres, slightly blurred."""
    grid = np.zeros((height, width), dtype=np.float64)
    count = max(1, int(width * height * density))
    x0 = rng.uniform(0, width, count)
    y0 = rng.uniform(0, height, count)
    angle = rng.uniform(0, np.pi, count)
    length = rng.uniform(4.0, 14.0, count)
    strength = rng.uniform(0.5, 1.0, count)
    t = np.linspace(-0.5, 0.5, 16)
    xs = (x0[:, None] + np.cos(angle)[:, None] * length[:, None] * t).ravel()
    ys = (y0[:, None] + np.sin(angle)[:, None] * length[:, None] * t).ravel()
    weights = np.repeat(strength, t.size)
    xi = np.mod(np.round(xs).astype(np.int64), width)
    yi = np.mod(np.round(ys).astype(np.int64), height)
    np.add.at(grid, (yi, xi), weights)
    grid = np.minimum(grid, 2.0)
    return _normalise(gaussian_filter(grid, 0.6, mode="wrap"))


def make_paper(width: int, height: int, seed: int, kind: str = "cold_press") -> np.ndarray:
    """Return the paper height field, float32 of shape (height, width), in [0, 1]."""
    if kind not in PAPER_KINDS:
        raise ValueError(f"unknown paper kind {kind!r}; choose from {sorted(PAPER_KINDS)}")
    spec = PAPER_KINDS[kind]
    rng = np.random.default_rng([int(seed), 0x9A9E5])
    noise = fractal_noise(width, height, rng, spec.base_sigma, spec.octaves, spec.persistence)
    fibres = fibre_grain(width, height, rng, spec.fibre_density)
    field = (1.0 - spec.fibre_weight) * noise + spec.fibre_weight * fibres
    lo, hi = np.percentile(field, [0.5, 99.5])
    field = np.clip((field - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    field = 0.5 + (field - 0.5) * spec.contrast
    return field.astype(np.float32)


def height_rank(paper_height: np.ndarray) -> np.ndarray:
    """Percentile rank of each cell's height, 0 for the lowest and 1 for the highest.

    The dry brush uses this so that "skip the lowest 60% of the paper" means the
    same thing on smooth and rough paper.
    """
    flat = paper_height.ravel()
    order = np.argsort(flat, kind="stable")
    rank = np.empty(flat.size, dtype=np.float32)
    rank[order] = np.linspace(0.0, 1.0, flat.size, dtype=np.float32)
    return rank.reshape(paper_height.shape)


def tooth(paper_height: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    """The fine-scale part of the paper height: local peaks and valleys.

    Height minus a blurred copy of itself, so a cell counts as a peak when it
    stands above its immediate surroundings, regardless of the slow swells in
    the sheet. The dry brush uses this to decide what it touches.
    """
    h = paper_height.astype(np.float32)
    return h - gaussian_filter(h, sigma, mode="wrap")


def paper_shading(paper_height: np.ndarray, strength: float = 0.05) -> np.ndarray:
    """Faint light-from-the-top-left shading factor around 1.0, float32 (H, W)."""
    gy, gx = np.gradient(paper_height.astype(np.float32))
    slope = -(gx + gy)
    scale = float(np.abs(slope).max()) or 1.0
    return (1.0 + strength * slope / scale).astype(np.float32)
