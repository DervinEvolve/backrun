"""The brush: turn a stroke's path into water and pigment on the grid.

A stroke is a Catmull-Rom spline through 2 to 8 control points. Every cell
gets a coverage between 0 and 1 from its distance to the spline, with a soft
falloff at the edge and a small seeded wobble in the radius so edges are not
machine-perfect. A brush carrying little water only touches the paper's peaks
(dry brush).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

Path = list[list[float]] | list[tuple[float, float]] | np.ndarray


@dataclass(frozen=True)
class BrushParams:
    """Brush settings. Design choices picked by eye."""

    jitter: float = 0.10  # edge wobble as a fraction of the radius (one std)
    softness: float = 0.30  # width of the soft edge as a fraction of the radius
    min_softness: float = 1.0  # soft edge never narrower than this, in pixels
    samples_per_pixel: float = 3.0  # spline samples per pixel of path length
    dry_brush_water: float = 0.35  # below this water load the brush skips valleys
    dry_brush_max_skip: float = 0.8  # fraction of paper skipped at zero water
    dry_brush_edge: float = 0.12  # softness of the skip threshold, in rank units


def catmull_rom(points: Path, samples_per_pixel: float = 3.0) -> np.ndarray:
    """Sample a uniform Catmull-Rom spline through ``points``.

    The end points are repeated so the curve starts and ends on them. Returns
    an (N, 2) float64 array of (x, y) positions, including every control point.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 2:
        raise ValueError("path must be a list of at least two [x, y] points")
    padded = np.vstack([pts[:1], pts, pts[-1:]])
    pieces = []
    for i in range(len(pts) - 1):
        p0, p1, p2, p3 = padded[i], padded[i + 1], padded[i + 2], padded[i + 3]
        length = float(np.hypot(*(p2 - p1)))
        n = max(2, int(np.ceil(length * samples_per_pixel)))
        t = np.linspace(0.0, 1.0, n, endpoint=False)[:, None]
        piece = 0.5 * (2 * p1 + (p2 - p0) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                       + (3 * p1 - p0 - 3 * p2 + p3) * t ** 3)
        pieces.append(piece)
    pieces.append(pts[-1:])
    return np.vstack(pieces)


def distance_to_samples(samples: np.ndarray, width: int, height: int, margin: int) -> np.ndarray:
    """Distance in pixels from every cell centre to the nearest spline sample.

    Marks the samples on a padded grid, finds each cell's nearest marked cell
    with a Euclidean distance transform, then measures to that sample's exact
    position, which keeps sub-pixel accuracy.
    """
    gw, gh = width + 2 * margin, height + 2 * margin
    ix = np.round(samples[:, 0]).astype(np.int64) + margin
    iy = np.round(samples[:, 1]).astype(np.int64) + margin
    keep = (ix >= 0) & (ix < gw) & (iy >= 0) & (iy < gh)
    ix, iy, sx, sy = ix[keep], iy[keep], samples[keep, 0], samples[keep, 1]
    big = np.full((height, width), np.float32(1e6))
    if ix.size == 0:
        return big
    sample_x = np.zeros((gh, gw))
    sample_y = np.zeros((gh, gw))
    sample_x[iy, ix] = sx  # later samples overwrite earlier ones; deterministic
    sample_y[iy, ix] = sy
    empty = np.ones((gh, gw), dtype=bool)
    empty[iy, ix] = False
    _, (ny, nx) = distance_transform_edt(empty, return_indices=True)
    ny = ny[margin:margin + height, margin:margin + width]
    nx = nx[margin:margin + height, margin:margin + width]
    yy, xx = np.mgrid[0:height, 0:width]
    d = np.hypot(xx - sample_x[ny, nx], yy - sample_y[ny, nx])
    return d.astype(np.float32)


def smoothstep(t: np.ndarray) -> np.ndarray:
    """Smooth 0-to-1 ramp for t in [0, 1]."""
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def edge_noise(width: int, height: int, rng: np.random.Generator, scale: float) -> np.ndarray:
    """Smooth unit-variance noise used to wobble the brush edge."""
    n = gaussian_filter(rng.standard_normal((height, width)), max(scale, 1.0), mode="reflect")
    std = float(n.std())
    return np.clip(n / (std if std > 0 else 1.0), -2.5, 2.5).astype(np.float32)


def stroke_coverage(path: Path, stroke_width: float, width: int, height: int,
                    rng: np.random.Generator, params: BrushParams = BrushParams()) -> np.ndarray:
    """How much of the brush touches each cell, float32 (H, W) in [0, 1].

    Full coverage near the spline, falling smoothly to zero at a radius of
    half the stroke width, with the radius wobbled by seeded noise.
    """
    radius = 0.5 * float(stroke_width)
    samples = catmull_rom(path, params.samples_per_pixel)
    margin = int(np.ceil(radius * (1 + 3 * params.jitter))) + 2
    dist = distance_to_samples(samples, width, height, margin)
    noise = edge_noise(width, height, rng, scale=float(np.clip(0.25 * radius, 1.5, 6.0)))
    r_eff = radius * (1.0 + params.jitter * noise)
    soft = max(params.min_softness, params.softness * radius)
    return smoothstep((r_eff - dist) / soft).astype(np.float32)


def dry_brush_contact(water: float, paper_rank: np.ndarray,
                      params: BrushParams = BrushParams()) -> np.ndarray:
    """Which parts of the paper a brush with this water load touches, (H, W).

    A well loaded brush touches everything (all ones). Below
    ``params.dry_brush_water`` the brush rides over the valleys: at zero water
    it skips ``dry_brush_max_skip`` of the paper, starting from the lowest.
    """
    if water >= params.dry_brush_water:
        return np.ones_like(paper_rank, dtype=np.float32)
    skip = params.dry_brush_max_skip * (1.0 - water / params.dry_brush_water)
    return smoothstep((paper_rank - skip) / params.dry_brush_edge + 0.5).astype(np.float32)


def brush_footprint(path: Path, stroke_width: float, water: float, paper_rank: np.ndarray,
                    rng: np.random.Generator, params: BrushParams = BrushParams()) -> np.ndarray:
    """Coverage times dry-brush contact: the share of the load each cell receives."""
    height, width = paper_rank.shape
    cover = stroke_coverage(path, stroke_width, width, height, rng, params)
    return cover * dry_brush_contact(water, paper_rank, params)
