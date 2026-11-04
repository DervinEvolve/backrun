"""Tests for spline rasterisation, edge jitter and dry brush."""

import numpy as np
import pytest

from backrun.brush import BrushParams, brush_footprint, catmull_rom, dry_brush_contact, stroke_coverage
from backrun.paper import height_rank, make_paper


def rng(seed=0):
    return np.random.default_rng(seed)


def test_spline_passes_through_control_points():
    pts = [[10.0, 10.0], [40.0, 30.0], [70.0, 12.0], [90.0, 50.0]]
    samples = catmull_rom(pts)
    for p in pts:
        assert np.min(np.hypot(*(samples - p).T)) < 1e-9


def test_spline_needs_two_points():
    with pytest.raises(ValueError):
        catmull_rom([[1.0, 2.0]])


def test_coverage_is_full_on_path_and_zero_far_away():
    cover = stroke_coverage([[20.0, 32.0], [100.0, 32.0]], 12.0, 128, 64, rng())
    assert cover.dtype == np.float32 and cover.shape == (64, 128)
    assert cover[32, 60] == pytest.approx(1.0)
    assert cover[5, 60] == 0.0
    assert cover[32, 120] == 0.0


def test_edge_is_soft():
    cover = stroke_coverage([[20.0, 32.0], [100.0, 32.0]], 20.0, 128, 64, rng())
    column = cover[:, 60]
    partial = (column > 0.05) & (column < 0.95)
    assert partial.sum() >= 2  # a ramp, not a step
    assert np.all(np.diff(column[32:]) <= 1e-6)  # falls off away from the centre line


def test_jitter_is_seeded():
    path = [[20.0, 32.0], [64.0, 20.0], [100.0, 40.0]]
    a = stroke_coverage(path, 16.0, 128, 64, rng(1))
    b = stroke_coverage(path, 16.0, 128, 64, rng(1))
    c = stroke_coverage(path, 16.0, 128, 64, rng(2))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    no_jitter = stroke_coverage(path, 16.0, 128, 64, rng(1), BrushParams(jitter=0.0))
    assert not np.array_equal(a, no_jitter)


def test_stroke_partly_off_canvas():
    cover = stroke_coverage([[-30.0, 10.0], [30.0, 10.0]], 10.0, 64, 32, rng())
    assert cover[10, 0] == pytest.approx(1.0)
    assert cover[10, 20] == pytest.approx(1.0)


def test_loaded_brush_touches_everything():
    rank = height_rank(make_paper(64, 48, seed=3))
    assert np.all(dry_brush_contact(0.8, rank) == 1.0)


def test_dry_brush_skips_valleys():
    paper = make_paper(128, 96, seed=3)
    rank = height_rank(paper)
    foot = brush_footprint([[0.0, 48.0], [128.0, 48.0]], 200.0, 0.05, rank, rng())
    valleys = rank < 0.3
    peaks = rank > 0.7
    assert foot[valleys].mean() < 0.1
    assert foot[peaks].mean() > 0.9
    # Less water skips more paper.
    wetter = brush_footprint([[0.0, 48.0], [128.0, 48.0]], 200.0, 0.25, rank, rng())
    assert wetter.mean() > foot.mean()
