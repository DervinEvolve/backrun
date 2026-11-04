"""Tests for target loading and synthetic targets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from backrun.targets import PAPER_RGB, SYNTHETIC_KINDS, load_target, synthetic_target


@pytest.mark.parametrize("kind", SYNTHETIC_KINDS)
def test_synthetic_shape_and_determinism(kind: str) -> None:
    a = synthetic_target(kind, 64, 48, seed=1)
    b = synthetic_target(kind, 64, 48, seed=1)
    c = synthetic_target(kind, 64, 48, seed=2)
    assert a.shape == (48, 64, 3) and a.dtype == np.uint8
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert a.std() > 5  # not a flat image


def test_horizon_is_lighter_above_than_below() -> None:
    t = synthetic_target("horizon", 128, 96, seed=0).astype(float)
    assert t[:20].mean() > t[-20:].mean() + 40


def test_unknown_kind_raises() -> None:
    with pytest.raises(ValueError):
        synthetic_target("teapot", 32, 32, 0)


def test_letterbox_wide_image(tmp_path: Path) -> None:
    src = np.zeros((50, 200, 3), dtype=np.uint8)
    src[..., 0] = 200
    p = tmp_path / "wide.png"
    Image.fromarray(src).save(p)
    out = load_target(p, 100, 100)
    assert out.shape == (100, 100, 3) and out.dtype == np.uint8
    # Fitted to 100 x 25, centred: bands above and below are paper.
    assert tuple(out[5, 50]) == PAPER_RGB
    assert tuple(out[95, 50]) == PAPER_RGB
    assert out[50, 50, 0] == 200 and out[50, 50, 1] == 0


def test_letterbox_tall_image_and_alpha(tmp_path: Path) -> None:
    src = np.zeros((200, 50, 4), dtype=np.uint8)
    src[..., 2] = 255
    src[..., 3] = 255
    src[:20, :, 3] = 0  # transparent top band becomes paper
    p = tmp_path / "tall.png"
    Image.fromarray(src, mode="RGBA").save(p)
    out = load_target(p, 80, 40)
    assert out.shape == (40, 80, 3)
    assert tuple(out[20, 2]) == PAPER_RGB  # left margin
    assert tuple(out[0, 40]) == PAPER_RGB  # transparent region
    assert tuple(out[30, 40]) == (0, 0, 255)


def test_load_grayscale(tmp_path: Path) -> None:
    p = tmp_path / "g.png"
    Image.fromarray(np.full((30, 40), 77, dtype=np.uint8), mode="L").save(p)
    out = load_target(p, 40, 30)
    assert out.shape == (30, 40, 3) and np.all(out == 77)
