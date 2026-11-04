"""Tests for the scorer."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from backrun import score as score_mod
from backrun.score import W_LAB, W_SSIM, W_STROKES, lab_l2, score, srgb_to_lab, ssim_luminance


def test_lab_reference_values() -> None:
    img = np.array([[[255, 255, 255], [0, 0, 0], [255, 0, 0], [0, 0, 255]]], dtype=np.uint8)
    lab = srgb_to_lab(img)[0]
    assert lab[0] == pytest.approx([100.0, 0.0, 0.0], abs=1e-3)
    assert lab[1] == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)
    assert lab[2] == pytest.approx([53.24, 80.09, 67.20], abs=0.02)
    assert lab[3] == pytest.approx([32.30, 79.19, -107.86], abs=0.02)


def test_uint8_and_float_inputs_agree() -> None:
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (8, 9, 3), dtype=np.uint8)
    assert np.allclose(srgb_to_lab(img), srgb_to_lab(img / 255.0))


def test_identical_images_score_perfectly() -> None:
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, (32, 40, 3), dtype=np.uint8)
    s = score(img, img, strokes=5)
    assert s["lab_l2"] == pytest.approx(0.0)
    assert s["ssim"] == pytest.approx(1.0)
    assert s["strokes"] == 5
    assert s["total"] == pytest.approx(W_STROKES * 5)


def test_uniform_offset_lab_l2() -> None:
    a = np.full((10, 10, 3), 120, dtype=np.uint8)
    b = np.full((10, 10, 3), 140, dtype=np.uint8)
    d = srgb_to_lab(a[:1, :1])[0, 0] - srgb_to_lab(b[:1, :1])[0, 0]
    assert lab_l2(a, b) == pytest.approx(float(np.sum(d * d)))


def test_total_is_documented_weighted_sum() -> None:
    rng = np.random.default_rng(2)
    a = rng.integers(0, 256, (24, 24, 3), dtype=np.uint8)
    b = rng.integers(0, 256, (24, 24, 3), dtype=np.uint8)
    s = score(a, b, strokes=3)
    assert s["total"] == pytest.approx(W_LAB * s["lab_l2"] + W_SSIM * (1 - s["ssim"]) + W_STROKES * 3)
    assert s["ssim"] == pytest.approx(ssim_luminance(a, b))


def test_closer_image_scores_lower() -> None:
    rng = np.random.default_rng(3)
    target = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    near = np.clip(target.astype(int) + rng.integers(-10, 11, target.shape), 0, 255).astype(np.uint8)
    far = np.full_like(target, 128)
    assert score(near, target)["total"] < score(far, target)["total"]


def test_target_cache_does_not_leak_between_targets() -> None:
    img = np.full((8, 8, 3), 100, dtype=np.uint8)
    t1 = np.full((8, 8, 3), 100, dtype=np.uint8)
    t2 = np.full((8, 8, 3), 200, dtype=np.uint8)
    assert score(img, t1)["lab_l2"] == pytest.approx(0.0)
    assert score(img, t2)["lab_l2"] > 100.0
    assert score(img, t1)["lab_l2"] == pytest.approx(0.0)


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        score(np.zeros((4, 4, 3), np.uint8), np.zeros((4, 5, 3), np.uint8))


def test_scorer_imports_nothing_from_painters() -> None:
    tree = ast.parse(Path(score_mod.__file__).read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert imported <= {"__future__", "numpy", "numpy.typing", "scipy"}, imported
