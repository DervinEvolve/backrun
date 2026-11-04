"""Compare a painting to a target and return numbers.

This module is the scorer. It is deliberately self-contained: it imports
nothing from any painter module, and painters treat it as a black box that
takes two images and returns a dictionary of numbers.

The returned dictionary has four entries:

``lab_l2``
    Mean over pixels of the squared Euclidean distance between the two
    images in CIELAB (D65 white point). Identical images give 0. A uniform
    colour error of Delta E = 10 gives 100.

``ssim``
    Structural similarity of the two luminance channels (BT.601 luma on the
    0..255 scale), computed with a Gaussian window of standard deviation 1.5
    pixels and the usual constants K1 = 0.01, K2 = 0.03. Identical images
    give 1.0. Higher is better.

``strokes``
    The stroke count, passed in by the caller. The scorer does not count
    strokes itself.

``total``
    A fixed weighted sum where lower is better::

        total = 0.01 * lab_l2 + 4.0 * (1 - ssim) + 0.002 * strokes

    The weights were chosen so that the colour term and the structure term
    are of similar size for a blank sheet against a typical target (both a
    few units), and so that one stroke costs about as much as a small detail
    stroke typically earns. They are fixed; changing them changes every
    published number, so do it only with a version bump.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

#: Weight on ``lab_l2`` in ``total``.
W_LAB: float = 0.01
#: Weight on ``1 - ssim`` in ``total``.
W_SSIM: float = 4.0
#: Weight on the stroke count in ``total``.
W_STROKES: float = 0.002

#: Gaussian window standard deviation for SSIM, in pixels.
SSIM_SIGMA: float = 1.5

# sRGB (D65) to CIE XYZ.
_RGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float64,
)
_WHITE_D65 = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)
_DELTA = 6.0 / 29.0


def _as_float_rgb(image: NDArray) -> NDArray[np.float64]:
    """Return an (H, W, 3) float64 array in 0..1 from a uint8 or float image."""
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"expected an (H, W, 3) image, got shape {arr.shape}")
    arr = arr[..., :3]
    if arr.dtype == np.uint8:
        return arr.astype(np.float64) / 255.0
    out = arr.astype(np.float64)
    if out.max(initial=0.0) > 1.0 + 1e-6:
        out = out / 255.0
    return np.clip(out, 0.0, 1.0)


def _srgb_to_linear(rgb: NDArray[np.float64]) -> NDArray[np.float64]:
    """Undo the sRGB transfer curve on values in 0..1."""
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


_SRGB_LUT = _srgb_to_linear(np.arange(256, dtype=np.float64) / 255.0)

# Single-entry cache of the target's Lab and luma, keyed by its bytes.
_target_cache: dict[str, object] = {"key": None, "lab": None, "luma": None}


def _target_features(target: NDArray) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return (Lab, luma) for the target, reusing the last result if unchanged."""
    arr = np.ascontiguousarray(np.asarray(target))
    key = (arr.shape, str(arr.dtype), hash(arr.tobytes()))
    if _target_cache["key"] != key:
        _target_cache["key"] = key
        _target_cache["lab"] = srgb_to_lab(arr)
        _target_cache["luma"] = luma(arr)
    return _target_cache["lab"], _target_cache["luma"]  # type: ignore[return-value]


def srgb_to_lab(image: NDArray) -> NDArray[np.float64]:
    """Convert an sRGB image to CIELAB under a D65 white point.

    Accepts uint8 (0..255) or float (0..1) input of shape (..., 3) where the
    leading dimensions form an image. Returns float64 with the same shape,
    channels L* (0..100), a*, b*.
    """
    arr = np.asarray(image)
    if arr.dtype == np.uint8 and arr.ndim == 3 and arr.shape[2] >= 3:
        linear = _SRGB_LUT[arr[..., :3]]
    else:
        linear = _srgb_to_linear(_as_float_rgb(image))
    xyz = linear @ _RGB_TO_XYZ.T
    t = xyz / _WHITE_D65
    f = np.cbrt(t)
    small = t <= _DELTA**3
    if small.any():
        f[small] = t[small] / (3.0 * _DELTA**2) + 4.0 / 29.0
    lab = np.empty_like(f)
    lab[..., 0] = 116.0 * f[..., 1] - 16.0
    lab[..., 1] = 500.0 * (f[..., 0] - f[..., 1])
    lab[..., 2] = 200.0 * (f[..., 1] - f[..., 2])
    return lab


def luma(image: NDArray) -> NDArray[np.float64]:
    """Return BT.601 luma of an sRGB image on the 0..255 scale."""
    arr = np.asarray(image)
    if arr.dtype == np.uint8 and arr.ndim == 3:
        rgb = arr[..., :3].astype(np.float64)
    else:
        rgb = _as_float_rgb(image) * 255.0
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def ssim_luminance(image: NDArray, target: NDArray, sigma: float = SSIM_SIGMA) -> float:
    """Mean structural similarity of the luma channels of two images.

    Uses a Gaussian window (``scipy.ndimage.gaussian_filter``) with the given
    standard deviation and the standard constants for an 8-bit dynamic
    range. Returns a float in [-1, 1]; 1 means identical.
    """
    return _ssim_from_luma(luma(image), luma(target), sigma)


def _ssim_from_luma(x: NDArray[np.float64], y: NDArray[np.float64], sigma: float) -> float:
    """SSIM between two luma arrays on the 0..255 scale."""
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2

    # Blur all five moment images in one call: sigma 0 along the stack axis.
    stack = np.stack([x, y, x * x, y * y, x * y])
    m = ndimage.gaussian_filter(stack, sigma=(0.0, sigma, sigma), truncate=3.5, mode="reflect")
    mu_x, mu_y = m[0], m[1]
    sxx = m[2] - mu_x * mu_x
    syy = m[3] - mu_y * mu_y
    sxy = m[4] - mu_x * mu_y
    num = (2.0 * mu_x * mu_y + c1) * (2.0 * sxy + c2)
    den = (mu_x * mu_x + mu_y * mu_y + c1) * (sxx + syy + c2)
    return float(np.mean(num / den))


def lab_l2(image: NDArray, target: NDArray) -> float:
    """Mean squared CIELAB distance between two images of the same shape."""
    diff = srgb_to_lab(image) - srgb_to_lab(target)
    return float(np.mean(np.sum(diff * diff, axis=-1)))


def score(image: NDArray, target: NDArray, strokes: int = 0) -> dict[str, float | int]:
    """Score a painting against a target. Lower ``total`` is better.

    Both images must be (H, W, 3) with the same height and width, either
    uint8 or float in 0..1. ``strokes`` is the number of strokes used so far,
    supplied by the caller. See the module docstring for the definition of
    each entry and the fixed weights in ``total``.
    """
    img = np.asarray(image)
    tgt = np.asarray(target)
    if img.shape[:2] != tgt.shape[:2]:
        raise ValueError(f"image {img.shape} and target {tgt.shape} differ in size")
    tgt_lab, tgt_luma = _target_features(tgt)
    diff = srgb_to_lab(img) - tgt_lab
    l2 = float(np.mean(np.sum(diff * diff, axis=-1)))
    s = _ssim_from_luma(luma(img), tgt_luma, SSIM_SIGMA)
    n = int(strokes)
    total = W_LAB * l2 + W_SSIM * (1.0 - s) + W_STROKES * n
    return {"lab_l2": l2, "ssim": s, "strokes": n, "total": float(total)}
