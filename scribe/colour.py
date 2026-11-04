"""Colour helpers shared by the gate and the pool report."""

from __future__ import annotations

import numpy as np

# D65 white point and the sRGB-to-XYZ matrix.
_WHITE = np.array([0.95047, 1.0, 1.08883])
_M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])


def srgb_to_lab(image: np.ndarray) -> np.ndarray:
    """uint8 (H, W, 3) sRGB to CIELAB, float64 (H, W, 3)."""
    rgb = np.asarray(image, dtype=np.float64) / 255.0
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = lin @ _M.T / _WHITE
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16.0 / 116.0)
    return np.stack([116.0 * f[..., 1] - 16.0, 500.0 * (f[..., 0] - f[..., 1]),
                     200.0 * (f[..., 1] - f[..., 2])], axis=-1)


def hue_chroma(lab: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hue angle in degrees [0, 360) and chroma from Lab values (..., 3)."""
    hue = np.degrees(np.arctan2(lab[..., 2], lab[..., 1])) % 360.0
    return hue, np.hypot(lab[..., 1], lab[..., 2])


def painted_mask(image: np.ndarray, paper: np.ndarray, threshold: float) -> np.ndarray:
    """Pixels whose colour differs from the bare paper by Delta E 1976 above ``threshold``."""
    return np.linalg.norm(srgb_to_lab(image) - srgb_to_lab(paper), axis=-1) > threshold
