"""Turn simulator state into an RGB image with Kubelka-Munk compositing.

The deposited pigment forms one mixed layer on the paper. Pigment still
floating in the water forms a second mixed layer on top of it, so a wet
stroke is visible before it dries; once everything is dry that layer is
empty and has no effect. A faint shading from the paper height is applied
last. Colour values are treated as display RGB with no gamma conversion.
"""

from __future__ import annotations

import numpy as np

from backrun.pigments import km_composite, km_layer, mix_ks

# Design choice: a warm off-white, like an unbleached cotton sheet.
PAPER_RGB: tuple[float, float, float] = (0.965, 0.945, 0.905)


def composite(deposited: np.ndarray, fluid: np.ndarray | None, k_table: np.ndarray,
              s_table: np.ndarray, paper_rgb: tuple[float, float, float] = PAPER_RGB) -> np.ndarray:
    """Reflectance of paper plus pigment layers, float64 (H, W, 3) in [0, 1]."""
    paper = np.asarray(paper_rgb, dtype=np.float64)
    k, s = mix_ks(deposited, k_table, s_table)
    r_dep, t_dep = km_layer(k, s)
    refl, _ = km_composite(r_dep, t_dep, paper)
    if fluid is not None and np.any(fluid):
        k, s = mix_ks(fluid, k_table, s_table)
        r_fl, t_fl = km_layer(k, s)
        refl, _ = km_composite(r_fl, t_fl, refl)
    return refl


def to_uint8(rgb: np.ndarray) -> np.ndarray:
    """Clip to [0, 1] and quantise to uint8 with round-half-to-even."""
    return np.round(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)


def render_image(deposited: np.ndarray, fluid: np.ndarray | None, k_table: np.ndarray,
                 s_table: np.ndarray, shading: np.ndarray,
                 paper_rgb: tuple[float, float, float] = PAPER_RGB) -> np.ndarray:
    """Full render: Kubelka-Munk composite times paper shading, uint8 (H, W, 3)."""
    refl = composite(deposited, fluid, k_table, s_table, paper_rgb)
    return to_uint8(refl * shading[..., None])


def save_png(image: np.ndarray, path: str) -> None:
    """Write a uint8 (H, W, 3) image as PNG."""
    from PIL import Image

    Image.fromarray(np.ascontiguousarray(image)).save(path, optimize=False)
