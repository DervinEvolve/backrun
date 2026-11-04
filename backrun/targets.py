"""Targets for the painter: image files and simple synthetic test pictures.

``load_target`` reads an image file, fits it inside the canvas without
changing its aspect ratio, and pads the rest with a paper-coloured border
(letterboxing). ``synthetic_target`` draws small test pictures with Pillow so
tests and first demos need no image files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFilter, ImageOps

#: Default letterbox colour: an off-white close to blank watercolour paper.
PAPER_RGB: tuple[int, int, int] = (250, 248, 242)

#: Names accepted by ``synthetic_target``.
SYNTHETIC_KINDS: tuple[str, ...] = ("horizon", "discs", "sky")


def load_target(
    path: str | Path,
    width: int,
    height: int,
    background: tuple[int, int, int] = PAPER_RGB,
) -> NDArray[np.uint8]:
    """Load an image file and letterbox it to ``width`` x ``height``.

    The image is rotated according to its EXIF orientation, flattened onto
    ``background`` if it has transparency, scaled with Lanczos filtering so
    that it fits entirely inside the canvas, and centred. Returns uint8 with
    shape (height, width, 3).
    """
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        im.load()
    return letterbox(im, width, height, background)


def letterbox(
    image: Image.Image | NDArray,
    width: int,
    height: int,
    background: tuple[int, int, int] = PAPER_RGB,
) -> NDArray[np.uint8]:
    """Fit a Pillow image or uint8 array inside the canvas, padding the rest."""
    im = Image.fromarray(np.asarray(image)) if not isinstance(image, Image.Image) else image
    if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
        rgba = im.convert("RGBA")
        flat = Image.new("RGBA", rgba.size, background + (255,))
        flat.alpha_composite(rgba)
        im = flat
    im = im.convert("RGB")
    scale = min(width / im.width, height / im.height)
    new_w = max(1, min(width, round(im.width * scale)))
    new_h = max(1, min(height, round(im.height * scale)))
    if (new_w, new_h) != im.size:
        im = im.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), background)
    canvas.paste(im, ((width - new_w) // 2, (height - new_h) // 2))
    return np.asarray(canvas, dtype=np.uint8).copy()


def _vertical_gradient(
    width: int, height: int, top: tuple[float, float, float], bottom: tuple[float, float, float]
) -> NDArray[np.float64]:
    """Return an (H, W, 3) float array blending ``top`` into ``bottom``."""
    t = np.linspace(0.0, 1.0, height)[:, None, None]
    top_a = np.asarray(top, dtype=np.float64)
    bot_a = np.asarray(bottom, dtype=np.float64)
    col = top_a * (1.0 - t) + bot_a * t
    return np.broadcast_to(col, (height, width, 3)).copy()


def _jitter(rng: np.random.Generator, rgb: tuple[int, int, int], amount: float = 12.0) -> tuple[int, int, int]:
    """Nudge a colour by a small seeded amount."""
    out = np.clip(np.asarray(rgb, dtype=np.float64) + rng.uniform(-amount, amount, 3), 0, 255)
    return tuple(int(round(v)) for v in out)  # type: ignore[return-value]


def synthetic_target(kind: str, width: int, height: int, seed: int = 0) -> NDArray[np.uint8]:
    """Draw a simple test target and return uint8 (height, width, 3).

    Kinds:

    ``horizon``
        A pale sky over darker water or land, with a warm sun disc above the
        horizon line.
    ``discs``
        Two overlapping translucent discs on paper; the overlap shows the two
        colours mixed subtractively, as glazes would.
    ``sky``
        A vertical gradient from deep blue at the top to a warm pale glow at
        the bottom.

    The seed shifts positions and colours slightly; the same arguments
    always give the same pixels.
    """
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    rng = np.random.default_rng(seed)
    s = min(width, height)

    if kind == "horizon":
        horizon = int(height * rng.uniform(0.55, 0.65))
        sky = _vertical_gradient(width, horizon, _jitter(rng, (120, 160, 205)), _jitter(rng, (235, 215, 190)))
        ground = _vertical_gradient(
            width, height - horizon, _jitter(rng, (70, 95, 120)), _jitter(rng, (40, 55, 70))
        )
        arr = np.concatenate([sky, ground], axis=0)
        im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        r = s * rng.uniform(0.09, 0.12)
        cx = width * rng.uniform(0.6, 0.75)
        cy = horizon - r * rng.uniform(1.3, 1.8)
        sun = Image.new("RGB", (width, height), _jitter(rng, (240, 150, 60), 8))
        mask = Image.new("L", (width, height), 0)
        ImageDraw.Draw(mask).ellipse((cx - r, cy - r, cx + r, cy + r), fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(radius=max(0.5, s / 160)))
        im.paste(sun, (0, 0), mask)
        return np.asarray(im, dtype=np.uint8).copy()

    if kind == "discs":
        paper = np.ones((height, width, 3), dtype=np.float64)
        colours = [_jitter(rng, (220, 70, 110)), _jitter(rng, (50, 90, 190))]
        r = s * rng.uniform(0.26, 0.32)
        cy = height * rng.uniform(0.45, 0.55)
        centres = [(width * rng.uniform(0.33, 0.4), cy), (width * rng.uniform(0.6, 0.67), cy)]
        for (cx, cyy), col in zip(centres, colours):
            mask = Image.new("L", (width, height), 0)
            ImageDraw.Draw(mask).ellipse((cx - r, cyy - r, cx + r, cyy + r), fill=255)
            m = np.asarray(mask, dtype=np.float64)[..., None] / 255.0
            glaze = np.asarray(col, dtype=np.float64) / 255.0
            paper *= 1.0 - m + m * glaze
        out = paper * np.asarray(PAPER_RGB, dtype=np.float64)
        return np.clip(np.round(out), 0, 255).astype(np.uint8)

    if kind == "sky":
        arr = _vertical_gradient(width, height, _jitter(rng, (40, 70, 140)), _jitter(rng, (245, 220, 185)))
        return np.clip(np.round(arr), 0, 255).astype(np.uint8)

    raise ValueError(f"unknown synthetic target {kind!r}; choose from {SYNTHETIC_KINDS}")
