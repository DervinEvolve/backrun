"""The eight pigments, their Kubelka-Munk coefficients and their behaviour.

Each pigment is described by two colours, as in Curtis et al. (1997): how a
unit-thickness layer looks thinly laid over white paper, and how the same layer
looks over black. From that pair we derive the absorption ``K`` and scattering
``S`` per RGB channel. A transparent pigment looks nearly black over black; an
opaque one keeps much of its colour.

Every number in ``PIGMENTS`` (both colours and the four behaviour numbers) is a
design choice picked to look plausible, not a measurement of real paint.

Behaviour numbers, all in [0, 1]:

- ``density``: how fast the pigment settles out of the water onto the paper,
  as a fraction per step at full settling.
- ``staining``: how hard it is to lift back off once deposited. 1 never lifts.
- ``granulation``: how much it gathers in paper valleys.
- ``diffusion``: how fast it spreads through still water, per step.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

RGB = tuple[float, float, float]


def km_coefficients(on_white: RGB | np.ndarray, on_black: RGB | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Derive Kubelka-Munk absorption K and scattering S from a colour pair.

    ``on_white`` and ``on_black`` are reflectances in (0, 1) per channel of a
    unit-thickness layer over white and over black. Each channel needs
    ``0 < on_black < on_white < 1``. Follows Curtis et al. (1997), section 5.1.
    """
    rw = np.asarray(on_white, dtype=np.float64)
    rb = np.asarray(on_black, dtype=np.float64)
    if np.any(rb <= 0) or np.any(rw >= 1) or np.any(rb >= rw):
        raise ValueError("need 0 < on_black < on_white < 1 in every channel")
    a = 0.5 * (rw + (rb - rw + 1.0) / rb)
    b = np.sqrt(a * a - 1.0)
    x = (b * b - (a - rw) * (a - 1.0)) / (b * (1.0 - rw))
    s = np.arctanh(1.0 / x) / b  # arccoth(x) = arctanh(1 / x)
    k = s * (a - 1.0)
    return k, s


def km_layer(k: np.ndarray, s: np.ndarray, thickness: np.ndarray | float = 1.0,
             eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Reflectance R and transmittance T of a layer with coefficients K, S.

    ``k`` and ``s`` are per unit thickness and broadcast against ``thickness``.
    A tiny ``eps`` is added to both coefficients so an empty layer comes out
    as R = 0, T = 1 instead of 0 / 0.
    """
    k = np.asarray(k, dtype=np.float64) + eps
    s = np.asarray(s, dtype=np.float64) + eps
    a = 1.0 + k / s
    b = np.sqrt(a * a - 1.0)
    bsx = np.minimum(b * s * thickness, 30.0)
    sh = np.sinh(bsx)
    c = a * sh + b * np.cosh(bsx)
    return sh / c, b / c


def km_composite(r1: np.ndarray, t1: np.ndarray, r2: np.ndarray,
                 t2: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Lay layer 1 over layer 2 and return the combined R and T.

    Pass ``t2=None`` for an opaque substrate such as paper; the returned T is
    then zero.
    """
    denom = 1.0 - r1 * r2
    r = r1 + t1 * t1 * r2 / denom
    t = np.zeros_like(r) if t2 is None else t1 * t2 / denom
    return r, t


@dataclass(frozen=True)
class Pigment:
    """One paint: its colour pair and how it behaves in water."""

    name: str
    on_white: RGB
    on_black: RGB
    density: float
    staining: float
    granulation: float
    diffusion: float

    @property
    def ks(self) -> tuple[np.ndarray, np.ndarray]:
        """Kubelka-Munk (K, S), each a length-3 float64 array."""
        return km_coefficients(self.on_white, self.on_black)


# Design choices, not measurements. Transparent pigments are dark over black;
# cadmium yellow, cerulean and Indian red are the more opaque ones.
PIGMENTS: tuple[Pigment, ...] = (
    Pigment("quinacridone_rose", (0.90, 0.36, 0.56), (0.10, 0.03, 0.06),
            density=0.020, staining=0.85, granulation=0.05, diffusion=0.014),
    Pigment("cadmium_yellow", (0.97, 0.84, 0.20), (0.40, 0.33, 0.05),
            density=0.045, staining=0.35, granulation=0.15, diffusion=0.008),
    Pigment("french_ultramarine", (0.30, 0.36, 0.80), (0.04, 0.05, 0.16),
            density=0.035, staining=0.20, granulation=0.85, diffusion=0.010),
    Pigment("cerulean_blue", (0.42, 0.68, 0.86), (0.12, 0.23, 0.32),
            density=0.050, staining=0.15, granulation=0.70, diffusion=0.008),
    Pigment("burnt_umber", (0.60, 0.44, 0.30), (0.09, 0.05, 0.03),
            density=0.045, staining=0.35, granulation=0.55, diffusion=0.009),
    Pigment("hookers_green", (0.36, 0.60, 0.30), (0.03, 0.07, 0.03),
            density=0.020, staining=0.75, granulation=0.10, diffusion=0.013),
    Pigment("indian_red", (0.70, 0.32, 0.28), (0.20, 0.06, 0.05),
            density=0.050, staining=0.55, granulation=0.45, diffusion=0.008),
    Pigment("paynes_grey", (0.42, 0.45, 0.52), (0.05, 0.06, 0.08),
            density=0.030, staining=0.55, granulation=0.20, diffusion=0.011),
)

PIGMENT_NAMES: tuple[str, ...] = tuple(p.name for p in PIGMENTS)


def pigment_index(name: str) -> int:
    """Position of a pigment in ``PIGMENTS``; raises ``KeyError`` for unknown names."""
    try:
        return PIGMENT_NAMES.index(name)
    except ValueError:
        raise KeyError(f"unknown pigment {name!r}; choose from {list(PIGMENT_NAMES)}") from None


def ks_table(pigments: tuple[Pigment, ...] = PIGMENTS) -> tuple[np.ndarray, np.ndarray]:
    """K and S for all pigments, each of shape (n_pigments, 3), float64."""
    pairs = [p.ks for p in pigments]
    return np.stack([k for k, _ in pairs]), np.stack([s for _, s in pairs])


def behaviour_table(pigments: tuple[Pigment, ...] = PIGMENTS) -> dict[str, np.ndarray]:
    """Behaviour numbers as float32 arrays of shape (n_pigments,)."""
    return {
        field: np.array([getattr(p, field) for p in pigments], dtype=np.float32)
        for field in ("density", "staining", "granulation", "diffusion")
    }


def mix_ks(amounts: np.ndarray, k_table: np.ndarray, s_table: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mix pigments in one layer: total K and S per cell, shape (H, W, 3).

    ``amounts`` has shape (n_pigments, H, W); an amount of 1 is one unit
    layer of that pigment. Mixing sums each pigment's K and S weighted by its
    amount, so the result is used with thickness 1.
    """
    k = np.tensordot(amounts, k_table, axes=([0], [0]))
    s = np.tensordot(amounts, s_table, axes=([0], [0]))
    return k, s


def granulation_deposit_term(granulation: np.ndarray, paper_height: np.ndarray) -> np.ndarray:
    """How much faster each pigment settles in each cell, shape (n_pigments, H, W).

    Averages about 1. A granulating pigment settles up to twice as fast in the
    deepest valleys and hardly at all on the highest peaks; a non-granulating
    one settles at the same rate everywhere.
    """
    g = granulation.astype(np.float32)[:, None, None]
    valley = 1.0 - paper_height.astype(np.float32)[None]
    return (1.0 - g) + g * 2.0 * valley


def granulation_lift_term(granulation: np.ndarray, paper_height: np.ndarray) -> np.ndarray:
    """How much faster deposited pigment lifts in each cell, the mirror image.

    Peaks lose pigment more easily than valleys, which also pushes granulating
    pigments into the valleys over time.
    """
    g = granulation.astype(np.float32)[:, None, None]
    peak = paper_height.astype(np.float32)[None]
    return (1.0 - g) + g * 2.0 * peak
