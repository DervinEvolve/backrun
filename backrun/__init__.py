"""Backrun: a small, deterministic watercolor simulator.

The simulator lives in :mod:`backrun.sim`. Paper texture, pigments, the brush
and rendering each have their own module so they can be read and tested on
their own.
"""

from backrun.sim import SimParams, Watercolor

__all__ = ["SimParams", "Watercolor"]
__version__ = "0.0.1"
