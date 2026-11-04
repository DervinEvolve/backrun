"""The watercolor simulator: state arrays, the step loop and stroke replay.

A simplified version of Curtis et al., "Computer-Generated Watercolor"
(SIGGRAPH 1997). Each ``step()`` runs, inside the wet mask only:

1. move water down the gradient of ``water + slope_weight * paper_height``
   with an upwind scheme, then damp the velocity;
2. move fluid pigment with the same upwind fluxes and diffuse it;
3. evaporate, faster at the edge of the wet area;
4. deposit fluid pigment onto the paper and lift some back off;
5. update the wet mask.

Backruns are not coded anywhere. If they appear, they come out of these steps.
All numbers in ``SimParams`` are design choices picked by eye.
"""

from __future__ import annotations

import copy as _copy
import json
from dataclasses import dataclass, field
from pathlib import Path as _FsPath
from typing import Any

import numpy as np
from scipy.ndimage import uniform_filter

from backrun.brush import BrushParams, brush_footprint
from backrun.paper import height_rank, make_paper, paper_shading, tooth
from backrun.pigments import (PIGMENT_NAMES, PIGMENTS, behaviour_table, granulation_deposit_term,
                              granulation_lift_term, ks_table, pigment_index)
from backrun.render import render_image

F32 = np.float32


@dataclass(frozen=True)
class SimParams:
    """Physics and brush constants. Design choices, not measurements."""

    wet_threshold: float = 0.01  # water above this makes a cell wet
    slope_weight: float = 0.04  # how much paper height tilts the water surface
    gravity: float = 2.0  # how hard a surface gradient accelerates the water
    stability: float = 0.5  # cap on gravity * depth per face; keeps the explicit scheme from ringing
    damping: float = 0.8  # velocity kept from one step to the next
    max_speed: float = 0.24  # cells per step, per face
    evap_base: float = 0.0015  # water lost per step everywhere wet
    evap_edge: float = 0.03  # extra loss per step at the edge of the wet area
    edge_band: int = 3  # size of the neighbourhood used to find the edge
    water_ref: float = 0.4  # water depth treated as "fully wet" for deposit and lift
    lift_rate: float = 0.04  # deposited pigment lifted per step at full wetness, before staining
    deposit_rate: float = 1.0  # multiplier on every pigment's density
    diffusion_scale: float = 1.0  # multiplier on every pigment's diffusion
    water_load: float = 0.6  # water added by a stroke with water = 1, at full coverage
    pigment_load: float = 1.0  # pigment added by concentration 1, at full coverage
    max_water: float = 1.5  # water depth cap after a stroke
    paper_shading: float = 0.05  # strength of the paper texture shading in the render
    brush: BrushParams = field(default_factory=BrushParams)


@dataclass(frozen=True)
class Coefficients:
    """Per-pigment arrays the step needs, shaped to broadcast over (K, H, W)."""

    density: np.ndarray
    lift: np.ndarray
    diffusion: np.ndarray
    deposit_term: np.ndarray  # (K, H, W)
    lift_term: np.ndarray  # (K, H, W)


def edge_strength(wet: np.ndarray, band: int) -> np.ndarray:
    """0 deep inside the wet area, rising toward 1 at its boundary; 0 when dry."""
    frac = uniform_filter(wet.astype(F32), size=band, mode="constant")
    np.subtract(F32(1.0), frac, out=frac)
    frac *= wet
    return frac


def face_velocity(u: np.ndarray, surface_step: np.ndarray, depth: np.ndarray,
                  is_open: np.ndarray, p: SimParams) -> np.ndarray:
    """New velocity on one family of faces.

    ``surface_step`` is the difference in water surface height across each
    face (far side minus near side) and ``depth`` the deeper of the two cells.
    Water accelerates down the step, the velocity is damped and clipped, and
    closed faces carry nothing. Gravity is capped at ``stability / depth``:
    with this explicit update a checkerboard ripple grows once gravity times
    depth goes above about (1 + damping) / (4 * damping).
    """
    g = np.maximum(depth, F32(1e-3))
    np.divide(F32(p.stability), g, out=g)
    np.minimum(g, F32(p.gravity), out=g)
    g *= surface_step
    out = u - g
    out *= F32(p.damping)
    np.clip(out, F32(-p.max_speed), F32(p.max_speed), out=out)
    out *= is_open
    return out


def upwind_coefficients(u: np.ndarray, v: np.ndarray, shape: tuple[int, int],
                        budget: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-face multipliers for an upwind transport step.

    ``u`` lives on the faces between columns (shape (h, w-1)), positive to the
    right; ``v`` on the faces between rows (shape (h-1, w)), positive down. The
    flux of a quantity ``q`` across a face is ``left * q_left + right * q_right``
    (or ``top`` and ``bottom``). If a cell would send out more than ``budget``
    of its contents in one step, all its outgoing faces are scaled down, so
    nothing can go negative.
    """
    up, un = np.maximum(u, F32(0.0)), np.minimum(u, F32(0.0))
    vp, vn = np.maximum(v, F32(0.0)), np.minimum(v, F32(0.0))
    rate = np.zeros(shape, dtype=F32)
    rate[:, :-1] += up
    rate[:, 1:] -= un
    rate[:-1, :] += vp
    rate[1:, :] -= vn
    np.maximum(rate, F32(budget), out=rate)
    np.divide(F32(budget), rate, out=rate)  # rate now holds the per-cell scale
    up *= rate[:, :-1]
    un *= rate[:, 1:]
    vp *= rate[:-1, :]
    vn *= rate[1:, :]
    return up, un, vp, vn


def apply_fluxes(q: np.ndarray, fx: np.ndarray, fy: np.ndarray) -> None:
    """Move ``fx`` across column faces and ``fy`` across row faces, in place.

    Positive flux goes right (or down). Whatever leaves one cell enters its
    neighbour, so the sum of ``q`` is unchanged up to rounding.
    """
    q[..., :, :-1] -= fx
    q[..., :, 1:] += fx
    q[..., :-1, :] -= fy
    q[..., 1:, :] += fy


def upwind_flux(q: np.ndarray, faces: tuple[np.ndarray, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Upwind fluxes of ``q`` (shape (..., h, w)) for the given face multipliers."""
    left, right, top, bottom = faces
    fx = left * q[..., :, :-1]
    fx += right * q[..., :, 1:]
    fy = top * q[..., :-1, :]
    fy += bottom * q[..., 1:, :]
    return fx, fy


def diffusion_flux(p: np.ndarray, water: np.ndarray, open_x: np.ndarray, open_y: np.ndarray,
                   rate: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Fluxes that even out pigment concentration between wet neighbours.

    The flux across a face is ``rate * (c_a - c_b) * min(water_a, water_b)``
    where ``c`` is pigment per unit water, so pigment spreads toward weaker
    concentration rather than toward smaller amounts. A cell can lose at most
    ``4 * rate`` of its pigment this way in one step.
    """
    inv = np.maximum(water, F32(threshold))
    np.divide(F32(1.0), inv, out=inv)
    c = p * inv
    hx = np.minimum(water[:, :-1], water[:, 1:])
    hx *= open_x
    hy = np.minimum(water[:-1, :], water[1:, :])
    hy *= open_y
    fx = c[..., :, :-1] - c[..., :, 1:]
    fx *= hx
    fx *= rate
    fy = c[..., :-1, :] - c[..., 1:, :]
    fy *= hy
    fy *= rate
    return fx, fy


@dataclass
class PigmentGroup:
    """A contiguous block of pigment planes cropped to the working region.

    ``fluid`` and ``deposited`` are views into the simulator state, so the
    step updates them in place.
    """

    fluid: np.ndarray
    deposited: np.ndarray
    coeffs: Coefficients


def advance(water: np.ndarray, groups: list[PigmentGroup], u: np.ndarray, v: np.ndarray,
            paper: np.ndarray, p: SimParams) -> tuple[np.ndarray, np.ndarray]:
    """One physics step on a (cropped) region.

    Updates ``water`` and every group's ``fluid`` and ``deposited`` in place
    and returns the new face velocities ``(u, v)``. Shapes: water and paper
    (h, w); group arrays (k, h, w); u (h, w-1); v (h-1, w).
    """
    thr = F32(p.wet_threshold)
    wet = water > thr
    open_x = wet[:, :-1] & wet[:, 1:]
    open_y = wet[:-1, :] & wet[1:, :]

    # 1. Move water down the surface gradient, upwind, then damp.
    surface = F32(p.slope_weight) * paper
    surface += water
    u = face_velocity(u, np.diff(surface, axis=1), np.maximum(water[:, :-1], water[:, 1:]), open_x, p)
    v = face_velocity(v, np.diff(surface, axis=0), np.maximum(water[:-1, :], water[1:, :]), open_y, p)
    # Leave room in each cell's budget for diffusion, which runs alongside.
    max_diffusion = max((float(g.coeffs.diffusion.max()) for g in groups), default=0.0)
    faces = upwind_coefficients(u, v, water.shape, budget=1.0 - 4.0 * max_diffusion)

    # 2. Move pigment with the same fluxes and diffuse it, both from the
    # state at the start of the step; then move the water itself.
    for g in groups:
        ax, ay = upwind_flux(g.fluid, faces)
        dx, dy = diffusion_flux(g.fluid, water, open_x, open_y, g.coeffs.diffusion, p.wet_threshold)
        ax += dx
        ay += dy
        apply_fluxes(g.fluid, ax, ay)
        np.maximum(g.fluid, F32(0.0), out=g.fluid)
    wx, wy = upwind_flux(water, faces)
    apply_fluxes(water, wx, wy)
    np.maximum(water, F32(0.0), out=water)

    # 3. Evaporate; pigment stays behind and concentrates.
    loss = edge_strength(wet, p.edge_band)
    loss *= F32(p.evap_edge)
    loss += F32(p.evap_base)
    loss *= wet
    water -= loss
    np.maximum(water, F32(0.0), out=water)

    # 4. Deposit and lift: net = fluid * settle - deposited * lift.
    wetness = water * F32(1.0 / p.water_ref)
    np.minimum(wetness, F32(1.0), out=wetness)
    dryness = F32(1.0) - wetness
    for g in groups:
        settle = g.coeffs.density * dryness
        settle *= g.coeffs.deposit_term
        settle *= g.fluid
        lift = g.coeffs.lift * wetness
        lift *= g.coeffs.lift_term
        lift *= g.deposited
        settle -= lift
        g.fluid -= settle
        g.deposited += settle

    # Cells that fell below the threshold dry out: all fluid pigment deposits.
    newly_dry = wet & (water <= thr)
    if newly_dry.any():
        for g in groups:
            g.deposited[:, newly_dry] += g.fluid[:, newly_dry]
            g.fluid[:, newly_dry] = 0.0
        water[newly_dry] = 0.0
        # 5. Close faces next to newly dry cells.
        still = ~newly_dry
        u *= still[:, :-1] & still[:, 1:]
        v *= still[:-1, :] & still[1:, :]
    return u, v


def flush_tiny(a: np.ndarray, tiny: float = 1e-20) -> None:
    """Set values smaller than ``tiny`` in magnitude to exactly zero, in place.

    Decaying velocities and the far tails of advected pigment otherwise end up
    as float32 subnormal numbers, which make every later operation on them
    many times slower on common CPUs. The amounts removed are far below any
    visible or measurable level.
    """
    a[np.abs(a) < F32(tiny)] = 0.0


def _readonly(a: np.ndarray) -> np.ndarray:
    view = a.view()
    view.flags.writeable = False
    return view


class Watercolor:
    """A sheet of paper, the water on it and the pigment in and on it.

    ``Watercolor(width, height, seed)`` makes a fresh sheet. Use ``apply`` for
    one stroke or dry operation, ``step`` to advance the physics, ``render`` to
    get a uint8 (H, W, 3) image, ``copy`` for an independent deep copy.
    """

    def __init__(self, width: int = 256, height: int = 192, seed: int = 0,
                 paper: str = "cold_press", params: SimParams | None = None) -> None:
        if width < 4 or height < 4:
            raise ValueError("canvas must be at least 4 x 4")
        self.width = int(width)
        self.height = int(height)
        self.seed = int(seed)
        self.paper = paper
        self.params = params if params is not None else SimParams()
        n = len(PIGMENTS)
        shape = (self.height, self.width)
        self.paper_height = make_paper(self.width, self.height, self.seed, paper)
        self.water = np.zeros(shape, dtype=F32)
        self.wet_mask = np.zeros(shape, dtype=bool)
        self.pigment_fluid = np.zeros((n,) + shape, dtype=F32)
        self.pigment_deposited = np.zeros((n,) + shape, dtype=F32)
        self.flow_u = np.zeros(shape, dtype=F32)  # column j: face between j and j+1
        self.flow_v = np.zeros(shape, dtype=F32)  # row i: face between i and i+1
        self.steps_taken = 0
        self.strokes_applied = 0
        self._present = np.zeros(n, dtype=bool)  # pigments that have ever been used
        self._paper_rank = height_rank(tooth(self.paper_height))  # used by the dry brush
        self._shading = paper_shading(self.paper_height, self.params.paper_shading)
        self._k_table, self._s_table = ks_table()
        b = behaviour_table()
        self._coeffs = Coefficients(
            density=(b["density"] * F32(self.params.deposit_rate))[:, None, None],
            lift=(F32(self.params.lift_rate) * (1.0 - b["staining"]))[:, None, None],
            diffusion=np.minimum(b["diffusion"] * F32(self.params.diffusion_scale), F32(0.2))[:, None, None],
            deposit_term=granulation_deposit_term(b["granulation"], self.paper_height),
            lift_term=granulation_lift_term(b["granulation"], self.paper_height),
        )
        # The deposit and lift fractions are not clipped in the step, so check
        # once here that they can never exceed 1.
        if float((self._coeffs.density * self._coeffs.deposit_term).max()) > 1.0:
            raise ValueError("density * deposit_rate is too large: a cell could deposit more than it holds")
        if float((self._coeffs.lift * self._coeffs.lift_term).max()) > 1.0:
            raise ValueError("lift_rate is too large: a cell could lift more than it holds")

    # ------------------------------------------------------------------ ops
    def apply(self, op: dict[str, Any]) -> None:
        """Apply one operation: ``{"op": "stroke", ...}`` or ``{"op": "dry", "steps": n}``."""
        kind = op.get("op")
        if kind == "stroke":
            self._stroke(op)
        elif kind == "dry":
            steps = op.get("steps")
            if not isinstance(steps, int) or isinstance(steps, bool) or steps < 0:
                raise ValueError("dry op needs a non-negative integer 'steps'")
            self.step(steps)
        elif kind == "canvas":
            raise ValueError("the canvas header belongs on the first line of a file, not in apply()")
        else:
            raise ValueError(f"unknown op {kind!r}")

    def _stroke(self, op: dict[str, Any]) -> None:
        path = op.get("path")
        if not isinstance(path, list) or not 2 <= len(path) <= 8:
            raise ValueError("stroke 'path' must have 2 to 8 points")
        width = float(op.get("width", 0))
        if width <= 0:
            raise ValueError("stroke 'width' must be positive")
        water = float(op.get("water", 0.5))
        if not 0.0 <= water <= 1.0:
            raise ValueError("stroke 'water' must be in [0, 1]")
        pigments = op.get("pigments", {})
        if not isinstance(pigments, dict):
            raise ValueError("stroke 'pigments' must map names to concentrations")
        loads = []
        for name, conc in sorted(pigments.items()):
            conc = float(conc)
            if not 0.0 <= conc <= 1.0:
                raise ValueError(f"concentration of {name!r} must be in [0, 1]")
            loads.append((pigment_index(name), conc))

        rng = np.random.default_rng([self.seed, self.strokes_applied, 0xB2054])
        self.strokes_applied += 1
        p = self.params
        cover = brush_footprint(path, width, water, self._paper_rank, rng, p.brush)
        self.water = np.minimum(self.water + F32(water * p.water_load) * cover, F32(p.max_water))
        for idx, conc in loads:
            if conc > 0:
                self.pigment_fluid[idx] += F32(conc * p.pigment_load) * cover
                self._present[idx] = True
        self._settle_dry_cells()

    def _settle_dry_cells(self) -> None:
        """Deposit fluid pigment in cells that are not wet, and update the mask."""
        thr = F32(self.params.wet_threshold)
        dry = self.water <= thr
        if np.any(self.pigment_fluid[:, dry]):
            self.pigment_deposited[:, dry] += self.pigment_fluid[:, dry]
            self.pigment_fluid[:, dry] = 0.0
        self.water[dry] = 0.0
        self.wet_mask = ~dry

    # -------------------------------------------------------------- physics
    def step(self, n: int = 1) -> None:
        """Advance the physics by ``n`` steps."""
        for _ in range(int(n)):
            self._step_once()

    def _step_once(self) -> None:
        self.steps_taken += 1
        wet = self.wet_mask
        rows = np.flatnonzero(wet.any(axis=1))
        if rows.size == 0:
            return
        cols = np.flatnonzero(wet.any(axis=0))
        # Work on the bounding box of the wet area plus a one-cell margin.
        # Outside it nothing is wet, so nothing can move there.
        y0, y1 = max(int(rows[0]) - 1, 0), min(int(rows[-1]) + 2, self.height)
        x0, x1 = max(int(cols[0]) - 1, 0), min(int(cols[-1]) + 2, self.width)
        if y1 - y0 < 2:
            y0, y1 = (y0, y0 + 2) if y0 + 2 <= self.height else (y1 - 2, y1)
        if x1 - x0 < 2:
            x0, x1 = (x0, x0 + 2) if x0 + 2 <= self.width else (x1 - 2, x1)
        ys, xs = slice(y0, y1), slice(x0, x1)
        c = self._coeffs
        groups = [
            PigmentGroup(
                fluid=self.pigment_fluid[run, ys, xs],  # views: updated in place
                deposited=self.pigment_deposited[run, ys, xs],
                coeffs=Coefficients(
                    density=c.density[run], lift=c.lift[run], diffusion=c.diffusion[run],
                    deposit_term=c.deposit_term[run, ys, xs], lift_term=c.lift_term[run, ys, xs]),
            )
            for run in self._pigment_runs()
        ]
        water = self.water[ys, xs]  # a view: updated in place
        u, v = advance(water, groups, self.flow_u[ys, x0:x1 - 1], self.flow_v[y0:y1 - 1, xs],
                       self.paper_height[ys, xs], self.params)
        if self.steps_taken % 8 == 0:
            flush_tiny(u, 1e-12)
            flush_tiny(v, 1e-12)
            flush_tiny(self.pigment_fluid[:, ys, xs])
            flush_tiny(self.pigment_deposited[:, ys, xs])
        self.flow_u[ys, x0:x1 - 1] = u
        self.flow_v[y0:y1 - 1, xs] = v
        self.wet_mask[ys, xs] = water > F32(self.params.wet_threshold)

    def _pigment_runs(self) -> list[slice]:
        """The pigments used so far, as contiguous slices of the pigment axis.

        Slices give views, so the step can work in place without copying
        whole pigment planes in and out.
        """
        runs: list[slice] = []
        start = None
        for i, used in enumerate(list(self._present) + [False]):
            if used and start is None:
                start = i
            elif not used and start is not None:
                runs.append(slice(start, i))
                start = None
        return runs

    # ------------------------------------------------------------- outputs
    def render(self, include_fluid: bool = True, predict_dry: bool = False) -> np.ndarray:
        """The painting as uint8 (H, W, 3).

        Deposited pigment is always drawn. Pigment still in the water is drawn
        on top unless ``include_fluid`` is False; once the sheet is dry there
        is none left and the flag makes no difference.

        ``predict_dry=True`` is a cheap guess at how the sheet will look once
        it has dried: every cell's fluid pigment is treated as deposited where
        it is now, in the same layer as the pigment already on the paper. It
        runs no steps and changes no state, so it ignores whatever the water
        would still carry before drying (edge darkening, blooms, backruns).
        On a dry sheet it gives the same image as a normal render.
        """
        idx = np.flatnonzero(self._present)
        if predict_dry:
            return render_image(self.pigment_deposited[idx] + self.pigment_fluid[idx], None,
                                self._k_table[idx], self._s_table[idx], self._shading)
        fluid = self.pigment_fluid[idx] if include_fluid else None
        return render_image(self.pigment_deposited[idx], fluid, self._k_table[idx],
                            self._s_table[idx], self._shading)

    @property
    def state(self) -> dict[str, np.ndarray]:
        """Read-only views of the state arrays, keyed by the names in the design."""
        return {
            "paper_height": _readonly(self.paper_height),
            "water": _readonly(self.water),
            "wet_mask": _readonly(self.wet_mask),
            "pigment_fluid": _readonly(self.pigment_fluid),
            "pigment_deposited": _readonly(self.pigment_deposited),
            "flow_u": _readonly(self.flow_u),
            "flow_v": _readonly(self.flow_v),
        }

    @property
    def pigment_names(self) -> tuple[str, ...]:
        """Names of the pigments, in the order of the first axis of pigment arrays."""
        return PIGMENT_NAMES

    def total_pigment(self) -> np.ndarray:
        """Fluid plus deposited pigment summed over the sheet, float64 per pigment."""
        return (self.pigment_fluid.sum(axis=(1, 2), dtype=np.float64)
                + self.pigment_deposited.sum(axis=(1, 2), dtype=np.float64))

    def copy(self) -> "Watercolor":
        """An independent deep copy, including step and stroke counters."""
        return _copy.deepcopy(self)

    # -------------------------------------------------------------- replay
    @staticmethod
    def read_jsonl(path: str | _FsPath) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Read a stroke file: returns (canvas header, list of ops)."""
        lines = [ln for ln in _FsPath(path).read_text().splitlines() if ln.strip()]
        if not lines:
            raise ValueError(f"{path}: empty stroke file")
        header = json.loads(lines[0])
        if header.get("op") != "canvas":
            raise ValueError(f"{path}: first line must be a canvas header")
        return header, [json.loads(ln) for ln in lines[1:]]

    @classmethod
    def from_header(cls, header: dict[str, Any], params: SimParams | None = None) -> "Watercolor":
        """A fresh sheet from a canvas header dict."""
        return cls(width=int(header.get("width", 256)), height=int(header.get("height", 192)),
                   seed=int(header.get("seed", 0)), paper=header.get("paper", "cold_press"),
                   params=params)

    @classmethod
    def from_jsonl(cls, path: str | _FsPath, params: SimParams | None = None) -> "Watercolor":
        """Replay a stroke file from a fresh sheet and return the final state."""
        header, ops = cls.read_jsonl(path)
        sim = cls.from_header(header, params)
        for op in ops:
            sim.apply(op)
        return sim
