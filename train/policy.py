"""The learned painter's eyes and hand: observation, network and stroke format.

Observation
-----------

``build_observation`` stacks, at 128 x 96 (``OBS_WIDTH`` x ``OBS_HEIGHT``):

====== ==========================================================
planes content
====== ==========================================================
0-2    the target, RGB in 0..1
3-5    the sheet as it would look dry (``render(predict_dry=True)``), RGB in 0..1
6      the wet mask, 0 or 1
7      per-pixel colour error between target and dry render (CIELAB
       Delta E / 100, computed at canvas size before downsampling)
8-10   the painting phase as a one-hot (``washes``, ``medium``, ``detail``),
       each plane constant
====== ==========================================================

``docs/TRAINING.md`` lists exactly these inputs and calls them nine channels;
with three phases the one-hot alone takes three planes, so the stack has
eleven (``OBS_CHANNELS``). The phase names come from the search painter so
ledgers and observations agree.

Action
------

The network returns ``ACTION_DIM`` = 17 numbers in 0..1:

* 0-5: three path control points, (x, y) each, as fractions of the canvas
  (0 is the first pixel, 1 the last);
* 6: width, mapped linearly to ``WIDTH_RANGE`` (fractions of the canvas's
  shorter side);
* 7: water, as is;
* 8-15: the eight pigment concentrations, in ``backrun.pigments`` order;
  ``concentration_sparsity`` is the training penalty that keeps mixes to one
  or two pigments;
* 16: the wait, the number of drying steps after the stroke, on a log scale
  up to ``MAX_WAIT``.

``policy_output_to_ops`` turns those numbers into ordinary stroke-file ops,
rounded so that they survive JSON exactly; ``encode_stroke`` goes the other
way for the search painter's strokes, which is what Stage A trains on
(``ledger_to_examples``).

The observation builder, the stroke format and ``ledger_to_examples`` need
only NumPy, so CPU rollout workers can use them without PyTorch. The
network (``PolicyNet``) needs PyTorch.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from backrun.painter_search import PHASES, catmull_rom, rgb01_to_lab
from backrun.pigments import PIGMENT_NAMES
from train import data

try:  # PyTorch is only needed for the network itself.
    import torch
    from torch import nn
    from torch.nn import functional as F
except ModuleNotFoundError:  # pragma: no cover - exercised only without torch
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

data.declare_training_module(__name__)

#: Observation size.
OBS_WIDTH = 128
OBS_HEIGHT = 96
#: Phase names, in painting order, from the search painter.
PHASE_NAMES: tuple[str, ...] = tuple(p.name for p in PHASES)
#: Observation planes: target (3), dry render (3), wet mask, error map, phase one-hot.
OBS_CHANNELS = 3 + 3 + 1 + 1 + len(PHASE_NAMES)

#: Action layout.
POINTS = slice(0, 6)
WIDTH = 6
WATER = 7
CONC = slice(8, 8 + len(PIGMENT_NAMES))
WAIT = 8 + len(PIGMENT_NAMES)
ACTION_DIM = WAIT + 1

#: Stroke width range, as fractions of the canvas's shorter side.
WIDTH_RANGE: tuple[float, float] = (0.005, 0.40)
#: The longest wait, in simulator steps (the search painter's phase dry).
MAX_WAIT = 2000
#: Concentrations below this are dropped when turning an action into a stroke.
MIN_CONC = 0.01
#: Error map scale: Delta E of this size maps to 1.
ERROR_SCALE = 100.0


# ---------------------------------------------------------------------------
# Observation.
# ---------------------------------------------------------------------------


def phase_index(phase: str | int) -> int:
    """Index of a phase given by name or number."""
    if isinstance(phase, (int, np.integer)):
        if not 0 <= int(phase) < len(PHASE_NAMES):
            raise ValueError(f"phase index {phase} out of range")
        return int(phase)
    try:
        return PHASE_NAMES.index(str(phase))
    except ValueError:
        raise ValueError(f"unknown phase {phase!r}; expected one of {PHASE_NAMES}") from None


def resize_planes(planes: NDArray, height: int = OBS_HEIGHT, width: int = OBS_WIDTH) -> NDArray[np.float32]:
    """Resize (C, H, W) float planes to (C, height, width).

    Whole-number shrink factors use a block mean (256 x 192 to 128 x 96 is a
    2 x 2 mean); anything else uses Pillow's bilinear filter per plane.
    """
    planes = np.asarray(planes, dtype=np.float32)
    c, h, w = planes.shape
    if (h, w) == (height, width):
        return planes.copy()
    if h % height == 0 and w % width == 0:
        fy, fx = h // height, w // width
        return planes.reshape(c, height, fy, width, fx).mean(axis=(2, 4), dtype=np.float64).astype(np.float32)
    out = np.empty((c, height, width), dtype=np.float32)
    for i in range(c):
        im = Image.fromarray(planes[i], mode="F")
        out[i] = np.asarray(im.resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32)
    return out


def error_map(target: NDArray[np.uint8], render: NDArray[np.uint8]) -> NDArray[np.float32]:
    """Per-pixel CIELAB Delta E between target and render, divided by ``ERROR_SCALE``.

    Uses the painter's own colour conversion, not the scorer's.
    """
    a = rgb01_to_lab(np.asarray(target, dtype=np.float64) / 255.0)
    b = rgb01_to_lab(np.asarray(render, dtype=np.float64) / 255.0)
    return (np.sqrt(np.sum((a - b) ** 2, axis=-1)) / ERROR_SCALE).astype(np.float32)


def build_observation(
    target: NDArray[np.uint8],
    dry_render: NDArray[np.uint8],
    wet_mask: NDArray,
    phase: str | int,
) -> NDArray[np.float32]:
    """Stack the observation planes described in the module docstring.

    ``target`` and ``dry_render`` are uint8 (H, W, 3) at canvas size,
    ``wet_mask`` is (H, W). Returns float32 (``OBS_CHANNELS``, 96, 128).
    """
    target = np.asarray(target, dtype=np.uint8)
    dry_render = np.asarray(dry_render, dtype=np.uint8)
    wet = np.asarray(wet_mask, dtype=np.float32)
    if target.shape != dry_render.shape or target.shape[:2] != wet.shape:
        raise ValueError(f"shapes differ: target {target.shape}, render {dry_render.shape}, wet {wet.shape}")
    spatial = np.concatenate(
        [
            np.moveaxis(target.astype(np.float32) / 255.0, -1, 0),
            np.moveaxis(dry_render.astype(np.float32) / 255.0, -1, 0),
            wet[None],
            error_map(target, dry_render)[None],
        ],
        axis=0,
    )
    small = resize_planes(spatial)
    onehot = np.zeros((len(PHASE_NAMES), OBS_HEIGHT, OBS_WIDTH), dtype=np.float32)
    onehot[phase_index(phase)] = 1.0
    return np.concatenate([small, onehot], axis=0)


def observe(sim: Any, target: NDArray[np.uint8], phase: str | int) -> NDArray[np.float32]:
    """The observation of a live simulator: its dry preview and wet mask."""
    return build_observation(target, sim.render(predict_dry=True), sim.state["wet_mask"], phase)


# ---------------------------------------------------------------------------
# Stroke format.
# ---------------------------------------------------------------------------


def _to_numpy(out: Any) -> NDArray[np.float64]:
    """Turn a tensor or array-like holding one action into a flat float64 array."""
    if hasattr(out, "detach"):
        out = out.detach().cpu().numpy()
    arr = np.asarray(out, dtype=np.float64).reshape(-1)
    if arr.size != ACTION_DIM:
        raise ValueError(f"expected one action of {ACTION_DIM} numbers, got {arr.size}")
    return arr


def wait_from_unit(u: float) -> int:
    """Map 0..1 to a wait of 0..``MAX_WAIT`` steps on a log scale."""
    return int(round(math.expm1(float(np.clip(u, 0.0, 1.0)) * math.log1p(MAX_WAIT))))


def wait_to_unit(steps: int) -> float:
    """Inverse of ``wait_from_unit`` (waits above ``MAX_WAIT`` are clipped)."""
    return math.log1p(min(max(int(steps), 0), MAX_WAIT)) / math.log1p(MAX_WAIT)


def policy_output_to_ops(out: Any, canvas_size: tuple[int, int], min_conc: float = MIN_CONC) -> list[dict[str, Any]]:
    """Turn one action (17 numbers in 0..1) into stroke-file ops.

    ``canvas_size`` is (width, height). Returns a ``stroke`` op and, if the
    wait is at least one step, a ``dry`` op. Numbers are clipped to 0..1 and
    rounded (path to 0.01 px, the rest to 0.001), so writing the ops to JSON
    and reading them back gives the same ops and the same painting.
    Concentrations below ``min_conc`` are dropped; a stroke with none left
    is a clear-water stroke, which is a legitimate watercolour move.
    """
    a = np.clip(_to_numpy(out), 0.0, 1.0)
    w, h = int(canvas_size[0]), int(canvas_size[1])
    short = min(w, h)
    pts = a[POINTS].reshape(3, 2)
    path = [[round(float(x * (w - 1)), 2), round(float(y * (h - 1)), 2)] for x, y in pts]
    lo, hi = WIDTH_RANGE
    width = round(max(1.0, short * (lo + float(a[WIDTH]) * (hi - lo))), 3)
    pigments: dict[str, float] = {}
    for name, c in zip(PIGMENT_NAMES, a[CONC]):
        c = round(float(c), 3)
        if c >= min_conc:
            pigments[name] = c
    ops: list[dict[str, Any]] = [
        {"op": "stroke", "path": path, "width": width, "water": round(float(a[WATER]), 3), "pigments": pigments}
    ]
    wait = wait_from_unit(float(a[WAIT]))
    if wait > 0:
        ops.append({"op": "dry", "steps": wait})
    return ops


def three_points(path: Sequence[Sequence[float]]) -> NDArray[np.float64]:
    """Reduce a stroke path of 2 to 8 points to three: start, spline middle, end.

    The middle point is the Catmull-Rom spline's sample halfway along its
    parameter, so a three-point path comes back unchanged and a two-point
    path gets its midpoint.
    """
    p = np.asarray(path, dtype=np.float64)
    if len(p) == 3:
        return p.copy()
    dense = catmull_rom(p)
    return np.stack([p[0], dense[(len(dense) - 1) // 2], p[-1]])


def encode_stroke(op: Mapping[str, Any], wait: int, canvas_size: tuple[int, int]) -> NDArray[np.float32]:
    """Express a stroke op and the wait after it as an action in 0..1.

    The inverse of ``policy_output_to_ops`` up to rounding and clipping:
    paths are reduced to three points (``three_points``), widths and waits
    outside their ranges are clipped.
    """
    w, h = int(canvas_size[0]), int(canvas_size[1])
    short = min(w, h)
    a = np.zeros(ACTION_DIM, dtype=np.float64)
    pts = three_points(op["path"])
    a[POINTS] = (pts / np.array([max(w - 1, 1), max(h - 1, 1)])).reshape(-1)
    lo, hi = WIDTH_RANGE
    a[WIDTH] = (float(op["width"]) / short - lo) / (hi - lo)
    a[WATER] = float(op.get("water", 0.5))
    conc = op.get("pigments", {})
    a[CONC] = [float(conc.get(n, 0.0)) for n in PIGMENT_NAMES]
    a[WAIT] = wait_to_unit(wait)
    return np.clip(a, 0.0, 1.0).astype(np.float32)


def concentration_sparsity(conc: Any) -> Any:
    """Penalty that is zero when at most two pigments are loaded.

    The sum of all concentrations except the two largest, averaged over the
    batch. Works on a NumPy array or a PyTorch tensor of shape (..., 8).
    """
    if torch is not None and isinstance(conc, torch.Tensor):
        top2 = torch.topk(conc, k=2, dim=-1).values.sum(dim=-1)
        return (conc.sum(dim=-1) - top2).mean()
    c = np.asarray(conc, dtype=np.float64)
    top2 = np.sort(c, axis=-1)[..., -2:].sum(axis=-1)
    return float(np.mean(c.sum(axis=-1) - top2))


# ---------------------------------------------------------------------------
# Ledgers to training examples.
# ---------------------------------------------------------------------------


@dataclass
class Example:
    """One Stage A example: the state before an accepted stroke and that stroke."""

    observation: NDArray[np.float32]
    action: NDArray[np.float32]
    op: dict[str, Any]
    wait: int
    phase: str
    ledger_index: int


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSON Lines file."""
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def ledger_to_examples(run_dir: str | Path, strict: bool = True) -> list[Example]:
    """Turn a recorded search-painter run into (observation, accepted stroke) pairs.

    Replays the run's ledger step by step on a fresh simulator built from
    the canvas header in ``strokes.jsonl``. Before each stroke the
    observation is taken; the stroke's action is the stroke plus the wait,
    which is the sum of the ``dry`` steps that follow it up to the next
    stroke. With ``strict`` (the default) the ledger's ops must equal the
    stroke file's and the replayed sheet must match ``final.png``
    byte for byte.
    """
    from backrun.sim import Watercolor

    run = data.assert_not_sealed(run_dir)
    header, file_ops = Watercolor.read_jsonl(data.assert_not_sealed(run / "strokes.jsonl"))
    lines = _read_jsonl(data.assert_not_sealed(run / "ledger.jsonl"))
    target = data.load_training_image(run / "target.png")
    size = (int(header["width"]), int(header["height"]))
    if target.shape[:2] != (size[1], size[0]):
        raise ValueError(f"target is {target.shape[:2]}, canvas is {size[::-1]}")
    ops = [ln["op"] for ln in lines]
    if strict and ops != file_ops:
        raise ValueError(f"{run}: ledger ops differ from strokes.jsonl")
    sim = Watercolor.from_header(header)
    examples: list[Example] = []
    for i, (line, op) in enumerate(zip(lines, ops)):
        if op["op"] == "stroke":
            wait = 0
            for later in ops[i + 1 :]:
                if later["op"] != "dry":
                    break
                wait += int(later["steps"])
            phase = line.get("phase", PHASE_NAMES[0])
            examples.append(
                Example(
                    observation=observe(sim, target, phase),
                    action=encode_stroke(op, wait, size),
                    op=op,
                    wait=wait,
                    phase=phase,
                    ledger_index=int(line.get("index", i)),
                )
            )
        sim.apply(op)
    final = run / "final.png"
    if strict and final.is_file():
        if not np.array_equal(sim.render(), np.asarray(Image.open(final).convert("RGB"))):
            raise ValueError(f"{run}: replaying the ledger does not reproduce final.png")
    return examples


# ---------------------------------------------------------------------------
# The network.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PolicyConfig:
    """Shape of the policy network."""

    in_channels: int = OBS_CHANNELS
    widths: tuple[int, ...] = (40, 80, 160, 256)
    blocks: tuple[int, ...] = (1, 2, 2, 2)
    hidden: int = 768
    groups: int = 8
    action_dim: int = ACTION_DIM

    def to_json(self) -> dict[str, Any]:
        """The config as plain JSON types."""
        d = asdict(self)
        d["widths"], d["blocks"] = list(self.widths), list(self.blocks)
        return d

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> "PolicyConfig":
        """Inverse of ``to_json``."""
        d = dict(d)
        d["widths"], d["blocks"] = tuple(d["widths"]), tuple(d["blocks"])
        return cls(**d)


def parameter_count(config: PolicyConfig = PolicyConfig()) -> int:
    """Number of trainable parameters ``PolicyNet(config)`` has, worked out by hand.

    Convolutions carry no bias (GroupNorm follows each); GroupNorm has a
    scale and a shift per channel; two coordinate planes are added to the
    input; the head sees average and max pooling side by side.
    """
    w = config.widths
    cin = config.in_channels + 2
    n = 9 * cin * w[0] + 2 * w[0]
    prev = w[0]
    for i, (c, b) in enumerate(zip(w, config.blocks)):
        if i > 0:
            n += 9 * prev * c + 2 * c
        n += b * 2 * (9 * c * c + 2 * c)
        prev = c
    hd = config.hidden
    n += 2 * prev * hd + hd + hd * hd + hd + hd * config.action_dim + config.action_dim
    return n


def _require_torch() -> None:
    """Raise a clear error if PyTorch is missing."""
    if torch is None:
        raise ModuleNotFoundError(
            "PyTorch is needed for the policy network: pip install torch --index-url "
            "https://download.pytorch.org/whl/cpu (or the 'train' extra)"
        )


if torch is not None:

    class _ConvNorm(nn.Module):
        """3 x 3 convolution, GroupNorm, SiLU."""

        def __init__(self, cin: int, cout: int, stride: int, groups: int) -> None:
            super().__init__()
            self.conv = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
            self.norm = nn.GroupNorm(groups, cout)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return F.silu(self.norm(self.conv(x)))

    class _ResBlock(nn.Module):
        """Two 3 x 3 convolutions with a skip connection."""

        def __init__(self, c: int, groups: int) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
            self.norm1 = nn.GroupNorm(groups, c)
            self.conv2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
            self.norm2 = nn.GroupNorm(groups, c)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            y = F.silu(self.norm1(self.conv1(x)))
            y = self.norm2(self.conv2(y))
            return F.silu(x + y)

    class PolicyNet(nn.Module):
        """A small convolutional encoder with a global pooling head.

        Input: (B, ``OBS_CHANNELS``, 96, 128). Two coordinate planes (x and y
        from -1 to 1) are appended so pooled features can still say where
        things are. A stride-2 stem, then four stages (the first at the
        stem's resolution, each later one halving it) of residual blocks,
        then average and max pooling, then a three-layer MLP. Output:
        (B, ``ACTION_DIM``) in 0..1 (a sigmoid), laid out as described in
        the module docstring.
        """

        def __init__(self, config: PolicyConfig = PolicyConfig()) -> None:
            super().__init__()
            self.config = config
            w, g = config.widths, config.groups
            self.stem = _ConvNorm(config.in_channels + 2, w[0], 2, g)
            stages = []
            prev = w[0]
            for i, (c, b) in enumerate(zip(w, config.blocks)):
                layers: list[nn.Module] = []
                if i > 0:
                    layers.append(_ConvNorm(prev, c, 2, g))
                layers += [_ResBlock(c, g) for _ in range(b)]
                stages.append(nn.Sequential(*layers))
                prev = c
            self.stages = nn.Sequential(*stages)
            hd = config.hidden
            self.head = nn.Sequential(
                nn.Linear(2 * prev, hd), nn.SiLU(), nn.Linear(hd, hd), nn.SiLU(), nn.Linear(hd, config.action_dim)
            )

        def forward(self, obs: "torch.Tensor") -> "torch.Tensor":
            b, _, h, w = obs.shape
            ys = torch.linspace(-1.0, 1.0, h, dtype=obs.dtype, device=obs.device)
            xs = torch.linspace(-1.0, 1.0, w, dtype=obs.dtype, device=obs.device)
            gy, gx = torch.meshgrid(ys, xs, indexing="ij")
            coords = torch.stack([gx, gy]).unsqueeze(0).expand(b, -1, -1, -1)
            x = self.stages(self.stem(torch.cat([obs, coords], dim=1)))
            pooled = torch.cat([x.mean(dim=(2, 3)), x.amax(dim=(2, 3))], dim=1)
            return torch.sigmoid(self.head(pooled))

else:  # pragma: no cover

    class PolicyNet:  # type: ignore[no-redef]
        """Placeholder that explains PyTorch is missing."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_torch()


def make_policy(config: PolicyConfig = PolicyConfig(), seed: int = 0) -> "PolicyNet":
    """A freshly initialised policy; the same seed gives the same weights."""
    _require_torch()
    torch.manual_seed(seed)
    return PolicyNet(config)


def save_checkpoint(path: str | Path, model: "PolicyNet", **extra: Any) -> Path:
    """Save weights, config and any extra JSON-able fields to ``path``."""
    _require_torch()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": model.config.to_json(), "state_dict": model.state_dict(), "extra": extra}, path)
    return path


def load_checkpoint(path: str | Path) -> "PolicyNet":
    """Load a policy saved by ``save_checkpoint``, in evaluation mode."""
    _require_torch()
    blob = torch.load(Path(path), map_location="cpu", weights_only=True)
    model = PolicyNet(PolicyConfig.from_json(blob["config"]))
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model


def act(model: "PolicyNet", observation: NDArray[np.float32]) -> NDArray[np.float32]:
    """Run the policy on one observation and return its 17 numbers."""
    _require_torch()
    with torch.no_grad():
        out = model(torch.from_numpy(np.ascontiguousarray(observation, dtype=np.float32))[None])
    return out[0].numpy()


def phase_schedule(n_strokes: int) -> list[str]:
    """Phase of each stroke slot, split by the search painter's phase shares."""
    shares = np.array([p.share for p in PHASES], dtype=np.float64)
    counts = np.floor(shares / shares.sum() * n_strokes).astype(int)
    counts[-1] += n_strokes - counts.sum()
    return [p.name for p, c in zip(PHASES, counts) for _ in range(int(c))]


def rollout(
    policy: Callable[[NDArray[np.float32]], Any],
    target: NDArray[np.uint8],
    sim: Any,
    n_strokes: int,
) -> list[dict[str, Any]]:
    """Paint ``n_strokes`` strokes on ``sim`` with ``policy``; return the ops applied.

    ``policy`` maps one observation to one action (for a network, use
    ``lambda o: act(model, o)``). One observation and one stroke per slot,
    no search; the policy's wait decides the drying between strokes.
    """
    size = (int(target.shape[1]), int(target.shape[0]))
    ops: list[dict[str, Any]] = []
    for phase in phase_schedule(n_strokes):
        for op in policy_output_to_ops(policy(observe(sim, target, phase)), size):
            sim.apply(op)
            ops.append(op)
    return ops
