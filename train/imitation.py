"""Stage A: behaviour cloning of the search painter.

The policy learns to predict the stroke the search painter accepted from
the state of the sheet just before it (``train.policy.ledger_to_examples``).

Example cache
-------------

Replaying a ledger to rebuild observations costs a full simulation, so it
is done once per run folder and kept in a compact cache: one ``.npz`` shard
per run folder under the cache folder. A shard holds, at observation size
(96 x 128):

``target``      uint8 (3, H, W): the target planes, once per painting;
``planes``      uint8 (N, 5, H, W): per example, the dry render (3), the
                wet mask (1) and the error map (1), each scaled by 255 and
                rounded (the error map is clipped to 0..1 first);
``phase``       int8 (N,): phase index;
``action``      float32 (N, 17): the search painter's stroke as an action;
``wait``, ``ledger_index``   int32 (N,);
``meta``        a JSON string: run name and folder, canvas size, the
                ledger's SHA-256 and the cache version.

``decode_observations`` turns these back into the float32 observation with
the same scaling ``build_observation`` uses (uint8 / 255, phase one-hot), so
a decoded observation equals the original within uint8 quantisation (half
of 1/255), except where the error map was above 1 and is clipped. A shard
is rebuilt only if its ledger changed.

Training
--------

Paintings, never single strokes, are split between training and
validation, deterministically from the seed, so no validation painting
shares a target with a training one.

The loss has three parts, logged separately:

* a smooth L1 loss (``beta`` = ``SMOOTH_L1_BETA``) on the six path numbers,
  width, water and wait. Smooth L1 rather than MSE because the target is
  noisy: several candidate strokes score almost the same, the search
  painter keeps one of them at random, so the same state can map to quite
  different strokes. Squared error lets such far-off targets dominate the
  gradient; smooth L1 grows only linearly for them and stays quadratic,
  and so smooth, near the answer.
* binary cross-entropy on the eight concentrations, which are fractions in
  0..1 (soft targets). Its floor is the entropy of the targets, so it does
  not go to zero even for a perfect fit.
* ``train.policy.concentration_sparsity`` on the predicted concentrations,
  times ``sparsity_weight``.

Optimisation: AdamW (no weight decay on norms and biases), a cosine
learning-rate schedule after a linear warmup, gradient clipping, and an
exponential moving average (EMA) of the weights. The EMA copy is what is
validated and saved. Mixed precision is used only on CUDA (bfloat16 where
supported, otherwise float16 with a gradient scaler).

Every epoch writes ``last.pt`` (and ``best.pt`` when the validation loss
improves) through ``train.policy.save_checkpoint``, with the full training
state in ``extra`` so a run resumes from ``last.pt``, and appends one line
to ``train.jsonl``.

Command line::

    python -m train.imitation LEDGER_ROOT [LEDGER_ROOT ...] OUT_DIR --cache DIR
        [--epochs N] [--batch N] [--lr X] [--val-fraction X] [--seed N]
        [--workers N] [--device auto|cpu|cuda] [--limit N] [--config small|default]
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import multiprocessing as mp
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import numpy as np
from numpy.typing import NDArray

from train import data
from train import policy as pol

try:  # PyTorch is needed for training, not for the cache.
    import torch
    from torch.nn import functional as F
except ModuleNotFoundError:  # pragma: no cover - exercised only without torch
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

data.declare_training_module(__name__)

#: Bumped when the shard layout changes; older shards are rebuilt.
CACHE_VERSION = 1
#: Spatial planes stored per example: dry render (3), wet mask, error map.
EXAMPLE_PLANES = 5
#: Transition point of the smooth L1 loss, in action units (0..1).
SMOOTH_L1_BETA = 0.05
#: Action indices trained with the regression loss, and their names in the log.
REGRESSION_PARTS: tuple[tuple[str, Any], ...] = (
    ("points", pol.POINTS),
    ("width", slice(pol.WIDTH, pol.WIDTH + 1)),
    ("water", slice(pol.WATER, pol.WATER + 1)),
    ("wait", slice(pol.WAIT, pol.WAIT + 1)),
)
#: Network shapes selectable from the command line.
CONFIGS: dict[str, pol.PolicyConfig] = {
    "default": pol.PolicyConfig(),
    "small": pol.PolicyConfig(widths=(8, 16, 16, 32), blocks=(1, 1, 1, 1), hidden=64, groups=4),
}


# ---------------------------------------------------------------------------
# Finding run folders.
# ---------------------------------------------------------------------------


def find_runs(ledger_roots: Sequence[str | Path], limit: int | None = None) -> list[Path]:
    """Finished run folders under the ledger roots, sorted by name within each root.

    A root with an ``index.jsonl`` (``train.ledgers``) contributes the
    paintings its index marks done; a root without one contributes every
    subfolder whose ``run.json`` says ``finished``. ``limit`` keeps the
    first N overall.
    """
    from train import ledgers

    runs: list[Path] = []
    for root in ledger_roots:
        root = data.assert_not_sealed(root)
        if (root / ledgers.INDEX_NAME).is_file():
            rows = ledgers.read_index(root)
            found = [Path(r["run_dir"]) for _, r in sorted(rows.items()) if ledgers.is_done(r)]
        else:
            found = []
            for d in sorted(p for p in root.iterdir() if p.is_dir()):
                try:
                    if json.loads((d / "run.json").read_text()).get("status") == "finished":
                        found.append(d)
                except (OSError, json.JSONDecodeError):
                    continue
        runs += [data.assert_not_sealed(r) for r in found]
    return runs[:limit] if limit is not None else runs


# ---------------------------------------------------------------------------
# The example cache (NumPy only).
# ---------------------------------------------------------------------------


def quantise(planes: NDArray) -> NDArray[np.uint8]:
    """Float planes in 0..1 to uint8 (times 255, rounded; values outside 0..1 clipped)."""
    return np.clip(np.rint(np.asarray(planes, dtype=np.float32) * 255.0), 0, 255).astype(np.uint8)


def shard_name(run_dir: str | Path) -> str:
    """Shard file name for a run folder: its name plus a hash of its full path."""
    run = Path(run_dir).resolve()
    return f"{run.name}-{hashlib.sha256(str(run).encode()).hexdigest()[:10]}.npz"


def _read_shard_meta(path: Path) -> dict[str, Any] | None:
    """The meta dict of a shard, or ``None`` if the file is missing or unreadable."""
    try:
        with np.load(data.assert_not_sealed(path), allow_pickle=False) as z:
            return json.loads(str(z["meta"]))
    except (OSError, KeyError, ValueError):
        return None


def cache_run(run_dir: str | Path, cache_dir: str | Path) -> dict[str, Any]:
    """Write (or keep) the shard for one run folder; return its meta.

    The shard is kept if one exists with the current ``CACHE_VERSION`` and
    the same ledger hash. Otherwise the ledger is replayed with
    ``ledger_to_examples`` (strict) and the shard is written atomically.
    """
    run = data.assert_not_sealed(run_dir)
    cache = Path(cache_dir)
    if data.sealed_folder_of(cache) is not None:
        raise data.SealedSetError(f"refusing to write a cache inside a sealed folder: {cache}")
    cache.mkdir(parents=True, exist_ok=True)
    ledger_sha = data.file_sha256(data.assert_not_sealed(run / "ledger.jsonl"))
    path = cache / shard_name(run)
    meta = _read_shard_meta(path)
    if meta and meta.get("version") == CACHE_VERSION and meta.get("ledger_sha256") == ledger_sha:
        return dict(meta, shard=str(path), reused=True)
    examples = pol.ledger_to_examples(run)
    if not examples:
        raise ValueError(f"{run}: no strokes in the ledger")
    obs = np.stack([e.observation for e in examples])
    header = json.loads((run / "strokes.jsonl").read_text().splitlines()[0])
    meta = {
        "version": CACHE_VERSION,
        "run": run.name,
        "run_dir": str(run.resolve()),
        "width": int(header["width"]),
        "height": int(header["height"]),
        "n": len(examples),
        "ledger_sha256": ledger_sha,
        "error_clipped_fraction": float(np.mean(obs[:, 7] > 1.0)),
    }
    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        target=quantise(obs[0, 0:3]),
        planes=quantise(obs[:, 3 : 3 + EXAMPLE_PLANES]),
        phase=np.array([pol.phase_index(e.phase) for e in examples], dtype=np.int8),
        action=np.stack([e.action for e in examples]).astype(np.float32),
        wait=np.array([e.wait for e in examples], dtype=np.int32),
        ledger_index=np.array([e.ledger_index for e in examples], dtype=np.int32),
        meta=np.array(json.dumps(meta)),
    )
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(buf.getvalue())
    os.replace(tmp, path)
    return dict(meta, shard=str(path), reused=False)


def _cache_run_safe(args: tuple[str, str]) -> dict[str, Any]:
    """``cache_run`` that reports a failure as a row instead of raising."""
    run_dir, cache_dir = args
    try:
        return cache_run(run_dir, cache_dir)
    except data.SealedSetError:
        raise
    except Exception as exc:  # one bad ledger must not end a long job
        return {"run_dir": run_dir, "error": f"{type(exc).__name__}: {exc}"}


def build_cache(run_dirs: Sequence[str | Path], cache_dir: str | Path, workers: int = 1) -> dict[str, Any]:
    """Make sure every run folder has a shard; fan out over ``workers`` processes.

    Returns ``{"shards": [paths in run order], "failed": [rows], "built": n,
    "reused": n, "seconds": s}``.
    """
    t0 = time.perf_counter()
    jobs = [(str(r), str(cache_dir)) for r in run_dirs]
    if workers <= 1 or len(jobs) <= 1:
        rows = [_cache_run_safe(j) for j in jobs]
    else:
        with mp.get_context("spawn").Pool(processes=min(workers, len(jobs))) as pool:
            rows = list(pool.imap(_cache_run_safe, jobs, chunksize=1))
    ok = [r for r in rows if "error" not in r]
    return {
        "shards": [r["shard"] for r in ok],
        "failed": [r for r in rows if "error" in r],
        "built": sum(not r["reused"] for r in ok),
        "reused": sum(bool(r["reused"]) for r in ok),
        "seconds": round(time.perf_counter() - t0, 2),
    }


def decode_observations(target: NDArray[np.uint8], planes: NDArray[np.uint8], phase: NDArray) -> NDArray[np.float32]:
    """Cached uint8 planes back to float32 observations, as ``build_observation`` scales them.

    ``target`` is (3, H, W) shared by the batch or (B, 3, H, W); ``planes`` is
    (B, 5, H, W); ``phase`` is (B,). Returns (B, ``OBS_CHANNELS``, H, W).
    """
    planes = np.asarray(planes)
    b, _, h, w = planes.shape
    tgt = np.asarray(target)
    if tgt.ndim == 3:
        tgt = np.broadcast_to(tgt, (b, *tgt.shape))
    out = np.zeros((b, pol.OBS_CHANNELS, h, w), dtype=np.float32)
    out[:, 0:3] = tgt.astype(np.float32) / 255.0
    out[:, 3 : 3 + EXAMPLE_PLANES] = planes.astype(np.float32) / 255.0
    out[np.arange(b), 3 + EXAMPLE_PLANES + np.asarray(phase, dtype=np.int64)] = 1.0
    return out


@dataclass
class ExampleSet:
    """Every cached example of a set of paintings, held in memory as uint8."""

    names: list[str]
    targets: NDArray[np.uint8]  # (P, 3, H, W)
    planes: NDArray[np.uint8]  # (N, 5, H, W)
    painting: NDArray[np.int64]  # (N,) index into names / targets
    phase: NDArray[np.int8]  # (N,)
    action: NDArray[np.float32]  # (N, 17)

    def __len__(self) -> int:
        return int(self.planes.shape[0])

    def observations(self, idx: NDArray[np.int64]) -> NDArray[np.float32]:
        """Decoded float32 observations for example indices ``idx``."""
        return decode_observations(self.targets[self.painting[idx]], self.planes[idx], self.phase[idx])

    def subset(self, paintings: Sequence[int]) -> "ExampleSet":
        """The examples of the given paintings only (indices into ``names``)."""
        keep = np.asarray(sorted(paintings), dtype=np.int64)
        remap = -np.ones(len(self.names), dtype=np.int64)
        remap[keep] = np.arange(len(keep))
        mask = remap[self.painting] >= 0
        return ExampleSet(
            names=[self.names[i] for i in keep],
            targets=self.targets[keep],
            planes=self.planes[mask],
            painting=remap[self.painting[mask]],
            phase=self.phase[mask],
            action=self.action[mask],
        )


def load_examples(shards: Sequence[str | Path]) -> ExampleSet:
    """Load cache shards into one ``ExampleSet`` (paintings in shard order)."""
    names, targets, planes, painting, phase, action = [], [], [], [], [], []
    for i, s in enumerate(shards):
        with np.load(data.assert_not_sealed(s), allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            names.append(f"{meta['run']}:{Path(meta['run_dir']).parent.name}")
            targets.append(z["target"])
            planes.append(z["planes"])
            painting.append(np.full(len(z["phase"]), i, dtype=np.int64))
            phase.append(z["phase"])
            action.append(z["action"])
    if not names:
        raise ValueError("no cache shards to load")
    return ExampleSet(names, np.stack(targets), np.concatenate(planes), np.concatenate(painting),
                      np.concatenate(phase), np.concatenate(action))


def split_paintings(names: Sequence[str], val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    """Split painting indices into (train, validation), by painting, from the seed.

    The split depends only on the set of names, the fraction and the seed,
    not on their order. With at least two paintings and a fraction above
    zero, validation gets at least one painting and training keeps at least
    one.
    """
    order = sorted(range(len(names)), key=lambda i: names[i])
    perm = np.random.default_rng(seed).permutation(len(order))
    shuffled = [order[int(j)] for j in perm]
    n_val = int(round(val_fraction * len(names)))
    if val_fraction > 0 and len(names) >= 2:
        n_val = min(max(n_val, 1), len(names) - 1)
    else:
        n_val = 0
    return sorted(shuffled[n_val:]), sorted(shuffled[:n_val])


# ---------------------------------------------------------------------------
# Training (PyTorch).
# ---------------------------------------------------------------------------


@dataclass
class TrainSettings:
    """Everything that controls a Stage A run."""

    epochs: int = 20
    batch: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.05
    warmup_fraction: float = 0.05
    clip: float = 1.0
    ema_decay: float = 0.999
    sparsity_weight: float = 0.0
    label_smoothing: float = 0.05
    val_fraction: float = 0.1
    seed: int = 0
    workers: int = 1
    device: str = "auto"
    limit: int | None = None
    config: str = "default"
    threads: int | None = None
    resume: bool = True

    def to_json(self) -> dict[str, Any]:
        """The settings as a plain dict."""
        return asdict(self)


def loss_parts(out: "pol.PolicyOutput", target: dict[str, "torch.Tensor"], sparsity_weight: float,
               label_smoothing: float = 0.0) -> dict[str, "torch.Tensor"]:
    """The loss of a batch, by part, plus ``total``, and the head accuracies. Computed in float32.

    ``out`` is the network's output with its conditional head run at the
    true start cells; ``target`` holds the true ``cell`` (B,), ``offset``
    (B, 2) and ``bins`` (B, 15) from ``train.policy.action_to_bins``.

    Parts: ``cell`` (cross-entropy over the start cells), ``offset``
    (smooth L1 on the sub-cell offset at the true cell), one
    ``bin_<name>`` cross-entropy per binned number and their mean
    ``bins``, and ``sparsity`` (``train.policy.concentration_sparsity`` on
    the expected decoded concentrations). ``total`` = cell + offset + bins +
    ``sparsity_weight`` x sparsity. Accuracies (no gradient): ``cell_top1``,
    ``cell_top5`` and ``bins_within1``, the share of binned numbers whose
    most likely bin is within one bin of the true one.
    """
    cell_logits = out.cell_logits.float()
    bin_logits = out.bin_logits.float()
    cell, offset, tb = target["cell"], target["offset"].float(), target["bins"]
    b, n, k = bin_logits.shape
    parts: dict[str, torch.Tensor] = {}
    parts["cell"] = F.cross_entropy(cell_logits, cell, label_smoothing=label_smoothing)
    pred_off = out.offsets.float()[torch.arange(b, device=cell.device), cell]
    parts["offset"] = F.smooth_l1_loss(pred_off, offset, beta=SMOOTH_L1_BETA)
    per = F.cross_entropy(bin_logits.reshape(b * n, k), tb.reshape(-1), label_smoothing=label_smoothing,
                          reduction="none").view(b, n).mean(dim=0)
    for i, name in enumerate(pol.BIN_NAMES):
        parts[f"bin_{name}"] = per[i]
    parts["bins"] = per.mean()
    centres = (torch.arange(k, device=bin_logits.device, dtype=torch.float32) + 0.5) / k
    centres[0] = 0.0  # the lowest concentration bin decodes to no pigment
    conc = (torch.softmax(bin_logits[:, pol.BIN_CONC], dim=-1) * centres).sum(dim=-1)
    parts["sparsity"] = pol.concentration_sparsity(conc)
    parts["total"] = parts["cell"] + parts["offset"] + parts["bins"] + sparsity_weight * parts["sparsity"]
    with torch.no_grad():
        top5 = cell_logits.topk(min(5, cell_logits.shape[1]), dim=1).indices
        parts["cell_top1"] = (top5[:, 0] == cell).float().mean()
        parts["cell_top5"] = (top5 == cell[:, None]).any(dim=1).float().mean()
        parts["bins_within1"] = ((bin_logits.argmax(dim=-1) - tb).abs() <= 1).float().mean()
    return parts


def _require_torch() -> None:
    """Raise a clear error if PyTorch is missing."""
    if torch is None:
        raise ModuleNotFoundError("PyTorch is needed for training: pip install torch (or the 'train' extra)")


def pick_device(name: str) -> "torch.device":
    """``auto`` is CUDA when available, else CPU."""
    _require_torch()
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda asked for, but CUDA is not available")
    return torch.device(name)


def lr_factor(step: int, total_steps: int, warmup_steps: int) -> float:
    """Learning-rate multiplier: linear warmup to 1, then cosine down to 0."""
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    span = max(1, total_steps - warmup_steps)
    progress = min(1.0, (step - warmup_steps) / span)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


@dataclass
class _Ema:
    """Exponential moving average of a model's parameters and buffers.

    The decay ramps up as ``min(decay, (1 + t) / (10 + t))`` so short runs
    are not dominated by the initial weights.
    """

    model: Any
    decay: float
    updates: int = 0
    shadow: Any = field(default=None)

    def __post_init__(self) -> None:
        import copy

        self.shadow = copy.deepcopy(self.model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad() if torch is not None else (lambda f: f)
    def update(self) -> None:
        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))
        src = self.model.state_dict()
        for k, v in self.shadow.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(src[k].detach(), alpha=1 - d)
            else:
                v.copy_(src[k])


def _batches(n: int, batch: int, rng: np.random.Generator | None) -> Iterator[NDArray[np.int64]]:
    """Index batches over ``n`` examples, shuffled when ``rng`` is given."""
    order = rng.permutation(n) if rng is not None else np.arange(n)
    for i in range(0, n, batch):
        yield order[i : i + batch]


def _to_device(examples: ExampleSet, idx: NDArray[np.int64], device: "torch.device",
               config: "pol.PolicyConfig") -> tuple["torch.Tensor", dict[str, "torch.Tensor"]]:
    """Decode a batch on the device: uint8 goes over, float32 is made there.

    The actions are discretised here (``train.policy.action_to_bins``), so
    the cache keeps the plain 17 numbers.
    """
    b = len(idx)
    tgt = torch.from_numpy(examples.targets[examples.painting[idx]]).to(device, non_blocking=True)
    planes = torch.from_numpy(examples.planes[idx]).to(device, non_blocking=True)
    phase = torch.from_numpy(examples.phase[idx].astype(np.int64)).to(device)
    h, w = planes.shape[-2:]
    obs = torch.zeros((b, pol.OBS_CHANNELS, h, w), dtype=torch.float32, device=device)
    obs[:, 0:3] = tgt.float() / 255.0
    obs[:, 3 : 3 + EXAMPLE_PLANES] = planes.float() / 255.0
    obs[torch.arange(b, device=device), 3 + EXAMPLE_PLANES + phase] = 1.0
    ab = pol.action_to_bins(examples.action[idx], config)
    target = {"cell": torch.from_numpy(ab.cell).to(device), "offset": torch.from_numpy(ab.offset).to(device),
              "bins": torch.from_numpy(ab.bins).to(device)}
    return obs, target


def evaluate_loss(model: Any, examples: ExampleSet, settings: TrainSettings, device: "torch.device",
                  max_examples: int | None = None) -> dict[str, float]:
    """Mean loss parts of ``model`` over ``examples`` (the first ``max_examples`` if given)."""
    n = len(examples) if max_examples is None else min(len(examples), max_examples)
    if n == 0:
        return {}
    sums: dict[str, float] = {}
    model.eval()
    with torch.no_grad():
        for idx in _batches(n, settings.batch, None):
            obs, tgt = _to_device(examples, idx, device, model.config)
            with _autocast(device):
                pred = model(obs, tgt["cell"])
            for k, v in loss_parts(pred, tgt, settings.sparsity_weight, settings.label_smoothing).items():
                sums[k] = sums.get(k, 0.0) + float(v) * len(idx)
    return {k: v / n for k, v in sums.items()}


def _autocast(device: "torch.device") -> Any:
    """Mixed precision on CUDA, nothing elsewhere."""
    import contextlib

    if device.type != "cuda":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _param_groups(model: Any, weight_decay: float) -> list[dict[str, Any]]:
    """AdamW groups: decay on weight matrices and kernels, none on norms and biases."""
    decay, no_decay = [], []
    for p in model.parameters():
        if p.requires_grad:
            (decay if p.ndim > 1 else no_decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def prepare_examples(ledger_roots: Sequence[str | Path], cache_dir: str | Path, *, workers: int = 1,
                     limit: int | None = None) -> tuple[ExampleSet, dict[str, Any]]:
    """Find the finished runs, build their shards and load them. Returns (examples, cache summary)."""
    runs = find_runs(ledger_roots, limit)
    if not runs:
        raise ValueError(f"no finished run folders under {list(map(str, ledger_roots))}")
    summary = build_cache(runs, cache_dir, workers=workers)
    if not summary["shards"]:
        raise ValueError(f"no run folder could be cached: {summary['failed'][:3]}")
    return load_examples(summary["shards"]), summary


def train_imitation(
    ledger_roots: Sequence[str | Path],
    out_dir: str | Path,
    cache_dir: str | Path | None = None,
    settings: TrainSettings | None = None,
    log: Any = print,
    on_epoch: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run Stage A end to end and return a summary dict.

    Builds or reuses the example cache (``cache_dir``, default
    ``OUT_DIR/cache``), splits paintings, trains, and writes ``last.pt``,
    ``best.pt``, ``train.jsonl`` and ``summary.json`` to ``out_dir``. If
    ``out_dir`` already holds a ``last.pt`` and ``settings.resume`` is set,
    training continues from it (the network config must match).
    ``on_epoch``, if given, is called with each epoch's log row after its
    checkpoints are written (the Modal app commits its volume there).
    """
    _require_torch()
    s = settings or TrainSettings()
    out = Path(out_dir)
    if data.sealed_folder_of(out) is not None:
        raise data.SealedSetError(f"refusing to write checkpoints inside a sealed folder: {out}")
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(cache_dir) if cache_dir is not None else out / "cache"
    t_start = time.perf_counter()
    if s.threads:
        torch.set_num_threads(int(s.threads))
    device = pick_device(s.device)

    examples, cache_summary = prepare_examples(ledger_roots, cache, workers=s.workers, limit=s.limit)
    train_idx, val_idx = split_paintings(examples.names, s.val_fraction, s.seed)
    train_set, val_set = examples.subset(train_idx), examples.subset(val_idx)
    log(f"cache: {len(cache_summary['shards'])} paintings ({cache_summary['built']} built, "
        f"{cache_summary['reused']} reused, {len(cache_summary['failed'])} failed) in {cache_summary['seconds']} s")
    log(f"split: {len(train_set.names)} paintings / {len(train_set)} strokes for training, "
        f"{len(val_set.names)} / {len(val_set)} for validation")

    config = CONFIGS[s.config]
    model = pol.make_policy(config, seed=s.seed).to(device)
    ema = _Ema(model, s.ema_decay)
    opt = torch.optim.AdamW(_param_groups(model, s.weight_decay), lr=s.lr, betas=(0.9, 0.99))
    steps_per_epoch = max(1, math.ceil(len(train_set) / s.batch))
    total_steps = steps_per_epoch * s.epochs
    warmup = int(round(s.warmup_fraction * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: lr_factor(k, total_steps, warmup))
    use_scaler = device.type == "cuda" and not torch.cuda.is_bf16_supported()
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    start_epoch, step, best_val = 0, 0, math.inf
    last_path, best_path, log_path = out / "last.pt", out / "best.pt", out / "train.jsonl"
    if s.resume and last_path.is_file():
        blob = torch.load(last_path, map_location=device, weights_only=True)
        pol.check_checkpoint_format(blob, last_path)
        if pol.PolicyConfig.from_json(blob["config"]) != config:
            raise ValueError(f"{last_path} was trained with a different network config; use a new OUT_DIR")
        st = blob["extra"]["state"]
        model.load_state_dict(st["model"])
        ema.shadow.load_state_dict(blob["state_dict"])
        ema.updates = int(st["ema_updates"])
        opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["scheduler"])
        if use_scaler and st.get("scaler"):
            scaler.load_state_dict(st["scaler"])
        start_epoch, step, best_val = int(st["epoch"]) + 1, int(st["step"]), float(st["best_val"])
        log(f"resuming from {last_path} after epoch {start_epoch}")
    elif log_path.exists():
        log_path.unlink()  # a fresh run starts a fresh log

    probe = min(len(train_set), 4096)
    start_train = evaluate_loss(model, train_set, s, device, probe)
    history: list[dict[str, Any]] = []
    for epoch in range(start_epoch, s.epochs):
        t0 = time.perf_counter()
        model.train()
        rng = np.random.default_rng([s.seed, epoch])
        sums: dict[str, float] = {}
        seen = 0
        for idx in _batches(len(train_set), s.batch, rng):
            obs, tgt = _to_device(train_set, idx, device, config)
            with _autocast(device):
                pred = model(obs, tgt["cell"])
            parts = loss_parts(pred, tgt, s.sparsity_weight, s.label_smoothing)
            opt.zero_grad(set_to_none=True)
            scaler.scale(parts["total"]).backward()
            scaler.unscale_(opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), s.clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            ema.update()
            step += 1
            seen += len(idx)
            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + float(v.detach()) * len(idx)
            sums["grad_norm"] = sums.get("grad_norm", 0.0) + float(grad_norm) * len(idx)
        train_time = time.perf_counter() - t0
        train_loss = {k: v / max(seen, 1) for k, v in sums.items()}
        val_loss = evaluate_loss(ema.shadow, val_set, s, device) if len(val_set) else {}
        score_key = val_loss.get("total", train_loss.get("total", math.inf))
        is_best = score_key < best_val
        best_val = min(best_val, score_key)
        row = {
            "epoch": epoch + 1,
            "step": step,
            "train": {k: round(v, 6) for k, v in train_loss.items()},
            "val": {k: round(v, 6) for k, v in val_loss.items()},
            "lr": opt.param_groups[0]["lr"],
            "examples_per_s": round(seen / max(train_time, 1e-9), 1),
            "epoch_s": round(time.perf_counter() - t0, 2),
            "wall_s": round(time.perf_counter() - t_start, 2),
            "best": is_best,
        }
        state = {
            "epoch": epoch, "step": step, "best_val": best_val, "ema_updates": ema.updates,
            "model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
            "scaler": scaler.state_dict() if use_scaler else {},
        }
        extra = {"state": state, "settings": s.to_json(), "epoch": epoch + 1, "val_loss": val_loss,
                 "train_loss": train_loss, "train_paintings": train_set.names, "val_paintings": val_set.names}
        pol.save_checkpoint(last_path.with_suffix(".tmp"), ema.shadow, **extra)
        os.replace(last_path.with_suffix(".tmp"), last_path)
        if is_best:
            pol.save_checkpoint(best_path.with_suffix(".tmp"), ema.shadow, **extra)
            os.replace(best_path.with_suffix(".tmp"), best_path)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        history.append(row)
        if on_epoch is not None:
            on_epoch(row)
        log(f"epoch {epoch + 1}/{s.epochs}: train {train_loss.get('total', math.nan):.4f}"
            f" val {val_loss.get('total', math.nan):.4f} lr {row['lr']:.2e}"
            f" {row['examples_per_s']:.0f} ex/s{' best' if is_best else ''}")

    end_train = evaluate_loss(model, train_set, s, device, probe)
    final = history[-1] if history else None
    summary = {
        "out_dir": str(out),
        "device": str(device),
        "config": s.config,
        "parameters": sum(p.numel() for p in model.parameters()),
        "paintings": {"train": len(train_set.names), "val": len(val_set.names)},
        "examples": {"train": len(train_set), "val": len(val_set)},
        "epochs_run": len(history),
        "epochs": s.epochs,
        "train_loss_start": start_train.get("total"),
        "train_loss_end": end_train.get("total"),
        "final_train_loss": final["train"].get("total") if final else None,
        "final_val_loss": final["val"].get("total") if final else None,
        "best_val_loss": best_val if math.isfinite(best_val) else None,
        "best": str(best_path) if best_path.is_file() else None,
        "last": str(last_path) if last_path.is_file() else None,
        "cache": {k: v for k, v in cache_summary.items() if k != "shards"},
        "wall_s": round(time.perf_counter() - t_start, 2),
        "settings": s.to_json(),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def describe(summary: dict[str, Any]) -> str:
    """A short plain-English account of a run."""
    def f(x: Any) -> str:
        return "n/a" if x is None else f"{x:.4f}"

    return (
        f"Trained the {summary['config']} policy ({summary['parameters']:,} parameters) on {summary['device']} "
        f"for {summary['epochs_run']} epoch(s) this call ({summary['epochs']} in all) on "
        f"{summary['paintings']['train']} paintings ({summary['examples']['train']} strokes); "
        f"{summary['paintings']['val']} paintings ({summary['examples']['val']} strokes) held out for validation.\n"
        f"Training loss went from {f(summary['train_loss_start'])} to {f(summary['train_loss_end'])} "
        f"(measured on up to 4096 training strokes). Last validation loss {f(summary['final_val_loss'])}, "
        f"best {f(summary['best_val_loss'])}.\n"
        f"Best weights: {summary['best']}. Took {summary['wall_s']:.1f} s."
    )


def build_parser() -> argparse.ArgumentParser:
    """Command-line options."""
    ap = argparse.ArgumentParser(prog="python -m train.imitation", description="Stage A: behaviour cloning.")
    ap.add_argument("paths", nargs="+", metavar="LEDGER_ROOT ... OUT_DIR",
                    help="one or more ledger roots (train.ledgers output), then the output folder")
    ap.add_argument("--cache", default=None, help="example cache folder (default OUT_DIR/cache)")
    d = TrainSettings()
    ap.add_argument("--epochs", type=int, default=d.epochs)
    ap.add_argument("--batch", type=int, default=d.batch)
    ap.add_argument("--lr", type=float, default=d.lr)
    ap.add_argument("--weight-decay", type=float, default=d.weight_decay)
    ap.add_argument("--warmup-fraction", type=float, default=d.warmup_fraction)
    ap.add_argument("--clip", type=float, default=d.clip, help="gradient norm clip")
    ap.add_argument("--ema-decay", type=float, default=d.ema_decay)
    ap.add_argument("--sparsity-weight", type=float, default=d.sparsity_weight)
    ap.add_argument("--val-fraction", type=float, default=d.val_fraction)
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--workers", type=int, default=d.workers, help="processes for building the cache")
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default=d.device)
    ap.add_argument("--limit", type=int, default=None, help="only the first N paintings")
    ap.add_argument("--config", choices=sorted(CONFIGS), default=d.config, help="'small' is for smoke tests")
    ap.add_argument("--threads", type=int, default=None, help="CPU threads for PyTorch")
    ap.add_argument("--no-resume", action="store_true", help="start over even if OUT_DIR/last.pt exists")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line. Returns a process exit code."""
    ap = build_parser()
    a = ap.parse_args(argv)
    if len(a.paths) < 2:
        ap.error("need at least one LEDGER_ROOT and an OUT_DIR")
    *roots, out_dir = a.paths
    settings = TrainSettings(
        epochs=a.epochs, batch=a.batch, lr=a.lr, weight_decay=a.weight_decay, warmup_fraction=a.warmup_fraction,
        clip=a.clip, ema_decay=a.ema_decay, sparsity_weight=a.sparsity_weight, val_fraction=a.val_fraction,
        seed=a.seed, workers=a.workers, device=a.device, limit=a.limit, config=a.config, threads=a.threads,
        resume=not a.no_resume,
    )
    summary = train_imitation(roots, out_dir, a.cache, settings)
    print(describe(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
