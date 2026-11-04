"""Running the training pipeline on Modal, one command per stage.

From a machine that can reach Modal (with ``modal setup`` done)::

    modal run train/modal_app.py --stage targets
    modal run train/modal_app.py --stage ledgers
    modal run train/modal_app.py --stage imitation
    modal run train/modal_app.py --stage evaluate --painter search
    modal run train/modal_app.py --stage evaluate --painter stage_a/best.pt
    modal run train/modal_app.py --stage evaluate --painter blank
    modal run train/modal_app.py --stage evaluate --painter mean
    modal run train/modal_app.py --stage all

The image pins Python 3.11, installs NumPy, SciPy, Pillow and PyTorch, then
copies this repository's ``backrun/`` and ``train/`` folders and
``pyproject.toml`` into it and installs it with ``pip install -e .[train]``,
so the simulator in the cloud is the code on the laptop that ran the
command.

Two volumes:

``backrun-data`` at ``/data``
    ``sealed/`` (the sealed set, written once), ``train/`` (training
    targets), ``ledgers/`` (search-painter runs and ``index.jsonl``),
    ``cache/`` (the Stage A example cache) and ``eval/`` (reports).
``backrun-checkpoints`` at ``/checkpoints``
    One folder per training run (``last.pt``, ``best.pt``, ``train.jsonl``).

The image sets ``BACKRUN_DATA=/data``, so ``train.data`` registers the
sealed set's hashes there whenever a training module is imported.

This module imports without Modal installed (the decorators then leave the
functions as plain Python), so tests can check it anywhere.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

try:
    import modal
except ModuleNotFoundError:  # pragma: no cover - exercised in a subprocess test
    modal = None  # type: ignore[assignment]

REPO = Path(__file__).resolve().parent.parent
REMOTE_REPO = "/root/backrun"
DATA = "/data"
CHECKPOINTS = "/checkpoints"
#: Default GPU for Stage A; ``--gpu`` on the command line picks another.
DEFAULT_GPU = "A10G"
#: Ledger settings, as in docs/TRAINING.md.
LEDGER_DEFAULTS: dict[str, Any] = {"width": 256, "height": 192, "strokes": 40, "candidates": 16, "seed": 7}
#: CPUs per ledger container; paintings in a chunk run one per CPU.
LEDGER_CPUS = 8


class _Absent:
    """Stands in for Modal when it is not installed.

    Any attribute or call returns the same object, except that calling it
    with one callable and nothing else returns that callable, so
    ``@app.function(...)`` leaves a function as it is.
    """

    def __getattr__(self, name: str) -> "_Absent":
        return self

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return self


_modal: Any = modal if modal is not None else _Absent()

app = _modal.App("backrun")

_ignore = ["**/__pycache__", "**/*.pyc"]
image = (
    _modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy>=1.26", "scipy>=1.11", "pillow>=10", "torch>=2.3")
    .env({"BACKRUN_DATA": DATA, "PYTHONUNBUFFERED": "1"})
    .add_local_file(REPO / "pyproject.toml", f"{REMOTE_REPO}/pyproject.toml", copy=True)
    .add_local_file(REPO / "README.md", f"{REMOTE_REPO}/README.md", copy=True)
    .add_local_dir(REPO / "backrun", f"{REMOTE_REPO}/backrun", copy=True, ignore=_ignore)
    .add_local_dir(REPO / "train", f"{REMOTE_REPO}/train", copy=True, ignore=_ignore)
    .run_commands(f"pip install -e '{REMOTE_REPO}[train]'")
    .workdir(REMOTE_REPO)
)

data_volume = _modal.Volume.from_name("backrun-data", create_if_missing=True)
checkpoint_volume = _modal.Volume.from_name("backrun-checkpoints", create_if_missing=True)
VOLUMES = {DATA: data_volume, CHECKPOINTS: checkpoint_volume}


def _commit(volume: Any) -> None:
    """Commit a volume when running on Modal; do nothing elsewhere."""
    if modal is not None and hasattr(volume, "commit"):
        try:
            volume.commit()
        except Exception as exc:  # outside a Modal container there is nothing to commit
            print(f"volume commit skipped: {exc}")


def _reload(volume: Any) -> None:
    """Pick up other containers' commits when running on Modal."""
    if modal is not None and hasattr(volume, "reload"):
        try:
            volume.reload()
        except Exception as exc:
            print(f"volume reload skipped: {exc}")


# ---------------------------------------------------------------------------
# Stage 0: targets.
# ---------------------------------------------------------------------------


@app.function(image=image, volumes=VOLUMES, cpu=4, memory=8192, timeout=3 * 3600)
def make_targets(seeds_per_kind: int = 400) -> dict[str, Any]:
    """Write the sealed set (only if absent) and the training targets to the data volume.

    The sealed set is drawn once: if ``/data/sealed`` already holds a
    manifest it is verified and left alone, never overwritten.
    """
    from train import data

    sealed = Path(DATA) / "sealed"
    if (sealed / data.MANIFEST_NAME).is_file():
        problems = data.verify_manifest(sealed)
        if problems:
            raise data.SealedSetError(f"sealed set on the volume does not match its manifest: {problems}")
        sealed_note = "kept (already present, manifest verified)"
    else:
        data.write_sealed_set(sealed)
        sealed_note = "written"
    rows = data.write_training_set(Path(DATA) / "train", seeds_per_kind, sealed_dir=sealed)
    _commit(data_volume)
    out = {"sealed": sealed_note, "training_targets": len(rows)}
    print(json.dumps(out))
    return out


# ---------------------------------------------------------------------------
# Stage A data: search-painter ledgers.
# ---------------------------------------------------------------------------


@app.function(image=image, volumes=VOLUMES, cpu=LEDGER_CPUS, memory=8192, timeout=6 * 3600, retries=1)
def paint_chunk(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Paint one chunk of ledger jobs, one painting per CPU; return their index rows.

    Each finished run folder is committed to the data volume as it lands.
    """
    import multiprocessing as mp

    from train import ledgers

    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    items = [ledgers.LedgerJob(**j) for j in jobs]
    rows = []
    with mp.get_context("spawn").Pool(processes=max(1, min(LEDGER_CPUS, len(items)))) as pool:
        for row in pool.imap_unordered(ledgers._paint_safe, items, chunksize=1):
            rows.append(row)
            _commit(data_volume)
    return rows


@app.function(image=image, volumes=VOLUMES, cpu=1, memory=2048, timeout=12 * 3600)
def generate_ledgers(
    limit: int = 0, chunk: int = 8, width: int = 256, height: int = 192, strokes: int = 40,
    candidates: int = 16, seed: int = 7,
) -> dict[str, Any]:
    """Plan the outstanding paintings and fan them out with ``paint_chunk.map``.

    Resumable: paintings already marked done in ``/data/ledgers/index.jsonl``
    (with a finished ``run.json``) are skipped. Only this function writes
    the index, one line per painting as chunks come back, so containers
    never write the same file. ``limit`` 0 means every training target.
    """
    from train import ledgers

    _reload(data_volume)
    out_root = Path(DATA) / "ledgers"
    out_root.mkdir(parents=True, exist_ok=True)
    jobs, skipped = ledgers.plan_jobs(
        Path(DATA) / "train", out_root, width=width, height=height, strokes=strokes, candidates=candidates,
        seed=seed, limit=limit or None,
    )
    chunks = [[asdict(j) for j in jobs[i : i + chunk]] for i in range(0, len(jobs), max(1, chunk))]
    print(f"{len(jobs)} paintings to run in {len(chunks)} chunks, {skipped} already done")
    index = out_root / ledgers.INDEX_NAME
    done = failed = 0
    for rows in paint_chunk.map(chunks, order_outputs=False):
        _reload(data_volume)
        with index.open("a", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
                done += row["status"] == "done"
                failed += row["status"] != "done"
        _commit(data_volume)
        print(f"{done} done, {failed} failed of {len(jobs)}")
    return {"planned": len(jobs), "done": done, "failed": failed, "skipped_already_done": skipped}


# ---------------------------------------------------------------------------
# Stage A training.
# ---------------------------------------------------------------------------


@app.function(image=image, volumes=VOLUMES, gpu=DEFAULT_GPU, cpu=8, memory=32768, timeout=8 * 3600)
def train_imitation(
    run_name: str = "stage_a", epochs: int = 20, batch: int = 256, lr: float = 1e-3, val_fraction: float = 0.1,
    seed: int = 0, config: str = "default", limit: int = 0, sparsity_weight: float = 0.0,
    label_smoothing: float = 0.05,
) -> dict[str, Any]:
    """Stage A on one GPU: cache on the data volume, checkpoints in ``/checkpoints/RUN_NAME``.

    Resumes from ``last.pt`` if the run folder already has one. The
    checkpoint volume is committed after every epoch.
    """
    from train import imitation

    _reload(data_volume)
    settings = imitation.TrainSettings(
        epochs=epochs, batch=batch, lr=lr, val_fraction=val_fraction, seed=seed, config=config,
        limit=limit or None, sparsity_weight=sparsity_weight,
        label_smoothing=label_smoothing, workers=8, device="auto",
    )
    summary = imitation.train_imitation(
        [Path(DATA) / "ledgers"], Path(CHECKPOINTS) / run_name, Path(DATA) / "cache", settings,
        on_epoch=lambda _row: _commit(checkpoint_volume),
    )
    _commit(data_volume)
    _commit(checkpoint_volume)
    print(imitation.describe(summary))
    return summary


@app.function(image=image, volumes=VOLUMES, gpu=DEFAULT_GPU, timeout=8 * 3600)
def finetune_rl(**_: Any) -> None:
    """Stage B, reinforcement fine-tuning. Not written yet."""
    raise NotImplementedError("Stage B (train/rl.py) is not written yet; run Stage A with --stage imitation")


# ---------------------------------------------------------------------------
# Evaluation on the sealed set.
# ---------------------------------------------------------------------------


@app.function(image=image, volumes=VOLUMES, cpu=8, memory=8192, timeout=6 * 3600)
def evaluate(
    painter: str = "search", out_name: str = "", limit: int = 0, seeds: str = "7,8,9", width: int = 256,
    height: int = 192, strokes: int = 40, candidates: int = 16, workers: int = 8,
) -> str:
    """Paint the sealed set and write the report to ``/data/eval``; return the Markdown.

    ``painter`` is ``search``; ``blank`` (no strokes, one dry); ``mean``
    (the dataset-mean stroke of ``/data/ledgers`` in every slot); a
    checkpoint path relative to ``/checkpoints`` (``stage_a/best.pt``); or
    ``policy:`` followed by an absolute path; add ``@sample`` or ``@sample:T``
    to draw strokes from the policy instead of taking the most likely one.
    When ``blank`` and ``search``
    reports with the same canvas size and limit are already on the volume,
    a table placing this painter between them is appended.
    """
    from train import evaluate as ev

    _reload(data_volume)
    _reload(checkpoint_volume)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    spec, name = painter_spec(painter)
    suffix = f"_{width}x{height}" + (f"_limit{limit}" if limit else "")
    name = (out_name or name) + suffix
    report = ev.evaluate(
        ev.make_painter(spec, candidates), Path(DATA) / "eval" / name, sealed_dir=Path(DATA) / "sealed",
        seeds=[int(s) for s in seeds.split(",") if s.strip()], n_strokes=strokes, width=width, height=height,
        limit=limit or None, workers=workers,
    )
    md = ev.markdown_report(report) + comparison_section(Path(DATA) / "eval", name, suffix)
    (Path(DATA) / "eval" / name / "report.md").write_text(md)
    _commit(data_volume)
    print(md)
    return md


def painter_spec(painter: str) -> tuple[str, str]:
    """The ``train.evaluate`` painter spec and the report folder name (before its size suffix) for ``--painter``."""
    if painter in ("search", "blank"):
        return painter, painter
    if painter == "mean":
        return f"mean:{Path(DATA) / 'ledgers'}", "mean"
    if painter.startswith("policy:"):
        painter = painter.split(":", 1)[1]
    else:
        painter = str(Path(CHECKPOINTS) / painter)
    # ``NAME@sample`` or ``NAME@sample:T`` asks for sampled decoding (see train.evaluate.parse_policy_spec).
    path, sep, tail = painter.partition("@")
    rel = path[len(CHECKPOINTS):] if path.startswith(CHECKPOINTS + "/") else path
    folder = rel.strip("/").replace("/", "_").removesuffix(".pt")
    if sep:
        folder += "_" + tail.replace(":", "")
    return f"policy:{painter}", folder


def comparison_section(eval_root: Path, name: str, suffix: str) -> str:
    """A Markdown table of the blank, search and ``name`` reports under ``eval_root``, or "" if one is missing."""
    from train import evaluate as ev

    dirs = [eval_root / f"blank{suffix}", eval_root / f"search{suffix}", eval_root / name]
    if not all((d / "report.json").is_file() for d in dirs):
        return ""
    unique = list(dict.fromkeys(d.resolve() for d in dirs))
    return ("\nWhere this painter stands between a blank sheet and the search painter (same canvas and "
            "targets):\n\n" + ev.compare_markdown([ev.load_report(d) for d in unique]))


# ---------------------------------------------------------------------------
# One command per stage.
# ---------------------------------------------------------------------------

STAGES = ("targets", "ledgers", "imitation", "evaluate", "all")


@app.local_entrypoint()
def main(
    stage: str = "all",
    seeds_per_kind: int = 400,
    limit: int = 0,
    chunk: int = 8,
    strokes: int = 40,
    candidates: int = 16,
    width: int = 256,
    height: int = 192,
    run_name: str = "stage_a",
    epochs: int = 20,
    batch: int = 256,
    lr: float = 1e-3,
    config: str = "default",
    gpu: str = DEFAULT_GPU,
    painter: str = "",
    eval_limit: int = 0,
    seeds: str = "7,8,9",
) -> None:
    """Run one stage, or all of them in order, on Modal.

    ``--limit`` caps the number of training targets painted and trained on
    (0 is all). ``--painter`` picks what ``evaluate`` paints: ``search``,
    ``blank``, ``mean`` or a checkpoint relative to the checkpoint volume; with ``--stage all`` the
    default is the new run's ``best.pt``.
    """
    if stage not in STAGES:
        raise SystemExit(f"--stage must be one of {', '.join(STAGES)}")
    if stage in ("targets", "all"):
        print(make_targets.remote(seeds_per_kind=seeds_per_kind))
    if stage in ("ledgers", "all"):
        print(generate_ledgers.remote(limit=limit, chunk=chunk, width=width, height=height, strokes=strokes,
                                      candidates=candidates))
    if stage in ("imitation", "all"):
        fn = train_imitation if gpu == DEFAULT_GPU else train_imitation.with_options(gpu=gpu)
        summary = fn.remote(run_name=run_name, epochs=epochs, batch=batch, lr=lr, config=config, limit=limit)
        print(json.dumps({k: summary[k] for k in ("best", "final_train_loss", "final_val_loss", "wall_s")}))
    if stage in ("evaluate", "all"):
        target = painter or ("search" if stage == "evaluate" else f"{run_name}/best.pt")
        print(evaluate.remote(painter=target, limit=eval_limit, seeds=seeds, strokes=strokes,
                              candidates=candidates))
