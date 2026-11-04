"""Targets for training, and the sealed evaluation set that training never sees.

Two folders come out of this module, both made of 256 x 192 PNGs letterboxed
through ``backrun.targets``:

``train/`` (the training targets)
    The built-in synthetic scenes (every kind in
    ``backrun.targets.SYNTHETIC_KINDS``, many seeds) plus any images in a
    folder the owner supplies. ``index.jsonl`` lists each file with its
    source and hashes.

``sealed/`` (the sealed evaluation set)
    Fifty targets drawn once with a fixed seed, written as PNGs next to a
    ``sealed.sha256`` manifest in the format ``sha256sum -c`` reads, and a
    ``sealed.json`` with each target's source and pixel hash. Synthetic
    sealed targets use seeds from a reserved range that the training set
    never uses, and an owner photo drawn into the sealed set is left out of
    the training set.

The guard
---------

A folder that holds a ``sealed.sha256`` file is sealed, wherever it lives.
Training modules call ``declare_training_module(__name__)`` when they are
imported. From then on:

* ``assert_not_sealed(path)`` raises ``SealedSetError`` for any path inside a
  sealed folder, and for any file whose bytes match a hash in a sealed
  manifest the process has seen (so a copied sealed PNG is caught too).
  Every loader in the training code goes through it.
* ``load_sealed_set`` refuses to run when it is called from a declared
  training module. Only the evaluation code, which is not a training
  module, may load the sealed set.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from backrun.targets import SYNTHETIC_KINDS, letterbox, load_target, synthetic_target

#: Target size used for every stored target.
TARGET_WIDTH = 256
TARGET_HEIGHT = 192

#: Size of the sealed evaluation set.
SEALED_SIZE = 50
#: The fixed seed the sealed set is drawn with.
SEALED_SEED = 20261104
#: Synthetic sealed targets use seeds at or above this; training seeds stay below.
SEALED_SEED_BASE = 1_000_000
#: Name of the manifest that marks a folder as sealed.
MANIFEST_NAME = "sealed.sha256"
#: Name of the sealed set's metadata file.
SEALED_META_NAME = "sealed.json"
#: Name of the training set's index file.
INDEX_NAME = "index.jsonl"

#: Image file extensions picked up from an owner's photo folder.
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")


class SealedSetError(RuntimeError):
    """Raised when training code reaches for the sealed evaluation set."""


# ---------------------------------------------------------------------------
# Hashing.
# ---------------------------------------------------------------------------


def file_sha256(path: str | Path) -> str:
    """SHA-256 of a file's bytes, as lowercase hex."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pixel_sha256(image: NDArray[np.uint8]) -> str:
    """SHA-256 of an image's shape and uint8 pixels, independent of file encoding."""
    arr = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
    h = hashlib.sha256(repr(arr.shape).encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def write_manifest(folder: str | Path, names: Sequence[str]) -> Path:
    """Write ``sealed.sha256`` for the named files in ``folder`` and return its path.

    One line per file, ``<hex>  <name>``, sorted by name, as ``sha256sum``
    writes it.
    """
    folder = Path(folder)
    lines = [f"{file_sha256(folder / n)}  {n}" for n in sorted(names)]
    path = folder / MANIFEST_NAME
    path.write_text("\n".join(lines) + "\n")
    return path


def read_manifest(path: str | Path) -> dict[str, str]:
    """Read a ``sha256sum``-style manifest into {file name: hex digest}."""
    out: dict[str, str] = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        out[name.lstrip("*").strip()] = digest.lower()
    return out


def verify_manifest(folder: str | Path) -> list[str]:
    """Check every file listed in the folder's manifest; return the problems found.

    An empty list means every listed file exists and matches its hash, and
    no extra PNG sits in the folder unlisted.
    """
    folder = Path(folder)
    manifest = read_manifest(folder / MANIFEST_NAME)
    problems: list[str] = []
    for name, digest in sorted(manifest.items()):
        p = folder / name
        if not p.is_file():
            problems.append(f"missing: {name}")
        elif file_sha256(p) != digest:
            problems.append(f"hash mismatch: {name}")
    for p in sorted(folder.glob("*.png")):
        if p.name not in manifest:
            problems.append(f"not in manifest: {p.name}")
    return problems


# ---------------------------------------------------------------------------
# The guard.
# ---------------------------------------------------------------------------

#: Modules that declared themselves training code.
_TRAINING_MODULES: set[str] = set()
#: File hashes of every sealed target in every manifest this process has seen.
_SEALED_HASHES: set[str] = set()
#: Pixel hashes of sealed targets, from ``sealed.json`` files seen.
_SEALED_PIXEL_HASHES: set[str] = set()
#: Sealed folders whose hashes have been registered.
_REGISTERED: set[Path] = set()


def declare_training_module(module_name: str) -> None:
    """Mark ``module_name`` as training code; call it at import time.

    After this, ``load_sealed_set`` refuses calls coming from that module.
    The default sealed folder's hashes, if it exists, are registered so a
    copied sealed file is recognised anywhere.
    """
    _TRAINING_MODULES.add(module_name)
    default = default_data_root() / "sealed"
    if (default / MANIFEST_NAME).is_file():
        register_sealed(default)


def training_modules() -> frozenset[str]:
    """The module names that have declared themselves training code."""
    return frozenset(_TRAINING_MODULES)


def register_sealed(folder: str | Path) -> None:
    """Remember the hashes of a sealed folder so copies of its files are refused."""
    folder = Path(folder).resolve()
    if folder in _REGISTERED:
        return
    manifest = folder / MANIFEST_NAME
    if not manifest.is_file():
        raise FileNotFoundError(f"{folder} has no {MANIFEST_NAME}")
    _SEALED_HASHES.update(read_manifest(manifest).values())
    meta = folder / SEALED_META_NAME
    if meta.is_file():
        for item in json.loads(meta.read_text()).get("targets", []):
            if item.get("pixel_sha256"):
                _SEALED_PIXEL_HASHES.add(item["pixel_sha256"])
    _REGISTERED.add(folder)


def sealed_folder_of(path: str | Path) -> Path | None:
    """The sealed folder containing ``path`` (or ``path`` itself), or ``None``."""
    p = Path(path).resolve()
    for d in (p, *p.parents):
        if (d / MANIFEST_NAME).is_file():
            return d
    return None


def is_sealed_path(path: str | Path) -> bool:
    """True if ``path`` lies inside a sealed folder or its bytes match a sealed hash."""
    folder = sealed_folder_of(path)
    if folder is not None:
        register_sealed(folder)
        return True
    p = Path(path)
    return bool(_SEALED_HASHES) and p.is_file() and file_sha256(p) in _SEALED_HASHES


def assert_not_sealed(path: str | Path) -> Path:
    """Return ``path`` as a Path, or raise ``SealedSetError`` if it is sealed.

    Every file the training code reads goes through this check.
    """
    if is_sealed_path(path):
        raise SealedSetError(f"refusing to read {path}: it belongs to the sealed evaluation set")
    return Path(path)


def assert_not_sealed_image(image: NDArray[np.uint8]) -> None:
    """Raise ``SealedSetError`` if the pixels match a registered sealed target."""
    if _SEALED_PIXEL_HASHES and pixel_sha256(image) in _SEALED_PIXEL_HASHES:
        raise SealedSetError("refusing to use an image identical to a sealed evaluation target")


def load_training_image(path: str | Path) -> NDArray[np.uint8]:
    """Load a stored target PNG for training, after the sealed-set checks."""
    p = assert_not_sealed(path)
    with Image.open(p) as im:
        arr = np.asarray(im.convert("RGB"), dtype=np.uint8).copy()
    assert_not_sealed_image(arr)
    return arr


def _calling_module(depth: int = 2) -> str | None:
    """Name of the module ``depth`` frames up the stack."""
    frame = inspect.currentframe()
    try:
        for _ in range(depth):
            if frame is None:
                return None
            frame = frame.f_back
        return None if frame is None else frame.f_globals.get("__name__")
    finally:
        del frame


# ---------------------------------------------------------------------------
# Building the sets.
# ---------------------------------------------------------------------------


def default_data_root() -> Path:
    """``runs/data`` under the repository, where targets live by default."""
    return Path(__file__).resolve().parent.parent / "runs" / "data"


@dataclass(frozen=True)
class TargetSpec:
    """Where a target comes from: a synthetic scene or an owner image."""

    source: str  # "synthetic" or "photo"
    kind: str = ""
    seed: int = 0
    path: str = ""

    @property
    def name(self) -> str:
        """A file-safe name for the target."""
        if self.source == "synthetic":
            return f"syn_{self.kind}_{self.seed:07d}"
        stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in Path(self.path).stem)
        return f"photo_{stem}_{hashlib.sha256(self.path.encode()).hexdigest()[:8]}"

    def render(self, width: int = TARGET_WIDTH, height: int = TARGET_HEIGHT) -> NDArray[np.uint8]:
        """Draw or load the target, letterboxed to ``width`` x ``height``."""
        if self.source == "synthetic":
            # Drawn at the stored size, then letterboxed, so every target goes
            # through the same backrun.targets path.
            return letterbox(synthetic_target(self.kind, width, height, self.seed), width, height)
        return load_target(self.path, width, height)

    def to_json(self) -> dict[str, Any]:
        """The spec as a plain dict."""
        out: dict[str, Any] = {"source": self.source, "name": self.name}
        if self.source == "synthetic":
            out.update(kind=self.kind, seed=self.seed)
        else:
            out["path"] = self.path
        return out


def photo_specs(folder: str | Path | None) -> list[TargetSpec]:
    """Specs for every image file in ``folder`` (recursively), sorted by path."""
    if folder is None:
        return []
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"photo folder not found: {folder}")
    files = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    out = []
    for p in files:
        if sealed_folder_of(p) is not None:
            continue  # never pull images out of a sealed folder
        out.append(TargetSpec("photo", path=str(p.resolve())))
    return out


def synthetic_specs(seeds_per_kind: int, seed_start: int = 0) -> list[TargetSpec]:
    """Training specs for every synthetic kind with seeds ``seed_start`` upward."""
    if seed_start + seeds_per_kind > SEALED_SEED_BASE:
        raise ValueError(f"training seeds must stay below {SEALED_SEED_BASE}")
    return [
        TargetSpec("synthetic", kind=k, seed=s)
        for s in range(seed_start, seed_start + seeds_per_kind)
        for k in SYNTHETIC_KINDS
    ]


def draw_sealed_specs(
    n: int = SEALED_SIZE, seed: int = SEALED_SEED, photos: Sequence[TargetSpec] = ()
) -> list[TargetSpec]:
    """Choose the sealed set's ``n`` targets with a fixed seed.

    The pool is ``n`` synthetic scenes per kind with seeds from the reserved
    range, plus the owner's photos. The same arguments always give the same
    list.
    """
    pool = [
        TargetSpec("synthetic", kind=k, seed=SEALED_SEED_BASE + i)
        for k in SYNTHETIC_KINDS
        for i in range(n)
    ] + list(photos)
    if n > len(pool):
        raise ValueError(f"cannot draw {n} sealed targets from a pool of {len(pool)}")
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(pool), size=n, replace=False)
    return [pool[i] for i in sorted(int(i) for i in picks)]


def _save_png(image: NDArray[np.uint8], path: Path) -> None:
    """Write a uint8 image as PNG with fixed settings so bytes are reproducible."""
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(path, format="PNG", optimize=False, compress_level=6)


def write_sealed_set(
    out_dir: str | Path,
    photo_dir: str | Path | None = None,
    n: int = SEALED_SIZE,
    seed: int = SEALED_SEED,
    force: bool = False,
) -> Path:
    """Draw the sealed set, write its PNGs, manifest and metadata; return the folder.

    The set is drawn once: if ``out_dir`` already holds a manifest the call
    refuses unless ``force`` is set.
    """
    out = Path(out_dir)
    if (out / MANIFEST_NAME).exists() and not force:
        raise FileExistsError(f"{out} already holds a sealed set; it is drawn once (pass force=True to redraw)")
    if out.exists() and force:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    specs = draw_sealed_specs(n, seed, photo_specs(photo_dir))
    meta = []
    names = []
    for i, spec in enumerate(specs):
        img = spec.render()
        name = f"sealed_{i:03d}.png"
        _save_png(img, out / name)
        names.append(name)
        meta.append({"file": name, **spec.to_json(), "pixel_sha256": pixel_sha256(img)})
    (out / SEALED_META_NAME).write_text(
        json.dumps({"seed": seed, "size": n, "width": TARGET_WIDTH, "height": TARGET_HEIGHT, "targets": meta}, indent=2)
        + "\n"
    )
    write_manifest(out, names)
    register_sealed(out)
    return out


def write_training_set(
    out_dir: str | Path,
    seeds_per_kind: int = 400,
    photo_dir: str | Path | None = None,
    sealed_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Write the training targets and ``index.jsonl``; return the index rows.

    Synthetic scenes of every kind with seeds ``0 .. seeds_per_kind - 1``,
    then the owner's photos. Any target whose pixels match the sealed set
    (``sealed_dir``, by default ``runs/data/sealed``) is left out, so a photo
    drawn into the sealed set never trains. ``out_dir`` must not be inside a
    sealed folder.
    """
    out = Path(out_dir)
    if sealed_folder_of(out) is not None:
        raise SealedSetError(f"refusing to write training targets inside a sealed folder: {out}")
    sealed = Path(sealed_dir) if sealed_dir is not None else default_data_root() / "sealed"
    excluded: set[str] = set()
    if (sealed / MANIFEST_NAME).is_file():
        register_sealed(sealed)
        meta = sealed / SEALED_META_NAME
        if meta.is_file():
            excluded = {t["pixel_sha256"] for t in json.loads(meta.read_text())["targets"]}
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for spec in [*synthetic_specs(seeds_per_kind), *photo_specs(photo_dir)]:
        img = spec.render()
        ph = pixel_sha256(img)
        if ph in excluded:
            continue
        name = f"{spec.name}.png"
        _save_png(img, out / name)
        rows.append({"file": name, **spec.to_json(), "pixel_sha256": ph, "sha256": file_sha256(out / name)})
    (out / INDEX_NAME).write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows


def read_index(folder: str | Path) -> list[dict[str, Any]]:
    """Read a training folder's ``index.jsonl``, refusing a sealed folder."""
    folder = assert_not_sealed(folder)
    path = assert_not_sealed(Path(folder) / INDEX_NAME)
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def iter_training_targets(folder: str | Path) -> Iterable[tuple[str, NDArray[np.uint8]]]:
    """Yield (name, image) for every target listed in a training folder."""
    folder = Path(folder)
    for row in read_index(folder):
        yield Path(row["file"]).stem, load_training_image(folder / row["file"])


def load_sealed_set(folder: str | Path | None = None, verify: bool = True) -> list[tuple[str, NDArray[np.uint8]]]:
    """Load the sealed evaluation targets as (name, image) pairs, in file order.

    Only evaluation code may call this. A call from a module that declared
    itself training code raises ``SealedSetError``. With ``verify`` (the
    default) every file is checked against the manifest first and any
    mismatch raises.
    """
    caller = _calling_module(2)
    if caller in _TRAINING_MODULES:
        raise SealedSetError(f"{caller} is training code and may not load the sealed set")
    folder = Path(folder) if folder is not None else default_data_root() / "sealed"
    if verify:
        problems = verify_manifest(folder)
        if problems:
            raise SealedSetError(f"sealed set at {folder} does not match its manifest: {'; '.join(problems)}")
    register_sealed(folder)
    out = []
    for name in sorted(read_manifest(folder / MANIFEST_NAME)):
        with Image.open(folder / name) as im:
            out.append((Path(name).stem, np.asarray(im.convert("RGB"), dtype=np.uint8).copy()))
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """Command line: ``python -m train.data [--root runs/data] [--seeds-per-kind N] [--photos DIR]``."""
    import argparse

    ap = argparse.ArgumentParser(prog="python -m train.data", description="Write the training targets and the sealed set.")
    ap.add_argument("--root", default=str(default_data_root()), help="data folder (gets train/ and sealed/)")
    ap.add_argument("--seeds-per-kind", type=int, default=400, help="synthetic seeds per scene kind")
    ap.add_argument("--photos", default=None, help="folder of the owner's images to include")
    ap.add_argument("--redraw-sealed", action="store_true", help="redraw the sealed set even if it exists")
    args = ap.parse_args(argv)
    root = Path(args.root)
    sealed = root / "sealed"
    if not (sealed / MANIFEST_NAME).exists() or args.redraw_sealed:
        write_sealed_set(sealed, args.photos, force=args.redraw_sealed)
        print(f"sealed set: {sealed}")
    else:
        problems = verify_manifest(sealed)
        print(f"sealed set exists at {sealed}: {'ok' if not problems else problems}")
    rows = write_training_set(root / "train", args.seeds_per_kind, args.photos, sealed)
    print(f"training targets: {len(rows)} in {root / 'train'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
