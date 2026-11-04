"""Tests for train.data: the training targets, the sealed set and its guard."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from train import data
from train.data import SealedSetError


@pytest.fixture()
def photos(tmp_path: Path) -> Path:
    """A folder with two small owner images of odd sizes."""
    d = tmp_path / "photos"
    d.mkdir()
    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 255, (60, 100, 3), dtype=np.uint8)).save(d / "a.png")
    Image.fromarray(rng.integers(0, 255, (90, 40, 3), dtype=np.uint8)).save(d / "b.jpg")
    return d


def test_sealed_set_is_deterministic_and_hashed(tmp_path: Path, photos: Path) -> None:
    a = data.write_sealed_set(tmp_path / "a" / "sealed", photos)
    b = data.write_sealed_set(tmp_path / "b" / "sealed", photos)
    pngs = sorted(p.name for p in a.glob("*.png"))
    assert len(pngs) == data.SEALED_SIZE
    manifest = data.read_manifest(a / data.MANIFEST_NAME)
    assert sorted(manifest) == pngs
    for name, digest in manifest.items():
        assert data.file_sha256(a / name) == digest
        img = np.asarray(Image.open(a / name))
        assert img.shape == (data.TARGET_HEIGHT, data.TARGET_WIDTH, 3)
    # Same seed, same draw, same bytes.
    assert (a / data.MANIFEST_NAME).read_bytes() == (b / data.MANIFEST_NAME).read_bytes()
    assert data.verify_manifest(a) == []
    # The manifest is in sha256sum format, if the tool is around to check it.
    if shutil.which("sha256sum"):
        subprocess.run(["sha256sum", "-c", "--quiet", data.MANIFEST_NAME], cwd=a, check=True)
    # Drawn once: a second write to the same folder is refused.
    with pytest.raises(FileExistsError):
        data.write_sealed_set(a, photos)


def test_manifest_catches_tampering(tmp_path: Path) -> None:
    sealed = data.write_sealed_set(tmp_path / "sealed", n=5)
    victim = sorted(sealed.glob("*.png"))[0]
    img = np.asarray(Image.open(victim)).copy()
    img[0, 0] ^= 1
    Image.fromarray(img).save(victim)
    (sealed / "extra.png").write_bytes(victim.read_bytes())
    problems = data.verify_manifest(sealed)
    assert any("hash mismatch" in p for p in problems)
    assert any("not in manifest: extra.png" in p for p in problems)
    with pytest.raises(SealedSetError):
        data.load_sealed_set(sealed)


def test_training_set_is_disjoint_from_sealed(tmp_path: Path, photos: Path) -> None:
    # A sealed set drawn from a pool that is only photos must take both photos.
    sealed = tmp_path / "sealed"
    specs = data.draw_sealed_specs(2, photos=data.photo_specs(photos))
    assert {s.source for s in specs} <= {"synthetic", "photo"}
    sealed_dir = data.write_sealed_set(sealed, photos, n=8)
    rows = data.write_training_set(tmp_path / "train", seeds_per_kind=3, photo_dir=photos, sealed_dir=sealed_dir)
    sealed_meta = json.loads((sealed_dir / data.SEALED_META_NAME).read_text())["targets"]
    sealed_px = {t["pixel_sha256"] for t in sealed_meta}
    assert all(r["pixel_sha256"] not in sealed_px for r in rows)
    assert all(r["seed"] < data.SEALED_SEED_BASE for r in rows if r["source"] == "synthetic")
    kinds = {r["kind"] for r in rows if r["source"] == "synthetic"}
    assert kinds == set(data.SYNTHETIC_KINDS)
    sealed_photos = sum(t["source"] == "photo" for t in sealed_meta)
    assert sum(r["source"] == "photo" for r in rows) == 2 - sealed_photos
    loaded = list(data.iter_training_targets(tmp_path / "train"))
    assert len(loaded) == len(rows)
    assert all(img.shape == (192, 256, 3) for _, img in loaded)


def test_guard_refuses_sealed_paths_and_copies(tmp_path: Path) -> None:
    sealed = data.write_sealed_set(tmp_path / "sealed", n=4)
    first = sorted(sealed.glob("*.png"))[0]
    with pytest.raises(SealedSetError):
        data.assert_not_sealed(first)
    with pytest.raises(SealedSetError):
        data.load_training_image(first)
    with pytest.raises(SealedSetError):
        data.read_index(sealed)
    # A copy outside the sealed folder is still recognised by its hash.
    copy = tmp_path / "elsewhere" / "innocent.png"
    copy.parent.mkdir()
    copy.write_bytes(first.read_bytes())
    with pytest.raises(SealedSetError):
        data.load_training_image(copy)
    # Writing training targets into the sealed folder is refused.
    with pytest.raises(SealedSetError):
        data.write_training_set(sealed / "train", seeds_per_kind=1)
    # Ordinary files pass.
    ok = tmp_path / "ok.png"
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(ok)
    assert data.load_training_image(ok).shape == (4, 4, 3)


def test_training_modules_may_not_load_sealed_set(tmp_path: Path) -> None:
    sealed = data.write_sealed_set(tmp_path / "sealed", n=3)
    # Not training code: the evaluation path may load it.
    assert len(data.load_sealed_set(sealed)) == 3
    # A module that declares itself training code is refused.
    namespace: dict = {"__name__": "train.fake_training_module"}
    exec(
        "from train import data\n"
        "data.declare_training_module(__name__)\n"
        "def load(folder):\n"
        "    return data.load_sealed_set(folder)\n",
        namespace,
    )
    assert "train.fake_training_module" in data.training_modules()
    with pytest.raises(SealedSetError):
        namespace["load"](sealed)


def test_real_training_modules_declare_themselves() -> None:
    import importlib

    for name in ("train.ledgers", "train.policy"):
        try:
            importlib.import_module(name)
        except ModuleNotFoundError as exc:  # not built yet, or torch missing
            if exc.name in (name, "torch"):
                continue
            raise
        assert name in data.training_modules()
