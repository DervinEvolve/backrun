"""Tests for the rated pool and the rating page logic."""

from __future__ import annotations

import json
import threading
import urllib.request
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scribe.gate import _blank
from scribe.pool import Pool
from scribe.rate import PAGE, RatingApp, make_handler

SEED = 3
BLUE = (70, 110, 190)
YELLOW = (225, 200, 70)
GREEN = (80, 150, 80)


def _add(root: Path, cid: str, model: str, prompt_id: str, colours, status: str = "ok") -> None:
    folder = root / cid
    folder.mkdir(parents=True)
    img = _blank(256, 192, SEED, "cold_press").copy()
    for i, c in enumerate(colours):
        img[20 + 50 * i: 60 + 50 * i, 30:200] = c
    if status == "ok":
        Image.fromarray(img).save(folder / "final.png")
        Image.fromarray(img).resize((512, 384)).save(folder / "preview.png")
        (folder / "strokes.jsonl").write_text(json.dumps({"op": "canvas", "seed": SEED}) + "\n" + cid + "\n")
    meta = {"id": cid, "model": model, "prompt_id": prompt_id, "subject": f"subject {prompt_id}",
            "prompt": f"Paint {prompt_id}.", "seed": SEED, "status": status}
    (folder / "meta.json").write_text(json.dumps(meta))
    with (root / "index.jsonl").open("a") as fh:
        fh.write(json.dumps({"id": cid, "prompt_id": prompt_id, "model": model, "status": status, "dir": cid}) + "\n")


@pytest.fixture()
def pool(tmp_path: Path) -> Pool:
    root = tmp_path / "pool"
    root.mkdir()
    for i in range(6):
        _add(root, f"p{i % 2}__m{i % 3}__{i:02d}", f"m{i % 3}", f"p{i % 2}", [BLUE, YELLOW] if i < 4 else [GREEN])
    _add(root, "p0__m0__99", "m0", "p0", [], status="rejected")
    return Pool(root)


def test_only_gated_candidates_are_rateable(pool: Pool) -> None:
    assert len(pool.candidates()) == 6
    with pytest.raises(KeyError):
        pool.rate("p0__m0__99", "love")


def test_rating_last_line_wins_and_summary(pool: Pool) -> None:
    ids = [c["id"] for c in pool.candidates()]
    pool.rate(ids[0], "nope")
    pool.rate(ids[0], "love")
    pool.rate(ids[1], "okay")
    assert pool.ratings() == {ids[0]: "love", ids[1]: "okay"}
    s = pool.summary()
    assert s["rated"] == 2 and s["total"] == 6
    assert s["tiers"] == {"love": 1, "okay": 1, "nope": 0, "unrated": 4}
    assert sum(sum(v.values()) for v in s["models"].values()) == 6
    with pytest.raises(ValueError):
        pool.rate(ids[2], "meh")


def test_undo_walks_back(pool: Pool) -> None:
    ids = [c["id"] for c in pool.candidates()]
    pool.rate(ids[0], "okay")
    pool.rate(ids[1], "nope")
    pool.rate(ids[1], "love")
    assert pool.undo() == ids[1]
    assert pool.ratings()[ids[1]] == "nope"
    assert pool.undo() == ids[1]
    assert ids[1] not in pool.ratings()
    assert pool.undo() == ids[0]
    assert pool.ratings() == {}
    assert pool.undo() is None


def test_next_unrated_follows_rating_order(pool: Pool) -> None:
    order = [c["id"] for c in pool.candidates()]
    assert pool.next_unrated()["id"] == order[0]
    pool.rate(order[0], "okay")
    assert pool.next_unrated()["id"] == order[1]


def test_sample_references_is_deterministic_and_respects_tiers(pool: Pool) -> None:
    ids = [c["id"] for c in pool.candidates()]
    for cid, tier in zip(ids, ["love", "love", "okay", "okay", "nope", "nope"]):
        pool.rate(cid, tier)
    a = pool.sample_references(11, k=2)
    b = pool.sample_references(np.random.default_rng(11), k=2)
    assert a == b and len({r["id"] for r in a}) == 2
    assert all(r["tier"] in ("love", "okay") for r in a)
    loves = pool.sample_references(5, k=2, mix=1.0)
    assert all(r["tier"] == "love" for r in loves)
    assert all(Path(r["path"]).is_file() for r in loves)
    with pytest.raises(ValueError):
        pool.sample_references(1, k=5)


def test_manifest_changes_with_ratings(pool: Pool) -> None:
    ids = [c["id"] for c in pool.candidates()]
    pool.rate(ids[0], "love")
    m1 = pool.manifest()
    assert m1["rated"] == 1 and m1["tiers"]["love"] == 1 and len(m1["sha256"]) == 64
    assert pool.manifest() == m1
    pool.rate(ids[0], "okay")
    assert pool.manifest()["sha256"] != m1["sha256"]


def test_colour_coverage_names_thin_bins(pool: Pool) -> None:
    for c in pool.candidates():
        pool.rate(c["id"], "okay")
    cov = pool.colour_coverage()
    assert cov["paintings"] == 6
    present = {b["name"] for b in cov["hue"] if b["paintings"] >= cov["thin_floor"]}
    assert present and "violet" in cov["thin_hues"]
    assert "Thin in 6 rated paintings" in cov["line"] and "violet" in cov["line"]
    assert len(cov["lightness"]) == 10


def test_colour_coverage_empty_pool(tmp_path: Path) -> None:
    assert "No paintings rated" in Pool(tmp_path).colour_coverage()["line"]


def test_rating_app_state_rate_and_undo(pool: Pool) -> None:
    app = RatingApp(pool)
    s = app.state()
    assert s["rated"] == 0 and s["total"] == 6 and s["current"]["image"].startswith("/img/")
    first = s["current"]["id"]
    s = app.rate(first, "love")
    assert s["rated"] == 1 and s["counts"]["love"] == 1 and s["current"]["id"] != first
    s = app.undo()
    assert s["current"]["id"] == first and s["rated"] == 0
    assert app.image_path(first).name == "preview.png"
    with pytest.raises(KeyError):
        app.image_path("../../etc/passwd")


def test_http_round_trip(pool: Pool) -> None:
    from http.server import ThreadingHTTPServer

    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(RatingApp(pool)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    direct = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        assert b"rated" in direct.open(base + "/").read()
        state = json.loads(direct.open(base + "/api/state").read())
        img = direct.open(base + state["current"]["image"]).read()
        assert img[:4] == b"\x89PNG"
        req = urllib.request.Request(base + "/api/rate", method="POST",
                                     data=json.dumps({"id": state["current"]["id"], "tier": "nope"}).encode())
        after = json.loads(direct.open(req).read())
        assert after["counts"]["nope"] == 1
        line = json.loads(pool.ratings_path.read_text().splitlines()[-1])
        assert line["tier"] == "nope" and "at" in line
        bad = urllib.request.Request(base + "/api/rate", method="POST", data=b'{"id": "x", "tier": "love"}')
        with pytest.raises(urllib.error.HTTPError) as info:
            direct.open(bad)
        assert info.value.code == 400
    finally:
        srv.shutdown()


def test_page_has_keys_and_drawers() -> None:
    for fragment in ('data-tier="love"', 'data-tier="okay"', 'data-tier="nope"', 'k==="u"', "rated ${fmt"):
        assert fragment in PAGE


def test_sample_references_never_returns_the_excluded_painting(pool: Pool) -> None:
    for c in pool.candidates():
        pool.rate(c["id"], "okay")
    ids = [c["id"] for c in pool.candidates() if c["prompt_id"] == "p0"]
    for _ in range(20):
        refs = pool.sample_references(np.random.default_rng(_), k=2, prompt_id="p0", exclude=ids[0])
        assert ids[0] not in {r["id"] for r in refs}
        assert all(r["prompt_id"] == "p0" for r in refs)


def test_export_writes_a_pool_of_the_kept_candidates(pool: Pool, tmp_path: Path) -> None:
    ids = [c["id"] for c in pool.candidates()]
    pool.rate(ids[0], "love")
    pool.rate(ids[1], "okay")
    pool.rate(ids[2], "nope")
    info = pool.export(tmp_path / "keeps")
    small = Pool(tmp_path / "keeps")
    assert info["exported"] == 2 and info["manifest"] == small.manifest()["sha256"]
    assert {c["id"] for c in small.candidates()} == {ids[0], ids[1]}
    assert small.ratings() == {ids[0]: "love", ids[1]: "okay"}
    assert (tmp_path / "keeps" / ids[0] / "final.png").is_file()
    assert small.sample_references(0, k=2, mix=0.0) and len(small.summary()["models"]) >= 1
