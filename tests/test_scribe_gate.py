"""Tests for the scribe stroke-language description, extraction and gate."""

from __future__ import annotations

import json

import pytest

from scribe import gate as G
from scribe.format import system_prompt
from scribe.gate import extract_jsonl, gate
from scribe_fixtures import CANVAS, as_reply, st, valid_painting


def test_system_prompt_is_short_and_names_every_pigment() -> None:
    text = system_prompt(seed=7)
    assert len(text.splitlines()) <= 25
    for name in ("quinacridone_rose", "paynes_grey", "hookers_green"):
        assert name in text
    assert "256 x 192" in text and "seed 7" in text


def test_extract_from_prose_and_fence() -> None:
    ops = valid_painting()
    lines = extract_jsonl(as_reply(ops))
    assert [json.loads(x) for x in lines] == ops


def test_extract_bare_lines_amid_prose() -> None:
    text = "Sure!\n" + "\n".join(json.dumps(o) for o in valid_painting()[:3]) + "\nHope you like it."
    assert len(extract_jsonl(text)) == 3


def test_extract_keeps_stray_trailing_line_and_strips_commas() -> None:
    ops = valid_painting()
    body = "\n".join(json.dumps(o) + "," for o in ops[:-1])
    text = f"```\n{body}\n```\n{json.dumps(ops[-1])}\nDone."
    lines = extract_jsonl(text)
    assert [json.loads(x) for x in lines] == ops


def test_extract_unclosed_fence() -> None:
    text = "```json\n" + "\n".join(json.dumps(o) for o in valid_painting())
    assert len(extract_jsonl(text)) == len(valid_painting())


def test_valid_painting_passes() -> None:
    res = gate(as_reply(valid_painting()), CANVAS)
    assert res.ok, res.reasons
    assert res.image is not None and res.image.shape == (192, 256, 3)
    assert G.MIN_COVERAGE < res.stats["coverage"] < G.MAX_COVERAGE
    assert res.to_dict()["rendered"] is True


def _fails(ops, fragment: str) -> None:
    res = gate(ops, CANVAS)
    assert not res.ok
    assert res.image is None
    assert any(fragment in r for r in res.reasons), res.reasons


def _with(index: int, **changes):
    ops = valid_painting()
    ops[index] = {**ops[index], **changes}
    return ops


def test_unparseable_line() -> None:
    lines = [json.dumps(o) for o in valid_painting()]
    lines[2] = '{"op": "stroke", "path": [[1, 2]'
    _fails(lines, "not valid JSON")


def test_unknown_op() -> None:
    _fails(valid_painting() + [{"op": "splash", "power": 9}], "unknown op")


def test_path_too_short_and_too_long() -> None:
    _fails(_with(0, path=[[10, 10]]), "path needs")
    _fails(_with(0, path=[[10, 10]] * 9), "path needs")


def test_path_outside_canvas() -> None:
    _fails(_with(0, path=[[10, 10], [300, 10]]), "outside")


def test_width_limits() -> None:
    _fails(_with(0, width=0.5), "width")
    _fails(_with(0, width=0.41 * 192), "width")


def test_water_range() -> None:
    _fails(_with(0, water=1.5), "water")


def test_unknown_pigment_and_bad_concentration() -> None:
    _fails(_with(0, pigments={"titanium_white": 0.5}), "unknown pigment")
    _fails(_with(0, pigments={"cerulean_blue": 1.2}), "concentration")


def test_too_many_pigments() -> None:
    _fails(_with(0, pigments={"cerulean_blue": 0.1, "cadmium_yellow": 0.1,
                              "burnt_umber": 0.1, "indian_red": 0.1}), "at most 3")


def test_stroke_count_limits() -> None:
    few = [st([[10, 10], [50, 50]], 5, 0.5, {"indian_red": 0.3})] * 5 + [{"op": "dry", "steps": 100}]
    _fails(few, "need 6 to 200")
    many = [st([[10, 10], [50, 50]], 5, 0.5, {"indian_red": 0.3})] * 201 + [{"op": "dry", "steps": 100}]
    _fails(many, "need 6 to 200")


def test_needs_a_dry_op() -> None:
    _fails([o for o in valid_painting() if o["op"] != "dry"], "no dry op")


def test_total_dry_steps_cap() -> None:
    _fails(valid_painting() + [{"op": "dry", "steps": 20_000}], "dry steps in total")


def test_canvas_line_from_model_is_ignored() -> None:
    lines = [{"op": "canvas", "width": 999, "height": 999, "seed": 1}] + valid_painting()
    _, ops, _ = G.static_check(lines, CANVAS)
    assert all(o["op"] != "canvas" for o in ops)


def test_giant_paynes_grey_wash_is_rejected_by_rendered_checks() -> None:
    cheat = [st([[0, y], [128, y], [256, y]], 76, 1.0, {"paynes_grey": 0.5}) for y in range(0, 193, 38)]
    cheat.append({"op": "dry", "steps": 1500})
    res = gate(cheat, CANVAS)
    assert not res.ok
    assert res.image is not None  # it passed the static checks and got rendered
    assert any("leave some paper" in r for r in res.reasons)
    assert any("one flat colour" in r for r in res.reasons)


def test_all_black_wash_is_rejected() -> None:
    heavy = {"paynes_grey": 1.0, "burnt_umber": 1.0, "french_ultramarine": 1.0}
    ops = [st([[0, y], [128, y], [256, y]], 76, 1.0, heavy) for y in range(0, 193, 38)] * 2
    res = gate(ops + [{"op": "dry", "steps": 1500}], CANVAS)
    assert not res.ok and any("too dark" in r for r in res.reasons)


def test_single_flat_wash_is_rejected() -> None:
    band = [st([[52, y], [128, y + 1], [204, y]], 40, 0.8, {"french_ultramarine": 0.45}) for y in (62, 90, 118)]
    res = gate(band * 2 + [{"op": "dry", "steps": 1200}], CANVAS)
    assert not res.ok
    assert res.reasons == ["only one flat colour; use at least two hues or two clearly different values"]


def test_nearly_blank_sheet_is_rejected() -> None:
    dots = [st([[100, 100], [102, 101]], 1, 0.3, {"cadmium_yellow": 0.1})] * 6
    res = gate(dots + [{"op": "dry", "steps": 100}], CANVAS)
    assert not res.ok and any("at least 5%" in r for r in res.reasons)


@pytest.mark.parametrize("bad", ["", "no strokes here at all"])
def test_empty_reply(bad: str) -> None:
    res = gate(bad, CANVAS)
    assert not res.ok and "no stroke lines found" in res.reasons
