"""Shared stroke files and helpers for the scribe tests."""

from __future__ import annotations

import json

from scribe.format import canvas_header

CANVAS = canvas_header(seed=7)


def st(path, width, water, pigments):
    return {"op": "stroke", "path": path, "width": width, "water": water, "pigments": pigments}


def valid_painting() -> list[dict]:
    """A small landscape: sky and field washes, dry, hills, dry, trees and a sun."""
    return [
        st([[10, 20], [128, 18], [246, 22]], 40, 0.9, {"cerulean_blue": 0.25}),
        st([[10, 50], [128, 52], [246, 48]], 30, 0.9, {"cerulean_blue": 0.15, "quinacridone_rose": 0.05}),
        st([[10, 150], [128, 148], [246, 152]], 50, 0.8, {"cadmium_yellow": 0.3, "hookers_green": 0.1}),
        {"op": "dry", "steps": 1500},
        st([[0, 110], [60, 90], [130, 105], [200, 88], [256, 100]], 24, 0.7, {"hookers_green": 0.45}),
        st([[200, 30], [204, 34]], 14, 0.6, {"cadmium_yellow": 0.7}),
        {"op": "dry", "steps": 1500},
        st([[60, 140], [62, 112]], 4, 0.4, {"burnt_umber": 0.7}),
        st([[150, 140], [151, 108]], 4, 0.4, {"burnt_umber": 0.7}),
        st([[60, 112], [62, 100]], 12, 0.5, {"hookers_green": 0.8, "paynes_grey": 0.2}),
        st([[150, 108], [152, 96]], 12, 0.5, {"hookers_green": 0.8, "paynes_grey": 0.2}),
        {"op": "dry", "steps": 800},
    ]


def as_reply(ops: list[dict], prose: bool = True) -> str:
    body = "\n".join(json.dumps(o) for o in ops)
    if not prose:
        return body
    return f"Here is a quiet landscape.\n\n```jsonl\n{body}\n```\n\nI kept the sky pale."
