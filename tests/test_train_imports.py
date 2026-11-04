"""The training code treats the scorer as a black box, as docs/TRAINING.md requires."""

from __future__ import annotations

import ast
from pathlib import Path

TRAIN = Path(__file__).resolve().parent.parent / "train"


def test_training_code_imports_only_score_from_the_scorer() -> None:
    files = sorted(TRAIN.glob("*.py"))
    assert files
    for path in files:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "backrun.score":
                assert [a.name for a in node.names] == ["score"], path.name
            if isinstance(node, ast.ImportFrom) and node.module == "backrun":
                assert "score" not in [a.name for a in node.names], path.name
            if isinstance(node, ast.Import):
                assert "backrun.score" not in [a.name for a in node.names], path.name
