"""The Modal app parses, imports with or without Modal, and has its five functions."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FUNCTIONS = ("make_targets", "generate_ledgers", "train_imitation", "evaluate", "finetune_rl")


def test_module_defines_the_functions_and_entrypoint() -> None:
    tree = ast.parse((ROOT / "train" / "modal_app.py").read_text())
    defs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    for name in (*FUNCTIONS, "main"):
        assert name in defs and defs[name].decorator_list, name
    args = [a.arg for a in defs["main"].args.args]
    assert args[0] == "stage"


def test_imports_and_exposes_functions() -> None:
    from train import modal_app

    for name in FUNCTIONS:
        assert hasattr(modal_app, name), name
    assert set(modal_app.STAGES) == {"targets", "ledgers", "imitation", "evaluate", "all"}


def test_imports_without_modal() -> None:
    code = (
        "import sys; sys.modules['modal'] = None\n"
        "from train import modal_app as m\n"
        "assert m.modal is None\n"
        f"assert all(callable(getattr(m, n)) for n in {FUNCTIONS!r})\n"
        "try:\n"
        "    m.finetune_rl()\n"
        "except NotImplementedError as e:\n"
        "    assert 'Stage B' in str(e)\n"
        "else:\n"
        "    raise AssertionError('finetune_rl should raise')\n"
    )
    res = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr


def test_painter_specs() -> None:
    from train import modal_app as m

    assert m.painter_spec("search") == ("search", "search")
    assert m.painter_spec("blank") == ("blank", "blank")
    assert m.painter_spec("mean") == ("mean:/data/ledgers", "mean")
    assert m.painter_spec("stage_a/best.pt") == ("policy:/checkpoints/stage_a/best.pt", "stage_a_best")
    assert m.painter_spec("policy:/x/y.pt") == ("policy:/x/y.pt", "x_y")


def test_comparison_section(tmp_path: Path) -> None:
    import json

    from train import modal_app as m

    def report(name: str, total: float) -> None:
        d = tmp_path / f"{name}_24x18"
        (d / "paintings").mkdir(parents=True)
        summary = {k: {"mean": 1.0, "std": 0.0, "n": 1} for k in ("lab_l2", "ssim", "strokes", "start_std_x_px",
                                                                   "start_std_y_px", "distinct_strokes")}
        summary["total"] = {"mean": total, "std": 0.0, "n": 1}
        rows = [{"distinct_strokes": 1}]
        (d / "report.json").write_text(json.dumps({"painter": name, "n_paintings": 1, "summary": summary,
                                                   "paintings": rows}))

    assert m.comparison_section(tmp_path, "pol_24x18", "_24x18") == ""
    report("blank", 24.0)
    report("search", 6.0)
    report("pol", 15.0)
    md = m.comparison_section(tmp_path, "pol_24x18", "_24x18")
    assert "| pol | 1 | 15.000 |" in md and md.rstrip().endswith("0.5000 |")
