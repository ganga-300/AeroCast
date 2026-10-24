"""Structural checks for the LaTeX report (TeX is not installed, so it cannot be compiled here).

Reads saved artifacts only; never accesses raw data or test labels.
"""

import hashlib
import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
MAIN = REPORT / "main.tex"
pytestmark = pytest.mark.skipif(not MAIN.exists(), reason="report not present")


def strip_comments(tex: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", tex)


def load_generator():
    spec = importlib.util.spec_from_file_location("make_report_tables", ROOT / "scripts" / "make_report_tables.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def main_tex():
    return strip_comments(MAIN.read_text())


def test_inputs_and_figures_exist(main_tex):
    for name in re.findall(r"\\input\{([^}]+)\}", main_tex):
        assert (REPORT / f"{name}.tex").is_file(), name
    for fig in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", main_tex):
        assert (REPORT / "figures" / fig).is_file(), fig


def test_citations_resolve(main_tex):
    bib = (REPORT / "references.bib").read_text()
    keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    cited = {k.strip() for group in re.findall(r"\\cite\{([^}]+)\}", main_tex) for k in group.split(",")}
    assert cited and cited <= keys, sorted(cited - keys)


def test_refs_resolve(main_tex):
    generated = "".join(p.read_text() for p in (REPORT / "generated").glob("tab_*.tex"))
    labels = set(re.findall(r"\\label\{([^}]+)\}", main_tex + generated))
    refs = set(re.findall(r"\\ref\{([^}]+)\}", main_tex + generated))
    assert refs <= labels, sorted(refs - labels)


def test_result_macros_defined(main_tex):
    defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", (REPORT / "generated" / "numbers.tex").read_text()))
    used = set(re.findall(r"\\([A-Z][A-Za-z]+)", main_tex)) - {"IEEEauthorblockN", "IEEEauthorblockA",
                                                                 "IEEEoverridecommandlockouts", "S",
                                                                 "Big", "Pr"}  # standard LaTeX math commands
    assert used <= defined, sorted(used - defined)


def test_braces_balanced():
    for p in [MAIN, *(REPORT / "generated").glob("*.tex")]:
        text = re.sub(r"\\[{}]", "", strip_comments(p.read_text()))
        assert text.count("{") == text.count("}"), p.name


def test_generated_files_match_fresh_build():
    """Committed tables/macros must equal a rebuild from the saved JSON (no stale or edited numbers)."""
    gen = load_generator()
    src = {k: json.loads(p.read_text()) for k, p in gen.SOURCES.items()}
    from flight_delay.config import load_config
    macros, tabs = gen.build(src, load_config())
    numbers = (REPORT / "generated" / "numbers.tex").read_text()
    for name, value in macros.items():
        assert f"\\newcommand{{\\{name}}}{{{value}}}" in numbers, name
    for name, tex in tabs.items():
        assert (REPORT / "generated" / f"{name}.tex").read_text() == tex, name


def test_key_numbers_trace_to_json():
    t = json.loads((ROOT / "reports" / "final" / "test_evaluation.json").read_text())
    numbers = (REPORT / "generated" / "numbers.tex").read_text()
    lr = t["models"]["logistic_regression_sigmoid"]["metrics"]
    assert f"\\newcommand{{\\TestAPLR}}{{{lr['average_precision']:.3f}}}" in numbers
    assert f"\\newcommand{{\\TestROCLR}}{{{lr['roc_auc']:.3f}}}" in numbers
    assert f"\\newcommand{{\\TestPrev}}{{{t['partition_summary']['prevalence']:.3f}}}" in numbers


def test_frozen_predictions_match_recorded_hash():
    t = json.loads((ROOT / "reports" / "final" / "test_evaluation.json").read_text())
    pred = ROOT / "reports" / "final" / "test_predictions.csv"
    assert hashlib.sha256(pred.read_bytes()).hexdigest() == t["test_predictions_sha256"]
