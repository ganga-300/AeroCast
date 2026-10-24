import math

import pytest

from flight_delay.reporting.latex import (MINUS, escape, integer, macro, num, pct, signed, table,
                                          with_ci)


def test_number_formatting():
    assert num(0.36281) == "0.363"
    assert num(-0.0793, 3) == MINUS + "0.079"
    assert num(-0.0001, 3) == "0.000"            # no negative zero
    assert signed(0.033) == "+0.033" and signed(-0.004) == MINUS + "0.004"
    assert integer(53991) == "53,991"
    assert pct(0.2583) == "25.8\\%"
    with pytest.raises(ValueError):
        num(math.nan)
    with pytest.raises(ValueError):
        integer(1.5)


def test_ci_formatting():
    assert with_ci(0.3628, 0.3106, 0.4197) == "0.363 [0.311, 0.420]"
    assert with_ci(0.033, 0.0177, 0.049, sign=True) == "+0.033 [+0.018, +0.049]"


def test_escape_and_macro():
    assert escape("a_b & 50%") == r"a\_b \& 50\%"
    assert macro("TestAPLR", "0.363") == r"\newcommand{\TestAPLR}{0.363}"
    for bad in ("Test1", "test_ap", ""):
        with pytest.raises(ValueError):
            macro(bad, "x")


def test_table_structure():
    t = table("Cap", "tab:x", ["a", "b"], [["1", "2"], ["3", "4"]], "lr", source="x.json")
    assert t.count(r"\\") == 3 and r"\begin{table}" in t and r"\label{tab:x}" in t
    assert "x.json" in t.splitlines()[0]
    assert r"\begin{table*}" in table("C", "t", ["a"], [["1"]], "l", wide=True)
    with pytest.raises(ValueError):
        table("C", "t", ["a", "b"], [["1"]], "ll")
