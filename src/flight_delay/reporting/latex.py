"""LaTeX formatting for numbers read from saved result artifacts.

The report never contains hand-typed results. Every number is formatted here from a JSON value
and emitted either as a macro (`\\newcommand`) or inside a generated table.
"""

from __future__ import annotations

import math
import re
from typing import Sequence

_ESCAPES = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
            "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
MINUS = "$-$"


def escape(text: str) -> str:
    """Escape LaTeX special characters in plain text."""
    return "".join(_ESCAPES.get(ch, ch) for ch in str(text))


def num(x: float, digits: int = 3) -> str:
    """Fixed-decimal number with a typographic minus. Non-finite values raise."""
    x = float(x)
    if not math.isfinite(x):
        raise ValueError(f"Non-finite value cannot be reported: {x}")
    s = f"{abs(x):.{digits}f}"
    if float(s) == 0:  # avoid "-0.000"
        return s
    return (MINUS + s) if x < 0 else s


def signed(x: float, digits: int = 3) -> str:
    """Like num() but with an explicit + for positive values (for differences)."""
    s = num(x, digits)
    return s if s.startswith(MINUS) or float(x) == 0 else "+" + s


def integer(n: int) -> str:
    """Integer with thousands separators (e.g. 53,991)."""
    if float(n) != int(n):
        raise ValueError(f"Not an integer: {n}")
    return f"{int(n):,}"


def pct(x: float, digits: int = 1) -> str:
    return num(100 * float(x), digits) + r"\%"


def with_ci(point: float, low: float, high: float, digits: int = 3, sign: bool = False) -> str:
    f = signed if sign else num
    return f"{f(point, digits)} [{f(low, digits)}, {f(high, digits)}]"


_MACRO_NAME = re.compile(r"^[A-Za-z]+$")


def macro(name: str, value: str) -> str:
    """\\newcommand line; LaTeX macro names may contain letters only."""
    if not _MACRO_NAME.match(name):
        raise ValueError(f"Invalid LaTeX macro name: {name!r}")
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def table(caption: str, label: str, header: Sequence[str], rows: Sequence[Sequence[str]],
          colspec: str, wide: bool = False, note: str | None = None, source: str = "") -> str:
    """booktabs table; cells are inserted verbatim (format/escape them before calling)."""
    if any(len(r) != len(header) for r in rows):
        raise ValueError("Every row must have as many cells as the header")
    env = "table*" if wide else "table"
    lines = [f"% Generated from {source}. Do not edit by hand." if source else "% Generated. Do not edit by hand.",
             f"\\begin{{{env}}}[!t]", "\\centering", "\\footnotesize",
             f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             f"\\begin{{tabular}}{{{colspec}}}", "\\toprule",
             " & ".join(header) + r" \\", "\\midrule"]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}"]
    if note:
        lines += [f"\\par\\vspace{{2pt}}\\parbox{{\\linewidth}}{{\\scriptsize {note}}}"]
    lines += [f"\\end{{{env}}}", ""]
    return "\n".join(lines)
