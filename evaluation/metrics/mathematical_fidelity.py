"""
Metric 6: Mathematical Accuracy Score (MAS)
============================================
Validates that the *mathematical content* of generated Manim code matches
the prompt's ground-truth equations — not just that the code runs.

Strategy
--------
1.  Parse the generated code with the ``ast`` module.
2.  Walk every ``ast.Call`` node looking for:
        • ``MathTex(...)``
        • ``Tex(...)``
        • ``Axes.plot(lambda x: ..., ...)``   (extracts the lambda body)
3.  Collect each LaTeX string / lambda expression.
4.  Compare against ``ground_truth_equations`` from the problem spec using
    SymPy's symbolic equality (``sympy.simplify(a - b) == 0``).

The result is reported as:
    ``mas_score`` — fraction of expected equations that were found and matched.
    ``mas_details`` — per-equation breakdown for error analysis.

Dependencies
------------
    sympy  (pip install sympy)        — symbolic comparison
    latex2sympy2 (pip install latex2sympy2) — LaTeX → SymPy parsing (optional but recommended)

The metric degrades gracefully: if a ground-truth equation cannot be parsed
by SymPy, it falls back to a *normalised string similarity* check so that
the score is never silently inflated.
"""

from __future__ import annotations

import ast
import re
import warnings
from typing import Any

# ── Optional symbolic dependencies ─────────────────────────────────────────
_SYMPY_AVAILABLE = False
_L2S_AVAILABLE   = False

try:
    import sympy                    # noqa: F401
    from sympy import sympify, simplify, Symbol, symbols
    from sympy.parsing.sympy_parser import (
        parse_expr,
        standard_transformations,
        implicit_multiplication_application,
    )
    _SYMPY_AVAILABLE = True
except ImportError:
    pass

try:
    from latex2sympy2 import latex2sympy
    _L2S_AVAILABLE = True
except ImportError:
    pass


# ── LaTeX normalisation helpers ─────────────────────────────────────────────

def _strip_display_env(latex: str) -> str:
    """Remove \\[ \\] and $$ $$ display math wrappers."""
    latex = re.sub(r"^\s*\$\$|\$\$\s*$", "", latex)
    latex = re.sub(r"^\s*\\\[|\\\]\s*$", "", latex)
    latex = re.sub(r"^\s*\$|\$\s*$", "", latex)
    return latex.strip()


def _normalise_latex(latex: str) -> str:
    """Light normalisation for string-level fallback comparison."""
    s = _strip_display_env(latex)
    s = re.sub(r"\s+", " ", s).strip()
    # Remove spacing commands that carry no semantic weight
    s = re.sub(r"\\[,;!quad]+", "", s)
    return s


def _latex_to_sympy(latex_str: str):
    """Try to convert a LaTeX string to a SymPy expression.
    Returns None on failure."""
    if not _SYMPY_AVAILABLE:
        return None
    clean = _strip_display_env(latex_str)
    # Attempt latex2sympy2 first (handles more LaTeX idioms)
    if _L2S_AVAILABLE:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return latex2sympy(clean)
        except Exception:
            pass
    # Fallback: sympy's own LaTeX parser
    try:
        from sympy.parsing.latex import parse_latex
        return parse_latex(clean)
    except Exception:
        pass
    # Last resort: bare sympify (works for simple algebraic strings)
    try:
        t = standard_transformations + (implicit_multiplication_application,)
        return parse_expr(clean, transformations=t)
    except Exception:
        return None


def _sympy_equal(a_expr, b_expr) -> bool:
    """Check symbolic equality via simplify(a - b) == 0."""
    try:
        diff = simplify(a_expr - b_expr)
        return diff == 0
    except Exception:
        return False


def _string_similarity(a: str, b: str) -> float:
    """Normalised edit-distance similarity in [0, 1]."""
    a, b = _normalise_latex(a), _normalise_latex(b)
    if a == b:
        return 1.0
    # Levenshtein ratio (simple DP)
    la, lb = len(a), len(b)
    if la == 0 and lb == 0:
        return 1.0
    if la == 0 or lb == 0:
        return 0.0
    # Cap length to avoid O(n²) on huge strings
    a, b = a[:300], b[:300]
    la, lb = len(a), len(b)
    dp = list(range(lb + 1))
    for i in range(1, la + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev[j - 1] + cost)
    dist = dp[lb]
    return 1.0 - dist / max(la, lb)


# ── AST extraction ──────────────────────────────────────────────────────────

_MATH_CALL_NAMES = {
    "MathTex", "Tex", "TexText",          # direct string-based
    "Title", "Paragraph",
}

_PLOT_CALL_NAMES = {
    "plot",                                # axes.plot(lambda x: ...) / axes.plot(func)
}


def _extract_string_args(node: ast.Call) -> list[str]:
    """Pull all string-literal positional arguments from a Call node."""
    result = []
    for arg in node.args:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            result.append(arg.value)
        elif isinstance(arg, ast.JoinedStr):
            # f-string: collect constant parts as a rough approximation
            parts = [
                v.value for v in arg.values
                if isinstance(v, ast.Constant) and isinstance(v.value, str)
            ]
            if parts:
                result.append("".join(parts))
    return result


def _lambda_body_to_str(node: ast.Lambda) -> str | None:
    """Try to unparse a lambda body to a string (Python ≥ 3.9)."""
    try:
        return ast.unparse(node.body)
    except Exception:
        return None


def extract_math_expressions(code: str) -> dict[str, list[str]]:
    """
    Walk the AST and collect:
      - ``latex_strings``: all strings passed to MathTex/Tex/etc.
      - ``plot_lambdas``:  unparsed lambda bodies from Axes.plot calls.

    Returns a dict with those two keys.
    """
    latex_strings: list[str] = []
    plot_lambdas:  list[str] = []

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {"latex_strings": latex_strings, "plot_lambdas": plot_lambdas}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        # Resolve function name (handles obj.method() and bare func())
        func = node.func
        func_name = ""
        if isinstance(func, ast.Name):
            func_name = func.id
        elif isinstance(func, ast.Attribute):
            func_name = func.attr

        # MathTex / Tex / etc.
        if func_name in _MATH_CALL_NAMES:
            latex_strings.extend(_extract_string_args(node))

        # Axes.plot(lambda x: ...)
        if func_name in _PLOT_CALL_NAMES:
            for arg in node.args:
                if isinstance(arg, ast.Lambda):
                    body = _lambda_body_to_str(arg)
                    if body:
                        plot_lambdas.append(body)

    return {"latex_strings": latex_strings, "plot_lambdas": plot_lambdas}


# ── Core comparison logic ───────────────────────────────────────────────────

def _compare_equation(
    generated: str,
    ground_truth: str,
    *,
    sympy_threshold: float = 0.0,
    string_threshold: float = 0.75,
) -> dict[str, Any]:
    """
    Compare one generated equation against one ground-truth equation.

    Returns:
        {
            "match": bool,
            "method": "symbolic" | "string" | "none",
            "confidence": float,
            "generated": str,
            "ground_truth": str,
        }
    """
    gen_sympy  = _latex_to_sympy(generated)
    gt_sympy   = _latex_to_sympy(ground_truth)

    # ── Symbolic path ──────────────────────────────────────────────────────
    if gen_sympy is not None and gt_sympy is not None:
        match = _sympy_equal(gen_sympy, gt_sympy)
        return {
            "match": match,
            "method": "symbolic",
            "confidence": 1.0 if match else 0.0,
            "generated": generated,
            "ground_truth": ground_truth,
        }

    # ── String fallback ────────────────────────────────────────────────────
    sim = _string_similarity(generated, ground_truth)
    match = sim >= string_threshold
    return {
        "match": match,
        "method": "string",
        "confidence": sim,
        "generated": generated,
        "ground_truth": ground_truth,
    }


def _best_match(generated_list: list[str], ground_truth: str) -> dict[str, Any]:
    """
    Find the best-matching generated expression for a given ground-truth.
    Returns the comparison result with the highest confidence.
    """
    if not generated_list:
        return {
            "match": False,
            "method": "none",
            "confidence": 0.0,
            "generated": "",
            "ground_truth": ground_truth,
        }
    results = [_compare_equation(g, ground_truth) for g in generated_list]
    return max(results, key=lambda r: (r["match"], r["confidence"]))


# ── Public API ──────────────────────────────────────────────────────────────

def compute_mathematical_fidelity(
    code: str,
    ground_truth_equations: list[str] | None = None,
) -> dict[str, Any]:
    """
    Compute the Mathematical Accuracy Score (MAS) for a generated Manim script.

    Args:
        code:                    Generated Python/Manim source code.
        ground_truth_equations:  List of LaTeX equation strings from the
                                 problem spec (``problem["ground_truth_equations"]``).
                                 If empty/None the score is computed solely from
                                 the presence and parsability of math expressions
                                 in the code (a 0–1 density signal).

    Returns:
        {
            "mas_score":             float in [0.0, 1.0],
            "equations_found":       int,   # extracted from code
            "equations_expected":    int,   # from ground truth list
            "equations_matched":     int,
            "sympy_available":       bool,
            "latex2sympy_available": bool,
            "details":               list[dict],  # per-GT-equation breakdown
            "extracted": {
                "latex_strings": list[str],
                "plot_lambdas":  list[str],
            }
        }
    """
    extracted = extract_math_expressions(code)
    all_generated = extracted["latex_strings"] + extracted["plot_lambdas"]
    n_found = len(all_generated)

    # ── No ground truth → density-only score ──────────────────────────────
    if not ground_truth_equations:
        # Reward for having *any* mathematical content at all
        density = min(1.0, n_found / 3.0)   # 3+ expressions → full score
        return {
            "mas_score": round(density, 4),
            "equations_found": n_found,
            "equations_expected": 0,
            "equations_matched": 0,
            "sympy_available": _SYMPY_AVAILABLE,
            "latex2sympy_available": _L2S_AVAILABLE,
            "details": [],
            "extracted": extracted,
            "note": "No ground-truth equations provided; score reflects math expression density.",
        }

    # ── Ground-truth comparison ────────────────────────────────────────────
    details = []
    matched = 0
    for gt in ground_truth_equations:
        res = _best_match(all_generated, gt)
        if res["match"]:
            matched += 1
        details.append(res)

    n_expected = len(ground_truth_equations)
    mas_score  = matched / n_expected if n_expected > 0 else 0.0

    return {
        "mas_score":             round(mas_score, 4),
        "equations_found":       n_found,
        "equations_expected":    n_expected,
        "equations_matched":     matched,
        "sympy_available":       _SYMPY_AVAILABLE,
        "latex2sympy_available": _L2S_AVAILABLE,
        "details":               details,
        "extracted":             extracted,
    }
