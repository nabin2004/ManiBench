"""
Metric 7: Code Maintainability Index (CMI)
==========================================
Assesses the *structural quality* of generated Manim code — not just
correctness, but how clean, readable, and maintainable the code is.

Sub-metrics (composited into CMI ∈ [0, 1]):
    1. Cyclomatic Complexity (CC)        — via radon (or AST fallback)
    2. Halstead Volume (HV)              — via radon
    3. PEP-8 Compliance (P8)            — violation count via pycodestyle
    4. Maintainability Index (MI)        — radon's built-in formula
    5. Monolith Penalty                  — extra penalty for huge construct()

The final ``cmi_score`` is in [0.0, 1.0]; higher is better.

Dependencies
------------
    radon       (pip install radon)
    pycodestyle (pip install pycodestyle)

Both degrade gracefully if not installed.
"""

from __future__ import annotations

import ast
import io
import math
import tempfile
from pathlib import Path
from typing import Any

# ── Optional dependencies ───────────────────────────────────────────────────
_RADON_AVAILABLE        = False
_PYCODESTYLE_AVAILABLE  = False

try:
    from radon.complexity  import cc_visit, average_complexity
    from radon.metrics     import mi_visit, h_visit
    from radon.raw         import analyze
    _RADON_AVAILABLE = True
except ImportError:
    pass

try:
    import pycodestyle
    _PYCODESTYLE_AVAILABLE = True
except ImportError:
    pass


# ── Tuneable thresholds ─────────────────────────────────────────────────────
# Cyclomatic complexity: A=1-5, B=6-10, C=11-15, D=16-20, E=21-25, F>25
CC_RANKS = {"A": 1.0, "B": 0.80, "C": 0.55, "D": 0.30, "E": 0.10, "F": 0.0}

MI_PERFECT   = 100.0   # radon MI upper bound
MI_THRESHOLD = 20.0    # MI < 20 is considered "very low maintainability"

MAX_CONSTRUCT_LINES = 80    # Lines in construct() before monolith penalty kicks in
PEP8_PER_ERROR_PENALTY = 0.02   # Each PEP-8 violation subtracts this from P8 score
MAX_HALSTEAD_VOLUME = 10_000.0  # Volume above this → HV score 0


# ── AST-level helpers ───────────────────────────────────────────────────────

def _count_branches_ast(code: str) -> int:
    """Estimate cyclomatic complexity from AST (fallback for no radon)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 0
    branch_nodes = (
        ast.If, ast.For, ast.While, ast.ExceptHandler,
        ast.With, ast.Assert, ast.comprehension,
    )
    return sum(1 for _ in ast.walk(tree) if isinstance(_, branch_nodes)) + 1


def _construct_method_lines(code: str) -> int:
    """Return the line-count of the largest ``construct`` method found."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 0
    max_lines = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "construct":
            # end_lineno available in Python 3.8+
            end = getattr(node, "end_lineno", node.lineno)
            length = end - node.lineno + 1
            max_lines = max(max_lines, length)
    return max_lines


def _helper_method_count(code: str) -> int:
    """Count non-construct instance methods (proxy for decomposition quality)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 0
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in ast.walk(node):
                if isinstance(item, ast.FunctionDef) and item.name != "construct":
                    count += 1
    return count


# ── Sub-score computations ──────────────────────────────────────────────────

def _cyclomatic_score(code: str) -> tuple[float, dict]:
    """Score in [0, 1] from cyclomatic complexity; A=best, F=worst."""
    if _RADON_AVAILABLE:
        try:
            blocks = cc_visit(code)
            if blocks:
                avg_cc = sum(b.complexity for b in blocks) / len(blocks)
                max_cc = max(b.complexity for b in blocks)
                # Map average CC to a rank score
                if avg_cc <= 5:
                    score = CC_RANKS["A"]
                elif avg_cc <= 10:
                    score = CC_RANKS["B"]
                elif avg_cc <= 15:
                    score = CC_RANKS["C"]
                elif avg_cc <= 20:
                    score = CC_RANKS["D"]
                elif avg_cc <= 25:
                    score = CC_RANKS["E"]
                else:
                    score = CC_RANKS["F"]
                return score, {"avg_cc": round(avg_cc, 2), "max_cc": max_cc, "n_blocks": len(blocks)}
        except Exception:
            pass
    # AST fallback
    cc = _count_branches_ast(code)
    score = max(0.0, 1.0 - (cc - 1) / 40.0)
    return round(score, 4), {"avg_cc": cc, "source": "ast_fallback"}


def _halstead_score(code: str) -> tuple[float, dict]:
    """Score in [0, 1] from Halstead volume (lower volume → higher score)."""
    if _RADON_AVAILABLE:
        try:
            h = h_visit(code)
            if h and hasattr(h[0], "volume"):
                vol = h[0].volume
                score = max(0.0, 1.0 - vol / MAX_HALSTEAD_VOLUME)
                return round(score, 4), {"halstead_volume": round(vol, 2)}
        except Exception:
            pass
    return 0.5, {"halstead_volume": None, "note": "radon unavailable"}


def _maintainability_index_score(code: str) -> tuple[float, dict]:
    """Score from radon Maintainability Index (0–100 → 0–1)."""
    if _RADON_AVAILABLE:
        try:
            mi_raw = mi_visit(code, multi=True)
            if mi_raw is not None:
                # radon MI is in [0, 100]
                score = max(0.0, min(1.0, mi_raw / MI_PERFECT))
                return round(score, 4), {"mi_raw": round(mi_raw, 2)}
        except Exception:
            pass
    return 0.5, {"mi_raw": None, "note": "radon unavailable"}


def _pep8_score(code: str) -> tuple[float, dict]:
    """Score from PEP-8 compliance; each violation subtracts from 1.0."""
    if not _PYCODESTYLE_AVAILABLE:
        return 0.5, {"violation_count": None, "note": "pycodestyle unavailable"}
    try:
        violations = []
        checker = pycodestyle.StyleGuide(quiet=True)
        # Feed via virtual file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            tmp_path = f.name
        result = checker.check_files([tmp_path])
        Path(tmp_path).unlink(missing_ok=True)
        count = result.total_errors
        score = max(0.0, 1.0 - count * PEP8_PER_ERROR_PENALTY)
        return round(score, 4), {"violation_count": count}
    except Exception:
        return 0.5, {"violation_count": None, "note": "pycodestyle error"}


def _monolith_penalty(code: str) -> tuple[float, dict]:
    """
    Extra penalty for oversized ``construct()`` methods.
    Score 1.0 if construct ≤ MAX_CONSTRUCT_LINES, linearly decays to 0 at 3×.
    Also gives a small *bonus* for helper method decomposition.
    """
    lines = _construct_method_lines(code)
    helpers = _helper_method_count(code)

    if lines == 0:
        score = 1.0
    elif lines <= MAX_CONSTRUCT_LINES:
        score = 1.0
    else:
        # Linear decay from 1.0 to 0.0 between MAX_CONSTRUCT_LINES and 3×MAX
        over = lines - MAX_CONSTRUCT_LINES
        total_over = 2 * MAX_CONSTRUCT_LINES  # 3×MAX - MAX
        score = max(0.0, 1.0 - over / total_over)

    # Helper method bonus (up to +0.2, capped at 1.0)
    bonus = min(0.2, helpers * 0.05)
    score = min(1.0, score + bonus)
    return round(score, 4), {
        "construct_lines": lines,
        "helper_methods": helpers,
    }


# ── Weights ─────────────────────────────────────────────────────────────────
_WEIGHTS = {
    "cyclomatic":     0.25,
    "halstead":       0.15,
    "maintainability": 0.30,
    "pep8":           0.20,
    "monolith":       0.10,
}


# ── Public API ──────────────────────────────────────────────────────────────

def compute_code_quality(code: str) -> dict[str, Any]:
    """
    Compute the Code Maintainability Index (CMI) for a generated Manim script.

    Args:
        code: Generated Python/Manim source code string.

    Returns:
        {
            "cmi_score":      float in [0.0, 1.0],   higher = better
            "sub_scores": {
                "cyclomatic":      float,
                "halstead":        float,
                "maintainability": float,
                "pep8":            float,
                "monolith":        float,
            },
            "details": {
                "cyclomatic":      dict,
                "halstead":        dict,
                "maintainability": dict,
                "pep8":            dict,
                "monolith":        dict,
            },
            "radon_available":       bool,
            "pycodestyle_available": bool,
        }
    """
    cc_score,  cc_det  = _cyclomatic_score(code)
    hv_score,  hv_det  = _halstead_score(code)
    mi_score,  mi_det  = _maintainability_index_score(code)
    p8_score,  p8_det  = _pep8_score(code)
    ml_score,  ml_det  = _monolith_penalty(code)

    sub_scores = {
        "cyclomatic":      cc_score,
        "halstead":        hv_score,
        "maintainability": mi_score,
        "pep8":            p8_score,
        "monolith":        ml_score,
    }
    cmi_score = sum(_WEIGHTS[k] * sub_scores[k] for k in _WEIGHTS)

    return {
        "cmi_score":  round(cmi_score, 4),
        "sub_scores": sub_scores,
        "details": {
            "cyclomatic":      cc_det,
            "halstead":        hv_det,
            "maintainability": mi_det,
            "pep8":            p8_det,
            "monolith":        ml_det,
        },
        "radon_available":       _RADON_AVAILABLE,
        "pycodestyle_available": _PYCODESTYLE_AVAILABLE,
    }
