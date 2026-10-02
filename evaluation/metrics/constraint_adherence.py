"""
Metric 8: Constraint Adherence Rate (CAR)
==========================================
Measures whether the generated Manim code *follows the specific constraints*
stated in the original prompt (colour choices, object prohibitions, background
settings, scene structure, etc.).

The evaluator uses an **LLM-as-a-Judge** pipeline:
    1. Build a structured rubric from ``problem["constraints"]``.
    2. Feed [system prompt + original task prompt + generated code] to a
       strong judge model (configurable, defaults to the same client used
       for generation).
    3. Parse the judge's per-constraint scores (1–5 Likert) and aggregate.

Offline / cheap fallback
------------------------
When ``use_llm_judge=False`` (or the judge API is unavailable), a deterministic
rule-based pass/fail check runs against each constraint using regex and AST
inspection.  This gives a coarser but reproducible signal.

Result schema
-------------
    {
        "car_score":         float [0.0, 1.0],   mean normalised Likert
        "method":            "llm_judge" | "rule_based",
        "constraint_count":  int,
        "details": [
            {
                "constraint":   str,
                "score":        float,    # 0.0–1.0 (LLM 1–5 → 0.0–1.0)
                "passed":       bool,
                "rationale":    str | None,
            }, ...
        ],
        "judge_model":       str | None,
        "judge_raw":         str | None,    # full judge output for auditing
    }
"""

from __future__ import annotations

import ast
import json
import re
import time
from typing import Any

# ── LLM judge system prompt ─────────────────────────────────────────────────
_JUDGE_SYSTEM = """You are an expert evaluator of Manim animation code quality.
Your task is to assess whether the generated Manim Python code satisfies each
of the listed constraints from the original animation task description.

For EACH constraint, provide:
  - A score from 1 to 5 (1 = completely violated, 5 = fully satisfied)
  - A one-sentence rationale

Return your evaluation as valid JSON only, with no extra prose, in this exact format:
{
  "evaluations": [
    {"constraint": "<constraint text>", "score": <1-5>, "rationale": "<one sentence>"},
    ...
  ]
}
"""

_JUDGE_USER_TEMPLATE = """## Original Animation Task Prompt
{task_prompt}

## Constraints to Evaluate
{constraints_list}

## Generated Manim Code
```python
{code}
```

Evaluate each constraint above and return JSON only."""


# ── Rule-based fallback detectors ───────────────────────────────────────────

def _rule_check(code: str, constraint: str) -> tuple[bool, str]:
    """
    Best-effort deterministic check for common constraint types.
    Returns (passed: bool, evidence: str).
    """
    c_low = constraint.lower()
    code_low = code.lower()

    # Dark/light background
    if "dark background" in c_low or "black background" in c_low:
        passed = bool(re.search(r"background_color\s*=\s*['\"]?BLACK|#000|dark", code_low))
        return passed, "background_color=BLACK found" if passed else "no dark background set"

    if "light background" in c_low or "white background" in c_low:
        passed = bool(re.search(r"background_color\s*=\s*['\"]?WHITE|#FFF|#fff", code_low))
        return passed, "background_color=WHITE found" if passed else "no light background set"

    # Colour constraints: "make the X red/blue/..."
    colour_match = re.search(
        r"(?:make|use|set|colour|color).*?\b(red|blue|green|yellow|orange|purple|white|black)\b",
        c_low,
    )
    if colour_match:
        colour = colour_match.group(1).upper()
        passed = colour in code.upper()
        return passed, f"colour {colour} {'found' if passed else 'not found'} in code"

    # "Do not use 3D" / "no 3D"
    if "3d" in c_low and ("no " in c_low or "not" in c_low or "avoid" in c_low):
        three_d_used = bool(re.search(r"ThreeDScene|ThreeDAxes|Surface|ParametricSurface", code))
        return not three_d_used, "3D objects " + ("absent ✓" if not three_d_used else "present ✗")

    # "Do not use text" / "no text"
    if "no text" in c_low or "without text" in c_low:
        text_used = bool(re.search(r"\bText\s*\(|\bTex\s*\(|\bMathTex\s*\(", code))
        return not text_used, "Text/Tex " + ("absent ✓" if not text_used else "present ✗")

    # Generic keyword presence check
    keywords = [w for w in re.findall(r"[a-z_]{4,}", c_low) if w not in {
        "must", "should", "animation", "scene", "manim", "code", "that",
        "with", "this", "from", "into", "have", "make", "does", "does",
    }]
    if keywords:
        for kw in keywords:
            if kw in code_low:
                return True, f"keyword '{kw}' found in code"
    return False, "constraint keywords not detected"


# ── LLM judge helpers ────────────────────────────────────────────────────────

def _call_judge(
    client,
    judge_model_id: str,
    task_prompt: str,
    constraints: list[str],
    code: str,
    max_tokens: int = 1024,
    timeout: int = 60,
) -> tuple[str, list[dict]]:
    """
    Call the LLM judge and parse its JSON response.
    Returns (raw_response: str, evaluations: list[dict]).
    """
    constraints_list = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(constraints))
    user_msg = _JUDGE_USER_TEMPLATE.format(
        task_prompt=task_prompt[:2000],
        constraints_list=constraints_list,
        code=code[:4000],           # truncate very long code
    )
    messages = [
        {"role": "system", "content": _JUDGE_SYSTEM},
        {"role": "user", "content": user_msg},
    ]

    from evaluation.config import ModelSpec
    judge_spec = ModelSpec(
        id=judge_model_id,
        short_name="judge",
        provider="judge",
        temperature=0.0,
        max_tokens=max_tokens,
    )

    raw = ""
    try:
        result = client.generate(
            model=judge_spec,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0,
        )
        # The client returns {"code": ..., ...} but we want the raw content
        # Prefer raw_content if the client stores it, otherwise use code field
        raw = result.get("raw_content") or result.get("code", "") or ""
    except Exception as e:
        return f"JUDGE_ERROR: {e}", []

    # Extract JSON from response (may be surrounded by markdown code fences)
    json_match = re.search(r"\{.*\"evaluations\".*\}", raw, re.DOTALL)
    if not json_match:
        return raw, []
    try:
        parsed = json.loads(json_match.group())
        return raw, parsed.get("evaluations", [])
    except json.JSONDecodeError:
        return raw, []


# ── Public API ──────────────────────────────────────────────────────────────

def compute_constraint_adherence(
    code: str,
    problem: dict,
    client=None,
    judge_model_id: str = "google/gemini-2.5-flash-preview",
    use_llm_judge: bool = True,
) -> dict[str, Any]:
    """
    Compute the Constraint Adherence Rate (CAR).

    Args:
        code:           Generated Manim Python code.
        problem:        Problem dict from ManiBench dataset.
                        Reads ``problem["constraints"]`` and
                        ``problem["prompt"]`` / ``problem["task"]``.
        client:         Evaluation API client (OpenRouterClient, etc.).
                        Required when use_llm_judge=True.
        judge_model_id: Model to use as judge.
        use_llm_judge:  If False, fall back to deterministic rule-based checks.

    Returns:
        See module docstring for full schema.
    """
    # Extract constraints from problem spec
    constraints: list[str] = []
    raw_constraints = problem.get("constraints", [])
    if isinstance(raw_constraints, list):
        for c in raw_constraints:
            if isinstance(c, str):
                constraints.append(c)
            elif isinstance(c, dict):
                desc = c.get("description") or c.get("constraint") or c.get("text") or ""
                if desc:
                    constraints.append(desc)
    elif isinstance(raw_constraints, str):
        constraints = [raw_constraints]

    if not constraints:
        return {
            "car_score":        None,
            "method":           "none",
            "constraint_count": 0,
            "details":          [],
            "judge_model":      None,
            "judge_raw":        None,
            "note":             "No constraints defined in problem spec.",
        }

    task_prompt = (
        problem.get("prompt")
        or problem.get("task")
        or problem.get("description")
        or problem.get("title", "")
    )

    # ── LLM Judge path ──────────────────────────────────────────────────────
    if use_llm_judge and client is not None:
        raw_output, evaluations = _call_judge(
            client, judge_model_id, task_prompt, constraints, code
        )

        details = []
        scores = []
        for i, c in enumerate(constraints):
            # Match judge output by position (may have fewer if truncated)
            if i < len(evaluations):
                ev = evaluations[i]
                likert = float(ev.get("score", 3))
                likert = max(1.0, min(5.0, likert))
                norm_score = (likert - 1) / 4.0   # 1→0.0, 5→1.0
                rationale = ev.get("rationale", "")
            else:
                norm_score = 0.5
                rationale  = "Not evaluated by judge"

            scores.append(norm_score)
            details.append({
                "constraint": c,
                "score":      round(norm_score, 4),
                "passed":     norm_score >= 0.5,
                "rationale":  rationale,
            })

        car_score = sum(scores) / len(scores) if scores else 0.0
        return {
            "car_score":        round(car_score, 4),
            "method":           "llm_judge",
            "constraint_count": len(constraints),
            "details":          details,
            "judge_model":      judge_model_id,
            "judge_raw":        raw_output[:3000] if raw_output else None,
        }

    # ── Rule-based fallback ──────────────────────────────────────────────────
    details = []
    scores = []
    for c in constraints:
        passed, evidence = _rule_check(code, c)
        norm_score = 1.0 if passed else 0.0
        scores.append(norm_score)
        details.append({
            "constraint": c,
            "score":      norm_score,
            "passed":     passed,
            "rationale":  evidence,
        })

    car_score = sum(scores) / len(scores) if scores else 0.0
    return {
        "car_score":        round(car_score, 4),
        "method":           "rule_based",
        "constraint_count": len(constraints),
        "details":          details,
        "judge_model":      None,
        "judge_raw":        None,
    }
