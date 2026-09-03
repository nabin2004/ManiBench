"""
Core Evaluation Helper Module for ManiBench Kaggle Benchmarks Tasks
===================================================================
Provides static analysis and metric assertion utilities for evaluating
LLM-generated Manim CE code in Kaggle Benchmark tasks.
"""

import ast
import re
from typing import Any

# Standard Manim GL vs CE pattern checks
GL_ONLY_PATTERNS = [
    r"from\s+manim_imports_ext\s+import",
    r"from\s+manimlib\s+import",
    r"import\s+manimlib",
    r"ShowCreation\(",
    r"ShowCreationThenDestruction\(",
    r"ShowCreationThenFadeOut\(",
    r"ApplyMethod\(",
    r"TexMobject\(",
    r"TextMobject\(",
    r"CONFIG\s*=\s*\{",
    r"self\.add_sound\(",
    r"get_piecewise_linear_function\(",
    r"get_graph\(",
    r"VGroup\.\*\(",
]


def extract_python_code(response_text: str) -> str:
    """Extract Python code block from LLM markdown response."""
    match = re.search(r"```python\s*(.*?)\s*```", response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\s*(.*?)\s*```", response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response_text.strip()


def check_syntax(code: str) -> dict[str, Any]:
    """Check if Python code is syntactically valid."""
    try:
        ast.parse(code)
        return {"valid": True, "error": None, "error_line": None}
    except SyntaxError as e:
        return {"valid": False, "error": str(e), "error_line": e.lineno}


def check_scene_class(code: str) -> dict[str, Any]:
    """Check if code defines at least one Manim Scene subclass."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {"has_scene": False, "scene_names": [], "scene_count": 0}

    scene_names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                base_name = ""
                if isinstance(base, ast.Name):
                    base_name = base.id
                elif isinstance(base, ast.Attribute):
                    base_name = base.attr

                if base_name in (
                    "Scene", "MovingCameraScene", "ThreeDScene",
                    "ZoomedScene", "VectorScene",
                ):
                    scene_names.append(node.name)

    return {
        "has_scene": len(scene_names) > 0,
        "scene_names": scene_names,
        "scene_count": len(scene_names),
    }


def check_manim_imports(code: str) -> dict[str, Any]:
    """Check that code uses Manim CE imports and no GL imports."""
    has_manim = bool(re.search(r"from\s+manim\s+import|import\s+manim", code))
    has_gl = bool(re.search(
        r"from\s+manim_imports_ext|from\s+manimlib|from\s+manim_gl|import\s+manimlib",
        code,
    ))
    return {
        "has_manim_import": has_manim,
        "has_gl_import": has_gl,
        "is_ce_compliant": has_manim and not has_gl,
    }


def detect_version_conflicts(code: str) -> dict[str, Any]:
    """Scan code for deprecated or GL-only API patterns."""
    conflicts = []
    lines = code.split("\n")

    for pattern_str in GL_ONLY_PATTERNS:
        pattern = re.compile(pattern_str, re.MULTILINE)
        for i, line in enumerate(lines, 1):
            match = pattern.search(line)
            if match:
                conflicts.append({
                    "pattern": pattern_str,
                    "line": i,
                    "match": match.group(0).strip()[:80],
                })

    return {
        "conflicts_found": len(conflicts),
        "details": conflicts,
        "has_conflicts": len(conflicts) > 0,
    }


def evaluate_manibench_code(code: str) -> dict[str, Any]:
    """
    Perform complete static evaluation of generated Manim code.
    Returns composite score (0.0 - 1.0) and breakdown metrics.
    """
    syntax_res = check_syntax(code)
    scene_res = check_scene_class(code)
    import_res = check_manim_imports(code)
    conflict_res = detect_version_conflicts(code)

    syntax_score = 1.0 if syntax_res["valid"] else 0.0
    scene_score = 1.0 if scene_res["has_scene"] else 0.0
    import_score = 1.0 if import_res["is_ce_compliant"] else (0.5 if import_res["has_manim_import"] else 0.0)
    conflict_score = 1.0 if not conflict_res["has_conflicts"] else max(0.0, 1.0 - (conflict_res["conflicts_found"] * 0.2))

    composite_score = (syntax_score * 0.35) + (scene_score * 0.25) + (import_score * 0.20) + (conflict_score * 0.20)

    return {
        "composite_score": round(composite_score, 4),
        "syntax_valid": syntax_res["valid"],
        "syntax_error": syntax_res["error"],
        "has_scene": scene_res["has_scene"],
        "scene_names": scene_res["scene_names"],
        "is_ce_compliant": import_res["is_ce_compliant"],
        "conflicts_found": conflict_res["conflicts_found"],
        "code_length": len(code),
    }
