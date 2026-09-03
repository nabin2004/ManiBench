# %%
import kaggle_benchmarks as kbench

# %%
import ast
import re
from typing import Any

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
    match = re.search(r"```python\s*(.*?)\s*```", response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\s*(.*?)\s*```", response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response_text.strip()

def evaluate_manibench_code(code: str) -> dict[str, Any]:
    syntax_valid = True
    syntax_error = None
    try:
        ast.parse(code)
    except SyntaxError as e:
        syntax_valid = False
        syntax_error = str(e)

    has_scene = False
    scene_names = []
    if syntax_valid:
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    for base in node.bases:
                        bname = base.id if isinstance(base, ast.Name) else (base.attr if isinstance(base, ast.Attribute) else "")
                        if bname in ("Scene", "MovingCameraScene", "ThreeDScene", "ZoomedScene", "VectorScene"):
                            has_scene = True
                            scene_names.append(node.name)
        except SyntaxError:
            pass

    has_manim = bool(re.search(r"from\s+manim\s+import|import\s+manim", code))
    has_gl = bool(re.search(r"from\s+manim_imports_ext|from\s+manimlib|from\s+manim_gl|import\s+manimlib", code))
    is_ce_compliant = has_manim and not has_gl

    conflicts = []
    lines = code.split("\n")
    for pattern_str in GL_ONLY_PATTERNS:
        pattern = re.compile(pattern_str, re.MULTILINE)
        for i, line in enumerate(lines, 1):
            if pattern.search(line):
                conflicts.append(pattern_str)

    syntax_score = 1.0 if syntax_valid else 0.0
    scene_score = 1.0 if has_scene else 0.0
    import_score = 1.0 if is_ce_compliant else (0.5 if has_manim else 0.0)
    conflict_score = 1.0 if len(conflicts) == 0 else max(0.0, 1.0 - (len(conflicts) * 0.2))

    composite_score = (syntax_score * 0.35) + (scene_score * 0.25) + (import_score * 0.20) + (conflict_score * 0.20)

    return {
        "composite_score": round(composite_score, 4),
        "syntax_valid": syntax_valid,
        "syntax_error": syntax_error,
        "has_scene": has_scene,
        "scene_names": scene_names,
        "is_ce_compliant": is_ce_compliant,
        "conflicts_found": len(conflicts),
    }


# %%
SYSTEM_PROMPT = """You are an expert Manim CE (Community Edition) developer.
Write complete, executable Python code using Manim CE to animate the requested mathematical concept.
Ensure:
1. Use `from manim import *` (Do NOT use deprecated ManimGL / manimlib imports like TexMobject, ShowCreation, or CONFIG).
2. Define a subclass of `Scene` (or MovingCameraScene/ThreeDScene).
3. Enclose code in ```python ... ``` blocks."""

PROMPT_MB_003 = """Animate the convolution operation between a signal and a kernel. Show:
1. A 1D signal plotted on a horizontal axis (bar chart or curve)
2. A 1D kernel (filter) displayed as a sliding window
3. The sliding window moves left-to-right along the signal
4. At each position, show the element-wise product (animation or highlight)
5. Show the integral (sum) accumulating in a separate output graph
6. Animate the output graph building up point-by-point
7. Label: 'Signal', 'Kernel', 'Convolution Output'"""

# %%
@kbench.task(name="manibench-mb-003-but-what-is-a-convolution")
def run_mb_003(llm) -> float:
    """ManiBench Problem MB-003: But What Is a Convolution?"""
    full_prompt = f"{SYSTEM_PROMPT}\n\nProblem: But What Is a Convolution?\n{PROMPT_MB_003}"
    response = llm.prompt(full_prompt, temperature=0.2)
    code = extract_python_code(response)
    metrics = evaluate_manibench_code(code)

    kbench.assertions.assert_true(metrics["syntax_valid"], expectation="Python code must be syntactically valid")
    kbench.assertions.assert_true(metrics["has_scene"], expectation="Code must define a Manim Scene subclass")
    kbench.assertions.assert_true(metrics["is_ce_compliant"], expectation="Code must use Manim CE imports")
    kbench.assertions.assert_true(metrics["conflicts_found"] == 0, expectation="Code must not use deprecated or GL-only APIs")

    return metrics["composite_score"]

run_mb_003.run(kbench.llm)
