#!/usr/bin/env python3
"""
Generate Kaggle Benchmarks Task Files for ManiBench
====================================================
Reads ManiBench_Pilot_Dataset.json and outputs self-contained Jupytext percent format
task files for each problem into tasks/.
"""

import json
import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
DATASET_PATH = ROOT_DIR / "ManiBench_Pilot_Dataset.json"
TASKS_DIR = ROOT_DIR / "tasks"

HELPER_CODE = '''import ast
import re
from typing import Any

GL_ONLY_PATTERNS = [
    r"from\\s+manim_imports_ext\\s+import",
    r"from\\s+manimlib\\s+import",
    r"import\\s+manimlib",
    r"ShowCreation\\(",
    r"ShowCreationThenDestruction\\(",
    r"ShowCreationThenFadeOut\\(",
    r"ApplyMethod\\(",
    r"TexMobject\\(",
    r"TextMobject\\(",
    r"CONFIG\\s*=\\s*\\{",
    r"self\\.add_sound\\(",
    r"get_piecewise_linear_function\\(",
    r"get_graph\\(",
    r"VGroup\\.\\*\\(",
]

def extract_python_code(response_text: str) -> str:
    match = re.search(r"```python\\s*(.*?)\\s*```", response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\\s*(.*?)\\s*```", response_text, re.DOTALL)
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

    has_manim = bool(re.search(r"from\\s+manim\\s+import|import\\s+manim", code))
    has_gl = bool(re.search(r"from\\s+manim_imports_ext|from\\s+manimlib|from\\s+manim_gl|import\\s+manimlib", code))
    is_ce_compliant = has_manim and not has_gl

    conflicts = []
    lines = code.split("\\n")
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
'''

SYSTEM_PROMPT = """You are an expert Manim CE (Community Edition) developer.
Write complete, executable Python code using Manim CE to animate the requested mathematical concept.
Ensure:
1. Use `from manim import *` (Do NOT use deprecated ManimGL / manimlib imports like TexMobject, ShowCreation, or CONFIG).
2. Define a subclass of `Scene` (or MovingCameraScene/ThreeDScene).
3. Enclose code in ```python ... ``` blocks.
"""

def slugify(text: str) -> str:
    text = text.lower().replace("π", "pi")
    text = re.sub(r"[^\w\s-]", "", text)
    return re.sub(r"[-\s]+", "-", text).strip("-")

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    TASKS_DIR.mkdir(exist_ok=True)
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    problems = data.get("problems", [])
    print(f"Generating Kaggle Benchmark tasks for {len(problems)} problems...")

    task_list = []

    for prob in problems:
        prob_id = prob["id"].lower().replace("_", "-")
        title_slug = slugify(prob["title"])
        task_slug = f"manibench-{prob_id}-{title_slug}"
        filename = f"{prob_id.replace('-', '_')}_{title_slug.replace('-', '_')}.py"
        filepath = TASKS_DIR / filename
        func_name = f"run_{prob_id.replace('-', '_')}"

        prompt_escaped = prob["full_prompt"].replace('"""', '\\"\\"\\"')

        code_content = f'''# %%
import kaggle_benchmarks as kbench

# %%
{HELPER_CODE}

# %%
SYSTEM_PROMPT = """{SYSTEM_PROMPT.strip()}"""

PROMPT_{prob_id.replace('-', '_').upper()} = """{prompt_escaped.strip()}"""

# %%
@kbench.task(name="{task_slug}")
def {func_name}(llm) -> float:
    """ManiBench Problem {prob['id']}: {prob['title']}"""
    full_prompt = f"{{SYSTEM_PROMPT}}\\n\\nProblem: {prob['title']}\\n{{PROMPT_{prob_id.replace('-', '_').upper()}}}"
    response = llm.prompt(full_prompt, temperature=0.2)
    code = extract_python_code(response)
    metrics = evaluate_manibench_code(code)

    kbench.assertions.assert_true(metrics["syntax_valid"], expectation="Python code must be syntactically valid")
    kbench.assertions.assert_true(metrics["has_scene"], expectation="Code must define a Manim Scene subclass")
    kbench.assertions.assert_true(metrics["is_ce_compliant"], expectation="Code must use Manim CE imports")
    kbench.assertions.assert_true(metrics["conflicts_found"] == 0, expectation="Code must not use deprecated or GL-only APIs")

    return metrics["composite_score"]

{func_name}.run(kbench.llm)
'''
        with open(filepath, "w", encoding="utf-8") as pf:
            pf.write(code_content)

        task_list.append({"slug": task_slug, "file": str(filepath.relative_to(ROOT_DIR))})
        print(f"Generated task file: {filepath.relative_to(ROOT_DIR)} (slug: {task_slug})")

    # Generate master manifest / summary
    manifest_path = TASKS_DIR / "tasks_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as mf:
        json.dump(task_list, mf, indent=2)

    print(f"\nTask generation complete! Manifest saved to {manifest_path.relative_to(ROOT_DIR)}")

if __name__ == "__main__":
    main()
