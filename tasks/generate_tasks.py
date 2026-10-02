#!/usr/bin/env python3
"""
Generate Kaggle Benchmarks Task Files for ManiBench
====================================================
Reads ManiBench_Pilot_Dataset.json and outputs self-contained Jupytext percent format
task files for each problem into tasks/. Evaluates all metrics from
ManiBench_Evaluation_Rubric.md:
  1. Executability (Binary: 0 or 1)
  2. Version-Conflict Error Rate (VCER: 0.0 - 1.0)
  3. Alignment Score (0.0 - 1.0)
  4. Coverage Score (0.0 - 1.0)
  5. Visual Similarity (DINOv2 + DTW: 0.0 - 1.0, with Graceful Fallback)
"""

import json
import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
DATASET_PATH = ROOT_DIR / "ManiBench_Pilot_Dataset.json"
TASKS_DIR = ROOT_DIR / "tasks"

SYSTEM_PROMPT = """You are an expert Manim CE (Community Edition) developer.
Write complete, executable Python code using Manim CE to animate the requested mathematical concept.
Ensure:
1. Use `from manim import *` (Do NOT use deprecated ManimGL / manimlib imports like TexMobject, ShowCreation, or CONFIG).
2. Define a subclass of `Scene` (or MovingCameraScene/ThreeDScene).
3. Enclose code in ```python ... ``` blocks.
"""

SLUG_OVERRIDES = {
    "MB-001": "manibench-mb-001-colliding-blocks-compute-pi",
    "MB-002": "manibench-mb-002-gradient-descent-how-neural-networks-learn",
    "MB-003": "manibench-mb-003-but-what-is-a-convolution",
    "MB-004": "manibench-mb-004-eigenvectors-eigenvalues-chapter-14",
    "MB-005": "manibench-mb-005-the-determinant-chapter-6",
    "MB-006": "manibench-006-central-limit-theorem",
    "MB-007": "manibench-007-the-medical-test-paradox",
    "MB-008": "manibench-008-visualizing-the-chain-rule",
    "MB-009": "manibench-009-integration-and-ftc",
    "MB-010": "manibench-010-taylor-series",
    "MB-011": "manibench-011-the-hairy-ball-theorem",
    "MB-012": "manibench-mb-012-the-unexpectedly-hard-windmill-problem",
}

def slugify(text: str) -> str:
    text = text.lower().replace("π", "pi")
    text = re.sub(r"[^\w\s-]", "", text)
    return re.sub(r"[-\s]+", "-", text).strip("-")

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    TASKS_DIR.mkdir(exist_ok=True)

    # Read helper code directly from manibench_eval_core.py to ensure 100% consistency
    core_path = TASKS_DIR / "manibench_eval_core.py"
    with open(core_path, "r", encoding="utf-8") as cf:
        helper_code = cf.read()

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    problems = data.get("problems", [])
    print(f"Generating Kaggle Benchmark tasks for {len(problems)} problems...")

    task_list = []

    for prob in problems:
        prob_id = prob["id"]
        prob_id_clean = prob_id.lower().replace("_", "-")
        title_slug = slugify(prob["title"])
        task_slug = SLUG_OVERRIDES.get(prob_id, f"manibench-{prob_id_clean}-{title_slug}")

        filename = f"{prob_id_clean.replace('-', '_')}_{title_slug.replace('-', '_')}.py"
        filepath = TASKS_DIR / filename
        func_name = f"run_{prob_id_clean.replace('-', '_')}"

        prompt_escaped = prob["full_prompt"].replace('"""', '\\"\\"\\"')

        req_visual_events = repr(prob.get("required_visual_events", []))
        known_incompatibilities = repr(prob.get("version_conflict_notes", {}).get("known_incompatibilities", []))
        cov_requirements = repr(prob.get("coverage_requirements", []))

        code_content = f'''# %%
import kaggle_benchmarks as kbench

# %%
{helper_code}

# %%
SYSTEM_PROMPT = """{SYSTEM_PROMPT.strip()}"""

PROMPT_{prob_id_clean.replace('-', '_').upper()} = """{prompt_escaped.strip()}"""

REQUIRED_VISUAL_EVENTS_{prob_id_clean.replace('-', '_').upper()} = {req_visual_events}

KNOWN_INCOMPATIBILITIES_{prob_id_clean.replace('-', '_').upper()} = {known_incompatibilities}

COVERAGE_REQUIREMENTS_{prob_id_clean.replace('-', '_').upper()} = {cov_requirements}

# %%
@kbench.task(name="{task_slug}")
def {func_name}(llm) -> dict:
    """ManiBench Problem {prob['id']}: {prob['title']}"""
    full_prompt = f"{{SYSTEM_PROMPT}}\\n\\nProblem: {prob['title']}\\n{{PROMPT_{prob_id_clean.replace('-', '_').upper()}}}"
    response = llm.prompt(full_prompt, temperature=0.2)
    code = extract_python_code(response)

    ref_video = "media/references/{prob_id_clean}_ref.mp4"

    metrics = evaluate_manibench_submission(
        code=code,
        required_visual_events=REQUIRED_VISUAL_EVENTS_{prob_id_clean.replace('-', '_').upper()},
        known_incompatibilities=KNOWN_INCOMPATIBILITIES_{prob_id_clean.replace('-', '_').upper()},
        coverage_requirements=COVERAGE_REQUIREMENTS_{prob_id_clean.replace('-', '_').upper()},
        ref_video_path=ref_video,
    )

    # Metric 1: Executability (Binary: 0 or 1)
    kbench.assertions.assert_true(
        metrics["executability"] == 1,
        expectation=f"Metric 1 - Executability: Code runs without errors (Got: {{metrics['executability']}})",
    )

    # Metric 2: Version-Conflict Error Rate (VCER: 0.0 - 1.0)
    kbench.assertions.assert_true(
        metrics["vcer"] == 0.0,
        expectation=f"Metric 2 - Version-Conflict Error Rate: No deprecated or ManimGL APIs (VCER: {{metrics['vcer']:.1%}}, conflicts: {{metrics['conflicts_found']}})",
    )

    # Metric 3: Alignment Score (0.0 - 1.0)
    kbench.assertions.assert_true(
        metrics["alignment_score"] >= 0.70,
        expectation=f"Metric 3 - Alignment Score: Required visual events present (Score: {{metrics['alignment_score']:.2f}} >= 0.70)",
    )

    # Metric 4: Coverage Score (0.0 - 1.0)
    kbench.assertions.assert_true(
        metrics["coverage_score"] >= 0.70,
        expectation=f"Metric 4 - Coverage Score: Pedagogical elements and annotations (Score: {{metrics['coverage_score']:.2f}} >= 0.70)",
    )

    if metrics.get("visual_similarity") is not None:
        kbench.assertions.assert_true(
            metrics["visual_similarity"] >= 0.70,
            expectation=f"Metric 5 - Visual Similarity (DINOv2+DTW): Frame alignment score (Score: {{metrics['visual_similarity']:.2f}} >= 0.70)",
        )

    # Concise dictionary format for clean display in Kaggle Benchmarks main UI
    return {{
        "Exec": metrics["executability"],
        "VCER": metrics["vcer"],
        "Align": metrics["alignment_score"],
        "Cover": metrics["coverage_score"],
        "VisSim": metrics["visual_similarity"],
    }}

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
