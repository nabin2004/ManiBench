# %%
import kaggle_benchmarks as kbench

# %%
"""
Core Evaluation Helper Module for ManiBench Kaggle Benchmarks Tasks
===================================================================
Implements the 5 evaluation metrics:
  1. Executability (Binary: 0 or 1)
  2. Version-Conflict Error Rate (VCER: 0.0 - 1.0)
  3. Alignment Score (0.0 - 1.0)
  4. Coverage Score (0.0 - 1.0)
  5. Visual Similarity (DINOv2 + DTW: 0.0 - 1.0, with Graceful Fallback to null)
"""

import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

# Standard Manim GL vs CE pattern checks
GL_ONLY_PATTERNS = [
    r"from\s+manim_imports_ext\s+import",
    r"from\s+manimlib\s+import",
    r"import\s+manimlib",
    r"from\s+manim_gl\s+import",
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
    r"InteractiveScene",
    r"GraphScene",
    r"ReconfigurableScene",
    r"FadeInFrom\(",
    r"OldTex",
    r"self\.frame\.reorient\(",
]

_ANIMATION_KEYWORDS = {
    "create": ["Create", "Write", "DrawBorderThenFill", "FadeIn", "GrowFromCenter"],
    "transform": ["Transform", "ReplacementTransform", "TransformFromCopy", "MoveToTarget", "LaggedStart", "Succession"],
    "fade": ["FadeIn", "FadeOut", "FadeTransform"],
    "indicate": ["Indicate", "Flash", "Circumscribe", "ShowPassingFlash", "Wiggle"],
    "move": ["shift", "move_to", "next_to", "to_edge", "to_corner", "animate"],
    "rotate": ["Rotate", "rotate", "set_angle"],
    "scale": ["scale", "Scale", "ScaleInPlace"],
    "color": ["set_color", "set_fill", "set_stroke", "color", "Color", "YELLOW", "RED", "BLUE", "GREEN", "WHITE"],
    "label": ["Text", "Tex", "MathTex", "DecimalNumber", "Integer", "label", "title"],
    "graph": ["Axes", "NumberPlane", "CoordinateSystem", "plot"],
    "group": ["VGroup", "Group"],
    "3d": ["ThreeDScene", "ThreeDAxes", "Surface", "ParametricSurface"],
    "updater": ["add_updater", "always_redraw", "ValueTracker"],
    "wait": ["wait", "Wait"],
    "shape": ["Circle", "Square", "Rectangle", "Triangle", "Polygon", "Dot", "Line", "Arc", "Ellipse", "Vector", "Arrow"],
}

_DIM_PATTERNS = {
    "math_annotations": [
        (r'\bTex\s*\(', "Tex object"),
        (r'\bMathTex\s*\(', "MathTex object"),
        (r'\bText\s*\(', "Text object"),
        (r'\bTitle\s*\(', "Title object"),
        (r'\\frac|\\int|\\sum|\\prod|\\lim|\\sqrt|\\mathbb|\\mathcal', "LaTeX math"),
    ],
    "visual_mapping": [
        (r'set_color\s*\(', "color setter"),
        (r'set_fill\s*\(', "fill styling"),
        (r'\bArrow\s*\(|\bVector\s*\(', "Arrow indicator"),
        (r'\bDot\s*\(', "Dot marker"),
        (r'always_redraw\s*\(', "dynamic redraw"),
        (r'color\s*=\s*[A-Z_]+', "named color"),
    ],
    "numeric_evidence": [
        (r'\bDecimalNumber\s*\(|\bInteger\s*\(', "number display"),
        (r'\bValueTracker\s*\(', "ValueTracker"),
        (r'\bAxes\s*\(|\bNumberPlane\s*\(', "coordinate system"),
        (r'\.plot\s*\(', "function plot"),
    ],
    "structural_clarity": [
        (r'\bVGroup\s*\(|\bGroup\s*\(', "group organization"),
        (r'\.arrange\s*\(', "arrange layout"),
        (r'self\.wait\s*\(', "paced wait"),
        (r'\bFadeIn\s*\(|\bFadeOut\s*\(', "fade transition"),
        (r'\bLaggedStart\s*\(|\bSuccession\s*\(', "sequencing"),
    ],
}

_DIM_WEIGHTS = {
    "math_annotations": 0.35,
    "visual_mapping": 0.30,
    "numeric_evidence": 0.20,
    "structural_clarity": 0.15,
}


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
    try:
        ast.parse(code)
        return {"valid": True, "error": None}
    except SyntaxError as e:
        return {"valid": False, "error": f"{type(e).__name__}: {e}"}


def check_scene_class(code: str) -> dict[str, Any]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {"has_scene": False, "scene_names": [], "has_construct": False}

    scene_names = []
    has_construct = False
    valid_bases = {"Scene", "MovingCameraScene", "ThreeDScene", "ZoomedScene", "VectorScene"}

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            is_scene = False
            for base in node.bases:
                base_name = ""
                if isinstance(base, ast.Name):
                    base_name = base.id
                elif isinstance(base, ast.Attribute):
                    base_name = base.attr
                if base_name in valid_bases:
                    is_scene = True
                    scene_names.append(node.name)
                    break
            if is_scene:
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "construct":
                        has_construct = True

    return {
        "has_scene": len(scene_names) > 0,
        "scene_names": scene_names,
        "has_construct": has_construct,
    }


def check_manim_imports(code: str) -> dict[str, Any]:
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


def compute_executability(code: str, timeout: int = 30) -> dict[str, Any]:
    """
    Metric 1: Executability (Pass@1)
    Binary 1 or 0: Code must parse, have a Scene subclass with construct(),
    use CE imports, and render cleanly if manim CLI is available.
    """
    syntax = check_syntax(code)
    if not syntax["valid"]:
        return {"executability": 0, "error_type": "SyntaxError", "error_message": syntax["error"]}

    scene = check_scene_class(code)
    if not scene["has_scene"]:
        return {"executability": 0, "error_type": "NoSceneClass", "error_message": "No Manim Scene subclass defined"}

    imports = check_manim_imports(code)
    if not imports["has_manim_import"]:
        return {"executability": 0, "error_type": "MissingManimImport", "error_message": "No 'from manim import *' found"}

    if imports["has_gl_import"]:
        return {"executability": 0, "error_type": "ManimGLImportError", "error_message": "ManimGL / manimlib import used in CE code"}

    manim_bin = shutil.which("manim")
    if manim_bin and scene["scene_names"]:
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
                f.write(code)
                temp_file = f.name

            scene_name = scene["scene_names"][0]
            cmd = [manim_bin, "-ql", "-s", "--media_dir", tempfile.gettempdir(), temp_file, scene_name]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            if res.returncode != 0:
                err_line = res.stderr.strip().splitlines()[-1] if res.stderr.strip() else "Render exit code non-zero"
                return {"executability": 0, "error_type": "RenderError", "error_message": err_line[:200]}
        except subprocess.TimeoutExpired:
            return {"executability": 0, "error_type": "TimeoutError", "error_message": f"Render timed out after {timeout}s"}
        except Exception:
            pass

    return {"executability": 1, "error_type": None, "error_message": None}


def detect_version_conflicts(code: str, known_incompatibilities: list[str] | None = None) -> dict[str, Any]:
    """
    Metric 2: Version-Conflict Error Rate (VCER)
    Checks for deprecated or GL-only API patterns + problem-specific incompatibilities.
    """
    conflicts = []
    lines = code.split("\n")
    patterns_to_check = list(GL_ONLY_PATTERNS)

    if known_incompatibilities:
        for item in known_incompatibilities:
            left = item.split("→")[0].split("->")[0].strip()
            ident = re.findall(r'[A-Za-z_]\w+', left)
            if ident:
                pat = rf"\b{ident[0]}\b"
                if pat not in patterns_to_check:
                    patterns_to_check.append(pat)

    for pat_str in patterns_to_check:
        try:
            pat = re.compile(pat_str)
        except re.error:
            continue
        for idx, line in enumerate(lines, start=1):
            if pat.search(line):
                conflicts.append({"line": idx, "pattern": pat_str, "match": line.strip()[:60]})

    conflicting_lines = len(set(c["line"] for c in conflicts))
    code_lines = len([l for l in lines if l.strip() and not l.strip().startswith("#")])
    vcer = round(conflicting_lines / max(code_lines, 1), 4)

    return {
        "vcer": vcer,
        "conflicts_found": len(conflicts),
        "conflict_details": conflicts,
    }


def compute_alignment(code: str, required_visual_events: list[dict] | None = None) -> dict[str, Any]:
    """
    Metric 3: Alignment Score (0.0 - 1.0)
    Weighted fraction of required visual events detected in code.
    """
    if not required_visual_events:
        return {"alignment_score": 1.0, "events_detected": 0, "events_total": 0, "per_event": []}

    code_lower = code.lower()
    per_event = []
    total_weight = 0.0
    weighted_detected = 0.0

    for ev in required_visual_events:
        desc = ev.get("description", "")
        weight = float(ev.get("weight", 1.0))
        total_weight += weight

        detected = False
        evidence = ""

        words = re.findall(r'[a-zA-Z]{4,}', desc)
        matches = [w for w in words if w.lower() in code_lower]
        if len(matches) >= 2:
            detected = True
            evidence = f"Keywords: {', '.join(matches[:3])}"
        else:
            for cat, kws in _ANIMATION_KEYWORDS.items():
                if cat in desc.lower():
                    for kw in kws:
                        if kw in code:
                            detected = True
                            evidence = f"Category '{cat}': {kw}"
                            break
                if detected:
                    break

        if detected:
            weighted_detected += weight

        per_event.append({"description": desc[:60], "weight": weight, "detected": detected, "evidence": evidence})

    score = round(weighted_detected / max(total_weight, 1e-9), 4)
    return {
        "alignment_score": score,
        "events_detected": sum(1 for e in per_event if e["detected"]),
        "events_total": len(per_event),
        "per_event": per_event,
    }


def compute_coverage(code: str, coverage_requirements: list[dict] | None = None) -> dict[str, Any]:
    """
    Metric 4: Coverage Score (0.0 - 1.0)
    Density of pedagogical elements across 4 dimensions:
      - Math Annotations (0.35)
      - Visual Mapping (0.30)
      - Numeric Evidence (0.20)
      - Structural Clarity (0.15)
    """
    dim_scores = {}
    for dim_name, patterns in _DIM_PATTERNS.items():
        found = 0
        for pat, _ in patterns:
            if re.search(pat, code):
                found += 1
        dim_scores[dim_name] = min(1.0, found / max(len(patterns) * 0.5, 1))

    total_score = sum(dim_scores[d] * _DIM_WEIGHTS[d] for d in _DIM_WEIGHTS)
    return {
        "coverage_score": round(min(1.0, total_score), 4),
        "dimension_scores": dim_scores,
    }


def compute_visual_similarity_if_available(
    code: str,
    ref_video_path: str | Path | None = None,
    scene_name: str | None = None,
    timeout: int = 60,
    target_fps: int = 4,
) -> dict[str, Any]:
    """
    Metric 5: Visual Embedding Similarity (DINOv2 + DTW) with Graceful Fallback.
    If PyTorch, OpenCV, Transformers, and a reference .mp4 are detected,
    renders the video and computes the DINOv2 similarity.
    If not available (e.g. running on Kaggle without the video dataset),
    reports visual_similarity: None without crashing.
    """
    if not ref_video_path or not Path(ref_video_path).exists():
        return {
            "visual_similarity": None,
            "dtw_distance": None,
            "reason": "Reference video not found in environment",
        }

    try:
        import torch
        from transformers import AutoImageProcessor, AutoModel
        import cv2
        from PIL import Image
        import numpy as np
    except ImportError as e:
        return {
            "visual_similarity": None,
            "dtw_distance": None,
            "reason": f"Vision dependencies not available: {e}",
        }

    manim_bin = shutil.which("manim")
    if not manim_bin:
        return {
            "visual_similarity": None,
            "dtw_distance": None,
            "reason": "Manim executable not available for video rendering",
        }

    try:
        with tempfile.TemporaryDirectory(prefix="manibench_vis_") as tmpdir:
            temp_code = Path(tmpdir) / "candidate_scene.py"
            temp_code.write_text(code, encoding="utf-8")

            if not scene_name:
                scene_info = check_scene_class(code)
                if not scene_info["scene_names"]:
                    return {"visual_similarity": None, "dtw_distance": None, "reason": "No Scene subclass"}
                scene_name = scene_info["scene_names"][0]

            media_dir = Path(tmpdir) / "media"
            cmd = [
                manim_bin, "-ql",
                "--media_dir", str(media_dir),
                str(temp_code),
                scene_name,
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            if res.returncode != 0:
                return {"visual_similarity": None, "dtw_distance": None, "reason": "Candidate render failed"}

            video_files = list(media_dir.rglob("*.mp4"))
            if not video_files:
                return {"visual_similarity": None, "dtw_distance": None, "reason": "Rendered video not found"}

            cand_video_path = video_files[0]
            device = "cuda" if torch.cuda.is_available() else "cpu"
            processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
            model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
            model.eval()

            def extract_video_embs(vid_path: Path) -> np.ndarray:
                cap = cv2.VideoCapture(str(vid_path))
                if not cap.isOpened():
                    return np.empty((0, 768))
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                interval = max(1, int(round(fps / target_fps)))
                embs = []
                idx = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if idx % interval == 0:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil = Image.fromarray(rgb)
                        with torch.no_grad():
                            inputs = processor(images=pil, return_tensors="pt").to(device)
                            out = model(**inputs)
                            cls_emb = out.last_hidden_state[:, 0, :].cpu().numpy().flatten()
                            norm = np.linalg.norm(cls_emb)
                            embs.append(cls_emb / norm if norm > 0 else cls_emb)
                    idx += 1
                cap.release()
                return np.array(embs) if embs else np.empty((0, 768))

            ref_embs = extract_video_embs(Path(ref_video_path))
            cand_embs = extract_video_embs(cand_video_path)

            if len(ref_embs) == 0 or len(cand_embs) == 0:
                return {"visual_similarity": None, "dtw_distance": None, "reason": "Failed to extract frames"}

            cosine_sim = np.dot(ref_embs, cand_embs.T)
            cosine_dist = np.clip(1.0 - cosine_sim, 0.0, 2.0)
            N, M = cosine_dist.shape
            cost = np.zeros((N, M))
            cost[0, 0] = cosine_dist[0, 0]
            for i in range(1, N):
                cost[i, 0] = cost[i - 1, 0] + cosine_dist[i, 0]
            for j in range(1, M):
                cost[0, j] = cost[0, j - 1] + cosine_dist[0, j]
            for i in range(1, N):
                for j in range(1, M):
                    cost[i, j] = cosine_dist[i, j] + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
            dtw_dist = float(cost[-1, -1] / (N + M))
            alignment_score = round(max(0.0, float(np.exp(-1.5 * dtw_dist))), 4)

            return {
                "visual_similarity": alignment_score,
                "dtw_distance": round(dtw_dist, 4),
            }

    except Exception as e:
        return {
            "visual_similarity": None,
            "dtw_distance": None,
            "reason": f"Visual similarity error: {e}",
        }


def evaluate_manibench_submission(
    code: str,
    required_visual_events: list[dict] | None = None,
    known_incompatibilities: list[str] | None = None,
    coverage_requirements: list[dict] | None = None,
    ref_video_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Complete evaluation across all metrics from ManiBench_Evaluation_Rubric.md
    plus optional Visual Similarity (DINOv2 + DTW).
    """
    exec_res = compute_executability(code)
    vcer_res = detect_version_conflicts(code, known_incompatibilities)
    align_res = compute_alignment(code, required_visual_events)
    cov_res = compute_coverage(code, coverage_requirements)
    vis_res = compute_visual_similarity_if_available(code, ref_video_path)

    return {
        "executability": exec_res["executability"],
        "error_type": exec_res["error_type"],
        "error_message": exec_res["error_message"],
        "vcer": vcer_res["vcer"],
        "conflicts_found": vcer_res["conflicts_found"],
        "alignment_score": align_res["alignment_score"],
        "events_detected": align_res["events_detected"],
        "events_total": align_res["events_total"],
        "coverage_score": cov_res["coverage_score"],
        "visual_similarity": vis_res.get("visual_similarity"),
        "dtw_distance": vis_res.get("dtw_distance"),
    }


# %%
SYSTEM_PROMPT = """You are an expert Manim CE (Community Edition) developer.
Write complete, executable Python code using Manim CE to animate the requested mathematical concept.
Ensure:
1. Use `from manim import *` (Do NOT use deprecated ManimGL / manimlib imports like TexMobject, ShowCreation, or CONFIG).
2. Define a subclass of `Scene` (or MovingCameraScene/ThreeDScene).
3. Enclose code in ```python ... ``` blocks."""

PROMPT_MB_002 = """Create a Manim scene animating gradient descent on a 2D loss landscape. Show:
1. A parametric surface z = L(w₁, w₂) representing loss as a function of two parameters
2. A dot starting at a high-loss location
3. At each step: (a) compute gradient ∇L at the dot's position, (b) move dot in direction of -∇L, (c) update a loss curve showing historical loss values
4. Animate 5–10 steps of descent with diminishing step size
5. Show arrows indicating gradient direction
6. Label axes: 'w₁', 'w₂', 'Loss'"""

REQUIRED_VISUAL_EVENTS_MB_002 = [{'id': 'evt_002_surface', 'description': '3D loss surface/landscape visualized as parametric surface z = L(w₁, w₂)', 'weight': 0.8, 'is_critical': True, 'timing': 'scene_start'}, {'id': 'evt_002_initial_dot', 'description': 'Dot positioned at initial high-loss location on the surface', 'weight': 0.8, 'is_critical': True, 'timing': 'after_surface_render'}, {'id': 'evt_002_gradient_arrow', 'description': "Gradient arrow (∇L) shown at dot's position and updates direction/magnitude each step", 'weight': 0.7, 'is_critical': True, 'timing': 'each_descent_step'}, {'id': 'evt_002_downhill', 'description': 'Dot moves downhill along −∇L direction on the surface', 'weight': 0.9, 'is_critical': True, 'timing': 'each_descent_step_after_gradient'}, {'id': 'evt_002_loss_curve', 'description': 'Separate loss curve (Loss vs iteration) plots historical values, updating synchronously with dot movement', 'weight': 0.8, 'is_critical': True, 'timing': 'synchronized_with_downhill'}, {'id': 'evt_002_step_shrink', 'description': 'Step size (learning rate) diminishes visually — shorter gradient arrows or explicit α label decreasing', 'weight': 0.6, 'is_critical': False, 'timing': 'progressive_across_steps'}]

KNOWN_INCOMPATIBILITIES_MB_002 = ['manim_imports_ext → from manim import *', 'OldTex/OldTexText → Tex/MathTex in CE', 'CONFIG dict class pattern → __init__ parameters in CE', 'ContinualEdgeUpdate → custom updater in CE', 'Eyes/PiCreature ecosystem → not available in CE', 'NetworkScene/NetworkMobject → custom implementation needed', 'MNistMobject/PixelsFromVect → custom pixel rendering in CE', 'ExternallyAnimatedScene/TODOStub → not available in CE', 'FRAME_X_RADIUS/FRAME_Y_RADIUS → config.frame_width/2 in CE', 'TeacherStudentsScene → custom implementation needed', 'force_skipping/revert_to_original_skipping_status → not in CE', 'GraphScene API → Axes object methods in CE', 'UnitInterval → NumberLine in CE']

COVERAGE_REQUIREMENTS_MB_002 = ["Axis labels present ('w₁', 'w₂', 'Loss')", 'Gradient arrows visible and pointing in −∇L direction', 'Loss curve displayed and updating synchronously with dot', 'Step count or iteration number displayed', 'Learning rate value or α label shown']

# %%
@kbench.task(name="manibench-mb-002-gradient-descent-how-neural-networks-learn")
def run_mb_002(llm) -> dict:
    """ManiBench Problem MB-002: Gradient Descent, How Neural Networks Learn"""
    full_prompt = f"{SYSTEM_PROMPT}\n\nProblem: Gradient Descent, How Neural Networks Learn\n{PROMPT_MB_002}"
    response = llm.prompt(full_prompt, temperature=0.2)
    code = extract_python_code(response)

    ref_video = "media/references/mb-002_ref.mp4"

    metrics = evaluate_manibench_submission(
        code=code,
        required_visual_events=REQUIRED_VISUAL_EVENTS_MB_002,
        known_incompatibilities=KNOWN_INCOMPATIBILITIES_MB_002,
        coverage_requirements=COVERAGE_REQUIREMENTS_MB_002,
        ref_video_path=ref_video,
    )

    # Metric 1: Executability (Binary: 0 or 1)
    kbench.assertions.assert_true(
        metrics["executability"] == 1,
        expectation=f"Metric 1 - Executability: Code runs without errors (Got: {metrics['executability']})",
    )

    # Metric 2: Version-Conflict Error Rate (VCER: 0.0 - 1.0)
    kbench.assertions.assert_true(
        metrics["vcer"] == 0.0,
        expectation=f"Metric 2 - Version-Conflict Error Rate: No deprecated or ManimGL APIs (VCER: {metrics['vcer']:.1%}, conflicts: {metrics['conflicts_found']})",
    )

    # Metric 3: Alignment Score (0.0 - 1.0)
    kbench.assertions.assert_true(
        metrics["alignment_score"] >= 0.70,
        expectation=f"Metric 3 - Alignment Score: Required visual events present (Score: {metrics['alignment_score']:.2f} >= 0.70)",
    )

    # Metric 4: Coverage Score (0.0 - 1.0)
    kbench.assertions.assert_true(
        metrics["coverage_score"] >= 0.70,
        expectation=f"Metric 4 - Coverage Score: Pedagogical elements and annotations (Score: {metrics['coverage_score']:.2f} >= 0.70)",
    )

    if metrics.get("visual_similarity") is not None:
        kbench.assertions.assert_true(
            metrics["visual_similarity"] >= 0.70,
            expectation=f"Metric 5 - Visual Similarity (DINOv2+DTW): Frame alignment score (Score: {metrics['visual_similarity']:.2f} >= 0.70)",
        )

    # Concise dictionary format for clean display in Kaggle Benchmarks main UI
    return {
        "Exec": metrics["executability"],
        "VCER": metrics["vcer"],
        "Align": metrics["alignment_score"],
        "Cover": metrics["coverage_score"],
        "VisSim": metrics["visual_similarity"],
    }

run_mb_002.run(kbench.llm)
