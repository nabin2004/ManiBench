#!/usr/bin/env python3
"""
ManiBench - Local self-test for the one-command Kaggle runner
=============================================================

Runs on any machine (CPU only, no GPU, no model weights, no network needed):

  python scripts/selftest_kaggle_run.py

Checks performed
----------------
  1. The dependency probe reports a healthy local stack.
  2. The probe *detects* the Kaggle NumPy corruption it was written for
     ("cannot import name '_center' from 'numpy._core.umath'") - simulated with a
     `sitecustomize.py` shim that removes the symbol before the probe runs.
  3. The pip constraints file pins NumPy/PyTorch to the installed versions.
  4. The whole evaluation pipeline (dry-run engine -> metrics -> LaTeX/CSV/figures
     -> zip bundle -> manifest) completes and produces non-empty artifacts, and the
     checkpoint makes a second run resume instead of recomputing.
  5. The Kaggle launcher notebook is byte-identical to its generator and all of its
     code cells compile.

Exit code 0 means everything passed.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PASS = "  [PASS]"
FAIL = "  [FAIL]"
_results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{PASS if ok else FAIL} {name}" + (f" - {detail}" if detail else ""), flush=True)
    _results.append((name, ok, detail))


# ═══════════════════════════════════════════════════════════════════════════

def test_probe_healthy() -> None:
    print("\n[1/9] Dependency probe on the current interpreter")
    from scripts import kaggle_run as kr

    probe = kr.probe_stack()
    check("stack probe succeeds", probe["ok"],
          "" if probe["ok"] else "; ".join(probe["errors"])[:200])
    check("probe reports library versions", bool(probe["versions"]),
          ", ".join(f"{k}={v}" for k, v in list(probe["versions"].items())[:5]))


def test_probe_detects_numpy_corruption() -> None:
    print("\n[2/9] Probe detects the 'numpy._core.umath._center' corruption")
    from scripts import kaggle_run as kr

    shim_dir = Path(tempfile.mkdtemp(prefix="manibench_shim_"))
    try:
        (shim_dir / "sitecustomize.py").write_text(
            "try:\n"
            "    import numpy._core.umath as _umath\n"
            "    del _umath._center          # simulate a mixed / half-upgraded numpy\n"
            "except Exception:\n"
            "    pass\n",
            encoding="utf-8",
        )
        env = dict(os.environ, PYTHONPATH=str(shim_dir) + os.pathsep + str(ROOT))
        result = subprocess.run([sys.executable, "-c", kr.PROBE_SRC],
                                capture_output=True, text=True, env=env, timeout=900)
        report = {}
        for line in (result.stdout or "").splitlines():
            if line.startswith(kr.PROBE_MARKER):
                report = json.loads(line[len(kr.PROBE_MARKER):])
                break
        detected = (not report.get("ok", True)) and any(
            "_center" in err or "numpy" in err for err in report.get("errors", [])
        )
        check("corrupted numpy is detected (not silently accepted)", detected,
              "; ".join(report.get("errors", []))[:200])
    finally:
        shutil.rmtree(shim_dir, ignore_errors=True)


def test_constraints_pin() -> None:
    print("\n[3/9] pip constraints file pins the pre-installed stack")
    from scripts import kaggle_run as kr

    target = Path(tempfile.mkdtemp(prefix="manibench_constraints_")) / "constraints.txt"
    try:
        kr.write_constraints(target)
        text = target.read_text(encoding="utf-8")
        numpy_pin = kr.pkg_version("numpy")
        torch_pin = kr.pkg_version("torch")
        check("constraints file created", target.exists())
        if numpy_pin:
            check("numpy pinned to the installed version", f"numpy=={numpy_pin}" in text, numpy_pin)
        if torch_pin:
            check("torch pinned to the installed version", f"torch=={torch_pin}" in text, torch_pin)
    finally:
        shutil.rmtree(target.parent, ignore_errors=True)


def test_end_to_end_pipeline() -> None:
    print("\n[4/9] End-to-end dry-run pipeline (metrics -> tables -> figures -> zip)")
    from scripts.kaggle_run import generate_figures
    from scripts.run_kaggle_benchmark import run_benchmark
    from scripts.model_catalog import catalog

    workdir = Path(tempfile.mkdtemp(prefix="manibench_e2e_"))
    try:
        problems = json.loads((ROOT / "ManiBench_Pilot_Dataset.json").read_text(encoding="utf-8"))["problems"]
        problems = [p for p in problems if p["id"] in {"MB-001", "MB-005"}]
        started = time.time()
        result = run_benchmark(
            models=[catalog.get("Qwen3-8B-Base")],
            problems=problems,
            output_dir=workdir,
            trials=1,
            strategy="zero_shot",
            skip_render=True,
            dry_run=True,
            compute_visual_sim=False,
            resume=False,
        )
        elapsed = time.time() - started

        records = result["records"]
        check("records produced", len(records) == 2, f"{len(records)} records in {elapsed:.1f}s")
        check("every record carries the full metric key set",
              all({"executability", "vcer", "alignment_score", "coverage_score",
                   "mas", "cmi", "car", "visual_similarity"} <= set(r) for r in records))
        check("checkpoint file written", (workdir / "checkpoint_records.jsonl").exists())

        exported = result["exported_files"]
        check("LaTeX tables exported", len([k for k in exported if k.startswith("latex_table")]) == 4)
        csv_summary = exported["csv_summary"]
        check("summary CSV is non-empty",
              csv_summary.exists() and csv_summary.stat().st_size > 100,
              f"{csv_summary.stat().st_size if csv_summary.exists() else 0} bytes")
        check("leaderboard mentions the model",
              "Qwen3-8B-Base" in exported["markdown_leaderboard"].read_text(encoding="utf-8"))

        figures = generate_figures(result["model_summaries"], workdir)
        check("publication figures generated", len(figures) >= 4, f"{len(figures)} files")

        zip_path = Path(shutil.make_archive(str(workdir / "bundle"), "zip", str(workdir)))
        check("submission bundle created", zip_path.exists() and zip_path.stat().st_size > 1000,
              f"{zip_path.stat().st_size / 1024:.1f} KB")

        # Resume semantics: a second identical run must skip all work.
        result2 = run_benchmark(
            models=[catalog.get("Qwen3-8B-Base")],
            problems=problems,
            output_dir=workdir,
            trials=1,
            strategy="zero_shot",
            skip_render=True,
            dry_run=True,
            resume=True,
        )
        check("resume reuses the checkpoint", result2["completed_evaluations"] == 0
              and len(result2["records"]) == 2)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def test_notebook_is_current() -> None:
    print("\n[5/9] Kaggle launcher notebook matches its generator")
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / "build_kaggle_notebook.py"), "--check"],
                            capture_output=True, text=True, timeout=120)
    check("notebook is in sync with scripts/build_kaggle_notebook.py", result.returncode == 0,
          (result.stdout or result.stderr or "").strip()[:120])

    notebook = json.loads((ROOT / "ManiBench_Kaggle_T4x2_Benchmark.ipynb").read_text(encoding="utf-8"))
    code_cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    broken = []
    for index, cell in enumerate(code_cells):
        try:
            compile("".join(cell["source"]), f"<cell {index}>", "exec")
        except SyntaxError as exc:
            broken.append(f"cell {index}: {exc}")
    check(f"all {len(code_cells)} notebook code cells compile", not broken, "; ".join(broken)[:150])
    check("notebook keeps GPU + internet Kaggle metadata",
          bool(notebook["metadata"].get("kaggle", {}).get("isGpuEnabled"))
          and bool(notebook["metadata"].get("kaggle", {}).get("isInternetEnabled")))


def test_trio_script() -> None:
    print("\n[6/9] Trio script: model wiring and publication suite")
    from scripts.run_model_trio import TRIO, resolve_trio  # noqa: F401

    specs = resolve_trio()
    by_short = {s.short_name: s for s in specs}
    check("trio resolves 3 model variants", len(specs) == 3,
          ", ".join(s.short_name for s in specs))
    base = by_short.get("Qwen3-8B-Base")
    check("baseline is the Qwen3-8B foundation model",
          base is not None and base.id == "Qwen/Qwen3-8B" and base.format == "base")
    sft = by_short.get("AOS-Qwen3-8B-Merged")
    check("SFT variant is the merged repo on the Qwen3-8B base",
          sft is not None and sft.id == "nabin2004/AOS-Qwen3-8B-Merged"
          and sft.format == "merged" and sft.base_model == "Qwen/Qwen3-8B"
          and sft.training_method == "SFT")
    grpo = by_short.get("qwen-Manimator-1-grpo")
    check("GRPO variant is the LoRA adapter on the Qwen3-8B base",
          grpo is not None and grpo.id == "nabin2004/qwen-Manimator-1-grpo"
          and grpo.is_lora and grpo.base_model == "Qwen/Qwen3-8B")

    result = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_model_trio.py"), "--help"],
                            capture_output=True, text=True, timeout=120)
    check("trio CLI responds to --help", result.returncode == 0
          and "--from-json" in (result.stdout or ""))

    from scripts.publication_figures import DEFAULT_STEMS
    check("publication figure suite defines 9 figures", len(DEFAULT_STEMS) == 9,
          f"{len(DEFAULT_STEMS)} stems: {', '.join(DEFAULT_STEMS)}")


def test_quantization() -> None:
    print("\n[7/9] 4-bit quantization: readiness, strict enforcement, provenance")
    import torch

    from scripts.model_catalog import ModelSpec
    from scripts.run_kaggle_benchmark import KaggleInferenceEngine
    from scripts.run_model_trio import build_parser

    status = KaggleInferenceEngine.quantization_ready()
    check("readiness probe reports each dependency",
          {"ready", "cuda", "accelerate", "bitsandbytes", "reasons"} <= set(status)
          and isinstance(status["ready"], bool),
          f"ready={status['ready']} ({'; '.join(status['reasons']) or 'all present'})")

    def bare_engine(load_in_4bit: bool, strict: bool):
        engine = object.__new__(KaggleInferenceEngine)   # skip __init__/_load (no weights)
        engine.spec = ModelSpec(id="org/tiny", short_name="tiny", family="baseline", format="base")
        engine.load_in_4bit = load_in_4bit
        engine.strict_quantization = strict
        return engine

    raised = False
    try:
        bare_engine(True, True)._resolve_quantization(torch, has_accelerate=False)
    except RuntimeError as exc:
        raised = "4-bit" in str(exc) and "--precision fp16" in str(exc)
    check("strict mode refuses to silently downgrade to FP16", raised)

    use_4bit, error = bare_engine(True, False)._resolve_quantization(torch, has_accelerate=False)
    check("fallback mode reports why 4-bit was skipped", use_4bit is False and error is not None,
          (error or {}).get("reason", "")[:60])

    use_4bit, error = bare_engine(False, True)._resolve_quantization(torch, has_accelerate=False)
    check("fp16 request never enters the quantization path", use_4bit is False and error is None)

    args = build_parser().parse_args([])
    check("trio defaults to unquantized fp16 for every variant",
          args.precision == "fp16" and args.allow_quant_fallback is False,
          f"precision={args.precision}")

    from scripts.publication_figures import precision_label
    check("precision is rendered for figures/reports",
          precision_label("nf4-4bit (all models)") == "4-bit NF4 (all models)",
          precision_label("nf4-4bit (all models)"))


def test_metric_key_mapping() -> None:
    """
    Regression guard: every metric module must expose one of the keys the runner
    reads. Three metrics previously reported constants for every run (MAS 0.0,
    CMI 70.0, CAR 0.80) because the runner looked up keys the modules never return.
    """
    print("\n[8/9] Metric key mapping (no silently fabricated defaults)")
    import warnings

    warnings.filterwarnings("ignore")

    from evaluation.metrics import (
        compute_alignment,
        compute_code_quality,
        compute_constraint_adherence,
        compute_coverage,
        compute_executability,
        compute_mathematical_fidelity,
        detect_version_conflicts,
    )
    from scripts.run_kaggle_benchmark import METRIC_RESULT_KEYS, metric_value

    problems = json.loads((ROOT / "ManiBench_Pilot_Dataset.json").read_text(encoding="utf-8"))["problems"]
    code = (
        "from manim import *\n\n"
        "class Demo(Scene):\n"
        "    def construct(self):\n"
        "        t = MathTex(r'\\pi = 4\\sum_{k=0}^{\\infty} \\frac{(-1)^k}{2k+1}')\n"
        "        axes = Axes(x_range=[-3, 3], y_range=[-1, 1])\n"
        "        self.play(Write(t), Create(axes))\n"
        "        self.play(axes.plot(lambda x: x ** 2, color=BLUE))\n"
        "        self.wait(1)\n"
    )
    calls = {
        "executability": lambda p: compute_executability(code, skip_render=True),
        "version_conflicts": lambda p: detect_version_conflicts(code),
        "alignment": lambda p: compute_alignment(code, p.get("required_visual_events", [])),
        "coverage": lambda p: compute_coverage(code, p.get("coverage_requirements", [])),
        "mathematical_fidelity": lambda p: compute_mathematical_fidelity(
            code, p.get("ground_truth_equations") or []),
        "code_quality": lambda p: compute_code_quality(code),
        "constraint_adherence": lambda p: compute_constraint_adherence(code, p),
    }
    unresolved: list[str] = []
    seen: set[str] = set()
    for problem in problems:
        for metric, fn in calls.items():
            try:
                result = fn(problem)
            except Exception as exc:  # noqa: BLE001
                unresolved.append(f"{problem['id']}/{metric}: {type(exc).__name__}")
                continue
            if not isinstance(result, dict):
                continue
            matched = [k for k in METRIC_RESULT_KEYS[metric] if result.get(k) is not None]
            if matched:
                seen.add(f"{metric}={matched[0]}")
            elif metric != "constraint_adherence":
                # constraint_adherence honestly returns None for unconstrained tasks
                unresolved.append(f"{problem['id']}/{metric}: no accepted key "
                                  f"(has {sorted(result.keys())[:4]})")

    check("all metrics resolve to a real key on every task", not unresolved,
          "; ".join(sorted(set(unresolved))[:3]))
    check("every metric key is exercised", len(seen) >= len(calls),
          ", ".join(sorted(seen))[:160])
    check("metric_value() falls back only when no accepted key is present",
          metric_value({"cmi_score": 0.42}, "code_quality", None) == 0.42
          and metric_value({}, "code_quality", None) is None
          and metric_value({"cmi_score": None}, "code_quality", None) is None)
    check("CAR scores real constraints from the dataset schema",
          any("constraint_adherence=car_score" == entry for entry in seen))


def test_peft_probe_shim() -> None:
    """
    Reproduce the Kaggle failure that cost a 2-hour run and assert the fix.

    A stale torchao (0.10.0) makes PEFT's `is_torchao_available()` *raise* while the
    LoRA dispatcher probes it, so attaching any adapter dies with
    "ImportError: Found an incompatible version of torchao". A stub `peft` package
    reproduces that exact code path without touching the real environment.
    """
    print("\n[9/9] PEFT optional-backend probe shim (stale torchao)")
    tmp = Path(tempfile.mkdtemp(prefix="fake_peft_"))
    try:
        (tmp / "peft" / "tuners" / "lora").mkdir(parents=True)
        (tmp / "peft" / "__init__.py").write_text("__version__ = '9.9.9-fake'\n", encoding="utf-8")
        (tmp / "peft" / "import_utils.py").write_text(
            "def is_torchao_available():\n"
            "    raise ImportError(\n"
            "        'Found an incompatible version of torchao. Found version 0.10.0, '\n"
            "        'but only versions above 0.16.0 are supported')\n",
            encoding="utf-8")
        (tmp / "peft" / "tuners" / "__init__.py").write_text("", encoding="utf-8")
        (tmp / "peft" / "tuners" / "lora" / "__init__.py").write_text("", encoding="utf-8")
        (tmp / "peft" / "tuners" / "lora" / "torchao.py").write_text(
            "from peft.import_utils import is_torchao_available\n\n\n"
            "def dispatch_torchao():\n"
            "    if not is_torchao_available():\n"
            "        return 'skipped'\n"
            "    return 'used'\n",
            encoding="utf-8")

        probe = (
            "from scripts.run_kaggle_benchmark import (\n"
            "    neutralise_incompatible_peft_probes, validate_adapter_runtime)\n"
            "from peft.tuners.lora.torchao import dispatch_torchao\n"
            "before = 'none'\n"
            "try:\n"
            "    dispatch_torchao()\n"
            "except ImportError as exc:\n"
            "    before = str(exc)[:48]\n"
            "print('BEFORE:' + before)\n"
            "neutralised = neutralise_incompatible_peft_probes()\n"
            "print('NEUTRALISED:' + ','.join(neutralised))\n"
            "print('AFTER:' + dispatch_torchao())\n"
            "print('VALIDATE_OK:' + str(validate_adapter_runtime()['ok']))\n"
        )
        env = dict(os.environ, PYTHONPATH=str(tmp) + os.pathsep + str(ROOT))
        result = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                                text=True, env=env, cwd=str(ROOT), timeout=300)
        out = result.stdout or ""
        check("reproduces the torchao ImportError without the shim",
              "BEFORE:Found an incompatible version of torchao" in out, out[:70])
        check("shim neutralises the incompatible probe",
              any(line.startswith("NEUTRALISED:") and "is_torchao_available" in line
                  for line in out.splitlines()))
        check("LoRA dispatch proceeds after the shim", "AFTER:skipped" in out)
        check("validate_adapter_runtime() reports the adapter path usable",
              "VALIDATE_OK:True" in out, (result.stderr or "")[-120:])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    print("=" * 78)
    print("  ManiBench - kaggle_run.py self-test")
    print("=" * 78)
    for test in (test_probe_healthy, test_probe_detects_numpy_corruption,
                 test_constraints_pin, test_end_to_end_pipeline, test_notebook_is_current,
                 test_trio_script, test_quantization, test_metric_key_mapping,
                 test_peft_probe_shim):
        try:
            test()
        except Exception as exc:  # noqa: BLE001
            import traceback
            check(test.__name__, False, f"{type(exc).__name__}: {exc}")
            traceback.print_exc()

    failed = [name for name, ok, _ in _results if not ok]
    print("\n" + "=" * 78)
    print(f"  {len(_results) - len(failed)}/{len(_results)} checks passed")
    print("=" * 78)
    if failed:
        for name in failed:
            print(f"  FAILED: {name}")
        return 1
    print("  All good: the Kaggle runner is ready.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
