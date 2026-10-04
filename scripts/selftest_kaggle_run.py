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
    print("\n[1/5] Dependency probe on the current interpreter")
    from scripts import kaggle_run as kr

    probe = kr.probe_stack()
    check("stack probe succeeds", probe["ok"],
          "" if probe["ok"] else "; ".join(probe["errors"])[:200])
    check("probe reports library versions", bool(probe["versions"]),
          ", ".join(f"{k}={v}" for k, v in list(probe["versions"].items())[:5]))


def test_probe_detects_numpy_corruption() -> None:
    print("\n[2/5] Probe detects the 'numpy._core.umath._center' corruption")
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
    print("\n[3/5] pip constraints file pins the pre-installed stack")
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
    print("\n[4/5] End-to-end dry-run pipeline (metrics -> tables -> figures -> zip)")
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
    print("\n[5/5] Kaggle launcher notebook matches its generator")
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


def main() -> int:
    print("=" * 78)
    print("  ManiBench - kaggle_run.py self-test")
    print("=" * 78)
    for test in (test_probe_healthy, test_probe_detects_numpy_corruption,
                 test_constraints_pin, test_end_to_end_pipeline, test_notebook_is_current):
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
