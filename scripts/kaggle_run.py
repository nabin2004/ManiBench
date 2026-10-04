#!/usr/bin/env python3
"""
ManiBench - One-Command Kaggle Runner (T4 x2)
=============================================

This single script replaces the whole
`ManiBench_Kaggle_T4x2_Benchmark.ipynb` notebook. It performs, in order:

  Phase 1  Environment detection (Kaggle / Colab / local, GPU inventory)
  Phase 2  Dependency bootstrap - system libs + pip, WITHOUT ever breaking the
           pre-installed NumPy/PyTorch stack (numpy is pinned through a pip
           constraints file, then verified, then repaired/restarted on demand).
  Phase 3  Repo resolution (shallow clone when run from a bare Kaggle notebook)
  Phase 4  Dataset + model-catalog resolution and dry-run validation
  Phase 5  Hugging Face preflight (skip repos that do not exist / are gated)
  Phase 6  Benchmark execution (resumable, crash-safe checkpointing, VRAM teardown)
  Phase 7  300-DPI publication figures
  Phase 8  Submission bundle (.zip) + run manifest + console summary

Typical Kaggle usage (single cell, no arguments needed):

    !git clone https://github.com/nabin2004/ManiBench.git
    %cd ManiBench
    !python scripts/kaggle_run.py --preset paper

Local CPU smoke test (no GPU, no model weights, no pip installs):

    python scripts/kaggle_run.py --preset smoke --no-install

Everything is non-interactive: no prompts, safe to run head-less.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Quiet, deterministic execution: no HF symlink chatter, no tokenizer fork warnings,
# no third-party deprecation noise drowning the actual benchmark output.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore", message=".*antlr4.error.ErrorListener.*")
warnings.filterwarnings("ignore", category=SyntaxWarning, module="mistune")

SCRIPT_PATH = Path(__file__).resolve()
REPO_URL = os.environ.get("MANIBENCH_REPO_URL", "https://github.com/nabin2004/ManiBench.git")
BOOTSTRAP_ENV = "MANIBENCH_BOOTSTRAPPED"
REPO_DIR_ENV = "MANIBENCH_REPO_DIR"
NO_RESTART_ENV = "MANIBENCH_NO_RESTART"
PROBE_MARKER = "__MANIBENCH_PROBE__"

# Packages that must never be silently swapped on Kaggle: the CUDA/NumPy ABI of
# the pre-installed image is what every other library was compiled against.
PINNED_PACKAGES = ("numpy", "torch", "torchvision", "torchaudio")

APT_PACKAGES = ["ffmpeg", "libcairo2-dev", "libpango1.0-dev", "pkg-config"]

PY_CORE = [
    "manim",
    "radon",
    "pycodestyle",
    "dtaidistance",
    "sympy",
    "pandas",
    "matplotlib",
    "tabulate",
    "tqdm",
    "rich",
    "huggingface_hub",
]
PY_INFERENCE = ["transformers", "accelerate", "peft", "safetensors"]
PY_QUANT = ["bitsandbytes"]
PY_VISION = ["opencv-python-headless", "pillow", "scipy"]

# ── Model cohorts (mirrors the notebook's SELECTED_MODELS) ──────────────────
PAPER_MODELS = [
    "nabin2004/qwen-Manimator-1-grpo-merged",
    "nabin2004/qwen-Manimator-1-merged",
    "nabin2004/qwen-Manimator-1-sft",
    "nabin2004/AOS-qwen3-8b-narrated-sft-merged",
    "nabin2004/AOS-qwen3-8b-grpo-merged",
    "nabin2004/AOS-qwen3-8b-narrated-merged",
    "nabin2004/AOS-qwen25-coder-7b-manim-merged",
    "Qwen/Qwen3-8B",
]
SMOKE_PROBLEMS = ["MB-001", "MB-005"]

# Base / SFT-merged / GRPO-adapter trio (see scripts/run_model_trio.py for the
# full publication bundle: 9 figures, LaTeX tables, narrative report, zip).
TRIO_MODELS = [
    "Qwen/Qwen3-8B",
    "nabin2004/AOS-Qwen3-8B-Merged",
    "nabin2004/qwen-Manimator-1-grpo",
]

PRESETS: Dict[str, Dict[str, Any]] = {
    # Full publication run: the 8 headline models x all 12 tasks, rendered.
    "paper": dict(models=PAPER_MODELS, all_problems=True, trials=1, skip_render=False),
    # Same backbone, three checkpoints: untuned base vs SFT-merged vs GRPO adapter.
    "trio": dict(models=TRIO_MODELS, all_problems=True, trials=1, skip_render=False),
    # Everything in the registry (merged + LoRA, no GGUF) - long, multi-session.
    "full": dict(all_models=True, all_problems=True, trials=1, skip_render=False),
    # Fast single-model sanity check with rendering disabled.
    "quick": dict(models=["Qwen/Qwen3-8B"], problems=SMOKE_PROBLEMS, trials=1, skip_render=True),
    # Fully offline pipeline test: synthetic code, no weights, no rendering.
    "smoke": dict(dry_run=True, models=["Qwen/Qwen3-8B"], problems=SMOKE_PROBLEMS, trials=1,
                  skip_render=True),
    # Offline pipeline test WITH real Manim CE rendering (validates Cairo/FFmpeg).
    "smoke-render": dict(dry_run=True, models=["Qwen/Qwen3-8B"], problems=["MB-005"], trials=1,
                         skip_render=False),
}


# ═══════════════════════════════════════════════════════════════════════════
# Console helpers
# ═══════════════════════════════════════════════════════════════════════════

def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78, flush=True)


def log(message: str = "") -> None:
    print(f"  {message}", flush=True)


def warn(message: str) -> None:
    print(f"  [WARN] {message}", flush=True)


def die(message: str, code: int = 1) -> None:
    print(f"\n  [FATAL] {message}\n", flush=True)
    raise SystemExit(code)


def hr() -> None:
    print("-" * 78, flush=True)


# ═══════════════════════════════════════════════════════════════════════════
# Subprocess helpers
# ═══════════════════════════════════════════════════════════════════════════

def run_cmd(cmd: List[str], check: bool = False, capture: bool = False,
            timeout: Optional[int] = None, cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    """Run a command, streaming its output unless `capture` is requested."""
    if capture:
        return subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, cwd=cwd)
    return subprocess.run(cmd, timeout=timeout, cwd=cwd, check=check)


def pip_install(args: List[str], constraints: Optional[Path] = None,
                extra: Optional[List[str]] = None, quiet: bool = True) -> bool:
    """Invoke pip in the current interpreter, optionally under a constraints file."""
    cmd = [sys.executable, "-m", "pip", "install"]
    if quiet:
        cmd.append("-q")
    cmd += list(args)
    if extra:
        cmd += list(extra)
    if constraints and constraints.exists():
        cmd += ["-c", str(constraints)]
    log("$ " + " ".join(cmd))
    try:
        result = run_cmd(cmd)
    except Exception as exc:  # noqa: BLE001
        warn(f"pip invocation failed: {type(exc).__name__}: {exc}")
        return False
    if result.returncode != 0:
        warn(f"pip exited with code {result.returncode}")
        return False
    return True


# ═══════════════════════════════════════════════════════════════════════════
# Environment introspection
# ═══════════════════════════════════════════════════════════════════════════

def environment_kind() -> str:
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or Path("/kaggle/working").exists():
        return "kaggle"
    if "COLAB_GPU" in os.environ or Path("/content").exists():
        return "colab"
    return "local"


def pkg_version(name: str) -> Optional[str]:
    """Read an installed package version WITHOUT importing it."""
    try:
        from importlib.metadata import PackageNotFoundError, version
        try:
            return version(name)
        except PackageNotFoundError:
            return None
    except Exception:  # noqa: BLE001
        return None


PROBE_SRC = r'''
import json
report = {"ok": True, "errors": [], "versions": {}, "cuda": False}

def note(msg):
    report["ok"] = False
    report["errors"].append(msg)

for name in ("numpy", "pandas", "scipy", "torch", "transformers", "manim",
             "sympy", "radon", "pycodestyle", "matplotlib"):
    try:
        mod = __import__(name)
        report["versions"][name] = str(getattr(mod, "__version__", "?"))
    except Exception as exc:
        note("import %s: %s: %s" % (name, type(exc).__name__, exc))

# --- NumPy integrity -------------------------------------------------------
# The classic Kaggle corruption is a *mixed* install: newer pure-Python modules
# (numpy/_core/strings.py) on disk next to an older compiled extension, which
# fails with "cannot import name '_center' from 'numpy._core.umath'".
# NOTE: numpy._core._multiarray_umath.__version__ reports the C-ABI version
# (e.g. "3.1"), NOT the package version - it cannot be used for this check.
try:
    import numpy
    numpy.array([1.0, 2.0, 3.0]).sum()
    parts = tuple(int(p) for p in numpy.__version__.split(".")[:2] if p.isdigit())
    if parts >= (2, 1):
        from numpy._core.umath import _center as _numpy_center      # noqa: F401
        from numpy._core import strings as _numpy_strings           # noqa: F401
        _numpy_strings.center("x", 3)
except Exception as exc:
    note("numpy runtime: %s: %s" % (type(exc).__name__, exc))

# --- The exact failure seen on Kaggle: pandas/numpy._core.strings import ----
try:
    import pandas  # noqa: F401
    pandas.DataFrame({"a": [1, 2, 3]}).sum()
except Exception as exc:
    note("pandas runtime: %s: %s" % (type(exc).__name__, exc))

try:
    import torch
    report["cuda"] = bool(torch.cuda.is_available())
    if report["cuda"]:
        report["gpu_count"] = torch.cuda.device_count()
except Exception:
    pass

print("__MANIBENCH_PROBE__" + json.dumps(report))
'''


def probe_stack(timeout: int = 900) -> Dict[str, Any]:
    """Import the whole scientific stack in a subprocess and report integrity."""
    try:
        result = run_cmd([sys.executable, "-c", PROBE_SRC], capture=True, timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "errors": [f"probe failed to run: {type(exc).__name__}: {exc}"],
                "versions": {}, "cuda": False}
    for line in (result.stdout or "").splitlines():
        if line.startswith(PROBE_MARKER):
            try:
                return json.loads(line[len(PROBE_MARKER):])
            except json.JSONDecodeError:
                break
    return {"ok": False,
            "errors": [(result.stderr or result.stdout or "unknown probe failure")[-500:]],
            "versions": {}, "cuda": False}


def restart_runner(reason: str) -> None:
    """
    Re-exec this script so freshly installed libraries are imported from scratch.

    Kaggle keeps the notebook kernel alive across pip installs, so a NumPy (or
    PyTorch) that was replaced on disk stays stale in memory and every later
    import explodes with confusing ABI errors. Re-exec'ing this script gives us a
    pristine interpreter without asking the user to restart the kernel.
    """
    if os.environ.get(NO_RESTART_ENV) == "1":
        warn(f"restart suppressed by {NO_RESTART_ENV}; continuing in-process ({reason}).")
        return
    if "ipykernel" in sys.modules or "IPython" in sys.modules:
        warn("running inside an interactive kernel - cannot self-restart automatically; "
             "restart the session if you hit import errors.")
        return
    os.environ[BOOTSTRAP_ENV] = "1"
    log(f"Restarting the runner so the refreshed libraries load cleanly ({reason}) ...")
    sys.stdout.flush()
    sys.stderr.flush()
    os.execv(sys.executable, [sys.executable, str(SCRIPT_PATH), *sys.argv[1:]])


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2 - dependency bootstrap (NumPy-safe)
# ═══════════════════════════════════════════════════════════════════════════

def ensure_system_packages(args: argparse.Namespace) -> None:
    """Install FFmpeg / Cairo / Pango on Debian-based images (Kaggle, Colab)."""
    if platform.system() != "Linux":
        has_ffmpeg = shutil.which("ffmpeg") is not None
        log(f"Non-Linux host: skipping apt. ffmpeg {'found' if has_ffmpeg else 'NOT found'} "
            f"(Manim rendering may fail without it).")
        return
    if shutil.which("apt-get") is None:
        warn("apt-get not available; assuming FFmpeg/Cairo/Pango are already present.")
        return

    missing = []
    if shutil.which("ffmpeg") is None:
        missing.append("ffmpeg")
    try:
        import ctypes.util
        if ctypes.util.find_library("cairo") is None:
            missing.append("libcairo2-dev")
        if ctypes.util.find_library("pango-1.0") is None:
            missing.append("libpango1.0-dev")
    except Exception:  # noqa: BLE001
        pass

    if not missing and not args.force_install:
        log("System libraries already present (ffmpeg, cairo, pango).")
        return

    log(f"Installing system packages: {', '.join(APT_PACKAGES)}")
    run_cmd(["apt-get", "update", "-qq"], capture=True, timeout=600)
    result = run_cmd(["apt-get", "install", "-y", "-qq", *APT_PACKAGES], capture=True, timeout=1800)
    if result.returncode != 0:
        warn("apt-get install failed - Manim rendering may be unavailable:")
        warn(((result.stderr or "") + (result.stdout or ""))[-400:])
    else:
        log("System packages installed.")


def write_constraints(path: Path) -> Path:
    """
    Freeze the currently healthy NumPy/PyTorch versions into a constraints file.

    This is the preventive half of the `numpy._core.umath` fix: with
    `pip install -c constraints.txt`, no dependency resolver can ever replace the
    image's NumPy or CUDA-enabled PyTorch build while installing Manim/PEFT/etc.
    """
    lines = [
        "# Auto-generated by scripts/kaggle_run.py - do not edit.",
        "# Pins the pre-installed scientific stack so pip can never corrupt the",
        "# NumPy ABI (the 'cannot import name _center from numpy._core.umath' bug).",
    ]
    for name in PINNED_PACKAGES:
        version = pkg_version(name)
        if version:
            lines.append(f"{name}=={version}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"Wrote dependency constraints to {path}")
    for line in lines[2:]:
        log(f"    pin: {line}")
    return path


def repair_numpy(baseline: Dict[str, Optional[str]]) -> bool:
    """Force-reinstall a consistent NumPy when the on-disk install is mixed."""
    # Never downgrade below what the image was built with: only restore the
    # captured version, then fall back to the newest release.
    candidates = [v for v in (baseline.get("numpy"),) if v] + ["numpy"]
    for spec in candidates:
        if spec is None:
            continue
        log(f"Repairing NumPy with '{spec}' ...")
        ok = pip_install(["--force-reinstall", "--no-cache-dir", "--no-deps", spec], quiet=False)
        if not ok:
            continue
        probe = probe_stack()
        if probe["ok"]:
            log(f"NumPy repaired successfully ({spec}).")
            return True
        log(f"Still unhealthy after '{spec}': {probe['errors'][:2]}")
    return False


def bootstrap_dependencies(args: argparse.Namespace) -> None:
    section("Phase 2/8 - Dependency Bootstrap (NumPy-safe)")
    log(f"Python {sys.version.split()[0]} ({sys.executable})")
    log(f"Platform: {platform.platform()} | Environment: {environment_kind()}")

    if args.no_install:
        log("--no-install given: leaving the Python environment untouched.")
        return

    restarted = os.environ.get(BOOTSTRAP_ENV) == "1"
    baseline = {name: pkg_version(name) for name in PINNED_PACKAGES}
    log("Baseline stack versions: " +
        ", ".join(f"{k}={v or 'absent'}" for k, v in baseline.items()))

    if restarted:
        probe = probe_stack()
        if probe["ok"]:
            log("Environment was already bootstrapped in this session and is healthy.")
            return
        warn("Stack still unhealthy after restart - running targeted repair.")
        if "numpy" in " ".join(probe["errors"]).lower():
            repair_numpy(baseline)
            restart_runner("numpy repaired after restart")
        return

    ensure_system_packages(args)

    pre_probe = probe_stack()
    if not pre_probe["ok"]:
        warn("The Python stack is ALREADY broken before we install anything:")
        for err in pre_probe["errors"][:4]:
            warn(f"    {err}")
        if any("numpy" in e.lower() for e in pre_probe["errors"]):
            repair_numpy(baseline)

    constraints = write_constraints(args.constraints_path)

    log("Installing ManiBench evaluation dependencies (NumPy/PyTorch pinned) ...")
    pip_install(PY_CORE, constraints=constraints)
    if args.with_vision or args.compute_visual_sim:
        pip_install(PY_VISION, constraints=constraints)
    if not args.dry_run:
        pip_install(PY_INFERENCE, constraints=constraints)
        pip_install(PY_QUANT, constraints=constraints)

    post_probe = probe_stack()
    if not post_probe["ok"]:
        warn("Post-install stack probe reported problems:")
        for err in post_probe["errors"][:6]:
            warn(f"    {err}")
        if any("numpy" in e.lower() for e in post_probe["errors"]):
            if repair_numpy(baseline):
                post_probe = probe_stack()
    else:
        log("Post-install stack probe: OK")
        if post_probe.get("cuda"):
            log(f"CUDA visible to PyTorch: {post_probe.get('gpu_count', '?')} device(s)")

    changed = [name for name in PINNED_PACKAGES
               if baseline[name] and pkg_version(name) != baseline[name]]
    if changed:
        warn(f"pip changed pinned package(s): {', '.join(changed)}")
        restart_runner(f"pinned package changed: {', '.join(changed)}")


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3 - repository resolution
# ═══════════════════════════════════════════════════════════════════════════

def ensure_repo(args: argparse.Namespace) -> Path:
    section("Phase 3/8 - Repository & Dataset Resolution")

    candidates: List[Path] = []
    if args.repo_dir:
        candidates.append(Path(args.repo_dir).expanduser())
    if os.environ.get(REPO_DIR_ENV):
        candidates.append(Path(os.environ[REPO_DIR_ENV]))
    cwd = Path.cwd()
    candidates += [cwd, cwd / "ManiBench", cwd.parent]

    for candidate in candidates:
        if (candidate / "ManiBench_Pilot_Dataset.json").exists():
            os.chdir(candidate)
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            log(f"Using ManiBench checkout at {candidate}")
            if args.update and (candidate / ".git").exists():
                log("Updating checkout (git pull --ff-only) ...")
                run_cmd(["git", "-C", str(candidate), "pull", "--ff-only"], capture=True, timeout=300)
            os.environ[REPO_DIR_ENV] = str(candidate)
            return candidate

    if not args.allow_clone:
        die("ManiBench_Pilot_Dataset.json not found and --no-clone was given. "
            "Run from the repository root or pass --repo-dir.")

    destination = Path(args.repo_dir).expanduser() if args.repo_dir else (cwd / "ManiBench")
    log(f"Dataset not found in the current tree - cloning {REPO_URL}")
    log(f"    destination: {destination}")
    if destination.exists() and not (destination / ".git").exists():
        die(f"{destination} exists but is not a git checkout; refusing to overwrite it.")
    run_cmd(["git", "clone", "--depth", "1", REPO_URL, str(destination)], timeout=1800)
    if not (destination / "ManiBench_Pilot_Dataset.json").exists():
        die(f"Clone into {destination} did not produce the dataset. "
            f"Check network access to {REPO_URL}.")
    os.chdir(destination)
    if str(destination) not in sys.path:
        sys.path.insert(0, str(destination))
    os.environ[REPO_DIR_ENV] = str(destination)
    log(f"Repository ready at {destination}")
    return destination


# ═══════════════════════════════════════════════════════════════════════════
# Phase 4/5 - dataset + model catalog + hub preflight
# ═══════════════════════════════════════════════════════════════════════════

def load_problems(repo: Path, wanted: Optional[List[str]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    dataset_path = repo / "ManiBench_Pilot_Dataset.json"
    with open(dataset_path, "r", encoding="utf-8") as fh:
        dataset = json.load(fh)
    problems = dataset.get("problems", [])
    if wanted:
        wanted_set = {p.upper() for p in wanted}
        problems = [p for p in problems if p["id"].upper() in wanted_set]
    if not problems:
        die("No benchmark problems matched the requested selection.")
    return dataset, problems


def resolve_models(args: argparse.Namespace) -> List[Any]:
    sys.path.insert(0, str(Path.cwd()))
    from scripts.model_catalog import catalog  # noqa: WPS433 (deliberate late import)

    if args.all_models:
        specs = catalog.list(family=args.family, format_type=args.format_type, exclude_gguf=not args.include_gguf)
    elif args.models:
        specs = [catalog.get_or_create(identifier) for identifier in args.models]
    elif args.family or args.format_type:
        specs = catalog.list(family=args.family, format_type=args.format_type, exclude_gguf=not args.include_gguf)
    else:
        specs = [catalog.get_or_create(identifier) for identifier in PAPER_MODELS]

    if args.add_models:
        specs.extend(catalog.get_or_create(identifier) for identifier in args.add_models)

    # De-duplicate while preserving order
    seen, unique = set(), []
    for spec in specs:
        if spec.id not in seen:
            seen.add(spec.id)
            unique.append(spec)
    return unique


def _model_info(api: Any, repo_id: str) -> Any:
    """huggingface_hub API compatibility: `timeout` is not available everywhere."""
    try:
        return api.model_info(repo_id, timeout=30)
    except TypeError:
        return api.model_info(repo_id)


def hf_preflight(specs: List[Any], args: argparse.Namespace) -> Tuple[List[Any], List[Dict[str, str]]]:
    usable: List[Any] = []
    skipped: List[Dict[str, str]] = []

    if args.skip_preflight:
        log("Preflight disabled (--skip-preflight).")
        return specs, skipped

    try:
        from huggingface_hub import HfApi, get_token
        api = HfApi()
        token = get_token()
    except Exception as exc:  # noqa: BLE001
        warn(f"huggingface_hub unavailable ({exc}); skipping preflight.")
        return specs, skipped

    log(f"Hugging Face preflight for {len(specs)} model(s) "
        f"({'authenticated' if token else 'no HF token'}) ...")
    for spec in specs:
        try:
            info = _model_info(api, spec.id)
            gated = getattr(info, "gated", False)
            if gated and not token:
                skipped.append({"id": spec.id, "short_name": spec.short_name,
                                "reason": "gated repository and no HF token configured"})
                log(f"  [SKIP] {spec.short_name:<40} gated repo, no HF token")
                continue
            usable.append(spec)
            note = "gated (token present)" if gated else "accessible"
            log(f"  [ OK ] {spec.short_name:<40} {note}")
        except Exception as exc:  # noqa: BLE001
            reason = f"{type(exc).__name__}: {exc}"
            skipped.append({"id": spec.id, "short_name": spec.short_name, "reason": reason[:200]})
            log(f"  [SKIP] {spec.short_name:<40} {reason[:90]}")

    if skipped:
        warn(f"{len(skipped)} model(s) will be skipped (missing/gated repositories).")
    return usable, skipped


# ═══════════════════════════════════════════════════════════════════════════
# Phase 7/8 - figures, bundle, manifest
# ═══════════════════════════════════════════════════════════════════════════

def generate_figures(
    summaries: Dict[str, Any],
    output_dir: Path,
    records: Optional[List[Dict[str, Any]]] = None,
    formatter: Any = None,
    basic: bool = False,
) -> List[str]:
    """
    Publication figures (Phase 7).

    Default: the full publication suite - methodology diagram, leaderboard, radar,
    per-task heatmaps, domain breakdown, failure taxonomy, significance forest,
    metric correlations and a contact sheet (see scripts/publication_figures.py).
    `basic=True` keeps only the two legacy figures (fastest smoke runs).
    """
    section("Phase 7/8 - Publication Figures (300 DPI)")
    if not summaries:
        warn("No model summaries available - skipping figure generation.")
        return []

    figures_dir = output_dir / "figures"
    produced: List[str] = []
    try:
        if basic:
            from scripts.generate_paper_plots import plot_family_comparison, plot_vcer_vs_executability
            plot_vcer_vs_executability(summaries, figures_dir)
            plot_family_comparison(summaries, figures_dir)
        else:
            from scripts.publication_figures import build_all
            from scripts.publication_report import paired_comparisons

            baseline = None
            for name, summary in summaries.items():
                if str(summary.get("family", "")).lower() == "baseline" or \
                   str(summary.get("training_method", "")).lower() == "base":
                    baseline = name
                    break
            if baseline is None:
                baseline = min(summaries, key=lambda n: float(summaries[n].get("pass_rate_pct") or 0.0))
            comparisons = paired_comparisons(records or [], baseline=baseline) if records else []

            build_all(
                output_dir=output_dir,
                summaries=summaries,
                records=records or [],
                domain_summaries=getattr(formatter, "domain_summaries", None),
                error_taxonomy=getattr(formatter, "error_taxonomy", None),
                comparisons=comparisons,
                baseline=baseline,
                meta={"trials": getattr(formatter, "metadata", {}).get("trials", 1)
                      if formatter else 1},
            )

        produced = sorted(str(p) for p in figures_dir.glob("*") if p.is_file())
        for path in produced:
            log(f"  {path}")
    except Exception as exc:  # noqa: BLE001
        warn(f"Figure generation failed: {type(exc).__name__}: {exc}")
    return produced


def build_bundle(output_dir: Path, bundle_dir: Optional[Path] = None,
                 name: str = "manibench_paper_submission_bundle") -> Optional[Path]:
    """
    Zip every paper artifact. The archive is written next to the results and, when
    a different `bundle_dir` is given (e.g. Kaggle's /kaggle/working), copied there
    so it shows up in the notebook output panel.
    """
    section("Phase 8/8 - Submission Bundle")
    archive = output_dir / name
    try:
        zip_path = Path(shutil.make_archive(str(archive), "zip", str(output_dir)))
    except Exception as exc:  # noqa: BLE001
        warn(f"Could not create the zip bundle: {type(exc).__name__}: {exc}")
        return None
    size_mb = zip_path.stat().st_size / (1024 * 1024)
    log(f"Bundle: {zip_path}  ({size_mb:.2f} MB)")

    if bundle_dir is not None and bundle_dir.resolve() != output_dir.resolve():
        try:
            bundle_dir.mkdir(parents=True, exist_ok=True)
            copy_target = bundle_dir / zip_path.name
            if copy_target.resolve() != zip_path.resolve():
                shutil.copy2(zip_path, copy_target)
                log(f"Copy  : {copy_target}")
        except OSError as exc:
            warn(f"Could not copy the bundle to {bundle_dir}: {exc}")
    return zip_path


def write_manifest(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kaggle_run.py",
        description="ManiBench one-command Kaggle / local benchmark runner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Presets:\n"
            "  paper        8 headline models x 12 tasks, rendered (default)\n"
            "  trio         base vs SFT-merged vs GRPO-adapter on Qwen3-8B\n"
            "  full         every non-GGUF model in the registry x 12 tasks\n"
            "  quick        1 baseline model x 2 tasks, no rendering\n"
            "  smoke        synthetic code, no weights, no rendering (offline)\n"
            "  smoke-render synthetic code with real Manim CE rendering\n"
            "\n"
            "For the extended publication bundle (methodology diagram, radar, heatmaps,\n"
            "forest plot, narrative report) use: python scripts/run_model_trio.py\n"
        ),
    )
    parser.add_argument("--preset", choices=sorted(PRESETS), default="paper",
                        help="Pre-baked run configuration (default: paper)")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Explicit HF repo ids or catalog short names")
    parser.add_argument("--add-models", nargs="+", default=None,
                        help="Additional models appended to the selection")
    parser.add_argument("--all-models", action="store_true", help="Use every catalog model")
    parser.add_argument("--include-gguf", action="store_true", help="Include GGUF entries (llama.cpp only)")
    parser.add_argument("--family", default=None,
                        choices=["qwen-manimator", "aos-qwen3", "aos-qwen2.5", "gemma", "baseline"])
    parser.add_argument("--format-type", default=None, choices=["merged", "lora", "gguf", "base"])
    parser.add_argument("--problems", nargs="+", default=None, help="Problem ids, e.g. MB-001 MB-005")
    parser.add_argument("--trials", type=int, default=None, help="Independent generations per problem")
    parser.add_argument("--strategy", default=None, choices=["zero_shot", "version_aware", "cot"])
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=int, default=45, help="Per-render timeout in seconds")
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--skip-render", action="store_true",
                        help="Static analysis only (no Manim execution)")
    parser.add_argument("--render", action="store_true", help="Force rendering on (overrides preset)")
    parser.add_argument("--load-in-4bit", action="store_true", help="NF4 quantization for 31B models")
    parser.add_argument("--compute-visual-sim", action="store_true",
                        help="DINOv2+DTW and SSIM similarity against reference clips")
    parser.add_argument("--with-vision", action="store_true",
                        help="Install OpenCV/Pillow for visual metrics")
    parser.add_argument("--basic-figures", action="store_true",
                        help="Only the two legacy figures instead of the full publication suite")
    parser.add_argument("--dry-run", action="store_true",
                        help="Synthetic code generation (no model weights needed)")

    parser.add_argument("--output-dir", default=None, help="Results directory (default: results/kaggle_paper_run)")
    parser.add_argument("--bundle-dir", default=None,
                        help="Also copy the .zip here (default: current working directory, "
                             "which is what Kaggle shows in the Output panel)")
    parser.add_argument("--no-bundle", action="store_true", help="Skip the zip bundle")
    parser.add_argument("--fresh", action="store_true",
                        help="Delete previous checkpoints/results in the output dir first")
    parser.add_argument("--no-resume", action="store_true", help="Do not resume from checkpoint")

    parser.add_argument("--repo-dir", default=None, help="Path to an existing ManiBench checkout")
    parser.add_argument("--no-clone", action="store_true", help="Never clone the repository")
    parser.add_argument("--update", action="store_true", help="git pull --ff-only before running")
    parser.add_argument("--no-install", action="store_true",
                        help="Skip apt/pip installs (use the current environment as-is)")
    parser.add_argument("--force-install", action="store_true", help="Reinstall apt/pip deps unconditionally")
    parser.add_argument("--skip-preflight", action="store_true",
                        help="Skip the Hugging Face repository check")
    parser.add_argument("--list-models", action="store_true", help="Print the model registry and exit")
    parser.add_argument("--constraints-path", default=None,
                        help="Where to write the pip constraints file (default: <repo>/manibench_constraints.txt)")
    return parser


def apply_preset(args: argparse.Namespace, raw_argv: List[str]) -> argparse.Namespace:
    preset = PRESETS[args.preset]
    explicit = {a.split("=")[0].lstrip("-").replace("-", "_") for a in raw_argv if a.startswith("--")}

    def default(field: str, value: Any) -> Any:
        """Preset value applies unless the user passed that flag explicitly."""
        current = getattr(args, field)
        if field in explicit:
            return current
        return value

    if preset.get("dry_run"):
        args.dry_run = default("dry_run", True)
    if preset.get("all_models"):
        args.all_models = default("all_models", True)
    if preset.get("models") and not args.models and not args.all_models:
        args.models = preset["models"]
    if preset.get("problems") and not args.problems:
        args.problems = preset["problems"]
    if preset.get("trials") is not None and args.trials is None:
        args.trials = preset["trials"]
    if "skip_render" in preset:
        args.skip_render = default("skip_render", preset["skip_render"])
    if args.render:
        args.skip_render = False
    if args.trials is None:
        args.trials = 1
    if args.strategy is None:
        args.strategy = "zero_shot"
    return args


def print_summary(result: Dict[str, Any], skipped: List[Dict[str, str]], zip_path: Optional[Path],
                  elapsed: float) -> None:
    section("Run Summary")
    summaries = result.get("model_summaries", {}) or {}
    records = result.get("records", []) or []

    log(f"Evaluations recorded : {len(records)}")
    log(f"Models with results  : {len(summaries)}")
    log(f"Wall-clock time      : {elapsed / 60:.1f} min")

    if summaries:
        log("")
        log(f"  {'Model':<40} {'Pass@1':>8} {'VCER':>8} {'Align':>8} {'Cov':>8}")
        hr()
        for name, summary in sorted(summaries.items(), key=lambda kv: -kv[1]["pass_rate_pct"]):
            log(f"  {name:<40} {summary['pass_rate_pct']:>7.1f}% {summary['vcer_pct']:>7.1f}% "
                f"{summary['alignment_mean']:>8.3f} {summary['coverage_mean']:>8.3f}")

    if skipped:
        log("")
        warn("Skipped models (fix the repository id or add an HF token and re-run):")
        for entry in skipped:
            log(f"    {entry['short_name']:<40} {entry['reason'][:80]}")

    statuses = result.get("model_statuses", {}) or {}
    failed = {k: v for k, v in statuses.items() if v.get("status") == "load_failed"}
    if failed:
        warn("Model load failures - full tracebacks are stored in the run JSON:")
        for name, status in failed.items():
            log(f"    {name:<40} {status.get('error', '')[:80]}")

    log("")
    log("Artifacts:")
    for name, path in (result.get("exported_files") or {}).items():
        log(f"    {name:<22} {path}")
    if result.get("json_path"):
        log(f"    {'raw_json_archive':<22} {result['json_path']}")
    if zip_path:
        log(f"    {'submission_bundle':<22} {zip_path}")

    if zip_path and environment_kind() == "kaggle":
        print("\n  Kaggle: open the notebook 'Output' panel and download "
              f"{zip_path.name} (or use the file browser on the right).\n")


# ═══════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════

def main(argv: Optional[List[str]] = None) -> int:
    started = time.time()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(raw_argv)
    args = apply_preset(args, raw_argv)

    print("=" * 78)
    print("  ManiBench - End-to-End Benchmark Runner")
    print(f"  preset={args.preset}  dry_run={args.dry_run}  render={'no' if args.skip_render else 'yes'}")
    print("=" * 78, flush=True)

    # ── Phase 1: environment ───────────────────────────────────────────────
    section("Phase 1/8 - Environment Detection")
    log(f"Host: {platform.node()} | {platform.system()} {platform.release()}")
    log(f"Environment: {environment_kind()}")

    # ── Phase 3 first: the repo must exist before any project import ───────
    repo = ensure_repo(args)

    if args.list_models:
        from scripts.model_catalog import catalog
        for index, spec in enumerate(catalog.list(), 1):
            print(f"  {index:>3}. {spec.short_name:<40} {spec.format:<7} {spec.id}")
        return 0

    if args.constraints_path is None:
        args.constraints_path = repo / "manibench_constraints.txt"
    else:
        args.constraints_path = Path(args.constraints_path).expanduser()

    output_dir = Path(args.output_dir).expanduser() if args.output_dir else repo / "results" / "kaggle_paper_run"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.fresh:
        for stale in list(output_dir.glob("checkpoint_records.jsonl")) + list(output_dir.glob("benchmark_run_*.json")):
            stale.unlink()
        log(f"--fresh: cleared previous checkpoints in {output_dir}")

    bootstrap_dependencies(args)

    # ── Phase 4: dataset + models ──────────────────────────────────────────
    section("Phase 4/8 - Dataset & Model Registry")
    dataset, problems = load_problems(repo, args.problems)
    log(f"Benchmark : {dataset.get('benchmark_name', 'ManiBench')} v{dataset.get('version', '?')}")
    log(f"Problems  : {len(problems)} -> {', '.join(p['id'] for p in problems)}")

    models = resolve_models(args)
    log(f"Models    : {len(models)}")
    for spec in models:
        log(f"    - {spec.short_name:<40} {spec.id}")

    # ── Phase 5: preflight ─────────────────────────────────────────────────
    section("Phase 5/8 - Model Preflight")
    if args.dry_run:
        log("Dry-run mode: skipping Hugging Face preflight.")
        usable, skipped = models, []
    else:
        usable, skipped = hf_preflight(models, args)
        if not usable:
            die("Every requested model failed preflight. Check the repository ids / HF token.")

    # ── Phase 6: execute ───────────────────────────────────────────────────
    section("Phase 6/8 - Benchmark Execution")
    from scripts.run_kaggle_benchmark import run_benchmark

    result = run_benchmark(
        models=usable,
        problems=problems,
        output_dir=output_dir,
        trials=args.trials,
        strategy=args.strategy,
        skip_render=args.skip_render,
        timeout=args.timeout,
        dry_run=args.dry_run,
        load_in_4bit=args.load_in_4bit,
        compute_visual_sim=args.compute_visual_sim,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        resume=not args.no_resume,
        reference_dir=repo / "media" / "references",
    )

    # ── Phase 7: figures ───────────────────────────────────────────────────
    figures = generate_figures(
        result.get("model_summaries", {}),
        output_dir,
        records=result.get("records"),
        formatter=result.get("formatter"),
        basic=args.basic_figures,
    )

    # ── Phase 8: bundle + manifest ─────────────────────────────────────────
    zip_path = None
    if not args.no_bundle:
        bundle_dir = Path(args.bundle_dir).expanduser() if args.bundle_dir else Path.cwd()
        zip_path = build_bundle(output_dir, bundle_dir)

    manifest_path = output_dir / "run_manifest.json"
    write_manifest(manifest_path, {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "argv": sys.argv[1:],
        "preset": args.preset,
        "config": {
            "models": [s.id for s in usable],
            "skipped_models": skipped,
            "problems": [p["id"] for p in problems],
            "trials": args.trials,
            "strategy": args.strategy,
            "skip_render": args.skip_render,
            "dry_run": args.dry_run,
            "load_in_4bit": args.load_in_4bit,
            "compute_visual_sim": args.compute_visual_sim,
            "max_new_tokens": args.max_new_tokens,
            "timeout": args.timeout,
        },
        "hardware": result.get("hardware"),
        "model_statuses": result.get("model_statuses"),
        "counts": {
            "records": len(result.get("records", [])),
            "resumed_skips": result.get("skipped_evaluations", 0),
        },
        "figures": figures,
        "exported_files": {k: str(v) for k, v in (result.get("exported_files") or {}).items()},
        "raw_json": str(result.get("json_path", "")),
        "bundle": str(zip_path) if zip_path else None,
        "elapsed_seconds": round(time.time() - started, 1),
    })
    log(f"Manifest: {manifest_path}")

    print_summary(result, skipped, zip_path, time.time() - started)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n  Interrupted by user - the checkpoint file lets you resume with the same command.\n")
        raise SystemExit(130)
