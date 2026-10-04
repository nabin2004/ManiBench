#!/usr/bin/env python3
"""
ManiBench - Base vs. SFT vs. GRPO Trio Benchmark (publication package)
=====================================================================

One script that evaluates three model variants of the same Qwen3-8B backbone and
emits a camera-ready publication bundle (figures, LaTeX, CSVs, narrative report):

    Qwen3-8B-Base            Qwen/Qwen3-8B                        (foundation baseline)
    AOS-Qwen3-8B-Merged      nabin2004/AOS-Qwen3-8B-Merged        (SFT, merged weights)
    qwen-Manimator-1-grpo    nabin2004/qwen-Manimator-1-grpo      (GRPO LoRA adapter)

Usage
-----
    # full publication run on Kaggle T4x2 (recommended: 3 trials)
    python scripts/run_model_trio.py --trials 3

    # quick local pipeline check (synthetic code, no weights, no GPU)
    python scripts/run_model_trio.py --dry-run --problems MB-001 MB-005

    # rebuild ONLY the figures/report from a finished run (no GPU needed)
    python scripts/run_model_trio.py --from-json results/trio_publication/benchmark_run_*.json

Outputs (in `--output-dir`, default `results/trio_publication`):

    figures/      9 publication figures, each as vector PDF + 300-DPI PNG
    tables/       LaTeX (booktabs) + CSV exports, incl. trio_summary.tex
    REPORT.md     narrative report with embedded figures and statistics
    latex_includes.tex   paste-ready \\input / \\includegraphics lines
    ARTIFACTS.md  index of every artifact and what it is for
    run_manifest.json    exact config, hardware and model provenance
    *.zip         everything above, for Overleaf / reviewers
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

from scripts.model_catalog import ModelCatalog, ModelSpec, catalog as global_catalog
from scripts.paper_formatter import PaperFormatter, describe_hardware
from scripts.publication_figures import build_all as build_figures
from scripts.publication_report import (
    build_artifacts_index,
    build_markdown_report,
    compact_latex_table,
    latex_includes,
    paired_comparisons,
    significance_latex_table,
)

DATASET_PATH = ROOT_DIR / "ManiBench_Pilot_Dataset.json"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "results" / "trio_publication"

# ── The trio: same backbone, three checkpoints ─────────────────────────────
TRIO: List[Dict[str, Any]] = [
    dict(
        id="Qwen/Qwen3-8B",
        short_name="Qwen3-8B-Base",
        family="baseline",
        format="base",
        base_model=None,
        param_size="8B",
        training_method="Base",
        chat_template_family="chatml",
        description="Pretrained Qwen3-8B foundation model - the untuned control.",
    ),
    dict(
        id="nabin2004/AOS-Qwen3-8B-Merged",
        short_name="AOS-Qwen3-8B-Merged",
        family="aos-qwen3",
        format="merged",
        base_model="Qwen/Qwen3-8B",
        param_size="8B",
        training_method="SFT",
        chat_template_family="chatml",
        description="Supervised fine-tuned Manim model, adapter merged into the base weights.",
    ),
    dict(
        id="nabin2004/qwen-Manimator-1-grpo",
        short_name="qwen-Manimator-1-grpo",
        family="qwen-manimator",
        format="lora",
        base_model="Qwen/Qwen3-8B",
        param_size="8B",
        training_method="GRPO",
        chat_template_family="chatml",
        description="GRPO reinforcement-learned LoRA adapter (r=16) on the Qwen3-8B base.",
    ),
]


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78, flush=True)


def resolve_trio(registry: Optional[ModelCatalog] = None) -> List[ModelSpec]:
    """Look the three variants up in the catalog, registering any missing entry."""
    registry = registry or global_catalog
    specs: List[ModelSpec] = []
    for entry in TRIO:
        spec = registry.get(entry["id"]) or registry.get(entry["short_name"])
        if spec is None:
            spec = ModelSpec(**entry)
            registry.register(spec)
            print(f"  [catalog] registered new model: {spec.short_name} ({spec.id})")
        specs.append(spec)
    return specs


def load_problems(wanted: Optional[Sequence[str]]) -> List[Dict[str, Any]]:
    if not DATASET_PATH.exists():
        raise SystemExit(f"[FATAL] dataset not found: {DATASET_PATH}")
    data = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    problems = data.get("problems", [])
    if wanted:
        wanted_set = {p.upper() for p in wanted}
        problems = [p for p in problems if p["id"].upper() in wanted_set]
    if not problems:
        raise SystemExit("[FATAL] no benchmark problems matched the requested selection")
    return problems


def preflight(specs: List[ModelSpec], skip: bool) -> List[ModelSpec]:
    if skip:
        print("  preflight skipped (--skip-preflight)")
        return specs
    try:
        from scripts.kaggle_run import hf_preflight
        usable, skipped = hf_preflight(specs, argparse.Namespace(skip_preflight=False))
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] preflight unavailable ({type(exc).__name__}: {exc}); continuing")
        return specs
    return usable or specs


# ═══════════════════════════════════════════════════════════════════════════
# Publication package
# ═══════════════════════════════════════════════════════════════════════════

def build_publication_package(
    output_dir: Path,
    records: List[Dict[str, Any]],
    meta: Dict[str, Any],
    specs: List[ModelSpec],
    baseline_short: str,
    first_timestamp: str,
) -> Dict[str, Any]:
    """Tables -> statistics -> figures -> report -> includes -> index -> zip."""
    section("Building the publication package")

    print("  [1/7] LaTeX tables, leaderboard and CSV exports ...")
    formatter = PaperFormatter(raw_records=records, metadata=meta)
    exported = formatter.export_all(output_dir)
    for name, path in exported.items():
        print(f"        {name:<22} {path}")

    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    print("  [2/7] Paired statistics against the baseline ...")
    comparisons = paired_comparisons(records, baseline=baseline_short)
    for entry in comparisons:
        diff = entry.get("mean_diff")
        p_holm = entry.get("p_holm")
        print(f"        {entry['model']:<28} "
              f"Δ={('n/a' if diff is None else f'{diff * 100:+.1f} pp'):>9}  "
              f"p_holm={('n/a' if p_holm is None else f'{p_holm:.4f}'):>7}  "
              f"pairs={entry.get('n_pairs')}")

    extra_tables: List[Path] = []
    summary_tex = tables_dir / "trio_summary.tex"
    summary_tex.write_text(compact_latex_table(formatter.model_summaries, baseline_short), encoding="utf-8")
    extra_tables.append(summary_tex)
    significance_tex = tables_dir / "trio_significance.tex"
    significance_tex.write_text(significance_latex_table(comparisons, baseline_short), encoding="utf-8")
    extra_tables.append(significance_tex)

    print("  [3/7] Publication figures (PDF + 300-DPI PNG) ...")
    model_labels = [
        {
            "label": spec.short_name,
            "sublabel": f"{spec.training_method} / {spec.format}"
                        + (f"\nbase: {spec.base_model.split('/')[-1]}" if spec.base_model else ""),
            "edge": "#4C72B0",
            "face": "#FFFFFF",
        }
        for spec in specs
    ]
    figures = build_figures(
        output_dir=output_dir,
        summaries=formatter.model_summaries,
        records=records,
        domain_summaries=formatter.domain_summaries,
        error_taxonomy=formatter.error_taxonomy,
        comparisons=comparisons,
        baseline=baseline_short,
        model_labels=model_labels,
        meta=meta,
    )

    print("  [4/7] Narrative report ...")
    report_path = build_markdown_report(
        output_dir=output_dir,
        summaries=formatter.model_summaries,
        records=records,
        comparisons=comparisons,
        baseline=baseline_short,
        figures=figures,
        meta=meta,
        model_specs=[s.to_dict() for s in specs],
    )
    print(f"        {report_path}")

    print("  [5/7] LaTeX include snippets ...")
    all_tables = sorted(tables_dir.glob("*.tex"))
    includes_path = output_dir / "latex_includes.tex"
    includes_path.write_text(latex_includes(figures, all_tables), encoding="utf-8")
    print(f"        {includes_path}")

    print("  [6/7] Artifact index ...")
    index_path = build_artifacts_index(output_dir, figures)
    print(f"        {index_path}")

    print("  [7/7] Bundle ...")
    zip_path: Optional[Path] = None
    try:
        zip_path = Path(shutil.make_archive(
            str(output_dir / f"manibench_trio_{first_timestamp}"), "zip", str(output_dir)))
        print(f"        {zip_path} ({zip_path.stat().st_size / (1024 * 1024):.2f} MB)")
    except Exception as exc:  # noqa: BLE001
        print(f"        [warn] zip failed: {type(exc).__name__}: {exc}")

    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "argparse_argv": sys.argv[1:],
        "models": [s.to_dict() for s in specs],
        "baseline": baseline_short,
        "config": {k: meta.get(k) for k in
                   ("trials", "strategy", "skip_render", "max_new_tokens", "timeout", "seed")},
        "counts": {"records": len(records),
                   "tasks": len({r.get("problem_id") for r in records}),
                   "models_with_results": len(formatter.model_summaries)},
        "hardware": meta.get("hardware"),
        "model_statuses": meta.get("model_statuses"),
        "comparisons": comparisons,
        "figures": [str(p) for p in figures],
        "tables": [str(p) for p in all_tables],
        "report": str(report_path),
        "latex_includes": str(includes_path),
        "bundle": str(zip_path) if zip_path else None,
    }, indent=2, default=str), encoding="utf-8")

    return {
        "formatter": formatter,
        "exported": exported,
        "comparisons": comparisons,
        "figures": figures,
        "report": report_path,
        "includes": includes_path,
        "index": index_path,
        "manifest": manifest_path,
        "zip": zip_path,
    }


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_model_trio.py",
        description="Evaluate base vs. SFT-merged vs. GRPO-adapter and build a publication bundle.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/run_model_trio.py --trials 3                 # Kaggle T4x2, full\n"
            "  python scripts/run_model_trio.py --problems MB-001 MB-002   # subset\n"
            "  python scripts/run_model_trio.py --dry-run                  # no GPU / no weights\n"
            "  python scripts/run_model_trio.py --from-json results/trio_publication/benchmark_run_*.json\n"
        ),
    )
    parser.add_argument("--problems", nargs="+", default=None,
                        help="Problem ids to evaluate (default: all 12)")
    parser.add_argument("--trials", type=int, default=1,
                        help="Trials per task (use 3+ for publishable significance tests)")
    parser.add_argument("--strategy", default="zero_shot",
                        choices=["zero_shot", "version_aware", "cot"])
    parser.add_argument("--skip-render", action="store_true",
                        help="Static metrics only (much faster, no Manim execution)")
    parser.add_argument("--load-in-4bit", action="store_true", help="NF4 quantization")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=int, default=60, help="Per-render timeout (seconds)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true",
                        help="Synthetic code engine - full pipeline without model weights")
    parser.add_argument("--baseline", default="Qwen3-8B-Base",
                        help="Short name of the baseline model for paired statistics")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--fresh", action="store_true", help="Discard the checkpoint and start over")
    parser.add_argument("--no-resume", action="store_true", help="Ignore the checkpoint file")
    parser.add_argument("--skip-preflight", action="store_true",
                        help="Do not verify the Hugging Face repositories")
    parser.add_argument("--from-json", default=None,
                        help="Rebuild the publication package from a finished benchmark_run_*.json")
    parser.add_argument("--no-bundle", action="store_true", help="Skip the zip archive")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    started = time.time()
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("  ManiBench - Base vs SFT vs GRPO publication run")
    print("=" * 78, flush=True)

    specs = resolve_trio()

    # ── Rebuild-only mode ──────────────────────────────────────────────────
    if args.from_json:
        import glob
        matches = sorted(glob.glob(args.from_json))
        if not matches:
            raise SystemExit(f"[FATAL] no run JSON matched: {args.from_json}")
        source = Path(matches[-1])
        print(f"  Reloading records from {source}")
        data = json.loads(source.read_text(encoding="utf-8"))
        records = data.get("detailed_records", [])
        if not records:
            raise SystemExit("[FATAL] the run JSON contains no detailed_records")
        saved_meta = data.get("metadata", {})
        meta = {
            **saved_meta,
            "benchmark": saved_meta.get("benchmark", "ManiBench"),
            "trials": saved_meta.get("trials", args.trials),
            "strategy": saved_meta.get("strategy", args.strategy),
            "skip_render": saved_meta.get("skip_render", args.skip_render),
            "n_problems": len({r.get("problem_id") for r in records}),
            "hardware": saved_meta.get("hardware"),
            "hardware_text": describe_hardware(saved_meta.get("hardware")),
            "model_statuses": data.get("model_statuses", {}),
            "command": "python " + " ".join(sys.argv),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }
        stamp = source.stem.replace("benchmark_run_", "")
        package = build_publication_package(output_dir, records, meta, specs, args.baseline, stamp)
        _print_summary(package, specs, output_dir, time.time() - started)
        return 0

    # ── Full run ───────────────────────────────────────────────────────────
    problems = load_problems(args.problems)
    print(f"  Problems : {len(problems)} -> {', '.join(p['id'] for p in problems)}")
    print(f"  Models   : {len(specs)}")
    for spec in specs:
        print(f"      - {spec.short_name:<26} {spec.id}  [{spec.format}/{spec.training_method}]")
    print(f"  Trials   : {args.trials}   Strategy: {args.strategy}   "
          f"Render: {'off' if args.skip_render else 'on'}")

    if args.fresh:
        stale_files = list(output_dir.glob("checkpoint_records.jsonl"))
        stale_files += list(output_dir.glob("benchmark_run_*.json"))
        for stale in stale_files:
            stale.unlink()
        print(f"  --fresh: cleared previous checkpoints in {output_dir}")

    section("Model preflight")
    usable = specs if args.dry_run else preflight(specs, args.skip_preflight)
    if not usable:
        raise SystemExit("[FATAL] every model failed preflight")
    if args.dry_run:
        print("  dry-run: preflight skipped, synthetic engine will be used")

    section("Benchmark execution")
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
        compute_visual_sim=False,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        resume=not args.no_resume,
        reference_dir=ROOT_DIR / "media" / "references",
    )

    records = result.get("records", [])
    if not records:
        print("\n  [WARNING] no evaluation records were produced - the report will state this.\n")

    hardware = result.get("hardware") or {}
    meta: Dict[str, Any] = {
        "benchmark": "ManiBench (pilot v1.0)",
        "timestamp": result.get("metadata", {}).get("timestamp", ""),
        "trials": args.trials,
        "strategy": args.strategy,
        "skip_render": args.skip_render,
        "max_new_tokens": args.max_new_tokens,
        "timeout": args.timeout,
        "seed": args.seed,
        "n_problems": len({r.get("problem_id") for r in records}) or len(problems),
        "hardware": hardware,
        "hardware_text": describe_hardware(hardware),
        "model_statuses": result.get("model_statuses", {}),
        "command": "python " + " ".join(sys.argv),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "records_file": Path(result.get("json_path", "")).name,
    }
    stamp = result.get("metadata", {}).get("timestamp") or datetime.now().strftime("%Y%m%d_%H%M%S")

    package = build_publication_package(output_dir, records, meta, usable, args.baseline, stamp)

    if args.no_bundle and package.get("zip"):
        try:
            Path(package["zip"]).unlink()
            package["zip"] = None
        except OSError:
            pass

    _print_summary(package, usable, output_dir, time.time() - started)
    return 0


def _print_summary(package: Dict[str, Any], specs: Sequence[ModelSpec],
                   output_dir: Path, elapsed: float) -> None:
    section("Publication package ready")
    formatter: PaperFormatter = package["formatter"]
    summaries = formatter.model_summaries
    print(f"  Wall clock      : {elapsed / 60:.1f} min")
    print(f"  Evaluated runs  : {len(formatter.records)}")
    print()
    if summaries:
        print(f"  {'Model':<28} {'Pass@1':>8} {'VCER':>8} {'MAS':>7} {'Align':>7} {'Cover':>7}")
        print("  " + "-" * 70)
        for name in sorted(summaries, key=lambda n: -summaries[n]["pass_rate_pct"]):
            s = summaries[name]
            print(f"  {name:<28} {s['pass_rate_pct']:>7.1f}% {s['vcer_pct']:>7.1f}% "
                  f"{(s['mas_mean'] or 0):>7.3f} {s['alignment_mean']:>7.3f} {s['coverage_mean']:>7.3f}")
        print()
    print("  Artifacts:")
    print(f"      figures      {output_dir / 'figures'}  ({len(package['figures'])} files)")
    print(f"      tables       {output_dir / 'tables'}")
    print(f"      report       {package['report']}")
    print(f"      latex        {package['includes']}")
    print(f"      index        {package['index']}")
    print(f"      manifest     {package['manifest']}")
    if package.get("zip"):
        print(f"      bundle       {package['zip']}")
    print()
    print("  Next steps:")
    print("      1. open REPORT.md for the narrative + embedded figures")
    print("      2. \\input the files listed in latex_includes.tex (booktabs required)")
    print("      3. cite run_manifest.json for exact config and hardware provenance")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n  Interrupted - the checkpoint file lets you resume with the same command.\n")
        raise SystemExit(130)
