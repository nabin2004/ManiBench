#!/usr/bin/env python3
"""
ManiBench - Publication Report & Statistical Analysis
=====================================================

Turns raw benchmark records into the written half of a paper package:

  * paired per-task statistics against a baseline (Wilcoxon signed-rank with a
    bootstrap 95% CI on the mean difference, Holm-Bonferroni corrected),
  * extra LaTeX tables (compact model comparison, significance matrix),
  * a ready-to-read Markdown report that embeds the generated figures,
  * `latex_includes.tex` with copy-paste \\input / \\includegraphics lines,
  * `ARTIFACTS.md` indexing everything in the bundle.

Everything degrades gracefully: with 1 model, no render, or a handful of tasks
the report still renders and states exactly what could not be computed.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:  # scipy is a hard dependency of the evaluation stack, but stay defensive
    from scipy import stats as _scipy_stats
except Exception:  # pragma: no cover
    _scipy_stats = None

from scripts.publication_figures import (
    METRIC_SPECS,
    _short,
    _sorted_models,
    model_colors,
    per_problem_values,
)


# ═══════════════════════════════════════════════════════════════════════════
# Statistics
# ═══════════════════════════════════════════════════════════════════════════

def bootstrap_ci(
    values: Sequence[float], n_resamples: int = 10000, confidence: float = 0.95, seed: int = 0,
) -> Tuple[float, float, float]:
    """Mean and percentile bootstrap CI of `values`."""
    data = np.asarray([v for v in values if v is not None], dtype=float)
    if data.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(data))
    if data.size == 1 or float(np.std(data)) == 0.0:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    samples = rng.choice(data, size=(n_resamples, data.size), replace=True).mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return mean, float(np.quantile(samples, alpha)), float(np.quantile(samples, 1 - alpha))


def paired_test(a: Sequence[float], b: Sequence[float]) -> Tuple[Optional[float], str]:
    """Two-sided paired test on matched per-task values -> (p_value, test name)."""
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.size != y.size or x.size < 2:
        return None, "insufficient pairs"
    diff = x - y
    if np.allclose(diff, 0.0):
        return 1.0, "no difference"
    if _scipy_stats is None:
        return None, "scipy unavailable"
    try:
        if x.size >= 6 and np.count_nonzero(diff) >= 2:
            return float(_scipy_stats.wilcoxon(x, y, alternative="two-sided",
                                               zero_method="wilcox").pvalue), "Wilcoxon signed-rank"
        return float(_scipy_stats.ttest_rel(x, y).pvalue), "paired t-test"
    except Exception as exc:  # noqa: BLE001
        return None, f"test failed ({type(exc).__name__})"


def holm_correction(p_values: Sequence[Optional[float]]) -> List[Optional[float]]:
    """Holm-Bonferroni step-down adjusted p-values (None-safe)."""
    indexed = [(i, p) for i, p in enumerate(p_values) if p is not None]
    adjusted: List[Optional[float]] = [None] * len(p_values)
    if not indexed:
        return adjusted
    indexed.sort(key=lambda t: t[1])
    m = len(indexed)
    running = 0.0
    for rank, (original_index, p_value) in enumerate(indexed):
        value = min(1.0, (m - rank) * p_value)
        running = max(running, value)
        adjusted[original_index] = running
    return adjusted


def paired_comparisons(
    records: Sequence[Dict[str, Any]],
    baseline: str,
    models: Optional[Sequence[str]] = None,
    metric: str = "executability",
    problems: Optional[Sequence[str]] = None,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    """
    Per-task paired comparison of every model against `baseline`.

    Returns a list of dicts with mean difference, bootstrap CI, raw and
    Holm-corrected p-values, effect size (Cohen's dz) and the pair count.
    """
    if problems is None:
        problems = sorted({str(r.get("problem_id")) for r in records if r.get("problem_id")})
    models = list(models) if models else sorted({str(r.get("short_name")) for r in records
                                                 if r.get("short_name")})
    base_values, base_problems = per_problem_values(records, baseline, problems, metric)

    results: List[Dict[str, Any]] = []
    raw_p: List[Optional[float]] = []
    for model in models:
        if model == baseline:
            continue
        values, used_problems = per_problem_values(records, model, problems, metric)
        # Match on the intersection of tasks present for both models
        paired_base = [base_values[base_problems.index(p)] for p in used_problems if p in base_problems]
        paired_model = [values[used_problems.index(p)] for p in used_problems if p in base_problems]
        entry: Dict[str, Any] = {
            "model": model,
            "baseline": baseline,
            "metric": metric,
            "n_pairs": len(paired_model),
            "mean_model": float(np.mean(paired_model)) if paired_model else None,
            "mean_baseline": float(np.mean(paired_base)) if paired_base else None,
        }
        if len(paired_model) >= 2:
            differences = np.asarray(paired_model) - np.asarray(paired_base)
            mean_diff, ci_low, ci_high = bootstrap_ci(differences, seed=seed)
            entry.update({"mean_diff": mean_diff, "ci_low": ci_low, "ci_high": ci_high})
            if float(np.std(differences, ddof=1)) > 0:
                entry["cohens_dz"] = float(np.mean(differences) / np.std(differences, ddof=1))
            else:
                entry["cohens_dz"] = 0.0
            p_value, test_name = paired_test(paired_model, paired_base)
        else:
            entry.update({"mean_diff": None, "ci_low": None, "ci_high": None,
                          "cohens_dz": None})
            p_value, test_name = None, "insufficient pairs"
        entry["p_value"] = p_value
        entry["test"] = test_name
        raw_p.append(p_value)
        results.append(entry)

    for entry, adjusted in zip(results, holm_correction(raw_p)):
        entry["p_holm"] = adjusted

    results.sort(key=lambda e: (e["mean_diff"] is None, -(e["mean_diff"] or 0.0)))
    return results


# ═══════════════════════════════════════════════════════════════════════════
# LaTeX helpers
# ═══════════════════════════════════════════════════════════════════════════

def _latex_escape(text: str) -> str:
    for old, new in (("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#")):
        text = text.replace(old, new)
    return text


def _fmt(value: Optional[float], digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "--"
    return f"{value:.{digits}f}"


def compact_latex_table(
    summaries: Dict[str, Dict[str, Any]], baseline: Optional[str] = None,
) -> str:
    """Small 3-6 model comparison table sized for a single column / slide."""
    models = _sorted_models(summaries)
    columns = [s for s in METRIC_SPECS if any(summaries[m].get(s[0]) is not None for m in models)]
    header = " & ".join(
        [r"\textbf{Model}"] + [f"\\textbf{{{_latex_escape(label)}}}" for _k, label, *_ in columns]
    )
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{ManiBench results for the evaluated model variants. "
        r"Pass@1 and mathematical accuracy are treated as primary; VCER is a failure rate "
        r"(lower is better). Best value per column in bold.}",
        r"\label{tab:trio_summary}",
        r"\begin{tabular}{l" + "c" * len(columns) + "}",
        r"\toprule",
        header + r" \\",
        r"\midrule",
    ]
    for model in models:
        summary = summaries[model]
        row = [_latex_escape(_short(model))]
        for key, _label, higher, scale, unit in columns:
            raw = summary.get(key)
            value = None if raw is None else float(raw)
            present = [float(summaries[m][key]) for m in models if summaries[m].get(key) is not None]
            best = (max(present) if higher else min(present)) if present else None
            text = "--" if value is None else f"{value:.1f}" if scale == 100 else f"{value:.3f}"
            if value is not None and best is not None and abs(value - best) < 1e-9:
                text = r"\textbf{" + text + "}"
            row.append(text)
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def significance_latex_table(comparisons: Sequence[Dict[str, Any]], baseline: str) -> str:
    if not comparisons:
        return "% No paired comparisons available.\n"
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Paired per-task comparison against \texttt{" + _latex_escape(_short(baseline)) +
        r"}. $\Delta$ is the mean per-task difference in Pass@1 (percentage points) with a "
        r"bootstrap 95\% CI; $p$ values are Holm-Bonferroni adjusted.}",
        r"\label{tab:trio_significance}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{$\Delta$ Pass@1 (pp)} & \textbf{95\% CI} & "
        r"\textbf{$p_{\text{Holm}}$} & \textbf{Effect size} \\",
        r"\midrule",
    ]
    for entry in comparisons:
        diff = entry.get("mean_diff")
        if diff is None:
            lines.append(f"{_latex_escape(_short(entry['model']))} & -- & -- & -- & -- \\\\")
            continue
        ci = f"[{entry['ci_low'] * 100:+.1f}, {entry['ci_high'] * 100:+.1f}]"
        p_value = entry.get("p_holm", entry.get("p_value"))
        p_text = "--" if p_value is None else (r"$<0.001$" if p_value < 0.001 else f"{p_value:.3f}")
        effect = entry.get("cohens_dz")
        lines.append(
            f"{_latex_escape(_short(entry['model']))} & {diff * 100:+.1f} & {ci} & {p_text} & "
            f"{_fmt(effect, 2)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def latex_includes(
    figures: Sequence[Path], tables: Sequence[Path], figure_dir_rel: str = "figures",
) -> str:
    """Ready-to-paste LaTeX include lines for every artifact."""
    lines = [
        "% ── Auto-generated by scripts/publication_report.py ──────────────────",
        "% Add these lines to your manuscript (graphicx + booktabs required).",
        "%",
        "% Figures",
    ]
    for path in figures:
        if path.suffix != ".pdf":
            continue
        lines += [
            f"\\begin{{figure}}[t]",
            r"  \centering",
            f"  \\includegraphics[width=\\columnwidth]{{{figure_dir_rel}/{path.name}}}",
            f"  \\caption{{% TODO caption for {path.stem}}}",
            f"  \\label{{fig:{path.stem}}}",
            r"\end{figure}",
            "",
        ]
    lines.append("% Tables")
    for path in tables:
        if path.suffix != ".tex":
            continue
        lines.append(f"\\input{{{path.parent.name}/{path.name}}}")
    lines += [
        "",
        "% Wide figures (full page): swap \\columnwidth for \\textwidth in figure*.",
        "",
    ]
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Markdown report
# ═══════════════════════════════════════════════════════════════════════════

def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join(":---" for _ in headers) + " |"]
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def _figure_block(path: Path, caption: str, rel_prefix: str = "figures") -> str:
    if not path.exists():
        return ""
    return f"![{caption}]({rel_prefix}/{path.name})\n\n*{caption}*\n"


def build_markdown_report(
    output_dir: Path,
    summaries: Dict[str, Dict[str, Any]],
    records: Sequence[Dict[str, Any]],
    comparisons: Sequence[Dict[str, Any]],
    baseline: Optional[str],
    figures: Sequence[Path],
    meta: Optional[Dict[str, Any]] = None,
    model_specs: Optional[Sequence[Dict[str, Any]]] = None,
) -> Path:
    meta = meta or {}
    figures_dir = Path(output_dir) / "figures"
    by_stem = {p.stem: p for p in figures if p.suffix == ".png"}

    models = _sorted_models(summaries)
    n_records = len(records)
    n_tasks = len({r.get("problem_id") for r in records if r.get("problem_id")})
    trials = meta.get("trials", 1)

    lines: List[str] = []
    lines.append("# ManiBench: model comparison report")
    lines.append("")
    lines.append(f"**Benchmark:** {meta.get('benchmark', 'ManiBench')} "
                 f"({n_tasks} tasks x {trials} trial(s), prompt strategy "
                 f"`{meta.get('strategy', 'zero_shot')}`)  ")
    lines.append(f"**Evaluated runs:** {n_records}  ")
    lines.append(f"**Models:** {len(models)}  ")
    lines.append(f"**Render validation:** "
                 f"{'disabled (static analysis only)' if meta.get('skip_render') else 'enabled (headless Manim CE)'}  ")
    lines.append(f"**Hardware:** {meta.get('hardware_text', 'see run_manifest.json')}  ")
    lines.append(f"**Generated:** {meta.get('generated_at', '')}")
    lines.append("")

    # ── Models ─────────────────────────────────────────────────────────────
    lines.append("## 1. Evaluated models")
    lines.append("")
    if model_specs:
        rows = [
            [f"`{spec.get('short_name')}`", spec.get("id", ""), spec.get("format", ""),
             spec.get("training_method", ""), spec.get("base_model") or "--"]
            for spec in model_specs
        ]
        lines.append(_markdown_table(["Short name", "Hugging Face repo", "Format", "Method", "Base model"], rows))
    else:
        rows = [[f"`{m}`", summaries[m].get("format", ""), summaries[m].get("training_method", "")]
                for m in models]
        lines.append(_markdown_table(["Model", "Format", "Method"], rows))
    lines.append("")

    # ── Methodology diagram ────────────────────────────────────────────────
    if "fig01_methodology" in by_stem:
        lines.append("## 2. Methodology")
        lines.append("")
        lines.append(_figure_block(by_stem["fig01_methodology"],
                                   "Figure 1. End-to-end ManiBench evaluation pipeline."))
        lines.append("")

    # ── Headline results ───────────────────────────────────────────────────
    lines.append("## 3. Headline results")
    lines.append("")
    best_pass = max((float(summaries[m].get("pass_rate_pct") or 0.0) for m in models), default=0.0)
    rows = []
    for model in models:
        s = summaries[model]
        pass_value = float(s.get("pass_rate_pct") or 0.0)
        pass_text = f"{_fmt(pass_value)}%"
        if abs(pass_value - best_pass) < 1e-9 and len(models) > 1:
            pass_text = f"**{pass_text}**"
        rows.append([
            f"`{_short(model)}`",
            s.get("training_method", ""),
            pass_text,
            f"{_fmt(s.get('vcer_pct'))}%",
            _fmt(s.get("mas_mean"), 3),
            _fmt(s.get("cmi_mean"), 1),
            _fmt(s.get("car_mean"), 3),
            _fmt(s.get("alignment_mean"), 3),
            _fmt(s.get("coverage_mean"), 3),
            _fmt(s.get("avg_gen_time_s"), 1),
        ])
    lines.append(_markdown_table(
        ["Model", "Method", "Pass@1", "VCER", "Math (density)", "CMI", "CAR",
         "Alignment", "Coverage", "Latency (s)"],
        rows))
    lines.append("")
    if "fig02_leaderboard" in by_stem:
        lines.append(_figure_block(by_stem["fig02_leaderboard"],
                                   "Figure 2. Headline leaderboard: executability and version-conflict rate."))
    if "fig03_metric_radar" in by_stem:
        lines.append(_figure_block(by_stem["fig03_metric_radar"],
                                   "Figure 3. Normalised metric profile across seven axes."))

    best = models[0] if models else None
    if best:
        lines.append(f"**Top model:** `{_short(best)}` with "
                     f"{_fmt(summaries[best].get('pass_rate_pct'))}% Pass@1 and "
                     f"{_fmt(summaries[best].get('vcer_pct'))}% VCER "
                     f"({summaries[best].get('samples', 0)} evaluated runs).")
        lines.append("")

    # ── Drift tradeoff ─────────────────────────────────────────────────────
    if "fig04_drift_pareto" in by_stem:
        lines.append("## 4. Syntactic drift vs. executability")
        lines.append("")
        lines.append(_figure_block(by_stem["fig04_drift_pareto"],
                                   "Figure 4. Version-conflict error rate against executability "
                                   "(bubble area = pedagogical coverage)."))
        lowest = min(models, key=lambda m: float(summaries[m].get("vcer_pct") or 1e9)) if models else None
        highest = max(models, key=lambda m: float(summaries[m].get("vcer_pct") or -1)) if models else None
        if lowest and highest and lowest != highest:
            low_v = float(summaries[lowest].get("vcer_pct") or 0.0)
            high_v = float(summaries[highest].get("vcer_pct") or 0.0)
            ratio = (high_v / low_v) if low_v > 0 else float("inf")
            reduction = f"{ratio:.1f}x fewer" if ratio != float("inf") else "essentially zero"
            lines.append(f"`{_short(lowest)}` shows the lowest version-conflict error rate at "
                         f"**{low_v:.1f}%**, against **{high_v:.1f}%** for `{_short(highest)}` - "
                         f"{reduction} legacy ManimGL/3Blue1Brown API hallucinations. "
                         f"Because Pass@1 measures successful Manim CE rendering, a drift gap of "
                         f"this size separates 'the model writes the wrong dialect' from "
                         f"'the model writes the wrong animation'.")
            lines.append("")

    # ── Per-task landscape ─────────────────────────────────────────────────
    if "fig05_problem_heatmap" in by_stem:
        lines.append("## 5. Per-task landscape")
        lines.append("")
        lines.append(_figure_block(by_stem["fig05_problem_heatmap"],
                                   "Figure 5. Pass@1 and VCER for every model/task pair."))
        lines.append("")

    if "fig06_domain_breakdown" in by_stem:
        lines.append("## 6. Domain breakdown")
        lines.append("")
        lines.append(_figure_block(by_stem["fig06_domain_breakdown"],
                                   "Figure 6. Domain-level executability and drift."))
        lines.append("")

    # ── Failure taxonomy ───────────────────────────────────────────────────
    if "fig07_error_taxonomy" in by_stem:
        lines.append("## 7. Failure taxonomy")
        lines.append("")
        lines.append(_figure_block(by_stem["fig07_error_taxonomy"],
                                   "Figure 7. Share of runs by failure mode."))
        lines.append("")

    # ── Statistics ─────────────────────────────────────────────────────────
    lines.append("## 8. Statistical significance")
    lines.append("")
    if comparisons and baseline:
        lines.append(f"Paired per-task comparison against the baseline "
                     f"`{_short(baseline)}` (metric: Pass@1 executability).")
        lines.append("")
        rows = []
        for entry in comparisons:
            diff = entry.get("mean_diff")
            p_value = entry.get("p_holm", entry.get("p_value"))
            rows.append([
                f"`{_short(entry['model'])}`",
                "--" if diff is None else f"{diff * 100:+.1f} pp",
                "--" if diff is None else f"[{entry['ci_low'] * 100:+.1f}, {entry['ci_high'] * 100:+.1f}]",
                "--" if p_value is None else (f"< 0.001" if p_value < 0.001 else f"{p_value:.3f}"),
                "--" if entry.get("cohens_dz") is None else f"{entry['cohens_dz']:+.2f}",
                entry.get("test", ""),
                str(entry.get("n_pairs", 0)),
            ])
        lines.append(_markdown_table(
            ["Model", "Δ Pass@1", "Bootstrap 95% CI", "p (Holm)", "Cohen's dz", "Test", "Pairs"],
            rows))
        lines.append("")
        if "fig08_significance_forest" in by_stem:
            lines.append(_figure_block(by_stem["fig08_significance_forest"],
                                       "Figure 8. Paired mean differences with bootstrap CIs."))
        significant = [e for e in comparisons
                       if (e.get("p_holm") or e.get("p_value") or 1.0) < 0.05]
        if significant:
            lines.append("Differences that survive Holm-Bonferroni correction at alpha = 0.05: "
                         + ", ".join(f"`{_short(e['model'])}`" for e in significant) + ".")
        else:
            lines.append("No pairwise difference survived Holm-Bonferroni correction at "
                         "alpha = 0.05 - treat the ranking as descriptive and increase the "
                         "number of trials/tasks before claiming significance.")
        lines.append("")
    else:
        lines.append("_Insufficient paired data for significance testing "
                     "(needs at least two models scored on the same tasks)._")
        lines.append("")

    if "fig09_metric_correlation" in by_stem:
        lines.append("## 9. Metric inter-correlation")
        lines.append("")
        lines.append(_figure_block(by_stem["fig09_metric_correlation"],
                                   "Figure 9. Correlation between the evaluation metrics."))
        lines.append("")

    # ── Threats to validity ────────────────────────────────────────────────
    lines.append("## 10. Threats to validity")
    lines.append("")
    lines.append(f"- **Sample size.** {n_tasks} tasks x {trials} trial(s) per model "
                 f"({n_records} runs). Ranking stability improves materially with 3+ trials.")
    if meta.get("skip_render"):
        lines.append("- **Rendering disabled.** Pass@1 reflects static validity "
                     "(syntax, imports, scene class) rather than a successful Manim render; "
                     "absolute executability is therefore optimistic.")
    else:
        lines.append("- **Render timeout.** Scenes exceeding the per-scene timeout are counted "
                     "as failures; long pedagogical scenes are penalised.")
    lines.append("- **Reference-free visual metrics.** Alignment and coverage are AST/heuristic "
                 "proxies; DINOv2+DTW similarity requires the reference clips and is reported "
                 "only when it was computed.")
    lines.append("- **MAS semantics.** The pilot dataset ships no `ground_truth_equations`, so "
                 "the mathematical score reports *annotation density* (math expressions found, "
                 "saturating at three) rather than symbolic equivalence to a reference formula.")
    lines.append("- **Decoding.** Temperature 0 with a single trial per task is deterministic "
                 "but not a sample of the model's output distribution.")
    lines.append("")

    # ── Reproducibility ────────────────────────────────────────────────────
    lines.append("## 11. Reproducibility")
    lines.append("")
    lines.append("```bash")
    lines.append(meta.get("command", "python scripts/run_model_trio.py --problems MB-001 ..."))
    lines.append("```")
    lines.append("")
    lines.append(f"- Run manifest: `run_manifest.json`  ")
    lines.append(f"- Raw records: `{meta.get('records_file', 'benchmark_run_*.json')}`  ")
    lines.append(f"- Checkpoint (resume state): `checkpoint_records.jsonl`")
    lines.append("")
    lines.append("## 12. Artifact index")
    lines.append("")
    lines.append("| Artifact | Purpose |")
    lines.append("| :--- | :--- |")
    for path in sorted(figures_dir.glob("*")):
        if path.is_file():
            lines.append(f"| `figures/{path.name}` | {path.stem.replace('_', ' ')} |")
    tables_dir = Path(output_dir) / "tables"
    if tables_dir.exists():
        for path in sorted(tables_dir.glob("*")):
            if path.is_file():
                lines.append(f"| `tables/{path.name}` | LaTeX / CSV export |")
    for extra in ("latex_includes.tex", "ARTIFACTS.md", "run_manifest.json"):
        if (Path(output_dir) / extra).exists():
            lines.append(f"| `{extra}` | bundle metadata |")
    lines.append("")

    report_path = Path(output_dir) / "REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def build_artifacts_index(output_dir: Path, figures: Sequence[Path]) -> Path:
    """ARTIFACTS.md: what every file in the bundle is for."""
    figures_dir = Path(output_dir) / "figures"
    tables_dir = Path(output_dir) / "tables"
    rows = [
        ("figures/*.pdf", "Vector figures - include directly in LaTeX (300 dpi equivalent)"),
        ("figures/*.png", "Raster 300-DPI figures - slides, PR, README"),
        ("tables/table1_main_benchmark.tex", "Main results table (booktabs)"),
        ("tables/table2_domain_breakdown.tex", "Per-domain breakdown"),
        ("tables/table3_error_taxonomy.tex", "Failure-mode taxonomy"),
        ("tables/table4_statistical_significance.tex", "Significance analysis"),
        ("tables/trio_summary.tex", "Compact model comparison (single column)"),
        ("tables/trio_significance.tex", "Paired comparison vs. baseline"),
        ("tables/results_summary.csv", "Model-level metrics (plotting / spreadsheets)"),
        ("tables/results_per_trial.csv", "Every single run (audit trail)"),
        ("tables/leaderboard.md", "Markdown leaderboard"),
        ("tables/paper_results_section.md", "Draft prose for the results section"),
        ("REPORT.md", "Full narrative report with embedded figures"),
        ("latex_includes.tex", "Paste-ready \\input / \\includegraphics lines"),
        ("run_manifest.json", "Exact configuration, environment and hardware"),
        ("checkpoint_records.jsonl", "Resume state - re-running continues here"),
    ]
    lines = ["# Publication bundle index", "",
             "Generated by `scripts/run_model_trio.py`.", "",
             "| Path | Purpose |", "| :--- | :--- |"]
    lines += [f"| `{path}` | {purpose} |" for path, purpose in rows]
    lines.append("")
    present_figures = sorted(p.name for p in figures if p.suffix == ".pdf")
    if present_figures:
        lines.append("Figures in this bundle:")
        lines.append("")
        lines += [f"- `{name}`" for name in present_figures]
        lines.append("")
    path = Path(output_dir) / "ARTIFACTS.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
