#!/usr/bin/env python3
"""
ManiBench - Publication-Grade Figure Suite
==========================================

Nine camera-ready figures plus a contact sheet, generated from the records and
summary dictionaries produced by `scripts/run_kaggle_benchmark.run_benchmark`
(via `scripts.paper_formatter.PaperFormatter`).

  fig01  Methodology / pipeline diagram            (schematic, no data needed)
  fig02  Headline leaderboard (Pass@1 + VCER, 95% CI)
  fig03  Metric profile radar (7 normalized axes)
  fig04  Syntactic-drift tradeoff scatter + Pareto frontier
  fig05  Per-problem heatmaps (Pass@1, VCER)
  fig06  Domain breakdown
  fig07  Failure taxonomy (stacked shares)
  fig08  Statistical significance vs. baseline (forest plot)
  fig09  Metric correlation matrix (Spearman)
  contact sheet: 2x3 montage of the key figures for quick review

Every figure is written twice - vector PDF (for LaTeX) and 300-DPI PNG (for
slides/PR) - into `<output_dir>/figures/`.

Design rules: colour-blind-safe palette, no chartjunk, every value annotated,
direction of "better" stated on the axis, all text legible when printed at
column width (3.5 in) and at full page width (7 in).
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ═══════════════════════════════════════════════════════════════════════════
# Style
# ═══════════════════════════════════════════════════════════════════════════

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Liberation Serif", "serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "figure.titlesize": 12,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.06,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.28,
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "legend.frameon": True,
    "legend.framealpha": 0.92,
    "legend.edgecolor": "0.75",
})

# Colour-blind-safe palette (Okabe-Ito derived) + stable model -> colour mapping
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
           "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]

ROLE_COLORS = {
    "base": "#7F7F7F",
    "sft": "#4C72B0",
    "dpo": "#8172B3",
    "grpo": "#55A868",
    "custom": "#DD8452",
}

# (summary key, axis label, higher_is_better, normalisation max, unit)
METRIC_SPECS: List[Tuple[str, str, bool, float, str]] = [
    ("pass_rate_pct", "Pass@1 (executability)", True, 100.0, "%"),
    ("vcer_pct", "Version-conflict rate", False, 100.0, "%"),
    ("mas_mean", "Math annotation density", True, 1.0, ""),
    ("car_mean", "Constraint adherence", True, 1.0, ""),
    ("alignment_mean", "Visual alignment", True, 1.0, ""),
    ("coverage_mean", "Pedagogical coverage", True, 1.0, ""),
    # code_quality.cmi_score is normalised to [0, 1] (radon MI 0-100 / 100)
    ("cmi_mean", "Maintainability index", True, 1.0, ""),
]

RADAR_METRICS = METRIC_SPECS


def precision_label(weight_precision: Optional[str]) -> str:
    """
    Compact weight-precision label for figures/reports.

    "nf4-4bit (all models)" -> "4-bit NF4 (all models)". Falls back to a neutral
    label when the metadata does not record a precision.
    """
    if not weight_precision:
        return "FP16"
    text = str(weight_precision)
    if text.startswith("nf4-4bit"):
        return "4-bit NF4" + text[len("nf4-4bit"):]
    if text.startswith("synthetic"):
        return "no weights (synthetic)"
    return text


def _short(name: str) -> str:
    """
    Compact model label for axes and tables.

    Keeps the informative parts of the catalog short name (family + variant) and
    only drops the Hugging Face org prefix, so `AOS-Qwen3-8B-Merged` never gets
    confused with the bare `Qwen3-8B-Base`. Case-insensitive on the abbreviation.
    """
    short = re.sub(r"^nabin2004/", "", str(name))
    short = re.sub(r"qwen-Manimator-1", "Manimator-1", short, flags=re.IGNORECASE)
    return short[:30]


def model_roles(summaries: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    """Infer each model's role (base / sft / dpo / grpo / custom) for colouring."""
    roles: Dict[str, str] = {}
    for name, summary in summaries.items():
        method = str(summary.get("training_method", "")).lower()
        family = str(summary.get("family", "")).lower()
        fmt = str(summary.get("format", "")).lower()
        if method in ("base", "") and family == "baseline":
            roles[name] = "base"
        elif "grpo" in method:
            roles[name] = "grpo"
        elif "dpo" in method:
            roles[name] = "dpo"
        elif "sft" in method:
            roles[name] = "sft"
        elif fmt == "base":
            roles[name] = "base"
        else:
            roles[name] = "custom"
    return roles


def model_colors(summaries: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    """Stable colour per model, with the role palette applied where possible."""
    roles = model_roles(summaries)
    colors: Dict[str, str] = {}
    used = set()
    for index, (name, role) in enumerate(roles.items()):
        color = ROLE_COLORS.get(role)
        if color is None or (color in used and role != "custom"):
            color = PALETTE[index % len(PALETTE)]
        colors[name] = color
        used.add(color)
    return colors


def _save(fig: plt.Figure, out_dir: Path, stem: str, also_png: bool = True) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{stem}.pdf"]
    fig.savefig(paths[0])
    if also_png:
        paths.append(out_dir / f"{stem}.png")
        fig.savefig(paths[-1])
    plt.close(fig)
    return paths


def _sorted_models(summaries: Dict[str, Dict[str, Any]]) -> List[str]:
    """Best Pass@1 first; ties broken by lower VCER."""
    return sorted(
        summaries,
        key=lambda n: (-float(summaries[n].get("pass_rate_pct") or 0.0),
                       float(summaries[n].get("vcer_pct") or 0.0)),
    )


def _fmt(value: Optional[float], spec_unit: str = "", digits: int = 1) -> str:
    if value is None:
        return "n/a"
    if spec_unit == "%":
        return f"{value:.1f}%"
    return f"{value:.{digits}f}"


# ═══════════════════════════════════════════════════════════════════════════
# Aggregation helpers
# ═══════════════════════════════════════════════════════════════════════════

def per_problem_matrix(
    records: Sequence[Dict[str, Any]],
    models: Sequence[str],
    problems: Sequence[str],
    metric: str = "executability",
    scale: float = 1.0,
) -> np.ndarray:
    """Mean of `metric` per (model, problem); NaN where a cell has no record."""
    matrix = np.full((len(models), len(problems)), np.nan)
    for i, model in enumerate(models):
        for j, problem in enumerate(problems):
            values = [
                float(r.get(metric))
                for r in records
                if r.get("short_name") == model
                and r.get("problem_id") == problem
                and r.get(metric) is not None
            ]
            if values:
                matrix[i, j] = float(np.mean(values)) * scale
    return matrix


def per_problem_values(
    records: Sequence[Dict[str, Any]],
    model: str,
    problems: Sequence[str],
    metric: str = "executability",
) -> Tuple[List[float], List[str]]:
    """One averaged value per problem (used for paired statistics)."""
    values: List[float] = []
    labels: List[str] = []
    for problem in problems:
        found = [
            float(r[metric])
            for r in records
            if r.get("short_name") == model
            and r.get("problem_id") == problem
            and r.get(metric) is not None
        ]
        if found:
            values.append(float(np.mean(found)))
            labels.append(problem)
    return values, labels


def metric_matrix(
    summaries: Dict[str, Dict[str, Any]], models: Sequence[str],
) -> Tuple[np.ndarray, List[str], List[bool]]:
    """(normalized[0,1] matrix, raw labels, higher_is_better flags) for the radar."""
    normalized = np.zeros((len(models), len(RADAR_METRICS)))
    labels: List[str] = []
    higher: List[bool] = []
    for j, (key, label, high, scale, _unit) in enumerate(RADAR_METRICS):
        labels.append(label)
        higher.append(high)
        for i, model in enumerate(models):
            raw = summaries.get(model, {}).get(key)
            value = 0.0 if raw is None else float(raw) / scale
            value = min(max(value, 0.0), 1.0)
            normalized[i, j] = value if high else 1.0 - value
    return normalized, labels, higher


# ═══════════════════════════════════════════════════════════════════════════
# fig01 - methodology / pipeline diagram
# ═══════════════════════════════════════════════════════════════════════════

def fig01_methodology(
    out_dir: Path,
    model_labels: Sequence[Dict[str, str]],
    n_problems: int = 12,
    trials: int = 1,
    strategy: str = "zero_shot",
    benchmark: str = "ManiBench",
    quant_label: str = "FP16",
) -> List[Path]:
    """Schematic of the evaluation pipeline: datasets -> models -> metrics -> paper."""
    fig, ax = plt.subplots(figsize=(13.6, 5.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.grid(False)

    def box(x, y, w, h, title, body="", face="#FFFFFF", edge="#333333",
            title_size=8.4, body_size=7.2, title_dy=0.055):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.018",
            linewidth=0.9, edgecolor=edge, facecolor=face, zorder=2,
        ))
        ax.text(x + w / 2, y + h - title_dy * 0.55, title, ha="center", va="top",
                fontsize=title_size, fontweight="bold", zorder=3)
        if body:
            ax.text(x + w / 2, y + h - title_dy - 0.055, body, ha="center", va="top",
                    fontsize=body_size, color="#333333", zorder=3, linespacing=1.45)

    def arrow(x1, y1, x2, y2, rad=0.0):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10,
            linewidth=1.1, color="#4A4A4A", zorder=1, connectionstyle=f"arc3,rad={rad}",
        ))

    def column_header(x_centre, text):
        ax.text(x_centre, 0.975, text, ha="center", va="bottom",
                fontsize=9.2, fontweight="bold", color="#222222")

    # ── Column geometry (x, width) ─────────────────────────────────────────
    C1, W1 = 0.005, 0.160     # benchmark
    C2, W2 = 0.200, 0.196     # model variants
    C3, W3 = 0.432, 0.150     # generation
    C4, W4 = 0.618, 0.200     # metrics
    C5, W5 = 0.855, 0.140     # analysis / artifacts

    # ── Column 1: benchmark ───────────────────────────────────────────────
    # Keep the box title short enough for the column width: drop a trailing
    # parenthetical (e.g. "ManiBench (pilot v1.0)" -> "ManiBench").
    bench_title = re.sub(r"\s*\(.*?\)\s*", " ", str(benchmark)).strip()[:18] or "ManiBench"
    box(C1, 0.30, W1, 0.40, f"{bench_title} pilot suite",
        f"{n_problems} Manim tasks\nphysics - calculus\nlinear algebra - ML\n"
        f"probability - geometry\n\n{trials} trial(s) per task\nprompt: {strategy}",
        face="#F2F6FB", edge="#4C72B0")
    column_header(C1 + W1 / 2, "Benchmark")

    # ── Column 2: model variants ──────────────────────────────────────────
    n = max(len(model_labels), 1)
    height = min(0.215, 0.74 / n)
    gap = (0.76 - n * height) / max(n - 1, 1) if n > 1 else 0.0
    top = 0.915
    model_centres: List[float] = []
    for index, spec in enumerate(model_labels):
        y = top - (index + 1) * height - index * gap
        model_centres.append(y + height / 2)
        box(C2, y, W2, height, spec.get("label", "model"), spec.get("sublabel", ""),
            face=spec.get("face", "#FFFFFF"), edge=spec.get("edge", "#4C72B0"),
            title_size=8.4, body_size=6.9, title_dy=0.05)
        arrow(C1 + W1 + 0.004, 0.50, C2 - 0.006, model_centres[-1])
    column_header(C2 + W2 / 2, "Model variants")

    # ── Column 3: generation ──────────────────────────────────────────────
    box(C3, 0.325, W3, 0.35, "Generation",
        f"chat template\ntemperature 0\nstreaming decode\n{quant_label}\nsharded over 2x T4",
        face="#FFF8F0", edge="#DD8452")
    column_header(C3 + W3 / 2, "Inference")
    for centre in model_centres:
        arrow(C2 + W2 + 0.004, centre, C3 - 0.006, 0.50)

    # ── Column 4: metrics ─────────────────────────────────────────────────
    box(C4, 0.575, W4, 0.29, "Executability (Pass@1)",
        "headless Manim CE render\nsandboxed subprocess\nrender timeout +\nscene detection",
        face="#F4FAF4", edge="#55A868")
    box(C4, 0.145, W4, 0.325, "Static & semantic metrics",
        "VCER: legacy ManimGL API\nMAS: math annotation density\n"
        "CMI: Radon complexity\nCAR: visual constraints\n"
        "Alignment + coverage:\nAST choreography detection",
        face="#F4FAF4", edge="#55A868")
    column_header(C4 + W4 / 2, "Evaluation metrics")
    arrow(C3 + W3 + 0.004, 0.50, C4 - 0.006, 0.72)
    arrow(C3 + W3 + 0.004, 0.50, C4 - 0.006, 0.31)

    # ── Column 5: statistics -> artifacts ─────────────────────────────────
    box(0.842, 0.415, 0.152, 0.40, "Statistics",
        "paired per-task tests\n(Wilcoxon)\nbootstrap 95% CIs\nHolm-Bonferroni\n"
        "Spearman correlations",
        face="#F7F4FB", edge="#8172B3")
    column_header(0.918, "Analysis")
    arrow(C4 + W4 + 0.004, 0.72, 0.838, 0.66)
    arrow(C4 + W4 + 0.004, 0.31, 0.838, 0.52)
    arrow(0.918, 0.41, 0.918, 0.295)

    ax.text(0.918, 0.275,
            "Paper artifacts\nLaTeX tables (booktabs)\n300-DPI vector figures\n"
            "CSV + narrative report\nreproducibility manifest",
            ha="center", va="top", fontsize=7.0, style="italic", color="#333333",
            linespacing=1.4)

    fig.suptitle("ManiBench evaluation methodology: from benchmark tasks to publication artifacts",
                 fontweight="bold", y=0.995)
    return _save(fig, out_dir, "fig01_methodology")


# ═══════════════════════════════════════════════════════════════════════════
# fig02 - headline leaderboard
# ═══════════════════════════════════════════════════════════════════════════

def fig02_leaderboard(summaries: Dict[str, Dict[str, Any]], out_dir: Path) -> List[Path]:
    models = _sorted_models(summaries)
    colors = model_colors(summaries)
    labels = [_short(m) for m in models]
    y = np.arange(len(models))[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 0.62 * len(models) + 2.0), sharey=True)

    pass_rate = [float(summaries[m].get("pass_rate_pct") or 0.0) for m in models]
    pass_std = [float(summaries[m].get("exec_std") or 0.0) * 100 for m in models]
    vcer = [float(summaries[m].get("vcer_pct") or 0.0) for m in models]
    vcer_std = [float(summaries[m].get("vcer_std") or 0.0) * 100 for m in models]

    ax = axes[0]
    ax.barh(y, pass_rate, xerr=pass_std, color=[colors[m] for m in models],
            height=0.62, edgecolor="black", linewidth=0.5,
            error_kw=dict(ecolor="#333333", elinewidth=0.8, capsize=2.5))
    for yi, value, err in zip(y, pass_rate, pass_std):
        ax.text(min(value + err + 2.5, 108), yi, f"{value:.1f}%", va="center",
                fontsize=8.4, fontweight="bold")
    ax.set_xlim(0, 118)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Pass@1 executability (%, higher is better)")
    ax.set_title("A. Executability", fontweight="bold", loc="left")
    ax.axvline(np.mean(pass_rate), color="#666666", linestyle=":", linewidth=0.9)
    ax.text(np.mean(pass_rate), len(models) - 0.3, f"mean {np.mean(pass_rate):.1f}%",
            fontsize=7.4, color="#555555", ha="center", va="bottom")

    ax = axes[1]
    ax.barh(y, vcer, xerr=vcer_std, color=[colors[m] for m in models],
            height=0.62, edgecolor="black", linewidth=0.5, hatch="///", alpha=0.92,
            error_kw=dict(ecolor="#333333", elinewidth=0.8, capsize=2.5))
    for yi, value, err in zip(y, vcer, vcer_std):
        ax.text(min(value + err + 1.8, 96), yi, f"{value:.1f}%", va="center", fontsize=8.4)
    ax.set_xlim(0, 105)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Version-conflict error rate (%, lower is better)")
    ax.set_title("B. ManimGL / legacy-API hallucination", fontweight="bold", loc="left")

    ax = axes[0]
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    for tick, model in zip(ax.get_yticklabels(), models):
        method = summaries[model].get("training_method", "")
        tick.set_text(f"{tick.get_text()}  [{method}]")

    handles = [Line2D([0], [0], marker="s", color="w", markerfacecolor=colors[m],
                      markersize=7, label=_short(m)) for m in models]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(models), 4),
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("ManiBench headline leaderboard (error bars: 1 SD across tasks)",
                 fontweight="bold", y=1.0)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    return _save(fig, out_dir, "fig02_leaderboard")


# ═══════════════════════════════════════════════════════════════════════════
# fig03 - metric radar
# ═══════════════════════════════════════════════════════════════════════════

def fig03_radar(summaries: Dict[str, Dict[str, Any]], out_dir: Path) -> List[Path]:
    models = _sorted_models(summaries)
    colors = model_colors(summaries)
    matrix, labels, higher = metric_matrix(summaries, models)

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6.4, 5.6), subplot_kw=dict(polar=True))
    for i, model in enumerate(models):
        values = matrix[i].tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1.7, color=colors[model], label=_short(model))
        ax.fill(angles, values, color=colors[model], alpha=0.13)
        ax.scatter(angles[:-1], matrix[i], s=11, color=colors[model], zorder=5)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f"{lbl}\n({'>' if h else '<'} better)" for lbl, h in zip(labels, higher)],
                       fontsize=7.8)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=7, color="#666666")
    ax.set_title("Normalised metric profile (1.0 = best observed range on every axis)",
                 fontweight="bold", pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.10))
    return _save(fig, out_dir, "fig03_metric_radar")


# ═══════════════════════════════════════════════════════════════════════════
# fig04 - syntactic drift tradeoff (Pareto)
# ═══════════════════════════════════════════════════════════════════════════

def fig04_pareto(summaries: Dict[str, Dict[str, Any]], out_dir: Path) -> List[Path]:
    models = _sorted_models(summaries)
    colors = model_colors(summaries)

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.axvspan(0, 10, color="#55A868", alpha=0.07, zorder=0)
    ax.text(0.6, 101, "preferred region\n(low drift, high executability)",
            fontsize=7.4, color="#2E6B42", style="italic", va="top")

    xs, ys = [], []
    for model in models:
        x = float(summaries[model].get("vcer_pct") or 0.0)
        y = float(summaries[model].get("pass_rate_pct") or 0.0)
        coverage = float(summaries[model].get("coverage_mean") or 0.0)
        xs.append(x)
        ys.append(y)
        ax.scatter(x, y, s=90 + 420 * coverage, color=colors[model],
                   edgecolors="black", linewidths=0.6, alpha=0.9, zorder=4,
                   label=f"{_short(model)} (coverage {coverage:.2f})")
        ax.annotate(_short(model), (x, y), xytext=(7, -4), textcoords="offset points",
                    fontsize=7.6, zorder=5)

    # Pareto frontier: no other point has both lower VCER and higher Pass@1
    frontier_idx = []
    for i, (x, y) in enumerate(zip(xs, ys)):
        dominated = any((xs[j] <= x and ys[j] >= y) and (xs[j] < x or ys[j] > y)
                        for j in range(len(xs)) if j != i)
        if not dominated:
            frontier_idx.append(i)
    if len(frontier_idx) > 1:
        order = sorted(frontier_idx, key=lambda i: xs[i])
        ax.plot([xs[i] for i in order], [ys[i] for i in order],
                linestyle="--", linewidth=1.1, color="#333333", alpha=0.7,
                label="Pareto frontier", zorder=3)

    ax.set_xlabel("Version-conflict error rate (%) - lower is better ->")
    ax.set_ylabel("Pass@1 executability (%) - higher is better ^")
    ax.set_xlim(-1, max(12.0, max(xs) * 1.35 + 1 if xs else 12))
    ax.set_ylim(max(0.0, min(ys) - 8) if ys else 0, 106)
    ax.set_title("Syntactic drift vs. executability (bubble = pedagogical coverage)",
                 fontweight="bold")
    ax.legend(loc="lower left", fontsize=7.6)
    return _save(fig, out_dir, "fig04_drift_pareto")


# ═══════════════════════════════════════════════════════════════════════════
# fig05 - per-problem heatmaps
# ═══════════════════════════════════════════════════════════════════════════

def fig05_problem_heatmap(
    records: Sequence[Dict[str, Any]],
    summaries: Dict[str, Dict[str, Any]],
    out_dir: Path,
) -> List[Path]:
    models = _sorted_models(summaries)
    problems = sorted({str(r.get("problem_id")) for r in records if r.get("problem_id")})
    if not problems:
        return []

    exec_matrix = per_problem_matrix(records, models, problems, "executability", 100.0)
    vcer_matrix = per_problem_matrix(records, models, problems, "vcer", 100.0)

    fig, axes = plt.subplots(2, 1, figsize=(max(7.4, 0.62 * len(problems) + 2.4),
                                            1.15 * len(models) + 4.0))
    for panel, (ax, matrix, title, cmap) in enumerate((
        (axes[0], exec_matrix, "A. Pass@1 executability (%) - higher is better", "YlGn"),
        (axes[1], vcer_matrix, "B. Version-conflict error rate (%) - lower is better", "OrRd"),
    )):
        data = np.ma.masked_invalid(matrix)
        im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=0, vmax=100)
        ax.set_xticks(range(len(problems)))
        ax.set_xticklabels(problems, rotation=45, ha="right", fontsize=7.6)
        if panel == 0:
            # the shared x axis is only labelled once, at the bottom
            ax.tick_params(labelbottom=False, bottom=False)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels([_short(m) for m in models], fontsize=8)
        ax.set_title(title, fontweight="bold", loc="left", fontsize=9.8)
        ax.grid(False)
        for i in range(len(models)):
            for j in range(len(problems)):
                value = matrix[i, j]
                if np.isnan(value):
                    ax.text(j, i, "-", ha="center", va="center", fontsize=7, color="#999999")
                else:
                    ax.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=7,
                            color="black" if value < 65 else "white")
        fig.colorbar(im, ax=ax, fraction=0.022, pad=0.012).ax.tick_params(labelsize=7)

    fig.suptitle("Per-task performance landscape", fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=2.2)
    return _save(fig, out_dir, "fig05_problem_heatmap")


# ═══════════════════════════════════════════════════════════════════════════
# fig06 - domain breakdown
# ═══════════════════════════════════════════════════════════════════════════

def fig06_domain(
    domain_summaries: Dict[str, Dict[str, Dict[str, float]]],
    out_dir: Path,
) -> List[Path]:
    models = [m for m in domain_summaries if domain_summaries[m]]
    if not models:
        return []
    domains = sorted({d for m in models for d in domain_summaries[m]})
    if not domains:
        return []

    colors = {}
    for index, model in enumerate(models):
        colors[model] = PALETTE[index % len(PALETTE)]

    x = np.arange(len(domains))
    width = 0.8 / max(len(models), 1)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for ax, metric, scale, title in (
        (axes[0], "pass_rate", 1.0, "A. Pass@1 executability by domain (%)"),
        (axes[1], "vcer", 1.0, "B. Version-conflict rate by domain (%)"),
    ):
        for i, model in enumerate(models):
            values = [domain_summaries[model].get(d, {}).get(metric, 0.0) for d in domains]
            offset = (i - (len(models) - 1) / 2) * width
            bars = ax.bar(x + offset, values, width * 0.92, label=_short(model),
                          color=colors[model], edgecolor="black", linewidth=0.4)
            for rect, value in zip(bars, values):
                if value > 0:
                    ax.text(rect.get_x() + rect.get_width() / 2, value + 1.2, f"{value:.0f}",
                            ha="center", fontsize=6.6)
        ax.set_xticks(x)
        ax.set_xticklabels([d[:14] for d in domains], rotation=30, ha="right", fontsize=7.4)
        ax.set_ylim(0, 112)
        ax.set_ylabel("Percent (%)")
        ax.set_title(title, fontweight="bold", loc="left", fontsize=9.8)
    axes[0].legend(fontsize=7.6, ncol=min(len(models), 3))
    fig.suptitle("Domain-level breakdown (bars annotated with raw values)", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, out_dir, "fig06_domain_breakdown")


# ═══════════════════════════════════════════════════════════════════════════
# fig07 - failure taxonomy
# ═══════════════════════════════════════════════════════════════════════════

def fig07_errors(
    error_taxonomy: Dict[str, Dict[str, int]],
    summaries: Dict[str, Dict[str, Any]],
    out_dir: Path,
) -> List[Path]:
    models = [m for m in _sorted_models(summaries) if m in error_taxonomy]
    if not models:
        return []

    types = sorted({t for m in models for t in error_taxonomy[m]})
    if not types:
        return []

    totals = {m: max(sum(error_taxonomy[m].values()), 1) for m in models}
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(max(6.6, 1.5 * len(models) + 2.6), 4.4))
    bottom = np.zeros(len(models))
    for index, err in enumerate(types):
        shares = np.array([100.0 * error_taxonomy[m].get(err, 0) / totals[m] for m in models])
        ax.bar(x, shares, 0.6, bottom=bottom, label=err,
               color=PALETTE[index % len(PALETTE)], edgecolor="white", linewidth=0.5)
        for xi, (share, base) in enumerate(zip(shares, bottom)):
            if share >= 5:
                ax.text(xi, base + share / 2, f"{share:.0f}%", ha="center", va="center",
                        fontsize=7, color="white", fontweight="bold")
        bottom += shares

    ax.set_xticks(x)
    ax.set_xticklabels([f"{_short(m)}\n(n={totals[m]})" for m in models], fontsize=8)
    ax.set_ylabel("Share of runs (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Failure taxonomy: what actually breaks the generated Manim code",
                 fontweight="bold")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7.6)
    fig.tight_layout()
    return _save(fig, out_dir, "fig07_error_taxonomy")


# ═══════════════════════════════════════════════════════════════════════════
# fig08 - significance forest plot
# ═══════════════════════════════════════════════════════════════════════════

def fig08_significance(
    comparisons: Sequence[Dict[str, Any]],
    baseline_label: str,
    out_dir: Path,
    metric_label: str = "Pass@1 executability (percentage points)",
) -> List[Path]:
    comparisons = [c for c in comparisons if c.get("mean_diff") is not None]
    if not comparisons:
        return []

    labels = [_short(c["model"]) for c in comparisons]
    diffs = [c["mean_diff"] * 100.0 for c in comparisons]
    lows = [(c["mean_diff"] - c.get("ci_low", c["mean_diff"])) * 100.0 for c in comparisons]
    highs = [(c.get("ci_high", c["mean_diff"]) - c["mean_diff"]) * 100.0 for c in comparisons]

    fig, ax = plt.subplots(figsize=(7.8, 0.62 * len(comparisons) + 2.3))
    y = np.arange(len(comparisons))[::-1]
    ax.axvline(0, color="#333333", linewidth=1.0, linestyle="-")
    span = max(abs(min(diffs + [0])), abs(max(diffs + [0])), 5.0)
    ax.axvspan(-0.05 * span, 0.05 * span, color="#999999", alpha=0.10, zorder=0)

    for yi, diff, low, high, comp in zip(y, diffs, lows, highs, comparisons):
        color = "#2E6B42" if diff > 0 else ("#A33A3A" if diff < 0 else "#666666")
        ax.errorbar(diff, yi, xerr=[[low], [high]], fmt="o", color=color,
                    ecolor=color, elinewidth=1.2, capsize=3.2, markersize=6.5, zorder=4)
        p = comp.get("p_value")
        p_txt = "n/a" if p is None else ("p<0.001" if p < 0.001 else f"p={p:.3f}")
        stars = "" if p is None else ("***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s.")
        # Annotate in whichever half is empty, so text never collides with the CI bar.
        if diff >= 0:
            ax.text(-span * 1.20, yi, f"{diff:+.1f} pp   {p_txt} {stars}", va="center",
                    ha="left", fontsize=7.8, zorder=5)
        else:
            ax.text(span * 1.20, yi, f"{diff:+.1f} pp   {p_txt} {stars}", va="center",
                    ha="right", fontsize=7.8, zorder=5)

    ax.set_yticks(y)
    ax.set_yticklabels([_short(c["model"]) for c in comparisons], fontsize=8.2)
    ax.set_xlabel(metric_label + "   (paired mean difference vs. baseline, bootstrap 95% CI)")
    ax.set_title(f"Statistical significance relative to {baseline_label}",
                 fontweight="bold")
    ax.set_xlim(-span * 1.30, span * 1.30)
    ax.set_xticks([t for t in ax.get_xticks() if -span * 1.3 <= t <= span * 1.3])
    fig.tight_layout()
    return _save(fig, out_dir, "fig08_significance_forest")


# ═══════════════════════════════════════════════════════════════════════════
# fig09 - metric correlation
# ═══════════════════════════════════════════════════════════════════════════

def fig09_metric_correlation(records: Sequence[Dict[str, Any]], out_dir: Path) -> List[Path]:
    keys = ["executability", "vcer", "mas", "cmi", "car", "alignment_score", "coverage_score"]
    labels = ["Pass@1", "VCER", "MAS", "CMI", "CAR", "Alignment", "Coverage"]

    columns: List[List[float]] = []
    keep: List[int] = []
    for index, key in enumerate(keys):
        values = [float(r[key]) for r in records if r.get(key) is not None]
        if len(values) >= 3 and float(np.std(values)) > 1e-9:
            columns.append(values)
            keep.append(index)
    if len(keep) < 2:
        return []

    # Align rows: keep only records where every retained metric is present
    rows: List[List[float]] = []
    for record in records:
        if all(record.get(keys[i]) is not None for i in keep):
            rows.append([float(record[keys[i]]) for i in keep])
    if len(rows) < 4:
        return []

    matrix = np.array(rows)
    n = len(keep)
    corr = np.eye(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                a, b = matrix[:, i], matrix[:, j]
                if np.std(a) < 1e-12 or np.std(b) < 1e-12:
                    corr[i, j] = 0.0
                else:
                    corr[i, j] = float(np.corrcoef(a, b)[0, 1])

    fig, ax = plt.subplots(figsize=(1.0 * n + 2.4, 1.0 * n + 2.0))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    names = [labels[i] for i in keep]
    ax.set_xticks(range(n))
    ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(names, fontsize=8)
    ax.grid(False)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{corr[i, j]:+.2f}", ha="center", va="center", fontsize=7.4,
                    color="white" if abs(corr[i, j]) > 0.55 else "black")
    ax.set_title(f"Metric correlation across {len(rows)} evaluated runs (Pearson r)",
                 fontweight="bold", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.042, pad=0.03).ax.tick_params(labelsize=7)
    fig.tight_layout()
    return _save(fig, out_dir, "fig09_metric_correlation")


# ═══════════════════════════════════════════════════════════════════════════
# Contact sheet
# ═══════════════════════════════════════════════════════════════════════════

def build_contact_sheet(out_dir: Path, stems: Sequence[str], name: str = "figure_contact_sheet") -> List[Path]:
    """2-column montage of the generated PNGs, for quick review and appendices."""
    available = [out_dir / f"{s}.png" for s in stems]
    available = [p for p in available if p.exists()]
    if not available:
        return []

    columns = 2
    rows = math.ceil(len(available) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(13.0, 5.4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, path in zip(axes, available):
        ax.imshow(plt.imread(path))
        ax.set_title(path.stem, fontsize=9, fontweight="bold")
        ax.axis("off")
    for ax in axes[len(available):]:
        ax.axis("off")
    fig.suptitle("ManiBench publication figure set", fontweight="bold", y=0.999)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    return _save(fig, out_dir, name)


# ═══════════════════════════════════════════════════════════════════════════
# Orchestration
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_STEMS = [
    "fig01_methodology", "fig02_leaderboard", "fig03_metric_radar", "fig04_drift_pareto",
    "fig05_problem_heatmap", "fig06_domain_breakdown", "fig07_error_taxonomy",
    "fig08_significance_forest", "fig09_metric_correlation",
]


def build_all(
    output_dir: Path,
    summaries: Dict[str, Dict[str, Any]],
    records: Sequence[Dict[str, Any]],
    domain_summaries: Optional[Dict[str, Any]] = None,
    error_taxonomy: Optional[Dict[str, Any]] = None,
    comparisons: Optional[Sequence[Dict[str, Any]]] = None,
    baseline: Optional[str] = None,
    model_labels: Optional[Sequence[Dict[str, str]]] = None,
    meta: Optional[Dict[str, Any]] = None,
    contact_sheet: bool = True,
) -> List[Path]:
    """
    Generate the complete publication figure set.

    `comparisons` comes from `scripts.publication_report.paired_comparisons`;
    when omitted, fig08 is skipped. Every figure is independent: one failing
    figure never prevents the others from being written.
    """
    meta = meta or {}
    figures_dir = Path(output_dir) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    def attempt(label: str, fn, *args, **kwargs) -> None:
        try:
            result = fn(*args, **kwargs)
            paths = result if isinstance(result, list) else [result]
            paths = [p for p in paths if p]
            if not paths:
                print(f"    {label:<26} -- skipped (not enough variance/data for this figure)")
                return
            written.extend(paths)
            print(f"    {label:<26} -> {', '.join(p.name for p in paths)}")
        except Exception as exc:  # noqa: BLE001 - one bad figure must not kill the bundle
            print(f"    {label:<26} !! skipped ({type(exc).__name__}: {exc})")

    if model_labels is None:
        model_labels = [
            {
                "label": _short(name),
                "sublabel": f"{summaries[name].get('training_method', '')} / "
                            f"{summaries[name].get('format', '')}",
                "edge": model_colors(summaries).get(name, "#4C72B0"),
                "face": "#FFFFFF",
            }
            for name in _sorted_models(summaries)
        ]

    attempt("fig01 methodology", fig01_methodology, figures_dir, model_labels,
            n_problems=int(meta.get("n_problems", 12) or 12),
            trials=int(meta.get("trials", 1) or 1),
            strategy=str(meta.get("strategy", "zero_shot")),
            benchmark=str(meta.get("benchmark", "ManiBench")),
            quant_label=precision_label(meta.get("weight_precision")))
    if summaries:
        attempt("fig02 leaderboard", fig02_leaderboard, summaries, figures_dir)
        attempt("fig03 metric radar", fig03_radar, summaries, figures_dir)
        attempt("fig04 drift/pareto", fig04_pareto, summaries, figures_dir)
        attempt("fig05 problem heatmap", fig05_problem_heatmap, records, summaries, figures_dir)
        attempt("fig06 domain breakdown", fig06_domain,
                domain_summaries if domain_summaries is not None
                else {m: {} for m in summaries}, figures_dir)
        attempt("fig07 error taxonomy", fig07_errors,
                error_taxonomy or {}, summaries, figures_dir)
        attempt("fig09 metric correlation", fig09_metric_correlation, records, figures_dir)
    if comparisons and baseline:
        attempt("fig08 significance", fig08_significance, comparisons, _short(baseline), figures_dir)

    if contact_sheet and written:
        stems = [p.stem for p in written if p.suffix == ".png"]
        attempt("contact sheet", build_contact_sheet, figures_dir, stems)

    return written


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Generate the ManiBench publication figure set")
    parser.add_argument("--results-json", required=True, help="benchmark_run_*.json or checkpoint")
    parser.add_argument("--output-dir", default="figures")
    args = parser.parse_args()

    data = json.loads(Path(args.results_json).read_text(encoding="utf-8"))
    summaries = data.get("model_summaries", {})
    records = data.get("detailed_records", [])
    out = Path(args.output_dir)
    written = build_all(out, summaries, records, meta={"trials": data.get("metadata", {}).get("trials", 1)})
    print(f"\n{len(written)} figure file(s) written to {out.resolve()}")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
