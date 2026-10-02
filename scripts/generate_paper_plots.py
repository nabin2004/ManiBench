#!/usr/bin/env python3
"""
ManiBench — Publication-Quality Figures Generator for Research Papers
====================================================================
Generates camera-ready vector PDFs and 300-DPI PNGs from benchmark outputs:
  1. Figure 1: Syntactic Drift Tradeoff (VCER vs. Executability Scatter Plot with Pareto Frontier)
  2. Figure 2: Model Family Performance Comparison (Grouped Bar Chart)
  3. Figure 3: Pedagogical Coverage Dimensions Radar Chart
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Publication style configuration
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
})

FAMILY_COLORS = {
    "qwen-manimator": "#1f77b4",  # Blue
    "aos-qwen3": "#ff7f0e",       # Orange
    "aos-qwen2.5": "#2ca02c",     # Green
    "gemma": "#d62728",           # Red
    "baseline": "#7f7f7f",        # Gray
    "custom": "#9467bd",          # Purple
}

METHOD_MARKERS = {
    "Base": "o",
    "SFT": "s",
    "DPO": "^",
    "GRPO": "D",
    "Custom": "v",
}


def plot_vcer_vs_executability(model_summaries: Dict[str, Dict[str, Any]], out_dir: Path) -> None:
    """Figure 1: VCER vs Executability Scatter Plot."""
    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    for name, s in model_summaries.items():
        fam = s.get("family", "custom")
        color = FAMILY_COLORS.get(fam, "#333333")
        method = s.get("training_method", "SFT")
        marker = METHOD_MARKERS.get(method, "o")

        x = s.get("vcer_pct", s.get("vcer_mean", 0.0) * 100.0)
        y = s.get("pass_rate_pct", s.get("executability_mean", 0.0) * 100.0)

        ax.scatter(x, y, color=color, marker=marker, s=80, alpha=0.85, edgecolors="k", linewidth=0.6, zorder=4)
        ax.annotate(
            name.replace("AOS-", "").replace("nabin2004/", "")[:18],
            (x, y),
            fontsize=7,
            xytext=(4, 2),
            textcoords="offset points",
            alpha=0.85,
        )

    ax.set_xlabel("Version-Conflict Error Rate (VCER) % [Lower is Better] →")
    ax.set_ylabel("Pass@1 Executability % [Higher is Better] ↑")
    ax.set_title("Syntactic Drift vs. Executability in Manim Code Generation", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4, zorder=0)

    # Highlight optimal region (top-left)
    ax.axvspan(0, 15, ymin=0.7, ymax=1.0, color="#2ca02c", alpha=0.08, zorder=1)
    ax.text(2, 92, "Optimal Frontier\n(High Exec, Low VCER)", color="#1b5e20", fontsize=8, style="italic")

    # Legend for families
    from matplotlib.lines import Line2D
    family_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=col, markersize=8, label=fam.upper())
        for fam, col in FAMILY_COLORS.items() if any(s.get("family") == fam for s in model_summaries.values())
    ]
    ax.legend(handles=family_handles, loc="lower right", framealpha=0.9)

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig1_vcer_vs_executability.pdf")
    fig.savefig(out_dir / "fig1_vcer_vs_executability.png")
    plt.close(fig)


def plot_family_comparison(model_summaries: Dict[str, Dict[str, Any]], out_dir: Path) -> None:
    """Figure 2: Clustered Bar Chart of Family Aggregates."""
    family_groups: Dict[str, List[Dict[str, Any]]] = {}
    for s in model_summaries.values():
        family_groups.setdefault(s.get("family", "custom"), []).append(s)

    families = list(family_groups.keys())
    if not families:
        return

    exec_means = [np.mean([s["pass_rate_pct"] for s in family_groups[f]]) for f in families]
    vcer_means = [np.mean([s["vcer_pct"] for s in family_groups[f]]) for f in families]
    align_means = [np.mean([s["alignment_mean"] * 100.0 for s in family_groups[f]]) for f in families]
    cov_means = [np.mean([s["coverage_mean"] * 100.0 for s in family_groups[f]]) for f in families]

    x = np.arange(len(families))
    width = 0.20

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    rects1 = ax.bar(x - 1.5 * width, exec_means, width, label="Pass@1 (%)", color="#1f77b4")
    rects2 = ax.bar(x - 0.5 * width, vcer_means, width, label="VCER (%)", color="#d62728")
    rects3 = ax.bar(x + 0.5 * width, align_means, width, label="Alignment (x100)", color="#2ca02c")
    rects4 = ax.bar(x + 1.5 * width, cov_means, width, label="Coverage (x100)", color="#ff7f0e")

    ax.set_ylabel("Metric Score (%)")
    ax.set_title("Cross-Family Performance Breakdown on ManiBench", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f.upper() for f in families])
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.savefig(out_dir / "fig2_family_comparison.pdf")
    fig.savefig(out_dir / "fig2_family_comparison.png")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate ManiBench publication figures")
    parser.add_argument("--results-json", type=str, required=True, help="Path to benchmark_run_*.json")
    parser.add_argument("--output-dir", type=str, default="figures", help="Output directory for figures")
    args = parser.parse_args()

    json_path = Path(args.results_json)
    if not json_path.exists():
        print(f"ERROR: Results file not found at: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    summaries = data.get("model_summaries", {})
    if not summaries:
        print("No model_summaries found in JSON.")
        return

    out = Path(args.output_dir)
    plot_vcer_vs_executability(summaries, out)
    plot_family_comparison(summaries, out)
    print(f"Publication figures successfully saved to: {out.resolve()}")


if __name__ == "__main__":
    main()
