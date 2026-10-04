#!/usr/bin/env python3
"""
ManiBench - Publication-Quality Figures Generator
=================================================

Thin compatibility CLI over the publication figure suite in
`scripts/publication_figures.py` (single source of truth for styling and layout).

    python scripts/generate_paper_plots.py --results-json results/<run>/benchmark_run_*.json \
                                           --output-dir results/<run>/figures

The legacy entry points `plot_vcer_vs_executability` and
`plot_family_comparison` are kept so existing callers (e.g. the Kaggle runner)
keep working; they now delegate to the shared, upgraded implementations.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Allow `python scripts/generate_paper_plots.py` (sys.path[0] is scripts/, not the
# repo root) as well as `import scripts.generate_paper_plots` from the runner.
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.publication_figures import (  # noqa: F401  (re-exported API)
    build_contact_sheet,
    fig02_leaderboard,
    fig03_radar,
    fig04_pareto,
    fig06_domain,
    fig09_metric_correlation,
    model_colors,
    per_problem_matrix,
)

# Historically these two functions produced fig1/fig2; keep the names.
plot_vcer_vs_executability = fig04_pareto
plot_family_comparison = fig02_leaderboard
plot_metric_radar = fig03_radar
plot_domain_breakdown = fig06_domain
plot_metric_correlation = fig09_metric_correlation


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate ManiBench publication figures")
    parser.add_argument("--results-json", required=True, help="Path to benchmark_run_*.json")
    parser.add_argument("--output-dir", default="figures", help="Output directory for figures")
    parser.add_argument("--legacy-two-figures", action="store_true",
                        help="Only emit fig04 (drift/pareto) and fig02 (leaderboard)")
    args = parser.parse_args()

    json_path = Path(args.results_json)
    if not json_path.exists():
        print(f"ERROR: results file not found at: {json_path}")
        return 1

    data = json.loads(json_path.read_text(encoding="utf-8"))
    summaries: Dict[str, Dict[str, Any]] = data.get("model_summaries", {})
    records: List[Dict[str, Any]] = data.get("detailed_records", [])
    metadata = data.get("metadata", {})
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    formatter = None
    if not summaries and records:
        # Tolerate archives that only carry per-trial records (e.g. a checkpoint
        # export) by deriving the model-level summaries here.
        from scripts.paper_formatter import PaperFormatter
        formatter = PaperFormatter(raw_records=records, metadata=metadata)
        summaries = formatter.model_summaries

    if not summaries:
        print("No model summaries could be derived from the JSON - nothing to plot.")
        return 1

    written: List[Path] = []
    if args.legacy_two_figures:
        written += fig04_pareto(summaries, out)
        written += fig02_leaderboard(summaries, out)
    else:
        from scripts.publication_figures import build_all
        from scripts.publication_report import paired_comparisons

        # Pick a baseline for the significance figure: an explicit baseline family
        # first, otherwise the weakest model by Pass@1.
        baseline = None
        for name, summary in summaries.items():
            if str(summary.get("family", "")).lower() == "baseline" or \
               str(summary.get("training_method", "")).lower() == "base":
                baseline = name
                break
        if baseline is None and summaries:
            baseline = min(summaries, key=lambda n: float(summaries[n].get("pass_rate_pct") or 0.0))
        comparisons = paired_comparisons(records, baseline=baseline) if baseline else []

        # `build_all` writes into <base>/figures - so a caller passing
        # ".../figures" gets ".../figures", and a caller passing the run directory
        # gets "<run>/figures".
        base = out.parent if out.name == "figures" else out
        written = build_all(
            output_dir=base,
            summaries=summaries,
            records=records,
            domain_summaries=getattr(formatter, "domain_summaries", None),
            error_taxonomy=getattr(formatter, "error_taxonomy", None),
            comparisons=comparisons,
            baseline=baseline,
            meta={"trials": metadata.get("trials", 1),
                  "strategy": metadata.get("strategy", "zero_shot"),
                  "skip_render": metadata.get("skip_render", False)},
        )

    target = written[0].parent if written else out
    print(f"{len(written)} figure file(s) written to: {target.resolve()}")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
