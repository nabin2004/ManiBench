#!/usr/bin/env python3
"""
ManiBench — W&B Training Dynamics Extraction Pipeline
=====================================================
Directly connects to Weights & Biases API to query CoreWeave training runs,
extracts granular training loss, reward dynamics, entropy, and learning rates,
exports structured CSVs, and produces 300-DPI publication-grade vector graphics.
"""

from __future__ import annotations

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import sys
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Academic publication aesthetic styling
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

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_CSV_DIR = ROOT_DIR / "results" / "wandb_training_curves"
OUTPUT_FIG_DIR = ROOT_DIR / "results" / "wandb_figures"
OUTPUT_CSV_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FIG_DIR.mkdir(parents=True, exist_ok=True)

# Target representative runs identified across the 4 CoreWeave projects
TARGET_RUNS = [
    # SFT Phase
    {
        "project": "aos-qwen-sft",
        "id": "3o121asn",
        "label": "Qwen-Manimator-1 (SFT, 4500 ctx)",
        "phase": "SFT",
        "family": "qwen-manimator",
        "keys": ["train/loss", "train/learning_rate", "train/grad_norm", "train/epoch"]
    },
    {
        "project": "aos-qwen-sft",
        "id": "6ar66uy9",
        "label": "AOS-Qwen2.5-Coder-7B (SFT)",
        "phase": "SFT",
        "family": "aos-qwen2.5",
        "keys": ["train/loss", "train/learning_rate", "train/grad_norm", "train/epoch"]
    },
    {
        "project": "aos-sft",
        "id": "5u1muot6",
        "label": "AOS-Gemma4-Manim (SFT)",
        "phase": "SFT",
        "family": "gemma",
        "keys": ["train/loss", "train/learning_rate", "train/grad_norm"]
    },
    # DPO Phase
    {
        "project": "huggingface",
        "id": "7o2bifgt",
        "label": "AOS-Qwen3-8B (Narrated DPO)",
        "phase": "DPO",
        "family": "aos-qwen3",
        "keys": [
            "train/loss", "train/rewards/chosen", "train/rewards/rejected", 
            "train/rewards/margins", "train/rewards/accuracies", "train/learning_rate"
        ]
    },
    # GRPO Phase
    {
        "project": "aos-grpo",
        "id": "4ejpywei",
        "label": "Qwen-Manimator-1 (GRPO RL)",
        "phase": "GRPO",
        "family": "qwen-manimator",
        "keys": [
            "train/loss", "train/reward", "train/reward_std", 
            "train/rewards/combined_reward/mean", "train/clip_ratio/high_mean"
        ]
    },
    {
        "project": "aos-grpo",
        "id": "pxowg7x7",
        "label": "AOS-Qwen3-8B (GRPO Ensemble)",
        "phase": "GRPO",
        "family": "aos-qwen3",
        "keys": [
            "train/loss", "train/reward", "train/reward_std", 
            "train/rewards/combined_reward/mean", "train/clip_ratio/high_mean"
        ]
    },
]


def extract_wandb_histories(api_key: str) -> Dict[str, pd.DataFrame]:
    """Connects to W&B API and pulls full metric logs for target runs."""
    import wandb
    os.environ["WANDB_API_KEY"] = api_key
    api = wandb.Api()

    extracted_dfs = {}

    print("=" * 75)
    print("  Extracting Empirical Training Trajectories from W&B (CoreWeave)")
    print("=" * 75)

    for spec in TARGET_RUNS:
        run_path = f"nabinoli2004-wiseyak/{spec['project']}/{spec['id']}"
        print(f"Fetching run: {run_path} ({spec['label']})...")
        try:
            run = api.run(run_path)
            history = run.history(samples=5000)
            if history.empty:
                print(f"  [WARN] History empty for {spec['id']}. Scanning scan_history...")
                rows = [row for row in run.scan_history()]
                history = pd.DataFrame(rows)
            
            # Filter to available keys
            present_keys = [k for k in spec["keys"] if k in history.columns]
            if "_step" in history.columns:
                present_keys.insert(0, "_step")
            
            filtered_df = history[present_keys].copy()
            filtered_df["run_id"] = spec["id"]
            filtered_df["run_label"] = spec["label"]
            filtered_df["phase"] = spec["phase"]
            filtered_df["family"] = spec["family"]

            # Save raw CSV
            csv_name = f"{spec['phase'].lower()}_{spec['id']}_{spec['project']}.csv"
            csv_path = OUTPUT_CSV_DIR / csv_name
            filtered_df.to_csv(csv_path, index=False)
            print(f"  [OK] Saved {len(filtered_df)} metric steps -> {csv_path.name}")

            extracted_dfs[spec["id"]] = filtered_df
        except Exception as e:
            print(f"  [FAIL] Failed to fetch {run_path}: {e}")

    return extracted_dfs


def plot_publication_figures(extracted_dfs: Dict[str, pd.DataFrame]) -> None:
    """Generates 300-DPI vector PDF and PNG training curves with print-grade styling."""
    print("\nGenerating publication-quality training dynamics figures...")

    # Palette
    palette = {
        "qwen-manimator": "#1f77b4",  # Blue
        "aos-qwen2.5": "#2ca02c",     # Green
        "aos-qwen3": "#ff7f0e",       # Orange
        "gemma": "#d62728",           # Red
    }

    # ──────────────────────────────────────────────────────────────────────────
    # FIGURE 1: SFT Convergence Dynamics (Loss Trajectories Across Architectures)
    # ──────────────────────────────────────────────────────────────────────────
    sft_runs = [spec for spec in TARGET_RUNS if spec["phase"] == "SFT" and spec["id"] in extracted_dfs]
    if sft_runs:
        fig, ax = plt.subplots(figsize=(6.8, 4.0))
        for spec in sft_runs:
            df = extracted_dfs[spec["id"]]
            loss_cols = [c for c in ["train/loss", "loss", "train_loss"] if c in df.columns]
            if not loss_cols:
                continue
            loss_col = loss_cols[0]
            df = df.dropna(subset=[loss_col])
            if df.empty:
                continue
            steps = df["_step"] if "_step" in df.columns else np.arange(len(df))
            loss = df[loss_col].values
            color = palette.get(spec["family"], "#333333")

            # Raw curve with low alpha
            ax.plot(steps, loss, color=color, alpha=0.25, linewidth=0.8)
            # Exponential moving average / rolling smooth
            smooth_loss = pd.Series(loss).rolling(window=max(3, len(loss)//20), min_periods=1).mean()
            ax.plot(steps, smooth_loss, color=color, linewidth=2.0, label=spec["label"])

        ax.set_xlabel("Optimization Step (Gradient Updates)")
        ax.set_ylabel("Training Cross-Entropy Loss")
        ax.set_title("Supervised Fine-Tuning (SFT) Convergence Across Model Families", fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="upper right", framealpha=0.9)
        fig.savefig(OUTPUT_FIG_DIR / "fig_sft_loss_dynamics.pdf")
        fig.savefig(OUTPUT_FIG_DIR / "fig_sft_loss_dynamics.png")
        plt.close(fig)
        print("  [OK] Saved: fig_sft_loss_dynamics.pdf / .png")

    # ──────────────────────────────────────────────────────────────────────────
    # FIGURE 2: GRPO Reinforcement Learning Reward Progression & Variance
    # ──────────────────────────────────────────────────────────────────────────
    grpo_runs = [spec for spec in TARGET_RUNS if spec["phase"] == "GRPO" and spec["id"] in extracted_dfs]
    if grpo_runs:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2))

        for spec in grpo_runs:
            df = extracted_dfs[spec["id"]]
            steps = df["_step"] if "_step" in df.columns else np.arange(len(df))
            color = palette.get(spec["family"], "#333333")

            # Left Panel: Mean Reward Convergence
            reward_col = "train/rewards/combined_reward/mean" if "train/rewards/combined_reward/mean" in df.columns else "train/reward"
            if reward_col in df.columns:
                rewards = df[reward_col].dropna().values
                r_steps = steps[:len(rewards)]
                ax1.plot(r_steps, rewards, color=color, alpha=0.3, linewidth=0.8)
                smooth_r = pd.Series(rewards).rolling(window=max(3, len(rewards)//15), min_periods=1).mean()
                ax1.plot(r_steps, smooth_r, color=color, linewidth=2.0, label=spec["label"])

            # Right Panel: Objective Loss & PPO/GRPO Clipping
            if "train/loss" in df.columns:
                loss = df["train/loss"].dropna().values
                l_steps = steps[:len(loss)]
                smooth_l = pd.Series(loss).rolling(window=max(3, len(loss)//15), min_periods=1).mean()
                ax2.plot(l_steps, smooth_l, color=color, linewidth=2.0, label=spec["label"])

        ax1.set_xlabel("GRPO Policy Step")
        ax1.set_ylabel("Normalized Aggregate Reward $R(\\hat{y})$")
        ax1.set_title("GRPO Policy Reward Convergence (AST + CE Executability)", fontweight="bold")
        ax1.grid(True, linestyle="--", alpha=0.4)
        ax1.legend(loc="lower right", framealpha=0.9)

        ax2.set_xlabel("GRPO Policy Step")
        ax2.set_ylabel("Surrogate Policy Loss $\\mathcal{L}_{\\text{GRPO}}$")
        ax2.set_title("Policy Gradient Loss Dynamics under $\\varepsilon=0.2$ Clipping", fontweight="bold")
        ax2.grid(True, linestyle="--", alpha=0.4)
        ax2.legend(loc="upper right", framealpha=0.9)

        fig.tight_layout()
        fig.savefig(OUTPUT_FIG_DIR / "fig_grpo_reward_dynamics.pdf")
        fig.savefig(OUTPUT_FIG_DIR / "fig_grpo_reward_dynamics.png")
        plt.close(fig)
        print("  [OK] Saved: fig_grpo_reward_dynamics.pdf / .png")

    # ──────────────────────────────────────────────────────────────────────────
    # FIGURE 3: DPO Margin Separation Dynamics
    # ──────────────────────────────────────────────────────────────────────────
    dpo_runs = [spec for spec in TARGET_RUNS if spec["phase"] == "DPO" and spec["id"] in extracted_dfs]
    if dpo_runs:
        spec = dpo_runs[0]
        df = extracted_dfs[spec["id"]].dropna(subset=["train/rewards/chosen"])
        if not df.empty and "train/rewards/chosen" in df.columns:
            fig, ax = plt.subplots(figsize=(6.8, 4.0))
            steps = df["_step"] if "_step" in df.columns else np.arange(len(df))
            chosen = df["train/rewards/chosen"].values
            rejected = df["train/rewards/rejected"].values if "train/rewards/rejected" in df.columns else None

            ax.plot(steps, chosen, color="#2ca02c", linewidth=2.0, label=r"Implicit Reward (Chosen: CE Compliant $y_w$)")
            if rejected is not None:
                ax.plot(steps, rejected, color="#d62728", linewidth=2.0, linestyle="--", label=r"Implicit Reward (Rejected: GL Hallucination $y_l$)")
                ax.fill_between(steps, rejected, chosen, color="#2ca02c", alpha=0.15, label="Preference Margin Gap")

            ax.set_xlabel("DPO Optimization Step")
            ax.set_ylabel(r"Implicit Reward $\beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$")
            ax.set_title("Direct Preference Optimization (DPO) Margin Widening", fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend(loc="center left", framealpha=0.9)

            fig.savefig(OUTPUT_FIG_DIR / "fig_dpo_preference_margin.pdf")
            fig.savefig(OUTPUT_FIG_DIR / "fig_dpo_preference_margin.png")
            plt.close(fig)
            print("  [OK] Saved: fig_dpo_preference_margin.pdf / .png")


def main():
    api_key = os.environ.get(
        "WANDB_API_KEY",
        "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"
    )
    extracted = extract_wandb_histories(api_key)
    if extracted:
        plot_publication_figures(extracted)
        print(f"\nAll W&B training trajectories and publication figures successfully exported!")
    else:
        print("No training trajectories were extracted.")


if __name__ == "__main__":
    main()
