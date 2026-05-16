#!/usr/bin/env python3
"""plot_results.py — generate comparison graphs from simulation CSV output.

Usage::

    python scripts/plot_results.py ../results/
    python scripts/plot_results.py ../results/ --format pdf

Generates:
    1. Hit rate over time (all policies overlaid)
    2. Final hit rate bar chart
    3. Cache utilisation over time
    4. ML prediction accuracy (training steps timeline)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ── Styling ───────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="Set2", font_scale=1.1)
COLORS = {
    "LRU": "#e74c3c",
    "LFU": "#3498db",
    "FIFO": "#95a5a6",
    "ML-Driven": "#2ecc71",
    "ML_Driven": "#2ecc71",
}


def load_results(results_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all CSV files in the results directory."""
    csvs = sorted(results_dir.glob("*.csv"))
    if not csvs:
        print(f"No CSV files found in {results_dir}", file=sys.stderr)
        sys.exit(1)

    data: dict[str, pd.DataFrame] = {}
    for csv_path in csvs:
        name = csv_path.stem.replace("_", "-").title().replace("-", "-")
        # Clean up name: lru -> LRU, ml_driven -> ML-Driven
        name_map = {"Lru": "LRU", "Lfu": "LFU", "Fifo": "FIFO", "Ml-Driven": "ML-Driven"}
        name = name_map.get(name, name)
        df = pd.read_csv(csv_path)
        data[name] = df
        print(f"  Loaded {csv_path.name} → {name} ({len(df)} rows)")

    return data


def plot_hit_rate_over_time(
    data: dict[str, pd.DataFrame],
    output_dir: Path,
    fmt: str,
) -> None:
    """Line chart: windowed hit rate over request index."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for name, df in data.items():
        color = COLORS.get(name, None)
        # Subsample for large traces to keep the plot readable.
        step = max(1, len(df) // 2000)
        subset = df.iloc[::step]
        ax.plot(
            subset["request_idx"],
            subset["windowed_hit_rate"] * 100,
            label=name,
            color=color,
            linewidth=1.5,
            alpha=0.9,
        )

    ax.set_xlabel("Request Index")
    ax.set_ylabel("Windowed Hit Rate (%)")
    ax.set_title("Cache Hit Rate Over Time — Policy Comparison")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    path = output_dir / f"hit_rate_over_time.{fmt}"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


def plot_final_comparison(
    data: dict[str, pd.DataFrame],
    output_dir: Path,
    fmt: str,
) -> None:
    """Bar chart: final cumulative hit rate for each policy."""
    fig, ax = plt.subplots(figsize=(8, 5))

    names = []
    rates = []
    colors = []
    for name, df in data.items():
        final_hr = df["cumulative_hit_rate"].iloc[-1] * 100
        names.append(name)
        rates.append(final_hr)
        colors.append(COLORS.get(name, "#888888"))

    bars = ax.bar(names, rates, color=colors, edgecolor="white", linewidth=1.2)

    # Value labels on bars.
    for bar, rate in zip(bars, rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{rate:.2f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
        )

    ax.set_ylabel("Final Hit Rate (%)")
    ax.set_title("Final Cache Hit Rate — Policy Comparison")
    ax.set_ylim(0, max(rates) * 1.15 if rates else 100)
    ax.grid(axis="y", alpha=0.3)

    path = output_dir / f"final_comparison.{fmt}"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


def plot_cache_utilisation(
    data: dict[str, pd.DataFrame],
    output_dir: Path,
    fmt: str,
) -> None:
    """Line chart: cache utilisation over time."""
    fig, ax = plt.subplots(figsize=(12, 5))

    for name, df in data.items():
        color = COLORS.get(name, None)
        step = max(1, len(df) // 2000)
        subset = df.iloc[::step]
        ax.plot(
            subset["request_idx"],
            subset["cache_utilisation"] * 100,
            label=name,
            color=color,
            linewidth=1.2,
            alpha=0.85,
        )

    ax.set_xlabel("Request Index")
    ax.set_ylabel("Cache Utilisation (%)")
    ax.set_title("Cache Store Utilisation Over Time")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    path = output_dir / f"cache_utilisation.{fmt}"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


def plot_ml_convergence(
    data: dict[str, pd.DataFrame],
    output_dir: Path,
    fmt: str,
) -> None:
    """Plot ML model training convergence (only for ML-Driven)."""
    ml_df = data.get("ML-Driven")
    if ml_df is None or "ml_training_steps" not in ml_df.columns:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Training steps over time.
    step = max(1, len(ml_df) // 2000)
    subset = ml_df.iloc[::step]

    ax1.plot(
        subset["request_idx"],
        subset["ml_training_steps"],
        color=COLORS["ML-Driven"],
        linewidth=1.5,
    )
    ax1.set_xlabel("Request Index")
    ax1.set_ylabel("Training Steps")
    ax1.set_title("ML Model Training Progress")
    ax1.grid(True, alpha=0.3)

    # Hit rate improvement: compare first 10% vs last 10% of requests.
    n = len(ml_df)
    early = ml_df.iloc[: n // 10]
    late = ml_df.iloc[-n // 10 :]
    early_hr = early["windowed_hit_rate"].mean() * 100
    late_hr = late["windowed_hit_rate"].mean() * 100

    bars = ax2.bar(
        ["First 10%", "Last 10%"],
        [early_hr, late_hr],
        color=[COLORS["ML-Driven"] + "80", COLORS["ML-Driven"]],
        edgecolor="white",
    )
    for bar, val in zip(bars, [early_hr, late_hr]):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{val:.2f}%",
            ha="center", va="bottom", fontweight="bold",
        )
    ax2.set_ylabel("Windowed Hit Rate (%)")
    ax2.set_title("ML Learning Effect: Early vs Late")
    ax2.set_ylim(0, max(early_hr, late_hr) * 1.2)
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = output_dir / f"ml_convergence.{fmt}"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot PEC simulation results")
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Directory containing per-policy CSV files",
    )
    parser.add_argument(
        "--format",
        default="png",
        choices=["png", "pdf", "svg"],
        help="Output image format (default: png)",
    )
    args = parser.parse_args()

    print(f"Loading results from {args.results_dir} …")
    data = load_results(args.results_dir)

    print("Generating plots …")
    plot_hit_rate_over_time(data, args.results_dir, args.format)
    plot_final_comparison(data, args.results_dir, args.format)
    plot_cache_utilisation(data, args.results_dir, args.format)
    plot_ml_convergence(data, args.results_dir, args.format)

    print("Done.")


if __name__ == "__main__":
    main()
