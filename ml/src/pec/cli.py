"""CLI entry point — ``pec-simulate`` or ``python -m pec.cli``.

Usage examples::

    # Use defaults (reads ../data/generated_trace.csv)
    pec-simulate

    # Specify trace and cache size
    pec-simulate --trace ../data/generated_trace.csv --cache-mb 50

    # Run only specific policies
    pec-simulate --policies lru ml_driven

    # Custom ML tuning
    pec-simulate --lr 0.1 --retrain-interval 128
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import CacheConfig, MLConfig, SimulationConfig
from .simulator import run_comparison


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pec-simulate",
        description="Predictive Edge Caching — ML vs baseline cache comparison",
    )
    p.add_argument(
        "--trace",
        default="../data/generated_trace.csv",
        help="Path to request trace CSV (default: ../data/generated_trace.csv)",
    )
    p.add_argument(
        "--output-dir",
        default="../results",
        help="Directory for metrics CSV output (default: ../results)",
    )
    p.add_argument(
        "--cache-mb",
        type=int,
        default=100,
        help="Cache capacity in MiB (default: 100)",
    )
    p.add_argument(
        "--policies",
        nargs="+",
        default=["lru", "lfu", "fifo", "ml_driven"],
        help="Policies to compare (default: all four)",
    )
    p.add_argument(
        "--history-window",
        type=int,
        default=256,
        help="History window size for feature extraction (default: 256)",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=0.05,
        help="XGBoost learning rate (default: 0.05)",
    )
    p.add_argument(
        "--retrain-interval",
        type=int,
        default=256,
        help="Retrain ML model every N requests (default: 256)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    p.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable DEBUG logging",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = SimulationConfig(
        trace_path=args.trace,
        output_dir=args.output_dir,
        history_window=args.history_window,
        ml=MLConfig(
            learning_rate=args.lr,
            retrain_interval=args.retrain_interval,
            seed=args.seed,
        ),
        cache=CacheConfig(
            capacity_bytes=args.cache_mb * 1024 * 1024,
        ),
        policies_to_compare=args.policies,
        seed=args.seed,
    )

    results = run_comparison(cfg)

    # ── Summary table ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  FINAL COMPARISON")
    print("=" * 60)
    print(f"  {'Policy':<15} {'Hit Rate':>10} {'Hits':>10} {'Misses':>10}")
    print("  " + "-" * 45)
    for name, reporter in sorted(results.items()):
        hr = reporter.final_hit_rate
        h = reporter.snapshots[-1].cumulative_hits if reporter.snapshots else 0
        m = reporter.snapshots[-1].cumulative_misses if reporter.snapshots else 0
        print(f"  {name:<15} {hr:>9.2%} {h:>10,} {m:>10,}")
    print("=" * 60)

    # ── Write CSV results ─────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    for name, reporter in results.items():
        path = reporter.write_csv(output_dir)
        print(f"  Wrote {path}")

    print(f"\nRun 'python scripts/plot_results.py {output_dir}' to generate graphs.")


if __name__ == "__main__":
    main()
