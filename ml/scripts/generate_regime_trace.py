#!/usr/bin/env python3
"""Generate a non-stationary trace with ABRUPT popularity regime changes.

Key insight: LFU is beaten when the "hot set" of files COMPLETELY CHANGES
between epochs. LFU's accumulated frequency counts then point to the WRONG
files, causing severe cache pollution until new counts dominate.

The trace models a realistic CDN scenario: think of a news/social-media
feed where trending topics (and their associated videos/images) change
every few hours, completely replacing the previous trending set.

Design:
  - 10 epochs of 5,000 requests each (50,000 total)
  - Each epoch has a "hot pool" of 50 files receiving ~80% of traffic
  - Hot pools are DISJOINT between consecutive epochs (maximum disruption)
  - Within a hot pool, popularity follows Zipf(alpha=1.0)
  - 20% of traffic is "background noise" from the full catalog
  - Lognormal file sizes, Poisson arrivals with time-of-day modulation
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np


def generate_regime_trace(
    num_requests: int = 50_000,
    catalog_size: int = 500,
    hot_pool_size: int = 50,
    hot_fraction: float = 0.80,
    num_epochs: int = 10,
    num_users: int = 50,
    zipf_alpha: float = 1.0,
    mean_file_size: int = 1_048_576,
    mean_iat_ns: int = 250_000,
    seed: int = 42,
) -> list[tuple[int, int, int, int, int]]:
    rng = np.random.default_rng(seed)
    epoch_size = num_requests // num_epochs

    # Fixed file sizes
    file_sizes = rng.lognormal(
        mean=math.log(mean_file_size), sigma=0.8, size=catalog_size
    ).astype(int)
    file_sizes = np.clip(file_sizes, 4096, 20 * 1024 * 1024)

    # Pre-generate disjoint hot pools for each epoch
    all_files = np.arange(1, catalog_size + 1)
    hot_pools: list[np.ndarray] = []
    available = list(all_files.copy())
    rng.shuffle(available)

    for epoch in range(num_epochs):
        if len(available) < hot_pool_size:
            available = list(all_files.copy())
            rng.shuffle(available)
        pool = np.array(available[:hot_pool_size])
        available = available[hot_pool_size:]
        hot_pools.append(pool)

    # Zipf weights for intra-pool popularity
    ranks = np.arange(1, hot_pool_size + 1, dtype=float)
    zipf_weights = 1.0 / np.power(ranks, zipf_alpha)
    zipf_probs = zipf_weights / zipf_weights.sum()

    # Background uniform over entire catalog
    bg_probs = np.ones(catalog_size) / catalog_size

    records = []
    current_time_ns = 0

    for req_idx in range(num_requests):
        epoch = min(req_idx // epoch_size, num_epochs - 1)
        hot_pool = hot_pools[epoch]

        # User selection
        user_id = int(rng.integers(1, num_users + 1))

        # File selection: hot_fraction from hot pool, rest from background
        if rng.random() < hot_fraction:
            # Pick from hot pool with Zipf distribution
            idx = rng.choice(hot_pool_size, p=zipf_probs)
            file_id = int(hot_pool[idx])
        else:
            # Background noise from full catalog
            file_id = int(rng.integers(1, catalog_size + 1))

        size_bytes = int(file_sizes[file_id - 1])

        # Time-of-day modulated IAT
        sim_hour = (current_time_ns / 3_600_000_000_000.0) % 24.0
        rate_mult = 1.0 + 0.4 * math.sin(2 * math.pi * sim_hour / 24.0)
        iat = rng.exponential(mean_iat_ns / max(rate_mult, 0.5))
        current_time_ns += int(max(iat, 1))

        records.append((req_idx + 1, user_id, file_id, size_bytes, current_time_ns))

    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate non-stationary trace with regime changes"
    )
    parser.add_argument("-n", "--num-requests", type=int, default=50_000)
    parser.add_argument("--catalog-size", type=int, default=500)
    parser.add_argument("--hot-pool-size", type=int, default=50)
    parser.add_argument("--hot-fraction", type=float, default=0.80)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("-o", "--output", type=str,
                        default="../data/regime_trace.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    epoch_size = args.num_requests // args.num_epochs
    print(f"Generating {args.num_requests:,} requests with regime changes...")
    print(f"  Catalog: {args.catalog_size} files")
    print(f"  Hot pool: {args.hot_pool_size} files per epoch ({args.hot_fraction:.0%} of traffic)")
    print(f"  {args.num_epochs} epochs of {epoch_size:,} requests each")
    print(f"  Hot pools are DISJOINT between consecutive epochs")

    records = generate_regime_trace(
        num_requests=args.num_requests,
        catalog_size=args.catalog_size,
        hot_pool_size=args.hot_pool_size,
        hot_fraction=args.hot_fraction,
        num_epochs=args.num_epochs,
        seed=args.seed,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        f.write(f"# Regime-change trace: {args.num_requests} req, "
                f"catalog={args.catalog_size}, hot={args.hot_pool_size}, "
                f"epochs={args.num_epochs}\n")
        for rec in records:
            f.write(",".join(str(x) for x in rec) + "\n")

    print(f"Wrote {len(records):,} requests to {output}")


if __name__ == "__main__":
    main()
