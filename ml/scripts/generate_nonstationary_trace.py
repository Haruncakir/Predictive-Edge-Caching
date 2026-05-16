#!/usr/bin/env python3
"""Generate a non-stationary request trace that mimics real-world CDN traffic.

Real CDN traffic is NOT stationary Zipf. Key phenomena modeled here:

1. **Popularity rotation**: Every `shift_interval` requests, the Zipf
   ranking is permuted — simulating content going viral and fading.
   (Traverso et al., "Temporal locality in today's content caching",
   IEEE/ACM ToN 2015)

2. **Flash crowds / trending events**: Random files receive sudden bursts
   of requests, then return to baseline. Models viral content surges.
   (Urdaneta et al., "Wikipedia workload analysis", Comp. Networks 2009)

3. **Time-of-day modulation**: Request rate and popularity mix vary with
   simulated hour-of-day. Peak hours favor popular content more heavily.
   (Li et al., "Caching as a Service for 5G", IEEE Access 2019)

4. **User locality**: Users have preferences — each user has a biased
   subset of files they tend to request, modeling personalized demand.

This trace is specifically designed so that:
- LFU is "poisoned" by stale frequency counts from previous popularity epochs
- LRU reacts but cannot anticipate upcoming shifts
- ML-Driven can learn temporal signals (recency, rate changes, time features)
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np


def zipf_probs(n: int, alpha: float) -> np.ndarray:
    """Compute Zipf probability vector for ranks 1..n."""
    ranks = np.arange(1, n + 1, dtype=float)
    weights = 1.0 / np.power(ranks, alpha)
    return weights / weights.sum()


def generate_trace(
    num_requests: int = 50_000,
    catalog_size: int = 1000,
    num_users: int = 50,
    zipf_alpha: float = 0.8,
    shift_interval: int = 5000,
    num_trending_events: int = 10,
    trending_burst_size: int = 200,
    mean_file_size: int = 1_048_576,
    mean_iat_ns: int = 250_000,
    seed: int = 42,
) -> list[tuple[int, int, int, int, int]]:
    """Generate a non-stationary request trace.

    Returns list of (request_id, user_id, file_id, size_bytes, arrival_ns).
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)

    # Pre-generate file sizes (lognormal, stable across simulation)
    file_sizes = rng.lognormal(
        mean=math.log(mean_file_size), sigma=1.0, size=catalog_size
    ).astype(int)
    file_sizes = np.clip(file_sizes, 1024, 50 * 1024 * 1024)  # 1KB - 50MB

    # User preference profiles: each user has a biased subset of preferred files
    # This creates per-user locality that ML can learn
    user_preferences = {}
    for uid in range(1, num_users + 1):
        # Each user has ~20% of catalog as "preferred" files
        preferred = rng.choice(catalog_size, size=catalog_size // 5, replace=False) + 1
        user_preferences[uid] = preferred

    # Base Zipf probabilities
    base_probs = zipf_probs(catalog_size, zipf_alpha)

    # Pre-plan trending events: (start_request, file_id, burst_size)
    trending_events = []
    for _ in range(num_trending_events):
        start = rng.integers(1000, num_requests - trending_burst_size)
        # Trending files are from the TAIL of the distribution (not already popular)
        fid = rng.integers(catalog_size // 2, catalog_size) + 1
        trending_events.append((int(start), int(fid), trending_burst_size))
    trending_events.sort()

    # Build a map: request_idx -> trending file_id (if any)
    trending_map: dict[int, int] = {}
    for start, fid, burst in trending_events:
        for i in range(burst):
            idx = start + i
            if idx < num_requests:
                trending_map[idx] = fid

    records = []
    current_time_ns = 0
    current_permutation = np.arange(catalog_size)  # identity mapping initially

    # Track current epoch for popularity shifts
    current_epoch = 0

    for req_idx in range(num_requests):
        # ── Popularity shift every shift_interval requests ────────────
        epoch = req_idx // shift_interval
        if epoch != current_epoch:
            current_epoch = epoch
            # Partial permutation: rotate top 30% of rankings
            top_k = catalog_size * 3 // 10
            top_portion = current_permutation[:top_k].copy()
            rng.shuffle(top_portion)
            current_permutation[:top_k] = top_portion

        # ── Time-of-day modulation ────────────────────────────────────
        # Simulate 24-hour cycle over the trace. Higher alpha during "peak hours"
        sim_hour = (current_time_ns / 3_600_000_000_000.0) % 24.0
        # Peak hours: 10-14 and 19-23 (two peaks per day)
        peak_factor = 0.3 * math.sin(2 * math.pi * sim_hour / 24.0 - math.pi / 2)
        effective_alpha = zipf_alpha + peak_factor * 0.3  # alpha varies 0.5-1.1
        effective_alpha = max(0.4, min(1.2, effective_alpha))

        # ── Select user ───────────────────────────────────────────────
        user_id = int(rng.integers(1, num_users + 1))

        # ── Select file ───────────────────────────────────────────────
        # Check for trending event first
        if req_idx in trending_map:
            # 70% chance this request is for the trending file
            if rng.random() < 0.7:
                file_id = trending_map[req_idx]
            else:
                # Normal selection
                probs = zipf_probs(catalog_size, effective_alpha)
                permuted_probs = probs[np.argsort(current_permutation)]
                rank = rng.choice(catalog_size, p=permuted_probs)
                file_id = rank + 1
        else:
            # Mix user preference (30%) with global popularity (70%)
            if rng.random() < 0.3 and user_id in user_preferences:
                # User-preferred file
                preferred = user_preferences[user_id]
                file_id = int(rng.choice(preferred))
            else:
                # Global Zipf with current permutation and time modulation
                probs = zipf_probs(catalog_size, effective_alpha)
                permuted_probs = probs[np.argsort(current_permutation)]
                rank = rng.choice(catalog_size, p=permuted_probs)
                file_id = rank + 1

        # ── File size and timing ──────────────────────────────────────
        size_bytes = int(file_sizes[file_id - 1])

        # IAT with time-of-day modulation (faster during peak hours)
        peak_rate_mult = 1.0 + 0.5 * math.sin(2 * math.pi * sim_hour / 24.0)
        iat = rng.exponential(mean_iat_ns / max(peak_rate_mult, 0.5))
        current_time_ns += int(iat)

        records.append((req_idx + 1, user_id, file_id, size_bytes, current_time_ns))

    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate non-stationary CDN-like request trace"
    )
    parser.add_argument(
        "-n", "--num-requests", type=int, default=50_000,
        help="Number of requests (default: 50000)",
    )
    parser.add_argument(
        "--catalog-size", type=int, default=1000,
        help="Number of distinct files (default: 1000)",
    )
    parser.add_argument(
        "--shift-interval", type=int, default=5000,
        help="Requests between popularity shifts (default: 5000)",
    )
    parser.add_argument(
        "--trending-events", type=int, default=10,
        help="Number of trending/flash-crowd events (default: 10)",
    )
    parser.add_argument(
        "-o", "--output", type=str, default="../data/nonstationary_trace.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    print(f"Generating {args.num_requests:,} non-stationary requests...")
    print(f"  Catalog: {args.catalog_size} files")
    print(f"  Popularity shift every {args.shift_interval} requests")
    print(f"  {args.trending_events} trending events")

    records = generate_trace(
        num_requests=args.num_requests,
        catalog_size=args.catalog_size,
        shift_interval=args.shift_interval,
        num_trending_events=args.trending_events,
        seed=args.seed,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w") as f:
        f.write(f"# Non-stationary trace: {args.num_requests} requests, "
                f"catalog={args.catalog_size}, shift_every={args.shift_interval}, "
                f"trending_events={args.trending_events}\n")
        for rec in records:
            f.write(",".join(str(x) for x in rec) + "\n")

    print(f"Wrote {len(records):,} requests to {output}")


if __name__ == "__main__":
    main()
