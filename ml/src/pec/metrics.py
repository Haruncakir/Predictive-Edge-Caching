"""MetricsReporter — time-series logging and CSV export.

Records per-request and per-epoch metrics so the final comparison graphs
(ML vs LRU vs LFU vs FIFO) can be generated offline.
"""

from __future__ import annotations

import csv
import os
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from .types import CacheOutcome


@dataclass
class MetricSnapshot:
    """One row in the metrics time-series."""

    request_idx: int
    timestamp_ns: int
    cumulative_hits: int
    cumulative_misses: int
    cumulative_hit_rate: float
    windowed_hit_rate: float
    cache_utilisation: float
    ml_training_steps: int = 0
    ml_has_model: bool = False


class MetricsReporter:
    """Accumulates simulation metrics and writes them to CSV.

    Parameters
    ----------
    policy_name : str
        Label for this run (e.g. "LRU", "ML-Driven").
    window_size : int
        Rolling window size for windowed hit-rate calculation.
    """

    def __init__(self, policy_name: str, window_size: int = 1000) -> None:
        self.policy_name = policy_name
        self._window_size = window_size

        self._hits: int = 0
        self._misses: int = 0
        self._window: deque[int] = deque(maxlen=window_size)  # 1=hit, 0=miss
        self._snapshots: list[MetricSnapshot] = []

    def record(
        self,
        request_idx: int,
        timestamp_ns: int,
        outcome: CacheOutcome,
        cache_utilisation: float = 0.0,
        ml_training_steps: int = 0,
        ml_has_model: bool = False,
    ) -> None:
        """Record one request outcome."""
        is_hit = outcome == CacheOutcome.HIT
        if is_hit:
            self._hits += 1
        else:
            self._misses += 1
        self._window.append(1 if is_hit else 0)

        total = self._hits + self._misses
        cum_hr = self._hits / total if total else 0.0
        win_hr = sum(self._window) / len(self._window) if self._window else 0.0

        self._snapshots.append(
            MetricSnapshot(
                request_idx=request_idx,
                timestamp_ns=timestamp_ns,
                cumulative_hits=self._hits,
                cumulative_misses=self._misses,
                cumulative_hit_rate=cum_hr,
                windowed_hit_rate=win_hr,
                cache_utilisation=cache_utilisation,
                ml_training_steps=ml_training_steps,
                ml_has_model=ml_has_model,
            )
        )

    @property
    def snapshots(self) -> list[MetricSnapshot]:
        return self._snapshots

    @property
    def final_hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total else 0.0

    @property
    def total_requests(self) -> int:
        return self._hits + self._misses

    def write_csv(self, output_dir: str | Path) -> Path:
        """Write the time-series to a CSV file.  Returns the output path."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{self.policy_name.lower().replace(' ', '_').replace('-', '_')}.csv"
        path = output_dir / filename

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "request_idx",
                "timestamp_ns",
                "cumulative_hits",
                "cumulative_misses",
                "cumulative_hit_rate",
                "windowed_hit_rate",
                "cache_utilisation",
                "ml_training_steps",
                "ml_has_model",
            ])
            for s in self._snapshots:
                writer.writerow([
                    s.request_idx,
                    s.timestamp_ns,
                    s.cumulative_hits,
                    s.cumulative_misses,
                    f"{s.cumulative_hit_rate:.6f}",
                    f"{s.windowed_hit_rate:.6f}",
                    f"{s.cache_utilisation:.6f}",
                    s.ml_training_steps,
                    int(s.ml_has_model),
                ])

        return path

    def reset(self) -> None:
        self._hits = 0
        self._misses = 0
        self._window.clear()
        self._snapshots.clear()
