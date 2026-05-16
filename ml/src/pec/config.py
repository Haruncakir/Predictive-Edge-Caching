"""Configuration dataclasses for the ML predictor and cache controller.

All knobs are collected here so a parameter sweep is a YAML/JSON file (or
CLI overrides) rather than scattered magic numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MLConfig:
    """Tuning knobs for the XGBoost predictor."""

    learning_rate: float = 0.05
    n_estimators: int = 100
    max_depth: int = 4
    prediction_horizon_ns: int = 50_000_000  # 50 ms (~200 requests lookahead)
    ewma_alpha: float = 0.1                   # for popularity smoothing
    retrain_interval: int = 128               # retrain every N requests
    min_training_samples: int = 32            # don't train until this many
    seed: int = 42


@dataclass
class CacheConfig:
    """Tuning knobs for the edge cache controller."""

    capacity_bytes: int = 100 * 1024 * 1024  # 100 MiB
    policy: str = "ml_driven"                # lru | lfu | fifo | ml_driven


@dataclass
class SimulationConfig:
    """Top-level simulation parameters."""

    trace_path: str = "../data/generated_trace.csv"
    output_dir: str = "../results"
    history_window: int = 256          # last K requests for feature extraction
    ml: MLConfig = field(default_factory=MLConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    policies_to_compare: list[str] = field(
        default_factory=lambda: ["lru", "lfu", "fifo", "ml_driven"]
    )
    metrics_window: int = 1000          # rolling window for windowed hit rate
    seed: int = 42
