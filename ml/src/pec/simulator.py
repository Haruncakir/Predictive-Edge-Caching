"""Simulator — replays a C++ trace through the full ML + cache pipeline.

This is the integration layer that wires:

    Request trace (CSV) → FeatureExtractor → MLPredictor → EdgeCacheController → MetricsReporter

For each policy in the comparison set, the *same* trace is replayed from
scratch so results are directly comparable.
"""

from __future__ import annotations

import logging
from collections import deque
from pathlib import Path

import pandas as pd

from .cache.controller import EdgeCacheController
from .cache.policies import (
    FIFOPolicy,
    LFUPolicy,
    LRUPolicy,
    MLDrivenPolicy,
    ReplacementPolicy,
)
from .cache.store import CacheStore
from .config import SimulationConfig
from .metrics import MetricsReporter
from .model import MLPredictor
from .types import CacheOutcome, Request

logger = logging.getLogger(__name__)


def load_trace(path: str | Path) -> list[Request]:
    """Load a CSV trace file into a list of Request objects.

    Expected CSV columns: ``request_id,user_id,file_id,size_bytes,arrival_ns``
    Lines starting with ``#`` are comments (SNIA IOTTA convention).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Trace file not found: {path}")

    df = pd.read_csv(
        path,
        comment="#",
        header=None,
        names=["request_id", "user_id", "file_id", "size_bytes", "arrival_ns"],
        dtype={
            "request_id": int,
            "user_id": int,
            "file_id": int,
            "size_bytes": int,
            "arrival_ns": int,
        },
    )

    requests = [
        Request(
            request_id=row.request_id,
            user_id=row.user_id,
            file_id=row.file_id,
            size_bytes=row.size_bytes,
            arrival_ns=row.arrival_ns,
        )
        for row in df.itertuples(index=False)
    ]
    logger.info("Loaded %d requests from %s", len(requests), path)
    return requests


def _make_policy(name: str) -> ReplacementPolicy:
    """Factory for replacement policies."""
    name_lower = name.lower().replace("-", "_")
    if name_lower == "lru":
        return LRUPolicy()
    if name_lower == "lfu":
        return LFUPolicy()
    if name_lower == "fifo":
        return FIFOPolicy()
    if name_lower in ("ml_driven", "ml"):
        return MLDrivenPolicy()
    raise ValueError(f"Unknown policy: {name!r}")


def run_single_policy(
    requests: list[Request],
    policy_name: str,
    cfg: SimulationConfig,
) -> MetricsReporter:
    """Run the full simulation for a single replacement policy.

    Returns a populated MetricsReporter.
    """
    policy = _make_policy(policy_name)
    store = CacheStore(cfg.cache.capacity_bytes)

    # ML predictor is only instantiated for the ML-driven policy.
    predictor: MLPredictor | None = None
    if isinstance(policy, MLDrivenPolicy):
        predictor = MLPredictor(cfg.ml)

    controller = EdgeCacheController(store, policy, predictor)
    reporter = MetricsReporter(policy.name, window_size=cfg.metrics_window)

    # Sliding history window (like the C++ HistoryBuffer).
    history: deque[Request] = deque(maxlen=cfg.history_window)

    # Track recently-seen file IDs for ML label generation.
    recent_file_ids: set[int] = set()
    recent_window: deque[int] = deque(maxlen=cfg.history_window)

    total = len(requests)
    log_interval = max(total // 20, 1)

    for idx, req in enumerate(requests):
        history.append(req)
        recent_file_ids.add(req.file_id)
        recent_window.append(req.file_id)
        # Keep recent_file_ids in sync with the sliding window.
        if len(recent_window) == recent_window.maxlen:
            recent_file_ids = set(recent_window)

        # ── ML observation + label + train ────────────────────────────
        if predictor is not None:
            predictor.observe(req)
            predictor.label_and_train(req.arrival_ns, recent_file_ids)

            # Record features for future labelling.
            if store.count > 0 and idx % cfg.ml.retrain_interval == 0:
                from .feature_extractor import FeatureExtractor

                extractor = FeatureExtractor()
                features = extractor.extract(
                    list(history), store.file_ids, req.arrival_ns
                )
                predictor.record_prediction(features, req.arrival_ns)

        # ── Process request through cache controller ──────────────────
        feedback = controller.process_request(req, list(history))

        # ── Record metrics ────────────────────────────────────────────
        reporter.record(
            request_idx=idx,
            timestamp_ns=req.arrival_ns,
            outcome=feedback.outcome,
            cache_utilisation=store.utilisation,
            ml_training_steps=predictor.training_steps if predictor else 0,
            ml_has_model=predictor.has_model if predictor else False,
        )

        if (idx + 1) % log_interval == 0:
            logger.info(
                "[%s] %d/%d  hit_rate=%.4f  cache=%.1f%%",
                policy.name,
                idx + 1,
                total,
                reporter.final_hit_rate,
                store.utilisation * 100,
            )

    # Final ML training flush.
    if predictor is not None:
        predictor.force_train()

    logger.info(
        "[%s] DONE — %d requests, hit_rate=%.4f",
        policy.name,
        total,
        reporter.final_hit_rate,
    )
    return reporter


def run_comparison(cfg: SimulationConfig) -> dict[str, MetricsReporter]:
    """Run the simulation for every policy in the comparison set.

    Returns a dict mapping policy name → MetricsReporter.
    """
    requests = load_trace(cfg.trace_path)
    results: dict[str, MetricsReporter] = {}

    for policy_name in cfg.policies_to_compare:
        logger.info("━" * 60)
        logger.info("Running policy: %s", policy_name)
        logger.info("━" * 60)
        reporter = run_single_policy(requests, policy_name, cfg)
        results[reporter.policy_name] = reporter

    return results
