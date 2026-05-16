"""Replacement policies — LRU, LFU, FIFO, and ML-driven.

Strategy pattern (mirrors ``TrafficEngine`` in the C++ codebase).  Each
policy implements ``on_access`` (called on every cache hit *and* miss) and
``select_victim`` (called when the cache is full and a new file must be
admitted).

DESIGN CHOICES
──────────────
* All four policies share the same abstract interface so the
  ``EdgeCacheController`` is policy-agnostic — Open/Closed Principle
  (Meyer, *OO Software Construction*, 1988).

* ``MLDrivenPolicy`` holds a reference to the prediction scores.  The
  controller calls ``update_predictions()`` before each eviction round.

* Tie-breaking in all policies is by insertion order (FIFO among equals),
  which makes behaviour deterministic and reproducible.
"""

from __future__ import annotations

import abc
from collections import OrderedDict, defaultdict

from ..types import PredictionResult


class ReplacementPolicy(abc.ABC):
    """Abstract eviction-policy interface."""

    @abc.abstractmethod
    def on_access(self, file_id: int, time_ns: int) -> None:
        """Notify the policy that *file_id* was accessed at *time_ns*."""

    @abc.abstractmethod
    def select_victim(self, cached_ids: set[int]) -> int:
        """Choose a file to evict from *cached_ids*."""

    @abc.abstractmethod
    def on_evict(self, file_id: int) -> None:
        """Notify the policy that *file_id* was evicted (clean up state)."""

    @abc.abstractmethod
    def reset(self) -> None:
        """Clear all internal state."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable policy name for reports."""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LRU — Least Recently Used
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class LRUPolicy(ReplacementPolicy):
    """Evict the file that was accessed least recently."""

    def __init__(self) -> None:
        self._order: OrderedDict[int, int] = OrderedDict()  # file_id → last access ns

    def on_access(self, file_id: int, time_ns: int) -> None:
        self._order[file_id] = time_ns
        self._order.move_to_end(file_id)

    def select_victim(self, cached_ids: set[int]) -> int:
        for fid in self._order:
            if fid in cached_ids:
                return fid
        # Fallback: pick arbitrary (shouldn't happen if state is consistent).
        return next(iter(cached_ids))

    def on_evict(self, file_id: int) -> None:
        self._order.pop(file_id, None)

    def reset(self) -> None:
        self._order.clear()

    @property
    def name(self) -> str:
        return "LRU"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LFU — Least Frequently Used
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class LFUPolicy(ReplacementPolicy):
    """Evict the file with the fewest total accesses (tie-break: oldest)."""

    def __init__(self) -> None:
        self._freq: dict[int, int] = defaultdict(int)
        self._first_seen: dict[int, int] = {}

    def on_access(self, file_id: int, time_ns: int) -> None:
        self._freq[file_id] += 1
        if file_id not in self._first_seen:
            self._first_seen[file_id] = time_ns

    def select_victim(self, cached_ids: set[int]) -> int:
        return min(
            cached_ids,
            key=lambda fid: (self._freq.get(fid, 0), self._first_seen.get(fid, 0)),
        )

    def on_evict(self, file_id: int) -> None:
        self._freq.pop(file_id, None)
        self._first_seen.pop(file_id, None)

    def reset(self) -> None:
        self._freq.clear()
        self._first_seen.clear()

    @property
    def name(self) -> str:
        return "LFU"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIFO — First In, First Out
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class FIFOPolicy(ReplacementPolicy):
    """Evict the file that was *inserted* earliest (ignoring accesses)."""

    def __init__(self) -> None:
        self._insertion_order: OrderedDict[int, None] = OrderedDict()

    def on_access(self, file_id: int, time_ns: int) -> None:
        # FIFO tracks insertion, not access. Only add if new.
        if file_id not in self._insertion_order:
            self._insertion_order[file_id] = None

    def select_victim(self, cached_ids: set[int]) -> int:
        for fid in self._insertion_order:
            if fid in cached_ids:
                return fid
        return next(iter(cached_ids))

    def on_evict(self, file_id: int) -> None:
        self._insertion_order.pop(file_id, None)

    def reset(self) -> None:
        self._insertion_order.clear()

    @property
    def name(self) -> str:
        return "FIFO"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ML-Driven — evict file with lowest predicted re-request probability
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MLDrivenPolicy(ReplacementPolicy):
    """Evict the file the ML model considers least likely to be re-requested.

    Falls back to LRU when no prediction scores are available (cold-start).
    """

    def __init__(self) -> None:
        self._predictions: dict[int, float] = {}  # file_id → probability
        self._lru_fallback = LRUPolicy()           # cold-start fallback

    def update_predictions(self, preds: list[PredictionResult]) -> None:
        """Inject fresh prediction scores from the ML Predictor."""
        self._predictions = {p.file_id: p.probability for p in preds}

    def on_access(self, file_id: int, time_ns: int) -> None:
        self._lru_fallback.on_access(file_id, time_ns)

    def select_victim(self, cached_ids: set[int]) -> int:
        # If we have predictions for at least some cached files, use them.
        scored = {
            fid: self._predictions[fid]
            for fid in cached_ids
            if fid in self._predictions
        }

        if scored:
            # Evict the file with the *lowest* predicted probability.
            return min(scored, key=scored.get)  # type: ignore[arg-type]

        # Fallback to LRU.
        return self._lru_fallback.select_victim(cached_ids)

    def on_evict(self, file_id: int) -> None:
        self._predictions.pop(file_id, None)
        self._lru_fallback.on_evict(file_id)

    def reset(self) -> None:
        self._predictions.clear()
        self._lru_fallback.reset()

    @property
    def name(self) -> str:
        return "ML-Driven"
