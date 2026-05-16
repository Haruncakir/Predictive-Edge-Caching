"""EdgeCacheController — orchestrates store + policy + ML predictor.

This is the top-level "Edge Cache Controller" box in the architecture
diagram.  For each incoming request it:

1. Checks the CacheStore for a **hit** or **miss**.
2. On a miss, if the cache is full, asks the ReplacementPolicy for a
   victim to evict (optionally refreshing ML predictions first).
3. Admits the new file.
4. Returns a ``Feedback`` record for the feedback loop.

DESIGN CHOICES
──────────────
* The controller is completely policy-agnostic — it never inspects the
  concrete policy type.  Swapping LRU for ML-driven is a one-line config
  change (Dependency Inversion Principle).

* Eviction may need to free *multiple* files when the incoming file is
  larger than any single cached file.  The loop continues evicting until
  enough space is available or the cache is empty.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ..types import CacheOutcome, Feedback, Request
from .policies import MLDrivenPolicy, ReplacementPolicy
from .store import CacheStore

if TYPE_CHECKING:
    from ..model import MLPredictor

logger = logging.getLogger(__name__)


class EdgeCacheController:
    """Wires ``CacheStore`` + ``ReplacementPolicy`` (+ optional ``MLPredictor``).

    Parameters
    ----------
    store : CacheStore
        The byte-addressed content store.
    policy : ReplacementPolicy
        Eviction strategy.
    predictor : MLPredictor | None
        If provided and the policy is ``MLDrivenPolicy``, predictions are
        refreshed before every eviction decision.
    history : list[Request] | None
        Reference to a shared history window — updated externally by the
        simulator.  Used for ML feature extraction.
    """

    def __init__(
        self,
        store: CacheStore,
        policy: ReplacementPolicy,
        predictor: "MLPredictor | None" = None,
    ) -> None:
        self._store = store
        self._policy = policy
        self._predictor = predictor

        self._hits: int = 0
        self._misses: int = 0

    # ── Public API ────────────────────────────────────────────────────

    def process_request(
        self,
        req: Request,
        history: list[Request] | None = None,
    ) -> Feedback:
        """Process a single request.  Returns a Feedback record."""
        self._policy.on_access(req.file_id, req.arrival_ns)

        if self._store.contains(req.file_id):
            self._hits += 1
            return Feedback(
                request_id=req.request_id,
                outcome=CacheOutcome.HIT,
                served_ns=req.arrival_ns,
            )

        # ── MISS ──────────────────────────────────────────────────────
        self._misses += 1

        # Make room if needed.
        self._ensure_space(req.size_bytes, req.arrival_ns, history)

        # Admit the new file (may still fail if it's larger than total
        # cache capacity — in that case we just don't cache it).
        self._store.admit(req.file_id, req.size_bytes)

        return Feedback(
            request_id=req.request_id,
            outcome=CacheOutcome.MISS,
            served_ns=req.arrival_ns,
        )

    # ── Metrics ───────────────────────────────────────────────────────

    @property
    def total_hits(self) -> int:
        return self._hits

    @property
    def total_misses(self) -> int:
        return self._misses

    @property
    def total_requests(self) -> int:
        return self._hits + self._misses

    @property
    def hit_rate(self) -> float:
        total = self.total_requests
        return self._hits / total if total else 0.0

    @property
    def store(self) -> CacheStore:
        return self._store

    @property
    def policy(self) -> ReplacementPolicy:
        return self._policy

    def reset_metrics(self) -> None:
        self._hits = 0
        self._misses = 0

    # ── Internals ─────────────────────────────────────────────────────

    def _ensure_space(
        self,
        needed_bytes: int,
        current_time_ns: int,
        history: list[Request] | None,
    ) -> None:
        """Evict files until *needed_bytes* are free."""
        if self._store.free_bytes() >= needed_bytes:
            return

        # Refresh ML predictions if applicable.
        if (
            isinstance(self._policy, MLDrivenPolicy)
            and self._predictor is not None
            and history
        ):
            preds = self._predictor.predict(
                history, self._store.file_ids, current_time_ns
            )
            self._policy.update_predictions(preds)

        # Evict until enough space or cache is empty.
        while self._store.free_bytes() < needed_bytes and self._store.count > 0:
            victim = self._policy.select_victim(self._store.file_ids)
            freed = self._store.evict(victim)
            self._policy.on_evict(victim)
            logger.debug(
                "Evicted file %d (freed %d bytes)", victim, freed
            )
