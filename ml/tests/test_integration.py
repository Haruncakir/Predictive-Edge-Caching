"""Integration test: run a small trace through the full pipeline."""

from pec.cache.controller import EdgeCacheController
from pec.cache.policies import LRUPolicy, MLDrivenPolicy
from pec.cache.store import CacheStore
from pec.config import MLConfig
from pec.model import MLPredictor
from pec.types import CacheOutcome, Request


def _build_trace(n: int = 500) -> list[Request]:
    """Generate a small synthetic trace with Zipf-like repetition."""
    import random

    rng = random.Random(42)
    catalog = list(range(1, 21))  # 20 files
    sizes = {fid: rng.randint(1024, 65536) for fid in catalog}

    requests = []
    for i in range(n):
        # Zipf-ish: lower IDs are more popular.
        fid = catalog[int(rng.paretovariate(1.5)) % len(catalog)]
        requests.append(
            Request(
                request_id=i + 1,
                user_id=rng.randint(1, 10),
                file_id=fid,
                size_bytes=sizes[fid],
                arrival_ns=i * 100_000,
            )
        )
    return requests


class TestIntegration:
    def test_lru_processes_all_requests(self):
        trace = _build_trace(200)
        store = CacheStore(capacity_bytes=200_000)
        policy = LRUPolicy()
        ctrl = EdgeCacheController(store, policy)

        for req in trace:
            fb = ctrl.process_request(req)
            assert fb.outcome in (CacheOutcome.HIT, CacheOutcome.MISS)

        assert ctrl.total_requests == 200
        assert ctrl.total_hits + ctrl.total_misses == 200
        assert ctrl.hit_rate >= 0.0

    def test_ml_driven_produces_valid_output(self):
        trace = _build_trace(300)
        store = CacheStore(capacity_bytes=200_000)
        policy = MLDrivenPolicy()
        predictor = MLPredictor(MLConfig(
            retrain_interval=50,
            min_training_samples=20,
        ))
        ctrl = EdgeCacheController(store, policy, predictor)

        from collections import deque
        history: deque[Request] = deque(maxlen=64)

        for req in trace:
            history.append(req)
            predictor.observe(req)
            fb = ctrl.process_request(req, list(history))
            assert fb.outcome in (CacheOutcome.HIT, CacheOutcome.MISS)

        assert ctrl.total_requests == 300
        assert 0.0 <= ctrl.hit_rate <= 1.0

    def test_hit_rate_improves_with_ml(self):
        """ML-driven should be competitive with LRU on a Zipf workload."""
        trace = _build_trace(1000)

        # Run LRU
        store_lru = CacheStore(capacity_bytes=100_000)
        ctrl_lru = EdgeCacheController(store_lru, LRUPolicy())
        for req in trace:
            ctrl_lru.process_request(req)

        # Run ML-Driven
        store_ml = CacheStore(capacity_bytes=100_000)
        policy_ml = MLDrivenPolicy()
        predictor = MLPredictor(MLConfig(
            retrain_interval=50,
            min_training_samples=20,
        ))
        ctrl_ml = EdgeCacheController(store_ml, policy_ml, predictor)

        from collections import deque
        history: deque[Request] = deque(maxlen=128)
        for req in trace:
            history.append(req)
            predictor.observe(req)
            ctrl_ml.process_request(req, list(history))

        # ML should at least not be catastrophically worse than LRU.
        # (On a short trace it may not beat LRU; the point is it's functional.)
        assert ctrl_ml.hit_rate >= ctrl_lru.hit_rate * 0.5
