"""Tests for CacheStore and replacement policies."""

import pytest

from pec.cache.store import CacheStore
from pec.cache.policies import LRUPolicy, LFUPolicy, FIFOPolicy, MLDrivenPolicy
from pec.types import PredictionResult


class TestCacheStore:
    def test_admit_and_contains(self):
        store = CacheStore(1024)
        assert store.admit(1, 256)
        assert store.contains(1)
        assert not store.contains(2)

    def test_capacity_enforcement(self):
        store = CacheStore(512)
        assert store.admit(1, 256)
        assert store.admit(2, 256)
        assert not store.admit(3, 256)  # would exceed 512

    def test_evict_frees_space(self):
        store = CacheStore(512)
        store.admit(1, 300)
        store.admit(2, 200)
        assert store.free_bytes() == 12

        freed = store.evict(1)
        assert freed == 300
        assert not store.contains(1)
        assert store.free_bytes() == 312

    def test_double_admit_is_noop(self):
        store = CacheStore(512)
        store.admit(1, 256)
        assert store.admit(1, 256)  # returns True, no-op
        assert store.used_bytes == 256  # not doubled

    def test_utilisation(self):
        store = CacheStore(1000)
        store.admit(1, 500)
        assert abs(store.utilisation - 0.5) < 0.001

    def test_invalid_capacity_raises(self):
        with pytest.raises(ValueError):
            CacheStore(0)


class TestLRUPolicy:
    def test_evicts_least_recently_used(self):
        pol = LRUPolicy()
        pol.on_access(1, 100)
        pol.on_access(2, 200)
        pol.on_access(3, 300)

        victim = pol.select_victim({1, 2, 3})
        assert victim == 1  # accessed earliest

    def test_access_refreshes_position(self):
        pol = LRUPolicy()
        pol.on_access(1, 100)
        pol.on_access(2, 200)
        pol.on_access(1, 300)  # re-access → moves to end

        victim = pol.select_victim({1, 2})
        assert victim == 2  # 2 is now least recent


class TestLFUPolicy:
    def test_evicts_least_frequent(self):
        pol = LFUPolicy()
        pol.on_access(1, 100)
        pol.on_access(1, 200)  # freq=2
        pol.on_access(2, 300)  # freq=1

        victim = pol.select_victim({1, 2})
        assert victim == 2  # least frequent


class TestFIFOPolicy:
    def test_evicts_first_inserted(self):
        pol = FIFOPolicy()
        pol.on_access(10, 100)
        pol.on_access(20, 200)
        pol.on_access(30, 300)

        victim = pol.select_victim({10, 20, 30})
        assert victim == 10  # first in

    def test_access_doesnt_reorder(self):
        pol = FIFOPolicy()
        pol.on_access(10, 100)
        pol.on_access(20, 200)
        pol.on_access(10, 300)  # re-access — should NOT change order

        victim = pol.select_victim({10, 20})
        assert victim == 10  # still first inserted


class TestMLDrivenPolicy:
    def test_evicts_lowest_probability(self):
        pol = MLDrivenPolicy()
        pol.on_access(1, 100)
        pol.on_access(2, 200)
        pol.on_access(3, 300)

        pol.update_predictions([
            PredictionResult(file_id=1, probability=0.9),
            PredictionResult(file_id=2, probability=0.1),
            PredictionResult(file_id=3, probability=0.5),
        ])

        victim = pol.select_victim({1, 2, 3})
        assert victim == 2  # lowest probability

    def test_falls_back_to_lru(self):
        pol = MLDrivenPolicy()
        pol.on_access(1, 100)
        pol.on_access(2, 200)
        # No predictions → falls back to LRU
        victim = pol.select_victim({1, 2})
        assert victim == 1  # LRU fallback
