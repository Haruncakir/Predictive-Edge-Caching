"""Tests for the FeatureExtractor."""

from pec.feature_extractor import FeatureExtractor
from pec.types import Request


def _make_requests(n: int, file_id: int = 1, base_time: int = 0) -> list[Request]:
    """Helper: generate N requests for the same file."""
    return [
        Request(
            request_id=i + 1,
            user_id=(i % 5) + 1,
            file_id=file_id,
            size_bytes=1024 * (i + 1),
            arrival_ns=base_time + i * 1_000_000,
        )
        for i in range(n)
    ]


class TestFeatureExtractor:
    def test_empty_history(self):
        ext = FeatureExtractor()
        result = ext.extract([], {1, 2}, 0)
        assert result == []

    def test_empty_cached_ids(self):
        ext = FeatureExtractor()
        reqs = _make_requests(5)
        result = ext.extract(reqs, set(), 10_000_000)
        assert result == []

    def test_basic_features(self):
        ext = FeatureExtractor()
        reqs = _make_requests(10, file_id=42)
        cached = {42}
        result = ext.extract(reqs, cached, 20_000_000)

        assert len(result) == 1
        ff = result[0]
        assert ff.file_id == 42
        assert ff.frequency > 0
        assert 0 <= ff.recency <= 1.5  # normalised
        assert ff.user_diversity > 0

    def test_uncached_file_gets_zero_features(self):
        ext = FeatureExtractor()
        reqs = _make_requests(5, file_id=42)
        # File 99 is cached but never appears in history.
        cached = {42, 99}
        result = ext.extract(reqs, cached, 10_000_000)

        assert len(result) == 2
        for ff in result:
            if ff.file_id == 99:
                assert ff.frequency == 0.0
                assert ff.recency == 0.0

    def test_feature_vector_length(self):
        ext = FeatureExtractor()
        reqs = _make_requests(5, file_id=1)
        result = ext.extract(reqs, {1}, 10_000_000)

        assert len(result) == 1
        vec = result[0].to_vector()
        assert len(vec) == 9  # 9 features

    def test_multiple_files(self):
        ext = FeatureExtractor()
        reqs = (
            _make_requests(5, file_id=1, base_time=0)
            + _make_requests(3, file_id=2, base_time=100_000)
        )
        cached = {1, 2}
        result = ext.extract(reqs, cached, 10_000_000)

        assert len(result) == 2
        ids = {ff.file_id for ff in result}
        assert ids == {1, 2}
