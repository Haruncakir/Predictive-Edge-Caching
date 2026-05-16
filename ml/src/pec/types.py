"""Canonical data types mirroring the C++ Request/Feedback PODs.

These dataclasses are the Python equivalents of the C++ types defined in
``include/Request.hpp``. They are intentionally kept simple — plain data
containers with no behaviour — so that every module can import them
without pulling in heavy dependencies.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field


@dataclass(slots=True)
class Request:
    """A single user-equipment file request.

    Fields mirror ``pec::Request`` in the C++ codebase:
    - **request_id** — monotonically increasing, globally unique.
    - **user_id** — identifies UE 1 … UE N.
    - **file_id** — content identifier (the "video" or "page").
    - **size_bytes** — payload size; needed for size-aware caching.
    - **arrival_ns** — simulation-time arrival in nanoseconds.
    """

    request_id: int
    user_id: int
    file_id: int
    size_bytes: int
    arrival_ns: int


class CacheOutcome(enum.IntEnum):
    """Result of a cache lookup — mirrors ``pec::CacheOutcome``."""

    PENDING = 0
    HIT = 1
    MISS = 2


@dataclass(slots=True)
class Feedback:
    """Cache outcome posted back through the feedback loop.

    Mirrors ``pec::Feedback``.
    """

    request_id: int
    outcome: CacheOutcome
    served_ns: int = 0


@dataclass(slots=True)
class FileFeatures:
    """Per-file feature vector produced by the FeatureExtractor.

    Each field corresponds to one input dimension for the ML model.
    """

    file_id: int = 0
    frequency: float = 0.0          # request count in window
    recency: float = 0.0            # normalised time since last request
    inter_arrival_mean: float = 0.0  # mean gap between consecutive reqs
    inter_arrival_cv: float = 0.0   # coeff. of variation of gaps
    user_diversity: float = 0.0     # distinct users / total users
    norm_size: float = 0.0          # normalised file size
    request_rate: float = 0.0       # frequency / window duration
    time_sin: float = 0.0           # cyclical hour-of-day (sin)
    time_cos: float = 0.0           # cyclical hour-of-day (cos)

    # Convenience: return feature vector as a list (excludes file_id).
    def to_vector(self) -> list[float]:
        return [
            self.frequency,
            self.recency,
            self.inter_arrival_mean,
            self.inter_arrival_cv,
            self.user_diversity,
            self.norm_size,
            self.request_rate,
            self.time_sin,
            self.time_cos,
        ]

    @staticmethod
    def feature_names() -> list[str]:
        return [
            "frequency",
            "recency",
            "inter_arrival_mean",
            "inter_arrival_cv",
            "user_diversity",
            "norm_size",
            "request_rate",
            "time_sin",
            "time_cos",
        ]


@dataclass
class PredictionResult:
    """ML model output: predicted probability a file will be requested soon."""

    file_id: int
    probability: float  # P(requested again within horizon T)
