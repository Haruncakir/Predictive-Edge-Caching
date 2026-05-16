"""Feature Extractor — transforms a window of raw Requests into per-file
feature vectors suitable for the ML model.

This is the "Feature Extractor" box inside the ML Predictor block of the
architecture diagram.  Given a sliding window of the most recent K requests
(provided by the C++ HistoryBuffer / or our Python replay), it computes a
feature vector for *every file currently in the cache*.

DESIGN CHOICES
──────────────
* All features are scalar and normalised to roughly [0, 1] so that
  gradient-boosted trees and any future neural-net variant converge well
  without manual feature scaling.

* Cyclical hour-of-day encoding via sin/cos avoids the discontinuity at
  midnight that a raw hour value would introduce (Géron, *Hands-On Machine
  Learning*, 3rd ed., §2).

* Coefficient of variation (CV) of inter-arrival times distinguishes
  bursty content (high CV) from steady periodic content (low CV), which
  is critical for proactive prefetching (Li et al., "Caching as a Service
  for 5G Networks", IEEE Access, 2019).
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from .types import FileFeatures, Request


class FeatureExtractor:
    """Stateless extractor — call :meth:`extract` with a history snapshot."""

    # Simulation-time → "hour of day" conversion factor.  The C++ simulator
    # uses nanosecond timestamps starting from 0; we map one simulated hour
    # to 3.6 × 10¹² ns.
    _NS_PER_HOUR: float = 3_600_000_000_000.0

    def extract(
        self,
        history: list[Request],
        cached_file_ids: set[int],
        current_time_ns: int,
    ) -> list[FileFeatures]:
        """Build a feature vector for every file in *cached_file_ids*.

        Parameters
        ----------
        history:
            Sliding window of the most recent requests (oldest → newest).
        cached_file_ids:
            Set of ``file_id`` values currently stored in the cache.
        current_time_ns:
            Simulation clock at the moment of prediction.

        Returns
        -------
        One :class:`FileFeatures` per cached file, in arbitrary order.
        """
        if not history or not cached_file_ids:
            return []

        # ── per-file accumulators ─────────────────────────────────────
        timestamps: dict[int, list[int]] = defaultdict(list)
        users: dict[int, set[int]] = defaultdict(set)
        sizes: dict[int, int] = {}

        for req in history:
            fid = req.file_id
            timestamps[fid].append(req.arrival_ns)
            users[fid].add(req.user_id)
            sizes[fid] = req.size_bytes  # last-seen size

        window_start = history[0].arrival_ns
        window_duration = max(current_time_ns - window_start, 1)

        # Global stats for normalisation.
        all_users = {r.user_id for r in history}
        num_users = max(len(all_users), 1)
        max_size = max((r.size_bytes for r in history), default=1) or 1

        features: list[FileFeatures] = []
        for fid in cached_file_ids:
            ff = FileFeatures(file_id=fid)

            ts = timestamps.get(fid)
            if ts is None:
                # File is cached but never appeared in the history window —
                # all features stay at their zero-initialised defaults.
                features.append(ff)
                continue

            ts_sorted = sorted(ts)
            count = len(ts_sorted)

            # frequency — raw count, capped at window size for normalisation
            ff.frequency = count / max(len(history), 1)

            # recency — time since last request (lower = more recent)
            last_ts = ts_sorted[-1]
            ff.recency = (current_time_ns - last_ts) / window_duration

            # inter-arrival statistics
            if count >= 2:
                gaps = np.diff(ts_sorted).astype(float)
                mean_gap = float(gaps.mean())
                ff.inter_arrival_mean = mean_gap / window_duration
                std_gap = float(gaps.std())
                ff.inter_arrival_cv = (std_gap / mean_gap) if mean_gap > 0 else 0.0
            else:
                ff.inter_arrival_mean = 1.0  # only one observation
                ff.inter_arrival_cv = 0.0

            # user diversity
            ff.user_diversity = len(users[fid]) / num_users

            # normalised file size
            ff.norm_size = sizes.get(fid, 0) / max_size

            # request rate (per ns, normalised by window)
            ff.request_rate = count / window_duration * 1e9  # per-second

            # cyclical time-of-day (from the *last* request for this file)
            hour = (last_ts / self._NS_PER_HOUR) % 24.0
            ff.time_sin = math.sin(2 * math.pi * hour / 24.0)
            ff.time_cos = math.cos(2 * math.pi * hour / 24.0)

            features.append(ff)

        return features
