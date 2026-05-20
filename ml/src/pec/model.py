"""ML Predictor — XGBoost-based popularity forecaster.

This module implements the "Classification Model" and "Popularity Forecaster"
boxes from the architecture diagram.  It wraps an XGBoost binary classifier
that predicts ``P(file requested again within horizon T)`` for each file
currently in the cache.

DESIGN CHOICES
──────────────
* XGBoost (Chen & Guestrin, KDD 2016) is the primary model.  Gradient-
  boosted trees handle the tabular features from FeatureExtractor far better
  than logistic regression, and the library supports incremental training
  via ``xgb_model`` which we use for online updates.

* An EWMA (exponentially weighted moving average) of per-file request rates
  is blended with the XGBoost score.  This provides a reasonable baseline
  prediction even before the model has been trained (cold-start).

* Training uses a sliding window of labelled examples.  The label for a
  prediction made at time *t* is determined retrospectively: if the file
  *was* requested again within [t, t + horizon], the label is 1.

* Thread-safety is NOT a concern here — the Python simulation is single-
  threaded by design.  Concurrency lives in the C++ producer side.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
import xgboost as xgb

from .config import MLConfig
from .feature_extractor import FeatureExtractor
from .types import FileFeatures, PredictionResult, Request

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class MLPredictor:
    """XGBoost-based content popularity predictor.

    Parameters
    ----------
    cfg : MLConfig
        Model hyper-parameters and training schedule.
    """

    def __init__(self, cfg: MLConfig | None = None) -> None:
        self.cfg = cfg or MLConfig()
        self._extractor = FeatureExtractor()

        # XGBoost booster — initialised lazily on first train().
        self._booster: xgb.Booster | None = None
        self._xgb_params: dict = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "learning_rate": self.cfg.learning_rate,
            "max_depth": self.cfg.max_depth,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "seed": self.cfg.seed,
            "verbosity": 0,
        }

        # EWMA of per-file request rates (file_id → smoothed rate).
        self._ewma: dict[int, float] = defaultdict(float)

        # Training buffer — (feature_vector, label) pairs.
        self._train_X: list[list[float]] = []
        self._train_y: list[int] = []
        self._training_steps: int = 0
        self._last_trained_samples: int = 0

        # Pending predictions awaiting labels.
        self._pending: list[_PendingPrediction] = []

    # ── Public API ────────────────────────────────────────────────────

    def predict(
        self,
        history: list[Request],
        cached_file_ids: set[int],
        current_time_ns: int,
    ) -> list[PredictionResult]:
        """Predict re-request probability for each cached file.

        Returns a list sorted by *descending* probability (most likely to
        be re-requested first).
        """
        features = self._extractor.extract(history, cached_file_ids, current_time_ns)
        if not features:
            return []

        results: list[PredictionResult] = []

        if self._booster is not None:
            X = np.array([f.to_vector() for f in features], dtype=np.float32)
            dmat = xgb.DMatrix(X, feature_names=FileFeatures.feature_names())
            ml_scores = self._booster.predict(dmat)
        else:
            ml_scores = None

        for i, ff in enumerate(features):
            # EWMA component — always available, acts as cold-start fallback.
            ewma_score = self._ewma.get(ff.file_id, 0.0)

            if ml_scores is not None:
                # Blend ML score with EWMA.
                alpha = self.cfg.ewma_alpha
                score = (1 - alpha) * float(ml_scores[i]) + alpha * ewma_score
            else:
                score = ewma_score

            score = float(np.clip(score, 0.0, 1.0))
            results.append(PredictionResult(file_id=ff.file_id, probability=score))

        # Sort descending — highest probability first.
        results.sort(key=lambda r: r.probability, reverse=True)
        return results

    def observe(self, request: Request) -> None:
        """Update EWMA counters with a new request observation."""
        fid = request.file_id
        alpha = self.cfg.ewma_alpha
        self._ewma[fid] = alpha * 1.0 + (1 - alpha) * self._ewma.get(fid, 0.0)

        # Decay all other files slightly.
        decay = 1 - alpha * 0.01
        for other_fid in list(self._ewma):
            if other_fid != fid:
                self._ewma[other_fid] *= decay

    def record_prediction(
        self,
        features: list[FileFeatures],
        current_time_ns: int,
    ) -> None:
        """Store features for later labelling once the horizon elapses."""
        for ff in features:
            self._pending.append(
                _PendingPrediction(
                    file_id=ff.file_id,
                    features=ff.to_vector(),
                    timestamp_ns=current_time_ns,
                )
            )

    def label_and_train(
        self,
        current_time_ns: int,
        recent_file_ids: set[int],
    ) -> None:
        """Label pending predictions and retrain if enough data.

        A pending prediction is labelled **1** if the file appeared in
        ``recent_file_ids`` (i.e. was requested within the horizon), else **0**.
        """
        horizon = self.cfg.prediction_horizon_ns
        still_pending: list[_PendingPrediction] = []

        for pp in self._pending:
            if current_time_ns - pp.timestamp_ns >= horizon:
                label = 1 if pp.file_id in recent_file_ids else 0
                self._train_X.append(pp.features)
                self._train_y.append(label)
            else:
                still_pending.append(pp)

        self._pending = still_pending

        # Retrain periodically once we have enough labelled samples.
        sample_count = len(self._train_y)
        if sample_count < self.cfg.min_training_samples:
            return

        # Train only when we have accumulated a full new interval of labels.
        # This prevents expensive repeated retrains over nearly identical data.
        if sample_count - self._last_trained_samples >= self.cfg.retrain_interval:
            self._train()

    def force_train(self) -> None:
        """Force a training step (useful at end of simulation)."""
        if len(self._train_y) >= self.cfg.min_training_samples:
            self._train()

    # ── Observability ─────────────────────────────────────────────────

    @property
    def training_steps(self) -> int:
        return self._training_steps

    @property
    def training_samples(self) -> int:
        return len(self._train_y)

    @property
    def has_model(self) -> bool:
        return self._booster is not None

    # ── Internals ─────────────────────────────────────────────────────

    def _train(self) -> None:
        """(Re)train the XGBoost model on accumulated labelled data."""
        X = np.array(self._train_X[-10_000:], dtype=np.float32)  # sliding cap
        y = np.array(self._train_y[-10_000:], dtype=np.float32)

        dtrain = xgb.DMatrix(
            X, label=y, feature_names=FileFeatures.feature_names()
        )

        # Retrain from scratch on the sliding window each time.
        # Incremental training (xgb_model=self._booster) causes unbounded
        # tree accumulation — after many retrains the booster grows to
        # tens of thousands of trees and serialization hangs. Retraining
        # from scratch on the most recent 10k samples is fast and still
        # captures evolving popularity patterns.
        self._booster = xgb.train(
            self._xgb_params,
            dtrain,
            num_boost_round=min(self.cfg.n_estimators, 50),
            verbose_eval=False,
        )
        self._training_steps += 1
        self._last_trained_samples = len(self._train_y)
        logger.debug(
            "Trained step %d on %d samples (pos_rate=%.3f)",
            self._training_steps,
            len(y),
            float(y.mean()),
        )


class _PendingPrediction:
    """Internal bookkeeping for a prediction awaiting its label."""

    __slots__ = ("file_id", "features", "timestamp_ns")

    def __init__(
        self, file_id: int, features: list[float], timestamp_ns: int
    ) -> None:
        self.file_id = file_id
        self.features = features
        self.timestamp_ns = timestamp_ns
