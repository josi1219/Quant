"""
Purged Walk-Forward Cross-Validation.

Standard TimeSeriesSplit is not safe for financial data because triple-barrier
labels can span multiple bars. If a training sample's label period extends
into the test set, you get data leakage.

Purged Walk-Forward CV fixes this by:
  1. Removing training samples whose label period (t0→t1) overlaps with
     the test period ("purging").
  2. Adding an embargo buffer after the training set to prevent information
     from bleeding through autocorrelation.

Reference:
    López de Prado, M. (2018). Advances in Financial Machine Learning.
    Chapter 7: Cross-Validation in Finance.

Usage:
    from src.training.purged_cv import PurgedWalkForwardCV
    cv = PurgedWalkForwardCV(n_splits=5, embargo_pct=0.01)
    for train_idx, test_idx in cv.split(X, labels_df):
        ...
"""

from __future__ import annotations

import logging
from typing import Generator, Optional

import numpy as np
import pandas as pd

from src.config import TrainingConfig, cfg

logger = logging.getLogger(__name__)


class PurgedWalkForwardCV:
    """
    Purged Walk-Forward Cross-Validation splitter.

    Compatible with scikit-learn's cross-validation interface.

    Parameters
    ----------
    n_splits : int
        Number of CV folds.
    embargo_pct : float
        Fraction of the test set size to use as an embargo buffer.
        The embargo is added at the end of the training set.
    """

    def __init__(
        self,
        n_splits: int = 5,
        embargo_pct: float = 0.01,
    ):
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def split(
        self,
        X: pd.DataFrame,
        labels_df: Optional[pd.DataFrame] = None,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train/test index pairs.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with DatetimeIndex.
        labels_df : pd.DataFrame, optional
            Labels DataFrame with 't1' column (event end times).
            If provided, purging is applied. If None, only embargo is applied.

        Yields
        ------
        train_indices : np.ndarray
            Integer indices into X for training.
        test_indices : np.ndarray
            Integer indices into X for testing.
        """
        n_samples = len(X)
        indices = np.arange(n_samples)

        # Calculate test size (expanding training window)
        test_size = n_samples // (self.n_splits + 1)
        embargo_size = int(test_size * self.embargo_pct)

        logger.info(
            "PurgedWalkForwardCV: n_splits=%d, test_size=%d, embargo=%d",
            self.n_splits,
            test_size,
            embargo_size,
        )

        for i in range(self.n_splits):
            # Test set: sliding window
            test_start = test_size + i * test_size
            test_end = test_start + test_size
            test_end = min(test_end, n_samples)

            test_indices = indices[test_start:test_end]

            # Training set: everything before test set minus embargo
            train_end = test_start - embargo_size
            train_indices = indices[:train_end]

            # Purging: remove training samples whose labels overlap with test
            if labels_df is not None and "t1" in labels_df.columns:
                train_indices = self._purge(
                    train_indices, test_indices, X, labels_df
                )

            if len(train_indices) == 0:
                logger.warning("Fold %d: empty training set after purging!", i)
                continue

            logger.info(
                "  Fold %d: train=%d [%s → %s], test=%d [%s → %s]",
                i,
                len(train_indices),
                X.index[train_indices[0]],
                X.index[train_indices[-1]],
                len(test_indices),
                X.index[test_indices[0]],
                X.index[test_indices[-1]],
            )

            yield train_indices, test_indices

    def _purge(
        self,
        train_indices: np.ndarray,
        test_indices: np.ndarray,
        X: pd.DataFrame,
        labels_df: pd.DataFrame,
    ) -> np.ndarray:
        """
        Remove training samples whose label period overlaps with the test set.

        A training sample at time t0 with label end time t1 is purged if
        t1 >= test_start (its label "leaks" into the test period).
        """
        test_start_time = X.index[test_indices[0]]

        # Get the t1 values for training samples
        train_times = X.index[train_indices]
        t1_values = labels_df.reindex(train_times)["t1"]

        # Keep training samples where t1 < test_start
        # (their label period doesn't overlap with test)
        keep_mask = t1_values.isna() | (t1_values < test_start_time)
        purged_count = (~keep_mask).sum()

        if purged_count > 0:
            logger.debug(
                "    Purged %d training samples (%.1f%%)",
                purged_count,
                100 * purged_count / len(train_indices),
            )

        return train_indices[keep_mask.values]

    def get_n_splits(self) -> int:
        """Return the number of splits."""
        return self.n_splits


# ─── Convenience ──────────────────────────────────────────────────────────────


def validate_no_leakage(
    X: pd.DataFrame,
    labels_df: pd.DataFrame,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> bool:
    """
    Verify that no training sample's label period overlaps with the test set.

    This is a safety check you can run on each fold to confirm purging worked.

    Returns True if no leakage detected, raises ValueError otherwise.
    """
    test_start = X.index[test_indices[0]]
    test_end = X.index[test_indices[-1]]

    train_times = X.index[train_indices]
    t1_values = labels_df.reindex(train_times)["t1"].dropna()

    leaking = t1_values[t1_values >= test_start]
    if len(leaking) > 0:
        raise ValueError(
            f"DATA LEAKAGE DETECTED: {len(leaking)} training samples have "
            f"label periods extending into the test set "
            f"(test starts at {test_start}). "
            f"Leaking t1 values: {leaking.head()}"
        )

    return True
