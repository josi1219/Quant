"""
Sample weights — assign importance weights to training samples based on
their uniqueness (non-redundancy).

Problem: In a trending market, many consecutive bars get the same label
(e.g., "buy"). These overlapping events are highly correlated and effectively
duplicate information. If the model trains equally on all of them, it
over-weights the trend and under-weights the turns.

Solution: Weight each sample by its average uniqueness. If a sample's label
period doesn't overlap with any other sample, it gets weight 1.0. If it
overlaps with 9 other samples, it gets weight ≈ 0.1.

Reference:
    López de Prado, M. (2018). Advances in Financial Machine Learning.
    Chapter 4: Sample Weights.

Usage:
    from src.training.sample_weights import compute_sample_weights
    weights = compute_sample_weights(labels_df)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─── Public API ───────────────────────────────────────────────────────────────


def compute_sample_weights(
    labels_df: pd.DataFrame,
    close: Optional[pd.Series] = None,
) -> pd.Series:
    """
    Compute sample weights based on average uniqueness.

    Parameters
    ----------
    labels_df : pd.DataFrame
        Output of ``apply_triple_barrier()``. Must have columns:
        - 't1': datetime when the barrier was touched
        - 'ret': realized return
        - 'label': integer label
    close : pd.Series, optional
        Close prices (used for return-attribution weighting).
        If None, uses uniqueness-only weighting.

    Returns
    -------
    pd.Series
        Sample weights, indexed to match ``labels_df``.
        Weights are normalized to sum to ``len(labels_df)``.
    """
    # Drop rows without valid labels
    valid = labels_df.dropna(subset=["label", "t1"]).copy()

    if len(valid) == 0:
        logger.warning("No valid labels found for weight computation")
        return pd.Series(dtype=float)

    # Compute concurrency (how many events are active at each time step)
    concurrency = _compute_concurrency(valid)

    # Average uniqueness per event
    uniqueness = _compute_avg_uniqueness(valid, concurrency)

    # Normalize: weights sum to number of samples
    weights = uniqueness / uniqueness.sum() * len(uniqueness)

    logger.info(
        "Sample weights computed: min=%.3f, max=%.3f, mean=%.3f, "
        "effective_samples=%.0f / %d",
        weights.min(),
        weights.max(),
        weights.mean(),
        (weights.sum() ** 2) / (weights ** 2).sum() if len(weights) > 0 else 0,
        len(weights),
    )

    # Reindex to match the full labels_df (fill missing with 1.0)
    return weights.reindex(labels_df.index, fill_value=1.0)


# ─── Internals ────────────────────────────────────────────────────────────────


def _compute_concurrency(labels_df: pd.DataFrame) -> pd.Series:
    """
    Compute the number of concurrent (overlapping) events at each timestamp.
    Uses O(n) vectorized logic instead of O(n^2) loops.
    
    An event is "active" from its start time (the index) to its end time (t1) inclusive.
    Concurrency at time t = (total starts <= t) - (total ends < t).
    """
    t0_series = labels_df.index
    t1_series = labels_df["t1"].dropna()

    # Build a timeline of all relevant timestamps
    all_times = sorted(set(t0_series) | set(t1_series))
    df = pd.DataFrame(index=all_times)
    
    # Count how many events start and end at each timestamp
    df["starts"] = pd.Series(1, index=t0_series).groupby(level=0).sum()
    df["ends"] = pd.Series(1, index=t1_series).groupby(level=0).sum()
    df = df.fillna(0)
    
    # Concurrency at time t is total starts up to t, minus total ends strictly BEFORE t.
    starts_cumsum = df["starts"].cumsum()
    ends_cumsum_shifted = df["ends"].cumsum().shift(1).fillna(0)
    
    concurrency = starts_cumsum - ends_cumsum_shifted
    return concurrency.astype(int)


def _compute_avg_uniqueness(
    labels_df: pd.DataFrame,
    concurrency: pd.Series,
) -> pd.Series:
    """
    Compute the average uniqueness of each event.
    Uses O(n) vectorized cumulative sum lookups instead of O(n^2) loops.
    """
    valid_mask = labels_df["t1"].notna()
    valid_t0 = labels_df.index[valid_mask]
    valid_t1 = labels_df.loc[valid_mask, "t1"]
    
    # Precompute 1/c at each timestamp
    u_t = 1.0 / concurrency.replace(0, np.nan)
    u_t = u_t.fillna(0)
    
    # Cumulative sum of uniqueness and counts
    U = u_t.cumsum()
    C = pd.Series(1, index=concurrency.index).cumsum()
    
    # Shifted cumsums to subtract the strictly-before values
    U_shifted = U.shift(1).fillna(0)
    C_shifted = C.shift(1).fillna(0)
    
    # Fast vectorized O(1) lookups per row
    sum_u = U.loc[valid_t1].values - U_shifted.loc[valid_t0].values
    count = C.loc[valid_t1].values - C_shifted.loc[valid_t0].values
    
    avg_u = sum_u / np.maximum(count, 1)
    
    # Create final series, filling missing t1 with 1.0
    result = pd.Series(1.0, index=labels_df.index)
    result.loc[valid_t0] = avg_u
    
    return result


# ─── Convenience: run as script ───────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    from src.data.loader import load_raw_data
    from src.labels.triple_barrier import apply_triple_barrier

    df = load_raw_data()
    labels = apply_triple_barrier(df["close"])
    weights = compute_sample_weights(labels)

    print(f"\nWeight statistics:")
    print(weights.describe())
