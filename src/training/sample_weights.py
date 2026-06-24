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
        (weights ** 2).sum() ** -1 * weights.sum() if len(weights) > 0 else 0,
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

    Uniqueness at time t = 1 / concurrency(t).
    Average uniqueness of event i = mean(uniqueness(t)) for t in [t0_i, t1_i].
    """
    uniqueness_values = []

    for t0, row in labels_df.iterrows():
        t1 = row["t1"]
        if pd.isna(t1):
            uniqueness_values.append(1.0)
            continue

        # Get concurrency values in [t0, t1]
        mask = (concurrency.index >= t0) & (concurrency.index <= t1)
        event_concurrency = concurrency.loc[mask]

        if len(event_concurrency) == 0 or event_concurrency.sum() == 0:
            uniqueness_values.append(1.0)
        else:
            # Uniqueness = 1/c at each timestep, averaged over the event
            avg_u = (1.0 / event_concurrency).mean()
            uniqueness_values.append(avg_u)

    return pd.Series(uniqueness_values, index=labels_df.index)


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
