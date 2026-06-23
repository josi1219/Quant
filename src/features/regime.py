"""
Volatility regime detection and momentum quality features.

Markets alternate between trending and mean-reverting regimes.
A model that treats all market conditions the same will fail.
These features help the model adapt its behavior to the current regime.

Features:
  - vol_regime: short-term vol / long-term vol ratio
  - vol_of_vol: instability of volatility itself
  - momentum_quality: how "clean" is the current move?
  - return_acceleration: is momentum increasing or fading?
  - consecutive_direction: persistence of directional bars

Usage:
    from src.features.regime import compute_regime_features
    df = compute_regime_features(df, config)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.config import FeatureConfig, cfg

logger = logging.getLogger(__name__)


# ─── Public API ───────────────────────────────────────────────────────────────


def compute_regime_features(
    df: pd.DataFrame,
    config: Optional[FeatureConfig] = None,
) -> pd.DataFrame:
    """
    Compute volatility regime and momentum quality features.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with at least 'close' and 'high'/'low' columns.
    config : FeatureConfig, optional
        Feature configuration.

    Returns
    -------
    pd.DataFrame
        DataFrame with regime feature columns appended.
    """
    config = config or cfg.features
    result = df.copy()
    close = result["close"]

    log_returns = np.log(close / close.shift(1))

    # ── Volatility regime ─────────────────────────────────────────────────
    # Ratio of short-term vol to long-term vol
    # > 1.0 = expanding volatility (trending market)
    # < 1.0 = contracting volatility (ranging market)
    short_vol = log_returns.rolling(
        window=config.vol_regime_fast, min_periods=10
    ).std()
    long_vol = log_returns.rolling(
        window=config.vol_regime_slow, min_periods=50
    ).std()
    result["vol_regime"] = short_vol / long_vol.replace(0, np.nan)

    # ── Volatility of volatility ──────────────────────────────────────────
    # How unstable is volatility itself?
    # High vol-of-vol = regime is changing = be cautious
    atr_pct = result.get("atr_pct")
    if atr_pct is None:
        # Compute ATR % if not already present
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                result["high"] - result["low"],
                (result["high"] - prev_close).abs(),
                (result["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr_pct = tr.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean() / close

    result["vol_of_vol"] = atr_pct.rolling(
        window=config.vol_regime_slow, min_periods=50
    ).std()

    # ── Momentum quality ──────────────────────────────────────────────────
    # Ratio of net return to sum of absolute returns over N bars
    # 1.0 = perfectly straight move (strong persistent momentum)
    # Near 0 = choppy, directionless (noise)
    lookback = config.momentum_quality_lookback
    net_return = close.pct_change(lookback).abs()
    abs_returns_sum = (
        close.pct_change(1).abs()
        .rolling(window=lookback, min_periods=lookback)
        .sum()
    )
    result["momentum_quality"] = net_return / abs_returns_sum.replace(0, np.nan)

    # ── Return acceleration ───────────────────────────────────────────────
    # Second derivative of price: is momentum increasing or fading?
    # Positive = momentum accelerating, Negative = momentum decelerating
    returns_1 = close.pct_change(1)
    returns_prev = returns_1.shift(1)
    result["return_acceleration"] = returns_1 - returns_prev

    # Smoothed version (less noisy)
    result["return_acceleration_smooth"] = result["return_acceleration"].rolling(
        window=6, min_periods=3  # 30 minutes of M5 bars
    ).mean()

    # ── Consecutive direction count ───────────────────────────────────────
    # How many consecutive bars have moved in the same direction?
    # Large counts = strong persistent momentum
    direction = np.sign(returns_1)
    result["consecutive_direction"] = _consecutive_count(direction)

    n_features = 6  # vol_regime, vol_of_vol, momentum_quality,
    # return_acceleration, return_acceleration_smooth, consecutive_direction
    logger.info("Computed %d regime/momentum features", n_features)

    return result


# ─── Internals ────────────────────────────────────────────────────────────────


def _consecutive_count(direction: pd.Series) -> pd.Series:
    """
    Count consecutive bars moving in the same direction.

    Returns positive values for consecutive up bars, negative for down.
    Resets to 0 when direction changes.

    Example: [1, 1, 1, -1, -1] → [1, 2, 3, -1, -2]
    """
    result = pd.Series(0, index=direction.index, dtype=int)
    count = 0
    prev_dir = 0

    values = direction.values
    counts = np.zeros(len(values), dtype=int)

    for i in range(len(values)):
        if np.isnan(values[i]):
            counts[i] = 0
            count = 0
            prev_dir = 0
            continue

        current_dir = int(values[i])
        if current_dir == prev_dir and current_dir != 0:
            count += current_dir  # +1 for up, -1 for down
        elif current_dir != 0:
            count = current_dir
        else:
            count = 0

        counts[i] = count
        prev_dir = current_dir

    return pd.Series(counts, index=direction.index, dtype=int)
