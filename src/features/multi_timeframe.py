"""
Multi-timeframe feature alignment — merge H1 context features onto
M5 execution data using the leak-proof Shift+Merge pattern.

The core problem: An H1 candle timestamped 09:00 represents the period
09:00→10:00. At M5 bar 09:05, that H1 candle hasn't finished yet.
Naive forward-fill would leak 55 minutes of future data.

The solution:
  1. Compute all H1 features on the raw H1 DataFrame
  2. shift(1) all H1 features — row at 09:00 now holds completed 08:00→09:00 data
  3. merge_asof(direction='backward') onto M5 timestamps

Result: At M5 bar 09:25, the model sees H1 features from the fully closed
08:00→09:00 candle. Zero leakage.

Usage:
    from src.features.multi_timeframe import build_h1_features, merge_h1_to_m5
    h1_features = build_h1_features(df_h1, config)
    df_merged = merge_h1_to_m5(df_m5, h1_features)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.config import FeatureConfig, cfg

logger = logging.getLogger(__name__)


# ─── Public API ───────────────────────────────────────────────────────────────


def build_h1_features(
    df_h1: pd.DataFrame,
    config: Optional[FeatureConfig] = None,
) -> pd.DataFrame:
    """
    Compute all H1-level context features.

    These features provide the "big picture" trend/regime context.
    They will be shifted and merged onto M5 data separately.

    Parameters
    ----------
    df_h1 : pd.DataFrame
        Clean H1 OHLCV DataFrame with DatetimeIndex.
    config : FeatureConfig, optional
        Feature configuration.

    Returns
    -------
    pd.DataFrame
        H1 features with DatetimeIndex. Column names suffixed with '_h1'.
    """
    config = config or cfg.features
    close = df_h1["close"]
    high = df_h1["high"]
    low = df_h1["low"]

    features = pd.DataFrame(index=df_h1.index)

    # ── Trend indicators ──────────────────────────────────────────────────

    # RSI
    features["rsi_h1"] = _rsi(close, period=config.h1_rsi_period)

    # MACD family (normalized to pips-relative)
    macd_line, macd_signal, macd_hist = _macd(
        close,
        fast=config.h1_ema_fast,
        slow=config.h1_ema_slow,
        signal=config.h1_macd_signal,
    )
    features["macd_pct_h1"] = (macd_line / close) * 10_000
    features["macd_signal_pct_h1"] = (macd_signal / close) * 10_000
    features["macd_hist_pct_h1"] = (macd_hist / close) * 10_000

    # EMA crossover (fast - slow) as pips-relative
    ema_f = close.ewm(span=config.h1_ema_fast, adjust=False).mean()
    ema_s = close.ewm(span=config.h1_ema_slow, adjust=False).mean()
    features["ema_cross_h1"] = ((ema_f - ema_s) / close) * 10_000

    # ── Volatility context ────────────────────────────────────────────────

    # ATR as percentage of price
    features["atr_pct_h1"] = (
        _atr(high, low, close, period=config.h1_atr_period) / close
    )

    # Bollinger Band position (0 = lower band, 1 = upper band)
    bb_upper, bb_middle, bb_lower = _bollinger_bands(
        close,
        period=config.h1_bb_period,
        std_mult=config.h1_bb_std,
    )
    bb_width = bb_upper - bb_lower
    features["bb_position_h1"] = np.where(
        bb_width > 0,
        (close - bb_lower) / bb_width,
        0.5,
    )

    # Realized volatility (rolling std of log returns, daily scale)
    log_ret_h1 = np.log(close / close.shift(1))
    features["realized_vol_h1"] = (
        log_ret_h1.rolling(window=24, min_periods=10).std() * np.sqrt(24)
    )

    # ── Regime detection ──────────────────────────────────────────────────

    # Volatility regime: short-term vol / long-term vol
    # > 1.0 = expanding volatility (trend), < 1.0 = contracting (range)
    short_vol = log_ret_h1.rolling(window=4, min_periods=2).std()
    long_vol = log_ret_h1.rolling(window=48, min_periods=10).std()
    features["vol_regime_h1"] = short_vol / long_vol.replace(0, np.nan)

    # Simplified Hurst exponent (rolling)
    features["hurst_h1"] = _rolling_hurst(close, lookback=24)

    # ── Momentum ──────────────────────────────────────────────────────────

    # Multi-horizon returns on H1
    for h in config.h1_return_horizons:
        features[f"returns_{h}_h1"] = close.pct_change(h)

    n_features = len(features.columns)
    logger.info("Computed %d H1 context features", n_features)

    return features


def merge_h1_to_m5(
    df_m5: pd.DataFrame,
    h1_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge H1 features onto M5 timestamps using the leak-proof
    Shift+Merge pattern.

    Steps:
      1. shift(1) the H1 features so each row holds the COMPLETED
         previous hour's data
      2. merge_asof(direction='backward') so each M5 bar gets the
         most recent completed H1 features

    Parameters
    ----------
    df_m5 : pd.DataFrame
        M5 DataFrame (with or without M5 features already added).
    h1_features : pd.DataFrame
        Output of ``build_h1_features()``.

    Returns
    -------
    pd.DataFrame
        M5 DataFrame with H1 feature columns appended.
    """
    logger.info("Merging H1 features onto M5 using Shift+Merge pattern ...")

    # Step 1: Shift H1 features forward by 1 row
    # Row at 09:00 now holds features from the COMPLETED 08:00→09:00 candle
    h1_shifted = h1_features.shift(1).copy()

    # Drop the first row (it's NaN after shifting)
    h1_shifted = h1_shifted.dropna(how="all")

    logger.info(
        "  H1 features shifted: %d rows (first valid: %s)",
        len(h1_shifted),
        h1_shifted.index[0] if len(h1_shifted) > 0 else "N/A",
    )

    # Step 2: merge_asof with direction='backward'
    # For M5 bar at 09:25, find the most recent H1 row ≤ 09:25
    # That's the shifted row at 09:00, which contains 08:00→09:00 data
    m5_reset = df_m5.reset_index()
    h1_reset = h1_shifted.reset_index()

    # Ensure both time columns are datetime
    m5_reset["time"] = pd.to_datetime(m5_reset["time"])
    h1_reset["time"] = pd.to_datetime(h1_reset["time"])

    merged = pd.merge_asof(
        m5_reset,
        h1_reset,
        on="time",
        direction="backward",
    )

    merged.set_index("time", inplace=True)

    n_h1_cols = len(h1_features.columns)
    n_null = merged[h1_features.columns].isna().any(axis=1).sum()
    logger.info(
        "  Merge complete: %d M5 rows × %d H1 features "
        "(%d rows missing H1 data at start)",
        len(merged),
        n_h1_cols,
        n_null,
    )

    return merged


def build_cross_tf_features(
    df: pd.DataFrame,
    config: Optional[FeatureConfig] = None,
) -> pd.DataFrame:
    """
    Compute cross-timeframe alignment features.

    These features measure the RELATIONSHIP between M5 and H1 signals,
    which is where the real edge lies.

    Parameters
    ----------
    df : pd.DataFrame
        Merged M5+H1 DataFrame (output of ``merge_h1_to_m5()``
        after M5 features have been added).
    config : FeatureConfig, optional
        Feature configuration.

    Returns
    -------
    pd.DataFrame
        Same DataFrame with cross-TF feature columns appended.
    """
    config = config or cfg.features
    result = df.copy()

    # Trend alignment: do M5 and H1 EMA crossovers agree?
    # +1 = both bullish, -1 = both bearish, mixed = conflict
    if "ema_cross_pct" in result.columns and "ema_cross_h1" in result.columns:
        m5_direction = np.sign(result["ema_cross_pct"])
        h1_direction = np.sign(result["ema_cross_h1"])
        result["trend_alignment"] = m5_direction * h1_direction

    # RSI divergence: M5 RSI minus H1 RSI
    # Large divergence = potential reversal
    if "rsi" in result.columns and "rsi_h1" in result.columns:
        result["m5_vs_h1_rsi_divergence"] = result["rsi"] - result["rsi_h1"]

    # MACD divergence
    if "macd_hist_pct" in result.columns and "macd_hist_pct_h1" in result.columns:
        m5_macd_dir = np.sign(result["macd_hist_pct"])
        h1_macd_dir = np.sign(result["macd_hist_pct_h1"])
        result["macd_alignment"] = m5_macd_dir * h1_macd_dir

    n_new = len(result.columns) - len(df.columns)
    logger.info("Computed %d cross-timeframe alignment features", n_new)

    return result


# ─── Internal indicator calculations (duplicated from indicators.py to
#     keep the H1 feature builder self-contained and avoid circular deps) ──────


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI using Wilder's smoothing method."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple:
    """MACD line, signal, and histogram."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - macd_signal
    return macd_line, macd_signal, macd_hist


def _atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Average True Range."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def _bollinger_bands(
    close: pd.Series,
    period: int = 20,
    std_mult: float = 2.0,
) -> tuple:
    """Bollinger Bands: (upper, middle, lower)."""
    middle = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper = middle + std_mult * std
    lower = middle - std_mult * std
    return upper, middle, lower


def _rolling_hurst(
    series: pd.Series,
    lookback: int = 24,
    min_periods: int = 10,
) -> pd.Series:
    """
    Vectorized rolling Hurst exponent estimate using the R/S method.

    H > 0.5 → trending (persistent)
    H ≈ 0.5 → random walk
    H < 0.5 → mean-reverting (anti-persistent)

    Uses numpy stride tricks to build the rolling window matrix once and
    compute R/S on all windows in parallel, avoiding the per-window Python
    callback overhead of ``rolling().apply()``.
    """
    from numpy.lib.stride_tricks import as_strided

    log_returns = np.log(series / series.shift(1)).values
    n = len(log_returns)
    result = np.full(n, np.nan)

    if n < lookback:
        return pd.Series(result, index=series.index)

    # Build a (n - lookback + 1, lookback) view using stride tricks — zero copy.
    itemsize = log_returns.strides[0]
    windows = as_strided(
        log_returns,
        shape=(n - lookback + 1, lookback),
        strides=(itemsize, itemsize),
    ).copy()  # copy so downstream ops don't corrupt the original array

    # Count non-NaN values per window to enforce min_periods
    non_nan_counts = np.sum(~np.isnan(windows), axis=1)

    # Mean-adjust each row (ignore NaN)
    means = np.nanmean(windows, axis=1, keepdims=True)
    mean_adj = windows - means

    # Cumulative sum along each row (treating NaN as 0 for robustness)
    cumsums = np.nancumsum(mean_adj, axis=1)

    # R = max - min of cumulative deviations within each window
    r = cumsums.max(axis=1) - cumsums.min(axis=1)

    # S = std (ddof=1) of the raw returns in each window
    s = np.nanstd(windows, axis=1, ddof=1)

    # Valid mask: enough data, non-zero S and R
    valid = (s > 0) & (r > 0) & (non_nan_counts >= min_periods)

    hurst_vals = np.where(
        valid,
        np.log(r / np.where(valid, s, 1.0)) / np.log(lookback),
        np.nan,
    )

    # Place results at the END of each window (standard rolling convention)
    result[lookback - 1 :] = hurst_vals

    return pd.Series(result, index=series.index)
