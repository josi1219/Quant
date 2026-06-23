"""
Scale-invariant technical indicators.

Every indicator is normalized so it does NOT depend on the absolute price
level. This means the model won't break when EURUSD moves from 1.08 to 1.15.

Rules:
  - RSI: already bounded [0, 100] → no change needed.
  - MACD, EMA crossovers: divide by close price, multiply by 10000 (express in pips).
  - Bollinger position: normalize to [0, 1] within the bands.
  - ATR: divide by close (express as % of price).
  - Returns: already scale-invariant (percentage changes).

Usage:
    from src.features.indicators import compute_all_indicators
    features = compute_all_indicators(df)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.config import FeatureConfig, cfg

logger = logging.getLogger(__name__)


# ─── Public API ───────────────────────────────────────────────────────────────


def compute_all_indicators(
    df: pd.DataFrame,
    config: Optional[FeatureConfig] = None,
) -> pd.DataFrame:
    """
    Compute all scale-invariant technical indicators.

    Parameters
    ----------
    df : pd.DataFrame
        OHLC DataFrame with DatetimeIndex.
    config : FeatureConfig, optional
        Feature configuration. Defaults to ``cfg.features``.

    Returns
    -------
    pd.DataFrame
        DataFrame with all indicator columns appended.
        Original OHLC columns are preserved.
    """
    config = config or cfg.features
    result = df.copy()

    # RSI (already bounded 0-100)
    result["rsi"] = _rsi(result["close"], period=config.rsi_period)

    # MACD family (normalized to pips-relative)
    macd_line, macd_signal, macd_hist = _macd(
        result["close"],
        fast=config.ema_fast,
        slow=config.ema_slow,
        signal=config.macd_signal,
    )
    result["macd_pct"] = (macd_line / result["close"]) * 10_000
    result["macd_signal_pct"] = (macd_signal / result["close"]) * 10_000
    result["macd_hist_pct"] = (macd_hist / result["close"]) * 10_000

    # Bollinger Band position (0 = at lower band, 1 = at upper band)
    bb_upper, bb_middle, bb_lower = _bollinger_bands(
        result["close"],
        period=config.bb_period,
        std_mult=config.bb_std,
    )
    bb_width = bb_upper - bb_lower
    # Avoid division by zero when bands collapse
    result["bb_position"] = np.where(
        bb_width > 0,
        (result["close"] - bb_lower) / bb_width,
        0.5,  # midpoint when bands are flat
    )
    # Bollinger bandwidth as % of price (volatility proxy)
    result["bb_width_pct"] = (bb_width / result["close"]) * 10_000

    # ATR as percentage of price
    result["atr_pct"] = (
        _atr(result["high"], result["low"], result["close"], period=config.atr_period)
        / result["close"]
    )

    # EMA crossover (fast - slow) as pips-relative
    ema_f = result["close"].ewm(span=config.ema_fast, adjust=False).mean()
    ema_s = result["close"].ewm(span=config.ema_slow, adjust=False).mean()
    result["ema_cross_pct"] = ((ema_f - ema_s) / result["close"]) * 10_000

    # Multi-horizon returns (already scale-invariant)
    for h in config.return_horizons:
        result[f"returns_{h}"] = result["close"].pct_change(h)

    # Log returns (for statistical properties)
    result["log_return_1"] = np.log(result["close"] / result["close"].shift(1))

    # Realized volatility (rolling std of log returns)
    result["realized_vol_96"] = (
        result["log_return_1"].rolling(window=96).std() * np.sqrt(96)
    )

    n_features = len(result.columns) - len(df.columns)
    logger.info("Computed %d scale-invariant indicators", n_features)

    return result


# ─── Internal indicator calculations ─────────────────────────────────────────


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index.

    Uses exponential moving average for the up/down components
    (Wilder's smoothing method).
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    MACD (Moving Average Convergence Divergence).

    Returns the raw (un-normalized) MACD line, signal line, and histogram.
    Normalization is done in the caller.
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - macd_signal
    return macd_line, macd_signal, macd_hist


def _bollinger_bands(
    close: pd.Series,
    period: int = 20,
    std_mult: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands. Returns (upper, middle, lower)."""
    middle = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper = middle + std_mult * std
    lower = middle - std_mult * std
    return upper, middle, lower


def _atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Average True Range.

    True Range = max(high-low, abs(high-prev_close), abs(low-prev_close))
    ATR = EMA of True Range.
    """
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return atr


# ─── Convenience: run as script ───────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    from src.data.loader import load_raw_data

    df = load_raw_data()
    features = compute_all_indicators(df)

    # Show which columns were added
    new_cols = [c for c in features.columns if c not in df.columns]
    print(f"\nNew feature columns ({len(new_cols)}):")
    for col in new_cols:
        non_null = features[col].dropna().shape[0]
        print(f"  {col:25s}  non-null: {non_null:6d}  mean: {features[col].mean():10.4f}")
