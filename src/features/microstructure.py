"""
Market microstructure features — volume dynamics, spread analysis,
price impact, and temporal patterns.

These features capture the *how* of price movements, not just the *what*.
Volume spikes, widening spreads, and abnormal price impact often precede
significant moves.

Usage:
    from src.features.microstructure import compute_microstructure_features
    features = compute_microstructure_features(df)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.config import FeatureConfig, cfg

logger = logging.getLogger(__name__)


# ─── Public API ───────────────────────────────────────────────────────────────


def compute_microstructure_features(
    df: pd.DataFrame,
    config: Optional[FeatureConfig] = None,
) -> pd.DataFrame:
    """
    Compute market microstructure features.

    Parameters
    ----------
    df : pd.DataFrame
        OHLC DataFrame with tick_volume and spread columns.
    config : FeatureConfig, optional
        Defaults to ``cfg.features``.

    Returns
    -------
    pd.DataFrame
        DataFrame with microstructure columns appended.
    """
    config = config or cfg.features
    result = df.copy()
    window = config.rolling_window

    # --- Volume features ---
    result["volume_zscore"] = _rolling_zscore(
        result["tick_volume"].astype(float), window
    )
    # Volume relative to daily average
    vol_daily_avg = result["tick_volume"].rolling(window=config.micro_vol_lookback, min_periods=20).mean()
    result["volume_ratio"] = result["tick_volume"] / vol_daily_avg.replace(0, np.nan)

    # --- Spread features ---
    result["spread_zscore"] = _rolling_zscore(
        result["spread"].astype(float), window
    )

    # --- Price impact (Kyle's Lambda proxy) ---
    # How much does price move per unit of volume?
    returns = result["close"].pct_change()
    # Avoid division by zero
    safe_volume = result["tick_volume"].replace(0, np.nan).astype(float)
    result["kyle_lambda"] = returns.abs() / safe_volume
    # Smooth it with a rolling median (more robust than mean for this)
    result["kyle_lambda_smooth"] = (
        result["kyle_lambda"].rolling(window=config.micro_vol_lookback, min_periods=20).median()
    )

    # --- Amihud Illiquidity ---
    # abs(return) / (volume × price) — higher = more illiquid
    result["amihud"] = returns.abs() / (safe_volume * result["close"])
    result["amihud_smooth"] = (
        result["amihud"].rolling(window=config.micro_vol_lookback, min_periods=20).median()
    )

    # --- VPIN (simplified) ---
    # Volume-synchronized probability of informed trading
    # Simplified version: ratio of buy vs sell volume estimated from candle shape
    result["vpin"] = _compute_vpin(result, lookback=config.micro_vpin_lookback)

    # --- Bar structure features ---
    # Candle body ratio: (close - open) / (high - low)
    candle_range = (result["high"] - result["low"]).replace(0, np.nan)
    result["body_ratio"] = (result["close"] - result["open"]) / candle_range

    # Upper/lower shadow ratios
    result["upper_shadow_ratio"] = (
        result["high"] - result[["open", "close"]].max(axis=1)
    ) / candle_range
    result["lower_shadow_ratio"] = (
        result[["open", "close"]].min(axis=1) - result["low"]
    ) / candle_range

    # --- Temporal features (cyclical encoding) ---
    if isinstance(df.index, pd.DatetimeIndex):
        hour = df.index.hour + df.index.minute / 60.0
        result["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        result["hour_cos"] = np.cos(2 * np.pi * hour / 24)

        dow = df.index.dayofweek  # 0=Monday, 4=Friday
        result["dow_sin"] = np.sin(2 * np.pi * dow / 5)
        result["dow_cos"] = np.cos(2 * np.pi * dow / 5)
    else:
        logger.warning("Index is not DatetimeIndex; skipping temporal features")

    n_features = len(result.columns) - len(df.columns)
    logger.info("Computed %d microstructure features", n_features)

    return result


# ─── Internals ────────────────────────────────────────────────────────────────


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """
    Compute rolling z-score using ONLY past data.

    z[t] = (x[t] - mean(x[t-W:t])) / std(x[t-W:t])

    This is leak-proof: at time t, we only use data up to and including t.
    """
    rolling_mean = series.rolling(window=window, min_periods=20).mean()
    rolling_std = series.rolling(window=window, min_periods=20).std()
    # Avoid division by zero
    zscore = (series - rolling_mean) / rolling_std.replace(0, np.nan)
    return zscore


def _compute_vpin(df: pd.DataFrame, lookback: int = 48) -> pd.Series:
    """
    Simplified VPIN (Volume-synchronized Probability of Informed Trading).

    Estimates the fraction of volume that is "directional" (informed)
    vs. "random" (uninformed) using the bulk volume classification method.

    For each candle, we estimate buy volume using:
        buy_vol = tick_volume * (close - low) / (high - low)
        sell_vol = tick_volume - buy_vol

    VPIN = rolling_mean(|buy_vol - sell_vol|) / rolling_mean(tick_volume)

    Higher VPIN → more informed trading → potential for large moves.
    """
    candle_range = (df["high"] - df["low"]).replace(0, np.nan)
    buy_fraction = (df["close"] - df["low"]) / candle_range
    buy_fraction = buy_fraction.fillna(0.5)  # If high == low, assume balanced

    buy_vol = df["tick_volume"] * buy_fraction
    sell_vol = df["tick_volume"] * (1 - buy_fraction)
    order_imbalance = (buy_vol - sell_vol).abs()

    vpin = (
        order_imbalance.rolling(window=lookback, min_periods=20).mean()
        / df["tick_volume"].rolling(window=lookback, min_periods=20).mean().replace(0, np.nan)
    )
    return vpin


# ─── Convenience: run as script ───────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    from src.data.loader import load_raw_data

    df = load_raw_data()
    features = compute_microstructure_features(df)

    new_cols = [c for c in features.columns if c not in df.columns]
    print(f"\nMicrostructure features ({len(new_cols)}):")
    for col in new_cols:
        non_null = features[col].dropna().shape[0]
        print(f"  {col:25s}  non-null: {non_null:6d}")
