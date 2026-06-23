"""
Alternative bar construction — Volume Bars and Tick Bars.

Standard time bars (15-min candles) distort the data because low-activity
periods get the same weight as high-activity ones. Volume bars create a
new bar every N units of volume, producing bars that carry roughly equal
informational content.

Usage:
    from src.data.bars import make_volume_bars
    vol_bars = make_volume_bars(df, volume_threshold=5000)

Reference:
    López de Prado, M. (2018). Advances in Financial Machine Learning.
    Chapter 2: Financial Data Structures.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.config import FeatureConfig, cfg

logger = logging.getLogger(__name__)


# ─── Public API ───────────────────────────────────────────────────────────────


def make_volume_bars(
    df: pd.DataFrame,
    volume_threshold: Optional[int] = None,
    volume_col: str = "tick_volume",
) -> pd.DataFrame:
    """
    Aggregate time-based candles into volume bars.

    A new bar is emitted every time cumulative ``volume_col`` reaches
    ``volume_threshold``. OHLC values are computed correctly from the
    aggregated candles:
      - open  = first candle's open
      - high  = max of all candle highs
      - low   = min of all candle lows
      - close = last candle's close
      - tick_volume = sum of all candle volumes
      - spread = volume-weighted average spread

    Parameters
    ----------
    df : pd.DataFrame
        Time-bar DataFrame with DatetimeIndex and columns:
        open, high, low, close, tick_volume, spread.
    volume_threshold : int, optional
        Cumulative volume needed to emit a bar.
        Defaults to ``cfg.features.volume_bar_size``.
    volume_col : str
        Column to accumulate. Default 'tick_volume'.

    Returns
    -------
    pd.DataFrame
        Volume-bar DataFrame with irregular DatetimeIndex.
        Same columns as input, plus 'bar_count' (number of source
        candles aggregated into each bar).
    """
    volume_threshold = volume_threshold or cfg.features.volume_bar_size

    logger.info(
        "Constructing volume bars (threshold=%d) from %d time bars ...",
        volume_threshold,
        len(df),
    )

    bars = _aggregate_volume_bars(df, volume_threshold, volume_col)
    result = pd.DataFrame(bars)

    if len(result) == 0:
        logger.warning(
            "No volume bars produced! Threshold %d may be too high "
            "(max cumulative volume in data: %d).",
            volume_threshold,
            df[volume_col].sum(),
        )
        return result

    result.set_index("time", inplace=True)

    logger.info(
        "Produced %d volume bars (%.1f%% of original %d time bars)",
        len(result),
        100 * len(result) / len(df),
        len(df),
    )
    logger.info(
        "  Avg candles/bar: %.1f  |  Avg volume/bar: %.0f",
        result["bar_count"].mean(),
        result["tick_volume"].mean(),
    )

    return result


def make_tick_bars(
    df: pd.DataFrame,
    tick_threshold: int = 1000,
) -> pd.DataFrame:
    """
    Aggregate time-based candles into tick bars.

    Similar to volume bars, but a new bar is emitted every
    ``tick_threshold`` ticks (using tick_volume as a proxy since
    we don't have individual tick data).

    This is functionally identical to volume bars for our data,
    but kept as a separate function for clarity and future
    extensibility (e.g., if we later get true tick data).

    Parameters
    ----------
    df : pd.DataFrame
        Time-bar DataFrame.
    tick_threshold : int
        Number of ticks per bar.

    Returns
    -------
    pd.DataFrame
        Tick-bar DataFrame.
    """
    return make_volume_bars(df, volume_threshold=tick_threshold, volume_col="tick_volume")


def suggest_volume_threshold(
    df: pd.DataFrame,
    target_bar_count: int = 5000,
    volume_col: str = "tick_volume",
) -> int:
    """
    Suggest a volume threshold that would produce approximately
    ``target_bar_count`` bars from the data.

    Parameters
    ----------
    df : pd.DataFrame
        Time-bar DataFrame.
    target_bar_count : int
        Desired number of output bars.
    volume_col : str
        Volume column to use.

    Returns
    -------
    int
        Suggested volume threshold.
    """
    total_volume = df[volume_col].sum()
    threshold = int(total_volume / target_bar_count)
    logger.info(
        "Suggested volume threshold for ~%d bars: %d "
        "(total volume=%d)",
        target_bar_count,
        threshold,
        total_volume,
    )
    return threshold


# ─── Internals ────────────────────────────────────────────────────────────────


def _aggregate_volume_bars(
    df: pd.DataFrame,
    threshold: int,
    volume_col: str,
) -> list[dict]:
    """
    Core aggregation loop — iterate through candles, accumulate volume,
    and emit a bar when the threshold is reached.

    Uses numpy arrays for performance on 50k+ rows.
    """
    # Extract arrays for fast iteration
    times = df.index.values
    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    volumes = df[volume_col].values
    spreads = df["spread"].values

    bars = []
    cum_volume = 0
    bar_open = opens[0]
    bar_high = highs[0]
    bar_low = lows[0]
    bar_start_time = times[0]
    bar_spread_sum = 0.0
    bar_count = 0

    for i in range(len(df)):
        cum_volume += volumes[i]
        bar_high = max(bar_high, highs[i])
        bar_low = min(bar_low, lows[i])
        bar_spread_sum += spreads[i] * volumes[i]  # Volume-weighted spread
        bar_count += 1

        if cum_volume >= threshold:
            # Emit bar
            avg_spread = (
                bar_spread_sum / cum_volume if cum_volume > 0 else spreads[i]
            )
            bars.append(
                {
                    "time": pd.Timestamp(bar_start_time),
                    "open": bar_open,
                    "high": bar_high,
                    "low": bar_low,
                    "close": closes[i],
                    "tick_volume": cum_volume,
                    "spread": avg_spread,
                    "bar_count": bar_count,
                }
            )

            # Reset accumulators for next bar
            cum_volume = 0
            bar_count = 0
            bar_spread_sum = 0.0
            if i + 1 < len(df):
                bar_open = opens[i + 1]
                bar_high = highs[i + 1]
                bar_low = lows[i + 1]
                bar_start_time = times[i + 1]

    # Note: We intentionally discard the last incomplete bar.
    # Partial bars introduce bias (they always have less volume).

    return bars


# ─── Convenience: run as script ───────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    from src.data.loader import load_raw_data

    df = load_raw_data()

    # Suggest a threshold
    threshold = suggest_volume_threshold(df, target_bar_count=5000)

    # Build volume bars
    vol_bars = make_volume_bars(df, volume_threshold=threshold)
    print(f"\nVolume bars sample:")
    print(vol_bars.head(10))
    print(f"\n... total {len(vol_bars)} volume bars")
