"""
Data loader — load, validate, and clean raw EURUSD CSV data.

Supports dual-timeframe loading (M5 + H1) for the multi-timeframe
pipeline, plus backward-compatible single-file loading.

Usage:
    from src.data.loader import load_dual_timeframe, load_raw_data
    df_m5, df_h1 = load_dual_timeframe()     # New MTF pipeline
    df = load_raw_data()                       # Legacy single-TF
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src.config import DataConfig, cfg

logger = logging.getLogger(__name__)


# ─── Public API ───────────────────────────────────────────────────────────────


def load_dual_timeframe(
    config: Optional[DataConfig] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load both M5 and H1 data for the multi-timeframe pipeline.

    Parameters
    ----------
    config : DataConfig, optional
        Data configuration. Defaults to ``cfg.data``.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (df_m5, df_h1) — both cleaned and validated.

    Raises
    ------
    FileNotFoundError
        If either CSV file does not exist.
    ValueError
        If date ranges don't overlap or data quality checks fail.
    """
    config = config or cfg.data

    logger.info("=" * 60)
    logger.info("LOADING DUAL-TIMEFRAME DATA")
    logger.info("=" * 60)

    df_m5 = load_m5_data(config)
    df_h1 = load_h1_data(config)

    # Validate that both datasets cover a similar date range
    _validate_date_overlap(df_m5, df_h1)

    logger.info("=" * 60)
    logger.info("DUAL-TIMEFRAME LOAD COMPLETE")
    logger.info("  M5: %d candles (%s → %s)", len(df_m5), df_m5.index.min(), df_m5.index.max())
    logger.info("  H1: %d candles (%s → %s)", len(df_h1), df_h1.index.min(), df_h1.index.max())
    logger.info("  Ratio: %.1f M5 bars per H1 bar", len(df_m5) / max(len(df_h1), 1))
    logger.info("=" * 60)

    return df_m5, df_h1


def load_m5_data(
    config: Optional[DataConfig] = None,
    csv_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load and clean the M5 (5-minute) CSV data.

    Parameters
    ----------
    config : DataConfig, optional
        Data configuration. Defaults to ``cfg.data``.
    csv_path : Path, optional
        Override path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Columns: open, high, low, close, tick_volume, spread
        Index: DatetimeIndex named 'time'
    """
    config = config or cfg.data
    path = csv_path or config.raw_csv_m5

    logger.info("Loading M5 data from %s", path)
    df = _read_csv(path)
    _validate_schema(df)
    df = _clean(df, config)
    _report_stats(df, "M5")
    return df


def load_h1_data(
    config: Optional[DataConfig] = None,
    csv_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load and clean the H1 (1-hour) CSV data.

    Parameters
    ----------
    config : DataConfig, optional
        Data configuration. Defaults to ``cfg.data``.
    csv_path : Path, optional
        Override path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Columns: open, high, low, close, tick_volume, spread
        Index: DatetimeIndex named 'time'
    """
    config = config or cfg.data
    path = csv_path or config.raw_csv_h1

    logger.info("Loading H1 data from %s", path)
    df = _read_csv(path)
    _validate_schema(df)
    df = _clean(df, config)
    _report_stats(df, "H1")
    return df


def load_raw_data(
    config: Optional[DataConfig] = None,
    csv_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load the raw CSV and return a clean, validated DataFrame.

    BACKWARD COMPATIBLE: Works with the old M15 pipeline.
    For the new MTF pipeline, use ``load_dual_timeframe()`` instead.

    Parameters
    ----------
    config : DataConfig, optional
        Data configuration. Defaults to ``cfg.data``.
    csv_path : Path, optional
        Override path to the CSV file (useful in Colab where paths differ).

    Returns
    -------
    pd.DataFrame
        Columns: open, high, low, close, tick_volume, spread
        Index: DatetimeIndex named 'time'

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist.
    ValueError
        If critical data quality checks fail.
    """
    config = config or cfg.data
    path = csv_path or config.raw_csv

    logger.info("Loading raw data from %s", path)
    df = _read_csv(path)
    _validate_schema(df)
    df = _clean(df, config)
    _report_stats(df, "RAW")
    return df


# ─── Internals ────────────────────────────────────────────────────────────────


def _read_csv(path: Path) -> pd.DataFrame:
    """Read the CSV with proper dtypes and datetime parsing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(
        path,
        parse_dates=["time"],
        dtype={
            "open": np.float64,
            "high": np.float64,
            "low": np.float64,
            "close": np.float64,
            "tick_volume": np.int64,
            "spread": np.int64,
            "real_volume": np.int64,
        },
    )
    df.set_index("time", inplace=True)
    df.sort_index(inplace=True)
    return df


def _validate_schema(df: pd.DataFrame) -> None:
    """Verify the DataFrame has the expected columns and no critical issues."""
    required = {"open", "high", "low", "close", "tick_volume", "spread"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Check for NaN in OHLC
    ohlc_cols = ["open", "high", "low", "close"]
    nan_counts = df[ohlc_cols].isna().sum()
    if nan_counts.any():
        logger.warning("NaN values found in OHLC:\n%s", nan_counts[nan_counts > 0])

    # Sanity: high >= low, high >= open, high >= close, etc.
    bad_hl = (df["high"] < df["low"]).sum()
    if bad_hl > 0:
        logger.warning("%d candles have high < low (will be dropped)", bad_hl)


def _clean(df: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
    """Apply quality filters and drop unnecessary columns."""
    initial_len = len(df)

    # Drop the real_volume column (always 0 for retail forex)
    if "real_volume" in df.columns:
        df = df.drop(columns=["real_volume"])

    # Drop rows with NaN in OHLC
    ohlc_cols = ["open", "high", "low", "close"]
    df = df.dropna(subset=ohlc_cols)

    # Drop candles where high < low (corrupt data)
    df = df[df["high"] >= df["low"]]

    # Drop low-liquidity candles
    df = df[df["tick_volume"] >= config.min_tick_volume]

    # Drop abnormal spread candles
    # Spread in the CSV is in points (1 point = 0.00001 for 5-digit brokers)
    # Convert max_spread_pips to points for comparison
    max_spread_points = config.max_spread_pips * 10  # 1 pip = 10 points
    df = df[df["spread"] <= max_spread_points]

    dropped = initial_len - len(df)
    if dropped > 0:
        logger.info(
            "Dropped %d of %d candles (%.1f%%) during cleaning",
            dropped,
            initial_len,
            100 * dropped / initial_len,
        )

    return df


def _validate_date_overlap(df_m5: pd.DataFrame, df_h1: pd.DataFrame) -> None:
    """Verify that M5 and H1 datasets cover overlapping date ranges."""
    m5_start, m5_end = df_m5.index.min(), df_m5.index.max()
    h1_start, h1_end = df_h1.index.min(), df_h1.index.max()

    # Check for overlap
    overlap_start = max(m5_start, h1_start)
    overlap_end = min(m5_end, h1_end)

    if overlap_start > overlap_end:
        raise ValueError(
            f"M5 and H1 datasets do not overlap! "
            f"M5: {m5_start} → {m5_end}, H1: {h1_start} → {h1_end}"
        )

    overlap_days = (overlap_end - overlap_start).days
    logger.info(
        "Date overlap: %s → %s (%d days)",
        overlap_start,
        overlap_end,
        overlap_days,
    )

    if overlap_days < 90:
        logger.warning(
            "Only %d days of overlap between M5 and H1. "
            "Recommend at least 365 days for robust training.",
            overlap_days,
        )


def _report_stats(df: pd.DataFrame, label: str = "") -> None:
    """Log summary statistics about the cleaned data."""
    prefix = f"[{label}] " if label else ""
    logger.info("=" * 60)
    logger.info("%sDATA SUMMARY", prefix)
    logger.info("=" * 60)
    logger.info("  Rows:        %d", len(df))
    logger.info("  Date range:  %s → %s", df.index.min(), df.index.max())
    logger.info(
        "  Price range: %.5f → %.5f",
        df["close"].min(),
        df["close"].max(),
    )
    logger.info(
        "  Avg tick vol: %.0f  (min=%d, max=%d)",
        df["tick_volume"].mean(),
        df["tick_volume"].min(),
        df["tick_volume"].max(),
    )
    logger.info(
        "  Avg spread:   %.1f points (max=%d)",
        df["spread"].mean(),
        df["spread"].max(),
    )

    # Detect gaps > 3 hours (expected on weekends, suspicious on weekdays)
    time_diffs = df.index.to_series().diff()
    large_gaps = time_diffs[time_diffs > pd.Timedelta(hours=3)]
    if len(large_gaps) > 0:
        logger.info("  Time gaps > 3h: %d (mostly weekends)", len(large_gaps))

    logger.info("=" * 60)


# ─── Convenience: run as script ───────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    # Try dual-timeframe first, fall back to legacy
    try:
        df_m5, df_h1 = load_dual_timeframe()
        print(f"\nM5: {len(df_m5)} candles")
        print(df_m5.head())
        print(f"\nH1: {len(df_h1)} candles")
        print(df_h1.head())
    except FileNotFoundError:
        logger.info("M5/H1 files not found, falling back to legacy loader")
        data = load_raw_data()
        print(f"\nLoaded {len(data)} clean candles.")
        print(data.head())
