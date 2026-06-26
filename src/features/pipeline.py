"""
Master feature pipeline — orchestrates all feature generation from raw
dual-timeframe data to model-ready feature matrix.

Multi-Timeframe Architecture:
  1. Build H1 context features (trend, regime, volatility)
  2. Shift(1) + merge_asof(backward) to align H1 → M5 (leak-proof)
  3. Build M5 execution features (indicators, microstructure)
  4. Compute cross-timeframe alignment features
  5. Add session-aware features
  6. Add regime/momentum features
  7. Rolling z-score normalization of unbounded features
  8. NaN cleanup

Usage:
    from src.features.pipeline import build_features, build_features_mtf
    # New MTF pipeline:
    X, meta = build_features_mtf(df_m5, df_h1)
    # Legacy single-TF pipeline (backward compatible):
    X, meta = build_features(df_raw)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.config import Config, cfg
from src.data.bars import make_volume_bars, suggest_volume_threshold
from src.features.fractional_diff import find_min_d, frac_diff_fast
from src.features.indicators import compute_all_indicators
from src.features.microstructure import compute_microstructure_features
from src.features.multi_timeframe import (
    build_cross_tf_features,
    build_h1_features,
    merge_h1_to_m5,
)
from src.features.regime import compute_regime_features
from src.features.sessions import compute_session_features

logger = logging.getLogger(__name__)

# Features that are already bounded and should NOT be z-scored
BOUNDED_FEATURES = {
    # M5 indicators — already bounded or normalized
    "rsi", "bb_position",
    "body_ratio", "upper_shadow_ratio", "lower_shadow_ratio",
    "vpin", "volume_zscore", "spread_zscore",
    # Cyclical time encodings — bounded [-1, 1]
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    # H1 indicators — already bounded
    "rsi_h1", "bb_position_h1",
    # H1 regime — Hurst exponent is [0, 1]; 0.5 = random walk boundary must be preserved
    "hurst_h1",
    # Cross-TF alignment — sign-based, bounded {-1, 0, 1}
    "trend_alignment", "macd_alignment",
    # Session — bounded by construction
    "session_id", "is_london_ny_overlap", "session_elapsed_pct",
    # Regime — vol_regime and vol_regime_h1 are short/long vol ratios;
    # 1.0 is the neutral boundary and must not be shifted by z-scoring
    "vol_regime", "vol_regime_h1",
    # RSI divergence is bounded [-100, 100] with 0 as natural midpoint
    "m5_vs_h1_rsi_divergence",
    # Momentum quality [0, 1] and consecutive direction (integer count)
    "momentum_quality", "consecutive_direction",
}

# Columns that are raw data, not features (will be excluded from final X)
NON_FEATURE_COLS = {
    "open",
    "high",
    "low",
    "close",
    "tick_volume",
    "spread",
    "bar_count",
}


# ─── Public API ───────────────────────────────────────────────────────────────


def build_features_mtf(
    df_m5: pd.DataFrame,
    df_h1: pd.DataFrame,
    config: Optional[Config] = None,
    frac_d: Optional[float] = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Full MTF pipeline: raw M5 + H1 data → model-ready feature matrix.

    Parameters
    ----------
    df_m5 : pd.DataFrame
        Raw M5 OHLCV data from ``load_m5_data()``.
    df_h1 : pd.DataFrame
        Raw H1 OHLCV data from ``load_h1_data()``.
    config : Config, optional
        Master configuration. Defaults to ``cfg``.
    frac_d : float, optional
        Fractional differentiation order. If None, auto-determined via ADF.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        - Feature matrix (DatetimeIndex, all feature columns, no NaN).
        - Metadata dict with pipeline details.
    """
    config = config or cfg
    metadata = {
        "n_bars_raw_m5": len(df_m5),
        "n_bars_raw_h1": len(df_h1),
        "pipeline": "mtf",
    }

    # ── Step 1: Volume bars on M5 (optional) ──────────────────────────────
    if config.features.use_volume_bars:
        logger.info("Step 1/8: Constructing volume bars from M5 data ...")
        df = make_volume_bars(
            df_m5,
            volume_threshold=config.features.volume_bar_size,
        )
        if len(df) < 500:
            msg = (
                f"Volume bars produced only {len(df)} bars (requires >=500). "
                "The model was trained on volume bars and cannot safely fall back to "
                "time bars. Please fetch more historical data."
            )
            logger.error(msg)
            raise RuntimeError(msg)
        else:
            metadata["used_volume_bars"] = True
    else:
        logger.info("Step 1/8: Skipping volume bars (use_volume_bars=False)")
        df = df_m5.copy()
        metadata["used_volume_bars"] = False

    # ── Step 2: Build H1 context features ─────────────────────────────────
    logger.info("Step 2/8: Building H1 context features ...")
    h1_features = build_h1_features(df_h1, config.features)

    # ── Step 3: Merge H1 → M5 using Shift+Merge pattern ──────────────────
    logger.info("Step 3/8: Merging H1 features onto M5 (shift+merge_asof) ...")
    df = merge_h1_to_m5(df, h1_features)

    # ── Step 4: Fractional differentiation of M5 close ────────────────────
    logger.info("Step 4/8: Fractional differentiation of M5 close ...")
    if frac_d is None:
        frac_d = find_min_d(df["close"])
    fd_close = frac_diff_fast(df["close"], d=frac_d)
    df["frac_diff_close"] = fd_close
    metadata["frac_d"] = frac_d

    # ── Step 5: M5 execution features (indicators + microstructure) ───────
    logger.info("Step 5/8: Computing M5 scale-invariant indicators ...")
    df = compute_all_indicators(df, config.features)

    logger.info("Step 5/8: Computing M5 microstructure features ...")
    df = compute_microstructure_features(df, config.features)

    # ── Step 6: Cross-timeframe alignment features ────────────────────────
    logger.info("Step 6/8: Computing cross-timeframe alignment features ...")
    df = build_cross_tf_features(df, config.features)

    # ── Step 7: Session + Regime features ─────────────────────────────────
    logger.info("Step 7/8: Computing session and regime features ...")
    df = compute_session_features(df, config.features)
    df = compute_regime_features(df, config.features)

    # ── Step 8: Rolling z-score normalization + cleanup ────────────────────
    logger.info("Step 8/8: Applying rolling z-score normalization + cleanup ...")
    df = _apply_rolling_normalization(df, config.features.rolling_window)

    # Separate feature columns from raw data columns
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feature_cols].copy()

    # Drop rows where ANY feature is NaN (warmup period)
    before = len(X)
    X = X.dropna()
    after = len(X)
    logger.info(
        "Dropped %d warmup rows (%d → %d)",
        before - after,
        before,
        after,
    )

    metadata["n_features"] = len(X.columns)
    metadata["feature_names"] = list(X.columns)
    metadata["n_bars_final"] = len(X)

    logger.info("=" * 60)
    logger.info("MTF FEATURE PIPELINE COMPLETE")
    logger.info("  Input M5 bars:   %d", metadata["n_bars_raw_m5"])
    logger.info("  Input H1 bars:   %d", metadata["n_bars_raw_h1"])
    logger.info("  Output bars:     %d", metadata["n_bars_final"])
    logger.info("  Features:        %d", metadata["n_features"])
    logger.info("  Frac diff d:     %.3f", metadata["frac_d"])
    logger.info("  Volume bars:     %s", metadata["used_volume_bars"])
    logger.info("=" * 60)

    return X, metadata


def build_features(
    df_raw: pd.DataFrame,
    config: Optional[Config] = None,
    frac_d: Optional[float] = None,
) -> tuple[pd.DataFrame, dict]:
    """
    LEGACY: Single-timeframe pipeline for backward compatibility.

    Full pipeline: raw M15 data → model-ready feature matrix.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Raw OHLC data from ``load_raw_data()``.
    config : Config, optional
        Master configuration. Defaults to ``cfg``.
    frac_d : float, optional
        Fractional differentiation order.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        - Feature matrix (DatetimeIndex, all feature columns, no NaN).
        - Metadata dict.
    """
    config = config or cfg
    metadata = {"n_bars_raw": len(df_raw), "pipeline": "legacy"}

    # ── Step 1: Volume bars (optional) ────────────────────────────────────
    if config.features.use_volume_bars:
        logger.info("Step 1/6: Constructing volume bars ...")
        df = make_volume_bars(
            df_raw,
            volume_threshold=config.features.volume_bar_size,
        )
        if len(df) < 200:
            msg = (
                f"Volume bars produced only {len(df)} bars (requires >=200). "
                "The model was trained on volume bars and cannot safely fall back to "
                "time bars. Please fetch more historical data."
            )
            logger.error(msg)
            raise RuntimeError(msg)
        else:
            metadata["used_volume_bars"] = True
    else:
        logger.info("Step 1/6: Skipping volume bars (use_volume_bars=False)")
        df = df_raw.copy()
        metadata["used_volume_bars"] = False

    # ── Step 2: Fractional differentiation ────────────────────────────────
    logger.info("Step 2/6: Fractional differentiation of close ...")
    if frac_d is None:
        frac_d = find_min_d(df["close"])
    fd_close = frac_diff_fast(df["close"], d=frac_d)
    df["frac_diff_close"] = fd_close
    metadata["frac_d"] = frac_d

    # ── Step 3: Technical indicators ──────────────────────────────────────
    logger.info("Step 3/6: Computing scale-invariant indicators ...")
    df = compute_all_indicators(df, config.features)

    # ── Step 4: Microstructure features ───────────────────────────────────
    logger.info("Step 4/6: Computing microstructure features ...")
    df = compute_microstructure_features(df, config.features)

    # ── Step 5: Rolling z-score normalization ─────────────────────────────
    logger.info("Step 5/6: Applying rolling z-score normalization ...")
    df = _apply_rolling_normalization(df, config.features.rolling_window)

    # ── Step 6: Cleanup ───────────────────────────────────────────────────
    logger.info("Step 6/6: Cleaning up NaN values ...")
    # Separate feature columns from raw data columns
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feature_cols].copy()

    # Drop rows where ANY feature is NaN (warmup period)
    before = len(X)
    X = X.dropna()
    after = len(X)
    logger.info(
        "Dropped %d warmup rows (%d → %d)",
        before - after,
        before,
        after,
    )

    metadata["n_features"] = len(X.columns)
    metadata["feature_names"] = list(X.columns)
    metadata["n_bars_final"] = len(X)

    logger.info("=" * 60)
    logger.info("FEATURE PIPELINE COMPLETE")
    logger.info("  Input bars:    %d", metadata["n_bars_raw"])
    logger.info("  Output bars:   %d", metadata["n_bars_final"])
    logger.info("  Features:      %d", metadata["n_features"])
    logger.info("  Frac diff d:   %.3f", metadata["frac_d"])
    logger.info("  Volume bars:   %s", metadata["used_volume_bars"])
    logger.info("=" * 60)

    return X, metadata


def get_aligned_close(
    df_raw: pd.DataFrame,
    X: pd.DataFrame,
) -> pd.Series:
    """
    Get the close prices aligned with the feature matrix index.

    Useful for labeling and evaluation (we need prices aligned with features).

    Parameters
    ----------
    df_raw : pd.DataFrame
        Original raw data (or volume bars) with 'close' column.
    X : pd.DataFrame
        Feature matrix from ``build_features()`` or ``build_features_mtf()``.

    Returns
    -------
    pd.Series
        Close prices indexed to match X.
    """
    return df_raw["close"].reindex(X.index)


# ─── Internals ────────────────────────────────────────────────────────────────


def _apply_rolling_normalization(
    df: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    """
    Apply rolling z-score normalization to all unbounded features.

    Bounded features (RSI, bb_position, sin/cos encodings, etc.) are
    left unchanged.

    The z-score uses ONLY past data:
        z[t] = (x[t] - mean(x[t-W:t])) / std(x[t-W:t])

    This is critical for preventing data leakage.
    """
    result = df.copy()

    # Identify columns to normalize (all feature columns except bounded ones)
    cols_to_normalize = [
        c
        for c in result.columns
        if c not in NON_FEATURE_COLS and c not in BOUNDED_FEATURES
    ]

    n_normalized = 0
    for col in cols_to_normalize:
        series = result[col]
        if series.dtype in [np.float64, np.float32, float]:
            rolling_mean = series.rolling(window=window, min_periods=20).mean()
            rolling_std = series.rolling(window=window, min_periods=20).std()
            # Replace in-place
            result[col] = (series - rolling_mean) / rolling_std.replace(0, np.nan)
            n_normalized += 1

    logger.info(
        "Z-score normalized %d features (window=%d). "
        "Kept %d features unbounded (already bounded).",
        n_normalized,
        window,
        len(BOUNDED_FEATURES),
    )

    return result


# ─── Convenience: run as script ───────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    # Try MTF pipeline first, fall back to legacy
    try:
        from src.data.loader import load_dual_timeframe

        df_m5, df_h1 = load_dual_timeframe()
        X, meta = build_features_mtf(df_m5, df_h1)
    except FileNotFoundError:
        logger.info("M5/H1 files not found, falling back to legacy pipeline")
        from src.data.loader import load_raw_data

        df_raw = load_raw_data()
        X, meta = build_features(df_raw)

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Features: {meta['feature_names']}")
    print(f"\nSample:")
    print(X.head())
    print(f"\nDescriptive stats:")
    print(X.describe().T[["mean", "std", "min", "max"]])
