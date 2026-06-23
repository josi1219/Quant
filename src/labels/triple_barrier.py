"""
Triple-Barrier Labeling Method — calibrated for M5 intraday swing trades.

Instead of naive "will price go up next bar?" labels, this creates labels
based on actual trade outcomes: which barrier does price hit first?

Three barriers:
  1. Upper barrier (Take-Profit): price rises by pt_multiplier × daily_vol
  2. Lower barrier (Stop-Loss): price falls by sl_multiplier × daily_vol
  3. Time barrier: after max_holding_period bars, check where price is

Labels:
   1 = Upper barrier hit first (Buy signal)
  -1 = Lower barrier hit first (Sell signal)
   0 = Time barrier hit, price stayed flat (No signal)

Enhancements over baseline:
  - Symmetric barriers (PT = SL) for unbiased labels
  - Dynamic barrier widths based on volatility regime
  - Minimum holding period to avoid noise-triggered exits
  - min_return_pips filter for time-barrier exits
  - Class distribution monitoring with warnings

Reference:
    López de Prado, M. (2018). Advances in Financial Machine Learning.
    Chapter 3: Labeling.

Usage:
    from src.labels.triple_barrier import apply_triple_barrier
    labels = apply_triple_barrier(close_prices)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.config import LabelConfig, cfg

logger = logging.getLogger(__name__)


# ─── Public API ───────────────────────────────────────────────────────────────


def apply_triple_barrier(
    close: pd.Series,
    config: Optional[LabelConfig] = None,
) -> pd.DataFrame:
    """
    Apply the triple-barrier labeling method.

    Parameters
    ----------
    close : pd.Series
        Close prices with DatetimeIndex.
    config : LabelConfig, optional
        Labeling configuration. Defaults to ``cfg.labels``.

    Returns
    -------
    pd.DataFrame
        Columns:
        - 'label': int in {-1, 0, 1}
        - 't1': datetime when the barrier was touched
        - 'ret': realized return when barrier was touched (in price units)
        - 'ret_pips': realized return in pips
        - 'barrier_type': str — 'upper', 'lower', or 'time'
        - 'holding_period': int — number of bars held

        The last ``max_holding_period`` rows will be NaN (can't compute
        forward-looking labels at the end of the dataset).
    """
    config = config or cfg.labels
    pip_value = cfg.costs.pip_value

    # Step 1: Compute daily volatility (rolling)
    daily_vol = _estimate_daily_volatility(close, lookback=config.vol_lookback)

    # Step 2: Compute regime-adaptive multipliers (optional)
    if config.dynamic_barriers:
        vol_multiplier = _compute_vol_regime_multiplier(
            close, config.vol_regime_fast, config.vol_regime_slow
        )
    else:
        vol_multiplier = pd.Series(1.0, index=close.index)

    # Step 3: Compute barriers for each bar
    logger.info(
        "Applying triple-barrier (PT=%.1f×vol, SL=%.1f×vol, hold=%d-%d bars, "
        "min_ret=%.1f pips, dynamic=%s) ...",
        config.pt_multiplier,
        config.sl_multiplier,
        config.min_holding_period,
        config.max_holding_period,
        config.min_return_pips,
        config.dynamic_barriers,
    )

    results = _compute_barriers(close, daily_vol, vol_multiplier, config, pip_value)
    labels_df = pd.DataFrame(results, index=close.index[: len(results)])

    # Report label distribution
    valid_labels = labels_df["label"].dropna()
    if len(valid_labels) > 0:
        label_counts = valid_labels.value_counts().sort_index()
        total = len(valid_labels)
        logger.info("Label distribution:")
        for label_val, count in label_counts.items():
            pct = 100 * count / total
            name = {-1: "Sell", 0: "Hold", 1: "Buy"}.get(int(label_val), "?")
            logger.info("  %+d (%s): %5d (%.1f%%)", int(label_val), name, count, pct)

            # Class imbalance warning
            if pct / 100 > config.max_class_pct:
                logger.warning(
                    "⚠️  Class '%s' exceeds %.0f%% (at %.1f%%). "
                    "Consider adjusting barrier widths or min_return_pips.",
                    name,
                    config.max_class_pct * 100,
                    pct,
                )

    # Report barrier type distribution
    valid_barriers = labels_df["barrier_type"].dropna()
    if len(valid_barriers) > 0:
        barrier_counts = valid_barriers.value_counts()
        logger.info("Barrier hit distribution:")
        for barrier_type, count in barrier_counts.items():
            pct = 100 * count / len(valid_barriers)
            logger.info("  %s: %5d (%.1f%%)", barrier_type, count, pct)

    return labels_df


def get_labels_for_features(
    close: pd.Series,
    feature_index: pd.DatetimeIndex,
    config: Optional[LabelConfig] = None,
) -> pd.DataFrame:
    """
    Compute triple-barrier labels and align them with the feature matrix index.

    Parameters
    ----------
    close : pd.Series
        Full close price series.
    feature_index : pd.DatetimeIndex
        Index of the feature matrix (from ``build_features``).
    config : LabelConfig, optional
        Labeling configuration.

    Returns
    -------
    pd.DataFrame
        Labels aligned with feature_index. Rows where labels couldn't
        be computed (e.g., end of dataset) are dropped.
    """
    labels_df = apply_triple_barrier(close, config)
    # Align with feature index
    aligned = labels_df.reindex(feature_index)
    # Drop rows without valid labels
    aligned = aligned.dropna(subset=["label"])
    aligned["label"] = aligned["label"].astype(int)
    return aligned


# ─── Internals ────────────────────────────────────────────────────────────────


def _estimate_daily_volatility(
    close: pd.Series,
    lookback: int = 288,
) -> pd.Series:
    """
    Estimate daily volatility using rolling standard deviation of returns.

    For M5 bars, 288 bars = 1 day.
    For M15 bars, 96 bars = 1 day (backward compat).
    """
    log_returns = np.log(close / close.shift(1))
    # Rolling std of log returns, scaled to daily
    daily_vol = log_returns.rolling(window=lookback, min_periods=20).std()
    # Scale to daily price movement
    daily_vol_price = daily_vol * close * np.sqrt(lookback)
    return daily_vol_price


def _compute_vol_regime_multiplier(
    close: pd.Series,
    fast_window: int,
    slow_window: int,
) -> pd.Series:
    """
    Compute a dynamic barrier multiplier based on the volatility regime.

    In high-vol regimes (expanding): widen barriers (multiplier > 1)
    In low-vol regimes (contracting): tighten barriers (multiplier < 1)

    The multiplier is clipped to [0.5, 2.0] to prevent extreme values.
    """
    log_returns = np.log(close / close.shift(1))
    short_vol = log_returns.rolling(window=fast_window, min_periods=10).std()
    long_vol = log_returns.rolling(window=slow_window, min_periods=50).std()

    # Ratio of short to long vol
    ratio = short_vol / long_vol.replace(0, np.nan)

    # Scale: ratio of 1.0 = neutral, >1 = expanding, <1 = contracting
    # Apply a dampening factor so barriers don't swing too wildly
    multiplier = 1.0 + 0.3 * (ratio - 1.0)  # Dampen by 30%
    multiplier = multiplier.clip(0.5, 2.0).fillna(1.0)

    return multiplier


def _compute_barriers(
    close: pd.Series,
    daily_vol: pd.Series,
    vol_multiplier: pd.Series,
    config: LabelConfig,
    pip_value: float,
) -> list[dict]:
    """
    Core barrier computation loop.

    For each bar i, look forward up to max_holding_period bars and
    determine which barrier is hit first.

    Enhancements:
      - Dynamic barriers scaled by vol_multiplier
      - min_holding_period: barriers hit before this are recorded but
        the trade continues
    """
    prices = close.values
    vols = daily_vol.values
    vol_mults = vol_multiplier.values
    n = len(prices)
    max_hold = config.max_holding_period
    min_hold = config.min_holding_period

    results = []
    for i in range(n):
        # Can't label if we can't look forward enough
        if i + max_hold >= n:
            results.append(
                {
                    "label": np.nan,
                    "t1": pd.NaT,
                    "ret": np.nan,
                    "ret_pips": np.nan,
                    "barrier_type": None,
                    "holding_period": np.nan,
                }
            )
            continue

        if np.isnan(vols[i]) or vols[i] <= 0:
            results.append(
                {
                    "label": np.nan,
                    "t1": pd.NaT,
                    "ret": np.nan,
                    "ret_pips": np.nan,
                    "barrier_type": None,
                    "holding_period": np.nan,
                }
            )
            continue

        entry_price = prices[i]
        vm = vol_mults[i]
        upper_barrier = entry_price + config.pt_multiplier * vm * vols[i]
        lower_barrier = entry_price - config.sl_multiplier * vm * vols[i]

        # Scan forward
        hit_barrier = False
        for j in range(1, max_hold + 1):
            idx = i + j
            current_price = prices[idx]

            # Only register barrier hits AFTER min_holding_period
            if j >= min_hold:
                # Check upper barrier (take-profit → buy signal)
                if current_price >= upper_barrier:
                    results.append(
                        {
                            "label": 1,
                            "t1": close.index[idx],
                            "ret": current_price - entry_price,
                            "ret_pips": (current_price - entry_price) / pip_value,
                            "barrier_type": "upper",
                            "holding_period": j,
                        }
                    )
                    hit_barrier = True
                    break

                # Check lower barrier (stop-loss → sell signal)
                if current_price <= lower_barrier:
                    results.append(
                        {
                            "label": -1,
                            "t1": close.index[idx],
                            "ret": current_price - entry_price,
                            "ret_pips": (current_price - entry_price) / pip_value,
                            "barrier_type": "lower",
                            "holding_period": j,
                        }
                    )
                    hit_barrier = True
                    break

        # Time barrier hit
        if not hit_barrier:
            final_idx = i + max_hold
            final_price = prices[final_idx]
            ret = final_price - entry_price
            ret_pips = ret / pip_value

            # Assign label based on return at time barrier
            if abs(ret_pips) < config.min_return_pips:
                label = 0
            elif ret > 0:
                label = 1
            else:
                label = -1

            results.append(
                {
                    "label": label,
                    "t1": close.index[final_idx],
                    "ret": ret,
                    "ret_pips": ret_pips,
                    "barrier_type": "time",
                    "holding_period": max_hold,
                }
            )

    return results


# ─── Convenience: run as script ───────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    from src.data.loader import load_raw_data

    df = load_raw_data()
    labels = apply_triple_barrier(df["close"])

    print(f"\nLabels shape: {labels.shape}")
    print(f"\nBarrier type distribution:")
    print(labels["barrier_type"].value_counts())
    print(f"\nAverage holding period: {labels['holding_period'].mean():.1f} bars")
    print(f"Average return (pips): {labels['ret_pips'].mean():.2f}")
    print(f"\nSample:")
    print(labels.dropna().head(10))
