"""
Fractional Differentiation — make price series stationary while preserving
long-term memory.

Regular differencing (d=1) gives you returns: stationary but memoryless.
No differencing (d=0) gives you raw prices: full memory but non-stationary.
Fractional differencing (d ≈ 0.3-0.5) finds the sweet spot.

The key insight: we want the *minimum* value of d that makes the series
stationary (passes the ADF test). This preserves maximum memory.

Reference:
    López de Prado, M. (2018). Advances in Financial Machine Learning.
    Chapter 5: Fractionally Differentiated Features.

Usage:
    from src.features.fractional_diff import frac_diff, find_min_d
    d_star = find_min_d(df['close'])      # Find optimal d
    fd_close = frac_diff(df['close'], d=d_star)  # Apply
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from src.config import FeatureConfig, cfg

logger = logging.getLogger(__name__)


# ─── Public API ───────────────────────────────────────────────────────────────


def get_weights(d: float, threshold: float = 1e-4) -> np.ndarray:
    """
    Compute the fractional differentiation weights.

    The weights are computed recursively:
        w_0 = 1
        w_k = -w_{k-1} * (d - k + 1) / k

    We stop when abs(w_k) < threshold.

    Parameters
    ----------
    d : float
        Fractional differentiation order (0 < d < 1).
    threshold : float
        Minimum absolute weight to keep. Smaller = more memory preserved
        but slower computation.

    Returns
    -------
    np.ndarray
        Array of weights [w_0, w_1, w_2, ...].
    """
    weights = [1.0]
    k = 1
    while True:
        w_k = -weights[-1] * (d - k + 1) / k
        if abs(w_k) < threshold:
            break
        weights.append(w_k)
        k += 1
        # Safety: cap at 10,000 terms (for very small d or threshold)
        if k > 10_000:
            logger.warning(
                "Weight computation capped at 10,000 terms (d=%.3f, threshold=%.1e)",
                d,
                threshold,
            )
            break
    return np.array(weights)


def frac_diff(
    series: pd.Series,
    d: float,
    threshold: float = 1e-4,
) -> pd.Series:
    """
    Apply fixed-width window fractional differentiation.

    Parameters
    ----------
    series : pd.Series
        The price series to differentiate (e.g., close prices).
    d : float
        Fractional differentiation order (0 < d < 1).
    threshold : float
        Weight threshold (see ``get_weights``).

    Returns
    -------
    pd.Series
        Fractionally differentiated series. The first ``len(weights)-1``
        values will be NaN (warmup period).
    """
    weights = get_weights(d, threshold)
    width = len(weights)
    result = pd.Series(index=series.index, dtype=np.float64)

    logger.debug(
        "Frac diff: d=%.3f, weight_count=%d, warmup=%d bars",
        d,
        width,
        width - 1,
    )

    values = series.values
    for i in range(width - 1, len(values)):
        # Dot product of weights with the window [x_i, x_{i-1}, ..., x_{i-w+1}]
        window = values[i - width + 1 : i + 1][::-1]
        result.iloc[i] = np.dot(weights, window)

    return result


def frac_diff_fast(
    series: pd.Series,
    d: float,
    threshold: float = 1e-4,
) -> pd.Series:
    """
    Vectorized fractional differentiation using convolution.

    Significantly faster than the loop-based ``frac_diff`` for large
    datasets. Produces identical results.

    Parameters
    ----------
    series : pd.Series
        The price series to differentiate.
    d : float
        Fractional differentiation order.
    threshold : float
        Weight threshold.

    Returns
    -------
    pd.Series
        Fractionally differentiated series.
    """
    weights = get_weights(d, threshold)
    width = len(weights)

    # Use numpy convolution in 'full' mode, then trim
    # We want: result[t] = sum(w[k] * x[t-k]) for k=0..width-1
    # This is a standard convolution (not correlation)
    conv = np.convolve(series.values, weights, mode="full")[: len(series)]

    # The first (width-1) values are based on incomplete windows → set to NaN
    result = pd.Series(conv, index=series.index, dtype=np.float64)
    result.iloc[: width - 1] = np.nan

    return result


def find_min_d(
    series: pd.Series,
    d_range: Optional[tuple[float, float]] = None,
    step: float = 0.05,
    threshold: float = 1e-4,
    adf_pvalue: float = 0.05,
) -> float:
    """
    Find the minimum fractional differentiation order ``d`` such that the
    resulting series is stationary (ADF test p-value < ``adf_pvalue``).

    Iterates d from ``d_range[0]`` to ``d_range[1]`` in steps of ``step``,
    and returns the smallest d where the ADF null hypothesis (unit root)
    is rejected.

    Parameters
    ----------
    series : pd.Series
        Original price series.
    d_range : tuple, optional
        (min_d, max_d). Default (0.0, 1.0).
    step : float
        Step size for the search. Default 0.05.
    threshold : float
        Weight threshold for frac diff computation.
    adf_pvalue : float
        Significance level for the ADF test. Default 0.05.

    Returns
    -------
    float
        Minimum d that achieves stationarity.

    Raises
    ------
    ValueError
        If no d in the range produces a stationary series (unlikely for d→1).
    """
    d_range = d_range or (0.0, 1.0)
    d_values = np.arange(d_range[0], d_range[1] + step, step)

    logger.info("Searching for minimum d in [%.2f, %.2f] with step %.2f ...", *d_range, step)

    results = []
    for d in d_values:
        if d == 0:
            fd_series = series.copy()
        else:
            fd_series = frac_diff_fast(series, d, threshold)

        fd_clean = fd_series.dropna()
        if len(fd_clean) < 100:
            logger.warning("d=%.2f: too few samples after frac diff (%d)", d, len(fd_clean))
            continue

        adf_stat, pvalue, *_ = adfuller(fd_clean, maxlag=1, regression="c", autolag=None)
        is_stationary = pvalue < adf_pvalue

        results.append(
            {
                "d": round(d, 3),
                "adf_stat": round(adf_stat, 4),
                "p_value": round(pvalue, 6),
                "stationary": is_stationary,
                "n_samples": len(fd_clean),
            }
        )

        logger.info(
            "  d=%.2f → ADF=%.4f, p=%.6f  %s",
            d,
            adf_stat,
            pvalue,
            "✓ STATIONARY" if is_stationary else "",
        )

        if is_stationary:
            logger.info("Found minimum d = %.2f (ADF p-value = %.6f)", d, pvalue)
            return round(d, 3)

    raise ValueError(
        f"No d in range {d_range} produced a stationary series. "
        f"Last ADF p-value: {results[-1]['p_value'] if results else 'N/A'}"
    )


def get_adf_summary(
    series: pd.Series,
    d_range: Optional[tuple[float, float]] = None,
    step: float = 0.05,
    threshold: float = 1e-4,
) -> pd.DataFrame:
    """
    Compute ADF test statistics for a range of d values.
    Useful for plotting the stationarity vs. memory tradeoff.

    Returns
    -------
    pd.DataFrame
        Columns: d, adf_stat, p_value, corr_with_original
    """
    d_range = d_range or (0.0, 1.0)
    d_values = np.arange(d_range[0], d_range[1] + step, step)
    original = series.dropna()

    rows = []
    for d in d_values:
        if d == 0:
            fd_series = series.copy()
        else:
            fd_series = frac_diff_fast(series, d, threshold)

        fd_clean = fd_series.dropna()
        if len(fd_clean) < 100:
            continue

        adf_stat, pvalue, *_ = adfuller(fd_clean, maxlag=1, regression="c", autolag=None)

        # Correlation with original series (measures memory preservation)
        # Align indices before computing correlation
        common_idx = original.index.intersection(fd_clean.index)
        if len(common_idx) > 0:
            corr = original.loc[common_idx].corr(fd_clean.loc[common_idx])
        else:
            corr = np.nan

        rows.append(
            {
                "d": round(d, 3),
                "adf_stat": round(adf_stat, 4),
                "p_value": round(pvalue, 6),
                "corr_with_original": round(corr, 4) if not np.isnan(corr) else np.nan,
            }
        )

    return pd.DataFrame(rows)


# ─── Convenience: run as script ───────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    from src.data.loader import load_raw_data

    df = load_raw_data()
    d_star = find_min_d(df["close"])
    fd_close = frac_diff_fast(df["close"], d=d_star)
    print(f"\nOptimal d = {d_star}")
    print(f"Frac diff series (non-NaN): {fd_close.dropna().shape[0]} values")
    print(fd_close.dropna().describe())
