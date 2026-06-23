"""
Trading session detection and session-relative features.

EURUSD behavior changes dramatically across trading sessions:
  - Asian session (00:00-07:00 UTC): Low volume, tight ranges, mean-reverting
  - London session (07:00-16:00 UTC): 60%+ of daily volume, trending
  - New York session (13:00-22:00 UTC): High volatility, especially overlap
  - London/NY overlap (13:00-16:00 UTC): Most volatile period

These features let the model learn session-specific patterns instead of
treating all hours equally.

Usage:
    from src.features.sessions import compute_session_features
    df = compute_session_features(df, config)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.config import FeatureConfig, cfg

logger = logging.getLogger(__name__)

# Session definitions: (name, start_hour_utc, end_hour_utc)
# Note: NY session wraps and overlaps with London
SESSION_DEFS = {
    "asian":  (0, 7),
    "london": (7, 16),
    "ny":     (13, 22),
}


# ─── Public API ───────────────────────────────────────────────────────────────


def compute_session_features(
    df: pd.DataFrame,
    config: Optional[FeatureConfig] = None,
) -> pd.DataFrame:
    """
    Compute trading-session-aware features.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with DatetimeIndex (M5 bars).
    config : FeatureConfig, optional
        Feature configuration with session boundary hours.

    Returns
    -------
    pd.DataFrame
        DataFrame with session feature columns appended.
    """
    config = config or cfg.features
    result = df.copy()

    if not isinstance(result.index, pd.DatetimeIndex):
        logger.warning("Index is not DatetimeIndex; skipping session features")
        return result

    hours = result.index.hour + result.index.minute / 60.0

    # ── Session identification ────────────────────────────────────────────
    # Determine which session each bar belongs to
    # Priority: London/NY overlap > London > NY > Asian > Off-hours
    session_id = pd.Series(3, index=result.index, dtype=int)  # Default: off-hours

    asian_mask = (hours >= config.session_asian_start) & (hours < config.session_asian_end)
    london_mask = (hours >= config.session_london_start) & (hours < config.session_london_end)
    ny_mask = (hours >= config.session_ny_start) & (hours < config.session_ny_end)
    overlap_mask = london_mask & ny_mask

    session_id[asian_mask] = 0   # Asian
    session_id[london_mask] = 1  # London
    session_id[ny_mask] = 2      # New York
    # Overlap gets its own treatment via the overlap feature below

    result["session_id"] = session_id
    result["is_london_ny_overlap"] = overlap_mask.astype(int)

    # ── Session elapsed percentage ────────────────────────────────────────
    # How far into the current session are we? (0.0 = just started, 1.0 = ending)
    session_elapsed = pd.Series(0.5, index=result.index, dtype=float)

    for mask, (start, end) in [
        (asian_mask, (config.session_asian_start, config.session_asian_end)),
        (london_mask, (config.session_london_start, config.session_london_end)),
        (ny_mask, (config.session_ny_start, config.session_ny_end)),
    ]:
        duration = end - start
        if duration > 0:
            elapsed = (hours[mask] - start) / duration
            session_elapsed[mask] = pd.Series(elapsed).clip(0, 1).values

    result["session_elapsed_pct"] = session_elapsed

    # ── Session-relative range ────────────────────────────────────────────
    # How does today's session range compare to the average?
    # Group by date and session, compute session high-low range
    result["_date"] = result.index.date
    result["_session_range"] = np.nan

    for session_name, (start_h, end_h) in SESSION_DEFS.items():
        mask = (hours >= start_h) & (hours < end_h)
        if mask.sum() == 0:
            continue

        # For each date, compute the session's running range
        # (rolling max high - rolling min low within the session so far)
        session_data = result.loc[mask].copy()
        for date in session_data["_date"].unique():
            date_mask = session_data["_date"] == date
            date_data = session_data.loc[date_mask]
            if len(date_data) == 0:
                continue
            running_high = date_data["high"].expanding().max()
            running_low = date_data["low"].expanding().min()
            running_range = (running_high - running_low) / date_data["close"]
            result.loc[date_data.index, "_session_range"] = running_range.values

    # Compare to rolling average session range
    avg_session_range = result["_session_range"].rolling(
        window=288 * 5, min_periods=288  # 5 trading days
    ).mean()
    result["session_range_pct"] = (
        result["_session_range"] / avg_session_range.replace(0, np.nan)
    )

    # ── Session-relative volume ───────────────────────────────────────────
    # Is the current session's volume higher or lower than average?
    vol_rolling_mean = result["tick_volume"].rolling(
        window=288 * 5, min_periods=288
    ).mean()
    result["session_volume_pct"] = (
        result["tick_volume"] / vol_rolling_mean.replace(0, np.nan)
    )

    # ── Cleanup temp columns ──────────────────────────────────────────────
    result.drop(columns=["_date", "_session_range"], inplace=True)

    n_features = 5  # session_id, is_london_ny_overlap, session_elapsed_pct,
    # session_range_pct, session_volume_pct
    logger.info("Computed %d session features", n_features)

    return result
