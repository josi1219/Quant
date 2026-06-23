"""
Trade Filtering and Meta-Labeling Module.

This module acts as the "Chief Risk Officer" for the primary trading model.
Even if the primary model identifies a Buy or Sell signal, this module can
veto the trade if the confidence is too low.

Usage:
    from src.training.trade_filter import apply_confidence_filter
    y_pred_filtered = apply_confidence_filter(y_pred, y_pred_proba, config)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.config import MetaLabelConfig, cfg

logger = logging.getLogger(__name__)


def apply_confidence_filter(
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
    config: Optional[MetaLabelConfig] = None,
) -> tuple[np.ndarray, dict]:
    """
    Filter out trades where the model's confidence is below the threshold.

    Parameters
    ----------
    y_pred : np.ndarray
        Array of predictions in LightGBM format {0: Sell, 1: Hold, 2: Buy}.
    y_pred_proba : np.ndarray
        Array of prediction probabilities, shape (N, 3).
    config : MetaLabelConfig, optional
        Configuration for thresholding.

    Returns
    -------
    tuple[np.ndarray, dict]
        - y_filtered: The filtered predictions (low confidence forced to 1/Hold).
        - stats: Dictionary of filtering statistics.
    """
    config = config or cfg.meta_label
    
    if not config.enabled:
        return y_pred, {"filtered_out": 0, "pct_filtered": 0.0}

    y_filtered = y_pred.copy()
    
    # max_probs is the probability of the *chosen* class
    max_probs = np.max(y_pred_proba, axis=1)
    
    # We only care about filtering actual trades (Buy=2, Sell=0)
    # If the model already predicted Hold(1), it's not a trade anyway.
    is_trade = (y_pred == 0) | (y_pred == 2)
    low_confidence = max_probs < config.confidence_threshold
    
    # Mask of trades that we want to cancel
    cancel_mask = is_trade & low_confidence
    
    # Force canceled trades to Hold (1)
    y_filtered[cancel_mask] = 1
    
    n_trades_before = is_trade.sum()
    n_canceled = cancel_mask.sum()
    n_trades_after = n_trades_before - n_canceled
    
    pct_filtered = (n_canceled / n_trades_before * 100) if n_trades_before > 0 else 0.0
    
    stats = {
        "trades_before": int(n_trades_before),
        "trades_after": int(n_trades_after),
        "filtered_out": int(n_canceled),
        "pct_filtered": pct_filtered,
        "threshold_used": config.confidence_threshold,
    }
    
    if n_canceled > 0:
        logger.debug(
            "Trade Filter: Canceled %d/%d trades (%.1f%%) below %.2f confidence.",
            n_canceled,
            n_trades_before,
            pct_filtered,
            config.confidence_threshold,
        )
        
    return y_filtered, stats


def determine_position_size(prob: float, config: Optional[MetaLabelConfig] = None) -> float:
    """
    Determine the position size (fraction of max risk) based on confidence.
    
    Returns a float between 0.0 and 1.0.
    """
    config = config or cfg.meta_label
    
    if prob >= config.full_position_threshold:
        return 1.0
    elif prob >= config.half_position_threshold:
        return 0.5
    elif prob >= config.quarter_position_threshold:
        return 0.25
    else:
        return 0.0
