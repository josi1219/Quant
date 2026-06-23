"""
Cost-aware evaluation metrics for the trading model.

The single most important metric is expected return per trade MINUS
transaction costs. If this is negative, the model is worthless regardless
of accuracy.

Metrics computed:
  - Standard ML metrics (accuracy, precision, recall, F1 per class)
  - Trading metrics (Sharpe ratio, max drawdown, profit factor)
  - Cost-aware metrics (expected return after costs, cost-adjusted PnL)

Usage:
    from src.evaluation.metrics import compute_all_metrics
    metrics = compute_all_metrics(y_true, y_pred, labels_df, cost_config)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from src.config import CostConfig, cfg

logger = logging.getLogger(__name__)


# ─── Public API ───────────────────────────────────────────────────────────────


def compute_all_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    labels_df: Optional[pd.DataFrame] = None,
    cost_config: Optional[CostConfig] = None,
) -> dict:
    """
    Compute all evaluation metrics.

    Parameters
    ----------
    y_true : pd.Series
        True labels ({-1, 0, 1}).
    y_pred : pd.Series
        Predicted labels ({-1, 0, 1}).
    labels_df : pd.DataFrame, optional
        Full labels DataFrame with 'ret_pips' and 'holding_period' columns.
        If provided, trading metrics are computed.
    cost_config : CostConfig, optional
        Transaction cost config. Defaults to ``cfg.costs``.

    Returns
    -------
    dict
        All computed metrics.
    """
    cost_config = cost_config or cfg.costs
    metrics = {}

    # ── ML Metrics ────────────────────────────────────────────────────────
    metrics.update(_ml_metrics(y_true, y_pred))

    # ── Trading Metrics (need labels_df with returns) ─────────────────────
    if labels_df is not None and "ret_pips" in labels_df.columns:
        trading = _trading_metrics(y_pred, labels_df, cost_config)
        metrics.update(trading)

    return metrics


def print_classification_report(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> str:
    """
    Print a formatted classification report.

    Maps labels back to human-readable names:
      -1 → Sell, 0 → Hold, 1 → Buy
    """
    target_names = ["Sell (-1)", "Hold (0)", "Buy (1)"]
    labels = [-1, 0, 1]
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0,
    )
    return report


# ─── Internals ────────────────────────────────────────────────────────────────


def _ml_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """Standard ML classification metrics."""
    metrics = {}

    metrics["accuracy"] = accuracy_score(y_true, y_pred)

    # Per-class precision & recall
    for label, name in [(-1, "sell"), (0, "hold"), (1, "buy")]:
        binary_true = (y_true == label).astype(int)
        binary_pred = (y_pred == label).astype(int)

        if binary_true.sum() > 0:
            metrics[f"precision_{name}"] = precision_score(
                binary_true, binary_pred, zero_division=0
            )
            metrics[f"recall_{name}"] = recall_score(
                binary_true, binary_pred, zero_division=0
            )

    # Weighted F1
    metrics["f1_weighted"] = f1_score(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # Prediction distribution
    total = len(y_pred)
    for label, name in [(-1, "sell"), (0, "hold"), (1, "buy")]:
        count = (y_pred == label).sum()
        metrics[f"pred_pct_{name}"] = count / total if total > 0 else 0

    return metrics


def _trading_metrics(
    y_pred: pd.Series,
    labels_df: pd.DataFrame,
    cost_config: CostConfig,
) -> dict:
    """
    Trading-specific metrics: Sharpe, drawdown, profit factor, etc.

    Simulates trading based on predictions:
    - If model predicts Buy (1): we go long, P&L = actual return
    - If model predicts Sell (-1): we go short, P&L = -actual return
    - If model predicts Hold (0): no trade, P&L = 0
    """
    metrics = {}

    # Align predictions with label returns
    aligned = labels_df.reindex(y_pred.index)
    ret_pips = aligned["ret_pips"].fillna(0)
    holding_period = aligned["holding_period"].fillna(0)

    # Compute PnL per prediction
    # Long: pnl = ret_pips; Short: pnl = -ret_pips; Hold: pnl = 0
    pnl_pips = pd.Series(0.0, index=y_pred.index)
    long_mask = y_pred == 1
    short_mask = y_pred == -1
    pnl_pips[long_mask] = ret_pips[long_mask]
    pnl_pips[short_mask] = -ret_pips[short_mask]

    # Trades = predictions that are NOT hold
    trade_mask = y_pred != 0
    n_trades = trade_mask.sum()
    metrics["n_trades"] = int(n_trades)

    if n_trades == 0:
        logger.warning("No trades predicted — all metrics will be zero")
        metrics["expected_return_pips"] = 0.0
        metrics["expected_return_after_costs_pips"] = 0.0
        metrics["sharpe_ratio"] = 0.0
        metrics["max_drawdown_pips"] = 0.0
        metrics["profit_factor"] = 0.0
        metrics["win_rate"] = 0.0
        return metrics

    trade_pnl = pnl_pips[trade_mask]
    total_cost = cost_config.total_cost_pips

    # ── Core metrics ──────────────────────────────────────────────────
    metrics["expected_return_pips"] = trade_pnl.mean()
    metrics["expected_return_after_costs_pips"] = trade_pnl.mean() - total_cost
    metrics["total_pnl_pips"] = trade_pnl.sum()
    metrics["total_pnl_after_costs_pips"] = (
        trade_pnl.sum() - n_trades * total_cost
    )

    # Win rate
    wins = (trade_pnl > 0).sum()
    metrics["win_rate"] = wins / n_trades

    # Average win / average loss
    winning_trades = trade_pnl[trade_pnl > 0]
    losing_trades = trade_pnl[trade_pnl < 0]
    metrics["avg_win_pips"] = winning_trades.mean() if len(winning_trades) > 0 else 0.0
    metrics["avg_loss_pips"] = losing_trades.mean() if len(losing_trades) > 0 else 0.0

    # Profit factor
    gross_profit = winning_trades.sum() if len(winning_trades) > 0 else 0.0
    gross_loss = abs(losing_trades.sum()) if len(losing_trades) > 0 else 0.0
    metrics["profit_factor"] = (
        gross_profit / gross_loss if gross_loss > 0 else float("inf")
    )

    # Sharpe ratio (annualized)
    # Assume ~250 trading days, ~96 M15 bars/day = 24000 bars/year
    if trade_pnl.std() > 0:
        bars_per_year = 250 * 96
        sharpe = (trade_pnl.mean() / trade_pnl.std()) * np.sqrt(
            bars_per_year / max(1, len(y_pred) / n_trades)
        )
        metrics["sharpe_ratio"] = sharpe
    else:
        metrics["sharpe_ratio"] = 0.0

    # Maximum drawdown
    cum_pnl = pnl_pips.cumsum()
    running_max = cum_pnl.cummax()
    drawdown = cum_pnl - running_max
    metrics["max_drawdown_pips"] = drawdown.min()

    # Average holding period (for trades only)
    trade_holding = holding_period[trade_mask]
    metrics["avg_holding_period"] = (
        trade_holding.mean() if len(trade_holding) > 0 else 0.0
    )

    # Cost analysis
    metrics["total_cost_per_trade_pips"] = total_cost
    metrics["cost_as_pct_of_avg_return"] = (
        total_cost / abs(metrics["expected_return_pips"])
        if metrics["expected_return_pips"] != 0
        else float("inf")
    )

    return metrics


# ─── Convenience: formatting ─────────────────────────────────────────────────


def format_metrics(metrics: dict) -> str:
    """Format metrics dict as a readable string."""
    lines = []
    lines.append("=" * 60)
    lines.append("EVALUATION METRICS")
    lines.append("=" * 60)

    sections = {
        "ML Metrics": [
            "accuracy",
            "f1_weighted",
            "precision_buy",
            "recall_buy",
            "precision_sell",
            "recall_sell",
        ],
        "Trading Metrics": [
            "n_trades",
            "win_rate",
            "profit_factor",
            "sharpe_ratio",
            "max_drawdown_pips",
            "avg_holding_period",
            "pct_filtered",
        ],
        "Cost-Aware Metrics": [
            "expected_return_pips",
            "expected_return_after_costs_pips",
            "total_pnl_pips",
            "total_pnl_after_costs_pips",
            "total_cost_per_trade_pips",
        ],
    }

    for section_name, keys in sections.items():
        lines.append(f"\n  --- {section_name} ---")
        for key in keys:
            val = metrics.get(key)
            if val is not None:
                if isinstance(val, float):
                    lines.append(f"  {key:40s}: {val:10.4f}")
                else:
                    lines.append(f"  {key:40s}: {val}")

    lines.append("=" * 60)
    return "\n".join(lines)
