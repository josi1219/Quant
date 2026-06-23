"""
Hyperparameter Tuning Module with Optuna.

Unlike standard ML tuning that only optimizes the model (LightGBM), this
module optimizes BOTH the label generation (Barrier widths, hold times)
AND the model hyperparameters simultaneously.

This allows the AI to find the exact trade duration and Take-Profit/Stop-Loss
ratio that works best for the specific feature set.

Usage:
    from src.training.tuner import run_optuna_tuning
    best_config, best_result = run_optuna_tuning(X, close_aligned)
"""

from __future__ import annotations

import logging
from typing import Optional
import copy

import optuna
import pandas as pd

from src.config import Config, cfg
from src.labels.triple_barrier import get_labels_for_features
from src.training.sample_weights import compute_sample_weights
from src.training.trainer import train_model, TrainingResult

logger = logging.getLogger(__name__)


def run_optuna_tuning(
    X: pd.DataFrame,
    close_aligned: pd.Series,
    base_config: Optional[Config] = None,
    n_trials: int = 100,
) -> tuple[Config, TrainingResult]:
    """
    Run Optuna optimization sweeping both Barrier Config and LightGBM Config.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    close_aligned : pd.Series
        Close prices aligned with X (for label generation).
    base_config : Config, optional
        Base configuration to start from.
    n_trials : int
        Number of trials to run.

    Returns
    -------
    tuple[Config, TrainingResult]
        The optimal configuration and its resulting trained model.
    """
    base_config = base_config or cfg

    # We want a clean console during tuning, only showing major progress
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        """Optuna objective function."""
        # --- 1. Sample Barrier Parameters ---
        # Instead of guessing 1.5x vol and 8 hrs, let Optuna find it!
        pt_mult = trial.suggest_float("pt_multiplier", 0.5, 3.0, step=0.1)
        sl_mult = trial.suggest_float("sl_multiplier", 0.5, 3.0, step=0.1)
        # Hold time: 24 bars (2 hrs) up to 144 bars (12 hrs)
        max_hold = trial.suggest_int("max_holding_period", 24, 144, step=12)

        # Build trial config
        trial_config = copy.deepcopy(base_config)
        trial_config.labels.pt_multiplier = pt_mult
        trial_config.labels.sl_multiplier = sl_mult
        trial_config.labels.max_holding_period = max_hold

        # --- 2. Generate Labels for this trial ---
        labels_df = get_labels_for_features(
            close=close_aligned,
            feature_index=X.index,
            config=trial_config.labels,
        )

        # If labels are completely unbalanced, fail fast
        valid_labels = labels_df["label"].value_counts()
        if len(valid_labels) < 3:
            return -999.0  # Missing buy, sell, or hold labels completely
        # NEW GUARD: Ensure each class has enough samples to be split across folds
        if valid_labels.min() < 100:
            return -999.0  # Too few samples in the minority class (avoids fold crashes)

        # --- 3. Compute Sample Weights ---
        sample_weights = compute_sample_weights(labels_df)

        # --- 4. Sample LightGBM Parameters ---
        lgbm_params = {
            "objective": "multiclass",
            "num_class": 3,
            "metric": "multi_logloss",
            "boosting_type": "gbdt",
            "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "min_child_samples": trial.suggest_int("min_child_samples", 50, 300),
            "subsample": trial.suggest_float("subsample", 0.6, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-2, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
            "random_state": trial_config.training.seed,
            "verbose": -1,
        }
        trial_config.training.lgbm_params = lgbm_params

        # --- 5. Train and Evaluate ---
        try:
            # We use an internal logger override to avoid spamming the console during 100 trials
            logger_lgbm = logging.getLogger("src.training.trainer")
            old_level = logger_lgbm.level
            logger_lgbm.setLevel(logging.ERROR)

            result = train_model(
                X=X,
                labels_df=labels_df,
                sample_weights=sample_weights,
                config=trial_config,
            )

            logger_lgbm.setLevel(old_level)

            # --- 6. The Objective Score ---
            metrics = result.avg_metrics
            if not metrics:
                return -999.0

            sharpe = metrics.get("sharpe_ratio", 0)
            pf = metrics.get("profit_factor", 0)
            net_pnl = metrics.get("total_pnl_after_costs_pips", -999)
            win_rate = metrics.get("win_rate", 0)
            n_trades = metrics.get("n_trades", 0)

            # Penalize models that don't trade enough (we need statistical significance)
            if n_trades < 100:
                return -999.0

            # Penalize models that lose money after costs
            if net_pnl <= 0:
                # Still return the sharpe, but heavily penalized so it ranks them, just poorly
                return sharpe - 10.0

            # Composite Score: Reward high Sharpe, but only if Profit Factor > 1.05 and Win Rate > 45%
            if pf < 1.05 or win_rate < 0.45:
                return sharpe - 5.0

            # If it's a solidly profitable model, optimize for pure Sharpe Ratio
            return sharpe

        except Exception as e:
            logger.error("Trial %d failed: %s", trial.number, e)
            return -999.0

    # Create the study
    logger.info("Starting Optuna Study: Joint Optimization of Barriers + Model")
    logger.info("Trials to run: %d", n_trials)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info("\n" + "=" * 60)
    logger.info("OPTUNA SEARCH COMPLETE")
    logger.info("=" * 60)
    logger.info("Best Trial: %d", study.best_trial.number)
    logger.info("Best Score (Sharpe): %.4f", study.best_value)
    
    best_params = study.best_params
    logger.info("\nOptimal Barrier Settings:")
    logger.info("  pt_multiplier: %.2f", best_params["pt_multiplier"])
    logger.info("  sl_multiplier: %.2f", best_params["sl_multiplier"])
    logger.info("  max_holding_period: %d bars", best_params["max_holding_period"])
    
    # Re-train the best model to return it
    best_config = copy.deepcopy(base_config)
    best_config.labels.pt_multiplier = best_params["pt_multiplier"]
    best_config.labels.sl_multiplier = best_params["sl_multiplier"]
    best_config.labels.max_holding_period = best_params["max_holding_period"]
    
    lgbm_keys = [k for k in best_params.keys() if k not in ["pt_multiplier", "sl_multiplier", "max_holding_period"]]
    best_lgbm_params = {k: best_params[k] for k in lgbm_keys}
    best_lgbm_params.update({
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "random_state": best_config.training.seed,
        "verbose": -1,
    })
    best_config.training.lgbm_params = best_lgbm_params

    logger.info("\nRe-training optimal model to get final metrics...")
    
    # Generate labels one last time with optimal settings
    labels_df = get_labels_for_features(
        close=close_aligned,
        feature_index=X.index,
        config=best_config.labels,
    )
    sample_weights = compute_sample_weights(labels_df)
    
    best_result = train_model(
        X=X,
        labels_df=labels_df,
        sample_weights=sample_weights,
        config=best_config,
    )

    return best_config, best_result
