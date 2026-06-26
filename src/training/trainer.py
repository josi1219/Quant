"""
Model training orchestration — train LightGBM with purged walk-forward CV,
sample weights, and Optuna hyperparameter tuning.

This is the central training module. It ties together:
  - Feature matrix + labels
  - Purged walk-forward cross-validation
  - Sample weights (uniqueness-based)
  - LightGBM training with cost-aware evaluation
  - Optional Optuna hyperparameter optimization

Usage:
    from src.training.trainer import train_model, train_with_optuna
    result = train_model(X, labels_df, sample_weights)
    best_result = train_with_optuna(X, labels_df, sample_weights)
"""

from __future__ import annotations

import logging
import copy
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from src.config import Config, cfg
from src.evaluation.metrics import compute_all_metrics
from src.training.purged_cv import PurgedWalkForwardCV, validate_no_leakage
from src.training.sample_weights import compute_sample_weights

logger = logging.getLogger(__name__)


# ─── Result container ─────────────────────────────────────────────────────────


@dataclass
class TrainingResult:
    """Container for training results across all CV folds."""

    model: Any = None  # Final model (trained on all data)
    fold_models: list = field(default_factory=list)
    fold_metrics: list = field(default_factory=list)
    fold_predictions: list = field(default_factory=list)
    feature_importances: Optional[pd.DataFrame] = None
    avg_metrics: Optional[dict] = None
    config: Optional[dict] = None

    def summary(self) -> str:
        """Return a human-readable summary of training results."""
        if not self.avg_metrics:
            return "No metrics computed yet."

        lines = ["=" * 60, "TRAINING RESULTS SUMMARY", "=" * 60]
        for key, val in self.avg_metrics.items():
            if isinstance(val, float):
                lines.append(f"  {key:35s}: {val:10.4f}")
            else:
                lines.append(f"  {key:35s}: {val}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ─── Public API ───────────────────────────────────────────────────────────────


def train_model(
    X: pd.DataFrame,
    labels_df: pd.DataFrame,
    sample_weights: Optional[pd.Series] = None,
    config: Optional[Config] = None,
) -> TrainingResult:
    """
    Train a LightGBM model using purged walk-forward CV.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (output of ``build_features``).
    labels_df : pd.DataFrame
        Labels (output of ``apply_triple_barrier`` or ``get_labels_for_features``).
        Must have 'label' and 't1' columns.
    sample_weights : pd.Series, optional
        Sample weights. If None, computed automatically from labels_df.
    config : Config, optional
        Master configuration.

    Returns
    -------
    TrainingResult
        Contains fold models, metrics, predictions, and feature importances.
    """
    config = config or cfg
    tc = config.training

    # Align X and labels on their common index
    common_idx = X.index.intersection(labels_df.index)
    X_aligned = X.loc[common_idx]
    labels_aligned = labels_df.loc[common_idx]
    y = labels_aligned["label"].astype(int)

    # Map labels: {-1, 0, 1} → {0, 1, 2} for LightGBM
    label_map = {-1: 0, 0: 1, 1: 2}
    y_mapped = y.map(label_map)

    logger.info(
        "Training on %d samples with %d features",
        len(X_aligned),
        X_aligned.shape[1],
    )

    # Compute sample weights if not provided
    if sample_weights is None:
        logger.info("Computing sample weights from label overlaps ...")
        sample_weights = compute_sample_weights(labels_aligned)
    weights_aligned = sample_weights.reindex(common_idx, fill_value=1.0)

    # Set up purged walk-forward CV
    cv = PurgedWalkForwardCV(
        n_splits=tc.n_splits,
        embargo_pct=tc.embargo_pct,
    )

    result = TrainingResult(config=tc.lgbm_params.copy())
    all_importances = []

    # ── Cross-validation loop ─────────────────────────────────────────────
    for fold_idx, (train_idx, test_idx) in enumerate(
        cv.split(X_aligned, labels_aligned)
    ):
        logger.info("─── Fold %d/%d ───", fold_idx + 1, tc.n_splits)

        # Safety check: verify no leakage
        validate_no_leakage(X_aligned, labels_aligned, train_idx, test_idx)

        # Split data
        X_train = X_aligned.iloc[train_idx]
        X_test = X_aligned.iloc[test_idx]
        y_train = y_mapped.iloc[train_idx]
        y_test = y_mapped.iloc[test_idx]
        w_train = weights_aligned.iloc[train_idx]

        logger.info(
            "  Train: %d samples (%s → %s)",
            len(X_train),
            X_train.index[0],
            X_train.index[-1],
        )
        logger.info(
            "  Test:  %d samples (%s → %s)",
            len(X_test),
            X_test.index[0],
            X_test.index[-1],
        )

        # Train model
        model = LGBMClassifier(**tc.lgbm_params)
        model.fit(
            X_train,
            y_train,
            sample_weight=w_train.values,
            eval_set=[(X_test, y_test)],
            callbacks=[
                _log_eval_callback(period=100),
            ],
        )

        # Predict
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

        # Apply Meta-Labeling Trade Filter
        from src.training.trade_filter import apply_confidence_filter
        y_pred_filtered, filter_stats = apply_confidence_filter(
            y_pred, y_pred_proba, config.meta_label
        )

        # Map predictions back: {0, 1, 2} → {-1, 0, 1}
        inv_map = {0: -1, 1: 0, 2: 1}
        y_pred_original = pd.Series(y_pred_filtered, index=X_test.index).map(inv_map)
        y_test_original = y_test.map(inv_map)

        # Get corresponding labels for metric computation
        test_labels = labels_aligned.loc[X_test.index]

        # Compute metrics
        fold_metrics = compute_all_metrics(
            y_true=y_test_original,
            y_pred=y_pred_original,
            labels_df=test_labels,
            cost_config=config.costs,
        )
        fold_metrics["pct_filtered"] = filter_stats["pct_filtered"]

        logger.info("  Fold %d metrics (Filtered %.1f%% of trades):", fold_idx + 1, filter_stats["pct_filtered"])
        for k, v in fold_metrics.items():
            if isinstance(v, float):
                logger.info("    %s: %.4f", k, v)

        # Store results
        result.fold_models.append(model)
        result.fold_metrics.append(fold_metrics)
        result.fold_predictions.append(
            {
                "index": X_test.index,
                "y_true": y_test_original,
                "y_pred": y_pred_original,
                "y_pred_proba": y_pred_proba,
            }
        )

        # Feature importance
        imp = pd.Series(
            model.feature_importances_,
            index=X_aligned.columns,
            name=f"fold_{fold_idx}",
        )
        all_importances.append(imp)

    # ── Aggregate results ─────────────────────────────────────────────────
    result.feature_importances = pd.concat(all_importances, axis=1)
    result.feature_importances["mean"] = result.feature_importances.mean(axis=1)
    result.feature_importances = result.feature_importances.sort_values(
        "mean", ascending=False
    )

    # Average metrics across folds
    result.avg_metrics = _average_metrics(result.fold_metrics)

    # ── Train final model on all data ─────────────────────────────────────
    logger.info("Training final model on all %d samples ...", len(X_aligned))
    final_model = LGBMClassifier(**tc.lgbm_params)
    final_model.fit(X_aligned, y_mapped, sample_weight=weights_aligned.values)
    result.model = final_model

    logger.info("\n%s", result.summary())

    return result


def train_with_optuna(
    X: pd.DataFrame,
    labels_df: pd.DataFrame,
    sample_weights: Optional[pd.Series] = None,
    config: Optional[Config] = None,
) -> TrainingResult:
    """
    Train with Optuna hyperparameter optimization.

    Uses purged walk-forward CV as the evaluation strategy.
    Optimizes for expected return per trade (cost-aware).

    Parameters
    ----------
    X, labels_df, sample_weights, config
        Same as ``train_model``.

    Returns
    -------
    TrainingResult
        Best model and metrics from Optuna search.
    """
    import optuna

    config = config or cfg
    tc = config.training

    # Suppress Optuna info logs
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        """Optuna objective: maximize cost-adjusted expected return."""
        params = {
            "objective": "multiclass",
            "num_class": 3,
            "metric": "multi_logloss",
            "boosting_type": "gbdt",
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "random_state": tc.seed,
            "verbose": -1,
        }

        # Override config with trial params
        trial_config = copy.deepcopy(config)
        trial_config.training.lgbm_params = params
        trial_config.training.n_splits = tc.n_splits
        trial_config.training.embargo_pct = tc.embargo_pct

        try:
            trial_result = train_model(
                X, labels_df, sample_weights, config=trial_config
            )
            # Optimize for cost-adjusted expected return per trade
            score = trial_result.avg_metrics.get("expected_return_pips", -999)
            return score
        except Exception as e:
            logger.warning("Trial %d failed: %s", trial.number, e)
            return -999

    pruner = (
        optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
        if tc.optuna_pruning
        else optuna.pruners.NopPruner()
    )
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(
        objective,
        n_trials=tc.optuna_n_trials,
        timeout=tc.optuna_timeout,
    )

    logger.info("Optuna best trial: %d", study.best_trial.number)
    logger.info("  Best score: %.4f", study.best_value)
    logger.info("  Best params: %s", study.best_params)

    # Retrain with best params
    best_config = copy.deepcopy(config)
    best_params = {
        **tc.lgbm_params,
        **study.best_params,
    }
    best_config.training.lgbm_params = best_params

    best_result = train_model(X, labels_df, sample_weights, config=best_config)
    best_result.config = best_params

    return best_result


# ─── Internals ────────────────────────────────────────────────────────────────


def _average_metrics(fold_metrics: list[dict]) -> dict:
    """Average numeric metrics across all folds."""
    if not fold_metrics:
        return {}

    avg = {}
    all_keys = fold_metrics[0].keys()
    for key in all_keys:
        values = [m[key] for m in fold_metrics if isinstance(m.get(key), (int, float))]
        if values:
            avg[key] = np.mean(values)
            avg[f"{key}_std"] = np.std(values)

    return avg


def _log_eval_callback(period: int = 100):
    """LightGBM callback that logs evaluation results periodically."""

    def callback(env):
        if env.iteration % period == 0 or env.iteration == env.end_iteration - 1:
            if env.evaluation_result_list:
                msg_parts = []
                for item in env.evaluation_result_list:
                    # item format: (dataset_name, metric_name, value, is_higher_better)
                    msg_parts.append(f"{item[1]}={item[2]:.4f}")
                logger.debug(
                    "    Iter %d: %s", env.iteration, ", ".join(msg_parts)
                )

    callback.order = 10
    return callback
