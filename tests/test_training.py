"""
Tests for the training pipeline.

Validates:
  - Purged CV produces no data leakage
  - Sample weights sum correctly
  - Metrics are computed without errors
"""

import numpy as np
import pandas as pd
import pytest

from src.config import Config, LabelConfig


# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def synthetic_data():
    """
    Create synthetic features, labels, and prices for testing.
    """
    np.random.seed(42)
    n = 2000
    dates = pd.date_range("2024-01-01", periods=n, freq="15min")

    # Synthetic features (5 features)
    X = pd.DataFrame(
        np.random.randn(n, 5),
        index=dates,
        columns=[f"feat_{i}" for i in range(5)],
    )

    # Synthetic labels
    labels = np.random.choice([-1, 0, 1], size=n, p=[0.3, 0.4, 0.3])
    holding = np.random.randint(1, 40, size=n)
    t1 = [dates[min(i + h, n - 1)] for i, h in enumerate(holding)]

    labels_df = pd.DataFrame(
        {
            "label": labels,
            "t1": t1,
            "ret": np.random.normal(0, 0.001, n),
            "ret_pips": np.random.normal(0, 10, n),
            "barrier_type": np.random.choice(
                ["upper", "lower", "time"], size=n
            ),
            "holding_period": holding,
        },
        index=dates,
    )

    return X, labels_df


# ─── Purged CV ────────────────────────────────────────────────────────────────


class TestPurgedCV:
    def test_no_overlap_between_train_test(self, synthetic_data):
        """Train and test indices should never overlap."""
        from src.training.purged_cv import PurgedWalkForwardCV

        X, labels_df = synthetic_data
        cv = PurgedWalkForwardCV(n_splits=3, embargo_pct=0.01)

        for train_idx, test_idx in cv.split(X, labels_df):
            overlap = set(train_idx) & set(test_idx)
            assert len(overlap) == 0, (
                f"Train and test overlap: {len(overlap)} indices"
            )

    def test_train_before_test(self, synthetic_data):
        """All training indices should be before all test indices."""
        from src.training.purged_cv import PurgedWalkForwardCV

        X, labels_df = synthetic_data
        cv = PurgedWalkForwardCV(n_splits=3, embargo_pct=0.01)

        for train_idx, test_idx in cv.split(X, labels_df):
            assert train_idx.max() < test_idx.min(), (
                f"Training max ({train_idx.max()}) should be < "
                f"test min ({test_idx.min()})"
            )

    def test_purging_removes_leaking_samples(self, synthetic_data):
        """Purging should remove training samples whose t1 overlaps test."""
        from src.training.purged_cv import PurgedWalkForwardCV, validate_no_leakage

        X, labels_df = synthetic_data
        cv = PurgedWalkForwardCV(n_splits=3, embargo_pct=0.01)

        for train_idx, test_idx in cv.split(X, labels_df):
            # This should NOT raise — purging should have worked
            assert validate_no_leakage(X, labels_df, train_idx, test_idx)

    def test_embargo_creates_gap(self, synthetic_data):
        """Embargo should create a gap between train and test."""
        from src.training.purged_cv import PurgedWalkForwardCV

        X, labels_df = synthetic_data
        cv = PurgedWalkForwardCV(n_splits=3, embargo_pct=0.05)

        for train_idx, test_idx in cv.split(X):
            gap = test_idx.min() - train_idx.max()
            assert gap > 1, f"Embargo should create gap > 1, got {gap}"

    def test_correct_number_of_folds(self, synthetic_data):
        """Should produce the specified number of folds."""
        from src.training.purged_cv import PurgedWalkForwardCV

        X, labels_df = synthetic_data
        cv = PurgedWalkForwardCV(n_splits=5, embargo_pct=0.01)
        folds = list(cv.split(X, labels_df))
        assert len(folds) == 5, f"Expected 5 folds, got {len(folds)}"


# ─── Sample Weights ──────────────────────────────────────────────────────────


class TestSampleWeights:
    def test_weights_sum_to_n(self, synthetic_data):
        """Weights should sum to the number of samples."""
        from src.training.sample_weights import compute_sample_weights

        _, labels_df = synthetic_data
        weights = compute_sample_weights(labels_df)
        valid_weights = weights[labels_df["label"].notna()]
        assert abs(valid_weights.sum() - len(valid_weights)) < 1.0, (
            f"Weights should sum to ~{len(valid_weights)}, "
            f"got {valid_weights.sum():.1f}"
        )

    def test_weights_are_positive(self, synthetic_data):
        """All weights should be positive."""
        from src.training.sample_weights import compute_sample_weights

        _, labels_df = synthetic_data
        weights = compute_sample_weights(labels_df)
        assert (weights > 0).all(), "All weights should be positive"

    def test_non_overlapping_events_get_weight_one(self):
        """Events that don't overlap should get weight ≈ 1.0."""
        from src.training.sample_weights import compute_sample_weights

        dates = pd.date_range("2024-01-01", periods=100, freq="1h")
        labels_df = pd.DataFrame(
            {
                "label": [1] * 100,
                "t1": dates + pd.Timedelta(minutes=30),  # Each event is 30 min
                "ret": [0.001] * 100,
            },
            index=dates,
        )
        weights = compute_sample_weights(labels_df)
        # Events don't overlap (1h apart, 30min duration), so weights ≈ 1.0
        assert weights.std() < 0.2, (
            f"Non-overlapping events should have similar weights (std={weights.std():.3f})"
        )


# ─── Metrics ─────────────────────────────────────────────────────────────────


class TestMetrics:
    def test_metrics_compute_without_error(self):
        """Metrics should compute without errors on valid inputs."""
        from src.evaluation.metrics import compute_all_metrics
        from src.config import CostConfig

        y_true = pd.Series([1, -1, 0, 1, -1, 1, 0, -1])
        y_pred = pd.Series([1, -1, 1, 1, 0, -1, 0, -1])

        metrics = compute_all_metrics(y_true, y_pred)
        assert "accuracy" in metrics
        assert "f1_weighted" in metrics
        assert 0 <= metrics["accuracy"] <= 1

    def test_perfect_predictions(self):
        """Perfect predictions should give accuracy = 1.0."""
        from src.evaluation.metrics import compute_all_metrics

        y = pd.Series([1, -1, 0, 1, -1])
        metrics = compute_all_metrics(y, y)
        assert metrics["accuracy"] == 1.0

    def test_trading_metrics_with_labels(self):
        """Trading metrics should work when labels_df is provided."""
        from src.evaluation.metrics import compute_all_metrics
        from src.config import CostConfig

        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="15min")
        y_true = pd.Series(
            np.random.choice([-1, 0, 1], n), index=dates
        )
        y_pred = pd.Series(
            np.random.choice([-1, 0, 1], n), index=dates
        )

        labels_df = pd.DataFrame(
            {
                "ret_pips": np.random.normal(0, 5, n),
                "holding_period": np.random.randint(1, 20, n),
            },
            index=dates,
        )

        metrics = compute_all_metrics(
            y_true, y_pred, labels_df, CostConfig()
        )
        assert "expected_return_pips" in metrics
        assert "sharpe_ratio" in metrics
        assert "n_trades" in metrics
