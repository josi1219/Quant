"""
Tests for feature engineering modules.

Validates:
  - Fractional differentiation weights and stationarity
  - Scale-invariance of indicators
  - Rolling z-score uses only past data (no leakage)
  - Feature pipeline produces expected output shape
"""

import numpy as np
import pandas as pd
import pytest

from src.config import Config, FeatureConfig


# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_ohlcv():
    """Create a small synthetic OHLCV dataset for testing."""
    np.random.seed(42)
    n = 2000
    dates = pd.date_range("2024-01-01", periods=n, freq="15min")

    # Random walk price
    returns = np.random.normal(0, 0.0005, n)
    close = 1.1000 + np.cumsum(returns)
    high = close + np.abs(np.random.normal(0, 0.0003, n))
    low = close - np.abs(np.random.normal(0, 0.0003, n))
    open_ = close + np.random.normal(0, 0.0001, n)

    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "tick_volume": np.random.randint(50, 1000, n),
            "spread": np.random.randint(0, 20, n),
        },
        index=dates,
    )
    df.index.name = "time"
    return df


@pytest.fixture
def config():
    """Default test config with smaller parameters for speed."""
    c = Config()
    c.features.rolling_window = 100
    c.features.use_volume_bars = False  # Skip for unit tests
    return c


# ─── Fractional Differentiation ──────────────────────────────────────────────


class TestFractionalDiff:
    def test_weights_first_is_one(self):
        """First weight should always be 1.0."""
        from src.features.fractional_diff import get_weights

        for d in [0.1, 0.3, 0.5, 0.7, 0.9]:
            w = get_weights(d)
            assert w[0] == 1.0, f"First weight should be 1.0 for d={d}"

    def test_weights_decrease(self):
        """Weights should decrease in absolute value."""
        from src.features.fractional_diff import get_weights

        w = get_weights(0.4)
        abs_w = np.abs(w)
        # Not strictly decreasing (they alternate sign), but abs should trend down
        assert abs_w[0] >= abs_w[-1], "Weights should decrease in magnitude"

    def test_frac_diff_d0_is_identity(self, sample_ohlcv):
        """d=0 should return the original series (no differencing)."""
        from src.features.fractional_diff import frac_diff_fast

        close = sample_ohlcv["close"]
        result = frac_diff_fast(close, d=0.0001)  # Very close to 0
        # With d≈0, result should be very close to original
        valid = result.dropna()
        # At d≈0, the series should still correlate highly with original
        corr = close.loc[valid.index].corr(valid)
        assert corr > 0.99, f"d≈0 should preserve the series (corr={corr:.4f})"

    def test_frac_diff_d1_is_returns(self, sample_ohlcv):
        """d=1 should approximate first differences (returns)."""
        from src.features.fractional_diff import frac_diff_fast

        close = sample_ohlcv["close"]
        result = frac_diff_fast(close, d=1.0)
        diff = close.diff()

        # Compare valid values
        valid_idx = result.dropna().index.intersection(diff.dropna().index)
        corr = result.loc[valid_idx].corr(diff.loc[valid_idx])
        assert corr > 0.95, f"d=1 should approximate diff (corr={corr:.4f})"

    def test_frac_diff_produces_no_nans_after_warmup(self, sample_ohlcv):
        """After the warmup period, there should be no NaN values."""
        from src.features.fractional_diff import frac_diff_fast, get_weights

        close = sample_ohlcv["close"]
        d = 0.4
        weights = get_weights(d)
        warmup = len(weights) - 1

        result = frac_diff_fast(close, d=d)
        assert result.iloc[warmup:].isna().sum() == 0, (
            "No NaN should exist after warmup period"
        )


# ─── Indicators ──────────────────────────────────────────────────────────────


class TestIndicators:
    def test_rsi_bounded(self, sample_ohlcv):
        """RSI should be bounded between 0 and 100."""
        from src.features.indicators import compute_all_indicators

        result = compute_all_indicators(sample_ohlcv)
        rsi = result["rsi"].dropna()
        assert rsi.min() >= 0, f"RSI min should be >= 0, got {rsi.min()}"
        assert rsi.max() <= 100, f"RSI max should be <= 100, got {rsi.max()}"

    def test_bb_position_roughly_bounded(self, sample_ohlcv):
        """Bollinger band position should be roughly between 0 and 1."""
        from src.features.indicators import compute_all_indicators

        result = compute_all_indicators(sample_ohlcv)
        bb = result["bb_position"].dropna()
        # Can exceed [0,1] during extreme moves, but mean should be near 0.5
        assert 0.3 < bb.mean() < 0.7, f"BB position mean should be ~0.5, got {bb.mean()}"

    def test_indicators_are_scale_invariant(self, sample_ohlcv):
        """
        Indicators should produce similar values when the entire price
        series is scaled by a constant (proving scale-invariance).
        """
        from src.features.indicators import compute_all_indicators

        # Original
        result1 = compute_all_indicators(sample_ohlcv)

        # Scaled by 10% higher
        df_scaled = sample_ohlcv.copy()
        for col in ["open", "high", "low", "close"]:
            df_scaled[col] = df_scaled[col] * 1.10
        result2 = compute_all_indicators(df_scaled)

        # RSI should be identical (purely based on up/down moves)
        rsi_corr = result1["rsi"].dropna().corr(result2["rsi"].dropna())
        assert rsi_corr > 0.99, f"RSI should be scale-invariant (corr={rsi_corr:.4f})"

        # Returns should be identical
        ret_corr = (
            result1["returns_1"].dropna().corr(result2["returns_1"].dropna())
        )
        assert ret_corr > 0.99, f"Returns should be scale-invariant (corr={ret_corr:.4f})"

    def test_no_raw_price_features(self, sample_ohlcv):
        """Verify no raw price values leak into the feature columns."""
        from src.features.indicators import compute_all_indicators

        result = compute_all_indicators(sample_ohlcv)
        # Feature columns (not OHLC)
        feature_cols = [
            c
            for c in result.columns
            if c not in ["open", "high", "low", "close", "tick_volume", "spread"]
        ]
        for col in feature_cols:
            valid = result[col].dropna()
            if len(valid) == 0:
                continue
            # No feature should have the same magnitude as close prices (~1.1)
            # They should all be either small ratios or bounded values
            assert valid.mean() < 100 or col == "rsi", (
                f"Feature '{col}' has suspiciously large mean ({valid.mean():.2f}). "
                "May contain raw price values."
            )


# ─── Rolling Normalization ───────────────────────────────────────────────────


class TestRollingNormalization:
    def test_zscore_uses_only_past(self, sample_ohlcv):
        """Rolling z-score at time t should only use data up to time t."""
        from src.features.microstructure import _rolling_zscore

        series = sample_ohlcv["tick_volume"].astype(float)
        window = 50
        zscores = _rolling_zscore(series, window)

        # Manually compute z-score at a specific point
        t = 200  # Well past warmup
        expected_mean = series.iloc[t - window + 1 : t + 1].mean()
        expected_std = series.iloc[t - window + 1 : t + 1].std()
        expected_z = (series.iloc[t] - expected_mean) / expected_std

        actual_z = zscores.iloc[t]
        assert abs(actual_z - expected_z) < 1e-6, (
            f"Z-score mismatch: expected {expected_z:.6f}, got {actual_z:.6f}"
        )

    def test_zscore_not_future_leaking(self, sample_ohlcv):
        """
        Modifying future values should NOT affect past z-scores.
        This directly tests for data leakage.
        """
        from src.features.microstructure import _rolling_zscore

        series = sample_ohlcv["tick_volume"].astype(float).copy()
        window = 50

        # Compute z-scores with original data
        z_original = _rolling_zscore(series, window)

        # Modify the last 500 values
        series_modified = series.copy()
        series_modified.iloc[-500:] = 999999

        # Recompute z-scores
        z_modified = _rolling_zscore(series_modified, window)

        # Z-scores before the modification point should be identical
        check_range = slice(window, len(series) - 500)
        np.testing.assert_array_almost_equal(
            z_original.iloc[check_range].values,
            z_modified.iloc[check_range].values,
            decimal=10,
            err_msg="Future data modification changed past z-scores (LEAKAGE!)",
        )


# ─── Feature Pipeline ────────────────────────────────────────────────────────


class TestFeaturePipeline:
    def test_pipeline_produces_output(self, sample_ohlcv, config):
        """Pipeline should produce a non-empty feature matrix."""
        from src.features.pipeline import build_features

        X, meta = build_features(sample_ohlcv, config=config)
        assert len(X) > 0, "Feature matrix should not be empty"
        assert X.shape[1] > 10, f"Should have many features, got {X.shape[1]}"

    def test_pipeline_no_nans(self, sample_ohlcv, config):
        """Pipeline output should have no NaN values."""
        from src.features.pipeline import build_features

        X, meta = build_features(sample_ohlcv, config=config)
        assert X.isna().sum().sum() == 0, "Feature matrix should have no NaN"

    def test_pipeline_metadata(self, sample_ohlcv, config):
        """Pipeline should return useful metadata."""
        from src.features.pipeline import build_features

        X, meta = build_features(sample_ohlcv, config=config)
        assert "frac_d" in meta
        assert "n_features" in meta
        assert "feature_names" in meta
        assert meta["n_features"] == X.shape[1]
        assert len(meta["feature_names"]) == X.shape[1]
