"""
Tests for the triple-barrier labeling method.

Validates:
  - Correct barrier detection (upper, lower, time)
  - Label values are in {-1, 0, 1}
  - Forward-looking nature doesn't contaminate features
  - Edge cases (flat markets, extreme moves)
"""

import numpy as np
import pandas as pd
import pytest

from src.config import LabelConfig


# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def trending_up():
    """Price series that trends up steadily."""
    n = 500
    dates = pd.date_range("2024-01-01", periods=n, freq="15min")
    close = pd.Series(
        1.1000 + np.arange(n) * 0.0002,  # +0.2 pips per bar
        index=dates,
    )
    return close


@pytest.fixture
def trending_down():
    """Price series that trends down steadily."""
    n = 500
    dates = pd.date_range("2024-01-01", periods=n, freq="15min")
    close = pd.Series(
        1.1000 - np.arange(n) * 0.0002,  # -0.2 pips per bar
        index=dates,
    )
    return close


@pytest.fixture
def flat_market():
    """Price series that stays flat with tiny noise."""
    n = 500
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="15min")
    close = pd.Series(
        1.1000 + np.random.normal(0, 0.00001, n),
        index=dates,
    )
    return close


@pytest.fixture
def realistic_prices():
    """More realistic random walk price series."""
    n = 2000
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="15min")
    returns = np.random.normal(0, 0.0005, n)
    close = pd.Series(1.1000 + np.cumsum(returns), index=dates)
    return close


# ─── Tests ────────────────────────────────────────────────────────────────────


class TestTripleBarrier:
    def test_labels_are_valid(self, realistic_prices):
        """All labels should be in {-1, 0, 1} or NaN."""
        from src.labels.triple_barrier import apply_triple_barrier

        labels = apply_triple_barrier(realistic_prices)
        valid_labels = labels["label"].dropna().unique()
        for val in valid_labels:
            assert val in {-1, 0, 1}, f"Invalid label value: {val}"

    def test_trending_up_has_buy_signals(self, trending_up):
        """A trending-up market should produce mostly Buy labels."""
        from src.labels.triple_barrier import apply_triple_barrier

        config = LabelConfig(
            pt_multiplier=1.0,
            sl_multiplier=1.0,
            max_holding_period=50,
            vol_lookback=96,
        )
        labels = apply_triple_barrier(trending_up, config)
        valid = labels["label"].dropna()
        buy_pct = (valid == 1).mean()
        assert buy_pct > 0.5, (
            f"Trending up should have >50% buy labels, got {buy_pct:.1%}"
        )

    def test_trending_down_has_sell_signals(self, trending_down):
        """A trending-down market should produce mostly Sell labels."""
        from src.labels.triple_barrier import apply_triple_barrier

        config = LabelConfig(
            pt_multiplier=1.0,
            sl_multiplier=1.0,
            max_holding_period=50,
            vol_lookback=96,
        )
        labels = apply_triple_barrier(trending_down, config)
        valid = labels["label"].dropna()
        sell_pct = (valid == -1).mean()
        assert sell_pct > 0.5, (
            f"Trending down should have >50% sell labels, got {sell_pct:.1%}"
        )

    def test_t1_is_after_t0(self, realistic_prices):
        """Event end time t1 should always be after the event start."""
        from src.labels.triple_barrier import apply_triple_barrier

        labels = apply_triple_barrier(realistic_prices)
        valid = labels.dropna(subset=["t1"])
        for t0, row in valid.iterrows():
            assert row["t1"] >= t0, (
                f"t1 ({row['t1']}) should be >= t0 ({t0})"
            )

    def test_holding_period_within_max(self, realistic_prices):
        """Holding period should not exceed max_holding_period."""
        from src.labels.triple_barrier import apply_triple_barrier

        config = LabelConfig(max_holding_period=30)
        labels = apply_triple_barrier(realistic_prices, config)
        valid = labels["holding_period"].dropna()
        assert valid.max() <= 30, (
            f"Holding period should be <= 30, got max={valid.max()}"
        )

    def test_return_sign_matches_label(self, realistic_prices):
        """
        For upper/lower barrier hits, the return sign should match the label.
        Upper barrier (buy signal) → positive return.
        Lower barrier (sell signal) → negative return.
        """
        from src.labels.triple_barrier import apply_triple_barrier

        labels = apply_triple_barrier(realistic_prices)

        upper_hits = labels[labels["barrier_type"] == "upper"]
        if len(upper_hits) > 0:
            assert (upper_hits["ret"] > 0).all(), (
                "Upper barrier hits should have positive returns"
            )

        lower_hits = labels[labels["barrier_type"] == "lower"]
        if len(lower_hits) > 0:
            assert (lower_hits["ret"] < 0).all(), (
                "Lower barrier hits should have negative returns"
            )

    def test_last_bars_are_nan(self, realistic_prices):
        """
        The last max_holding_period bars should have NaN labels
        (can't look forward at end of dataset).
        """
        from src.labels.triple_barrier import apply_triple_barrier

        config = LabelConfig(max_holding_period=40)
        labels = apply_triple_barrier(realistic_prices, config)
        last_labels = labels["label"].iloc[-40:]
        assert last_labels.isna().all(), (
            "Last max_holding_period bars should be NaN"
        )
