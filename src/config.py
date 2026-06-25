"""
Central configuration for the EURUSD Trading Model.

Multi-Timeframe Architecture: H1 (context) + M5 (execution).
All hyperparameters, paths, and constants live here.
Never hardcode magic numbers in other modules — import from config.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Project root (resolved at import time, works on any machine)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class DataConfig:
    """Paths and raw-data settings for dual-timeframe pipeline."""

    # Primary execution timeframe (M5)
    raw_csv_m5: Path = PROJECT_ROOT / "data" / "raw" / "EURUSD_M5.csv"
    # Context timeframe (H1)
    raw_csv_h1: Path = PROJECT_ROOT / "data" / "raw" / "EURUSD_H1.csv"
    # Legacy single-TF path (backward compatibility)
    raw_csv: Path = PROJECT_ROOT / "data" / "raw" / "EURUSD_M15.csv"

    symbol: str = "EURUSD"
    execution_tf: str = "M5"
    context_tf: str = "H1"
    bars_per_day_m5: int = 288   # 24h × 12 bars/hour for 5-min candles
    bars_per_day_h1: int = 24    # 24h × 1 bar/hour
    # Legacy
    timeframe: str = "M15"
    bars_per_day: int = 96       # For backward compat with old pipeline

    # Minimum quality thresholds for raw candles
    min_tick_volume: int = 5     # Lower for M5 (less volume per bar)
    max_spread_pips: float = 30.0  # Drop candles with abnormal spreads


@dataclass
class CostConfig:
    """Transaction cost assumptions (in pips) — realistic retail ECN."""

    pip_value: float = 0.0001    # 1 pip = 0.0001 for EURUSD
    spread: float = 0.8          # Avg spread on a decent ECN broker
    commission_per_side: float = 0.35  # Common ECN commission per side
    slippage: float = 0.2        # M5 execution = faster fills = less slippage

    @property
    def total_cost_pips(self) -> float:
        """Total round-trip cost per trade in pips."""
        return self.spread + (2 * self.commission_per_side) + self.slippage


@dataclass
class FeatureConfig:
    """Feature engineering hyperparameters for dual-timeframe pipeline."""

    # Fractional differentiation
    frac_diff_d: float = 0.4     # Starting guess; will be tuned via ADF
    frac_diff_threshold: float = 1e-4  # Drop weights below this

    # Volume bars
    use_volume_bars: bool = True
    volume_bar_size: int = 1500  # Reduced from 5000 (M5 bars have ~3x less vol)

    # Rolling normalization window (in M5 bars)
    rolling_window: int = 1440   # ~5 trading days of M5 bars

    # --- M5 Indicator periods ---
    rsi_period: int = 14
    atr_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0
    ema_fast: int = 12
    ema_slow: int = 26
    macd_signal: int = 9

    # M5 multi-horizon return lookbacks (in M5 bars)
    # 1=5min, 3=15min, 6=30min, 12=1hr, 48=4hr
    return_horizons: List[int] = field(
        default_factory=lambda: [1, 3, 6, 12, 48]
    )

    # --- H1 Indicator periods ---
    h1_rsi_period: int = 14
    h1_atr_period: int = 14
    h1_bb_period: int = 20
    h1_bb_std: float = 2.0
    h1_ema_fast: int = 12
    h1_ema_slow: int = 26
    h1_macd_signal: int = 9
    # H1 return lookbacks (in H1 bars): 1=1hr, 4=4hr, 24=1day
    h1_return_horizons: List[int] = field(
        default_factory=lambda: [1, 4, 24]
    )

    # --- Regime detection ---
    vol_regime_fast: int = 48    # 4 hours of M5 bars (short-term vol)
    vol_regime_slow: int = 576   # 2 days of M5 bars (long-term vol)
    hurst_lookback: int = 288    # 1 day of M5 bars for Hurst exponent
    momentum_quality_lookback: int = 12  # 1 hour of M5 bars

    # --- Session boundaries (UTC hours) ---
    session_asian_start: int = 0
    session_asian_end: int = 7
    session_london_start: int = 7
    session_london_end: int = 16
    session_ny_start: int = 13
    session_ny_end: int = 22

    # --- Microstructure lookbacks (M5 bars) ---
    micro_vol_lookback: int = 288   # 1 day for volume z-score
    micro_vpin_lookback: int = 48   # 4 hours for VPIN


@dataclass
class LabelConfig:
    """Triple-barrier labeling parameters for M5 intraday swing trades."""

    # Barrier widths as multiples of daily volatility — ASYMMETRIC
    pt_multiplier: float = 2.4   # Take-profit (Optimized)
    sl_multiplier: float = 1.7   # Stop-loss (Optimized)

    # Holding period in M5 bars
    max_holding_period: int = 48   # 48 M5 bars = 4 hours
    min_holding_period: int = 24   # 24 M5 bars = 2 hours (NEW)
    min_return_pips: float = 3.0   # Trades < 3 pips are "Hold" (was 0.0)

    # Daily volatility estimation (in M5 bars)
    vol_lookback: int = 288        # 1 day of M5 bars (was 96 for M15)

    # Dynamic barriers — scale by regime volatility
    dynamic_barriers: bool = True
    vol_regime_fast: int = 48      # 4 hours of M5 for short-term vol
    vol_regime_slow: int = 576     # 2 days of M5 for long-term vol

    # Class balance warning threshold
    max_class_pct: float = 0.50    # Warn if any class exceeds 50%


@dataclass
class TrainingConfig:
    """Training and cross-validation settings."""

    # Purged walk-forward CV
    n_splits: int = 5
    embargo_pct: float = 0.01  # Fraction of test size used as embargo

    # LightGBM defaults (will be tuned by Optuna)
    lgbm_params: dict = field(
        default_factory=lambda: {
            "objective": "multiclass",
            "num_class": 3,
            "metric": "multi_logloss",
            "boosting_type": "gbdt",
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 6,
            "num_leaves": 31,
            "min_child_samples": 50,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "verbose": -1,
        }
    )

    # Optuna
    optuna_n_trials: int = 80    # Increased from 50
    optuna_timeout: int = 7200   # 2 hours
    optuna_pruning: bool = True  # Use MedianPruner

    # Concurrent Trade Filtering
    max_open_trades: int = 3     # Maximum concurrent trades allowed
    cooldown_bars: int = 6       # Wait 6 M5 bars (30m) between trades

    # Random seed for reproducibility
    seed: int = 42


@dataclass
class MetaLabelConfig:
    """Meta-labeling and trade filtering settings."""

    enabled: bool = True
    confidence_threshold: float = 0.55  # Min probability to take a trade

    # Secondary meta-model parameters
    meta_model_params: dict = field(
        default_factory=lambda: {
            "objective": "binary",
            "metric": "binary_logloss",
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.05,
            "min_child_samples": 30,
            "random_state": 42,
            "verbose": -1,
        }
    )

    # Position sizing tiers (confidence → risk fraction)
    full_position_threshold: float = 0.75    # Risk 1.0% of account
    half_position_threshold: float = 0.60    # Risk 0.5% of account
    quarter_position_threshold: float = 0.45  # Risk 0.25% of account


@dataclass
class LiveConfig:
    """Production deployment safeguards."""

    max_spread_pips: float = 1.5       # Reject trades above this spread
    max_slippage_pips: float = 1.0     # Cancel if slippage exceeds this
    trading_sessions: List[str] = field(
        default_factory=lambda: ["london", "new_york"]
    )
    news_blackout_minutes_before: int = 30
    news_blackout_minutes_after: int = 15
    paper_trade_weeks: int = 4         # Min paper trading before live


@dataclass
class ExportConfig:
    """Model export settings."""

    models_dir: Path = PROJECT_ROOT / "models"
    onnx_opset: int = 13


@dataclass
class Config:
    """Master configuration — aggregates all sub-configs."""

    data: DataConfig = field(default_factory=DataConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    meta_label: MetaLabelConfig = field(default_factory=MetaLabelConfig)
    live: LiveConfig = field(default_factory=LiveConfig)
    export: ExportConfig = field(default_factory=ExportConfig)


# ---------------------------------------------------------------------------
# Default singleton — import this in other modules
# ---------------------------------------------------------------------------
cfg = Config()
