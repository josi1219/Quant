# ============================================================================
# EURUSD Trading Model — Phase 4: Optuna Tuning
# ============================================================================
# Joint optimization of Triple-Barrier logic and LightGBM parameters.
# Finds the exact TP/SL multiplier and duration to maximize Sharpe.
# ============================================================================

# %% [markdown]
# # EURUSD Multi-Timeframe Trading Model — Optuna Tuning
#
# **Goal:** Maximize net profitability and Sharpe Ratio.
# **Method:** Sweep through different barrier configurations (pt_multiplier, sl_multiplier, holding_period) 
# and LightGBM hyperparameters simultaneously.

# %% — Mount Drive & Setup Path
"""
# Uncomment these lines when running on Google Colab:

from google.colab import drive
drive.mount('/content/drive')

import sys
sys.path.insert(0, '/content/drive/MyDrive/trading_model')

# Install dependencies
!pip install -r /content/drive/MyDrive/trading_model/requirements.txt -q
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# %% — Imports
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)

from src.config import cfg
from src.data.loader import load_dual_timeframe
from src.features.pipeline import build_features_mtf, get_aligned_close
from src.training.tuner import run_optuna_tuning
from src.evaluation.metrics import format_metrics

# %% — Step 1: Load Data & Build Features
print("=" * 60)
print("Loading data and building features ...")
print("=" * 60)

df_m5, df_h1 = load_dual_timeframe()
X, meta = build_features_mtf(df_m5, df_h1)
close_aligned = get_aligned_close(df_m5, X)

print(f"Feature matrix ready: {X.shape}")

# %% — Step 2: Run Optuna Tuning
print("\n" + "=" * 60)
print("Starting Joint Optuna Tuning (Barriers + Model)")
print("WARNING: This will take several hours on Colab.")
print("=" * 60)

# We use fewer trials for the notebook demo, but for production, use 100+
best_config, best_result = run_optuna_tuning(X, close_aligned, n_trials=50)

# %% — Step 3: View Results
print("\n" + "=" * 60)
print("OPTIMAL SETTINGS FOUND")
print("=" * 60)
print(f"Take Profit:        {best_config.labels.pt_multiplier:.2f}x Daily Volatility")
print(f"Stop Loss:          {best_config.labels.sl_multiplier:.2f}x Daily Volatility")
print(f"Max Holding Period: {best_config.labels.max_holding_period} M5 bars")

print("\n" + format_metrics(best_result.avg_metrics))

print("\nTo use these settings permanently, update your src/config.py file.")
