# ============================================================================
# EURUSD Trading Model — Multi-Timeframe Colab Training Notebook
# ============================================================================
# H1 (context) + M5 (execution) architecture.
# Upload the project + M5/H1 CSV files to Google Drive, then run this.
# ============================================================================

# %% [markdown]
# # EURUSD Multi-Timeframe Trading Model — Training
#
# **Architecture:** H1 provides trend/regime context, M5 provides execution signals.
# **Holding Period:** 2-8 hours (intraday swing)
# **Barriers:** Symmetric (1.5× vol TP, 1.5× vol SL), dynamic regime scaling
#
# ## Setup
# 1. Upload `trading_model/` folder to Google Drive
# 2. Place `EURUSD_M5.csv` and `EURUSD_H1.csv` in `data/raw/`
# 3. Run the cells below

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

# For local testing, just use the project root
import sys
from pathlib import Path

# Auto-detect: Colab vs Local
try:
    from google.colab import drive  # noqa: F401
    PROJECT_ROOT = Path("/content/drive/MyDrive/trading_model")
except ImportError:
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
from src.labels.triple_barrier import get_labels_for_features
from src.training.sample_weights import compute_sample_weights
from src.training.trainer import train_model
from src.evaluation.metrics import format_metrics
from src.export.onnx_export import export_model

# %% — Step 1: Load Dual-Timeframe Data
print("=" * 60)
print("STEP 1: Loading M5 + H1 data")
print("=" * 60)

df_m5, df_h1 = load_dual_timeframe()
print(f"\nM5: {len(df_m5)} candles ({df_m5.index.min()} → {df_m5.index.max()})")
print(f"H1: {len(df_h1)} candles ({df_h1.index.min()} → {df_h1.index.max()})")

# %% — Step 2: Build MTF Features
print("\n" + "=" * 60)
print("STEP 2: Building multi-timeframe features (H1 context + M5 execution)")
print("=" * 60)

X, meta = build_features_mtf(df_m5, df_h1)
print(f"\nFeature matrix: {X.shape}")
print(f"Pipeline: {meta['pipeline']}")
print(f"Frac diff d: {meta['frac_d']}")
print(f"\nFeatures ({meta['n_features']}):")
for i, name in enumerate(meta["feature_names"]):
    print(f"  {i+1:3d}. {name}")

# %% — Step 3: Generate Labels
print("\n" + "=" * 60)
print("STEP 3: Generating triple-barrier labels (2-8hr hold, symmetric)")
print("=" * 60)

close_aligned = get_aligned_close(df_m5, X)
labels_df = get_labels_for_features(close_aligned, X.index)

print(f"\nLabels shape: {labels_df.shape}")
print(f"Label distribution:\n{labels_df['label'].value_counts().sort_index()}")
print(f"\nBarrier type distribution:")
print(labels_df["barrier_type"].value_counts())
print(f"\nAvg holding period: {labels_df['holding_period'].mean():.1f} M5 bars "
      f"({labels_df['holding_period'].mean() * 5 / 60:.1f} hours)")

# %% — Step 4: Compute Sample Weights
print("\n" + "=" * 60)
print("STEP 4: Computing sample weights")
print("=" * 60)

weights = compute_sample_weights(labels_df)
print(f"\nWeight stats:\n{weights.describe()}")

# %% — Step 5: Train Model
print("\n" + "=" * 60)
print("STEP 5: Training with Purged Walk-Forward CV")
print("=" * 60)

result = train_model(X, labels_df, weights)

# Print results
print("\n" + format_metrics(result.avg_metrics))

# Feature importances
print("\nTop 20 features by importance:")
print(result.feature_importances["mean"].head(20))

# %% — Step 6: Export Model
print("\n" + "=" * 60)
print("STEP 6: Exporting model")
print("=" * 60)

saved = export_model(
    model=result.model,
    feature_names=meta["feature_names"],
    metadata=meta,
)
print(f"\nSaved files: {saved}")

# %% — Done!
print("\n" + "=" * 60)
print("MTF TRAINING COMPLETE")
print("=" * 60)
print("\nKey metrics:")
print(f"  Accuracy:                 {result.avg_metrics.get('accuracy', 0):.4f}")
print(f"  Expected Return (pips):   {result.avg_metrics.get('expected_return_pips', 0):.4f}")
print(f"  After Costs (pips):       {result.avg_metrics.get('expected_return_after_costs_pips', 0):.4f}")
print(f"  Profit Factor:            {result.avg_metrics.get('profit_factor', 0):.4f}")
print(f"  Sharpe Ratio:             {result.avg_metrics.get('sharpe_ratio', 0):.4f}")
print(f"  Win Rate:                 {result.avg_metrics.get('win_rate', 0):.4f}")
print(f"  Avg Holding Period:       {result.avg_metrics.get('avg_holding_period', 0):.1f} bars "
      f"({result.avg_metrics.get('avg_holding_period', 0) * 5 / 60:.1f} hours)")
