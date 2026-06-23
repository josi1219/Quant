# ============================================================================
# EURUSD Trading Model — Colab Training Notebook
# ============================================================================
# This is a THIN WRAPPER. All logic lives in src/.
# Upload the project to Google Drive, mount it, then run this notebook.
#
# To use: File > Save a copy to Google Colab
# ============================================================================

# %% [markdown]
# # EURUSD Trading Model — Training
# 
# **This notebook runs on Google Colab.** It imports all logic from the
# `src/` package. No model logic is written here — only orchestration.
#
# ## Setup
# 1. Upload the entire `trading_model/` folder to Google Drive
# 2. Run the cells below

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
from src.data.loader import load_raw_data
from src.features.pipeline import build_features
from src.labels.triple_barrier import get_labels_for_features
from src.training.sample_weights import compute_sample_weights
from src.training.trainer import train_model
from src.evaluation.metrics import format_metrics
from src.export.onnx_export import export_model

# %% — Step 1: Load Data
print("=" * 60)
print("STEP 1: Loading raw data")
print("=" * 60)

df_raw = load_raw_data()
print(f"\nLoaded {len(df_raw)} candles")
print(df_raw.head())

# %% — Step 2: Build Features
print("\n" + "=" * 60)
print("STEP 2: Building features")
print("=" * 60)

X, meta = build_features(df_raw)
print(f"\nFeature matrix: {X.shape}")
print(f"Features: {meta['feature_names']}")

# %% — Step 3: Generate Labels
print("\n" + "=" * 60)
print("STEP 3: Generating triple-barrier labels")
print("=" * 60)

# We need the close prices aligned with the feature matrix
# For volume bars, use the volume-bar close; for time bars, use raw close
from src.features.pipeline import get_aligned_close

close_aligned = get_aligned_close(df_raw, X)
labels_df = get_labels_for_features(close_aligned, X.index)

print(f"\nLabels shape: {labels_df.shape}")
print(f"Label distribution:\n{labels_df['label'].value_counts().sort_index()}")

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
print("\nTop 15 features by importance:")
print(result.feature_importances["mean"].head(15))

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
print("TRAINING COMPLETE")
print("=" * 60)
print("\nKey metrics:")
print(f"  Accuracy:                 {result.avg_metrics.get('accuracy', 0):.4f}")
print(f"  Expected Return (pips):   {result.avg_metrics.get('expected_return_pips', 0):.4f}")
print(f"  After Costs (pips):       {result.avg_metrics.get('expected_return_after_costs_pips', 0):.4f}")
print(f"  Sharpe Ratio:             {result.avg_metrics.get('sharpe_ratio', 0):.4f}")
print(f"  Win Rate:                 {result.avg_metrics.get('win_rate', 0):.4f}")
