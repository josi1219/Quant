# EURUSD Production Trading Model

A production-grade quantitative trading model for EURUSD, built with quant-level feature engineering, leak-proof training, and cost-aware evaluation.

## Architecture

```
src/              ← All logic lives here (proper Python package)
  config.py       ← Central hyperparameters & constants
  data/           ← Data loading & bar construction
  features/       ← Feature engineering (frac diff, indicators, microstructure)
  labels/         ← Triple-barrier labeling
  training/       ← Purged walk-forward CV, model training
  evaluation/     ← Cost-aware metrics
  export/         ← ONNX model export

notebooks/        ← Thin Colab wrappers (import from src/)
tests/            ← Unit tests
models/           ← Saved trained models
```

## Quick Start

```bash
# Install in development mode
pip install -e ".[dev,export,tune]"

# Run tests
python -m pytest tests/ -v
```

## Colab Usage

```python
from google.colab import drive
drive.mount('/content/drive')

import sys
sys.path.insert(0, '/content/drive/MyDrive/trading_model')

from src.features.pipeline import build_features
from src.training.trainer import train_model
```

## Design Principles

1. **No notebook logic** — Notebooks only call `src/` functions.
2. **No data leakage** — Rolling normalization, purged CV, proper time-series splits.
3. **Scale-invariant features** — No raw prices ever enter the model.
4. **Cost-aware** — Every metric accounts for spread, commission, and slippage.
5. **Reproducible** — All hyperparameters in `config.py`, all seeds fixed.
