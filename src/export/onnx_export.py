"""
Export trained LightGBM model to ONNX format for deployment.

ONNX (Open Neural Network Exchange) is a language-agnostic model format.
Once exported, the model can be loaded in C++, C#, Java, or any runtime
that supports ONNX — including MetaTrader 5's MQL5.

Usage:
    from src.export.onnx_export import export_model
    export_model(trained_result, feature_names, save_dir="models/")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import ExportConfig, cfg

logger = logging.getLogger(__name__)


# ─── Public API ───────────────────────────────────────────────────────────────


def export_model(
    model,
    feature_names: list[str],
    metadata: dict,
    save_dir: Optional[Path] = None,
    model_name: str = "eurusd_model",
) -> dict:
    """
    Export a trained LightGBM model to ONNX and save associated metadata.

    Parameters
    ----------
    model : LGBMClassifier
        Trained LightGBM model.
    feature_names : list[str]
        Ordered list of feature names (must match training order).
    metadata : dict
        Pipeline metadata from ``build_features`` (frac_d, etc.).
    save_dir : Path, optional
        Directory to save files. Defaults to ``cfg.export.models_dir``.
    model_name : str
        Base name for the exported files.

    Returns
    -------
    dict
        Paths to all saved files.
    """
    save_dir = Path(save_dir or cfg.export.models_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    saved_files = {}

    # ── 1. Save as ONNX ──────────────────────────────────────────────────
    try:
        onnx_path = save_dir / f"{model_name}.onnx"
        _save_onnx(model, feature_names, onnx_path)
        saved_files["onnx"] = str(onnx_path)
        logger.info("ONNX model saved: %s", onnx_path)
    except ImportError as e:
        logger.warning("ONNX export skipped (missing dependency): %s", e)
    except Exception as e:
        logger.error("ONNX export failed: %s", e)

    # ── 2. Save as native LightGBM ────────────────────────────────────────
    lgbm_path = save_dir / f"{model_name}.lgbm"
    model.booster_.save_model(str(lgbm_path))
    saved_files["lgbm"] = str(lgbm_path)
    logger.info("LightGBM model saved: %s", lgbm_path)

    # ── 3. Save feature spec (for C++ engine) ─────────────────────────────
    spec_path = save_dir / f"{model_name}_feature_spec.json"
    feature_spec = {
        "model_name": model_name,
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "n_classes": 3,
        "class_labels": {0: "sell", 1: "hold", 2: "buy"},
        "pipeline_metadata": {
            k: v
            for k, v in metadata.items()
            if isinstance(v, (str, int, float, bool))
        },
    }
    with open(spec_path, "w") as f:
        json.dump(feature_spec, f, indent=2)
    saved_files["feature_spec"] = str(spec_path)
    logger.info("Feature spec saved: %s", spec_path)

    # ── 4. Save feature importances ───────────────────────────────────────
    imp_path = save_dir / f"{model_name}_importances.csv"
    importances = pd.Series(
        model.feature_importances_,
        index=feature_names,
        name="importance",
    ).sort_values(ascending=False)
    importances.to_csv(imp_path)
    saved_files["importances"] = str(imp_path)
    logger.info("Feature importances saved: %s", imp_path)

    logger.info("Export complete. Files: %s", list(saved_files.keys()))
    return saved_files


def load_model(
    model_path: Path,
) -> lgb.Booster:
    """
    Load a saved LightGBM model as a native Booster.

    Returns a ``lgb.Booster`` whose ``.predict()`` method returns class
    probabilities of shape (N, 3) for the three-class problem directly.
    Use ``booster.predict(X)`` instead of ``predict_proba``.

    Parameters
    ----------
    model_path : Path
        Path to the .lgbm file.

    Returns
    -------
    lgb.Booster
        Native LightGBM booster ready for prediction.
    """
    booster = lgb.Booster(model_file=str(model_path))
    logger.info("Model loaded from %s", model_path)
    return booster


# ─── Internals ────────────────────────────────────────────────────────────────


def _save_onnx(model, feature_names: list[str], path: Path) -> None:
    """Convert LightGBM model to ONNX format."""
    from onnxmltools import convert_lightgbm
    from onnxmltools.convert.common.data_types import FloatTensorType

    # Define input shape: (batch_size, n_features)
    initial_type = [
        ("features", FloatTensorType([None, len(feature_names)]))
    ]

    onnx_model = convert_lightgbm(
        model.booster_,
        initial_types=initial_type,
        target_opset=cfg.export.onnx_opset,
    )

    with open(path, "wb") as f:
        f.write(onnx_model.SerializeToString())


# ─── Convenience: run as script ───────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    print("Export module loaded. Use export_model() after training.")
