"""Evaluate a saved model on the eval split.

This script is intentionally simple: load model, load eval data,
predict, and compute MAE/RMSE/R2.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DEFAULT_EVAL = Path("data/processed/eval_final.csv")
DEFAULT_MODEL = Path("models/xgb_model.pkl")


def _maybe_sample(
    df: pd.DataFrame, sample_frac: Optional[float], random_state: int
) -> pd.DataFrame:
    """Optionally subsample eval set for quick checks."""
    if sample_frac is None:
        return df
    sample_frac = float(sample_frac)
    if sample_frac <= 0 or sample_frac >= 1:
        return df
    return df.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)


def evaluate_model(
    model_path: Path | str = DEFAULT_MODEL,
    eval_path: Path | str = DEFAULT_EVAL,
    sample_frac: Optional[float] = None,
    random_state: int = 42,
) -> Dict[str, float]:
    # 1) Load eval dataset created by feature pipeline.
    eval_df = pd.read_csv(eval_path)
    eval_df = _maybe_sample(eval_df, sample_frac, random_state)

    target = "price"
    if target not in eval_df.columns:
        raise ValueError("Missing target column: 'price'")
    X_eval, y_eval = eval_df.drop(columns=[target]), eval_df[target]

    # 2) Load trained model and run predictions.
    model = load(model_path)
    y_pred = model.predict(X_eval)

    # 3) Compute standard regression metrics.
    mae = float(mean_absolute_error(y_eval, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_eval, y_pred)))
    r2 = float(r2_score(y_eval, y_pred))
    metrics = {"mae": mae, "rmse": rmse, "r2": r2}

    print("📊 Evaluation:")
    print(f"   MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.4f}")
    return metrics


if __name__ == "__main__":
    evaluate_model()
