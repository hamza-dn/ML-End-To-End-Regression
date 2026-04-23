# tests/test_inference.py
from pathlib import Path

import pandas as pd
import pytest
from src.inference_pipeline.inference import predict

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def sample_df():
    """Load a small sample from eval_clean.csv for inference testing.

    We use `eval_clean` (not `eval_final`) because inference should receive
    raw-like data and apply encoders itself.
    """
    sample_path = ROOT / "data/processed/eval_clean.csv"
    df = pd.read_csv(sample_path).sample(5, random_state=42).reset_index(drop=True)
    return df


def test_inference_runs_and_returns_predictions(sample_df):
    """Ensure inference pipeline runs and returns predicted_price column."""
    # Use smoke model generated during training checks.
    model_path = ROOT / "models/xgb_model_smoke.pkl"
    preds_df = predict(sample_df, model_path=model_path)

    # Check output is not empty
    assert not preds_df.empty

    # Must include prediction column
    assert "predicted_price" in preds_df.columns

    # Predictions should be numeric
    assert pd.api.types.is_numeric_dtype(preds_df["predicted_price"])

    print("✅ Inference pipeline test passed. Predictions:")
    print(preds_df[["predicted_price"]].head())
