"""Inference pipeline for Housing Regression MLE.

Flow:
1) Receive raw-like input rows.
2) Apply the same preprocessing/feature logic as training.
3) Reuse saved encoders from training (no re-fit in inference).
4) Align columns to training schema and predict.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from joblib import load

# Import preprocessing + feature engineering helpers
from src.feature_pipeline.preprocess import (
    clean_and_merge,
    drop_duplicates,
    remove_outliers,
)
from src.feature_pipeline.feature_engineering import (
    add_date_features,
    drop_unused_columns,
)

# ----------------------------
# Default paths
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODEL = PROJECT_ROOT / "models" / "xgb_best_model.pkl"
DEFAULT_FREQ_ENCODER = PROJECT_ROOT / "models" / "freq_encoder.pkl"
DEFAULT_TARGET_ENCODER = PROJECT_ROOT / "models" / "target_encoder.pkl"
TRAIN_FE_PATH = PROJECT_ROOT / "data" / "processed" / "train_final.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "predictions.csv"


def _resolve_model_path(model_path: Path | str) -> Path:
    """Resolve a usable model path with a safe fallback for local smoke runs."""
    requested = Path(model_path)
    if requested.exists():
        return requested

    # Fallback order helps beginners run inference even before final model naming is fixed.
    fallback_candidates = [
        PROJECT_ROOT / "models" / "xgb_model.pkl",
        PROJECT_ROOT / "models" / "xgb_model_smoke.pkl",
        PROJECT_ROOT / "models" / "xgb_best_model_smoke.pkl",
    ]
    for candidate in fallback_candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"No model file found. Checked: {requested} and fallback candidates {fallback_candidates}"
    )


# Load training feature columns (strict schema from training dataset)
if TRAIN_FE_PATH.exists():
    _train_cols = pd.read_csv(TRAIN_FE_PATH, nrows=1)
    TRAIN_FEATURE_COLUMNS = [
        c for c in _train_cols.columns if c != "price"
    ]  # excluding target
else:
    TRAIN_FEATURE_COLUMNS = None


# ----------------------------
# Core inference function
# ----------------------------
def predict(
    input_df: pd.DataFrame,
    model_path: Path | str = DEFAULT_MODEL,
    freq_encoder_path: Path | str = DEFAULT_FREQ_ENCODER,
    target_encoder_path: Path | str = DEFAULT_TARGET_ENCODER,
) -> pd.DataFrame:
    """Run inference on an input dataframe and return predictions.

    The function accepts raw-like rows (before encoding), applies the same
    transformation logic as training, and returns a dataframe with
    `predicted_price`.
    """
    # Step 1: Preprocess raw-like input.
    df = clean_and_merge(input_df)
    df = drop_duplicates(df)
    df = remove_outliers(df)

    # Step 2: Recreate date features used during training.
    if "date" in df.columns:
        df = add_date_features(df)

    # Step 3: Reapply training encoders.
    # Frequency encoding for zipcode.
    if Path(freq_encoder_path).exists() and "zipcode" in df.columns:
        freq_map = load(freq_encoder_path)
        df["zipcode_freq"] = df["zipcode"].map(freq_map).fillna(0)
        df = df.drop(columns=["zipcode"], errors="ignore")

    # Target encoding for city_full.
    if Path(target_encoder_path).exists() and "city_full" in df.columns:
        target_encoder = load(target_encoder_path)
        df["city_full_encoded"] = target_encoder.transform(df[["city_full"]]).squeeze()
        df = df.drop(columns=["city_full"], errors="ignore")

    # Drop raw columns not expected by model inputs.
    df, _ = drop_unused_columns(df.copy(), df.copy())

    # Step 4: Keep actuals (if provided) for quick validation/debug.
    y_true = None
    if "price" in df.columns:
        y_true = df["price"].tolist()
        df = df.drop(columns=["price"])

    # Step 5: Align columns with training schema to prevent serving skew.
    if TRAIN_FEATURE_COLUMNS is not None:
        df = df.reindex(columns=TRAIN_FEATURE_COLUMNS, fill_value=0)

    # Step 6: Load model and predict.
    resolved_model_path = _resolve_model_path(model_path)
    model = load(resolved_model_path)
    preds = model.predict(df)

    # Step 7: Build output
    out = df.copy()
    out["predicted_price"] = preds
    if y_true is not None:
        out["actual_price"] = y_true

    return out


# ----------------------------
# CLI entrypoint
# ----------------------------
# Allows running inference directly from terminal.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference on new housing data (raw)."
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Path to input RAW CSV file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Path to save predictions CSV",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(DEFAULT_MODEL),
        help="Path to trained model file",
    )
    parser.add_argument(
        "--freq_encoder",
        type=str,
        default=str(DEFAULT_FREQ_ENCODER),
        help="Path to frequency encoder pickle",
    )
    parser.add_argument(
        "--target_encoder",
        type=str,
        default=str(DEFAULT_TARGET_ENCODER),
        help="Path to target encoder pickle",
    )

    args = parser.parse_args()

    raw_df = pd.read_csv(args.input)
    preds_df = predict(
        raw_df,
        model_path=args.model,
        freq_encoder_path=args.freq_encoder,
        target_encoder_path=args.target_encoder,
    )

    preds_df.to_csv(args.output, index=False)
    print(f"✅ Predictions saved to {args.output}")
