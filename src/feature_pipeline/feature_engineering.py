"""Feature engineering for the housing project.

This step turns cleaned data into model-ready data:
- add date parts
- encode zipcode frequency
- encode city with target encoding
- remove raw columns that could leak information
"""

from pathlib import Path
import pandas as pd
from category_encoders import TargetEncoder
from joblib import dump

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ---------- feature functions ----------


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create simple time features from the date column.

    These features help the model understand seasonality and long-term
    market changes.
    """
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["quarter"] = df["date"].dt.quarter
    df["month"] = df["date"].dt.month

    # Reorder columns so the new date features are easy to inspect.
    df.insert(1, "year", df.pop("year"))
    df.insert(2, "quarter", df.pop("quarter"))
    df.insert(3, "month", df.pop("month"))
    return df


# Creates a frequency encoding (how often a value appears).
# Fit only on train, then applied to eval.
def frequency_encode(train: pd.DataFrame, eval: pd.DataFrame, col: str):
    """Encode a category by how often it appears in the train set.

    We fit only on train to avoid leaking evaluation information.
    """
    freq_map = train[col].value_counts()
    train[f"{col}_freq"] = train[col].map(freq_map)
    eval[f"{col}_freq"] = eval[col].map(freq_map).fillna(0)
    return train, eval, freq_map


# Uses target encoding (replace category with average of target variable).
# Fitted only on train (prevents leakage).
def target_encode(train: pd.DataFrame, eval: pd.DataFrame, col: str, target: str):
    """
    Use TargetEncoder on `col`, consistently name as <col>_encoded.
    For city_full → city_full_encoded (keeps schema aligned with inference).
    """
    # TargetEncoder learns the average target value per category.
    # We fit only on train so the model never sees future information.
    te = TargetEncoder(cols=[col])
    encoded_col = f"{col}_encoded" if col != "city_full" else "city_full_encoded"
    train[encoded_col] = te.fit_transform(train[[col]], train[target]).squeeze()
    eval[encoded_col] = te.transform(eval[[col]]).squeeze()
    return train, eval, te


def drop_unused_columns(train: pd.DataFrame, eval: pd.DataFrame):
    """Drop raw columns that should not go to the model.

    We remove the original text columns after encoding them.
    We also remove the raw target-like column `median_sale_price` because the
    project predicts `price` and we do not want leakage from similar fields.
    """
    drop_cols = ["date", "city_full", "city", "zipcode", "median_sale_price"]
    train = train.drop(
        columns=[c for c in drop_cols if c in train.columns], errors="ignore"
    )
    eval = eval.drop(
        columns=[c for c in drop_cols if c in eval.columns], errors="ignore"
    )
    return train, eval


# ---------- pipeline ----------


# Handles full pipeline:
# reads cleaned CSVs → applies feature engineering → saves engineered data + encoders.
def run_feature_engineering(
    in_train_path: Path | str | None = None,
    in_eval_path: Path | str | None = None,
    in_holdout_path: Path | str | None = None,
    output_dir: Path | str = PROCESSED_DIR,
):
    """Run feature engineering and write outputs + encoders to disk.

    The important rule is that anything learned from data is fit on train only.
    Then the same learned objects are applied to eval and holdout.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Defaults for inputs
    if in_train_path is None:
        in_train_path = PROCESSED_DIR / "train_clean.csv"
    if in_eval_path is None:
        in_eval_path = PROCESSED_DIR / "eval_clean.csv"
    if in_holdout_path is None:
        in_holdout_path = PROCESSED_DIR / "holdout_clean.csv"

    train_df = pd.read_csv(in_train_path)
    eval_df = pd.read_csv(in_eval_path)
    holdout_df = pd.read_csv(in_holdout_path)

    print("Train date range:", train_df["date"].min(), "to", train_df["date"].max())
    print("Eval date range:", eval_df["date"].min(), "to", eval_df["date"].max())
    print(
        "Holdout date range:", holdout_df["date"].min(), "to", holdout_df["date"].max()
    )

    # Add the same date-based features to all splits.
    train_df = add_date_features(train_df)
    eval_df = add_date_features(eval_df)
    holdout_df = add_date_features(holdout_df)

    # Frequency encoding: learn the mapping on train, then reuse it everywhere.
    freq_map = None
    if "zipcode" in train_df.columns:
        train_df, eval_df, freq_map = frequency_encode(train_df, eval_df, "zipcode")
        holdout_df["zipcode_freq"] = holdout_df["zipcode"].map(freq_map).fillna(0)
        dump(freq_map, MODELS_DIR / "freq_encoder.pkl")

    # Target encoding: same idea, but the mapping is based on the target value.
    target_encoder = None
    if "city_full" in train_df.columns:
        train_df, eval_df, target_encoder = target_encode(
            train_df, eval_df, "city_full", "price"
        )
        holdout_df["city_full_encoded"] = target_encoder.transform(
            holdout_df[["city_full"]]
        ).squeeze()
        dump(target_encoder, MODELS_DIR / "target_encoder.pkl")

    # After encoding, keep only model-ready columns.
    train_df, eval_df = drop_unused_columns(train_df, eval_df)
    holdout_df, _ = drop_unused_columns(holdout_df.copy(), holdout_df.copy())

    # Save the final datasets used by the training pipeline.
    out_train_path = output_dir / "train_final.csv"
    out_eval_path = output_dir / "eval_final.csv"
    out_holdout_path = output_dir / "holdout_final.csv"
    train_df.to_csv(out_train_path, index=False)
    eval_df.to_csv(out_eval_path, index=False)
    holdout_df.to_csv(out_holdout_path, index=False)

    print("✅ Feature engineering complete.")
    print("   Train shape:", train_df.shape)
    print("   Eval  shape:", eval_df.shape)
    print("   Holdout shape:", holdout_df.shape)
    print("   Encoders saved to models/")

    return train_df, eval_df, holdout_df, freq_map, target_encoder


if __name__ == "__main__":
    run_feature_engineering()
