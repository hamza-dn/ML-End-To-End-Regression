"""Load the raw dataset and create time-based train/eval/holdout splits.

This file is the first step of the feature pipeline.
It keeps the split deterministic so we do not mix future data into training.
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw")


def load_and_split_data(
    raw_path: str | Path = "data/raw/housets_original.csv",
    output_dir: Path | str = DATA_DIR,
):
    """Load the raw file, split by date, and save `*_raw.csv` files.

    We split on date because this is a time series style housing problem.
    The goal is to mimic the real production order: past data for training,
    later data for evaluation, and the most recent data for holdout.
    """
    df = pd.read_csv(raw_path)

    if "date" not in df.columns:
        raise ValueError("Missing required column: 'date'")

    # Convert the date column once so the temporal split is reliable.
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # These dates come from the notebook workflow and stay fixed here.
    cutoff_date_eval = pd.Timestamp("2020-01-01")
    cutoff_date_holdout = pd.Timestamp("2022-01-01")

    # Train = past, Eval = middle period, Holdout = most recent data.
    train_df = df[df["date"] < cutoff_date_eval]
    eval_df = df[(df["date"] >= cutoff_date_eval) & (df["date"] < cutoff_date_holdout)]
    holdout_df = df[df["date"] >= cutoff_date_holdout]

    # Save files with the naming convention used by the rest of the project.
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(outdir / "train_raw.csv", index=False)
    eval_df.to_csv(outdir / "eval_raw.csv", index=False)
    holdout_df.to_csv(outdir / "holdout_raw.csv", index=False)

    print(f"✅ Data split completed (saved to {outdir}).")
    print(
        f"   Train: {train_df.shape}, Eval: {eval_df.shape}, Holdout: {holdout_df.shape}"
    )

    return train_df, eval_df, holdout_df


if __name__ == "__main__":
    load_and_split_data()
