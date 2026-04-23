"""Preprocessing for the housing project.

This step cleans the raw split before feature engineering:
- normalize city names
- merge metro coordinates
- remove duplicates
- remove extreme price outliers
"""

import re
from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Manual fixes for known mismatches (normalized form)
CITY_MAPPING = {
    "las vegas-henderson-paradise": "las vegas-henderson-north las vegas",
    "denver-aurora-lakewood": "denver-aurora-centennial",
    "houston-the woodlands-sugar land": "houston-pasadena-the woodlands",
    "austin-round rock-georgetown": "austin-round rock-san marcos",
    "miami-fort lauderdale-pompano beach": "miami-fort lauderdale-west palm beach",
    "san francisco-oakland-berkeley": "san francisco-oakland-fremont",
    "dc_metro": "washington-arlington-alexandria",
    "atlanta-sandy springs-alpharetta": "atlanta-sandy springs-roswell",
}


def normalize_city(s: str) -> str:
    """Normalize a city name so matching becomes more stable.

    We lowercase text, trim spaces, and unify dash characters.
    This helps city names match even if the raw source uses slightly
    different formatting.
    """
    if pd.isna(s):
        return s
    s = str(s).strip().lower()
    s = re.sub(r"[–—-]", "-", s)  # unify dashes
    s = re.sub(r"\s+", " ", s)  # collapse spaces
    return s


def normalize_metro_full(s: str) -> str:
    """Normalize metros `metro_full` and remove the state suffix.

    Example:
    "New York-Newark-Jersey City, NY-NJ" -> "new york-newark-jersey city"
    """
    s = normalize_city(s)
    if pd.isna(s):
        return s
    return str(s).split(",", 1)[0].strip()


def clean_and_merge(
    df: pd.DataFrame, metros_path: str | None = "data/raw/us_metros.csv"
) -> pd.DataFrame:
    """
    Normalize city names, optionally merge lat/lng from metros dataset.
    If `city_full` column or `metros_path` is missing, skip gracefully.
    """

    if "city_full" not in df.columns:
        print("⚠️ Skipping city merge: no 'city_full' column present.")
        return df

    # First we normalize the city column used for matching.
    df["city_full"] = df["city_full"].apply(normalize_city)

    # Then we replace known aliases so the city names match our metro file.
    norm_mapping = {
        normalize_city(k): normalize_city(v) for k, v in CITY_MAPPING.items()
    }
    df["city_full"] = df["city_full"].replace(norm_mapping)

    # If lat/lng already exist, we do not need to merge again.
    if {"lat", "lng"}.issubset(df.columns):
        print("⚠️ Skipping lat/lng merge: already present in DataFrame.")
        return df

    # If no metros file provided / exists, skip merge
    if not metros_path or not Path(metros_path).exists():
        print("⚠️ Skipping lat/lng merge: metros file not provided or not found.")
        return df

    # Merge coordinates from the reference metro dataset.
    metros = pd.read_csv(metros_path)
    if "metro_full" not in metros.columns or not {"lat", "lng"}.issubset(
        metros.columns
    ):
        print("⚠️ Skipping lat/lng merge: metros file missing required columns.")
        return df

    metros["metro_full"] = metros["metro_full"].apply(normalize_metro_full)
    df = df.merge(
        metros[["metro_full", "lat", "lng"]],
        how="left",
        left_on="city_full",
        right_on="metro_full",
    )
    df.drop(columns=["metro_full"], inplace=True, errors="ignore")

    missing = df[df["lat"].isnull()]["city_full"].unique()
    if len(missing) > 0:
        print("⚠️ Still missing lat/lng for:", missing)
    else:
        print("✅ All cities matched with metros dataset.")
    return df


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicates while keeping the first version of each row.

    We ignore date/year in the duplicate check because the same listing can
    appear in multiple periods, and we only want to remove repeated rows.
    """
    before = df.shape[0]
    dedup_subset = [c for c in df.columns if c not in {"date", "year"}]
    df = df.drop_duplicates(subset=dedup_subset, keep="first")
    after = df.shape[0]
    print(f"✅ Dropped {before - after} duplicate rows (excluding date/year).")
    return df


def remove_outliers(
    df: pd.DataFrame, max_median_list_price: float = 19_000_000
) -> pd.DataFrame:
    """Remove extreme outliers in median_list_price.

    This is a simple safety filter to avoid training on absurd values that
    can distort the model.
    """
    if "median_list_price" not in df.columns:
        return df
    before = df.shape[0]
    df = df[df["median_list_price"] <= max_median_list_price].copy()
    after = df.shape[0]
    print(f"✅ Removed {before - after} rows with median_list_price > 19M.")
    return df


def preprocess_split(
    split: str,
    raw_dir: Path | str = RAW_DIR,
    processed_dir: Path | str = PROCESSED_DIR,
    metros_path: str | None = "data/raw/us_metros.csv",
    max_median_list_price: float = 19_000_000,
) -> pd.DataFrame:
    """Run preprocessing for one split and save the cleaned CSV.

    Input example:  data/raw/train_raw.csv
    Output example: data/processed/train_clean.csv
    """
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    path = raw_dir / f"{split}_raw.csv"
    df = pd.read_csv(path)

    # The order matters: normalize names first, then clean duplicates and outliers.
    df = clean_and_merge(df, metros_path=metros_path)
    df = drop_duplicates(df)
    df = remove_outliers(df, max_median_list_price=max_median_list_price)

    out_path = processed_dir / f"{split}_clean.csv"
    df.to_csv(out_path, index=False)
    print(f"✅ Preprocessed {split} saved to {out_path} ({df.shape})")
    return df


def run_preprocess(
    splits: tuple[str, ...] = ("train", "eval", "holdout"),
    raw_dir: Path | str = RAW_DIR,
    processed_dir: Path | str = PROCESSED_DIR,
    metros_path: str | None = "data/raw/us_metros.csv",
    max_median_list_price: float = 19_000_000,
):
    for s in splits:
        preprocess_split(
            s,
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            metros_path=metros_path,
            max_median_list_price=max_median_list_price,
        )


if __name__ == "__main__":
    run_preprocess()
