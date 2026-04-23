import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# ============================
# Configuration
# ============================
# API endpoint (local by default, can be overridden by environment variable).
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000/predict")

# Local data paths (from feature pipeline outputs).
# holdout_final.csv = training-ready data (with engineered features).
# holdout_clean.csv = raw-cleaned data (for joining metadata like date/city).
PROJECT_ROOT = Path(__file__).resolve().parent
HOLDOUT_FINAL_PATH = PROJECT_ROOT / "data" / "processed" / "holdout_final.csv"
HOLDOUT_CLEAN_PATH = PROJECT_ROOT / "data" / "processed" / "holdout_clean.csv"


# ============================
# Data loading
# ============================
@st.cache_data
def load_data():
    """Load holdout datasets and prepare for filtering/prediction.

    Returns:
        fe: Full holdout_final.csv (engineered features for API input).
        disp: Display dataframe with date, region, actual price (for filtering).
    """
    # Load feature-engineered data (used as input to API).
    fe = pd.read_csv(HOLDOUT_FINAL_PATH)

    # Load raw-cleaned data to extract metadata (date, city for filtering).
    meta = pd.read_csv(HOLDOUT_CLEAN_PATH, parse_dates=["date"])

    # Ensure both files have same row count.
    if len(fe) != len(meta):
        st.warning("Holdout lengths differ. Aligning by index.")
        min_len = min(len(fe), len(meta))
        fe = fe.iloc[:min_len].copy()
        meta = meta.iloc[:min_len].copy()

    # Build display dataframe for filtering and results.
    disp = pd.DataFrame(index=fe.index)
    disp["date"] = meta["date"]
    disp["region"] = meta["city_full"]
    disp["year"] = disp["date"].dt.year
    disp["month"] = disp["date"].dt.month
    disp["actual_price"] = fe["price"]  # Actual target from training.

    return fe, disp


fe_df, disp_df = load_data()

# ============================
# Streamlit UI
# ============================
st.title("Housing Price Prediction - Holdout Set Explorer")
st.write(
    "Select a time period and region, then click Show Predictions to run batch inference."
)

years = sorted(disp_df["year"].unique())
months = list(range(1, 13))
regions = ["All"] + sorted(disp_df["region"].dropna().unique())

col1, col2, col3 = st.columns(3)
with col1:
    year = st.selectbox("Select Year", years, index=0)
with col2:
    month = st.selectbox("Select Month", months, index=0)
with col3:
    region = st.selectbox("Select Region", regions, index=0)

if st.button("Show Predictions"):
    # Filter by selected year, month, and region.
    mask = (disp_df["year"] == year) & (disp_df["month"] == month)
    if region != "All":
        mask &= disp_df["region"] == region

    idx = disp_df.index[mask]

    if len(idx) == 0:
        st.warning("No data found for these filters.")
    else:
        st.write(f"Running predictions for {year}-{month:02d} | Region: {region}")

        # Build API request: wrap rows in {"rows": [...]} format.
        # The API expects this structure for validation (Pydantic model).
        request_data = fe_df.loc[idx].to_dict(orient="records")
        payload = {"rows": request_data}

        try:
            # Call the API with the batch.
            resp = requests.post(API_URL, json=payload, timeout=60)
            resp.raise_for_status()
            out = resp.json()

            # Extract predictions and actuals from response.
            preds = out.get("predictions", [])
            actuals = out.get("actuals", None)

            # Build results view.
            view = disp_df.loc[idx, ["date", "region", "actual_price"]].copy()
            view = view.sort_values("date")

            # Extract predicted prices from response.
            view["prediction"] = [p.get("predicted_price", None) for p in preds]

            # Update actuals if API returned them (should match).
            if actuals is not None and len(actuals) == len(view):
                view["actual_price"] = pd.Series(actuals, index=view.index).astype(
                    float
                )

            # Compute validation metrics.
            mae = (view["prediction"] - view["actual_price"]).abs().mean()
            rmse = ((view["prediction"] - view["actual_price"]) ** 2).mean() ** 0.5
            avg_pct_error = (
                (view["prediction"] - view["actual_price"]).abs() / view["actual_price"]
            ).mean() * 100

            # Display results table.
            st.subheader("Predictions vs Actuals")
            st.dataframe(
                view[["date", "region", "actual_price", "prediction"]].reset_index(
                    drop=True
                ),
                use_container_width=True,
            )

            # Show key metrics.
            st.subheader("Performance Metrics")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("MAE", f"${mae:,.0f}")
            with c2:
                st.metric("RMSE", f"${rmse:,.0f}")
            with c3:
                st.metric("Avg % Error", f"{avg_pct_error:.2f}%")

            # ============================
            # Yearly Trend Chart
            # ============================
            # Generate monthly average trend for the selected year.
            if region == "All":
                # Predict for all records in the selected year.
                yearly_data = disp_df[disp_df["year"] == year].copy()
                idx_all = yearly_data.index
                request_data_all = fe_df.loc[idx_all].to_dict(orient="records")
                payload_all = {"rows": request_data_all}

                resp_all = requests.post(API_URL, json=payload_all, timeout=60)
                resp_all.raise_for_status()
                preds_all = resp_all.json().get("predictions", [])

                yearly_data["prediction"] = [
                    p.get("predicted_price", None) for p in preds_all
                ]

            else:
                # Predict for selected region in the year.
                yearly_data = disp_df[
                    (disp_df["year"] == year) & (disp_df["region"] == region)
                ].copy()
                idx_region = yearly_data.index
                request_data_region = fe_df.loc[idx_region].to_dict(orient="records")
                payload_region = {"rows": request_data_region}

                resp_region = requests.post(API_URL, json=payload_region, timeout=60)
                resp_region.raise_for_status()
                preds_region = resp_region.json().get("predictions", [])

                yearly_data["prediction"] = [
                    p.get("predicted_price", None) for p in preds_region
                ]

            # Aggregate predictions and actuals by month.
            monthly_avg = (
                yearly_data.groupby("month")[["actual_price", "prediction"]]
                .mean()
                .reset_index()
            )

            # Create interactive chart.
            region_suffix = "" if region == "All" else f" - {region}"
            fig = px.line(
                monthly_avg,
                x="month",
                y=["actual_price", "prediction"],
                markers=True,
                labels={"value": "Price ($)", "month": "Month"},
                title=f"Yearly Trend - {year}{region_suffix}",
            )

            # Highlight the selected month in the chart.
            fig.add_vrect(
                x0=month - 0.5,
                x1=month + 0.5,
                fillcolor="red",
                opacity=0.1,
                layer="below",
                line_width=0,
            )

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"API call failed: {e}")
            st.exception(e)

else:
    st.info("Choose filters and click Show Predictions to run batch inference.")
