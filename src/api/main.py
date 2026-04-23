"""FastAPI app exposing local prediction endpoints.

This module wraps the inference pipeline behind HTTP routes so we can
serve predictions locally (and later in Docker/CI/CD).
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.inference_pipeline.inference import predict


class PredictRequest(BaseModel):
    """Request payload for batch prediction.

    `rows` is a list of records where each record is one house observation.
    Keys should match the columns expected by the inference pipeline.
    """

    rows: list[dict[str, Any]] = Field(..., min_length=1)


class PredictResponse(BaseModel):
    """Response payload returned by the prediction route.

    Contains predictions and optionally actual prices for validation.
    """

    n_rows: int
    predictions: list[dict[str, Any]]
    actuals: list[float] | None = None


app = FastAPI(
    title="Housing MLE API",
    description="Local API for house price predictions using trained artifacts.",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    """Basic health endpoint used by local checks and containers."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict_route(payload: PredictRequest) -> PredictResponse:
    """Run inference on a batch of rows and return predictions.

    We keep this route intentionally simple for the first production step.
    """
    try:
        input_df = pd.DataFrame(payload.rows)
        if input_df.empty:
            raise HTTPException(status_code=400, detail="Input rows are empty.")

        # Extract actual prices if present (for validation/metrics display).
        actuals = None
        if "price" in input_df.columns:
            actuals = input_df["price"].tolist()

        preds_df = predict(input_df)
        return PredictResponse(
            n_rows=len(preds_df),
            predictions=preds_df.to_dict(orient="records"),
            actuals=actuals,
        )
    except HTTPException:
        raise
    except Exception as exc:
        # Return a clear API error instead of an internal traceback.
        raise HTTPException(
            status_code=500, detail=f"Prediction failed: {exc}"
        ) from exc
