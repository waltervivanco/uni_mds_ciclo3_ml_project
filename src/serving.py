from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "artifacts" / "models" / "baseline_model.joblib"
FEATURES_PATH = PROJECT_ROOT / "artifacts" / "meta" / "feature_columns.joblib"

app = FastAPI(title="WA1200 Model API", version="1.0.0")

model: Any = None
feature_columns: list[str] = []


class PredictionInput(BaseModel):
    features: dict[str, float]


@app.on_event("startup")
def load_artifacts() -> None:
    global model
    global feature_columns

    if not MODEL_PATH.exists() or not FEATURES_PATH.exists():
        raise RuntimeError(
            "Model artifacts not found. Run `python src/train.py` first."
        )

    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURES_PATH)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: PredictionInput) -> dict[str, Any]:
    if model is None or not feature_columns:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    row = {col: payload.features.get(col, np.nan) for col in feature_columns}
    X = pd.DataFrame([row], columns=feature_columns)

    try:
        pred = int(model.predict(X)[0])
        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(X)[0, 1])
        else:
            prob = None
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction error: {exc}") from exc

    return {
        "prediction": pred,
        "probability_high_consumption": prob,
    }
