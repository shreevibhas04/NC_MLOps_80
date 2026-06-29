"""
app.py — Stage 4: FastAPI inference service for hospital readmission prediction.

Loads the trained pipeline (artifacts/best_model.pkl), exposes:
  GET  /health   -> service + model status
  POST /predict  -> readmission probability + label for one encounter
Includes request validation, error handling, and prediction logging (governance).

Run:  uvicorn app:app --host 0.0.0.0 --port 8000
"""
import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
THRESHOLD = 0.5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(ARTIFACTS / "predictions.log"),
              logging.StreamHandler()],
)
log = logging.getLogger("readmission-api")

app = FastAPI(title="Hospital Readmission Predictor", version="1.0")

_model = None
_columns = None


def _load():
    global _model, _columns
    if _model is None:
        _model = joblib.load(ARTIFACTS / "best_model.pkl")
        _columns = json.loads((ARTIFACTS / "input_columns.json").read_text())
    return _model, _columns


class PredictRequest(BaseModel):
    features: dict = Field(..., description="Raw encounter features (column -> value)")


@app.get("/health")
def health():
    try:
        _load()
        return {"status": "ok", "model_loaded": True}
    except Exception as e:  # noqa: BLE001
        return {"status": "degraded", "model_loaded": False, "error": str(e)}


@app.post("/predict")
def predict(req: PredictRequest):
    model, columns = _load()
    if not req.features:
        raise HTTPException(status_code=422, detail="features payload is empty")
    # build a single-row frame aligned to the training columns (missing -> NaN)
    row = pd.DataFrame([req.features]).reindex(columns=columns)
    try:
        prob = float(model.predict_proba(row)[:, 1][0])
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"prediction failed: {e}")
    result = {"readmission_probability": round(prob, 4),
              "readmitted_30d": int(prob >= THRESHOLD),
              "threshold": THRESHOLD}
    log.info(f"prediction={result}")
    return result
