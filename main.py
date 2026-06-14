"""
FastAPI application exposing a trained Iris classifier for inference.

Run locally:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Docs available at:
    http://localhost:8000/docs
"""
import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

from app.schemas import (
    IrisFeatures,
    PredictionResponse,
    BatchIrisFeatures,
    BatchPredictionResponse,
    HealthResponse,
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")

model_artifacts = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    if os.path.exists(MODEL_PATH):
        artifacts = joblib.load(MODEL_PATH)
        model_artifacts["model"] = artifacts["model"]
        model_artifacts["target_names"] = artifacts["target_names"]
        model_artifacts["feature_names"] = artifacts["feature_names"]
    else:
        model_artifacts["model"] = None
    yield
    model_artifacts.clear()


app = FastAPI(
    title="Iris Classifier API",
    description="A simple API serving a RandomForest Iris classifier",
    version="1.0.0",
    lifespan=lifespan,
)


def _features_to_array(features: IrisFeatures) -> np.ndarray:
    return np.array(
        [[
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width,
        ]]
    )


def _predict_single(features: IrisFeatures) -> PredictionResponse:
    model = model_artifacts["model"]
    target_names = model_artifacts["target_names"]

    X = _features_to_array(features)
    pred_idx = int(model.predict(X)[0])
    probs = model.predict_proba(X)[0]

    probabilities = {
        target_names[i]: round(float(p), 4) for i, p in enumerate(probs)
    }

    return PredictionResponse(
        predicted_class=target_names[pred_idx],
        predicted_class_index=pred_idx,
        probabilities=probabilities,
    )


@app.get("/", tags=["Meta"])
def root():
    return {
        "message": "Iris Classifier API",
        "docs": "/docs",
        "endpoints": ["/health", "/predict", "/predict/batch"],
    }


@app.get("/health", response_model=HealthResponse, tags=["Meta"])
def health():
    return HealthResponse(
        status="ok",
        model_loaded=model_artifacts.get("model") is not None,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(features: IrisFeatures):
    if model_artifacts.get("model") is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return _predict_single(features)


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch(batch: BatchIrisFeatures):
    if model_artifacts.get("model") is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not batch.instances:
        raise HTTPException(status_code=400, detail="No instances provided")

    results = [_predict_single(f) for f in batch.instances]
    return BatchPredictionResponse(predictions=results)
