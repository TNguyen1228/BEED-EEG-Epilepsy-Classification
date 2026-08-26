import os
from contextlib import asynccontextmanager
from typing import Dict, Any
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    SingleSampleRequest,
    BatchSampleRequest,
    PredictionResponse,
    BatchPredictionResponse,
    HealthResponse
)
from src.data.data_loader import load_config
from src.utils.logger import get_logger

logger = get_logger("APIService")

# Global state
MODEL = None
CONFIG = {}
LABELS = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to load resources on startup."""
    global MODEL, CONFIG, LABELS
    try:
        CONFIG = load_config()
        model_path = CONFIG["model"]["saved_model_path"]
        LABELS = {int(k): v for k, v in CONFIG["labels"].items()}
        
        if os.path.exists(model_path):
            logger.info("Loading model from %s", model_path)
            MODEL = joblib.load(model_path)
            logger.info("Model loaded successfully.")
        else:
            logger.warning("Model file not found at %s. Train the model first.", model_path)
    except Exception as e:
        logger.error("Error during startup: %s", str(e))
    yield
    logger.info("Shutting down API service.")

app = FastAPI(
    title="BEED EEG Epilepsy Classification API",
    description="Production-ready REST API for real-time epileptic seizure classification using Random Forest.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=MODEL is not None,
        version="1.0.0"
    )

@app.get("/info", tags=["Metadata"])
def get_info():
    """Returns dataset and class label metadata."""
    return {
        "project": CONFIG.get("project_name", "BEED Epilepsy MLOps"),
        "features": CONFIG.get("data", {}).get("feature_columns", []),
        "labels": LABELS
    }

def _make_prediction(features_list: list) -> PredictionResponse:
    """Helper to run model inference on a single 16-channel vector."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Please ensure training has run.")
    
    cols = CONFIG["data"]["feature_columns"]
    df_input = pd.DataFrame([features_list], columns=cols)
    
    pred_id = int(MODEL.predict(df_input)[0])
    probs = MODEL.predict_proba(df_input)[0]
    
    prob_dict = {
        LABELS.get(i, f"Class {i}"): round(float(p), 4)
        for i, p in enumerate(probs)
    }
    
    return PredictionResponse(
        class_id=pred_id,
        class_name=LABELS.get(pred_id, f"Class {pred_id}"),
        probabilities=prob_dict
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(request: SingleSampleRequest):
    """Predicts seizure class for a single 16-channel EEG reading."""
    try:
        return _make_prediction(request.features)
    except Exception as e:
        logger.error("Inference error: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch(request: BatchSampleRequest):
    """Predicts seizure classes for multiple EEG samples."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    
    cols = CONFIG["data"]["feature_columns"]
    for sample in request.samples:
        if len(sample) != len(cols):
            raise HTTPException(
                status_code=400,
                detail=f"Each sample must contain exactly {len(cols)} channels."
            )
            
    df_input = pd.DataFrame(request.samples, columns=cols)
    preds = MODEL.predict(df_input)
    probs = MODEL.predict_proba(df_input)
    
    results = []
    for pred_id, prob in zip(preds, probs):
        pid = int(pred_id)
        prob_dict = {
            LABELS.get(i, f"Class {i}"): round(float(p), 4)
            for i, p in enumerate(prob)
        }
        results.append(
            PredictionResponse(
                class_id=pid,
                class_name=LABELS.get(pid, f"Class {pid}"),
                probabilities=prob_dict
            )
        )
        
    return BatchPredictionResponse(
        total_samples=len(results),
        predictions=results
    )

