"""
FastAPI service exposing the trained AI-vs-human text detector.

Endpoints:
  GET  /health              - liveness check
  POST /predict               - score arbitrary text (the core feature)
  GET  /metrics                - saved model evaluation metrics
  GET  /example/{subtype}       - generate a synthetic example of a chosen
                                   style, scored, for demoing without needing
                                   to paste your own text

Run locally:  uvicorn api.app:app --reload --port 8000
Docker:       see Dockerfile / docker-compose.yml
"""
import json
import time
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from data.generate_text_dataset import SUBTYPES, generate_sample_of_subtype
from src.detector import ALERT_THRESHOLD, AITextDetector

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "random_forest.joblib"
METRICS_PATH = ROOT / "results" / "metrics.json"

app = FastAPI(
    title="AI Text Detector API",
    description="ML-based classifier distinguishing human-written from AI-generated text.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_detector: AITextDetector | None = None
_example_rng = np.random.default_rng()


def get_detector() -> AITextDetector:
    global _detector
    if _detector is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail="Model not found. Run `python main.py` to train and save models first.",
            )
        _detector = AITextDetector(MODEL_PATH)
    return _detector


class TextRequest(BaseModel):
    text: str = Field(..., min_length=20, max_length=20000)


class PredictionResponse(BaseModel):
    is_ai_generated: bool
    confidence: float
    threshold: float
    features: dict


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: TextRequest):
    detector = get_detector()
    result = detector.score_text(request.text)
    return PredictionResponse(**result, threshold=ALERT_THRESHOLD)


@app.get("/metrics")
def metrics():
    if not METRICS_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Metrics not found. Run `python main.py` to train models first.",
        )
    with open(METRICS_PATH) as f:
        return json.load(f)


ExampleSubtype = Literal["ai_standard", "ai_casual", "human_casual", "human_formal"]


@app.get("/example/{subtype}")
def example(subtype: ExampleSubtype):
    """
    Generate one synthetic example of a chosen style and score it — lets
    the dashboard demo the detector without requiring pasted text.
    """
    if subtype not in SUBTYPES:
        raise HTTPException(status_code=400, detail=f"Unknown subtype: {subtype}")

    detector = get_detector()
    sample = generate_sample_of_subtype(subtype, _example_rng)
    result = detector.score_text(sample["text"])
    return {
        "timestamp": time.time(),
        "text": sample["text"],
        "true_label": sample["label"],
        "subtype": sample["subtype"],
        "topic": sample["topic"],
        **result,
        "threshold": ALERT_THRESHOLD,
    }
