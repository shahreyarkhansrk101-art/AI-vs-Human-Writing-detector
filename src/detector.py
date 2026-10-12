"""
Wraps a trained pipeline to score arbitrary text as human-written or
AI-generated, and exposes the engineered linguistic features behind the
score for explainability.
"""
from pathlib import Path

import pandas as pd

from src.features import extract_features
from src.train import load_model

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "random_forest.joblib"
ALERT_THRESHOLD = 0.6  # probability above which text is flagged as AI-generated


class AITextDetector:
    """Wraps a trained pipeline to score text and expose feature breakdowns."""

    def __init__(self, model_path: str | Path = MODEL_PATH, threshold: float = ALERT_THRESHOLD):
        self.pipeline = load_model(model_path)
        self.threshold = threshold

    def score_text(self, text: str) -> dict:
        """Score a single piece of text."""
        X = pd.DataFrame({"text": [text]})
        proba = self.pipeline.predict_proba(X)[0, 1]
        return {
            "is_ai_generated": bool(proba >= self.threshold),
            "confidence": round(float(proba), 4),
            "features": extract_features(text),
        }
