from pathlib import Path

import pytest

from src.evaluate import evaluate_all, evaluate_model
from src.train import CANDIDATE_MODELS, make_pipeline, train_test_data

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "text_samples.csv"


@pytest.fixture(scope="module")
def split_data():
    return train_test_data(DATA_PATH)


def test_train_test_data_split_sizes(split_data):
    X_train, X_test, y_train, y_test = split_data
    total = len(X_train) + len(X_test)
    assert abs(len(X_test) / total - 0.2) < 0.02


def test_single_model_trains_and_predicts(split_data):
    X_train, X_test, y_train, y_test = split_data
    pipeline = make_pipeline(CANDIDATE_MODELS["logistic_regression"])
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    assert len(preds) == len(X_test)
    assert set(preds).issubset({0, 1})


def test_evaluate_model_returns_expected_keys(split_data):
    X_train, X_test, y_train, y_test = split_data
    pipeline = make_pipeline(CANDIDATE_MODELS["logistic_regression"])
    pipeline.fit(X_train, y_train)
    metrics = evaluate_model(pipeline, X_test, y_test)
    for key in ("accuracy", "precision", "recall", "f1_score", "roc_auc", "confusion_matrix"):
        assert key in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_evaluate_all_ranks_models(split_data):
    X_train, X_test, y_train, y_test = split_data
    fitted = {}
    for name in ("logistic_regression", "random_forest"):
        pipeline = make_pipeline(CANDIDATE_MODELS[name])
        pipeline.fit(X_train, y_train)
        fitted[name] = pipeline

    results_df = evaluate_all(fitted, X_test, y_test)
    assert len(results_df) == 2
    assert results_df["roc_auc"].is_monotonic_decreasing


def test_model_detects_obvious_ai_text(split_data):
    """Sanity check: blatant AI-style text should score high."""
    X_train, X_test, y_train, y_test = split_data
    pipeline = make_pipeline(CANDIDATE_MODELS["random_forest"])
    pipeline.fit(X_train, y_train)

    import pandas as pd
    text = pd.DataFrame({"text": [
        "Furthermore, it is worth noting that artificial intelligence continues to reshape "
        "modern industries. Moreover, cloud computing has transformed daily productivity. "
        "In summary, these factors are closely interconnected."
    ]})
    proba = pipeline.predict_proba(text)[0, 1]
    assert proba > 0.7
