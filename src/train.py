"""Train and compare candidate models for AI-vs-human text classification."""
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from src.preprocessing import build_preprocessor, load_data, split_features_target

RANDOM_SEED = 42

CANDIDATE_MODELS = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
    "random_forest": RandomForestClassifier(
        n_estimators=250, max_depth=20, random_state=RANDOM_SEED, n_jobs=-1
    ),
    # LinearSVC doesn't natively expose predict_proba; calibrate it so the
    # API can still return a confidence score like the other models.
    "linear_svm": CalibratedClassifierCV(LinearSVC(random_state=RANDOM_SEED), cv=3),
}


def make_pipeline(model) -> Pipeline:
    return Pipeline(steps=[("preprocessor", build_preprocessor()), ("model", model)])


def train_test_data(data_path: str | Path, test_size: float = 0.2):
    df = load_data(data_path)
    X, y = split_features_target(df)
    return train_test_split(X, y, test_size=test_size, random_state=RANDOM_SEED, stratify=y)


def train_all(data_path: str | Path) -> tuple[dict, tuple]:
    X_train, X_test, y_train, y_test = train_test_data(data_path)
    fitted = {}
    for name, model in CANDIDATE_MODELS.items():
        pipeline = make_pipeline(model)
        pipeline.fit(X_train, y_train)
        fitted[name] = pipeline
    return fitted, (X_train, X_test, y_train, y_test)


def save_model(pipeline: Pipeline, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)


def load_model(path: str | Path) -> Pipeline:
    return joblib.load(path)


if __name__ == "__main__":
    data_path = Path(__file__).resolve().parent.parent / "data" / "text_samples.csv"
    fitted_models, _ = train_all(data_path)
    for name, pipeline in fitted_models.items():
        save_model(pipeline, Path(__file__).resolve().parent.parent / "models" / f"{name}.joblib")
    print(f"Trained and saved {len(fitted_models)} models.")
