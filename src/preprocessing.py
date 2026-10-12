"""Data loading and preprocessing for the AI-vs-human text classifier."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import FEATURE_NAMES, extract_features

TARGET_COLUMN = "label"
TEXT_COLUMN = "text"


class LinguisticFeatureExtractor(BaseEstimator, TransformerMixin):
    """Sklearn-compatible transformer wrapping extract_features() so it can
    sit inside a ColumnTransformer/Pipeline alongside TF-IDF."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        texts = X.iloc[:, 0] if hasattr(X, "iloc") else X
        rows = [extract_features(t) for t in texts]
        return pd.DataFrame(rows, columns=FEATURE_NAMES).values

    def get_feature_names_out(self, input_features=None):
        return np.array(FEATURE_NAMES)


def load_data(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the text column from the binary label (1 = AI-generated)."""
    X = df[[TEXT_COLUMN]].copy()
    y = (df[TARGET_COLUMN] == "AI").astype(int)
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """
    Combine:
      - TF-IDF over the raw text (unigrams + bigrams, capped vocabulary)
      - engineered linguistic features (burstiness, hedge-phrase rate, etc.), scaled
    """
    return ColumnTransformer(transformers=[
        ("tfidf", TfidfVectorizer(max_features=1500, ngram_range=(1, 2), min_df=2), TEXT_COLUMN),
        ("linguistic", Pipeline(steps=[
            ("extract", LinguisticFeatureExtractor()),
            ("scale", StandardScaler()),
        ]), [TEXT_COLUMN]),
    ])
