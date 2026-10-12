import pandas as pd
import pytest

from src.preprocessing import build_preprocessor, split_features_target


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "sample_id": ["TXT-00001", "TXT-00002"],
        "text": [
            "Furthermore, artificial intelligence continues to reshape modern industries.",
            "honestly idk why but this is pretty cool lol, anyway it's neat.",
        ],
        "label": ["AI", "HUMAN"],
        "topic": ["technology", "technology"],
        "subtype": ["ai_standard", "human_casual"],
    })


def test_split_features_target_shapes(sample_df):
    X, y = split_features_target(sample_df)
    assert list(X.columns) == ["text"]
    assert len(y) == len(sample_df)


def test_split_features_target_encodes_binary(sample_df):
    _, y = split_features_target(sample_df)
    assert set(y.unique()).issubset({0, 1})
    assert y.iloc[0] == 1  # AI
    assert y.iloc[1] == 0  # HUMAN


def test_preprocessor_fits_and_transforms(sample_df):
    X, _ = split_features_target(sample_df)
    # min_df=2 (the production default) needs shared vocabulary across at
    # least 2 docs, which this 2-row fixture doesn't guarantee — swap in
    # min_df=1 to test the pipeline mechanics, not the production threshold
    # (that threshold is exercised by the full-dataset tests in
    # test_train_evaluate.py).
    preprocessor = build_preprocessor()
    preprocessor.transformers[0] = (
        "tfidf",
        preprocessor.transformers[0][1].set_params(min_df=1),
        "text",
    )
    transformed = preprocessor.fit_transform(X)
    assert transformed.shape[0] == len(sample_df)
    assert transformed.shape[1] > 0
