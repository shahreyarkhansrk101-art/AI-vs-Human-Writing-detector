from src.features import FEATURE_NAMES, extract_features


def test_extract_features_returns_all_keys():
    feats = extract_features("This is a simple test sentence. It has two sentences.")
    assert set(feats.keys()) == set(FEATURE_NAMES)


def test_extract_features_handles_empty_string():
    feats = extract_features("")
    assert all(v == 0.0 for v in feats.values())


def test_extract_features_detects_ai_phrases():
    ai_text = "Furthermore, this is important. Moreover, it is worth noting that this matters."
    plain_text = "This is important. It matters a lot to people."
    ai_feats = extract_features(ai_text)
    plain_feats = extract_features(plain_text)
    assert ai_feats["ai_phrase_rate"] > plain_feats["ai_phrase_rate"]


def test_extract_features_detects_contractions():
    contracted = "I'm happy you're here, it's great, don't leave."
    plain = "I am happy you are here, it is great, do not leave."
    assert extract_features(contracted)["contraction_rate"] > extract_features(plain)["contraction_rate"]


def test_extract_features_word_count():
    feats = extract_features("one two three four five")
    assert feats["num_words"] == 5
