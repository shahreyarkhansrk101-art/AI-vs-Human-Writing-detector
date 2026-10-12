"""
Linguistic feature extraction for the AI-vs-human text classifier.

All features are computed with plain regex/string operations — no NLTK
corpora, no downloaded models — so this runs fully offline and works
identically in training and at inference time.
"""
import re

FEATURE_NAMES = [
    "num_words",
    "avg_sentence_len",
    "sentence_len_std",
    "avg_word_len",
    "type_token_ratio",
    "exclamation_rate",
    "question_rate",
    "contraction_rate",
    "ai_phrase_rate",
    "first_person_rate",
    "comma_rate",
]

AI_PHRASES = [
    "furthermore", "moreover", "in addition", "it is worth noting",
    "additionally", "that said", "notably", "as a result",
    "in summary", "overall,", "taken together", "generally speaking",
    "in many cases", "it's generally true", "typically,",
    "certainly!", "let's break this down", "i'd be happy to explain",
]

CONTRACTION_PATTERN = re.compile(r"\b\w+'(s|re|ve|ll|d|t|m)\b", re.IGNORECASE)
WORD_PATTERN = re.compile(r"[A-Za-z']+")
SENTENCE_SPLIT_PATTERN = re.compile(r"[.!?]+")
FIRST_PERSON_PATTERN = re.compile(r"\b(i|i'm|i've|i'll|i'd|my|me|mine|myself)\b", re.IGNORECASE)


def extract_features(text: str) -> dict:
    """Compute the linguistic feature vector for one piece of text."""
    text = text or ""
    words = WORD_PATTERN.findall(text)
    num_words = len(words)

    sentences = [s.strip() for s in SENTENCE_SPLIT_PATTERN.split(text) if s.strip()]
    sentence_lengths = [len(WORD_PATTERN.findall(s)) for s in sentences] or [0]

    if num_words == 0:
        return {name: 0.0 for name in FEATURE_NAMES}

    avg_sentence_len = sum(sentence_lengths) / len(sentence_lengths)
    if len(sentence_lengths) > 1:
        mean = avg_sentence_len
        variance = sum((x - mean) ** 2 for x in sentence_lengths) / len(sentence_lengths)
        sentence_len_std = variance ** 0.5
    else:
        sentence_len_std = 0.0

    avg_word_len = sum(len(w) for w in words) / num_words

    unique_words = {w.lower() for w in words}
    type_token_ratio = len(unique_words) / num_words

    per_100 = 100.0 / num_words
    exclamation_rate = text.count("!") * per_100
    question_rate = text.count("?") * per_100
    comma_rate = text.count(",") * per_100
    contraction_rate = len(CONTRACTION_PATTERN.findall(text)) * per_100
    first_person_rate = len(FIRST_PERSON_PATTERN.findall(text)) * per_100

    text_lower = text.lower()
    ai_phrase_count = sum(text_lower.count(phrase) for phrase in AI_PHRASES)
    ai_phrase_rate = ai_phrase_count * per_100

    return {
        "num_words": float(num_words),
        "avg_sentence_len": avg_sentence_len,
        "sentence_len_std": sentence_len_std,
        "avg_word_len": avg_word_len,
        "type_token_ratio": type_token_ratio,
        "exclamation_rate": exclamation_rate,
        "question_rate": question_rate,
        "contraction_rate": contraction_rate,
        "ai_phrase_rate": ai_phrase_rate,
        "first_person_rate": first_person_rate,
        "comma_rate": comma_rate,
    }
