"""
Generates a synthetic-but-realistic corpus of human-written and
AI-generated text, so the classifier trains fully offline with no
internet access to real corpora or LLM APIs.

Text is built from topic-based sentence templates, then passed through a
style transform: an "AI style" (formal transitions, hedge words, uniform
sentence length, no typos) or a "human style" (contractions, informal
connectors, variable sentence length/burstiness, occasional typos).

To avoid a trivially separable dataset, ~12% of each class is generated
in the *other* class's style ("human_formal" writing that reads like AI
output; "ai_casual" output where the model was asked to write casually) —
these are the genuinely hard cases a real detector has to handle.

Run: python data/generate_text_dataset.py
Output: data/text_samples.csv
"""
import re
import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_SEED = 42
N_SAMPLES = 3000

TOPICS = {
    "technology": {
        "subjects": ["smartphone technology", "cloud computing", "artificial intelligence", "electric vehicle adoption", "5G connectivity", "wearable device technology"],
        "verbs": ["has transformed", "continues to reshape", "is changing", "has improved", "is disrupting"],
        "objects": ["how people communicate", "modern industries", "daily productivity", "the way businesses operate", "consumer habits"],
    },
    "travel": {
        "subjects": ["budget airline growth", "remote work culture", "travel app adoption", "off-season tourism", "the rise of local guides"],
        "verbs": ["has made", "is making", "has simplified", "continues to influence"],
        "objects": ["international travel more accessible", "trip planning easier", "tourism more sustainable", "cultural exchange more common"],
    },
    "food": {
        "subjects": ["the plant-based diet trend", "food delivery culture", "home cooking", "meal prepping", "the farmers market movement"],
        "verbs": ["has grown", "is becoming", "continues to gain", "has increased in"],
        "objects": ["popularity among young adults", "relevance in urban areas", "importance for busy households", "appeal over the past few years"],
    },
    "environment": {
        "subjects": ["renewable energy investment", "recycling infrastructure", "urban green space planning", "carbon pricing policy", "reforestation work"],
        "verbs": ["plays a growing role in", "is essential to", "contributes significantly to", "helps support"],
        "objects": ["reducing emissions", "protecting biodiversity", "improving air quality", "building sustainable cities"],
    },
    "health": {
        "subjects": ["regular exercise", "sleep quality", "mental health awareness", "preventive care", "a balanced diet"],
        "verbs": ["is linked to", "plays a key role in", "significantly affects", "contributes to"],
        "objects": ["overall wellbeing", "long-term health outcomes", "reduced stress levels", "better quality of life"],
    },
    "finance": {
        "subjects": ["index fund investing", "personal budgeting", "compound interest", "financial literacy", "automated saving"],
        "verbs": ["can help build", "is central to", "plays an important role in", "supports"],
        "objects": ["long-term wealth", "financial independence", "sound investment decisions", "a stable financial future"],
    },
    "education": {
        "subjects": ["online learning", "project-based learning", "personalized tutoring", "early literacy programming"],
        "verbs": ["has reshaped", "continues to improve", "is changing", "has expanded access to"],
        "objects": ["how students learn", "educational outcomes", "classroom engagement", "learning opportunities worldwide"],
    },
    "sports": {
        "subjects": ["data analytics", "youth training programming", "sports science", "wearable fitness tracking"],
        "verbs": ["has changed", "continues to influence", "plays a growing role in", "has improved"],
        "objects": ["how teams prepare for competition", "athlete performance", "injury prevention strategies", "coaching decisions"],
    },
}

AI_TRANSITIONS = [
    "Furthermore, ", "Moreover, ", "In addition, ", "It is worth noting that ",
    "Additionally, ", "That said, ", "Notably, ", "As a result, ",
]
AI_OPENERS = [
    "Certainly! ", "Sure, let's break this down. ", "Great question. ",
    "I'd be happy to explain. ", "", "", "",  # empty strings = no opener, more common
]
AI_CLOSERS = [
    " Overall, this reflects a broader trend worth considering.",
    " In summary, these factors are closely interconnected.",
    " Taken together, these points highlight an important shift.",
    "", "", "",
]
AI_HEDGES = ["generally speaking, ", "in many cases, ", "it's generally true that ", "typically, "]

HUMAN_CONNECTORS = ["also, ", "honestly, ", "so ", "anyway, ", "plus, ", "and yeah, "]
HUMAN_OPENERS = ["so ", "honestly, ", "ok so ", "not gonna lie, ", "", "", ""]
HUMAN_REACTIONS = [
    "honestly love this.", "not bad at all.", "still kind of amazes me.",
    "took a while to notice tbh.", "pretty underrated imo.", "wish more people talked about this.",
]
HUMAN_ASIDES = [
    " honestly not sure why but it works.",
    " idk, just my take.",
    " at least that's been my experience.",
    " could be wrong tho.",
    "",
    "",
]
CONTRACTIONS = {
    "it is": "it's", "do not": "don't", "does not": "doesn't", "cannot": "can't",
    "I am": "I'm", "you are": "you're", "they are": "they're", "we are": "we're",
    "is not": "isn't", "was not": "wasn't", "have not": "haven't", "will not": "won't",
    "I have": "I've", "I will": "I'll", "that is": "that's",
}


def _fill_sentence(topic: str, rng: np.random.Generator) -> str:
    t = TOPICS[topic]
    subject = rng.choice(t["subjects"])
    verb = rng.choice(t["verbs"])
    obj = rng.choice(t["objects"])
    sentence = f"{subject[0].upper()}{subject[1:]} {verb} {obj}"
    return sentence


def _apply_typo(word: str, rng: np.random.Generator) -> str:
    if len(word) < 4:
        return word
    i = rng.integers(1, len(word) - 1)
    kind = rng.choice(["swap", "drop"])
    if kind == "swap":
        chars = list(word)
        j = i + 1 if i + 1 < len(chars) else i - 1
        chars[i], chars[j] = chars[j], chars[i]
        return "".join(chars)
    return word[:i] + word[i + 1:]


def _ai_style(sentences: list[str], rng: np.random.Generator, casual: bool = False) -> str:
    parts = []
    opener = rng.choice(AI_OPENERS)
    for i, s in enumerate(sentences):
        if i > 0 and rng.random() < (0.55 if not casual else 0.2):
            s = rng.choice(AI_TRANSITIONS) + s[0].lower() + s[1:]
        if rng.random() < (0.25 if not casual else 0.1):
            s = rng.choice(AI_HEDGES) + s[0].lower() + s[1:]
        parts.append(s.strip().rstrip(".") + ".")
    text = opener + " ".join(parts)
    if not casual:
        text += rng.choice(AI_CLOSERS)
    if casual:
        # "write casually" AI output: a couple of light contractions, no typos
        for full, contr in CONTRACTIONS.items():
            if rng.random() < 0.4:
                text = re.sub(full, contr, text, flags=re.IGNORECASE)
    return text.strip()


def _human_style(sentences: list[str], rng: np.random.Generator, formal: bool = False) -> str:
    parts = []
    opener = "" if formal else rng.choice(HUMAN_OPENERS)
    for i, s in enumerate(sentences):
        if i > 0 and not formal and rng.random() < 0.35:
            s = rng.choice(HUMAN_CONNECTORS) + s[0].lower() + s[1:]

        # apply contractions
        for full, contr in CONTRACTIONS.items():
            if rng.random() < (0.75 if not formal else 0.3):
                s = re.sub(full, contr, s, flags=re.IGNORECASE)

        # occasional typo
        if not formal and rng.random() < 0.12:
            words = s.split()
            idx = rng.integers(0, len(words))
            words[idx] = _apply_typo(words[idx], rng)
            s = " ".join(words)

        punct = "."
        if not formal and rng.random() < 0.15:
            punct = rng.choice(["!", "!!", "..."])
        parts.append(s.strip().rstrip(".") + punct)

        # occasionally insert a short reaction sentence for burstiness
        # (a genuinely short sentence, not a mid-sentence break)
        if not formal and rng.random() < 0.18:
            parts.append(rng.choice(HUMAN_REACTIONS))

    text = opener + " ".join(parts)
    if not formal:
        text += rng.choice(HUMAN_ASIDES)
    if not formal and rng.random() < 0.1:
        text = text[0].lower() + text[1:]  # occasional no-capital start
    return text.strip()


def generate_sample(rng: np.random.Generator) -> dict:
    topic = rng.choice(list(TOPICS.keys()))
    n_sentences = int(rng.integers(4, 8))
    sentences = [_fill_sentence(topic, rng) for _ in range(n_sentences)]

    roll = rng.random()
    if roll < 0.44:
        subtype = "ai_standard"
    elif roll < 0.50:
        subtype = "ai_casual"
    elif roll < 0.94:
        subtype = "human_casual"
    else:
        subtype = "human_formal"

    return generate_sample_of_subtype(subtype, rng, topic=topic, sentences=sentences)


SUBTYPES = ("ai_standard", "ai_casual", "human_casual", "human_formal")


def generate_sample_of_subtype(
    subtype: str, rng: np.random.Generator, topic: str | None = None, sentences: list | None = None
) -> dict:
    """Generate one sample of a specific style subtype — used for the API's
    on-demand example generator so a dashboard can request a chosen style."""
    if subtype not in SUBTYPES:
        raise ValueError(f"Unknown subtype: {subtype}. Choose from {SUBTYPES}")

    if topic is None:
        topic = rng.choice(list(TOPICS.keys()))
    if sentences is None:
        n_sentences = int(rng.integers(4, 8))
        sentences = [_fill_sentence(topic, rng) for _ in range(n_sentences)]

    if subtype == "ai_standard":
        text, label = _ai_style(sentences, rng, casual=False), "AI"
    elif subtype == "ai_casual":
        text, label = _ai_style(sentences, rng, casual=True), "AI"
    elif subtype == "human_casual":
        text, label = _human_style(sentences, rng, formal=False), "HUMAN"
    else:
        text, label = _human_style(sentences, rng, formal=True), "HUMAN"

    return {"text": text, "label": label, "topic": topic, "subtype": subtype}


def generate_dataset(n=N_SAMPLES, seed=RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = [generate_sample(rng) for _ in range(n)]
    df = pd.DataFrame(rows)
    df.insert(0, "sample_id", [f"TXT-{i:05d}" for i in range(1, len(df) + 1)])
    return df


if __name__ == "__main__":
    df = generate_dataset()
    out_path = Path(__file__).parent / "text_samples.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} samples to {out_path}")
    print(df["label"].value_counts())
    print(df["subtype"].value_counts())
